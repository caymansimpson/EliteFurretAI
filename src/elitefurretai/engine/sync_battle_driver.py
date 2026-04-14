"""Synchronous Rust battle driving for the simplified fallback backend.

This module keeps the Rust runtime understandable rather than maximally tuned.
The maintained design is:

- local model inference inside the worker
- batched LSTM inference across snapshots when that is naturally available
- sequential Transformer inference instead of context-length batching tricks
- automatic use of the faster embedder array path when the embedder exposes it

Stage 2 benchmarking showed that some Rust optimizations were worth retaining
only as defaults, not as configurable branches. The websocket backend is now the
primary RL path, so the Rust backend stays as a documented, debuggable fallback.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from elitefurretai.engine.battle_snapshot import BattleSnapshot
from elitefurretai.engine.rust_battle_engine import (
    RustBattleEngine,
    adapt_rust_battle,
    create_standalone_double_battle,
    pokepaste_to_rust_team,
    pokepaste_to_rust_team_json,
)
from elitefurretai.etl.battle_data import team_from_json
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.fast_action_mask import (
    ACTIONS_PER_SLOT,
    PASS_ACTION,
    SWITCH_ACTION_BASE,
    TARGET_TO_OFFSET,
    get_valid_targets_for_request_move,
)
from elitefurretai.rl.players import RNaDAgent


def _normalize_choice_message(message: str) -> str:
    if message.startswith("/choose "):
        return message[len("/choose ") :]
    if message.startswith("/team "):
        return "team " + message[len("/team ") :]
    return message


@dataclass
class RolloutStep:
    state: Optional[Any]
    action: int
    log_prob: float
    value: float
    mask: Optional[np.ndarray]
    is_teampreview: bool
    opponent_fainted: int
    reward: float = 0.0


@dataclass
class SyncPolicyProfile:
    embed_seconds: float = 0.0
    inference_seconds: float = 0.0
    action_decode_seconds: float = 0.0
    rollout_record_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return (
            self.embed_seconds
            + self.inference_seconds
            + self.action_decode_seconds
            + self.rollout_record_seconds
        )


@dataclass
class SyncDriverProfile:
    battle_setup_seconds: float = 0.0
    snapshot_build_seconds: float = 0.0
    baseline_choice_seconds: float = 0.0
    batched_policy_seconds: float = 0.0
    engine_step_seconds: float = 0.0
    trajectory_finalize_seconds: float = 0.0
    policy_embed_seconds: float = 0.0
    policy_inference_seconds: float = 0.0
    policy_action_decode_seconds: float = 0.0
    policy_rollout_record_seconds: float = 0.0

    @property
    def total_profiled_seconds(self) -> float:
        return (
            self.battle_setup_seconds
            + self.snapshot_build_seconds
            + self.baseline_choice_seconds
            + self.batched_policy_seconds
            + self.engine_step_seconds
            + self.trajectory_finalize_seconds
            + self.policy_embed_seconds
            + self.policy_inference_seconds
            + self.policy_action_decode_seconds
            + self.policy_rollout_record_seconds
        )

@dataclass
class SyncDriverStats:
    completed_battles: int
    p1_wins: int
    truncated_battles: int
    turn_limit_truncations: int
    stalled_limit_truncations: int
    dual_limit_truncations: int
    duration_seconds: float
    p1_decisions: int
    rollout_steps: int
    p1_rejected_choices: int
    p2_rejected_choices: int
    p1_fallback_recoveries: int
    p2_fallback_recoveries: int
    p1_unrecovered_rejections: int
    p2_unrecovered_rejections: int
    profile: SyncDriverProfile = field(default_factory=SyncDriverProfile)

    @property
    def battles_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.completed_battles / self.duration_seconds

    @property
    def non_truncated_battles(self) -> int:
        return self.completed_battles - self.truncated_battles

    @property
    def non_truncated_battles_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.non_truncated_battles / self.duration_seconds

    @property
    def decisions_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.p1_decisions / self.duration_seconds


class SyncPolicyPlayer:
    def __init__(
        self,
        agent: RNaDAgent,
        format_id: str,
        *,
        device: str = "cpu",
        feature_set: str = Embedder.FULL,
        collect_trajectories: bool = False,
        probabilistic: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_battle_steps: int = 40,
        opponent_type: str = "self_play",
    ):
        self.agent = agent
        self.device = device
        self.embedder = Embedder(format=format_id, feature_set=feature_set, omniscient=False)
        self.collect_trajectories = collect_trajectories
        self.probabilistic = probabilistic
        self.temperature = temperature
        self.top_p = top_p
        self.max_battle_steps = max_battle_steps
        self.opponent_type = opponent_type
        self.hidden_states: Dict[str, Any] = {}
        self.current_trajectories: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self.completed_trajectories: List[Dict[str, Any]] = []
        self._battle_opponent_types: Dict[str, str] = {}
        self._discarded_battles: set[str] = set()
        self._is_transformer = getattr(agent, "_is_transformer", False)
        self.profile = SyncPolicyProfile()

    def _trim_transformer_context(self, hidden_state: Any) -> Any:
        if not self._is_transformer or hidden_state is None:
            return hidden_state

        max_seq_len = getattr(self.agent.model, "max_seq_len", None)
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            return hidden_state
        if hidden_state.size(1) <= max_seq_len:
            return hidden_state
        return hidden_state[:, -max_seq_len:, :].contiguous()

    def _transformer_context_length(self, hidden_state: Any) -> int:
        if hidden_state is None:
            return 0
        return int(hidden_state.size(1))

    def _embed_battle_state(self, battle: Any) -> np.ndarray:
        embed_to_array = getattr(self.embedder, "embed_to_array", None)
        if callable(embed_to_array):
            return cast(np.ndarray, embed_to_array(battle))
        return np.asarray(self.embedder.embed_to_vector(battle), dtype=np.float32)

    def update_sampling(self, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p

    def choose_action(
        self,
        battle_tag: str,
        battle: Any,
        legal_actions: List[Tuple[int, str]],
        opponent_type: Optional[str] = None,
    ) -> Tuple[str, Optional[RolloutStep]]:
        if opponent_type is not None:
            self._battle_opponent_types[battle_tag] = opponent_type

        snapshot = BattleSnapshot(
            battle_tag=battle_tag,
            side="p1",
            battle=battle,
            request={},
            legal_actions=legal_actions,
            action_mask=None,
            action_to_choice={},
            is_teampreview=bool(battle.teampreview),
            opponent_fainted=sum(1 for mon in battle.opponent_team.values() if mon.fainted),
        )
        return self.choose_action_from_snapshot(snapshot, opponent_type=opponent_type)

    def choose_action_from_snapshot(
        self,
        snapshot: BattleSnapshot,
        opponent_type: Optional[str] = None,
    ) -> Tuple[str, Optional[RolloutStep]]:
        if opponent_type is not None:
            self._battle_opponent_types[snapshot.battle_tag] = opponent_type
        current_steps = len(self.current_trajectories.get(snapshot.battle_tag, []))
        record_trajectory = current_steps < self.max_battle_steps

        state: Any = snapshot.state_vector
        if state is None:
            embed_start = time.perf_counter()
            try:
                state = self._embed_battle_state(snapshot.battle)
            except Exception:
                return "default", None
            finally:
                self.profile.embed_seconds += time.perf_counter() - embed_start

        if snapshot.is_teampreview:
            action_idx, choice, log_prob, value = self._sample_teampreview(snapshot.battle_tag, state)
            rollout_step = self._build_rollout_step(
                battle=snapshot.battle,
                state=state,
                action=action_idx,
                log_prob=log_prob,
                value=value,
                mask=None,
                record_trajectory=record_trajectory,
            )
            return choice, rollout_step

        valid_actions = [action for action, _ in snapshot.legal_actions if action >= 0]
        if not valid_actions:
            return "default", None

        mask = snapshot.action_mask
        action_to_choice = snapshot.action_to_choice
        if mask is None:
            mask = np.zeros(MDBO.action_space(), dtype=np.int8)
            action_to_choice = {}
            for action, choice in snapshot.legal_actions:
                if action < 0:
                    continue
                mask[action] = 1
                action_to_choice.setdefault(action, choice)

        action_idx, log_prob, value = self._sample_masked_action(
            snapshot.battle_tag,
            state,
            mask,
            is_teampreview=False,
        )
        choice = action_to_choice.get(action_idx, "default")
        rollout_step = self._build_rollout_step(
            battle=snapshot.battle,
            state=state,
            action=action_idx,
            log_prob=log_prob,
            value=value,
            mask=mask,
            record_trajectory=record_trajectory,
        )
        return choice, rollout_step

    def choose_actions_from_snapshots(
        self,
        snapshots: Sequence[BattleSnapshot],
        opponent_types: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Tuple[str, Optional[RolloutStep]]]:
        if not snapshots:
            return []

        if opponent_types is None:
            opponent_types = [None] * len(snapshots)

        if len(snapshots) == 1:
            return [
                self.choose_action_from_snapshot(snapshot, opponent_type=opponent_type)
                for snapshot, opponent_type in zip(snapshots, opponent_types)
            ]

        if self._is_transformer:
            return [
                self.choose_action_from_snapshot(snapshot, opponent_type=opponent_type)
                for snapshot, opponent_type in zip(snapshots, opponent_types)
            ]

        prepared: List[Dict[str, Any]] = []
        results: List[Tuple[str, Optional[RolloutStep]]] = [("default", None)] * len(snapshots)

        for index, (snapshot, opponent_type) in enumerate(zip(snapshots, opponent_types)):
            if opponent_type is not None:
                self._battle_opponent_types[snapshot.battle_tag] = opponent_type
            current_steps = len(self.current_trajectories.get(snapshot.battle_tag, []))
            record_trajectory = current_steps < self.max_battle_steps

            state: Any = snapshot.state_vector
            if state is None:
                embed_start = time.perf_counter()
                try:
                    state = self._embed_battle_state(snapshot.battle)
                except Exception:
                    continue
                finally:
                    self.profile.embed_seconds += time.perf_counter() - embed_start

            if snapshot.is_teampreview:
                prepared.append(
                    {
                        "index": index,
                        "snapshot": snapshot,
                        "state": state,
                        "record_trajectory": record_trajectory,
                        "mask": None,
                        "action_to_choice": {},
                        "is_teampreview": True,
                    }
                )
                continue

            valid_actions = [action for action, _ in snapshot.legal_actions if action >= 0]
            if not valid_actions:
                continue

            mask = snapshot.action_mask
            action_to_choice = snapshot.action_to_choice
            if mask is None:
                mask = np.zeros(MDBO.action_space(), dtype=np.int8)
                action_to_choice = {}
                for action, choice in snapshot.legal_actions:
                    if action < 0:
                        continue
                    mask[action] = 1
                    action_to_choice.setdefault(action, choice)

            prepared.append(
                {
                    "index": index,
                    "snapshot": snapshot,
                    "state": state,
                    "record_trajectory": record_trajectory,
                    "mask": mask,
                    "action_to_choice": action_to_choice,
                    "is_teampreview": False,
                }
            )

        if not prepared:
            return results

        states_np = np.asarray([entry["state"] for entry in prepared], dtype=np.float32)
        state_tensor = torch.as_tensor(states_np, dtype=torch.float32, device=self.device).unsqueeze(1)

        hidden_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for entry in prepared:
            battle_tag = entry["snapshot"].battle_tag
            hidden = self.hidden_states.get(battle_tag)
            if hidden is None:
                hidden = self.agent.get_initial_state(1, self.device)
            hidden_states.append(cast(Tuple[torch.Tensor, torch.Tensor], hidden))

        hidden = (
            torch.cat([state[0] for state in hidden_states], dim=1),
            torch.cat([state[1] for state in hidden_states], dim=1),
        )

        inference_start = time.perf_counter()
        with torch.no_grad():
            turn_logits, tp_logits, values, _, next_hidden = self.agent(state_tensor, hidden)
        self.profile.inference_seconds += time.perf_counter() - inference_start

        next_hidden = cast(Tuple[torch.Tensor, torch.Tensor], next_hidden)
        for batch_index, entry in enumerate(prepared):
            snapshot = cast(BattleSnapshot, entry["snapshot"])
            logits = tp_logits[batch_index, 0] if entry["is_teampreview"] else turn_logits[batch_index, 0]
            mask = cast(Optional[np.ndarray], entry["mask"])
            decode_start = time.perf_counter()
            action_idx, log_prob = self._select_action_from_logits(logits, mask)
            choice = (
                MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW).message
                if entry["is_teampreview"]
                else cast(Dict[int, str], entry["action_to_choice"]).get(action_idx, "default")
            )
            if entry["is_teampreview"]:
                choice = _normalize_choice_message(choice)
            self.profile.action_decode_seconds += time.perf_counter() - decode_start

            self.hidden_states[snapshot.battle_tag] = (
                next_hidden[0][:, batch_index : batch_index + 1, :].cpu(),
                next_hidden[1][:, batch_index : batch_index + 1, :].cpu(),
            )
            rollout_step = self._build_rollout_step(
                battle=snapshot.battle,
                state=cast(List[float], entry["state"]),
                action=action_idx,
                log_prob=log_prob,
                value=float(values[batch_index, 0].item()),
                mask=mask,
                record_trajectory=bool(entry["record_trajectory"]),
            )
            results[cast(int, entry["index"])] = (choice, rollout_step)

        return results

    def finish_battle(self, battle_tag: str, won: bool, truncated: bool = False) -> None:
        if battle_tag in self._discarded_battles or truncated:
            self._discarded_battles.discard(battle_tag)
            self.current_trajectories.pop(battle_tag, None)
            self.hidden_states.pop(battle_tag, None)
            self._battle_opponent_types.pop(battle_tag, None)
            return

        if not self.collect_trajectories:
            self.current_trajectories.pop(battle_tag, None)
            self.hidden_states.pop(battle_tag, None)
            self._battle_opponent_types.pop(battle_tag, None)
            return

        traj = self.current_trajectories.pop(battle_tag, None)
        self.hidden_states.pop(battle_tag, None)
        opponent_type = self._battle_opponent_types.pop(battle_tag, self.opponent_type)
        if not traj:
            return

        for index, step in enumerate(traj):
            if step is None:
                continue
            step_reward = -0.005
            if index == len(traj) - 1:
                step_reward += 1.0 if won else -1.0
            prev_step = traj[index - 1] if index > 0 else None
            prev_fainted = int(prev_step["opponent_fainted"]) if prev_step is not None else 0
            ko_delta = step["opponent_fainted"] - prev_fainted
            if ko_delta > 0:
                step_reward += 0.05 * ko_delta
            step["reward"] = step_reward

        filtered_traj = [step for step in traj if step is not None]
        if filtered_traj:
            self.completed_trajectories.append(
                {
                    "steps": filtered_traj,
                    "opponent_type": opponent_type,
                    "won": won,
                    "battle_length": len(filtered_traj),
                    "forfeited": False,
                }
            )

    def consume_completed_trajectories(self) -> List[Dict[str, Any]]:
        completed = list(self.completed_trajectories)
        self.completed_trajectories.clear()
        return completed

    def _build_rollout_step(
        self,
        *,
        battle: Any,
        state: Any,
        action: int,
        log_prob: float,
        value: float,
        mask: Optional[np.ndarray],
        record_trajectory: bool,
    ) -> Optional[RolloutStep]:
        if not self.collect_trajectories or action < 0 or not record_trajectory:
            return None
        rollout_start = time.perf_counter()
        step = {
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": 0.0,
            "is_teampreview": bool(battle.teampreview),
            "mask": mask,
            "opponent_fainted": sum(1 for mon in battle.opponent_team.values() if mon.fainted),
        }
        self.current_trajectories.setdefault(getattr(battle, "battle_tag", "unknown"), []).append(step)
        self.profile.rollout_record_seconds += time.perf_counter() - rollout_start
        return RolloutStep(
            state=state,
            action=action,
            log_prob=log_prob,
            value=value,
            mask=mask,
            is_teampreview=bool(battle.teampreview),
            opponent_fainted=cast(int, step["opponent_fainted"]),
        )

    def _sample_teampreview(self, battle_tag: str, state: Any) -> Tuple[int, str, float, float]:
        action, log_prob, value = self._sample_masked_action(
            battle_tag,
            state,
            mask=None,
            is_teampreview=True,
        )
        message = MDBO.from_int(action, type=MDBO.TEAMPREVIEW).message
        return action, _normalize_choice_message(message), log_prob, value

    def _sample_masked_action(
        self,
        battle_tag: str,
        state: Any,
        mask: Optional[np.ndarray],
        *,
        is_teampreview: bool,
    ) -> Tuple[int, float, float]:
        state_tensor = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device).unsqueeze(1)
        hidden = self.hidden_states.get(battle_tag)
        if hidden is None:
            hidden = self.agent.get_initial_state(1, self.device)
        hidden = self._trim_transformer_context(hidden)

        inference_start = time.perf_counter()
        with torch.no_grad():
            turn_logits, tp_logits, values, _, next_hidden = self.agent(state_tensor, hidden)
        self.profile.inference_seconds += time.perf_counter() - inference_start

        logits = tp_logits[0, 0] if is_teampreview else turn_logits[0, 0]
        decode_start = time.perf_counter()
        action, log_prob = self._select_action_from_logits(logits, mask)
        self.profile.action_decode_seconds += time.perf_counter() - decode_start

        if self._is_transformer:
            next_hidden = self._trim_transformer_context(next_hidden)
            self.hidden_states[battle_tag] = next_hidden.cpu() if next_hidden is not None else None
        else:
            self.hidden_states[battle_tag] = (next_hidden[0].cpu(), next_hidden[1].cpu())

        return action, log_prob, float(values[0, 0].item())

    def _select_action_from_logits(
        self,
        logits: torch.Tensor,
        mask: Optional[np.ndarray],
    ) -> Tuple[int, float]:
        unscaled_log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1).cpu().numpy()

        if mask is not None:
            probs = probs * mask
            if probs.sum() == 0:
                valid_indices = np.flatnonzero(mask)
                if len(valid_indices) == 0:
                    return -1, 0.0
                probs = mask / mask.sum()
            else:
                probs = probs / probs.sum()

            if self.top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                cumulative = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumulative, self.top_p) + 1
                keep = sorted_idx[:cutoff]
                filtered = np.zeros_like(probs)
                filtered[keep] = probs[keep]
                probs = filtered / filtered.sum()

        action = int(np.random.choice(np.arange(len(probs)), p=probs) if self.probabilistic else int(np.argmax(probs)))
        if mask is not None:
            valid_mask = mask.astype(bool)
            log_valid_mass = np.log(np.exp(unscaled_log_probs[valid_mask]).sum())
            log_prob = float(unscaled_log_probs[action] - log_valid_mass)
        else:
            log_prob = float(unscaled_log_probs[action])
        return action, log_prob


class SyncBaselineController:
    def __init__(self, chooser: Any, opponent_type: str):
        self.chooser = chooser
        self.opponent_type = opponent_type

    def choose(self, battle: Any) -> str:
        if battle.teampreview and hasattr(self.chooser, "teampreview"):
            return _normalize_choice_message(str(self.chooser.teampreview(battle)))

        choice = self.chooser.choose_move(battle)
        message = choice if isinstance(choice, str) else getattr(choice, "message", str(choice))
        return _normalize_choice_message(str(message))


class SyncRustBattleDriver:
    def __init__(
        self,
        format_id: str,
        p1_team: str,
        p2_team: str,
        *,
        battle_tag_prefix: str = "rust-sync",
        feature_set: Optional[str] = None,
        collect_rollouts: bool = False,
        seed: Optional[int] = None,
        max_turns_per_battle: int = 100,
        max_stalled_steps_per_battle: int = 25,
        p1_policy: Optional[SyncPolicyPlayer] = None,
        p2_policy: Optional[Any] = None,
        p1_team_supplier: Optional[Callable[[], str]] = None,
        p2_team_supplier: Optional[Callable[[], str]] = None,
        battle_setup_callback: Optional[Callable[[int], Dict[str, Any]]] = None,
        include_binding_snapshots: bool = True,
        diagnostic_log_path: Optional[str] = None,
        error_battle_record_path: Optional[str] = None,
    ):
        from pokemon_showdown_py import RustBattle  # type: ignore[attr-defined]

        self._rust_battle_cls = RustBattle
        self._format_id = format_id
        self._battle_tag_prefix = battle_tag_prefix
        self._p1_team_supplier = p1_team_supplier
        self._p2_team_supplier = p2_team_supplier
        self._p1_team_text = p1_team
        self._p2_team_text = p2_team
        self._p1_team_struct = pokepaste_to_rust_team(p1_team)
        self._p2_team_struct = pokepaste_to_rust_team(p2_team)
        self._p1_team_json = pokepaste_to_rust_team_json(p1_team)
        self._p2_team_json = pokepaste_to_rust_team_json(p2_team)
        self._collect_rollouts = collect_rollouts
        self._max_turns_per_battle = max_turns_per_battle
        self._max_stalled_steps_per_battle = max_stalled_steps_per_battle
        self._p1_policy = p1_policy
        self._p2_policy = p2_policy
        self._battle_setup_callback = battle_setup_callback
        self._include_binding_snapshots = include_binding_snapshots
        self._diagnostic_log_path = diagnostic_log_path
        self._error_battle_record_path = error_battle_record_path
        self._selected_error_battle_tag: Optional[str] = None
        self._error_battle_record_limit = 10
        self._embedder = (
            Embedder(format=format_id, feature_set=feature_set or Embedder.SIMPLE)
            if collect_rollouts
            else None
        )
        self._rng = np.random.default_rng(seed)

    def _embed_battle_state(self, battle: Any) -> np.ndarray:
        if self._embedder is None:
            raise RuntimeError("Embedder is not configured for this driver")

        embed_to_array = getattr(self._embedder, "embed_to_array", None)
        if callable(embed_to_array):
            return cast(np.ndarray, embed_to_array(battle))
        return np.asarray(self._embedder.embed_to_vector(battle), dtype=np.float32)

    @staticmethod
    def _all_teampreview_legal_actions() -> List[Tuple[int, str]]:
        return [
            (action, _normalize_choice_message(MDBO.from_int(action, type=MDBO.TEAMPREVIEW).message))
            for action in range(MDBO.teampreview_space())
        ]

    @staticmethod
    def _apply_teampreview_choice_to_team(
        team: List[Dict[str, object]],
        choice: str,
    ) -> List[Dict[str, object]]:
        digits = [int(char) for char in choice if char.isdigit()]
        if len(digits) != 4:
            return [dict(member) for member in team]

        ordered_indices = [digit - 1 for digit in digits]
        used_indices = set(ordered_indices)
        reordered = [dict(team[index]) for index in ordered_indices]
        reordered.extend(dict(member) for index, member in enumerate(team) if index not in used_indices)
        return reordered

    @staticmethod
    def _build_teampreview_request(team: List[Dict[str, object]], side: str) -> Dict[str, Any]:
        pokemon_entries: List[Dict[str, Any]] = []
        for request_index, member in enumerate(team):
            species = str(member.get("species", member.get("name", f"Pokemon{request_index + 1}")))
            name = str(member.get("name", species))
            raw_level = member.get("level", 50)
            level = int(raw_level) if isinstance(raw_level, (int, float, str)) else 50
            gender = str(member.get("gender", "") or "")
            gender_suffix = f", {gender}" if gender else ""
            pokemon_entries.append(
                {
                    "ident": f"{side}: {name}",
                    "details": f"{species}, L{level}{gender_suffix}",
                    "active": False,
                    "_request_index": request_index,
                }
            )
        return {
            "teamPreview": True,
            "side": {
                "id": side,
                "pokemon": pokemon_entries,
            },
        }

    def _build_teampreview_battle(
        self,
        *,
        battle_tag: str,
        perspective: str,
        player_username: str,
        opponent_username: str,
        team: List[Dict[str, object]],
        opponent_team: List[Dict[str, object]],
    ):
        battle = create_standalone_double_battle(
            battle_tag=battle_tag,
            player_username=player_username,
            opponent_username=opponent_username,
            perspective=perspective,
            team=team,
        )
        battle._teampreview = True
        battle._teampreview_opponent_team = [mon.to_pokemon() for mon in team_from_json(opponent_team)]
        return battle

    def _choose_teampreview_action(
        self,
        *,
        battle_tag: str,
        side: str,
        battle: Any,
        request: Dict[str, Any],
        policy: Optional[Any],
        opponent_type: str,
    ) -> Dict[str, Any]:
        legal_actions = self._all_teampreview_legal_actions()
        state_vector: Any = None
        if self._embedder is not None:
            try:
                state_vector = self._embed_battle_state(battle)
            except Exception:
                state_vector = None

        snapshot = BattleSnapshot(
            battle_tag=battle_tag if side == "p1" else battle_tag + ":p2",
            side=side,
            battle=battle,
            request=request,
            legal_actions=legal_actions,
            action_mask=None,
            action_to_choice={},
            is_teampreview=True,
            opponent_fainted=0,
            state_vector=state_vector,
            binding_snapshot=None,
        )

        if isinstance(policy, SyncPolicyPlayer):
            choice, rollout_step = policy.choose_action_from_snapshot(
                snapshot,
                opponent_type=opponent_type if side == "p1" else None,
            )
            return {
                "choice": choice,
                "fallback_choice": legal_actions[0][1],
                "rollout_step": rollout_step,
                "snapshot": snapshot,
            }

        if isinstance(policy, SyncBaselineController):
            choice = policy.choose(battle)
            return {
                "choice": choice,
                "fallback_choice": choice,
                "rollout_step": None,
                "snapshot": snapshot,
            }

        action, choice, _ = self._sample_action_and_mask(snapshot)
        return {
            "choice": choice,
            "fallback_choice": legal_actions[0][1],
            "rollout_step": RolloutStep(
                state=snapshot.state_vector,
                action=action,
                log_prob=0.0,
                value=0.0,
                mask=None,
                is_teampreview=True,
                opponent_fainted=0,
            ),
            "snapshot": snapshot,
        }

    def _resolve_synthetic_teampreview(
        self,
        *,
        battle_tag: str,
        p1_team_struct: List[Dict[str, object]],
        p2_team_struct: List[Dict[str, object]],
        p1_policy: Optional[Any],
        p2_policy: Optional[Any],
        opponent_type: str,
    ) -> Dict[str, Any]:
        p1_battle = self._build_teampreview_battle(
            battle_tag=battle_tag,
            perspective="p1",
            player_username="EliteFurretAI-p1",
            opponent_username="EliteFurretAI-p2",
            team=p1_team_struct,
            opponent_team=p2_team_struct,
        )
        p2_battle = self._build_teampreview_battle(
            battle_tag=battle_tag,
            perspective="p2",
            player_username="EliteFurretAI-p2",
            opponent_username="EliteFurretAI-p1",
            team=p2_team_struct,
            opponent_team=p1_team_struct,
        )
        p1_request = self._build_teampreview_request(p1_team_struct, "p1")
        p2_request = self._build_teampreview_request(p2_team_struct, "p2")
        p1_step = self._choose_teampreview_action(
            battle_tag=battle_tag,
            side="p1",
            battle=p1_battle,
            request=p1_request,
            policy=p1_policy,
            opponent_type=opponent_type,
        )
        p2_step = self._choose_teampreview_action(
            battle_tag=battle_tag,
            side="p2",
            battle=p2_battle,
            request=p2_request,
            policy=p2_policy,
            opponent_type=opponent_type,
        )

        trace = [
            {
                "event_type": "decision_window",
                "turn_before": 0,
                "stalled_steps_before": 0,
                "p1": {
                    "battle_tag": battle_tag,
                    "side": "p1",
                    "request_type": "teamPreview",
                    "sanitized_request": p1_request,
                    "raw_request": None,
                    "legal_choice_count": len(p1_step["snapshot"].legal_actions),
                    "legal_choices": [choice for _, choice in p1_step["snapshot"].legal_actions],
                    "battle_state_poke_env": self._battle_state_to_string(p1_battle),
                    "battle_state_request": self._request_state_to_string(p1_request),
                    "protocol_log_length_before": 0,
                    "protocol_history": [],
                },
                "p2": {
                    "battle_tag": battle_tag + ":p2",
                    "side": "p2",
                    "request_type": "teamPreview",
                    "sanitized_request": p2_request,
                    "raw_request": None,
                    "legal_choice_count": len(p2_step["snapshot"].legal_actions),
                    "legal_choices": [choice for _, choice in p2_step["snapshot"].legal_actions],
                    "battle_state_poke_env": self._battle_state_to_string(p2_battle),
                    "battle_state_request": self._request_state_to_string(p2_request),
                    "protocol_log_length_before": 0,
                    "protocol_history": [],
                },
                "submitted_choices": {
                    "p1": p1_step["choice"],
                    "p2": p2_step["choice"],
                },
                "fallback_choices": {
                    "p1": p1_step["fallback_choice"],
                    "p2": p2_step["fallback_choice"],
                },
            },
            {
                "event_type": "decision_result",
                "turn_before": 0,
                "turn_after": 0,
                "initial_acceptance": {"p1": True, "p2": True},
                "final_acceptance": {"p1": True, "p2": True},
                "fallback_recovered": {"p1": False, "p2": False},
                "protocol_log_lengths": {"p1": 0, "p2": 0},
            },
        ]

        return {
            "p1_choice": p1_step["choice"],
            "p2_choice": p2_step["choice"],
            "p1_team_struct": self._apply_teampreview_choice_to_team(p1_team_struct, p1_step["choice"]),
            "p2_team_struct": self._apply_teampreview_choice_to_team(p2_team_struct, p2_step["choice"]),
            "p1_rollout_step": p1_step["rollout_step"],
            "trace": trace,
        }

    def run(self, total_battles: int, max_concurrent: int = 1) -> SyncDriverStats:
        if total_battles <= 0:
            raise ValueError("total_battles must be positive")
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")

        active_battles: List[Dict[str, Any]] = []
        next_battle_index = 0
        completed_battles = 0
        p1_wins = 0
        truncated_battles = 0
        turn_limit_truncations = 0
        stalled_limit_truncations = 0
        dual_limit_truncations = 0
        p1_decisions = 0
        rollout_steps = 0
        p1_rejected_choices = 0
        p2_rejected_choices = 0
        p1_fallback_recoveries = 0
        p2_fallback_recoveries = 0
        p1_unrecovered_rejections = 0
        p2_unrecovered_rejections = 0
        driver_profile = SyncDriverProfile()
        tracked_policies: Dict[int, SyncPolicyPlayer] = {}

        start_time = time.perf_counter()
        diagnostic_handle = self._open_diagnostic_log()

        try:
            while next_battle_index < total_battles and len(active_battles) < max_concurrent:
                battle_setup_start = time.perf_counter()
                active_battles.append(self._new_active_battle(next_battle_index))
                driver_profile.battle_setup_seconds += time.perf_counter() - battle_setup_start
                next_battle_index += 1

            while active_battles:
                next_active: List[Dict[str, Any]] = []
                pending_actions: List[Dict[str, Any]] = []
                batched_policy_requests: Dict[SyncPolicyPlayer, List[Dict[str, Any]]] = {}
                for active in active_battles:
                    engine: RustBattleEngine = active["engine"]
                    turn_limit_hit = engine.turn > self._max_turns_per_battle
                    stalled_limit_hit = active["stalled_steps"] > self._max_stalled_steps_per_battle
                    truncated = turn_limit_hit or stalled_limit_hit
                    if engine.ended or truncated:
                        p1_won = bool(engine.p1_battle.won) if engine.ended else False
                        p1_policy = active.get("p1_policy")
                        p2_policy = active.get("p2_policy")
                        finalize_start = time.perf_counter()
                        if p1_policy is not None:
                            p1_policy.finish_battle(engine.battle_tag, p1_won, truncated=truncated and not engine.ended)
                        else:
                            self._finalize_trajectory(active["trajectory"], p1_won)
                        if isinstance(p2_policy, SyncPolicyPlayer):
                            p2_policy.finish_battle(engine.battle_tag + ":p2", not p1_won if engine.ended else False, truncated=truncated and not engine.ended)
                        completed_battles += 1
                        if truncated and not engine.ended:
                            truncated_battles += 1
                            if turn_limit_hit and stalled_limit_hit:
                                dual_limit_truncations += 1
                            elif turn_limit_hit:
                                turn_limit_truncations += 1
                            else:
                                stalled_limit_truncations += 1
                            self._emit_diagnostic_event(
                                diagnostic_handle,
                                {
                                    "event_type": "battle_truncated",
                                    "battle_tag": engine.battle_tag,
                                    "turn": engine.turn,
                                    "turn_limit_hit": turn_limit_hit,
                                    "stalled_limit_hit": stalled_limit_hit,
                                    "stalled_steps": active["stalled_steps"],
                                    "max_turns_per_battle": self._max_turns_per_battle,
                                    "max_stalled_steps_per_battle": self._max_stalled_steps_per_battle,
                                    "p1_request_type": self._engine_request_type(engine, "p1"),
                                    "p2_request_type": self._engine_request_type(engine, "p2"),
                                    "p1_request": self._engine_request_json(engine, "p1"),
                                    "p2_request": self._engine_request_json(engine, "p2"),
                                },
                            )
                        self._finalize_error_battle_record(
                            active,
                            engine=engine,
                            truncated=truncated and not engine.ended,
                            p1_won=p1_won,
                        )
                        driver_profile.trajectory_finalize_seconds += time.perf_counter() - finalize_start
                        if p1_won:
                            p1_wins += 1
                        rollout_steps += len(active["trajectory"])
                        if next_battle_index < total_battles:
                            battle_setup_start = time.perf_counter()
                            next_active.append(self._new_active_battle(next_battle_index))
                            driver_profile.battle_setup_seconds += time.perf_counter() - battle_setup_start
                            next_battle_index += 1
                        continue

                    pending: Dict[str, Any] = {
                        "active": active,
                        "p1_choice": None,
                        "p2_choice": None,
                        "p1_fallback_choice": None,
                        "p2_fallback_choice": None,
                        "p1_snapshot": None,
                        "p2_snapshot": None,
                    }

                    if engine.needs_action("p1"):
                        p1_decisions += 1
                        p1_policy = active.get("p1_policy")
                        if isinstance(p1_policy, SyncPolicyPlayer):
                            tracked_policies[id(p1_policy)] = p1_policy
                            snapshot_start = time.perf_counter()
                            snapshot = self._build_battle_snapshot(engine, "p1")
                            driver_profile.snapshot_build_seconds += time.perf_counter() - snapshot_start
                            pending["p1_snapshot"] = snapshot
                            batched_policy_requests.setdefault(p1_policy, []).append(
                                {
                                    "pending": pending,
                                    "side": "p1",
                                    "snapshot": snapshot,
                                    "opponent_type": active.get("opponent_type", "self_play"),
                                }
                            )
                        else:
                            choice_start = time.perf_counter()
                            step = self._build_choice_context(
                                engine,
                                side="p1",
                                policy=p1_policy,
                                opponent_type=active.get("opponent_type", "self_play"),
                            )
                            driver_profile.baseline_choice_seconds += time.perf_counter() - choice_start
                            pending["p1_choice"] = step["choice"]
                            pending["p1_fallback_choice"] = step.get("fallback_choice")
                            pending["p1_snapshot"] = step.get("snapshot")
                            if self._collect_rollouts and step.get("rollout_step") is not None:
                                active["trajectory"].append(step["rollout_step"])

                    if engine.needs_action("p2"):
                        p2_policy = active.get("p2_policy")
                        if isinstance(p2_policy, SyncPolicyPlayer):
                            tracked_policies[id(p2_policy)] = p2_policy
                            snapshot_start = time.perf_counter()
                            snapshot = self._build_battle_snapshot(engine, "p2")
                            driver_profile.snapshot_build_seconds += time.perf_counter() - snapshot_start
                            pending["p2_snapshot"] = snapshot
                            batched_policy_requests.setdefault(p2_policy, []).append(
                                {
                                    "pending": pending,
                                    "side": "p2",
                                    "snapshot": snapshot,
                                    "opponent_type": None,
                                }
                            )
                        else:
                            choice_start = time.perf_counter()
                            p2_choice, p2_fallback_choice, p2_battle_snapshot = self._sample_choice(
                                engine,
                                side="p2",
                                policy=p2_policy,
                            )
                            driver_profile.baseline_choice_seconds += time.perf_counter() - choice_start
                            pending["p2_choice"] = p2_choice
                            pending["p2_fallback_choice"] = p2_fallback_choice
                            pending["p2_snapshot"] = p2_battle_snapshot

                    pending_actions.append(pending)

                for policy, requests in batched_policy_requests.items():
                    tracked_policies[id(policy)] = policy
                    batched_policy_start = time.perf_counter()
                    decisions = policy.choose_actions_from_snapshots(
                        [cast(BattleSnapshot, item["snapshot"]) for item in requests],
                        [cast(Optional[str], item["opponent_type"]) for item in requests],
                    )
                    driver_profile.batched_policy_seconds += time.perf_counter() - batched_policy_start
                    for request, (choice, rollout_step) in zip(requests, decisions):
                        pending = cast(Dict[str, Any], request["pending"])
                        side = cast(str, request["side"])
                        snapshot = cast(BattleSnapshot, request["snapshot"])
                        fallback_choice = snapshot.legal_actions[0][1] if snapshot.legal_actions else "default"
                        if side == "p1":
                            pending["p1_choice"] = choice
                            pending["p1_fallback_choice"] = fallback_choice
                            if self._collect_rollouts and rollout_step is not None:
                                active_entry = cast(Dict[str, Any], pending["active"])
                                cast(List[RolloutStep], active_entry["trajectory"]).append(rollout_step)
                        else:
                            pending["p2_choice"] = choice
                            pending["p2_fallback_choice"] = fallback_choice

                for pending in pending_actions:
                    active = cast(Dict[str, Any], pending["active"])
                    engine = cast(RustBattleEngine, active["engine"])
                    submit_p1_choice = cast(Optional[str], pending["p1_choice"])
                    submit_p2_choice = cast(Optional[str], pending["p2_choice"])
                    p1_fallback_choice = cast(Optional[str], pending["p1_fallback_choice"])
                    p2_fallback_choice = cast(Optional[str], pending["p2_fallback_choice"])
                    p1_snapshot: Optional[BattleSnapshot] = cast(Optional[BattleSnapshot], pending["p1_snapshot"])
                    p2_pending_snapshot: Optional[BattleSnapshot] = cast(Optional[BattleSnapshot], pending["p2_snapshot"])

                    if submit_p1_choice is None and submit_p2_choice is None:
                        raise RuntimeError(
                            f"Battle {engine.battle_tag} is active but neither side has a legal action request"
                        )

                    previous_turn = engine.turn
                    p1_protocol_log_before_step = self._engine_protocol_log(engine, "p1")
                    p2_protocol_log_before_step = self._engine_protocol_log(engine, "p2")
                    selected_error_battle_tag = getattr(self, "_selected_error_battle_tag", None)
                    if selected_error_battle_tag is None or selected_error_battle_tag == engine.battle_tag:
                        self._append_battle_trace_entry(
                            active,
                            {
                                "event_type": "decision_window",
                                "turn_before": previous_turn,
                                "stalled_steps_before": active["stalled_steps"],
                                "p1": self._build_trace_side_state(engine, p1_snapshot),
                                "p2": self._build_trace_side_state(engine, p2_pending_snapshot),
                                "submitted_choices": {
                                    "p1": submit_p1_choice,
                                    "p2": submit_p2_choice,
                                },
                                "fallback_choices": {
                                    "p1": p1_fallback_choice,
                                    "p2": p2_fallback_choice,
                                },
                            },
                        )
                    step_start = time.perf_counter()
                    step_result = cast(
                        Tuple[bool, bool, bool],
                        engine.step(p1_choice=submit_p1_choice, p2_choice=submit_p2_choice),
                    )
                    _, p1_accepted, p2_accepted = step_result
                    p1_initially_accepted = p1_accepted
                    p2_initially_accepted = p2_accepted
                    if not p1_initially_accepted:
                        p1_rejected_choices += 1
                        self._select_error_battle(engine.battle_tag)
                    if not p2_initially_accepted:
                        p2_rejected_choices += 1
                        self._select_error_battle(engine.battle_tag)
                    if not p1_initially_accepted and p1_fallback_choice is not None and p1_fallback_choice != submit_p1_choice:
                        fallback_result = cast(Tuple[bool, bool, bool], engine.step(p1_choice=p1_fallback_choice))
                        _, p1_accepted, _ = fallback_result
                        if p1_accepted:
                            p1_fallback_recoveries += 1
                    if not p1_accepted:
                        p1_unrecovered_rejections += 1
                    if not p2_initially_accepted and p2_fallback_choice is not None and p2_fallback_choice != submit_p2_choice:
                        fallback_result = cast(Tuple[bool, bool, bool], engine.step(p2_choice=p2_fallback_choice))
                        _, _, p2_accepted = fallback_result
                        if p2_accepted:
                            p2_fallback_recoveries += 1
                    driver_profile.engine_step_seconds += time.perf_counter() - step_start
                    if not p2_accepted:
                        p2_unrecovered_rejections += 1
                    p1_protocol_log_after_step = self._engine_protocol_log(engine, "p1")
                    p2_protocol_log_after_step = self._engine_protocol_log(engine, "p2")
                    if getattr(self, "_selected_error_battle_tag", None) == engine.battle_tag:
                        self._append_battle_trace_entry(
                            active,
                            {
                                "event_type": "decision_result",
                                "turn_before": previous_turn,
                                "turn_after": engine.turn,
                                "initial_acceptance": {
                                    "p1": p1_initially_accepted,
                                    "p2": p2_initially_accepted,
                                },
                                "final_acceptance": {
                                    "p1": p1_accepted,
                                    "p2": p2_accepted,
                                },
                                "fallback_recovered": {
                                    "p1": (not p1_initially_accepted) and bool(p1_accepted),
                                    "p2": (not p2_initially_accepted) and bool(p2_accepted),
                                },
                                "protocol_log_lengths": {
                                    "p1": len(self._engine_protocol_log(engine, "p1")),
                                    "p2": len(self._engine_protocol_log(engine, "p2")),
                                },
                            },
                        )
                    if not p1_initially_accepted:
                        self._log_rejected_choice(
                            diagnostic_handle,
                            engine=engine,
                            side="p1",
                            snapshot=p1_snapshot,
                            submitted_choice=submit_p1_choice,
                            fallback_choice=p1_fallback_choice,
                            fallback_recovered=bool(p1_accepted),
                            turn_before=previous_turn,
                            protocol_log_before_step=p1_protocol_log_before_step,
                            protocol_log_after_step=p1_protocol_log_after_step,
                        )
                    if not p2_initially_accepted:
                        self._log_rejected_choice(
                            diagnostic_handle,
                            engine=engine,
                            side="p2",
                            snapshot=p2_pending_snapshot,
                            submitted_choice=submit_p2_choice,
                            fallback_choice=p2_fallback_choice,
                            fallback_recovered=bool(p2_accepted),
                            turn_before=previous_turn,
                            protocol_log_before_step=p2_protocol_log_before_step,
                            protocol_log_after_step=p2_protocol_log_after_step,
                        )
                    if engine.turn <= active["last_turn"]:
                        active["stalled_steps"] += 1
                    else:
                        active["last_turn"] = engine.turn
                        active["stalled_steps"] = 0
                    next_active.append(active)

                active_battles = next_active
        finally:
            if diagnostic_handle is not None:
                diagnostic_handle.close()

        duration_seconds = time.perf_counter() - start_time
        for policy in tracked_policies.values():
            driver_profile.policy_embed_seconds += policy.profile.embed_seconds
            driver_profile.policy_inference_seconds += policy.profile.inference_seconds
            driver_profile.policy_action_decode_seconds += policy.profile.action_decode_seconds
            driver_profile.policy_rollout_record_seconds += policy.profile.rollout_record_seconds
        return SyncDriverStats(
            completed_battles=completed_battles,
            p1_wins=p1_wins,
            truncated_battles=truncated_battles,
            turn_limit_truncations=turn_limit_truncations,
            stalled_limit_truncations=stalled_limit_truncations,
            dual_limit_truncations=dual_limit_truncations,
            duration_seconds=duration_seconds,
            p1_decisions=p1_decisions,
            rollout_steps=rollout_steps,
            p1_rejected_choices=p1_rejected_choices,
            p2_rejected_choices=p2_rejected_choices,
            p1_fallback_recoveries=p1_fallback_recoveries,
            p2_fallback_recoveries=p2_fallback_recoveries,
            p1_unrecovered_rejections=p1_unrecovered_rejections,
            p2_unrecovered_rejections=p2_unrecovered_rejections,
            profile=driver_profile,
        )

    def _new_active_battle(self, battle_index: int) -> Dict[str, Any]:
        battle_setup = self._battle_setup_callback(battle_index) if self._battle_setup_callback is not None else {}
        p1_team_text = battle_setup.get("p1_team_text")
        if p1_team_text is None:
            p1_team_text = self._p1_team_supplier() if self._p1_team_supplier is not None else self._p1_team_text
        p2_team_text = battle_setup.get("p2_team_text")
        if p2_team_text is None:
            p2_team_text = self._p2_team_supplier() if self._p2_team_supplier is not None else self._p2_team_text
        p1_team_struct = pokepaste_to_rust_team(p1_team_text)
        p2_team_struct = pokepaste_to_rust_team(p2_team_text)
        battle_tag = f"{self._battle_tag_prefix}-{battle_index}"
        p1_policy = battle_setup.get("p1_policy", self._p1_policy)
        p2_policy = battle_setup.get("p2_policy", self._p2_policy)
        opponent_type = battle_setup.get("opponent_type", "self_play")
        teampreview = self._resolve_synthetic_teampreview(
            battle_tag=battle_tag,
            p1_team_struct=p1_team_struct,
            p2_team_struct=p2_team_struct,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            opponent_type=opponent_type,
        )
        p1_team_struct = teampreview["p1_team_struct"]
        p2_team_struct = teampreview["p2_team_struct"]
        p1_team_json = json.dumps(p1_team_struct)
        p2_team_json = json.dumps(p2_team_struct)
        rust_battle = adapt_rust_battle(
            self._rust_battle_cls(
                format_id=self._format_id,
                p1_team_json=p1_team_json,
                p2_team_json=p2_team_json,
                seed=self._battle_seed(),
                p1_name="EliteFurretAI-p1",
                p2_name="EliteFurretAI-p2",
            ),
        )
        engine = RustBattleEngine(
            rust_battle=rust_battle,
            battle_tag=battle_tag,
            p1_username="EliteFurretAI-p1",
            p2_username="EliteFurretAI-p2",
            p1_team=p1_team_struct,
            p2_team=p2_team_struct,
        )
        return {
            "engine": engine,
            "trajectory": [teampreview["p1_rollout_step"]] if teampreview["p1_rollout_step"] is not None else [],
            "last_turn": engine.turn,
            "stalled_steps": 0,
            "p1_policy": p1_policy,
            "p2_policy": p2_policy,
            "opponent_type": opponent_type,
            "battle_trace": list(teampreview["trace"]),
        }

    def _battle_seed(self) -> List[int]:
        return [int(value) for value in self._rng.integers(0, 2**32 - 1, size=4)]

    def _build_choice_context(
        self,
        engine: RustBattleEngine,
        side: str,
        policy: Optional[SyncPolicyPlayer] = None,
        opponent_type: str = "self_play",
    ) -> Dict[str, Any]:
        snapshot = self._build_battle_snapshot(engine, side)
        if policy is not None:
            choice, rollout_step = policy.choose_action_from_snapshot(
                snapshot=snapshot,
                opponent_type=opponent_type if side == "p1" else None,
            )
            fallback_choice = snapshot.legal_actions[0][1] if snapshot.legal_actions else "default"
            return {
                "choice": choice,
                "fallback_choice": fallback_choice,
                "rollout_step": rollout_step,
                "snapshot": snapshot,
            }

        action, choice, mask = self._sample_action_and_mask(snapshot)
        return {
            "choice": choice,
            "fallback_choice": snapshot.legal_actions[0][1] if snapshot.legal_actions else "default",
            "snapshot": snapshot,
            "rollout_step": RolloutStep(
                state=snapshot.state_vector,
                action=action,
                log_prob=0.0,
                value=0.0,
                mask=mask,
                is_teampreview=snapshot.is_teampreview,
                opponent_fainted=snapshot.opponent_fainted,
            ),
        }

    def _build_battle_snapshot(self, engine: RustBattleEngine, side: str) -> BattleSnapshot:
        battle = engine.battle_for(side)
        side_snapshot = engine.side_snapshot(side) if self._include_binding_snapshots else None
        request = (side_snapshot.request if side_snapshot is not None else engine.request_json(side)) or {}
        legal_pairs = (
            self._all_teampreview_legal_actions()
            if battle.teampreview
            else self._enumerate_valid_action_pairs(battle, request)
        )
        action_mask: Optional[np.ndarray] = None
        action_to_choice: Dict[int, str] = {}
        if not battle.teampreview and legal_pairs:
            action_mask = np.zeros(MDBO.action_space(), dtype=np.int8)
            for action, choice in legal_pairs:
                if action < 0:
                    continue
                action_mask[action] = 1
                action_to_choice.setdefault(action, choice)

        state_vector: Any = None
        if self._embedder is not None:
            try:
                state_vector = self._embed_battle_state(battle)
            except Exception:
                state_vector = None

        return BattleSnapshot(
            battle_tag=engine.battle_tag if side == "p1" else engine.battle_tag + ":p2",
            side=side,
            battle=battle,
            request=request,
            legal_actions=legal_pairs,
            action_mask=action_mask,
            action_to_choice=action_to_choice,
            is_teampreview=bool(battle.teampreview),
            opponent_fainted=sum(1 for mon in battle.opponent_team.values() if mon.fainted),
            state_vector=state_vector,
            binding_snapshot=side_snapshot,
        )

    def _sample_choice(
        self,
        engine: RustBattleEngine,
        side: str,
        policy: Optional[Any] = None,
    ) -> Tuple[str, Optional[str], BattleSnapshot]:
        if isinstance(policy, SyncPolicyPlayer):
            step = self._build_choice_context(engine, side, policy=policy)
            fallback_choice = cast(Optional[str], step.get("fallback_choice"))
            return cast(str, step["choice"]), fallback_choice, cast(BattleSnapshot, step["snapshot"])
        if isinstance(policy, SyncBaselineController):
            snapshot = self._build_battle_snapshot(engine, side)
            choice = policy.choose(engine.battle_for(side))
            return choice, choice, snapshot
        snapshot = self._build_battle_snapshot(engine, side)
        _, choice, _ = self._sample_action_and_mask(snapshot)
        fallback_choice = cast(Optional[str], snapshot.legal_actions[0][1] if snapshot.legal_actions else "default")
        return choice, fallback_choice, snapshot

    def _sample_action_and_mask(
        self,
        snapshot: BattleSnapshot,
    ) -> Tuple[int, str, Optional[np.ndarray]]:
        if snapshot.is_teampreview:
            action = int(self._rng.integers(MDBO.teampreview_space()))
            message = MDBO.from_int(action, type=MDBO.TEAMPREVIEW).message
            return action, _normalize_choice_message(message), None

        if not snapshot.request:
            raise RuntimeError("Battle request is missing; cannot generate a legal action")

        if not snapshot.legal_actions:
            return -1, "default", None

        choice_index = int(self._rng.integers(len(snapshot.legal_actions)))
        action, message = snapshot.legal_actions[choice_index]
        action = int(action)
        message = cast(str, message)
        mask = snapshot.action_mask
        if mask is None:
            mask = np.zeros(MDBO.action_space(), dtype=np.int8)
            for valid_action, _ in snapshot.legal_actions:
                mask[valid_action] = 1
        return action, message, mask

    def _enumerate_valid_action_pairs(
        self,
        battle,
        request: Dict[str, Any],
    ) -> List[Tuple[int, str]]:
        slot0_actions = self._get_slot_actions(battle, request, slot=0)
        slot1_actions = self._get_slot_actions(battle, request, slot=1)
        valid_pairs: List[Tuple[int, str]] = []

        for action0 in slot0_actions:
            for action1 in slot1_actions:
                if action0["tera"] and action1["tera"]:
                    continue
                if (
                    action0["switch_target"] is not None
                    and action1["switch_target"] is not None
                    and action0["switch_target"] == action1["switch_target"]
                ):
                    continue
                if action0["encoded"] == PASS_ACTION and action1["encoded"] == PASS_ACTION:
                    continue

                action_idx = action0["encoded"] * ACTIONS_PER_SLOT + action1["encoded"]
                if action_idx >= MDBO.action_space():
                    continue
                valid_pairs.append(
                    (action_idx, f"{action0['choice']}, {action1['choice']}")
                )

        return valid_pairs

    def _get_slot_actions(
        self,
        battle,
        request: Dict[str, Any],
        *,
        slot: int,
    ) -> List[Dict[str, Any]]:
        def pass_only_action() -> List[Dict[str, Any]]:
            return [
                {
                    "encoded": PASS_ACTION,
                    "choice": "pass",
                    "tera": False,
                    "switch_target": None,
                }
            ]

        slot_actions: List[Dict[str, Any]] = []
        force_switch = request.get("forceSwitch", [False, False])
        any_force_switch = any(bool(value) for value in force_switch)
        force_switch_required = bool(slot < len(force_switch) and force_switch[slot])
        forced_slot_count = sum(bool(value) for value in force_switch)

        if any_force_switch and not force_switch_required:
            return pass_only_action()

        if force_switch_required:
            switch_actions = self._get_switch_actions(request)
            slot_actions.extend(switch_actions)
            allow_pass = len(switch_actions) < max(1, forced_slot_count)
            if not switch_actions or allow_pass:
                slot_actions.append(
                    {
                        "encoded": PASS_ACTION,
                        "choice": "pass",
                        "tera": False,
                        "switch_target": None,
                    }
                )
            return slot_actions

        active_requests = request.get("active") or []
        if slot >= len(active_requests) or battle.active_pokemon[slot] is None:
            return pass_only_action()

        side = request.get("side") or {}
        pokemon_list = side.get("pokemon") or []
        active_side_pokemon = [pokemon for pokemon in pokemon_list if pokemon.get("active")]
        active_side_entry = active_side_pokemon[slot] if slot < len(active_side_pokemon) else None
        if isinstance(active_side_entry, dict) and bool(active_side_entry.get("commanding")):
            return pass_only_action()

        active_request = active_requests[slot]
        moves = active_request.get("moves") or []
        can_tera = active_request.get("canTerastallize") is not None
        for move_idx, move_info in enumerate(moves):
            if move_info.get("disabled"):
                continue
            if move_info.get("pp", 1) <= 0:
                continue
            targets = get_valid_targets_for_request_move(battle, slot, move_info)
            for target in targets:
                target_offset = TARGET_TO_OFFSET.get(target, 2)
                target_suffix = "" if target == 0 else f" {target}"
                slot_actions.append(
                    {
                        "encoded": move_idx * 10 + target_offset,
                        "choice": f"move {move_idx + 1}{target_suffix}",
                        "tera": False,
                        "switch_target": None,
                    }
                )
                if can_tera:
                    slot_actions.append(
                        {
                            "encoded": move_idx * 10 + target_offset + 5,
                            "choice": f"move {move_idx + 1}{target_suffix} terastallize",
                            "tera": True,
                            "switch_target": None,
                        }
                    )

        if not active_request.get("trapped"):
            slot_actions.extend(self._get_switch_actions(request))

        if not slot_actions:
            slot_actions.extend(pass_only_action())

        return slot_actions

    def _get_switch_actions(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        switch_actions: List[Dict[str, Any]] = []
        side = request.get("side") or {}
        pokemon_list = side.get("pokemon") or []
        bench_position = 0
        for request_index, pokemon in enumerate(pokemon_list):
            if pokemon.get("active"):
                continue
            condition = pokemon.get("condition", "")
            if not isinstance(condition, str) or "fnt" in condition:
                continue
            if bench_position >= 4:
                break
            original_request_index = pokemon.get("_request_index", request_index)
            if not isinstance(original_request_index, int):
                original_request_index = request_index
            switch_actions.append(
                {
                    "encoded": SWITCH_ACTION_BASE + bench_position,
                    "choice": f"switch {original_request_index + 1}",
                    "tera": False,
                    "switch_target": original_request_index,
                }
            )
            bench_position += 1
        return switch_actions

    @staticmethod
    def _slot_is_fainted(request: Dict[str, Any], slot: int) -> bool:
        side = request.get("side") or {}
        pokemon_list = side.get("pokemon") or []
        active_pokemon = [pokemon for pokemon in pokemon_list if pokemon.get("active")]
        if slot >= len(active_pokemon):
            return True
        condition = active_pokemon[slot].get("condition", "")
        return isinstance(condition, str) and "fnt" in condition

    def _open_diagnostic_log(self):
        diagnostic_log_path = getattr(self, "_diagnostic_log_path", None)
        if not diagnostic_log_path:
            return None
        path = Path(diagnostic_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")

    @staticmethod
    def _emit_diagnostic_event(handle, event: Dict[str, Any]) -> None:
        if handle is None:
            return
        handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _log_rejected_choice(
        self,
        handle,
        *,
        engine: RustBattleEngine,
        side: str,
        snapshot: Optional[BattleSnapshot],
        submitted_choice: Optional[str],
        fallback_choice: Optional[str],
        fallback_recovered: bool,
        turn_before: int,
        protocol_log_before_step: Sequence[Dict[str, str]],
        protocol_log_after_step: Sequence[Dict[str, str]],
    ) -> None:
        request = snapshot.request if snapshot is not None else (self._engine_request_json(engine, side) or {})
        legal_actions = snapshot.legal_actions if snapshot is not None else []
        battle = snapshot.battle if snapshot is not None else self._engine_battle_for(engine, side)
        raw_request = snapshot.binding_snapshot.raw_request if snapshot is not None and snapshot.binding_snapshot is not None else None
        request_type = self._request_type_from_request(request)
        if request_type is None:
            request_type = self._engine_request_type(engine, side)
        self._emit_diagnostic_event(
            handle,
            {
                "event_type": "rejected_choice",
                "battle_tag": engine.battle_tag,
                "side": side,
                "turn_before": turn_before,
                "turn_after": engine.turn,
                "request_type": request_type,
                "choice": submitted_choice,
                "choice_kind": self._classify_choice(submitted_choice),
                "choice_slot_kinds": self._split_choice_kinds(submitted_choice),
                "choice_action_index": self._action_index_for_choice(legal_actions, submitted_choice),
                "choice_in_legal_actions": self._choice_in_legal_actions(legal_actions, submitted_choice),
                "fallback_choice": fallback_choice,
                "fallback_kind": self._classify_choice(fallback_choice),
                "fallback_slot_kinds": self._split_choice_kinds(fallback_choice),
                "fallback_in_legal_actions": self._choice_in_legal_actions(legal_actions, fallback_choice),
                "fallback_recovered": fallback_recovered,
                "legal_choice_count": len(legal_actions),
                "legal_choices": [choice for _, choice in legal_actions],
                "legal_choice_preview": [choice for _, choice in legal_actions[:8]],
                "force_switch": request.get("forceSwitch"),
                "active_trapped": [
                    bool(active.get("trapped")) for active in (request.get("active") or []) if isinstance(active, dict)
                ],
                "can_tera": [
                    active.get("canTerastallize") is not None
                    for active in (request.get("active") or [])
                    if isinstance(active, dict)
                ],
                "protocol_history": self._engine_protocol_history(engine, side, limit=5),
                "protocol_log_before_step": [dict(entry) for entry in protocol_log_before_step],
                "protocol_log_after_step": [dict(entry) for entry in protocol_log_after_step],
                "stalled_steps": None,
                "battle_state_poke_env": self._battle_state_to_string(battle),
                "battle_state_request": self._request_state_to_string(request),
                "raw_request": raw_request,
            },
        )

    @staticmethod
    def _battle_state_to_string(battle: Any) -> str:
        if battle is None:
            return "<battle state unavailable>"

        def mon_label(mon: Any) -> str:
            if mon is None:
                return "None"
            species = getattr(mon, "species", "<unknown>")
            return f"{species} [active={getattr(mon, 'active', None)}, fainted={getattr(mon, 'fainted', None)}]"

        def team_lines(team: Dict[str, Any]) -> List[str]:
            lines: List[str] = []
            for ident, mon in team.items():
                lines.append(
                    f"  - {ident}: {getattr(mon, 'species', '<unknown>')} [active={getattr(mon, 'active', None)}, fainted={getattr(mon, 'fainted', None)}]"
                )
            return lines

        def preview_lines(prefix: str, team: Sequence[Any]) -> List[str]:
            lines: List[str] = []
            for index, mon in enumerate(team, start=1):
                species = getattr(mon, "species", getattr(mon, "name", f"mon-{index}"))
                lines.append(f"  - {prefix}{index}: {species} [active={getattr(mon, 'active', False)}, fainted={getattr(mon, 'fainted', False)}]")
            return lines

        if bool(getattr(battle, "teampreview", False)):
            return "\n".join(
                [
                    f"Battle {getattr(battle, 'battle_tag', '<unknown battle>')} turn={getattr(battle, 'turn', '<unknown turn>')}",
                    f"Perspective: {getattr(battle, 'player_username', '<unknown player>')} vs {getattr(battle, 'opponent_username', '<unknown opponent>')}",
                    "My Teampreview Team:",
                    *preview_lines("slot ", getattr(battle, "teampreview_team", []) or []),
                    "Opp Teampreview Team:",
                    *preview_lines("slot ", getattr(battle, "teampreview_opponent_team", []) or []),
                ]
            )

        weather = ", ".join(effect.name for effect in getattr(battle, "weather", {})) or "None"
        fields = ", ".join(effect.name for effect in getattr(battle, "fields", {})) or "None"
        side_conditions = ", ".join(effect.name for effect in getattr(battle, "side_conditions", {})) or "None"
        opp_side_conditions = ", ".join(effect.name for effect in getattr(battle, "opponent_side_conditions", {})) or "None"
        active_pokemon = getattr(battle, "active_pokemon", []) or []
        opponent_active_pokemon = getattr(battle, "opponent_active_pokemon", []) or []
        lines = [
            f"Battle {getattr(battle, 'battle_tag', '<unknown battle>')} turn={getattr(battle, 'turn', '<unknown turn>')}",
            f"Perspective: {getattr(battle, 'player_username', '<unknown player>')} vs {getattr(battle, 'opponent_username', '<unknown opponent>')}",
            f"My Active: [{', '.join(mon_label(mon) for mon in active_pokemon)}]",
            f"Opp Active: [{', '.join(mon_label(mon) for mon in opponent_active_pokemon)}]",
            f"Weather: [{weather}]",
            f"Fields: [{fields}]",
            f"My Side Conditions: [{side_conditions}]",
            f"Opp Side Conditions: [{opp_side_conditions}]",
            f"Force Switch: {getattr(battle, 'force_switch', None)}",
            f"Trapped: {getattr(battle, 'trapped', None)}",
            f"Can Tera: {getattr(battle, 'can_tera', None)}",
            "My Team:",
            *team_lines(getattr(battle, "team", {})),
            "Opp Team:",
            *team_lines(getattr(battle, "opponent_team", {})),
        ]
        return "\n".join(lines)

    @staticmethod
    def _request_state_to_string(request: Dict[str, Any]) -> str:
        if not request:
            return "<no request>"

        lines = [
            f"Request type: {SyncRustBattleDriver._request_type_from_request(request)}",
            f"wait={request.get('wait', False)} teamPreview={request.get('teamPreview', False)} forceSwitch={request.get('forceSwitch')}",
        ]

        active_entries = request.get("active") or []
        if isinstance(active_entries, list):
            lines.append("Active payloads:")
            for index, active in enumerate(active_entries):
                if not isinstance(active, dict):
                    lines.append(f"  - slot {index}: <non-dict payload>")
                    continue
                move_parts = []
                for move in active.get("moves", []):
                    if not isinstance(move, dict):
                        continue
                    move_parts.append(f"{move.get('id')}[target={move.get('target')!r}]")
                lines.append(
                    f"  - slot {index}: trapped={active.get('trapped', False)} canTera={active.get('canTerastallize')} moves=[{', '.join(move_parts)}]"
                )

        side = request.get("side") or {}
        pokemon_list = side.get("pokemon") or []
        if isinstance(pokemon_list, list):
            lines.append("Side pokemon:")
            for index, pokemon in enumerate(pokemon_list):
                if not isinstance(pokemon, dict):
                    lines.append(f"  - index {index}: <non-dict pokemon>")
                    continue
                lines.append(
                    "  - index {index}: {ident} active={active} request_index={request_index}".format(
                        index=index,
                        ident=pokemon.get("ident"),
                        active=pokemon.get("active"),
                        request_index=pokemon.get("_request_index"),
                    )
                )

        return "\n".join(lines)

    def _select_error_battle(self, battle_tag: str) -> None:
        error_battle_record_path = getattr(self, "_error_battle_record_path", None)
        if not error_battle_record_path:
            return

        path = Path(error_battle_record_path)
        if path.suffix:
            if getattr(self, "_selected_error_battle_tag", None) is None:
                self._selected_error_battle_tag = battle_tag
            return

        selected_tags = getattr(self, "_selected_error_battle_tags", None)
        if selected_tags is None:
            selected_tags = set()
            self._selected_error_battle_tags = selected_tags
        record_limit = getattr(self, "_error_battle_record_limit", 10)
        if battle_tag not in selected_tags and len(selected_tags) < record_limit:
            selected_tags.add(battle_tag)

    def _append_battle_trace_entry(self, active: Dict[str, Any], event: Dict[str, Any]) -> None:
        if not getattr(self, "_error_battle_record_path", None):
            return
        active.setdefault("battle_trace", []).append(event)

    def _build_trace_side_state(
        self,
        engine: RustBattleEngine,
        snapshot: Optional[BattleSnapshot],
    ) -> Optional[Dict[str, Any]]:
        if snapshot is None:
            return None
        binding_snapshot = snapshot.binding_snapshot
        protocol_log_length_before = len(self._engine_protocol_log(engine, snapshot.side))
        return {
            "battle_tag": snapshot.battle_tag,
            "side": snapshot.side,
            "request_type": self._request_type_from_request(snapshot.request),
            "sanitized_request": snapshot.request,
            "raw_request": binding_snapshot.raw_request if binding_snapshot is not None else None,
            "legal_choice_count": len(snapshot.legal_actions),
            "legal_choices": [choice for _, choice in snapshot.legal_actions],
            "battle_state_poke_env": self._battle_state_to_string(snapshot.battle),
            "battle_state_request": self._request_state_to_string(snapshot.request),
            "protocol_log_length_before": protocol_log_length_before,
            "protocol_history": binding_snapshot.pending_messages if binding_snapshot is not None else None,
        }

    def _finalize_error_battle_record(
        self,
        active: Dict[str, Any],
        *,
        engine: RustBattleEngine,
        truncated: bool,
        p1_won: bool,
    ) -> None:
        error_battle_record_path = getattr(self, "_error_battle_record_path", None)
        if not error_battle_record_path:
            return
        path = Path(error_battle_record_path)
        if path.suffix:
            if getattr(self, "_selected_error_battle_tag", None) != engine.battle_tag:
                return
        else:
            selected_tags: set[str] = getattr(self, "_selected_error_battle_tags", set())
            if engine.battle_tag not in selected_tags:
                return
            path = path / f"{engine.battle_tag}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "battle_tag": engine.battle_tag,
            "format_id": self._format_id,
            "completed": True,
            "truncated": truncated,
            "winner": engine.winner,
            "p1_won": p1_won,
            "final_turn": engine.turn,
            "final_request_types": {
                "p1": self._engine_request_type(engine, "p1"),
                "p2": self._engine_request_type(engine, "p2"),
            },
            "final_requests": {
                "p1": self._engine_request_json(engine, "p1"),
                "p2": self._engine_request_json(engine, "p2"),
            },
            "trace": active.get("battle_trace", []),
            "protocol_log": {
                "p1": self._engine_protocol_log(engine, "p1"),
                "p2": self._engine_protocol_log(engine, "p2"),
            },
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _classify_choice(choice: Optional[str]) -> Optional[str]:
        if choice is None:
            return None
        if choice.startswith("move "):
            return "move"
        if choice.startswith("switch "):
            return "switch"
        if choice == "pass":
            return "pass"
        if choice.startswith("team "):
            return "team"
        if choice == "default":
            return "default"
        return "other"

    def _split_choice_kinds(self, choice: Optional[str]) -> List[Optional[str]]:
        if choice is None:
            return []
        return [self._classify_choice(part.strip()) for part in choice.split(",")]

    @staticmethod
    def _choice_in_legal_actions(legal_actions: Sequence[Tuple[int, str]], choice: Optional[str]) -> bool:
        if choice is None:
            return False
        return any(legal_choice == choice for _, legal_choice in legal_actions)

    @staticmethod
    def _action_index_for_choice(
        legal_actions: Sequence[Tuple[int, str]],
        choice: Optional[str],
    ) -> Optional[int]:
        if choice is None:
            return None
        for action_index, legal_choice in legal_actions:
            if legal_choice == choice:
                return action_index
        return None

    @staticmethod
    def _engine_request_type(engine: Any, side: str) -> Optional[str]:
        request_type = getattr(engine, "request_type", None)
        if request_type is None:
            return None
        return cast(Optional[str], request_type(side))

    @staticmethod
    def _engine_request_json(engine: Any, side: str) -> Optional[Dict[str, Any]]:
        request_json = getattr(engine, "request_json", None)
        if request_json is None:
            return None
        return cast(Optional[Dict[str, Any]], request_json(side))

    @staticmethod
    def _engine_battle_for(engine: Any, side: str) -> Any:
        battle_for = getattr(engine, "battle_for", None)
        if battle_for is None:
            return None
        return battle_for(side)

    @staticmethod
    def _engine_protocol_history(engine: Any, side: str, limit: int = 5) -> List[Dict[str, str]]:
        protocol_history = getattr(engine, "protocol_history", None)
        if protocol_history is None:
            return []
        return cast(List[Dict[str, str]], protocol_history(side, limit))

    @staticmethod
    def _engine_protocol_log(engine: Any, side: str) -> List[Dict[str, str]]:
        protocol_log = getattr(engine, "protocol_log", None)
        if protocol_log is None:
            return []
        return cast(List[Dict[str, str]], protocol_log(side))

    @staticmethod
    def _request_type_from_request(request: Optional[Dict[str, Any]]) -> Optional[str]:
        if not request:
            return None
        if request.get("wait"):
            return "wait"
        if request.get("teamPreview"):
            return "teamPreview"
        force_switch = request.get("forceSwitch")
        if isinstance(force_switch, list) and any(force_switch):
            return "switch"
        if "active" in request:
            return "move"
        return None

    @staticmethod
    def _finalize_trajectory(trajectory: Sequence[RolloutStep], won: bool) -> None:
        for index, step in enumerate(trajectory):
            step_reward = -0.005
            if index == len(trajectory) - 1:
                step_reward += 1.0 if won else -1.0
            prev_fainted = trajectory[index - 1].opponent_fainted if index > 0 else 0
            ko_delta = step.opponent_fainted - prev_fainted
            if ko_delta > 0:
                step_reward += 0.05 * ko_delta
            step.reward = step_reward

    def consume_completed_trajectories(self) -> List[Dict[str, Any]]:
        if self._p1_policy is None:
            return []
        return self._p1_policy.consume_completed_trajectories()
