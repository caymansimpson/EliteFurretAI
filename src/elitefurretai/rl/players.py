import asyncio
import concurrent.futures
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.calc import calculate_damage
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.data import GenData
from poke_env.player import BattleOrder, DoubleBattleOrder, Player
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.fast_action_mask import fast_get_action_mask
from elitefurretai.supervised.model_archs import (
    FlexibleThreeHeadedModel,
    TransformerThreeHeadedModel,
)

logger = logging.getLogger("MaxDamagePlayer")


# Registry of worker-specific executors: worker_id -> ThreadPoolExecutor
_WORKER_EXECUTORS: Dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = Lock()

# Global fallback executor for non-worker contexts (e.g., testing)
_FALLBACK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="fallback_inference"
)


def get_worker_executor(worker_id: Optional[int] = None) -> ThreadPoolExecutor:
    """Get or create a ThreadPoolExecutor for a specific worker."""
    if worker_id is None:
        return _FALLBACK_EXECUTOR

    with _EXECUTOR_LOCK:
        if worker_id not in _WORKER_EXECUTORS:
            _WORKER_EXECUTORS[worker_id] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"worker{worker_id}_inference",
            )
        return _WORKER_EXECUTORS[worker_id]


def cleanup_worker_executors() -> None:
    """Shutdown all worker executors (call at training end)."""
    with _EXECUTOR_LOCK:
        for executor in _WORKER_EXECUTORS.values():
            executor.shutdown(wait=False)
        _WORKER_EXECUTORS.clear()


class BatchInferencePlayer(Player):
    """High-performance player with async batched inference."""

    def __init__(
        self,
        model: "RNaDAgent",
        device="cpu",
        batch_size=16,
        batch_timeout=0.01,
        probabilistic=True,
        trajectory_queue=None,
        accept_open_team_sheet=False,
        worker_id: Optional[int] = None,
        embedder: Optional[Embedder] = None,
        max_battle_steps: int = 40,
        opponent_type: str = "self_play",
        **kwargs,
    ):
        battle_format = kwargs.get("battle_format", "gen9vgc2023regc")

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.probabilistic = probabilistic
        self.trajectory_queue = trajectory_queue
        self.worker_id = worker_id
        self.opponent_type = opponent_type
        self.embedder = (
            embedder
            if embedder is not None
            else Embedder(format=battle_format, feature_set=Embedder.FULL, omniscient=False)
        )
        self.queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue)
        self.hidden_states: Dict[str, Any] = {}  # (h,c) for LSTM or context tensor for Transformer
        self._is_transformer: bool = isinstance(
            model.model if isinstance(model, RNaDAgent) else model,
            TransformerThreeHeadedModel,
        )
        self._inference_task: Optional[asyncio.Task] = None
        self._inference_future: Optional[concurrent.futures.Future] = None
        self.temperature: float = 1.0  # Sampling temperature (set by trainer)
        self.top_p: float = 1.0  # Nucleus sampling threshold (set by trainer)
        self.current_trajectories: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self.completed_trajectories: List[Dict[str, Any]] = []
        self._discarded_battles: set[str] = set()
        # Per-battle request generation counter. Incremented on each battle
        # request handler invocation to suppress stale async sends.
        self._request_generation: Dict[str, int] = {}
        self.max_battle_steps = max_battle_steps
        # Timeout for waiting on a batched inference response.
        # Why: if the inference loop stalls, we prefer fallback behavior over hanging
        # a battle coroutine indefinitely.
        self.inference_request_timeout_s = 8.0

        super().__init__(accept_open_team_sheet=accept_open_team_sheet, **kwargs)

    def clear_completed_trajectories(self) -> None:
        self.completed_trajectories.clear()

    async def stop_listening(self):
        await self.ps_client.stop_listening()

    def stop_inference_loop(self, timeout_s: float = 1.0) -> None:
        """Stop the background inference loop if it is running.

        Why: during worker-side rebuilds, we must stop old loop tasks before creating
        fresh players, or stale loops can keep references alive and leak pending work.
        """
        if self._inference_future is None:
            return

        # First request cooperative cancellation.
        self._inference_future.cancel()

        # Then drain completion briefly so the loop has a chance to unwind cleanly.
        try:
            self._inference_future.result(timeout=timeout_s)
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            pass
        except Exception:
            pass
        finally:
            self._inference_future = None

    def teardown_runtime(self, timeout_s: float = 1.5) -> None:
        """Best-effort teardown of inference + websocket listener state.

        Why: this is the explicit teardown step needed before rebuilding agents.
        Without it, old clients can remain logged in, causing `|nametaken|` collisions
        and repeated timeout loops after a desync event.
        """
        # Stop inference first so we don't enqueue decisions while disconnecting.
        self.stop_inference_loop(timeout_s=timeout_s)

        # Ask poke-env to stop websocket listening on this player.
        try:
            fut = asyncio.run_coroutine_threadsafe(self.stop_listening(), POKE_LOOP)
            fut.result(timeout=timeout_s)
        except Exception:
            pass

    def start_inference_loop(self) -> None:
        self._inference_future = asyncio.run_coroutine_threadsafe(
            self._inference_loop(), POKE_LOOP
        )

    async def _inference_loop(self):
        while True:
            batch: List[Any] = []
            futures: List[Any] = []
            battle_tags: List[str] = []
            is_tps: List[bool] = []
            masks: List[Any] = []
            try:
                item = await self.queue.get()
                self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)

                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        self._add_to_batch(
                            batch, futures, battle_tags, is_tps, masks, item
                        )
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break

            if batch:
                await self._run_batch(batch, futures, battle_tags, is_tps, masks)

    def _add_to_batch(self, batch, futures, battle_tags, is_tps, masks, item):
        batch.append(item[0])
        futures.append(item[1])
        battle_tags.append(item[2])
        is_tps.append(item[3])
        masks.append(item[4])

    def _gpu_inference_sync(self, states_np, hidden_cpu):
        states_tensor = (
            torch.tensor(states_np, dtype=torch.float32).to(self.device).unsqueeze(1)
        )

        if self._is_transformer:
            # Transformer: hidden_cpu is context tensor or None
            hidden = hidden_cpu.to(self.device) if hidden_cpu is not None else None
        else:
            # LSTM: hidden_cpu is (h, c)
            hidden = (hidden_cpu[0].to(self.device), hidden_cpu[1].to(self.device))

        with torch.no_grad():
            turn_logits, tp_logits, values, _, next_hidden = self.model(states_tensor, hidden)

        # Temperature-scaled probs for action SELECTION (exploration)
        temp = max(self.temperature, 1e-6)
        turn_probs = torch.softmax(turn_logits / temp, dim=-1).cpu().numpy()
        tp_probs = torch.softmax(tp_logits / temp, dim=-1).cpu().numpy()

        # Unscaled (T=1) log-probs for PPO importance ratio computation
        # Critical: log_probs must come from the policy distribution, not the
        # temperature-scaled sampling distribution, to avoid biasing PPO ratios.
        turn_log_probs = torch.log_softmax(turn_logits, dim=-1).cpu().numpy()
        tp_log_probs = torch.log_softmax(tp_logits, dim=-1).cpu().numpy()

        values_np = values.cpu().numpy()

        if self._is_transformer:
            # next_hidden is context tensor (batch, T, H)
            next_hidden_cpu = next_hidden.cpu()
        else:
            # next_hidden is (h, c)
            next_hidden_cpu = (next_hidden[0].cpu(), next_hidden[1].cpu())

        return turn_probs, tp_probs, turn_log_probs, tp_log_probs, values_np, next_hidden_cpu

    async def _run_batch(self, states, futures, battle_tags, is_tps, masks):
        states_np = np.array(states)

        if self._is_transformer:
            # Transformer: hidden is a context tensor per battle, or None.
            # We need to handle variable-length contexts.  For simplicity,
            # we set hidden to None for each battle (the model handles it)
            # and store per-battle contexts separately.  For batched inference
            # we process each battle individually since contexts may differ in length.
            # TODO: pad contexts for true batched Transformer inference.
            all_results: List[Dict[str, Any]] = []
            for i, tag in enumerate(battle_tags):
                single_state = states_np[i : i + 1]
                ctx = self.hidden_states.get(tag, None)
                loop = asyncio.get_running_loop()
                executor = get_worker_executor(self.worker_id)
                (
                    turn_probs,
                    tp_probs,
                    turn_log_probs,
                    tp_log_probs,
                    values,
                    next_ctx,
                ) = await loop.run_in_executor(
                    executor,
                    self._gpu_inference_sync,
                    single_state,
                    ctx,
                )
                self.hidden_states[tag] = next_ctx
                all_results.append({
                    "turn_probs": turn_probs[0, 0],
                    "tp_probs": tp_probs[0, 0],
                    "turn_log_probs": turn_log_probs[0, 0],
                    "tp_log_probs": tp_log_probs[0, 0],
                    "value": values[0, 0],
                })

            for i, future in enumerate(futures):
                is_tp = is_tps[i]
                mask = masks[i]
                r = all_results[i]

                if is_tp:
                    probs = r["tp_probs"]
                    unscaled_log_probs = r["tp_log_probs"]
                    valid_actions = list(range(len(probs)))
                else:
                    probs = r["turn_probs"]
                    unscaled_log_probs = r["turn_log_probs"]
                    if mask is not None:
                        probs = probs * mask
                        if probs.sum() == 0:
                            probs = mask / mask.sum()
                        else:
                            probs = probs / probs.sum()

                        if self.top_p < 1.0:
                            sorted_idx = np.argsort(-probs)
                            cum = np.cumsum(probs[sorted_idx])
                            cutoff = np.searchsorted(cum, self.top_p) + 1
                            keep = sorted_idx[:cutoff]
                            filtered = np.zeros_like(probs)
                            filtered[keep] = probs[keep]
                            probs = filtered / filtered.sum()

                    valid_actions = list(range(len(probs)))

                action = (
                    np.random.choice(valid_actions, p=probs)
                    if self.probabilistic
                    else np.argmax(probs)
                )

                # Compute log_prob from the MASKED distribution so PPO
                # ratios are consistent with the learner (which also masks).
                if mask is not None:
                    valid_mask = mask.astype(bool)
                    log_valid_mass = np.log(
                        np.exp(unscaled_log_probs[valid_mask]).sum()
                    )
                    log_prob = float(
                        unscaled_log_probs[action] - log_valid_mass
                    )
                else:
                    log_prob = float(unscaled_log_probs[action])

                future.set_result({
                    "action": action,
                    "log_prob": log_prob,
                    "value": r["value"],
                    "probs": probs,
                })
            return

        # ---- LSTM path (original batched inference) ----
        h_list = []
        c_list = []
        for tag in battle_tags:
            if tag not in self.hidden_states:
                init_state = self.model.get_initial_state(1, "cpu")
                assert init_state is not None  # LSTM always returns (h, c)
                h, c = init_state
                self.hidden_states[tag] = (h, c)
            h_list.append(self.hidden_states[tag][0])
            c_list.append(self.hidden_states[tag][1])

        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)

        loop = asyncio.get_running_loop()
        executor = get_worker_executor(self.worker_id)
        (
            turn_probs,
            tp_probs,
            turn_log_probs,
            tp_log_probs,
            values,
            next_hidden_cpu,
        ) = await loop.run_in_executor(
            executor,
            self._gpu_inference_sync,
            states_np,
            (h_batch, c_batch),
        )

        h_next_cpu, c_next_cpu = next_hidden_cpu
        for i, tag in enumerate(battle_tags):
            self.hidden_states[tag] = (
                h_next_cpu[:, i : i + 1, :],
                c_next_cpu[:, i : i + 1, :],
            )

        for i, future in enumerate(futures):
            is_tp = is_tps[i]
            mask = masks[i]

            if is_tp:
                probs = tp_probs[i, 0]
                unscaled_log_probs = tp_log_probs[i, 0]
                valid_actions = list(range(len(probs)))
            else:
                probs = turn_probs[i, 0]
                unscaled_log_probs = turn_log_probs[i, 0]
                if mask is not None:
                    probs = probs * mask
                    if probs.sum() == 0:
                        probs = mask / mask.sum()
                    else:
                        probs = probs / probs.sum()

                    # Top-p (nucleus) filtering: zero out the tail of the
                    # temperature-scaled sampling distribution so that only
                    # the smallest set of actions whose cumulative probability
                    # exceeds top_p are kept.
                    if self.top_p < 1.0:
                        sorted_idx = np.argsort(-probs)
                        cum = np.cumsum(probs[sorted_idx])
                        cutoff = np.searchsorted(cum, self.top_p) + 1
                        keep = sorted_idx[:cutoff]
                        filtered = np.zeros_like(probs)
                        filtered[keep] = probs[keep]
                        probs = filtered / filtered.sum()

                valid_actions = list(range(len(probs)))

            action = (
                np.random.choice(valid_actions, p=probs)
                if self.probabilistic
                else np.argmax(probs)
            )

            # Use the UNSCALED (T=1) log-prob for PPO importance ratios.
            # The temperature-scaled probs only determine which action is
            # *selected*, but the policy gradient must use the true policy
            # distribution to avoid biased ratio estimates.
            # Apply mask correction so old_log_prob matches the learner's
            # masked Categorical distribution.
            if mask is not None:
                valid_mask = mask.astype(bool)
                log_valid_mass = np.log(
                    np.exp(unscaled_log_probs[valid_mask]).sum()
                )
                log_prob = float(unscaled_log_probs[action] - log_valid_mass)
            else:
                log_prob = float(unscaled_log_probs[action])

            future.set_result(
                {
                    "action": action,
                    "log_prob": log_prob,
                    "value": values[i, 0],
                    "probs": probs,
                }
            )

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):

        # Defensive guard: do not attempt to send orders for battles that are already
        # marked finished locally.
        if getattr(battle, "finished", False):
            return

        request_generation = self._request_generation.get(battle.battle_tag, 0) + 1
        self._request_generation[battle.battle_tag] = request_generation

        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
                logger.warning(
                    "DEFAULT_SEND_FAILURE "
                    f"tag={battle.battle_tag} turn={getattr(battle, 'turn', '?')} "
                    f"message={message} err={repr(exc)}"
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self.hidden_states.pop(battle.battle_tag, None)
            return

        choice = await self._choose_move_async(
            battle,
            request_generation=request_generation,
        )

        # Drop if a newer request superseded this one while inference/order
        # computation was in flight.
        if self._request_generation.get(battle.battle_tag, -1) != request_generation:
            return

        # Drop if battle finished before send.
        if getattr(battle, "finished", False):
            return

        if isinstance(choice, str):
            message = choice
        elif hasattr(choice, "message"):
            message = choice.message
        else:
            message = str(choice)

        # Root-cause guardrail #1: TODO fix
        # If battle requires a forced switch but the chosen message is a move command,
        # rewrite to default order to avoid guaranteed invalid-choice errors.
        if (
            isinstance(battle, DoubleBattle)
            and any(battle.force_switch)
            and isinstance(message, str)
            and message.startswith("/choose move")
        ):
            message = self.choose_default_move().message

        # Root-cause guardrail #2:
        # If no active slot can tera, strip accidental tera directive from message.
        if (
            isinstance(battle, DoubleBattle)
            and isinstance(message, str)
            and "terastallize" in message
            and hasattr(battle, "can_tera")
            and not any(getattr(battle, "can_tera", []))
        ):
            message = message.replace(" terastallize", "")

        if message:
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
                # Rich context to diagnose first trigger root causes:
                # - which battle tag failed
                # - what message we attempted
                # - legal move/switch context where available
                active_names = []
                legal_move_names = []
                legal_switch_names = []
                if isinstance(battle, DoubleBattle):
                    active_names = [p.species for p in battle.active_pokemon if p is not None]
                    for moves in battle.available_moves:
                        legal_move_names.append([m.id for m in moves])
                    for switches in battle.available_switches:
                        legal_switch_names.append([p.species for p in switches])

                logger.warning(
                    "SEND_FAILURE "
                    f"tag={battle.battle_tag} turn={getattr(battle, 'turn', '?')} "
                    f"teampreview={getattr(battle, 'teampreview', False)} "
                    f"force_switch={getattr(battle, 'force_switch', None)} "
                    f"message={message} active={active_names} legal_moves={legal_move_names} "
                    f"legal_switches={legal_switch_names} err={repr(exc)}"
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self.hidden_states.pop(battle.battle_tag, None)

    def choose_move(self, battle: AbstractBattle) -> Any:
        return self._choose_move_async(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        raise RuntimeError(
            "teampreview() should not be called; decisions are routed through choose_move()"
        )

    async def _choose_move_async(
        self,
        battle: AbstractBattle,
        request_generation: Optional[int] = None,
    ):
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()

        current_steps = len(self.current_trajectories.get(battle.battle_tag, []))
        if current_steps >= self.max_battle_steps:
            self._discarded_battles.add(battle.battle_tag)
            self.current_trajectories.pop(battle.battle_tag, None)
            self.hidden_states.pop(battle.battle_tag, None)
            return "/forfeit" if self.trajectory_queue is not None else DefaultBattleOrder()

        try:
            state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))
        except Exception:
            if battle.teampreview:
                return "/team 1234"
            try:
                return self.choose_random_doubles_move(battle)
            except Exception:
                return DefaultBattleOrder()
        mask = None if battle.teampreview else fast_get_action_mask(battle)

        # Snapshot request-time battle state so we can detect stale inference outputs.
        request_turn = getattr(battle, "turn", -1)
        request_teampreview = battle.teampreview
        request_force_switch = (
            tuple(bool(x) for x in battle.force_switch)
            if isinstance(battle, DoubleBattle)
            else tuple()
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((state, future, battle.battle_tag, battle.teampreview, mask))
        try:
            result = await asyncio.wait_for(future, timeout=self.inference_request_timeout_s)
        except asyncio.TimeoutError:
            logger.debug(
                "INFERENCE_TIMEOUT tag=%s turn=%s teampreview=%s queue_size=%s",
                battle.battle_tag, getattr(battle, 'turn', '?'),
                battle.teampreview,
                self.queue.qsize() if hasattr(self.queue, 'qsize') else '?',
            )
            self.current_trajectories.pop(battle.battle_tag, None)
            self.hidden_states.pop(battle.battle_tag, None)
            return DefaultBattleOrder()

        # Request-generation guard: if a newer request is already active for this
        # battle tag, drop this stale result before decoding/sending.
        if request_generation is not None:
            latest_generation = self._request_generation.get(battle.battle_tag, -1)
            if latest_generation != request_generation:
                logger.debug(
                    "STALE_REQUEST_GENERATION_DROP tag=%s request_gen=%d latest_gen=%d",
                    battle.battle_tag, request_generation, latest_generation,
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self.hidden_states.pop(battle.battle_tag, None)
                return DefaultBattleOrder()

        action_idx = result["action"]

        # Root-cause guardrail #3:
        # If battle state changed while waiting on batched inference, discard this
        # decision instead of sending a potentially invalid/stale command.
        current_turn = getattr(battle, "turn", -1)
        current_teampreview = battle.teampreview
        current_force_switch = (
            tuple(bool(x) for x in battle.force_switch)
            if isinstance(battle, DoubleBattle)
            else tuple()
        )
        if (
            current_turn != request_turn
            or current_teampreview != request_teampreview
            or current_force_switch != request_force_switch
        ):
            logger.debug(
                "STALE_INFERENCE_DROP tag=%s turn=%d->%d tp=%s->%s fs=%s->%s",
                battle.battle_tag, request_turn, current_turn,
                request_teampreview, current_teampreview,
                request_force_switch, current_force_switch,
            )
            self.current_trajectories.pop(battle.battle_tag, None)
            self.hidden_states.pop(battle.battle_tag, None)
            return DefaultBattleOrder()

        if mask is not None and action_idx < len(mask) and mask[action_idx] == 0:
            logger.warning(
                "MASK_MISMATCH tag=%s turn=%s action_idx=%d mask_value=%s",
                battle.battle_tag, getattr(battle, 'turn', '?'),
                action_idx, mask[action_idx],
            )

        if self.trajectory_queue is not None:
            self.current_trajectories.setdefault(battle.battle_tag, []).append(
                {
                    "state": state,
                    "action": action_idx,
                    "log_prob": result["log_prob"],
                    "value": result["value"],
                    "reward": 0,
                    "is_teampreview": battle.teampreview,
                    "mask": mask,
                    "opponent_fainted": sum(
                        1 for m in battle.opponent_team.values() if m.fainted
                    ),
                }
            )
        else:
            self.current_trajectories.setdefault(battle.battle_tag, []).append(None)

        if battle.teampreview:
            try:
                return MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW).message
            except (ValueError, AssertionError):
                return "/team 123456"

        try:
            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            mdbo = MDBO.from_int(action_idx, type=action_type)
            return mdbo.to_double_battle_order(battle)
        except (ValueError, KeyError, AttributeError, IndexError, AssertionError):
            return DefaultBattleOrder()

    def _battle_finished_callback(self, battle: AbstractBattle):
        self._request_generation.pop(battle.battle_tag, None)

        if battle.battle_tag in self._discarded_battles:
            self._discarded_battles.discard(battle.battle_tag)
            self.current_trajectories.pop(battle.battle_tag, None)
            self.hidden_states.pop(battle.battle_tag, None)
            return

        if self.trajectory_queue is None:
            self.current_trajectories.pop(battle.battle_tag, None)
            self.hidden_states.pop(battle.battle_tag, None)
            return

        if battle.battle_tag in self.current_trajectories:
            traj = self.current_trajectories.pop(battle.battle_tag)
            for t, step in enumerate(traj):
                if step is None:
                    continue
                step_reward = -0.005
                if t == len(traj) - 1:
                    step_reward += 1.0 if battle.won else -1.0
                # KO reward: +0.05 for each opponent pokemon KO'd this step
                prev_fainted = 0
                prev_step = traj[t - 1] if t > 0 else None
                if prev_step is not None:
                    prev_fainted = prev_step["opponent_fainted"]
                ko_delta = step["opponent_fainted"] - prev_fainted
                if ko_delta > 0:
                    step_reward += 0.05 * ko_delta
                step["reward"] = step_reward

            filtered_traj = [step for step in traj if step is not None]
            self.trajectory_queue.put(
                {
                    "steps": filtered_traj,
                    "opponent_type": self.opponent_type,
                    "won": battle.won,
                    "battle_length": len(filtered_traj),
                    "forfeited": False,
                }
            )
            self.hidden_states.pop(battle.battle_tag, None)


class RNaDAgent(torch.nn.Module):
    """
    RL Agent wrapper around FlexibleThreeHeadedModel or TransformerThreeHeadedModel.
    Handles hidden states and value function transformation.
    """

    def __init__(self, model: Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel]):
        super().__init__()
        self.model = model
        self._is_transformer = isinstance(model, TransformerThreeHeadedModel)

    def get_initial_state(self, batch_size: int, device: str):
        if self._is_transformer:
            # Transformer has no initial hidden state — context starts as None.
            # Return a sentinel None that players.py checks.
            return None
        assert isinstance(self.model, FlexibleThreeHeadedModel)
        num_directions = 2
        num_layers: int = self.model.lstm.num_layers
        hidden_size: int = self.model.lstm_hidden_size
        h = torch.zeros(
            num_layers * num_directions,
            batch_size,
            hidden_size,
            device=device,
        )
        c = torch.zeros(
            num_layers * num_directions,
            batch_size,
            hidden_size,
            device=device,
        )
        return (h, c)

    def forward(self, x, hidden_state=None, mask=None):
        turn_logits, tp_logits, value, win_dist_logits, next_hidden = (
            self.model.forward_with_hidden(x, hidden_state, mask)
        )
        return turn_logits, tp_logits, value, win_dist_logits, next_hidden


class MaxDamagePlayer(Player):
    def __init__(
        self,
        battle_format: str = "gen9vgc2023regc",
        switch_threshold: float = 1.5,
        temperature: float = 0.5,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, battle_format=battle_format, **kwargs)
        self.switch_threshold = switch_threshold
        self.temperature = temperature
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[MDP] %(message)s"))
                logger.addHandler(handler)

    @staticmethod
    def _estimate_evs_and_nature(
        base_stats: Dict[str, int],
    ) -> Tuple[List[int], str]:
        hp, atk, dfn, spa, spd, spe = (
            base_stats["hp"],
            base_stats["atk"],
            base_stats["def"],
            base_stats["spa"],
            base_stats["spd"],
            base_stats["spe"],
        )

        stat_values = {
            "hp": hp,
            "atk": atk,
            "def": dfn,
            "spa": spa,
            "spd": spd,
            "spe": spe,
        }
        highest = max(stat_values, key=lambda k: stat_values[k])

        evs = [0, 0, 0, 0, 0, 0]

        if highest == "hp":
            evs[0] = 252
            evs[2] = 128
            evs[4] = 128
            nature = "calm"
        elif highest == "atk":
            evs[1] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "adamant"
        elif highest == "def":
            evs[2] = 252
            evs[0] = 252
            nature = "bold"
        elif highest == "spa":
            evs[3] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "modest"
        elif highest == "spd":
            evs[4] = 252
            evs[0] = 252
            nature = "calm"
        else:
            evs[5] = 252
            if atk >= spa:
                evs[1] = 252
                nature = "adamant"
            else:
                evs[3] = 252
                nature = "modest"

        return evs, nature

    def _estimate_opponent_stats(self, battle) -> None:
        gen_data = GenData.from_gen(battle.gen)

        for mon in battle.opponent_team.values():
            if mon.stats.get("hp") is not None:
                continue

            base_stats = mon.base_stats
            if not base_stats:
                continue

            evs, nature = self._estimate_evs_and_nature(base_stats)
            ivs = [31, 31, 31, 31, 31, 31]
            level = mon.level if mon.level else 50

            try:
                raw_stats = compute_raw_stats(
                    mon.species, evs, ivs, level, nature, gen_data
                )
                mon.stats = {
                    "hp": raw_stats[0],
                    "atk": raw_stats[1],
                    "def": raw_stats[2],
                    "spa": raw_stats[3],
                    "spd": raw_stats[4],
                    "spe": raw_stats[5],
                }
            except Exception:
                continue

    def teampreview(self, battle) -> str:  # type: ignore
        self._estimate_opponent_stats(battle)
        return self._select_teampreview_by_damage(battle)

    def _select_teampreview_by_damage(self, battle) -> str:
        my_team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())

        if not my_team or not opponent_team:
            return self.random_teampreview(battle)

        pokemon_total_damages: List[Tuple[Pokemon, float]] = []
        for mon in my_team:
            total_damage = 0.0

            if mon.moves:
                for move in mon.moves.values():
                    if not move or move.base_power == 0:
                        continue

                    for opp_mon in opponent_team:
                        try:
                            damage_range = calculate_damage(
                                mon.identifier(battle.player_role),
                                opp_mon.identifier(battle.opponent_role),
                                move,
                                battle,
                            )
                            if damage_range and damage_range[0] is not None:
                                total_damage += (damage_range[0] + damage_range[1]) / 2.0
                        except Exception:
                            continue

            pokemon_total_damages.append((mon, total_damage))

        pokemon_total_damages.sort(key=lambda x: x[1], reverse=True)

        team_list = list(battle.team.values())
        selected_indices: List[int] = []
        for selected_mon, _ in pokemon_total_damages[:4]:
            for idx, mon in enumerate(team_list):
                if mon.species == selected_mon.species and (idx + 1) not in selected_indices:
                    selected_indices.append(idx + 1)
                    break

        if len(selected_indices) >= 4:
            return "/team " + "".join(str(i) for i in selected_indices)

        return self.random_teampreview(battle)

    def choose_move(self, battle) -> BattleOrder:  # type: ignore
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._estimate_opponent_stats(battle)

        if self.debug:
            active = [m.species if m else "None" for m in battle.active_pokemon]
            opp_active = [m.species if m else "None" for m in battle.opponent_active_pokemon]
            logger.debug(f"Turn {battle.turn}: Active={active} vs Opp={opp_active}")
            for m in battle.opponent_active_pokemon:
                if m:
                    logger.debug(f"  Opp {m.species} stats={m.stats}")

        used_switches: Set[str] = set()
        slot_orders: List[BattleOrder] = []

        for slot in range(2):
            active_mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None

            if slot < len(battle.force_switch) and battle.force_switch[slot]:
                switches = [
                    s for s in battle.available_switches[slot]
                    if s.species not in used_switches
                ]
                if switches:
                    switch_candidates = [
                        (
                            cast(BattleOrder, self.create_order(s)),
                            self._get_best_move_damage(battle, s)[0],
                        )
                        for s in switches
                    ]
                    chosen_order, _ = self._softmax_sample(switch_candidates, self.temperature)
                    slot_orders.append(chosen_order)
                    chosen_payload = self._get_order_payload(chosen_order)
                    if isinstance(chosen_payload, Pokemon):
                        used_switches.add(chosen_payload.species)
                else:
                    slot_orders.append(DefaultBattleOrder())
                continue

            if any(battle.force_switch):
                slot_orders.append(PassBattleOrder())
                continue

            if active_mon is None:
                slot_orders.append(PassBattleOrder())
                continue

            candidates = self._score_available_actions(battle, slot, used_switches)

            if candidates:
                chosen_order, chosen_score = self._softmax_sample(candidates, self.temperature)

                chosen_payload = self._get_order_payload(chosen_order)
                if isinstance(chosen_payload, Pokemon):
                    used_switches.add(chosen_payload.species)

                slot_orders.append(chosen_order)
                if self.debug:
                    if isinstance(chosen_payload, Pokemon):
                        order_desc = f"SWITCH to {chosen_payload.species}"
                    elif chosen_payload is not None and hasattr(chosen_payload, "id"):
                        order_desc = (
                            f"{chosen_payload.id} "
                            f"target={getattr(chosen_order, 'move_target', None)}"
                        )
                    else:
                        order_desc = str(chosen_order)
                    logger.debug(
                        f"  Slot {slot} ({active_mon.species}): {order_desc}"
                        f" score={chosen_score:.0f} (from {len(candidates)} candidates, temp={self.temperature})"
                    )
            else:
                slot_orders.append(DefaultBattleOrder())
                if self.debug:
                    logger.debug(f"  Slot {slot} ({active_mon.species if active_mon else 'None'}): DEFAULT (no valid move or switch)")

        if len(slot_orders) == 2:
            return DoubleBattleOrder(
                first_order=cast(SingleBattleOrder, slot_orders[0]),
                second_order=cast(SingleBattleOrder, slot_orders[1]),
            )
        elif len(slot_orders) == 1:
            return DoubleBattleOrder(first_order=cast(SingleBattleOrder, slot_orders[0]))

        return self.choose_random_doubles_move(battle)

    @staticmethod
    def _softmax_sample(
        candidates: Sequence[Tuple[BattleOrder, float]], temperature: float
    ) -> Tuple[BattleOrder, float]:
        if not candidates:
            raise ValueError("Cannot sample from empty candidates list")
        if len(candidates) == 1:
            return candidates[0]

        scores = [s for _, s in candidates]

        if temperature <= 0:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx]

        max_score = max(scores)
        if max_score > 0:
            normalized = [s / max_score for s in scores]
        else:
            idx = random.randrange(len(candidates))
            return candidates[idx]

        scaled = [s / temperature for s in normalized]
        max_scaled = max(scaled)
        exp_scores = [math.exp(s - max_scaled) for s in scaled]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return candidates[i]

        return candidates[-1]

    def _score_available_actions(
        self, battle: DoubleBattle, slot: int, used_switches: Set[str]
    ) -> List[Tuple[BattleOrder, float]]:
        available_moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        active_mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        candidates: List[Tuple[BattleOrder, float]] = []

        if available_moves and active_mon is not None:
            for move in available_moves:
                targets = battle.get_possible_showdown_targets(move, active_mon)

                for target in targets:
                    if target < 0:
                        continue

                    target_mon = None
                    if target == 1:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[0] if len(opp_active) > 0 else None
                    elif target == 2:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[1] if len(opp_active) > 1 else None
                    elif target == 0:
                        for opp in battle.opponent_active_pokemon:
                            if opp is not None:
                                target_mon = opp
                                break

                    if target_mon is None:
                        continue

                    try:
                        player_role = battle.player_role or "p1"
                        opponent_role = battle.opponent_role or "p2"
                        damage_range = calculate_damage(
                            active_mon.identifier(player_role),
                            target_mon.identifier(opponent_role),
                            move,
                            battle,
                        )
                        if damage_range and damage_range[0] is not None:
                            avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                            if self.debug:
                                logger.debug(
                                    f"    {active_mon.species} {move.id} -> {target_mon.species}"
                                    f" (target={target}): {damage_range[0]}-{damage_range[1]}"
                                    f" (avg={avg_damage:.0f})"
                                )
                            candidates.append(
                                (
                                    cast(BattleOrder, self.create_order(move, move_target=target)),
                                    avg_damage,
                                )
                            )
                    except Exception as e:
                        if self.debug:
                            logger.debug(
                                f"    {active_mon.species} {move.id} -> {target_mon.species}"
                                f" (target={target}): EXCEPTION: {e}"
                            )
                        continue

            if not candidates and available_moves:
                move = available_moves[0]
                targets = battle.get_possible_showdown_targets(move, active_mon)
                opp_targets = [t for t in targets if t > 0]
                target = opp_targets[0] if opp_targets else (0 if 0 in targets else targets[0] if targets else 0)
                candidates.append(
                    (cast(BattleOrder, self.create_order(move, move_target=target)), 0.0)
                )
                if self.debug:
                    logger.debug(
                        f"    {active_mon.species}: FALLBACK {move.id} target={target}"
                        f" (all_targets={targets})"
                    )

        available_switches = [
            s for s in battle.available_switches[slot]
            if s.species not in used_switches
        ]
        for switch_mon in available_switches:
            switch_damage = self._get_best_move_damage(battle, switch_mon)[0]
            switch_score = switch_damage / self.switch_threshold
            candidates.append((cast(BattleOrder, self.create_order(switch_mon)), switch_score))

        return candidates

    def _get_best_available_move(
        self, battle: DoubleBattle, slot: int
    ) -> Tuple[Optional[BattleOrder], float]:
        candidates = self._score_available_actions(battle, slot, set())
        if not candidates:
            return (None, 0.0)
        move_candidates = [
            (order, score)
            for order, score in candidates
            if not isinstance(self._get_order_payload(order), Pokemon)
        ]
        if not move_candidates:
            return (None, 0.0)
        best = max(move_candidates, key=lambda x: x[1])
        return best

    def _get_best_move_damage(
        self, battle, attacker: Pokemon
    ) -> Tuple[float, Optional[str], Optional[int]]:
        if not attacker.moves:
            return (0.0, None, None)

        max_damage = 0.0
        best_move_id: Optional[str] = None
        best_target_idx: Optional[int] = None

        targets = [
            (idx, mon)
            for idx, mon in enumerate(battle.opponent_active_pokemon)
            if mon is not None
        ]
        if not targets:
            return (0.0, None, None)

        for move_id, move in attacker.moves.items():
            if not move or move.current_pp == 0:
                continue

            for idx, target in targets:
                try:
                    damage_range = calculate_damage(
                        attacker.identifier(battle.player_role or "p1"),
                        target.identifier(battle.opponent_role or "p2"),
                        move,
                        battle,
                    )
                    if damage_range and damage_range[0] is not None:
                        avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                        if avg_damage >= max_damage:
                            max_damage = avg_damage
                            best_move_id = move_id
                            best_target_idx = idx
                except Exception:
                    continue

        return (max_damage, best_move_id, best_target_idx)

    @staticmethod
    def _get_order_payload(order: BattleOrder):
        return getattr(order, "order", None)


__all__ = [
    "RNaDAgent",
    "MaxDamagePlayer",
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
