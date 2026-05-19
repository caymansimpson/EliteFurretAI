"""SimpleModelPlayer — eval-time Player wrapping a trained model.

Loads a checkpoint, runs inference inline in ``choose_move`` (no IPC, no
batching). Trades training-time throughput for setup simplicity — callers
don't need to spawn an InferenceService process or wire up
request/response queues. Use ``rl/players.py``'s BatchInferencePlayer
when you need the trainer-side centralized inference pattern.

See agents/AGENTS.md for usage.

Moved here from ``rl/players.py`` on 2026-05-19 as part of the agents/
directory reorganization (see planning/stage2/2026-05-19-09-30-agents-directory-reorg.md).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.masking import fast_get_action_mask
from elitefurretai.rl.players import RNaDAgent


class SimpleModelPlayer(Player):
    """In-process Player wrapping an RNaDAgent. Use for benchmarks/evaluations.

    Loads a checkpoint into a torch.nn.Module and runs inference inline in
    ``choose_move`` (no IPC, no batching). Trades training-time throughput
    for setup simplicity — callers don't need to spawn an InferenceService
    process or wire up request/response queues. Use BatchInferencePlayer
    when you need the trainer-side centralized inference pattern.

    Subclass and override ``_on_action_selected`` to hook in extra behavior
    per turn (e.g. ``VerboseModelPlayer`` prints top-k probabilities).
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        battle_format: str,
        probabilistic: bool = False,
        embedder: Optional[Embedder] = None,
        **player_kwargs: Any,
    ):
        super().__init__(battle_format=battle_format, **player_kwargs)
        # Late import: learners.py imports RNaDAgent from this module,
        # so a top-level import here would be circular.
        from elitefurretai.rl.learners import load_agent_from_checkpoint

        self.agent: RNaDAgent = load_agent_from_checkpoint(model_path, device)
        self.device = device
        self.probabilistic = probabilistic
        self.embedder = embedder or Embedder(
            format=battle_format, feature_set=Embedder.FULL, omniscient=False
        )
        # Hidden state shape is opaque (None for fresh battles; a context
        # tensor for Transformer after the first step). Don't decompose it.
        self.hidden_states: Dict[str, Any] = {}

    def _get_hidden(self, battle_tag: str) -> Any:
        if battle_tag not in self.hidden_states:
            self.hidden_states[battle_tag] = self.agent.get_initial_state(1, self.device)
        return self.hidden_states[battle_tag]

    def _select_action(
        self, battle: DoubleBattle
    ) -> Tuple[int, "np.ndarray[Any, Any]", float, bool]:
        """Run one inference step. Returns (action_idx, probs, value, is_teampreview)."""
        if battle.battle_tag in self.hidden_states and battle.finished:
            del self.hidden_states[battle.battle_tag]

        state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))
        state_tensor = (
            torch.tensor(state, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        hidden = self._get_hidden(battle.battle_tag)

        with torch.no_grad():
            turn_logits, tp_logits, value, _win_dist_logits, next_hidden = self.agent(
                state_tensor, hidden
            )

        self.hidden_states[battle.battle_tag] = next_hidden

        is_teampreview = battle.teampreview
        if is_teampreview:
            probs = torch.softmax(tp_logits[0, 0], dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(turn_logits[0, 0], dim=-1).cpu().numpy()
            mask = fast_get_action_mask(battle)
            probs = probs * mask
            probs = probs / probs.sum() if probs.sum() > 0 else mask / mask.sum()

        selected = (
            int(np.random.choice(np.arange(len(probs)), p=probs))
            if self.probabilistic
            else int(np.argmax(probs))
        )
        return selected, probs, float(value[0, 0].item()), is_teampreview

    def _on_action_selected(
        self,
        battle: DoubleBattle,
        probs: "np.ndarray[Any, Any]",
        value: float,
        selected: int,
        is_teampreview: bool,
    ) -> None:
        """Hook for subclasses (e.g. verbose logging). Default: no-op."""

    def _build_order(
        self, battle: DoubleBattle, selected: int, is_teampreview: bool
    ) -> Any:
        try:
            if is_teampreview:
                return MDBO.from_int(selected, type=MDBO.TEAMPREVIEW).message
            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            return MDBO.from_int(selected, type=action_type).to_double_battle_order(battle)
        except Exception:
            return DefaultBattleOrder()

    def choose_move(self, battle: AbstractBattle) -> Any:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        selected, probs, value, is_teampreview = self._select_action(battle)
        self._on_action_selected(battle, probs, value, selected, is_teampreview)
        return self._build_order(battle, selected, is_teampreview)
