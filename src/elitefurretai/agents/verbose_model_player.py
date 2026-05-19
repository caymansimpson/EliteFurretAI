"""VerboseModelPlayer — SimpleModelPlayer + per-turn top-k logging to stdout.

Use for ad-hoc inspection of model behavior during a battle. Subclasses
SimpleModelPlayer; overrides _on_action_selected to print the top-k
action probabilities at each decision point.

See agents/AGENTS.md for usage.

Moved here from ``rl/players.py`` on 2026-05-19.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.etl.encoder import MDBO
from elitefurretai.inference.inference_utils import battle_to_str


class VerboseModelPlayer(SimpleModelPlayer):
    """SimpleModelPlayer that prints top-k action probabilities each turn."""

    def __init__(
        self,
        model_path: str,
        device: str,
        battle_format: str,
        probabilistic: bool,
        top_k: int,
        print_summary: bool,
        account_configuration: AccountConfiguration,
        server_configuration: ServerConfiguration,
        max_concurrent_battles: int,
        start_timer_on_battle_start: bool,
        team: Optional[str] = None,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            battle_format=battle_format,
            probabilistic=probabilistic,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            accept_open_team_sheet=True,
            max_concurrent_battles=max_concurrent_battles,
            start_timer_on_battle_start=start_timer_on_battle_start,
            team=team,
        )
        self.top_k = top_k
        self.print_summary = print_summary

    def _describe_action(
        self, battle: DoubleBattle, action_idx: int, is_teampreview: bool
    ) -> str:
        try:
            if is_teampreview:
                return MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW).message
            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            mdbo = MDBO.from_int(action_idx, type=action_type)
            order = mdbo.to_double_battle_order(battle)
            if hasattr(order, "message"):
                return str(order.message)
            return str(order)
        except Exception:
            return f"action[{action_idx}]"

    def _print_debug(
        self,
        battle: DoubleBattle,
        probs: "np.ndarray[Any, Any]",
        value: float,
        selected: int,
        is_teampreview: bool,
    ) -> None:
        print("\n" + "=" * 80)
        print(
            f"Battle: {battle.battle_tag} | Turn: {battle.turn} | Teampreview: {battle.teampreview}"
        )
        print(f"State value estimate: {value:.4f}")

        topk_indices = np.argsort(probs)[-self.top_k :][::-1]
        print(f"Top-{self.top_k} actions:")
        for rank, idx in enumerate(topk_indices, start=1):
            desc = self._describe_action(battle, int(idx), is_teampreview)
            print(f"  {rank}. p={probs[idx]:.4f} | {desc}")

        selected_desc = self._describe_action(battle, int(selected), is_teampreview)
        print(f"Selected: p={probs[selected]:.4f} | {selected_desc}")

        if self.print_summary:
            print("\nBattle summary:")
            print(battle_to_str(battle))

    def _on_action_selected(
        self,
        battle: DoubleBattle,
        probs: "np.ndarray[Any, Any]",
        value: float,
        selected: int,
        is_teampreview: bool,
    ) -> None:
        self._print_debug(battle, probs, value, selected, is_teampreview)
