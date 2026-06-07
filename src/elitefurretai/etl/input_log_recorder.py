# -*- coding: utf-8 -*-
"""input_log_recorder.py

``InputLogRecorder`` — a mixin for poke-env ``Player`` subclasses that records every
order the player makes, in decision order, keyed by ``battle_tag``, in the
``BattleData.input_logs`` (`">pX ..."`) format.

Self-play battles have no server-side ``inputLog``, so to serialize them into a
replayable ``BattleData`` we must capture each player's own inputs as they are made.
Mix this into the players used for self-play data generation, then combine the two
players' recorded logs with ``etl.build_self_play_battle_data`` (or ``merge_input_logs``)::

    class RecordingMaxDamage(InputLogRecorder, MaxDamagePlayer):
        pass

    p1, p2 = RecordingMaxDamage(...), RecordingMaxDamage(...)
    await p1.battle_against(p2, n_battles=1)
    bd = build_self_play_battle_data(
        p1_battle, p2_battle, p1.get_input_log(tag), p2.get_input_log(tag)
    )

Works with both synchronous and ``async`` ``choose_move`` / ``teampreview`` players;
ordering is preserved because poke-env requests (and therefore our overrides) fire
once per decision, in order.
"""

import inspect
from typing import Any, Awaitable, Dict, List, Union

from poke_env.battle import AbstractBattle

from elitefurretai.etl.self_play import (
    battle_order_to_input,
    teampreview_order_to_input,
)


class InputLogRecorder:
    """Mixin that records a player's orders into ``BattleData.input_logs`` format.

    Place it BEFORE the concrete ``Player`` in the MRO so ``super()`` resolves to the
    real agent (e.g. ``class Rec(InputLogRecorder, MaxDamagePlayer)``).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # battle_tag -> ordered list of ">pX ..." input lines
        self._recorded_input_logs: Dict[str, List[str]] = {}

    def get_input_log(self, battle_tag: str) -> List[str]:
        """The recorded inputs for a battle, in the order they were made."""
        return self._recorded_input_logs.get(battle_tag, [])

    def clear_input_logs(self) -> None:
        self._recorded_input_logs.clear()

    def _record(self, battle: AbstractBattle, line: Union[str, None]) -> None:
        if line is not None:
            self._recorded_input_logs.setdefault(battle.battle_tag, []).append(line)

    def choose_move(self, battle: AbstractBattle) -> Any:
        result = super().choose_move(battle)  # type: ignore[misc]
        if inspect.isawaitable(result):

            async def _await_and_record() -> Any:
                order = await result
                self._record(battle, battle_order_to_input(order, battle))
                return order

            return _await_and_record()
        self._record(battle, battle_order_to_input(result, battle))
        return result

    def teampreview(self, battle: AbstractBattle) -> Union[str, Awaitable[str]]:
        result = super().teampreview(battle)  # type: ignore[misc]
        role = battle.player_role
        assert role is not None
        if inspect.isawaitable(result):

            async def _await_and_record() -> str:
                order = await result
                self._record(battle, teampreview_order_to_input(order, role))
                return order

            return _await_and_record()
        self._record(battle, teampreview_order_to_input(result, role))
        return result
