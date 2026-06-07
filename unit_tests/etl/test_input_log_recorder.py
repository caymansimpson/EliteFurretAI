# -*- coding: utf-8 -*-
import asyncio
from types import SimpleNamespace
from typing import cast

from poke_env.battle import AbstractBattle, Move
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

from elitefurretai.etl.input_log_recorder import InputLogRecorder


def _battle(role="p1", tag="battle-1") -> AbstractBattle:
    # The recorder + move/pass conversion only read player_role and battle_tag.
    return cast(AbstractBattle, SimpleNamespace(player_role=role, battle_tag=tag))


def _dbo(move_id, target, second=None):
    first = SingleBattleOrder(order=Move(move_id, gen=9), move_target=target)
    return DoubleBattleOrder(first, second if second is not None else PassBattleOrder())


class _SyncStub:
    def __init__(self, orders, tp="/team 1, 2, 3, 4"):
        self._orders = list(orders)
        self._i = 0
        self._tp = tp

    def choose_move(self, battle):
        order = self._orders[self._i]
        self._i += 1
        return order

    def teampreview(self, battle):
        return self._tp


class _AsyncStub(_SyncStub):
    async def choose_move(self, battle):
        return super().choose_move(battle)

    async def teampreview(self, battle):
        return super().teampreview(battle)


class _SyncRecorder(InputLogRecorder, _SyncStub):
    pass


class _AsyncRecorder(InputLogRecorder, _AsyncStub):
    pass


def test_records_teampreview_and_orders_in_decision_order():
    player = _SyncRecorder(
        [
            _dbo("uturn", 2, SingleBattleOrder(order=Move("protect", gen=9))),
            _dbo("fakeout", 1),
        ],
        tp="/team 4, 5, 1, 3",
    )
    battle = _battle(role="p1", tag="t1")

    player.teampreview(battle)
    player.choose_move(battle)
    player.choose_move(battle)

    assert player.get_input_log("t1") == [
        ">p1 team 4, 5, 1, 3",
        ">p1 move uturn +2, move protect",
        ">p1 move fakeout +1, pass",  # second slot defaults to a pass
    ]


def test_separates_logs_by_battle_tag_and_role():
    player = _SyncRecorder([_dbo("surf", 0)], tp="/team 1, 2, 3, 4")
    player.teampreview(_battle(role="p2", tag="A"))
    player.choose_move(_battle(role="p2", tag="A"))
    assert player.get_input_log("A") == [">p2 team 1, 2, 3, 4", ">p2 move surf, pass"]
    assert player.get_input_log("missing") == []


def test_works_with_async_players():
    player = _AsyncRecorder([_dbo("spore", 1)], tp="/team 2, 4, 1, 6")
    battle = _battle(role="p1", tag="async-1")

    # teampreview's declared return is Union[str, Awaitable[str]]; the async stub
    # yields a coroutine here, which asyncio.run accepts.
    asyncio.run(player.teampreview(battle))  # type: ignore[arg-type]
    asyncio.run(player.choose_move(battle))

    assert player.get_input_log("async-1") == [
        ">p1 team 2, 4, 1, 6",
        ">p1 move spore +1, pass",
    ]
