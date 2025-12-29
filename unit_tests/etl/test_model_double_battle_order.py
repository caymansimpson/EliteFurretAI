# -*- coding: utf-8 -*-
from typing import Union

from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    SingleBattleOrder,
)

from elitefurretai.etl import MDBO, BattleData, BattleIterator


def orders_equal(
    o1: Union[DoubleBattleOrder, DefaultBattleOrder],
    o2: Union[DoubleBattleOrder, DefaultBattleOrder],
) -> bool:
    """
    Compare two DoubleBattleOrders by their semantic content.

    Move objects don't have __eq__ defined, so we compare by move ID.
    Pokemon objects are compared by species.
    """
    # Handle DefaultBattleOrder
    if isinstance(o1, DefaultBattleOrder) and isinstance(o2, DefaultBattleOrder):
        return True
    if isinstance(o1, DefaultBattleOrder) or isinstance(o2, DefaultBattleOrder):
        return False

    # Now both are DoubleBattleOrder
    assert isinstance(o1, DoubleBattleOrder)
    assert isinstance(o2, DoubleBattleOrder)

    def single_order_equal(s1: SingleBattleOrder, s2: SingleBattleOrder) -> bool:
        if s1.mega != s2.mega:
            return False
        if s1.z_move != s2.z_move:
            return False
        if s1.dynamax != s2.dynamax:
            return False
        if s1.terastallize != s2.terastallize:
            return False
        if s1.move_target != s2.move_target:
            return False

        # Compare order (Move or Pokemon)
        if isinstance(s1.order, Move) and isinstance(s2.order, Move):
            return s1.order.id == s2.order.id
        elif isinstance(s1.order, Pokemon) and isinstance(s2.order, Pokemon):
            return s1.order.species == s2.order.species
        elif s1.order is None and s2.order is None:
            return True
        else:
            return s1.order == s2.order

    return single_order_equal(o1.first_order, o2.first_order) and single_order_equal(
        o1.second_order, o2.second_order
    )


def test_mdbo():
    order = MDBO(MDBO.DEFAULT)
    assert order.to_int() == -1
    assert MDBO.from_int(-1, MDBO.DEFAULT).to_int() == -1

    order = MDBO(MDBO.TEAMPREVIEW, "/team 1234")
    assert order.to_int() == 0
    assert MDBO.from_int(order.to_int(), MDBO.TEAMPREVIEW).to_int() == 0
    assert order.message == "/team 1234"

    order = MDBO(MDBO.TEAMPREVIEW, "/team 4321")
    assert order.message == "/team 3412"  # Remember we sort pairing and back mons

    order = MDBO(MDBO.FORCE_SWITCH, "/choose pass, switch 1")
    assert order.to_int() == 2020
    assert MDBO.from_int(order.to_int(), MDBO.FORCE_SWITCH).to_int() == 2020

    order = MDBO(MDBO.TURN, "/choose move 1 -2 terastallize, move 2")
    assert order.to_int() == 237
    assert MDBO.from_int(order.to_int(), MDBO.TURN).to_int() == 237


def test_mdbo_2(vgc_json_anon2):
    bd = BattleData.from_showdown_json(vgc_json_anon2)

    iter = BattleIterator(bd)
    assert iter.log == vgc_json_anon2["log"][0]
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 1, 4, 5, 2"
    assert iter.last_order().message == "/team 1425"
    assert (
        iter.last_order().to_int()
        == MDBO.from_int(iter.last_order().to_int(), MDBO.TEAMPREVIEW).to_int()
    )

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2 terastallize, move heavyslam +1"
    assert iter.last_order().message == "/choose move 2 2 terastallize, move 4 1"

    o1 = iter.last_order().to_double_battle_order(iter.battle)  # type: ignore
    o2 = DoubleBattleOrder(  # type: ignore
        first_order=SingleBattleOrder(
            order=Move("ruination", gen=9), terastallize=True, move_target=2
        ),
        second_order=SingleBattleOrder(order=Move("heavyslam", gen=9), move_target=1),
    )

    # Test conversion (use orders_equal since Move objects don't have __eq__)
    assert orders_equal(o1, o2)

    # Test self-target
    assert orders_equal(
        MDBO(
            MDBO.TURN, "/choose move 2 -2 terastallize, move 4 -1"
        ).to_double_battle_order(iter.battle),  # type: ignore
        DoubleBattleOrder(
            first_order=SingleBattleOrder(
                order=Move("ruination", gen=9), terastallize=True, move_target=-2
            ),
            second_order=SingleBattleOrder(order=Move("heavyslam", gen=9), move_target=-1),
        ),
    )

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +1"
    assert iter.last_order().message == "/choose switch 3, move 1 1"

    # Test switch
    key = list(iter.battle.team.keys())[2]
    assert orders_equal(
        iter.last_order().to_double_battle_order(iter.battle),  # type: ignore
        DoubleBattleOrder(
            first_order=SingleBattleOrder(order=iter.battle.team[key]),
            second_order=SingleBattleOrder(order=Move("drainpunch", gen=9), move_target=1),
        ),
    )

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +1"
    assert iter.last_order().message == "/choose switch 3, move 1 1"

    iter.next_input()
    assert iter.last_input == ">p1 move whirlwind +2, move wildcharge +1"
    assert iter.last_order().message == "/choose move 3 2, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"
    assert iter.last_order().message == "/choose pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +1, move overheat +2"
    assert iter.last_order().message == "/choose move 2 1, move 1 2"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2, move discharge"
    assert iter.last_order().message == "/choose move 2 2, move 2"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2, move discharge"
    assert iter.last_order().message == "/choose move 2 2, move 2"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move discharge"
    assert iter.last_order().message == "/choose move 4, move 2"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"
    assert iter.last_order().message == "/choose switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move phantomforce +2, move discharge"
    assert iter.last_order().message == "/choose move 3 2, move 2"

    iter.next_input()
    assert iter.last_input is None
    assert iter.last_order().message == "/choose default"

    iter = BattleIterator(bd, perspective="p2")
    assert iter.log == vgc_json_anon2["log"][0]
    assert iter.last_input is None
    assert iter.next_input() == ">p2 team 1, 5, 4, 3"
    assert iter.last_order().message == "/team 1534"

    iter.next_input()
    assert iter.last_input == ">p2 move mortalspin terastallize, switch 3"
    assert iter.last_order().message == "/choose move 4 terastallize, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +2, move yawn +1"
    assert iter.last_order().message == "/choose move 3 2, move 3 1"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, move earthpower +2"
    assert iter.last_order().message == "/choose switch 3, move 2 2"

    iter.next_input()
    assert iter.last_input == ">p2 move waterfall +2, move yawn +1"
    assert iter.last_order().message == "/choose move 1 2, move 3 1"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, pass"
    assert iter.last_order().message == "/choose switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move yawn +2, move mortalspin"
    assert iter.last_order().message == "/choose move 3 2, move 4"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"
    assert iter.last_order().message == "/choose pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +1, move hex +1"
    assert iter.last_order().message == "/choose move 2 1, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +1, move substitute"
    assert iter.last_order().message == "/choose move 2 1, move 3"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +1, move protect"
    assert iter.last_order().message == "/choose move 2 1, move 4"

    iter.next_input()
    assert iter.last_input == ">p2 move yawn +1, move moonblast +1"
    assert iter.last_order().message == "/choose move 3 1, move 1 1"


def test_mdbo_3(vgc_json_anon3):
    bd = BattleData.from_showdown_json(vgc_json_anon3)

    iter = BattleIterator(bd, perspective="p2")
    assert iter.last_input is None
    iter.next_input()
    assert iter.last_input == ">p2 team 3, 6, 1, 2"
    assert iter.last_order().message == "/team 3612"

    iter.next_input()
    assert iter.last_input == ">p2 move reflect, move sacredsword +1"
    assert iter.last_order().message == "/choose move 2, move 3 1"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move suckerpunch +1"
    assert iter.last_order().message == "/choose move 4 2, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"
    assert iter.last_order().message == "/choose pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move protect"
    assert iter.last_order().message == "/choose move 4 2, move 4"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move ragepowder"
    assert iter.last_order().message == "/choose move 4 2, move 3"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"
    assert iter.last_order().message == "/choose pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move lightscreen, move bellydrum terastallize"
    assert iter.last_order().message == "/choose move 1, move 4 terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move protect"
    assert iter.last_order().message == "/choose pass, move 3"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move drainpunch +2"
    assert iter.last_order().message == "/choose pass, move 1 2"


def test_mdbo_4(vgc_json_anon4):
    bd = BattleData.from_showdown_json(vgc_json_anon4)

    iter = BattleIterator(bd, perspective="p1")
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 1, 6, 2, 5"
    assert iter.last_order().message == "/team 1625"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, move voltswitch +2"
    assert iter.last_order().message == "/choose move 4, move 2 2"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"
    assert iter.last_order().message == "/choose pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, move throatchop +1"
    assert iter.last_order().message == "/choose move 4, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"
    assert iter.last_order().message == "/choose pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam terastallize, move spore +2"
    assert iter.last_order().message == "/choose move 4 terastallize, move 4 2"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, switch 4"
    assert iter.last_order().message == "/choose move 4, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"
    assert iter.last_order().message == "/choose switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move spore +2, move fakeout +1"
    assert iter.last_order().message == "/choose move 4 2, move 4 1"


def test_mdbo_5(vgc_json_anon5):
    bd = BattleData.from_showdown_json(vgc_json_anon5)

    iter = BattleIterator(bd, perspective="p1")
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 3, 6, 2, 4"
    assert iter.last_order().message == "/team 3624"

    iter.next_input()
    assert iter.last_input == ">p1 move hydropump +1, move flareblitz +1"
    assert iter.last_order().message == "/choose move 3 1, move 1 1"

    iter.next_input()
    assert iter.last_input == ">p1 move willowisp +2, move protect"
    assert iter.last_order().message == "/choose move 4 2, move 4"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +1, switch 4"
    assert iter.last_order().message == "/choose move 1 1, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, move protect"
    assert iter.last_order().message == "/choose move 1 2, move 3"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, switch 4"
    assert iter.last_order().message == "/choose move 1 2, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"
    assert iter.last_order().message == "/choose pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, move protect"
    assert iter.last_order().message == "/choose move 1 2, move 3"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"
    assert iter.last_order().message == "/choose switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move suckerpunch +1 terastallize, move waterfall +2"
    assert iter.last_order().message == "/choose move 2 1 terastallize, move 2 2"

    iter.next_input()
    assert iter.last_input == ">p1 move suckerpunch +2, move waterfall +2"
    assert iter.last_order().message == "/choose move 2 2, move 2 2"

    iter = BattleIterator(bd, perspective="p2")
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 3, 5, 6, 2"
    assert iter.last_order().message == "/team 3526"

    iter.next_input()
    assert iter.last_input == ">p2 move trickroom, move protect"
    assert iter.last_order().message == "/choose move 3, move 4"

    iter.next_input()
    assert iter.last_input == ">p2 move beatup -2, move bulkup"
    assert iter.last_order().message == "/choose move 1 -2, move 3"

    iter.next_input()
    assert iter.last_input == ">p2 move allyswitch, move ragefist +2"
    assert iter.last_order().message == "/choose move 4, move 1 2"

    iter.next_input()
    assert iter.last_input == ">p2 move ragefist +2, move hypervoice"
    assert iter.last_order().message == "/choose move 1 2, move 2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"
    assert iter.last_order().message == "/choose pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move ragefist +1, move eruption terastallize"
    assert iter.last_order().message == "/choose move 1 1, move 1 terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, move flamethrower +1"
    assert iter.last_order().message == "/choose switch 4, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p2 move dazzlinggleam, move flamethrower +1"
    assert iter.last_order().message == "/choose move 1, move 2 1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"
    assert iter.last_order().message == "/choose pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move drainpunch +1"
    assert iter.last_order().message == "/choose pass, move 2 1"


class DummyBattle:
    def __init__(self):
        self.player_role = "p1"
        self.active_pokemon = [None, None]
        self.team = {str(i): i for i in range(6)}

    def __getattr__(self, name):
        return None


def test_mdbo_teampreview_to_int_and_back():
    # Test that teampreview orders round-trip
    for i in range(MDBO.teampreview_space()):
        mdbo = MDBO.from_int(i, MDBO.TEAMPREVIEW)
        assert mdbo.to_int() == i
        assert mdbo._type == MDBO.TEAMPREVIEW
        assert mdbo.message.startswith("/team")


def test_mdbo_turn_to_int_and_back():
    # Test that turn orders round-trip
    for i in range(10):  # Just test a few
        mdbo = MDBO.from_int(i, MDBO.TURN)
        assert mdbo.to_int() == i
        assert mdbo._type == MDBO.TURN
        assert mdbo.message.startswith("/choose")


def test_mdbo_default():
    mdbo = MDBO(MDBO.DEFAULT)
    assert mdbo.to_int() == -1
    assert mdbo.message == "/choose default"


def test_mdbo_to_double_battle_order():
    mdbo = MDBO.from_int(0, MDBO.TURN)
    battle = DummyBattle()
    battle.player_role = "p1"
    battle.active_pokemon = [
        type("DummyMon", (), {"moves": {"a": 1}})(),
        type("DummyMon", (), {"moves": {"a": 1}})(),
    ]
    battle.team = {str(i): i for i in range(6)}
    dbo = mdbo.to_double_battle_order(battle)  # type: ignore
    assert dbo is not None


def test_mdbo_action_space():
    assert isinstance(MDBO.action_space(), int)
    assert MDBO.action_space() > 0


def test_mdbo_teampreview_space():
    # Ensure that the teampreview_space method of MDBO returns an integer
    assert isinstance(MDBO.teampreview_space(), int)
    assert MDBO.teampreview_space() > 0
