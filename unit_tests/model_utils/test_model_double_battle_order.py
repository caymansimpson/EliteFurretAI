# -*- coding: utf-8 -*-
from poke_env.environment import Move
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder

from elitefurretai.model_utils import MDBO, BattleData, BattleIterator


def test_mdbo():
    order = MDBO(MDBO.DEFAULT)
    assert order.to_int() == -1
    assert MDBO.from_int(-1, MDBO.DEFAULT).to_int() == -1

    order = MDBO(MDBO.TEAMPREVIEW, "/team 1234")
    assert order.to_int() == 0
    assert MDBO.from_int(order.to_int(), MDBO.TEAMPREVIEW).to_int() == 0

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
    assert iter.last_order().message == "/team 1, 4, 5, 2"
    assert (
        iter.last_order().to_int()
        == MDBO.from_int(iter.last_order().to_int(), MDBO.TEAMPREVIEW).to_int()
    )

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2 terastallize, move heavyslam +1"
    assert iter.last_order().message == "/choose move 2 2 terastallize, move 4 1"

    # Test conversion
    assert iter.last_order().to_double_battle_order(iter.battle) == DoubleBattleOrder(  # type: ignore
        first_order=BattleOrder(
            order=Move("ruination", gen=9), terastallize=True, move_target=2
        ),
        second_order=BattleOrder(order=Move("heavyslam", gen=9), move_target=1),
    )

    # Test self-target
    assert MDBO(MDBO.TURN, "/choose move 2 -2 terastallize, move 4 -1").to_double_battle_order(iter.battle) == DoubleBattleOrder(  # type: ignore
        first_order=BattleOrder(
            order=Move("ruination", gen=9), terastallize=True, move_target=-2
        ),
        second_order=BattleOrder(order=Move("heavyslam", gen=9), move_target=-1),
    )

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +1"
    assert iter.last_order().message == "/choose switch 3, move 1 1"

    # Test switch
    key = list(iter.battle.team.keys())[2]
    assert iter.last_order().to_double_battle_order(iter.battle) == DoubleBattleOrder(  # type: ignore
        first_order=BattleOrder(order=iter.battle.team[key]),
        second_order=BattleOrder(order=Move("drainpunch", gen=9), move_target=1),
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
    assert iter.last_order().message == "/team 1, 5, 4, 3"

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
    assert iter.last_order().message == "/team 3, 6, 1, 2"

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
    assert iter.last_order().message == "/team 1, 6, 2, 5"

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
    assert iter.last_order().message == "/team 3, 6, 2, 4"

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
    assert iter.last_order().message == "/team 3, 5, 6, 2"

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
