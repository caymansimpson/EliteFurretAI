# -*- coding: utf-8 -*-
import pytest

from elitefurretai.model_utils import BattleData, BattleIterator


def test_battle_iterator(vgc_json_anon):
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle = bd.to_battle(perspective="p1")
    iter = BattleIterator(battle, bd)
    assert iter

    # Go to teampreview
    iter.next_input()
    assert iter.log == "|"
    assert vgc_json_anon["inputLog"][iter._input_nums[0] : iter._input_nums[1]] == [
        ">p1 team 4, 5, 1, 3",
        ">p2 team 1, 2, 6, 3",
    ]
    assert iter._input_nums == [0, 2]

    # Test ability to move to the next
    iter.next()
    iter.next()
    assert iter.log == "|switch|p1a: 780b3dada7|Arcanine, L50, F|191/191"

    iter.next_turn()
    assert iter.log == "|"
    assert iter._input_nums == [2, 4]

    iter.next_input()
    assert iter.log == "|"
    assert iter._input_nums == [4, 5]

    iter.finish()
    assert battle.finished

    with pytest.raises(StopIteration):
        iter.next()


def test_next_input(vgc_json_anon2):
    bd = BattleData.from_showdown_json(vgc_json_anon2)
    battle = bd.to_battle(perspective="p1")

    iter = BattleIterator(battle, bd)
    assert iter.log == vgc_json_anon2["log"][0]
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 1, 4, 5, 2"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2 terastallize, move heavyslam +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +1"

    iter.next_input()
    assert iter.last_input == ">p1 move whirlwind +2, move wildcharge +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +1, move overheat +2"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2, move discharge"

    iter.next_input()
    assert iter.last_input == ">p1 move ruination +2, move discharge"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move discharge"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move phantomforce +2, move discharge"

    iter.next_input()
    assert iter.last_input is None

    iter.next_input()
    assert iter.finished

    battle = bd.to_battle(perspective="p2")
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective="p2"
    )
    assert iter.log == vgc_json_anon2["log"][0]
    assert iter.last_input is None
    assert iter.next_input() == ">p2 team 1, 5, 4, 3"
    assert iter.next_input() == ">p2 move mortalspin terastallize, switch 3"
    assert iter.next_input() == ">p2 move earthpower +2, move yawn +1"
    assert iter.next_input() == ">p2 switch 3, move earthpower +2"
    assert iter.next_input() == ">p2 move waterfall +2, move yawn +1"
    assert iter.next_input() == ">p2 switch 3, pass"
    assert iter.next_input() == ">p2 move yawn +2, move mortalspin"
    assert iter.next_input() == ">p2 pass, switch 4"
    assert iter.next_input() == ">p2 move earthpower +1, move hex +1"
    assert iter.next_input() == ">p2 move earthpower +1, move substitute"
    assert iter.next_input() == ">p2 move earthpower +1, move protect"
    assert iter.next_input() == ">p2 move yawn +1, move moonblast +1"
    assert iter.next_input() is None


def test_next_input_3(vgc_json_anon3):
    bd = BattleData.from_showdown_json(vgc_json_anon3)

    battle = bd.to_battle(perspective="p2")
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective="p2"
    )
    assert iter.last_input is None
    iter.next_input()
    assert iter.last_input == ">p2 team 3, 6, 1, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move reflect, move sacredsword +1"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move suckerpunch +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move protect"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +2, move ragepowder"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move lightscreen, move bellydrum terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move protect"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move drainpunch +2"


def test_next_input_4(vgc_json_anon4):
    bd = BattleData.from_showdown_json(vgc_json_anon4)

    battle = bd.to_battle(perspective="p1")
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective="p1"
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 1, 6, 2, 5"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, move voltswitch +2"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, move throatchop +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam terastallize, move spore +2"

    iter.next_input()
    assert iter.last_input == ">p1 move dazzlinggleam, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move spore +2, move fakeout +1"


def test_next_input_5(vgc_json_anon5):
    bd = BattleData.from_showdown_json(vgc_json_anon5)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 3, 6, 2, 4"

    iter.next_input()
    assert iter.last_input == ">p1 move hydropump +1, move flareblitz +1"

    iter.next_input()
    assert iter.last_input == ">p1 move willowisp +2, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +1, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +2, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move suckerpunch +1 terastallize, move waterfall +2"

    iter.next_input()
    assert iter.last_input == ">p1 move suckerpunch +2, move waterfall +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 3, 5, 6, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move trickroom, move protect"

    iter.next_input()
    assert iter.last_input == ">p2 move beatup -2, move bulkup"

    iter.next_input()
    assert iter.last_input == ">p2 move allyswitch, move ragefist +2"

    iter.next_input()
    assert iter.last_input == ">p2 move ragefist +2, move hypervoice"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move ragefist +1, move eruption terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, move flamethrower +1"

    iter.next_input()
    assert iter.last_input == ">p2 move dazzlinggleam, move flamethrower +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move drainpunch +1"


def test_next_input_6(vgc_json_anon6):
    bd = BattleData.from_showdown_json(vgc_json_anon6)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 6, 1, 5, 4"

    iter.next_input()
    assert iter.last_input == ">p1 move fakeout +2, move uturn +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move powergem +2"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move bravebird +2, move hydropump +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move fakeout +2, move icywind"

    iter.next_input()
    assert iter.last_input == ">p1 move drainpunch +1, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move uturn +1"

    iter.next_input()
    assert iter.last_input == ">p1 move hydropump +2, move uturn +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move icywind, move fakeout +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move drainpunch +2"

    iter.next_input()
    assert iter.last_input == ">p1 move bravebird +2, move drainpunch +2"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move bravebird +2, move icywind"

    iter.next_input()
    assert iter.last_input == ">p1 pass, move icywind"

    iter.next_input()
    assert iter.last_input == ">p1 pass, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 pass, move hydropump +2"


def test_next_input_7(vgc_json_anon7):
    bd = BattleData.from_showdown_json(vgc_json_anon7)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 2, 5, 4, 6"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderbolt +1, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move spore +1, move bitterblade +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 1, 3, 5, 4"

    iter.next_input()
    assert iter.last_input == ">p2 move partingshot +1, move flareblitz +2"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move freezedry +2, move flareblitz +1"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move spiritbreak +1, move willowisp +2"

    iter.next_input()
    assert iter.last_input == ">p2 move reflect, switch 3"


def test_next_input_8(vgc_json_anon8):
    bd = BattleData.from_showdown_json(vgc_json_anon8)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 2, 1, 3, 6"

    iter.next_input()
    assert iter.last_input == ">p1 move fakeout +1, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move wildcharge +1, move spore +2"

    iter.next_input()
    assert iter.last_input == ">p1 move wildcharge +2, move ragepowder"

    iter.next_input()
    assert iter.last_input == ">p1 move wildcharge +2, move pollenpuff -1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move bulkup terastallize, move spore +2"

    iter.next_input()
    assert iter.last_input == ">p1 move bitterblade +2, move spore +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 2, 1, 6, 4"

    iter.next_input()
    assert iter.last_input == ">p2 move tailwind, move earthquake"

    iter.next_input()
    assert iter.last_input == ">p2 move tailwind, move phantomforce +1"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move fakeout +1, move phantomforce +1"

    iter.next_input()
    assert iter.last_input == ">p2 move thunderpunch +1, move shadowclaw +1"

    iter.next_input()
    assert iter.last_input == ">p2 move revivalblessing, move phantomforce +1"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move closecombat +1, move phantomforce +1"


def test_next_input_9(vgc_json_anon9):
    bd = BattleData.from_showdown_json(vgc_json_anon9)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 5, 2, 3, 6"

    iter.next_input()
    assert iter.last_input == ">p1 move thunderwave +1, move dragondance terastallize"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move shadowball +1, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move shadowball +1, move terablast +2"

    iter.next_input()
    assert iter.last_input == ">p1 move struggle, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move struggle, move rockslide"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, move protect"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 1, 6, 3, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move howl, move populationbomb +1"

    iter.next_input()
    assert iter.last_input == ">p2 move encore +2, move populationbomb +2"

    iter.next_input()
    assert iter.last_input == ">p2 move disable +1, move populationbomb +2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move howl, move protect"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move icespinner +1, move extremespeed +2 terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 move suckerpunch +2, move aquajet +1"


def test_next_input_10(vgc_json_anon10):
    bd = BattleData.from_showdown_json(vgc_json_anon10)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 1, 6, 5, 3"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move hydropump +1, move suckerpunch +2"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move icespinner +1"

    iter.next_input()
    assert iter.last_input == ">p1 move heatwave, move icespinner +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 1, 5, 6, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move fakeout +2, move trickroom"

    iter.next_input()
    assert iter.last_input == ">p2 move drainpunch +2, move dazzlinggleam"

    iter.next_input()
    assert iter.last_input == ">p2 move voltswitch +2, move protect"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move spore +1, move dazzlinggleam"


def test_next_input_11(vgc_json_anon11):
    bd = BattleData.from_showdown_json(vgc_json_anon11)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 5, 2, 4, 3"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move fakeout +2"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 move ragingbull +1, move psychic +2"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move psychic +1"

    iter.next_input()
    assert iter.last_input == ">p1 move closecombat +1, move psychic +2"

    iter.next_input()
    assert iter.last_input == ">p1 move ragingbull +1, move psychic +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move jetpunch +1 terastallize, move psychic +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 6, 4, 5, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move uturn +2, move chillyreception"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move auroraveil, move uturn +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move leechseed +2, move chillingwater +1"

    iter.next_input()
    assert iter.last_input == ">p2 move leechseed +1, move yawn +2"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move swordsdance terastallize, move trickroom"

    iter.next_input()
    assert iter.last_input == ">p2 move terablast +1, move chillyreception"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move thunderpunch +1, move uturn +2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"


def test_next_input_12(vgc_json_anon12):
    bd = BattleData.from_showdown_json(vgc_json_anon12)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 4, 3, 2, 1"

    iter.next_input()
    assert iter.last_input == ">p1 move tailwind, move voltswitch +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move bravebird +2, move extremespeed +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, switch 3"

    iter.next_input()
    assert iter.last_input == ">p1 move iceshard +1, move hydropump +2"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move hydropump +1"

    iter.next_input()
    assert iter.last_input == ">p1 move iceshard +1, move voltswitch +1 terastallize"

    iter.next_input()
    assert iter.last_input == ">p1 pass, move voltswitch +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 4, 6, 1, 2"

    iter.next_input()
    assert iter.last_input == ">p2 move freezedry +1, move moonblast +1"

    iter.next_input()
    assert iter.last_input == ">p2 move hydropump +2, move moonblast +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, move bravebird +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move iceshard +1 terastallize, move freezedry +2"

    iter.next_input()
    assert iter.last_input == ">p2 move iceshard +1, move freezedry +2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move freezedry +2"


def test_next_input_13(vgc_json_anon13):
    bd = BattleData.from_showdown_json(vgc_json_anon13)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 6, 5, 1, 2"

    iter.next_input()
    assert iter.last_input == ">p1 move flipturn +2, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move heatwave, move hydropump +1"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move jetpunch +2, move hydropump +1"

    iter.next_input()
    assert iter.last_input == ">p1 move jetpunch +2 terastallize, move freezedry +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 6, 2, 1, 3"

    iter.next_input()
    assert iter.last_input == ">p2 move irondefense, move followme"

    iter.next_input()
    assert iter.last_input == ">p2 move bodypress +1 terastallize, move followme"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move recover, move tailwind"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 pass, move earthquake"


def test_next_input_14(vgc_json_anon14):
    bd = BattleData.from_showdown_json(vgc_json_anon14)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 5, 2, 4, 1"

    iter.next_input()
    assert iter.last_input == ">p1 move protect, move spikyshield"

    iter.next_input()
    assert iter.last_input == ">p1 move disable +1, move powergem +2"

    iter.next_input()
    assert iter.last_input == ">p1 move encore +1, move powergem +2"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 5, 3, 2, 4"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +2, move uturn +2 terastallize"

    iter.next_input()
    assert iter.last_input == ">p2 move earthpower +2, move uturn +2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move thunderbolt +1, move sacredsword +2"


def test_next_input_15(vgc_json_anon15):
    bd = BattleData.from_showdown_json(vgc_json_anon15)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 4, 5, 6, 1"

    iter.next_input()
    assert iter.last_input == ">p1 move tailwind, move protect"

    iter.next_input()
    assert iter.last_input == ">p1 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move dragondarts +1 terastallize, move sacredsword +2"

    iter.next_input()
    assert iter.last_input == ">p1 move dragondarts +1, move sacredsword +1"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 move dragondarts +1, move dazzlinggleam"

    iter.next_input()
    assert iter.last_input == ">p1 move dragondarts +2, move dazzlinggleam"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 4, 3, 1, 5"

    iter.next_input()
    assert iter.last_input == ">p2 move voltswitch +1, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move willowisp +1 terastallize, move flareblitz +2"

    iter.next_input()
    assert iter.last_input == ">p2 move voltswitch +1, move extremespeed +2"

    iter.next_input()
    assert iter.last_input == ">p2 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move tailwind, move voltswitch +2"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 3"

    iter.next_input()
    assert iter.last_input == ">p2 move bravebird +1, move earthquake"


def test_next_input_16(vgc_json_anon16):
    bd = BattleData.from_showdown_json(vgc_json_anon16)
    p = "p1"

    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p1 team 2, 4, 1, 3"

    iter.next_input()
    assert iter.last_input == ">p1 move trickroom, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p1 switch 4, pass"

    iter.next_input()
    assert iter.last_input == ">p1 move spore +2, move ruination +1"

    iter.next_input()
    assert iter.last_input == ">p1 move spore +1, move ruination +2"

    iter.next_input()
    assert iter.last_input == ">p1 move pollenpuff +1, move ruination +1"

    p = "p2"
    battle = bd.to_battle(perspective=p)
    iter = BattleIterator(
        battle, bd, custom_parse=BattleData.showdown_translation, perspective=p
    )
    assert iter.last_input is None

    iter.next_input()
    assert iter.last_input == ">p2 team 2, 6, 3, 1"

    iter.next_input()
    assert iter.last_input == ">p2 move flipturn +2, move finalgambit +1"

    iter.next_input()
    assert iter.last_input == ">p2 pass, switch 4"

    iter.next_input()
    assert iter.last_input == ">p2 move flipturn +2, move perishsong"

    iter.next_input()
    assert iter.last_input == ">p2 switch 3, pass"

    iter.next_input()
    assert iter.last_input == ">p2 move fakeout +2, move reflect"

    iter.next_input()
    assert iter.last_input == ">p2 move protect, move reflect"
