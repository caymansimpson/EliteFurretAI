# -*- coding: utf-8 -*-
"""Unit tests for the HumanPlayer action grammar parser.

The parser turns terminal input strings like ``"astralbarrage 1 tera, surgingstrikes 1"``
into ``BattleOrder`` instances. It's a pure function — no I/O, no battle mutation —
so it's straightforward to test on a hand-built ``DoubleBattle`` stub.
"""

from logging import Logger

import pytest
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.player.battle_order import (
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder


def _calyrex() -> Pokemon:
    tb = ConstantTeambuilder(
        """Calyrex-Shadow @ Life Orb
        Ability: As One (Spectrier)
        Level: 50
        Tera Type: Ghost
        - Astral Barrage
        - Psyshock
        - Nasty Plot
        - Protect"""
    )
    mon = Pokemon(gen=9, teambuilder=tb.team[0])
    mon._species = "calyrexshadow"
    mon._current_hp = 100
    mon._max_hp = 100
    mon._active = True
    return mon


def _urshifu() -> Pokemon:
    tb = ConstantTeambuilder(
        """Urshifu-Rapid-Strike @ Mystic Water
        Ability: Unseen Fist
        Level: 50
        Tera Type: Water
        - Surging Strikes
        - Close Combat
        - Aqua Jet
        - Detect"""
    )
    mon = Pokemon(gen=9, teambuilder=tb.team[0])
    mon._species = "urshifurapidstrike"
    mon._current_hp = 100
    mon._max_hp = 100
    mon._active = True
    return mon


def _incineroar() -> Pokemon:
    tb = ConstantTeambuilder(
        """Incineroar @ Sitrus Berry
        Ability: Intimidate
        Level: 50
        Tera Type: Ghost
        - Fake Out
        - Knock Off
        - Parting Shot
        - Flare Blitz"""
    )
    mon = Pokemon(gen=9, teambuilder=tb.team[0])
    mon._species = "incineroar"
    mon._current_hp = 100
    mon._max_hp = 100
    return mon


def _rillaboom() -> Pokemon:
    tb = ConstantTeambuilder(
        """Rillaboom @ Assault Vest
        Ability: Grassy Surge
        Level: 50
        Tera Type: Fire
        - Wood Hammer
        - Fake Out
        - U-turn
        - Knock Off"""
    )
    mon = Pokemon(gen=9, teambuilder=tb.team[0])
    mon._species = "rillaboom"
    mon._current_hp = 100
    mon._max_hp = 100
    return mon


def _battle(tera=(True, True)) -> DoubleBattle:
    battle = DoubleBattle("tag", "me", Logger("ex"), gen=9)
    battle.player_role = "p1"
    battle._format = "gen9vgc2025regg"
    calyrex = _calyrex()
    urshifu = _urshifu()
    incineroar = _incineroar()
    rillaboom = _rillaboom()
    battle._team = {
        "p1: Calyrex": calyrex,
        "p1: Urshifu": urshifu,
        "p1: Incineroar": incineroar,
        "p1: Rillaboom": rillaboom,
    }
    battle._active_pokemon = {"p1a": calyrex, "p1b": urshifu}
    battle._available_moves = [list(calyrex.moves.values()), list(urshifu.moves.values())]
    battle._available_switches = [[incineroar, rillaboom], [incineroar, rillaboom]]
    battle._can_tera = list(tera)
    return battle


class TestParseActionHappyPaths:
    def test_two_moves_with_tera(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("astralbarrage 1 tera, surgingstrikes 1", _battle())
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.second_order is not None
        assert order.first_order.order.id == "astralbarrage"  # pyright: ignore
        assert order.first_order.move_target == 1
        assert order.first_order.terastallize is True
        assert order.second_order.order.id == "surgingstrikes"  # pyright: ignore
        assert order.second_order.move_target == 1
        assert order.second_order.terastallize is False

    def test_self_target_move_no_target_token(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("protect, aquajet -1", _battle())
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.second_order is not None
        assert order.first_order.order.id == "protect"  # pyright: ignore
        assert order.second_order.move_target == -1

    def test_switch_by_species_name(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("incineroar, surgingstrikes 1", _battle())
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.first_order.order.species == "incineroar"  # pyright: ignore

    def test_numeric_move_reference(self):
        from elitefurretai.agents._human_action_parser import parse_action

        # 1 = astralbarrage, 4 = protect (self target)
        order = parse_action("1 1, 4", _battle())
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.second_order is not None
        assert order.first_order.order.id == "astralbarrage"  # pyright: ignore
        assert order.second_order.order.id == "detect"  # pyright: ignore

    def test_pass_first_slot(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("pass, detect", _battle())
        assert isinstance(order, DoubleBattleOrder)
        assert isinstance(order.first_order, PassBattleOrder)

    def test_quit_returns_forfeit(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("quit", _battle())
        assert isinstance(order, ForfeitBattleOrder)


class TestParseActionForceSwitch:
    def test_single_forced_slot_pads_the_other(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("incineroar", _battle(), force_switch=[True, False])
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.first_order.order.species == "incineroar"  # pyright: ignore
        assert isinstance(order.second_order, PassBattleOrder)

    def test_two_forced_slots_take_two_switches(self):
        from elitefurretai.agents._human_action_parser import parse_action

        order = parse_action("incineroar, rillaboom", _battle(), force_switch=[True, True])
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.second_order is not None
        assert order.first_order.order.species == "incineroar"  # pyright: ignore
        assert order.second_order.order.species == "rillaboom"  # pyright: ignore


class TestParseActionTeraGating:
    def test_tera_silently_ignored_when_unavailable(self):
        from elitefurretai.agents._human_action_parser import parse_action

        battle = _battle(tera=(False, False))
        order = parse_action("astralbarrage 1 tera, surgingstrikes 1", battle)
        assert isinstance(order, DoubleBattleOrder)
        assert order.first_order is not None
        assert order.first_order.terastallize is False


class TestParseActionErrors:
    def test_unknown_move_raises(self):
        from elitefurretai.agents._human_action_parser import (
            ActionParseError,
            parse_action,
        )

        with pytest.raises(ActionParseError):
            parse_action("nopesuchmove 1, protect", _battle())

    def test_unknown_switch_species_raises(self):
        from elitefurretai.agents._human_action_parser import (
            ActionParseError,
            parse_action,
        )

        with pytest.raises(ActionParseError):
            parse_action("nopebenchmon, protect", _battle())

    def test_wrong_part_count_raises(self):
        from elitefurretai.agents._human_action_parser import (
            ActionParseError,
            parse_action,
        )

        with pytest.raises(ActionParseError):
            parse_action("protect, surgingstrikes 1, extra", _battle())

    def test_missing_second_action_raises(self):
        from elitefurretai.agents._human_action_parser import (
            ActionParseError,
            parse_action,
        )

        with pytest.raises(ActionParseError):
            parse_action("protect", _battle())
