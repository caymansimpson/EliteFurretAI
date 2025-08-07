# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

import pytest
from poke_env.battle import DoubleBattle, Pokemon

from elitefurretai.inference.battle_inference import BattleInference


def get_battle_inference():
    furret = Pokemon(gen=9, details="Furret, L50, F")
    sentret = Pokemon(gen=9, details="Sentret, L50, F")
    battle = DoubleBattle("tag", "elitefurretai", MagicMock(), gen=9)
    battle._player_role = "p2"
    battle_inference = BattleInference(battle)
    battle_inference._opponent_mons = {
        "p1: Furret": BattleInference.load_opponent_set(furret)
    }
    battle._opponent_team = {"p1: Sentret": sentret, "p1: Furret": furret}
    return battle_inference


def test_battle_inference():
    battle_inference = get_battle_inference()
    assert battle_inference


def test_is_tracking():
    battle_inference = get_battle_inference()

    # Test if not in opponent team
    assert battle_inference.is_tracking("p1: Furret")
    assert not battle_inference.is_tracking("p1: Sentret")

    battle_inference.get_flag("p1: Sentret", "can_be_choice")
    assert battle_inference.is_tracking("p1: Sentret")

    # Test if my team
    assert not battle_inference.is_tracking("p2: Sentret")


def test_set_flag():
    battle_inference = get_battle_inference()

    assert battle_inference.get_flag("p1: Furret", "can_be_choice")
    battle_inference.set_flag("p1: Furret", "can_be_choice", False)
    assert not battle_inference.get_flag("p1: Furret", "can_be_choice")

    # Test trying to set a flag of a mon not in opponent team
    with pytest.raises(KeyError):
        battle_inference.set_flag("p1: Arceus", "can_be_choice", True)

    # Test trying to set a flag of a mon thats not being tracked
    battle_inference.set_flag("p1: Sentret", "can_be_choice", False)
    assert not battle_inference.get_flag("p1: Sentret", "can_be_choice")

    # Try setting a flag that im not expecting
    with pytest.raises(KeyError):
        battle_inference.set_flag("p1: Furret", "dope_af", True)


def test_get_flag():
    battle_inference = get_battle_inference()

    battle_inference.set_flag("p1: Furret", "can_be_choice", False)
    assert not battle_inference.get_flag("p1: Furret", "can_be_choice")

    # Test trying to get a flag im not tracking
    with pytest.raises(KeyError):
        battle_inference.get_flag("p1: Furret", "dope_af")

    # Test trying to get a flag of a mon not tracking
    assert battle_inference.get_flag("p1: Sentret", "can_be_choice")

    # Test trying to get a flag of a mon thats not being tracked
    with pytest.raises(KeyError):
        battle_inference.get_flag("p1: Arceus", "can_be_choice")
