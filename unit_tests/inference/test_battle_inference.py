# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from elitefurretai.inference.battle_inference import BattleInference


def test_battle_inference():
    mock = MagicMock()
    mock.gen = 9
    mock.player_username = "elitefurretai"
    battle_inference = BattleInference(mock)
    assert battle_inference


# TODO: implement
def test_is_tracking():
    raise NotImplementedError


# TODO: implement
def test_set_flag():
    raise NotImplementedError


# TODO: implement
def test_get_flag():
    raise NotImplementedError
