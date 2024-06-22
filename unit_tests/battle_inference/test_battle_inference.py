# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from elitefurretai.battle_inference import BattleInference


def test_battle_inference():
    battle_inference = BattleInference(MagicMock())
    assert battle_inference
    raise NotImplementedError
