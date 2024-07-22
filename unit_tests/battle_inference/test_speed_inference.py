# -*- coding: utf-8 -*-
"""Unit Tests to ensure InfostateNetwork is good
"""
from elitefurretai.battle_inference.speed_inference import SpeedInference


# Will fail once I implement, which is intended
def test_speed_inference():
    # TODO: generate a bunch of edge-case speed data
    # Test whether speedparser can get it all since we have the ground truth
    raise NotImplementedError


# TODO:
def test_clean_orders():
    orders = [
        [("p2: Ting-Lu", 1.0), ("p1: Raichu", 1.0), ("p1: Wo-Chien", 0.67)],
        [("p1b: Raichu", 1.0)],
        [("p1a: Wo-Chien", 1.0), ("p2b: Ting-Lu", 1.0)],
        [("p1a: Wo-Chien", 1.0), ("p2b: Ting-Lu", 1.0), ("p1b: Raichu", 0.25)],
    ]

    cleaned = SpeedInference.clean_orders(orders)
    assert [("p2: Ting-Lu", 1.0), ("p1: Raichu", 1.0)] in cleaned
    assert [("p1: Raichu", 1.0), ("p1: Wo-Chien", 0.67)] in cleaned
    assert [("p1: Wo-Chien", 1.0), ("p2: Ting-Lu", 1.0)] in cleaned
    assert [("p1: Wo-Chien", 1.0), ("p2: Ting-Lu", 1.0)] in cleaned
    assert [("p2: Ting-Lu", 1.0), ("p1: Raichu", 0.25)] in cleaned
    assert len(cleaned) == 5
