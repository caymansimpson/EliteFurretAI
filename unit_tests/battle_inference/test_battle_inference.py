# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

import pytest
from poke_env.data.gen_data import GenData
from poke_env.environment import DoubleBattle, Observation, Pokemon

from elitefurretai.battle_inference.battle_inference import BattleInference


def test_battle_inference():
    mock = MagicMock()
    mock.gen = 9
    mock.player_username = "elitefurretai"
    battle_inference = BattleInference(mock)
    assert battle_inference


def test_check_items_covertcloak():

    # icywind
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {
        Pokemon(gen=9, species="Furret"),
        Pokemon(gen=9, species="Shuckle"),
    }

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.get_item("p2: Shuckle") == "covertcloak"
    assert bi.get_item("p2: Furret") == "airballoon"

    # bulldoze
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Bulldoze", "p2a: Furret", "[spread] p1b,p2a,p2b"],
        ["", "-damage", "p1b: Raichu", "97/100"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p1b: Raichu", "spe", "1"],
        ["", "-unboost", "p2b: Shuckle", "spe", "1"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {
        Pokemon(gen=9, species="Furret"),
        Pokemon(gen=9, species="Shuckle"),
    }

    bi = BattleInference(battle)
    with pytest.raises(ValueError):
        bi.check_items(Observation(events=events))

    # fakeout
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "move", "p1a: Smeargle", "Fake Out", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {
        Pokemon(gen=9, species="Furret"),
        Pokemon(gen=9, species="Shuckle"),
    }

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.get_item("p2: Furret") == "covertcloak"
    assert bi.get_item("p2: Shuckle") == GenData.UNKNOWN_ITEM

    # nuzzle
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "move", "p1b: Raichu", "Nuzzle", "p2b: Shuckle"],
        ["", "-damage", "p2b: Shuckle", "98/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {
        Pokemon(gen=9, species="Furret"),
        Pokemon(gen=9, species="Shuckle"),
    }

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.get_item("p2: Shuckle") == "covertcloak"
    assert bi.get_item("p2: Furret") == GenData.UNKNOWN_ITEM


def test_check_items_lightclay():
    # test lightclay
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p1a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "turn", "1"],
        ["", "turn", "2"],
        ["", "turn", "3"],
        ["", "turn", "4"],
        ["", "turn", "5"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {Pokemon(gen=9, species="Furret")}

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.get_item("p2: Furret") == GenData.UNKNOWN_ITEM
    battle.parse_message(["", "turn", "6"])
    bi.check_items(Observation(events=["", "turn", "6"]))
    assert bi.get_item("p2: Furret") == "lightclay"

    # Test setting it twice
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "switch", "p2b: Sentret", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p1a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "move", "p2b: Sentret", "Reflect", "[still]"],
        ["", "-fail", "p2b: Sentret"],
        ["", "turn", "1"],
        ["", "turn", "2"],
        ["", "turn", "3"],
        ["", "turn", "4"],
        ["", "move", "p2a: Furret", "Reflect", "[still]"],
        ["", "-fail", "p2a: Furret"],
        ["", "turn", "5"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {
        Pokemon(gen=9, species="Furret"),
        Pokemon(gen=9, species="Sentret"),
    }

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.get_item("p2: Furret") == GenData.UNKNOWN_ITEM
    assert bi.get_item("p2: Sentret") == GenData.UNKNOWN_ITEM
    battle.parse_message(["", "turn", "6"])
    bi.check_items(Observation(events=["", "turn", "6"]))
    assert bi.get_item("p2: Furret") == "lightclay"
    assert bi.get_item("p2: Sentret") == GenData.UNKNOWN_ITEM


def test_check_items_can_have_choice_item():
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {Pokemon(gen=9, species="Furret")}

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.can_have_choice_item("p2: Furret")

    new_events = [
        ["", "turn", "1"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in new_events:
        battle.parse_message(event)
    bi.check_items(Observation(events=new_events))
    assert bi.can_have_choice_item("p2: Furret")

    newer_events = [
        ["", "turn", "2"],
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
    ]
    for event in newer_events:
        battle.parse_message(event)
    bi.check_items(Observation(events=newer_events))
    assert not bi.can_have_choice_item("p2: Furret")


def test_check_items_can_have_assault_vest():
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in events:
        battle.parse_message(event)

    battle._teampreview_opponent_team = {Pokemon(gen=9, species="Furret")}

    bi = BattleInference(battle)
    bi.check_items(Observation(events=events))
    assert bi.can_have_assault_vest("p2: Furret")

    new_events = [
        ["", "turn", "1"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    for event in new_events:
        battle.parse_message(event)
    bi.check_items(Observation(events=new_events))
    assert bi.can_have_assault_vest("p2: Furret")

    newer_events = [
        ["", "turn", "2"],
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
    ]
    for event in newer_events:
        battle.parse_message(event)
    bi.check_items(Observation(events=newer_events))
    assert not bi.can_have_assault_vest("p2: Furret")
