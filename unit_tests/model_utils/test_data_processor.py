# -*- coding: utf-8 -*-
import os.path
from typing import Iterator
from unittest.mock import MagicMock

from poke_env.environment import PokemonType

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.data_processor import DataProcessor


def test_stream_data():
    files = [
        os.path.join("data/fixture", f)
        for f in ["gen9randombattle-222.log.json", "gen7anythinggoesanon.json"]
    ]
    dp = DataProcessor(omniscient=True, double_data=True)

    assert dp.omniscient
    assert dp.double_data

    iterator = dp.stream_data(files=files)

    assert isinstance(iterator, Iterator)
    assert len([x for x in iterator]) == 4


def test_load_data():
    files = [
        os.path.join("data/fixture", f)
        for f in ["gen9randombattle-222.log.json", "gen7anythinggoesanon.json"]
    ]
    dp = DataProcessor(omniscient=True, double_data=False)

    data = dp.load_data(files=files)

    assert len(data) == len(files)
    assert isinstance(list(data.values())[0], BattleData)
    assert data.get("battle-gen9randombattle-222", None)
    assert data["battle-gen9randombattle-222"].roomid == "battle-gen9randombattle-222"
    assert data["battle-gen9randombattle-222"].p1rating == 1000


def json_to_battledata_showdown_singles(single_battle_json):
    dp = DataProcessor(omniscient=True)
    bd1 = dp.json_to_battledata(single_battle_json, perspective="p1")
    assert bd1.format == "gen9randombattle"
    assert bd1.p1 == "RandomPlayer 16"

    # Tests we have the right player perspective, and we're recording turns well
    assert bd1.observations[2].active_pokemon.species == "armarouge"

    # Tests we have omniscient data, on both sides
    assert bd1.p1_team[0].tera_type == PokemonType.GHOST
    assert bd1.p2_team[0].item == "Heavy-Duty Boots"
    assert bd1.p2_team[1].tera_type == PokemonType.DRAGON
    assert len(bd1.p2_team[0].moves) == 4

    # Tests we're recording the right gen
    assert 9 in dp._gen_data

    bd2 = dp.json_to_battledata(single_battle_json, perspective="p2")
    assert bd2.format == "gen9randombattle"
    assert bd2.p1 == "RandomPlayer 16"

    # Test whether perspective has changed
    assert bd2.observations[2].opponent_active_pokemon.species == "armarouge"

    assert bd2.p1_team[0].tera_type == PokemonType.GHOST
    assert bd2.p2_team[0].item == "Heavy-Duty Boots"
    assert bd2.p2_team[1].tera_type == PokemonType.DRAGON
    assert len(bd2.p2_team[0].moves) == 4

    # Test whether we don't have omniscient data
    dp = DataProcessor(omniscient=False)
    bd3 = dp.json_to_battledata(single_battle_json, perspective="p1")
    assert bd3.format == "gen9randombattle"
    assert bd3.p1 == "RandomPlayer 16"
    assert bd3.p1_team[0].tera_type == PokemonType.GHOST

    # We don't know this information
    assert bd3.p2_team[1].tera_type is None
    assert bd3.p2_team[0].item is None
    assert len(bd3.p2_team[0].moves) == 0


def test_json_to_battledata_showdown_doubles(double_battle_json):
    dp = DataProcessor(omniscient=True)
    bd1 = dp.json_to_battledata(double_battle_json, perspective="p1")
    assert bd1.format == "gen6doublesou"
    assert bd1.p1 == "test-player-b"

    # Tests we have the right player perspective, and we're recording turns well
    assert bd1.observations[1].active_pokemon[0].species == "keldeoresolute"

    # Tests we have omniscient data, on both sides

    # First test that we have a Charizard and find it
    charizard_index = None
    for i, mon in enumerate(bd1.p1_team):
        if mon.species == "charizard":
            charizard_index = i
    assert charizard_index is not None

    assert bd1.p1_team[charizard_index].tera_type is None
    assert bd1.p1_team[charizard_index].item == "charizarditey"

    amoonguss_index = None
    for i, mon in enumerate(bd1.p2_team):
        if mon.species == "amoonguss":
            amoonguss_index = i
    assert amoonguss_index is not None

    assert bd1.p2_team[amoonguss_index].tera_type is None
    assert bd1.p2_team[amoonguss_index].item == "rockyhelmet"
    assert len(bd1.p2_team[amoonguss_index].moves) == 4

    # Tests we're recording the right gen
    assert 6 in dp._gen_data

    bd2 = dp.json_to_battledata(double_battle_json, perspective="p2")
    assert bd2.format == "gen6doublesou"
    assert bd2.p1 == "test-player-b"

    # Test whether perspective has changed
    assert "keldeoresolute" in map(
        lambda x: x.species, bd2.observations[1].opponent_active_pokemon
    )

    charizard_index = None
    for i, mon in enumerate(bd2.p1_team):
        if mon.species == "charizard":
            charizard_index = i
    assert charizard_index is not None

    assert bd2.p1_team[charizard_index].tera_type is None
    assert bd2.p1_team[charizard_index].item == "charizarditey"

    cresselia_index = None
    for i, mon in enumerate(bd2.p2_team):
        if mon.species == "cresselia":
            cresselia_index = i
    assert cresselia_index is not None

    assert bd1.p2_team[cresselia_index].tera_type is None
    assert bd1.p2_team[cresselia_index].item == "safetygoggles"
    assert len(bd1.p2_team[cresselia_index].moves) == 4

    # Test whether we don't have omniscient data
    dp = DataProcessor(omniscient=False)
    bd3 = dp.json_to_battledata(double_battle_json, perspective="p1")
    assert bd3.format == "gen6doublesou"
    assert bd3.p1 == "test-player-b"

    charizard_index = None
    for i, mon in enumerate(bd3.p1_team):
        if mon.species == "charizard":
            charizard_index = i
    assert charizard_index is not None

    assert bd3.p1_team[charizard_index].item == "charizarditey"

    # We don't know this information
    cresselia_index = None
    for i, mon in enumerate(bd3.p2_team):
        if mon.species == "cresselia":
            cresselia_index = i
    assert cresselia_index is not None
    assert bd3.p2_team[cresselia_index].item is None
    assert len(bd3.p2_team[cresselia_index].moves) == 0


def test_battle_data(single_battle_json_anon):
    battle_data = BattleData(
        roomid="elitefurretai",
        format=single_battle_json_anon["format"],
        p1=single_battle_json_anon["p1"],
        p2=single_battle_json_anon["p2"],
        p1_teampreview_team=single_battle_json_anon["p1team"],
        p2_teampreview_team=single_battle_json_anon["p2team"],
        p1_team=single_battle_json_anon["p1team"],
        p2_team=single_battle_json_anon["p2team"],
        p1rating=single_battle_json_anon["p1rating"],
        p2rating=single_battle_json_anon["p2rating"],
        score=single_battle_json_anon["score"],
        winner=single_battle_json_anon["winner"],
        end_type=single_battle_json_anon["endType"],
        observations={0: MagicMock()},
    )

    assert single_battle_json_anon["format"] == battle_data.format
    assert single_battle_json_anon["p1"] == battle_data.p1
    assert single_battle_json_anon["p2"] == battle_data.p2
    assert single_battle_json_anon["p1team"] == battle_data.p1_teampreview_team
    assert single_battle_json_anon["p2team"] == battle_data.p2_teampreview_team
    assert single_battle_json_anon["p1team"] == battle_data.p1_team
    assert single_battle_json_anon["p2team"] == battle_data.p2_team
    assert single_battle_json_anon["p1rating"] == battle_data.p1rating
    assert single_battle_json_anon["p2rating"] == battle_data.p2rating
    assert single_battle_json_anon["score"][0] == battle_data.score[0]
    assert single_battle_json_anon["winner"] == battle_data.winner
    assert single_battle_json_anon["endType"] == battle_data.end_type


def battle_to_json(double_battle_json):

    # 1: Read Showdown data into BattleData
    dp = DataProcessor(omniscient=True)
    bd = dp.json_to_battledata(double_battle_json, perspective="p1")

    # 2: Test BattleData
    assert double_battle_json["format"] == bd.format
    assert double_battle_json["p1"] == bd.p1
    assert double_battle_json["p2"] == bd.p2
    assert double_battle_json["p1team"] == bd.p1_team
    assert double_battle_json["p2team"] == bd.p2_team
    assert double_battle_json["p1rating"] == bd.p1rating
    assert double_battle_json["p2rating"] == bd.p2rating
    assert double_battle_json["score"][0] == bd.score[0]
    assert double_battle_json["winner"] == bd.winner
    assert double_battle_json["endType"] == bd.end_type

    # 3: Convert JSON to Battle, and back to JSON
    battle = DataProcessor.json_to_battle(double_battle_json)
    json = DataProcessor.battle_to_json(battle)

    # 4: Test that it works correctly
    assert json["p1"] == double_battle_json["p1"]
    assert json["p2"] == double_battle_json["p2"]
    assert json["p2team"] == double_battle_json["p2team"]
    assert json["winner"] == double_battle_json["winner"]
    assert json["turns"] == double_battle_json["turns"]

    assert json["p1team"][0]["species"] == double_battle_json["p1team"][0]["species"]
    assert json["p1team"][0]["item"] == double_battle_json["p1team"][0]["item"]
    assert json["p1team"][0]["moves"][0] == double_battle_json["p1team"][0]["moves"][0]
    assert json["p1team"][0]["moves"][1] == double_battle_json["p1team"][0]["moves"][1]
    assert json["p1team"][0]["moves"][2] == double_battle_json["p1team"][0]["moves"][2]
    assert json["p1team"][0]["moves"][3] == double_battle_json["p1team"][0]["moves"][3]
    assert json["p1team"][0]["level"] == double_battle_json["p1team"][0]["level"]

    assert json["p1team"][0]["nature"] == double_battle_json["p1team"][0]["nature"]

    assert json["p1team"][0]["evs"]["hp"] == double_battle_json["p1team"][0]["evs"]["hp"]
    assert json["p1team"][0]["evs"]["atk"] == double_battle_json["p1team"][0]["evs"]["atk"]
    assert json["p1team"][0]["evs"]["def"] == double_battle_json["p1team"][0]["evs"]["def"]
    assert json["p1team"][0]["evs"]["spa"] == double_battle_json["p1team"][0]["evs"]["spa"]
    assert json["p1team"][0]["evs"]["spd"] == double_battle_json["p1team"][0]["evs"]["spd"]

    assert json["p1team"][1]["species"] == double_battle_json["p1team"][1]["species"]
    assert json["eliteFurretAIGenerated"]

    for i, log in enumerate(json["logs"]):
        assert log == double_battle_json["logs"][i]

    # 5. I convert that back into BattleData
    bd2 = dp.json_to_battledata(json, perspective="p1")

    # 6. I test that everything is the same as #2
    assert bd == bd2
    assert bd.format == bd2.format
    assert bd.p1 == bd2.p1
    assert bd.p2 == bd2.p2
    assert bd.p1_team[0] == bd2.p1_team[0]
    assert bd.p2_team[0] == bd2.p2_team[0]
    assert bd.p1rating == bd2.p1rating
    assert bd.p2rating == bd2.p2rating
    assert bd.score[0] == bd2.score[0]
    assert bd.winner == bd2.winner
    assert bd.end_type == bd2.end_type

    for i in range(len(bd.p1_team)):
        compare_obs_mon(bd.p1_team[i], bd2.p2_team[i])

    for i in range(len(bd.p1_teampreview_team)):
        compare_obs_mon(bd.p1_teampreview_team[i], bd2.p2_teampreview_team[i])


def compare_obs_mon(obs_mon1, obs_mon2):
    assert obs_mon1.species == obs_mon2.species
    assert obs_mon1.level == obs_mon2.level

    assert obs_mon1.stats["hp"] == obs_mon2.stats["hp"]
    assert obs_mon1.stats["atk"] == obs_mon2.stats["atk"]
    assert obs_mon1.stats["def"] == obs_mon2.stats["def"]
    assert obs_mon1.stats["spa"] == obs_mon2.stats["spa"]
    assert obs_mon1.stats["spd"] == obs_mon2.stats["spd"]
    assert obs_mon1.stats["spe"] == obs_mon2.stats["spe"]

    # Test that we're copying the moves correctly
    assert list(obs_mon1.moves.keys())[0] == list(obs_mon2.moves.keys())[0]
    assert list(obs_mon1.moves.keys())[1] == list(obs_mon2.moves.keys())[1]
    assert list(obs_mon1.moves.keys())[2] == list(obs_mon2.moves.keys())[2]
    assert list(obs_mon1.moves.keys())[3] == list(obs_mon2.moves.keys())[3]
    assert obs_mon1.ability == obs_mon2.ability
    assert obs_mon1.item == obs_mon2.item
    assert str(obs_mon1.gender) == str(obs_mon2.gender)
    assert obs_mon1.tera_type == obs_mon2.tera_type
    assert obs_mon1.shiny == obs_mon2.shiny
