# -*- coding: utf-8 -*-
import os.path
from typing import Iterator

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


def test_process_battle_singles(single_battle_json):
    dp = DataProcessor(omniscient=True)
    bd1 = dp._process_battle(single_battle_json, perspective="p1")
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

    bd2 = dp._process_battle(single_battle_json, perspective="p2")
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
    bd3 = dp._process_battle(single_battle_json, perspective="p1")
    assert bd3.format == "gen9randombattle"
    assert bd3.p1 == "RandomPlayer 16"
    assert bd3.p1_team[0].tera_type == PokemonType.GHOST

    # We don't know this information
    assert bd3.p2_team[1].tera_type is None
    assert bd3.p2_team[0].item is None
    assert len(bd3.p2_team[0].moves) == 0


def test_process_battle_doubles(double_battle_json):
    dp = DataProcessor(omniscient=True)
    bd1 = dp._process_battle(double_battle_json, perspective="p1")
    assert bd1.format == "gen6doublesou"
    assert bd1.p1 == "test-player-b"

    # Tests we have the right player perspective, and we're recording turns well
    assert bd1.observations[1].active_pokemon[0].species == "keldeoresolute"

    # Tests we have omniscient data, on both sides
    assert bd1.p1_team[0].tera_type is None
    assert bd1.p1_team[0].item == "charizarditey"
    assert bd1.p2_team[0].tera_type is None
    assert bd1.p2_team[0].item == "rockyhelmet"
    assert len(bd1.p2_team[0].moves) == 4

    # Tests we're recording the right gen
    assert 6 in dp._gen_data

    bd2 = dp._process_battle(double_battle_json, perspective="p2")
    assert bd2.format == "gen6doublesou"
    assert bd2.p1 == "test-player-b"

    # Test whether perspective has changed
    assert bd2.observations[1].opponent_active_pokemon[0].species == "keldeoresolute"

    assert bd1.p1_team[0].tera_type is None
    assert bd1.p1_team[0].item == "charizarditey"
    assert bd1.p2_team[0].tera_type is None
    assert bd1.p2_team[0].item == "rockyhelmet"
    assert len(bd1.p2_team[0].moves) == 4

    # Test whether we don't have omniscient data
    dp = DataProcessor(omniscient=False)
    bd3 = dp._process_battle(double_battle_json, perspective="p1")
    assert bd3.format == "gen6doublesou"
    assert bd3.p1 == "test-player-b"
    assert bd1.p1_team[0].item == "charizarditey"

    # We don't know this information
    assert bd3.p2_team[0].item is None
    assert len(bd3.p2_team[0].moves) == 0


def test_pokemon_to_json(double_battle_json):
    raise NotImplementedError


def battle_to_json(double_battle_json):
    raise NotImplementedError
