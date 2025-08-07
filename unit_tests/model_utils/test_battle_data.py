# -*- coding: utf-8 -*-
import os.path

import orjson
from poke_env.battle import PokemonGender
from poke_env.data.normalize import to_id_str

from elitefurretai.model_utils.battle_data import BattleData


def get_bd_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "data/fixture/test_battledata",
    )


def test_write_and_load():
    bd = BattleData(
        battle_tag="test",
        format="gen9vgc2024regg",
        p1="elitefurretai",
        p2="joeschmoe",
        p1_team=[],
        p2_team=[],
        p1_rating=1000,
        p2_rating=1000,
        turns=0,
        score=[0, 0],
        winner="elitefurretai",
        end_type="",
        logs=[],
        input_logs=[],
    )

    bd.save(os.path.join(get_bd_path(), "test_write.json"))
    assert os.path.exists(os.path.join(get_bd_path(), "test_write.json"))

    with open(os.path.join(get_bd_path(), "test_write.json")) as f:
        data = orjson.loads(f.read())
        assert data["format"] == "gen9vgc2024regg"
        assert data["p1"] == "elitefurretai"
        assert data["p2"] == "joeschmoe"
        assert data["turns"] == 0
        assert data["score"] == [0, 0]
        assert data["winner"] == "elitefurretai"
        assert data["end_type"] == ""
        assert data["logs"] == []
        assert data["input_logs"] == []
        assert data["p1_rating"] == 1000
        assert data["p2_rating"] == 1000


def test_convert_anonymous_showdown_log_to_battledata(single_battle_json_anon):
    bd = BattleData.from_showdown_json(single_battle_json_anon)
    assert bd.format == single_battle_json_anon["format"]
    assert bd.end_type == single_battle_json_anon["endType"]
    assert bd.score == single_battle_json_anon["score"]
    assert bd.p1_rating == 1900
    assert bd.p2_rating == 1450
    omon = bd.p1_team[0]
    assert omon.species == single_battle_json_anon["p1team"][0]["species"]
    assert omon.item == single_battle_json_anon["p1team"][0]["item"]
    assert omon.ability == to_id_str(single_battle_json_anon["p1team"][0]["ability"])
    assert omon.stats == {
        "hp": 397,
        "atk": 284,
        "def": 146,
        "spa": 174,
        "spd": 147,
        "spe": 226,
    }
    assert omon.level == single_battle_json_anon["p1team"][0]["level"]
    assert omon.gender == PokemonGender.NEUTRAL

    assert bd.p2_team[0].species == single_battle_json_anon["p2team"][0]["species"]
    assert bd.p2_team[0].item == single_battle_json_anon["p2team"][0]["item"]

    assert bd.p1_team[0].species == single_battle_json_anon["p1team"][0]["species"]
    assert bd.p2_team[0].species == single_battle_json_anon["p2team"][0]["species"]

    assert bd.p1 == single_battle_json_anon["p1"]
    assert bd.p2 == single_battle_json_anon["p2"]
    assert bd.winner == single_battle_json_anon["winner"]
    assert bd.turns == single_battle_json_anon["turns"]
    assert len(bd.input_logs) == 39


def test_saving_and_loading_showdown_anon(
    single_battle_json_anon, vgc_json_anon, vgc_json_anon2
):
    for i, json in enumerate([single_battle_json_anon, vgc_json_anon, vgc_json_anon2]):
        bd = BattleData.from_showdown_json(json)

        # Test saving and loading
        filename = f"test_write_{i}.json"
        bd.save(os.path.join(get_bd_path(), filename))
        assert os.path.exists(os.path.join(get_bd_path(), filename))

        with open(os.path.join(get_bd_path(), filename)) as f:
            data = orjson.loads(f.read())
            bd2 = BattleData.from_elite_furret_ai_json(data)
            assert bd.to_dict() == bd2.to_dict()
