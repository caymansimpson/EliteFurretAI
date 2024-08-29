# -*- coding: utf-8 -*-
import os.path
from unittest.mock import MagicMock

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import PokemonGender, PokemonType

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.data_processor import DataProcessor


def get_bd_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        "data/fixture/test_battledata",
    )


def test_write_battle_data_to_file():
    bd = BattleData(
        roomid="test",
        format="gen9vgc2024regg",
        p1="elitefurretai",
        p2="joeschmoe",
        p1_teampreview_team=[],
        p2_teampreview_team=[],
        p1_team=[],
        p2_team=[],
        p1rating=1000,
        p2rating=1000,
        score=[0, 0],
        winner="elitefurretai",
        end_type="",
        observations={},
        inputs={},
        source="test",
    )
    dp = DataProcessor()
    dp.write_battle_data_to_file(bd, os.path.join(get_bd_path(), "test_write.pickle"))
    assert os.path.exists(os.path.join(get_bd_path(), "test_write.pickle"))


def test_convert_anonymous_showdown_log_to_battledata(single_battle_json_anon):
    dp = DataProcessor()
    bd = dp.convert_anonymous_showdown_log_to_battledata(single_battle_json_anon)

    assert bd.format == single_battle_json_anon["format"]
    assert bd.end_type == single_battle_json_anon["endType"]
    assert bd.score == single_battle_json_anon["score"]
    assert bd.p1rating == single_battle_json_anon["p1rating"]
    assert bd.p2rating == single_battle_json_anon["p2rating"]
    omon = bd.p1_teampreview_team[0]
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

    assert (
        bd.p2_teampreview_team[0].species
        == single_battle_json_anon["p2team"][0]["species"]
    )
    assert bd.p2_teampreview_team[0].item == single_battle_json_anon["p2team"][0]["item"]

    assert bd.p1_team[0].species == single_battle_json_anon["p1team"][0]["species"]
    assert bd.p2_team[0].species == single_battle_json_anon["p2team"][0]["species"]

    assert bd.p1 == single_battle_json_anon["p1"]
    assert bd.p2 == single_battle_json_anon["p2"]
    assert bd.winner == single_battle_json_anon["winner"]

    # Because we count and record a 0th turn
    assert len(bd.observations) == single_battle_json_anon["turns"] + 1

    assert bd.source == BattleData.SOURCE_SHOWDOWN_ANON

    # Will fail once I implement inputs. This will force me to add the test
    assert len(bd.inputs) == 39

    dp.write_battle_data_to_file(
        bd, os.path.join(get_bd_path(), "single_battle_json_anon.pickle")
    )
    assert os.path.exists(os.path.join(get_bd_path(), "single_battle_json_anon.pickle"))


# Need to first pickle the data
def test_stream_data():
    files = [
        os.path.join(get_bd_path(), "test_write.pickle"),
        os.path.join(get_bd_path(), "single_battle_json_anon.pickle"),
    ]
    dp = DataProcessor()

    data = [bd for bd in dp.stream_data(files=files)]

    assert len(data) == len(files)
    assert isinstance(data[0], BattleData)
    assert data[0].format == "gen9vgc2024regg"
    assert data[1].format == "gen7anythinggoes"


def test_self_play_to_battle_data(vgc_battle_p1, vgc_battle_p2):
    p1 = MagicMock()
    p1._battles = {vgc_battle_p1.battle_tag: vgc_battle_p1}
    p1.prestige = 100

    p2 = MagicMock()
    p2._battles = {vgc_battle_p2.battle_tag: vgc_battle_p2}
    p2.prestige = 150

    dp = DataProcessor()
    bd = dp.self_play_to_battle_data(p1, p2, vgc_battle_p1.battle_tag)

    assert bd.format == vgc_battle_p1.format
    assert bd.end_type == "normal"
    assert bd.score == [1, 0]
    assert bd.p1rating == 100
    assert bd.p2rating == 150

    # Test correct storage of pokemon
    omon = bd.p1_team[1]
    assert omon.species == "smeargle"
    assert omon.item == "blacksludge"
    assert omon.ability == "moody"
    assert omon.stats == {
        "hp": 130,
        "atk": 72,
        "def": 55,
        "spa": 36,
        "spd": 66,
        "spe": 139,
    }
    assert omon.level == 50

    # omon which is from teampreview had no specified gender
    assert bd.p1_teampreview_team[0].gender is None

    # Gender is randomly assigned in battle
    assert bd.observations[1].team["p1: Smeargle"].gender == PokemonGender.MALE

    # Gender should be picked up by team
    assert omon.gender == PokemonGender.MALE

    # Test omniscience of battledata fields; should check teratype of wo-chien
    omon = None
    for mon in bd.p2_teampreview_team:
        if mon.species == "sentret":
            omon = mon

    assert omon
    assert omon.species == "sentret"
    assert omon.item == "kingsrock"
    assert bd.p1_team[3].species == "sentret"
    assert bd.p2_team[3].item == "kingsrock"
    assert bd.p2_team[3].tera_type == PokemonType.NORMAL

    assert bd.p1 == "elitefurretai"
    assert bd.p2 == "CustomPlayer 1"
    assert bd.winner == "p1"
    assert bd.source == BattleData.SOURCE_EFAI

    # Observations should NOT have omniscience because they come from a battle object
    # with one perspective. However, BattleData object has all omniscient characteristics
    assert bd.observations[10].opponent_team["p2: Wo-Chien"].tera_type is None
    assert bd.p2_team[2].species == "wochien"
    assert bd.p2_team[2].tera_type == PokemonType.DARK


# TODO: basically copy everything above, but ensure we don't have omniscience and only things we've observed
def test_online_play_to_battle_data(vgc_battle_p1):
    p1 = MagicMock()
    p1._battles = {vgc_battle_p1.battle_tag: vgc_battle_p1}
    p1.prestige = 100

    dp = DataProcessor()
    bd = dp.online_play_to_battle_data(p1, vgc_battle_p1.battle_tag)

    assert bd.format == vgc_battle_p1.format
    assert bd.end_type == "normal"
    assert bd.score == [1, 0]
    assert bd.p1rating == 100
    assert bd.p2rating == 0

    # Test correct storage of pokemon
    omon = bd.p1_team[1]
    assert omon.species == "smeargle"
    assert omon.item == "blacksludge"
    assert omon.ability == "moody"
    assert omon.stats == {
        "hp": 130,
        "atk": 72,
        "def": 55,
        "spa": 36,
        "spd": 66,
        "spe": 139,
    }
    assert omon.level == 50

    # omon which is from teampreview had no specified gender
    assert bd.p1_teampreview_team[0].gender is None

    # Gender is randomly assigned in battle
    assert bd.observations[1].team["p1: Smeargle"].gender == PokemonGender.MALE

    # Gender should be picked up by team
    assert omon.gender == PokemonGender.MALE

    # Test omniscience of battledata fields; should check teratype of wo-chien
    omon = None
    for mon in bd.p2_teampreview_team:
        if mon.species == "sentret":
            omon = mon

    assert omon
    assert omon.species == "sentret"
    assert omon.item == GenData.UNKNOWN_ITEM
    assert bd.p2_team[2].species == "sentret"
    assert bd.p2_team[2].item == GenData.UNKNOWN_ITEM
    assert bd.p2_team[2].tera_type is None

    assert bd.p1 == "elitefurretai"
    assert bd.p2 == "CustomPlayer 1"
    assert bd.winner == "p1"
    assert bd.source == BattleData.SOURCE_SHOWDOWN

    # Observations should NOT have omniscience because they come from a battle object
    # with one perspective. However, BattleData object has all omniscient characteristics
    assert bd.observations[10].opponent_team["p2: Wo-Chien"].tera_type is None
    assert bd.p2_team[3].species == "wochien"
    assert bd.p2_team[3].tera_type is None
