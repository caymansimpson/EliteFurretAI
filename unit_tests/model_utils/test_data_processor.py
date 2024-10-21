# -*- coding: utf-8 -*-
import os.path
from logging import Logger
from unittest.mock import MagicMock

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import DoubleBattle, Pokemon, PokemonGender, PokemonType
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

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


# TODO: create new logs from these two teams with different perspectives
def test_self_play_to_battle_data(vgc_battle_p1_logs, vgc_battle_p2_logs, vgc_battle_team):

    # Set up battle and players
    p1 = MagicMock()
    p2 = MagicMock()

    p1.username = "elitefurretai"
    p2.username = "CustomPlayer 1"

    # These would be normally set during requests and player._create_battle
    tag = "example-vgc2024regg-battle"
    p1_battle = DoubleBattle(tag, "elitefurretai", Logger("example"), gen=9)
    p2_battle = DoubleBattle(tag, "CustomPlayer 1", Logger("example"), gen=9)

    p1_battle.player_role = "p1"
    p2_battle.player_role = "p2"

    p1_battle.teampreview_team = set(
        [
            Pokemon(gen=9, teambuilder=tb_mon)
            for tb_mon in ConstantTeambuilder(vgc_battle_team).team
        ]
    )
    p2_battle.teampreview_team = set(
        [
            Pokemon(gen=9, teambuilder=tb_mon)
            for tb_mon in ConstantTeambuilder(vgc_battle_team).team
        ]
    )

    p1._battles = {tag: p1_battle}
    p2._battles = {tag: p2_battle}

    p1_battle.team = {"p1: " + mon.name: mon for mon in p1_battle.teampreview_team}
    p2_battle.team = {"p2: " + mon.name: mon for mon in p2_battle.teampreview_team}

    p1.prestige = 100
    p2.prestige = 150

    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

    for turn in vgc_battle_p2_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p2_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p2_battle.won_by(log[2])

    # Test Data Processor
    dp = DataProcessor()
    bd = dp.self_play_to_battle_data(p1, p2, tag)

    assert bd.format == p1_battle.format
    assert bd.end_type == "normal"
    assert bd.score == [0, 1]
    assert bd.p1rating == 100
    assert bd.p2rating == 150

    # Test correct storage of pokemon
    omon = None
    for mon in bd.p1_team:
        if mon.species == "smeargle":
            omon = mon

    assert omon.species == "smeargle"
    assert omon.item == "covertcloak"
    assert omon.ability == "moody"
    assert omon.stats == {
        "hp": 130,
        "atk": 40,
        "def": 55,
        "spa": 40,
        "spd": 65,
        "spe": 95,
    }
    assert omon.level == 50

    # omon which is from teampreview had no specified gender
    assert bd.p1_teampreview_team[0].gender is None

    # TODO: Randomly assigned gender is not caught in battle. Will need to fix
    # this if there is gender-based mechanics in the future

    # Test omniscience of battledata fields
    omon = None
    for mon in bd.p2_teampreview_team:
        if mon.species == "pikachu":
            omon = mon

    assert omon
    assert omon.species == "pikachu"
    assert omon.item == "lightball"
    assert omon.tera_type == PokemonType.ELECTRIC

    # Omniscient information also on team
    omon = None
    for mon in bd.p2_team:
        if mon.species == "pikachu":
            omon = mon
    assert omon.species == "pikachu"
    assert omon.tera_type == PokemonType.ELECTRIC

    assert bd.p1 == "elitefurretai"
    assert bd.p2 == "CustomPlayer 1"
    assert bd.winner == "p2"
    assert bd.source == BattleData.SOURCE_EFAI

    # Observations should NOT have omniscience because they come from a battle object
    # with one perspective. However, BattleData object has all omniscient characteristics
    assert bd.observations[10].opponent_team["p2: gagaga"].tera_type is None
    omon = None
    for mon in bd.p2_team:
        if mon.species == "zamazentacrowned":
            omon = mon
    assert omon.species == "zamazentacrowned"
    assert omon.tera_type == PokemonType.FIGHTING


def test_online_play_to_battle_data(vgc_battle_p1_logs, vgc_battle_team):
    # Set up battle and player
    p1 = MagicMock()
    p1.username = "elitefurretai"
    p1._team = {"team": vgc_battle_team}

    # These would be normally set during requests and player._create_battle
    tag = "example-vgc2024regg-battle"
    p1_battle = DoubleBattle(tag, "elitefurretai", Logger("example"), gen=9)
    p1_battle.player_role = "p1"
    p1_battle.teampreview_team = set(
        [
            Pokemon(gen=9, teambuilder=tb_mon)
            for tb_mon in ConstantTeambuilder(vgc_battle_team).team
        ]
    )
    p1._battles = {tag: p1_battle}
    p1_battle.team = {"p1: " + mon.name: mon for mon in p1_battle.teampreview_team}
    p1.prestige = 100

    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

    dp = DataProcessor()
    bd = dp.online_play_to_battle_data(p1, tag)

    assert bd.format == p1_battle.format
    assert bd.end_type == "normal"
    assert bd.score == [0, 1]
    assert bd.p1rating == 100
    assert bd.p2rating == 0

    # Test correct storage of pokemon
    omon = None
    for mon in bd.p1_team:
        if mon.species == "smeargle":
            omon = mon

    assert omon.species == "smeargle"
    assert omon.item == "covertcloak"
    assert omon.ability == "moody"
    assert omon.stats == {
        "hp": 130,
        "atk": 40,
        "def": 55,
        "spa": 40,
        "spd": 65,
        "spe": 95,
    }
    assert omon.level == 50

    # Test omniscience of battledata fields; should check teratype of pikachu
    omon = None
    for mon in bd.p2_teampreview_team:
        if mon.species == "pikachu":
            omon = mon

    assert omon
    assert omon.species == "pikachu"
    assert omon.item == GenData.UNKNOWN_ITEM

    omon = None
    for mon in bd.p2_team:
        if mon.species == "pikachu":
            omon = mon

    assert omon.species == "pikachu"
    assert omon.item == GenData.UNKNOWN_ITEM
    assert omon.tera_type is None

    assert bd.p1 == "elitefurretai"
    assert bd.p2 == "CustomPlayer 1"
    assert bd.winner == "p2"
    assert bd.source == BattleData.SOURCE_SHOWDOWN

    # Observations should NOT have omniscience because they come from a battle object
    # with one perspective. However, BattleData object has all omniscient characteristics
    assert bd.observations[10].opponent_team["p2: gagaga"].tera_type is None
    omon = None
    for mon in bd.p2_team:
        if mon.species == "zamazentacrowned":
            omon = mon
    assert omon.species == "zamazentacrowned"
    assert omon.tera_type is None
