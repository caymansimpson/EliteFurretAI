# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from elitefurretai.model_utils.battle_data import BattleData


# TODO: add non-anonymized data
def test_battle_data(single_battle_json_anon):
    battle_data = BattleData(
        roomid="elitefurretai",
        format=single_battle_json_anon["format"],
        p1=single_battle_json_anon["p1"],
        p2=single_battle_json_anon["p2"],
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
    assert single_battle_json_anon["p1team"] == battle_data.p1_team
    assert single_battle_json_anon["p2team"] == battle_data.p2_team
    assert single_battle_json_anon["p1rating"] == battle_data.p1rating
    assert single_battle_json_anon["p2rating"] == battle_data.p2rating
    assert single_battle_json_anon["score"][0] == battle_data.score[0]
    assert single_battle_json_anon["winner"] == battle_data.winner
    assert single_battle_json_anon["endType"] == battle_data.end_type


def test_embed_team_preview():
    raise NotImplementedError


def test_embed_turn():
    raise NotImplementedError
