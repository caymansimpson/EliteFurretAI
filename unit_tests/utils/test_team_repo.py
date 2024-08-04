# -*- coding: utf-8 -*-
import os

import pytest

from elitefurretai.utils.team_repo import TeamRepo


@pytest.fixture
def tr():
    return TeamRepo("data/teams")


# helper function
def get_paste(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def test_team_repo(tr):
    # Test 'teams' getter
    assert "speed_ability1" in tr.teams["gen9vgc2024regg"]
    assert "s6_yan" in tr.teams["gen8vgc2020"]
    assert "s2_koutesh" in tr.teams["gen8vgc2020"]
    assert "worlds_rattle" in tr.teams["gen8vgc2022"]
    print(tr.teams["gen8vgc2020"].keys())
    assert len(tr.teams["gen8vgc2022"]) > 3
    for team in tr.teams["gen8vgc2020"]:
        assert isinstance(tr.teams["gen8vgc2020"][team], str)

    # Test formats getter
    assert len(tr.formats) >= 5
    assert "gen8vgc2021" in tr.formats
    assert "gen9vgc2023regd" in tr.formats
    assert "invalidformat" not in tr.formats

    # Test 'get()'
    assert (
        tr.get(format="gen9vgc2023regd", name="worlds_kelsch")
        == tr.teams["gen9vgc2023regd"]["worlds_kelsch"]
    )
    assert (
        tr.get(format="gen8vgc2021", name="s7_zheng")
        == tr.teams["gen8vgc2021"]["s7_zheng"]
    )
    assert isinstance(tr.get("gen9vgc2023regd", "worlds_nielsen"), str)
    with pytest.raises(KeyError):
        tr.get(format="invalidformat", name="invalidteam")
        tr.get(format="gen9vgc2023regd", name="invalidteam")

    # Test 'get_all()'
    assert tr.get_all(format="gen8vgc2020") == tr.teams["gen8vgc2020"]
    assert len(tr.get_all(format="gen8vgc2022")) == len(tr.teams["gen8vgc2022"])
    assert isinstance(tr.get_all("gen8vgc2021"), dict)
    with pytest.raises(KeyError):
        tr.get_all(format="invalidformat")


def test_validator(tr):
    teams_folder = "data/teams"
    formats = os.listdir(teams_folder)

    # test invalidity for all formats
    blank = get_paste("data/fixture/test_teams/blank")
    gibberish = get_paste("data/fixture/test_teams/gibberish.txt")
    illegal_set = get_paste("data/fixture/test_teams/illegal_moveset.txt")
    invalid_pastes = [blank, gibberish, illegal_set]

    for format_name in formats:
        for paste in invalid_pastes:
            assert tr.validate_team(paste, format_name) is False

    # test validity for all formats
    format_name = "gen8vgc2020"
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s2_koutesh"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s3_eakes"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s4_morgan"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s5_tarquinio"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s6_yan"], format_name)

    # gen8vgc2021 uses series 7/9 rules, so 8/10 should be invalid!
    format_name = "gen8vgc2021"
    s8_team = get_paste("data/fixture/test_teams/s8_duff.txt")
    s10_team = get_paste("data/fixture/test_teams/s10_patel.txt")
    assert tr.validate_team(s8_team, format_name) is False
    assert tr.validate_team(s10_team, format_name) is False
    assert tr.validate_team(tr.teams["gen8vgc2021"]["s7_zheng"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2021"]["s9_silva"], format_name)

    format_name = "gen8vgc2022"
    assert tr.validate_team(tr.teams["gen8vgc2022"]["s11_clover"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2022"]["s12_mott"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_baek"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_chua"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_rattle"], format_name)

    format_name = "gen9vgc2023regd"
    assert tr.validate_team(tr.teams["gen9vgc2023regd"]["worlds_kelsch"], format_name)
    assert tr.validate_team(tr.teams["gen9vgc2023regd"]["worlds_nielsen"], format_name)

    format_name = "gen9vgc2024regg"
    # These two teams should be flagged as invalid
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["p1"], format_name) is False
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["p2"], format_name) is False
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["residual_speed1"], format_name)
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["residual_speed2"], format_name)
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["speed_ability1"], format_name)
    assert tr.validate_team(tr.teams["gen9vgc2024regg"]["miraidon_bal"], format_name)

    # Use wrong formats to test we aren't getting false positives
    reg_e_team = get_paste("data/fixture/test_teams/reg_e_team.txt")
    assert tr.validate_team(reg_e_team, "gen9vgc2023regd") is False
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg"]["miraidon_bal"], "gen9vgc2023regd")
        is False
    )
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg"]["miraidon_bal"], "gen8vgc2022")
        is False
    )
    assert (
        tr.validate_team(tr.teams["gen8vgc2022"]["worlds_baek"], "gen9vgc2024regg")
        is False
    )
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_chua"], "gen8vgc2021") is False
    assert (
        tr.validate_team(tr.teams["gen8vgc2022"]["worlds_rattle"], "gen8vgc2020") is False
    )
