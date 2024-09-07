# -*- coding: utf-8 -*-
import pytest

from elitefurretai.utils.team_repo import TeamRepo


# helper function
def get_paste(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def test_team_repo():
    tr = TeamRepo("data/teams")

    # Test 'teams' getter
    assert "speed_ability1" in tr.teams["gen9vgc2024regg"]
    assert "s6_yan" in tr.teams["gen8vgc2020"]
    assert "s2_koutesh" in tr.teams["gen8vgc2020"]
    assert "worlds_rattle" in tr.teams["gen8vgc2022"]

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


def test_validator():
    tr = TeamRepo("data/teams")

    # test invalidity for all formats
    gibberish = get_paste("data/fixture/test_teams/gibberish.txt")
    illegal_set = get_paste("data/fixture/test_teams/illegal_moveset.txt")
    invalid_pastes = [gibberish, illegal_set]

    for paste in invalid_pastes:
        assert tr.validate_team(paste, "gen9vgc2024regg") is False

    # test validity for formats
    format_name = "gen8vgc2020"
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s2_koutesh"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s6_yan"], format_name)

    # gen8vgc2021 uses series 7/9 rules, so 8/10 should be invalid!
    format_name = "gen8vgc2021"
    s8_team = get_paste("data/fixture/test_teams/s8_duff.txt")
    assert tr.validate_team(s8_team, format_name) is False
    assert tr.validate_team(tr.teams["gen8vgc2021"]["s7_zheng"], format_name)

    format_name = "gen8vgc2022"
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_baek"], format_name)

    # Use wrong formats to test we aren't getting false positives
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg"]["miraidon_bal"], "gen9vgc2023regd")
        is False
    )
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_chua"], "gen8vgc2021") is False
