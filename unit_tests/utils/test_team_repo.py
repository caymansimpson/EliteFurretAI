# -*- coding: utf-8 -*-
import os

from elitefurretai.utils.team_repo import TeamRepo


def test_team_repo():
    tr = TeamRepo("data/teams")
    assert "gen9vgc2024regg_speed_ability1" in tr.teams
    assert "gen8vgc2020_s6_yan" in tr.teams
    assert len(tr.teams) > 5
    for team in tr.teams:
        assert isinstance(tr.teams[team], str)


# helper function
def get_paste(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def test_validator():
    teams_folder = "data/teams"
    tr = TeamRepo(teams_folder)
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
    assert tr.validate_team(tr.teams["gen8vgc2020_s2_koutesh"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2020_s3_eakes"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2020_s4_morgan"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2020_s5_tarquinio"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2020_s6_yan"], format_name) is True

    # gen8vgc2021 uses series 7/9 rules, so 8/10 should be invalid!
    format_name = "gen8vgc2021"
    s8_team = get_paste("data/fixture/test_teams/s8_duff.txt")
    s10_team = get_paste("data/fixture/test_teams/s10_patel.txt")
    assert tr.validate_team(s8_team, format_name) is False
    assert tr.validate_team(s10_team, format_name) is False
    assert tr.validate_team(tr.teams["gen8vgc2021_s7_zheng"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2021_s9_silva"], format_name) is True

    format_name = "gen8vgc2022"
    assert tr.validate_team(tr.teams["gen8vgc2022_s11_clover"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2022_s12_mott"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2022_worlds_baek"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2022_worlds_chua"], format_name) is True
    assert tr.validate_team(tr.teams["gen8vgc2022_worlds_rattle"], format_name) is True

    format_name = "gen9vgc2023regd"
    assert tr.validate_team(tr.teams["gen9vgc2023regd_worlds_kelsch"], format_name) is True
    assert (
        tr.validate_team(tr.teams["gen9vgc2023regd_worlds_nielsen"], format_name) is True
    )

    format_name = "gen9vgc2024regg"
    assert tr.validate_team(tr.teams["gen9vgc2024regg_p1"], format_name) is True
    assert tr.validate_team(tr.teams["gen9vgc2024regg_p2"], format_name) is True
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg_residual_speed1"], format_name) is True
    )
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg_residual_speed2"], format_name) is True
    )
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg_speed_ability1"], format_name) is True
    )
    assert tr.validate_team(tr.teams["gen9vgc2024regg_miraidon_bal"], format_name) is True

    # Use wrong formats to test we aren't getting flase positives
    reg_e_team = get_paste("data/fixture/test_teams/reg_e_team.txt")
    assert tr.validate_team(reg_e_team, "gen9vgc2023regd") is False
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg_miraidon_bal"], "gen9vgc2023regd")
        is False
    )
    assert (
        tr.validate_team(tr.teams["gen9vgc2024regg_miraidon_bal"], "gen8vgc2022") is False
    )
    assert (
        tr.validate_team(tr.teams["gen8vgc2022_worlds_baek"], "gen9vgc2024regg") is False
    )
    assert tr.validate_team(tr.teams["gen8vgc2022_worlds_chua"], "gen8vgc2021") is False
    assert tr.validate_team(tr.teams["gen8vgc2022_worlds_rattle"], "gen8vgc2020") is False
