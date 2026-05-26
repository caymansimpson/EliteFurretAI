# -*- coding: utf-8 -*-
import pytest

from elitefurretai.etl.team_repo import TeamRepo


# helper function
def get_paste(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def test_team_repo():
    tr = TeamRepo("data/fixture/teams")

    # Test 'teams' getter
    assert "s6_yan" in tr.teams["gen8vgc2020"]
    assert "s2_koutesh" in tr.teams["gen8vgc2020"]
    assert "worlds_rattle" in tr.teams["gen8vgc2022"]

    assert len(tr.teams["gen8vgc2022"]) > 3
    for team in tr.teams["gen8vgc2020"]:
        assert isinstance(tr.teams["gen8vgc2020"][team], str)

    # Test formats getter
    assert len(tr.formats) == 4
    assert "gen8vgc2021" in tr.formats
    assert "invalidformat" not in tr.formats

    # Test 'get()'
    assert (
        tr.get(format="gen8vgc2020", name="s2_koutesh")
        == tr.teams["gen8vgc2020"]["s2_koutesh"]
    )

    assert isinstance(tr.get("gen8vgc2020", "s2_koutesh"), str)
    with pytest.raises(KeyError):
        tr.get(format="invalidformat", name="invalidteam")
        tr.get(format="gen8vgc2020", name="invalidteam")

    # Test 'get_all()'
    assert tr.get_all(format="gen8vgc2020") == tr.teams["gen8vgc2020"]
    assert len(tr.get_all(format="gen8vgc2022")) == len(tr.teams["gen8vgc2022"])
    assert isinstance(tr.get_all("gen8vgc2021"), dict)
    with pytest.raises(KeyError):
        tr.get_all(format="invalidformat")


@pytest.mark.slow
def test_validator():
    tr = TeamRepo("data/fixture/teams", verbose=True)

    # test invalidity for all formats
    gibberish = get_paste("data/fixture/teams/test_format/gibberish.txt")
    illegal_set = get_paste("data/fixture/teams/test_format/illegal_moveset.txt")
    invalid_pastes = [gibberish, illegal_set]

    for paste in invalid_pastes:
        assert tr.validate_team(paste, "gen8vgc2020") is False

    # test validity for formats
    format_name = "gen8vgc2020"
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s2_koutesh"], format_name)
    assert tr.validate_team(tr.teams["gen8vgc2020"]["s6_yan"], format_name)

    # gen8vgc2021 uses series 7/9 rules, so 8/10 should be invalid!
    format_name = "gen8vgc2021"
    s8_team = get_paste("data/fixture/teams/test_format/s8_duff.txt")
    assert tr.validate_team(s8_team, format_name) is False
    assert tr.validate_team(tr.teams["gen8vgc2021"]["s7_zheng"], format_name)

    format_name = "gen8vgc2022"
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_baek"], format_name)

    # Use wrong formats to test we aren't getting false positives
    assert tr.validate_team(tr.teams["gen8vgc2022"]["worlds_chua"], "gen8vgc2021") is False


def test_sample_team_delegates_to_sample_team_name(tmp_path, monkeypatch):
    """sample_team must look the name up via sample_team_name (single source of
    truth for uniform sampling) and then shuffle. After the merge, calling
    sample_team_name and resolving via _teams[fmt][name] yields the same
    string sample_team returns when shuffle is disabled."""
    # Build a tiny repo with shuffle off. TeamRepo's loader skips VGC teams
    # whose Ability-line count is not 6, so each fixture lists 6 mons with an
    # Ability line apiece.
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    alpha_team = "\n\n".join(
        f"Furret {i} @ Choice Band\nAbility: Frisk\n- Quick Attack" for i in range(6)
    )
    beta_team = "\n\n".join(
        f"Sentret {i} @ Eviolite\nAbility: Run Away\n- Tackle" for i in range(6)
    )
    (fmt_dir / "alpha.txt").write_text(alpha_team)
    (fmt_dir / "beta.txt").write_text(beta_team)

    from elitefurretai.etl.team_repo import TeamRepo

    repo = TeamRepo(filepath=str(tmp_path), shuffle=False)

    # Force deterministic sample_team_name → "alpha" by stubbing random.choice
    import random as _r

    monkeypatch.setattr(_r, "choice", lambda seq: "alpha" if "alpha" in seq else seq[0])

    name = repo.sample_team_name("gen9vgc2024regg")
    team_by_lookup = repo._teams["gen9vgc2024regg"][name]
    team_via_sample = repo.sample_team("gen9vgc2024regg")
    assert team_via_sample == team_by_lookup
