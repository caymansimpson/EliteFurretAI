# -*- coding: utf-8 -*-
"""Tests for ``team_provider.parse_team_specification``."""

from __future__ import annotations

from pathlib import Path

import pytest

from elitefurretai.rl.analyze.analysis_utils import parse_team_specification

_SAMPLE_MON = """\
Calyrex-Shadow @ Life Orb
Ability: As One (Spectrier)
Tera Type: Normal
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Astral Barrage
- Psychic
- Nasty Plot
- Protect
"""

# TeamRepo rejects VGC teams without exactly 6 "Ability:" lines, so the
# fixture pads with copies of the same mon. The contents are never used
# in battle by these unit tests.
_SAMPLE_TEAM = "\n\n".join([_SAMPLE_MON] * 6)


def test_file_provider_returns_fixed_team(tmp_path):
    f = tmp_path / "team.txt"
    f.write_text(_SAMPLE_TEAM)
    provider = parse_team_specification(str(f), battle_format="gen9vgc2024regg")
    assert provider() == _SAMPLE_TEAM
    # Calling twice returns the same team — fixed source.
    assert provider() == _SAMPLE_TEAM


def test_directory_provider_samples_team(tmp_path):
    # Build a tiny team repo: data/teams/<format>/team.txt structure.
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    (fmt_dir / "t1.txt").write_text(_SAMPLE_TEAM)

    provider = parse_team_specification(str(tmp_path), battle_format="gen9vgc2024regg")
    out = provider()
    # TeamRepo shuffles Pokemon order by default, so we just check that
    # the sampled team contains the expected number of mons.
    assert out.strip().count("Ability:") == 6


def test_none_falls_back_to_default(monkeypatch, tmp_path):
    # Stub the default-provider's underlying repo so we don't depend on
    # the real data/teams directory in unit tests.
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    (fmt_dir / "t1.txt").write_text(_SAMPLE_TEAM)

    # The default provider constructs TeamRepo(filepath="data/teams").
    # Patch the constructor in the team_provider module to redirect.
    from elitefurretai.etl import TeamRepo

    real_init = TeamRepo.__init__

    def fake_init(self, filepath, *args, **kwargs):
        # Redirect default "data/teams" to our tmp_path.
        redirected = str(tmp_path) if filepath == "data/teams" else filepath
        return real_init(self, redirected, *args, **kwargs)

    monkeypatch.setattr(TeamRepo, "__init__", fake_init)

    provider = parse_team_specification(None, battle_format="gen9vgc2024regg")
    out = provider()
    # TeamRepo shuffles Pokemon order by default, so we just check that
    # the sampled team contains the expected number of mons.
    assert out.strip().count("Ability:") == 6


def test_invalid_path_raises(tmp_path):
    bogus = tmp_path / "definitely_does_not_exist"
    with pytest.raises(ValueError, match="not a file and not a directory"):
        parse_team_specification(str(bogus), battle_format="gen9vgc2024regg")


def test_empty_string_falls_back_to_default(monkeypatch, tmp_path):
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    (fmt_dir / "t1.txt").write_text(_SAMPLE_TEAM)

    from elitefurretai.etl import TeamRepo

    real_init = TeamRepo.__init__

    def fake_init(self, filepath, *args, **kwargs):
        redirected = str(tmp_path) if filepath == "data/teams" else filepath
        return real_init(self, redirected, *args, **kwargs)

    monkeypatch.setattr(TeamRepo, "__init__", fake_init)

    provider = parse_team_specification("", battle_format="gen9vgc2024regg")
    # TeamRepo shuffles by default; verify the team came through by mon count.
    assert provider().strip().count("Ability:") == 6


def test_real_vgcbench_team_file_loads():
    """Smoke check against the actual repo team file used by the acceptance test."""
    team_path = Path("data/teams/gen9vgc2024regg/vgcbench.txt")
    if not team_path.is_file():
        pytest.skip("vgcbench.txt not present in working dir")
    provider = parse_team_specification(str(team_path), battle_format="gen9vgc2024regg")
    out = provider()
    assert len(out) > 0
    assert provider() == out  # stable
