"""Unit tests for Change 7 — agent-team-axis adaptive curriculum.

Each test exercises one slice of the per-(battle_format, agent_team)
sampling design from
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.
"""

from __future__ import annotations

from pathlib import Path

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import CurriculumConfig


def test_curriculum_config_has_team_axis_defaults():
    """CurriculumConfig exposes three new team-axis fields with documented defaults."""
    cfg = CurriculumConfig()
    assert cfg.team_axis_enabled is True
    assert cfg.team_warmup_threshold == 20
    assert cfg.team_per_team_floor == 0.005


def _write_team_file(path: Path, name: str) -> None:
    """Write a placeholder team file with the given filename stem.

    TeamRepo's VGC-format loader requires six Pokemon entries per team
    (it counts ``Ability:`` lines). We write six identical Pikachu blocks
    so the file passes the count check; only the filename matters for
    sampling tests.
    """
    pokemon_block = (
        "Pikachu @ Light Ball\nAbility: Static\nLevel: 50\n"
        "EVs: 252 Atk / 4 Def / 252 Spe\nNature: Jolly\n- Volt Tackle\n"
    )
    path.write_text("\n".join([pokemon_block] * 6))


def test_sample_team_name_uniform_returns_known_name(tmp_path):
    """sample_team_name draws from the configured format and returns a known name."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    _write_team_file(fmt_dir / "team_alpha.txt", "team_alpha")
    _write_team_file(fmt_dir / "team_beta.txt", "team_beta")
    repo = TeamRepo(filepath=str(tmp_path))

    seen = set()
    for _ in range(50):
        seen.add(repo.sample_team_name("gen9vgc2024regg"))

    assert seen <= {"team_alpha", "team_beta"}
    assert len(seen) >= 1  # at least one name returned across 50 draws


def test_sample_team_name_respects_subdirectory(tmp_path):
    """sample_team_name restricts draws to the specified subdirectory."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    sub_dir = fmt_dir / "constrained"
    sub_dir.mkdir(parents=True)
    other_dir = fmt_dir / "other"
    other_dir.mkdir()
    _write_team_file(sub_dir / "in_pool.txt", "constrained/in_pool")
    _write_team_file(other_dir / "out_of_pool.txt", "other/out_of_pool")
    repo = TeamRepo(filepath=str(tmp_path))

    for _ in range(50):
        name = repo.sample_team_name("gen9vgc2024regg", subdirectory="constrained")
        assert name.startswith("constrained/"), name
