"""Unit tests for Change 7 — agent-team-axis adaptive curriculum.

Each test exercises one slice of the per-(battle_format, agent_team)
sampling design from
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.
"""

from __future__ import annotations

from pathlib import Path

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import CurriculumConfig
from elitefurretai.rl.opponents import OpponentPool


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


def _two_format_repo(tmp_path) -> TeamRepo:
    """Build a TeamRepo with two formats, two teams each.

    Used by several tests in this module to exercise per-format
    isolation.
    """
    for fmt in ("gen9vgc2024regg", "gen9vgc2023regc"):
        d = tmp_path / fmt
        d.mkdir()
        _write_team_file(d / "alpha.txt", f"{fmt}/alpha")
        _write_team_file(d / "beta.txt", f"{fmt}/beta")
    return TeamRepo(filepath=str(tmp_path))


def _make_opponent_pool(
    tmp_path,
    *,
    team_axis_enabled: bool = True,
    team_warmup_threshold: int = 20,
    team_per_team_floor: float = 0.005,
    half_life: float = 50.0,
    pfsp_exponent: float = 1.0,
) -> OpponentPool:
    """Construct an OpponentPool with the multi-format test repo."""
    repo = _two_format_repo(tmp_path)
    return OpponentPool(
        curriculum={"self_play": 1.0},
        team_repo=repo,
        battle_formats={
            "gen9vgc2024regg": 0.5,
            "gen9vgc2023regc": 0.5,
        },
        opponent_team_subdirectories={
            "gen9vgc2024regg": None,
            "gen9vgc2023regc": None,
        },
        team_axis_enabled=team_axis_enabled,
        team_warmup_threshold=team_warmup_threshold,
        team_per_team_floor=team_per_team_floor,
        half_life=half_life,
        pfsp_exponent=pfsp_exponent,
    )


def test_opponent_pool_initializes_per_format_team_state(tmp_path):
    """OpponentPool builds per-format team_win_rates / sample_counts / known_teams at init."""
    pool = _make_opponent_pool(tmp_path)
    assert set(pool.known_teams.keys()) == {"gen9vgc2024regg", "gen9vgc2023regc"}
    assert set(pool.known_teams["gen9vgc2024regg"]) == {"alpha", "beta"}
    assert set(pool.known_teams["gen9vgc2023regc"]) == {"alpha", "beta"}
    # EWMA state starts at zero for every (format, team).
    for fmt, teams in pool.known_teams.items():
        for t in teams:
            assert pool.team_win_rates[fmt][t] == (0.0, 0.0)
            assert pool.team_sample_counts[fmt][t] == 0
    # Warm flag starts False for every format.
    for fmt in pool.known_teams:
        assert pool._team_axis_warm.get(fmt, False) is False


def test_record_battle_result_per_format_team_routes_correctly(tmp_path):
    """Per-(format, team) EWMA updates route correctly; other cells untouched."""
    pool = _make_opponent_pool(tmp_path, half_life=1e9)  # ~no decay

    # 10 wins, 10 losses on (gen9vgc2024regg, alpha).
    for _ in range(10):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )
    for _ in range(10):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )

    wins, n = pool.team_win_rates["gen9vgc2024regg"]["alpha"]
    assert abs(n - 20.0) < 1e-6
    assert abs(wins / n - 0.5) < 1e-6
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 20

    # Other (format, team) cells untouched.
    assert pool.team_win_rates["gen9vgc2024regg"]["beta"] == (0.0, 0.0)
    assert pool.team_win_rates["gen9vgc2023regc"]["alpha"] == (0.0, 0.0)
    assert pool.team_sample_counts["gen9vgc2024regg"]["beta"] == 0
    assert pool.team_sample_counts["gen9vgc2023regc"]["alpha"] == 0


def test_record_battle_result_forfeit_skips_team_update(tmp_path):
    """Forfeits do not update the per-team EWMA or the sample count."""
    pool = _make_opponent_pool(tmp_path, half_life=1e9)

    for _ in range(5):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )
    for _ in range(5):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=True,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )

    wins, n = pool.team_win_rates["gen9vgc2024regg"]["alpha"]
    assert abs(n - 5.0) < 1e-6
    assert abs(wins - 5.0) < 1e-6
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 5


def test_record_battle_result_no_team_args_is_noop_for_team_state(tmp_path):
    """Calls without battle_format / team_name don't touch team state.

    Preserves the existing record_battle_result contract for opponent-
    type-only tracking paths that haven't migrated yet.
    """
    pool = _make_opponent_pool(tmp_path)

    pool.record_battle_result(
        opponent_type="self_play",
        won=True,
        battle_length=10,
        forfeited=False,
    )

    for fmt, teams in pool.known_teams.items():
        for t in teams:
            assert pool.team_win_rates[fmt][t] == (0.0, 0.0)
            assert pool.team_sample_counts[fmt][t] == 0
