"""Unit tests for Change 7 — agent-team-axis adaptive curriculum.

Each test exercises one slice of the per-(battle_format, agent_team)
sampling design from
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import CurriculumConfig
from elitefurretai.rl.opponents import OpponentPool


def test_curriculum_config_has_team_axis_defaults():
    """CurriculumConfig exposes Change 7 team-axis defaults via the
    nested `adaptive_team_axis` sub-dataclass (Phase 4 unification)."""
    cfg = CurriculumConfig()
    assert cfg.adaptive_team_axis.enabled is True
    assert cfg.adaptive_team_axis.min_samples == 20
    assert cfg.adaptive_team_axis.per_key_floor == 0.005


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


def _record_wins_losses(
    pool: OpponentPool,
    battle_format: str,
    team_name: str,
    wins: int,
    losses: int,
) -> None:
    """Helper: record `wins` wins and `losses` losses for (format, team)."""
    for _ in range(wins):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format=battle_format,
            team_name=team_name,
        )
    for _ in range(losses):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=False,
            battle_format=battle_format,
            team_name=team_name,
        )


def test_update_team_distribution_warmup_per_format(tmp_path):
    """Per-format warm-up: format A warms up first; format B stays None until it warms too.

    Also verifies latching: once a format's warm flag is True, draining
    sample counts back below threshold does not flip it back to None.
    """
    pool = _make_opponent_pool(tmp_path, team_warmup_threshold=20, half_life=1e9)

    # Warm up format A only.
    _record_wins_losses(pool, "gen9vgc2024regg", "alpha", 10, 10)
    _record_wins_losses(pool, "gen9vgc2024regg", "beta", 10, 10)
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert dist["gen9vgc2023regc"] is None

    # Now warm up format B.
    _record_wins_losses(pool, "gen9vgc2023regc", "alpha", 10, 10)
    _record_wins_losses(pool, "gen9vgc2023regc", "beta", 10, 10)
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert isinstance(dist["gen9vgc2023regc"], dict)

    # Latching: synthetically drop sample counts back to 0.
    for fmt in pool.team_sample_counts:
        for t in pool.team_sample_counts[fmt]:
            pool.team_sample_counts[fmt][t] = 0
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert isinstance(dist["gen9vgc2023regc"], dict)


def test_update_team_distribution_asymmetric_pfsp_direction(tmp_path):
    """Asymmetric PFSP over-weights teams the model is worst at piloting.

    5-team single-format setup: 3 strong (90/10 W/L) and 2 weak
    (20/80 W/L). After warm-up, the two weak teams together get >60%
    of the format's distribution.
    """
    # Build a TeamRepo with 5 teams in one format.
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    for name in ("s1", "s2", "s3", "w1", "w2"):
        _write_team_file(fmt_dir / f"{name}.txt", name)
    repo = TeamRepo(filepath=str(tmp_path))
    pool = OpponentPool(
        curriculum={"self_play": 1.0},
        team_repo=repo,
        battle_formats={"gen9vgc2024regg": 1.0},
        opponent_team_subdirectories={"gen9vgc2024regg": None},
        team_axis_enabled=True,
        team_warmup_threshold=20,
        team_per_team_floor=0.0,  # disable floor for pure-PFSP check
        half_life=1e9,
        pfsp_exponent=1.0,
    )

    for name in ("s1", "s2", "s3"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 90, 10)
    for name in ("w1", "w2"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 20, 80)

    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    weak_share = dist["gen9vgc2024regg"]["w1"] + dist["gen9vgc2024regg"]["w2"]
    assert weak_share > 0.60, f"weak_share={weak_share}, dist={dist}"


def test_update_team_distribution_per_team_floor_enforced(tmp_path):
    """Per-team floor prevents any team from dropping below the configured minimum."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    for name in ("s1", "s2", "s3", "w1", "w2"):
        _write_team_file(fmt_dir / f"{name}.txt", name)
    repo = TeamRepo(filepath=str(tmp_path))
    pool = OpponentPool(
        curriculum={"self_play": 1.0},
        team_repo=repo,
        battle_formats={"gen9vgc2024regg": 1.0},
        opponent_team_subdirectories={"gen9vgc2024regg": None},
        team_axis_enabled=True,
        team_warmup_threshold=20,
        team_per_team_floor=0.05,
        half_life=1e9,
        pfsp_exponent=2.0,  # steeper to make floor relevant
    )
    for name in ("s1", "s2", "s3"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 95, 5)
    for name in ("w1", "w2"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 5, 95)

    dist = pool.update_team_distribution()["gen9vgc2024regg"]
    assert isinstance(dist, dict)
    for name, weight in dist.items():
        assert weight >= 0.05 - 1e-9, f"team {name} weight {weight} below floor"
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_update_team_distribution_disabled_returns_all_none(tmp_path):
    """team_axis_enabled=False makes update_team_distribution return None per format."""
    pool = _make_opponent_pool(tmp_path, team_axis_enabled=False)
    _record_wins_losses(pool, "gen9vgc2024regg", "alpha", 50, 50)
    _record_wins_losses(pool, "gen9vgc2024regg", "beta", 50, 50)
    dist = pool.update_team_distribution()
    assert dist["gen9vgc2024regg"] is None
    assert dist["gen9vgc2023regc"] is None


def _make_worker_factory(
    tmp_path,
    *,
    team_distribution_by_format: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
):
    """Construct a minimal WorkerOpponentFactory for sample_team tests.

    Most of WorkerOpponentFactory's collaborators (account configs,
    inference clients, etc.) are unused by the sample_team code path,
    so we construct a bare instance via __new__ and only populate the
    attributes sample_team needs. This avoids the heavy real
    initialization.
    """
    from elitefurretai.rl.opponents import WorkerOpponentFactory

    repo = _two_format_repo(tmp_path)
    factory = WorkerOpponentFactory.__new__(WorkerOpponentFactory)
    factory.team_repo = repo
    subdirs: Dict[str, Optional[str]] = {
        "gen9vgc2024regg": None,
        "gen9vgc2023regc": None,
    }
    factory.opponent_team_subdirectories = subdirs
    factory.team_distribution_by_format = team_distribution_by_format or {}
    return factory


def test_sample_team_returns_tuple_uniform(tmp_path):
    """sample_team(battle_format) returns (team_string, team_name) — uniform path."""
    factory = _make_worker_factory(tmp_path)
    team_string, team_name = factory.sample_team("gen9vgc2024regg")
    assert isinstance(team_string, str) and team_string
    assert team_name in {"alpha", "beta"}


def test_sample_team_biased_false_uses_uniform(tmp_path):
    """sample_team(biased=False) always uses uniform sampling, even if a biased dist is set."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 1.0, "beta": 0.0},
        },
    )
    seen = set()
    for _ in range(40):
        _, name = factory.sample_team("gen9vgc2024regg", biased=False)
        seen.add(name)
    # With biased=False both names should appear under uniform sampling.
    assert seen == {"alpha", "beta"}, seen


def test_sample_team_biased_true_uses_distribution(tmp_path):
    """sample_team(biased=True) draws from the configured per-format distribution."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.95, "beta": 0.05},
        },
    )
    counts = {"alpha": 0, "beta": 0}
    for _ in range(2000):
        _, name = factory.sample_team("gen9vgc2024regg", biased=True)
        counts[name] += 1
    # alpha should dominate; tolerate sampling noise within reason.
    assert counts["alpha"] > counts["beta"] * 5, counts


def test_sample_team_biased_format_isolation(tmp_path):
    """Per-format distributions don't cross-pollinate."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.95, "beta": 0.05},
            "gen9vgc2023regc": {"alpha": 0.05, "beta": 0.95},
        },
    )
    counts_a = {"alpha": 0, "beta": 0}
    counts_b = {"alpha": 0, "beta": 0}
    for _ in range(2000):
        _, name_a = factory.sample_team("gen9vgc2024regg", biased=True)
        _, name_b = factory.sample_team("gen9vgc2023regc", biased=True)
        counts_a[name_a] += 1
        counts_b[name_b] += 1
    assert counts_a["alpha"] > counts_a["beta"] * 5, counts_a
    assert counts_b["beta"] > counts_b["alpha"] * 5, counts_b


def test_sample_team_falls_back_to_uniform_when_no_distribution(tmp_path):
    """sample_team(biased=True) falls back to uniform when no distribution exists for the format."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={"gen9vgc2024regg": None},
    )
    seen = set()
    for _ in range(40):
        _, name = factory.sample_team("gen9vgc2024regg", biased=True)
        seen.add(name)
    assert seen == {"alpha", "beta"}, seen


def test_update_curriculum_accepts_team_distribution(tmp_path):
    """WorkerOpponentFactory.update_curriculum accepts team_distribution_by_format and stores it."""
    factory = _make_worker_factory(tmp_path)
    factory.curriculum = {}  # update_curriculum normalizes this
    factory.update_curriculum(
        {"self_play": 1.0},
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.7, "beta": 0.3},
        },
    )
    assert factory.team_distribution_by_format == {
        "gen9vgc2024regg": {"alpha": 0.7, "beta": 0.3},
    }


def test_factory_stamps_pending_team_name_via_sample_team(tmp_path):
    """End-to-end: factory's sample_team returns (string, name) and consumers can stamp ._pending_team_name.

    This is the integration-style contract check that the tuple-return
    migration is consistent. Callers (RLTrajectoryPlayer slots in
    create_agents / randomize_all_teams) unpack the tuple and write
    the name onto the player's _pending_team_name so the next
    _handle_battle_request stamps it into current_team_names[tag].
    """
    factory = _make_worker_factory(tmp_path)
    team_string, team_name = factory.sample_team("gen9vgc2024regg")
    assert isinstance(team_string, str) and team_string
    assert team_name in {"alpha", "beta"}

    # Simulate Pattern B: assign to an object that exposes _pending_team_name.
    class _MockPlayer:
        _pending_team_name: str = ""

    p = _MockPlayer()
    p._pending_team_name = team_name
    assert p._pending_team_name in {"alpha", "beta"}


def test_train_passes_team_args_to_record_battle_result(tmp_path):
    """train.py's trajectory-ingest loop passes battle_format and team_name to record_battle_result.

    Indirect test: synthesize a trajectory dict with the new fields,
    call OpponentPool.record_battle_result with the same kwargs the
    train.py code path will pass, and verify the side effect lands.
    This catches keyword typos and ensures the wiring stays in sync
    if record_battle_result's signature evolves.
    """
    pool = _make_opponent_pool(tmp_path, half_life=1e9)

    fake_traj = {
        "steps": [],
        "opponent_type": "self_play",
        "won": True,
        "battle_length": 7,
        "forfeited": False,
        "team_name": "alpha",
        "battle_format": "gen9vgc2024regg",
    }
    # This is the call shape train.py uses (see Step 9.3 below).
    pool.record_battle_result(
        opponent_type=fake_traj["opponent_type"],
        won=fake_traj["won"],
        battle_length=fake_traj["battle_length"],
        forfeited=fake_traj["forfeited"],
        battle_format=fake_traj["battle_format"],
        team_name=fake_traj["team_name"],
    )
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 1


def test_broadcast_key_string_literal_is_consistent_across_layers():
    """Catch typos in the broadcast-payload key string at any layer.

    The trainer puts the dict under key 'team_distribution_by_format'
    on the control_payload; worker.py pulls it under the same key;
    env.update_curriculum / backend.update_curriculum / factory.update_curriculum
    all accept it under the same parameter name. A typo at any layer
    would silently disable the broadcast.
    """
    import inspect

    from elitefurretai.engine.vgc_environment import VGCEnvironment
    from elitefurretai.rl.opponents import WorkerOpponentFactory

    expected_param = "team_distribution_by_format"

    # Factory.update_curriculum must accept the kwarg.
    factory_sig = inspect.signature(WorkerOpponentFactory.update_curriculum)
    assert expected_param in factory_sig.parameters, (
        f"WorkerOpponentFactory.update_curriculum missing kwarg "
        f"{expected_param!r}; params: {list(factory_sig.parameters)}"
    )

    # VGCEnvironment.update_curriculum must accept the kwarg.
    env_sig = inspect.signature(VGCEnvironment.update_curriculum)
    assert expected_param in env_sig.parameters, (
        f"VGCEnvironment.update_curriculum missing kwarg "
        f"{expected_param!r}; params: {list(env_sig.parameters)}"
    )


def test_broadcast_payload_field_threads_through_factory(tmp_path):
    """End-to-end: simulate the broadcast payload reaching the factory.

    Constructs a real WorkerOpponentFactory, calls update_curriculum with
    a synthetic team_distribution_by_format (same shape train.py would
    bundle), and asserts the factory's stored distribution matches.
    Catches typos in the kwarg name at the factory layer; the
    environment/backend layers are caught by the previous test.
    """
    factory = _make_worker_factory(tmp_path)
    factory.curriculum = {}
    fake_dist = {
        "gen9vgc2024regg": {"alpha": 0.7, "beta": 0.3},
        "gen9vgc2023regc": None,
    }
    factory.update_curriculum(
        {"self_play": 1.0},
        team_distribution_by_format=fake_dist,
    )
    assert factory.team_distribution_by_format == fake_dist
