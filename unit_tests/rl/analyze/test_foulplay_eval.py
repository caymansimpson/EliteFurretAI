# -*- coding: utf-8 -*-
"""Unit tests for the foulplay_eval driver.

Mocks ``run_eval_parallel`` so tests run without a live Showdown
server or the FoulPlay venv. The end-to-end smoke test
(planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md
Task 9) verifies the real flow.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from elitefurretai.rl.analyze.evaluate import EvalResult
from elitefurretai.rl.analyze.foulplay_eval import FoulplayEvalResult, run
from elitefurretai.rl.config import CurriculumConfig, FoulplayEvalConfig


def _make_curriculum_single(tmp_path) -> CurriculumConfig:
    base = tmp_path / "teams"
    fmt_dir = base / "gen9vgc2024regg"
    pool_dir = fmt_dir / "constrained"
    pool_dir.mkdir(parents=True)
    (pool_dir / "team_a.txt").write_text("team a body")
    agent_dir = fmt_dir / "agent"
    agent_dir.mkdir()
    (agent_dir / "team_a.txt").write_text("agent team body")
    return CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 1.0},
        base_team_path=str(base),
        agent_team_path="agent",
        opponent_team_pool_path="constrained",
    )


def _make_curriculum_multi(tmp_path) -> CurriculumConfig:
    base = tmp_path / "teams"
    for fmt in ("gen9vgc2024regg", "gen9vgc2024regh"):
        fmt_dir = base / fmt
        pool_dir = fmt_dir / "constrained"
        pool_dir.mkdir(parents=True)
        (pool_dir / "team_a.txt").write_text("team a body")
        agent_dir = fmt_dir / "agent"
        agent_dir.mkdir()
        (agent_dir / "team_a.txt").write_text("agent team body")
    return CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3},
        base_team_path=str(base),
        agent_team_path="agent",
        opponent_team_pool_path="constrained",
    )


def _make_config(tmp_path) -> FoulplayEvalConfig:
    fake_py = tmp_path / "python"
    fake_py.write_text("")
    return FoulplayEvalConfig(
        enabled=True,
        n_battles_per_format=20,
        search_time_ms=250,
        parallelism=2,
        python_executable=str(fake_py),
    )


def _stub_eval_result(p1_wins: int, p2_wins: int, ties: int = 0) -> EvalResult:
    return EvalResult(
        label="model_vs_foul_play",
        player1_wins=p1_wins,
        player2_wins=p2_wins,
        ties=ties,
        battles_played=p1_wins + p2_wins + ties,
    )


# ── FoulplayEvalResult aggregation ────────────────────────────────────────────


def test_result_aggregates_across_formats():
    """Aggregate properties sum across per-format results."""
    result = FoulplayEvalResult(
        per_format={
            "gen9vgc2024regg": _stub_eval_result(p1_wins=12, p2_wins=8),
            "gen9vgc2024regh": _stub_eval_result(p1_wins=6, p2_wins=14),
        },
        battle_formats_weights={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3},
    )
    assert result.battles_played == 40
    assert result.model_wins == 18
    assert result.foulplay_wins == 22
    assert result.ties == 0


def test_overall_win_rate_is_weight_averaged():
    """
    overall_win_rate weight-averages per-format win rates by the
    curriculum's battle_formats weights — not by battle count. This
    matters when formats have unequal weights: a high-weight format's
    win rate dominates the aggregate.
    """
    result = FoulplayEvalResult(
        per_format={
            "gen9vgc2024regg": _stub_eval_result(p1_wins=60, p2_wins=40),  # 60%
            "gen9vgc2024regh": _stub_eval_result(p1_wins=20, p2_wins=80),  # 20%
        },
        battle_formats_weights={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3},
    )
    # 0.7 * 0.60 + 0.3 * 0.20 = 0.42 + 0.06 = 0.48
    assert result.overall_win_rate == pytest.approx(0.48)


def test_overall_win_rate_zero_battles_is_zero():
    """No battles played → 0.0, not a division-by-zero."""
    result = FoulplayEvalResult(per_format={}, battle_formats_weights={})
    assert result.overall_win_rate == 0.0
    assert result.battles_played == 0


# ── run() multi-format loop ──────────────────────────────────────────────────


def test_run_single_format_calls_run_eval_parallel_once(tmp_path):
    """Single-format eval calls run_eval_parallel exactly once."""
    curriculum = _make_curriculum_single(tmp_path)
    config = _make_config(tmp_path)
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")

    with patch(
        "elitefurretai.rl.analyze.foulplay_eval.run_eval_parallel",
        return_value=_stub_eval_result(p1_wins=15, p2_wins=5),
    ) as mock_parallel:
        result = run(
            checkpoint_path=str(fake_ckpt),
            config=config,
            curriculum=curriculum,
            device="cpu",
            server_urls=["localhost:8000"],
            run_tag="abcd",
        )

    assert mock_parallel.call_count == 1
    assert result.battles_played == 20
    assert result.model_wins == 15
    assert result.foulplay_wins == 5
    # Weight-averaged WR with one format equals that format's WR.
    assert result.overall_win_rate == pytest.approx(0.75)


def test_run_multi_format_iterates_once_per_format(tmp_path):
    """Multi-format eval calls run_eval_parallel once per active format."""
    curriculum = _make_curriculum_multi(tmp_path)
    config = _make_config(tmp_path)
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")

    side_effects = [
        _stub_eval_result(p1_wins=12, p2_wins=8),  # regg: 60%
        _stub_eval_result(p1_wins=4, p2_wins=16),  # regh: 20%
    ]
    with patch(
        "elitefurretai.rl.analyze.foulplay_eval.run_eval_parallel",
        side_effect=side_effects,
    ) as mock_parallel:
        result = run(
            checkpoint_path=str(fake_ckpt),
            config=config,
            curriculum=curriculum,
            device="cpu",
            server_urls=["localhost:8000"],
            run_tag="abcd",
        )

    assert mock_parallel.call_count == 2
    assert set(result.per_format.keys()) == {"gen9vgc2024regg", "gen9vgc2024regh"}
    assert result.battles_played == 40
    # Overall WR = 0.7 * 0.60 + 0.3 * 0.20 = 0.48
    assert result.overall_win_rate == pytest.approx(0.48)


def test_run_uses_explicit_foulplay_team_pool_paths_when_set(tmp_path):
    """
    When config.foulplay_team_pool_paths is set, the per-format
    eval uses those paths verbatim instead of falling back to the
    curriculum's opponent_team_pool_paths.
    """
    curriculum = _make_curriculum_single(tmp_path)
    config = _make_config(tmp_path)
    # Distinct directory for FoulPlay's side — not the curriculum pool.
    foulplay_pool = tmp_path / "foulplay_only"
    foulplay_pool.mkdir()
    (foulplay_pool / "fp_team.txt").write_text("fp team body")
    config.foulplay_team_pool_paths = {"gen9vgc2024regg": str(foulplay_pool)}

    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")

    with patch(
        "elitefurretai.rl.analyze.foulplay_eval.run_eval_parallel",
        return_value=_stub_eval_result(p1_wins=10, p2_wins=10),
    ) as mock_parallel:
        run(
            checkpoint_path=str(fake_ckpt),
            config=config,
            curriculum=curriculum,
            device="cpu",
            server_urls=["localhost:8000"],
            run_tag="abcd",
        )

    # Inspect the foul_play spec passed as p2 — its team_pool_path
    # param should be the override, not the curriculum's pool.
    _, kwargs = mock_parallel.call_args
    foul_play_spec = kwargs["p2"]
    assert foul_play_spec.params["team_pool_path"] == str(foulplay_pool)
