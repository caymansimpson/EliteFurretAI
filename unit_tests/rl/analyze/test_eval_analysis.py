# -*- coding: utf-8 -*-
"""Tests for ``eval_analysis`` Q1/Q2/Q3 win-rate functions + CI helper.

Each test builds a synthetic ``battles`` DataFrame with a known
distribution and asserts the aggregation produces the expected
shape. The Wilson CI is checked against a published reference value
(p=0.5, n=100, z=1.96 → [0.404, 0.595]) so a regression in the
formula is caught without a heavyweight dependency.
"""

from __future__ import annotations

import math

import pandas as pd

from elitefurretai.rl.analyze.eval_analysis import (
    q1_agent_team_win_rate,
    q2_opp_team_win_rate,
    q3_opp_type_win_rate,
    wilson_ci,
)

# ─── Wilson CI ──────────────────────────────────────────────────────


def test_wilson_ci_published_50_pct():
    """At p=0.5, n=100, z=1.96 the Wilson interval is [0.404, 0.595]."""
    lo, hi = wilson_ci(50, 100)
    assert math.isclose(lo, 0.404, abs_tol=0.001)
    assert math.isclose(hi, 0.596, abs_tol=0.001)


def test_wilson_ci_handles_zero_n():
    """Empty sample returns (0, 0) without dividing by zero."""
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_zero_wins_bounded():
    """0 wins of 100 → upper bound stays under 0.05 (sanity)."""
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0
    assert hi < 0.05


def test_wilson_ci_all_wins_bounded():
    """All wins → lower bound stays above 0.95; upper hits ~1.0."""
    lo, hi = wilson_ci(100, 100)
    assert lo > 0.95
    assert math.isclose(hi, 1.0, abs_tol=1e-9)


# ─── Q3: opp_player_name aggregation ────────────────────────────────


def _make_battles(rows):
    """Convenience: build a battles DataFrame with sensible defaults.

    ``rows`` is a list of dicts with at least ``outcome``,
    ``agent_team_hash``, ``opp_team_hash``, and ``opp_player_name``.
    Missing fields are filled with placeholder values that don't affect
    the aggregation under test.
    """
    full = []
    for i, row in enumerate(rows):
        full.append(
            {
                "battle_id": f"b{i}",
                "eval_run_id": "r0",
                "agent_ckpt": "data/models/foo.pt",
                "agent_team_hash": row.get("agent_team_hash", "aaaa"),
                "opp_player_kind": "baseline",
                "opp_player_name": row["opp_player_name"],
                "opp_team_hash": row.get("opp_team_hash", "bbbb"),
                "battle_format": "gen9vgc2024regg",
                "outcome": row["outcome"],
                "final_turn": 15,
                "agent_final_pokemon_alive": 2,
                "opp_final_pokemon_alive": 0,
                "timestamp_started": 1700000000.0 + i,
                "replay_saved": False,
            }
        )
    return pd.DataFrame(full)


def test_q3_one_opp_type_uniform_wins():
    battles = _make_battles(
        [{"outcome": 1.0, "opp_player_name": "max_damage"}] * 50
        + [{"outcome": 0.0, "opp_player_name": "max_damage"}] * 50
    )
    result = q3_opp_type_win_rate(battles)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["opp_player_name"] == "max_damage"
    assert row["n_battles"] == 100
    assert row["wins"] == 50
    assert row["losses"] == 50
    assert row["ties"] == 0
    assert math.isclose(row["win_rate"], 0.5)
    # Wilson CI for 50/100 → [0.404, 0.596]
    assert math.isclose(row["ci_low"], 0.404, abs_tol=0.005)
    assert math.isclose(row["ci_high"], 0.596, abs_tol=0.005)


def test_q3_multiple_opp_types_segregated():
    battles = _make_battles(
        [{"outcome": 1.0, "opp_player_name": "max_damage"}] * 80
        + [{"outcome": 0.0, "opp_player_name": "max_damage"}] * 20
        + [{"outcome": 1.0, "opp_player_name": "simple_heuristic"}] * 30
        + [{"outcome": 0.0, "opp_player_name": "simple_heuristic"}] * 70
    )
    result = q3_opp_type_win_rate(battles).set_index("opp_player_name")
    assert math.isclose(result.loc["max_damage", "win_rate"], 0.8)
    assert math.isclose(result.loc["simple_heuristic", "win_rate"], 0.3)
    assert result.loc["max_damage", "n_battles"] == 100
    assert result.loc["simple_heuristic", "n_battles"] == 100


def test_q3_ties_excluded_from_win_rate_but_counted():
    battles = _make_battles(
        [{"outcome": 1.0, "opp_player_name": "vgc_bench"}] * 40
        + [{"outcome": 0.0, "opp_player_name": "vgc_bench"}] * 40
        + [{"outcome": float("nan"), "opp_player_name": "vgc_bench"}] * 20
    )
    result = q3_opp_type_win_rate(battles)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["n_battles"] == 100
    assert row["wins"] == 40
    assert row["losses"] == 40
    assert row["ties"] == 20
    # WR = 40/80 (NaNs not in denom), not 40/100.
    assert math.isclose(row["win_rate"], 0.5)


def test_q3_empty_input_returns_empty_df():
    result = q3_opp_type_win_rate(pd.DataFrame())
    assert result.empty
    assert "win_rate" in result.columns


# ─── Q1: agent_team_hash × opp_player_name ──────────────────────────


def test_q1_per_team_per_opp_breakdown():
    rows = []
    # team_A vs max_damage: 80% WR
    rows += [
        {"agent_team_hash": "A", "opp_player_name": "max_damage", "outcome": 1.0}
    ] * 40
    rows += [
        {"agent_team_hash": "A", "opp_player_name": "max_damage", "outcome": 0.0}
    ] * 10
    # team_B vs max_damage: 20% WR — should sort to top of "worst"
    rows += [
        {"agent_team_hash": "B", "opp_player_name": "max_damage", "outcome": 1.0}
    ] * 10
    rows += [
        {"agent_team_hash": "B", "opp_player_name": "max_damage", "outcome": 0.0}
    ] * 40

    result = q1_agent_team_win_rate(_make_battles(rows))
    # Expect 2 cells: (A, max_damage) and (B, max_damage).
    assert len(result) == 2
    keyed = result.set_index("agent_team_hash")
    assert math.isclose(keyed.loc["A", "win_rate"], 0.8)
    assert math.isclose(keyed.loc["B", "win_rate"], 0.2)


def test_q1_sorts_worst_to_top_when_caller_requests():
    """The default order is groupby order; caller is responsible for sort.

    Test that the CLI's documented sort_values("win_rate") puts the
    worst-performing team first.
    """
    rows = (
        [{"agent_team_hash": "good", "opp_player_name": "x", "outcome": 1.0}] * 8
        + [{"agent_team_hash": "good", "opp_player_name": "x", "outcome": 0.0}] * 2
        + [{"agent_team_hash": "bad", "opp_player_name": "x", "outcome": 1.0}] * 2
        + [{"agent_team_hash": "bad", "opp_player_name": "x", "outcome": 0.0}] * 8
    )
    result = q1_agent_team_win_rate(_make_battles(rows)).sort_values("win_rate")
    assert result.iloc[0]["agent_team_hash"] == "bad"
    assert result.iloc[-1]["agent_team_hash"] == "good"


# ─── Q2: opp_team_hash × opp_player_name ────────────────────────────


def test_q2_per_opp_team_breakdown():
    rows = (
        [{"opp_team_hash": "OT1", "opp_player_name": "x", "outcome": 1.0}] * 30
        + [{"opp_team_hash": "OT1", "opp_player_name": "x", "outcome": 0.0}] * 20
        + [{"opp_team_hash": "OT2", "opp_player_name": "x", "outcome": 1.0}] * 5
        + [{"opp_team_hash": "OT2", "opp_player_name": "x", "outcome": 0.0}] * 45
    )
    result = q2_opp_team_win_rate(_make_battles(rows)).set_index("opp_team_hash")
    assert math.isclose(result.loc["OT1", "win_rate"], 0.6)
    assert math.isclose(result.loc["OT2", "win_rate"], 0.1)


def test_q2_picks_out_consistently_losing_opp_teams():
    """The user's Q8 category (c): teams the agent consistently loses to are
    those with WR > 0.75 from the opponent's perspective, equivalently
    WR < 0.25 from agent's perspective."""
    rows = (
        [{"opp_team_hash": "tough", "opp_player_name": "x", "outcome": 1.0}] * 2
        + [{"opp_team_hash": "tough", "opp_player_name": "x", "outcome": 0.0}] * 18
        + [{"opp_team_hash": "easy", "opp_player_name": "x", "outcome": 1.0}] * 16
        + [{"opp_team_hash": "easy", "opp_player_name": "x", "outcome": 0.0}] * 4
    )
    result = q2_opp_team_win_rate(_make_battles(rows))
    tough_wr = result.set_index("opp_team_hash").loc["tough", "win_rate"]
    assert tough_wr < 0.25  # category-(c) threshold
