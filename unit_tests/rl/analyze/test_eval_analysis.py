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
    q5_short_loss_patterns,
    q7_value_calibration,
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


def _make_turns(rows):
    """Build a turns DataFrame with sensible defaults; `rows` provides at
    minimum battle_id + value_predicted + is_teampreview."""
    full = []
    for i, row in enumerate(rows):
        full.append(
            {
                "battle_id": row["battle_id"],
                "turn_number": row.get("turn_number", i),
                "is_teampreview": row.get("is_teampreview", False),
                "action_chosen": row.get("action_chosen", 42),
                "action_chosen_str": row.get("action_chosen_str", "act"),
                "top_k_actions_json": "[]",
                "policy_entropy": row.get("policy_entropy", 1.0),
                "value_predicted": row["value_predicted"],
                "heuristic_adv": row.get("heuristic_adv", 0.0),
                "agent_hp_frac_sum": 3.0,
                "opp_hp_frac_sum": 3.0,
                "agent_alive_count": 4,
                "opp_alive_count": 4,
                "agent_switch_this_turn": False,
            }
        )
    return pd.DataFrame(full)


# ─── Q5: short-loss patterns ────────────────────────────────────────


def test_q5_filters_to_short_losses():
    """Only battles with outcome=0 AND final_turn ≤ max_turn count."""
    rows = (
        # Short losses to OT1 (over-represented).
        [{"opp_team_hash": "OT1", "opp_player_name": "x", "outcome": 0.0, "final_turn": 3}]
        * 10
        # Long losses to OT2 (NOT over-represented).
        + [
            {
                "opp_team_hash": "OT2",
                "opp_player_name": "x",
                "outcome": 0.0,
                "final_turn": 20,
            }
        ]
        * 10
        # Wins to both (not counted as losses).
        + [
            {
                "opp_team_hash": "OT1",
                "opp_player_name": "x",
                "outcome": 1.0,
                "final_turn": 12,
            }
        ]
        * 5
    )
    battles_df = _make_battles(rows)
    # Manually set final_turn since _make_battles defaults to 15.
    for i, r in enumerate(rows):
        battles_df.at[i, "final_turn"] = r["final_turn"]

    freq, _ = q5_short_loss_patterns(battles_df)
    keyed = freq.set_index("opp_team_hash")
    # OT1: 10 short of 10 losses → fully over-represented (rate 1.0).
    # OT2: 0 short of 10 losses → 0 over-represented.
    assert keyed.loc["OT1", "n_short_losses"] == 10
    assert keyed.loc["OT1", "n_losses"] == 10
    assert math.isclose(keyed.loc["OT1", "short_fraction"], 1.0)
    assert keyed.loc["OT2", "n_short_losses"] == 0
    assert math.isclose(keyed.loc["OT2", "short_fraction"], 0.0)
    # Over-representation: baseline = 10/20 = 0.5; OT1 ratio = 1.0/0.5 = 2.0.
    assert math.isclose(keyed.loc["OT1", "over_representation"], 2.0)


def test_q5_action_dist_returned_with_turns():
    """When turns provided, action_dist surfaces over-/under-represented actions."""
    battles_df = _make_battles(
        [{"opp_player_name": "x", "outcome": 0.0}] * 4
        + [{"opp_player_name": "x", "outcome": 1.0}] * 4
    )
    for i in range(4):
        battles_df.at[i, "final_turn"] = 3  # short losses
    for i in range(4, 8):
        battles_df.at[i, "final_turn"] = 12  # long wins

    # In losing battles (b0..b3) the model always plays "bad_action".
    # In winning battles (b4..b7) it plays "good_action".
    turn_rows = []
    for bid in ["b0", "b1", "b2", "b3"]:
        turn_rows.append(
            {"battle_id": bid, "value_predicted": 0.0, "action_chosen_str": "bad_action"}
        )
    for bid in ["b4", "b5", "b6", "b7"]:
        turn_rows.append(
            {"battle_id": bid, "value_predicted": 0.0, "action_chosen_str": "good_action"}
        )

    freq, actions = q5_short_loss_patterns(battles_df, _make_turns(turn_rows))
    assert actions is not None
    assert "chi_sq_residual" in actions.columns
    # bad_action over-represented in short losses; good_action under.
    by_action = actions.set_index("action_chosen_str")
    assert by_action.loc["bad_action", "chi_sq_residual"] > 0
    assert by_action.loc["good_action", "chi_sq_residual"] < 0


def test_q5_empty_inputs_safe():
    freq, actions = q5_short_loss_patterns(pd.DataFrame())
    assert freq.empty
    assert actions is None


# ─── Q7: value calibration ──────────────────────────────────────────


def test_q7_perfect_calibration_zero_ece():
    """A model whose value predictions match outcomes exactly has ECE ≈ 0.

    Construct: 50 turns with value=+1 in won battles, 50 turns with
    value=-1 in lost battles. After rescaling to [0, 1], that's
    pred_prob=1.0 for wins and pred_prob=0.0 for losses. Reliability
    is perfect.
    """
    win_battles = [{"opp_player_name": "x", "outcome": 1.0} for _ in range(50)]
    lose_battles = [{"opp_player_name": "x", "outcome": 0.0} for _ in range(50)]
    battles_df = _make_battles(win_battles + lose_battles)
    # battle_ids generated as b0..b99.
    turn_rows = []
    for i in range(50):
        turn_rows.append(
            {"battle_id": f"b{i}", "value_predicted": 1.0, "is_teampreview": False}
        )
    for i in range(50, 100):
        turn_rows.append(
            {"battle_id": f"b{i}", "value_predicted": -1.0, "is_teampreview": False}
        )

    reliability, ece = q7_value_calibration(battles_df, _make_turns(turn_rows))
    assert math.isclose(ece, 0.0, abs_tol=1e-6)
    # Two non-empty bins: predicted ~0 (loss) and predicted ~1 (win).
    assert len(reliability) >= 1
    win_bin = reliability[reliability["bin_lower"] > 0.5].iloc[0]
    loss_bin = reliability[reliability["bin_upper"] < 0.5].iloc[0]
    assert math.isclose(win_bin["observed_win_rate"], 1.0)
    assert math.isclose(loss_bin["observed_win_rate"], 0.0)


def test_q7_miscalibration_increases_ece():
    """A wildly overconfident model has ECE > 0.5.

    All turns predict value=+1 (confident win) but the model loses
    half the time. ECE should reflect the gap.
    """
    battles_df = _make_battles(
        [{"opp_player_name": "x", "outcome": 1.0}] * 50
        + [{"opp_player_name": "x", "outcome": 0.0}] * 50
    )
    turn_rows = [
        {"battle_id": f"b{i}", "value_predicted": 1.0, "is_teampreview": False}
        for i in range(100)
    ]
    _, ece = q7_value_calibration(battles_df, _make_turns(turn_rows))
    # Predicted prob=1.0; observed=0.5. ECE = |1.0 - 0.5| = 0.5.
    assert math.isclose(ece, 0.5, abs_tol=0.01)


def test_q7_teampreview_turns_excluded():
    """Teampreview turns have degenerate value and must be filtered out."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": 1.0}] * 10)
    turn_rows = []
    for i in range(10):
        # Teampreview turn with wildly wrong value=-1.
        turn_rows.append(
            {"battle_id": f"b{i}", "value_predicted": -1.0, "is_teampreview": True}
        )
        # Mid-battle turn with correct value=+1.
        turn_rows.append(
            {"battle_id": f"b{i}", "value_predicted": 1.0, "is_teampreview": False}
        )
    _, ece = q7_value_calibration(battles_df, _make_turns(turn_rows))
    # Only the +1 mid-battle turns are in the diagram; outcome=1.0;
    # pred_prob=1.0; ECE ≈ 0.
    assert math.isclose(ece, 0.0, abs_tol=0.01)


def test_q7_empty_inputs_safe():
    rel, ece = q7_value_calibration(pd.DataFrame(), pd.DataFrame())
    assert rel.empty
    assert math.isnan(ece)


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
