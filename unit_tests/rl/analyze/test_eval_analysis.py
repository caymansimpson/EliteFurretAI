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
    compute_ensemble_advantage,
    q1_agent_team_win_rate,
    q2_opp_team_win_rate,
    q3_opp_type_win_rate,
    q5_short_loss_patterns,
    q6_confidence_in_poor_situations,
    q7_value_calibration,
    q9a_persistent_disagreement,
    q9b_agree_then_diverge,
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


# ─── compute_ensemble_advantage (helper) ────────────────────────────


def test_ensemble_advantage_matches_battle_dataset_semantics():
    """Ported function matches the original _compute_ensemble_advantage.

    Reference values computed manually for n=5, heuristic ≡ 0,
    outcome=+1: each blended[i] = outcome_weight × 1.0 since the
    position term contributes 0.
    """
    n = 5
    heuristic = [0.0] * n
    out = compute_ensemble_advantage(heuristic, 1.0)
    assert len(out) == n
    # progress: [0/4, 1/4, 2/4, 3/4, 4/4] = [0, .25, .5, .75, 1]
    # outcome_weight = clip(p², 0.05, 0.95)
    # → [0.05, 0.0625, 0.25, 0.5625, 0.95]
    expected = [0.05, 0.0625, 0.25, 0.5625, 0.95]
    for o, e in zip(out, expected):
        assert math.isclose(o, e, abs_tol=1e-6)


def test_ensemble_advantage_n1_returns_outcome():
    assert compute_ensemble_advantage([0.5], 1.0) == [1.0]
    assert compute_ensemble_advantage([], 1.0) == []


def test_ensemble_advantage_blends_heuristic_and_outcome():
    """Early: mostly heuristic. Late: mostly outcome."""
    heuristic = [0.8] * 10  # consistent "we're winning"
    out = compute_ensemble_advantage(heuristic, -1.0)  # but we lose
    # Early turn should weight heuristic heavily → blended positive.
    assert out[0] > 0  # heuristic dominates
    # Late turn should weight outcome heavily → blended negative.
    assert out[-1] < 0


# ─── Q6: confidence in poor situations ──────────────────────────────


def test_q6_swing_detection_finds_collapses_in_losses():
    """A loss where heuristic_adv drops from +0.5 to -0.5 over K=3 turns
    should surface as a swing event."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": 0.0}] * 1)
    # Turn 0: tp. Turns 1-6: heuristic goes from +0.5 → -0.5 around turn 4.
    turn_rows = [
        {
            "battle_id": "b0",
            "turn_number": 0,
            "is_teampreview": True,
            "value_predicted": 0.0,
            "heuristic_adv": 0.0,
        },
        {
            "battle_id": "b0",
            "turn_number": 1,
            "is_teampreview": False,
            "value_predicted": 0.5,
            "heuristic_adv": 0.5,
        },
        {
            "battle_id": "b0",
            "turn_number": 2,
            "is_teampreview": False,
            "value_predicted": 0.4,
            "heuristic_adv": 0.4,
        },
        {
            "battle_id": "b0",
            "turn_number": 3,
            "is_teampreview": False,
            "value_predicted": 0.3,
            "heuristic_adv": 0.3,
        },
        {
            "battle_id": "b0",
            "turn_number": 4,
            "is_teampreview": False,
            "value_predicted": -0.5,
            "heuristic_adv": -0.5,
        },
    ]
    _, swings, _ = q6_confidence_in_poor_situations(
        battles_df, _make_turns(turn_rows), swing_window=3, swing_threshold=-0.5
    )
    assert len(swings) >= 1
    row = swings.iloc[0]
    assert math.isclose(row["swing"], -1.0, abs_tol=0.01)  # -0.5 - 0.5 = -1.0
    assert math.isclose(row["heuristic_adv_before"], 0.5, abs_tol=0.01)
    assert math.isclose(row["heuristic_adv_after"], -0.5, abs_tol=0.01)
    # The model's value at t-K was high (0.5) → it didn't see this coming.
    assert math.isclose(row["value_at_t_minus_k"], 0.5, abs_tol=0.01)


def test_q6_swing_only_in_losses():
    """Wins don't appear in swing_events even if they have negative swings."""
    battles_df = _make_battles(
        [{"opp_player_name": "x", "outcome": 1.0}]  # WIN
    )
    turn_rows = [
        {
            "battle_id": "b0",
            "turn_number": t,
            "is_teampreview": False,
            "value_predicted": 0.0,
            "heuristic_adv": 0.5 if t < 4 else -0.5,
        }
        for t in range(1, 6)
    ]
    _, swings, _ = q6_confidence_in_poor_situations(
        battles_df, _make_turns(turn_rows), swing_window=3, swing_threshold=-0.5
    )
    assert len(swings) == 0


def test_q6_poor_situation_summary_compares_buckets():
    """Summary table includes both 'all' and 'poor situation' rows."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": 0.0}] * 2)
    turn_rows = []
    # Battle b0: 10 turns at heuristic_adv = +0.5 (not poor) with entropy 1.0
    for t in range(1, 11):
        turn_rows.append(
            {
                "battle_id": "b0",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": 0.0,
                "heuristic_adv": 0.5,
                "policy_entropy": 1.0,
            }
        )
    # Battle b1: 10 turns at heuristic_adv = -0.5 (poor) with entropy 3.0
    for t in range(1, 11):
        turn_rows.append(
            {
                "battle_id": "b1",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": 0.0,
                "heuristic_adv": -0.5,
                "policy_entropy": 3.0,
            }
        )

    _, _, summary = q6_confidence_in_poor_situations(
        battles_df, _make_turns(turn_rows), poor_threshold=-0.3
    )
    assert len(summary) == 2
    summary_by_subset = summary.set_index("subset")
    assert summary_by_subset.loc["all", "n_turns"] == 20
    # The "poor" subset is the 10 turns at -0.5 with entropy 3.0.
    poor_row = summary_by_subset.iloc[1]
    assert poor_row["n_turns"] == 10
    assert math.isclose(poor_row["mean_entropy"], 3.0)


def test_q6_empty_inputs_safe():
    by_b, sw, summ = q6_confidence_in_poor_situations(pd.DataFrame(), pd.DataFrame())
    assert by_b.empty
    assert sw.empty
    assert summ.empty


# ─── Q9a: persistent disagreement ───────────────────────────────────


def test_q9a_ranks_by_mean_abs_difference():
    """The battle with the most consistent value vs ensemble gap ranks first."""
    battles_df = _make_battles(
        [
            {"opp_player_name": "x", "outcome": 1.0},
            {"opp_player_name": "x", "outcome": 1.0},
        ]
    )
    # Battle b0: value=+1, ensemble blends from heuristic=+1 + outcome=+1 → ~+1. Diff ≈ 0.
    # Battle b1: value=-1 (overconfident loss prediction), heuristic=+1, outcome=+1.
    #            ensemble blends toward +1, so |value - ensemble| ≈ 2 throughout.
    turn_rows = []
    for t in range(1, 11):
        turn_rows.append(
            {
                "battle_id": "b0",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": 1.0,
                "heuristic_adv": 1.0,
            }
        )
    for t in range(1, 11):
        turn_rows.append(
            {
                "battle_id": "b1",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": -1.0,
                "heuristic_adv": 1.0,
            }
        )

    result = q9a_persistent_disagreement(battles_df, _make_turns(turn_rows))
    assert len(result) == 2
    # b1 should rank first (bigger gap).
    assert result.iloc[0]["battle_id"] == "b1"
    assert result.iloc[0]["mean_abs_diff"] > 1.5
    assert result.iloc[1]["battle_id"] == "b0"
    assert result.iloc[1]["mean_abs_diff"] < 0.5


def test_q9a_ignores_ties():
    """Battles with NaN outcome (ties) get NaN ensemble_adv → excluded."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": float("nan")}])
    turn_rows = [
        {
            "battle_id": "b0",
            "turn_number": t,
            "is_teampreview": False,
            "value_predicted": 0.0,
            "heuristic_adv": 0.0,
        }
        for t in range(1, 6)
    ]
    result = q9a_persistent_disagreement(battles_df, _make_turns(turn_rows))
    assert result.empty


# ─── Q9b: agree-then-diverge ────────────────────────────────────────


def test_q9b_detects_split_pattern():
    """Battle where early diff is tiny and late diff is large should match."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": 1.0}])
    turn_rows = []
    # Turns 1-5: value=heuristic=+0.8 → ensemble~+0.8 → diff ~0.
    for t in range(1, 6):
        turn_rows.append(
            {
                "battle_id": "b0",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": 0.8,
                "heuristic_adv": 0.8,
            }
        )
    # Turns 6-15: value plummets to -0.9 while heuristic stays +0.8 → big diff.
    for t in range(6, 16):
        turn_rows.append(
            {
                "battle_id": "b0",
                "turn_number": t,
                "is_teampreview": False,
                "value_predicted": -0.9,
                "heuristic_adv": 0.8,
            }
        )

    result = q9b_agree_then_diverge(
        battles_df,
        _make_turns(turn_rows),
        early_window=5,
        early_threshold=0.15,
        late_threshold=0.30,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["battle_id"] == "b0"
    assert row["pre_split_diff"] < 0.15
    assert row["post_split_diff"] > 0.30
    assert row["diff_increase"] > 0


def test_q9b_skips_battles_too_short():
    """Battles with ≤early_window in-battle turns get filtered out."""
    battles_df = _make_battles([{"opp_player_name": "x", "outcome": 1.0}])
    turn_rows = [
        {
            "battle_id": "b0",
            "turn_number": t,
            "is_teampreview": False,
            "value_predicted": 0.0,
            "heuristic_adv": 0.0,
        }
        for t in range(1, 4)  # only 3 in-battle turns
    ]
    result = q9b_agree_then_diverge(battles_df, _make_turns(turn_rows), early_window=5)
    assert result.empty


def test_q9b_empty_inputs_safe():
    result = q9b_agree_then_diverge(pd.DataFrame(), pd.DataFrame())
    assert result.empty
