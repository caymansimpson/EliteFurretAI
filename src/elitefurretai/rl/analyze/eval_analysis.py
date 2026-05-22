# -*- coding: utf-8 -*-
"""Offline analysis CLI for Plan B trajectory collection runs.

Reads parquet shards produced by ``evaluate.py --collect-trajectories``
and computes the answers to Q1–Q9 from
``planning/stage2/2026-05-17-...-plan-b-model-analysis.md``.

Subcommands
-----------
``summary``         — top-level Q3 table: WR by opp_player_name.
``agent_team``      — Q1: WR by (agent_team_hash, opp_player_name).
``opp_team``        — Q2: WR by (opp_team_hash, opp_player_name).
``short_loss``      — Q5: patterns in losses ≤ 5 turns.        (TODO)
``confidence``      — Q6: policy entropy vs heuristic_adv.      (TODO)
``value_calibration``— Q7: reliability diagram + ECE.            (TODO)
``value_ensemble``  — Q9: model-value vs ensemble disagreement. (TODO)
``save_games``      — Q8: dump 3 games per category.            (TODO)
``report``          — render everything as one HTML / markdown. (TODO)

Each subcommand reads ``<run_dir>/battles.parquet`` (glob across
worker shards) and optionally ``<run_dir>/turns.parquet``, then
prints a table to stdout (default) or writes one of csv/markdown/json
to ``--output``.

The analysis functions are pure ``(battles_df, turns_df) -> pd.DataFrame``
shape so they can be unit-tested on synthetic input without needing
a real eval run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd

from elitefurretai.rl.analyze.eval_schema import read_battles, read_turns

# ─── Statistics helpers ──────────────────────────────────────────────


def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson is preferred over the normal approximation because it
    doesn't degenerate at the boundaries (p=0 or p=1) and gives
    sensible intervals at low n.

    Returns (lower, upper). For n=0 returns (0.0, 0.0).
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ─── Analysis functions (Q1, Q2, Q3) ────────────────────────────────


def q3_opp_type_win_rate(battles: pd.DataFrame) -> pd.DataFrame:
    """Q3: win rate aggregated by ``opp_player_name``.

    Returns one row per opp_player_name with columns:
    ``n_battles, wins, losses, ties, win_rate, ci_low, ci_high``.

    Ties (outcome=NaN) are excluded from win_rate denominators but
    counted in the ``ties`` column so the audit trail is complete.
    """
    return _group_win_rate(battles, ["opp_player_name"])


def q_format_opp_type_win_rate(battles: pd.DataFrame) -> pd.DataFrame:
    """Win rate grouped by (battle_format, opp_player_name).

    Used for the Stage II per-format graduation check. Returns
    ``n_battles, wins, losses, ties, win_rate, ci_low, ci_high`` per cell.
    Requires a ``battle_format`` column on the input DataFrame, which the
    eval pipeline already populates (see eval_schema.py).
    """
    return _group_win_rate(battles, ["battle_format", "opp_player_name"])


def graduation_summary(
    battles: pd.DataFrame,
    threshold: float = 0.60,
    required_opp_types: tuple = (
        "max_damage",
        "vgc_bench",
        "bc_player",
        "simple_heuristic",
    ),
) -> dict:
    """Stage II graduation check across (format x opp_type) cells.

    For each (battle_format, opp_type) present in ``battles`` whose
    ``opp_player_name`` is in ``required_opp_types``, emit a cell:
    ``{battle_format, opp_player_name, n_battles, win_rate, passed}``.
    For any (format, opp_type) pair where opp_type is required but no
    battles exist, emit a cell with ``missing=True, passed=False``.

    Overall ``passed`` is True iff every cell passes (no missing
    required opp_types, every win_rate >= threshold).
    """
    if battles.empty or "battle_format" not in battles.columns:
        raise ValueError(
            "graduation_summary requires a non-empty battles DataFrame with a "
            "'battle_format' column"
        )
    per_cell = q_format_opp_type_win_rate(battles)
    formats = sorted({str(f) for f in battles["battle_format"].unique()})
    cells: List[dict] = []
    for fmt in formats:
        for opp in required_opp_types:
            row = per_cell[
                (per_cell["battle_format"] == fmt) & (per_cell["opp_player_name"] == opp)
            ]
            if row.empty:
                cells.append(
                    {
                        "battle_format": fmt,
                        "opp_player_name": opp,
                        "n_battles": 0,
                        "win_rate": float("nan"),
                        "passed": False,
                        "missing": True,
                    }
                )
            else:
                wr = float(row["win_rate"].iloc[0])
                cells.append(
                    {
                        "battle_format": fmt,
                        "opp_player_name": opp,
                        "n_battles": int(row["n_battles"].iloc[0]),
                        "win_rate": wr,
                        "passed": wr >= threshold,
                        "missing": False,
                    }
                )
    return {
        "threshold": threshold,
        "required_opp_types": list(required_opp_types),
        "formats": formats,
        "cells": cells,
        "passed": all(c["passed"] for c in cells),
    }


def q1_agent_team_win_rate(battles: pd.DataFrame) -> pd.DataFrame:
    """Q1: win rate by (agent_team_hash, opp_player_name).

    "Which of our teams perform poorly?" — sort the output by
    ``win_rate`` ascending to surface the worst-performing teams.
    """
    return _group_win_rate(battles, ["agent_team_hash", "opp_player_name"])


def q2_opp_team_win_rate(battles: pd.DataFrame) -> pd.DataFrame:
    """Q2: win rate by (opp_team_hash, opp_player_name).

    "Which opponent teams crush us?" — sort ascending by ``win_rate``
    to surface the worst matchups for our model.
    """
    return _group_win_rate(battles, ["opp_team_hash", "opp_player_name"])


def q5_short_loss_patterns(
    battles: pd.DataFrame,
    turns: Optional[pd.DataFrame] = None,
    *,
    max_turn: int = 5,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Q5: short-loss (≤``max_turn`` turns) patterns.

    Returns ``(opp_team_freq, action_dist)``:

    * ``opp_team_freq``: per (opp_team_hash, opp_player_name) the
      number of short losses, total battles, and the over-representation
      ratio ``(short_loss_rate / overall_loss_rate)``. A ratio >1
      means this opp_team crushes us *faster* than average. Sort
      descending to surface the worst short-loss offenders.
    * ``action_dist``: if ``turns`` is provided, per action_chosen_str
      the count in short losses vs the same count in all battles, and
      a chi-square-style residual. Tells you which actions are
      over-/under-represented when we lose fast. ``None`` if no turns.

    Why two tables: opp_team_freq answers "which matchups blow up
    fast?" (team-level diagnostic); action_dist answers "what is the
    model doing wrong on turn 1-4 when it gets crushed?" (behavior
    diagnostic).
    """
    if battles.empty:
        empty = pd.DataFrame(
            columns=[
                "opp_team_hash",
                "opp_player_name",
                "n_short_losses",
                "n_losses",
                "short_fraction",
                "over_representation",
            ]
        )
        return empty, None

    losses_all = battles.dropna(subset=["outcome"])
    losses_all = losses_all[losses_all["outcome"] == 0.0]
    short_losses = losses_all[losses_all["final_turn"] <= max_turn]

    # Per-opp-team frequency.
    overall = (
        losses_all.groupby(["opp_team_hash", "opp_player_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_losses"})
    )
    short = (
        short_losses.groupby(["opp_team_hash", "opp_player_name"], as_index=False)
        .size()
        .rename(columns={"size": "n_short_losses"})
    )
    if overall.empty:
        opp_team_freq = pd.DataFrame(
            columns=[
                "opp_team_hash",
                "opp_player_name",
                "n_short_losses",
                "n_losses",
                "short_fraction",
                "over_representation",
            ]
        )
    else:
        # Outer join so opp_teams with zero short losses still appear
        # (over_representation = 0 for them).
        opp_team_freq = overall.merge(
            short, on=["opp_team_hash", "opp_player_name"], how="left"
        )
        opp_team_freq["n_short_losses"] = (
            opp_team_freq["n_short_losses"].fillna(0).astype(int)
        )
        opp_team_freq["short_fraction"] = (
            opp_team_freq["n_short_losses"] / opp_team_freq["n_losses"]
        )
        total_losses = len(losses_all)
        total_short_losses = len(short_losses)
        baseline_rate = total_short_losses / total_losses if total_losses > 0 else 0.0
        # Over-representation: how much more likely is this team to
        # short-loss us vs. the average opp_team. 1.0 = same as
        # average; >1 = worse than average; <1 = better than average.
        opp_team_freq["over_representation"] = (
            opp_team_freq["short_fraction"] / baseline_rate
            if baseline_rate > 0
            else float("nan")
        )

    # Action distribution in short losses vs overall.
    if turns is None or turns.empty:
        return opp_team_freq, None

    short_battle_ids = set(short_losses["battle_id"])
    short_turn_actions = turns[turns["battle_id"].isin(short_battle_ids)][
        "action_chosen_str"
    ].value_counts()
    all_turn_actions = turns["action_chosen_str"].value_counts()
    actions_df = pd.DataFrame(
        {
            "action_chosen_str": all_turn_actions.index,
            "n_short_loss_turns": [
                int(short_turn_actions.get(a, 0)) for a in all_turn_actions.index
            ],
            "n_all_turns": all_turn_actions.values,
        }
    )
    total_short = actions_df["n_short_loss_turns"].sum()
    total_all = actions_df["n_all_turns"].sum()
    if total_short > 0 and total_all > 0:
        actions_df["short_rate"] = actions_df["n_short_loss_turns"] / total_short
        actions_df["overall_rate"] = actions_df["n_all_turns"] / total_all
        # Expected count under the null = overall_rate × total_short.
        expected = actions_df["overall_rate"] * total_short
        actions_df["chi_sq_residual"] = (
            actions_df["n_short_loss_turns"] - expected
        ) / expected.pow(0.5)
    else:
        actions_df["short_rate"] = float("nan")
        actions_df["overall_rate"] = float("nan")
        actions_df["chi_sq_residual"] = float("nan")
    return opp_team_freq, actions_df


def q7_value_calibration(
    battles: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    n_bins: int = 10,
    per_opp_type: bool = False,
) -> Tuple[pd.DataFrame, float]:
    """Q7: value-head calibration via reliability diagram + ECE.

    For each turn we know the model's ``value_predicted`` (a scalar in
    roughly [-1, 1]) and, by joining on ``battle_id``, the eventual
    outcome of that turn's battle (1 / 0 / NaN). The C51 head is
    trained to predict a distribution that integrates to expected
    return; in expectation, ``value_predicted ≈ E[outcome | state]``,
    so calibration is "does the predicted value match the actual win
    rate from states with that value?"

    Returns ``(reliability_df, ece)``:

    * ``reliability_df``: one row per bin with columns
      ``bin_lower, bin_upper, n_turns, mean_predicted, observed_win_rate``.
      Bins are equal-width over the value range.
    * ``ece``: scalar Expected Calibration Error = Σ wᵢ·|predᵢ - obsᵢ|
      where wᵢ is the fraction of turns in bin i.

    Mapping value_predicted (∈ [-1, 1]) to "win probability" assumes
    the model's value support is symmetric and outcome is 0/1. We
    rescale via ``(value + 1) / 2`` so bins are on the probability
    scale and the ECE is directly interpretable.

    ``per_opp_type=True`` would facet, but we currently do not — the
    caller can compute per-slice ECE by pre-filtering ``battles`` /
    ``turns``.
    """
    if turns.empty or battles.empty:
        return pd.DataFrame(
            columns=[
                "bin_lower",
                "bin_upper",
                "n_turns",
                "mean_predicted",
                "observed_win_rate",
            ]
        ), float("nan")

    # Join turn rows with their battle's outcome.
    outcome_by_battle = battles.set_index("battle_id")["outcome"]
    turns_with_outcome = turns.assign(outcome=turns["battle_id"].map(outcome_by_battle))
    # Drop teampreview turns — value_predicted there is degenerate
    # (the value head sees no in-battle state) and would skew the
    # diagram. Also drop ties (NaN outcome).
    turns_with_outcome = turns_with_outcome[
        ~turns_with_outcome["is_teampreview"] & turns_with_outcome["outcome"].notna()
    ]
    if turns_with_outcome.empty:
        return pd.DataFrame(), float("nan")

    # Rescale value_predicted from [-1, 1] to [0, 1] (probability scale).
    turns_with_outcome = turns_with_outcome.assign(
        pred_prob=(turns_with_outcome["value_predicted"] + 1) / 2
    )
    # Clip to [0, 1] in case the value head over- or under-shoots
    # (C51 support is [-1, 1] but numerical noise can push slightly out).
    turns_with_outcome["pred_prob"] = turns_with_outcome["pred_prob"].clip(0.0, 1.0)

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    # right=True (default) so 1.0 falls in the last bin; left edge is
    # inclusive only for the very first bin via include_lowest.
    bin_idx = pd.cut(
        turns_with_outcome["pred_prob"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )
    turns_with_outcome = turns_with_outcome.assign(bin_idx=bin_idx)

    grouped = turns_with_outcome.groupby("bin_idx", as_index=False).agg(
        n_turns=("pred_prob", "size"),
        mean_predicted=("pred_prob", "mean"),
        observed_win_rate=("outcome", "mean"),
    )
    grouped["bin_lower"] = (
        grouped["bin_idx"].astype(int).map({i: bin_edges[i] for i in range(n_bins)})
    )
    grouped["bin_upper"] = (
        grouped["bin_idx"].astype(int).map({i: bin_edges[i + 1] for i in range(n_bins)})
    )
    grouped = (
        grouped[
            ["bin_lower", "bin_upper", "n_turns", "mean_predicted", "observed_win_rate"]
        ]
        .sort_values("bin_lower")
        .reset_index(drop=True)
    )

    # ECE — bucket-weighted absolute calibration gap.
    total_n = grouped["n_turns"].sum()
    ece = float(
        (
            (grouped["mean_predicted"] - grouped["observed_win_rate"]).abs()
            * grouped["n_turns"]
            / total_n
        ).sum()
    )
    return grouped, ece


def compute_ensemble_advantage(
    heuristic_advs: List[float], final_outcome: float
) -> List[float]:
    """Per-turn blended advantage: heuristic + outcome with step-dependent weights.

    Ported from ``etl/battle_dataset.py:_compute_ensemble_advantage``
    so the analysis pipeline (Q9) can compare model values against
    the same label distribution BC was trained on. The blend ramps
    smoothly from "pure heuristic" early to "pure outcome" late:

      outcome_weight   = clip(progress², 0.05, 0.95)
      position_weight  = 1 - outcome_weight
      blended[t]       = position_weight × (0.5·heuristic[t] + 0.5·avg_next_3)
                       + outcome_weight × final_outcome

    where ``progress = t / (n-1)`` and ``avg_next_3`` averages
    heuristic_advs over the 3 turns following t (inclusive). The
    bounds [0.05, 0.95] keep heuristic from being completely ignored
    at the end and outcome from being completely ignored at the start.

    For battles with n=1 just returns ``[final_outcome]``.
    ``final_outcome`` should be in {-1, +1} (loss/win) or 0 for ties;
    the function does not validate this.
    """
    n = len(heuristic_advs)
    if n == 0:
        return []
    if n == 1:
        return [final_outcome]
    out: List[float] = []
    for i in range(n):
        progress = i / max(n - 1, 1)
        outcome_weight = min(max(progress * progress, 0.05), 0.95)
        position_weight = 1.0 - outcome_weight
        current = heuristic_advs[i]
        next_window = heuristic_advs[i + 1 : min(i + 4, n)]
        avg_next = sum(next_window) / len(next_window) if next_window else current
        blended = (
            position_weight * (0.5 * current + 0.5 * avg_next)
            + outcome_weight * final_outcome
        )
        out.append(blended)
    return out


def _attach_ensemble_advantage(battles: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Return turns_df with an ``ensemble_adv`` column attached.

    For each battle, computes ``compute_ensemble_advantage`` over the
    in-battle (non-teampreview) turns and writes it back per row.
    Battles missing an outcome (ties) get NaN ensemble_adv across all
    their turns.
    """
    if turns.empty:
        return turns.assign(ensemble_adv=pd.Series(dtype=float))

    # Map battle_id -> +1/-1/0(NaN-tie).
    outcome_map = battles.set_index("battle_id")["outcome"]
    enriched_rows: List[pd.Series] = []
    for battle_id, group in turns.groupby("battle_id", sort=False):
        outcome = outcome_map.get(battle_id, float("nan"))
        if pd.isna(outcome):
            ensemble = [float("nan")] * len(group)
        else:
            # +1 / -1 sign convention from _compute_ensemble_advantage.
            signed_outcome = 1.0 if outcome > 0.5 else -1.0
            # Compute over non-teampreview turns; teampreview turns
            # don't have meaningful heuristic_adv, hold them at NaN.
            in_battle = group[~group["is_teampreview"]]
            heuristic_seq = in_battle["heuristic_adv"].tolist()
            ensemble_seq = compute_ensemble_advantage(heuristic_seq, signed_outcome)
            ensemble_by_index = dict(zip(in_battle.index, ensemble_seq))
            ensemble = [ensemble_by_index.get(idx, float("nan")) for idx in group.index]
        g = group.assign(ensemble_adv=ensemble)
        enriched_rows.append(g)
    return pd.concat(enriched_rows).sort_index()


def q6_confidence_in_poor_situations(
    battles: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    swing_window: int = 3,
    swing_threshold: float = -0.5,
    poor_threshold: float = -0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Q6: did the model see "poor situations" coming?

    Three tables:

    * ``confidence_by_bucket``: per heuristic_adv quartile, mean
      policy_entropy and mean value_predicted. A healthy model has
      *bimodal* entropy (confident in extremes, uncertain in messy
      mid-game), so a flat-entropy profile is a red flag.

    * ``swing_events``: per losing battle, the worst (most negative)
      swing ``heuristic_adv[t] - heuristic_adv[t-swing_window]``. For
      each, ``value_at_t_minus_k`` is the model's value prediction at
      the start of the swing — was it worried (low) or oblivious
      (high)? Top rows surface the worst "sudden collapse" battles.

    * ``poor_situation_summary``: aggregate confidence comparison
      between ``heuristic_adv < poor_threshold`` (the "poor situation"
      bucket) and all turns. Tells you whether the model is more or
      less confident than average when the heuristic says we're
      losing.

    All inputs use ``heuristic_adv`` (the raw position score, no
    outcome leak) — the question is whether the model's confidence
    *at the time* tracked the heuristic's *at the time*, not whether
    in hindsight it should have known.
    """
    confidence_cols = [
        "bucket",
        "n_turns",
        "mean_entropy",
        "mean_value_predicted",
        "mean_heuristic_adv",
    ]
    swing_cols = [
        "battle_id",
        "t_after",
        "t_before",
        "swing",
        "heuristic_adv_before",
        "heuristic_adv_after",
        "value_at_t_minus_k",
        "value_at_t",
        "entropy_at_t_minus_k",
        "entropy_at_t",
    ]
    summary_cols = [
        "subset",
        "n_turns",
        "mean_entropy",
        "mean_value_predicted",
    ]
    if turns.empty or battles.empty:
        return (
            pd.DataFrame(columns=confidence_cols),
            pd.DataFrame(columns=swing_cols),
            pd.DataFrame(columns=summary_cols),
        )

    # Drop teampreview turns from analysis — heuristic_adv is not
    # meaningful there.
    in_battle = turns[~turns["is_teampreview"]].copy()
    in_battle = in_battle.dropna(subset=["heuristic_adv"])

    # Table 1: confidence by heuristic_adv quartile.
    quartiles = pd.qcut(in_battle["heuristic_adv"], q=4, duplicates="drop")
    by_q = in_battle.groupby(quartiles, observed=True, as_index=False).agg(
        n_turns=("policy_entropy", "size"),
        mean_entropy=("policy_entropy", "mean"),
        mean_value_predicted=("value_predicted", "mean"),
        mean_heuristic_adv=("heuristic_adv", "mean"),
    )
    by_q = by_q.rename(columns={by_q.columns[0]: "bucket"})
    by_q["bucket"] = by_q["bucket"].astype(str)
    confidence_by_bucket = by_q

    # Table 2: worst negative swings in losses.
    losing_battle_ids = set(battles[battles["outcome"] == 0.0]["battle_id"].tolist())
    swing_records = []
    for battle_id, group in in_battle.groupby("battle_id", sort=False):
        if battle_id not in losing_battle_ids:
            continue
        group = group.sort_values("turn_number").reset_index(drop=True)
        if len(group) <= swing_window:
            continue
        # Compute swing[t] = adv[t] - adv[t - k]
        shifted = group["heuristic_adv"].shift(swing_window)
        swing = group["heuristic_adv"] - shifted
        # Find the most negative swing.
        worst_idx = swing.idxmin()
        if pd.isna(worst_idx) or pd.isna(swing[worst_idx]):
            continue
        worst_swing = float(swing[worst_idx])
        if worst_swing > swing_threshold:
            continue  # Not a "large" negative swing.
        t_before_pos = worst_idx - swing_window
        if t_before_pos < 0:
            continue
        before_row = group.iloc[t_before_pos]
        after_row = group.iloc[worst_idx]
        swing_records.append(
            {
                "battle_id": battle_id,
                "t_before": int(before_row["turn_number"]),
                "t_after": int(after_row["turn_number"]),
                "swing": worst_swing,
                "heuristic_adv_before": float(before_row["heuristic_adv"]),
                "heuristic_adv_after": float(after_row["heuristic_adv"]),
                "value_at_t_minus_k": float(before_row["value_predicted"]),
                "value_at_t": float(after_row["value_predicted"]),
                "entropy_at_t_minus_k": float(before_row["policy_entropy"]),
                "entropy_at_t": float(after_row["policy_entropy"]),
            }
        )
    swing_events = pd.DataFrame(swing_records, columns=swing_cols).sort_values("swing")

    # Table 3: poor-situation summary.
    poor = in_battle[in_battle["heuristic_adv"] < poor_threshold]
    summary_records = [
        {
            "subset": "all",
            "n_turns": len(in_battle),
            "mean_entropy": float(in_battle["policy_entropy"].mean()),
            "mean_value_predicted": float(in_battle["value_predicted"].mean()),
        },
        {
            "subset": f"heuristic_adv < {poor_threshold}",
            "n_turns": len(poor),
            "mean_entropy": (
                float(poor["policy_entropy"].mean()) if len(poor) > 0 else float("nan")
            ),
            "mean_value_predicted": (
                float(poor["value_predicted"].mean()) if len(poor) > 0 else float("nan")
            ),
        },
    ]
    poor_situation_summary = pd.DataFrame(summary_records, columns=summary_cols)
    return confidence_by_bucket, swing_events, poor_situation_summary


def q9a_persistent_disagreement(
    battles: pd.DataFrame, turns: pd.DataFrame, *, top_n: int = 20
) -> pd.DataFrame:
    """Q9a: battles where the value head persistently disagrees with the
    ensemble-advantage signal.

    For each non-tied battle, compute the per-turn absolute difference
    between rescaled value (mapped to the ensemble's [-1, +1] support)
    and ``ensemble_adv``, then average across the battle. Returns the
    top-N battles ranked by mean disagreement.

    The rescaling matches Q7's convention (value sits on [-1, 1] from
    C51 support; ensemble_adv sits on the same support after blending
    heuristic + outcome). The two are directly comparable without
    further normalization.
    """
    cols = [
        "battle_id",
        "opp_player_name",
        "mean_abs_diff",
        "n_turns",
        "outcome",
        "final_turn",
    ]
    if turns.empty or battles.empty:
        return pd.DataFrame(columns=cols)

    enriched = _attach_ensemble_advantage(battles, turns)
    in_battle = enriched[~enriched["is_teampreview"]].dropna(
        subset=["ensemble_adv", "value_predicted"]
    )
    if in_battle.empty:
        return pd.DataFrame(columns=cols)

    in_battle = in_battle.assign(
        abs_diff=(in_battle["value_predicted"] - in_battle["ensemble_adv"]).abs()
    )

    agg = in_battle.groupby("battle_id", as_index=False).agg(
        mean_abs_diff=("abs_diff", "mean"),
        n_turns=("abs_diff", "size"),
    )
    meta = battles.set_index("battle_id")[["opp_player_name", "outcome", "final_turn"]]
    agg = agg.merge(meta, left_on="battle_id", right_index=True, how="left")
    agg = agg.sort_values("mean_abs_diff", ascending=False).head(top_n)
    return agg[cols].reset_index(drop=True)


def q9b_agree_then_diverge(
    battles: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    early_threshold: float = 0.15,
    late_threshold: float = 0.40,
    early_window: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """Q9b: battles where value and ensemble agree early then diverge.

    Detection: for each battle (≥``early_window+1`` in-battle turns),
    compute |value - ensemble| per turn. A battle "agrees-then-
    diverges" when the mean over the first ``early_window`` turns is
    < ``early_threshold`` AND the mean over the rest is >
    ``late_threshold``. Returns the top-N by ``post_split_diff -
    pre_split_diff`` (most dramatic break).

    Operational meaning: the model's read of the battle started in
    line with the heuristic-plus-outcome reference, then the value
    head's trajectory diverged. Could indicate the model picked a
    strategy the heuristic doesn't understand, OR the value head got
    confused mid-battle. Either way it's worth eyeballing — Q8 will
    save these for human inspection.
    """
    cols = [
        "battle_id",
        "opp_player_name",
        "pre_split_diff",
        "post_split_diff",
        "diff_increase",
        "n_turns",
        "outcome",
        "final_turn",
    ]
    if turns.empty or battles.empty:
        return pd.DataFrame(columns=cols)

    enriched = _attach_ensemble_advantage(battles, turns)
    in_battle = enriched[~enriched["is_teampreview"]].dropna(
        subset=["ensemble_adv", "value_predicted"]
    )
    if in_battle.empty:
        return pd.DataFrame(columns=cols)
    in_battle = in_battle.assign(
        abs_diff=(in_battle["value_predicted"] - in_battle["ensemble_adv"]).abs()
    )

    records = []
    for battle_id, group in in_battle.groupby("battle_id", sort=False):
        group = group.sort_values("turn_number")
        if len(group) <= early_window:
            continue
        pre = group.iloc[:early_window]["abs_diff"]
        post = group.iloc[early_window:]["abs_diff"]
        pre_mean = float(pre.mean())
        post_mean = float(post.mean())
        if pre_mean >= early_threshold:
            continue
        if post_mean <= late_threshold:
            continue
        records.append(
            {
                "battle_id": battle_id,
                "pre_split_diff": pre_mean,
                "post_split_diff": post_mean,
                "diff_increase": post_mean - pre_mean,
                "n_turns": int(len(group)),
            }
        )
    if not records:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(records)
    meta = battles.set_index("battle_id")[["opp_player_name", "outcome", "final_turn"]]
    df = df.merge(meta, left_on="battle_id", right_index=True, how="left")
    df = df.sort_values("diff_increase", ascending=False).head(top_n)
    return df[cols].reset_index(drop=True)


def q8_save_games(
    battles: pd.DataFrame,
    turns: pd.DataFrame,
    *,
    run_dir: str,
    n_per_category: int = 3,
    short_loss_max_turn: int = 5,
    team_wr_threshold: float = 0.25,
    persistent_diff_min: float = 0.5,
    diverge_diff_increase_min: float = 0.3,
    min_battles_per_team: int = 10,
    rng_seed: Optional[int] = 0,
) -> Dict[str, List[str]]:
    """Q8: dump ``n_per_category`` battles for each of 5 categories.

    Categories:
      * ``short_loss`` — outcome=0 AND final_turn ≤ short_loss_max_turn.
      * ``team_we_lose_with`` — outcome=0 AND agent_team's overall WR
        across ≥min_battles_per_team battles is below team_wr_threshold.
      * ``team_we_lose_to`` — outcome=0 AND opp_team's "our-WR" across
        ≥min_battles_per_team battles is below team_wr_threshold (i.e.
        their WR over us is > 1-threshold).
      * ``value_vs_ensemble_persistent`` — top Q9a battles whose
        mean_abs_diff ≥ ``persistent_diff_min``.
      * ``value_vs_ensemble_diverge`` — top Q9b battles whose
        diff_increase ≥ ``diverge_diff_increase_min``.

    Selection: from each candidate pool we keep only battles with
    ``replay_saved=True`` (so we can dump the Showdown log), then
    uniformly random-sample ``n_per_category``. Ties in ranking are
    broken by random sample, per the user's earlier choice.

    For each selected battle, write to ``<run_dir>/saved_games/<category>/``:
      * ``<battle_id>.log.gz`` — the gzipped Showdown protocol log
        (copied from ``<run_dir>/replays/``).
      * ``<battle_id>.json`` — sidecar with the battle row and the
        per-turn records (action, value, entropy, top-K, heuristic_adv,
        ensemble_adv). Lets future-you (or someone unfamiliar with the
        run) read the model's reasoning alongside the replay without
        loading parquet.

    Returns a dict ``{category: [battle_id, ...]}`` for the caller to
    log / verify.
    """
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    saved_root = os.path.join(run_dir, "saved_games")
    os.makedirs(saved_root, exist_ok=True)

    saved: Dict[str, List[str]] = {}

    # Pre-compute Q1/Q2 WR tables for category filters; restrict to
    # team_hashes with at least min_battles_per_team battles.
    q1 = q1_agent_team_win_rate(battles)
    q2 = q2_opp_team_win_rate(battles)
    agent_team_overall = (
        q1.groupby("agent_team_hash")
        .agg(n=("n_battles", "sum"), wins=("wins", "sum"), losses=("losses", "sum"))
        .reset_index()
    )
    agent_team_overall["wr"] = agent_team_overall["wins"] / (
        agent_team_overall["wins"] + agent_team_overall["losses"]
    ).replace(0, float("nan"))
    bad_agent_teams = set(
        agent_team_overall[
            (agent_team_overall["n"] >= min_battles_per_team)
            & (agent_team_overall["wr"] < team_wr_threshold)
        ]["agent_team_hash"]
    )

    opp_team_overall = (
        q2.groupby("opp_team_hash")
        .agg(n=("n_battles", "sum"), wins=("wins", "sum"), losses=("losses", "sum"))
        .reset_index()
    )
    opp_team_overall["wr"] = opp_team_overall["wins"] / (
        opp_team_overall["wins"] + opp_team_overall["losses"]
    ).replace(0, float("nan"))
    tough_opp_teams = set(
        opp_team_overall[
            (opp_team_overall["n"] >= min_battles_per_team)
            & (opp_team_overall["wr"] < team_wr_threshold)
        ]["opp_team_hash"]
    )

    # Build candidate pools (battle_ids) per category.
    only_with_replay = battles[battles["replay_saved"]]
    decisive_losses = only_with_replay[only_with_replay["outcome"] == 0.0]

    pools: Dict[str, List[str]] = {}
    pools["short_loss"] = decisive_losses[
        decisive_losses["final_turn"] <= short_loss_max_turn
    ]["battle_id"].tolist()
    pools["team_we_lose_with"] = decisive_losses[
        decisive_losses["agent_team_hash"].isin(bad_agent_teams)
    ]["battle_id"].tolist()
    pools["team_we_lose_to"] = decisive_losses[
        decisive_losses["opp_team_hash"].isin(tough_opp_teams)
    ]["battle_id"].tolist()

    # Q9-driven pools — require Q9a/Q9b to be computed with the same
    # filter thresholds. We take a generous top_n then restrict to
    # battles that have replays.
    q9a = q9a_persistent_disagreement(battles, turns, top_n=200)
    q9a_pool = q9a[q9a["mean_abs_diff"] >= persistent_diff_min]["battle_id"].tolist()
    q9b = q9b_agree_then_diverge(battles, turns, top_n=200)
    q9b_pool = q9b[q9b["diff_increase"] >= diverge_diff_increase_min]["battle_id"].tolist()
    replay_ids = set(only_with_replay["battle_id"])
    pools["value_vs_ensemble_persistent"] = [bid for bid in q9a_pool if bid in replay_ids]
    pools["value_vs_ensemble_diverge"] = [bid for bid in q9b_pool if bid in replay_ids]

    # Sample n_per_category from each pool and dump.
    battle_by_id = battles.set_index("battle_id")
    if turns.empty:
        turns_by_battle: Dict[str, List[dict]] = {}
    else:
        turns_by_battle = {
            bid: g.sort_values("turn_number").to_dict(orient="records")
            for bid, g in turns.groupby("battle_id", sort=False)
        }

    for category, pool in pools.items():
        cat_dir = os.path.join(saved_root, category)
        os.makedirs(cat_dir, exist_ok=True)
        n = min(n_per_category, len(pool))
        picks = rng.sample(pool, n) if n > 0 else []
        saved[category] = picks
        for battle_id in picks:
            _dump_saved_game(
                battle_id=battle_id,
                category=category,
                battle_record=battle_by_id.loc[battle_id].to_dict(),
                turn_records=turns_by_battle.get(battle_id, []),
                run_dir=run_dir,
                cat_dir=cat_dir,
            )

    return saved


def _dump_saved_game(
    *,
    battle_id: str,
    category: str,
    battle_record: dict,
    turn_records: List[dict],
    run_dir: str,
    cat_dir: str,
) -> None:
    """Copy replay log + write JSON sidecar for one saved battle."""
    src_log = os.path.join(run_dir, "replays", f"{battle_id}.log.gz")
    dst_log = os.path.join(cat_dir, f"{battle_id}.log.gz")
    if os.path.exists(src_log):
        shutil.copy2(src_log, dst_log)
    sidecar = {
        "battle_id": battle_id,
        "category": category,
        "battle": _jsonify(battle_record),
        "turns": [_jsonify(t) for t in turn_records],
    }
    sidecar_path = os.path.join(cat_dir, f"{battle_id}.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)


def _jsonify(d: dict) -> dict:
    """Best-effort conversion of non-JSON-serializable types in row dicts.

    Pandas / numpy scalars (int64, float64, NaN, Timestamp) don't
    JSON-encode by default. We coerce them to Python natives so the
    sidecars stay readable without a custom decoder.
    """
    out = {}
    for k, v in d.items():
        if pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            try:
                out[k] = v.item()
                continue
            except (ValueError, AttributeError):
                pass
        else:
            out[k] = v
    return out


def _group_win_rate(battles: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Group + aggregate win rate with Wilson CIs.

    Shared implementation for Q1/Q2/Q3. Ties (NaN outcome) are
    excluded from win_rate but reported in a ``ties`` column.
    """
    if battles.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["n_battles", "wins", "losses", "ties", "win_rate", "ci_low", "ci_high"]
        )

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        ties = int(group["outcome"].isna().sum())
        decisive = group.dropna(subset=["outcome"])
        wins = int(decisive["outcome"].sum())
        losses = len(decisive) - wins
        denom = len(decisive)
        wr = wins / denom if denom > 0 else float("nan")
        ci_low, ci_high = wilson_ci(wins, denom)
        return pd.Series(
            {
                "n_battles": n,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate": wr,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    grouped = (
        battles.groupby(group_cols, as_index=False, sort=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    # Restore group_col values dropped by include_groups=False.
    return grouped


# ─── Output formatting ───────────────────────────────────────────────


def _format_df(df: pd.DataFrame, fmt: str) -> str:
    """Render a DataFrame as csv / markdown / json / table (default)."""
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "markdown":
        return df.to_markdown(index=False, floatfmt=".4f")
    if fmt == "json":
        return df.to_json(orient="records", indent=2)
    # default: pandas string repr with sensible width
    with pd.option_context(
        "display.max_rows", 200, "display.max_columns", 20, "display.width", 200
    ):
        return df.to_string(index=False)


def _emit(df: pd.DataFrame, *, output: Optional[str], fmt: str) -> None:
    rendered = _format_df(df, fmt)
    if output:
        with open(output, "w") as f:
            f.write(rendered)
        print(f"wrote {output}")
    else:
        print(rendered)


# ─── CLI ─────────────────────────────────────────────────────────────


_TODO_MESSAGE = "(not yet implemented — pending Plan B Phase B5)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a Plan B trajectory-collection run dir"
    )
    parser.add_argument("run_dir", type=str, help="Trajectory collection run dir")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the result table to this path instead of stdout",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "markdown", "json"],
        default="table",
        help="Output format (default: table — pandas string repr)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="Q3: WR by opp_player_name")
    sub.add_parser("agent_team", help="Q1: WR by (agent_team_hash, opp_player_name)")
    sub.add_parser("opp_team", help="Q2: WR by (opp_team_hash, opp_player_name)")
    sub.add_parser("short_loss", help="Q5: short-loss patterns")
    sub.add_parser("confidence", help="Q6: confidence vs heuristic_adv")
    sub.add_parser("value_calibration", help="Q7: reliability + ECE")
    sub.add_parser("value_ensemble", help="Q9: value vs ensemble disagreement")
    sub.add_parser("save_games", help="Q8: dump 3 games per category")
    sub.add_parser("report", help="Render full report (HTML)")

    args = parser.parse_args()
    battles = read_battles(args.run_dir)
    if battles.empty:
        print(f"No battle records found under {args.run_dir!r}")
        return

    if args.cmd == "summary":
        result = q3_opp_type_win_rate(battles)
    elif args.cmd == "agent_team":
        result = q1_agent_team_win_rate(battles).sort_values("win_rate")
    elif args.cmd == "opp_team":
        result = q2_opp_team_win_rate(battles).sort_values("win_rate")
    elif args.cmd == "short_loss":
        turns = read_turns(args.run_dir)
        opp_freq, actions = q5_short_loss_patterns(battles, turns)
        opp_freq = opp_freq.sort_values("over_representation", ascending=False)
        print("=== Over-represented opp_teams in short losses ===")
        _emit(opp_freq, output=None, fmt=args.format)
        if actions is not None:
            print("\n=== Action distribution: short-loss turns vs all turns ===")
            actions = actions.sort_values("chi_sq_residual", ascending=False).head(20)
            _emit(actions, output=args.output, fmt=args.format)
        return
    elif args.cmd == "value_calibration":
        turns = read_turns(args.run_dir)
        reliability, ece = q7_value_calibration(battles, turns)
        print(f"=== Value-head reliability (ECE = {ece:.4f}) ===")
        result = reliability
    elif args.cmd == "confidence":
        turns = read_turns(args.run_dir)
        by_bucket, swings, summary = q6_confidence_in_poor_situations(battles, turns)
        print("=== Confidence by heuristic_adv quartile ===")
        _emit(by_bucket, output=None, fmt=args.format)
        print("\n=== Poor-situation summary ===")
        _emit(summary, output=None, fmt=args.format)
        print("\n=== Worst negative swings in losses (top 20) ===")
        _emit(swings.head(20), output=args.output, fmt=args.format)
        return
    elif args.cmd == "value_ensemble":
        turns = read_turns(args.run_dir)
        persistent = q9a_persistent_disagreement(battles, turns, top_n=20)
        diverge = q9b_agree_then_diverge(battles, turns, top_n=20)
        print("=== Q9a: persistent value-vs-ensemble disagreement (top 20) ===")
        _emit(persistent, output=None, fmt=args.format)
        print("\n=== Q9b: agree-then-diverge (top 20) ===")
        _emit(diverge, output=args.output, fmt=args.format)
        return
    elif args.cmd == "save_games":
        turns = read_turns(args.run_dir)
        saved = q8_save_games(battles, turns, run_dir=args.run_dir)
        for category, ids in saved.items():
            print(f"=== {category}: {len(ids)} games saved ===")
            for bid in ids:
                print(f"  {bid}")
        return
    elif args.cmd == "report":
        print(f"report: {_TODO_MESSAGE}")
        return
    else:
        raise AssertionError(f"unreachable subcommand: {args.cmd!r}")

    _emit(result, output=args.output, fmt=args.format)


if __name__ == "__main__":
    main()
