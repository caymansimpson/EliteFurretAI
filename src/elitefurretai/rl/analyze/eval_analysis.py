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
import math
from typing import Optional, Tuple

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
    elif args.cmd in {
        "confidence",
        "value_ensemble",
        "save_games",
        "report",
    }:
        print(f"{args.cmd}: {_TODO_MESSAGE}")
        return
    else:
        raise AssertionError(f"unreachable subcommand: {args.cmd!r}")

    _emit(result, output=args.output, fmt=args.format)


if __name__ == "__main__":
    main()
