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

from elitefurretai.rl.analyze.eval_schema import read_battles

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
    elif args.cmd in {
        "short_loss",
        "confidence",
        "value_calibration",
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
