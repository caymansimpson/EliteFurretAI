#!/usr/bin/env python3
"""Parse `run.log` files from run_rl_throughput_benchmark.sh and emit a
summary.json with per-update throughput, win rates, and aggregate stats.

Usage:
    python scripts/parse_rl_benchmark.py <log_dir> [<log_dir> ...]

Each log_dir must contain a `run.log` produced by the runner. Writes
`summary.json` next to it. Also prints a one-line headline per run for
quick inspection.

Schema of summary.json:
{
  "config": "...",
  "git_head": "...",
  "elapsed_seconds": int,
  "warmup_skipped": int,            // first N updates excluded from aggregates
  "updates": [                      // raw per-update parse
      {"update": int, "b_per_s": float, "learner_steps_per_s": float,
       "elapsed_s_total": int, "win_rates": {opponent: float}, ...},
      ...
  ],
  "aggregate": {
      "b_per_s_mean": float, "b_per_s_std": float,
      "learner_steps_per_s_mean": float, "learner_steps_per_s_std": float,
      "wall_clock_per_update_mean_s": float
  },
  "win_rate_at_update": {
      "50": {opponent: float, ...},
      "100": {opponent: float, ...},
      "150": {opponent: float, ...},
      "200": {opponent: float, ...}
  }
}
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

# `Update 12: Loss=1.23, ... | Total Battles=148 in 0h 5m 23s (0.46 b/s)
#   | Learner Steps=312 (10.50 steps/s) | Learner Trajectories=148 (0.46 traj/s)
#   | Win rates: self_play: 50.0%, bc_player: 47.0%`
UPDATE_RE = re.compile(
    r"Update (?P<update>\d+):[^\n]*?\|\s*Total Battles=(?P<battles>\d+)\s+in\s+"
    r"(?P<h>\d+)h\s+(?P<m>\d+)m\s+(?P<s>\d+)s\s+\((?P<bps>[\d.]+)\s*b/s\)\s*"
    r"\|\s*Learner Steps=(?P<steps>\d+)\s+\((?P<sps>[\d.]+)\s*steps/s\)"
    r"(?:[^\n]*?\|\s*Win rates:\s*(?P<winrates>[^\n]+))?",
)
WINRATE_PAIR_RE = re.compile(r"(?P<opponent>[\w_]+):\s*(?P<pct>[\d.]+)%")
GIT_HEAD_RE = re.compile(r"git_head=(?P<head>[a-f0-9]+)")
CONFIG_RE = re.compile(r"config=(?P<config>\S+)")
END_RE = re.compile(r"=== END .*?\(exit=(?P<exit>-?\d+),\s*elapsed=(?P<elapsed>\d+)s\)")

WARMUP_UPDATES = 3  # Skip first N updates from aggregate; matches plan §0.2
WIN_RATE_MILESTONES = [50, 100, 150, 200]


def parse_log(path: Path) -> Dict[str, Any]:
    text = path.read_text()

    git_head_m = GIT_HEAD_RE.search(text)
    config_m = CONFIG_RE.search(text)
    end_m = END_RE.search(text)

    updates: List[Dict[str, Any]] = []
    for m in UPDATE_RE.finditer(text):
        h, mn, s = int(m["h"]), int(m["m"]), int(m["s"])
        winrates: Dict[str, float] = {}
        if m["winrates"]:
            for wm in WINRATE_PAIR_RE.finditer(m["winrates"]):
                winrates[wm["opponent"]] = float(wm["pct"]) / 100.0

        updates.append(
            {
                "update": int(m["update"]),
                "b_per_s": float(m["bps"]),
                "learner_steps_per_s": float(m["sps"]),
                "learner_steps": int(m["steps"]),
                "battles": int(m["battles"]),
                "elapsed_s_total": h * 3600 + mn * 60 + s,
                "win_rates": winrates,
            }
        )

    warm = [u for u in updates if u["update"] > WARMUP_UPDATES]
    bps = [u["b_per_s"] for u in warm]
    sps = [u["learner_steps_per_s"] for u in warm]

    # Wall-clock per update (steady state) = (elapsed at last warm update -
    # elapsed at first warm update) / (count - 1). Falls back to mean of
    # first-difference if the steady-state span is too short.
    wall_clock_per_update_mean_s = 0.0
    if len(warm) >= 2:
        spans = [
            warm[i + 1]["elapsed_s_total"] - warm[i]["elapsed_s_total"]
            for i in range(len(warm) - 1)
        ]
        spans = [s for s in spans if s >= 0]
        if spans:
            wall_clock_per_update_mean_s = statistics.mean(spans)

    win_rate_at_update: Dict[str, Dict[str, float]] = {}
    by_update_no = {u["update"]: u for u in updates}
    for milestone in WIN_RATE_MILESTONES:
        if milestone in by_update_no:
            win_rate_at_update[str(milestone)] = by_update_no[milestone]["win_rates"]

    return {
        "config": config_m["config"] if config_m else None,
        "git_head": git_head_m["head"] if git_head_m else None,
        "exit_code": int(end_m["exit"]) if end_m else None,
        "elapsed_seconds": int(end_m["elapsed"]) if end_m else None,
        "warmup_skipped": WARMUP_UPDATES,
        "updates": updates,
        "aggregate": {
            "n_updates_total": len(updates),
            "n_updates_warm": len(warm),
            "b_per_s_mean": statistics.mean(bps) if bps else 0.0,
            "b_per_s_std": statistics.stdev(bps) if len(bps) >= 2 else 0.0,
            "learner_steps_per_s_mean": statistics.mean(sps) if sps else 0.0,
            "learner_steps_per_s_std": statistics.stdev(sps) if len(sps) >= 2 else 0.0,
            "wall_clock_per_update_mean_s": wall_clock_per_update_mean_s,
        },
        "win_rate_at_update": win_rate_at_update,
    }


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    for arg in argv:
        log_dir = Path(arg)
        log = log_dir / "run.log"
        if not log.exists():
            print(f"[skip] {log} does not exist")
            continue

        summary = parse_log(log)
        out = log_dir / "summary.json"
        out.write_text(json.dumps(summary, indent=2))

        agg = summary["aggregate"]
        print(
            f"{log_dir.name}: "
            f"updates={agg['n_updates_total']} "
            f"b/s={agg['b_per_s_mean']:.3f}±{agg['b_per_s_std']:.3f} "
            f"steps/s={agg['learner_steps_per_s_mean']:.2f}±{agg['learner_steps_per_s_std']:.2f} "
            f"sec/update={agg['wall_clock_per_update_mean_s']:.1f} "
            f"elapsed={summary['elapsed_seconds']}s "
            f"exit={summary['exit_code']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
