"""Run the Stage II graduation matrix (4 baselines x N formats) for a checkpoint.

Reads `battle_formats` from an RL config (or accepts `--formats` directly),
shells out to `evaluate.py` once per format with each of the four required
baselines, concatenates the parquet shards, and prints a graduation matrix.

VGCBench v1 is used uniformly across formats — cross-format VGCBench numbers
are slightly noisier but acceptable per the user's spec for this plan.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from elitefurretai.rl.analyze.eval_analysis import graduation_summary
from elitefurretai.rl.config import RNaDConfig

BASELINES = [
    "max_damage",
    "vgc_bench",
    "bc_player",
    "simple_heuristic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to RL .pt checkpoint")
    parser.add_argument(
        "--config",
        default=None,
        help="RL yaml config. If given, --formats defaults to its battle_formats keys.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Override format list (space-separated).",
    )
    parser.add_argument("--battles-per-cell", type=int, default=200)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    if args.formats:
        formats = args.formats
    elif args.config:
        cfg = RNaDConfig.load(args.config)
        formats = list(cfg.curriculum.battle_formats.keys())
    else:
        sys.exit("Must provide --formats or --config")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        for baseline in BASELINES:
            print(f"[run] {fmt} vs {baseline}")
            subprocess.run(
                [
                    sys.executable,
                    "src/elitefurretai/rl/analyze/evaluate.py",
                    "--player1",
                    f"model:{args.checkpoint}",
                    "--player2",
                    baseline,
                    "--team1",
                    f"data/teams/{fmt}/constrained",
                    "--team2",
                    f"data/teams/{fmt}/constrained",
                    "--battle-format",
                    fmt,
                    "--battles",
                    str(args.battles_per_cell),
                    "--collect-trajectories",
                    str(run_dir),
                ],
                check=True,
            )

    shards = list(run_dir.rglob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet shards found under {run_dir}")
    battles = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    summary = graduation_summary(
        battles,
        threshold=args.threshold,
        required_opp_types=tuple(BASELINES),
    )
    print("\n=== Graduation Matrix ===")
    print(f"Threshold: {summary['threshold']:.0%}")
    for cell in summary["cells"]:
        mark = "PASS" if cell["passed"] else "FAIL"
        wr = cell["win_rate"]
        wr_str = "MISSING" if cell.get("missing") else f"{wr:.1%}"
        print(
            f"  {mark} {cell['battle_format']:>22} vs {cell['opp_player_name']:>18}: "
            f"{wr_str:>8}  (n={cell['n_battles']})"
        )
    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}")
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
