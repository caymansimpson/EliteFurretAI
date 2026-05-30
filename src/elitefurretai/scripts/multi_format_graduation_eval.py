"""Run the Stage II graduation matrix (4 baselines x N formats) for a checkpoint.

Reads `battle_formats` from an RL config (or accepts `--formats` directly),
shells out to `evaluate.py` once per format with each of the four required
baselines, concatenates the parquet shards, and prints a graduation matrix.

VGCBench v1 is used uniformly across formats — cross-format VGCBench numbers
are slightly noisier but acceptable per the user's spec for this plan.

The four required baselines are passed to evaluate.py as opponent specs that
``parse_player_specification`` understands:

* ``max_damage`` / ``vgc_bench`` / ``simple_heuristic`` — canonical baseline names.
* BC — passed as the bare path to a BC checkpoint .pt file (resolves to
  ``kind="model"``). evaluate.py records its ``opp_player_name`` as the
  checkpoint's filename stem, which the driver remaps to ``"bc_player"``
  before calling ``graduation_summary`` so the default required_opp_types
  match.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from elitefurretai.rl.analyze.matchup_analysis import graduation_summary
from elitefurretai.rl.config import RNaDConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to RL .pt checkpoint")
    parser.add_argument(
        "--bc-checkpoint",
        required=True,
        help="Path to the BC checkpoint .pt file. Used as the bc_player opponent.",
    )
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

    if args.formats and args.config:
        parser.error("Pass either --formats or --config, not both")
    if args.formats:
        formats = args.formats
    elif args.config:
        cfg = RNaDConfig.load(args.config)
        formats = list(cfg.curriculum.battle_formats.keys())
    else:
        sys.exit("Must provide --formats or --config")

    if not Path(args.checkpoint).is_file():
        sys.exit(f"--checkpoint path does not exist: {args.checkpoint}")

    bc_checkpoint_path = Path(args.bc_checkpoint)
    if not bc_checkpoint_path.is_file():
        sys.exit(f"--bc-checkpoint path does not exist: {bc_checkpoint_path}")
    # evaluate.py records opp_player_name as the checkpoint filename stem for
    # model-kind opponents. We remap this stem to "bc_player" after collecting
    # shards so graduation_summary's default required_opp_types matches.
    bc_player_name = bc_checkpoint_path.stem

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # (display_name, opponent_spec_for_evaluate.py)
    baselines = [
        ("max_damage", "max_damage"),
        ("vgc_bench", "vgc_bench"),
        ("bc_player", str(bc_checkpoint_path)),
        ("simple_heuristic", "simple_heuristic"),
    ]

    for fmt in formats:
        for opp_display, opp_spec in baselines:
            print(f"[run] {fmt} vs {opp_display}")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "elitefurretai.rl.analyze.analysis_utils",
                    "--player1",
                    args.checkpoint,
                    "--player2",
                    opp_spec,
                    "--battle-format",
                    fmt,
                    "--battles",
                    str(args.battles_per_cell),
                    "--collect-trajectories",
                    str(run_dir),
                    "--launch-servers",
                ],
                check=True,
            )

    shards = list(run_dir.rglob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet shards found under {run_dir}")
    battles = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    # evaluate.py records opp_player_name as the checkpoint filename stem for
    # model opponents; remap to the canonical "bc_player" tag so
    # graduation_summary's default required_opp_types matches.
    battles["opp_player_name"] = battles["opp_player_name"].replace(
        {bc_player_name: "bc_player"}
    )

    summary = graduation_summary(
        battles,
        threshold=args.threshold,
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
