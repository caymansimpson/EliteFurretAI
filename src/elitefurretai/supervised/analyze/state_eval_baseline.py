#!/usr/bin/env python3
"""
State Evaluation vs Win Advantage Correlation Analysis

This script analyzes the correlation between simple heuristic state evaluation
(from evaluate_state.py) and the actual win advantage targets used in training.
This provides a baseline to compare against model predictions in win_model_diagnostics.py.

The win advantage target is an ensemble of:
1. Current position evaluation (heuristic)
2. Near-future position evaluation (3 turns ahead)
3. Final outcome (who actually won)

This script helps us understand:
- How well does the simple heuristic correlate with actual outcomes?
- When does the heuristic fail to predict the final result?
- What is the baseline performance we should expect from a learned model?

Usage:
    python state_eval_baseline.py <filter_json> <num_battles> [--output results.json] [--both-perspectives]

Example:
    python state_eval_baseline.py data/battles/supervised_battle_files_no_commander.json 1000 --output baseline.json

Results from a 10K battle run (both perspectives) on VGC 2023 Regulation C:
================================================================================
STATE EVALUATION BASELINE ANALYSIS
================================================================================

--- Overall Statistics ---
Correlation with Win Advantage: 0.4544
MAE vs Win Advantage: 0.3730
MSE vs Win Advantage: 0.2613

Outcome Prediction Accuracy: 41.48%
Precision: 0.00%
Recall: 0.00%
Total Samples: 87,444

--- Correlation by Turn Number ---
Turn     Correlation    Accuracy     Samples
--------------------------------------------------
0        0.6620         99.96%       10,004
1        0.5442         7.58%        12,250
2        0.6105         27.50%       12,421
3        0.5563         36.13%       11,781
4        0.4855         41.22%       10,564
34       1.0000         50.00%       2
35       1.0000         50.00%       2
36       1.0000         50.00%       2
37       nan            100.00%      1
38       nan            100.00%      1

--- Correlation by Turns Until End ---
Turns Left   Correlation    Accuracy     Samples
----------------------------------------------------
0            0.6119         41.72%       10,000
1            0.4025         43.10%       9,964
2            0.5530         41.90%       9,698
3            0.6721         41.09%       9,347
4            0.7712         39.02%       8,839
5            0.8521         37.48%       8,220
6            0.9086         37.95%       7,426
7            0.9470         40.73%       6,376
8            0.9667         44.30%       5,066
9            0.9768         44.80%       3,757

--- Correlation by Actual Outcome ---

Losing Games:
  Correlation: 0.4544
  MAE: 0.3730
  Samples: 87,444

"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import orjson
from tqdm import tqdm

from elitefurretai.etl.battle_iterator import BattleIterator
from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.evaluate_state import evaluate_position_advantage


class StateEvaluationBaseline:
    """Analyzes correlation between heuristic evaluation and win advantage."""

    def __init__(self):
        # Storage for analysis
        self.by_turn: Dict[int, List[Tuple[float, float, bool]]] = defaultdict(list)
        self.by_turns_to_end: Dict[int, List[Tuple[float, float, bool]]] = defaultdict(list)
        self.by_outcome: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.all_data: List[Tuple[float, float, bool]] = []  # (heuristic, win_adv, actual_win)

    def analyze_battle(
        self,
        battle_data: BattleData,
        perspective: str = "p1",
    ) -> None:
        """
        Analyze a single battle by replaying it with BattleIterator.

        Args:
            battle_data: BattleData object containing the battle logs
            perspective: Which player's perspective to analyze ("p1" or "p2")
        """
        try:
            # Create battle iterator
            iterator = BattleIterator(battle_data, perspective=perspective, omniscient=False)

            # Get the final outcome (who actually won)
            actual_win = battle_data.winner == perspective

            # Track all steps in this battle
            battle_evaluations = []
            turn_numbers = []

            # Iterate through all decision points in the battle, similar to BattleDataset
            while not iterator.battle.finished and iterator.next_input():
                try:
                    # Get current battle state
                    battle = iterator.battle
                    if battle is None:
                        continue

                    # Compute heuristic evaluation
                    heuristic_eval = evaluate_position_advantage(battle)  # type: ignore
                    turn_num = battle.turn

                    battle_evaluations.append(heuristic_eval)
                    turn_numbers.append(turn_num)

                except Exception:
                    # Skip problematic steps but continue iterating
                    continue

            # Now compute synthetic win advantages using ensemble logic similar to BattleDataset
            # This combines: current heuristic, near-future heuristic, and final outcome
            num_steps = len(battle_evaluations)
            if num_steps == 0:
                return

            final_outcome = 1.0 if actual_win else -1.0

            for i, (heuristic_eval, turn_num) in enumerate(zip(battle_evaluations, turn_numbers)):
                turns_to_end = num_steps - i - 1

                # Compute synthetic win advantage using ensemble weighting
                # Weight transitions from heuristic (early) to outcome (late)
                # Using quadratic weighting similar to BattleDataset._compute_ensemble_advantage
                if num_steps > 1:
                    # Quadratic weight: more weight on outcome as we approach end
                    weight = min(max((i / (num_steps - 1)) ** 2, 0.05), 0.95)
                else:
                    weight = 0.5

                # Ensemble: weighted combination of heuristic and outcome
                synthetic_win_adv = (1 - weight) * heuristic_eval + weight * final_outcome

                # Store data points for analysis
                self.by_turn[turn_num].append((heuristic_eval, synthetic_win_adv, actual_win))
                self.by_turns_to_end[turns_to_end].append((heuristic_eval, synthetic_win_adv, actual_win))

                outcome_key = "winning" if actual_win else "losing"
                self.by_outcome[outcome_key].append((heuristic_eval, synthetic_win_adv))

                self.all_data.append((heuristic_eval, synthetic_win_adv, actual_win))

        except Exception:
            # Skip battles that can't be processed
            pass

    def compute_correlation(self, data: List[Tuple[float, float, bool]]) -> Dict[str, float]:
        """Compute correlation metrics between heuristic and win advantage."""
        if not data:
            return {"correlation": 0.0, "mse": 0.0, "mae": 0.0, "samples": 0}

        heuristics = np.array([x[0] for x in data])
        win_advs = np.array([x[1] for x in data])

        correlation = float(np.corrcoef(heuristics, win_advs)[0, 1])
        mse = float(np.mean((heuristics - win_advs) ** 2))
        mae = float(np.mean(np.abs(heuristics - win_advs)))

        return {
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "samples": len(data),
        }

    def compute_prediction_accuracy(self, data: List[Tuple[float, float, bool]]) -> Dict[str, float]:
        """
        Compute how well heuristic predicts actual outcome.

        Uses sign of heuristic evaluation as prediction (positive = player wins).
        """
        if not data:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "samples": 0}

        heuristics = np.array([x[0] for x in data])
        actuals = np.array([x[2] for x in data])

        # Predict win if heuristic > 0
        predicted_wins = heuristics > 0
        actual_wins = actuals

        accuracy = float(np.mean(predicted_wins == actual_wins))

        # True positives, false positives, false negatives
        tp = np.sum(predicted_wins & actual_wins)
        fp = np.sum(predicted_wins & ~actual_wins)
        fn = np.sum(~predicted_wins & actual_wins)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "samples": len(data),
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        report: Dict[str, Any] = {
            "overall": {},
            "by_turn": {},
            "by_turns_to_end": {},
            "by_outcome": {},
        }

        # Overall statistics
        if self.all_data:
            report["overall"]["correlation"] = self.compute_correlation(self.all_data)
            report["overall"]["prediction"] = self.compute_prediction_accuracy(self.all_data)

        # By turn number
        for turn in sorted(self.by_turn.keys()):
            data = self.by_turn[turn]
            report["by_turn"][turn] = {
                "correlation": self.compute_correlation(data),
                "prediction": self.compute_prediction_accuracy(data),
            }

        # By turns to end
        for turns_to_end in sorted(self.by_turns_to_end.keys()):
            data = self.by_turns_to_end[turns_to_end]
            report["by_turns_to_end"][turns_to_end] = {
                "correlation": self.compute_correlation(data),
                "prediction": self.compute_prediction_accuracy(data),
            }

        # By outcome
        for outcome in ["winning", "losing"]:
            if outcome in self.by_outcome:
                data = self.by_outcome[outcome]  # type: ignore
                report["by_outcome"][outcome] = self.compute_correlation(data)  # type: ignore

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable summary of the analysis."""
        print("\n" + "=" * 80)
        print("STATE EVALUATION BASELINE ANALYSIS")
        print("=" * 80)

        # Overall statistics
        if "overall" in report and report["overall"]:
            print("\n--- Overall Statistics ---")
            corr = report["overall"]["correlation"]
            pred = report["overall"]["prediction"]
            print(f"Correlation with Win Advantage: {corr['correlation']:.4f}")
            print(f"MAE vs Win Advantage: {corr['mae']:.4f}")
            print(f"MSE vs Win Advantage: {corr['mse']:.4f}")
            print(f"\nOutcome Prediction Accuracy: {pred['accuracy']:.2%}")
            print(f"Precision: {pred['precision']:.2%}")
            print(f"Recall: {pred['recall']:.2%}")
            print(f"Total Samples: {pred['samples']:,}")

        # By turn analysis (first few turns and last few turns)
        if "by_turn" in report and report["by_turn"]:
            print("\n--- Correlation by Turn Number ---")
            print(f"{'Turn':<8} {'Correlation':<14} {'Accuracy':<12} {'Samples':<10}")
            print("-" * 50)

            turns = sorted(report["by_turn"].keys())
            # Show first 5 and last 5 turns
            display_turns = sorted(set(turns[:5] + turns[-5:]))

            for turn in display_turns:
                data = report["by_turn"][turn]
                corr = data["correlation"]["correlation"]
                acc = data["prediction"]["accuracy"]
                samples = data["correlation"]["samples"]
                print(f"{turn:<8} {corr:<14.4f} {acc:<12.2%} {samples:<10,}")

        # By turns to end
        if "by_turns_to_end" in report and report["by_turns_to_end"]:
            print("\n--- Correlation by Turns Until End ---")
            print(f"{'Turns Left':<12} {'Correlation':<14} {'Accuracy':<12} {'Samples':<10}")
            print("-" * 52)

            for turns_to_end in sorted(report["by_turns_to_end"].keys())[:10]:
                data = report["by_turns_to_end"][turns_to_end]
                corr = data["correlation"]["correlation"]
                acc = data["prediction"]["accuracy"]
                samples = data["correlation"]["samples"]
                print(f"{turns_to_end:<12} {corr:<14.4f} {acc:<12.2%} {samples:<10,}")

        # By outcome
        if "by_outcome" in report and report["by_outcome"]:
            print("\n--- Correlation by Actual Outcome ---")
            for outcome, data in report["by_outcome"].items():
                print(f"\n{outcome.capitalize()} Games:")
                print(f"  Correlation: {data['correlation']:.4f}")
                print(f"  MAE: {data['mae']:.4f}")
                print(f"  Samples: {data['samples']:,}")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between state evaluation and win advantage"
    )
    parser.add_argument(
        "filter_json",
        type=str,
        help="Path to JSON file containing list of valid battle file paths",
    )
    parser.add_argument(
        "num_battles",
        type=int,
        help="Number of battles to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--both-perspectives",
        action="store_true",
        help="Analyze from both players' perspectives (doubles the data)",
    )

    args = parser.parse_args()

    print(f"Loading filtered battle list from: {args.filter_json}")
    print(f"Number of battles to analyze: {args.num_battles}")

    # Initialize analyzer
    analyzer = StateEvaluationBaseline()

    # Load filtered battle file list
    with open(args.filter_json, "r") as f:
        battle_files = json.load(f)

    print(f"Loaded {len(battle_files)} battles from filter file")

    # Limit to requested number
    battle_files = battle_files[:args.num_battles]

    print(f"Will process {len(battle_files)} battles")

    # Process battles
    print("\nProcessing battles...")
    successful = 0
    failed = 0

    for battle_file in tqdm(battle_files, desc="Analyzing"):
        try:
            # Load battle data
            with open(battle_file, "r") as f:
                raw_data = orjson.loads(f.read())

            battle_data = BattleData.from_showdown_json(raw_data)

            # Analyze from p1 perspective
            analyzer.analyze_battle(battle_data, perspective="p1")

            # Optionally analyze from p2 perspective as well
            if args.both_perspectives:
                analyzer.analyze_battle(battle_data, perspective="p2")

            successful += 1

        except Exception:
            failed += 1
            continue

    print(f"\nSuccessfully processed: {successful} battles")
    print(f"Failed to process: {failed} battles")
    print(f"Total data points collected: {len(analyzer.all_data)}")

    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_report()

    # Print summary
    analyzer.print_summary(report)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
