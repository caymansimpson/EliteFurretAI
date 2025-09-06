#!/usr/bin/env python3
"""
Analyze the correlation between evaluate_state and actual battle outcomes.

This script loads battles using BattleIteratorDataset and analyzes how well
the poke-engine evaluation function predicts battle outcomes across various factors.


"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elitefurretai.model_utils.battle_data import BattleData

# Project imports
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.utils.evaluate_state import evaluate_position_advantage


def load_battle_files(data_path: str, num_battles: int) -> List[str]:
    """Load battle file paths"""
    battle_files = glob.glob(os.path.join(data_path, "*.json"))
    if len(battle_files) == 0:
        raise ValueError(f"No battle files found in {data_path}")

    # Limit to requested number of battles
    return battle_files[:num_battles]


def analyze_single_battle(file_path: str, perspective: str) -> Dict:
    """
    Analyze a single battle from one perspective.
    Returns dictionary with analysis results.
    """
    with open(file_path, "r") as f:
        bd = BattleData.from_showdown_json(json.loads(f.read()))

    # Create battle iterator
    iterator = BattleIterator(bd, perspective=perspective, omniscient=True)

    # Track data for each step
    step_data = []
    step = 0

    while (
        not iterator.battle.finished and iterator.next_input() and step < 50
    ):  # Cap at 50 steps
        if iterator.last_input is None:
            continue

        try:
            # Calculate evaluation advantage
            advantage = evaluate_position_advantage(iterator.battle)  # type: ignore

            # Get battle state info
            my_alive = sum(1 for p in iterator.battle.team.values() if not p.fainted)
            opp_alive = sum(
                1 for p in iterator.battle.opponent_team.values() if not p.fainted
            )
            my_hp = sum(
                p.current_hp_fraction
                for p in iterator.battle.team.values()
                if not p.fainted
            )
            opp_hp = sum(
                p.current_hp_fraction
                for p in iterator.battle.opponent_team.values()
                if not p.fainted
            )

            # Speed information (simplified)
            my_speeds = []
            opp_speeds = []
            for p in iterator.battle.active_pokemon:
                if p and not p.fainted and p.stats:
                    my_speeds.append(p.stats.get("spe", 0))
            for p in iterator.battle.opponent_active_pokemon:
                if p and not p.fainted and p.stats:
                    opp_speeds.append(p.stats.get("spe", 0))

            avg_my_speed = np.mean(my_speeds) if my_speeds else 0
            avg_opp_speed = np.mean(opp_speeds) if opp_speeds else 0

            step_data.append(
                {
                    "step": step,
                    "elo": (iterator.bd.p1_rating + iterator.bd.p2_rating) / 2,
                    "turn": iterator.battle.turn,
                    "advantage": advantage,
                    "my_alive": my_alive,
                    "opp_alive": opp_alive,
                    "my_hp": my_hp,
                    "opp_hp": opp_hp,
                    "avg_my_speed": avg_my_speed,
                    "avg_opp_speed": avg_opp_speed,
                    "material_advantage": (my_alive - opp_alive) + 0.5 * (my_hp - opp_hp),
                }
            )

        except Exception as e:
            print(f"Error analyzing step {step} in {file_path}: {e}")
            continue

        step += 1

    # Final outcome
    final_outcome = int(bd.winner == iterator.battle.player_username)
    total_steps = len(step_data)

    return {
        "file_path": file_path,
        "perspective": perspective,
        "final_outcome": final_outcome,
        "total_steps": total_steps,
        "step_data": step_data,
        "winner": bd.winner,
        "player_name": iterator.battle.player_username,
    }


def calculate_correlations(battle_results: List[Dict]) -> Dict:
    """Calculate various correlation metrics"""
    all_steps = []

    for battle in battle_results:
        final_outcome = battle["final_outcome"]
        total_steps = battle["total_steps"]

        for step_info in battle["step_data"]:
            step_info["final_outcome"] = final_outcome
            step_info["total_steps"] = total_steps
            step_info["turns_left"] = max(0, total_steps - step_info["step"])
            step_info["progress"] = step_info["step"] / max(1, total_steps - 1)
            all_steps.append(step_info)

    if not all_steps:
        return {}

    df = pd.DataFrame(all_steps)

    correlations = {}

    # Overall correlation
    if len(df) > 1:
        correlations["overall"] = df["advantage"].corr(df["final_outcome"])

    # Correlation by game phase
    early_game = df[df["progress"] <= 0.33]
    mid_game = df[(df["progress"] > 0.33) & (df["progress"] <= 0.66)]
    late_game = df[df["progress"] > 0.66]

    for phase, phase_df in [("early", early_game), ("mid", mid_game), ("late", late_game)]:
        if len(phase_df) > 1:
            correlations[f"{phase}_game"] = phase_df["advantage"].corr(
                phase_df["final_outcome"]
            )

    # Correlation by turn number
    for turn_range in [(1, 5), (6, 10), (11, 20), (21, 50)]:
        turn_df = df[(df["turn"] >= turn_range[0]) & (df["turn"] <= turn_range[1])]
        if len(turn_df) > 1:
            correlations[f"turns_{turn_range[0]}-{turn_range[1]}"] = turn_df[
                "advantage"
            ].corr(turn_df["final_outcome"])

    # Correlation by material situation
    material_advantage = df[df["material_advantage"] > 0.5]
    material_disadvantage = df[df["material_advantage"] < -0.5]
    material_even = df[abs(df["material_advantage"]) <= 0.5]

    for situation, situation_df in [
        ("advantage", material_advantage),
        ("disadvantage", material_disadvantage),
        ("even", material_even),
    ]:
        if len(situation_df) > 1:
            correlations[f"material_{situation}"] = situation_df["advantage"].corr(
                situation_df["final_outcome"]
            )

    return correlations


def analyze_advantage_switches(battle_results: List[Dict]) -> Dict:
    """Analyze how often the advantage switches between players"""
    switch_analysis: Dict[str, Any] = {
        "total_battles": len(battle_results),
        "battles_with_switches": 0,
        "avg_switches_per_battle": 0.0,
        "switch_details": [],
    }

    total_switches = 0

    for battle in battle_results:
        step_data = battle["step_data"]
        if len(step_data) < 2:
            continue

        switches = 0
        last_advantage_sign = None

        for step_info in step_data:
            current_sign = 1 if step_info["advantage"] > 0 else -1
            if last_advantage_sign is not None and current_sign != last_advantage_sign:
                switches += 1
            last_advantage_sign = current_sign

        if switches > 0:
            switch_analysis["battles_with_switches"] += 1

        total_switches += switches

        switch_analysis["switch_details"].append(
            {
                "file": battle["file_path"],
                "switches": switches,
                "total_steps": battle["total_steps"],
                "final_outcome": battle["final_outcome"],
            }
        )

    if len(battle_results) > 0:
        switch_analysis["avg_switches_per_battle"] = total_switches / len(battle_results)

    return switch_analysis


def find_poorly_predicted_battles(
    battle_results: List[Dict], threshold: float = 0.3
) -> List[Dict]:
    """Find battles where the evaluation performs poorly"""
    poor_predictions = []

    for battle in battle_results:
        step_data = battle["step_data"]
        if len(step_data) < 3:
            continue

        final_outcome = battle["final_outcome"]
        advantages = [step["advantage"] for step in step_data]

        # Check if evaluation consistently disagrees with final outcome
        if final_outcome == 1:  # Player won
            # Evaluation should be mostly positive
            negative_evals = sum(1 for adv in advantages if adv < -threshold)
            poor_ratio = negative_evals / len(advantages)
        else:  # Player lost
            # Evaluation should be mostly negative
            positive_evals = sum(1 for adv in advantages if adv > threshold)
            poor_ratio = positive_evals / len(advantages)

        if poor_ratio > 0.6:  # More than 60% of evaluations disagree with outcome
            poor_predictions.append(
                {
                    "file": battle["file_path"],
                    "perspective": battle["perspective"],
                    "final_outcome": final_outcome,
                    "poor_ratio": poor_ratio,
                    "avg_advantage": np.mean(advantages),
                    "total_steps": len(step_data),
                    "winner": battle.get("winner", "unknown"),
                    "player_name": battle.get("player_name", "unknown"),
                }
            )

    # Sort by poor_ratio descending
    poor_predictions.sort(key=lambda x: x["poor_ratio"], reverse=True)
    return poor_predictions


def create_visualizations(battle_results: List[Dict], output_dir: str):
    """Create visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all step data
    all_steps = []
    for battle in battle_results:
        final_outcome = battle["final_outcome"]
        for step_info in battle["step_data"]:
            step_info["final_outcome"] = final_outcome
            step_info["progress"] = step_info["step"] / max(1, battle["total_steps"] - 1)
            all_steps.append(step_info)

    if not all_steps:
        print("No step data available for visualization")
        return

    df = pd.DataFrame(all_steps)

    # Plot 1: Advantage vs Final Outcome
    plt.figure(figsize=(10, 6))
    plt.scatter(df["advantage"], df["final_outcome"], alpha=0.5)
    plt.xlabel("Evaluation Advantage")
    plt.ylabel("Final Outcome (0=Loss, 1=Win)")
    plt.title("Evaluation Advantage vs Final Outcome")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "advantage_vs_outcome.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Advantage over game progress
    plt.figure(figsize=(12, 6))
    for outcome in [0, 1]:
        subset = df[df["final_outcome"] == outcome]
        plt.scatter(
            subset["progress"],
            subset["advantage"],
            alpha=0.5,
            label=f'Final Outcome: {"Win" if outcome else "Loss"}',
        )
    plt.xlabel("Game Progress (0=Start, 1=End)")
    plt.ylabel("Evaluation Advantage")
    plt.title("Evaluation Advantage Over Game Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "advantage_vs_progress.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Correlation by turn
    turn_correlations = []
    for turn in range(1, min(21, int(df["turn"].max()) + 1)):
        turn_data = df[df["turn"] == turn]
        if len(turn_data) > 5:
            corr = turn_data["advantage"].corr(turn_data["final_outcome"])
            if not np.isnan(corr):
                turn_correlations.append({"turn": turn, "correlation": corr})

    if turn_correlations:
        corr_df = pd.DataFrame(turn_correlations)
        plt.figure(figsize=(10, 6))
        plt.plot(corr_df["turn"], corr_df["correlation"], marker="o")
        plt.xlabel("Turn Number")
        plt.ylabel("Correlation with Final Outcome")
        plt.title("Evaluation Correlation by Turn Number")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(output_dir, "correlation_by_turn.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def plot_elo_vs_correlation(battle_results: List[Dict], output_dir: str):
    """Plot and print the relationship between player ELO and correlation with final outcome."""
    # Collect all steps with elo
    all_steps = []
    for battle in battle_results:
        final_outcome = battle["final_outcome"]
        for step_info in battle["step_data"]:
            step_info["final_outcome"] = final_outcome
            all_steps.append(step_info)

    if not all_steps or "elo" not in all_steps[0]:
        print("No ELO data available for correlation plot.")
        return

    df = pd.DataFrame(all_steps)
    # Bin by ELO (50 point bins)
    df["elo_bin"] = (df["elo"] // 50) * 50
    elo_corrs = []
    for elo_bin, group in df.groupby("elo_bin"):
        if len(group) > 5:
            corr = group["advantage"].corr(group["final_outcome"])
            if not np.isnan(corr):
                elo_corrs.append(
                    {"elo_bin": elo_bin, "correlation": corr, "count": len(group)}
                )

    if not elo_corrs:
        print("No ELO bins with enough data for correlation plot.")
        return

    corr_df = pd.DataFrame(elo_corrs)
    print("\n=== ELO vs Correlation ===")
    for _, row in corr_df.iterrows():
        print(
            f"ELO {int(row.loc('elo_bin')):4d} - {int(row.loc('elo_bin') + 99):4d}: Correlation={row['correlation']:.3f} (n={row['count']})"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(corr_df["elo_bin"], corr_df["correlation"], marker="o")
    plt.xlabel("Player ELO (bin)")
    plt.ylabel("Correlation with Final Outcome")
    plt.title("ELO vs Evaluation/Outcome Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "elo_vs_correlation.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Collect all step data
    all_steps = []
    for battle in battle_results:
        final_outcome = battle["final_outcome"]
        for step_info in battle["step_data"]:
            step_info["final_outcome"] = final_outcome
            step_info["progress"] = step_info["step"] / max(1, battle["total_steps"] - 1)
            all_steps.append(step_info)

    if not all_steps:
        print("No step data available for visualization")
        return

    df = pd.DataFrame(all_steps)

    # Plot 1: Advantage vs Final Outcome
    plt.figure(figsize=(10, 6))
    plt.scatter(df["advantage"], df["final_outcome"], alpha=0.5)
    plt.xlabel("Evaluation Advantage")
    plt.ylabel("Final Outcome (0=Loss, 1=Win)")
    plt.title("Evaluation Advantage vs Final Outcome")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "advantage_vs_outcome.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Advantage over game progress
    plt.figure(figsize=(12, 6))
    for outcome in [0, 1]:
        subset = df[df["final_outcome"] == outcome]
        plt.scatter(
            subset["progress"],
            subset["advantage"],
            alpha=0.5,
            label=f'Final Outcome: {"Win" if outcome else "Loss"}',
        )
    plt.xlabel("Game Progress (0=Start, 1=End)")
    plt.ylabel("Evaluation Advantage")
    plt.title("Evaluation Advantage Over Game Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, "advantage_vs_progress.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Correlation by turn
    turn_correlations = []
    for turn in range(1, min(21, int(df["turn"].max()) + 1)):
        turn_data = df[df["turn"] == turn]
        if len(turn_data) > 5:
            corr = turn_data["advantage"].corr(turn_data["final_outcome"])
            if not np.isnan(corr):
                turn_correlations.append({"turn": turn, "correlation": corr})

    if turn_correlations:
        corr_df = pd.DataFrame(turn_correlations)
        plt.figure(figsize=(10, 6))
        plt.plot(corr_df["turn"], corr_df["correlation"], marker="o")
        plt.xlabel("Turn Number")
        plt.ylabel("Correlation with Final Outcome")
        plt.title("Evaluation Correlation by Turn Number")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(output_dir, "correlation_by_turn.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


# Example usage: python3 src/elitefurretai/scripts/analyze/state_evaluation.py path/to/battle/file --num-battles 1000 --output-dir ~/Desktop --perspectives both
def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluate_state correlation with battle outcomes"
    )
    parser.add_argument(
        "battle_files_json",
        type=str,
        help="Path to JSON file containing list of battle files",
    )
    parser.add_argument(
        "--num-battles",
        type=int,
        default=None,
        help="Number of battles to analyze (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/analyze/output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--perspectives",
        default="both",
        choices=["p1", "p2", "both"],
        help="Which perspective(s) to analyze",
    )

    args = parser.parse_args()

    print(f"Loading battle files from {args.battle_files_json}")

    # Load battle file paths from JSON
    try:
        with open(args.battle_files_json, "r") as f:
            battle_files = json.load(f)
        if not isinstance(battle_files, list):
            raise ValueError("battle_files.json must be a list of file paths")
        if args.num_battles is not None:
            battle_files = battle_files[: args.num_battles]
        print(f"Found {len(battle_files)} battle files")
    except Exception as e:
        print(f"Error loading battle files: {e}")
        return 1

    # Analyze battles
    battle_results = []
    perspectives = ["p1", "p2"] if args.perspectives == "both" else [args.perspectives]

    total_analyses = len(battle_files) * len(perspectives)
    completed = 0

    for file_path in battle_files:
        for perspective in perspectives:
            try:
                result = analyze_single_battle(file_path, perspective)
                battle_results.append(result)
                completed += 1

                if completed % 10 == 0:
                    print(
                        f"\033[2K\rCompleted {completed}/{total_analyses} analyses...",
                        end="",
                    )

            except Exception as e:
                print(f"Error analyzing {file_path} from {perspective}: {e}")
                continue

    print(f"Successfully analyzed {len(battle_results)} battle perspectives")

    if not battle_results:
        print("No battles were successfully analyzed")
        return 1

    # Calculate correlations
    print("\n=== CORRELATION ANALYSIS ===")
    correlations = calculate_correlations(battle_results)
    for metric, correlation in correlations.items():
        print(f"{metric:20s}: {correlation:.4f}")

    # Analyze advantage switches
    print("\n=== ADVANTAGE SWITCH ANALYSIS ===")
    switch_analysis = analyze_advantage_switches(battle_results)
    print(f"Total battles analyzed: {switch_analysis['total_battles']}")
    print(f"Battles with advantage switches: {switch_analysis['battles_with_switches']}")
    print(f"Average switches per battle: {switch_analysis['avg_switches_per_battle']:.2f}")

    # Find poorly predicted battles
    print("\n=== POORLY PREDICTED BATTLES ===")
    poor_battles = find_poorly_predicted_battles(battle_results)
    print(f"Found {len(poor_battles)} poorly predicted battles:")

    for i, battle in enumerate(poor_battles[:10]):  # Show top 10
        print(
            f"{i + 1:2d}. {os.path.basename(battle['file'])} "
            f"| Outcome: {battle['final_outcome']} "
            f"| Poor ratio: {battle['poor_ratio']:.3f} "
            f"| Avg advantage: {battle['avg_advantage']:+.3f}"
        )

    # Threshold analysis
    print("\n=== THRESHOLD ANALYSIS ===")
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    all_steps = []
    for battle in battle_results:
        for step_info in battle["step_data"]:
            step_info["final_outcome"] = battle["final_outcome"]
            all_steps.append(step_info)

    if all_steps:
        print("Threshold | Accuracy | Precision | Recall")
        print("-" * 45)

        for threshold in thresholds:
            predictions = [1 if step["advantage"] > threshold else 0 for step in all_steps]
            actuals = [step["final_outcome"] for step in all_steps]

            if len(set(predictions)) > 1 and len(set(actuals)) > 1:
                tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
                fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
                tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
                fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)

                accuracy = (
                    (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
                )
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                print(
                    f"{threshold:8.1f} | {accuracy:8.3f} | {precision:9.3f} | {recall:6.3f}"
                )

    # Create visualizations
    print(f"\n=== Creating visualizations in {args.output_dir} ===")
    create_visualizations(battle_results, args.output_dir)
    plot_elo_vs_correlation(battle_results, args.output_dir)

    # Save detailed results
    results_file = os.path.join(args.output_dir, "analysis_results.json")
    os.makedirs(args.output_dir, exist_ok=True)

    summary_results = {
        "correlations": correlations,
        "switch_analysis": switch_analysis,
        "poor_battles": poor_battles,
        "total_battles_analyzed": len(battle_results),
        "total_steps_analyzed": sum(len(b["step_data"]) for b in battle_results),
    }

    with open(results_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"Detailed results saved to {results_file}")
    print("Analysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
