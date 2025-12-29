#!/usr/bin/env python3
"""
Win Prediction Model Analysis

Analyzes a trained model's win prediction performance across various dimensions:
1. Prediction accuracy by turn number (early/mid/late game)
2. Prediction accuracy by turns from game end (proximity to outcome)
3. Calibration curves (predicted vs actual win rates)
4. Performance by game outcome (winning vs losing positions)
5. Confidence distribution analysis
6. Prediction trajectory over the course of games
7. Error analysis: when does the model get it most wrong?

Usage:
    python win_prediction_analysis.py <model_path> <data_path> [--output output.json]
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from elitefurretai.etl import Embedder, OptimizedBattleDataLoader
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


class WinPredictionAnalyzer:
    """Comprehensive analyzer for win prediction model performance."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()

        # Storage for analysis
        self.predictions_by_turn: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
        self.predictions_by_turns_to_end: Dict[int, List[Tuple[float, bool]]] = (
            defaultdict(list)
        )
        self.predictions_by_outcome: Dict[str, List[Tuple[float, bool]]] = defaultdict(
            list
        )
        self.all_predictions: List[Tuple[float, bool]] = []

        # Synthetic win advantage analysis (comparing against training target)
        self.synthetic_predictions_by_turn: Dict[int, List[Tuple[float, float]]] = (
            defaultdict(list)
        )
        self.synthetic_predictions_by_turns_to_end: Dict[
            int, List[Tuple[float, float]]
        ] = defaultdict(list)
        self.all_synthetic_predictions: List[Tuple[float, float]] = []

        self.prediction_trajectories: List[List[float]] = []
        self.win_trajectories: List[bool] = []

        # For calibration analysis
        self.calibration_bins = 20
        self.calibration_data: List[Tuple[float, bool]] = []

    def analyze_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_seq_len: int = 17,
    ) -> None:
        """
        Analyze a single batch of predictions.

        Args:
            batch: Dictionary with keys 'states', 'wins', 'masks'
            max_seq_len: Maximum sequence length for trajectories
        """
        with torch.no_grad():
            states = batch["states"].to(torch.float32).to(self.device)
            wins = batch["wins"].to(self.device)
            masks = batch["masks"].to(self.device)

            # Get predictions
            _, _, win_preds = self.model(states, masks)

            # Convert predictions from [-1, 1] to [0, 1] (win probability)
            win_probs = (win_preds + 1.0) / 2.0

            # Process each sequence in the batch
            batch_size, seq_len = states.shape[:2]

            for b in range(batch_size):
                valid_steps = masks[b].sum().item()
                if valid_steps == 0:
                    continue

                # Extract valid predictions and outcomes for this sequence
                sequence_preds = win_probs[b, : int(valid_steps)].cpu().numpy()
                sequence_wins = wins[b, : int(valid_steps)].cpu().numpy()

                # Get actual outcome (last valid step)
                actual_win = bool(sequence_wins[-1] > 0.5)

                # Store trajectory
                self.prediction_trajectories.append(sequence_preds.tolist())
                self.win_trajectories.append(actual_win)

                # Analyze each step in the sequence
                num_steps = int(valid_steps)
                for step in range(num_steps):
                    pred = float(sequence_preds[step])
                    outcome = actual_win

                    # Synthetic win advantage (training target)
                    # Convert from [-1, 1] to [0, 1]
                    synthetic_adv = float(sequence_wins[step])
                    synthetic_prob = (synthetic_adv + 1.0) / 2.0

                    # Store for overall analysis
                    self.all_predictions.append((pred, outcome))
                    self.all_synthetic_predictions.append((pred, synthetic_prob))
                    self.calibration_data.append((pred, outcome))

                    # By turn number (0 = teampreview, 1-16 = turns)
                    turn_number = step
                    self.predictions_by_turn[turn_number].append((pred, outcome))
                    self.synthetic_predictions_by_turn[turn_number].append(
                        (pred, synthetic_prob)
                    )

                    # By turns to end (0 = final turn, 1 = one turn before end, etc.)
                    turns_to_end = num_steps - step - 1
                    self.predictions_by_turns_to_end[turns_to_end].append((pred, outcome))
                    self.synthetic_predictions_by_turns_to_end[turns_to_end].append(
                        (pred, synthetic_prob)
                    )

                    # By game outcome
                    outcome_key = "winning_positions" if outcome else "losing_positions"
                    self.predictions_by_outcome[outcome_key].append((pred, outcome))

    def compute_metrics(self, predictions: List[Tuple[float, bool]]) -> Dict[str, float]:
        """
        Compute various metrics for a set of predictions.

        Args:
            predictions: List of (predicted_win_prob, actual_outcome) tuples

        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}

        preds = np.array([p[0] for p in predictions])
        actuals = np.array([float(p[1]) for p in predictions])

        # Correlation
        correlation = float(np.corrcoef(preds, actuals)[0, 1])

        # MSE
        mse = float(np.mean((preds - actuals) ** 2))

        # MAE
        mae = float(np.mean(np.abs(preds - actuals)))

        # Classification accuracy (threshold at 0.5)
        binary_preds = (preds > 0.5).astype(float)
        accuracy = float(np.mean(binary_preds == actuals))

        # Brier score (calibration metric)
        brier_score = float(np.mean((preds - actuals) ** 2))

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(preds, actuals)

        # Confidence statistics
        mean_confidence = float(np.mean(np.abs(preds - 0.5) * 2))
        std_confidence = float(np.std(np.abs(preds - 0.5) * 2))

        return {
            "count": len(predictions),
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "accuracy": accuracy,
            "brier_score": brier_score,
            "ece": ece,
            "mean_confidence": mean_confidence,
            "std_confidence": std_confidence,
            "mean_prediction": float(np.mean(preds)),
            "std_prediction": float(np.std(preds)),
        }

    def compute_synthetic_metrics(
        self, predictions: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Compute metrics for synthetic predictions (comparing against training targets).

        Args:
            predictions: List of (predicted_win_prob, synthetic_win_advantage) tuples

        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {}

        preds = np.array([p[0] for p in predictions])
        targets = np.array([p[1] for p in predictions])

        # Correlation
        correlation = float(np.corrcoef(preds, targets)[0, 1])

        # MSE
        mse = float(np.mean((preds - targets) ** 2))

        # MAE
        mae = float(np.mean(np.abs(preds - targets)))

        return {
            "count": len(predictions),
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "mean_prediction": float(np.mean(preds)),
            "std_prediction": float(np.std(preds)),
            "mean_target": float(np.mean(targets)),
            "std_target": float(np.std(targets)),
        }

    def _compute_ece(
        self, predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE measures how well predicted probabilities match actual frequencies.
        Lower is better (0 = perfect calibration).
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find predictions in this bin
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            if i == n_bins - 1:  # Include upper boundary for last bin
                in_bin = (predictions >= bin_lower) & (predictions <= bin_upper)

            if np.sum(in_bin) > 0:
                avg_confidence = np.mean(predictions[in_bin])
                avg_accuracy = np.mean(actuals[in_bin])
                ece += np.sum(in_bin) * np.abs(avg_confidence - avg_accuracy)

        ece /= len(predictions)
        return float(ece)

    def get_calibration_curve(
        self, n_bins: int = 20
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Generate calibration curve data.

        Returns:
            (bin_centers, actual_frequencies, counts)
        """
        if not self.calibration_data:
            return [], [], []

        preds = np.array([p[0] for p in self.calibration_data])
        actuals = np.array([float(p[1]) for p in self.calibration_data])

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        actual_frequencies = []
        counts = []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (preds >= bin_lower) & (preds < bin_upper)
            if i == n_bins - 1:
                in_bin = (preds >= bin_lower) & (preds <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_centers.append(float((bin_lower + bin_upper) / 2))
                actual_frequencies.append(float(np.mean(actuals[in_bin])))
                counts.append(int(np.sum(in_bin)))

        return bin_centers, actual_frequencies, counts

    def analyze_error_cases(
        self, threshold: float = 0.3
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Find cases where model predictions were most wrong.

        Args:
            threshold: Error threshold for "large error" classification

        Returns:
            Dictionary with 'overconfident_wrong' and 'underconfident_right' cases
            Each case is a tuple of (trajectory_index, step_index, prediction)
        """
        overconfident_wrong = []  # Predicted win strongly but lost
        underconfident_right = []  # Predicted loss but won

        for traj_idx, (pred_traj, outcome) in enumerate(
            zip(self.prediction_trajectories, self.win_trajectories)
        ):
            for step_idx, pred in enumerate(pred_traj):
                error = abs(pred - float(outcome))

                if error > threshold:
                    if outcome and pred < 0.5:
                        # Actually won but predicted loss
                        underconfident_right.append((traj_idx, step_idx, pred))
                    elif not outcome and pred > 0.5:
                        # Actually lost but predicted win
                        overconfident_wrong.append((traj_idx, step_idx, pred))

        return {
            "overconfident_wrong": overconfident_wrong,
            "underconfident_right": underconfident_right,
        }

    def analyze_prediction_stability(self) -> Dict[str, float]:
        """
        Analyze how much predictions change over the course of games.

        Returns:
            Dictionary with stability metrics
        """
        if not self.prediction_trajectories:
            return {}

        # Compute variance within trajectories
        trajectory_variances = []
        trajectory_ranges = []
        trajectory_trend_strengths = []

        for traj in self.prediction_trajectories:
            if len(traj) < 2:
                continue

            traj_array = np.array(traj)

            # Variance
            trajectory_variances.append(float(np.var(traj_array)))

            # Range
            trajectory_ranges.append(float(np.max(traj_array) - np.min(traj_array)))

            # Trend strength (correlation with step number)
            if len(traj) > 2:
                steps = np.arange(len(traj))
                correlation = float(np.corrcoef(steps, traj_array)[0, 1])
                trajectory_trend_strengths.append(abs(correlation))

        return {
            "mean_trajectory_variance": float(np.mean(trajectory_variances)),
            "mean_trajectory_range": float(np.mean(trajectory_ranges)),
            "mean_trend_strength": float(np.mean(trajectory_trend_strengths)),
            "trajectories_analyzed": len(trajectory_variances),
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.

        Returns:
            Dictionary containing all analysis results
        """
        report: Dict[str, Any] = {}

        # Overall metrics
        report["overall"] = self.compute_metrics(self.all_predictions)

        # Synthetic metrics (training target)
        report["synthetic_overall"] = self.compute_synthetic_metrics(
            self.all_synthetic_predictions
        )

        # By turn number
        report["by_turn_number"] = {}
        report["synthetic_by_turn_number"] = {}
        for turn in sorted(self.predictions_by_turn.keys()):
            report["by_turn_number"][turn] = self.compute_metrics(
                self.predictions_by_turn[turn]
            )
            if turn in self.synthetic_predictions_by_turn:
                report["synthetic_by_turn_number"][turn] = self.compute_synthetic_metrics(
                    self.synthetic_predictions_by_turn[turn]
                )

        # By turns to end
        report["by_turns_to_end"] = {}
        report["synthetic_by_turns_to_end"] = {}
        for turns_to_end in sorted(self.predictions_by_turns_to_end.keys()):
            report["by_turns_to_end"][turns_to_end] = self.compute_metrics(
                self.predictions_by_turns_to_end[turns_to_end]
            )
            if turns_to_end in self.synthetic_predictions_by_turns_to_end:
                report["synthetic_by_turns_to_end"][turns_to_end] = (
                    self.compute_synthetic_metrics(
                        self.synthetic_predictions_by_turns_to_end[turns_to_end]
                    )
                )

        # By outcome
        report["by_outcome"] = {}
        for outcome_type, preds in self.predictions_by_outcome.items():
            report["by_outcome"][outcome_type] = self.compute_metrics(preds)

        # Calibration curve
        bin_centers, actual_freqs, counts = self.get_calibration_curve()
        report["calibration"] = {
            "bin_centers": bin_centers,
            "actual_frequencies": actual_freqs,
            "counts": counts,
        }

        # Prediction stability
        report["stability"] = self.analyze_prediction_stability()

        # Error analysis
        report["error_cases"] = self.analyze_error_cases()

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable summary of analysis."""
        print("\n" + "=" * 80)
        print("WIN PREDICTION MODEL ANALYSIS")
        print("=" * 80)

        # Overall performance
        print("\n### OVERALL PERFORMANCE ###")
        overall = report["overall"]
        print(f"  Total Predictions: {overall['count']:,}")
        print(f"  Correlation:       {overall['correlation']:.4f}")
        print(f"  Accuracy:          {overall['accuracy']:.4f}")
        print(f"  MSE:               {overall['mse']:.4f}")
        print(f"  MAE:               {overall['mae']:.4f}")
        print(f"  Brier Score:       {overall['brier_score']:.4f}")
        print(f"  ECE:               {overall['ece']:.4f}")
        print(
            f"  Mean Confidence:   {overall['mean_confidence']:.4f} ± {overall['std_confidence']:.4f}"
        )

        # Synthetic performance (Training Target)
        if "synthetic_overall" in report:
            print("\n### SYNTHETIC WIN ADVANTAGE (TRAINING TARGET) ###")
            syn = report["synthetic_overall"]
            print(f"  Correlation:       {syn['correlation']:.4f}")
            print(f"  MSE:               {syn['mse']:.4f}")
            print(f"  MAE:               {syn['mae']:.4f}")

        # By turn number
        print("\n### PERFORMANCE BY TURN NUMBER ###")
        print(f"{'Turn':<6} {'Count':<10} {'Corr':<8} {'Acc':<8} {'MSE':<8} {'ECE':<8}")
        print("-" * 60)
        for turn in sorted(report["by_turn_number"].keys())[:17]:
            metrics = report["by_turn_number"][turn]
            turn_name = "T-Pre" if turn == 0 else f"T{turn}"
            print(
                f"{turn_name:<6} {metrics['count']:<10,} "
                f"{metrics['correlation']:<8.4f} {metrics['accuracy']:<8.4f} "
                f"{metrics['mse']:<8.4f} {metrics['ece']:<8.4f}"
            )

        # By turns to end
        print("\n### PERFORMANCE BY TURNS FROM GAME END ###")
        print(f"{'T-End':<6} {'Count':<10} {'Corr':<8} {'Acc':<8} {'MSE':<8} {'ECE':<8}")
        print("-" * 60)
        for turns_to_end in sorted(report["by_turns_to_end"].keys())[:10]:
            metrics = report["by_turns_to_end"][turns_to_end]
            print(
                f"{turns_to_end:<6} {metrics['count']:<10,} "
                f"{metrics['correlation']:<8.4f} {metrics['accuracy']:<8.4f} "
                f"{metrics['mse']:<8.4f} {metrics['ece']:<8.4f}"
            )

        # By outcome
        print("\n### PERFORMANCE BY GAME OUTCOME ###")
        for outcome_type, metrics in report["by_outcome"].items():
            print(f"\n{outcome_type.replace('_', ' ').title()}:")
            print(f"  Count:       {metrics['count']:,}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Mean Pred:   {metrics['mean_prediction']:.4f}")

        # Stability
        print("\n### PREDICTION STABILITY ###")
        stability = report["stability"]
        print(f"  Mean Trajectory Variance: {stability['mean_trajectory_variance']:.4f}")
        print(f"  Mean Trajectory Range:    {stability['mean_trajectory_range']:.4f}")
        print(f"  Mean Trend Strength:      {stability['mean_trend_strength']:.4f}")

        # Calibration summary
        print("\n### CALIBRATION ###")
        calibration = report["calibration"]
        if calibration["bin_centers"]:
            print("  Predicted → Actual Win Rate (sample)")
            for i in range(0, len(calibration["bin_centers"]), 2):
                pred = calibration["bin_centers"][i]
                actual = calibration["actual_frequencies"][i]
                count = calibration["counts"][i]
                error = abs(pred - actual)
                print(f"    {pred:.2f} → {actual:.2f} (n={count:,}, error={error:.3f})")

        # Error cases
        print("\n### ERROR ANALYSIS ###")
        error_cases = report["error_cases"]
        print(f"  Overconfident Wrong Cases: {len(error_cases['overconfident_wrong'])}")
        print(f"  Underconfident Right Cases: {len(error_cases['underconfident_right'])}")

        print("\n" + "=" * 80)


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint missing 'config' key")

    # Create embedder to get input size and group sizes
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set="full", omniscient=False
    )
    input_size = embedder.embedding_size
    group_sizes = (
        embedder.group_embedding_sizes
        if config.get("use_grouped_encoder", False)
        else None
    )

    # Reconstruct model
    model = FlexibleThreeHeadedModel(
        input_size=input_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config["lstm_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
        dropout=config["dropout"],
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 8),
        late_attention_heads=config.get("late_attention_heads", 8),
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=group_sizes,
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        turn_head_layers=config.get("turn_head_layers", []),
        max_seq_len=17,
    )

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Training steps: {checkpoint.get('steps', 'unknown')}")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Analyze win prediction model performance"
    )
    parser.add_argument("model_path", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("data_path", help="Path to validation/test data directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for detailed results",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument("--num-workers", type=int, default=7, help="DataLoader workers")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit number of batches to process (for quick testing)",
    )

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    if args.output is None:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_dir = "data/models/supervised"
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{model_name}_diagnostics.json")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model_from_checkpoint(args.model_path, device=device)

    # Setup embedder
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set="full", omniscient=False
    )

    # Create dataloader
    print(f"Loading data from {args.data_path}...")
    dataloader = OptimizedBattleDataLoader(
        args.data_path,
        embedder=embedder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=False,  # WSL2 compatibility
    )

    # Create analyzer
    analyzer = WinPredictionAnalyzer(model, device=device)

    # Process batches
    print("Analyzing predictions...")
    num_batches_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        analyzer.analyze_batch(batch)
        num_batches_processed += 1

        if args.max_batches and num_batches_processed >= args.max_batches:
            print(f"Stopping at {args.max_batches} batches (as requested)")
            break

    # Generate report
    print("Generating report...")
    report = analyzer.generate_report()

    # Print summary
    analyzer.print_summary(report)

    # Save detailed report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to {args.output}")


if __name__ == "__main__":
    main()
