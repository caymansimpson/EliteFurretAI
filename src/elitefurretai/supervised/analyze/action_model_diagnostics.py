#!/usr/bin/env python3
"""
Comprehensive diagnostics for action prediction models.

Analyzes:
1. Action type distribution (predicted vs actual)
2. Top-k accuracy breakdown by action type
3. Loss contribution analysis
4. Action confidence scores
5. Move selection patterns (which moves are predicted vs actual)
6. Invalid action predictions
"""

import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from elitefurretai.etl import MDBO, Embedder, OptimizedBattleDataLoader
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


class ActionDiagnostics:
    """Diagnostic analyzer for action prediction models."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

        # Storage for analysis
        self.action_type_pred = defaultdict(int)
        self.action_type_actual = defaultdict(int)
        self.action_confidences = defaultdict(list)
        self.topk_hits: defaultdict = defaultdict(lambda: defaultdict(int))
        self.loss_contributions = defaultdict(list)
        self.move_predictions: Counter = Counter()
        self.move_actuals: Counter = Counter()
        self.invalid_predictions = 0
        self.total_predictions = 0

    def classify_action_type(self, action_idx: int, order_type: str) -> str:
        """
        Classify action type from action index by decoding it to MDBO.

        Args:
            action_idx: MDBO action index
            order_type: One of MDBO.TURN, MDBO.FORCE_SWITCH, MDBO.TEAMPREVIEW

        Returns:
            One of: MOVE, SWITCH, BOTH, OTHER, INVALID
        """
        try:
            # Use MDBO.from_int() static method to decode
            mdbo_order = MDBO.from_int(int(action_idx), order_type)
            if mdbo_order is None:
                return "INVALID"

            # Get the message string (e.g., "/choose move 1, switch 2")
            message = mdbo_order.message.lower()

            # Parse the message to determine action type
            # Remove the "/choose " prefix
            if message.startswith("/choose "):
                orders = message[8:].split(", ")
            else:
                return "OTHER"

            # Count moves and switches
            has_move = any("move" in order for order in orders)
            has_switch = any("switch" in order for order in orders)

            if has_move and has_switch:
                return "BOTH"
            elif has_switch:
                return "SWITCH"
            elif has_move:
                return "MOVE"
            else:
                return "OTHER"
        except Exception:
            return "INVALID"

    def analyze_batch(self, batch: Dict[str, torch.Tensor], feature_idx: Dict[str, int]):
        """
        Analyze a single batch of predictions.

        Args:
            batch: Dictionary with keys 'states', 'actions', 'action_masks', 'wins', 'masks'
            feature_idx: Dictionary mapping feature names to indices
        """
        self.model.eval()

        # Extract from batch dictionary
        states = batch["states"].to(torch.float32).to(self.device)
        actions = batch["actions"].to(self.device)
        masks = batch["action_masks"].to(self.device)
        padding_mask = batch["masks"].to(self.device)

        with torch.no_grad():
            # Get predictions
            turn_logits, teampreview_logits, win_logits = self.model(states, padding_mask)

            # Apply action masking
            turn_logits_masked = turn_logits.clone()
            turn_logits_masked[~masks.bool()] = float("-inf")

            # Get top-k predictions
            probs = torch.softmax(turn_logits_masked, dim=-1)
            topk_values, topk_indices = torch.topk(probs, k=10, dim=-1)

            # Analyze each timestep
            batch_size, seq_len = actions.shape

            for b in range(batch_size):
                for t in range(seq_len):
                    # Skip padding
                    if not padding_mask[b, t]:
                        continue

                    actual_action = int(actions[b, t].item())
                    predicted_action = int(topk_indices[b, t, 0].item())
                    confidence = topk_values[b, t, 0].item()

                    # Check if prediction is valid
                    if not masks[b, t, predicted_action]:
                        self.invalid_predictions += 1

                    self.total_predictions += 1

                    # Classify action types
                    # Exclude teampreview steps from all calculations
                    is_teampreview = states[b, t, feature_idx["teampreview"]].item() > 0.5
                    if is_teampreview:
                        continue

                    is_force_switch = False
                    for idx in feature_idx["force_switch_indices"]:  # type: ignore
                        if states[b, t, idx].item() > 0.5:
                            is_force_switch = True
                            break

                    if is_force_switch:
                        action_type = "FORCE_SWITCH"
                        order_type = MDBO.FORCE_SWITCH
                    else:
                        # Only analyze regular turn actions (MOVE/SWITCH/BOTH/OTHER/INVALID)
                        order_type = MDBO.TURN
                        action_type = self.classify_action_type(actual_action, order_type)

                    # Record action type distributions
                    self.action_type_actual[action_type] += 1
                    pred_type = self.classify_action_type(predicted_action, order_type)
                    self.action_type_pred[pred_type] += 1

                    # Record confidence
                    self.action_confidences[action_type].append(confidence)

                    # Check top-k accuracy
                    topk_actions = topk_indices[b, t, :].cpu().numpy()
                    for k in [1, 3, 5, 10]:
                        if actual_action in topk_actions[:k]:
                            self.topk_hits[action_type][f"top{k}"] += 1

                    # Calculate loss contribution
                    ce_loss = -torch.log(probs[b, t, actual_action] + 1e-10).item()
                    self.loss_contributions[action_type].append(ce_loss)

                    # Track move predictions (for MOVE action types)
                    if action_type == "MOVE":
                        self.move_actuals[actual_action] += 1
                        self.move_predictions[predicted_action] += 1

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""

        report: Dict[str, Dict] = {
            "action_type_distribution": {},
            "topk_accuracy_by_type": {},
            "confidence_stats": {},
            "loss_contribution": {},
            "move_analysis": {},
            "invalid_predictions": {
                "count": self.invalid_predictions,
                "rate": self.invalid_predictions / max(self.total_predictions, 1),
            },
        }

        # Action type distribution
        for action_type in set(
            list(self.action_type_actual.keys()) + list(self.action_type_pred.keys())
        ):
            report["action_type_distribution"][action_type] = {
                "actual_count": self.action_type_actual[action_type],
                "actual_pct": self.action_type_actual[action_type]
                / max(sum(self.action_type_actual.values()), 1),
                "predicted_count": self.action_type_pred[action_type],
                "predicted_pct": self.action_type_pred[action_type]
                / max(sum(self.action_type_pred.values()), 1),
                "prediction_bias": self.action_type_pred[action_type]
                / max(sum(self.action_type_pred.values()), 1)
                - self.action_type_actual[action_type]
                / max(sum(self.action_type_actual.values()), 1),
            }

        # Top-k accuracy by type
        for action_type, hits in self.topk_hits.items():
            total = self.action_type_actual[action_type]
            report["topk_accuracy_by_type"][action_type] = {
                "total": total,
                **{k: v / max(total, 1) for k, v in hits.items()},
            }

        # Overall top-k accuracy (aggregated across all action types)
        total_actions = sum(self.action_type_actual.values())
        overall_topk = {}
        for k in [1, 3, 5, 10]:
            total_hits = sum(hits.get(f"top{k}", 0) for hits in self.topk_hits.values())
            overall_topk[f"top{k}"] = total_hits / max(total_actions, 1)
        report["overall_accuracy"] = {"total": total_actions, **overall_topk}

        # Confidence stats
        for action_type, confidences in self.action_confidences.items():
            if confidences:
                report["confidence_stats"][action_type] = {
                    "mean": float(np.mean(confidences)),
                    "std": float(np.std(confidences)),
                    "median": float(np.median(confidences)),
                    "min": float(np.min(confidences)),
                    "max": float(np.max(confidences)),
                }

        # Loss contribution
        total_loss = sum(sum(losses) for losses in self.loss_contributions.values())
        for action_type, losses in self.loss_contributions.items():
            if losses:
                report["loss_contribution"][action_type] = {
                    "total_loss": float(np.sum(losses)),
                    "mean_loss": float(np.mean(losses)),
                    "pct_of_total": float(np.sum(losses) / max(total_loss, 1)),
                    "count": len(losses),
                }

        # Move analysis
        if self.move_actuals:
            most_common_actual = self.move_actuals.most_common(20)
            most_common_pred = self.move_predictions.most_common(20)

            report["move_analysis"] = {
                "unique_moves_actual": len(self.move_actuals),
                "unique_moves_predicted": len(self.move_predictions),
                "most_common_actual": [
                    {"action": action, "count": count}
                    for action, count in most_common_actual
                ],
                "most_common_predicted": [
                    {"action": action, "count": count}
                    for action, count in most_common_pred
                ],
                "overlap": len(
                    set(self.move_actuals.keys()) & set(self.move_predictions.keys())
                ),
            }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Pretty print the diagnostic report."""

        print("\n" + "=" * 60)
        print("ACTION MODEL DIAGNOSTIC REPORT")
        print("=" * 60)

        # Print overall accuracy first
        print("\n### OVERALL ACCURACY ###")
        oa = report["overall_accuracy"]
        print(f"Total predictions: {oa['total']}")
        print(f"Top-1:  {oa.get('top1', 0) * 100:>6.2f}%")
        print(f"Top-3:  {oa.get('top3', 0) * 100:>6.2f}%")
        print(f"Top-5:  {oa.get('top5', 0) * 100:>6.2f}%")
        print(f"Top-10: {oa.get('top10', 0) * 100:>6.2f}%")

        print("\n### ACTION TYPE DISTRIBUTION ###")
        print(f"{'Type':<15} {'Actual %':<12} {'Pred %':<12} {'Bias':<12}")
        print("-" * 60)
        for action_type, stats in sorted(report["action_type_distribution"].items()):
            print(
                f"{action_type:<15} {stats['actual_pct'] * 100:>10.2f}% {stats['predicted_pct'] * 100:>10.2f}% {stats['prediction_bias'] * 100:>+10.2f}%"
            )

        print("\n### TOP-K ACCURACY BY ACTION TYPE ###")
        print(
            f"{'Type':<15} {'Count':<10} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Top-10':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(report["topk_accuracy_by_type"].items()):
            print(
                f"{action_type:<15} {stats['total']:<10} {stats.get('top1', 0) * 100:>8.2f}% {stats.get('top3', 0) * 100:>8.2f}% {stats.get('top5', 0) * 100:>8.2f}% {stats.get('top10', 0) * 100:>8.2f}%"
            )

        print("\n### PREDICTION CONFIDENCE BY TYPE ###")
        print(
            f"{'Type':<15} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(report["confidence_stats"].items()):
            print(
                f"{action_type:<15} {stats['mean']:>8.4f}  {stats['std']:>8.4f}  {stats['median']:>8.4f}  {stats['min']:>8.4f}  {stats['max']:>8.4f}"
            )

        print("\n### LOSS CONTRIBUTION BY TYPE ###")
        print(
            f"{'Type':<15} {'Total Loss':<15} {'Mean Loss':<15} {'% of Total':<15} {'Count':<10}"
        )
        print("-" * 75)
        for action_type, stats in sorted(
            report["loss_contribution"].items(),
            key=lambda x: x[1]["total_loss"],
            reverse=True,
        ):
            print(
                f"{action_type:<15} {stats['total_loss']:>13.2f}  {stats['mean_loss']:>13.4f}  {stats['pct_of_total'] * 100:>13.2f}% {stats['count']:>10}"
            )

        if "move_analysis" in report and report["move_analysis"]:
            print("\n### MOVE PREDICTION ANALYSIS ###")
            ma = report["move_analysis"]
            print(f"Unique moves (actual): {ma['unique_moves_actual']}")
            print(f"Unique moves (predicted): {ma['unique_moves_predicted']}")
            print(f"Overlap: {ma['overlap']}")

            print("\nMost common actual moves:")
            for i, move in enumerate(ma["most_common_actual"][:10], 1):
                print(
                    f"  {i}. Action {move['action']}: {move['count']} times ({MDBO.from_int(move['action'], MDBO.TURN).message if MDBO.from_int(move['action'], MDBO.TURN) else 'N/A'})"
                )

            print("\nMost common predicted moves:")
            for i, move in enumerate(ma["most_common_predicted"][:10], 1):
                print(
                    f"  {i}. Action {move['action']}: {move['count']} times ({MDBO.from_int(move['action'], MDBO.TURN).message if MDBO.from_int(move['action'], MDBO.TURN) else 'N/A'})"
                )

        print("\n### INVALID PREDICTIONS ###")
        inv = report["invalid_predictions"]
        print(f"Invalid predictions: {inv['count']} ({inv['rate'] * 100:.2f}%)")

        print("\n" + "=" * 60)


def main(model_path: str, data_path: str, max_batches: Optional[int] = 100):
    """
    Run diagnostics on a trained model.

    Args:
        model_path: Path to saved model (.pt file)
        data_path: Path to evaluation data directory
        max_batches: Maximum number of batches to analyze
    """

    print(f"Loading model from {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize embedder
    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=config["embedder_feature_set"],
        omniscient=False,
    )

    # Get feature indices
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    feature_idx: Dict[str, Any] = {
        "teampreview": feature_names["teampreview"],
        "force_switch_indices": [feature_names[f"MON:{j}:force_switch"] for j in range(6)],
    }

    # Initialize model
    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config["lstm_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
        dropout=0.0,  # No dropout for evaluation
        gated_residuals=config["gated_residuals"],
        early_attention_heads=config["early_attention_heads"],
        late_attention_heads=config["late_attention_heads"],
        use_grouped_encoder=config["use_grouped_encoder"],
        group_sizes=(
            embedder.group_embedding_sizes if config["use_grouped_encoder"] else None
        ),
        grouped_encoder_hidden_dim=config["grouped_encoder_hidden_dim"],
        grouped_encoder_aggregated_dim=config["grouped_encoder_aggregated_dim"],
        pokemon_attention_heads=config["pokemon_attention_heads"],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        teampreview_head_layers=config["teampreview_head_layers"],
        teampreview_head_dropout=0.0,
        teampreview_attention_heads=config["teampreview_attention_heads"],
        turn_head_layers=config["turn_head_layers"],
        max_seq_len=config["max_seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully")

    # Load data
    print(f"Loading data from {data_path}")
    dataloader = OptimizedBattleDataLoader(
        data_path,
        embedder=embedder,
        batch_size=64,
        num_workers=7,
        prefetch_factor=8,
        files_per_worker=3,
    )

    # Run diagnostics
    print("Running diagnostics...")
    diagnostics = ActionDiagnostics(model, device)

    # Use tqdm for progress bar
    dataloader_iter = enumerate(dataloader)
    if max_batches is not None:
        dataloader_iter = enumerate(
            tqdm(dataloader, total=max_batches, desc="Processing batches")
        )
    else:
        dataloader_iter = enumerate(tqdm(dataloader, desc="Processing batches"))

    for i, batch in dataloader_iter:
        if max_batches is not None and i >= max_batches:
            break
        diagnostics.analyze_batch(batch, feature_idx)

    # Generate and print report
    report = diagnostics.generate_report()
    diagnostics.print_report(report)

    # Save report to JSON
    output_path = model_path.replace(".pt", "_action_diagnostics.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python action_model_diagnostics.py <model_path> <data_path> [max_batches]"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    if len(sys.argv) > 3:
        max_batches: Optional[int] = int(sys.argv[3])
    else:
        max_batches = None  # None means process all batches in the folder

    main(model_path, data_path, max_batches)
