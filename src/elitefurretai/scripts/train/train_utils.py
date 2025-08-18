"""
Model evaluation and analysis utilities for Pokémon battle prediction models.

This module provides functions to evaluate and analyze model performance on battle data,
supporting both one-headed (action prediction only) and two-headed (action + win prediction) models.
It includes functions for computing metrics, analyzing performance across different battle states,
and processing batch data.

Functions:
    evaluate: Evaluate model performance on a dataset.
    analyze: Analyze model performance across different battle characteristics.
    flatten_and_filter: Prepare batch data by flattening and filtering.
    process_batch: Process a batch for analysis.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from elitefurretai.model_utils.model_double_battle_order import MDBO


def topk_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    k: int = 1,
) -> torch.Tensor:
    """
    Returns 0 loss if target is in top-k predictions, else cross-entropy loss.

    Args:
        logits: [batch, num_classes] Logits from the model
        targets: [batch] Target class indices
        weights: Optional weights for different samples
        k: Number of top predictions to consider

    Returns:
        Mean loss value
    """
    # Get top-k indices
    topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1).indices  # [batch, k]
    # Check if target is in top-k
    in_topk = (topk == targets.unsqueeze(1)).any(dim=1)  # [batch]
    # Compute standard cross-entropy
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none") / 7.0
    # Only penalize if not in top-k
    loss = ce_loss * (~in_topk)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def flatten_and_filter(
    states: torch.Tensor,
    action_logits: torch.Tensor,
    actions: torch.Tensor,
    win_logits: Optional[torch.Tensor] = None,
    wins: Optional[torch.Tensor] = None,
    action_masks: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
) -> Optional[Tuple]:
    """
    Flatten batch dimensions and filter for valid samples.

    Args:
        states: [batch, seq_len, state_dim] State representations
        action_logits: [batch, seq_len, num_actions] Action logits
        actions: [batch, seq_len] Target actions
        win_logits: [batch, seq_len] Win prediction logits (optional)
        wins: [batch, seq_len] Win labels (optional)
        action_masks: [batch, seq_len, num_actions] Mask for valid actions
        masks: [batch, seq_len] Mask for valid steps

    Returns:
        Tuple of valid data or None if no valid data
    """
    # Flatten all tensors
    flat_states = states.view(-1, states.size(-1))

    if masks is not None:
        flat_mask = masks.bool().view(-1)
    else:
        flat_mask = torch.ones(flat_states.size(0), dtype=torch.bool, device=states.device)

    flat_action_logits = action_logits.view(-1, action_logits.size(-1))
    flat_actions = actions.view(-1)

    if win_logits is not None and wins is not None:
        flat_win_logits = win_logits.view(-1)
        flat_wins = wins.view(-1)
    else:
        flat_win_logits = None
        flat_wins = None

    if action_masks is not None:
        flat_action_masks = action_masks.view(-1, action_masks.size(-1))
    else:
        flat_action_masks = torch.ones_like(flat_action_logits, dtype=torch.bool)

    # Filter for valid (unpadded) positions
    valid_idx = flat_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None

    # Apply valid indices
    valid_states = flat_states[valid_idx]
    valid_action_logits = flat_action_logits[valid_idx]
    valid_actions = flat_actions[valid_idx]
    if flat_win_logits is not None and flat_wins is not None:
        valid_win_logits = flat_win_logits[valid_idx]
        valid_wins = flat_wins[valid_idx]
    else:
        valid_win_logits = None
        valid_wins = None
    valid_action_masks = flat_action_masks[valid_idx]

    # Filter for samples with at least one valid action
    valid_action_rows = valid_action_masks.sum(dim=1) > 0
    if valid_action_rows.sum() == 0:
        return None

    result = [
        valid_states[valid_action_rows],
        valid_action_logits[valid_action_rows],
        valid_actions[valid_action_rows],
    ]

    if valid_win_logits is not None and valid_wins is not None:
        result.extend([valid_win_logits[valid_action_rows], valid_wins[valid_action_rows]])

    return tuple(result)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    has_win_head: bool = True,
    action_mask_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        has_win_head: Whether the model has a win prediction head
        action_mask_fn: Optional function to apply action masking

    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_samples = 0
    running_action_loss = 0.0
    running_win_loss = 0.0 if has_win_head else 0.0
    running_win_acc = 0.0 if has_win_head else 0.0
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, wins, masks = batch
        states = states.to(device)
        actions = actions.to(device)
        action_masks = action_masks.to(device)

        if has_win_head:
            wins = wins.to(device)

        masks = masks.to(device)

        # Forward pass - handle both model types
        if has_win_head:
            action_logits, win_logits = model(states, masks)
        else:
            action_logits = model(states, masks)
            win_logits = None

        # Apply action masks if provided
        if action_mask_fn is not None:
            masked_action_logits = action_mask_fn(action_logits, action_masks)
        else:
            # Default masking
            masked_action_logits = action_logits.masked_fill(
                ~action_masks.bool(), float("-inf")
            )

        # Use helper for flattening and filtering
        if has_win_head:
            flat_data = flatten_and_filter(
                states,
                masked_action_logits,
                actions,
                win_logits,
                wins,
                action_masks,
                masks,
            )
        else:
            flat_data = flatten_and_filter(
                states, masked_action_logits, actions, None, None, action_masks, masks
            )

        if flat_data is None:
            continue

        if has_win_head:
            (
                valid_states,
                valid_action_logits,
                valid_actions,
                valid_win_logits,
                valid_wins,
            ) = flat_data
        else:
            valid_states, valid_action_logits, valid_actions = flat_data
            valid_win_logits = None
            valid_wins = None

        # Calculate action accuracy
        top1_preds = torch.argmax(valid_action_logits, dim=1)
        top3_preds = torch.topk(
            valid_action_logits, k=min(3, valid_action_logits.size(1)), dim=1
        ).indices
        top5_preds = torch.topk(
            valid_action_logits, k=min(5, valid_action_logits.size(1)), dim=1
        ).indices

        top1_correct = top1_preds == valid_actions
        top3_correct = (valid_actions.unsqueeze(1) == top3_preds).any(dim=1)
        top5_correct = (valid_actions.unsqueeze(1) == top5_preds).any(dim=1)

        # Win accuracy if applicable
        if has_win_head and valid_win_logits is not None and valid_wins is not None:
            win_preds = (torch.sigmoid(valid_win_logits) > 0.5).float()
            win_acc = (win_preds == valid_wins).float().sum().item()
            win_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                valid_win_logits, valid_wins.float()
            )
            running_win_loss += win_loss.item()
            running_win_acc += win_acc

        # Action loss
        action_loss = topk_cross_entropy_loss(valid_action_logits, valid_actions)
        running_action_loss += action_loss.item()

        # Accumulate metrics
        total_top1 += top1_correct.sum().item()
        total_top3 += top3_correct.sum().item()
        total_top5 += top5_correct.sum().item()
        total_samples += valid_actions.size(0)
        num_batches += 1

    # Calculate final metrics
    metrics = {}
    if total_samples > 0:
        metrics["action_top1"] = total_top1 / total_samples
        metrics["action_top3"] = total_top3 / total_samples
        metrics["action_top5"] = total_top5 / total_samples
        metrics["action_loss"] = (
            running_action_loss / num_batches if num_batches > 0 else float("inf")
        )

        if has_win_head:
            metrics["win_loss"] = (
                running_win_loss / num_batches if num_batches > 0 else float("inf")
            )
            metrics["win_acc"] = running_win_acc / total_samples
            metrics["loss"] = metrics["action_loss"] + metrics["win_loss"]
        else:
            metrics["loss"] = metrics["action_loss"]
    else:
        # Default values if no samples
        metrics["action_top1"] = 0.0
        metrics["action_top3"] = 0.0
        metrics["action_top5"] = 0.0
        metrics["action_loss"] = float("inf")
        if has_win_head:
            metrics["win_loss"] = float("inf")
            metrics["win_acc"] = 0.0
            metrics["loss"] = float("inf")
        else:
            metrics["loss"] = float("inf")

    return metrics


def mdbo_converter(action_idx):
    return MDBO.from_int(action_idx, MDBO.TURN)


@torch.no_grad()
def analyze(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    feature_names: List[str],
    device: str = "cpu",
    has_win_head: bool = True,
    action_mask_fn: Optional[Callable] = None,
    mdbo_fn: Optional[Callable] = mdbo_converter,
) -> Dict[str, Dict[str, Dict[str, Union[int, float]]]]:
    """
    Analyze model performance across different battle characteristics.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        feature_names: List of feature names corresponding to state vectors
        device: Device to run evaluation on
        has_win_head: Whether the model has a win prediction head
        action_mask_fn: Optional function to apply action masking
        mdbo_fn: Function to convert action index to MDBO (required for action type analysis)

    Returns:
        Nested dictionary with analysis results
    """
    model.eval()

    # Initialize analysis categories
    analysis_categories: Dict[str, Dict[str, Dict[str, Union[int, float]]]] = {
        "turns": defaultdict(lambda: create_empty_metrics()),
        "ko_opportunities": defaultdict(lambda: create_empty_metrics()),
        "mons_alive": defaultdict(lambda: create_empty_metrics()),
    }

    # Only add action_types if mdbo_fn is provided
    if mdbo_fn is not None:
        analysis_categories["action_types"] = defaultdict(lambda: create_empty_metrics())

    # Create feature name lookup and identify KO features
    feature_idx = {name: i for i, name in enumerate(feature_names)}
    ko_features = {v for k, v in feature_idx.items() if "KO" in k}

    for batch in dataloader:
        # Process batch and get predictions
        predictions = process_batch(model, batch, device, has_win_head, action_mask_fn)
        if predictions is None:
            continue

        if has_win_head:
            (
                states,
                valid_actions,
                top1_correct,
                top3_correct,
                top5_correct,
                win_correct,
            ) = predictions
        else:
            states, valid_actions, top1_correct, top3_correct, top5_correct = predictions
            win_correct = torch.zeros_like(top1_correct).float()  # dummy

        # Analyze each valid sample
        for i, state in enumerate(states):
            sample_keys = extract_sample_keys(
                state, valid_actions[i], feature_idx, ko_features, mdbo_fn
            )
            sample_metrics = [
                top1_correct[i].item(),
                top3_correct[i].item(),
                top5_correct[i].item(),
                win_correct[i].item() if has_win_head else 0.0,
                1,  # sample count
            ]

            # Update metrics for each category
            metric_names = [
                "total_top1",
                "total_top3",
                "total_top5",
                "win",
                "total_samples",
            ]
            for category, key in sample_keys.items():
                if (
                    category in analysis_categories
                ):  # Skip action_types if mdbo_fn not provided
                    for metric_name, value in zip(metric_names, sample_metrics):
                        analysis_categories[category][str(key)][metric_name] += value

    print_analysis_results(
        analysis_categories=analysis_categories, has_win_head=has_win_head
    )
    return analysis_categories


def process_batch(
    model: torch.nn.Module,
    batch: List[torch.Tensor],
    device: str,
    has_win_head: bool = True,
    action_mask_fn: Optional[Callable] = None,
) -> Optional[Tuple]:
    """
    Process a single batch and return predictions.

    Args:
        model: The model to evaluate
        batch: A batch of data
        device: Device to run on
        has_win_head: Whether the model has a win prediction head
        action_mask_fn: Optional function to apply action masking

    Returns:
        Tuple of processed data or None
    """
    states, actions, action_masks, wins, masks = batch

    # Move to device
    states = states.to(device)
    actions = actions.to(device)
    action_masks = action_masks.to(device)
    if has_win_head:
        wins = wins.to(device)
    masks = masks.to(device)

    # Forward pass
    if has_win_head:
        action_logits, win_logits = model(states, masks)
    else:
        action_logits = model(states, masks)
        win_logits = None

    # Apply action masks
    if action_mask_fn is not None:
        masked_action_logits = action_mask_fn(action_logits, action_masks)
    else:
        # Default masking
        masked_action_logits = action_logits.masked_fill(
            ~action_masks.bool(), float("-inf")
        )

    # Flatten and filter valid samples
    if has_win_head:
        flat_data = flatten_and_filter(
            states, masked_action_logits, actions, win_logits, wins, action_masks, masks
        )
    else:
        flat_data = flatten_and_filter(
            states, masked_action_logits, actions, None, None, action_masks, masks
        )

    if flat_data is None:
        return None

    if has_win_head:
        valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
            flat_data
        )
    else:
        valid_states, valid_action_logits, valid_actions = flat_data
        valid_win_logits = None
        valid_wins = None

    # Calculate predictions and correctness
    top1_preds = torch.argmax(valid_action_logits, dim=1)
    top3_preds = torch.topk(
        valid_action_logits, k=min(3, valid_action_logits.size(1)), dim=1
    ).indices
    top5_preds = torch.topk(
        valid_action_logits, k=min(5, valid_action_logits.size(1)), dim=1
    ).indices

    top1_correct = top1_preds == valid_actions
    top3_correct = (valid_actions.unsqueeze(1) == top3_preds).any(dim=1)
    top5_correct = (valid_actions.unsqueeze(1) == top5_preds).any(dim=1)

    result = [
        valid_states,
        valid_actions,
        top1_correct,
        top3_correct,
        top5_correct,
    ]

    if has_win_head and valid_win_logits is not None and valid_wins is not None:
        win_preds = (torch.sigmoid(valid_win_logits) > 0.5).float()
        win_correct = (win_preds == valid_wins).float()
        result.append(win_correct)

    return tuple(result)


def create_empty_metrics() -> Dict[str, Union[int, float]]:
    """Create empty metrics dictionary."""
    return {
        "total_top1": 0.0,
        "total_top3": 0.0,
        "total_top5": 0.0,
        "win": 0.0,
        "total_samples": 0.0,
    }


def extract_sample_keys(
    state: torch.Tensor,
    action: torch.Tensor,
    feature_idx: Dict[str, int],
    ko_features: set,
    mdbo_fn: Optional[Callable] = None,
) -> Dict[str, str]:
    """
    Extract categorization keys for a single sample.

    Args:
        state: State vector
        action: Action index
        feature_idx: Dictionary mapping feature names to indices
        ko_features: Set of KO feature indices
        mdbo_fn: Function to convert action index to MDBO (optional)

    Returns:
        Dictionary of category keys
    """
    result = {}

    # Basic features
    turn_key = "turn"
    if turn_key in feature_idx:
        turn = int(state[feature_idx[turn_key]].item())
        result["turns"] = str(turn)
    else:
        result["turns"] = "unknown"

    # KO opportunities
    if ko_features:
        can_ko = str(max(state[feature_idx].item() for feature_idx in ko_features))
        result["ko_opportunities"] = str(can_ko)
    else:
        result["ko_opportunities"] = "unknown"

    # Number of Pokémon alive
    num_fainted_key = "NUM_FAINTED"
    opp_num_fainted_key = "OPP_NUM_FAINTED"
    if num_fainted_key in feature_idx and opp_num_fainted_key in feature_idx:
        num_alive = int(
            8
            - state[feature_idx[opp_num_fainted_key]].item()
            - state[feature_idx[num_fainted_key]].item()
        )
        result["mons_alive"] = str(num_alive)
    else:
        result["mons_alive"] = "unknown"

    # Determine action type if mdbo_fn is provided
    if mdbo_fn is not None:
        try:
            result["action_types"] = determine_action_type(
                state, action, feature_idx, mdbo_fn
            )
        except Exception:
            result["action_types"] = "unknown"

    return result


def determine_action_type(
    state: torch.Tensor,
    action: torch.Tensor,
    feature_idx: Dict[str, int],
    mdbo_fn: Callable,
) -> str:
    """
    Determine the type of action being taken.

    Args:
        state: State vector
        action: Action index
        feature_idx: Dictionary mapping feature names to indices
        mdbo_fn: Function to convert action index to MDBO

    Returns:
        Action type string
    """
    # Check for teampreview
    if "teampreview" in feature_idx and int(state[feature_idx["teampreview"]].item()) == 1:
        return "teampreview"

    # Check for force switch
    for j in range(6):
        force_switch_key = f"MON:{j}:force_switch"
        if (
            force_switch_key in feature_idx
            and int(state[feature_idx[force_switch_key]].item()) == 1
        ):
            return "force_switch"

    # Parse action message using provided function
    try:
        action_msg = mdbo_fn(int(action.item())).message
        has_switch = "switch" in action_msg
        has_move = "move" in action_msg

        if has_switch and has_move:
            return "both"
        elif has_move:
            return "move"
        elif has_switch:
            return "switch"
        else:
            return "unknown"
    except Exception:
        return "unknown"


def print_analysis_results(
    analysis_categories: Dict[str, Dict[str, Dict[str, Union[int, float]]]],
    has_win_head: bool = True,
) -> None:
    """
    Print formatted analysis results.

    Args:
        analysis_categories: Analysis results to print
        has_win_head: Whether win metrics should be printed
    """
    print("\nAnalysis complete! Results:")

    category_display_names = {
        "turns": "Turn",
        "action_types": "Action Types",
        "ko_opportunities": "KO Opportunities",
        "mons_alive": "Pokemon Alive",
    }

    for category, display_name in category_display_names.items():
        if category in analysis_categories:
            results = analysis_categories[category]
            print(f"\nAnalysis by {display_name}:")

            for key in sorted(results.keys()):
                metrics = results[key]
                if metrics["total_samples"] > 0:
                    top1_acc = metrics["total_top1"] / metrics["total_samples"] * 100
                    top3_acc = metrics["total_top3"] / metrics["total_samples"] * 100
                    top5_acc = metrics["total_top5"] / metrics["total_samples"] * 100

                    output = (
                        f"  {key}: Top-1: {top1_acc:.1f}%, Top-3: {top3_acc:.1f}%, "
                        f"Top-5: {top5_acc:.1f}%"
                    )

                    if has_win_head:
                        win_acc = metrics["win"] / metrics["total_samples"] * 100
                        output += f", Win: {win_acc:.1f}%"

                    output += f" (n={metrics['total_samples']})"
                    print(output)
                else:
                    print(f"  {key}: No samples")


def format_time(seconds: float) -> str:
    """Format seconds into hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


"""
# For two-headed model:
metrics = evaluate(model, dataloader, device="cuda", has_win_head=True)

# For one-headed model:
metrics = evaluate(model, dataloader, device="cuda", has_win_head=False)

# Analysis with MDBO function:
from elitefurretai.model_utils import MDBO
def mdbo_converter(action_idx):
    return MDBO.from_int(action_idx, MDBO.TURN)

analysis_results = analyze(
    model,
    dataloader,
    embedder.feature_names,
    device="cuda",
    has_win_head=True,
    mdbo_fn=mdbo_converter
)

print_analysis_results(analysis_results, has_win_head=True)
"""
