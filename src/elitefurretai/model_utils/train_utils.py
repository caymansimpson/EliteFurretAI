from typing import Any, Callable, Dict, List, Optional, Tuple
import zstd
import io
import torch

from elitefurretai.model_utils.encoder import MDBO


def topk_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    k: int = 3,
) -> torch.Tensor:
    """
    Compute cross entropy loss, only considering the top-k predictions per example.

    Args:
        logits: [batch, num_classes] Unnormalized logits
        labels: [batch] Target class indices
        weights: [batch] Optional weights per example
        k: Number of top predictions to consider

    Returns:
        Weighted cross entropy loss
    """
    # Get top-k indices
    _, topk_indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
    label_indices = labels.unsqueeze(-1)

    # Check if labels are in top-k
    is_label_in_topk = (topk_indices == label_indices).any(dim=-1)

    # Compute loss only for examples where label is in top-k
    valid_indices = is_label_in_topk.nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits = logits[valid_indices]
    valid_labels = labels[valid_indices]

    # Compute cross entropy loss
    loss = torch.nn.functional.cross_entropy(valid_logits, valid_labels, reduction="none")

    # Apply weights if provided
    if weights is not None:
        valid_weights = weights[valid_indices]
        loss = loss * valid_weights
        loss = loss.sum() / (valid_weights.sum() + 1e-8)
    else:
        loss = loss.mean()

    return loss


def flatten_and_filter(
    states: torch.Tensor,
    action_logits: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    win_logits: Optional[torch.Tensor] = None,
    wins: Optional[torch.Tensor] = None,
    action_masks: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    move_orders: Optional[torch.Tensor] = None,
    move_order_logits: Optional[torch.Tensor] = None,
    kos: Optional[torch.Tensor] = None,
    ko_logits: Optional[torch.Tensor] = None,
    switches: Optional[torch.Tensor] = None,
    switch_logits: Optional[torch.Tensor] = None,
) -> Optional[Tuple]:
    """
    Flatten batch dimensions and filter for valid samples.

    Args:
        states: [batch, seq_len, state_dim] State representations
        action_logits: [batch, seq_len, num_actions] Action logits
        actions: [batch, seq_len] Target actions
        win_logits: [batch, seq_len] Win prediction logits
        wins: [batch, seq_len] Win labels
        action_masks: [batch, seq_len, num_actions] Mask for valid actions
        masks: [batch, seq_len] Mask for valid steps
        move_orders: [batch, seq_len] Move order labels
        move_order_logits: [batch, seq_len, 24] Move order logits
        kos: [batch, seq_len, 4] KO labels
        ko_logits: [batch, seq_len, 4] KO logits
        switches: [batch, seq_len, 4] Switch labels
        switch_logits: [batch, seq_len, 4] Switch logits

    Returns:
        Tuple of valid data or None if no valid data
    """
    # Flatten all tensors
    flat_states = states.view(-1, states.size(-1))

    if masks is not None:
        flat_mask = masks.bool().view(-1)
    else:
        flat_mask = torch.ones(flat_states.size(0), dtype=torch.bool, device=states.device)

    if action_logits is not None and actions is not None:
        flat_action_logits = action_logits.view(-1, action_logits.size(-1))
        flat_actions = actions.view(-1)
    else:
        flat_action_logits = None
        flat_actions = None

    # Flatten win tensors
    if win_logits is not None and wins is not None:
        flat_win_logits = win_logits.view(-1)
        flat_wins = wins.view(-1)
    else:
        flat_win_logits = None
        flat_wins = None

    # Flatten move order tensors
    if move_orders is not None and move_order_logits is not None:
        flat_move_orders = move_orders.view(-1)
        flat_move_order_logits = move_order_logits.view(-1, move_order_logits.size(-1))
    else:
        flat_move_orders = None
        flat_move_order_logits = None

    # Flatten KO tensors
    if kos is not None and ko_logits is not None:
        flat_kos = kos.view(-1, kos.size(-1))
        flat_ko_logits = ko_logits.view(-1, ko_logits.size(-1))
    else:
        flat_kos = None
        flat_ko_logits = None

    # Flatten switch tensors
    if switches is not None and switch_logits is not None:
        flat_switches = switches.view(-1, switches.size(-1))
        flat_switch_logits = switch_logits.view(-1, switch_logits.size(-1))
    else:
        flat_switches = None
        flat_switch_logits = None

    if action_masks is not None and flat_action_logits is not None:
        flat_action_masks = action_masks.view(-1, action_masks.size(-1))
    else:
        flat_action_masks = None

    # Filter for valid (unpadded) positions
    valid_idx = flat_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None

    # Apply valid indices
    valid_states = flat_states[valid_idx]

    # Apply valid indices to action tensors
    if (
        flat_action_logits is not None
        and flat_actions is not None
        and flat_action_masks is not None
    ):
        valid_action_logits = flat_action_logits[valid_idx]
        valid_actions = flat_actions[valid_idx]
        valid_action_masks = flat_action_masks[valid_idx]
    else:
        valid_action_logits = None
        valid_actions = None
        valid_action_masks = None

    # Apply valid indices to win tensors
    if flat_win_logits is not None and flat_wins is not None:
        valid_win_logits = flat_win_logits[valid_idx]
        valid_wins = flat_wins[valid_idx]
    else:
        valid_win_logits = None
        valid_wins = None

    # Apply valid indices to move order tensors
    if flat_move_orders is not None and flat_move_order_logits is not None:
        valid_move_orders = flat_move_orders[valid_idx]
        valid_move_order_logits = flat_move_order_logits[valid_idx]
    else:
        valid_move_orders = None
        valid_move_order_logits = None

    # Apply valid indices to KO tensors
    if flat_kos is not None and flat_ko_logits is not None:
        valid_kos = flat_kos[valid_idx]
        valid_ko_logits = flat_ko_logits[valid_idx]
    else:
        valid_kos = None
        valid_ko_logits = None

    # Apply valid indices to switch tensors
    if flat_switches is not None and flat_switch_logits is not None:
        valid_switches = flat_switches[valid_idx]
        valid_switch_logits = flat_switch_logits[valid_idx]
    else:
        valid_switches = None
        valid_switch_logits = None

    # Filter for samples with at least one valid action if we have an action head
    if valid_action_masks is not None:
        valid_action_rows = valid_action_masks.sum(dim=1) > 0
        if not valid_action_rows.any():
            return None
        final_idx = torch.nonzero(valid_action_rows, as_tuple=True)[0]
    else:
        final_idx = torch.nonzero(valid_idx, as_tuple=True)[0]

    # Make sure we have something
    if final_idx is None or (final_idx is not None and final_idx.numel() == 0):
        return None

    # Build result with all valid tensors
    result = [valid_states[final_idx]]

    if valid_action_logits is not None and valid_actions is not None:
        result.append(valid_action_logits[final_idx])
        result.append(valid_actions[final_idx])

    if valid_win_logits is not None and valid_wins is not None:
        result.append(valid_win_logits[final_idx])
        result.append(valid_wins[final_idx])

    if valid_move_order_logits is not None and valid_move_orders is not None:
        result.append(valid_move_order_logits[final_idx])
        result.append(valid_move_orders[final_idx])

    if valid_ko_logits is not None and valid_kos is not None:
        result.append(valid_ko_logits[final_idx])
        result.append(valid_kos[final_idx])

    if valid_switch_logits is not None and valid_switches is not None:
        result.append(valid_switch_logits[final_idx])
        result.append(valid_switches[final_idx])

    return tuple(result)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    has_action_head: bool = True,
    has_win_head: bool = True,
    has_move_order_head: bool = True,
    has_ko_head: bool = True,
    has_switch_head: bool = True,
    has_teampreview_head: bool = False,
    teampreview_idx: Optional[int] = None,
    action_mask_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        has_win_head: Whether model has a win prediction head
        action_mask_fn: Optional function to create action masks

    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = create_empty_metrics()
    steps = 0

    for batch in dataloader:
        states = batch["states"].to(device).to(torch.float32)
        actions = batch["actions"].to(device)
        action_masks = (
            batch["action_masks"].to(device) if "action_masks" in batch else None
        )
        masks = batch["masks"].to(device) if "masks" in batch else None
        wins = batch["wins"].to(device).to(torch.float32) if has_win_head and "wins" in batch else None
        move_orders = batch["move_orders"].to(device) if "move_orders" in batch else None
        kos = batch["kos"].to(device) if "kos" in batch else None
        switches = batch["switches"].to(device) if "switches" in batch else None

        # Apply action masking if provided
        if action_mask_fn and action_masks is None:
            action_masks = action_mask_fn(states)

        # Full forward pass with all heads
        action_logits = win_logits = move_order_logits = ko_logits = switch_logits = teampreview_logits = None
        if has_teampreview_head:
            # Three-headed model: (turn_action_logits, teampreview_logits, win_logits)
            if has_action_head and has_win_head:
                action_logits, teampreview_logits, win_logits = model(states, masks)
            else:
                raise ValueError("Three-headed model requires has_action_head and has_win_head")
        elif (
            has_win_head
            and has_action_head
            and has_move_order_head
            and has_ko_head
            and has_switch_head
        ):
            action_logits, win_logits, move_order_logits, ko_logits, switch_logits = model(
                states, masks
            )
        elif has_action_head and not has_win_head:
            action_logits = model(states, masks)
            win_logits = move_order_logits = ko_logits = switch_logits = None
        elif (
            has_action_head
            and has_win_head
            and not has_move_order_head
            and not has_ko_head
            and not has_switch_head
        ):
            action_logits, win_logits = model(states, masks)
            move_order_logits = ko_logits = switch_logits = None

        # For three-headed models, merge teampreview and turn action logits
        # Similar to ThreeHeadedModelWrapper logic
        if has_teampreview_head and teampreview_logits is not None and action_logits is not None:
            # Identify teampreview samples
            teampreview_mask = states[:, :, teampreview_idx] == 1  # (batch, seq)

            # Start with turn logits
            merged_action_logits = action_logits.clone()

            # For teampreview samples, replace with teampreview logits
            # Zero out all positions for teampreview samples
            tp_mask_expanded = teampreview_mask.unsqueeze(-1).expand_as(merged_action_logits)
            merged_action_logits = torch.where(tp_mask_expanded, torch.tensor(float("-inf"), device=merged_action_logits.device), merged_action_logits)

            # Fill in teampreview actions [0:90] for teampreview samples
            tp_space = MDBO.teampreview_space()
            merged_action_logits[:, :, :tp_space] = torch.where(
                teampreview_mask.unsqueeze(-1).expand(-1, -1, tp_space),
                teampreview_logits,
                merged_action_logits[:, :, :tp_space]
            )

            # Apply action masks to merged logits
            if action_masks is not None:
                masked_action_logits = merged_action_logits.masked_fill(
                    ~action_masks.bool(), float("-inf")
                )
            else:
                masked_action_logits = merged_action_logits
        else:
            # Single action head - apply masks normally
            if action_masks is not None and action_logits is not None:
                masked_action_logits = action_logits.masked_fill(
                    ~action_masks.bool(), float("-inf")
                )
            else:
                masked_action_logits = action_logits

        # Use helper for flattening and filtering
        flat_data = flatten_and_filter(
            states=states,
            action_logits=masked_action_logits,
            actions=actions,
            win_logits=win_logits,
            wins=wins,
            action_masks=action_masks,
            masks=masks,
            move_orders=move_orders,
            move_order_logits=move_order_logits,
            kos=kos,
            ko_logits=ko_logits,
            switches=switches,
            switch_logits=switch_logits,
        )

        if flat_data is None:
            continue

        # Unpack the tuples
        valid_states = flat_data[0]
        head_idx = 1

        valid_action_logits = valid_actions = None
        if has_action_head:
            valid_action_logits, valid_actions = flat_data[head_idx : head_idx + 2]
            head_idx += 2

        valid_win_logits = valid_wins = None
        if has_win_head and wins is not None:
            valid_win_logits, valid_wins = flat_data[head_idx : head_idx + 2]
            head_idx += 2

        valid_move_order_logits = valid_move_orders = None
        if has_move_order_head and move_orders is not None:
            valid_move_order_logits, valid_move_orders = flat_data[head_idx : head_idx + 2]
            head_idx += 2

        valid_ko_logits = valid_kos = None
        if has_ko_head and kos is not None:
            valid_ko_logits, valid_kos = flat_data[head_idx : head_idx + 2]
            head_idx += 2

        valid_switch_logits = valid_switches = None
        if has_switch_head and switches is not None:
            valid_switch_logits, valid_switches = flat_data[head_idx : head_idx + 2]
            head_idx += 2

        if (
            has_action_head
            and valid_action_logits is not None
            and valid_actions is not None
        ):
            # Separate teampreview and turn samples if we have teampreview head
            if has_teampreview_head and teampreview_idx is not None and valid_states is not None:
                # Identify teampreview vs turn samples
                teampreview_mask = valid_states[:, teampreview_idx] == 1
                turn_mask = ~teampreview_mask

                # Process turn samples
                if turn_mask.any():
                    turn_logits = valid_action_logits[turn_mask]
                    turn_actions = valid_actions[turn_mask]

                    # Update turn accuracy metrics
                    turn_preds = torch.argmax(turn_logits, dim=1)
                    turn_correct = (turn_preds == turn_actions).float().sum().item()
                    metrics["turn_acc"] += turn_correct
                    metrics["turn_steps"] += turn_actions.size(0)

                    # Turn topk loss
                    turn_topk_loss = topk_cross_entropy_loss(turn_logits, turn_actions, k=3)
                    metrics["turn_top3_loss"] += turn_topk_loss.item() * turn_actions.size(0)

                    # Turn top-k accuracy
                    for k in [1, 3, 5]:
                        if turn_logits.size(-1) >= k:
                            topk_preds = torch.topk(turn_logits, k=k, dim=-1)[1]
                            turn_topk_correct = (topk_preds == turn_actions.unsqueeze(-1)).any(dim=-1).float().sum().item()
                            metrics[f"turn_top{k}_acc"] += turn_topk_correct

                # Process teampreview samples
                if teampreview_mask.any():
                    tp_logits = valid_action_logits[teampreview_mask]
                    tp_actions = valid_actions[teampreview_mask]

                    # Update teampreview accuracy metrics
                    tp_preds = torch.argmax(tp_logits, dim=1)
                    tp_correct = (tp_preds == tp_actions).float().sum().item()
                    metrics["teampreview_acc"] += tp_correct
                    metrics["teampreview_steps"] += tp_actions.size(0)

                    # Teampreview loss (using regular CE, not topk)
                    tp_loss = torch.nn.functional.cross_entropy(tp_logits, tp_actions)
                    metrics["teampreview_top3_loss"] += tp_loss.item() * tp_actions.size(0)

                    # Teampreview top-k accuracy
                    for k in [1, 3, 5]:
                        if tp_logits.size(-1) >= k:
                            topk_preds = torch.topk(tp_logits, k=k, dim=-1)[1]
                            tp_topk_correct = (topk_preds == tp_actions.unsqueeze(-1)).any(dim=-1).float().sum().item()
                            metrics[f"teampreview_top{k}_acc"] += tp_topk_correct
            else:
                # Original combined metrics (no separation)
                # Update action accuracy metrics
                action_preds = torch.argmax(valid_action_logits, dim=1)
                top1_correct = (action_preds == valid_actions).float().sum().item()
                metrics["action_acc"] += top1_correct

                # Calculate topk_cross_entropy_loss with k=3
                topk_loss = topk_cross_entropy_loss(valid_action_logits, valid_actions, k=3)
                metrics["top3_loss"] += topk_loss.item() * valid_actions.size(0)

                # Calculate top-k accuracy
                for k in [1, 3, 5]:
                    if valid_action_logits.size(1) < k:
                        continue
                    topk_preds = torch.topk(valid_action_logits, k=k, dim=1)[1]
                    topk_correct = (
                        (topk_preds == valid_actions.unsqueeze(1))
                        .any(dim=1)
                        .float()
                        .sum()
                        .item()
                    )
                    metrics[f"top{k}_acc"] += topk_correct

        # Update win metrics if available
        if valid_win_logits is not None and valid_wins is not None:
            # For regression task (MSE)
            win_mse = torch.nn.functional.mse_loss(
                valid_win_logits, valid_wins, reduction="sum"
            )
            metrics["win_mse"] += win_mse.item()
            metrics["win_count"] += valid_wins.size(0)

            # Also calculate correlation
            if valid_wins.size(0) > 1:
                win_corr = torch.corrcoef(torch.stack([valid_win_logits, valid_wins]))[
                    0, 1
                ]
                if not torch.isnan(win_corr):
                    metrics["win_corr"] += win_corr.item()
                    metrics["win_corr_count"] += 1

        # Move order metrics if available
        if valid_move_order_logits is not None and valid_move_orders is not None:
            move_preds = torch.argmax(valid_move_order_logits, dim=1)
            move_correct = (move_preds == valid_move_orders).float().sum().item()
            metrics["move_order_acc"] += move_correct
            metrics["move_order_count"] += valid_move_orders.size(0)

        # KO metrics if available
        if valid_ko_logits is not None and valid_kos is not None:
            ko_preds = (torch.sigmoid(valid_ko_logits) > 0.5).float()
            ko_correct = (ko_preds == valid_kos).float().sum().item()
            metrics["ko_acc"] += ko_correct
            metrics["ko_count"] += valid_kos.numel()

        # Switch metrics if available
        if valid_switch_logits is not None and valid_switches is not None:
            switch_preds = (torch.sigmoid(valid_switch_logits) > 0.5).float()
            switch_correct = (switch_preds == valid_switches).float().sum().item()
            metrics["switch_acc"] += switch_correct
            metrics["switch_count"] += valid_switches.numel()

        # Update step count
        steps += valid_states.size(0)
        metrics["steps"] = steps

    # Normalize metrics
    if steps > 0:
        metrics["action_acc"] /= steps
        for k in [1, 3, 5]:
            if metrics[f"top{k}_acc"] > 0:
                metrics[f"top{k}_acc"] /= steps

        if metrics["top3_loss"] > 0:
            metrics["top3_loss"] /= steps

    # Normalize teampreview metrics separately
    if metrics.get("teampreview_steps", 0) > 0:
        metrics["teampreview_acc"] /= metrics["teampreview_steps"]
        for k in [1, 3, 5]:
            if metrics.get(f"teampreview_top{k}_acc", 0) > 0:
                metrics[f"teampreview_top{k}_acc"] /= metrics["teampreview_steps"]
        if metrics.get("teampreview_top3_loss", 0) > 0:
            metrics["teampreview_top3_loss"] /= metrics["teampreview_steps"]

    # Normalize turn metrics separately
    if metrics.get("turn_steps", 0) > 0:
        metrics["turn_acc"] /= metrics["turn_steps"]
        for k in [1, 3, 5]:
            if metrics.get(f"turn_top{k}_acc", 0) > 0:
                metrics[f"turn_top{k}_acc"] /= metrics["turn_steps"]
        if metrics.get("turn_top3_loss", 0) > 0:
            metrics["turn_top3_loss"] /= metrics["turn_steps"]

    # Normalize other metrics
    if steps > 0:
        if metrics["win_count"] > 0:
            metrics["win_mse"] /= metrics["win_count"]

        if metrics["win_corr_count"] > 0:
            metrics["win_corr"] /= metrics["win_corr_count"]

        if metrics["move_order_count"] > 0:
            metrics["move_order_acc"] /= metrics["move_order_count"]

        if metrics["ko_count"] > 0:
            metrics["ko_acc"] /= metrics["ko_count"]

        if metrics["switch_count"] > 0:
            metrics["switch_acc"] /= metrics["switch_count"]

    return metrics


@torch.no_grad()
def analyze(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    has_action_head: bool = True,
    has_win_head: bool = True,
    has_move_order_head: bool = False,
    has_ko_head: bool = False,
    has_switch_head: bool = False,
    has_teampreview_head: bool = False,
    action_mask_fn: Optional[Callable] = None,
    teampreview_idx: Optional[int] = None,
    force_switch_indices: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze model performance across different battle characteristics.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        has_action_head: Whether model has an action head
        has_win_head: Whether model has a win prediction head
        has_teampreview_head: Whether model has separate teampreview head
        action_mask_fn: Optional function to create action masks
        teampreview_idx: Index of teampreview feature in state (required if has_teampreview_head=True)
        force_switch_indices: Indices of force switch features in state

    Returns:
        Dictionary of analysis results grouped by categories
    """
    model.eval()
    analysis_categories: Dict[str, Any] = {
        "overall": {
            "all": create_empty_metrics(),
        },
        "action_type": {
            "teampreview": create_empty_metrics(),
            "force_switch": create_empty_metrics(),
            "move": create_empty_metrics(),
            "switch": create_empty_metrics(),
            "both": create_empty_metrics(),
        },
        "phase": {
            "early": create_empty_metrics(),
            "mid": create_empty_metrics(),
            "late": create_empty_metrics(),
        },
    }

    for batch in dataloader:
        states = batch["states"].to(device).to(torch.float32)
        actions = batch["actions"].to(device)
        action_masks = (
            batch["action_masks"].to(device) if "action_masks" in batch else None
        )
        masks = batch["masks"].to(device) if "masks" in batch else None
        wins = batch["wins"].to(device).to(torch.float32) if has_win_head and "wins" in batch else None
        move_orders = batch["move_orders"].to(device) if "move_orders" in batch else None
        kos = batch["kos"].to(device) if "kos" in batch else None
        switches = batch["switches"].to(device) if "switches" in batch else None

        # Apply action masking if provided
        if action_mask_fn and action_masks is None:
            action_masks = action_mask_fn(states)

        # Get model predictions with all heads
        action_logits = teampreview_logits = win_logits = move_order_logits = ko_logits = switch_logits = None
        if has_teampreview_head:
            # Three-headed model: turn actions, teampreview, win
            action_logits, teampreview_logits, win_logits = model(states, masks)
            move_order_logits = ko_logits = switch_logits = None
        elif (
            has_win_head
            and has_action_head
            and has_move_order_head
            and has_ko_head
            and has_switch_head
        ):
            action_logits, win_logits, move_order_logits, ko_logits, switch_logits = model(
                states, masks
            )
        elif has_action_head and not has_win_head:
            action_logits = model(states, masks)
            win_logits = move_order_logits = ko_logits = switch_logits = None
        elif (
            has_action_head
            and has_win_head
            and not has_move_order_head
            and not has_ko_head
            and not has_switch_head
        ):
            action_logits, win_logits = model(states, masks)
            move_order_logits = ko_logits = switch_logits = None

        # For three-headed models, merge teampreview and turn action logits
        if has_teampreview_head and teampreview_logits is not None and action_logits is not None:
            # Identify teampreview samples
            teampreview_mask = states[:, :, teampreview_idx] == 1  # (batch, seq)

            # Start with turn logits
            merged_action_logits = action_logits.clone()

            # For teampreview samples, replace with teampreview logits
            # Zero out all positions for teampreview samples
            tp_mask_expanded = teampreview_mask.unsqueeze(-1).expand_as(merged_action_logits)
            merged_action_logits = torch.where(tp_mask_expanded, torch.tensor(float("-inf"), device=merged_action_logits.device), merged_action_logits)

            # Fill in teampreview actions [0:90] for teampreview samples
            tp_space = MDBO.teampreview_space()
            merged_action_logits[:, :, :tp_space] = torch.where(
                teampreview_mask.unsqueeze(-1).expand(-1, -1, tp_space),
                teampreview_logits,
                merged_action_logits[:, :, :tp_space]
            )

            action_logits = merged_action_logits

        # Apply action masks
        if action_masks is not None and action_logits is not None:
            masked_action_logits = action_logits.masked_fill(
                ~action_masks.bool(), float("-inf")
            )
        else:
            masked_action_logits = action_logits

        # Extract individual samples for analysis
        samples = extract_sample_keys(
            states,
            masked_action_logits,
            actions,
            win_logits,
            wins,
            action_masks,
            masks,
            move_orders,
            move_order_logits,
            kos,
            ko_logits,
            switches,
            switch_logits,
        )

        # Analyze each valid sample
        for sample in samples:
            if not sample["valid"]:
                continue

            # Get action type and phase
            action_type = determine_action_type(
                sample, teampreview_idx, force_switch_indices
            )
            phase = determine_phase(sample["state_idx"], dataloader.dataset)

            # Update overall metrics
            update_metrics(sample, analysis_categories["overall"]["all"])

            # Update action type metrics
            if action_type in analysis_categories["action_type"]:
                update_metrics(sample, analysis_categories["action_type"][action_type])

            # Update phase metrics
            if phase in analysis_categories["phase"]:
                update_metrics(sample, analysis_categories["phase"][phase])

    # Normalize metrics for all categories
    for category_dict in analysis_categories.values():
        for subcat_metrics in category_dict.values():
            normalize_metrics(subcat_metrics)

    if verbose:
        print("\n======== DETAILED ANALYSIS RESULTS ========")
        for category, category_data in analysis_categories.items():
            print(f"\n=== {category.upper()} ===")
            for subcategory, subcategory_data in category_data.items():
                if (
                    isinstance(subcategory_data, dict)
                    and subcategory_data.get("steps", 0) > 0
                ):
                    print(
                        f"  {subcategory.upper()} (Steps: {subcategory_data.get('steps', 0)})"
                    )
                    print(
                        f"    Action accuracy: {subcategory_data.get('action_acc', 0):.4f}"
                    )
                    print(f"    Win Correlation: {subcategory_data.get('win_corr', 0):.4f}")
                    if "move_order_acc" in subcategory_data and has_move_order_head:
                        print(
                            f"    Move order accuracy: {subcategory_data.get('move_order_acc', 0):.4f}"
                        )
                    if "ko_acc" in subcategory_data and has_ko_head:
                        print(f"    KO accuracy: {subcategory_data.get('ko_acc', 0):.4f}")
                    if "switch_acc" in subcategory_data and has_switch_head:
                        print(
                            f"    Switch accuracy: {subcategory_data.get('switch_acc', 0):.4f}"
                        )
        print("\n===========================================")

    return analysis_categories


def create_empty_metrics() -> Dict[str, Any]:
    """Create empty metrics dictionary with all metrics initialized to 0."""
    return {
        "steps": 0,
        "action_acc": 0.0,
        "top1_acc": 0.0,
        "top3_acc": 0.0,
        "top5_acc": 0.0,
        "top3_loss": 0.0,
        # Separate teampreview metrics
        "teampreview_steps": 0,
        "teampreview_acc": 0.0,
        "teampreview_top1_acc": 0.0,
        "teampreview_top3_acc": 0.0,
        "teampreview_top5_acc": 0.0,
        "teampreview_top3_loss": 0.0,
        # Separate turn metrics
        "turn_steps": 0,
        "turn_acc": 0.0,
        "turn_top1_acc": 0.0,
        "turn_top3_acc": 0.0,
        "turn_top5_acc": 0.0,
        "turn_top3_loss": 0.0,
        # Win metrics
        "win_mse": 0.0,
        "win_corr": 0.0,
        "win_corr_count": 0,
        "win_count": 0,
        "win_preds": [],
        "win_targets": [],
        "move_order_acc": 0.0,
        "move_order_count": 0,
        "ko_acc": 0.0,
        "ko_count": 0,
        "switch_acc": 0.0,
        "switch_count": 0,
    }


def extract_sample_keys(
    states: torch.Tensor,
    action_logits: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    win_logits: Optional[torch.Tensor] = None,
    wins: Optional[torch.Tensor] = None,
    action_masks: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    move_orders: Optional[torch.Tensor] = None,
    move_order_logits: Optional[torch.Tensor] = None,
    kos: Optional[torch.Tensor] = None,
    ko_logits: Optional[torch.Tensor] = None,
    switches: Optional[torch.Tensor] = None,
    switch_logits: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    """
    Extract individual samples with keys for easier analysis.
    Handles both [batch, seq_len, ...] and [batch, ...] tensors.
    """
    result = []

    # Define pairs of related tensors and their names
    tensor_pairs = [
        (action_logits, actions, "action_logits", "action"),
        (win_logits, wins, "win_logits", "win"),
        (move_order_logits, move_orders, "move_order_logits", "move_order"),
        (ko_logits, kos, "ko_logits", "ko"),
        (switch_logits, switches, "switch_logits", "switch"),
    ]

    # Handle different tensor dimensions
    is_trajectory = states.dim() == 3
    batch_size = states.shape[0]
    seq_len = states.shape[1] if is_trajectory else 1

    # Process each sample
    for b in range(batch_size):
        for t in range(seq_len):
            # Check validity
            if not _is_valid_sample(masks, action_masks, b, t if is_trajectory else None):
                continue

            # Create basic sample dict
            sample = {
                "valid": True,
                "batch_idx": b,
                "state_idx": t if is_trajectory else 0,
                "state": states[b, t] if is_trajectory else states[b],
            }

            # Add tensor pairs to sample
            for logits, targets, logits_name, target_name in tensor_pairs:
                if logits is not None and targets is not None:
                    if is_trajectory:
                        sample[logits_name] = logits[b, t]
                        sample[target_name] = targets[b, t]
                    else:
                        sample[logits_name] = logits[b]
                        sample[target_name] = targets[b]

            result.append(sample)

    return result


def _is_valid_sample(masks, action_masks, batch_idx, seq_idx=None):
    """Helper to check if a sample is valid based on masks."""
    # Validity from masks
    is_valid = True
    if masks is not None:
        if seq_idx is not None and masks.dim() == 2:
            is_valid = bool(masks[batch_idx, seq_idx].item())
        elif masks.dim() == 1 or seq_idx is None:
            is_valid = bool(masks[batch_idx].item())

    if not is_valid:
        return False

    # Validity from action_masks
    has_valid_actions = True
    if action_masks is not None:
        if seq_idx is not None and action_masks.dim() == 3:
            has_valid_actions = action_masks[batch_idx, seq_idx].sum().item() > 0
        elif action_masks.dim() == 2 or seq_idx is None:
            has_valid_actions = action_masks[batch_idx].sum().item() > 0

    return has_valid_actions


def determine_action_type(
    sample: Dict[str, Any],
    teampreview_idx: Optional[int] = None,
    force_switch_indices: Optional[List[int]] = None,
) -> str:
    """
    Determine the type of action for a sample.
    """
    state = sample.get("state")
    if state is None:
        return "unknown"

    # Check if it's teampreview
    if teampreview_idx is not None and state[teampreview_idx].item() == 1:
        return "teampreview"

    # Check if it's force switch
    if force_switch_indices:
        force_switch = False
        for idx in force_switch_indices:
            if idx < len(state) and state[idx].item() == 1:
                force_switch = True
                break

        if force_switch:
            return "force_switch"

    # Determine from the action
    try:
        action = sample.get("action")
        if action is None:
            return "unknown"
        action_msg = MDBO.from_int(action.item(), MDBO.TURN).message
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


def determine_phase(state_idx: int, dataset: Any) -> str:
    """
    Determine the game phase based on state index.
    """
    # Simple heuristic
    if state_idx < 5:
        return "early"
    elif state_idx < 15:
        return "mid"
    else:
        return "late"


def update_metrics(sample: Dict[str, Any], metrics: Dict[str, Any]):
    """
    Update metrics based on a single sample.
    """
    # Action accuracy
    action_pred = torch.argmax(sample["action_logits"]).item()
    action = sample["action"].item()
    metrics["steps"] += 1
    metrics["action_acc"] += 1 if action_pred == action else 0

    # Calculate loss (topk for turn actions, regular CE for teampreview)
    action_logits = sample["action_logits"].unsqueeze(0)  # Add batch dimension
    action_label = sample["action"].unsqueeze(0)  # Add batch dimension
    # For analyze, we use topk since we don't distinguish action types here
    top3_loss = topk_cross_entropy_loss(action_logits, action_label, k=3)
    metrics["top3_loss"] += top3_loss.item()

    # Top-k accuracy
    for k in [1, 3, 5]:
        if sample["action_logits"].size(0) < k:
            continue
        topk_preds = torch.topk(sample["action_logits"], k=k)[1]
        metrics[f"top{k}_acc"] += 1 if action in topk_preds else 0

    # Win metrics if available
    if "win_logits" in sample and "win" in sample:
        win_logit = sample["win_logits"].item()
        win = sample["win"].item()
        metrics["win_mse"] += (win_logit - win) ** 2
        # Store predictions and targets for correlation calculation
        if "win_preds" not in metrics:
            metrics["win_preds"] = []
            metrics["win_targets"] = []
        metrics["win_preds"].append(win_logit)
        metrics["win_targets"].append(win)
        metrics["win_count"] += 1

    # Move order metrics if available
    if "move_order_logits" in sample and "move_order" in sample:
        move_order_pred = torch.argmax(sample["move_order_logits"]).item()
        move_order = sample["move_order"].item()
        metrics["move_order_acc"] += 1 if move_order_pred == move_order else 0
        metrics["move_order_count"] += 1

    # KO metrics if available
    if "ko_logits" in sample and "ko" in sample:
        ko_preds = (torch.sigmoid(sample["ko_logits"]) > 0.5).float()
        ko = sample["ko"]
        metrics["ko_acc"] += (ko_preds == ko).float().sum().item()
        metrics["ko_count"] += ko.numel()

    # Switch metrics if available
    if "switch_logits" in sample and "switch" in sample:
        switch_preds = (torch.sigmoid(sample["switch_logits"]) > 0.5).float()
        switch = sample["switch"]
        metrics["switch_acc"] += (switch_preds == switch).float().sum().item()
        metrics["switch_count"] += switch.numel()


def normalize_metrics(metrics: Dict[str, Any]):
    """
    Normalize metrics by step count.
    """
    steps = metrics["steps"]
    if steps > 0:
        metrics["action_acc"] /= steps
        metrics["top1_acc"] /= steps
        metrics["top3_acc"] /= steps
        metrics["top5_acc"] /= steps
        metrics["top3_loss"] /= steps

    # Normalize teampreview metrics separately
    if metrics.get("teampreview_steps", 0) > 0:
        metrics["teampreview_acc"] /= metrics["teampreview_steps"]
        metrics["teampreview_top1_acc"] /= metrics["teampreview_steps"]
        metrics["teampreview_top3_acc"] /= metrics["teampreview_steps"]
        metrics["teampreview_top5_acc"] /= metrics["teampreview_steps"]
        metrics["teampreview_top3_loss"] /= metrics["teampreview_steps"]

    # Normalize turn metrics separately
    if metrics.get("turn_steps", 0) > 0:
        metrics["turn_acc"] /= metrics["turn_steps"]
        metrics["turn_top1_acc"] /= metrics["turn_steps"]
        metrics["turn_top3_acc"] /= metrics["turn_steps"]
        metrics["turn_top5_acc"] /= metrics["turn_steps"]
        metrics["turn_top3_loss"] /= metrics["turn_steps"]

    # Normalize other metrics
    if steps > 0:
        if metrics["win_count"] > 0:
            metrics["win_mse"] /= metrics["win_count"]

            # Calculate win correlation if we have predictions
            if len(metrics["win_preds"]) > 1:
                win_preds_tensor = torch.tensor(metrics["win_preds"])
                win_targets_tensor = torch.tensor(metrics["win_targets"])
                win_corr = torch.corrcoef(torch.stack([win_preds_tensor, win_targets_tensor]))[0, 1]
                if not torch.isnan(win_corr):
                    metrics["win_corr"] = win_corr.item()
                else:
                    metrics["win_corr"] = 0.0
            else:
                metrics["win_corr"] = 0.0

        if metrics["move_order_count"] > 0:
            metrics["move_order_acc"] /= metrics["move_order_count"]

        if metrics["ko_count"] > 0:
            metrics["ko_acc"] /= metrics["ko_count"]

        if metrics["switch_count"] > 0:
            metrics["switch_acc"] /= metrics["switch_count"]


def print_analysis_results(
    results: Dict[str, Any],  # Using Any for flexibility with nested dictionaries
    selected_metrics: Optional[List[str]] = None,
) -> None:
    """
    Print analysis results in a readable format.

    Args:
        results: Dictionary of analysis results, potentially nested
        selected_metrics: List of metric names to display
    """
    if selected_metrics is None:
        selected_metrics = [
            "action_acc",
            "top3_acc",
            "top3_loss",
            "win_mse",
            "move_order_acc",
            "ko_acc",
            "switch_acc",
        ]

    for category, data in results.items():
        print(f"\n=== {category.upper()} ===")

        # Check if data is a nested dictionary of metrics
        if isinstance(data, dict):
            # Check if this is a metrics dict or a dict of metrics dicts
            if any(metric in data for metric in selected_metrics):
                # This is a metrics dict - handle directly
                if "steps" not in data or data.get("steps", 0) == 0:
                    continue

                print(f"  ({data.get('steps', 0)} steps):")
                _print_metrics(data, selected_metrics)
            else:
                # This is a dict of metrics dicts
                for subcategory, subdata in data.items():
                    if isinstance(subdata, dict):
                        if "steps" not in subdata or subdata.get("steps", 0) == 0:
                            continue

                        print(f"  {subcategory} ({subdata.get('steps', 0)} steps):")
                        _print_metrics(subdata, selected_metrics)


def _print_metrics(metrics_dict: Dict[str, Any], selected_metrics: List[str]) -> None:
    """Helper function to print metrics with appropriate formatting"""
    metrics_str = []
    for metric in selected_metrics:
        if metric in metrics_dict:
            value = metrics_dict[metric]
            if isinstance(value, (int, float)):
                if "acc" in metric:
                    metrics_str.append(f"{metric}: {value:.3f}")
                else:
                    metrics_str.append(f"{metric}: {value:.5f}")

    print(f"    {', '.join(metrics_str)}")


def format_time(seconds: float) -> str:
    """Format seconds into hours, minutes, seconds string."""
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    secs = int(seconds % 60)

    return f"{hours}h {minutes}m {secs}s"


def save_compressed(obj, out_path, level=3):
    buf = io.BytesIO()
    torch.save(obj, buf)
    compressed = zstd.compress(buf.getvalue(), level)
    with open(out_path, "wb") as f:
        f.write(compressed)


def load_compressed(in_path, map_location="cpu"):
    with open(in_path, "rb") as f:
        compressed = f.read()
    raw = zstd.decompress(compressed)
    return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=False)
