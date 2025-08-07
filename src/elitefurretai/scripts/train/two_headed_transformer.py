# -*- coding: utf-8 -*-
"""This script trains a supervised model for move and winprediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import random
import sys
import time
from collections import defaultdict

import torch

import wandb
from elitefurretai.model_utils import MDBO, Embedder, PreprocessedTrajectoryDataset


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)  # Add ReLU after addition


class TwoHeadedHybridModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=[1024, 512, 256],
        num_heads=4,
        num_lstm_layers=1,
        num_actions=MDBO.action_space(),
        max_seq_len=17,
        dropout=0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_layers[-1]
        self.num_actions = num_actions

        # Feedforward stack with residual blocks
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.ff_stack = torch.nn.Sequential(*layers)

        # Positional encoding (learned) for the final hidden size
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Multihead Self-Attention block
        self.self_attn = torch.nn.MultiheadAttention(
            self.hidden_size, num_heads, batch_first=True
        )

        # Normalize outputs
        self.norm = torch.nn.LayerNorm(self.hidden_size)

        # Output heads
        self.action_head = torch.nn.Linear(self.hidden_size, num_actions)
        self.win_head = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Feedforward stack with residuals
        x = self.ff_stack(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = (
            x + self.pos_embedding(positions) * mask.unsqueeze(-1)
            if mask is not None
            else x + self.pos_embedding(positions)
        )

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM (packed)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        lstm_out = self.lstm_proj(lstm_out)

        # Multihead Self-Attention
        attn_mask = ~mask.bool()
        attn_out, _ = self.self_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
        )
        out = self.norm(attn_out + lstm_out)

        # *** REMOVE POOLING ***
        # Output heads: now per-step
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len)

        return action_logits, win_logits

    def predict(self, x, mask=None):
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            win_prob = torch.sigmoid(win_logits)
        return action_probs, win_prob


def topk_cross_entropy_loss(logits, targets, weights=None, k=1):
    """
    Returns 0 loss if target is in top-k predictions, else cross-entropy loss.
    logits: [batch, num_classes]
    targets: [batch] (long)
    """
    # Get top-k indices
    topk = torch.topk(logits, k=k, dim=1).indices  # [batch, k]
    # Check if target is in top-k
    in_topk = (topk == targets.unsqueeze(1)).any(dim=1)  # [batch]
    # Compute standard cross-entropy
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none") / 7.0
    # Only penalize if not in top-k
    loss = ce_loss * (~in_topk)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def train_epoch(model, dataloader, prev_steps, optimizer, config):
    model.train()
    running_loss = 0
    running_action_loss = 0
    running_win_loss = 0
    running_win_acc = 0
    steps = 0
    num_batches = 0
    start = time.time()

    for batch in dataloader:
        states, actions, action_masks, wins, masks = batch
        states = states.to(config["device"])
        actions = actions.to(config["device"])
        action_masks = action_masks.to(config["device"])
        wins = wins.to(config["device"])
        masks = masks.to(config["device"])

        # Forward pass
        action_logits, win_logits = model(states, masks)
        masked_action_logits = action_logits.masked_fill(
            ~action_masks.bool(), float("-inf")
        )

        # Use helper for flattening and filtering
        flat_data = flatten_and_filter(
            states, masked_action_logits, actions, win_logits, wins, action_masks, masks
        )
        if flat_data is None:
            continue

        valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
            flat_data
        )

        # Build weights for loss
        weights = None
        if config.get("weights", False):
            weights = torch.ones(valid_states.shape[0], device=states.device)
            teampreview_mask = valid_states[:, config["teampreview_idx"]] == 1
            force_switch_mask = torch.stack(
                [valid_states[:, idx] == 1 for idx in config["force_switch_indices"]],
                dim=-1,
            ).any(dim=-1)
            turn_mask = ~(teampreview_mask | force_switch_mask)
            weights[teampreview_mask] = 8.55
            weights[force_switch_mask] = 125
            weights[turn_mask] = 1.14

        # Losses
        action_loss = topk_cross_entropy_loss(
            valid_action_logits, valid_actions, weights=weights, k=3
        )
        win_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_win_logits, valid_wins.float()
        )
        loss = config["action_weight"] * action_loss + config["win_weight"] * win_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_action_loss += action_loss.item()
        running_win_loss += win_loss.item()
        win_preds = (torch.sigmoid(valid_win_logits) > 0.5).float()
        running_win_acc += (win_preds == valid_wins).float().sum().item()
        steps += valid_actions.size(0)
        num_batches += 1

        # Logging progress
        wandb.log(
            {
                "steps": prev_steps + steps,
                "train_loss": running_loss / num_batches,
                "train_action_loss": running_action_loss / num_batches,
                "train_win_loss": running_win_loss / num_batches,
                "train_win_acc": running_win_acc / steps if steps > 0 else 0,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Print progress
        hours = int(time.time() - start) // 3600
        minutes = (int(time.time() - start) % 3600) // 60
        seconds = int(time.time() - start) % 60
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        est_time_left = (len(dataloader) - num_batches) * time_per_batch
        hours_left = int(est_time_left // 3600)
        minutes_left = int((est_time_left % 3600) // 60)
        seconds_left = int(est_time_left % 60)
        processed = f"Processed {num_batches * dataloader.batch_size} battles/trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {hours}h {minutes}m {seconds}s"
        left = f" with an estimated {hours_left}h {minutes_left}m {seconds_left}s left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    hours = int(time.time() - start) // 3600
    minutes = (int(time.time() - start) % 3600) // 60
    seconds = int(time.time() - start) % 60
    print(
        "\033[2K\rDone training in "
        + str(hours)
        + "h "
        + str(minutes)
        + "m "
        + str(seconds)
        + "s!"
    )

    return (
        running_loss / num_batches,
        steps,
        running_action_loss / num_batches,
        running_win_loss / num_batches,
    )


@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_samples = 0
    running_action_loss = 0
    running_win_loss = 0
    running_win_acc = 0

    for batch in dataloader:
        states, actions, action_masks, wins, masks = batch
        states = states.to(config["device"])
        actions = actions.to(config["device"])
        action_masks = action_masks.to(config["device"])
        wins = wins.to(config["device"])
        masks = masks.to(config["device"])

        # Forward pass
        action_logits, win_logits = model(states, masks)
        masked_action_logits = action_logits.masked_fill(
            ~action_masks.bool(), float("-inf")
        )

        # Use helper for flattening and filtering
        flat_data = flatten_and_filter(
            states, masked_action_logits, actions, win_logits, wins, action_masks, masks
        )
        if flat_data is None:
            continue

        valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
            flat_data
        )

        top1_preds = torch.argmax(valid_action_logits, dim=1)
        top3_preds = torch.topk(valid_action_logits, k=3, dim=1).indices
        top5_preds = torch.topk(valid_action_logits, k=5, dim=1).indices

        top1_correct = top1_preds == valid_actions
        top3_correct = (valid_actions.unsqueeze(1) == top3_preds).any(dim=1)
        top5_correct = (valid_actions.unsqueeze(1) == top5_preds).any(dim=1)

        win_preds = (torch.sigmoid(valid_win_logits) > 0.5).float()
        win_acc = (win_preds == valid_wins).float().sum().item()

        action_loss = topk_cross_entropy_loss(valid_action_logits, valid_actions)
        win_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_win_logits, valid_wins.float()
        )

        total_top1 += top1_correct.sum().item()
        total_top3 += top3_correct.sum().item()
        total_top5 += top5_correct.sum().item()
        total_samples += valid_actions.size(0)
        running_action_loss += action_loss.item()
        running_win_loss += win_loss.item()
        running_win_acc += win_acc

    action_top1 = total_top1 / total_samples if total_samples > 0 else 0
    action_top3 = total_top3 / total_samples if total_samples > 0 else 0
    action_top5 = total_top5 / total_samples if total_samples > 0 else 0
    avg_action_loss = running_action_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_win_loss = running_win_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_win_acc = running_win_acc / total_samples if total_samples > 0 else 0
    return (
        action_top1,
        action_top3,
        action_top5,
        avg_action_loss,
        avg_win_acc,
        avg_win_loss,
    )


@torch.no_grad()
def analyze(model, dataloader, feature_names, config):
    """Analyze model performance across different battle characteristics."""
    model.eval()

    # Initialize analysis categories
    analysis_categories = {
        "turns": defaultdict(lambda: create_empty_metrics()),
        "ko_opportunities": defaultdict(lambda: create_empty_metrics()),
        "mons_alive": defaultdict(lambda: create_empty_metrics()),
        "action_types": defaultdict(lambda: create_empty_metrics()),
    }

    # Create feature name lookup and identify KO features
    feature_idx = {name: i for i, name in enumerate(feature_names)}
    ko_features = {v for k, v in feature_idx.items() if "KO" in k}

    print("Starting model analysis...")

    for batch in dataloader:
        # Process batch and get predictions
        predictions = process_batch(model, batch, config)
        if predictions is None:
            continue

        states, valid_actions, top1_correct, top3_correct, top5_correct, win_correct = (
            predictions
        )

        # Analyze each valid sample
        for i, state in enumerate(states):
            sample_keys = extract_sample_keys(
                state, valid_actions[i], feature_idx, ko_features
            )
            sample_metrics = [
                top1_correct[i].item(),
                top3_correct[i].item(),
                top5_correct[i].item(),
                win_correct[i].item(),
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
                for metric_name, value in zip(metric_names, sample_metrics):
                    analysis_categories[category][key][metric_name] += value

    # Print results
    print_analysis_results(analysis_categories)


def create_empty_metrics():
    """Create empty metrics dictionary."""
    return {
        "total_top1": 0,
        "total_top3": 0,
        "total_top5": 0,
        "win": 0,
        "total_samples": 0,
    }


def process_batch(model, batch, config):
    """Process a single batch and return predictions."""
    states, actions, action_masks, wins, masks = batch

    # Move to device
    states = states.to(config["device"])
    actions = actions.to(config["device"])
    action_masks = action_masks.to(config["device"])
    wins = wins.to(config["device"])
    masks = masks.to(config["device"])

    # Forward pass
    action_logits, win_logits = model(states, masks)
    masked_action_logits = action_logits.masked_fill(~action_masks.bool(), float("-inf"))

    # Flatten and filter valid samples
    flat_data = flatten_and_filter(
        states, masked_action_logits, actions, win_logits, wins, action_masks, masks
    )

    if flat_data is None:
        return None

    valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
        flat_data
    )

    # Calculate predictions and correctness
    top1_preds = torch.argmax(valid_action_logits, dim=1)
    top3_preds = torch.topk(valid_action_logits, k=3, dim=1).indices
    top5_preds = torch.topk(valid_action_logits, k=5, dim=1).indices
    win_preds = (torch.sigmoid(valid_win_logits) > 0.5).float()

    top1_correct = top1_preds == valid_actions
    top3_correct = (valid_actions.unsqueeze(1) == top3_preds).any(dim=1)
    top5_correct = (valid_actions.unsqueeze(1) == top5_preds).any(dim=1)
    win_correct = (win_preds == valid_wins).float()

    return (
        valid_states,
        valid_actions,
        top1_correct,
        top3_correct,
        top5_correct,
        win_correct,
    )


def flatten_and_filter(
    states, action_logits, actions, win_logits, wins, action_masks, masks
):
    """Flatten batch dimensions and filter for valid samples."""
    # Flatten all tensors
    flat_states = states.view(-1, states.size(-1))
    flat_mask = masks.bool().view(-1)
    flat_action_logits = action_logits.view(-1, action_logits.size(-1))
    flat_actions = actions.view(-1)
    flat_win_logits = win_logits.view(-1)
    flat_wins = wins.view(-1)
    flat_action_masks = action_masks.view(-1, action_masks.size(-1))

    # Filter for valid (unpadded) positions
    valid_idx = flat_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return None

    # Apply valid indices
    valid_states = flat_states[valid_idx]
    valid_action_logits = flat_action_logits[valid_idx]
    valid_actions = flat_actions[valid_idx]
    valid_win_logits = flat_win_logits[valid_idx]
    valid_wins = flat_wins[valid_idx]
    valid_action_masks = flat_action_masks[valid_idx]

    # Filter for samples with at least one valid action
    valid_action_rows = valid_action_masks.sum(dim=1) > 0
    if valid_action_rows.sum() == 0:
        return None

    return (
        valid_states[valid_action_rows],
        valid_action_logits[valid_action_rows],
        valid_actions[valid_action_rows],
        valid_win_logits[valid_action_rows],
        valid_wins[valid_action_rows],
    )


def extract_sample_keys(state, action, feature_idx, ko_features):
    """Extract categorization keys for a single sample."""
    # Basic features
    turn = state[feature_idx["turn"]].item()
    can_ko = max(state[feature_idx] for feature_idx in ko_features).item()
    num_alive = int(
        8
        - state[feature_idx["OPP_NUM_FAINTED"]].item()
        - state[feature_idx["NUM_FAINTED"]].item()
    )

    # Determine action type
    action_type = determine_action_type(state, action, feature_idx)

    return {
        "turns": turn,
        "ko_opportunities": can_ko,
        "mons_alive": num_alive,
        "action_types": action_type,
    }


def determine_action_type(state, action, feature_idx):
    """Determine the type of action being taken."""
    if int(state[feature_idx["teampreview"]].item()) == 1:
        return "teampreview"

    # Check for force switch
    if any(int(state[feature_idx[f"MON:{j}:force_switch"]].item()) == 1 for j in range(6)):
        return "force_switch"

    # Parse action message
    action_msg = MDBO.from_int(int(action), MDBO.TURN).message
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


def print_analysis_results(analysis_categories):
    """Print formatted analysis results."""
    print("Analysis complete! Results:")

    category_display_names = {
        "turns": "Turn",
        "action_types": "Action Types",
        "ko_opportunities": "KO Opportunities",
        "mons_alive": "Pokemon Alive",
    }

    for category, display_name in category_display_names.items():
        results = analysis_categories[category]
        print(f"\nAnalysis by {display_name}:")

        for key in sorted(results.keys()):
            metrics = results[key]
            if metrics["total_samples"] > 0:
                top1_acc = metrics["total_top1"] / metrics["total_samples"] * 100
                top3_acc = metrics["total_top3"] / metrics["total_samples"] * 100
                top5_acc = metrics["total_top5"] / metrics["total_samples"] * 100
                win_acc = metrics["win"] / metrics["total_samples"] * 100

                print(
                    f"  {key}: Top-1: {top1_acc:.1f}%, Top-3: {top3_acc:.1f}%, "
                    f"Top-5: {top5_acc:.1f}%, Win: {win_acc:.1f}% (n={metrics['total_samples']})"
                )
            else:
                print(f"  {key}: No samples")


def main(train_path, test_path, val_path):
    print("Starting!")
    config = {
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 20,
        "hidden_layers": [1024, 512, 256],
        "num_heads": 4,
        "num_lstm_layers": 2,
        "action_weight": 1.0,
        "win_weight": 0.0,
        "max_grad_norm": 1.0,
        "device": (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "save_path": "data/models/",
        "seed": 21,
    }
    wandb.init(project="elitefurretai-scovillain", config=config)
    wandb.save(__file__)

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    random.seed(config["seed"])

    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print("Embedder initialized. Embedding size:", embedder.embedding_size)

    print("Loading datasets...")
    start = time.time()

    train_dataset = PreprocessedTrajectoryDataset(train_path, embedder=embedder)
    test_dataset = PreprocessedTrajectoryDataset(test_path, embedder=embedder)
    val_dataset = PreprocessedTrajectoryDataset(val_path, embedder=embedder)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Initialize model
    model = TwoHeadedHybridModel(
        input_size=embedder.embedding_size,
        hidden_layers=config["hidden_layers"],
        num_heads=config["num_heads"],
        num_lstm_layers=config["num_lstm_layers"],
        num_actions=MDBO.action_space(),
    ).to(config["device"])
    wandb.watch(model, log="all", log_freq=1000)
    print(
        f"Finished loading data and model! for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    print("Initialized model! Starting training...")

    # Training loop
    start, steps = time.time(), 0
    for epoch in range(config["num_epochs"]):
        train_loss, training_steps, train_action_loss, train_win_loss = train_epoch(
            model, train_loader, steps, optimizer, config
        )
        action_top1, action_top3, action_top5, action_loss, win, win_loss = evaluate(
            model, test_loader, config
        )
        steps += training_steps

        print(f"Epoch #{epoch + 1}:")
        print(f"=> Total Steps:       {steps}")
        print(f"=> Train Loss:        {train_loss:.4f}")
        print(f"=> Train Win Loss:    {train_win_loss:.4f}")
        print(f"=> Train Action Loss: {train_action_loss:.4f}")
        print(
            f"=> Test Loss:         {(win_loss * config['win_weight'] + action_loss * config['action_weight']):.4f}"
        )
        print(f"=> Test Win Acc:      {win * 100:.3f}%")
        print(f"=> Test Win Loss:     {win_loss:.4f}")
        print(f"=> Test Action Loss:  {action_loss:.4f}")
        print(
            f"=> Test Action Acc:   {(action_top1 * 100):.3f}% (Top-1), {(action_top3 * 100):.3f}% (Top-3) {(action_top5 * 100):.3f}% (Top-5)"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "steps": steps,
                "train_loss": train_loss,
                "train_win_loss": train_win_loss,
                "train_action_loss": train_action_loss,
                "test_loss": win_loss + action_loss,
                "test_action_loss": action_loss,
                "test_win_loss": win_loss,
                "test_win_acc": win,
                "test_action_top1": action_top1,
                "test_action_top3": action_top3,
                "test_action_top5": action_top5,
            }
        )

        total_time = time.time() - start
        h, m, s = (
            int(total_time / (60 * 60)),
            int((total_time % (60 * 60)) / 60),
            int(total_time % 60),
        )
        t_left = (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        h_left, m_left, s_left = (
            int(t_left / (60 * 60)),
            int((t_left % (60 * 60)) / 60),
            int(t_left % 60),
        )
        print(f"=> Time thus far: {h}h {m}m {s}s // ETA: {h_left}h {m_left}m {s_left}s")
        print()

        scheduler.step(win_loss + action_loss)

    torch.save(
        model.state_dict(),
        os.path.join(config["save_path"], f"{wandb.run.name}.pth"),  # type: ignore
    )
    print("\nEvaluating on Validation Dataset:")
    action_top1, action_top3, action_top5, action_loss, win, win_loss = evaluate(
        model, val_loader, config
    )
    print(
        f"=> Val Loss:          {(win_loss * config['win_weight'] + action_loss * config['action_weight']):.3f}"
    )
    print(f"=> Val Win Loss:      {win_loss:.3f}")
    print(f"=> Val Action Loss:   {action_loss:.3f}")
    print(f"=> Val Win Acc:       {win:.3f}")
    print(
        f"=> Val Action Acc:    {(action_top1 * 100):.3f}% (Top-1), {(action_top3 * 100):.3f}% (Top-3) {(action_top5 * 100):.3f}% (Top-5)"
    )
    wandb.log(
        {
            "val_loss": win_loss * config["win_weight"]
            + action_loss * config["action_weight"],
            "val_win_loss": win_loss,
            "val_win_acc": win,
            "val_action_loss": action_loss,
            "val_action_top1": action_top1,
            "val_action_top3": action_top3,
            "val_action_top5": action_top5,
        }
    )
    analyze(model, val_loader, embedder.feature_names, config)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
