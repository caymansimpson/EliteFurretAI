# -*- coding: utf-8 -*-
"""
This script trains a supervised model for move and win prediction using a flexible architecture.
It supports hyperparameter sweeping through Weights & Biases to find optimal configurations.

Usage:
    python two_headed_sweep.py <train_path> <test_path> <val_path> [--sweep]

Options:
    --sweep    Run as a sweep instead of a single training run
"""
import argparse
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import wandb
from elitefurretai.model_utils import MDBO, Embedder, PreprocessedTrajectoryDataset
from elitefurretai.scripts.train.train_utils import (
    evaluate,
    flatten_and_filter,
    format_time,
    topk_cross_entropy_loss,
)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        # Main path
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.ln1 = torch.nn.LayerNorm(out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)
        self.ln2 = torch.nn.LayerNorm(out_features)

        # Gate generation
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features), torch.nn.Sigmoid()
        )

        # Regularization
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Projection for residual if dimensions change
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # Main path
        main = self.linear1(x)
        main = self.ln1(main)
        main = self.relu(main)
        main = self.dropout(main)
        main = self.linear2(main)
        main = self.ln2(main)

        # Apply gate
        gate_value = self.gate(x)
        gated_output = gate_value * main

        return self.relu(residual + gated_output)


class FlexibleTwoHeadedModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        early_layers: List[int],
        late_layers: List[int],
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        num_actions: int = MDBO.action_space(),
        max_seq_len: int = 17,
        dropout: float = 0.1,
        gated_residuals: bool = False,
        early_attention: bool = False,
        late_attention: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.hidden_size = early_layers[-1] if early_layers else input_size
        self.num_actions = num_actions
        self.early_attention = early_attention
        self.late_attention = late_attention

        # Select residual block type
        ResBlock = GatedResidualBlock if gated_residuals else ResidualBlock

        # Build early feedforward stack with residual blocks
        early_ff_layers = []
        prev_size = input_size
        for h in early_layers:
            early_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.early_ff_stack = (
            torch.nn.Sequential(*early_ff_layers)
            if early_ff_layers
            else torch.nn.Identity()
        )

        # Early attention if enabled
        if early_attention:
            self.early_attn = torch.nn.MultiheadAttention(
                self.hidden_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.early_ln = torch.nn.LayerNorm(self.hidden_size)

        # Positional encoding (learned)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_proj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Late attention if enabled
        if late_attention:
            self.late_attn = torch.nn.MultiheadAttention(
                self.hidden_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln = torch.nn.LayerNorm(self.hidden_size)

        # Build late feedforward stack with residual blocks
        late_ff_layers = []
        prev_size = self.hidden_size
        for h in late_layers:
            late_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers) if late_ff_layers else torch.nn.Identity()
        )

        # Output size after late layers
        output_size = late_layers[-1] if late_layers else self.hidden_size

        # Output heads
        self.action_head = torch.nn.Linear(output_size, num_actions)
        self.win_head = torch.nn.Linear(output_size, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Early feedforward stack
        x = self.early_ff_stack(x)

        # Early attention if enabled
        if self.early_attention:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(x, x, x, key_padding_mask=attn_mask)
            x = self.early_ln(x + attn_out)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x = x + self.pos_embedding(positions)

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

        # Late attention if enabled
        if self.late_attention:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.late_attn(
                lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
            )
            out = self.late_ln(lstm_out + attn_out)
        else:
            out = lstm_out

        # Late feedforward stack
        out = self.late_ff_stack(out)

        # Output heads
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len)

        return action_logits, win_logits

    def predict(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            win_prob = torch.sigmoid(win_logits)
        return action_probs, win_prob


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    prev_steps: int,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Tuple[float, int, float, float]:
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_win_loss = 0.0
    running_win_acc = 0.0
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
        time_taken = format_time(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        time_left = format_time((len(dataloader) - num_batches) * time_per_batch)
        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1
        processed = f"Processed {num_batches * batch_size} battles/trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {time_taken}"
        left = f" with an estimated {time_left} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    time_taken = format_time(time.time() - start)
    print("\033[2K\rDone training in " + time_taken)

    return (
        running_loss / num_batches if num_batches > 0 else 0,
        steps,
        running_action_loss / num_batches if num_batches > 0 else 0,
        running_win_loss / num_batches if num_batches > 0 else 0,
    )


def train_model(
    config: Dict[str, Any], train_path: str, test_path: str, val_path: str
) -> Dict[str, float]:
    """Train a model with the given configuration and return metrics."""
    print(f"Starting training with config: {config}")

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    random.seed(config["seed"])

    # Initialize embedder
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )
    feature_names = embedder.feature_names
    config["teampreview_idx"] = (
        feature_names.index("teampreview") if "teampreview" in feature_names else 0
    )
    config["force_switch_indices"] = [
        i
        for i, name in enumerate(feature_names)
        if name.startswith("MON:") and name.endswith(":force_switch")
    ]
    print(f"Embedder initialized. Embedding size: {embedder.embedding_size}")

    # Load datasets
    print("Loading datasets...")
    start = time.time()

    train_dataset = PreprocessedTrajectoryDataset(train_path, embedder=embedder)
    test_dataset = PreprocessedTrajectoryDataset(test_path, embedder=embedder)
    val_dataset = PreprocessedTrajectoryDataset(val_path, embedder=embedder)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Initialize model
    model = FlexibleTwoHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        num_attention_heads=config["num_attention_heads"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        gated_residuals=config["gated_residuals"],
        early_attention=config["early_attention"],
        late_attention=config["late_attention"],
        num_actions=MDBO.action_space(),
    ).to(config["device"])

    # Log model size
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params} trainable parameters")
    wandb.log({"model_parameters": trainable_params})

    # Initialize optimizer
    optimizer_class = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }[config["optimizer"]]

    optimizer = optimizer_class(
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
        action_top1, action_top3, action_top5, action_loss, win_acc, win_loss = evaluate(
            model, test_loader, config["device"], has_win_head=True
        )
        steps += training_steps

        # Print epoch results
        print(f"Epoch #{epoch + 1}:")
        print(f"=> Total Steps:       {steps}")
        print(f"=> Train Loss:        {train_loss:.4f}")
        print(f"=> Train Win Loss:    {train_win_loss:.4f}")
        print(f"=> Train Action Loss: {train_action_loss:.4f}")
        print(
            f"=> Test Loss:         {(win_loss * config['win_weight'] + action_loss * config['action_weight']):.4f}"
        )
        print(f"=> Test Win Acc:      {win_acc * 100:.3f}%")
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
                "test_loss": win_loss * config["win_weight"]
                + action_loss * config["action_weight"],
                "test_action_loss": action_loss,
                "test_win_loss": win_loss,
                "test_win_acc": win_acc,
                "test_action_top1": action_top1,
                "test_action_top3": action_top3,
                "test_action_top5": action_top5,
            }
        )

        # Print time progress
        total_time = time.time() - start
        t_left = (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        print(f"=> Time thus far: {format_time(total_time)} // ETA: {format_time(t_left)}")
        print()

        scheduler.step(
            win_loss * config["win_weight"] + action_loss * config["action_weight"]
        )

    # Final evaluation on validation set
    print("\nEvaluating on Validation Dataset:")
    (
        val_action_top1,
        val_action_top3,
        val_action_top5,
        val_action_loss,
        val_win_acc,
        val_win_loss,
    ) = evaluate(model, val_loader, config["device"], has_win_head=True)

    print(
        f"=> Val Loss:          {(val_win_loss * config['win_weight'] + val_action_loss * config['action_weight']):.3f}"
    )
    print(f"=> Val Win Loss:      {val_win_loss:.3f}")
    print(f"=> Val Action Loss:   {val_action_loss:.3f}")
    print(f"=> Val Win Acc:       {val_win_acc:.3f}")
    print(
        f"=> Val Action Acc:    {(val_action_top1 * 100):.3f}% (Top-1), {(val_action_top3 * 100):.3f}% (Top-3) {(val_action_top5 * 100):.3f}% (Top-5)"
    )

    wandb.log(
        {
            "val_loss": val_win_loss * config["win_weight"]
            + val_action_loss * config["action_weight"],
            "val_win_loss": val_win_loss,
            "val_win_acc": val_win_acc,
            "val_action_loss": val_action_loss,
            "val_action_top1": val_action_top1,
            "val_action_top3": val_action_top3,
            "val_action_top5": val_action_top5,
        }
    )

    # Return metrics for the sweep
    return {
        "val_action_top1": float(val_action_top1),
        "val_action_top3": float(val_action_top3),
        "val_action_top5": float(val_action_top5),
        "val_win_acc": float(val_win_acc),
        "val_loss": float(
            val_win_loss * config["win_weight"] + val_action_loss * config["action_weight"]
        ),
    }


def sweep_agent(train_path: str, test_path: str, val_path: str) -> None:
    """WandB sweep agent function."""
    # Initialize sweep configuration
    sweep_config = {
        "method": "bayes",
        "name": "action_sweep",
        "metric": {
            "goal": "maximize",
            "name": "val_action_top3",
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
        },
        "parameters": {
            "batch_size": {"values": [256, 1024, 2048, 4096]},
            "optimizer": {"values": ["Adam", "AdamW"]},
            "learning_rate": {"values": [0.001, 0.0005, 0.0001]},
            "dropout": {"values": [0.1, 0.3, 0.5]},
            "gated_residuals": {"values": [True, False]},
            "num_attention_heads": {"values": [2, 4, 8]},
            "early_attention": {"values": [True, False]},
            "early_layers": {
                "values": [
                    [2048, 1024, 512],
                    [1024, 512],
                ]
            },
            "lstm_layers": {"values": [2, 4, 8]},
            "late_attention": {"values": [True, False]},
            "late_layers": {
                "values": [
                    [256, 128],
                    [256],
                    [],
                ]
            },
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="elitefurretai-scovillain")

    # Define the training function
    def train_sweep():
        # Initialize a new run
        wandb.init()

        # Get hyperparameters
        config = wandb.config

        # Add fixed parameters
        config.update(
            {
                "num_epochs": 20,
                "action_weight": 1.0,
                "win_weight": 0.0,
                "max_grad_norm": 1.0,
                "weight_decay": 1e-5,
                "device": (
                    "mps"
                    if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                ),
                "save_path": "data/models/",
                "seed": 21,
                "weights": False,
            }
        )

        # Train the model with this config
        return train_model(config, train_path, test_path, val_path)  # type: ignore

    # Run the agent
    wandb.agent(sweep_id, function=train_sweep)


def main():
    parser = argparse.ArgumentParser(
        description="Train a two-headed hybrid model for Pokemon battle prediction."
    )
    parser.add_argument("train_path", type=str, help="Path to training data")
    parser.add_argument("test_path", type=str, help="Path to testing data")
    parser.add_argument("val_path", type=str, help="Path to validation data")

    args = parser.parse_args()
    sweep_agent(args.train_path, args.test_path, args.val_path)


if __name__ == "__main__":
    main()
