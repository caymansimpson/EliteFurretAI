# -*- coding: utf-8 -*-
"""This script trains a supervised model for move and winprediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import random
import sys
import time
from typing import Any, Dict

import torch

import wandb
from elitefurretai.model_utils import MDBO, Embedder, OptimizedPreprocessedTrajectoryDataset, OptimizedPreprocessedTrajectorySampler
from elitefurretai.model_utils.train_utils import (
    analyze,
    evaluate,
    flatten_and_filter,
    format_time,
    topk_cross_entropy_loss,
)


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
        max_seq_len=40,
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


def train_epoch(model, dataloader, prev_steps, optimizer, config):
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_win_loss = 0.0
    steps = 0
    num_batches = 0
    start = time.time()
    scaler = torch.amp.GradScaler(config['device']) if config['device'] == 'cuda' else None

    for batch in dataloader:
        # Get data from dictionary
        states = batch["states"].to(config["device"])
        actions = batch["actions"].to(config["device"])
        action_masks = batch["action_masks"].to(config["device"])
        wins = batch["wins"].to(config["device"])
        masks = batch["masks"].to(config["device"])

        autocast = torch.amp.autocast if config['device'] == 'cuda' else torch.autocast
        with autocast(config['device']):

            # Forward pass
            action_logits, win_logits = model(states, masks)
            masked_action_logits = action_logits.masked_fill(
                ~action_masks.bool(), float("-inf")
            )

            # Use helper for flattening and filtering
            flat_data = flatten_and_filter(
                states=states,
                action_logits=masked_action_logits,
                actions=actions,
                win_logits=win_logits,
                wins=wins,
                action_masks=action_masks,
                masks=masks,
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
            win_loss = torch.nn.functional.mse_loss(valid_win_logits, valid_wins.float())
            loss = config["action_weight"] * action_loss + config["win_weight"] * win_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_action_loss += action_loss.item()
        running_win_loss += win_loss.item()

        steps += valid_actions.size(0)
        num_batches += 1

        # Logging progress
        wandb.log(
            {
                "steps": prev_steps + steps,
                "train_loss": running_loss / num_batches,
                "train_action_loss": running_action_loss / num_batches,
                "train_win_loss": running_win_loss / num_batches,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Print progress
        time_taken = format_time(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        time_left = format_time((len(dataloader) - num_batches) * time_per_batch)
        processed = f"Processed {num_batches * dataloader.batch_size} battles/trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {time_taken}"
        left = f" with an estimated {time_left} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    time_taken = format_time(time.time() - start)
    print("\033[2K\rDone training in " + time_taken)

    return {
        "loss": running_loss / num_batches,
        "steps": steps,
        "action_loss": running_action_loss / num_batches,
        "win_loss": running_win_loss / num_batches,
    }


def initialize(config):
    # Wandb defaults
    wandb.init(
        project="elitefurretai-scovillain-gpu",
        config=config,
        settings=wandb.Settings(
            x_service_wait=30,  # Increase service wait time
            start_method="thread"  # Use thread instead of fork
        )
    )
    try:
        # Try normal symlink first (fast)
        wandb.save(__file__)
    except OSError as e:
        if "WinError 1314" in str(e) or "privilege" in str(e).lower():
            try:
                # Fallback to copy method
                wandb.save(__file__, policy="now")
                print("Note: Using file copy instead of symlink for wandb")
            except Exception as copy_error:
                print(f"Warning: Could not save script to wandb: {copy_error}")
        else:
            raise  # Re-raise if it's a different error

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))


def main(train_path, test_path, val_path):

    print("Starting!")
    config: Dict[str, Any] = {
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
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        "save_path": "data/models/",
        "seed": 21,
    }
    initialize(config)

    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print(f"Embedder initialized. Embedding[{embedder.embedding_size}] on {config['device']}")

    print("Loading datasets...")
    start = time.time()

    num_workers = 2  # min(4, os.cpu_count() or 0)
    train_dataset = OptimizedPreprocessedTrajectoryDataset(train_path, embedder=embedder, num_workers=num_workers)
    test_dataset = OptimizedPreprocessedTrajectoryDataset(test_path, embedder=embedder, num_workers=num_workers)
    val_dataset = OptimizedPreprocessedTrajectoryDataset(val_path, embedder=embedder, num_workers=num_workers)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        sampler=OptimizedPreprocessedTrajectorySampler(train_dataset),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        sampler=OptimizedPreprocessedTrajectorySampler(test_dataset),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        sampler=OptimizedPreprocessedTrajectorySampler(val_dataset),
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
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config)
        metrics = evaluate(
            model,
            test_loader,
            config["device"],
            has_action_head=True,
            has_win_head=True,
            has_move_order_head=False,
            has_ko_head=False,
            has_switch_head=False,
        )
        steps += train_metrics["steps"]
        test_loss = metrics["win_mse"] * config["win_weight"] + metrics["top3_loss"] * config["action_weight"]

        log = {
            "Total Steps": steps,
            "Train Loss": train_metrics["loss"],
            "Train Win Loss": train_metrics["win_loss"],
            "Train Action Loss": train_metrics["action_loss"],
            "Test Loss": test_loss,
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Top3 Loss": metrics["top3_loss"],
            "Test Action Top1": metrics["top1_acc"],
            "Test Action Top3": metrics["top3_acc"],
            "Test Action Top5": metrics["top5_acc"],
        }

        print(f"Epoch #{epoch + 1}:")
        for metric, value in log.items():
            print(f"=> {metric:<25}: {value:>10.3f}")

        wandb.log(log)

        total_time = time.time() - start
        time_taken = format_time(total_time)
        time_left = format_time(
            (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        )
        print(f"=> Time thus far: {time_taken} // ETA: {time_left}")
        print()

        scheduler.step(test_loss)

    torch.save(
        model.state_dict(),
        os.path.join(config["save_path"], f"{wandb.run.name}.pth"),  # type: ignore
    )
    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(
        model,
        val_loader,
        config["device"],
        has_action_head=True,
        has_win_head=True,
        has_move_order_head=False,
        has_ko_head=False,
        has_switch_head=False,
    )
    val_log = {
        "Total Steps": steps,
        "Validation Loss": (
            metrics["win_mse"] * config["win_weight"]
            + metrics["top3_loss"] * config["action_weight"]
        ),
        "Validation Win Corr": metrics["win_corr"],
        "Validation Win MSE": metrics["win_mse"],
        "Validation Top3 Loss": metrics["top3_loss"],
        "Validation Action Top1": metrics["top1_acc"],
        "Validation Action Top3": metrics["top3_acc"],
        "Validation Action Top5": metrics["top5_acc"],
    }

    for metric, value in val_log.items():
        print(f"==> {metric:<25}: {value:>10.3f}")

    wandb.log(val_log)

    print("\nAnalyzing...")
    analyze(
        model,
        val_loader,
        device=config["device"],
        has_action_head=True,
        has_win_head=True,
        has_move_order_head=False,
        has_ko_head=False,
        has_switch_head=False,
        teampreview_idx=config["teampreview_idx"],
        force_switch_indices=config["force_switch_indices"],
        verbose=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
