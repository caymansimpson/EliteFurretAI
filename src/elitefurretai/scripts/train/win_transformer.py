# -*- coding: utf-8 -*-
"""This script trains a supervised model for move and winprediction; if you have collected replays, you can build a model
to try to play like humans.
"""
import os.path
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch

import wandb
from elitefurretai.model_utils import (
    MDBO,
    Embedder,
    MoveOrderEncoder,
    PreprocessedTrajectoryDataset,
)
from elitefurretai.model_utils.train_utils import (
    analyze,
    evaluate,
    flatten_and_filter,
    format_time,
    topk_cross_entropy_loss,
)


class TwoHeadedTransformerModel(torch.nn.Module):
    """
    Pure Transformer encoder architecture:

    Steps:
      1. Linear projection of input_size -> d_model
      2. Learned positional embedding added
      3. TransformerEncoder (N layers)
      4. LayerNorm
      5. Two per-step heads:
           - action logits (batch, seq_len, num_actions)
           - win logits (batch, seq_len)

    The forward interface matches the hybrid model.
    """

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        max_seq_len: int = 17,
        d_model: int = 256,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_actions = num_actions
        self.d_model = d_model

        self.input_proj = torch.nn.Linear(input_size, d_model)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=torch.nn.LayerNorm(d_model, eps=layer_norm_eps),
        )

        self.action_head = torch.nn.Linear(d_model, num_actions)
        self.win_head = torch.nn.Linear(d_model, 1)  # Regression now, predicts advantage
        self.move_order_head = torch.nn.Linear(
            d_model, MoveOrderEncoder.action_space()
        )  # 24 possible move orders
        self.ko_head = torch.nn.Linear(
            d_model, 4
        )  # Binary classification for each position
        self.switch_head = torch.nn.Linear(
            d_model, 4
        )  # Binary classification for each position

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_size)
        mask: (batch, seq_len) with 1 where valid, 0 where padding
        """
        b, seq_len, _ = x.shape
        if mask is None:
            mask = torch.ones(b, seq_len, device=x.device)

        x = self.input_proj(x)
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pos_embedding(pos_idx) * mask.unsqueeze(-1)

        # Transformer key_padding_mask expects True for padded positions
        key_padding_mask = ~mask.bool()  # (batch, seq_len)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        action_logits = self.action_head(x)
        win_logits = self.win_head(x).squeeze(-1)  # [batch, seq_len]
        move_order_logits = self.move_order_head(x)  # [batch, seq_len, 24]
        ko_logits = self.ko_head(x)  # [batch, seq_len, 4]
        switch_logits = self.switch_head(x)  # [batch, seq_len, 4]
        return action_logits, win_logits, move_order_logits, ko_logits, switch_logits

    def predict(self, x, mask=None):
        with torch.no_grad():
            action_logits, win_logits, move_order_logits, ko_logits, switch_logits = (
                self.forward(x, mask)
            )
            action_probs = torch.softmax(action_logits, dim=-1)
            win_values = win_logits  # Already regression values
            move_order_probs = torch.softmax(move_order_logits, dim=-1)
            ko_probs = torch.sigmoid(ko_logits)  # Binary classification per position
            switch_probs = torch.sigmoid(
                switch_logits
            )  # Binary classification per position
        return action_probs, win_values, move_order_probs, ko_probs, switch_probs


def train_epoch(model, dataloader, prev_steps, optimizer, config):
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_win_loss = 0.0
    running_move_order_loss = 0.0
    running_ko_loss = 0.0
    running_switch_loss = 0.0
    running_move_order_acc = 0.0
    running_ko_acc = 0.0
    running_switch_acc = 0.0
    steps = 0
    num_batches = 0
    start = time.time()

    for batch in dataloader:
        # Get data from dictionary
        states = batch["states"].to(config["device"])
        actions = batch["actions"].to(config["device"])
        action_masks = batch["action_masks"].to(config["device"])
        wins = batch["wins"].to(config["device"])
        masks = batch["masks"].to(config["device"])
        move_orders = batch["move_orders"].to(config["device"])
        kos = batch["kos"].to(config["device"])
        switches = batch["switches"].to(config["device"])

        # Forward pass with all heads
        action_logits, win_logits, move_order_logits, ko_logits, switch_logits = model(
            states, masks
        )
        masked_action_logits = action_logits.masked_fill(
            ~action_masks.bool(), float("-inf")
        )

        # Use helper for flattening and filtering all tensors
        flat_data = flatten_and_filter(
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
        if flat_data is None:
            continue

        (
            valid_states,
            valid_action_logits,
            valid_actions,
            valid_win_logits,
            valid_wins,
            valid_move_order_logits,
            valid_move_orders,
            valid_ko_logits,
            valid_kos,
            valid_switch_logits,
            valid_switches,
        ) = flat_data

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
        # Action loss (same as before)
        action_loss = topk_cross_entropy_loss(
            valid_action_logits, valid_actions, weights=weights, k=3
        )

        # Win loss (now MSE since it's regression)
        win_loss = torch.nn.functional.mse_loss(valid_win_logits, valid_wins.float())

        # Move order loss (categorical cross-entropy)
        move_order_loss = torch.nn.functional.cross_entropy(
            valid_move_order_logits, valid_move_orders.long()
        )

        # KO loss (binary cross-entropy per position)
        ko_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_ko_logits, valid_kos
        )

        # Switch loss (binary cross-entropy per position)
        switch_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_switch_logits, valid_switches
        )

        # Combined loss with weights from config
        loss = (
            config["action_weight"] * action_loss
            + config["win_weight"] * win_loss
            + config["move_order_weight"] * move_order_loss
            + config["ko_weight"] * ko_loss
            + config["switch_weight"] * switch_loss
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_action_loss += action_loss.item()
        running_win_loss += win_loss.item()
        running_move_order_loss += move_order_loss.item()
        running_ko_loss += ko_loss.item()
        running_switch_loss += switch_loss.item()

        # Move order accuracy
        move_order_preds = torch.argmax(valid_move_order_logits, dim=-1)
        running_move_order_acc = (
            (move_order_preds == valid_move_orders).float().mean().item()
        )

        # KO accuracy (binary classification)
        ko_preds = (torch.sigmoid(valid_ko_logits) > 0.5).float()
        running_ko_acc = (ko_preds == valid_kos).float().mean().item()

        # Switch accuracy (binary classification)
        switch_preds = (torch.sigmoid(valid_switch_logits) > 0.5).float()
        running_switch_acc = (switch_preds == valid_switches).float().mean().item()
        steps += valid_actions.size(0)
        num_batches += 1

        # Logging progress with all metrics
        wandb.log(
            {
                "steps": prev_steps + steps,
                "train_loss": running_loss / num_batches,
                "train_action_loss": running_action_loss / num_batches,
                "train_win_loss": running_win_loss / num_batches,
                "train_move_order_loss": running_move_order_loss / num_batches,
                "train_ko_loss": running_ko_loss / num_batches,
                "train_switch_loss": running_switch_loss / num_batches,
                "train_move_order_acc": running_move_order_acc,
                "train_ko_acc": running_ko_acc,
                "train_switch_acc": running_switch_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Print progress
        time_taken = format_time(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        time_left = format_time((len(dataloader) - num_batches) * time_per_batch)
        processed = f"Processed {num_batches * dataloader.batch_size} trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {time_taken}"
        left = f" with an estimated {time_left} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    time_taken = format_time(time.time() - start)
    print("\033[2K\rDone training epoch in " + time_taken)

    # Return all metrics
    return {
        "loss": running_loss / num_batches,
        "steps": steps,
        "action_loss": running_action_loss / num_batches,
        "win_loss": running_win_loss / num_batches,
        "move_order_loss": running_move_order_loss / num_batches,
        "ko_loss": running_ko_loss / num_batches,
        "switch_loss": running_switch_loss / num_batches,
    }


def main(train_path, test_path, val_path):
    print("Starting!")
    config: Dict[str, Any] = {
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "num_epochs": 20,
        "num_heads": 4,
        "action_weight": 0.0,
        "win_weight": 0.9,
        "move_order_weight": 0.025,
        "ko_weight": 0.05,
        "switch_weight": 0.025,
        "max_grad_norm": 1.0,
        "device": (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "save_path": "data/models/",
        "seed": 21,
        "d_model": 256,
        "num_transformer_layers": 4,
        "ff_dim": 1024,
        "dropout": 0.1,
    }
    wandb.init(project="elitefurretai-victini-new", config=config)
    wandb.save(__file__)

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))

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

    train_dataset = PreprocessedTrajectoryDataset(train_path, embedder=embedder)
    test_dataset = PreprocessedTrajectoryDataset(test_path, embedder=embedder)
    val_dataset = PreprocessedTrajectoryDataset(val_path, embedder=embedder)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4
    )

    # Initialize model
    model = TwoHeadedTransformerModel(
        input_size=embedder.embedding_size,
        num_actions=MDBO.action_space(),
        max_seq_len=17,
        d_model=config.get("d_model", 256),
        num_layers=config.get("num_transformer_layers", 4),
        nhead=config["num_heads"],
        dim_feedforward=config.get("ff_dim", 1024),
        dropout=config.get("dropout", 0.1),
    ).to(config["device"])
    # wandb.watch(model, log="all", log_freq=1000)
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
        metrics = evaluate(model, test_loader, config["device"], has_win_head=True)
        steps += train_metrics["steps"]

        log = {
            "Total Steps": steps,
            "Train Loss": train_metrics["loss"],
            "Train Win Loss": train_metrics["win_loss"],
            "Train Action Loss": train_metrics["action_loss"],
            "Test Loss": (
                metrics["win_mse"] * config["win_weight"]
                + metrics["top3_loss"] * config["action_weight"]
            ),
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Top3 Loss": metrics["top3_loss"],
            "Test Action Top1": metrics["top1_acc"],
            "Test Action Top3": metrics["top3_acc"],
            "Test Action Top5": metrics["top5_acc"],
            "Test Move Order Acc": metrics["move_order_acc"],
            "Test KO Acc": metrics["ko_acc"],
            "Test Switch Acc": metrics["switch_acc"],
        }

        if "move_order_acc" in metrics:
            log["Test Move Order Acc"] = metrics["move_order_acc"]
        if "ko_acc" in metrics:
            log["Test KO Acc"] = metrics["ko_acc"]
        if "switch_acc" in metrics:
            log["Test Switch Acc"] = metrics["switch_acc"]

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

        scheduler.step(
            float(
                metrics["win_mse"] * config["win_weight"]
                + metrics["top3_loss"] * config["action_weight"]
            )
        )

    torch.save(
        model.state_dict(),
        os.path.join(config["save_path"], f"{wandb.run.name}.pth"),  # type: ignore
    )
    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(model, val_loader, config["device"], has_win_head=True)
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

    if "move_order_acc" in metrics:
        val_log["Validation Move Order Acc"] = metrics["move_order_acc"]
    if "ko_acc" in metrics:
        val_log["Validation KO Acc"] = metrics["ko_acc"]
    if "switch_acc" in metrics:
        val_log["Validation Switch Acc"] = metrics["switch_acc"]

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
        has_move_order_head=True,
        has_ko_head=True,
        has_switch_head=True,
        teampreview_idx=config["teampreview_idx"],
        force_switch_indices=config["force_switch_indices"],
        verbose=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
