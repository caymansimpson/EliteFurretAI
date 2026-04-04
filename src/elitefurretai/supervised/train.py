# -*- coding: utf-8 -*-
"""
This script trains a supervised model for teampreview, turn action, and win prediction; if you have collected replays,
you can build a model to try to play like humans.
"""

import argparse
import gc
import os.path
import random
import time
from typing import Any, Dict, cast

import torch
import yaml

import wandb
from elitefurretai.etl import (
    MDBO,
    Embedder,
    OptimizedBattleDataLoader,
)
from elitefurretai.etl.system_utils import configure_torch_multiprocessing
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel
from elitefurretai.supervised.train_utils import (
    analyze,
    evaluate,
    focal_topk_cross_entropy_loss,
    format_time,
    topk_cross_entropy_loss,
)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    prev_steps: int,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    scaler=None,
) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    running_turn_loss = 0.0
    running_teampreview_loss = 0.0
    running_win_loss = 0.0
    running_entropy = 0.0
    steps = 0
    num_batches = 0
    start = time.time()

    # Gradient accumulation setup to reduce memory pressure
    accumulation_steps = config.get("accumulation_steps", 1)
    accumulation_counter = 0

    for batch in dataloader:
        # Transfer data to the right device
        if config["device"] == "cuda":
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}

        states = batch["states"].to(torch.float32)
        actions = batch["actions"]
        action_masks = batch["action_masks"]
        wins = batch["wins"].to(torch.float32)
        masks = batch["masks"]

        autocast = torch.amp.autocast if config["device"] == "cuda" else torch.autocast  # type: ignore
        with autocast(config["device"]):
            # Forward pass - returns four outputs (distributional value head)
            turn_action_logits, teampreview_logits, win_logits, _ = model(states, masks)

            # Determine which samples are teampreview vs turn decisions
            teampreview_mask = states[:, :, config["teampreview_idx"]] == 1  # (batch, seq)
            turn_mask = ~teampreview_mask  # (batch, seq)

            # Mask action logits appropriately
            # Teampreview: mask first 90 actions, Turn: mask all 2025 actions
            masked_turn_logits = turn_action_logits.masked_fill(
                ~action_masks.bool(), float("-inf")
            )
            # For teampreview, only use first 90 dimensions
            teampreview_action_masks = action_masks[:, :, : MDBO.teampreview_space()]
            masked_teampreview_logits = teampreview_logits.masked_fill(
                ~teampreview_action_masks.bool(), float("-inf")
            )

            # Flatten and filter valid samples
            valid_mask = masks.bool()  # (batch, seq)

            # Detect force switch states - any mon has force_switch=1
            # and mask them to force learning actions
            force_switch_mask = torch.zeros_like(turn_mask)
            for fs_idx in config["force_switch_indices"]:
                force_switch_mask = force_switch_mask | (states[:, :, fs_idx] > 0.5)

            # Conditionally exclude force switches from turn training based on config
            if not config.get("keep_force_switch", True):
                turn_valid_mask = valid_mask & turn_mask & ~force_switch_mask
            else:
                turn_valid_mask = valid_mask & turn_mask

            if turn_valid_mask.any():
                flat_turn_logits = masked_turn_logits[turn_valid_mask]
                flat_turn_actions = actions[turn_valid_mask]
                flat_turn_wins = wins[turn_valid_mask]
                flat_turn_win_logits = win_logits[turn_valid_mask]

                # Compute per-sample weights for move/switch action-type weighting.
                # Per-slot action indices: 0-39 are moves, 40-43 are switches, 44 is pass.
                move_w = config.get("move_loss_weight", 1.0)
                switch_w = config.get("switch_loss_weight", 1.0)
                if move_w != 1.0 or switch_w != 1.0:
                    slot1 = flat_turn_actions // 45
                    slot2 = flat_turn_actions % 45
                    has_switch = (
                        ((slot1 >= 40) & (slot1 <= 43))
                        | ((slot2 >= 40) & (slot2 <= 43))
                    )
                    sample_weights = torch.where(
                        has_switch,
                        torch.tensor(switch_w, device=states.device),
                        torch.tensor(move_w, device=states.device),
                    )
                else:
                    sample_weights = None

                # Choose loss function based on config
                loss_type = config.get("turn_loss_type", "topk")
                if loss_type == "focal":
                    turn_loss = focal_topk_cross_entropy_loss(
                        flat_turn_logits,
                        flat_turn_actions,
                        weights=sample_weights,
                        k=config.get("train_topk_k", 2025),
                        gamma=config.get("focal_gamma", 2.0),
                        alpha=config.get("focal_alpha", 0.25),
                        label_smoothing=config.get("label_smoothing", 0.0),
                    )
                else:  # "topk" or default
                    turn_loss = topk_cross_entropy_loss(
                        flat_turn_logits,
                        flat_turn_actions,
                        weights=sample_weights,
                        k=config.get("train_topk_k", 3),
                    )

                turn_win_loss = torch.nn.functional.mse_loss(
                    flat_turn_win_logits, flat_turn_wins.float()
                )

                # Compute entropy for regularization (encourages exploration)
                turn_probs = torch.nn.functional.softmax(flat_turn_logits, dim=-1)
                turn_entropy = (
                    -(turn_probs * torch.log(turn_probs + 1e-10)).sum(dim=-1).mean()
                )
            else:
                turn_loss = torch.tensor(0.0, device=states.device)
                turn_win_loss = torch.tensor(0.0, device=states.device)
                turn_entropy = torch.tensor(0.0, device=states.device)

            # Process teampreview samples
            teampreview_valid_mask = valid_mask & teampreview_mask
            if teampreview_valid_mask.any():
                flat_tp_logits = masked_teampreview_logits[teampreview_valid_mask]
                flat_tp_actions = actions[teampreview_valid_mask]
                flat_tp_wins = wins[teampreview_valid_mask]
                flat_tp_win_logits = win_logits[teampreview_valid_mask]

                # Filter out TP samples with turn-space encoded actions (data issue)
                tp_valid = flat_tp_actions < MDBO.teampreview_space()
                if tp_valid.any():
                    flat_tp_logits = flat_tp_logits[tp_valid]
                    flat_tp_actions = flat_tp_actions[tp_valid]
                    flat_tp_wins = flat_tp_wins[tp_valid]
                    flat_tp_win_logits = flat_tp_win_logits[tp_valid]

                    teampreview_loss = torch.nn.functional.cross_entropy(
                        flat_tp_logits, flat_tp_actions
                    )
                    teampreview_win_loss = torch.nn.functional.mse_loss(
                        flat_tp_win_logits, flat_tp_wins.float()
                    )
                else:
                    teampreview_loss = torch.tensor(0.0, device=states.device)
                    teampreview_win_loss = torch.tensor(0.0, device=states.device)
            else:
                teampreview_loss = torch.tensor(0.0, device=states.device)
                teampreview_win_loss = torch.tensor(0.0, device=states.device)

            # Combined loss with configurable weights
            # Entropy regularization: negative entropy encourages diversity (higher entropy = more uniform distribution)
            entropy_loss = (
                -turn_entropy
                if config.get("entropy_weight", 0.0) > 0
                else torch.tensor(0.0, device=states.device)
            )

            loss = (
                config["turn_loss_weight"] * turn_loss
                + config["teampreview_loss_weight"] * teampreview_loss
                + config["win_loss_weight"] * (turn_win_loss + teampreview_win_loss)
                + config.get("entropy_weight", 0.0) * entropy_loss
            )

            # Track number of samples
            num_turn_samples = turn_valid_mask.sum().item()
            num_tp_samples = teampreview_valid_mask.sum().item()

            # Scale loss by accumulation steps to maintain gradient magnitude
            loss = loss / accumulation_steps

        # Backpropagation with mixed precision (skip if loss has no grad)
        if loss.requires_grad:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            # No valid samples in batch — nothing to backprop
            continue

        accumulation_counter += 1

        # Only update weights every accumulation_steps
        if accumulation_counter >= accumulation_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["max_grad_norm"]
                )
                optimizer.step()

            optimizer.zero_grad()
            accumulation_counter = 0

        # Metrics (multiply back to get actual loss)
        running_loss += loss.item() * accumulation_steps
        running_turn_loss += turn_loss.item()
        running_teampreview_loss += teampreview_loss.item()
        running_win_loss += turn_win_loss.item() + teampreview_win_loss.item()
        running_entropy += turn_entropy.item() if turn_valid_mask.any() else 0.0
        steps += num_turn_samples + num_tp_samples
        num_batches += 1

        # Logging progress (only on actual optimizer steps)
        if num_batches % (100 * accumulation_steps) == 0:
            wandb.log({"Total Steps": prev_steps + steps, "grad_norm": grad_norm.item()})  # type: ignore

            # Periodic Python garbage collection every 100 batches
            gc.collect()

        if num_batches % (10 * accumulation_steps) == 0:
            wandb.log(
                {
                    "Total Steps": prev_steps + steps,
                    "train_loss": running_loss / num_batches,
                    "train_turn_loss": running_turn_loss / num_batches,
                    "train_teampreview_loss": running_teampreview_loss / num_batches,
                    "train_win_loss": running_win_loss / num_batches,
                    "train_entropy": running_entropy / num_batches,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Print progress
        assert dataloader.batch_size is not None
        batch_size = dataloader.batch_size
        time_taken = format_time(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        time_left = format_time((len(dataloader) - num_batches) * time_per_batch)
        processed = f"Processed {num_batches * batch_size} battles/trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {time_taken}"
        left = f" with an estimated {time_left} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    time_taken = format_time(time.time() - start)
    print("\033[2K\rDone training in " + time_taken)

    return {
        "loss": running_loss / num_batches,
        "steps": steps,
        "turn_loss": running_turn_loss / num_batches,
        "teampreview_loss": running_teampreview_loss / num_batches,
        "win_loss": running_win_loss / num_batches,
        "entropy": running_entropy / num_batches,
    }


def initialize(config):
    # Wandb defaults
    wandb.init(
        project="elitefurretai-hydreigon",
        config=config,
        settings=wandb.Settings(
            x_service_wait=30,  # Increase service wait time
            start_method="thread",  # Use thread instead of fork
        ),
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


def main(train_path, test_path, val_path, config={}, save_best=False):
    print("Starting!")
    default_config: Dict[str, Any] = {
        # Training config to optimize speed (see SUPERVISED.md DataLoader Performance)
        "worker_batch_size": 128,
        "num_workers": 3,
        "prefetch_factor": 2,
        "files_per_worker": 3,
        "persistent_workers": True,
        # Basic Training Params
        "learning_rate": 5e-5,
        "optimizer": "AdamW",
        "num_epochs": 30,
        # Regularization
        "batch_size": 512,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "max_grad_norm": 2.0,
        "teampreview_head_dropout": 0.1,
        # Loss Weights (three separate heads)
        "teampreview_loss_weight": 1,
        "turn_loss_weight": 1,
        "win_loss_weight": 1,
        "keep_force_switch": True,  # If True, include force switch examples from training
        "entropy_weight": 0.0,  # Entropy regularization to encourage prediction diversity (0.0 = off, try 0.01-0.1)
        # Loss Function Type
        "turn_loss_type": "topk",  # "topk" (standard topk CE) or "focal" (focal loss for hard examples)
        "train_topk_k": 3,  # Number of top predictions to consider (2025 = all actions, 3 = top-3 only)
        "focal_gamma": 2.0,  # Focal loss focusing parameter (higher = more focus on hard examples)
        "focal_alpha": 0.25,  # Focal loss weighting parameter
        # Architecture - Backbone
        "grouped_encoder_hidden_dim": 512,
        "grouped_encoder_aggregated_dim": 4096,
        "pokemon_attention_heads": 8,
        "early_layers": [4096, 4096, 2048, 2048, 1024],
        # Architecture - Transformer
        "transformer_layers": 6,
        "transformer_heads": 16,
        "transformer_ff_dim": 2048,
        "transformer_dropout": 0.1,
        "use_decision_tokens": True,
        "use_causal_mask": True,
        "late_layers": [2048, 1024, 1024, 512],
        # Architecture - Teampreview Head
        "teampreview_head_layers": [512, 256],
        "teampreview_attention_heads": 8,
        # Architecture - Turn Action Head
        "turn_head_layers": [512, 512, 512],
        # Distributional Value Head (C51)
        "num_value_bins": 51,
        "value_min": -1.0,
        "value_max": 1.0,
        # Feature set and format (saved in checkpoint for RL compatibility)
        "embedder_feature_set": "full",
        "battle_format": "gen9vgc2023regc",
        "max_seq_len": 40,
        # Other
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        "save_path": "data/models/supervised/",
        "seed": 21,
    }

    # Update config with any overrides provided in cfg
    for k, v in default_config.items():
        if k not in config:
            config[k] = v

    # Coerce numeric types that YAML may parse as strings (e.g. 3e-5)
    _float_keys = {
        "learning_rate", "dropout", "weight_decay", "max_grad_norm",
        "teampreview_head_dropout", "entropy_weight", "focal_gamma",
        "focal_alpha", "label_smoothing", "value_min", "value_max",
        "transformer_dropout",
    }
    _int_keys = {
        "batch_size", "worker_batch_size", "num_workers", "prefetch_factor",
        "files_per_worker", "num_epochs", "seed", "lstm_layers",
        "lstm_hidden_size", "num_value_bins", "max_seq_len",
        "early_attention_heads", "late_attention_heads",
        "pokemon_attention_heads", "teampreview_attention_heads",
        "grouped_encoder_hidden_dim", "grouped_encoder_aggregated_dim",
        "train_topk_k", "transformer_layers", "transformer_heads",
        "transformer_ff_dim",
    }
    for k in _float_keys:
        if k in config:
            config[k] = float(config[k])
    for k in _int_keys:
        if k in config:
            config[k] = int(config[k])

    config["accumulation_steps"] = int(config["batch_size"] // config["worker_batch_size"])
    print(f"Starting training with config: {config}")
    initialize(config)

    # Initialize Embedder and find indices of special features; these will be used
    # for weighting training and analyzing model performance
    embedder = Embedder(
        format=config["battle_format"],
        feature_set=config["embedder_feature_set"],
        omniscient=False,
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print(
        f"Embedder initialized. Embedding[{embedder.embedding_size}] on {config['device']}"
    )

    print("Loading datasets...")
    start = time.time()

    train_loader = OptimizedBattleDataLoader(
        train_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        files_per_worker=config["files_per_worker"],
        persistent_workers=config["persistent_workers"],
    )
    test_loader = OptimizedBattleDataLoader(
        test_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=4,
        prefetch_factor=2,
        files_per_worker=1,
    )
    val_loader = OptimizedBattleDataLoader(
        val_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=4,
        prefetch_factor=2,
        files_per_worker=1,
    )

    # Initialize model with flexible architecture
    raw_model = TransformerThreeHeadedModel(
        embedder=embedder,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        dropout=config["dropout"],
        grouped_encoder_hidden_dim=config["grouped_encoder_hidden_dim"],
        grouped_encoder_aggregated_dim=config["grouped_encoder_aggregated_dim"],
        pokemon_attention_heads=config["pokemon_attention_heads"],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        teampreview_head_layers=config["teampreview_head_layers"],
        teampreview_head_dropout=config["teampreview_head_dropout"],
        teampreview_attention_heads=config["teampreview_attention_heads"],
        turn_head_layers=config["turn_head_layers"],
        max_seq_len=config["max_seq_len"],
        num_value_bins=config["num_value_bins"],
        value_min=config["value_min"],
        value_max=config["value_max"],
        transformer_layers=config.get("transformer_layers", 6),
        transformer_heads=config.get("transformer_heads", 16),
        transformer_ff_dim=config.get("transformer_ff_dim", 2048),
        transformer_dropout=config.get("transformer_dropout", 0.1),
        use_decision_tokens=config.get("use_decision_tokens", True),
        use_causal_mask=config.get("use_causal_mask", True),
    ).to(config["device"])
    model = cast(torch.nn.Module, raw_model)

    # Auto-detect model class for RL checkpoint compatibility before compile wraps it
    config["use_transformer"] = isinstance(model, TransformerThreeHeadedModel)

    # torch.compile fuses GPU kernels for ~15-30% speedup on GPU-bound training.
    # First batch is slow (compilation), but all subsequent batches are faster.
    # With 30min epochs, this saves ~5-9 min/epoch.
    if config["device"] == "cuda":
        compiled_model = torch.compile(model)
        model = cast(torch.nn.Module, compiled_model)

    wandb.watch(model, log="all", log_freq=1000)

    # Count Parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Finished loading data and model! Total trainable parameters: {num_params:,}")
    wandb.log({"model_parameters": num_params})
    print("Starting training...")

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

    scheduler: Any
    scheduler_type = config.get("lr_schedule", "plateau")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if config["device"] == "cuda" else None  # type: ignore

    print("Initialized model! Starting training...")

    # Training loop
    start, steps = time.time(), 0
    best_test_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config, scaler)

        # Evaluate with native three-headed support
        metrics = evaluate(
            model,
            test_loader,
            config["device"],
            has_teampreview_head=True,
            teampreview_idx=config["teampreview_idx"],
            config=config,
        )
        steps += train_metrics["steps"]
        test_loss = (
            metrics["win_mse"] * config["win_loss_weight"]
            + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
            + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
        )

        log = {
            "Total Steps": steps,
            "Train Loss": train_metrics["loss"],
            "Train Win Loss": train_metrics["win_loss"],
            "Train Turn Loss": train_metrics["turn_loss"],
            "Train Teampreview Loss": train_metrics["teampreview_loss"],
            "Test Loss": test_loss,
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
            "Test Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
            "Test Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
            "Test Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
            "Test Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
            "Test Turn Top1": metrics.get("turn_top1_acc", 0),
            "Test Turn Top3": metrics.get("turn_top3_acc", 0),
            "Test Turn Top5": metrics.get("turn_top5_acc", 0),
            # Action-type specific metrics
            "Test MOVE Top1": metrics.get("move_top1_acc", 0),
            "Test MOVE Top3": metrics.get("move_top3_acc", 0),
            "Test MOVE Top5": metrics.get("move_top5_acc", 0),
            "Test SWITCH Top1": metrics.get("switch_top1_acc", 0),
            "Test SWITCH Top3": metrics.get("switch_top3_acc", 0),
            "Test SWITCH Top5": metrics.get("switch_top5_acc", 0),
            "Test BOTH Top1": metrics.get("both_top1_acc", 0),
            "Test BOTH Top3": metrics.get("both_top3_acc", 0),
            "Test BOTH Top5": metrics.get("both_top5_acc", 0),
        }

        print(f"Epoch #{epoch + 1}:")
        for metric, value in log.items():
            print(f"=> {metric:<30}: {value:>10.3f}")

        wandb.log(log)

        total_time = time.time() - start
        time_taken = format_time(total_time)
        time_left = format_time(
            (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        )
        print(f"=> Time thus far: {time_taken} // ETA: {time_left}")
        print()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        # Save best checkpoint if requested
        if save_best and test_loss < best_test_loss:
            best_test_loss = test_loss
            best_save_dict = {"model_state_dict": model.state_dict(), "config": config}
            best_path = os.path.join(config["save_path"], f"{wandb.run.name}_best.pt")  # type: ignore
            torch.save(best_save_dict, best_path)
            print(f"New best model saved to {best_path} (test_loss={test_loss:.4f})")

    # Save final model with config embedded
    save_dict = {"model_state_dict": model.state_dict(), "config": config}
    save_path = os.path.join(config["save_path"], f"{wandb.run.name}.pt")  # type: ignore
    torch.save(save_dict, save_path)
    print(f"\nModel and config saved to {save_path}")

    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(
        model,
        val_loader,
        config["device"],
        has_teampreview_head=True,
        teampreview_idx=config["teampreview_idx"],
        config=config,
    )
    val_log = {
        "Total Steps": steps,
        "Validation Loss": (
            metrics["win_mse"] * config["win_loss_weight"]
            + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
            + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
        ),
        "Validation Win Corr": metrics["win_corr"],
        "Validation Win MSE": metrics["win_mse"],
        "Validation Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
        "Validation Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
        "Validation Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
        "Validation Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
        "Validation Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
        "Validation Turn Top1": metrics.get("turn_top1_acc", 0),
        "Validation Turn Top3": metrics.get("turn_top3_acc", 0),
        "Validation Turn Top5": metrics.get("turn_top5_acc", 0),
    }

    for metric, value in val_log.items():
        print(f"==> {metric:<30}: {value:>10.3f}")

    wandb.log(val_log)

    print("\nAnalyzing...")
    analyze(
        model,
        val_loader,
        device=config["device"],
        has_teampreview_head=True,
        teampreview_idx=config["teampreview_idx"],
        force_switch_indices=config["force_switch_indices"],
    )


if __name__ == "__main__":
    configure_torch_multiprocessing(
        use_file_system_sharing=True,
        filter_socket_send_warning=True,
    )

    parser = argparse.ArgumentParser(
        description="Train a supervised BC model for teampreview, turn action, and win prediction."
    )
    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to data directory containing train/, test/, and val/ subdirectories",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides defaults)",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best model checkpoint (lowest test loss) during training",
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    main(
        os.path.join(args.data_directory, "train"),
        os.path.join(args.data_directory, "test"),
        os.path.join(args.data_directory, "val"),
        cfg,
        save_best=args.save_best,
    )
