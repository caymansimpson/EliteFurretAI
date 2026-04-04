# -*- coding: utf-8 -*-
"""
Wandb Bayesian hyperparameter sweep for supervised VGC action prediction.

Sweep configuration is loaded from a YAML config file in sweep_configs/.
See sweep_configs/first_config.yaml for an example with full documentation.

Usage:
    # Run a sweep with a YAML config
    python train_sweep.py data/battles/regc_final_v3 --config sweep_configs/second_config.yaml

    # Custom run count
    python train_sweep.py data/battles/regc_final_v3 --config sweep_configs/second_config.yaml --count 50

    # Resume an existing sweep
    python train_sweep.py data/battles/regc_final_v3 --config sweep_configs/second_config.yaml --sweep-id abc123

    # Use a specific wandb project
    python train_sweep.py data/battles/regc_final_v3 --config sweep_configs/second_config.yaml --project my-project
"""

import argparse
import gc
import os
import random
import time
from typing import Any, Dict, Union, cast

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
from elitefurretai.supervised.train import train_epoch
from elitefurretai.supervised.train_utils import (
    evaluate,
    format_time,
)

# ===========================================================================
# Configuration — loaded from YAML at startup, accessed by sweep_train()
# ===========================================================================

SWEEP_CONFIG: Dict[str, Any] = {}
LATE_LAYERS_VARIANTS: Dict[str, list] = {}
TURN_HEAD_LAYERS_VARIANTS: Dict[str, list] = {}
TEAMPREVIEW_HEAD_LAYERS_VARIANTS: Dict[str, list] = {}
FIXED_CONFIG: Dict[str, Any] = {}


def load_sweep_config(config_path: str) -> None:
    """Load sweep, fixed, and variant configs from a YAML file."""
    global SWEEP_CONFIG, FIXED_CONFIG
    global LATE_LAYERS_VARIANTS, TURN_HEAD_LAYERS_VARIANTS, TEAMPREVIEW_HEAD_LAYERS_VARIANTS

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    SWEEP_CONFIG = raw["sweep"]
    FIXED_CONFIG = raw.get("fixed", {})

    variants = raw.get("variants", {})
    LATE_LAYERS_VARIANTS = variants.get("late_layers", {})
    TURN_HEAD_LAYERS_VARIANTS = variants.get("turn_head_layers", {})
    TEAMPREVIEW_HEAD_LAYERS_VARIANTS = variants.get("teampreview_head_layers", {})


def _resolve_config(wandb_config: dict) -> Dict[str, Any]:
    """Merge wandb sweep params with fixed config and resolve variant names."""
    config: Dict[str, Any] = {**FIXED_CONFIG}

    # Copy all sweep parameters
    for key, value in wandb_config.items():
        if key.endswith("_variant"):
            continue  # Handled below
        config[key] = value

    # Resolve list-valued architecture variants
    config["late_layers"] = LATE_LAYERS_VARIANTS[
        wandb_config.get("late_layers_variant", "large")
    ]
    config["turn_head_layers"] = TURN_HEAD_LAYERS_VARIANTS[
        wandb_config.get("turn_head_layers_variant", "medium")
    ]
    config["teampreview_head_layers"] = TEAMPREVIEW_HEAD_LAYERS_VARIANTS[
        wandb_config.get("teampreview_head_layers_variant", "small")
    ]

    # Set device
    config["device"] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Compute gradient accumulation from batch sizes
    config["accumulation_steps"] = int(
        config["batch_size"] // config["worker_batch_size"]
    )

    # Coerce numeric types that YAML/wandb may parse as strings
    _float_keys = {
        "learning_rate", "dropout", "weight_decay", "max_grad_norm",
        "teampreview_head_dropout", "entropy_weight", "focal_gamma",
        "focal_alpha", "label_smoothing", "value_min", "value_max",
        "transformer_dropout", "move_loss_weight", "switch_loss_weight",
    }
    _int_keys = {
        "batch_size", "worker_batch_size", "num_workers", "prefetch_factor",
        "files_per_worker", "num_epochs", "seed", "num_value_bins",
        "max_seq_len", "pokemon_attention_heads", "teampreview_attention_heads",
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

    return config


def sweep_train() -> None:
    """One sweep trial: initialize wandb, build model, train, evaluate."""
    run = wandb.init()
    config = _resolve_config(dict(wandb.config))

    # Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    random.seed(config["seed"])

    # Initialize embedder and locate special feature indices
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

    # Log resolved config (including list-valued params)
    wandb.config.update(
        {
            "late_layers": config["late_layers"],
            "turn_head_layers": config["turn_head_layers"],
            "teampreview_head_layers": config["teampreview_head_layers"],
            "embedding_size": embedder.embedding_size,
        },
        allow_val_change=True,
    )

    print(f"\n{'='*60}")
    print(f"Sweep run: {run.name}")  # type: ignore[union-attr]
    print(f"Config: {config}")
    print(f"{'='*60}\n")

    # ---- Data loaders ----
    train_loader = OptimizedBattleDataLoader(
        _train_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        files_per_worker=config["files_per_worker"],
        persistent_workers=config["persistent_workers"],
    )
    test_loader = OptimizedBattleDataLoader(
        _test_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=2,
        prefetch_factor=2,
        files_per_worker=1,
    )

    # ---- Model ----
    model = cast(
        torch.nn.Module,
        TransformerThreeHeadedModel(
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
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        transformer_ff_dim=config["transformer_ff_dim"],
        transformer_dropout=config["transformer_dropout"],
        use_decision_tokens=config["use_decision_tokens"],
        use_causal_mask=config["use_causal_mask"],
        ).to(config["device"]),
    )

    # torch.compile fuses GPU kernels for ~15-30% speedup on GPU-bound training.
    # First batch is slow (compilation), but all subsequent batches are faster.
    # With 30min epochs, this saves ~5-9 min/epoch.
    if config["device"] == "cuda":
        compiled_model = torch.compile(model)
        model = cast(torch.nn.Module, compiled_model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"model_parameters": num_params})
    print(f"Model: {num_params:,} trainable parameters")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler: Union[
        torch.optim.lr_scheduler.CosineAnnealingLR,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ]
    scheduler_type = config.get("lr_schedule", "plateau")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["num_epochs"], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    scaler = (
        torch.amp.GradScaler("cuda") if config["device"] == "cuda" else None  # type: ignore
    )

    # ---- Training loop ----
    total_steps = 0
    start = time.time()

    for epoch in range(config["num_epochs"]):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model, train_loader, total_steps, optimizer, config, scaler
        )
        total_steps += train_metrics["steps"]

        # Evaluate on test set
        metrics = evaluate(
            model,
            test_loader,
            config["device"],
            has_teampreview_head=True,
            teampreview_idx=config["teampreview_idx"],
            config=config,
        )

        test_loss = (
            metrics["win_mse"] * config["win_loss_weight"]
            + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
            + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
        )

        epoch_time = time.time() - epoch_start

        # Log all metrics (sweep metric "Test Turn Top3" must be logged here)
        log = {
            "epoch": epoch + 1,
            "Total Steps": total_steps,
            "epoch_time_sec": epoch_time,
            # Training metrics
            "Train Loss": train_metrics["loss"],
            "Train Turn Loss": train_metrics["turn_loss"],
            "Train Teampreview Loss": train_metrics["teampreview_loss"],
            "Train Win Loss": train_metrics["win_loss"],
            "Train Entropy": train_metrics["entropy"],
            # Primary sweep target — Turn Top-3 accuracy
            "Test Turn Top1": metrics.get("turn_top1_acc", 0),
            "Test Turn Top3": metrics.get("turn_top3_acc", 0),
            "Test Turn Top5": metrics.get("turn_top5_acc", 0),
            "Test Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
            # Action-type breakdown (key diagnostic for switch-bias)
            "Test MOVE Top1": metrics.get("move_top1_acc", 0),
            "Test MOVE Top3": metrics.get("move_top3_acc", 0),
            "Test MOVE Top5": metrics.get("move_top5_acc", 0),
            "Test SWITCH Top1": metrics.get("switch_top1_acc", 0),
            "Test SWITCH Top3": metrics.get("switch_top3_acc", 0),
            "Test SWITCH Top5": metrics.get("switch_top5_acc", 0),
            "Test BOTH Top1": metrics.get("both_top1_acc", 0),
            "Test BOTH Top3": metrics.get("both_top3_acc", 0),
            "Test BOTH Top5": metrics.get("both_top5_acc", 0),
            # Teampreview metrics (secondary target)
            "Test Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
            "Test Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
            "Test Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
            "Test Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
            # Win prediction
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Loss": test_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        # Composite diagnostic: ratio of MOVE vs SWITCH accuracy reveals bias.
        # If SWITCH Top3 >> MOVE Top3, the model is switch-biased.
        move_top3 = metrics.get("move_top3_acc", 0)
        switch_top3 = metrics.get("switch_top3_acc", 0)
        if switch_top3 > 0:
            log["MOVE_SWITCH_Top3_Ratio"] = move_top3 / switch_top3
        log["MOVE_SWITCH_Top3_Gap"] = move_top3 - switch_top3

        wandb.log(log)

        # Print summary
        total_elapsed = format_time(time.time() - start)
        eta = format_time(
            (config["num_epochs"] - epoch - 1) * (time.time() - start) / (epoch + 1)
        )
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Turn Top3: {metrics.get('turn_top3_acc', 0):.3f} | "
            f"MOVE Top3: {move_top3:.3f} | "
            f"SWITCH Top3: {switch_top3:.3f} | "
            f"TP Top1: {metrics.get('teampreview_top1_acc', 0):.3f} | "
            f"Elapsed: {total_elapsed} | ETA: {eta}"
        )

        # Step LR scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        # Periodic GC to prevent memory buildup across epochs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()


# Module-level paths set by __main__ and read by sweep_train()
_train_path: str = ""
_test_path: str = ""


if __name__ == "__main__":
    configure_torch_multiprocessing(
        use_file_system_sharing=True,
        filter_socket_send_warning=True,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run a W&B Bayesian hyperparameter sweep for supervised VGC "
            "action prediction. Optimizes for Turn Top-3 accuracy."
        ),
    )
    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to data directory containing train/, test/, val/ subdirectories",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML sweep config file (see sweep_configs/ for examples)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=35,
        help="Number of sweep runs (default: 35)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing sweep by ID (skip sweep creation)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="elitefurretai-hydreigon",
        help="W&B project name",
    )
    args = parser.parse_args()

    # Load sweep config from YAML
    load_sweep_config(args.config)

    # Set module-level paths for sweep_train() to access
    _train_path = os.path.join(args.data_directory, "train")
    _test_path = os.path.join(args.data_directory, "test")

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(
            SWEEP_CONFIG,
            project=args.project,
        )
        print(f"Created sweep: {sweep_id}")

    print(f"Starting {args.count} sweep runs...")
    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)
