# -*- coding: utf-8 -*-
"""
Fine-tune a previously trained TransformerThreeHeadedModel.

This script:
1. Loads a saved model checkpoint with its embedded config
2. Optionally overrides config parameters with a new config file
3. Continues training from the loaded weights using the same training
   infrastructure as ``train.py`` (same train_epoch, evaluate, analyze,
   DataLoader defaults, torch.compile, lr_schedule choice, type coercion).
4. Saves the fine-tuned model with a new name; optionally saves the
   best-by-test-loss checkpoint along the way.

Usage:
    python fine_tune.py <data_directory> <model_path> <wandb_run_name> \
        [--config-override path/to/overrides.yaml] [--save-best]

The data_directory must contain train/, test/, and val/ subdirectories.
The config_override file (if provided) overrides matching keys from the
model's saved config — typically learning_rate, num_epochs, weight_decay,
dropout, lr_schedule, etc. Architecture keys must NOT change (they would
mismatch the loaded weights).
"""

import argparse
import os
import time
from typing import Any, Dict, Optional, cast

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
from elitefurretai.supervised.utils import analyze, evaluate, format_time

# Keys that may appear in YAML as e.g. "5e-5" / "1.0" but must be floats/ints.
# Mirrors train.py.
_FLOAT_KEYS = {
    "learning_rate",
    "dropout",
    "weight_decay",
    "max_grad_norm",
    "teampreview_head_dropout",
    "entropy_weight",
    "focal_gamma",
    "focal_alpha",
    "label_smoothing",
    "value_min",
    "value_max",
    "transformer_dropout",
    "move_loss_weight",
    "switch_loss_weight",
    "turn_loss_weight",
    "teampreview_loss_weight",
    "win_loss_weight",
}
_INT_KEYS = {
    "batch_size",
    "worker_batch_size",
    "num_workers",
    "prefetch_factor",
    "files_per_worker",
    "num_epochs",
    "seed",
    "num_value_bins",
    "max_seq_len",
    "pokemon_attention_heads",
    "teampreview_attention_heads",
    "grouped_encoder_hidden_dim",
    "grouped_encoder_aggregated_dim",
    "train_topk_k",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_dim",
}


def _coerce_numeric_types(config: Dict[str, Any]) -> None:
    for k in _FLOAT_KEYS:
        if k in config:
            config[k] = float(config[k])
    for k in _INT_KEYS:
        if k in config:
            config[k] = int(config[k])


def _build_transformer_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull only TransformerThreeHeadedModel constructor kwargs from config.

    Architecture keys must come from the saved checkpoint config so the
    constructed module matches the loaded weight shapes.
    """
    return dict(
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
        value_head_layers=config.get("value_head_layers", []),
        max_seq_len=config.get("max_seq_len", 40),
        num_value_bins=config.get("num_value_bins", 51),
        value_min=config.get("value_min", -1.0),
        value_max=config.get("value_max", 1.0),
        number_bank_hp_bins=config.get("number_bank_hp_bins", 100),
        number_bank_stat_bins=config.get("number_bank_stat_bins", 600),
        number_bank_power_bins=config.get("number_bank_power_bins", 250),
        number_bank_embedding_dim=config.get("number_bank_embedding_dim", 16),
        number_bank_damage_bins=config.get("number_bank_damage_bins", 600),
        number_bank_damage_embed_dim=config.get("number_bank_damage_embed_dim", 4),
        number_bank_turn_bins=config.get("number_bank_turn_bins", 40),
        number_bank_turn_embed_dim=config.get("number_bank_turn_embed_dim", 16),
        number_bank_rating_bins=config.get("number_bank_rating_bins", 100),
        number_bank_rating_embed_dim=config.get("number_bank_rating_embed_dim", 16),
        ability_embed_dim=config.get("ability_embed_dim", 16),
        item_embed_dim=config.get("item_embed_dim", 16),
        species_embed_dim=config.get("species_embed_dim", 32),
        move_embed_dim=config.get("move_embed_dim", 16),
        transformer_layers=config.get("transformer_layers", 6),
        transformer_heads=config.get("transformer_heads", 16),
        transformer_ff_dim=config.get("transformer_ff_dim", 2048),
        transformer_dropout=config.get("transformer_dropout", 0.1),
        use_decision_tokens=config.get("use_decision_tokens", True),
        use_causal_mask=config.get("use_causal_mask", True),
        value_to_trunk_grad_scale=config.get("value_to_trunk_grad_scale", 1.0),
    )


def load_model_and_config(model_path: str, device: str):
    """Load checkpoint, rebuild TransformerThreeHeadedModel, restore weights."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if "config" not in checkpoint:
        raise ValueError(
            f"Model checkpoint at {model_path} does not contain config. "
            "Please ensure the model was saved with both model_state_dict and config."
        )

    config: Dict[str, Any] = checkpoint["config"]
    print(f"Loaded config with {len(config)} parameters")

    embedder = Embedder(
        format=config.get("battle_format", "gen9vgc2023regc"),
        feature_set=config.get("embedder_feature_set", Embedder.FULL),
        omniscient=False,
    )

    model = TransformerThreeHeadedModel(
        embedder=embedder,
        **_build_transformer_kwargs(config),
    ).to(device)

    # Strip torch.compile's "_orig_mod." prefix (train.py wraps the model with
    # torch.compile before saving, so every key in checkpoints from train.py
    # carries this prefix; the un-compiled model we just built does not).
    raw_state_dict = checkpoint["model_state_dict"]
    state_dict = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in raw_state_dict.items()
    }

    # strict=False so older checkpoints missing newly-added keys (e.g. distributional
    # head buffers) get random init for those tensors only.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys (will use random init): {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys (ignored): {unexpected}")
    print("Model weights loaded successfully")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    return model, config, embedder


def finetune(
    train_path: str,
    test_path: str,
    val_path: str,
    model_path: str,
    wandb_run_name: str,
    config_override_path: Optional[str] = None,
    save_best: bool = False,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load checkpoint + rebuild model with the saved architecture
    raw_model, config, _ = load_model_and_config(model_path, device)

    # Apply YAML overrides (typically training hyperparams, not architecture)
    if config_override_path:
        print(f"\nLoading config overrides from {config_override_path}...")
        with open(config_override_path, "r") as f:
            config_override = yaml.safe_load(f) or {}
        for key, value in config_override.items():
            config[key] = value

    # Force the runtime device — saved configs may carry a stale "cuda" from a
    # different machine.
    config["device"] = device

    # Coerce types after overrides (YAML may parse numbers as strings)
    _coerce_numeric_types(config)

    # Recompute embedder-derived indices from a fresh embedder. Saved indices
    # could be stale if the feature schema evolved between training runs.
    embedder = Embedder(
        format=config.get("battle_format", "gen9vgc2023regc"),
        feature_set=config.get("embedder_feature_set", Embedder.FULL),
        omniscient=False,
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    config["state_input_dim"] = embedder.embedding_size
    print(
        f"Embedder initialized. Embedding[{embedder.embedding_size}] on {config['device']}"
    )

    # Effective batch size via gradient accumulation, matching train.py
    if "batch_size" in config and "worker_batch_size" in config:
        config["accumulation_steps"] = max(
            1, int(config["batch_size"] // config["worker_batch_size"])
        )
    else:
        config["accumulation_steps"] = config.get("accumulation_steps", 1)

    print(f"\nFine-tuning for: {config.get('num_epochs', 10)} epochs\n")

    # Initialize wandb
    wandb_config = config.copy()
    wandb_config["finetuned_from"] = os.path.basename(model_path)

    wandb.init(
        project="elitefurretai-hydreigon",
        name=wandb_run_name,
        config=wandb_config,
    )

    try:
        wandb.save(__file__)
    except OSError as e:
        if "WinError 1314" in str(e) or "privilege" in str(e).lower():
            try:
                wandb.save(__file__, policy="now")
                print("Note: Using file copy instead of symlink for wandb")
            except Exception as copy_error:
                print(f"Warning: Could not save script to wandb: {copy_error}")
        else:
            raise

    # DataLoaders — defaults match the optimal profile in SUPERVISED.md.
    print("Setting up data loaders...")
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

    # torch.compile fuses GPU kernels for ~15-30% speedup on CUDA. Apply AFTER
    # load_state_dict so we don't have to deal with _orig_mod-prefixed keys.
    model: torch.nn.Module = raw_model
    if config["device"] == "cuda":
        model = cast(torch.nn.Module, torch.compile(raw_model))

    wandb.watch(model, log="all", log_freq=1000)

    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Finished loading data and model! Total trainable parameters: {num_params:,}")
    wandb.log({"model_parameters": num_params})

    # Optimizer
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

    # LR schedule (parity with train.py)
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

    print("Initialized model! Starting fine-tuning...")

    start, steps = time.time(), 0
    best_test_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config)

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
            "Train Brier": train_metrics.get("brier", 0.0),
            "Test Loss": test_loss,
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Brier": metrics.get("brier_score", 0.0),
            "Test Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
            "Test Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
            "Test Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
            "Test Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
            "Test Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
            "Test Turn Top1": metrics.get("turn_top1_acc", 0),
            "Test Turn Top3": metrics.get("turn_top3_acc", 0),
            "Test Turn Top5": metrics.get("turn_top5_acc", 0),
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

        if save_best and test_loss < best_test_loss:
            best_test_loss = test_loss
            best_save_dict = {
                "model_state_dict": raw_model.state_dict(),
                "config": config,
            }
            best_path = os.path.join(config["save_path"], f"{wandb.run.name}_best.pt")  # type: ignore
            torch.save(best_save_dict, best_path)
            print(f"New best model saved to {best_path} (test_loss={test_loss:.4f})")

    # Final save — use raw_model.state_dict() to avoid _orig_mod prefixes from torch.compile
    save_dict = {"model_state_dict": raw_model.state_dict(), "config": config}
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
        "Validation Brier": metrics.get("brier_score", 0.0),
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
        state_input_dim=config.get("state_input_dim"),
    )

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    configure_torch_multiprocessing(
        use_file_system_sharing=True,
        filter_socket_send_warning=True,
    )

    parser = argparse.ArgumentParser(
        description="Fine-tune a previously trained TransformerThreeHeadedModel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune with original config
  python fine_tune.py \\
      data/battles/regc_final_v4 \\
      data/models/supervised/curious-darkness-77_best.pt \\
      curious-darkness-77_finetuned

  # Fine-tune with overridden hyperparameters and best-checkpoint saving
  python fine_tune.py \\
      data/battles/regc_final_v4 \\
      data/models/supervised/curious-darkness-77_best.pt \\
      curious-darkness-77_lower_lr \\
      --config-override src/elitefurretai/supervised/configs/finetune.yaml \\
      --save-best
        """,
    )

    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to data directory containing train/, test/, and val/ subdirectories",
    )
    parser.add_argument(
        "model_path", type=str, help="Path to saved model checkpoint (.pt file)"
    )
    parser.add_argument("wandb_run_name", type=str, help="Custom name for the wandb run")
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Optional path to YAML config file to override loaded model config",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the best model checkpoint (lowest test loss) during fine-tuning",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.data_directory):
        parser.error(f"Data directory not found: {args.data_directory}")
    if args.config_override and not os.path.exists(args.config_override):
        parser.error(f"Config override file not found: {args.config_override}")

    for subdir in ["train", "test", "val"]:
        path = os.path.join(args.data_directory, subdir)
        if not os.path.exists(path):
            print(f"Warning: {subdir} directory not found at {path}")

    finetune(
        os.path.join(args.data_directory, "train"),
        os.path.join(args.data_directory, "test"),
        os.path.join(args.data_directory, "val"),
        args.model_path,
        wandb_run_name=args.wandb_run_name,
        config_override_path=args.config_override,
        save_best=args.save_best,
    )
