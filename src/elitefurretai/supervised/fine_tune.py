# -*- coding: utf-8 -*-
"""
Fine-tune a previously trained three-headed transformer model.

This script:
1. Loads a saved model checkpoint with its embedded config
2. Optionally overrides config parameters with a new config file
3. Trains the loaded model with additional params
4. Saves the fine-tuned model with a new name

Usage:
    python finetune_three_headed_transformer.py <data_directory> <model_path> <wandb_run_name> [config_override_path]

The data_directory should contain train/, test/, and val/ subdirectories.
The config_override file (if provided) will override matching keys from the model's saved config.
"""

import os
import time
import argparse
import torch
import wandb
import orjson
from typing import Optional

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl import OptimizedBattleDataLoader
from elitefurretai.supervised.train_utils import evaluate, format_time

# Import the model and training components from the original script
from elitefurretai.supervised.three_headed_transformer import (
    FlexibleThreeHeadedModel,
    train_epoch,
)


def load_model_and_config(model_path: str, device: str):
    """
    Load a saved model checkpoint and its configuration.

    Args:
        model_path: Path to the saved .pt file
        device: Device to load model onto ('cuda' or 'cpu')

    Returns:
        Tuple of (model, config)
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError(f"Model checkpoint at {model_path} does not contain config. Please ensure the model was saved with both model_state_dict and config.")

    config = checkpoint['config']
    print(f"Loaded config with {len(config)} parameters")

    # Create embedder (same as in training)
    embedder = Embedder(
        format="gen9vgc2023regulationc",
        feature_set=Embedder.FULL,
        omniscient=False
    )

    # Create model with same architecture
    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config["lstm_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
        dropout=config["dropout"],
        gated_residuals=config["gated_residuals"],
        early_attention_heads=config["early_attention_heads"],
        late_attention_heads=config["late_attention_heads"],
        use_grouped_encoder=config["use_grouped_encoder"],
        group_sizes=embedder.group_embedding_sizes if config["use_grouped_encoder"] else None,
        grouped_encoder_hidden_dim=config["grouped_encoder_hidden_dim"],
        grouped_encoder_aggregated_dim=config["grouped_encoder_aggregated_dim"],
        pokemon_attention_heads=config["pokemon_attention_heads"],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        teampreview_head_layers=config["teampreview_head_layers"],
        teampreview_head_dropout=config["teampreview_head_dropout"],
        teampreview_attention_heads=config["teampreview_attention_heads"],
        turn_head_layers=config["turn_head_layers"],
        max_seq_len=17,
    ).to(device)

    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
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
):
    """
    Fine-tune a pre-trained model for additional epochs.

    Args:
        train_path: Path to training data
        test_path: Path to test data
        val_path: Path to validation data
        model_path: Path to saved model checkpoint
        wandb_run_name: Custom name for wandb run (optional)
        num_epochs: Number of additional epochs to train (default: 10)
        config_override_path: Optional path to JSON config file to override loaded config
    """
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and config
    model, config, embedder = load_model_and_config(model_path, device)

    # Override config with new values if provided
    if config_override_path:
        print(f"\nLoading config overrides from {config_override_path}...")
        with open(config_override_path, 'rb') as f:
            config_override = orjson.loads(f.read())

        for key, value in config_override.items():
            config[key] = value

    # Override num_epochs with the fine-tuning value
    original_epochs = config.get("num_epochs", "unknown")
    config["num_epochs"] = config.get("num_epochs", 10)
    print(f"\nOriginal training: {original_epochs} epochs")
    print(f"Fine-tuning for: {config['num_epochs']} additional epochs\n")

    # Initialize wandb with custom run name
    wandb_config = config.copy()
    wandb_config["finetuned_from"] = os.path.basename(model_path)
    wandb_config["original_epochs"] = original_epochs

    wandb.init(
        project="elitefurretai-hydreigon",  # Same project as original training
        name=wandb_run_name,
        config=wandb_config,
        settings=wandb.Settings(
            x_service_wait=30,
            start_method="thread"
        )
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

    # Set up data loaders
    print("Setting up data loaders...")
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader = OptimizedBattleDataLoader(
        train_path,
        embedder=embedder,
        num_workers=7,
        prefetch_factor=8,
        files_per_worker=3,
        persistent_workers=True,
        pin_memory=False,
    )
    test_loader = OptimizedBattleDataLoader(
        test_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=4,
        prefetch_factor=2,
        files_per_worker=1
    )
    val_loader = OptimizedBattleDataLoader(
        val_path,
        embedder=embedder,
        batch_size=config["worker_batch_size"],
        num_workers=4,
        prefetch_factor=2,
        files_per_worker=1
    )

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if config['device'] == 'cuda' else None  # type: ignore

    print("Initialized model! Starting training...")

    # Training loop
    start, steps = time.time(), 0
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

        scheduler.step(
            float(
                metrics["win_mse"] * config["win_loss_weight"]
                + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
                + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
            )
        )

    # Save model with config embedded
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
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

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a previously trained three-headed transformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune with original config
  python finetune_three_headed_transformer.py \\
      data/battles/regc_trajectory \\
      data/models/my_model.pt \\
      my_model_finetuned

  # Fine-tune with modified hyperparameters
  python finetune_three_headed_transformer.py \\
      data/battles/regc_trajectory \\
      data/models/my_model.pt \\
      my_model_with_lower_lr \\
      --config-override config_override.cfg
        """
    )

    parser.add_argument(
        "data_directory",
        type=str,
        help="Path to data directory containing train/, test/, and val/ subdirectories"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model checkpoint (.pt file)"
    )
    parser.add_argument(
        "wandb_run_name",
        type=str,
        help="Custom name for the wandb run"
    )
    parser.add_argument(
        "--config-override",
        type=str,
        default=None,
        help="Optional path to JSON config file to override loaded model config"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    if not os.path.exists(args.data_directory):
        parser.error(f"Data directory not found: {args.data_directory}")

    if args.config_override and not os.path.exists(args.config_override):
        parser.error(f"Config override file not found: {args.config_override}")

    # Verify subdirectories exist
    for subdir in ['train', 'test', 'val']:
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
    )
