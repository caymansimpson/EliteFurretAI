import os.path
import random
import sys
import time
from typing import Any, Dict

import torch

import wandb
from elitefurretai.etl import Embedder, PreprocessedBattleDataset
from elitefurretai.supervised.model_archs import DNN
from elitefurretai.supervised.train_utils import (
    analyze,
    evaluate,
    format_time,
    topk_cross_entropy_loss,
)


def train_epoch(model, dataloader, prev_steps, optimizer, config):
    model.train()
    running_action_loss = 0.0
    steps = 0
    start = time.time()
    num_batches = 0

    for batch in dataloader:
        states = batch["states"].to(config["device"])
        actions = batch["actions"].to(config["device"])
        action_masks = batch["action_masks"].to(config["device"])
        masks = batch["masks"].to(config["device"])

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward pass
        action_logits = model(states[valid_mask])

        # Mask invalid actions
        masked_action_logits = action_logits.clone()
        masked_action_logits[~action_masks[valid_mask].bool()] = float("-inf")

        # Filter out samples with no valid actions
        valid_action_rows = action_masks[valid_mask].sum(dim=1) > 0
        if valid_action_rows.sum() == 0:
            continue

        masked_action_logits = masked_action_logits[valid_action_rows]
        actions_for_loss = actions[valid_mask][valid_action_rows]

        # Build weights for loss
        weights = torch.ones(states[valid_mask].shape[0], device=states.device)
        teampreview_mask = states[valid_mask, config["teampreview_idx"]] == 1
        force_switch_mask = torch.stack(
            [states[valid_mask, idx] == 1 for idx in config["force_switch_indices"]],
            dim=-1,
        ).any(dim=-1)
        turn_mask = ~(teampreview_mask | force_switch_mask)
        weights[teampreview_mask] = 8.55
        weights[force_switch_mask] = 125
        weights[turn_mask] = 1.14

        # Calculate loss using masked logits
        mean_batch_loss = topk_cross_entropy_loss(
            masked_action_logits, actions_for_loss, weights, config["k"]
        )

        # Propogate loss
        optimizer.zero_grad()
        mean_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.action_head.parameters(), config["max_grad_norm"]
        )
        optimizer.step()

        # Accumulate batch means
        running_action_loss += mean_batch_loss.item()
        steps += valid_mask.sum().item()
        num_batches += 1

        # Logging progress
        wandb.log(
            {
                "steps": prev_steps + steps,
                "train_mean_action_loss": running_action_loss / num_batches,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # Print progress
        t = int(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        t_left = (len(dataloader) - num_batches) * time_per_batch

        processed = f"Processed {num_batches * dataloader.batch_size} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {format_time(t)}"
        left = f" with an estimated {format_time(t_left)} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    print(f"\033[2K\rDone training in {format_time(int(time.time() - start))}!")

    return running_action_loss / num_batches, steps


def main(train_path, test_path, val_path):
    print("Starting!")
    config: Dict[str, Any] = {
        "device": (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        ),
        "batch_size": 4096,
        "learning_rate": 1e-4,
        "dropout": 0.3,
        "hidden_sizes": [1024, 512, 256, 128],
        "weight_decay": 1e-4,
        "max_grad_norm": 2.0,
        "k": 3,
        "num_epochs": 20,
        "save_path": "data/models/",
        "seed": 21,
        "notes": "forewarn with Esmaller funnel and lower dropout",
    }
    wandb.init(project="elitefurretai-forewarn", config=config)
    wandb.save(__file__)

    # Set Seeds
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    random.seed(config["seed"])

    embedder = Embedder(
        format="gen9vgc2023regc", feature_set=Embedder.FULL, omniscient=True
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]

    print("Loading datasets...")
    start = time.time()

    # Create datasets
    train_dataset = PreprocessedBattleDataset(train_path, embedder=embedder)
    test_dataset = PreprocessedBattleDataset(test_path, embedder=embedder)
    val_dataset = PreprocessedBattleDataset(val_path, embedder=embedder)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    print(f"Loaded datasets in {round(time.time() - start, 2)}s")

    model = DNN(
        input_size=embedder.embedding_size,
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    ).to(config["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    print(
        f"Finished loading data and model! for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    wandb.watch(model, log="all", log_freq=100)
    print(f"Starting training with {config['num_epochs']} epochs!")

    # Assuming you have train_loader and val_loader from your dataset
    start, steps = time.time(), 0
    for epoch in range(config["num_epochs"]):
        train_loss, training_steps = train_epoch(
            model, train_loader, steps, optimizer, config
        )

        metrics = evaluate(
            model,
            test_loader,
            device=config["device"],
            has_action_head=True,
            has_win_head=False,
            has_move_order_head=False,
            has_ko_head=False,
            has_switch_head=False,
        )
        steps += training_steps

        # Format metrics nicely for console output
        print(f"Epoch #{epoch + 1}:")
        print(f"=> Total Steps              : {steps}")
        print(f"=> Train Loss               : {train_loss:.4f}")
        print(f"=> Test Action Loss         : {metrics['top3_loss']:.4f}")
        print(f"=> Test Action Acc (Top-1)  : {(metrics['top1_acc'] * 100):.2f}%")
        print(f"=> Test Action Acc (Top-3)  : {(metrics['top3_acc'] * 100):.2f}%")
        print(f"=> Test Action Acc (Top-5)  : {(metrics['top5_acc'] * 100):.2f}%")

        # Updated to use new metric names from evaluate function
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_action_loss": metrics["top3_loss"],
                "test_action_top1": metrics["top1_acc"],
                "test_action_top3": metrics["top3_acc"],
                "test_action_top5": metrics["top5_acc"],
            }
        )

        total_time = time.time() - start
        t_left = (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        print(
            f"=> Time thus far: {format_time(total_time)}s // ETA: {format_time(t_left)}"
        )
        print()

        scheduler.step(float(metrics["top3_loss"]))

    print("Done training! Saving model...")
    torch.save(
        model.state_dict(),
        os.path.join(config["save_path"], f"{wandb.run.name}.pth"),  # type: ignore
    )

    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(
        model,
        val_loader,
        device=config["device"],
        has_action_head=True,
        has_win_head=False,
        has_move_order_head=False,
        has_ko_head=False,
        has_switch_head=False,
    )
    print(f"=> Val Action Loss          : {metrics['top3_loss']:.4f}")
    print(f"=> Val Action Acc (Top-1)   : {(metrics['top1_acc'] * 100):.2f}%")
    print(f"=> Val Action Acc (Top-3)   : {(metrics['top3_acc'] * 100):.2f}%")
    print(f"=> Val Action Acc (Top-5)   : {(metrics['top5_acc'] * 100):.2f}%")

    wandb.log(
        {
            "val_action_loss": metrics["top3_loss"],
            "val_action_top1": metrics["top1_acc"],
            "val_action_top3": metrics["top3_acc"],
            "val_action_top5": metrics["top5_acc"],
        }
    )
    print("\nAnalyzing on Validation Dataset:")
    analyze(
        model,
        val_loader,
        device=config["device"],
        has_action_head=True,
        has_win_head=False,
        has_move_order_head=False,
        has_ko_head=False,
        has_switch_head=False,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
