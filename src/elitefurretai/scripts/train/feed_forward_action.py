import os.path
import random
import sys
import time
from collections import defaultdict

import torch

import wandb
from elitefurretai.model_utils import MDBO, Embedder, PreprocessedBattleDataset


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.BatchNorm1d(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)  # Add ReLU after addition


class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        # Build residual blocks
        for size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, size, dropout))
            prev_size = size

        self.backbone = torch.nn.Sequential(*layers)
        self.action_head = torch.nn.Linear(prev_size, MDBO.action_space())

        # Initialize weights
        for layer in self.backbone:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        torch.nn.init.xavier_normal_(self.action_head.weight)

    def forward(self, x):
        x = self.backbone(x)
        action_logits = self.action_head(x)
        return action_logits


def compute_loss(logits, targets, weights=None, k=1):
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
    running_action_loss = 0
    steps = 0
    start = time.time()
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.to(config["device"])
        actions = actions.to(config["device"])
        action_masks = action_masks.to(config["device"])
        masks = masks.to(config["device"])

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
        mean_batch_loss = compute_loss(masked_action_logits, actions_for_loss, config["k"])

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
        hours = int(time.time() - start) // 3600
        minutes = int(time.time() - start) // 60
        seconds = int(time.time() - start) % 60

        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        est_time_left = (len(dataloader) - num_batches) * time_per_batch
        hours_left = int(est_time_left // 3600)
        minutes_left = int((est_time_left % 3600) // 60)
        seconds_left = int(est_time_left % 60)

        processed = f"Processed {num_batches * dataloader.batch_size} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {hours}h {minutes}m {seconds}s"
        left = f" with an estimated {hours_left}h {minutes_left}m {seconds_left}s left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    hours = int(time.time() - start) // 3600
    minutes = int(time.time() - start) // 60
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

    return running_action_loss / num_batches, steps


@torch.no_grad()
def evaluate(model, dataloader, config):
    model.eval()
    total_top1 = 0
    total_top3 = 0
    total_top5 = 0
    total_samples = 0
    running_action_loss = 0
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.to(config["device"])
        actions = actions.to(config["device"])
        action_masks = action_masks.to(config["device"])
        masks = masks.to(config["device"])

        valid_mask = masks.bool()
        valid_action_rows = action_masks[valid_mask].sum(dim=1) > 0
        if valid_mask.sum() == 0 or valid_action_rows.sum() == 0:
            continue

        # Forward pass
        action_logits = model(states[valid_mask])
        masked_action_logits = action_logits.clone()
        masked_action_logits[~action_masks[valid_mask].bool()] = float("-inf")

        # Mask invalid actions
        masked_action_logits = masked_action_logits[valid_action_rows]
        actions_for_loss = actions[valid_mask][valid_action_rows]

        # no samples after filtering
        if masked_action_logits.size(0) == 0:
            print("no samples after filtering")
            continue

        # Action predictions
        top1_preds = torch.argmax(masked_action_logits, dim=1)
        top3_preds = torch.topk(masked_action_logits, k=3, dim=1).indices
        top5_preds = torch.topk(masked_action_logits, k=5, dim=1).indices

        # Check correctness
        top1_correct = top1_preds == actions_for_loss
        top3_correct = (actions_for_loss.unsqueeze(1) == top3_preds).any(dim=1)
        top5_correct = (actions_for_loss.unsqueeze(1) == top5_preds).any(dim=1)

        # Calculate loss using masked logits
        mean_batch_loss = compute_loss(masked_action_logits, actions_for_loss)

        # Accumulate metrics
        total_top1 += top1_correct.sum().item()
        total_top3 += top3_correct.sum().item()
        total_top5 += top5_correct.sum().item()
        total_samples += valid_action_rows.sum().item()
        running_action_loss += mean_batch_loss.item()
        num_batches += 1

    action_top1 = total_top1 / total_samples if total_samples > 0 else 0
    action_top3 = total_top3 / total_samples if total_samples > 0 else 0
    action_top5 = total_top5 / total_samples if total_samples > 0 else 0
    avg_action_loss = running_action_loss / num_batches if num_batches > 0 else 0
    return action_top1, action_top3, action_top5, avg_action_loss


# TODO: add calculation for turns til end
def analyze(model, dataloader, feature_names):
    model.eval()
    eval_metrics = {"total_top1": 0, "total_top3": 0, "total_top5": 0, "total_samples": 0}
    turns = defaultdict(lambda: eval_metrics.copy())
    ko_can_be_taken = defaultdict(lambda: eval_metrics.copy())
    mons_alive = defaultdict(lambda: eval_metrics.copy())
    available_actions = defaultdict(lambda: eval_metrics.copy())
    action_type = defaultdict(lambda: eval_metrics.copy())
    elos = defaultdict(lambda: eval_metrics.copy())

    feature_names = {name: i for i, name in enumerate(feature_names)}
    ko_features = {v for k, v in feature_names.items() if "KO" in k}

    print("Starting analysis...")
    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.to("mps")
        actions = actions.to("mps")
        action_masks = action_masks.to("mps")
        masks = masks.to("mps")

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
        if valid_action_rows.sum() == 0 or masked_action_logits.size(0) == 0:
            continue

        masked_action_logits = masked_action_logits[valid_action_rows]
        actions_for_loss = actions[valid_mask][valid_action_rows]

        # Action predictions
        top1_preds = torch.argmax(masked_action_logits, dim=1)
        top3_preds = torch.topk(masked_action_logits, k=3, dim=1).indices
        top5_preds = torch.topk(masked_action_logits, k=5, dim=1).indices

        # Check correctness
        top1_correct = top1_preds == actions_for_loss
        top3_correct = (actions_for_loss.unsqueeze(1) == top3_preds).any(dim=1)
        top5_correct = (actions_for_loss.unsqueeze(1) == top5_preds).any(dim=1)

        for i, state in enumerate(states[valid_mask]):

            # Generage Keys for analysis
            elo = int(state[feature_names["p1rating"]] // 100) * 100
            turn = state[feature_names["turn"]].item()
            num_actions = int(action_masks[valid_mask][i].sum().item() / 10) * 10
            turn_type = ""
            action_msg = MDBO.from_int(int(actions[valid_mask][i]), MDBO.TURN).message
            if int(state[feature_names["teampreview"]].item()) == 1:
                turn_type = "teampreview"
            elif any(
                int(state[feature_names[f"MON:{j}:force_switch"]].item()) == 1
                for j in range(6)
            ):
                turn_type = "force_switch"
            elif "switch" in action_msg and "move" in action_msg:
                turn_type = "both"
            elif "move" in action_msg:
                turn_type = "move"
            elif "switch" in action_msg:
                turn_type = "switch"
            else:
                continue  # Skip if we can't determine the turn type

            can_ko = max(state[feature_idx] for feature_idx in ko_features).item()
            num_alive = int(
                8
                - state[feature_names["OPP_NUM_FAINTED"]].item()
                - state[feature_names["NUM_FAINTED"]].item()
            )

            for key, value in zip(
                ["total_top1", "total_top3", "total_top5", "total_samples"],
                [
                    top1_correct[i].item(),
                    top3_correct[i].item(),
                    top5_correct[i].item(),
                    1,
                ],
            ):
                ko_can_be_taken[can_ko][key] += value
                mons_alive[num_alive][key] += value
                turns[turn][key] += value
                available_actions[num_actions][key] += value
                action_type[turn_type][key] += value
                elos[elo][key] += value

    # Print accuracy
    print("Analysis complete! Results:")
    data = [
        (turns, "Turn"),
        (available_actions, "Available Actions"),
        (action_type, "Action Types"),
        (elos, "Elos"),
        (ko_can_be_taken, "Can KO"),
        (mons_alive, "Alive"),
    ]
    for results, name in data:
        print(f"\nAnalysis by {name}:")
        for key in sorted(list(results.keys())):
            metrics = results[key]
            if metrics["total_samples"] > 0:
                top1_acc = metrics["total_top1"] / metrics["total_samples"] * 100
                top3_acc = metrics["total_top3"] / metrics["total_samples"] * 100
                top5_acc = metrics["total_top5"] / metrics["total_samples"] * 100
                print(
                    f"  {key}: Top-1: {top1_acc:.1f}%, Top-3: {top3_acc:.1f}%, Top-5: {top5_acc:.1f}% | {metrics['total_samples']} samples"
                )
            else:
                print(f"\t{key}: No samples")


def main(train_path, test_path, val_path):
    print("Starting!")
    config = {
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
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
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
        action_top1, action_top3, action_top5, action_loss = evaluate(
            model, test_loader, config
        )
        steps += training_steps

        print(f"Epoch #{epoch + 1}:")
        print(f"=> Total Steps:       {steps}")
        print(f"=> Train Loss:        {train_loss:.3f}")
        print(f"=> Test Action Loss:  {action_loss:.3f}")
        print(
            f"=> Test Action Acc:   {(action_top1 * 100):.3f}% (Top-1), {(action_top3 * 100):.3f}% (Top-3) {(action_top5 * 100):.3f}% (Top-5)"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_action_loss": action_loss,
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

        scheduler.step(action_loss)

    torch.save(
        model.state_dict(), os.path.join(config["save_path"], f"{wandb.run.name}.pth")
    )

    print("\nEvaluating on Validation Dataset:")
    action_top1, action_top3, action_top5, action_loss = evaluate(
        model, val_loader, config
    )
    print(f"=> Val Action Loss:   {action_loss:.3f}")
    print(
        f"=> Val Action Acc:    {(action_top1 * 100):.3f}% (Top-1), {(action_top3 * 100):.3f}% (Top-3) {(action_top5 * 100):.3f}% (Top-5)"
    )
    wandb.log(
        {
            "val_action_loss": action_loss,
            "val_action_top1": action_top1,
            "val_action_top3": action_top3,
            "val_action_top5": action_top5,
        }
    )
    analyze(model, val_loader, embedder.feature_names)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
