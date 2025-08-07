import random
import sys
import time

import orjson
import torch

from elitefurretai.model_utils import MDBO, BattleDataset, Embedder


# ---------------------------
# Residual Block Definition
# ---------------------------
class ResidualBlock(torch.nn.Module):
    """
    A standard residual block with optional projection for dimension change.
    Applies: Linear -> BatchNorm -> ReLU -> Dropout -> (Residual Add) -> ReLU
    """

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # If input and output dims differ, use a projection for the shortcut
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
        return self.relu(x + residual)


# ---------------------------
# Model Architecture Diagram
# ---------------------------
# Input: (batch_size, hidden_sizes[0])
# │
# ├─ Backbone (shared)
# │   ├─ ResidualBlock(hidden_sizes[0], hidden_sizes[1])
# │   ├─ ResidualBlock(hidden_sizes[1], hidden_sizes[2])
# │   └─ ResidualBlock(hidden_sizes[2], hidden_sizes[3])
# │
# ├─ Shared Representation: (batch_size, hidden_sizes[3])
# │
# ├─ Action Head
# │   ├─ ResidualBlock(hidden_sizes[3], hidden_sizes[4])
# │   └─ Linear(hidden_sizes[4], action_size)
# │
# ├─ Win Head
# │   ├─ ResidualBlock(hidden_sizes[3], hidden_sizes[4])
# │   └─ Linear(hidden_sizes[4], 1)
# │
# └─ Outputs:
#     ├─ action_logits: (batch_size, action_size)
#     └─ win_logits:    (batch_size,)


# ---------------------------
# MultiTaskDNN Model
# ---------------------------
class MultiTaskDNN(torch.nn.Module):
    """
    Multitask DNN with a shared backbone and two task-specific heads:
    - Action head: predicts the best action (classification)
    - Win head: predicts the probability of winning (binary classification)
    """

    def __init__(
        self,
        hidden_sizes=[2048, 1024, 512, 256, 128],
        action_size=530,
        dropout=0.3,
    ):
        super().__init__()

        # Shared backbone: 3 stacked residual blocks
        self.backbone = torch.nn.Sequential(
            ResidualBlock(hidden_sizes[0], hidden_sizes[1], dropout),
            ResidualBlock(hidden_sizes[1], hidden_sizes[2], dropout),
            ResidualBlock(hidden_sizes[2], hidden_sizes[3], dropout),
        )

        # Action head: another residual block + linear classifier
        self.action_head = torch.nn.Sequential(
            ResidualBlock(hidden_sizes[3], hidden_sizes[4], dropout),
            torch.nn.Linear(hidden_sizes[4], action_size),
        )
        # Win head: another residual block + linear output (single logit)
        self.win_head = torch.nn.Sequential(
            ResidualBlock(hidden_sizes[3], hidden_sizes[4], dropout),
            torch.nn.Linear(hidden_sizes[4], 1),
        )

        # Weight initialization for all Linear layers
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Pass through shared backbone
        x = self.backbone(x)
        # Task-specific heads
        action_logits = self.action_head(x)
        win_logits = self.win_head(x).squeeze(-1)
        return action_logits, win_logits


# ---------------------------
# Loss Computation
# ---------------------------
def compute_loss(action_logits, win_logits, actions, wins, action_masks):
    """
    Computes multitask loss:
    - Action loss: cross-entropy (with label smoothing)
    - Win loss: binary cross-entropy (with logits)
    - Action loss is scaled down to balance with win loss
    """
    action_loss = torch.nn.functional.cross_entropy(
        action_logits, actions, label_smoothing=0.1, reduction="mean"
    )
    win_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        win_logits, wins.float(), reduction="mean"
    )
    # Scale action loss given its many classes vs win's binary task
    scaled_action_loss = action_loss / 7.0
    mean_weighted_loss = scaled_action_loss + win_loss
    return mean_weighted_loss, scaled_action_loss, win_loss


# ---------------------------
# Training Loop (One Epoch)
# ---------------------------
def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """
    Trains the model for one epoch.
    Tracks running losses and prints progress.
    """
    model.train()
    running_loss = 0
    running_action_loss = 0
    running_win_loss = 0
    steps = 0
    start = time.time()
    num_batches = 0

    for batch in dataloader:
        # Unpack and flatten batch
        states, actions, action_masks, wins, masks = batch
        states = states.flatten(0, 1).to(device)
        actions = actions.flatten(0, 1).to(device)
        action_masks = action_masks.flatten(0, 1).to(device)
        wins = wins.flatten(0, 1).to(device)
        masks = masks.flatten(0, 1).to(device)

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward and loss
        action_logits, win_logits = model(states[valid_mask])
        mean_batch_loss, mean_action_loss, mean_win_loss = compute_loss(
            action_logits,
            win_logits,
            actions[valid_mask],
            wins[valid_mask],
            action_masks[valid_mask],
        )

        # Backpropagation
        optimizer.zero_grad()
        mean_batch_loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.win_head.parameters(), max_grad_norm)
        optimizer.step()

        # Track metrics
        running_loss += mean_batch_loss.item()
        running_action_loss += mean_action_loss.item()
        running_win_loss += mean_win_loss.item()
        steps += valid_mask.sum().item()
        num_batches += 1

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

    # Print epoch summary
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

    return (
        running_loss / num_batches,
        running_action_loss / num_batches,
        running_win_loss / num_batches,
        num_batches,
        steps,
    )


# -----------------------------------
# Checks for Vanishing Activations
# -----------------------------------
def check_activations(model, inputs, threshold=0.01):
    """
    Checks for vanishing activations in the model. Not used in this script.
    Returns the number of layers with max abs activation below the threshold.
    """
    with torch.no_grad():
        # Add hook to capture activations
        activations = {}

        def hook(name):
            def fn(module, input, output):
                activations[name] = output

            return fn

        # Register hooks
        handles = []
        for name, layer in model.named_modules():
            handles.append(layer.register_forward_hook(hook(name)))

        # Forward pass
        _ = model(inputs)

        # Check activations
        dead_neurons = 0
        for name, act in activations.items():
            if act.abs().max() < threshold:
                print(f"Vanishing activations in {name}")
                dead_neurons += 1

        # Cleanup
        for handle in handles:
            handle.remove()

        return dead_neurons


# ---------------------------
# Evaluation Loop
# ---------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Computes accuracy and loss for both tasks.
    """
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_win_correct = 0
    total_samples = 0
    running_loss = 0
    running_win_loss = 0
    running_action_loss = 0
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, wins, masks = batch
        states = states.flatten(0, 1).to(device)
        actions = actions.flatten(0, 1).to(device)
        action_masks = action_masks.flatten(0, 1).to(device)
        wins = wins.flatten(0, 1).to(device)
        masks = masks.flatten(0, 1).to(device)

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward pass
        action_logits, win_logits = model(states[valid_mask])

        # Action predictions (top-1 and top-5)
        top1_preds = torch.argmax(
            action_logits, dim=1
        )  # TODO: should I mask the actions first?
        top5_preds = torch.topk(action_logits, k=5, dim=1).indices

        # Check correctness
        top1_correct = top1_preds == actions[valid_mask]
        top5_correct = (actions[valid_mask].unsqueeze(1) == top5_preds).any(dim=1)

        # Win predictions
        win_probs = torch.sigmoid(win_logits)
        win_preds = (win_probs > 0.5).float()
        win_correct = (win_preds == wins[valid_mask]).float()

        # Calculate loss
        mean_batch_loss, mean_action_loss, mean_win_loss = compute_loss(
            action_logits,
            win_logits,
            actions[valid_mask],
            wins[valid_mask],
            action_masks[valid_mask],
        )

        # Accumulate metrics
        total_top1 += top1_correct.sum().item()
        total_top5 += top5_correct.sum().item()
        total_win_correct += win_correct.sum().item()
        total_samples += valid_mask.sum().item()
        running_loss += mean_batch_loss.item()
        running_action_loss += mean_action_loss.item()
        running_win_loss += mean_win_loss.item()
        num_batches += 1

    # Compute averages and accuracies
    action_top1 = total_top1 / total_samples if total_samples > 0 else 0
    action_top5 = total_top5 / total_samples if total_samples > 0 else 0
    win_acc = total_win_correct / total_samples if total_samples > 0 else 0
    avg_loss = running_loss / num_batches if num_batches > 0 else 0
    avg_win_loss = running_win_loss / num_batches if num_batches > 0 else 0
    avg_action_loss = running_action_loss / num_batches if num_batches > 0 else 0
    return action_top1, action_top5, win_acc, avg_loss, avg_action_loss, avg_win_loss


# ---------------------------
# Main Training Script
# ---------------------------
def main(battle_filepath):
    print("Starting!")

    # Load and shuffle battle data
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())
    random.shuffle(files)

    # Split files into train/validation sets (90/10 split)
    train_files = files[: int(len(files) * 0.9)]
    val_files = files[int(len(files) * 0.9) :]

    # Create embedder for feature extraction
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
    )

    # Create datasets
    train_dataset = BattleDataset(files=train_files, embedder=embedder)
    val_dataset = BattleDataset(
        files=val_files,
        embedder=embedder,
    )

    # Create dataloaders for batching
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, num_workers=4, pin_memory=True
    )

    # Initialize model
    model = MultiTaskDNN(
        hidden_sizes=[2048, 1024, 512, 256, 128],
        action_size=MDBO.action_space(),
        dropout=0.3,
    ).to("cpu")

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print("Finished loading data and model! Model Details:")
    print(model)
    print(
        f"for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )
    print(f"Starting training with {10} epochs!")

    # Training loop over epochs
    start, steps = time.time(), 0
    for epoch in range(10):
        # Train for one epoch
        train_loss, running_action_loss, running_win_loss, training_steps = train_epoch(
            model, train_loader, optimizer, "cpu", 1.0
        )
        # Evaluate on validation set
        action_top1, action_top5, win_acc, val_loss, action_loss, win_loss = evaluate(
            model, val_loader, "cpu"
        )
        steps += training_steps

        # Print epoch summary
        print(f"Epoch #{epoch + 1}:")
        print(f"=> Total Steps:       {steps}")
        print(f"=> Train Loss:        {train_loss:.3f}")
        print(f"=> Train Action Loss: {running_action_loss:.3f}")
        print(f"=> Train Win Loss:    {running_win_loss:.3f}")
        print(f"=> Val Loss:          {val_loss:.3f}")
        print(f"=> Val Action Loss:   {action_loss:.3f}")
        print(f"=> Val Win Loss:      {win_loss:.3f}")
        print(
            f"=> Val Action Acc:    {(action_top1 * 100):.3f}% (Top-1), {(action_top5 * 100):.3f}% (Top-5)"
        )
        print(f"=> Val Win Acc:       {(win_acc * 100):.3f}%")

        total_time = time.time() - start
        h, m, s = (
            int(total_time / (60 * 60)),
            int((total_time % (60 * 60)) / 60),
            int(total_time % 60),
        )
        t_left = (9 - epoch) * total_time / (epoch + 1)
        h_left, m_left, s_left = (
            int(t_left / (60 * 60)),
            int((t_left % (60 * 60)) / 60),
            int(t_left % 60),
        )
        print(f"=> Time thus far: {h}h {m}m {s}s // ETA: {h_left}h {m_left}m {s_left}s")
        print()

        # Step the learning rate scheduler
        scheduler.step()


if __name__ == "__main__":
    main(sys.argv[1])
