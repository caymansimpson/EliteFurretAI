"""
forewarn_sweep.py

Description:
    This file implements a sweepable feedforward neural network architecture for supervised Pokémon battle action prediction.
    It uses Weights & Biases (wandb) for hyperparameter optimization via Bayesian sweeps and supports various architectural choices
    such as gated residual blocks, block-wise attention, and flexible backbone/layer configurations. The model is trained and evaluated
    on feature-engineered battle data using the EliteFurretAI embedder and dataset utilities.

    NOTE: This file is a Work In Progress (WIP).
    The current architecture is a feedforward DNN with optional attention and gating, but it should be replaced with the
    transformer-based architecture (see two_headed_transformer.py) which has demonstrated significantly better performance
    for this task.
"""

import random

import orjson
import torch

import wandb
from elitefurretai.model_utils import MDBO, BattleDataset, Embedder


class GatedResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        # Main path
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)

        # Gate generation
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features), torch.nn.Sigmoid()
        )

        # Normalization and regularization
        self.bn1 = torch.nn.BatchNorm1d(out_features)
        self.bn2 = torch.nn.BatchNorm1d(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Projection for residual if dimensions change
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        # Compute gate values
        gate_values = self.gate(x)

        # Main path
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)

        # Apply gating
        gated = gate_values * out

        # Add residual
        residual = self.shortcut(x)
        return self.relu(gated + residual)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Projection shortcut if dimensions change
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


class PokemonBlockAttention(torch.nn.Module):
    def __init__(self, group_sizes, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.group_sizes = group_sizes
        self.num_blocks = len(group_sizes)
        self.embed_dim = embed_dim

        # Project each block to a common embedding size
        self.block_mlps = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(size, embed_dim),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(embed_dim),
                )
                for size in group_sizes
            ]
        )

        # Multihead attention across blocks
        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, sum(group_sizes))
        splits = torch.split(x, self.group_sizes, dim=1)  # List of (batch, group_size)
        block_embeds = [
            mlp(block) for mlp, block in zip(self.block_mlps, splits)
        ]  # Each (batch, embed_dim)
        blocks = torch.stack(block_embeds, dim=1)  # (batch, num_blocks, embed_dim)

        # Self-attention across blocks
        attn_out, _ = self.attn(blocks, blocks, blocks)
        out = self.norm(blocks + attn_out)
        return out  # (batch, num_blocks, embed_dim)


class SweepDNN(torch.nn.Module):
    def __init__(
        self,
        feature_group_sizes,
        action_size,
        dropout,
        gated_residuals,
        early_attention,
        hidden_sizes,
        late_attention,
        late_hidden_sizes,
        attn_embed_dim=256,
        attn_heads=4,
    ):
        super().__init__()
        self.num_blocks = len(feature_group_sizes)
        self.attn_embed_dim = attn_embed_dim

        # Choose block type
        Block = GatedResidualBlock if gated_residuals else ResidualBlock

        # Early Attention
        if early_attention:
            self.early_attn = PokemonBlockAttention(
                group_sizes=feature_group_sizes,
                embed_dim=attn_embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
            )
            backbone_input = attn_embed_dim * self.num_blocks
        else:
            self.early_attn = None
            backbone_input = sum(feature_group_sizes)

        # Shared Backbone
        layers = []
        prev_size = backbone_input
        for size in hidden_sizes:
            layers.append(Block(prev_size, size, dropout))
            prev_size = size
        self.backbone = torch.nn.Sequential(*layers)

        # Late Attention
        if late_attention:
            self.late_attn = torch.nn.MultiheadAttention(
                embed_dim=prev_size // self.num_blocks,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.late_norm = torch.nn.LayerNorm(prev_size // self.num_blocks)
            late_input = prev_size
        else:
            self.late_attn = None
            late_input = prev_size

        # Late Hidden Layers
        late_layers = []
        for size in late_hidden_sizes:
            late_layers.append(Block(late_input, size, dropout))
            late_input = size
        self.late_layers = torch.nn.Sequential(*late_layers)

        # Output head
        self.action_head = torch.nn.Linear(late_input, action_size)

    def forward(self, x):
        # Early attention (if enabled)
        if self.early_attn is not None:
            x = self.early_attn(x)
            x = x.flatten(1)

        # Backbone
        x = self.backbone(x)

        # Late attention (if enabled)
        if self.late_attn is not None:
            batch_size = x.size(0)
            x_blocks = x.view(batch_size, self.num_blocks, -1)
            attn_out, _ = self.late_attn(x_blocks, x_blocks, x_blocks)
            x_blocks = self.late_norm(x_blocks + attn_out)
            x = x_blocks.flatten(1)
        # Late hidden layers
        x = self.late_layers(x)
        # Output
        return self.action_head(x)


def compute_loss(action_logits, actions):
    return (
        torch.nn.functional.cross_entropy(
            action_logits, actions, label_smoothing=0.1, reduction="mean"
        )
        / 7.0
    )


def train_epoch(model, dataloader, prev_steps, optimizer, device, max_grad_norm=1.0):
    model.train()
    running_action_loss = 0
    steps = 0
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.flatten(0, 1).to(device)
        actions = actions.flatten(0, 1).to(device)
        action_masks = action_masks.flatten(0, 1).to(device)
        masks = masks.flatten(0, 1).to(device)

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward pass
        action_logits = model(states[valid_mask])

        # Mask invalid actions for loss calculation
        masked_action_logits = action_logits.clone()
        masked_action_logits[~action_masks[valid_mask].bool()] = float("-inf")

        # Calculate loss using masked logits
        mean_batch_loss = compute_loss(masked_action_logits, actions[valid_mask])

        # Propogate loss
        optimizer.zero_grad()
        mean_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.action_head.parameters(), max_grad_norm)
        optimizer.step()

        # Accumulate batch means
        running_action_loss += mean_batch_loss.item()
        steps += valid_mask.sum().item()
        num_batches += 1

    return running_action_loss / num_batches, steps


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    running_action_loss = 0
    num_batches = 0

    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.to(device)
        actions = actions.to(device)
        action_masks = action_masks.to(device)
        masks = masks.to(device)

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward pass
        action_logits = model(states[valid_mask])

        # Mask invalid actions for both prediction and loss
        masked_action_logits = action_logits.clone()
        masked_action_logits[~action_masks[valid_mask].bool()] = float("-inf")

        # Action predictions
        top1_preds = torch.argmax(masked_action_logits, dim=1)
        top5_preds = torch.topk(masked_action_logits, k=5, dim=1).indices

        # Check correctness
        top1_correct = top1_preds == actions[valid_mask]
        top5_correct = (actions[valid_mask].unsqueeze(1) == top5_preds).any(dim=1)

        # Calculate loss using masked logits
        mean_batch_loss = compute_loss(masked_action_logits, actions[valid_mask])

        # Accumulate metrics
        total_top1 += top1_correct.sum().item()
        total_top5 += top5_correct.sum().item()
        total_samples += valid_mask.sum().item()
        running_action_loss += mean_batch_loss.item()
        num_batches += 1

    action_top1 = total_top1 / total_samples if total_samples > 0 else 0
    action_top5 = total_top5 / total_samples if total_samples > 0 else 0
    avg_action_loss = running_action_loss / num_batches if num_batches > 0 else 0
    return action_top1, action_top5, avg_action_loss


def main(sweep_configuration):
    print("Starting!")

    with wandb.init(project="elitefurretai-forewarn-sweep") as run:

        config = {
            "battle_filepath": "data/battles/supervised_battle_files.json",
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            ),
            "max_grad_norm": 1.0,
            "train_slice": (0, 0.05),
            "val_slice": (0.21, 0.215),
            "weight_decay": 1e-5,
            "num_epochs": 10,
            "save_path": "data/models/",
            "seed": 21,
            "batch_size": run.config["batch_size"],
            "optimizer": run.config["optimizer"],
            "learning_rate": run.config["learning_rate"],
            "dropout": run.config["dropout"],
            "gated_residuals": run.config["gated_residuals"],
            "early_attention": run.config["early_attention"],
            "hidden_sizes": run.config["hidden_sizes"],
            "late_attention": run.config["late_attention"],
            "late_hidden_sizes": run.config["late_hidden_sizes"],
        }

        # Set Seeds
        torch.cuda.manual_seed_all(config["seed"])
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])

        # Prepare everything we need to initalize model
        files = []
        with open(config["battle_filepath"], "rb") as f:
            files = orjson.loads(f.read())
        random.shuffle(files)

        # Split files
        train_files = files[
            int(len(files) * config["train_slice"][0]) : int(
                len(files) * config["train_slice"][1]
            )
        ]
        val_files = files[
            int(len(files) * config["val_slice"][0]) : int(
                len(files) * config["val_slice"][1]
            )
        ]

        embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
        )

        # Create datasets
        train_dataset = BattleDataset(files=train_files, embedder=embedder)
        val_dataset = BattleDataset(files=val_files, embedder=embedder)

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config["batch_size"], num_workers=4, pin_memory=True
        )

        model = SweepDNN(
            feature_group_sizes=embedder.feature_group_sizes,
            action_size=MDBO.action_space(),
            dropout=config["dropout"],
            gated_residuals=config["gated_residuals"],
            early_attention=config["early_attention"],
            hidden_sizes=config["hidden_sizes"],
            late_attention=config["late_attention"],
            late_hidden_sizes=config["late_hidden_sizes"],
            attn_embed_dim=256,
            attn_heads=4,
        ).to(config["device"])

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.999),
        )
        if config["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
                betas=(0.9, 0.999),
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )

        print(
            f"Finished loading data and model! for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
        run.watch(model, log="all", log_freq=100)
        print(f"Starting training with {config['num_epochs']} epochs!")

        # Assuming you have train_loader and val_loader from your dataset
        steps = 0
        for epoch in range(config["num_epochs"]):
            train_loss, training_steps = train_epoch(
                model,
                train_loader,
                steps,
                optimizer,
                config["device"],
                config["max_grad_norm"],
            )
            action_top1, action_top5, action_loss = evaluate(
                model, val_loader, config["device"]
            )
            steps += training_steps

            run.log(
                {
                    "epoch": epoch + 1,
                    "steps": steps,
                    "train_action_loss": train_loss,
                    "val_action_loss": action_loss,
                    "val_action_top1": action_top1,
                    "val_action_top5": action_top5,
                }
            )

            scheduler.step(action_loss)


if __name__ == "__main__":

    sweep_configuration = {
        "method": "bayes",
        "name": "action_sweep",
        "metric": {
            "goal": "maximize",
            "name": "val_action_top3",
        },  # Choosing something we're not computing loss for to ensure generalized and non-overconfident results
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
            "early_attention": {"values": [True, False]},
            "hidden_sizes": {
                "values": [
                    [2048, 1024, 512],
                    [1024, 512],
                ]
            },
            "late_attention": {"values": [True, False]},
            "late_hidden_sizes": {
                "values": [
                    [256, 128],
                    [256],
                    [],
                ]
            },
        },
    }

    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="elitefurretai-forewarn-sweep"
    )
    print(f"Sweep ID: {sweep_id}")

    # Start the sweep job
    wandb.agent(sweep_id, function=main, project="elitefurretai-forewarn-sweep")

    # I have to run the following in the CLI to parallelize:
    # CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>
    # CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID>
    # CUDA_VISIBLE_DEVICES=2 wandb agent <SWEEP_ID>
    # CUDA_VISIBLE_DEVICES=3 wandb agent <SWEEP_ID>
