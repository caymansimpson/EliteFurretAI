# -*- coding: utf-8 -*-
"""
This script trains a supervised model for move and win prediction using a flexible architecture.
It supports hyperparameter sweeping through Weights & Biases to find optimal configurations.

Usage:
    python two_headed_sweep.py <train_path> <test_path> <val_path>
"""
import argparse
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import wandb
from elitefurretai.model_utils import MDBO, Embedder, OptimizedBattleDataLoader
from elitefurretai.model_utils.train_utils import (
    evaluate,
    flatten_and_filter,
    format_time,
    topk_cross_entropy_loss,
)


class ResidualBlock(torch.nn.Module):
    """
    Residual block without second ReLU to allow negative values.
    Architecture: Linear → LayerNorm → ReLU → Dropout → Add residual
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Initialize
        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear.bias, 0)

        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            shortcut_linear = torch.nn.Linear(in_features, out_features)
            torch.nn.init.kaiming_normal_(shortcut_linear.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(shortcut_linear.bias, 0)
            self.shortcut = torch.nn.Sequential(
                shortcut_linear,
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual  # No second ReLU - allows negative values


class GatedResidualBlock(torch.nn.Module):
    """
    Gated residual block without second ReLU.
    Architecture: Linear → LN → ReLU → Dropout → Linear → LN → Gate → Add residual
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        # Main path
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.ln1 = torch.nn.LayerNorm(out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)
        self.ln2 = torch.nn.LayerNorm(out_features)

        # Gate generation
        gate_linear = torch.nn.Linear(in_features, out_features)
        self.gate = torch.nn.Sequential(
            gate_linear, torch.nn.Sigmoid()
        )

        # Regularization
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Initialize
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_normal_(gate_linear.weight, gain=1.0)
        torch.nn.init.constant_(gate_linear.bias, 0)

        # Projection for residual if dimensions change
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            shortcut_linear = torch.nn.Linear(in_features, out_features)
            torch.nn.init.kaiming_normal_(shortcut_linear.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(shortcut_linear.bias, 0)
            self.shortcut = torch.nn.Sequential(
                shortcut_linear,
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # Main path
        main = self.linear1(x)
        main = self.ln1(main)
        main = self.relu(main)
        main = self.dropout(main)
        main = self.linear2(main)
        main = self.ln2(main)

        # Apply gate
        gate_value = self.gate(x)
        gated_output = gate_value * main

        return residual + gated_output  # No final ReLU


class GroupedFeatureEncoder(torch.nn.Module):
    """
    Encodes features by semantic groups (Pokemon, opponent Pokemon, battle state).
    Includes cross-attention for Pokemon features to learn team synergies.

    Input: (batch, seq, full_feature_dim)
    Output: (batch, seq, aggregated_dim)

    Groups from embedder.group_embedding_sizes:
        - Player Pokemon 0-5: [pokemon_emb_size] * 6
        - Opponent Pokemon 0-5: [opp_pokemon_emb_size] * 6
        - Battle state: [battle_emb_size]
        - Engineered features: [feature_eng_emb_size]
    """
    def __init__(self, group_sizes, hidden_dim=128, aggregated_dim=1024, dropout=0.1):
        super().__init__()
        self.group_sizes = group_sizes
        self.hidden_dim = hidden_dim

        # Per-group encoders
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(size, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            for size in group_sizes
        ])

        # Initialize per-group encoders
        for encoder in self.encoders:
            linear_layer = encoder[0]  # type: ignore
            torch.nn.init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(linear_layer.bias, 0)

        # Cross-attention for player Pokemon (first 6 groups)
        # Allows Pokemon to attend to each other (e.g., Incineroar + Rillaboom synergy)
        self.pokemon_cross_attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=2, batch_first=True, dropout=dropout
        )
        self.pokemon_norm = torch.nn.LayerNorm(hidden_dim)

        # Aggregate all group embeddings
        self.aggregator = torch.nn.Linear(hidden_dim * len(group_sizes), aggregated_dim)
        torch.nn.init.xavier_normal_(self.aggregator.weight)
        torch.nn.init.constant_(self.aggregator.bias, 0)

    def forward(self, x):
        # x: (batch, seq, full_feature_dim)
        batch, seq, _ = x.shape

        # Encode each group
        group_features = []
        start_idx = 0
        for encoder, size in zip(self.encoders, self.group_sizes):
            group = x[:, :, start_idx:start_idx + size]
            group_features.append(encoder(group))
            start_idx += size

        # Cross-attention among player Pokemon (first 6 groups)
        # This helps the model learn team compositions and synergies
        player_pokemon = torch.stack(group_features[:6], dim=2)  # (batch, seq, 6, hidden)
        player_pokemon_flat = player_pokemon.reshape(batch * seq, 6, -1)  # (batch*seq, 6, hidden)

        attn_out, _ = self.pokemon_cross_attn(
            player_pokemon_flat,
            player_pokemon_flat,
            player_pokemon_flat
        )  # (batch*seq, 6, hidden)

        attn_out = attn_out.reshape(batch, seq, 6, -1)  # (batch, seq, 6, hidden)

        # Apply residual connection and normalization
        for i in range(6):
            group_features[i] = self.pokemon_norm(group_features[i] + attn_out[:, :, i, :])

        # Concatenate all groups and aggregate
        concatenated = torch.cat(group_features, dim=-1)  # (batch, seq, hidden * num_groups)
        return self.aggregator(concatenated)  # (batch, seq, aggregated_dim)


class FlexibleTwoHeadedModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        early_layers: List[int],
        late_layers: List[int],
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        num_actions: int = MDBO.action_space(),
        max_seq_len: int = 40,
        dropout: float = 0.1,
        gated_residuals: bool = False,
        early_attention: bool = False,
        late_attention: bool = True,
        use_grouped_encoder: bool = False,
        group_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.use_grouped_encoder = use_grouped_encoder
        self.hidden_size = early_layers[-1] if early_layers else input_size
        self.num_actions = num_actions
        self.early_attention = early_attention
        self.late_attention = late_attention

        # Select residual block type
        ResBlock = GatedResidualBlock if gated_residuals else ResidualBlock

        # Feature encoding: either grouped or simple linear
        if use_grouped_encoder and group_sizes is not None:
            self.feature_encoder: Optional[torch.nn.Module] = GroupedFeatureEncoder(
                group_sizes=group_sizes,
                hidden_dim=128,
                aggregated_dim=early_layers[0],
                dropout=dropout
            )
            # early_ff_stack processes [early_layers[0] → early_layers[1] → ...]
            early_ff_layers = []
            prev_size = early_layers[0]
            for h in early_layers[1:]:
                early_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )
        else:
            # Simple linear projection
            self.feature_encoder = None
            input_proj = torch.nn.Linear(input_size, early_layers[0])
            torch.nn.init.kaiming_normal_(input_proj.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(input_proj.bias, 0)
            self.input_proj = input_proj

            # early_ff_stack processes [early_layers[0] → early_layers[1] → ...]
            early_ff_layers = []
            prev_size = early_layers[0]
            for h in early_layers[1:]:
                early_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )

        # Early attention if enabled
        if early_attention:
            self.early_attn = torch.nn.MultiheadAttention(
                self.hidden_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.early_ln = torch.nn.LayerNorm(self.hidden_size)

        # Positional encoding (learned)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM (no projection - uses natural bidirectional output)
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        # No lstm_proj - bidirectional output is 2 * hidden_size

        # Skip connection projection (only if grouped encoder is used)
        if use_grouped_encoder:
            self.skip_proj: Optional[torch.nn.Module] = torch.nn.Linear(self.hidden_size, self.hidden_size * 2)
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention if enabled (operates on LSTM output size)
        lstm_output_size = self.hidden_size * 2  # Bidirectional
        if late_attention:
            self.late_attn = torch.nn.MultiheadAttention(
                lstm_output_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln = torch.nn.LayerNorm(lstm_output_size)

        # Build late feedforward stack with residual blocks
        late_ff_layers = []
        prev_size = lstm_output_size
        for h in late_layers:
            late_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers) if late_ff_layers else torch.nn.Identity()
        )

        # Output size after late layers
        output_size = late_layers[-1] if late_layers else lstm_output_size

        # Action head
        self.action_head = torch.nn.Linear(output_size, num_actions)
        torch.nn.init.xavier_normal_(self.action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.action_head.bias, 0)

        # Win prediction head - LayerNorm instead of ReLU to allow negative values
        win_linear1 = torch.nn.Linear(output_size, 128)
        win_linear2 = torch.nn.Linear(128, 1)

        # Initialize with small values to prevent explosion
        torch.nn.init.xavier_normal_(win_linear1.weight, gain=0.01)
        torch.nn.init.constant_(win_linear1.bias, 0)
        torch.nn.init.xavier_normal_(win_linear2.weight, gain=0.01)
        torch.nn.init.constant_(win_linear2.bias, 0)

        self.win_head = torch.nn.Sequential(
            win_linear1,
            torch.nn.LayerNorm(128),  # LayerNorm instead of ReLU
            torch.nn.Dropout(dropout),
            win_linear2,
            torch.nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Feature encoding
        if self.feature_encoder is not None:
            # Grouped encoder path
            ff_out_early = self.feature_encoder(x)  # (batch, seq, early_layers[0])
        else:
            # Simple linear projection path
            ff_out_early = self.input_proj(x)  # (batch, seq, early_layers[0])

        # Early feedforward stack
        ff_out_early = self.early_ff_stack(ff_out_early)  # (batch, seq, hidden_size)

        # Early attention if enabled
        if self.early_attention:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask)
            ff_out_early = self.early_ln(ff_out_early + attn_out)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x_pos = ff_out_early + self.pos_embedding(positions)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM (packed)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_pos, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        # lstm_out is (batch, seq, hidden_size * 2) - no projection needed

        # Skip connection (only if grouped encoder is used)
        if self.skip_proj is not None:
            ff_out_early_proj = self.skip_proj(ff_out_early)  # (batch, seq, hidden_size) → (batch, seq, hidden_size * 2)
            lstm_out = lstm_out + ff_out_early_proj

        # Late attention if enabled
        if self.late_attention:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.late_attn(
                lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
            )
            out = self.late_ln(lstm_out + attn_out)
        else:
            out = lstm_out

        # Late feedforward stack
        out = self.late_ff_stack(out)

        # Output heads
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len), values in [-1, 1]

        return action_logits, win_logits

    def predict(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs, win_logits


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    prev_steps: int,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    scaler: Optional[torch.amp.GradScaler] = None,  # type: ignore
) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_win_loss = 0.0
    steps = 0
    num_batches = 0
    start = time.time()

    for batch in dataloader:
        # Transfer data to the right device
        if config['device'] == 'cuda':
            batch = {k: v.cuda(non_blocking=False) for k, v in batch.items()}
        else:
            batch = {k: v.to(config['device']) for k, v in batch.items()}

        states = batch["states"]
        actions = batch["actions"]
        action_masks = batch["action_masks"]
        wins = batch["wins"]
        masks = batch["masks"]

        # Mixed precision training (CUDA only)
        autocast = torch.amp.autocast if config['device'] == 'cuda' else torch.autocast  # type: ignore
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

            # Usually I would set weights=None, but I want to only train on turns
            # and not force_switch nor teampreview
            weights = torch.ones(valid_states.shape[0], device=states.device)
            teampreview_mask = valid_states[:, config["teampreview_idx"]] == 1
            force_switch_mask = torch.stack(
                [valid_states[:, idx] == 1 for idx in config["force_switch_indices"]],
                dim=-1,
            ).any(dim=-1)
            turn_mask = ~(teampreview_mask | force_switch_mask)
            weights[teampreview_mask] = 1
            weights[force_switch_mask] = 1
            weights[turn_mask] = 1

            # Losses
            action_loss = topk_cross_entropy_loss(
                valid_action_logits, valid_actions, weights=weights, k=3
            )
            win_loss = torch.nn.functional.mse_loss(valid_win_logits, valid_wins.float())
            loss = config["action_weight"] * action_loss + config["win_weight"] * win_loss

        # Backpropagation with mixed precision
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_action_loss += action_loss.item()
        running_win_loss += win_loss.item()
        steps += valid_actions.size(0)
        num_batches += 1

        # Logging progress
        if num_batches % 100 == 0:
            wandb.log({"Total Steps": prev_steps + steps, "grad_norm": grad_norm.item()})

        if num_batches % 10 == 0:
            wandb.log(
                {
                    "Total Steps": prev_steps + steps,
                    "train_loss": running_loss / num_batches,
                    "train_action_loss": running_action_loss / num_batches,
                    "train_win_loss": running_win_loss / num_batches,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Print progress
        assert dataloader.batch_size is not None
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


def train_model(
    config: Dict[str, Any], train_path: str, test_path: str,
) -> Dict[str, float]:
    """Train a model with the given configuration and return metrics."""
    print(f"Starting training with config: {config}")

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    random.seed(config["seed"])

    # Initialize embedder
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print(f"Embedder initialized. Embedding size: {embedder.embedding_size:,}")

    # Load datasets with OptimizedBattleDataLoader
    print("Loading datasets...")
    start = time.time()

    train_loader = OptimizedBattleDataLoader(train_path, embedder=embedder, batch_size=config["batch_size"])
    test_loader = OptimizedBattleDataLoader(test_path, embedder=embedder, batch_size=config["batch_size"], num_workers=0, prefetch_factor=None)

    # Initialize model
    model = FlexibleTwoHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        num_attention_heads=config["num_attention_heads"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        gated_residuals=config["gated_residuals"],
        early_attention=config["early_attention"],
        late_attention=config["late_attention"],
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=embedder.group_embedding_sizes if config.get("use_grouped_encoder", False) else None,
        num_actions=MDBO.action_space(),
        max_seq_len=40,
    ).to(config["device"])

    # Log model size
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {trainable_params:,} trainable parameters")
    wandb.log({"model_parameters": trainable_params})

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
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if config['device'] == 'cuda' else None  # type: ignore

    print("Initialized model! Starting training...")

    # Training loop
    start, steps = time.time(), 0
    final_log = {}
    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config, scaler)
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
        }
        final_log = log

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

    # Return metrics for the sweep
    return final_log


def sweep_agent(train_path: str, test_path: str) -> None:
    """WandB sweep agent function."""
    # Initialize sweep configuration
    sweep_config = {
        "method": "bayes",
        "name": "win_sweep_restrained",
        "metric": {
            "goal": "maximize",
            "name": "Test Action Top 3",
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
        },
        "parameters": {
            "learning_rate": {"values": [0.001, 0.0001]},
            "dropout": {"values": [0, 0.1]},
            "weight_decay": {"values": [0, 1e-5, 1e-2]},
            "early_layers": {
                "values": [
                    [2048, 1024, 1024, 512],
                    [1024, 512],
                ]
            },
            "late_layers": {
                "values": [
                    [512, 256],
                    [],
                ]
            },
            "gated_residuals": {"values": [True, False]},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="elitefurretai-scovillain-gpu")

    # Define the training function
    def train_sweep():
        # Initialize a new run
        wandb.init()

        # Get hyperparameters
        config = wandb.config

        # Add fixed parameters
        config.update(
            {
                "num_epochs": 10,
                "batch_size": 512,
                "lstm_layers": 2,
                "early_attention": True,
                "action_weight": 0.1,
                "num_attention_heads": 8,
                "win_weight": 0.9,
                "use_grouped_encoder": True,
                "max_grad_norm": 1.0,
                "optimizer": "Adam",
                "late_attention": True,
                "device": (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                ),
                "save_path": "data/models/",
                "seed": 21,
                "weights": False,
            }
        )

        # Train the model with this config
        return train_model(config, train_path, test_path)  # type: ignore

    # Run the agent
    wandb.agent(sweep_id, function=train_sweep)


def main():
    parser = argparse.ArgumentParser(
        description="Train a two-headed hybrid model for Pokemon battle prediction."
    )
    parser.add_argument("train_path", type=str, help="Path to training data")
    parser.add_argument("test_path", type=str, help="Path to testing data")

    args = parser.parse_args()
    sweep_agent(args.train_path, args.test_path)


if __name__ == "__main__":
    main()
