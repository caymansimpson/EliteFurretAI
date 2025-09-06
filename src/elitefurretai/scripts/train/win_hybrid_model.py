"""
train_entity_value_model.py

End-to-end training script for:
  - EntityTemporalValueModel (entity + temporal Transformer)
  - Uses BattleDatasetWithAux for supervised imitation (action) + per-step value + auxiliary tasks
"""

import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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


class EntityTemporalValueModel(torch.nn.Module):
    def __init__(
        self,
        group_sizes: List[int],
        num_actions: int,
        max_seq_len: int = 17,
        cfg: Dict[str, Any] = {},
    ):
        super().__init__()
        self.group_sizes = group_sizes
        self.num_groups = len(group_sizes)
        self.num_actions = num_actions
        self.max_seq_len = max_seq_len
        self.cfg = cfg

        d_entity = cfg["d_entity"]
        d_model = cfg["d_model"]

        # Per-group projections
        self.group_offsets = [0]
        acc = 0
        for gs in group_sizes[:-1]:
            acc += gs
            self.group_offsets.append(acc)

        self.group_proj = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(gs, d_entity),
                    torch.nn.LayerNorm(d_entity),
                    torch.nn.GELU(),
                )
                for gs in group_sizes
            ]
        )

        # Intra-step encoder (entity-level)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_entity,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["ff_mult"] * d_entity,
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.intra_encoder = torch.nn.TransformerEncoder(
            enc_layer, num_layers=cfg["intra_layers"], norm=torch.nn.LayerNorm(d_entity)
        )

        self.step_norm = torch.nn.LayerNorm(d_entity)
        self.step_proj = torch.nn.Linear(d_entity, d_model)

        # Temporal encoder
        temp_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg["nhead"],
            dim_feedforward=cfg["ff_mult"] * d_model,
            dropout=cfg["dropout"],
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = torch.nn.TransformerEncoder(
            temp_layer, num_layers=cfg["inter_layers"], norm=torch.nn.LayerNorm(d_model)
        )

        self.pos_embed = torch.nn.Embedding(max_seq_len, d_model)

        self.action_head = torch.nn.Linear(d_model, num_actions)
        self.win_head = torch.nn.Linear(d_model, 1)
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
        x: torch.Tensor,  # (B,T,D_flat)
        mask: Optional[torch.Tensor] = None,  # (B,T) 1 valid, 0 pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if mask is None:
            mask = torch.ones(B, T, device=x.device)

        # Slice groups
        entities = []
        start = 0
        for gi, gs in enumerate(self.group_sizes):
            end = start + gs
            g_slice = x[:, :, start:end]  # (B,T,gs)
            proj = self.group_proj[gi](g_slice.reshape(B * T, gs)).view(B, T, -1)
            entities.append(proj)
            start = end
        # entities list of (B,T,d_entity); stack -> (B,T,G,d_entity)
        ent_tensor = torch.stack(entities, dim=2)

        # Flatten (B,T,G,d_entity) -> (B*T,G,d_entity) for intra-step encoder
        ent_flat = ent_tensor.view(B * T, self.num_groups, -1)
        ent_encoded = self.intra_encoder(ent_flat)  # (B*T,G,d_entity)

        # Mean pool entities
        step_emb = ent_encoded.mean(dim=1)  # (B*T,d_entity)
        step_emb = self.step_norm(step_emb)
        step_emb = self.step_proj(step_emb).view(B, T, -1)

        # Positional + mask
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0)
        step_emb = step_emb + self.pos_embed(pos_idx) * mask.unsqueeze(-1)

        key_padding_mask = ~mask.bool()
        temporal_out = self.temporal_encoder(
            step_emb, src_key_padding_mask=key_padding_mask
        )  # (B,T,d_model)

        action_logits = self.action_head(temporal_out)  # (B,T, num_actions)
        win_logits = self.win_head(temporal_out).squeeze(-1)  # (B,T)
        move_order_logits = self.move_order_head(temporal_out)
        ko_logits = self.ko_head(temporal_out)
        switch_logits = self.switch_head(temporal_out)
        return action_logits, win_logits, move_order_logits, ko_logits, switch_logits

    def predict_mc_dropout(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        passes: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty. Returns (mean_win_prob, var_win_prob).
        """
        self.train()  # Keep dropout active
        outs = []
        with torch.no_grad():
            for _ in range(passes):
                _, wlogits, _, _, _ = self.forward(x, mask)
                outs.append(torch.sigmoid(wlogits).unsqueeze(0))
        stack = torch.cat(outs, 0)
        return stack.mean(0), stack.var(0)


def sample_prefix_lengths(masks: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    if not config["prefix_enabled"]:
        return masks.sum(-1).long()
    lengths = masks.sum(-1).long()
    min_keep = (lengths.float() * config["prefix_min_frac"]).clamp(min=1).long()
    sampled = torch.empty_like(lengths)
    for i, T in enumerate(lengths):
        if config["prefix_mode"] == "geometric":
            support = torch.arange(
                int(min_keep[i].item()), int(T) + 1, device=masks.device
            )
            probs = (1 - config["prefix_geo_p"]) ** (support - min_keep[i]) * config[
                "prefix_geo_p"
            ]
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, 1)
            sampled[i] = support[idx]
        else:  # uniform
            sampled[i] = torch.randint(
                low=int(min_keep[i].item()),
                high=int(T.item() + 1),
                size=(1,),
                device=masks.device,
            )
    return sampled


def compute_step_weights(valid_mask: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    if config["step_weight_mode"] == "none":
        return valid_mask.float()
    B, L = valid_mask.shape
    weights = torch.zeros_like(valid_mask, dtype=torch.float)
    for b in range(B):
        idxs = (valid_mask[b] > 0).nonzero(as_tuple=False).flatten()
        if len(idxs) == 0:
            continue
        T = idxs[-1].item() + 1
        for t in idxs.tolist():
            dist = T - 1 - t
            if config["step_weight_mode"] == "inv_distance":
                weights[b, t] = 1.0 / (1 + config["inv_distance_alpha"] * dist)
    # Normalize per sequence
    seq_sum = weights.sum(-1, keepdim=True).clamp(min=1e-6)
    weights = weights / seq_sum * (valid_mask.sum(-1, keepdim=True).float().clamp(min=1))
    return weights


def build_value_targets(
    win_logits: torch.Tensor,
    final_outcome: torch.Tensor,  # (B,)
    value_mask: torch.Tensor,  # (B,T)
    has_next: torch.Tensor,  # (B,T)
    config: Dict[str, Any],
) -> torch.Tensor:
    B, T = win_logits.shape
    y = final_outcome.float().unsqueeze(1).expand(B, T)
    targets = y.clone()
    if not config["td_bootstrap"]:
        return targets
    with torch.no_grad():
        preds = torch.sigmoid(win_logits)
        gamma = config["td_gamma"]
        mix = config["td_mix"]
        for b in range(B):
            valid_idxs = (value_mask[b] > 0).nonzero(as_tuple=False).flatten()
            for t in valid_idxs.tolist():
                if has_next[b, t] > 0:
                    # next valid assuming contiguous valid indexes
                    if t + 1 in valid_idxs:
                        boot = gamma * preds[b, t + 1] + (1 - gamma) * y[b, t]
                    else:
                        boot = y[b, t]
                else:
                    boot = y[b, t]
                targets[b, t] = (1 - mix) * y[b, t] + mix * boot
    return targets


def apply_early_label_smoothing(
    targets: torch.Tensor,
    value_mask: torch.Tensor,
    config: Dict[str, Any],
) -> torch.Tensor:
    if not config["early_label_smoothing"]:
        return targets
    B, T = targets.shape
    smoothed = targets.clone()
    # Exponential decay smoothing: epsilon_t = smoothing_max * exp(-decay * t_norm)
    for t in range(T):
        eps = config["smoothing_max"] * math.exp(-config["smoothing_decay"] * t)
        # Only adjust where mask ==1
        if eps > 1e-6:
            step_mask = value_mask[:, t] > 0
            if step_mask.any():
                y = targets[step_mask, t]
                smoothed_val = y * (1 - eps) + 0.5 * eps
                smoothed[step_mask, t] = smoothed_val
    return smoothed


def train_epoch(model, dataloader, prev_steps, optimizer, config):
    model.train()
    running_loss = 0.0
    running_action_loss = 0.0
    running_win_loss = 0.0
    running_move_order_loss = 0.0
    running_ko_loss = 0.0
    running_switch_loss = 0.0
    running_win_samples = 0
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

        # Create has_next tensor (1 if not the last step, 0 if last step)
        # For each batch item, set has_next to 0 for the last valid step
        B, T = masks.shape
        has_next = masks.clone()
        for b in range(B):
            # Find the last valid step (where mask is 1)
            valid_idxs = (masks[b] > 0).nonzero(as_tuple=False).flatten()
            if len(valid_idxs) > 0:
                has_next[b, valid_idxs[-1]] = 0  # Last valid step has no next step

        # Prefix truncation mask for value
        prefix_lengths = sample_prefix_lengths(masks, config)
        idxs = torch.arange(T, device=config["device"]).unsqueeze(0).expand(B, T)
        value_mask = (idxs < prefix_lengths.unsqueeze(1)) & (masks > 0)

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

        # Unpack in the same order
        head_idx = 0
        valid_states = flat_data[head_idx]
        head_idx += 1

        valid_action_logits, valid_actions = flat_data[head_idx : head_idx + 2]
        head_idx += 2

        valid_win_logits, valid_wins = flat_data[head_idx : head_idx + 2]
        head_idx += 2

        valid_move_order_logits, valid_move_orders = flat_data[head_idx : head_idx + 2]
        head_idx += 2

        valid_ko_logits, valid_kos = flat_data[head_idx : head_idx + 2]
        head_idx += 2

        valid_switch_logits, valid_switches = flat_data[head_idx : head_idx + 2]

        # Build weights for loss
        weights = None
        if config.get("action_weights", False):
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

        # Action loss (same as before)
        action_loss = topk_cross_entropy_loss(
            valid_action_logits, valid_actions, weights=weights, k=3
        )

        # Win loss (now MSE since it's regression)
        # Build value targets
        step_weights = compute_step_weights(value_mask, config)
        targets = build_value_targets(win_logits, wins[:, 0], value_mask, has_next, config)
        targets = apply_early_label_smoothing(targets, value_mask, config)

        # Apply MSE only to valid steps
        # Using reduction='none' to multiply by step weights
        win_mse = torch.nn.functional.mse_loss(win_logits, targets, reduction="none")
        # Apply mask and weights
        win_mse = win_mse * value_mask.float() * step_weights
        win_loss = win_mse.sum() / value_mask.float().sum().clamp(min=1.0)

        # Move order loss (categorical cross-entropy); handles -1 for invalid move orders
        valid_idx = (valid_move_orders >= 0) & (
            valid_move_orders < valid_move_order_logits.size(1)
        )
        if valid_idx.sum() > 0:
            move_order_loss = torch.nn.functional.cross_entropy(
                valid_move_order_logits[valid_idx], valid_move_orders[valid_idx].long()
            )
        else:
            move_order_loss = torch.tensor(0.0, device=valid_move_order_logits.device)

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
        running_win_loss += win_loss.item() * valid_wins.size(0)
        running_win_samples += valid_wins.size(0)
        running_move_order_loss += move_order_loss.item()
        running_ko_loss += ko_loss.item()
        running_switch_loss += switch_loss.item()
        steps += valid_actions.size(0)
        num_batches += 1

        # Logging progress with all metrics
        wandb.log(
            {
                "steps": prev_steps + steps,
                "train_loss": running_loss / num_batches,
                "train_action_loss": running_action_loss / num_batches,
                "train_win_loss": running_win_loss
                / (running_win_samples if running_win_samples > 0 else 1),
                "train_move_order_loss": running_move_order_loss / num_batches,
                "train_ko_loss": running_ko_loss / num_batches,
                "train_switch_loss": running_switch_loss / num_batches,
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
        "win_loss": running_win_loss
        / (running_win_samples if running_win_samples > 0 else 1),
        "move_order_loss": running_move_order_loss / num_batches,
        "ko_loss": running_ko_loss / num_batches,
        "switch_loss": running_switch_loss / num_batches,
    }


def load_file_list(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main(train_path: str, test_path: str, val_path: str):
    print("Starting!")
    config: Dict[str, Any] = {
        # Basic training params
        "project": "entity_value_model",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 32,
        "num_epochs": 20,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "log_interval": 50,
        "early_stop_patience": 5,
        "save_path": "data/models",
        "seed": 42,
        # Loss weights
        "action_weight": 0.475,
        "win_weight": 0.475,
        "move_order_weight": 0.01,
        "ko_weight": 0.03,
        "switch_weight": 0.01,
        "action_weights": False,
        # Model architecture params
        "d_entity": 128,
        "d_model": 256,
        "intra_layers": 2,
        "inter_layers": 3,
        "nhead": 4,
        "ff_mult": 4,
        "dropout": 0.1,
        # Prefix params
        "prefix_enabled": True,
        "prefix_min_frac": 0.3,
        "prefix_mode": "uniform",
        "prefix_geo_p": 0.4,
        # Value params
        "step_weight_mode": "inv_distance",
        "inv_distance_alpha": 1.0,
        "td_bootstrap": True,
        "td_gamma": 0.99,
        "td_mix": 0.3,
        "early_label_smoothing": True,
        "smoothing_max": 0.2,
        "smoothing_decay": 0.15,
    }

    wandb.init(project=config["project"], config=config)

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    random.seed(int(config["seed"]))

    # Embedder
    embedder = Embedder(
        format="gen9vgc2023regulationc",
        feature_set=Embedder.FULL,
        omniscient=True,
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print("Embedder initialized. Embedding size:", embedder.embedding_size)

    # Datasets
    train_dataset = PreprocessedTrajectoryDataset(train_path, embedder=embedder)
    test_dataset = PreprocessedTrajectoryDataset(test_path, embedder=embedder)
    val_dataset = PreprocessedTrajectoryDataset(val_path, embedder=embedder)
    print(
        f"Loaded datasets. Train: {len(train_dataset)}, Test: {len(test_dataset)}, Val: {len(val_dataset)}"
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4
    )
    print("DataLoaders created.")

    # Model
    model = EntityTemporalValueModel(
        group_sizes=embedder.group_embedding_sizes,
        num_actions=MDBO.action_space(),
        max_seq_len=120,
        cfg=config,
    ).to(config["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    start, steps = time.time(), 0
    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config)
        metrics = evaluate(
            model,
            test_loader,
            config["device"],
            has_action_head=True,
            has_win_head=True,
            has_move_order_head=True,
            has_ko_head=True,
            has_switch_head=True,
        )
        steps += train_metrics["steps"]

        log = {
            "epoch": epoch + 1,
            "total_steps": steps,
            "train_loss": train_metrics["loss"],
            "train_win_loss": train_metrics["win_loss"],
            "train_action_loss": train_metrics["action_loss"],
            "train_move_order_loss": train_metrics["move_order_loss"],
            "train_ko_loss": train_metrics["ko_loss"],
            "train_switch_loss": train_metrics["switch_loss"],
            "Test Loss": (
                metrics["win_mse"] * config["win_weight"]
                + metrics["top3_loss"] * config["action_weight"]
            ),
            "test_win_corr": metrics["win_corr"],
            "test_win_mse": metrics["win_mse"],
            "test_action_top1": metrics["top1_acc"],
            "test_action_top3": metrics["top3_acc"],
            "test_action_top5": metrics["top5_acc"],
            "test_move_order_acc": metrics["move_order_acc"],
            "test_ko_acc": metrics["ko_acc"],
            "test_switch_acc": metrics["switch_acc"],
        }

        print(f"\nEpoch #{epoch + 1}:")
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

    torch.save(
        model.state_dict(),
        os.path.join(config["save_path"], f"{wandb.run.name}.pth"),  # type: ignore
    )
    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(
        model,
        val_loader,
        config["device"],
        has_action_head=True,
        has_win_head=True,
        has_move_order_head=True,
        has_ko_head=True,
        has_switch_head=True,
    )

    val_log = {
        "Val Loss": (
            metrics["win_mse"] * config["win_weight"]
            + metrics["top3_loss"] * config["action_weight"]
        ),
        "val_win_corr": metrics["win_corr"],
        "val_win_mse": metrics["win_mse"],
        "val_action_top1": metrics["top1_acc"],
        "val_action_top3": metrics["top3_acc"],
        "val_action_top5": metrics["top5_acc"],
        "val_move_order_acc": metrics["move_order_acc"],
        "val_ko_acc": metrics["ko_acc"],
        "val_switch_acc": metrics["switch_acc"],
    }

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
    if len(sys.argv) != 4:
        print(
            "Usage: python -m elitefurretai.scripts.train.win_hybrid_model "
            "<train_file_list.json> <test_file_list.json> <val_file_list.json>"
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
