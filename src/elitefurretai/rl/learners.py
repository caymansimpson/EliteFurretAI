"""learners.py — Training math and model I/O for EliteFurretAI RL.

This file owns two concerns that are tightly coupled:

  1. LEARNING ALGORITHM (PortfolioRNaDLearner)
     Computes the RNaD loss (PPO + C51 value + KL regularization vs a portfolio of
     reference models) and applies gradient updates. See RL.md for the math.

     To replicate standard RNaD (single fixed reference, no portfolio), set
     max_portfolio_size=1 and portfolio_update_strategy="recent" in the config.
     PortfolioRNaDLearner reduces to the base algorithm in that configuration.

  2. MODEL I/O  (bottom of this file)
     Utilities for saving and loading model checkpoints. The checkpoint format
     stores optimizer state, training step, and the RNaDConfig that produced the
     model. train.py and worker.py import these from here.
"""

import copy
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR

from elitefurretai.etl import MDBO, Embedder
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel, twohot_encode


def _build_optimizer(
    model: nn.Module,
    config: RNaDConfig,
) -> optim.Optimizer:
    """Build optimizer with topology-aware parameter groups from config."""
    opt = config.optimizer

    head_keywords = [
        "turn_action_head",
        "teampreview_head",
        "win_head",
        "turn_ff_stack",
        "teampreview_ff_stack",
    ]

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if any(h in name for h in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": opt.backbone_lr,
            "weight_decay": opt.backbone_weight_decay,
        },
        {
            "params": head_params,
            "lr": opt.heads_lr,
            "weight_decay": opt.heads_weight_decay,
        },
    ]

    if opt.type == "adamw":
        return optim.AdamW(param_groups)
    else:
        return optim.Adam(param_groups)


def _build_scheduler(optimizer: optim.Optimizer, config: RNaDConfig) -> LambdaLR:
    """Build LR scheduler with warmup + decay from config."""
    return LambdaLR(optimizer, lr_lambda=lambda step: config.lr_lambda(step))


class PortfolioRNaDLearner:
    def __init__(
        self,
        model: RNaDAgent,
        ref_models: List[RNaDAgent],
        config: RNaDConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.ref_models = [ref.to(device) for ref in ref_models]
        self.config = config
        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)
        self.gamma = config.algorithm.gamma
        self.clip_range = config.algorithm.clip_range
        self.ent_coef = config.algorithm.ent_coef
        self.vf_coef = config.algorithm.vf_coef
        self.rnad_alpha = config.algorithm.rnad_alpha
        self.gradient_clip = config.algorithm.max_grad_norm
        self.device = device
        self.use_mixed_precision = config.hardware.use_mixed_precision
        self.max_portfolio_size = config.portfolio.max_portfolio_size
        self.portfolio_update_strategy = config.portfolio.portfolio_update_strategy
        self._step = 0

        # Distributional value head support
        self.num_value_bins = config.value_head.num_value_bins
        self.value_support = torch.linspace(
            config.value_head.value_min,
            config.value_head.value_max,
            config.value_head.num_value_bins,
        ).to(device)

        self.scaler = (
            torch.amp.GradScaler(device=self.device)  # type: ignore[attr-defined]
            if self.use_mixed_precision
            else None
        )

        for ref_model in self.ref_models:
            for param in ref_model.parameters():
                param.requires_grad = False

        self.portfolio_kl_history: List[List[float]] = [[] for _ in range(len(ref_models))]
        self.portfolio_selection_counts: List[int] = [0] * len(ref_models)

    def add_reference_model(self, new_ref: RNaDAgent):
        new_ref = new_ref.to(self.device)
        for param in new_ref.parameters():
            param.requires_grad = False

        self.ref_models.append(new_ref)
        self.portfolio_kl_history.append([])
        self.portfolio_selection_counts.append(0)

        if len(self.ref_models) > self.max_portfolio_size:
            self._prune_portfolio()

    def _prune_portfolio(self):
        if len(self.ref_models) <= 1:
            return

        if self.portfolio_update_strategy == "recent":
            idx = 0
        elif self.portfolio_update_strategy == "best":
            idx = int(np.argmin(self.portfolio_selection_counts))
        elif self.portfolio_update_strategy == "diverse":
            raise NotImplementedError("Diverse strategy not implemented yet")
        else:
            raise ValueError(
                f"Unknown portfolio update strategy: {self.portfolio_update_strategy}"
            )

        self.ref_models.pop(idx)
        self.portfolio_kl_history.pop(idx)
        self.portfolio_selection_counts.pop(idx)

    def update_main_reference(self):
        if len(self.ref_models) > 0:
            self.ref_models[-1].load_state_dict(self.model.state_dict())
        else:
            new_ref = RNaDAgent(copy.deepcopy(self.model.model))
            self.add_reference_model(new_ref)

    def _compute_portfolio_kl(
        self,
        curr_dist: Categorical,
        ref_logits_list: list,
    ) -> torch.Tensor:
        if len(ref_logits_list) == 0:
            return torch.tensor(0.0, device=self.device)

        min_kl = None
        best_ref_idx = 0

        for ref_idx, ref_logits in enumerate(ref_logits_list):
            ref_dist = Categorical(logits=ref_logits)
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist).mean()

            self.portfolio_kl_history[ref_idx].append(kl.item())
            if len(self.portfolio_kl_history[ref_idx]) > 100:
                self.portfolio_kl_history[ref_idx].pop(0)

            if min_kl is None or kl < min_kl:
                min_kl = kl
                best_ref_idx = ref_idx

        self.portfolio_selection_counts[best_ref_idx] += 1

        return min_kl if min_kl is not None else torch.tensor(0.0, device=self.device)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self._step += 1
        ent_coef = self.config.ent_coef_at_step(self._step)

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        is_teampreview = batch["is_teampreview"].to(self.device)
        padding_mask = batch.get(
            "padding_mask", torch.ones_like(actions, dtype=torch.bool)
        ).to(self.device)

        action_masks = batch.get("masks", None)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        valid_advantages = advantages[padding_mask]
        if len(valid_advantages) > 1:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std() + 1e-8
            )

        initial_hidden = batch.get("initial_hidden", None)
        is_transformer = getattr(self.model, "_is_transformer", False)
        if is_transformer:
            initial_hidden_state = None
        elif initial_hidden is None:
            initial_hidden_state = self.model.get_initial_state(
                states.shape[0], self.device
            )
        else:
            initial_hidden_state = (
                initial_hidden[0].to(self.device),
                initial_hidden[1].to(self.device),
            )

        with torch.amp.autocast(device_type=self.device, enabled=self.use_mixed_precision):  # type: ignore
            if is_transformer:
                # Transformer: call raw model.forward() for full trajectory
                turn_logits, tp_logits, values, win_dist_logits = self.model.model.forward(
                    states
                )
            else:
                turn_logits, tp_logits, values, win_dist_logits, _ = self.model(
                    states, initial_hidden_state
                )

            ref_outputs = []
            for ref_model in self.ref_models:
                with torch.no_grad():
                    if is_transformer:
                        ref_turn, ref_tp, _, _ = ref_model.model.forward(states)
                    else:
                        ref_turn, ref_tp, _, _, _ = ref_model(states, initial_hidden_state)
                    ref_outputs.append((ref_turn, ref_tp))

            flat_actions = actions.reshape(-1)
            flat_old_log_probs = old_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_is_tp = is_teampreview.reshape(-1).bool()
            flat_padding_mask = padding_mask.reshape(-1).bool()

            policy_loss_tp = torch.tensor(0.0, device=self.device)
            policy_loss_turn = torch.tensor(0.0, device=self.device)
            entropy_tp = torch.tensor(0.0, device=self.device)
            entropy_turn = torch.tensor(0.0, device=self.device)
            rnad_loss_tp = torch.tensor(0.0, device=self.device)
            rnad_loss_turn = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)

            valid_tp_mask = flat_is_tp & flat_padding_mask
            if valid_tp_mask.any():
                tp_indices = torch.nonzero(valid_tp_mask, as_tuple=False).squeeze(-1)
                curr_tp_logits = tp_logits.reshape(-1, tp_logits.shape[-1])[tp_indices]
                curr_dist = Categorical(logits=curr_tp_logits)

                ref_tp_logits_list = [
                    ref_tp.reshape(-1, ref_tp.shape[-1])[tp_indices]
                    for _, ref_tp in ref_outputs
                ]
                rnad_loss_tp = self._compute_portfolio_kl(curr_dist, ref_tp_logits_list)

                curr_log_probs = curr_dist.log_prob(flat_actions[tp_indices])
                ratio = torch.exp(curr_log_probs - flat_old_log_probs[tp_indices])
                surr1 = ratio * flat_advantages[tp_indices]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * flat_advantages[tp_indices]
                )
                policy_loss_tp = -torch.min(surr1, surr2).mean()

                entropy_tp = curr_dist.entropy().mean()

            valid_turn_mask = (~flat_is_tp) & flat_padding_mask
            if valid_turn_mask.any():
                turn_indices = torch.nonzero(valid_turn_mask, as_tuple=False).squeeze(-1)
                curr_turn_logits = turn_logits.reshape(-1, turn_logits.shape[-1])[
                    turn_indices
                ]

                ref_turn_logits_list = [
                    ref_turn.reshape(-1, ref_turn.shape[-1])[turn_indices]
                    for ref_turn, _ in ref_outputs
                ]

                # Apply action masks. Use a large finite negative value rather than
                # -inf: with -inf, Categorical.kl_divergence's backward computes
                # (-inf) - (-inf) = NaN at masked positions, then propagates NaN
                # gradients into the unmasked positions and back through the model.
                # -1e9 is small enough to make masked actions effectively zero
                # probability under softmax, while keeping the backward pass finite.
                if action_masks is not None:
                    flat_masks = action_masks.reshape(-1, action_masks.shape[-1])
                    curr_masks = flat_masks[turn_indices]
                    curr_turn_logits = curr_turn_logits.masked_fill(
                        ~curr_masks.bool(), -1e9
                    )
                    ref_turn_logits_list = [
                        ref_logits.masked_fill(~curr_masks.bool(), -1e9)
                        for ref_logits in ref_turn_logits_list
                    ]

                curr_dist = Categorical(logits=curr_turn_logits)
                rnad_loss_turn = self._compute_portfolio_kl(
                    curr_dist, ref_turn_logits_list
                )

                curr_log_probs = curr_dist.log_prob(flat_actions[turn_indices])
                ratio = torch.exp(curr_log_probs - flat_old_log_probs[turn_indices])
                surr1 = ratio * flat_advantages[turn_indices]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * flat_advantages[turn_indices]
                )
                policy_loss_turn = -torch.min(surr1, surr2).mean()

                entropy_turn = curr_dist.entropy().mean()

            if flat_padding_mask.any():
                # Distributional value loss (C51 cross-entropy with two-hot targets)
                flat_dist_logits = win_dist_logits.reshape(-1, self.num_value_bins)
                targets = twohot_encode(flat_returns, self.value_support)
                log_probs_dist = torch.log_softmax(flat_dist_logits, dim=-1)
                per_step_value_loss = -(targets * log_probs_dist).sum(dim=-1)
                value_loss = (
                    per_step_value_loss * flat_padding_mask.float()
                ).sum() / flat_padding_mask.sum().clamp(min=1.0)

            policy_loss = policy_loss_tp + policy_loss_turn
            entropy_loss = entropy_tp + entropy_turn
            rnad_loss = rnad_loss_tp + rnad_loss_turn

            total_loss = (
                policy_loss
                + self.vf_coef * value_loss
                - ent_coef * entropy_loss
                + self.rnad_alpha * rnad_loss
            )

        self.optimizer.zero_grad()

        grad_norm_before = 0.0
        grad_norm_after = 0.0

        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            ).item()
            grad_norm_after = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            ).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            ).item()
            grad_norm_after = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            ).item()
            self.optimizer.step()

        self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "rnad_loss": rnad_loss.item(),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "portfolio_size": len(self.ref_models),
            "portfolio_selections": dict(enumerate(self.portfolio_selection_counts)),
            "lr_backbone": self.optimizer.param_groups[0]["lr"],
            "lr_heads": self.optimizer.param_groups[1]["lr"],
            "ent_coef": ent_coef,
        }

    def get_portfolio_stats(self) -> Dict[str, Any]:
        stats = {
            "portfolio_size": len(self.ref_models),
            "selection_counts": self.portfolio_selection_counts.copy(),
            "avg_kl_per_ref": [
                np.mean(kls) if kls else 0.0 for kls in self.portfolio_kl_history
            ],
        }
        return stats


__all__ = ["PortfolioRNaDLearner"]


# ══════════════════════════════════════════════════════════════════════════════
# Model I/O
#
# These utilities build, save, and load model checkpoints. They live here
# (rather than a separate file) because the checkpoint format is tightly coupled
# to the learner: it stores optimizer state, training step, and the RNaDConfig
# that produced the model.
#
# Callers: train.py (save_checkpoint, load_checkpoint, build_model_from_config),
#          worker.py (load_model_from_checkpoint, load_agent_from_checkpoint),
#          exploiter_train.py, analyze/play_model.py.
# ══════════════════════════════════════════════════════════════════════════════

# Config keys that define model architecture — used to check checkpoint compatibility.
# Changing any of these makes old checkpoints incompatible with the current model.
MODEL_ARCH_CONFIG_KEYS = (
    "battle_format",
    "embedder_feature_set",
    "early_layers",
    "late_layers",
    "lstm_layers",
    "lstm_hidden_size",
    "dropout",
    "early_attention_heads",
    "late_attention_heads",
    "grouped_encoder_hidden_dim",
    "grouped_encoder_aggregated_dim",
    "pokemon_attention_heads",
    "teampreview_head_layers",
    "teampreview_head_dropout",
    "teampreview_attention_heads",
    "turn_head_layers",
    "max_seq_len",
    "num_value_bins",
    "value_min",
    "value_max",
    "number_bank_hp_bins",
    "number_bank_stat_bins",
    "number_bank_power_bins",
    "number_bank_embedding_dim",
    # Transformer
    "use_transformer",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_dim",
    "transformer_dropout",
    "use_decision_tokens",
    "use_causal_mask",
)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _config_to_flat_arch(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat dict of all config fields from either a flat or nested config dict.

    Nested configs (new format) store architecture fields under sub-sections like
    "architecture", "value_head", "curriculum", etc. Merging all sections gives a
    flat dict compatible with the legacy MODEL_ARCH_CONFIG_KEYS lookup.
    """
    from elitefurretai.rl.config import _is_nested_format

    if not _is_nested_format(d):
        return d
    flat: Dict[str, Any] = {}
    for section in d.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def is_checkpoint_compatible_with_model_config(
    checkpoint_path: str,
    model_config: Dict[str, Any],
) -> bool:
    """Return True when a checkpoint config matches the current model architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return False

    flat_checkpoint = _config_to_flat_arch(checkpoint_config)
    flat_model = _config_to_flat_arch(model_config)

    for key in MODEL_ARCH_CONFIG_KEYS:
        current_value = _normalize_config_value(flat_model.get(key))
        checkpoint_value = _normalize_config_value(flat_checkpoint.get(key))
        if current_value != checkpoint_value:
            return False

    return True


def build_model_from_config(
    model_config: Dict[str, Any],
    embedder: Embedder,
    device: str,
    state_dict: Optional[Dict[str, Any]] = None,
) -> Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel]:
    """Construct a model from a config dict (flat or nested) and optionally load weights."""
    model_config = _config_to_flat_arch(model_config)
    use_transformer = model_config.get("use_transformer", False)

    common_kwargs: Dict[str, Any] = dict(
        embedder=embedder,
        early_layers=model_config["early_layers"],
        late_layers=model_config["late_layers"],
        dropout=model_config.get("dropout", 0.1),
        grouped_encoder_hidden_dim=model_config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=model_config.get(
            "grouped_encoder_aggregated_dim", 1024
        ),
        pokemon_attention_heads=model_config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=model_config.get("teampreview_head_layers", []),
        teampreview_head_dropout=model_config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=model_config.get("teampreview_attention_heads", 4),
        turn_head_layers=model_config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=model_config.get("max_seq_len", 17),
        num_value_bins=model_config.get("num_value_bins", 51),
        value_min=model_config.get("value_min", -1.0),
        value_max=model_config.get("value_max", 1.0),
        number_bank_hp_bins=model_config.get("number_bank_hp_bins", 100),
        number_bank_stat_bins=model_config.get("number_bank_stat_bins", 600),
        number_bank_power_bins=model_config.get("number_bank_power_bins", 250),
        number_bank_embedding_dim=model_config.get("number_bank_embedding_dim", 16),
        number_bank_damage_bins=model_config.get("number_bank_damage_bins", 600),
        number_bank_damage_embed_dim=model_config.get("number_bank_damage_embed_dim", 4),
        number_bank_turn_bins=model_config.get("number_bank_turn_bins", 40),
        number_bank_turn_embed_dim=model_config.get("number_bank_turn_embed_dim", 16),
        number_bank_rating_bins=model_config.get("number_bank_rating_bins", 100),
        number_bank_rating_embed_dim=model_config.get("number_bank_rating_embed_dim", 16),
        ability_embed_dim=model_config.get("ability_embed_dim", 16),
        item_embed_dim=model_config.get("item_embed_dim", 16),
        species_embed_dim=model_config.get("species_embed_dim", 32),
        move_embed_dim=model_config.get("move_embed_dim", 16),
    )

    if use_transformer:
        model: Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel] = (
            TransformerThreeHeadedModel(
                **common_kwargs,
                transformer_layers=model_config.get("transformer_layers", 6),
                transformer_heads=model_config.get("transformer_heads", 16),
                transformer_ff_dim=model_config.get("transformer_ff_dim", 2048),
                transformer_dropout=model_config.get("transformer_dropout", 0.1),
                use_decision_tokens=model_config.get("use_decision_tokens", True),
                use_causal_mask=model_config.get("use_causal_mask", True),
            ).to(device)
        )
    else:
        model = FlexibleThreeHeadedModel(
            **common_kwargs,
            lstm_layers=model_config.get("lstm_layers", 2),
            lstm_hidden_size=model_config.get("lstm_hidden_size", 512),
            early_attention_heads=model_config.get("early_attention_heads", 8),
            late_attention_heads=model_config.get("late_attention_heads", 8),
        ).to(device)

    if state_dict:
        # Strip _orig_mod. prefix left by torch.compile() before loading
        cleaned_state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict)

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> Tuple[
    Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel], Embedder, Dict[str, Any]
]:
    """Load a model + embedder from a checkpoint file. Returns (model, embedder, config_dict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    if embedder is None:
        cfg = RNaDConfig.from_dict(config_dict)
        embedder = Embedder(
            format=cfg.curriculum.battle_format,
            feature_set=cfg.training.embedder_feature_set,
            omniscient=False,
        )

    model = build_model_from_config(config_dict, embedder, device, state_dict)
    return model, embedder, config_dict


def load_agent_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> RNaDAgent:
    """Load a checkpoint and wrap it in an RNaDAgent."""
    model, _, _ = load_model_from_checkpoint(checkpoint_path, device, embedder)
    return RNaDAgent(model)


def save_checkpoint(
    model: RNaDAgent,
    optimizer: Any,
    step: int,
    config: RNaDConfig,
    curriculum: Dict[str, Any],
    save_dir: str = "data/models",
) -> str:
    """Save model, optimizer, step, and config to a timestamped checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"main_model_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "curriculum": curriculum,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    model: RNaDAgent,
    optimizer: Any,
    device: str,
) -> Tuple[int, RNaDConfig]:
    """Load checkpoint weights and optimizer state into existing model/optimizer objects."""
    print(f"Resuming from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    old_config = RNaDConfig.from_dict(checkpoint["config"])

    print(f"Resumed from step {step}")
    return step, old_config
