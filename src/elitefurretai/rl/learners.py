"""Unified learner module.

Contains both standard `RNaDLearner` and `PortfolioRNaDLearner`.
"""

import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR

from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import twohot_encode


def _build_optimizer(
    model: nn.Module,
    config: RNaDConfig,
) -> optim.Optimizer:
    """Build optimizer with topology-aware parameter groups from config."""
    opt_cfg = config.optimizer
    param_groups_cfg = opt_cfg.get("param_groups", {})

    head_keywords = [
        "turn_action_head", "teampreview_head", "win_head",
        "turn_ff_stack", "teampreview_ff_stack",
    ]

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if any(h in name for h in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_cfg = param_groups_cfg.get("backbone", {})
    heads_cfg = param_groups_cfg.get("heads", {})

    param_groups = [
        {
            "params": backbone_params,
            "lr": backbone_cfg.get("lr", config.lr),
            "weight_decay": backbone_cfg.get("weight_decay", opt_cfg.get("weight_decay", 1e-4)),
        },
        {
            "params": head_params,
            "lr": heads_cfg.get("lr", config.lr * 3),
            "weight_decay": heads_cfg.get("weight_decay", 0.0),
        },
    ]

    if opt_cfg.get("type", "adamw") == "adamw":
        return optim.AdamW(param_groups)
    else:
        return optim.Adam(param_groups)


def _build_scheduler(optimizer: optim.Optimizer, config: RNaDConfig) -> LambdaLR:
    """Build LR scheduler with warmup + decay from config."""
    return LambdaLR(optimizer, lr_lambda=lambda step: config.lr_lambda(step))


class RNaDLearner:
    def __init__(
        self,
        model: RNaDAgent,
        ref_model: RNaDAgent,
        config: RNaDConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.config = config
        self.optimizer = _build_optimizer(self.model, config)
        self.scheduler = _build_scheduler(self.optimizer, config)
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.rnad_alpha = config.rnad_alpha
        self.device = device
        self.use_mixed_precision = config.use_mixed_precision
        self.gradient_clip = config.max_grad_norm
        self._step = 0

        # Distributional value head support
        self.num_value_bins = config.num_value_bins
        self.value_support = torch.linspace(
            config.value_min, config.value_max, config.num_value_bins
        ).to(device)

        self.scaler = torch.amp.GradScaler(device=device) if self.use_mixed_precision else None  # type: ignore

        for param in self.ref_model.parameters():
            param.requires_grad = False

    def update_ref_model(self):
        self.ref_model.load_state_dict(self.model.state_dict())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self._step += 1
        ent_coef = self.config.ent_coef_at_step(self._step)

        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        is_teampreview = batch["is_teampreview"].to(self.device)
        padding_mask = batch.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        action_masks = batch.get("masks", None)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        is_transformer = getattr(self.model, "_is_transformer", False)
        initial_hidden = batch.get("initial_hidden", None)
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
            turn_logits, tp_logits, values, win_dist_logits, _ = self.model(states, initial_hidden_state)

        with torch.no_grad():
            ref_turn_logits, ref_tp_logits, _, _, _ = self.ref_model(
                states, initial_hidden_state
            )

        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_is_tp = is_teampreview.reshape(-1).bool()
        flat_values = values.reshape(-1)

        if padding_mask is not None:
            flat_padding_mask = padding_mask.reshape(-1).bool()
            valid_advs = flat_advantages[flat_padding_mask]
            adv_mean = valid_advs.mean()
            adv_std = valid_advs.std()
            flat_advantages = (flat_advantages - adv_mean) / (adv_std + 1e-8)
        else:
            flat_padding_mask = torch.ones_like(flat_values, dtype=torch.bool)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_rnad_loss = 0.0

        valid_tp_mask = flat_is_tp & flat_padding_mask

        if valid_tp_mask.any():
            tp_indices = torch.nonzero(valid_tp_mask).squeeze()
            curr_tp_logits = tp_logits.reshape(-1, tp_logits.shape[-1])[tp_indices]
            ref_tp_logits_sel = ref_tp_logits.reshape(-1, ref_tp_logits.shape[-1])[tp_indices]

            curr_dist = Categorical(logits=curr_tp_logits)
            ref_dist = Categorical(logits=ref_tp_logits_sel)

            curr_log_probs = curr_dist.log_prob(flat_actions[tp_indices])
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist)
            total_rnad_loss += kl.mean().item()

            ratio = torch.exp(curr_log_probs - flat_old_log_probs[tp_indices])
            surr1 = ratio * flat_advantages[tp_indices]
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * flat_advantages[tp_indices]
            )
            total_policy_loss += -torch.min(surr1, surr2).mean().item()
            total_entropy_loss += curr_dist.entropy().mean()

        valid_turn_mask = (~flat_is_tp) & flat_padding_mask

        if valid_turn_mask.any():
            turn_indices = torch.nonzero(valid_turn_mask).squeeze()

            curr_turn_logits = turn_logits.reshape(-1, turn_logits.shape[-1])[turn_indices]
            ref_turn_logits_sel = ref_turn_logits.reshape(-1, ref_turn_logits.shape[-1])[turn_indices]

            if action_masks is not None:
                flat_masks = action_masks.reshape(-1, action_masks.shape[-1])
                curr_masks = flat_masks[turn_indices]

                curr_turn_logits = curr_turn_logits.masked_fill(
                    ~curr_masks.bool(), float("-inf")
                )
                ref_turn_logits_sel = ref_turn_logits_sel.masked_fill(
                    ~curr_masks.bool(), float("-inf")
                )

            curr_dist = Categorical(logits=curr_turn_logits)
            ref_dist = Categorical(logits=ref_turn_logits_sel)

            curr_log_probs = curr_dist.log_prob(flat_actions[turn_indices])
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist)
            total_rnad_loss += kl.mean().item()

            ratio = torch.exp(curr_log_probs - flat_old_log_probs[turn_indices])
            surr1 = ratio * flat_advantages[turn_indices]
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                * flat_advantages[turn_indices]
            )
            total_policy_loss += -torch.min(surr1, surr2).mean().item()
            total_entropy_loss += curr_dist.entropy().mean()

        # Distributional value loss (C51 cross-entropy with two-hot targets)
        flat_dist_logits = win_dist_logits.reshape(-1, self.num_value_bins)
        targets = twohot_encode(flat_returns, self.value_support)
        log_probs_dist = torch.log_softmax(flat_dist_logits, dim=-1)
        per_step_value_loss = -(targets * log_probs_dist).sum(dim=-1)
        if padding_mask is not None:
            value_loss = (
                (per_step_value_loss * flat_padding_mask.float()).sum()
                / flat_padding_mask.sum().clamp(min=1.0)
            )
        else:
            value_loss = per_step_value_loss.mean()

        loss = (
            total_policy_loss
            + self.vf_coef * value_loss
            - ent_coef * total_entropy_loss
            + self.rnad_alpha * total_rnad_loss
        )

        self.optimizer.zero_grad()

        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
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
            loss.backward()
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            ).item()
            grad_norm_after = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            ).item()
            self.optimizer.step()

        self.scheduler.step()

        return {
            "loss": loss.item(),
            "policy_loss": (
                total_policy_loss.item()
                if isinstance(total_policy_loss, torch.Tensor)
                else total_policy_loss
            ),
            "value_loss": value_loss.item(),
            "entropy": (
                total_entropy_loss.item()
                if isinstance(total_entropy_loss, torch.Tensor)
                else total_entropy_loss
            ),
            "rnad_loss": (
                total_rnad_loss.item()
                if isinstance(total_rnad_loss, torch.Tensor)
                else total_rnad_loss
            ),
            "grad_norm_before_clip": grad_norm_before,
            "grad_norm_after_clip": grad_norm_after,
            "lr_backbone": self.optimizer.param_groups[0]["lr"],
            "lr_heads": self.optimizer.param_groups[1]["lr"],
            "ent_coef": ent_coef,
        }


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
        self.gamma = config.gamma
        self.clip_range = config.clip_range
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.rnad_alpha = config.rnad_alpha
        self.gradient_clip = config.max_grad_norm
        self.device = device
        self.use_mixed_precision = config.use_mixed_precision
        self.max_portfolio_size = config.max_portfolio_size
        self.portfolio_update_strategy = config.portfolio_update_strategy
        self._step = 0

        # Distributional value head support
        self.num_value_bins = config.num_value_bins
        self.value_support = torch.linspace(
            config.value_min, config.value_max, config.num_value_bins
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
            raise ValueError(f"Unknown portfolio update strategy: {self.portfolio_update_strategy}")

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
            turn_logits, tp_logits, values, win_dist_logits, _ = self.model(states, initial_hidden_state)

            ref_outputs = []
            for ref_model in self.ref_models:
                with torch.no_grad():
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
                rnad_loss_tp = self._compute_portfolio_kl(
                    curr_dist, ref_tp_logits_list
                )

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
                curr_dist = Categorical(logits=curr_turn_logits)

                ref_turn_logits_list = [
                    ref_turn.reshape(-1, ref_turn.shape[-1])[turn_indices]
                    for ref_turn, _ in ref_outputs
                ]
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
                    (per_step_value_loss * flat_padding_mask.float()).sum()
                    / flat_padding_mask.sum().clamp(min=1.0)
                )

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


__all__ = ["RNaDLearner", "PortfolioRNaDLearner"]
