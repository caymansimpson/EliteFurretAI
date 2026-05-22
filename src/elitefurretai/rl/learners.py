"""learners.py — Training math and model I/O for EliteFurretAI RL.

What this file is
-----------------
The "brain" of the trainer. While workers are out playing battles and shipping
trajectories, this is what consumes those trajectories and turns them into
gradient updates for the model.

Two concerns live here, intentionally coupled because they share types:

  1. LEARNING ALGORITHM (`PortfolioRNaDLearner`)
     The actual RL loss: PPO policy loss + C51 distributional value loss +
     entropy bonus + KL regularization against a portfolio of reference models.

  2. MODEL I/O (bottom of file)
     Save/load checkpoints. We pickle (model_state_dict, optimizer_state,
     step, RNaDConfig) so a run can be resumed exactly.

For ML researchers new to RL: a one-paragraph algorithm primer
--------------------------------------------------------------
Plain PPO (Schulman 2017) computes a "policy ratio" between the current and
the data-collection-time policy, clips it to [1-ε, 1+ε], and uses that to
weight an advantage estimate (GAE) for the policy gradient. A value head is
trained to predict expected return so we have something to compute
advantages against. That's the policy_loss + value_loss part.

RNaD (Perolat et al., DeepMind 2022) adds a KL penalty against a slowly-moving
"reference" policy (the anchor). This stops the agent from cycling between
strategies in adversarial games and prevents catastrophic forgetting of what
behavior cloning taught it. The α weight on this term (`rnad_alpha`) is the
"how strongly do we tether the current policy to the reference" knob.

Portfolio RNaD (this code) keeps not one but several reference models and
uses min-KL across them: the regularization is satisfied as long as the
current policy is close to *any* of the references. This avoids the failure
mode where a single old reference becomes irrelevant.

C51 distributional value head (Bellemare et al. 2017): instead of regressing
the value as a scalar, we predict a distribution over discrete return-bins and
use cross-entropy on a "two-hot" target. This gives richer gradients and
empirically trains more stably than scalar MSE for our use case.

To replicate standard RNaD (single fixed reference, no portfolio), set
max_portfolio_size=1 and portfolio_update_strategy="recent" in the config.
PortfolioRNaDLearner reduces to the base algorithm in that configuration.

Where this fits in the bigger picture
-------------------------------------
train.py main loop:
  - Pulls trajectories from the multiprocessing queue.
  - When `train_batch_size` trajectories are accumulated, calls
    `learner.update(batch)` (this file).
  - Every N updates, broadcasts new model weights back to workers.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR

from elitefurretai.etl import MDBO, Embedder
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.rl_utils import is_cuda_device, timestamp_iso
from elitefurretai.rl.rnad_model import RNaDModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel, twohot_encode

logger = logging.getLogger(__name__)

# Action-mask fill value: large negative number that drives softmax probability
# to zero on illegal actions, but small enough in magnitude to fit in fp16
# (max representable ≈ 65504). Using -1e9 (the obvious choice) overflows fp16
# under torch.amp.autocast(device_type="cuda") and crashes
# `masked_fill(..., -1e9)` with "value cannot be converted to type at::Half
# without overflow". -1e4 is a) representable in fp16 with margin,
# b) softmax(-1e4 + valid_O(1)) underflows to zero in fp32 too,
# c) finite, so Categorical.kl_divergence's backward is well-defined on
# fully-masked rows.
ACTION_MASK_FILL = -1e4


def _build_optimizer(
    model: nn.Module,
    config: RNaDConfig,
    device: str = "cpu",
) -> optim.Optimizer:
    """Build optimizer with topology-aware parameter groups from config.

    Why two parameter groups (backbone vs heads)
    --------------------------------------------
    The model has a big shared "backbone" (transformer / LSTM + encoder layers)
    and several small "heads" (one each for turn actions, teampreview actions,
    and win/value prediction). They train differently:
      - Backbone: large, deep, already pretrained via BC. Wants a small LR
        and meaningful weight decay so we don't undo BC.
      - Heads: smaller, may need to adjust faster as the value/policy refines
        for the new (RL) reward signal. Higher LR, often no weight decay.
    Splitting them into two AdamW param groups gives each its own LR and WD.

    The keywords below are matched as substrings of named parameter paths.
    Anything matching is "head"; everything else is "backbone".
    """
    opt = config.optimizer

    head_keywords = [
        "turn_action_head",
        "teampreview_head",
        "win_head",
        "turn_ff_stack",
        "teampreview_ff_stack",
        "value_ff_stack",
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
        # fused AdamW is CUDA-only and ~10-20% faster than the default foreach path.
        use_fused = is_cuda_device(device)
        return optim.AdamW(param_groups, fused=use_fused)
    else:
        return optim.Adam(param_groups)


def _build_scheduler(optimizer: optim.Optimizer, config: RNaDConfig) -> LambdaLR:
    """Build LR scheduler with warmup + decay from config."""
    return LambdaLR(optimizer, lr_lambda=lambda step: config.lr_lambda(step))


class PortfolioRNaDLearner:
    def __init__(
        self,
        model: RNaDModel,
        ref_models: List[RNaDModel],
        config: RNaDConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.ref_models = [ref.to(device) for ref in ref_models]
        self.config = config
        self.optimizer = _build_optimizer(self.model, config, device=device)
        self.scheduler = _build_scheduler(self.optimizer, config)
        self.gamma = config.algorithm.gamma
        self.clip_range = config.algorithm.clip_range
        self.ent_coef = config.algorithm.ent_coef
        self.vf_coef = config.algorithm.vf_coef
        self.rnad_alpha = config.algorithm.rnad_alpha
        self.gradient_clip = config.algorithm.max_grad_norm
        self.ppo_epochs = max(1, config.algorithm.ppo_epochs)
        self.ppo_kl_early_stop = config.algorithm.ppo_kl_early_stop
        self.device = device
        self.max_portfolio_size = config.portfolio.max_portfolio_size
        self.portfolio_update_strategy = config.portfolio.portfolio_update_strategy
        self._step = 0

        # One-shot eval batch used by the "diverse" pruning strategy to
        # measure pairwise policy KL between refs. Lazily snapshotted from
        # the first real `update()` batch — random states would give a
        # meaningless KL. Only allocated when the strategy is actually
        # "diverse" to keep VRAM use unchanged for other strategies.
        self._diversity_eval_states: Optional[torch.Tensor] = None

        # Distributional value head support
        self.num_value_bins = config.value_head.num_value_bins
        self.value_support = torch.linspace(
            config.value_head.value_min,
            config.value_head.value_max,
            config.value_head.num_value_bins,
        ).to(device)
        # Mix a uniform component into the twohot C51 target before the
        # cross-entropy. 0.0 = stock twohot. See ValueHeadConfig docstring
        # and MODEL_EVALUATION.md Priority 2 for motivation.
        self.value_label_smoothing = config.value_head.value_label_smoothing

        self.scaler = (
            torch.amp.GradScaler(device=self.device)  # type: ignore[attr-defined]
            if is_cuda_device(self.device)
            else None
        )

        for ref_model in self.ref_models:
            for param in ref_model.parameters():
                param.requires_grad = False

        self.portfolio_kl_history: List[List[float]] = [[] for _ in range(len(ref_models))]
        self.portfolio_selection_counts: List[int] = [0] * len(ref_models)

    def load_resume_state(self, checkpoint: Dict[str, Any]) -> None:
        """Restore optimizer, scheduler, and `_step` from a checkpoint dict.

        Must be called AFTER the learner is constructed (so optimizer +
        scheduler exist) and AFTER the model weights have been loaded into
        the agent (so `ref_models[0]` deepcopied the trained weights, not
        the random init).

        Falls back gracefully on older checkpoints that pre-date the
        scheduler/_step persistence (those skip restoration and the global
        update counter is used as a best-effort anchor for scheduler
        `last_epoch`). New checkpoints written by `save_checkpoint` always
        carry both keys.
        """
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("step", 0))
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Pre-fix checkpoints: anchor warmup/cosine to the global step
            # so we don't replay the warmup from 0 on resume.
            self.scheduler.last_epoch = max(global_step - 1, -1)
        if "learner_step" in checkpoint:
            self._step = int(checkpoint["learner_step"])
        else:
            self._step = global_step

    def add_reference_model(self, new_ref: RNaDModel):
        new_ref = new_ref.to(self.device)
        for param in new_ref.parameters():
            param.requires_grad = False

        self.ref_models.append(new_ref)
        self.portfolio_kl_history.append([])
        self.portfolio_selection_counts.append(0)

        if len(self.ref_models) > self.max_portfolio_size:
            self._prune_portfolio()

    def _prune_portfolio(self):
        """Pick one reference to evict so the portfolio fits max_portfolio_size.

        Strategies trade off recency, load-bearing-ness, and exploration of
        the anchor space. The choice matters because the RNaD KL term anchors
        the current policy to whichever reference is selected as min-KL at
        each step — what we keep here directly shapes the next regularization
        signal.
        """
        if len(self.ref_models) <= 1:
            return

        if self.portfolio_update_strategy == "recent":
            # Drop the oldest reference (refs are appended in time order, so
            # idx 0 is the eldest). Cheapest + most stable rotation; keeps the
            # portfolio biased toward recent policies. Recommended default.
            idx = 0
        elif self.portfolio_update_strategy == "best":
            # Drop the reference that was the closest-to-current-policy LEAST
            # often. Under mean-KL loss this is the ref the policy has drifted
            # FARTHEST from on average — so keeping the others concentrates the
            # anchor field near the current policy (weaker but more locally
            # focused regularization).
            idx = int(np.argmin(self.portfolio_selection_counts))
        elif self.portfolio_update_strategy == "random":
            # Uniformly-random eviction. Cheapest unbiased alternative to
            # "recent" — wider mix of anchor ages on average without preferring
            # any selection signal. Useful as an ablation baseline.
            idx = int(np.random.randint(len(self.ref_models)))
        elif self.portfolio_update_strategy == "diverse":
            # Drop the ref whose policy is most redundant with another's,
            # measured by min pairwise KL on a one-shot eval batch
            # (`_diversity_eval_states`). Keeps survivors spread across
            # policy space instead of clustering around the most-recent
            # snapshots — guards against portfolios that fill with
            # near-duplicates added at every portfolio_add_interval.
            #
            # Falls back to "recent" until the first update() call has run
            # and stashed an eval batch.
            idx = self._diverse_eviction_index()
        else:
            raise ValueError(
                f"Unknown portfolio update strategy: {self.portfolio_update_strategy}"
            )

        self.ref_models.pop(idx)
        self.portfolio_kl_history.pop(idx)
        self.portfolio_selection_counts.pop(idx)

    def _diverse_eviction_index(self) -> int:
        """Pick the most-redundant ref for the "diverse" pruning strategy.

        Algorithm: for each ref, compute its min pairwise turn-policy KL to
        any other ref (on `_diversity_eval_states`). Return the index of the
        ref whose nearest neighbor is closest — i.e. the ref whose policy
        a sibling already covers.

        Returns 0 (fall back to "recent") if no eval batch has been
        snapshotted yet (no update() call has run).
        """
        if self._diversity_eval_states is None:
            return 0

        with torch.no_grad():
            ref_dists: List[Categorical] = []
            for r in self.ref_models:
                turn_logits, _, _, _ = r.model.forward(self._diversity_eval_states)
                logits_flat = turn_logits.reshape(-1, turn_logits.shape[-1])
                ref_dists.append(Categorical(logits=logits_flat))

            n = len(ref_dists)
            redundancy = []
            for i in range(n):
                nearest = float("inf")
                for j in range(n):
                    if i == j:
                        continue
                    kl_ij = (
                        torch.distributions.kl_divergence(ref_dists[i], ref_dists[j])
                        .mean()
                        .item()
                    )
                    nearest = min(nearest, kl_ij)
                redundancy.append(nearest)
        return int(np.argmin(redundancy))

    def _compute_portfolio_kl(
        self,
        curr_dist: Categorical,
        ref_logits_list: list,
        track: bool = True,
    ) -> torch.Tensor:
        """Return MEAN KL from curr_dist to all reference policies.

        Mean-KL anchors the policy against the average of the
        portfolio.

        track=True records the per-ref KL into history and bumps the
        selection counter for the closest reference — set False on PPO
        inner-loop epochs >0 to keep bookkeeping aligned with
        one-update-per-counter semantics. The "selected" counter still
        tracks the nearest reference for diagnostics even though the loss
        no longer uses it.
        """
        if len(ref_logits_list) == 0:
            return torch.tensor(0.0, device=self.device)

        kls: List[torch.Tensor] = []
        best_ref_idx = 0
        best_kl_val: Optional[float] = None

        for ref_idx, ref_logits in enumerate(ref_logits_list):
            ref_dist = Categorical(logits=ref_logits)
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist).mean()
            kls.append(kl)

            kl_val = kl.item()
            if track:
                self.portfolio_kl_history[ref_idx].append(kl_val)
                if len(self.portfolio_kl_history[ref_idx]) > 100:
                    self.portfolio_kl_history[ref_idx].pop(0)

            if best_kl_val is None or kl_val < best_kl_val:
                best_kl_val = kl_val
                best_ref_idx = ref_idx

        if track:
            self.portfolio_selection_counts[best_ref_idx] += 1

        return torch.stack(kls).mean()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """One gradient update from a batch of trajectories.

        High-level flow (this is the meat of the algorithm):
          1. Move batch tensors onto the learner device.
          2. Normalize advantages (helps training stability).
          3. Run reference models forward ONCE (their outputs are frozen
             across PPO inner-loop epochs).
          4. For each PPO mini-epoch (config.algorithm.ppo_epochs, default 1):
              a. Forward main model.
              b. Compute losses split by step type:
                  - Teampreview steps (90 actions, no mask): PPO loss + entropy + KL
                  - Turn steps (2025 actions, masked): same, but logits get the
                    action mask applied so illegal actions can't move
                    probability mass.
              c. Compute distributional value loss (C51 cross-entropy) over all
                 non-padded steps.
              d. Sum into total_loss = policy + vf*value - ent*entropy + α*KL.
              e. Backprop, clip gradients, optimizer step.
              f. Optionally early-stop on approximate KL drift.
          5. Scheduler steps once per update (not per epoch).
          6. Return a dict of metrics for wandb / console logging.

        With ppo_epochs=1 (default) this reproduces the original behavior with
        the only change being that ref_outputs are computed once before the
        (single-iteration) loop.

        Returns:
            metrics dict with keys: loss, policy_loss, value_loss, entropy,
            rnad_loss, grad norms, approx_kl, ppo_epochs_actual, LRs, ent_coef,
            portfolio_*.
        """
        self._step += 1
        ent_coef = self.config.ent_coef_at_step(self._step)

        states = batch["states"].to(self.device)
        # Lazy snapshot for the "diverse" pruning strategy. Done here (not
        # in __init__) because we need a representative real-data batch.
        if (
            self.portfolio_update_strategy == "diverse"
            and self._diversity_eval_states is None
        ):
            self._diversity_eval_states = states.detach().clone()
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        is_teampreview = batch["is_teampreview"].to(self.device)
        padding_mask = batch.get(
            "padding_mask", torch.ones_like(actions, dtype=torch.bool)
        ).to(self.device)

        action_masks = batch["masks"].to(self.device)

        # ── Advantage normalization ───────────────────────────────────────────
        # Why: PPO is sensitive to the *scale* of advantages. Large advantages
        # cause big policy updates; small ones cause tiny ones. Per-batch
        # standardization to zero-mean / unit-std makes training behave more
        # uniformly across reward magnitudes and is a well-known PPO trick.
        # We standardize using only valid (non-padded) entries so padding zeros
        # don't bias the mean/std.
        # ─────────────────────────────────────────────────────────────────────
        valid_advantages = advantages[padding_mask]
        if len(valid_advantages) > 1:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std() + 1e-8
            )

        # ── Pre-compute everything that's constant across PPO epochs ─────────
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_is_tp = is_teampreview.reshape(-1).bool()
        flat_padding_mask = padding_mask.reshape(-1).bool()

        valid_tp_mask = flat_is_tp & flat_padding_mask
        valid_turn_mask = (~flat_is_tp) & flat_padding_mask
        has_tp = bool(valid_tp_mask.any().item())
        has_turn = bool(valid_turn_mask.any().item())
        has_padded = bool(flat_padding_mask.any().item())

        tp_indices = (
            torch.nonzero(valid_tp_mask, as_tuple=False).squeeze(-1) if has_tp else None
        )
        turn_indices = (
            torch.nonzero(valid_turn_mask, as_tuple=False).squeeze(-1)
            if has_turn
            else None
        )

        # Action mask for turn steps. ACTION_MASK_FILL (not -inf) avoids NaN in
        # Categorical.kl_divergence backward when applied to fully-masked rows.
        if has_turn:
            flat_masks = action_masks.reshape(-1, action_masks.shape[-1])
            curr_masks_bool = flat_masks[turn_indices].bool()
            turn_mask_neg_inf = ~curr_masks_bool
        else:
            turn_mask_neg_inf = None

        # Distributional value targets — twohot is just a binning op, no grad.
        # Optional uniform smoothing: target = (1-ε)·twohot + ε/num_bins.
        if has_padded:
            value_targets = twohot_encode(flat_returns, self.value_support)
            if self.value_label_smoothing > 0.0:
                eps = self.value_label_smoothing
                value_targets = value_targets * (1.0 - eps) + (eps / self.num_value_bins)
            valid_count_clamped = flat_padding_mask.sum().clamp(min=1.0)
        else:
            value_targets = None
            valid_count_clamped = None

        # ── Reference model forwards (frozen across PPO epochs) ──────────────
        # Refs don't update during this call, so their logits at every step
        # are identical between epoch 0 and epoch K-1. Pre-slice + pre-mask
        # so the inner loop only computes the main model.
        ref_tp_logits_list: list = []
        ref_turn_logits_list: list = []
        if len(self.ref_models) > 0:
            with torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
                device_type=self.device
            ):
                with torch.no_grad():
                    for ref_model in self.ref_models:
                        ref_turn, ref_tp, _, _ = ref_model.model.forward(states)
                        if has_tp:
                            ref_tp_logits_list.append(
                                ref_tp.reshape(-1, ref_tp.shape[-1])[tp_indices]
                            )
                        if has_turn:
                            assert turn_mask_neg_inf is not None
                            r = ref_turn.reshape(-1, ref_turn.shape[-1])[turn_indices]
                            ref_turn_logits_list.append(
                                r.masked_fill(turn_mask_neg_inf, ACTION_MASK_FILL)
                            )
            if self.device == "cuda":
                torch.cuda.synchronize()

        # ── PPO inner loop ────────────────────────────────────────────────────
        metrics: Dict[str, Any] = {}
        epochs_run = 0
        approx_kl = 0.0

        for epoch in range(self.ppo_epochs):
            # Bookkeeping (selection counts / kl history) tracks ONCE per update,
            # on the first epoch — it represents "which ref was closest at the
            # *start* of this update", not how many forward passes we did.
            track_kl = epoch == 0

            with torch.amp.autocast(  # pyright: ignore[reportPrivateImportUsage]
                device_type=self.device
            ):
                turn_logits, tp_logits, values, win_dist_logits = self.model.model.forward(
                    states
                )

                policy_loss_tp = torch.tensor(0.0, device=self.device)
                policy_loss_turn = torch.tensor(0.0, device=self.device)
                entropy_tp = torch.tensor(0.0, device=self.device)
                entropy_turn = torch.tensor(0.0, device=self.device)
                rnad_loss_tp = torch.tensor(0.0, device=self.device)
                rnad_loss_turn = torch.tensor(0.0, device=self.device)
                value_loss = torch.tensor(0.0, device=self.device)
                approx_kl_sum = torch.tensor(0.0, device=self.device)
                approx_kl_n = 0

                if has_tp:
                    curr_tp_logits = tp_logits.reshape(-1, tp_logits.shape[-1])[tp_indices]
                    curr_dist = Categorical(logits=curr_tp_logits)
                    rnad_loss_tp = self._compute_portfolio_kl(
                        curr_dist, ref_tp_logits_list, track=track_kl
                    )

                    # ── PPO clipped surrogate objective (teampreview path) ──
                    # ratio = π_current(a | s) / π_old(a | s)
                    #       = exp(log_prob_current - log_prob_old)
                    # surr1 = ratio       * advantage      (unclipped)
                    # surr2 = clip(ratio) * advantage      (clipped to [1-ε, 1+ε])
                    # loss  = -min(surr1, surr2).mean()
                    # Why min: PPO's pessimistic bound — when an action looked
                    # good but ratio drifted high we use the clipped (smaller)
                    # value, preventing aggressive moves; symmetric for bad
                    # actions with low ratio.
                    curr_log_probs = curr_dist.log_prob(flat_actions[tp_indices])
                    log_ratio = curr_log_probs - flat_old_log_probs[tp_indices]
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * flat_advantages[tp_indices]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                        * flat_advantages[tp_indices]
                    )
                    policy_loss_tp = -torch.min(surr1, surr2).mean()
                    entropy_tp = curr_dist.entropy().mean()
                    # Schulman approximate KL (PPO standard, low-variance)
                    approx_kl_sum = (
                        approx_kl_sum + ((torch.exp(log_ratio) - 1) - log_ratio).sum()
                    )
                    approx_kl_n += log_ratio.numel()

                if has_turn:
                    assert turn_indices is not None
                    assert turn_mask_neg_inf is not None
                    curr_turn_logits = turn_logits.reshape(-1, turn_logits.shape[-1])[
                        turn_indices
                    ]
                    curr_turn_logits = curr_turn_logits.masked_fill(
                        turn_mask_neg_inf, ACTION_MASK_FILL
                    )
                    curr_dist = Categorical(logits=curr_turn_logits)
                    rnad_loss_turn = self._compute_portfolio_kl(
                        curr_dist, ref_turn_logits_list, track=track_kl
                    )

                    curr_log_probs = curr_dist.log_prob(flat_actions[turn_indices])
                    log_ratio = curr_log_probs - flat_old_log_probs[turn_indices]
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * flat_advantages[turn_indices]
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                        * flat_advantages[turn_indices]
                    )
                    policy_loss_turn = -torch.min(surr1, surr2).mean()
                    entropy_turn = curr_dist.entropy().mean()
                    approx_kl_sum = (
                        approx_kl_sum + ((torch.exp(log_ratio) - 1) - log_ratio).sum()
                    )
                    approx_kl_n += log_ratio.numel()

                if has_padded:
                    assert value_targets is not None
                    assert valid_count_clamped is not None
                    # ── C51 distributional value loss ────────────────────────
                    # Value head outputs a distribution over num_value_bins
                    # bins spanning [value_min, value_max]. The target is
                    # twohot-encoded: probability mass on the two bins
                    # straddling the true return, weighted by linear distance.
                    # Loss = cross-entropy(target, predicted), masked over
                    # padding. Richer gradient than scalar MSE — value head
                    # learns *uncertainty*, not just the mean.
                    flat_dist_logits = win_dist_logits.reshape(-1, self.num_value_bins)
                    log_probs_dist = torch.log_softmax(flat_dist_logits, dim=-1)
                    per_step_value_loss = -(value_targets * log_probs_dist).sum(dim=-1)
                    value_loss = (
                        per_step_value_loss * flat_padding_mask.float()
                    ).sum() / valid_count_clamped

                # Combine teampreview and turn-step contributions.
                policy_loss = policy_loss_tp + policy_loss_turn
                entropy_loss = entropy_tp + entropy_turn
                rnad_loss = rnad_loss_tp + rnad_loss_turn

                # total = policy + vf*value - ent*entropy + α*KL_to_reference
                # Entropy is *subtracted* because we want to *maximize* it
                # (encourage exploration).
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - ent_coef * entropy_loss
                    + self.rnad_alpha * rnad_loss
                )

            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                ).item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                ).item()
                self.optimizer.step()

            epochs_run += 1
            approx_kl = (approx_kl_sum / approx_kl_n).item() if approx_kl_n > 0 else 0.0

            metrics = {
                "loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy_loss.item(),
                "rnad_loss": rnad_loss.item(),
                "grad_norm_before_clip": grad_norm_before,
                "approx_kl": approx_kl,
                "ppo_epochs_actual": epochs_run,
            }

            # Optional safety: stop the inner loop if the policy has drifted
            # too far from the collection-time policy.
            if (
                self.ppo_kl_early_stop is not None
                and approx_kl > self.ppo_kl_early_stop
                and epoch + 1 < self.ppo_epochs
            ):
                break

        # Scheduler advances once per update, not per epoch.
        self.scheduler.step()

        metrics.update(
            {
                "portfolio_size": len(self.ref_models),
                "portfolio_selections": dict(enumerate(self.portfolio_selection_counts)),
                "lr_backbone": self.optimizer.param_groups[0]["lr"],
                "lr_heads": self.optimizer.param_groups[1]["lr"],
                "ent_coef": ent_coef,
            }
        )

        return metrics

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
#          worker.py (load_model_from_checkpoint).
# ══════════════════════════════════════════════════════════════════════════════

# Config keys that define model architecture — used to check checkpoint compatibility.
# Changing any of these makes old checkpoints incompatible with the current model.
MODEL_ARCH_CONFIG_KEYS = (
    "battle_format",
    "embedder_feature_set",
    "early_layers",
    "late_layers",
    "dropout",
    "grouped_encoder_hidden_dim",
    "grouped_encoder_aggregated_dim",
    "pokemon_attention_heads",
    "teampreview_head_layers",
    "teampreview_head_dropout",
    "teampreview_attention_heads",
    "turn_head_layers",
    "value_head_layers",
    "max_seq_len",
    "num_value_bins",
    "value_min",
    "value_max",
    "number_bank_hp_bins",
    "number_bank_stat_bins",
    "number_bank_power_bins",
    "number_bank_embedding_dim",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_dim",
    "transformer_dropout",
)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _config_to_flat_arch(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat dict of all config fields from either a flat or nested config dict.

    Nested configs (new format) store architecture fields under sub-sections like
    "architecture", "value_head", "curriculum", etc. Flat configs have all fields
    at the top level. Always-flatten is safe because no MODEL_ARCH_CONFIG_KEYS
    value is itself a dict — top-level dict values are always sections.
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
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
    strict: bool = True,
) -> TransformerThreeHeadedModel:
    """Construct a TransformerThreeHeadedModel from a config dict (flat or nested) and
    optionally load weights.

    When ``strict=True`` (default), state_dict keys must match the model exactly.
    When ``strict=False``, mismatched keys are skipped and missing/unexpected
    keys are logged — used by the ``initialize_path`` flow when the runtime
    architecture diverges from the checkpoint's architecture (e.g. a deep
    value head being added on top of a BC-trained trunk).
    """
    model_config = _config_to_flat_arch(model_config)

    model = TransformerThreeHeadedModel(
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
        value_head_layers=model_config.get("value_head_layers", []),
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
        transformer_layers=model_config.get("transformer_layers", 6),
        transformer_heads=model_config.get("transformer_heads", 16),
        transformer_ff_dim=model_config.get("transformer_ff_dim", 2048),
        transformer_dropout=model_config.get("transformer_dropout", 0.1),
        use_decision_tokens=model_config.get("use_decision_tokens", True),
        use_causal_mask=model_config.get("use_causal_mask", True),
        value_to_trunk_grad_scale=model_config.get("value_to_trunk_grad_scale", 1.0),
    ).to(device)

    if state_dict:
        # Strip _orig_mod. prefix left by torch.compile() before loading
        cleaned_state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
        }
        load_result = model.load_state_dict(cleaned_state_dict, strict=strict)
        if not strict:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            if missing:
                print(
                    f"Partial state_dict load: {len(missing)} missing keys "
                    f"(fresh-initialized): {missing[:20]}"
                )
            if unexpected:
                print(
                    f"Partial state_dict load: {len(unexpected)} unexpected keys "
                    f"(dropped): {unexpected[:20]}"
                )

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> Tuple[TransformerThreeHeadedModel, Embedder, Dict[str, Any]]:
    """Load a model + embedder from a checkpoint file. Returns (model, embedder, config_dict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    if embedder is None:
        cfg = RNaDConfig.from_dict(config_dict)
        embedder = Embedder(
            format=cfg.curriculum.primary_format,
            feature_set=cfg.training.embedder_feature_set,
            omniscient=False,
        )

    model = build_model_from_config(config_dict, embedder, device, state_dict)
    return model, embedder, config_dict


def save_checkpoint(
    model: RNaDModel,
    learner: "PortfolioRNaDLearner",
    step: int,
    config: RNaDConfig,
    curriculum: Dict[str, Any],
    save_dir: str = "data/models",
) -> str:
    """Save model + learner state (optimizer, scheduler, step counter), config, curriculum.

    Pulling state through the learner (instead of just `optimizer`) lets us
    persist the LR scheduler's `last_epoch` and the learner's internal
    `_step` counter. Without those, resume rewinds the LR warmup and the
    entropy-bonus annealing back to zero — see
    planning/stage2/2026-05-14-17-00-resume-state-bugs.md.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"main_model_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "scheduler_state_dict": learner.scheduler.state_dict(),
        "learner_step": learner._step,
        "step": step,
        "curriculum": curriculum,
        "config": config.to_dict(),
        "timestamp": timestamp_iso(),
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(filepath: str, device: str) -> Dict[str, Any]:
    """Read a checkpoint file from disk and return its raw contents.

    This function is intentionally side-effect-free. Callers must wire state
    into the model, optimizer, scheduler, and learner step counter in the
    correct order — model weights MUST be loaded into the model before the
    learner is constructed, otherwise `PortfolioRNaDLearner`'s initial
    `ref_models` deepcopy captures random weights and the RNaD KL term
    pulls the policy toward a random anchor on every resume. Use
    `PortfolioRNaDLearner.load_resume_state` for the optimizer/scheduler/
    step counter half.
    """
    print(f"Resuming from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    print(f"Resumed from step {checkpoint['step']}")
    return checkpoint
