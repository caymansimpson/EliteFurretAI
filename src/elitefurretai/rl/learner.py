import torch
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict

from elitefurretai.rl.agent import RNaDAgent


class RNaDLearner:
    def __init__(
        self,
        model: RNaDAgent,
        ref_model: RNaDAgent,
        lr: float = 1e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        rnad_alpha: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision: bool = True,
        gradient_clip: float = 0.5
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.rnad_alpha = rnad_alpha
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip = gradient_clip

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler(device=device) if use_mixed_precision else None  # type: ignore

        # Freeze ref model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def update_ref_model(self):
        """Updates the reference model with the current model's weights."""
        self.ref_model.load_state_dict(self.model.state_dict())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Performs a PPO + RNaD update step.

        Args:
            batch: Dictionary containing:
                - states: (batch, seq, dim)
                - actions: (batch, seq)
                - rewards: (batch, seq)
                - values: (batch, seq) - Old values
                - log_probs: (batch, seq) - Old log probs
                - masks: (batch, seq, action_dim) - Action masks (optional)
                - is_teampreview: (batch, seq) - Boolean mask
                - advantages: (batch, seq)
                - returns: (batch, seq)
                - initial_hidden: (h, c) tuple (optional)
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        is_teampreview = batch['is_teampreview'].to(self.device)
        padding_mask = batch.get('padding_mask', None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)

        action_masks = batch.get('masks', None)
        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        # Handle hidden states
        # If not provided, init to zero
        initial_hidden = batch.get('initial_hidden', None)
        if initial_hidden is None:
            initial_hidden_state = self.model.get_initial_state(states.shape[0], self.device)
        else:
            # Move to device
            initial_hidden_state = (initial_hidden[0].to(self.device), initial_hidden[1].to(self.device))

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=self.device, enabled=self.use_mixed_precision):  # type: ignore
            # Forward pass current model
            turn_logits, tp_logits, values, _ = self.model(states, initial_hidden_state)

        # Forward pass ref model (no grad, always FP32 for stability)
        with torch.no_grad():
            ref_turn_logits, ref_tp_logits, _, _ = self.ref_model(states, initial_hidden_state)

        # Flatten batch and seq dimensions for processing
        batch_size, seq_len = actions.shape
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_is_tp = is_teampreview.reshape(-1).bool()
        flat_values = values.reshape(-1)

        if padding_mask is not None:
            flat_padding_mask = padding_mask.reshape(-1).bool()
            # Filter out padded steps from all flat tensors?
            # Or just mask the loss.
            # Masking loss is easier but we need to be careful with indices.
            # Let's just use the mask to zero out loss elements.
        else:
            flat_padding_mask = torch.ones_like(flat_values, dtype=torch.bool)

        # Initialize losses
        total_policy_loss = 0.0
        total_entropy_loss = 0.0
        total_rnad_loss = 0.0

        # --- Process Teampreview Steps ---
        # Combine TP mask and Padding mask
        valid_tp_mask = flat_is_tp & flat_padding_mask

        if valid_tp_mask.any():
            tp_indices = torch.nonzero(valid_tp_mask).squeeze()

            # Get logits for TP steps
            curr_tp_logits = tp_logits.reshape(-1, tp_logits.shape[-1])[tp_indices]
            ref_tp_logits_sel = ref_tp_logits.reshape(-1, ref_tp_logits.shape[-1])[tp_indices]

            # Apply masks if available (assuming mask shape matches logits or is handled)
            # TP masks might be different from Turn masks.
            # For now, assume no masking for TP or handled elsewhere.

            curr_dist = Categorical(logits=curr_tp_logits)
            ref_dist = Categorical(logits=ref_tp_logits_sel)

            # Log probs
            curr_log_probs = curr_dist.log_prob(flat_actions[tp_indices])

            # RNaD KL Divergence
            # KL(pi || pi_ref)
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist)
            total_rnad_loss += kl.mean().item()

            # PPO Loss
            ratio = torch.exp(curr_log_probs - flat_old_log_probs[tp_indices])
            surr1 = ratio * flat_advantages[tp_indices]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * flat_advantages[tp_indices]
            total_policy_loss += -torch.min(surr1, surr2).mean().item()

            # Entropy
            total_entropy_loss += curr_dist.entropy().mean()

        # --- Process Turn Steps ---
        valid_turn_mask = (~flat_is_tp) & flat_padding_mask

        if valid_turn_mask.any():
            turn_indices = torch.nonzero(valid_turn_mask).squeeze()

            curr_turn_logits = turn_logits.reshape(-1, turn_logits.shape[-1])[turn_indices]
            ref_turn_logits_sel = ref_turn_logits.reshape(-1, ref_turn_logits.shape[-1])[turn_indices]

            # Apply masks
            if action_masks is not None:
                # Reshape masks: (batch, seq, action_dim) -> (batch*seq, action_dim)
                flat_masks = action_masks.reshape(-1, action_masks.shape[-1])
                curr_masks = flat_masks[turn_indices]

                curr_turn_logits = curr_turn_logits.masked_fill(~curr_masks.bool(), float('-inf'))
                ref_turn_logits_sel = ref_turn_logits_sel.masked_fill(~curr_masks.bool(), float('-inf'))

            curr_dist = Categorical(logits=curr_turn_logits)
            ref_dist = Categorical(logits=ref_turn_logits_sel)

            curr_log_probs = curr_dist.log_prob(flat_actions[turn_indices])

            # RNaD KL
            kl = torch.distributions.kl_divergence(curr_dist, ref_dist)
            total_rnad_loss += kl.mean().item()

            # PPO
            ratio = torch.exp(curr_log_probs - flat_old_log_probs[turn_indices])
            surr1 = ratio * flat_advantages[turn_indices]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * flat_advantages[turn_indices]
            total_policy_loss += -torch.min(surr1, surr2).mean().item()

            # Entropy
            total_entropy_loss += curr_dist.entropy().mean()

        # Value Loss (with padding mask)
        value_error = (flat_values - flat_returns) ** 2
        if padding_mask is not None:
            value_loss = 0.5 * (value_error * flat_padding_mask.float()).sum() / flat_padding_mask.sum().clamp(min=1.0)
        else:
            value_loss = 0.5 * value_error.mean()

        # Total Loss
        loss = (
            total_policy_loss
            + self.vf_coef * value_loss
            - self.ent_coef * total_entropy_loss
            + self.rnad_alpha * total_rnad_loss
        )

        # Backward pass with mixed precision
        self.optimizer.zero_grad()
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": total_policy_loss.item() if isinstance(total_policy_loss, torch.Tensor) else total_policy_loss,
            "value_loss": value_loss.item(),
            "entropy": total_entropy_loss.item() if isinstance(total_entropy_loss, torch.Tensor) else total_entropy_loss,
            "rnad_loss": total_rnad_loss.item() if isinstance(total_rnad_loss, torch.Tensor) else total_rnad_loss
        }
