"""
Portfolio-based RNaD Learner with Mixed Precision Training.

Maintains multiple reference models and regularizes against the best one,
inspired by Ataraxos paper. Uses mixed precision (FP16) for faster training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Any, List, Optional
import copy

from elitefurretai.rl2.agent import RNaDAgent


class PortfolioRNaDLearner:
    """
    Enhanced RNaD learner with portfolio regularization and mixed precision.
    
    Key improvements over base RNaDLearner:
    1. Portfolio of reference models (instead of single reference)
    2. Regularizes against closest/best reference model
    3. Mixed precision training (FP16) for 2x speedup
    4. Automatic portfolio management (add/remove references)
    """
    
    def __init__(
        self, 
        model: RNaDAgent, 
        ref_models: List[RNaDAgent],  # Portfolio of reference models
        lr: float = 1e-4, 
        gamma: float = 0.99, 
        clip_range: float = 0.2, 
        ent_coef: float = 0.01, 
        vf_coef: float = 0.5, 
        rnad_alpha: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_mixed_precision: bool = True,
        max_portfolio_size: int = 5,
        portfolio_update_strategy: str = "diverse"  # "diverse", "best", "recent"
    ):
        """
        Args:
            model: Main model being trained
            ref_models: List of reference models for portfolio regularization
            lr: Learning rate
            gamma: Discount factor
            clip_range: PPO clipping range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            rnad_alpha: RNaD regularization strength
            device: Device to train on
            use_mixed_precision: Enable FP16 mixed precision training
            max_portfolio_size: Maximum number of reference models to keep
            portfolio_update_strategy: How to manage portfolio ("diverse", "best", "recent")
        """
        self.model = model.to(device)
        self.ref_models = [ref.to(device) for ref in ref_models]
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.rnad_alpha = rnad_alpha
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.max_portfolio_size = max_portfolio_size
        self.portfolio_update_strategy = portfolio_update_strategy
        
        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Freeze ref models
        for ref_model in self.ref_models:
            for param in ref_model.parameters():
                param.requires_grad = False
        
        # Track portfolio statistics
        self.portfolio_kl_history = [[] for _ in range(len(ref_models))]
        self.portfolio_selection_counts = [0] * len(ref_models)
        
    def add_reference_model(self, new_ref: RNaDAgent):
        """Add a new reference model to the portfolio."""
        new_ref = new_ref.to(self.device)
        for param in new_ref.parameters():
            param.requires_grad = False
            
        self.ref_models.append(new_ref)
        self.portfolio_kl_history.append([])
        self.portfolio_selection_counts.append(0)
        
        # Prune portfolio if too large
        if len(self.ref_models) > self.max_portfolio_size:
            self._prune_portfolio()
    
    def _prune_portfolio(self):
        """Remove least useful reference model based on strategy."""
        if len(self.ref_models) <= 1:
            return  # Keep at least one
            
        if self.portfolio_update_strategy == "recent":
            # Remove oldest (first in list)
            self.ref_models.pop(0)
            self.portfolio_kl_history.pop(0)
            self.portfolio_selection_counts.pop(0)
            
        elif self.portfolio_update_strategy == "best":
            # Remove model selected least often
            min_idx = np.argmin(self.portfolio_selection_counts)
            self.ref_models.pop(min_idx)
            self.portfolio_kl_history.pop(min_idx)
            self.portfolio_selection_counts.pop(min_idx)
            
        elif self.portfolio_update_strategy == "diverse":
            # Remove model most similar to others (highest avg KL to other refs)
            # This keeps a diverse portfolio
            # For simplicity, remove least selected (similar to "best")
            min_idx = np.argmin(self.portfolio_selection_counts)
            self.ref_models.pop(min_idx)
            self.portfolio_kl_history.pop(min_idx)
            self.portfolio_selection_counts.pop(min_idx)
    
    def update_main_reference(self):
        """Update the main (most recent) reference model."""
        if len(self.ref_models) > 0:
            # Replace the last reference with current model
            self.ref_models[-1].load_state_dict(self.model.state_dict())
        else:
            # Add current model as first reference
            new_ref = RNaDAgent(copy.deepcopy(self.model.model))
            self.add_reference_model(new_ref)
    
    def _compute_portfolio_kl(self, curr_dist: Categorical, states: torch.Tensor, 
                            initial_hidden, is_teampreview: bool) -> torch.Tensor:
        """
        Compute KL divergence to all reference models and return minimum.
        
        This is the key innovation: regularize against the CLOSEST reference,
        preventing forgetting of multiple strategies.
        """
        min_kl = None
        best_ref_idx = 0
        
        for ref_idx, ref_model in enumerate(self.ref_models):
            with torch.no_grad():
                ref_turn_logits, ref_tp_logits, _, _ = ref_model(states, initial_hidden)
                
                if is_teampreview:
                    ref_logits = ref_tp_logits
                else:
                    ref_logits = ref_turn_logits
                    
                ref_dist = Categorical(logits=ref_logits)
                kl = torch.distributions.kl_divergence(curr_dist, ref_dist).mean()
                
                # Track KL for this reference
                self.portfolio_kl_history[ref_idx].append(kl.item())
                if len(self.portfolio_kl_history[ref_idx]) > 100:
                    self.portfolio_kl_history[ref_idx].pop(0)
                
                if min_kl is None or kl < min_kl:
                    min_kl = kl
                    best_ref_idx = ref_idx
        
        # Track which reference was selected
        self.portfolio_selection_counts[best_ref_idx] += 1
        
        return min_kl if min_kl is not None else torch.tensor(0.0, device=self.device)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Performs a PPO + Portfolio RNaD update step with mixed precision.
        
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
                - padding_mask: (batch, seq) - Valid timesteps
        
        Returns:
            Dictionary of loss components and metrics
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        returns = batch['returns'].to(self.device)
        is_teampreview = batch['is_teampreview'].to(self.device)
        padding_mask = batch.get('padding_mask', torch.ones_like(actions, dtype=torch.bool)).to(self.device)
        
        # Normalize advantages
        valid_advantages = advantages[padding_mask]
        if len(valid_advantages) > 1:
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        
        # Handle hidden states
        initial_hidden = batch.get('initial_hidden', None)
        if initial_hidden is None:
            initial_hidden = self.model.get_initial_state(states.shape[0], self.device)
        else:
            initial_hidden = (initial_hidden[0].to(self.device), initial_hidden[1].to(self.device))
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_mixed_precision):
            # Forward pass current model
            turn_logits, tp_logits, values, _ = self.model(states, initial_hidden)
            
            # Flatten for processing
            batch_size, seq_len = actions.shape
            flat_actions = actions.reshape(-1)
            flat_old_log_probs = old_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_is_tp = is_teampreview.reshape(-1).bool()
            flat_values = values.reshape(-1)
            flat_padding_mask = padding_mask.reshape(-1).bool()
            
            # Initialize losses
            policy_loss_tp = torch.tensor(0.0, device=self.device)
            policy_loss_turn = torch.tensor(0.0, device=self.device)
            entropy_tp = torch.tensor(0.0, device=self.device)
            entropy_turn = torch.tensor(0.0, device=self.device)
            rnad_loss_tp = torch.tensor(0.0, device=self.device)
            rnad_loss_turn = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)
            
            # --- Process Teampreview Steps ---
            valid_tp_mask = flat_is_tp & flat_padding_mask
            if valid_tp_mask.any():
                tp_indices = torch.nonzero(valid_tp_mask, as_tuple=False).squeeze(-1)
                curr_tp_logits = tp_logits.reshape(-1, tp_logits.shape[-1])[tp_indices]
                curr_dist = Categorical(logits=curr_tp_logits)
                
                # Portfolio KL
                rnad_loss_tp = self._compute_portfolio_kl(curr_dist, states, initial_hidden, is_teampreview=True)
                
                # PPO Loss
                curr_log_probs = curr_dist.log_prob(flat_actions[tp_indices])
                ratio = torch.exp(curr_log_probs - flat_old_log_probs[tp_indices])
                surr1 = ratio * flat_advantages[tp_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * flat_advantages[tp_indices]
                policy_loss_tp = -torch.min(surr1, surr2).mean()
                
                # Entropy
                entropy_tp = curr_dist.entropy().mean()
            
            # --- Process Turn Steps ---
            valid_turn_mask = (~flat_is_tp) & flat_padding_mask
            if valid_turn_mask.any():
                turn_indices = torch.nonzero(valid_turn_mask, as_tuple=False).squeeze(-1)
                curr_turn_logits = turn_logits.reshape(-1, turn_logits.shape[-1])[turn_indices]
                curr_dist = Categorical(logits=curr_turn_logits)
                
                # Portfolio KL
                rnad_loss_turn = self._compute_portfolio_kl(curr_dist, states, initial_hidden, is_teampreview=False)
                
                # PPO Loss
                curr_log_probs = curr_dist.log_prob(flat_actions[turn_indices])
                ratio = torch.exp(curr_log_probs - flat_old_log_probs[turn_indices])
                surr1 = ratio * flat_advantages[turn_indices]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * flat_advantages[turn_indices]
                policy_loss_turn = -torch.min(surr1, surr2).mean()
                
                # Entropy
                entropy_turn = curr_dist.entropy().mean()
            
            # --- Value Loss ---
            if flat_padding_mask.any():
                valid_indices = torch.nonzero(flat_padding_mask, as_tuple=False).squeeze(-1)
                value_loss = nn.MSELoss()(flat_values[valid_indices], flat_returns[valid_indices])
            
            # Total loss
            policy_loss = policy_loss_tp + policy_loss_turn
            entropy_loss = entropy_tp + entropy_turn
            rnad_loss = rnad_loss_tp + rnad_loss_turn
            
            total_loss = (
                policy_loss
                + self.vf_coef * value_loss
                - self.ent_coef * entropy_loss
                + self.rnad_alpha * rnad_loss
            )
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'rnad_loss': rnad_loss.item(),
            'portfolio_size': len(self.ref_models),
            'portfolio_selections': dict(enumerate(self.portfolio_selection_counts))
        }
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get statistics about the reference portfolio."""
        stats = {
            'portfolio_size': len(self.ref_models),
            'selection_counts': self.portfolio_selection_counts.copy(),
            'avg_kl_per_ref': [np.mean(kls) if kls else 0.0 for kls in self.portfolio_kl_history]
        }
        return stats
