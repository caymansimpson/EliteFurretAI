# -*- coding: utf-8 -*-
"""
Learner implementations for RL training.
Supports PPO with action masking and behavioral cloning initialization.
"""

import torch
from abc import ABC, abstractmethod
import time
from typing import Dict

from elitefurretai.rl.agent import ActorCritic


class BaseLearner(ABC):
    """
    An abstract base class for a Learner.

    It defines the main training loop, pulling experience from workers and pushing
    updated weights back to them. Subclasses must implement the specific update logic.
    """
    def __init__(self, state_dim, action_dim, weights_queue, experience_queue, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Learner using device: {self.device}")

        # Initialize or load BC model
        bc_model_path = config.get('bc_model_path', None)
        if bc_model_path:
            print(f"Loading BC model from {bc_model_path} for warm-start...")
            self.model = ActorCritic.create_from_bc_model(
                bc_model_path,
                state_dim,
                action_dim,
                hidden_sizes=config.get('hidden_sizes', [512, 256])
            ).to(self.device)
        else:
            self.model = ActorCritic(
                state_dim,
                action_dim,
                hidden_sizes=config.get('hidden_sizes', [512, 256])
            ).to(self.device)

        self.weights_queue = weights_queue
        self.experience_queue = experience_queue
        self.config = config

        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

        self._setup_optimizer()

    @abstractmethod
    def _setup_optimizer(self):
        """Initializes the optimizer(s) for the learning algorithm."""
        pass

    @abstractmethod
    def _update(self, experience) -> Dict[str, float]:
        """
        Performs the algorithm-specific update step using a batch of experience.
        This method must be implemented by subclasses.
        """
        pass

    def learn(self):
        """
        The main learning loop. This runs indefinitely.
        """
        # Distribute initial weights before starting
        self.distribute_weights()
        print("Initial weights distributed. Starting learning loop...")

        start_time = time.time()
        last_log_time = start_time

        while True:
            # 1. Get a batch of experience from a worker
            experience = self.experience_queue.get()

            # Extract stats
            worker_id = experience.get('worker_id', -1)
            episode_stats = experience.get('episode_stats', [])

            # Update statistics
            self.total_steps += len(experience['states'])
            self.total_episodes += len(episode_stats)

            # 2. Perform the algorithm-specific update
            update_info = self._update(experience)
            self.update_count += 1

            # 3. Distribute the new weights
            self.distribute_weights()

            # 4. Log progress
            current_time = time.time()
            if current_time - last_log_time >= 10.0:  # Log every 10 seconds
                elapsed = current_time - start_time
                steps_per_sec = self.total_steps / elapsed if elapsed > 0 else 0

                avg_reward = sum(ep['episode_reward'] for ep in episode_stats) / len(episode_stats) if episode_stats else 0
                avg_length = sum(ep['episode_length'] for ep in episode_stats) / len(episode_stats) if episode_stats else 0
                win_rate = sum(1 for ep in episode_stats if ep.get('won')) / len(episode_stats) if episode_stats else 0

                print(f"[Learner] Steps: {self.total_steps}, Episodes: {self.total_episodes}, "
                      f"Updates: {self.update_count}, Worker: {worker_id}")
                print(f"  Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                      f"Win Rate: {win_rate:.2%}, Steps/sec: {steps_per_sec:.1f}")
                if update_info:
                    print(f"  Update Info: {update_info}")

                last_log_time = current_time

    def distribute_weights(self):
        """
        Puts the current model's state_dict onto the weights queue.
        The weights are moved to CPU before being put on the queue.
        """
        weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        # Non-blocking put to avoid queue filling up
        try:
            while not self.weights_queue.empty():
                self.weights_queue.get_nowait()  # Clear old weights
        except Exception:
            pass
        self.weights_queue.put(weights)


class PPOLearner(BaseLearner):
    """
    A Learner that implements the Proximal Policy Optimization (PPO) algorithm.
    Supports action masking for valid move selection.
    """
    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

    def _update(self, experience) -> Dict[str, float]:
        """
        Performs the PPO update step with action masking support.
        """
        states = experience['states'].to(self.device)
        actions = experience['actions'].to(self.device)
        old_log_probs = experience['log_probs'].to(self.device)
        returns = experience['returns'].to(self.device)
        advantages = experience['advantages'].to(self.device)
        action_masks = experience.get('action_masks', None)

        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        batch_size = states.size(0)
        minibatch_size = batch_size // self.config['num_minibatches']

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.config['ppo_epochs']):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(0, batch_size, minibatch_size):
                end = min(i + minibatch_size, batch_size)
                mb_indices = indices[i:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_action_masks = action_masks[mb_indices] if action_masks is not None else None

                # Get new values, log probs, and entropy
                new_values, new_log_probs, entropy = self.model.get_value_and_log_prob(
                    mb_states, mb_actions, mb_action_masks
                )

                # PPO clipped objective
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = mb_advantages * ratio
                surr2 = mb_advantages * torch.clamp(
                    ratio,
                    1 - self.config['clip_coef'],
                    1 + self.config['clip_coef']
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((new_values.view(-1) - mb_returns) ** 2).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config['vf_coef'] * value_loss
                    - self.config['ent_coef'] * entropy.mean()
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        # Return update statistics
        num_updates = self.config['ppo_epochs'] * (batch_size // minibatch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }


class MMDLearner(BaseLearner):
    """
    A Learner that implements Magnetic Mirror Descent (MMD) with dilated entropy.

    MMD is a second-order method that can offer more stable updates. Dilated
    entropy helps in encouraging exploration while staying close to a reference policy.
    This implementation is based on interpretations of MMD for policy optimization.

    Supports action masking for valid move selection.
    """
    def _setup_optimizer(self):
        # MMD often uses separate optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor_head.parameters(),
            lr=self.config['learning_rate']
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.critic_head.parameters()) + list(self.model.shared_body.parameters()),
            lr=self.config['learning_rate']
        )

        # We also need a reference model for the dilated entropy calculation
        self.reference_model = ActorCritic(
            self.config['state_dim'],
            self.config['action_dim'],
            hidden_sizes=self.config.get('hidden_sizes', [512, 256])
        ).to(self.device)
        self.reference_model.load_state_dict(self.model.state_dict())
        self.reference_model.eval()

    def _update(self, experience) -> Dict[str, float]:
        """
        Performs the MMD update step with action masking support.
        This involves a policy loss based on advantages and a dilated entropy regularizer,
        and a standard value loss.
        """
        states = experience['states'].to(self.device)
        actions = experience['actions'].to(self.device)
        returns = experience['returns'].to(self.device)
        advantages = experience['advantages'].to(self.device)
        action_masks = experience.get('action_masks', None)

        if action_masks is not None:
            action_masks = action_masks.to(self.device)

        batch_size = states.size(0)
        minibatch_size = batch_size // self.config['num_minibatches']

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_div = 0.0

        for epoch in range(self.config.get('mmd_epochs', 10)):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(0, batch_size, minibatch_size):
                end = min(i + minibatch_size, batch_size)
                mb_indices = indices[i:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_action_masks = action_masks[mb_indices] if action_masks is not None else None

                # --- Critic Update ---
                # The critic update is standard: minimize MSE between predicted values and actual returns.
                new_values, _, _ = self.model.get_value_and_log_prob(
                    mb_states, mb_actions, mb_action_masks
                )
                value_loss = 0.5 * ((new_values.view(-1) - mb_returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.critic_head.parameters()) + list(self.model.shared_body.parameters()),
                    0.5
                )
                self.critic_optimizer.step()

                # --- Actor Update (MMD with Dilated Entropy) ---
                # Get action distribution from current and reference policies
                current_logits, _ = self.model.forward(mb_states, mb_action_masks)
                current_dist = torch.distributions.Categorical(logits=current_logits)

                with torch.no_grad():
                    ref_logits, _ = self.reference_model.forward(mb_states, mb_action_masks)
                    ref_dist = torch.distributions.Categorical(logits=ref_logits)

                # Dilated Entropy = KL(current || reference)
                kl_div = torch.distributions.kl.kl_divergence(current_dist, ref_dist).mean()

                # Policy Gradient Loss
                log_probs = current_dist.log_prob(mb_actions)
                policy_gradient_loss = -(mb_advantages * log_probs).mean()

                # Total actor loss for MMD
                # The 'mmd_beta' parameter balances the PG loss and the KL regularizer
                actor_loss = policy_gradient_loss + self.config.get('mmd_beta', 1.0) * kl_div

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor_head.parameters(), 0.5)
                self.actor_optimizer.step()

                total_policy_loss += policy_gradient_loss.item()
                total_value_loss += value_loss.item()
                total_kl_div += kl_div.item()

        # After all epochs, update the reference model to the new policy
        self.reference_model.load_state_dict(self.model.state_dict())

        # Return update statistics
        num_updates = self.config.get('mmd_epochs', 10) * (batch_size // minibatch_size)
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'kl_divergence': total_kl_div / num_updates
        }
