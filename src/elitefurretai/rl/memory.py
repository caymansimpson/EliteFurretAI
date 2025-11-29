import torch
import numpy as np
import os
from pathlib import Path
from typing import Optional


class ExperienceBuffer:
    """
    A buffer to store trajectories of experience collected by a worker.
    It is also responsible for calculating the advantages and returns
    using General Advantage Estimation (GAE) at the end of a rollout.

    Supports file-system based sharing for memory efficiency in WSL environments.
    """
    def __init__(self, buffer_size, gamma, gae_lambda, device, checkpoint_dir: Optional[str] = None):
        """
        Initializes the buffer.

        Args:
            buffer_size (int): The maximum number of steps to store (e.g., 2048).
            gamma (float): The discount factor.
            gae_lambda (float): The lambda for GAE.
            device (torch.device): The device to store tensors on (e.g., 'cpu').
            checkpoint_dir (str, optional): Directory to save buffer checkpoints for disk-based sharing.
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.clear()

    def clear(self):
        """Resets the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []  # Store action masks for masked policy training
        self.ptr = 0

    def store(self, state, action, reward, value, log_prob, done, action_mask=None):
        """
        Adds a single step of experience to the buffer.

        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode terminated
            action_mask: Binary mask of valid actions (optional)
        """
        if len(self.states) < self.buffer_size:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.values.append(None)
            self.log_probs.append(None)
            self.dones.append(None)
            self.action_masks.append(None)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask if action_mask is not None else None
        self.ptr = (self.ptr + 1) % self.buffer_size

    def calculate_advantages(self, last_value, done):
        """
        Calculates advantages and returns for the stored trajectory using GAE.

        Args:
            last_value (float): The value estimate of the final state in the
                                trajectory, used for bootstrapping.
            done (bool): Whether the final state was a terminal state.
        """
        path_length = len(self.rewards)
        self.advantages = np.zeros(path_length, dtype=np.float32)
        self.returns = np.zeros(path_length, dtype=np.float32)

        # If the last state was not terminal, bootstrap with its value
        last_gae_lam = 0
        next_value = last_value if not done else 0.0

        for t in reversed(range(path_length)):
            # GAE calculation
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae_lam
            self.advantages[t] = last_gae_lam
            next_value = self.values[t]

        # Calculate returns by adding advantages to values
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

    def save_to_disk(self, worker_id: int):
        """
        Save buffer contents to disk for file-system based sharing.

        Args:
            worker_id (int): Unique worker identifier.
        """
        if not self.checkpoint_dir:
            return

        checkpoint_path = os.path.join(self.checkpoint_dir, f"worker_{worker_id}_buffer.pt")

        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'dones': self.dones,
            'action_masks': self.action_masks,
            'advantages': self.advantages if hasattr(self, 'advantages') else None,
            'returns': self.returns if hasattr(self, 'returns') else None,
        }, checkpoint_path)

    @staticmethod
    def load_from_disk(checkpoint_dir: str, worker_id: int, device):
        """
        Load buffer contents from disk.

        Args:
            checkpoint_dir (str): Directory containing buffer checkpoints.
            worker_id (int): Unique worker identifier.
            device: Device to load tensors to.

        Returns:
            dict: Buffer data as dictionary of tensors.
        """
        checkpoint_path = os.path.join(checkpoint_dir, f"worker_{worker_id}_buffer.pt")

        if not os.path.exists(checkpoint_path):
            return None

        data = torch.load(checkpoint_path, map_location=device)
        return data

    def get(self):
        """
        Returns all stored data as a dictionary of tensors, ready for the learner.

        Returns:
            dict: A dictionary containing batches of states, actions, log_probs,
                  returns, advantages, and action_masks.
        """
        # Ensure the buffer is full before calling get
        assert self.ptr == 0 or len(self.states) == self.buffer_size, \
            "Buffer is not properly filled. Call get() only after rollout is complete."

        # Convert lists to tensors
        states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(self.actions), dtype=torch.int64).to(self.device)
        log_probs_tensor = torch.tensor(np.array(self.log_probs), dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(self.returns, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(self.advantages, dtype=torch.float32).to(self.device)

        # Handle action masks (may be None for some entries)
        if self.action_masks[0] is not None:
            action_masks_tensor = torch.tensor(np.array(self.action_masks), dtype=torch.int8).to(self.device)
        else:
            action_masks_tensor = None

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        return dict(
            states=states_tensor,
            actions=actions_tensor,
            log_probs=log_probs_tensor,
            returns=returns_tensor,
            advantages=advantages_tensor,
            action_masks=action_masks_tensor
        )
