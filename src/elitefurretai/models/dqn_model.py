# -*- coding: utf-8 -*-
"""A wrapper class for a model. DEPRECATED!!!!!!!!
"""
from typing import List

import torch
from stable_baselines3 import DQN

from elitefurretai.models.abstract_model import AbstractModel


class DQNModel(AbstractModel):

    def __init__(self):
        self._model = None
        raise NotImplementedError

    def create(self, **kwargs):

        self._model = DQN(
            kwargs.get("policy"),  # type: ignore
            kwargs.get("env"),  # type:ignore
            verbose=kwargs.get("verbose", 1),  # Print training progress
            learning_rate=kwargs.get(
                "learning_rate", 1e-4
            ),  # Learning rate for the optimizer
            buffer_size=kwargs.get("buffer_size", 10000),  # Size of the replay buffer
            learning_starts=kwargs.get(
                "learning_starts", 1000
            ),  # Number of steps before learning starts
            batch_size=kwargs.get("batch_size", 32),  # Batch size for training
            tau=kwargs.get("tau", 1.0),  # Target network update rate
            gamma=kwargs.get("gamma", 0.99),  # Discount factor
            train_freq=kwargs.get("train_freq", 4),  # Train the model every 4 steps
            gradient_steps=kwargs.get(
                "gradient_steps", 1
            ),  # Number of gradient steps per update
            target_update_interval=kwargs.get(
                "target_update_interval", 1000
            ),  # Update target network every 1000 steps
            exploration_fraction=kwargs.get(
                "exploration_fraction", 0.1
            ),  # Fraction of entire training period over which the exploration rate is reduced
            exploration_initial_eps=kwargs.get(
                "exploration_initial_eps", 1.0
            ),  # Initial exploration probability
            exploration_final_eps=kwargs.get(
                "exploration_final_eps", 0.05
            ),  # Final exploration probability
        )

    def train(self, **kwargs):
        assert self._model is not None
        self._model.learn(total_timesteps=kwargs.get("total_timesteps", 50000))

    def save(self, filepath: str):
        assert self._model is not None
        self._model.save(filepath)

    def load(self, filepath: str):
        assert self._model is not None
        self._model.load(filepath)

    def predict(self, observation: List[float]) -> List[float]:
        assert self._model is not None

        # Convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Get action probabilities
        with torch.no_grad():
            # Access the Q-network directly
            q_values = self._model.policy.q_net(obs_tensor)

            # Convert Q-values to probabilities using softmax
            action_probs = torch.softmax(q_values, dim=1)

            # Convert to numpy for easier handling
            return action_probs.numpy()[0]
