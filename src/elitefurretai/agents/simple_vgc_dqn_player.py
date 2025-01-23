from typing import List

import numpy as np
import torch
from poke_env.environment import AbstractBattle, DoubleBattle
from poke_env.player.battle_order import BattleOrder

from elitefurretai.agents.abstract_vgc_model_player import AbstractVGCModelPlayer
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.model_battle_order import ModelBattleOrder
from elitefurretai.models.abstract_model import AbstractModel


# We define our RL player
class SimpleVGCDQNPlayer(AbstractVGCModelPlayer):
    def __init__(
        self,
        battle_format: str = "gen9vgc2024regh",
        simple: bool = True,
        probabilistic=True,
        **kwargs,
    ):
        # pull in all player manually
        super().__init__(**kwargs)
        self._embedder = Embedder(format=battle_format, simple=simple)
        self._probabilistic = probabilistic

        # The model that we use to make predictions
        self._model = None

    """
    MODEL-BASED METHODS
    """

    def create_model(self, **kwargs):
        assert self._model is None
        self._model = AbstractModel(
            kwargs.get("policy"),  # type:ignore
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

    def save_model(self, filepath: str):
        assert self._model is not None
        self._model.save(filepath)

    def load_model(self, filepath: str):
        assert self._model is None
        self._model = AbstractModel.load(filepath)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def embed_battle_state(self, battle: AbstractBattle) -> List[float]:
        assert isinstance(battle, DoubleBattle)
        return self._embedder.feature_dict_to_vector(
            self._embedder.featurize_double_battle(battle)
        )

    def _get_model_output(self, observation: List[float]) -> List[float]:
        assert self._model is not None

        # Convert observation to tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():

            # Access the network directly
            return self._model.predict(obs_tensor)  # type:ignore

    # Same shit as embed_battle_state, but in environment-related language
    def get_observation(self, battle: AbstractBattle):
        return self.embed_battle_state(battle)

    """
    PLAYER-BASED METHODS
    """

    @property
    def probabilistic(self):
        return self._probabilistic

    @probabilistic.setter
    def probabilistic(self, value: bool):
        self._probabilistic = value

    def action_to_move(self, action: int, battle: AbstractBattle) -> BattleOrder:
        return ModelBattleOrder.from_int(action, battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)

        # Get model prediction based on the battle state
        emb = self.embed_battle_state(battle)
        output = self._get_model_output(emb)

        # Generate Action Mask
        mask = ModelBattleOrder.action_mask(battle)
        assert sum(mask) != 0, "No valid moves in battle"

        # Softmax
        probabilities = np.exp(output) * mask
        probabilities = probabilities / np.sum(probabilities)

        # If probabilistic, sample a move proportiaonal to the softmax
        # otherwise, choose the best move
        if self._probabilistic:
            choice = np.random.choice(range(len(probabilities)), p=probabilities)
        else:
            choice = int(np.argmax(probabilities))

        return self.action_to_move(choice, battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)

        # Get model prediction based on the battle state
        emb = self.embed_battle_state(battle)
        output = self._get_model_output(emb)

        # Generate Action Mask
        mask = ModelBattleOrder.action_mask(battle)
        assert sum(mask) != 0, "No valid moves in battle"

        # Softmax
        probabilities = np.exp(output) * mask
        probabilities = probabilities / np.sum(probabilities)

        # If probabilistic, sample a move proportiaonal to the softmax
        # otherwise, choose the best move
        if self._probabilistic:
            choice = np.random.choice(range(len(probabilities)), p=probabilities)
        else:
            choice = int(np.argmax(probabilities))
        return ModelBattleOrder.from_int(choice, battle).message

    # Save it to the battle_filepath using DataProcessor, using opponent information
    # to create omniscient BattleData object
    def _battle_finished_callback(self, battle: AbstractBattle):
        raise NotImplementedError

    """
    ENVIRONMENT-BASED METHODS
    """

    @property
    def action_space(self) -> int:
        return ModelBattleOrder.action_space()

    @property
    def observation_space(self) -> int:
        return self._embedder.embedding_size

    def calculate_reward(self, battle: AbstractBattle) -> float:
        if battle.won:
            return 1.0
        elif battle.lost:
            return -1.0
        else:
            return 0.0

    def get_info(self, battle: AbstractBattle) -> dict:
        return {}
