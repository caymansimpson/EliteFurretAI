import itertools
from typing import Dict, List

import numpy as np
import torch
from poke_env.environment import AbstractBattle, DoubleBattle
from poke_env.player.battle_order import BattleOrder, StringBattleOrder
from stable_baselines3 import DQN

from elitefurretai.agents.abstract_vgc_model_player import AbstractVGCModelPlayer
from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.utils.battle_order_validator import is_valid_order


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
        self._model = DQN(
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
        self._model = DQN.load(filepath)

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

            # Access the Q-network directly
            q_values = self._model.policy.q_net(obs_tensor)

            # Convert to pythonic list
            return q_values.numpy().tolist()

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

    def action_to_move(self, action: int) -> StringBattleOrder:
        first_order = action % len(_ORDER_MAPPINGS)
        second_order = int(action / len(_ORDER_MAPPINGS))
        return StringBattleOrder(
            _ORDER_MAPPINGS[first_order].message
            + ", "
            + _ORDER_MAPPINGS[second_order].message.replace("/choose ", "")
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)

        # Get model prediction based on the battle state
        emb = self.embed_battle_state(battle)
        output = self._get_model_output(emb)

        # Generate Action Mask
        action_mask = [0] * len(output)
        for index in range(len(output)):
            action_mask[index] = int(
                is_valid_order(self.action_to_move(index), battle)  # type:ignore
            )

        # If we find no valid actions, we need to throw an error
        if sum(action_mask) == 0:
            print(battle_to_str(battle))
            raise ValueError("No valid moves in battle")

        # Softmax
        probabilities = np.exp(output) * action_mask
        probabilities = probabilities / np.sum(probabilities)

        # If probabilistic, sample a move proportiaonal to the softmax
        # otherwise, choose the best move
        if self._probabilistic:
            return self.action_to_move(
                np.random.choice(range(len(probabilities)), p=probabilities)
            )
        else:
            return self.action_to_move(int(np.argmax(probabilities)))

    def teampreview(self, battle: AbstractBattle) -> str:
        assert isinstance(battle, DoubleBattle)

        # Get model prediction based on the battle state
        emb = self.embed_battle_state(battle)
        output = self._get_model_output(emb)

        # Generate Action Mask
        action_mask = [0] * len(output)
        for index in range(len(output)):
            action_mask[index] = int(index in _TEAMPREVIEW_MAPPINGS)

        # If we find no valid actions, we need to throw an error
        if sum(action_mask) == 0:
            print(battle_to_str(battle))
            raise ValueError("No valid teampreview in battle")

        # Softmax
        probabilities = np.exp(output) * action_mask
        probabilities = probabilities / np.sum(probabilities)

        # If probabilistic, sample a move proportiaonal to the softmax
        # otherwise, choose the best move
        if self._probabilistic:
            return _TEAMPREVIEW_MAPPINGS[
                np.random.choice(range(len(probabilities)), p=probabilities)
            ]
        else:
            return _TEAMPREVIEW_MAPPINGS[int(np.argmax(probabilities))]

    # Save it to the battle_filepath using DataProcessor, using opponent information
    # to create omniscient BattleData object
    def _battle_finished_callback(self, battle: AbstractBattle):
        raise NotImplementedError

    """
    ENVIRONMENT-BASED METHODS
    """

    @property
    def action_space(self) -> int:
        return len(_ORDER_MAPPINGS) * len(_ORDER_MAPPINGS)

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


_ORDER_MAPPINGS: Dict[int, StringBattleOrder] = {
    0: StringBattleOrder("/choose move 1 -1"),  # first move, self-target
    1: StringBattleOrder("/choose move 1 1"),  # first move, target first mon
    2: StringBattleOrder("/choose move 1 2"),  # first move, target second mon
    3: StringBattleOrder("/choose move 1 0"),  # first move, no target
    4: StringBattleOrder(
        "/choose move 1 -1 terastallize"
    ),  # first move, self-target, tera
    5: StringBattleOrder(
        "/choose move 1 1 terastallize"
    ),  # first move, target first mon, tera
    6: StringBattleOrder(
        "/choose move 1 2 terastallize"
    ),  # first move, target second mon, tera
    7: StringBattleOrder("/choose move 1 0 terastallize"),  # first move, no target, tera
    8: StringBattleOrder("/choose move 2 -1"),  # second move, self-target
    9: StringBattleOrder("/choose move 2 1"),  # second move, target first mon
    10: StringBattleOrder("/choose move 2 2"),  # second move, target second mon
    11: StringBattleOrder("/choose move 2 0"),  # second move, no target
    12: StringBattleOrder(
        "/choose move 2 -1 terastallize"
    ),  # second move, self-target, tera
    13: StringBattleOrder(
        "/choose move 2 1 terastallize"
    ),  # second move, target first mon, tera
    14: StringBattleOrder(
        "/choose move 2 2 terastallize"
    ),  # second move, target second mon, tera
    15: StringBattleOrder("/choose move 2 0 terastallize"),  # second move, no target, tera
    16: StringBattleOrder("/choose move 3 -1"),  # third move, self-target
    17: StringBattleOrder("/choose move 3 1"),  # third move, target first mon
    18: StringBattleOrder("/choose move 3 2"),  # third move, target second mon
    19: StringBattleOrder("/choose move 3 0"),  # third move, no target
    20: StringBattleOrder(
        "/choose move 3 -1 terastallize"
    ),  # third move, self-target, tera
    21: StringBattleOrder(
        "/choose move 3 1 terastallize"
    ),  # third move, target first mon, tera
    22: StringBattleOrder(
        "/choose move 3 2 terastallize"
    ),  # third move, target second mon, tera
    23: StringBattleOrder("/choose move 3 0 terastallize"),  # third move, no target, tera
    24: StringBattleOrder("/choose move 4 -1"),  # fourth move, self-target
    25: StringBattleOrder("/choose move 4 1"),  # fourth move, target first mon
    26: StringBattleOrder("/choose move 4 2"),  # fourth move, target second mon
    27: StringBattleOrder("/choose move 4 0"),  # fourth move, no target
    28: StringBattleOrder(
        "/choose move 4 -1 terastallize"
    ),  # fourth move, self-target, tera
    29: StringBattleOrder(
        "/choose move 4 1 terastallize"
    ),  # fourth move, target first mon, tera
    30: StringBattleOrder(
        "/choose move 4 2 terastallize"
    ),  # fourth move, target second mon, tera
    31: StringBattleOrder("/choose move 4 0 terastallize"),  # fourth move, no target, tera
    32: StringBattleOrder("/choose switch 1"),  # switch first mon
    33: StringBattleOrder("choose switch 2"),  # switch second mon
    34: StringBattleOrder("/choose switch 3"),  # switch third mon
    35: StringBattleOrder("/choose switch 4"),  # switch fourth mon
    36: StringBattleOrder("/choose pass"),
}

_TEAMPREVIEW_MAPPINGS: Dict[int, str] = {
    i: "/team " + "".join([str(x) for x in permutation])
    for i, permutation in enumerate(itertools.permutations(range(1, 7), 2))
}
