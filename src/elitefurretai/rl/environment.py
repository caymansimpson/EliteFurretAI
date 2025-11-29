# -*- coding: utf-8 -*-
"""
RL Environment for Pokemon VGC doubles battles.
Integrates poke_env with MDBO encoder and Embedder for RL training.
Uses poke-env's DoublesEnv as base but with custom MDBO action space and Embedder observations.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from gymnasium import spaces
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder
from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder

from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.utils.battle_order_validator import is_valid_order


class VGCDoublesEnv(Player):
    """
    RL Environment for VGC doubles battles using MDBO action encoding and Embedder observations.

    This is a Gym-style environment that wraps a poke-env Player for self-play training.
    Unlike poke-env's DoublesEnv which uses MultiDiscrete([107, 107]) actions,
    this uses MDBO's 2025-dimensional action space which includes all possible
    move/switch/target/gimmick combinations for both Pokemon.

    Observation space: Dict with:
        - 'observation': Box of shape (embedding_size,) from Embedder
        - 'action_mask': MultiBinary of shape (2025,) indicating valid actions

    Action space: Discrete(2025) representing MDBO encoded actions
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2023regulationc",
        opponent: Optional[Player] = None,
        embedder: Optional[Embedder] = None,
        account_configuration: Optional[AccountConfiguration] = None,
        server_configuration: Optional[ServerConfiguration] = None,
        team: Optional[Union[str, Teambuilder]] = None,
        **kwargs
    ):
        """
        Initialize the VGC Doubles RL environment.

        Args:
            battle_format: Pokemon Showdown battle format
            opponent: Opponent player (if None, uses self-play)
            embedder: Custom embedder for observations (if None, creates default)
            account_configuration: Account config for connecting to Showdown
            server_configuration: Server config (defaults to localhost)
            team: Team string or Teambuilder
            **kwargs: Additional Player arguments
        """
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            server_configuration=server_configuration or LocalhostServerConfiguration,
            team=team,
            max_concurrent_battles=1,
            **kwargs
        )

        # Initialize embedder for observations
        self.embedder = embedder or Embedder(
            format=battle_format,
            feature_set=Embedder.FULL,
            omniscient=False
        )

        # Store opponent
        self.opponent = opponent

        # Define Gym spaces
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.embedder.embedding_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.MultiBinary(MDBO.action_space())
        })

        self.action_space = spaces.Discrete(MDBO.action_space())

        # Track current battle
        self._current_battle: Optional[DoubleBattle] = None
        self._current_order: Optional[BattleOrder] = None
        self._reward = 0.0
        self._done = False

        # Initialize episode tracking fields
        self.__model = None
        self.__memory = None
        self.__steps = 0
        self.__max_steps = 0

    @property
    def model(self):
        """Get the current ActorCritic model."""
        return self.__model

    @model.setter
    def model(self, value):
        """Set the ActorCritic model for action selection."""
        self.__model = value

    @property
    def memory(self):
        """Get the current ExperienceBuffer."""
        return self.__memory

    @memory.setter
    def memory(self, value):
        """Set the ExperienceBuffer for storing experiences."""
        self.__memory = value

    @property
    def steps(self) -> int:
        """Get the current step count for the episode."""
        return self.__steps

    @steps.setter
    def steps(self, value: int):
        """Set the step count for the episode."""
        self.__steps = value

    @property
    def max_steps(self) -> int:
        """Get the maximum steps allowed per episode."""
        return self.__max_steps

    @max_steps.setter
    def max_steps(self, value: int):
        """Set the maximum steps allowed per episode."""
        self.__max_steps = value

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Called by poke-env when a move is needed.
        If a model and memory are set, uses the model to select actions
        and stores experiences in memory.
        """
        # Check if we have a model set (from worker)
        if self.model is None:
            return DefaultBattleOrder()

        # Check step limit
        if self.steps >= self.max_steps > 0:
            return DefaultBattleOrder()

        # Get current observation
        obs = self.embed_battle(battle)  # type: ignore
        state = obs['observation']
        action_mask = obs['action_mask']

        # Import torch here to avoid circular dependency
        import torch

        # Get action from model
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_mask_tensor = torch.tensor(action_mask, dtype=torch.int8).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.model.get_action_and_value(state_tensor, action_mask_tensor)

        action_int = action.item()

        # Store experience in memory
        if self.memory is not None:
            self.memory.store(
                state=state,
                action=action_int,
                reward=0.0,  # Will be updated at episode end
                value=value.item(),
                log_prob=log_prob.item(),
                done=False,
                action_mask=action_mask
            )

        # Increment step counter
        self.steps += 1

        # Convert action to battle order
        return self.action_to_move(action_int, battle)  # type: ignore

    def embed_battle(self, battle: DoubleBattle) -> Dict[str, np.ndarray]:
        """
        Convert battle state to observation dict with features and action mask.

        Args:
            battle: Current battle state

        Returns:
            Dict with 'observation' and 'action_mask' keys
        """
        # Get embedding features
        feature_dict = self.embedder.embed(battle)
        observation = self.embedder.feature_dict_to_vector(feature_dict)

        # Get action mask for valid MDBO actions
        action_mask = self._get_action_mask(battle)

        return {
            'observation': np.array(observation, dtype=np.float32),
            'action_mask': action_mask
        }

    def _get_action_mask(self, battle: DoubleBattle) -> np.ndarray:
        """
        Generate binary mask indicating which MDBO actions are valid.

        Args:
            battle: Current battle state

        Returns:
            Binary array of shape (2025,) where 1 = valid, 0 = invalid
        """
        mask = np.zeros(MDBO.action_space(), dtype=np.int8)

        if battle.teampreview:
            # During teampreview, only teampreview actions are valid
            mask[:MDBO.teampreview_space()] = 1
        else:
            # During regular turns, check each MDBO action
            for i in range(MDBO.action_space()):
                try:
                    # Try to convert to DoubleBattleOrder and validate
                    mdbo = MDBO.from_int(i, type=MDBO.TURN)
                    order = mdbo.to_double_battle_order(battle)
                    # Only check DoubleBattleOrder instances, skip DefaultBattleOrder
                    if isinstance(order, DoubleBattleOrder) and is_valid_order(order, battle):
                        mask[i] = 1
                except Exception:
                    # Invalid action
                    pass

        return mask

    def calc_reward(self, battle: DoubleBattle) -> float:
        """
        Calculate reward for the current battle state.

        Rewards are sparse and only given at the end of the battle:
        - +1 for winning
        - -1 for losing
        - 0 otherwise (battle not finished)

        Args:
            battle: Current battle state

        Returns:
            Reward value
        """
        if not battle.finished:
            return 0.0
        return 1.0 if battle.won else -1.0

    def action_to_move(self, action: int, battle: DoubleBattle) -> BattleOrder:
        """
        Convert MDBO integer action to BattleOrder.

        Args:
            action: Integer action from 0 to 2024
            battle: Current battle state

        Returns:
            BattleOrder to send to Showdown
        """
        if battle.teampreview:
            mdbo = MDBO.from_int(action, type=MDBO.TEAMPREVIEW)
        else:
            mdbo = MDBO.from_int(action, type=MDBO.TURN)

        return mdbo.to_double_battle_order(battle)

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> "tuple[Dict[str, np.ndarray], Dict[str, Any]]":
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset internal state
        self._reward = 0.0
        self._done = False

        # Start a new battle (this will be handled by poke-env's event loop)
        # For now, return empty observation - actual battle will be set by poke-env
        observation: Dict[str, np.ndarray] = {
            'observation': np.zeros(self.embedder.embedding_size, dtype=np.float32),
            'action_mask': np.zeros(MDBO.action_space(), dtype=np.int8)
        }

        info: Dict[str, Any] = {}

        return observation, info  # type: ignore[return-value]

    async def step(
        self,
        action: int
    ) -> "tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]":
        """
        Execute one timestep in the environment.

        Args:
            action: MDBO integer action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # This needs to be called asynchronously within the poke-env battle loop
        # For now, this is a placeholder structure
        # The actual implementation will be in the worker

        observation: Dict[str, np.ndarray] = {
            'observation': np.zeros(self.embedder.embedding_size, dtype=np.float32),
            'action_mask': np.zeros(MDBO.action_space(), dtype=np.int8)
        }
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        return observation, reward, terminated, truncated, info  # type: ignore[return-value]
