import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from bots.random_doubles_player import RandomDoublesPlayer
from poke_env.environment.battle import Battle
from poke_env.environment.field import Field
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.target_type import TargetType
from poke_env.environment.volatile_status import VolatileStatus
from poke_env.environment.weather import Weather
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)
from poke_env.player.env_player import EnvPlayer
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, MaxBoltzmannQPolicy
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# We define our RL player
class SimpleDQNPlayer(EnvPlayer):

    def __init__(self, num_battles=10000, **kwargs):
        super().__init__(**kwargs)
        _ACTION_SPACE = 536760

    # Takes the output of our policy (which chooses from a `_ACTION_SPACE`-dimensional array), and converts it into a battle order
    def _action_to_move(
        self, action: int, battle: DoubleBattle
    ) -> DoubleBattleOrder:  # pyre-ignore
        """Converts actions to move orders.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: DoubleBattleOrder
        """
        raise NotImplementedError

    # TODO: should we return a list?
    @property
    def action_space(self) -> List:
        """
        There are 530 possible moves w/ tera:
        1/ If two switches:
            = 2 possible actions
        2/ if one switch:
            First Mon: 2 possible switches
            Second Mon: 4 moves * 3 targets * 2 (tera)
            = 2 (first mon or second mon could switch) * 2 * 24 = 96
        3/ If no switch:
            If one Tera: 4 moves * 3 targets * 4 moves * 3 targets
            If no Tera: 4 moves * 3 targets * 4 moves * 3 targets
            = 2 * one_Tera + no_tera = 432
        = 2 + 96 + 432 = 530

        HOWEVER, we can consider for simplicity a mon's actions as 2 switches + 4 moves * 3 targets * tera = 26,
        and thus the action space is 26 x 26 = 676. We can prune illegal actions later!
        """
        return self._ACTION_SPACE

    def compute_reward(self, battle: Union[Battle, DoubleBattle]) -> float:
        """A helper function to compute rewards. We only give rewards for winning
        :param battle: The battle for which to compute rewards.
        :type battle: Union[Battle, DoubleBattle]
        :return: the reward
        :rtype: float
        """

        # Victory condition
        if battle.won:
            reward = 1
        elif battle.lost:
            reward = -1
        else:
            reward = 0

        self._reward_buffer[battle] = reward

        return reward

    # We choose a move
    def choose_move(self, battle: Union[Battle, DoubleBattle]) -> str:
        """
        With DQN approach:
        action = model.get(Embedder.embed_double_battle(battle))
        return self._action_to_move(action, battle)
        """
        raise NotImplementedError

    # Same as max damage for now - we return the mons who have the best average type advantages against the other team
    def teampreview(self, battle):

        # We have a dict that has index in battle.team -> average type advantage
        mon_performance = {}

        # For each of our pokemons
        for i, mon in enumerate(battle.team.values()):

            # We store their average performance against the opponent team
            mon_performance[i] = np.mean(
                [compute_type_advantage(mon, opp) for opp in battle.opponent_team.values()]
            )

        # We sort our mons by performance, and choose the top 4
        ordered_mons = sorted(mon_performance, key=lambda k: -mon_performance[k])[:4]

        # We start with the one we consider best overall
        # We use i + 1 as python indexes start from 0 but showdown's indexes start from 1, and return the first 4 mons, in term of importance
        # return "/team " + "".join([str(i + 1) for i in ordered_mons])
        return "/team " + "".join(
            random.sample(list(map(lambda x: str(x + 1), range(0, len(battle.team)))), k=4)
        )


# TODO: train this model and evaluate it
# also add embedder and embeddding past 10 turns using reuniclus
