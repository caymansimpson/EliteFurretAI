# -*- coding: utf-8 -*-
"""This module isn't yet implemented. It will host R-NaD-related code
that will take a policy and play according to that policy
"""

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.env_player import EnvPlayer


class ModelFreeActor(EnvPlayer):

    def __init__(self, policy, battle_filepath):
        raise NotImplementedError

    # Track opponent
    def add_opponent(self, opponent: EnvPlayer):
        raise NotImplementedError

    def load_policy(self, policy):
        raise NotImplementedError

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        raise NotImplementedError

    def teampreview(self, battle: AbstractBattle) -> str:
        raise NotImplementedError

    # Save it to the battle_filepath using DataProcessor, using opponent information
    # to create omniscient BattleData object
    def _battle_finished_callback(self, battle: AbstractBattle):
        raise NotImplementedError
