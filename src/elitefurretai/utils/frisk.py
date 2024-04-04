# -*- coding: utf-8 -*-
"""This module returns probabilities of infostates, from information learned in battle logs
"""

# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union

from poke_env.environment import DoubleBattle, Observation, ObservedPokemon


class Frisk:

    def __init__(self, fp: str, method: str = "database"):

        # We either load stats from a database, or we load a series of networks
        if method == "database":
            pass
        elif method == "network":
            pass
        else:
            raise ValueError("Invalid method")

    def get_speed_range(self, mon: ObservedPokemon, obs: Observation) -> List[int]:
        raise NotImplementedError

    def get_defensive_stat_range(
        self, move_log: str, damage_log: str, obs: Observation
    ) -> Dict[str, int]:
        raise NotImplementedError

    def get_offensive_stat_range(
        self, move_log: str, damage_log: str, obs: Observation
    ) -> Dict[str, int]:
        raise NotImplementedError

    # We should first identify what we don't know, and then using what we do know, predict
    # the information we don't
    # TODO: overload this
    def predict_infostate(
        self, battle: DoubleBattle, probabilities=False
    ) -> Union[DoubleBattle, List[Tuple[DoubleBattle, float]]]:
        raise NotImplementedError

    # Given a teampreview, predicts the spreads of the team.
    # TODO: overload this
    def predict_vgc_team(
        self, battle: DoubleBattle, probabilities=False
    ) -> Union[List[ObservedPokemon], List[Tuple[ObservedPokemon, float]]]:
        raise NotImplementedError
