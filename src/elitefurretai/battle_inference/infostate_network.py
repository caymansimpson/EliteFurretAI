# -*- coding: utf-8 -*-
"""This module returns probabilities of infostates, from a network
"""


# -*- coding: utf-8 -*-
from typing import List, Tuple, Union

from poke_env.environment import DoubleBattle, ObservedPokemon


class InfostateNetwork:

    _MODEL_DIR: str = "data/models"

    def __init__(self, fp: str = "frisk"):
        raise NotImplementedError

    # We should first identify what we don't know, and then using what we do know, predict
    # the information we don't
    def predict_infostate(
        self, battle: DoubleBattle, probabilities: bool = False
    ) -> Union[DoubleBattle, List[Tuple[DoubleBattle, float]]]:
        raise NotImplementedError

    # Given a teampreview, predicts the spreads of the team.
    def predict_vgc_team(
        self, battle: DoubleBattle, probabilities=False
    ) -> Union[List[ObservedPokemon], List[Tuple[ObservedPokemon, float]]]:
        raise NotImplementedError
