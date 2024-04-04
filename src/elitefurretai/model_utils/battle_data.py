# -*- coding: utf-8 -*-
"""This class stores the data of a battle, from a json file
"""

from dataclasses import dataclass
from typing import Dict, List, Union

from poke_env.environment.observation import Observation
from poke_env.environment.observed_pokemon import ObservedPokemon


@dataclass
class BattleData:
    roomid: str
    format: str

    p1: str
    p2: str

    p1rating: Union[int, Dict[str, int]]
    p2rating: Union[int, Dict[str, int]]

    p1_team: List[ObservedPokemon]
    p2_team: List[ObservedPokemon]

    score: List[int]
    winner: str
    end_type: str

    observations: Dict[int, Observation]

    def embed_team_preview(self) -> List[int]:
        """
        Returns a list of integers representing the team preview of the battle
        """
        raise NotImplementedError

    def embed_turn(self, turn: int, perspective: str = "p1") -> List[int]:
        """
        Returns a list of integers representing the turn of the battle, from the
        perspective of the player whose turn it is.
        """
        raise NotImplementedError
