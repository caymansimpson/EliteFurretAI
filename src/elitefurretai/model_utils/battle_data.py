# -*- coding: utf-8 -*-
"""This class stores the data of a battle, from a json file
"""

from dataclasses import dataclass
from typing import Dict, List, Union

from poke_env.environment.observation import Observation
from poke_env.environment.observed_pokemon import ObservedPokemon
from poke_env.environment.pokemon import Pokemon


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

    p1_teampreview_team: List[ObservedPokemon]
    p2_teampreview_team: List[ObservedPokemon]

    score: List[int]
    winner: str
    end_type: str

    observations: Dict[int, Observation]

    @staticmethod
    def observed_pokemon_to_pokemon(omon: ObservedPokemon) -> Pokemon:
        mon = Pokemon(gen=6, species=omon.species)
        mon._item = omon.item
        mon._level = omon.level
        mon._moves = omon.moves
        mon._ability = omon.ability
        stats = omon.stats  # Broken right now; we don't stoer hp
        if not mon._last_request:
            mon._last_request = {}
        mon._last_request["stats"] = stats
        mon._gender = omon.gender
        mon._shiny = omon.shiny
        return mon
