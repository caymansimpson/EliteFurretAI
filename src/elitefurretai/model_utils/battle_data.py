# -*- coding: utf-8 -*-
"""This class stores the data of a battle, from a json file
"""

from dataclasses import dataclass
from typing import Dict, List, Union
import logging

from elitefurretai.inference.inference_utils import get_showdown_identifier

from poke_env.environment.observation import Observation
from poke_env.environment.observed_pokemon import ObservedPokemon
from poke_env.environment.pokemon import Pokemon
from poke_env.environment import (
    Battle, DoubleBattle, AbstractBattle
)


@dataclass
class BattleData:
    SOURCE_EFAI = "elitefurretai"
    SOURCE_SHOWDOWN_ANON = "showdown_anon"
    SOURCE_SHOWDOWN = "showdown"

    roomid: str
    format: str

    p1: str
    p2: str

    p1rating: Union[int, Dict[str, int]]
    p2rating: Union[int, Dict[str, int]]

    # Not always four when unobservable information
    p1_team: List[ObservedPokemon]
    p2_team: List[ObservedPokemon]

    p1_teampreview_team: List[ObservedPokemon]
    p2_teampreview_team: List[ObservedPokemon]

    score: List[int]  # Represents non-fainted mons at end of battle for each player
    winner: str
    end_type: str

    # Turn --> Observation for that turn; these are from the perspective of one player
    # and so arent omniscient. But with the information from p1_teampreview_team and p1_team
    # we can reconstruct a battle with omnscience (we need both teampreview and team because
    # gender, if not explicitly assigned, will be generated at random)
    observations: Dict[int, Observation]

    inputs: List[str]

    # Where the log initially came from. Can be any of the class
    # variables. If showdown_anon, we don't get requests. If showdown,
    # it's incomplete information
    source: str

    @staticmethod
    def observed_pokemon_to_pokemon(omon: ObservedPokemon) -> Pokemon:
        mon = Pokemon(gen=6, species=omon.species)
        mon._item = omon.item
        mon._level = omon.level
        mon._moves = omon.moves
        mon._ability = omon.ability
        mon._terastallized_type = omon.tera_type

        mon._stats = {}
        if omon.stats:
            for stat in omon.stats:
                if isinstance(omon.stats[stat], int):
                    mon._stats[stat] = omon.stats[stat]  # pyright: ignore

        mon._gender = omon.gender
        mon._shiny = omon.shiny
        return mon
    
    # TODO: Update to handle singles and different generations
    def to_battle(self, perspective: str) -> Union[Battle, DoubleBattle, AbstractBattle]:
        player = self.p2
        team = self.p2_team
        if perspective == "p1":
            player = self.p1
            team = self.p1_team
       
        battle = DoubleBattle(
            self.roomid, 
            player, 
            logging.getLogger(player), 
            gen=9
        )

        my_team = {}
        for omon in team:
            mon = BattleData.observed_pokemon_to_pokemon(omon)
            my_team[get_showdown_identifier(mon, perspective)] = mon

        battle.team = my_team

        return battle