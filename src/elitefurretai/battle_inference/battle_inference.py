# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""


import sys
from typing import Any, Dict, List, Union

from poke_env.environment import Battle, DoubleBattle, Pokemon

from elitefurretai.battle_inference.inference_utils import _DISCERNABLE_ITEMS, _flags
from elitefurretai.battle_inference.speed_inference import SpeedInference


# TODO: extend class to take into account aggressiveness of assumptions
class BattleInference:

    def __init__(self, battle: Union[Battle, DoubleBattle]):
        self._mons: Dict[str, Any] = {}
        self._battle = battle
        self._observed_turns = (
            set()
        )  # Track which turns we've observed so we don't skip or double up

        self._speed_inference = SpeedInference(battle, self._mons)

        for ident in battle.opponent_team:
            self._mons[ident] = self._load_opponent_set(ident)
        for ident in battle.team:
            self._mons[ident] = self._load_pokemon_set(battle.team[ident])

    # TODO: take in metaDB and load in the pokemon's moves, items, abilities, and stats
    def _load_opponent_set(self, pokemon_ident: str) -> Dict[str, Any]:
        return {
            "items": list(_DISCERNABLE_ITEMS),
            "abilities": [],
            "moves": [],
            "hp": [0, sys.maxsize],
            "atk": [0, sys.maxsize],
            "def": [0, sys.maxsize],
            "spa": [0, sys.maxsize],
            "spd": [0, sys.maxsize],
            "spe": [0, sys.maxsize],
            "flags": {k: v for k, v in _flags.items()},
            "opponent": True,
        }

    def _load_pokemon_set(self, mon: Pokemon) -> Dict[str, Any]:
        return {
            "items": mon.item,
            "abilities": mon.ability,
            "moves": list(mon.moves.values()),
            "hp": mon.max_hp,
            "atk": mon.stats["atk"] if mon.stats else [0, sys.maxsize],
            "def": mon.stats["def"] if mon.stats else [0, sys.maxsize],
            "spa": mon.stats["spa"] if mon.stats else [0, sys.maxsize],
            "spd": mon.stats["spd"] if mon.stats else [0, sys.maxsize],
            "spe": mon.stats["def"] if mon.stats else [0, sys.maxsize],
            "flags": {k: v for k, v in _flags.items()},
            "opponent": False,
        }

    # We should call update before we decide what we should do each turn. The latest Observation will
    # be of the last turn, with events leading up to now
    def update(self):
        """
        Updates the inference with the latest observation

        :param observation: The latest observation
        :type observation: Observation
        """
        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        # If there's a mon I haven't seen, load it
        for species in self._battle.opponent_team:
            ident = str(self._battle.opponent_role) + species
            if ident not in self._mons:
                self._mons[ident] = self._load_opponent_set(species)

        for i, observation in self._battle.observations.items():
            if i not in self._observed_turns:
                self._speed_inference.check_speed(observation)
                # TODO: implement
                # self.check_damage(observation)
                # self.check_items(observation); should also update flags
                self._observed_turns.add(i)

        self._last_observed_turn = self._battle.turn

    def get_stat_range(self, species: str, stat: str) -> List[int]:
        """
        :return: Min and Max in a list of the pokemon's stat
        :rtype: List[int]
        """
        if self._battle.turn not in self._observed_turns:
            self.update()

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        if self._battle.opponent_role + ": " + species not in self._mons:
            key = self._battle.opponent_role + ": " + species
            raise KeyError(f"Can't get {key} in self._mons")

        if stat not in self._mons[self._battle.opponent_role + ": " + species]:
            raise KeyError(f"{stat} is not a stat that we store. Please check the key")

        return self._mons[self._battle.opponent_role + ": " + species][stat]

    def get_possible_items(self, species: str) -> List[str]:
        """
        :return: List of items the pokemon can have
        :rtype: List[str]
        """
        if self._battle.turn not in self._observed_turns:
            self.update()

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        return list(self._mons[self._battle.opponent_role + ": " + species]["items"])

    def get_possible_unknown_moves(self, species: str) -> List[str]:
        """
        :return: List of moves the pokemon can have
        :rtype: List[str]
        """
        if self._battle.turn not in self._observed_turns:
            self.update()

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        return list(self._mons[self._battle.opponent_role + ": " + species]["items"])
