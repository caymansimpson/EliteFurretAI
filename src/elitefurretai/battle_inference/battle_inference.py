# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""


import math
from typing import Any, Dict, List, Union

from poke_env.environment import Battle, DoubleBattle, Pokemon
from poke_env.stats import compute_raw_stats

from elitefurretai.battle_inference.inference_utils import _DISCERNABLE_ITEMS, _flags
from elitefurretai.battle_inference.speed_inference import SpeedInference


# TODO: need to give up all calculations until I identify a zoroark
class BattleInference:

    def __init__(self, battle: Union[Battle, DoubleBattle]):
        self._opponent_mons: Dict[str, Any] = {}
        self._battle = battle

        # Track which turns we've observed so we don't skip or double up
        self._observed_turns = set()

        for mon in battle.teampreview_opponent_team:
            key = battle.opponent_role + ": " + mon._data.pokedex[mon.species]["name"]
            self._opponent_mons[key] = self._load_opponent_set(mon)

        self._speed_inference = SpeedInference(battle, self._opponent_mons)

    def _load_opponent_set(self, mon: Pokemon) -> Dict[str, Any]:
        stats = {}
        for stat, minval, maxval in zip(
            ["hp", "atk", "def", "spa", "spd", "spe"],
            compute_raw_stats(
                mon.species, [0] * 6, [0] * 6, mon.level, "serious", mon._data
            ),
            compute_raw_stats(
                mon.species, [252] * 6, [31] * 6, mon.level, "serious", mon._data
            ),
        ):
            stats[stat] = [math.floor(minval * 0.9), math.floor(maxval * 1.1)]

        return {
            "items": list(_DISCERNABLE_ITEMS),
            "abilities": mon.possible_abilities,
            "moves": [],
            "hp": stats["hp"],
            "atk": stats["atk"],
            "def": stats["def"],
            "spa": stats["spa"],
            "spd": stats["spd"],
            "spe": stats["spe"],
            "flags": {k: v for k, v in _flags.items()},
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

        for i, observation in self._battle.observations.items():
            if i not in self._observed_turns:
                self._speed_inference.check_speed(observation)
                # TODO: implement
                # self.check_damage(observation)
                # self.check_items(observation); should also update flags
                self._observed_turns.add(i)

        self._last_observed_turn = self._battle.turn

    # TODO: implement properly
    def get_stat_range(self, mon_ident: str, stat: str) -> List[int]:
        """
        :return: Min and Max in a list of the pokemon's stat
        :rtype: List[int]
        """
        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        # TODO: handle case where is either species or full identifier
        if mon_ident not in self._mons:
            raise KeyError(f"Can't get {mon_ident} in self._mons")

        if stat not in self._mons[mon_ident]:
            raise KeyError(f"{stat} is not a stat that we store. Please check the key")

        return self._mons[mon_ident][stat]

    # TODO: implement properly; handle case where is either species or full identifier
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
