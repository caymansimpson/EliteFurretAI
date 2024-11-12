# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""

import sys
from typing import Any, Dict, Optional, Union

from elitefurretai.inference.inference_utils import battle_to_str

from poke_env.data.gen_data import GenData
from poke_env.environment import AbstractBattle, Battle, DoubleBattle, Pokemon
from poke_env.stats import compute_raw_stats

_FLAGS = {
    "has_status_move": False,  # Assault Vest Flag
    "can_be_choice": True,  # For checking Choice
    "can_be_clearamulet": None,  # Set None if we have no evidence; true if we have evidence, but not yet confirmed
    "can_be_covertcloak": None,  # Set None if we have no evidence; true if we have evidence, but not yet confirmed
    "last_move": None,  # For checking Choice
    "item": GenData.UNKNOWN_ITEM,
    "num_moves_since_switch": 0,  # Check Choice probs
    "num_moved": 0,
    "screens": [],  # Check Light Clay
    # To help disambiguate states in item tracking
    "debug_item_found_turn": -1,  # TODO: debugging; need to remove
    # Currently not checked, since we don't have a reverse damage calculator
    "1.2atk": 0,
    "1.5atk": 0,
    "1.2spa": 0,
    "1.5spa": 0,
    "1.5def": 0,
    "1.5spd": 0,
}


# This class is meant to work with other inference classes, and is updated by them. It is in essence
# a data class. It should eventually be combined with damage, speed and item inference classes. But we're keeping them
# seperate for now due to code complexity.
class BattleInference:
    __slots__ = (
        "_battle",
        "_opponent_mons",
    )

    def __init__(self, battle: Union[Battle, DoubleBattle, AbstractBattle]):
        self._battle: Union[Battle, DoubleBattle, AbstractBattle] = battle

        # Showdown identifier to flags
        self._opponent_mons: Dict[str, Any] = {}

    @staticmethod
    def load_opponent_set(mon: Pokemon) -> Dict[str, Any]:
        opponent_info = {}

        # Compute smallest and largest possible value for each stat
        for nature in mon._data.natures:
            for stat, minval, maxval in zip(
                ["hp", "atk", "def", "spa", "spd", "spe"],
                compute_raw_stats(
                    mon.species, [0] * 6, [0] * 6, mon.level, nature, mon._data
                ),
                compute_raw_stats(
                    mon.species, [252] * 6, [31] * 6, mon.level, nature, mon._data
                ),
            ):
                stats = opponent_info.get(stat, [sys.maxsize, 0])
                stats[0] = min(stats[0], minval)
                stats[1] = max(stats[1], maxval)
                opponent_info[stat] = stats

        opponent_info.update(_FLAGS)
        opponent_info["screens"] = []
        return opponent_info

    def is_tracking(self, ident: str) -> bool:
        """
        :return: Whether or not we are tracking the given mon
        :rtype: bool
        """
        return ident in self._opponent_mons

    def get_flags(self, mon_ident: str) -> Dict[str, Any]:
        """
        :return: The all the flags inferred from our battle observations
        :rtype: Dict[str, Any]
        """

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        # If we don't have it yet, we load the default
        if mon_ident not in self._opponent_mons:
            self._opponent_mons[mon_ident] = self.load_opponent_set(
                self._battle.opponent_team[mon_ident]
            )

        return self._opponent_mons[mon_ident]

    def get_flag(self, mon_ident: str, flag: str) -> Optional[Any]:
        """
        :return: The item that the pokemon is holding, derived from our observations
            from the battle. It behaves like defaultdict in that it will generate a new default
            for new idents that are valid
        :rtype: str
        """
        # Check if Battle is valid
        if self._battle.opponent_role is None:
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Battle {self._battle.battle_tag} must be initialized before inference; we have no opponent role"
            )

        # Check if mon_ident is valid
        if mon_ident not in self._battle.opponent_team:
            print(battle_to_str(self._battle))
            raise KeyError(
                f"Can't find {mon_ident} indentifier in self._battle.opponent_teams. Keys: {list(self._battle.opponent_team.keys())}"
            )

        # If we don't have it yet, we load the default
        if mon_ident not in self._opponent_mons:
            self._opponent_mons[mon_ident] = self.load_opponent_set(
                self._battle.opponent_team[mon_ident]
            )

        # If flag is invalid, we create an error
        if flag not in self._opponent_mons[mon_ident]:
            print(battle_to_str(self._battle))
            raise KeyError(
                f"We don't have {flag} in BattleInference. We have {str(list(self._opponent_mons[mon_ident].keys()))}"
            )

        return self._opponent_mons[mon_ident][flag]

    def set_flag(self, mon_ident: str, flag: str, val: Any):
        """
        Sets the flag to the given value. If we haven't seen the mon yet, we create a new entry
        """
        if self._battle.opponent_role is None:
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Battle {self._battle.battle_tag} must be initialized before inference; we have no opponent role"
            )

        if mon_ident not in self._battle.opponent_team:
            print(battle_to_str(self._battle))
            raise KeyError(
                f"Can't find {mon_ident} indentifier in self._battle.opponent_teams. Keys: {list(self._battle.opponent_team.keys())}"
            )

        # If we don't have it yet, we load the default
        if mon_ident not in self._opponent_mons:
            self._opponent_mons[mon_ident] = self.load_opponent_set(
                self._battle.opponent_team[mon_ident]
            )

        # Check to see if the flag is valid
        if flag not in self._opponent_mons[mon_ident]:
            print(battle_to_str(self._battle))
            raise KeyError(
                f"Can't find {flag} in BattleInference. Keys: {list(self._opponent_mons[mon_ident].keys())}"
            )

        # Check to see if typings are corrext
        if (
            isinstance(val, type(self._opponent_mons[mon_ident][flag]))
            or (
                isinstance(val, (int, float))
                and isinstance(self._opponent_mons[mon_ident][flag], (int, float))
            )
            or val is None
            or self._opponent_mons[mon_ident][flag] is None
        ):
            self._opponent_mons[mon_ident][flag] = val
        else:
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Can't set {flag} to {val} in BattleInference. val is {type(val)} and flag is {type(self._opponent_mons[mon_ident][flag])}, which are incompatible"
            )
