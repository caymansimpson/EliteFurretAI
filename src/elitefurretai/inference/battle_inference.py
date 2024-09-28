# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""

import math
from typing import Any, Dict, List, Optional, Union

from poke_env.data.gen_data import GenData
from poke_env.environment import Battle, DoubleBattle, Pokemon
from poke_env.stats import compute_raw_stats

from elitefurretai.inference.inference_utils import get_showdown_identifier
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference

_FLAGS = {
    "has_status_move": False,
    "can_be_choice": True,
    "last_move": None,
    "item": GenData.UNKNOWN_ITEM,
    "num_moves_since_switch": 0,
    "num_moved": 0,
    "screens": [],
    # Currently not checked, since we don't have a reverse damage calculator
    "1.2atk": 0,
    "1.5atk": 0,
    "1.2spa": 0,
    "1.5spa": 0,
    "1.5def": 0,
    "1.5spd": 0,
}


# This class is meant to be initiated by the Player in teampreview/choose_move (when the battle is initiated)
# and called w/ battle_inference.update(battle) each turn.
class BattleInference:
    __slots__ = (
        "_abort",
        "_battle",
        "_item_inference",
        "_last_observed_turn",
        "_observed_turns",
        "_opponent_mons",
        "_speed_inference",
    )

    def __init__(self, battle: Union[Battle, DoubleBattle], verbose=0):
        self._opponent_mons: Dict[str, Any] = {}
        self._battle = battle
        self._abort = False

        # Track which turns we've observed so we don't skip or double up
        self._observed_turns = set()

        for mon in battle.teampreview_opponent_team:
            key = get_showdown_identifier(mon, battle.opponent_role)
            self._opponent_mons[key] = self.load_opponent_set(mon)
            if mon.name == "Zoroark":
                self._abort = True

        self._speed_inference = SpeedInference(battle, self._opponent_mons, verbose)
        self._item_inference = ItemInference(battle, self._opponent_mons, verbose)

    @staticmethod
    def load_opponent_set(mon: Pokemon) -> Dict[str, Any]:
        opponent_info = {}
        for stat, minval, maxval in zip(
            ["hp", "atk", "def", "spa", "spd", "spe"],
            compute_raw_stats(
                mon.species, [0] * 6, [0] * 6, mon.level, "serious", mon._data
            ),
            compute_raw_stats(
                mon.species, [252] * 6, [31] * 6, mon.level, "serious", mon._data
            ),
        ):
            opponent_info[stat] = [math.floor(minval * 0.9), math.floor(maxval * 1.1)]

        opponent_info.update(_FLAGS)
        opponent_info["screens"] = []
        return opponent_info

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
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if self._abort:
            return

        for i, observation in self._battle.observations.items():
            if i not in self._observed_turns:
                self._speed_inference.check_speed(observation)
                self._item_inference.check_items(observation)
                # TODO implement: self.check_damage(observation)
                self._observed_turns.add(i)

                self._last_observed_turn = i

    def get_flags(self, mon_ident: str) -> Dict[str, Any]:
        """
        :return: The all the flags inferred from our battle observations
        :rtype: Dict[str, Any]
        """

        if self._abort:
            return {}

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}",
                self._battle,
            )

        return self._opponent_mons[mon_ident]

    def get_item(self, mon_ident: str) -> Optional[str]:
        """
        :return: The item that the pokemon is holding, derived from our observations
            from the battle.
        :rtype: str
        """

        if self._abort:
            return GenData.UNKNOWN_ITEM

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}",
                self._battle,
            )

        if (
            self._battle.opponent_team[mon_ident].item is not None
            and self._battle.opponent_team[mon_ident].item != GenData.UNKNOWN_ITEM
        ):
            return self._battle.opponent_team[mon_ident].item
        else:
            return self._opponent_mons[mon_ident]["item"]

    def get_stat_range(self, mon_ident: str, stat: str) -> List[int]:
        """
        :return: Min and Max in a list of the pokemon's stat, derived from our observations
            from the battle.
        :rtype: List[int]
        """

        if self._abort:
            return []

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}",
                self._battle,
            )

        if stat not in self._opponent_mons[mon_ident]:
            raise KeyError(
                f"{stat} is not a stat that we store. Keys: {self._opponent_mons[mon_ident].keys()}",
                self._battle,
            )

        return self._opponent_mons[mon_ident][stat]

    def can_have_choice_item(self, mon_ident: str) -> bool:
        """
        :return: Whether the mon has demonstrated it can move multiple moves
            and thus can't be a choice item.
        :rtype: bool
        """

        if self._abort:
            raise ValueError("BattleInference was not properly initalized", self._battle)

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}",
                self._battle,
            )

        return self._opponent_mons[mon_ident]["can_be_choice"]

    def can_have_assault_vest(self, mon_ident: str) -> bool:
        """
        :return: Whether the mon has demonstrated it has a status move
            and thus can't have an assault best
        :rtype: bool
        """

        if self._abort:
            raise ValueError("BattleInference was not properly initalized", self._battle)

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role",
                self._battle,
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}",
                self._battle,
            )

        return not self._opponent_mons[mon_ident]["has_status_move"]
