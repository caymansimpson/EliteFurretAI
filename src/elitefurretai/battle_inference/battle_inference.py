# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""

import math
from typing import Any, Dict, List, Union

from poke_env.environment import (
    Battle,
    DoubleBattle,
    MoveCategory,
    Observation,
    Pokemon,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.battle_inference.inference_utils import (
    get_segments,
    get_showdown_identifier,
    standardize_pokemon_ident,
)
from elitefurretai.battle_inference.speed_inference import SpeedInference

_FLAGS = {
    "has_status_move": False,
    "can_be_choice": True,
    "last_move": None,
    "item": None,
    "num_moves_since_switch": 0,
    "num_moved": 0,
    "1.2atk": 0,
    "1.5atk": 0,
    "1.2spa": 0,
    "1.5spa": 0,
    "1.5def": 0,
    "1.5spd": 0,
    "1.5spe": 0,
    "screen_turn": None,
}


# This class is meant to be initiated by the Player in teampreview/choose_move (when the battle is initiated)
# and called w/ battle_inference.update(battle) each turn
class BattleInference:

    def __init__(self, battle: Union[Battle, DoubleBattle], verbose=0):
        self._opponent_mons: Dict[str, Any] = {}
        self._battle = battle
        self._abort = False

        # Track which turns we've observed so we don't skip or double up
        self._observed_turns = set()

        for mon in battle.teampreview_opponent_team:
            key = get_showdown_identifier(mon, battle.opponent_role)
            self._opponent_mons[key] = self._load_opponent_set(mon)
            if mon.name == "Zoroark":
                self._abort = True

        self._speed_inference = SpeedInference(battle, self._opponent_mons, verbose)

    def _load_opponent_set(self, mon: Pokemon) -> Dict[str, Any]:
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
                "Battle must be initialized before inference; we have no opponent role"
            )

        if self._abort:
            return

        for i, observation in self._battle.observations.items():
            if i not in self._observed_turns:
                self.check_items(observation)
                self._speed_inference.check_speed(observation)
                # TODO implement: self.check_damage(observation)
                self._observed_turns.add(i)

        self._last_observed_turn = self._battle.turn

    def check_items(self, observation: Observation):
        """
        Updates the inference with the latest observation

        :param observation: The latest observation
        :type observation: Observation
        """

        # Check for status moves; eliminates assault vest
        for ident in self._opponent_mons:
            if ident in self._battle.opponent_team:
                for move in self._battle.opponent_team[ident].moves.values():
                    if move.category == MoveCategory.STATUS:
                        self._opponent_mons[ident]["has_status_move"] = True

                if self._battle.opponent_team[ident].item is not None:
                    self._opponent_mons[ident]["item"] = self._battle.opponent_team[
                        ident
                    ].item
                    self._opponent_mons[ident]["can_be_choice"] = False

        # Reset mons flags on switch-in
        segments = get_segments(observation.events)
        events = (
            segments.get("switch", [])
            + segments.get("preturn_switch", [])
            + segments.get("move", [])
        )
        for i, event in enumerate(events):
            if event[1] == "switch":
                ident = standardize_pokemon_ident(event[2])
                if self._battle.opponent_role and ident.startswith(
                    self._battle.opponent_role
                ):
                    self._opponent_mons[ident]["num_moves_since_switch"] = 0
                    self._opponent_mons[ident]["last_move"] = None

        # Add move counts and check if moves changed, setting choice=False if so
        events = segments.get("move", [])
        for event in events:
            if event[1] == "move":
                ident = standardize_pokemon_ident(event[2])
                if self._battle.opponent_role and ident.startswith(
                    self._battle.opponent_role
                ):
                    self._opponent_mons[ident]["num_moved"] += 1
                    self._opponent_mons[ident]["num_moves_since_switch"] += 1

                    if event[3] != self._opponent_mons[ident]["last_move"]:
                        self._opponent_mons[ident]["can_be_choice"] = False
                        self._opponent_mons[ident]["last_move"] = event[3]

        # TODO: Check for lack of residual messages for safetygoggles, covert cloak, heavy duty boots, light clay

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
                "Battle must be initialized before inference; we have no opponent role"
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}"
            )

        if stat not in self._opponent_mons[mon_ident]:
            raise KeyError(
                f"{stat} is not a stat that we store. Keys: {self._opponent_mons[mon_ident].keys()}"
            )

        return self._opponent_mons[mon_ident][stat]
