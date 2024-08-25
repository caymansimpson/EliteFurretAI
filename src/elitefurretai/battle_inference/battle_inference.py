# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""

import math
from typing import Any, Dict, List, Optional, Union

from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import (
    Battle,
    DoubleBattle,
    Move,
    MoveCategory,
    Observation,
    Pokemon,
    SideCondition,
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
    "item": GenData.UNKNOWN_ITEM,
    "num_moves_since_switch": 0,
    "num_moved": 0,
    "1.2atk": 0,
    "1.5atk": 0,
    "1.2spa": 0,
    "1.5spa": 0,
    "1.5def": 0,
    "1.5spd": 0,
    "1.5spe": 0,
    "screens": [],
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

    # TODO: Check for lack of residual messages for safetygoggles and heavy duty boots
    def check_items(self, observation: Observation):
        """
        Updates the inference with the latest observation

        :param observation: The latest observation
        :type observation: Observation
        """

        # Nothing to do if battle isnt initiated
        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        # Check for status moves; eliminates assault vest
        for ident in self._opponent_mons:
            if ident in self._battle.opponent_team:
                for move in self._battle.opponent_team[ident].moves.values():
                    if move.category == MoveCategory.STATUS:
                        self._opponent_mons[ident]["has_status_move"] = True

                if self._opp_has_item(ident):
                    item = self._battle.opponent_team[ident].item
                    self._opponent_mons[ident]["item"] = item
                    if item and item.startswith("choice"):
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
                if ident.startswith(self._battle.opponent_role):
                    self._opponent_mons[ident]["num_moves_since_switch"] = 0
                    self._opponent_mons[ident]["last_move"] = None

        # Remove screens that have been removed from the battle
        for ident in self._opponent_mons:
            for sc in self._opponent_mons[ident]["screens"]:
                if sc not in self._battle.opponent_side_conditions:
                    self._opponent_mons[ident]["screens"].remove(sc)

        # Add move counts and check if moves changed, setting choice=False if so
        # and add screens
        events = segments.get("move", [])
        for i, event in enumerate(events):
            if event[1] == "move":
                ident = standardize_pokemon_ident(event[2])
                if ident.startswith(self._battle.opponent_role):
                    # Move tracking for AI
                    self._opponent_mons[ident]["num_moved"] += 1
                    self._opponent_mons[ident]["num_moves_since_switch"] += 1

                    # Check choice
                    if self._opponent_mons[ident]["last_move"] not in [event[3], None]:
                        self._opponent_mons[ident]["can_be_choice"] = False
                        self._opponent_mons[ident]["last_move"] = event[3]

                    # Update last move
                    self._opponent_mons[ident]["last_move"] = event[3]

                    # Record who used a screen move to track for Light Clay
                    if event[3] in {"Reflect", "Light Screen", "Aurora Veil"} and not (
                        len(events) > i + 1 and events[i + 1][1] == "-fail"
                    ):
                        self._opponent_mons[ident]["screens"].append(
                            SideCondition.from_showdown_message(event[3])
                        )

                # Check for covert cloak
                move = Move(to_id_str(event[3]), self._battle.gen)
                target = (
                    standardize_pokemon_ident(event[4]) if event[4] != "[still]" else ""
                )
                if (
                    len(move.secondary) > 0
                    and move.secondary[0].get("chance", 0) == 100
                    and target in self._opponent_mons
                ):

                    # Get what we should be looking for
                    key = ""
                    if "boosts" in move.secondary[0]:
                        key = "boosts"
                    elif "status" in move.secondary[0]:
                        key = "status"
                    elif "volatileStatus" in move.secondary[0]:
                        key = move.secondary[0]["volatileStatus"]

                    # Find the index of the next move to know where to stop looking for
                    # boosts, status and some volatileStatuses
                    end = i + 1
                    while end < len(events) and events[end][1] != "move":
                        end += 1

                    # Not implementing healblock from psychicnoise or syrupbomb or dynamicpunch
                    # |move|p2b: Smeargle|Icy Wind|p1b: Incineroar|[spread] p1a,p1b
                    # |-supereffective|p1a: Koraidon
                    # |-resisted|p1b: Incineroar
                    # |-damage|p1a: Koraidon|97/100
                    # |-damage|p1b: Incineroar|99/100
                    # |-unboost|p1a: Koraidon|spe|1
                    # |-unboost|p1b: Incineroar|spe|1
                    if key == "boosts":

                        # We go through every opponent mon and check if they got hit, and if they did, we
                        # check if they got the secondary effect
                        idents = []
                        for j in range(i, end):
                            if events[j][1] == "-damage" and events[j][2].startswith(
                                self._battle.opponent_role
                            ):
                                idents.append(standardize_pokemon_ident(events[j][2]))
                            elif events[j][1] in ["-boost", "-unboost"] and events[j][
                                2
                            ].startswith(self._battle.opponent_role):
                                idents.remove(standardize_pokemon_ident(events[j][2]))

                        # This means that we get an ident that took damage, but didn't get a (un)boost
                        if len(idents) > 0:
                            if not self._opp_has_item(idents[0]):
                                self._opponent_mons[idents[0]]["can_be_choice"] = False
                                self._opponent_mons[idents[0]]["item"] = "covertcloak"
                                self._battle.opponent_team[idents[0]].item = "covertcloak"
                            else:
                                raise ValueError(
                                    f"We found Covert Cloak but {idents[0]} has {self._battle.opponent_team[idents[0]].item}"
                                )

                    # Check to see if someone has been damaged and if there is no status. If there is no status
                    # this means that a mon has covert cloak. Note that this will fail if there is a move that
                    # affects multiple targets and has 100% chance of incurring a status as a secondary
                    # |-status|p1b: Incineroar|par|[from] move: Nuzzle
                    elif (
                        key == "status"
                        and any(events[j][1] == "-damage" for j in range(i, end))
                        and not any(events[j][1] == "-status" for j in range(i, end))
                    ):
                        if not self._opp_has_item(target):
                            self._opponent_mons[target]["can_be_choice"] = False
                            self._opponent_mons[target]["item"] = "covertcloak"
                            self._battle.opponent_team[target].item = "covertcloak"
                        else:
                            raise ValueError(
                                f"We found Covert Cloak but {target} has {self._battle.opponent_team[ident].item}"
                            )

                    # |-start|p1a: Amoonguss|Salt Cure
                    elif key == "saltcure" and not any(
                        events[j][1] == "-start" and events[j][-1] == "Salt Cure"
                        for j in range(i, end)
                    ):
                        if not self._opp_has_item(target):
                            self._opponent_mons[target]["can_be_choice"] = False
                            self._opponent_mons[target]["item"] = "covertcloak"
                            self._battle.opponent_team[target].item = "covertcloak"
                        else:
                            raise ValueError(
                                f"We found Covert Cloak but {target} has {self._battle.opponent_team[ident].item}"
                            )

                    # |move|p2b: Smeargle|Fake Out|p1b: Urshifu
                    # |-damage|p1b: Urshifu|97/100
                    # |cant|p1b: Urshifu|flinch
                    elif key == "flinch":
                        # Flinch will only happen if we damage the pokemon
                        damaged = False
                        for j in range(i, end):
                            if (
                                events[j][1] == "-damage"
                                and standardize_pokemon_ident(events[j][2]) == target
                            ):
                                damaged = True
                                break

                        if damaged:
                            for j in range(end, len(events)):
                                if (
                                    events[j][1] == "move"
                                    and standardize_pokemon_ident(events[j][2]) == target
                                ):
                                    if not self._opp_has_item(target):
                                        self._opponent_mons[target][
                                            "can_be_choice"
                                        ] = False
                                        self._opponent_mons[target]["item"] = "covertcloak"
                                        self._battle.opponent_team[target].item = (
                                            "covertcloak"
                                        )
                                    else:
                                        raise ValueError(
                                            f"We found Covert Cloak but {target} has {self._battle.opponent_team[ident].item}"
                                        )
                                    break

        # Check battle for screen turns. If there's a screen that lasts longer than
        # 5, we should figure out who used it and then set them as lightclay
        for sc, turn_started in self._battle.opponent_side_conditions.items():
            # If the screen lasts longer than 5 turns, we find the pokemon that
            # set the screen and give them light clay.
            if self._battle.turn - turn_started > 5:
                for ident in self._opponent_mons:
                    if sc in self._opponent_mons[ident]["screens"]:
                        if not self._opp_has_item(ident):
                            self._opponent_mons[ident]["can_be_choice"] = False
                            self._opponent_mons[ident]["item"] = "lightclay"
                            self._battle.opponent_team[ident].item = "lightclay"
                        else:
                            raise ValueError(
                                f"We found Light Clay but {ident} has {self._battle.opponent_team[ident].item}"
                            )

        # Check for Safety Goggles; need to check sandstorm and rage powder
        # events = segments.get("residual", [])
        # for event in events:
        #     residual, identifier = get_residual_and_identifier(event)
        #     if residual == "Sandstorm":

    def _opp_has_item(self, ident):
        return (
            self._battle.opponent_team[ident].item is not None
            and self._battle.opponent_team[ident].item != GenData.UNKNOWN_ITEM
        )

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
                "Battle must be initialized before inference; we have no opponent role"
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}"
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

    def can_have_choice_item(self, mon_ident: str) -> bool:
        """
        :return: Whether the mon has demonstrated it can move multiple moves
            and thus can't be a choice item.
        :rtype: bool
        """

        if self._abort:
            raise ValueError("BattleInference was not properly initalized")

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}"
            )

        return self._opponent_mons[mon_ident]["can_be_choice"]

    def can_have_assault_vest(self, mon_ident: str) -> bool:
        """
        :return: Whether the mon has demonstrated it has a status move
            and thus can't have an assault best
        :rtype: bool
        """

        if self._abort:
            raise ValueError("BattleInference was not properly initalized")

        if self._battle.opponent_role is None:
            raise ValueError(
                "Battle must be initialized before inference; we have no opponent role"
            )

        if mon_ident not in self._opponent_mons:
            raise KeyError(
                f"Can't find {mon_ident} in self._mons. Keys: {list(self._opponent_mons.keys())}"
            )

        return not self._opponent_mons[mon_ident]["has_status_move"]
