# -*- coding: utf-8 -*-
"""This module makes items inferences throughout a battle by tracking pokemons' moves.
This object is a companion class to BattleInference.
"""

import inspect
from typing import Any, Dict, List, Union

from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import (
    Battle,
    DoubleBattle,
    Effect,
    Move,
    MoveCategory,
    Observation,
    Pokemon,
    SideCondition,
    Status,
    Target,
    Weather,
)

from elitefurretai.inference.inference_utils import (
    copy_battle,
    get_pokemon,
    get_segments,
    get_showdown_identifier,
    has_flinch_immunity,
    has_rage_powder_immunity,
    has_sandstorm_immunity,
    has_status_immunity,
    is_grounded,
    standardize_pokemon_ident,
    update_battle,
)
from elitefurretai.inference.battle_inference import BattleInference


class ItemInference:
    __slots__ = (
        "_battle",
        "_inferences",
        "_v",
        "_last_event",
    )

    # We need to keep a copy of the battle state because we need to recreate the turn and conditions of the turn to parse speeds
    def __init__(
        self,
        battle: Union[Battle, DoubleBattle],
        inferences: BattleInference,
        verbose: int = 0,
    ):
        self._inferences: BattleInference = inferences
        self._battle = copy_battle(battle)
        self._v = verbose
        self._last_event = 0

    def check_items(self, observation: Observation):
        """
        Updates the inference with the latest observation

        :param observation: The latest observation
        :type observation: Observation
        """

        # Nothing to do if battle isnt initiated
        if self._battle.opponent_role is None or self._battle.player_role is None:
            return

        segments = get_segments(observation.events[self._last_event :])

        if "init" in segments:
            for event in segments["init"]:
                update_battle(self._battle, event)

        # I don't parse the order of activations (right now: Quick Claw and Quick Draw) because of the nicheness
        if "activation" in segments:
            for event in segments["activation"]:
                update_battle(self._battle, event)

        # Reset mons flags on switch-in, check for heavydutyboots
        if "switch" in segments:
            i = 0
            while i < len(segments["switch"]):
                if segments["switch"][i][1] == "switch" and segments["switch"][i][
                    2
                ].startswith(self._battle.opponent_role):

                    end = i + 1
                    while (
                        end < len(segments["switch"])
                        and segments["switch"][end] != "switch"
                    ):
                        end += 1

                    self._check_opponent_switch(segments["switch"][i:end])

                update_battle(self._battle, segments["switch"][i])
                i += 1

        if "battle_mechanic" in segments:
            for event in segments["battle_mechanic"]:
                update_battle(self._battle, event)

        # Reset mons flags on switch-in, check for heavydutyboots
        # Check for covert cloak and safetygoggles
        # TODO: check knockoff and trick or fling or move that consumes items and set flags accordingly
        # Knock off, switcheroo, trick, fling,
        if "move" in segments:
            self._handle_moves(segments)

        if "state_upkeep" in segments:
            for event in segments["state_upkeep"]:
                update_battle(self._battle, event)

        # Check for safetygoggles;
        if "residual" in segments:

            # On normal turns (eg not at the start of the battle where we only have preturn_switch)
            # we should look for residuals if there is sandstorm. If there isn't any residuals, then
            # that says something too! If not, we have to apply the residual events to the battle
            # since _check_residuals_for_safety_goggles does this as well
            if "move" in segments:
                self._check_residuals_for_safety_goggles(segments.get("residual", []))
            else:
                for event in segments["residual"]:
                    update_battle(self._battle, event)

        # Reset mons flags on switch-in, check for heavydutyboots
        if "preturn_switch" in segments:
            i = 0
            while i < len(segments["preturn_switch"]):
                if segments["preturn_switch"][i][1] == "switch" and segments[
                    "preturn_switch"
                ][i][2].startswith(self._battle.opponent_role):
                    self._check_opponent_switch(segments["preturn_switch"][i:])

                update_battle(self._battle, segments["preturn_switch"][i])
                i += 1

        # If we see a turn, that means we're done with the observation and
        # we can reset last_event
        if "turn" in segments:
            for event in segments["turn"]:
                update_battle(self._battle, event)
            self._last_event = 0

        # If there's no turn, this method likely got called in the middle of
        # a turn, in a force_switch situation. So we save where we are so that
        # we can continue parsing from there
        else:
            self._last_event = len(obs.events)

    def _handle_moves(self, segments: Dict[str, List[List[str]]]):
        i = 0
        while i < len(segments["move"]):

            event = segments["move"][i]

            # An opponent is given an tem
            if event[1] == "-item" and event[2].startswith(self._battle.opponent_role):
                ident = standardize_pokemon_ident(event[2])
                item = to_id_str(event[3])
                self._inferences.set_flag(ident, "item", item)

                # Reset flags if Choice is being switched
                if item.startswith("choice"):
                    self._inferences.set_flag(ident, "can_be_choice", True)
                    self._inferences.set_flag(ident, "last_move", None)

            # An opponent loses an item
            elif event[1] == "-enditem" and event[2].startswith(
                self._battle.opponent_role
            ):
                ident = standardize_pokemon_ident(event[2])
                self._inferences.set_flag(ident, "item", None)
                self._inferences.set_flag(ident, "can_be_choice", False)

            # Someone uses a move
            elif event[1] == "move":
                self._update_move_tracking(segments["move"], i)

                # Get Move details
                move = Move(to_id_str(event[3]), self._battle.gen)
                actor = standardize_pokemon_ident(event[2])
                target = None
                if (
                    event[-1] != "[still]"
                    and len(event) >= 5
                    and event[4] not in [None, "null"]
                ):
                    target = standardize_pokemon_ident(event[4])
                active_pokemon = self._battle.active_pokemon

                # Check for covert cloak if an eligible move is made against an opponent
                if (
                    len(move.secondary) > 0
                    and move.secondary[0].get("chance", 0) == 100
                    and target in self._battle.opponent_team
                ):
                    # Goes ahead and looks for covert cloak
                    self._check_covert_cloak(segments["move"], i)

                # Check for safetygoggles if we have a mon that has a rage powder effect and two mons out on the field;
                # not implemented for singles
                if (
                    isinstance(active_pokemon, list)
                    and actor.startswith(self._battle.opponent_role)
                    and all(map(lambda x: x and not x.fainted, active_pokemon))
                    and any(
                        map(
                            lambda x: x and Effect.RAGE_POWDER in x.effects, active_pokemon
                        )
                    )
                ):
                    index = 1
                    if (
                        active_pokemon[0]
                        and Effect.RAGE_POWDER in active_pokemon[0].effects
                    ):  # pyright: ignore
                        index = 0

                    # If the actor targets the non rage-powdered mon and doesnt have immunity, it has safetygoggles
                    if active_pokemon[index] is None:
                        pass
                    elif (
                        target
                        and target
                        != get_showdown_identifier(
                            active_pokemon[index], self._battle.player_role
                        )  # pyright: ignore
                        and not has_rage_powder_immunity(get_pokemon(actor, self._battle))
                        and get_pokemon(actor, self._battle).item
                        in [None, GenData.UNKNOWN_ITEM, "safetygoggles"]
                        and (
                            not move.id == "snipeshot"
                            and move.target in [Target.ANY, Target.NORMAL]
                        )
                    ):
                        self._inferences.set_flag(actor, "can_be_choice", False)
                        self._inferences.set_flag(actor, "item", "safetygoggles")
                        self._battle.opponent_team[actor].item = "safetygoggles"
                    elif (
                        target
                        and target
                        != get_showdown_identifier(
                            active_pokemon[index], self._battle.player_role
                        )  # pyright: ignore
                        and not has_rage_powder_immunity(get_pokemon(actor, self._battle))
                        and (
                            not move.id == "snipeshot"
                            and move.target in [Target.ANY, Target.NORMAL]
                        )
                    ):
                        raise ValueError(
                            f"Found safetygoggles for {actor}, but it already has an item {get_pokemon(actor, self._battle).item}",
                            self._battle,
                            {
                                name: value
                                for name, value in inspect.getmembers(
                                    self._battle.opponent_team[actor]
                                )
                                if not name.startswith("__") or inspect.isfunction(value)
                            },
                        )

            # Check for heavydutyboots and reset tracking on switches; from pivot moves
            # This method updates iteration counter and parses batttle messages
            elif event[1] in ["switch", "drag"] and event[2].startswith(
                self._battle.opponent_role
            ):

                end = i + 1
                while end < len(segments["move"]) and segments["move"][end][1] not in [
                    "switch",
                    "drag",
                    "move",
                ]:
                    end += 1

                self._check_opponent_switch(segments["move"][i:end])

            update_battle(self._battle, event)
            i += 1

    def _handle_end_of_turn(self):
        # At the end of the turn, check for status moves; eliminates assault vest
        for ident in self._battle.opponent_team:
            if ident in self._battle.opponent_team:
                for move in self._battle.opponent_team[ident].moves.values():
                    if move.category == MoveCategory.STATUS:
                        self._inferences.set_flag(ident, "has_status_move", True)

                if self._opp_has_item(ident):
                    item = self._battle.opponent_team[ident].item
                    self._inferences.set_flag(ident, "item", item)
                    if item is not None and not item.startswith("choice"):
                        self._inferences.set_flag(ident, "can_be_choice", False)

        # At end of the turn, remove all the screens that have been removed from the battle
        for ident in self._battle.opponent_team:
            for sc in [item for item in self._inferences.get_flag(ident, "screens")]:
                if sc not in self._battle.opponent_side_conditions:
                    screens = self._inferences.get_flag(ident, "screens")
                    screens.remove(sc)
                    self._inferences.set_flag(ident, "screens", screens)

        # At end of turn, check battle for screen turns. If there's a screen that lasts longer than 5,
        # we should figure out who used it and then set them as lightclay
        for sc, turn_started in self._battle.opponent_side_conditions.items():
            # If the screen lasts longer than 5 turns, we find the pokemon that
            # set the screen and give them light clay.
            if self._battle.turn - turn_started > 5:
                for ident in self._battle.opponent_team:
                    if sc in self._inferences.get_flag(ident, "screens"):
                        if self._battle.opponent_team[ident].item in {
                            None,
                            GenData.UNKNOWN_ITEM,
                            "lightclay",
                        }:
                            self._inferences.set_flag(ident, "can_be_choice", False)
                            self._inferences.set_flag(ident, "item", "lightclay")
                            self._battle.opponent_team[ident].item = "lightclay"
                        else:
                            raise ValueError(
                                f"We found lightclay but {ident} has {self._battle.opponent_team[ident].item}",
                                self._battle,
                            )

    # Reset mons flags on switch-in, check for heavydutyboots
    # Doesnt actually iterate, and we need to pass it just the relevant events
    # (switch and everything up until the next switch or move). This assumes that the act of
    # switching won't affect whether heavydutyboots comes into play or not because we dont update
    # the battle state after the switch
    def _check_opponent_switch(self, events: List[List[str]]):
        if self._battle.opponent_role is None or self._battle.player_role is None:
            raise ValueError(
                "Cannot check opponent switches without roles",
                self._battle,
            )

        if events[0][1] not in ["switch", "drag"] or not events[0][2].startswith(
            self._battle.opponent_role
        ):
            raise ValueError(
                f"Expected switch event of opponent mon, but got {events[0]} instead",
                self._battle,
            )

        ident = standardize_pokemon_ident(events[0][2])
        mon = get_pokemon(ident, self._battle)

        # Reset mon flags
        self._inferences.set_flag(ident, "num_moves_since_switch", 0)
        self._inferences.set_flag(ident, "last_move", None)

        immune_to_spikes = (
            not is_grounded(ident, self._battle)
            or mon.ability == "magicguard"
            or (mon.ability is None and "magicguard" in mon.possible_abilities)
        )

        immune_to_toxic_spikes = not is_grounded(
            ident, self._battle
        ) or has_status_immunity(ident, Status.PSN, self._battle)

        immune_to_rocks = mon.ability == "magicguard" or (
            mon.ability is None and "magicguard" in mon.possible_abilities
        )

        immune_to_sticky_web = not is_grounded(ident, self._battle)

        # Only check for the ident
        affected_by_entry_hazard = any(
            map(
                lambda x: x[-1]
                in {
                    "psn",
                    "tox",
                    "[from] Spikes",
                    "[from] Stealth Rock",
                    "move: Sticky Web",
                }
                and standardize_pokemon_ident(x[2]) == ident,
                events,
            )
        )

        has_heavy_duty_boots = not affected_by_entry_hazard and (
            (
                SideCondition.SPIKES in self._battle.opponent_side_conditions
                and not immune_to_spikes
            )
            or (
                SideCondition.TOXIC_SPIKES in self._battle.opponent_side_conditions
                and not immune_to_toxic_spikes
            )
            or (
                SideCondition.STEALTH_ROCK in self._battle.opponent_side_conditions
                and not immune_to_rocks
            )
            or (
                SideCondition.STICKY_WEB in self._battle.opponent_side_conditions
                and not immune_to_sticky_web
            )
        )

        # Now evaluate if we have heavydutyboots
        if has_heavy_duty_boots and mon.item not in {
            None,
            GenData.UNKNOWN_ITEM,
            "heavydutyboots",
        }:
            raise ValueError(
                f"We found Heavy Duty Boots but {ident} has {mon.item}",
                self._battle,
                {
                    name: value
                    for name, value in inspect.getmembers(
                        self._battle.opponent_team[ident]
                    )
                    if not name.startswith("__")
                },
            )
        elif has_heavy_duty_boots:
            self._inferences.set_flag(ident, "can_be_choice", False)
            self._inferences.set_flag(ident, "item", "heavydutyboots")
            self._battle.opponent_team[ident].item = "heavydutyboots"

    # Updates move tracking; checks for choice items, and looks for screen_setting
    def _update_move_tracking(self, events: List[List[str]], i: int):
        if self._battle.opponent_role is None or self._battle.player_role is None:
            raise ValueError(
                "Cannot update move tracking without roles",
                self._battle,
            )

        ident = standardize_pokemon_ident(events[i][2])
        if ident.startswith(self._battle.opponent_role):
            # Move tracking for AI
            self._inferences.set_flag(
                ident, "num_moved", self._inferences.get_flag(ident, "num_moved") + 1
            )
            self._inferences.set_flag(
                ident,
                "num_moves_since_switch",
                self._inferences.get_flag(ident, "num_moves_since_switch") + 1,
            )

            # Check choice; mon can struggle if out of PP or Disabled
            if (
                self._inferences.get_flag(ident, "last_move") not in [events[i][3], None]
                and events[i][3] != "Struggle"
            ):
                self._inferences.set_flag(ident, "can_be_choice", False)
                self._inferences.set_flag(ident, "last_move", events[i][3])

            # Update last move
            self._inferences.set_flag(ident, "last_move", events[i][3])

            # Record who used a screen move to track for Light Clay
            if events[i][3] in {"Reflect", "Light Screen", "Aurora Veil"} and not (
                len(events) > i + 1 and events[i + 1][1] == "-fail"
            ):
                screens = self._inferences.get_flag(ident, "screens")
                screens.append(SideCondition.from_showdown_message(events[i][3]))
                self._inferences.set_flag(ident, "screens", screens)

    # Checks a series of events for covert cloak. Does not read events. This may fail with ejectpack;
    # I have not tested it with this yet
    def _check_covert_cloak(self, events: List[List[str]], i: int):
        if self._battle.opponent_role is None or self._battle.player_role is None:
            raise ValueError(
                "Cannot check for covert cloak without roles",
                self._battle,
            )

        move = Move(to_id_str(events[i][3]), self._battle.gen)

        # Means move didnt activate; showdown is not animating this
        target = (
            standardize_pokemon_ident(events[i][4]) if events[i][-1] != "[still]" else ""
        )

        if (
            len(move.secondary) == 0
            or move.secondary[0].get("chance", 0) != 100
            or target not in self._battle.opponent_team
        ):
            raise ValueError(
                f"Checking for Covert Cloak {events[i]}, but the event shouldn't trigger checking",
                self._battle,
                events,
            )

        # Get what effect we should be looking for to trigger based on the move
        key = ""
        if "boosts" in move.secondary[0]:
            key = "boosts"
        elif "status" in move.secondary[0]:
            key = "status"
        elif "volatileStatus" in move.secondary[0]:
            key = move.secondary[0]["volatileStatus"]

        # Find the index of the next move to know where to stop looking for
        # boosts, status and some volatileStatuses. We check for switches too to
        # handle the eject button/wimp out edge case
        end = i + 1
        while end < len(events) and events[end][1] not in ["move", "switch", "drag"]:
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
            # check if they got the secondary effect (or fainted). We go until we don't
            idents = []
            j = i
            while j <= end and events[j][1] in [
                "-damage",
                "-boost",
                "-unboost",
                "-resisted",
                "-immune",
                "faint",
                "-supereffective",
                "-crit",
                "-miss",
            ]:
                if events[j][1] == "-damage" and events[j][2].startswith(
                    self._battle.opponent_role
                ):
                    idents.append(standardize_pokemon_ident(events[j][2]))

                # We look back one event to ensure an ability (eg Stamina) or item doesn't trigger a boost
                elif events[j][1] in ["-boost", "-unboost"] and events[j][2].startswith(
                    self._battle.opponent_role
                ):
                    idents.remove(standardize_pokemon_ident(events[j][2]))
                elif events[j][1] in ["faint"] and events[j][2].startswith(
                    self._battle.opponent_role
                ):
                    idents.remove(standardize_pokemon_ident(events[j][2]))

                j += 1

            # This means that we get an ident that took damage, but didn't get a (un)boost
            if len(idents) > 0:
                if (
                    not self._opp_has_item(idents[0])
                    or self._battle.opponent_team[idents[0]].item == "covertcloak"
                ):
                    self._inferences.set_flag(idents[0], "can_be_choice", False)
                    self._inferences.set_flag(idents[0], "item", "covertcloak")
                    self._battle.opponent_team[idents[0]].item = "covertcloak"
                else:
                    raise ValueError(
                        f"We found Covert Cloak but {idents[0]} has {self._battle.opponent_team[idents[0]].item}",
                        self._battle,
                        events,
                        {
                            name: value
                            for name, value in inspect.getmembers(
                                self._battle.opponent_team[idents[0]]
                            )
                            if not name.startswith("__")
                        },
                    )

        # Check to see if someone has been damaged and if there is no status. If there is no status
        # this means that a mon has covert cloak. Note that this will fail if there is a move that
        # affects multiple targets and has 100% chance of incurring a status as a secondary
        # |-status|p1b: Incineroar|par|[from] move: Nuzzle
        elif (
            key == "status"
            and any(events[j][1] == "-damage" for j in range(i, end))
            and not any(events[j][1] in ["-status", "faint"] for j in range(i, end))
            and not has_status_immunity(
                target, Status[move.secondary[0].get("status", "").upper()], self._battle
            )
        ):
            if (
                not self._opp_has_item(target)
                or self._battle.opponent_team[target].item == "covertcloak"
            ):
                self._inferences.set_flag(target, "can_be_choice", False)
                self._inferences.set_flag(target, "item", "covertcloak")
                self._battle.opponent_team[target].item = "covertcloak"
            else:
                raise ValueError(
                    f"We found Covert Cloak but {target} has {self._battle.opponent_team[target].item}",
                    self._battle,
                    events,
                    {
                        name: value
                        for name, value in inspect.getmembers(
                            self._battle.opponent_team[target]
                        )
                        if not name.startswith("__")
                    },
                )

        # |-start|p1a: Amoonguss|Salt Cure
        elif (
            key == "saltcure"  # Salt Cure was used
            and events[i + 1][1] == "-damage"
            and events[i + 1][2] == target  # It hits
            and not any(
                events[j][1] in ["-start", "faint"] and events[j][-1] == "Salt Cure"
                for j in range(i, end)
            )  # we didn't faint or salt cure didnt activate
            and Effect.SALT_CURE
            not in self._battle.opponent_team[
                target
            ].effects  # we already didn't have salt cure
        ):
            if (
                not self._opp_has_item(target)
                or self._battle.opponent_team[target].item == "covertcloak"
            ):
                self._inferences.set_flag(target, "can_be_choice", False)
                self._inferences.set_flag(target, "item", "covertcloak")
                self._battle.opponent_team[target].item = "covertcloak"
            else:
                raise ValueError(
                    f"We found Covert Cloak but {target} has {self._battle.opponent_team[target].item}",
                    self._battle,
                    events,
                    {
                        name: value
                        for name, value in inspect.getmembers(
                            self._battle.opponent_team[target]
                        )
                        if not name.startswith("__")
                    },
                )

        # |move|p2b: Smeargle|Fake Out|p1b: Urshifu
        # |-damage|p1b: Urshifu|97/100
        # |cant|p1b: Urshifu|flinch
        elif key == "flinch":

            # Flinch will only happen if we damage the pokemon
            damaged = any(
                map(
                    lambda j: events[j][1] == "-damage"
                    and standardize_pokemon_ident(events[j][2]) == target,
                    range(i, end),
                )
            )
            fainted = any(
                map(
                    lambda j: events[j][1] == "faint"
                    and standardize_pokemon_ident(events[j][2]) == target,
                    range(i, end),
                )
            )

            # We have nothing to look for if we didnt hit the mon, or if the mon fainted
            if not damaged or fainted:
                return

            for j in range(end, len(events)):

                # If the mon successfully moves and it can be flinched, it has a covert cloak
                if (
                    events[j][1] == "move"
                    and standardize_pokemon_ident(events[j][2]) == target
                    and not has_flinch_immunity(get_pokemon(target, self._battle))
                ):
                    if (
                        not self._opp_has_item(target)
                        or self._battle.opponent_team[target].item == "covertcloak"
                    ):
                        self._inferences.set_flag(target, "can_be_choice", False)
                        self._inferences.set_flag(target, "item", "covertcloak")
                        self._battle.opponent_team[target].item = "covertcloak"
                    else:
                        raise ValueError(
                            f"We found Covert Cloak but {target} has {self._battle.opponent_team[target].item}",
                            self._battle,
                            events,
                            {
                                name: value
                                for name, value in inspect.getmembers(
                                    self._battle.opponent_team[target]
                                )
                                if not name.startswith("__")
                            },
                        )

                    return

    # Goes through residuals and looks for mons who should be hit by sandstorm, but aren't
    def _check_residuals_for_safety_goggles(self, events: List[List[str]]):
        if self._battle.opponent_role is None:
            raise ValueError(
                "Cannot check for SafetyGoggles without an opponent role", self._battle
            )

        # If there's no sandstorm, we just go through our events; nothing to check
        elif Weather.SANDSTORM not in self._battle.weather:
            for event in events:
                update_battle(self._battle, event)
            return

        # We are in doubles
        elif isinstance(self._battle.opponent_active_pokemon, list):
            # We track all mons that should be hit by sandstorm
            opp_actives = [
                get_showdown_identifier(mon, self._battle.opponent_role)
                for mon in self._battle.opponent_active_pokemon
                if mon
            ]

            for event in events:

                # Remove opponents that have been hit by Sandstorm
                if (
                    event[2].startswith(self._battle.opponent_role)
                    and event[-1] == "[from] Sandstorm"
                    and standardize_pokemon_ident(event[2]) in opp_actives
                ):
                    opp_actives.remove(standardize_pokemon_ident(event[2]))

                update_battle(self._battle, event)

            # Remove opponents remaining opponents that shouldnt have been hit by sandstorm
            opp_actives = [
                ident
                for ident in opp_actives
                if not has_sandstorm_immunity(get_pokemon(ident, self._battle))
            ]

            if len(opp_actives) == 1 and (
                not self._opp_has_item(opp_actives[0])
                or self._battle.opponent_team[opp_actives[0]].item == "safetygoggles"
            ):
                # We found SafetyGoggles, but we need to make sure the mon has it
                self._inferences.set_flag(opp_actives[0], "can_be_choice", False)
                self._inferences.set_flag(opp_actives[0], "item", "safetygoggles")
                self._battle.opponent_team[opp_actives[0]].item = "safetygoggles"

            elif len(opp_actives) > 1:
                # This technically could happen if we trick SafetyGoogles, and had another Safetygoggles during a sandstorm
                raise ValueError(
                    f"Only at max one expected pokemon can not be affected by Sandstorm, but got {opp_actives}",
                    self._battle,
                    events,
                )
            elif (
                len(opp_actives) == 1
                and self._opp_has_item(opp_actives[0])
                and self._battle.opponent_team[opp_actives[0]].item != "safetygoggles"
            ):
                raise ValueError(
                    f"We found SafetyGoggles but {opp_actives[0]} has {self._battle.opponent_team[opp_actives[0]].item}",
                    self._battle,
                    events,
                    {
                        name: value
                        for name, value in inspect.getmembers(
                            self._battle.opponent_team[opp_actives[0]]
                        )
                        if not name.startswith("__")
                    },
                )

        elif isinstance(self._battle.opponent_active_pokemon, Pokemon):

            # Singles; requirement are different and simpler. Haven't tested!
            if (
                self._battle.opponent_active_pokemon is not None
                and not has_sandstorm_immunity(self._battle.opponent_active_pokemon)
                and not any(
                    map(
                        lambda x: self._battle.opponent_role is not None
                        and x[1].startswith(self._battle.opponent_role)
                        and x[-1] == "[from] Sandstorm",
                        events,
                    )
                )
            ):
                # We found SafetyGoggles, but we need to make sure the mon has it
                ident = get_showdown_identifier(
                    self._battle.opponent_active_pokemon, self._battle.opponent_role
                )
                self._inferences.set_flag(ident, "can_be_choice", False)
                self._inferences.set_flag(ident, "item", "safetygoggles")
                self._battle.opponent_team[ident].item = "safetygoggles"

            for event in events:
                update_battle(self._battle, event)

    def _opp_has_item(self, ident):
        return (
            self._battle.opponent_team[ident].item is not None
            and self._battle.opponent_team[ident].item != GenData.UNKNOWN_ITEM
        )
