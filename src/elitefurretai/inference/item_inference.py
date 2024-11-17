# -*- coding: utf-8 -*-
"""This module makes items inferences throughout a battle by tracking pokemons' moves.
This object is a companion class to BattleInference.
"""

from typing import Dict, List, Union

from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import (
    battle_to_str,
    copy_bare_battle,
    get_pokemon,
    get_segments,
    get_showdown_identifier,
    has_flinch_immunity,
    has_rage_powder_immunity,
    has_sandstorm_immunity,
    has_status_immunity,
    has_unboost_immunity,
    is_grounded,
    standardize_pokemon_ident,
    update_battle,
)


class ItemInference:
    __slots__ = (
        "_battle",
        "_inferences",
        "_last_tracked_event",
        "_last_tracked_turn",
    )

    # We need to keep a copy of the battle state because we need to recreate the turn and conditions of the turn to parse speeds
    def __init__(
        self,
        battle: Union[Battle, DoubleBattle, AbstractBattle],
        inferences: BattleInference,
    ):
        self._battle = copy_bare_battle(battle)
        self._inferences: BattleInference = inferences
        self._last_tracked_event: int = 0
        self._last_tracked_turn: int = -1

        if battle.teampreview:
            self._battle.parse_request(battle.last_request)

    # Update our internal tracking of the battle, and apply inferences
    def update(self, battle: Union[Battle, DoubleBattle]):

        if len(battle.observations) == 0:
            return

        # Get most recent observation
        turn_of_events = max(battle.observations.keys())
        obs = battle.observations[max(battle.observations.keys())]

        # TODO: because of a poke_env bug, we should only do one inference per turn.
        # So if we've already touched this turn, we're just going to update the battle
        if (isinstance(battle.force_switch, list) and any(battle.force_switch)) or (
            isinstance(battle.force_switch, bool) and battle.force_switch
        ):
            turn_of_events = battle.turn
            obs = battle.current_observation

        # Check if we've already updated
        if (
            len(obs.events) > self._last_tracked_event
            and self._last_tracked_event > 0
            and turn_of_events == self._last_tracked_turn
        ) or turn_of_events > self._last_tracked_turn:

            # Go through logs; actual inference
            self.check_items(obs.events, start=self._last_tracked_event)

            # Parse the request to update our internal battle copy after updating
            # the events, since the request happened right before this call
            self._battle.parse_request(battle.last_request)

            # Update tracking
            self._last_tracked_event = len(obs.events)
            if len(obs.events) > 0 and obs.events[-1][1] == "turn":
                self._last_tracked_event = 0
            self._last_tracked_turn = turn_of_events

    def check_items(self, events: List[List[str]], start: int = 0):
        """
        Updates the inference with the latest events

        :param observation: The events of the turn to check, and which index to start
        """

        # Nothing to do if battle isnt initiated
        if self._battle.opponent_role is None or self._battle.player_role is None:
            return

        # If it's the first turn, we can't infer any items
        if self._battle.turn == 0:
            for event in events:
                update_battle(self._battle, event)
            return

        segments = get_segments(events, start=start)

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

                    update_battle(self._battle, segments["switch"][i])
                    self._check_opponent_switch(segments["switch"][i:end])
                else:
                    update_battle(self._battle, segments["switch"][i])

                i += 1

        if "battle_mechanic" in segments:
            for event in segments["battle_mechanic"]:
                update_battle(self._battle, event)

        # Reset mons flags on switch-in, check for heavydutyboots
        # Check for covert cloak and safetygoggles
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
                    update_battle(self._battle, segments["preturn_switch"][i])
                    self._check_opponent_switch(segments["preturn_switch"][i:])
                else:
                    update_battle(self._battle, segments["preturn_switch"][i])

                i += 1

        # If we see a turn, that means we're done with the observation and
        # we can reset last_event
        if "turn" in segments:
            self._handle_end_of_turn()
            for event in segments["turn"]:
                update_battle(self._battle, event)

    def _handle_moves(self, segments: Dict[str, List[List[str]]]):

        # Nothing to do if battle isnt initiated
        if self._battle.opponent_role is None or self._battle.player_role is None:
            return

        i = 0
        while i < len(segments["move"]):

            event = segments["move"][i]

            # An opponent is given an item
            if event[1] == "-item" and event[2].startswith(self._battle.opponent_role):
                ident = standardize_pokemon_ident(event[2])
                item = to_id_str(event[3])
                self._inferences.set_flag(ident, "item", item)
                self._inferences.set_flag(ident, "screens", [])

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
                    and len(event) > 4
                    and event[4] != "[still]"
                    and event[4]
                    not in [
                        None,
                        "null",
                        "",
                    ]  # is empty string for lifedew/lunarblessing for some reason
                ):
                    target = standardize_pokemon_ident(event[4])

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
                    isinstance(self._battle.active_pokemon, list)
                    and actor.startswith(self._battle.opponent_role)
                    and all(
                        map(lambda x: x and not x.fainted, self._battle.active_pokemon)
                    )
                    and any(
                        map(
                            lambda x: x and Effect.RAGE_POWDER in x.effects,
                            self._battle.active_pokemon,
                        )
                    )
                    and target is not None
                    and actor != target
                ):
                    self._check_rage_powder_safety_goggles(actor, target, move)

            # Note that I call update_battle before I check the opponent switch because I need
            # the event to register to have the mon in battle.opponent_team
            update_battle(self._battle, event)

            # Check for heavydutyboots and reset tracking on switches; from pivot moves
            # This method updates iteration counter and parses batttle messages.
            if event[1] in ["switch", "drag"] and event[2].startswith(
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

            # Now I increment the events
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
            screens = self._inferences.get_flag(ident, "screens")
            for sc in [item for item in (screens if screens is not None else [])]:
                if sc not in self._battle.opponent_side_conditions:
                    screens = self._inferences.get_flag(ident, "screens")
                    if screens is not None and sc in screens:
                        screens.remove(sc)
                    self._inferences.set_flag(ident, "screens", screens)

        # At end of turn, check battle for screen turns. If there's a screen that lasts longer than 5,
        # we should figure out who used it and then set them as lightclay
        for sc, turn_started in self._battle.opponent_side_conditions.items():
            # If the screen lasts longer than 5 turns, we find the pokemon that
            # set the screen and give them light clay.
            if self._battle.turn - turn_started > 5:
                for ident in self._battle.opponent_team:
                    screens = self._inferences.get_flag(ident, "screens")
                    if (
                        screens is not None
                        and sc in screens
                        and self._battle.opponent_team[ident].item
                        in {
                            None,
                            GenData.UNKNOWN_ITEM,
                            "lightclay",
                        }
                    ):
                        self._inferences.set_flag(ident, "can_be_choice", False)
                        self._inferences.set_flag(ident, "item", "lightclay")
                        self._inferences.set_flag(ident, "can_be_clearamulet", False)
                        self._inferences.set_flag(ident, "can_be_covertcloak", True)

    # Reset mons flags on switch-in, check for heavydutyboots
    # Doesnt actually iterate, and we need to pass it just the relevant events
    # (switch and everything up until the next switch or move). This assumes that the act of
    # switching won't affect whether heavydutyboots comes into play or not because we dont update
    # the battle state after the switch
    def _check_opponent_switch(self, events: List[List[str]]):
        if self._battle.opponent_role is None or self._battle.player_role is None:
            print(battle_to_str(self._battle))
            raise ValueError(
                "Cannot check opponent switches without roles",
            )

        if events[0][1] not in ["switch", "drag"] or not events[0][2].startswith(
            self._battle.opponent_role
        ):
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Expected switch event of opponent mon, but got {events[0]} instead",
            )

        ident = standardize_pokemon_ident(events[0][2])
        mon = get_pokemon(ident, self._battle)

        # Reset mon flags
        self._inferences.set_flag(ident, "num_moves_since_switch", 0)
        self._inferences.set_flag(ident, "last_move", None)

        # Check Intimidate
        # ['', '-ability', 'p1a: Incineroar', 'Intimidate', 'boost'],
        # ['', '-unboost', 'p2a: Whimsicott', 'atk', '1'],
        # ['', '-unboost', 'p2b: Ogerpon', 'atk', '1']
        if (
            len(events) > 1
            and len(events[1]) > 3
            and events[1][3] == "Intimidate"
            and events[1][2].startswith(self._battle.player_role)
        ):
            # We go through every opponent mon and check if they got hit, and if they did, we
            # check if they got the secondary effect (or fainted). We go until we don't
            if not isinstance(self._battle.opponent_active_pokemon, list):
                raise NotImplementedError()
            idents = [
                get_showdown_identifier(m, self._battle.opponent_role) if m else None
                for m in self._battle.opponent_active_pokemon
            ]

            j = 2
            while j < len(events):
                # We look back one event to ensure an ability (eg Stamina) or item doesn't trigger a boost
                if events[j][1] in ["-boost", "-unboost"] and events[j][2].startswith(
                    self._battle.opponent_role
                ):
                    if standardize_pokemon_ident(events[j][2]) in idents:
                        idents.remove(standardize_pokemon_ident(events[j][2]))

                j += 1

            # Remove any mons that are immune to Intimidate
            for mon_ident in idents[:]:
                if mon_ident is None:
                    idents.remove(mon_ident)
                elif has_unboost_immunity(mon_ident, "atk", self._battle):
                    idents.remove(mon_ident)
                elif get_pokemon(mon_ident, self._battle).ability == "innerfocus" or (
                    get_pokemon(mon_ident, self._battle).ability is None
                    and "innerfocus"
                    in get_pokemon(mon_ident, self._battle).possible_abilities
                ):
                    idents.remove(mon_ident)

            # If there's a mon left. It has clear amulet. Otherwise, we know other mons arent
            if len(idents) > 0 and idents[0]:
                self._inferences.set_flag(idents[0], "can_be_choice", False)
                self._inferences.set_flag(idents[0], "item", "clearamulet")
                self._inferences.set_flag(idents[0], "can_be_clearamulet", True)
                self._inferences.set_flag(idents[0], "can_be_covertcloak", False)

            for m in self._battle.opponent_active_pokemon:
                if m is not None:
                    other_ident = get_showdown_identifier(m, self._battle.opponent_role)
                    if other_ident not in idents and not has_unboost_immunity(
                        other_ident, "atk", self._battle
                    ):
                        self._inferences.set_flag(other_ident, "can_be_clearamulet", False)

        # Check HeavyDutyBoots
        immune_to_spikes = (
            not is_grounded(ident, self._battle)
            or mon.ability == "magicguard"
            or (mon.ability is None and "magicguard" in mon.possible_abilities)
        )

        immune_to_toxic_spikes = not is_grounded(
            ident, self._battle
        ) or has_status_immunity(ident, Status.PSN, self._battle)

        immune_to_rocks = (mon.ability == "magicguard") or (
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
        if has_heavy_duty_boots and mon.item in {
            None,
            GenData.UNKNOWN_ITEM,
            "heavydutyboots",
        }:
            self._inferences.set_flag(ident, "can_be_choice", False)
            self._inferences.set_flag(ident, "item", "heavydutyboots")
            self._inferences.set_flag(ident, "can_be_clearamulet", False)
            self._inferences.set_flag(ident, "can_be_covertcloak", False)

    def _check_rage_powder_safety_goggles(self, actor: str, target: str, move: Move):

        # Not implemented for singles
        if not isinstance(self._battle, DoubleBattle):
            raise NotImplementedError()

        active_pokemon = self._battle.active_pokemon
        index = 1
        if active_pokemon[0] and Effect.RAGE_POWDER in active_pokemon[0].effects:
            index = 0

        is_rage_powder_immune = has_rage_powder_immunity(get_pokemon(actor, self._battle))
        if move.target in [
            Target.ALL,
            Target.SELF,
            Target.ALL_ADJACENT,
            Target.ALL_ADJACENT_FOES,
            Target.ALLIES,
            Target.ALLY_SIDE,
            Target.FOE_SIDE,
        ]:
            is_rage_powder_immune = True
        if move.id == "snipeshot" and move.target in [Target.ANY, Target.NORMAL]:
            is_rage_powder_immune = True
        if (
            move.id == "terastarstorm"
            and PokemonType.STELLAR in self._battle.opponent_team[actor].types
        ):
            is_rage_powder_immune = True
        if move.id == "expandingforce" and Field.PSYCHIC_TERRAIN in self._battle.fields:
            is_rage_powder_immune = True
        if (
            active_pokemon[1 - index] is not None
            and Effect.FOLLOW_ME in active_pokemon[1 - index].effects  # pyright: ignore
        ):
            is_rage_powder_immune = True

        # If the actor targets the non rage-powdered mon and doesnt have immunity, it has safetygoggles
        if (
            active_pokemon[index] is not None
            and target is not None
            and target
            != get_showdown_identifier(
                active_pokemon[index], self._battle.player_role  # pyright: ignore
            )
            and not is_rage_powder_immune
            and get_pokemon(actor, self._battle).item
            in [None, GenData.UNKNOWN_ITEM, "safetygoggles"]
        ):
            self._inferences.set_flag(actor, "can_be_choice", False)
            self._inferences.set_flag(actor, "item", "safetygoggles")
            self._inferences.set_flag(actor, "can_be_clearamulet", False)
            self._inferences.set_flag(actor, "can_be_covertcloak", False)

    # Updates move tracking; checks for choice items, and looks for screen_setting
    def _update_move_tracking(self, events: List[List[str]], i: int):

        # Nothing to do if battle isnt initiated
        if self._battle.opponent_role is None or self._battle.player_role is None:
            return

        ident = standardize_pokemon_ident(events[i][2])
        if ident.startswith(self._battle.opponent_role):
            # Move tracking for AI
            num_moved = self._inferences.get_flag(ident, "num_moved")
            self._inferences.set_flag(
                ident, "num_moved", (num_moved if num_moved is not None else 0) + 1
            )

            nummoves_since_switch = self._inferences.get_flag(
                ident, "num_moves_since_switch"
            )
            self._inferences.set_flag(
                ident,
                "num_moves_since_switch",
                (nummoves_since_switch if nummoves_since_switch is not None else 0) + 1,
            )

            # Check choice; mon can struggle if out of PP or Disabled
            if (
                self._inferences.get_flag(ident, "last_move") not in [events[i][3], None]
                and events[i][3] != "Struggle"
                and "[from]move: Sleep Talk" not in events[i]
            ):
                self._inferences.set_flag(ident, "can_be_choice", False)

            # Update last move
            self._inferences.set_flag(ident, "last_move", events[i][3])

            # Record who used a screen move to track for Light Clay
            if events[i][3] in {"Reflect", "Light Screen", "Aurora Veil"} and not (
                len(events) > i + 1 and events[i + 1][1] == "-fail"
            ):
                screens = self._inferences.get_flag(ident, "screens")
                if screens is not None:
                    screens.append(SideCondition.from_showdown_message(events[i][3]))
                    self._inferences.set_flag(ident, "screens", screens)

    # Checks a series of events for covert cloak. Does not read events. This may fail with ejectpack;
    # I have not tested it with this yet
    def _check_covert_cloak(self, events: List[List[str]], i: int):
        if self._battle.opponent_role is None or self._battle.player_role is None:
            print(battle_to_str(self._battle))
            raise ValueError(
                "Cannot check for covert cloak without roles",
            )
        elif events[i][1] != "move":
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Checking for Covert Cloak {events[i]}, but the first event is not a move"
                + "and this is not expected behavior",
                events,
                i,
            )

        actor = events[i][2]
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
            print(battle_to_str(self._battle))
            raise ValueError(
                f"Checking for Covert Cloak {events[i]}, but the event shouldn't trigger checking",
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
            idents, affected, ineligible = [], [], []
            boost_types = move.secondary[0]["boosts"].keys()
            j = i + 1
            while j < end:
                if (
                    events[j][1] == "-damage"
                    and events[j][2].startswith(self._battle.opponent_role)
                    and events[j][2]
                    != actor  # can happen if I self-attack and I damage with Rocky Helmet
                ):
                    idents.append(standardize_pokemon_ident(events[j][2]))

                # We look back one event to ensure an ability (eg Stamina) or item doesn't trigger a boost
                elif (
                    events[j][1] in ["-boost", "-unboost"]
                    and events[j][2].startswith(self._battle.opponent_role)
                    and events[j][2] != actor
                ):
                    affected.append(standardize_pokemon_ident(events[j][2]))

                # If a mon faints, then it's ineligible to be considered
                elif (
                    events[j][1] == "faint"
                    and events[j][2].startswith(self._battle.opponent_role)
                    and events[j][2] != actor
                ):
                    ineligible.append(standardize_pokemon_ident(events[j][2]))

                j += 1

            # Store original idents so we can know who didn't pass
            orig_idents = idents[:]
            for ident in orig_idents:
                for boost_type in boost_types:
                    if (
                        has_unboost_immunity(ident, boost_type, self._battle)
                        or ident in ineligible
                        or ident in affected
                    ):
                        idents.remove(ident)

            # This means that we get an ident that took damage, but didn't get a (un)boost
            if len(idents) > 0:

                # We don't know if what we're seeing is a clear amulet or a covert cloak
                if (
                    not self._opp_has_item(idents[0])
                    or self._battle.opponent_team[idents[0]].item == "covertcloak"
                    or self._battle.opponent_team[idents[0]].item == "clearamulet"
                    or self._inferences.get_flag(idents[0], "item") == "covertcloak"
                    or self._inferences.get_flag(idents[0], "item") == "clearamulet"
                ) and (
                    self._inferences.get_flag(idents[0], "can_be_clearamulet")
                    in [True, None]
                    or self._inferences.get_flag(idents[0], "can_be_covertcloak")
                    in [True, None]
                ):
                    # Check flags
                    if self._inferences.get_flag(idents[0], "can_be_clearamulet") is False:
                        self._inferences.set_flag(idents[0], "can_be_choice", False)
                        self._inferences.set_flag(idents[0], "can_be_covertcloak", True)
                        self._inferences.set_flag(idents[0], "item", "covertcloak")
                    elif (
                        self._inferences.get_flag(idents[0], "can_be_covertcloak") is False
                    ):
                        self._inferences.set_flag(idents[0], "can_be_choice", False)
                        self._inferences.set_flag(idents[0], "can_be_clearamulet", True)
                        self._inferences.set_flag(idents[0], "item", "clearamulet")
                    else:
                        self._inferences.set_flag(idents[0], "can_be_choice", False)
                        self._inferences.set_flag(idents[0], "can_be_covertcloak", True)
                        self._inferences.set_flag(idents[0], "can_be_clearamulet", True)

            # Confirm that mons that did get hit with the unboosts dont have either item
            for ident in orig_idents:
                if ident in affected:
                    self._inferences.set_flag(ident, "can_be_clearamulet", False)
                    self._inferences.set_flag(ident, "can_be_covertcloak", False)

        # Check to see if someone has been damaged and if there is no status. If there is no status
        # this means that a mon has covert cloak. Note that this will fail if there is a move that
        # affects multiple targets and has 100% chance of incurring a status as a secondary
        # |-status|p1b: Incineroar|par|[from] move: Nuzzle
        elif (
            key == "status"
            and any(events[j][1] == "-damage" for j in range(i, end))
            and not any(events[j][1] in ["-status", "faint"] for j in range(i, end))
            and not has_status_immunity(
                target,
                Status[move.secondary[0].get("status", "").upper()],
                self._battle,
            )
        ):
            if (
                not self._opp_has_item(target)
                or self._battle.opponent_team[target].item == "covertcloak"
            ):
                self._inferences.set_flag(target, "can_be_choice", False)
                self._inferences.set_flag(target, "item", "covertcloak")
                self._inferences.set_flag(target, "can_be_clearamulet", False)
                self._inferences.set_flag(target, "can_be_covertcloak", True)

        # |-start|p1a: Amoonguss|Salt Cure
        elif (
            key == "saltcure"  # Salt Cure was used
            and events[i + 1][1] == "-damage"  # it hits
            and standardize_pokemon_ident(events[i + 1][2]) == target  # It hits our mon
            and not any(
                events[j][1] in ["-start", "faint"] and events[j][-1] == "Salt Cure"
                for j in range(i, end)
            )  # we didn't faint or salt cure didnt activate
            and Effect.SALT_CURE
            not in self._battle.opponent_team[
                target
            ].effects  # we already didn't have salt cure
            and not any(
                events[j][1] in ["switch", "drag"]
                and events[j][2].startswith(events[i][4][:3])
                for j in range(i, len(events))
            )  # we didn't switch out
        ):
            if (
                not self._opp_has_item(target)
                or self._battle.opponent_team[target].item == "covertcloak"
            ):
                self._inferences.set_flag(target, "can_be_choice", False)
                self._inferences.set_flag(target, "item", "covertcloak")
                self._inferences.set_flag(target, "can_be_clearamulet", False)
                self._inferences.set_flag(target, "can_be_covertcloak", True)

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
                # Note that showdown doesn't handle Magic Bounce well, so we have to add a special
                # condition: https://github.com/smogon/pokemon-showdown/issues/10660
                if (
                    events[j][1] == "move"
                    and standardize_pokemon_ident(events[j][2]) == target
                    and not has_flinch_immunity(get_pokemon(target, self._battle))
                    and "magicbounce"
                    not in get_pokemon(target, self._battle).possible_abilities
                ):
                    if (
                        not self._opp_has_item(target)
                        or self._battle.opponent_team[target].item == "covertcloak"
                    ):
                        self._inferences.set_flag(target, "can_be_choice", False)
                        self._inferences.set_flag(target, "item", "covertcloak")
                        self._inferences.set_flag(target, "can_be_clearamulet", False)
                        self._inferences.set_flag(target, "can_be_covertcloak", True)

                    return

    # Goes through residuals and looks for mons who should be hit by sandstorm, but aren't
    def _check_residuals_for_safety_goggles(self, events: List[List[str]]):
        if self._battle.opponent_role is None:
            print(battle_to_str(self._battle))
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
                or self._inferences.get_flag(opp_actives[0], "item") == "safetygoggles"
                or self._battle.opponent_team[opp_actives[0]].item == "safetygoggles"
            ):
                # We found SafetyGoggles, but we need to make sure the mon has it
                self._inferences.set_flag(opp_actives[0], "can_be_choice", False)
                self._inferences.set_flag(opp_actives[0], "can_be_covertcloak", False)
                self._inferences.set_flag(opp_actives[0], "can_be_clearamulet", False)
                self._inferences.set_flag(opp_actives[0], "item", "safetygoggles")

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
                self._inferences.set_flag(ident, "can_be_clearamulet", False)
                self._inferences.set_flag(ident, "can_be_covertcloak", False)
                self._inferences.set_flag(ident, "item", "safetygoggles")
                self._battle.opponent_team[ident].item = "safetygoggles"

            for event in events:
                update_battle(self._battle, event)

    def _opp_has_item(self, ident):
        return (
            self._battle.opponent_team[ident].item is not None
            and self._battle.opponent_team[ident].item != GenData.UNKNOWN_ITEM
        )
