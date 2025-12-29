# -*- coding: utf-8 -*-
"""This module makes speed inferences throughout a battle by tracking pokemons' moves, stats and items.
It even can even establish whether a mon has choice scarf. This object is a companion class to BattleInference
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import pulp
from poke_env.battle import (
    AbstractBattle,
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data.gen_data import GenData
from poke_env.stats import _raw_stat

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import (
    ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    FIRST_BLOCK_RESIDUALS,
    ITEMS_THAT_ACTIVATE_ON_SWITCH,
    MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    PRIORITY_ACTIVATION_ABILITIES,
    SECOND_BLOCK_RESIDUALS,
    THIRD_BLOCK_RESIDUALS,
    copy_bare_battle,
    get_ability_and_identifier,
    get_priority_and_identifier,
    get_residual_and_identifier,
    get_segments,
    is_ability_event,
    standardize_pokemon_ident,
    update_battle,
)


class SpeedInference:
    __slots__ = (
        "_battle",
        "_inferences",
        "_orders",
        "_last_tracked_event",
        "_last_tracked_turn",
    )

    # We need to keep a copy of the battle state because we need to recreate the turn and conditions of the turn to parse speeds
    def __init__(
        self,
        battle: Union[Battle, DoubleBattle, AbstractBattle],
        inferences: BattleInference,
    ):
        self._battle: Union[Battle, DoubleBattle, AbstractBattle] = copy_bare_battle(
            battle
        )
        self._inferences: BattleInference = inferences
        self._orders: List[List[Tuple[str, float]]] = []
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

        # TODO: showdown reversing requests and inputs should fix order of operations
        # We should revisit this tracking
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
            # Go through speeds
            self.track_speeds(obs.events, start=self._last_tracked_event)

            # Parse the request to update our internal battle copy after updating
            # the events, since the request happened right before this call
            self._battle.parse_request(battle.last_request)

            # Update tracking
            self._last_tracked_event = len(obs.events)
            if len(obs.events) > 0 and obs.events[-1][1] == "turn":
                self._last_tracked_event = 0
            self._last_tracked_turn = turn_of_events

    def track_speeds(self, events: List[List[str]], start: int = 0):
        """
        The main method of this class. It creates bounds for speed by going through events in Observation, supported by a copy
        of the battle that is played along the original. We look at each segment/chapter of the turn:
        - Switches (triggers from abilities)
        - Tera/Mega/Dynamax
        - Moves (sorted by priorities)
        - Residuals
        - Switches from fainted mons (and triggers from abilities)

        We store each order we observe and the associated multiplier of speed based on the current battle conditions. We turn each
        of these "orders" into a set of equations that we solve to minimize and maximize opponent stats to get the mons' stat ranges.
        We store these in the flags we share with BattleInference. This means this method can get speed inferences wrong if they
        could make sense when there's actually a choicescarf

        Right now, we assume abilities that affect speeds/priorities, we don't look at every interaction of moves, and only try to
        learn choice scarfs (not iron ball or lagging tail).
        """
        speed_orders: List[List[Tuple[str, Optional[float]]]] = []

        if start >= len(events) or len(events[start:]) == 0:
            return

        segments = get_segments(events, start=start)
        if start != 0:
            print(events)
            print(start)
            print(segments)

        if "init" in segments:
            for event in segments["init"]:
                update_battle(self._battle, event)

        # I don't parse the order of activations (right now: Quick Claw,
        # Custap Berry and Quick Draw) because of the nicheness
        if "activation" in segments:
            for event in segments["activation"]:
                update_battle(self._battle, event)

        if "switch" in segments:
            seg = self._parse_switch(segments["switch"])
            speed_orders += seg

        if "battle_mechanic" in segments:
            seg = self._parse_battle_mechanic(segments["battle_mechanic"])
            speed_orders += seg

        if "move" in segments:
            seg = self._parse_move(segments["move"])
            speed_orders += seg

        if "state_upkeep" in segments:
            for event in segments["state_upkeep"]:
                update_battle(self._battle, event)

        if "residual" in segments:
            seg = self._parse_residual(segments["residual"])
            speed_orders += seg

        if "preturn_switch" in segments:
            seg = self._parse_preturn_switch(segments["preturn_switch"])
            speed_orders += seg

        if "turn" in segments:
            for event in segments["turn"]:
                update_battle(self._battle, event)

        # Clean the orders by removing None speeds
        def none_filter(order: list) -> bool:
            return len(order) == 2 and order[0][1] is not None and order[1][1] is not None

        self._solve_speeds(list(filter(none_filter, speed_orders)))  # type: ignore

    # Uses pulp to solve for speeds. We create the set of equations and try to solve them. If we get an infeasible
    # set of equations, this means that a mon has choice scarf (or SpeedInference's parsing is incomplete/buggy).
    # If we find infeasibility, we multiply each mon's speed by 1.5 if they are choicescarf eligible to see if we
    # can find a speed that makes the set of equations work. If it's only one mon, we know it's choicescarf! If
    # we find an optimal solution, we set the new speed bounds.
    def _solve_speeds(self, orders: List[List[Tuple[str, float]]]):
        assert (
            self._battle.opponent_role is not None
        ), "Battle must be set before solving speeds"

        # Add all our orders to full set of orders we've observed in the battle
        self._orders.extend(orders)

        # Create a LP problems
        problems = [
            pulp.LpProblem("MinProblem", pulp.LpMinimize),
            pulp.LpProblem("MaxProblem", pulp.LpMaximize),
        ]

        # Create the variables we're trying to solve for, which are opponent mons' speeds
        variables: Dict[str, pulp.LpVariable] = {}
        for mon in self._battle.opponent_team.values():
            key = mon.identifier(self._battle.opponent_role)

            variables[key] = pulp.LpVariable(
                key,
                lowBound=_raw_stat(mon.base_stats["spe"], 0, 0, mon.level, 0.9),
                upBound=_raw_stat(mon.base_stats["spe"], 252, 31, mon.level, 1.1),
                cat="Integer",
            )

        # We record the constraints of our mons as their actual speeds, because we know their true value
        constraints: Dict[str, float] = {}
        for key, mon in self._battle.team.items():
            # This happens in teampreview, since battle mon are initiated with a request
            # and we solve for speeds before the request
            if mon.stats["spe"] is None:
                for tp_mon in self._battle.teampreview_team:
                    if mon.name == tp_mon.name and tp_mon.stats["spe"] is not None:
                        constraints[key] = tp_mon.stats["spe"]
            else:
                constraints[key] = mon.stats["spe"]

        # For each problem (a minimization and maximization one)
        for problem in problems:
            # This is our objective function; either maximize or minimize total speed stats
            # for opponent mons (this includes our mons; which is an int and invariant of solving)
            problem += pulp.lpSum([variables[key] for key in variables])

            # We add each order that we parsed, only if there's an opponent speed to parse
            for order in self._orders:
                first_mon, first_mult = order[0]
                second_mon, second_mult = order[1]

                if first_mon in self._battle.opponent_team:
                    if second_mon in self._battle.opponent_team:
                        problem += (
                            variables[first_mon] * first_mult
                            >= variables[second_mon] * second_mult
                        )
                    else:
                        problem += (
                            variables[first_mon] * first_mult
                            >= constraints[second_mon] * second_mult
                        )
                elif second_mon in self._battle.opponent_team:
                    problem += (
                        constraints[first_mon] * first_mult
                        >= variables[second_mon] * second_mult
                    )

            # Solve our set of linear equations (using an arbitrary solver, with no command line output)
            problem.solve(solver=pulp.apis.SCIP_PY(msg=False))

            # Flag for whether we can update our speed tracking
            success, choicescarfmon = False, ""

            # If we get infeasible conditions, we check for Choice Scarf (we don't check for iron ball given its rarity)
            # This assumes that our speed tracking is correct; we could get infeasible if I miss a speed interaction
            if pulp.LpStatus[problem.status] == "Infeasible":
                # Loop through every mon that could have choice scarf and test if the problem can be solved by assuming one
                # We can have a situation where multiple mons could have a choice scarf and the set of equations works
                # so we only claim success
                for mon_ident in self._battle.opponent_team:
                    if (
                        self._inferences.get_flag(mon_ident, "can_be_choice")
                        and variables[mon_ident].upBound
                    ):
                        # See if we can solve it if we let the speed be 1.5 the max
                        variables[mon_ident].upBound = variables[mon_ident].upBound * 1.5  # type: ignore
                        problem.solve(solver=pulp.apis.SCIP_PY(msg=False))

                        # If solvable, we found our choice scarf mon! We should record success and update the mon
                        if pulp.LpStatus[problem.status] == "Optimal":
                            # Need to go back and update the previous orders to note that there was a choice scarf there,
                            # since previous orders were wrong
                            self._update_orders(mon_ident, 1.5)

                            # Record that we found the item and update inferences
                            self._inferences.set_flag(mon_ident, "item", "choicescarf")
                            success, choicescarfmon = True, mon_ident

                        # Reset the upperbound to normal
                        variables[mon_ident].upBound = variables[mon_ident].upBound / 1.5  # type: ignore

                        # If we found success, leave
                        if success:
                            break

            elif pulp.LpStatus[problem.status] == "Optimal":
                success = True

            # If we got success, we should record the solved speeds
            if success:
                for key in variables:
                    if key in self._battle.opponent_team and variables[key].varValue:
                        # Get speed and adjust if choicescarf, since it's 1.5 what we think; the solved
                        # speed is the choicescarf speed
                        spe = variables[key].varValue
                        if choicescarfmon == key and spe is not None:
                            spe = spe / 1.5

                        # Update inferences
                        if problem.name == "MinProblem":
                            speeds = self._inferences.get_flag(key, "spe")
                            speeds[0] = math.ceil(spe)  # type: ignore
                            self._inferences.set_flag(key, "spe", speeds)
                        else:
                            speeds = self._inferences.get_flag(key, "spe")
                            speeds[1] = math.ceil(spe)  # type: ignore
                            self._inferences.set_flag(key, "spe", speeds)

    # We look at abilities that don't have priority and trigger, found here:
    # https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts
    # If the triggered ability is weather or field, we then look for strings of other
    # abilities/items that activate with that ability as their own order.
    def _parse_preturn_switch(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, Optional[float]]]]:
        orders: List[List[Tuple[str, Optional[float]]]] = []
        last_moved = None
        last_multipliers = None
        i = 0

        # preturn_switch can be infinite if you switch in mons and they die (edge-case). Example:
        # ['', 'switch', 'p1a: Smeargle', 'Smeargle, L50, M, tera:Rock', '32/130'],
        # ['', '-damage', 'p1a: Smeargle', '0 fnt', '[from] Spikes'],
        # ['', 'faint', 'p1a: Smeargle'],
        # ['', ''],
        # ['', 'switch', 'p1a: Raichu', 'Raichu, L50, F', '60/135'],
        # ['', '-damage', 'p1a: Raichu', '27/135', '[from] Spikes']
        while i < len(events):
            # If we hit the edge-case above, we should go until we start again
            while i < len(events) and events[i][1] not in ["switch"]:
                update_battle(self._battle, events[i])
                i += 1

            # Go through the switches, since we can't actually anything from switches
            while i < len(events) and events[i][1] in ["switch"]:
                update_battle(self._battle, events[i])
                i += 1

            # Go through priority abilities; assumes detailschange and heals are from priority
            # activationws
            while i < len(events) and (
                events[i][-1].replace("ability: ", "") in PRIORITY_ACTIVATION_ABILITIES
                or events[i][1] == "detailschange"
                or events[i][1] == "-heal"
            ):
                ability, mon_ident = get_ability_and_identifier(events[i])
                if ability:
                    if last_moved and last_multipliers and mon_ident:
                        orders.append(
                            [
                                (last_moved, last_multipliers[last_moved]),
                                (mon_ident, last_multipliers[mon_ident]),
                            ]
                        )

                    last_moved = mon_ident
                    last_multipliers = self._save_multipliers()

                if events[i][1] in ["-transform", "detailschange"]:
                    last_moved, last_multipliers = None, None
                    orders = self.scrub_orders(  # type: ignore
                        self.clean_orders(orders), standardize_pokemon_ident(events[i][2])
                    )

                    self._orders = self.scrub_orders(
                        self._orders, standardize_pokemon_ident(events[i][2])
                    )
                update_battle(self._battle, events[i])
                i += 1

            # Now we look for abilities, both priority abilities and regular abilities
            last_moved, last_multipliers = None, None
            while i < len(events) and (
                is_ability_event(events[i])
                or events[i][-1] in ITEMS_THAT_ACTIVATE_ON_SWITCH
            ):
                ability, mon_ident = None, None
                if is_ability_event(events[i]):
                    ability, mon_ident = get_ability_and_identifier(events[i])
                else:
                    mon_ident = standardize_pokemon_ident(events[i][2])

                # We ignore booster energy because it has it's own lower trigger priority
                # that I can't figure out
                if (
                    last_moved
                    and last_multipliers
                    and mon_ident
                    and events[i][-1] != "Booster Energy"
                ):
                    orders.append(
                        [
                            (last_moved, last_multipliers[last_moved]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                last_moved = mon_ident
                last_multipliers = self._save_multipliers()

                # If the ability triggers a field or a weather, this could trigger both abilities and items
                if ability in ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
                    activations, num_traversed = (
                        self._get_activations_from_weather_or_terrain(events, i)
                    )
                    if len(activations) > 0:
                        orders += activations
                    i += num_traversed
                else:
                    # If there is a detailschange or a transform, the pokemon's speed is changed.
                    # Because of this, we need to make sure we don't record any of the previous
                    # speed interaction in this preturn switch since the speeds may be the different.
                    # We also do this for previous turns.
                    if events[i][1] in ["-transform", "detailschange"]:
                        last_moved, last_multipliers = None, None
                        orders = self.scrub_orders(  # type: ignore
                            self.clean_orders(orders),
                            standardize_pokemon_ident(events[i][2]),
                        )

                        self._orders = self.scrub_orders(
                            self._orders, standardize_pokemon_ident(events[i][2])
                        )

                    update_battle(self._battle, events[i])

                i += 1

        return orders

    def _parse_battle_mechanic(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, Optional[float]]]]:
        orders: List[List[Tuple[str, Optional[float]]]] = []
        last_moved, last_multipliers = None, None
        i = 0

        while i < len(events):
            if events[i][1] in ["-terastallize", "-dynamax", "-mega"]:
                mon_ident = standardize_pokemon_ident(events[i][2])

                if last_moved and last_multipliers:
                    orders.append(
                        [
                            (last_moved, last_multipliers[last_moved]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                last_moved = mon_ident
                last_multipliers = self._save_multipliers()

                # If we hit a battle mechanic that can activate abilities or items (eg alakazite -> trace)
                # we go until we find the weather, then we update it and track activations from it
                if (
                    events[i][1] == "-mega"
                    and events[i][-1]
                    in MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS
                ):
                    activations, num_traversed = (
                        self._get_activations_from_weather_or_terrain(events, i)
                    )
                    if len(activations) > 0:
                        orders.extend(activations)
                    i += num_traversed

            # Update battle after the order calculation
            update_battle(self._battle, events[i])

            i += 1

        return orders

    # Right now, we only look at orders of moves, and not at things that can activate or
    # trigger because of them, just due to the sheer complexity of VGC (eg bulldoze boost
    # triggering) Assumes abilities that affect priority.  Logic to find priority sourced
    # from here: https://bulbapedia.bulbagarden.net/wiki/Priority
    def _parse_move(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, Optional[float]]]]:
        priority_orders: List[List[Tuple[str, Optional[float]]]] = []
        last_moved = None
        last_priority = None
        last_multipliers = {}
        temp_orders: List[List[Tuple[str, Optional[float]]]] = []

        for i, event in enumerate(events):
            # We only pay attention to moves; we don't consider if a pokemon can't move because
            # we don't know what move they chose. Eventually, we should include if WE moved, since
            # we should know what move we chose, and what the priority would have been
            if event[1] == "move":
                # Now we need to get the priority
                mon_ident, priority = get_priority_and_identifier(event, self._battle)

                # We can't find a priority, it means we can skip and not record anything
                # since a priority of None means we can't interpret the priority. We don't want
                # to reset the priority though, because of abilities like Dancer which don't
                # disturb other moves in the same priority bracket
                if priority is None:
                    pass

                # If we're in a new priority bracket, record everyone's multipliers and this move
                # If we had a temp order, we need to reset it because we don't know if the priority
                # was the slowest in the last priority, or first in this new priority
                elif last_priority != priority:
                    last_priority = priority
                    last_moved = mon_ident
                    temp_orders = []
                    last_multipliers = self._save_multipliers()

                # We are in the same priority bracket as the last recorded segment, so we can add a tuple
                # We record the last multipliers, since now we know that in the past, these two mons
                # were compared against each other
                else:
                    # This means that our temp_order from a "can't move" is sandwiched between
                    # two orders with same priority, guaranteeing that it can be compared against
                    # the other mons moving before and after it
                    if len(temp_orders) >= 0:
                        priority_orders += temp_orders
                        temp_orders = []

                    if last_moved and last_multipliers:
                        priority_orders.append(
                            [
                                (last_moved, last_multipliers[last_moved]),
                                (mon_ident, last_multipliers[mon_ident]),
                            ]
                        )

                    # Now update our tracking
                    last_priority = priority
                    last_moved = mon_ident
                    last_multipliers = self._save_multipliers()

            # If there's an event where a mon is sleeping or paralysis or hurt itself from
            # confusion, if it's an opponent mon, I don't know what move they used, and so what
            # priority they would be acting in. I will only know for sure if this event is
            # sandwiched between two moves that I know are the same priority
            # We have to also be careful of "cants" that are spawned due to ability events, like:
            # ['', 'cant', 'p2b: Farigiraf', 'ability: Armor Tail', 'Fake Out', '[of] p1a: Rillaboom']
            # TODO: right now, I don't store what move I chose, and so can't do the above
            # calculations properly. For now, I treat even my own mon's failure to move as a mystery
            elif (
                event[1] == "cant" or event[-1] == "[from] confusion"
            ) and not is_ability_event(event):
                # If there was a move before this
                if last_priority is not None and last_multipliers and last_moved:
                    mon_ident = standardize_pokemon_ident(event[2])

                    # Record the temp
                    temp_orders.append(
                        [
                            (last_moved, last_multipliers[last_moved]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                    # Now update our tracking
                    last_moved = mon_ident
                    last_multipliers = self._save_multipliers()

            # Update battle after calculation, since the calculation happens before we make a mvove
            update_battle(self._battle, event)

        return priority_orders

    # Order that resolutions parse according to: https://www.smogon.com/forums/threads/sword-shield-battle-mechanics-research.3655528/page-64#post-9244179
    # True order can be found in PS code, under onResidualOrder
    # block: Sand damage, Hail, Solar Power, Dry Skin, Ice Body, Rain Dish
    # block: Grassy Terrain, Healer, Hydration, Shed Skin, Black Sludge, Black Sludge, Leftovers
    # each one resolves, in this order: Aqua Ring, Ingrain, Leech Seed, Poison, Burn, Curse,
    # Binding moves, Octolock, Salt Cure, Taunt, Torment, Encore, Disable, Magnet Rise, Yawn, Perish count
    # block: Uproar,CudChew, Harvest, Moody, Slow Start, Speed Boost, Flame Orb, Sticky Barb, Toxic Orb, White Herb
    def _parse_residual(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, Optional[float]]]]:
        orders: Dict[str, List[List[Tuple[str, Optional[float]]]]] = {}
        last_moved, last_key, last_multipliers = None, None, {}

        # Go through each event
        for i, event in enumerate(events):
            residual, mon_ident = get_residual_and_identifier(event)
            if residual is None or mon_ident is None:
                update_battle(self._battle, event)
                continue

            key = None
            if residual in FIRST_BLOCK_RESIDUALS:
                key = "First Block"
            elif residual in SECOND_BLOCK_RESIDUALS:
                key = "Second Block"
            elif residual in THIRD_BLOCK_RESIDUALS:
                key = "Third Block"
            else:
                key = residual

            # We're in a new block/residual, so we don't record anything, and just update tracking
            if key != last_key:
                last_key = key
                last_moved = mon_ident
                last_multipliers = self._save_multipliers()

            else:
                if key not in orders:
                    orders[key] = []

                if last_moved and last_multipliers:
                    orders[key].append(
                        [
                            (last_moved, last_multipliers[last_moved]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                last_moved = mon_ident
                last_multipliers = self._save_multipliers()

            # Update battle after we process the order, since the calc happens before the event
            update_battle(self._battle, event)

        return [o for order in orders.values() for o in order]

    # Even at beginning of turns, abilities activate on switch
    def _parse_switch(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, Optional[float]]]]:
        assert (
            self._battle.opponent_role is not None
        ), "Battle must be set before solving speeds"
        last_switched = None
        last_multipliers: Dict[str, Optional[float]] = {}
        speed_orders: List[List[Tuple[str, Optional[float]]]] = []
        i = 0

        # We need to store these to understand what was there before the switch
        actives, opp_actives, info = None, None, {}
        if isinstance(self._battle, DoubleBattle):
            actives = [mon for mon in self._battle.active_pokemon]
            opp_actives = [mon for mon in self._battle.opponent_active_pokemon]

            # I have to store the info of the mons before they switch, lest temporary effects get erased after they
            # switch, and then we record the wrong multipliers
            for mon in actives + opp_actives:
                if mon:
                    info[mon] = {
                        "effects": {k: v for k, v in mon.effects.items()},
                        "ability": mon.ability,
                        "speed_boost": float(mon.boosts["spe"]),
                    }

        else:
            active = self._battle.active_pokemon
            opp_active = self._battle.opponent_active_pokemon
            for mon in [active, opp_active]:
                if mon:
                    info[mon] = {
                        "effects": {k: v for k, v in mon.effects.items()},
                        "ability": mon.ability,
                        "speed_boost": float(mon.boosts["spe"]),
                    }

        while i < len(events):
            update_battle(self._battle, events[i])

            # First we look at switches
            if events[i][1] == "switch":
                # Get mons on the right side before the switch
                old_actives = (
                    actives
                    if events[i][2][:2] == self._battle.player_role
                    else opp_actives
                )

                # Now get the mon_ident of the mon that used to be there before we switched
                mon = None
                if isinstance(old_actives, list):
                    index = 0 if events[i][2][:3] in ["p1a", "p2a"] else 1
                    mon = old_actives[index]
                else:
                    mon = old_actives

                # We can't find the mon that we're switching for
                if not mon:
                    raise ValueError(
                        f"We can't find a mon in our actives:\n{actives}\n"
                        + f"or opponent actives:\n{opp_actives}\nfor the switch {events[i]}. "
                        + "This shouldn't happen!",
                        self._battle.active_pokemon,
                        self._battle.opponent_active_pokemon,
                        self._battle.team,
                        self._battle.opponent_team,
                        events,
                    )

                # We grab the mon_ident, but without position information
                mon_ident = mon.identifier(events[i][2][:2])

                # If we have a previous switch, let's add it to speed_orders
                if last_switched is not None:
                    speed_orders.append(
                        [
                            (last_switched, last_multipliers[last_switched]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                # Regardless, we'll update what we found this time around
                last_switched = mon_ident
                last_multipliers = {}
                for ident, mon in self._battle.team.items():
                    last_multipliers[ident] = self._generate_multiplier(
                        ident,
                        override_ability=info[mon]["ability"] if mon in info else None,  # type: ignore
                        override_effects=info[mon]["effects"] if mon in info else None,  # type: ignore
                        override_speed_boost=(
                            info[mon]["speed_boost"] if mon in info else None  # type: ignore
                        ),
                    )

                # Go through opponents mons and save speed multipliers, but using the ones at the beginning of the turn
                for mon in self._battle.opponent_team.values():
                    ident = mon.identifier(self._battle.opponent_role)
                    last_multipliers[ident] = self._generate_multiplier(
                        ident,
                        override_ability=info[mon]["ability"] if mon in info else None,  # type: ignore
                        override_effects=info[mon]["effects"] if mon in info else None,  # type: ignore
                        override_speed_boost=(
                            info[mon]["speed_boost"] if mon in info else None  # type: ignore
                        ),
                    )

                for mon in self._battle.teampreview_opponent_team:
                    ident = mon.identifier(self._battle.opponent_role)
                    if ident not in last_multipliers:
                        last_multipliers[ident] = 1.0

            # Check if we have an ability event activated by the switch
            elif is_ability_event(events[i]):
                ability, _ = get_ability_and_identifier(events[i])

                # If the ability triggers a field or a weather, this could trigger both abilities and items.
                # First we update our observation class, and then we get activated orders of events.
                # This method also iterates through obs.events, so we continue afterwards to not iterate twice
                if ability in ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
                    order, num_traversed = self._get_activations_from_weather_or_terrain(
                        events, i
                    )
                    if len(order) > 0:
                        speed_orders += order
                    i += num_traversed

            i += 1

        return speed_orders

    # Continue iterating through events that may have been triggered by weather_or_terrain
    # The paramter `i` that's passed should be the event that is the activator (eg terrain or weather)
    # It should end exactly on the last event we've found
    def _get_activations_from_weather_or_terrain(
        self, events: List[List[str]], i: int
    ) -> Tuple[List[List[Tuple[str, Optional[float]]]], int]:
        # Ensure we were called correctly
        ability, mon_ident = get_ability_and_identifier(events[i])
        if ability not in ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
            raise ValueError(
                "_get_activations_from_weather_or_terrain was not passed the right parameter."
                + "it should always be passed the index of the event that could possibly activate "
                + "other abilities or items",
                self._battle,
            )

        # Set up tracking variables
        last_moved, last_multipliers = None, None
        orders: List[List[Tuple[str, Optional[float]]]] = []

        # Process and save where we started, and take a first step to look at what's next
        start = i
        update_battle(self._battle, events[i])
        i += 1

        while i < len(events) and events[i][1] in [
            "-activate",
            "-enditem",
            "-start",
            "-boost",
        ]:
            # Check to see if we're done with activations. If we see a Booster Energy activation, it means we're at the item
            # activation phase at the beginning of a turn and we should be done. Example activations:
            # ['', '-activate', 'p1b: Iron Hands', 'ability: Quark Drive']
            # ['', '-start', 'p1b: Iron Hands', 'quarkdriveatk']
            # ['', '-enditem', 'p2a: Hawlucha', 'Electric Seed']
            # ['', '-boost', 'p2a: Hawlucha', 'def', '1', '[from] item: Electric Seed']
            # We also stop if we're activating another independent ability
            if (
                events[i][1] not in ["-activate", "-enditem", "-start", "-boost"]
                or events[i][-1] == "Booster Energy"
                or get_ability_and_identifier(events[i])[0]
                in ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS
            ):
                break

            # We have to skip sources because of Orichalcum pulse:
            # ['', '-activate', 'p2a: Koraidon', 'Orichalcum Pulse', '[source]']
            if events[i][1] in ["-enditem", "-activate"] and events[i][-1] != "[source]":
                mon_ident = standardize_pokemon_ident(events[i][2])

                if last_moved is not None and last_multipliers is not None:
                    orders.append(
                        [
                            (last_moved, last_multipliers[last_moved]),
                            (mon_ident, last_multipliers[mon_ident]),
                        ]
                    )

                last_moved = mon_ident
                last_multipliers = self._save_multipliers()

            update_battle(self._battle, events[i])
            i += 1

        # We need to end on the exact last event we looked at. Because we go one further to see when to end,
        # we need to subtract a single additional iteration
        num_traversed = i - start - 1

        return orders, num_traversed

    # Method to generate a multiplier for a mon_ident given a self._battle
    def _generate_multiplier(
        self,
        mon_ident: str,
        override_ability: Optional[str] = None,
        override_effects: Optional[Dict[Effect, int]] = None,
        override_speed_boost: Optional[int] = None,
    ) -> Optional[float]:
        mon = self._battle.get_pokemon(mon_ident)
        sc = (
            self._battle.side_conditions
            if mon_ident.startswith(self._battle.player_role or "p1")
            else self._battle.opponent_side_conditions
        )

        # We need to override item if we previously found an item, because we only
        # store it in the inferences object; we don't update the battle object
        item = mon.item
        if item in [None, GenData.UNKNOWN_ITEM] and self._inferences.is_tracking(
            mon_ident
        ):
            item = self._inferences.get_flag(mon_ident, "item")

        return self.get_speed_multiplier(
            mon=mon,
            weathers=self._battle.weather,
            side_conditions=sc,
            fields=self._battle.fields,
            speed_boosts=(
                override_speed_boost if override_speed_boost else mon.boosts["spe"]
            ),
            item=item,
            ability=override_ability if override_ability else mon.ability,
            status=mon.status,
            effects=override_effects if override_effects else mon.effects,
        )

    # Supposed to be called at speed-order decision time (when we decide who goes next); this function
    # records all the speed multipliers that are at this moment so we can get what the calculation was
    def _save_multipliers(self) -> Dict[str, Optional[float]]:
        assert (
            self._battle.opponent_role is not None and self._battle.player_role is not None
        ), "Battle must be set before solving speeds"

        multipliers: Dict[str, Optional[float]] = {}
        if isinstance(self._battle, DoubleBattle):
            # Use internal variable because it already stores idents and also doesn't
            # make the mons null if they fainted. If my mon uses an attack, life orb dies
            # and then another mon moves, we can still use my mon's info to ascertain something
            # about the opponent's speed
            for mon in self._battle._active_pokemon.values():
                key = mon.identifier(self._battle.player_role)
                multipliers[key] = self._generate_multiplier(key)

            for mon in self._battle._opponent_active_pokemon.values():
                key = mon.identifier(self._battle.opponent_role)
                multipliers[key] = self._generate_multiplier(key)

        else:
            raise NotImplementedError("Not implemented for single battles")

        return multipliers

    # Updates self._orders when we find an item; we recorded them with the multipliers we thought they had,
    # but we were wrong, and they should have had a different multiplier based on their item. We only call
    # this method once we find out the item for sure; then we go back and adjust the multipliers to what
    # they should have been
    def _update_orders(self, mon_ident: str, mult: float):
        for order in self._orders:
            for i, tup in enumerate(order):
                if tup[0] == mon_ident:
                    order[i] = (mon_ident, tup[1] * mult)

    # Scrubs orders of a mon if previous orders are bad from a speedchange (eg transform,
    # detailschange, speedswap, etc)
    @staticmethod
    def scrub_orders(
        orders: List[List[Tuple[str, float]]], ident: str
    ) -> List[List[Tuple[str, float]]]:
        new_orders = []
        # First, we do linking (so if A > B and B > C, then A > C)
        # o1 = [('p1: Terapagos', 1.0), ('p2: Terapagos', 1.0)]
        for o1 in orders:
            for o2 in orders:
                # Need to check speed multipliers too
                if (
                    o1[1][0] == o2[0][0]
                    and o1[1][1] is not None
                    and o2[0][1] is not None
                    and o1[1][1] >= o2[0][1]
                ):
                    new_orders.append([o1[0], o2[1]])

        # Scrub all orders that contain an ident
        new_orders = list(
            filter(lambda x: x[0][0] != ident and x[1][0] != ident, new_orders + orders)
        )
        return new_orders

    @staticmethod
    def clean_orders(
        orders: List[List[Tuple[str, Optional[float]]]],
    ) -> List[List[Tuple[str, float]]]:
        return list(
            filter(
                lambda x: len(x) == 2 and x[0][1] is not None and x[1][1] is not None,  # type: ignore
                orders,
            )
        )  # pyright: ignore

    # Gets speed multipliers that apply to a pokemon. If a mon's position in the speed bracket has been affected (eg
    # via quash) we return None so that we know not to account for this mon's speed when deducing others'
    @staticmethod
    def get_speed_multiplier(
        mon: Optional[Pokemon] = None,
        weathers: Dict[Weather, int] = {},
        side_conditions: Dict[SideCondition, int] = {},
        fields: Dict[Field, int] = {},
        speed_boosts: int = 0,
        item: Optional[str] = None,
        ability: Optional[str] = None,
        status: Optional[Status] = None,
        effects: Optional[Dict[Effect, int]] = None,
    ) -> Optional[float]:
        # We should ignore speed calculations for mons whose place in the speed bracket has been affected
        if effects and (
            Effect.AFTER_YOU in effects
            or Effect.QUASH in effects
            or Effect.AFTER_YOU in effects
            or Effect.QUICK_CLAW in effects
            or Effect.CUSTAP_BERRY in effects
            or Effect.DANCER in effects
            or Effect.QUICK_DRAW in effects
        ):
            return None
        elif item in ["fullincense", "laggingtail"]:
            return None

        multiplier = 1.0

        # Check Tailwind
        if SideCondition.TAILWIND in side_conditions:
            multiplier *= 2

        # Check Grass Pledge (Swamp)
        if SideCondition.GRASS_PLEDGE in side_conditions:
            multiplier *= 0.5

        # Check Trick Room; speeds are reversed
        if Field.TRICK_ROOM in fields:
            multiplier *= -1

        # Check boosts
        if speed_boosts == 1:
            multiplier *= 1.5
        elif speed_boosts == 2:
            multiplier *= 2
        elif speed_boosts == 3:
            multiplier *= 2.5
        elif speed_boosts == 4:
            multiplier *= 3
        elif speed_boosts == 5:
            multiplier *= 3.5
        elif speed_boosts == 6:
            multiplier *= 4
        elif speed_boosts == -1:
            multiplier *= 2.0 / 3
        elif speed_boosts == -2:
            multiplier *= 2.0 / 4
        elif speed_boosts == -3:
            multiplier *= 2.0 / 5
        elif speed_boosts == -4:
            multiplier *= 2.0 / 6
        elif speed_boosts == -5:
            multiplier *= 2.0 / 7
        elif speed_boosts == -6:
            multiplier *= 2.0 / 8

        # Check items
        if item in [
            "ironball",
            "machobrace",
            "poweranklet",
            "powerband",
            "powerbelt",
            "powerbracer",
            "powerlens",
            "powerweight",
        ]:
            multiplier *= 0.5
        elif item == "choicescarf":
            multiplier *= 1.5
        elif item == "quickpowder" and mon is not None and mon.species == "ditto":
            multiplier *= 2

        # Check abilities
        if ability == "quickfeet" and status is not None:
            multiplier *= 1.5
        # Check status; have to do this w/ quickfeet check
        elif status == Status.PAR:
            multiplier *= 0.5

        # Check rest of abilities
        if (
            ability == "unburden"
            or (
                ability is None
                and mon is not None
                and "unburden" in mon.possible_abilities
            )
        ) and item is None:
            multiplier *= 2  # Assume mon lost item if this is the case
        elif (
            ability == "sandrush"
            or (
                ability is None
                and mon is not None
                and "sandrush" in mon.possible_abilities
            )
        ) and Weather.SANDSTORM in weathers:
            multiplier *= 2
        elif (
            ability == "swiftswim"
            or (
                ability is None
                and mon is not None
                and "swiftswim" in mon.possible_abilities
            )
        ) and Weather.RAINDANCE in weathers:
            multiplier *= 2
        elif (
            ability == "slushrush"
            or (
                ability is None
                and mon is not None
                and "slushrush" in mon.possible_abilities
            )
        ) and Weather.SNOW in weathers:
            multiplier *= 2
        elif (
            ability == "slushrush"
            or (
                ability is None
                and mon is not None
                and "slushrush" in mon.possible_abilities
            )
        ) and Weather.HAIL in weathers:
            multiplier *= 2
        elif (
            ability == "chlorophyll"
            or (
                ability is None
                and mon is not None
                and "chlorophyll" in mon.possible_abilities
            )
        ) and Weather.SUNNYDAY in weathers:
            multiplier *= 2

        if effects and Effect.SLOW_START in effects:
            multiplier *= 0.5
        elif effects and Effect.PROTOSYNTHESISSPE in effects:
            multiplier *= 1.5
        elif effects and Effect.QUARKDRIVESPE in effects:
            multiplier *= 1.5
        elif ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in fields:
            multiplier *= 2

        return multiplier
