# -*- coding: utf-8 -*-
"""This module makes speed inferences throughout a battle by tracking pokemons' moves, stats and items.
It even can even establish whether a mon has choice scarf. This object is a companion class to BattleInference
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pulp
from poke_env.environment import (
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Observation,
    SideCondition,
    Status,
    Weather,
)

from elitefurretai.inference.inference_utils import (
    ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    FIRST_BLOCK_RESIDUALS,
    ITEMS_THAT_ACTIVATE_ON_SWITCH,
    MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    PRIORITY_ACTIVATION_ABILITIES,
    SECOND_BLOCK_RESIDUALS,
    THIRD_BLOCK_RESIDUALS,
    copy_battle,
    get_ability_and_identifier,
    get_pokemon,
    get_priority_and_identifier,
    get_residual_and_identifier,
    get_segments,
    get_showdown_identifier,
    is_ability_event,
    print_battle,
    standardize_pokemon_ident,
    update_battle,
)


class SpeedInference:
    __slots__ = (
        "_battle",
        "_opponent_mons",
        "_orders",
        "_v",
    )

    # We share flags from opponent mons with BattleInference, and we need to keep a copy of the battle state
    # because we need to recreate the turn and conditions of the turn to parse speeds
    def __init__(
        self,
        battle: Union[Battle, DoubleBattle],
        opponent_mons: Dict[str, Any],
        verbose: int = 0,
    ):
        self._opponent_mons: Dict[str, Any] = opponent_mons
        self._battle = copy_battle(battle)
        self._orders: List[List[Tuple[str, float]]] = []
        self._v = verbose

    def check_speed(self, obs: Observation):
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
        We store these in the flags we share with BattleInference.

        Right now, we assume abilities that affect speeds/priorities, we don't look at every interaction of moves, and only try to
        learn choice scarfs (not iron ball or lagging tail).
        """
        speed_orders: List[List[Tuple[str, Optional[float]]]] = []

        segments = get_segments(obs.events)

        if "init" in segments:
            for event in segments["init"]:
                update_battle(self._battle, event)

        if self._v >= 2:
            self._debug("beginning_of_turn_state")

        if self._v >= 3:
            self._debug("segments", segments)

        # I don't parse the order of activations (right now: Quick Claw and Quick Draw) because of the nicheness
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

        # Clean the orders by making them into tuplets and removing duplicates
        self._solve_speeds(self.clean_orders(speed_orders))

    # Uses pulp to solve for speeds. We create the set of equations and try to solve them. If we get an infeasible
    # set of equations, this means that a mon has choice scarf (or SpeedInference's parsing is incomplete/buggy).
    # If we find infeasibility, we multiply each mon's speed by 1.5 if they are choicescarf eligible to see if we
    # can find a speed that makes the set of equations work. If it's only one mon, we know it's choicescarf! If
    # we find an optimal solution, we set the new speed bounds.

    # TODO: Need to check which mon is choice scarf
    def _solve_speeds(self, orders: List[List[Tuple[str, float]]]):

        # Add all our orders to full set of orders we've observed in the battle
        self._orders.extend(orders)

        if self._v >= 3:
            self._debug("orders", orders)

        # Create a LP problems
        problems = [
            pulp.LpProblem("MinProblem", pulp.LpMinimize),
            pulp.LpProblem("MaxProblem", pulp.LpMaximize),
        ]

        # Create the variables we're trying to solve for, which are opponent mons' speeds
        variables: Dict[str, pulp.LpVariable] = {}
        for mon in self._battle.teampreview_opponent_team:
            key = get_showdown_identifier(mon, self._battle.opponent_role)
            variables[key] = pulp.LpVariable(
                key,
                lowBound=self._opponent_mons[key]["spe"][0],
                upBound=self._opponent_mons[key]["spe"][1],
                cat="Integer",
            )

        # We record the constraints of our mons as their actual speeds, because we know their true value
        constraints: Dict[str, float] = {}
        for key, mon in self._battle.team.items():
            if mon.stats["spe"]:
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

            if self._v >= 2 and problem.name == "MinProblem":
                self._debug("lpproblem", problem)

            # Solve our set of linear equations (using an arbitrary solver, with no command line output)
            problem.solve(solver=pulp.apis.SCIP_PY(msg=False))

            # Flag for whether we can update our speed tracking
            success = False

            # If we get infeasible conditions, we check for Choice Scarf (we don't check for iron ball given its rarity)
            # This assumes that our speed tracking is correct; we could get infeasible if I miss a speed interaction
            if pulp.LpStatus[problem.status] == "Infeasible":

                if self._v >= 1 and problem.name == "MinProblem":
                    self._debug("infeasible")

                # Loop through every mon that could have choice scarf and test if the problem can be solved by assuming one
                # We can have a situation where multiple mons could have a choice scarf and the set of equations works
                # so we only claim success
                for mon_ident in self._opponent_mons:
                    if self._v >= 3 and problem.name == "MinProblem":
                        self._debug("item_testing", mon_ident)

                    if (
                        self._opponent_mons[mon_ident]["can_be_choice"]
                        and variables[mon_ident].upBound
                    ):
                        variables[mon_ident].upBound = int(variables[mon_ident].upBound * 1.5)  # type: ignore
                        problem.solve(solver=pulp.apis.SCIP_PY(msg=False))

                        # We found our choice scarf mon! We should record success and update the mon
                        if pulp.LpStatus[problem.status] == "Optimal":
                            self._update_orders(mon_ident, 1.5)
                            self._battle.opponent_team[mon_ident].item = "choicescarf"
                            self._opponent_mons[mon_ident]["item"] = "choicescarf"
                            success = True
                            if self._v >= 2 and problem.name == "MinProblem":
                                self._debug("item_found", mon_ident)
                            break

                        variables[mon_ident].upBound = int(variables[mon_ident].upBound / 1.5)  # type: ignore

                if self._v >= 1 and not success and problem.name == "MinProblem":
                    self._debug("item_not_found", problem)

            elif pulp.LpStatus[problem.status] == "Optimal":
                success = True
                if self._v >= 3 and problem.name == "MinProblem":
                    self._debug("solution_found")
            else:
                raise ValueError(
                    f"Got Pulp status of {pulp.LpStatus[problem.status]}. Pulp setup: {problem}",
                    self._battle,
                )

            # If we got success, we should record the solved speeds
            if success:
                for key in variables:
                    if key in self._opponent_mons and variables[key].varValue:
                        if problem.name == "MinProblem":
                            self._opponent_mons[key]["spe"][0] = variables[key].varValue
                        else:
                            self._opponent_mons[key]["spe"][1] = variables[key].varValue

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

            # Go through priority abilities
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

                if last_moved and last_multipliers and mon_ident:
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
        last_moved, last_priority, last_multipliers, temp_orders = None, None, {}, []

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

                    # TODO: bug here; im pretty sure we're not keeping active pokemon up-to-date
                    # right, which is messing with save_multipliers, stored in last_multipliers
                    if last_moved and last_multipliers:
                        try:
                            priority_orders.append(
                                [
                                    (last_moved, last_multipliers[last_moved]),
                                    (mon_ident, last_multipliers[mon_ident]),
                                ]
                            )
                        except KeyError as e:

                            print("In Last Multipliers and had a Key Error")
                            print(e)
                            print("Last Battle:")
                            print(print_battle(self._battle))

                    # Now update our tracking
                    last_priority = priority
                    last_moved = mon_ident
                    last_multipliers = self._save_multipliers()

            # If there's an event where a mon is sleeping or paralysis or hurt itself from
            # confusion, if it's an opponent mon, I don't know what move they used, and so what
            # priority they would be acting in. I will only know for sure if this event is
            # sandwiched between two moves that I know are the same priority
            # TODO: right now, I don't store what move I chose, and so can't do the above
            # calculations properly. For now, I treat even my own mon's failure to move as a mystery
            elif event[1] == "cant" or event[-1] == "[from] confusion":

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
        last_switched, last_multipliers = None, {}
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
            actives = self._battle.active_pokemon
            opp_actives = self._battle.opponent_active_pokemon
            for mon in [actives, opp_actives]:
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
                        f"We can't find a mon in our actives: {actives} "
                        + f"or opponent actives: {opp_actives} for the switch {events[i]}. "
                        + "This shouldn't happen!",
                        self._battle.active_pokemon,
                        self._battle.opponent_active_pokemon,
                        self._battle.team,
                        self._battle.opponent_team,
                        events,
                    )

                # We grab the mon_ident, but without position information
                mon_ident = get_showdown_identifier(mon, events[i][2][:2])

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
                for mon in self._battle.team.values():
                    ident = get_showdown_identifier(mon, self._battle.player_role)
                    last_multipliers[ident] = self._generate_multiplier(
                        ident,
                        override_ability=info[mon]["ability"] if mon in info else None,
                        override_effects=info[mon]["effects"] if mon in info else None,
                        override_speed_boost=(
                            info[mon]["speed_boost"] if mon in info else None
                        ),
                    )

                # Go through opponents mons and save speed multipliers, but using the ones at the beginning of the turn
                for mon in self._battle.opponent_team.values():
                    ident = get_showdown_identifier(mon, self._battle.opponent_role)
                    last_multipliers[ident] = self._generate_multiplier(
                        ident,
                        override_ability=info[mon]["ability"] if mon in info else None,
                        override_effects=info[mon]["effects"] if mon in info else None,
                        override_speed_boost=(
                            info[mon]["speed_boost"] if mon in info else None
                        ),
                    )

                for mon in self._battle.teampreview_opponent_team:
                    ident = get_showdown_identifier(mon, self._battle.opponent_role)
                    if ident not in last_multipliers:
                        last_multipliers[ident] = 1.0

            # Check if we have an ability event activated by the switch
            elif is_ability_event(events[i]):
                ability, mon_ident = get_ability_and_identifier(events[i])

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
            if (
                events[i][1] not in ["-activate", "-enditem", "-start", "-boost"]
                or events[i][-1] == "Booster Energy"
            ):
                break

            if events[i][1] in ["-enditem", "-activate"]:
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

        mon = get_pokemon(mon_ident, self._battle)
        sc = (
            self._battle.side_conditions
            if mon_ident.startswith(self._battle.player_role or "p1")
            else self._battle.opponent_side_conditions
        )

        return self.get_speed_multiplier(
            species=mon.species,
            weathers=self._battle.weather,
            side_conditions=sc,
            fields=self._battle.fields,
            speed_boosts=(
                override_speed_boost if override_speed_boost else mon.boosts["spe"]
            ),
            item=mon.item,
            ability=override_ability if override_ability else mon.ability,
            status=mon.status,
            effects=override_effects if override_effects else mon.effects,
        )

    # Supposed to be called at speed-order decision time (when we decide who goes next); this function
    # records all the speed multipliers that are at this moment so we can get what the calculation was
    def _save_multipliers(self) -> Dict[str, Optional[float]]:

        multipliers: Dict[str, Optional[float]] = {}
        if isinstance(self._battle, DoubleBattle):
            for mon in self._battle.active_pokemon:
                if mon is not None:
                    key = get_showdown_identifier(mon, self._battle.player_role)
                    multipliers[key] = self._generate_multiplier(key)

            for mon in self._battle.opponent_active_pokemon:
                if mon is not None:
                    key = get_showdown_identifier(mon, self._battle.opponent_role)
                    multipliers[key] = self._generate_multiplier(key)

        else:
            mon = self._battle.active_pokemon
            if mon is not None:
                key = get_showdown_identifier(mon, self._battle.player_role)
                multipliers[key] = self._generate_multiplier(key)

            opp_mon = self._battle.opponent_active_pokemon
            if opp_mon is not None:
                opp_key = get_showdown_identifier(opp_mon, self._battle.opponent_role)
                multipliers[opp_key] = self._generate_multiplier(opp_key)

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

    # I use this horrendous code because debugging can be quite tricky with LP and Showdown; it can
    # be unclear whether the problem is with the parsing or the LP, and this allows for more easy future
    # readability of the main bulk of code, given we will likely have to revisit in future generations
    def _debug(self, key: str, arg: Any = None):
        if key == "beginning_of_turn_state":
            print(
                f"\nTurn # {self._battle.turn} in SpeedInference is starting with the following conditions:"
            )
            if isinstance(self._battle.active_pokemon, list):
                print(
                    f"\tMy Active Mon:  {list(map(lambda x: x.name if x else 'None', self._battle.active_pokemon))}"
                )
            else:
                print(
                    f"\tMy Active Mon:  {self._battle.active_pokemon.name if self._battle.active_pokemon else 'None'}"
                )
            if isinstance(self._battle.opponent_active_pokemon, list):
                print(
                    f"\tOpp Active Mon: {list(map(lambda x: x.name if x else 'None', self._battle.opponent_active_pokemon))}"
                )
            else:
                print(
                    f"\tOpp Active Mon: {self._battle.opponent_active_pokemon.name if self._battle.opponent_active_pokemon else 'None'}"
                )
            print(
                f"\tMy Side Conditions:  {list(map(lambda x: x.name, self._battle.side_conditions))}"
            )
            print(
                f"\tOpp Side Conditions: {list(map(lambda x: x.name, self._battle.opponent_side_conditions))}"
            )
            print(f"\tWeather: {list(map(lambda x: x.name, self._battle.weather))}")
            print(f"\tFields: {list(map(lambda x: x.name, self._battle.fields))}")
            print("\tMy Team:")
            for mon in self._battle.team.values():
                print(
                    f"\t\t{mon.name} => [Speed: {mon.stats['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"
                )
            print("\tOpp Team:")
            for mon in self._battle.teampreview_opponent_team:
                ident = get_showdown_identifier(mon, self._battle.opponent_role)
                if ident in self._battle.opponent_team:
                    mon = self._battle.opponent_team[ident]
                print(
                    f"\t\t{mon.name} => [Speed Range: {self._opponent_mons[ident]['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"
                )
            print()
        elif key == "segments":
            print("Segmented Events:")
            for segment in arg:
                print(f"\t{arg}:")
                for event in arg[segment]:
                    print(f"\t\t{event}")
            print()
        elif key == "orders":
            print("Orders from this turn:")
            for order in arg:
                print(f"\t{order}")
            print("\nThis makes total orders:", self._orders)
            print()
        elif key == "lpproblem":
            print(arg)
        elif key == "infeasible":
            print("Got an infeasible solution... looking for Choice Scarf...")
        elif key == "item_testing":
            print(
                f"Testing {arg} for Choice Scarf; Eligibility for Choice Items: {self._opponent_mons[arg]['can_be_choice']}"
            )
        elif key == "item_found":
            print(f"Found choice scarf mon: {arg}")
            print()
        elif key == "item_not_found":
            print("Couldn't find a mon with Choice Scarf...")
            print("Here is the LP problem:")
            print(arg)
            print()
        elif key == "solution_found":
            print("We found an optimal solution!")
            print()

    # Takes the list of move orders and prunes non-comparables, converts list of 3
    # or more into pairs, standardizes mon_ident and dedupes
    @staticmethod
    def clean_orders(
        orders: List[List[Tuple[str, Optional[float]]]]
    ) -> List[List[Tuple[str, float]]]:
        tuples = []

        for order in orders:
            for i in range(len(order)):
                if (
                    i < len(order) - 1
                    and order[i][1] is not None
                    and order[i + 1][1] is not None
                ):
                    tuples.append([order[i], order[i + 1]])

        cleaned_tuples = []
        for order in tuples:
            cleaned_tuples.append(
                list(
                    map(
                        lambda x: (
                            (
                                x[0][:2] + x[0][3:]
                                if not x[0].startswith("p1:")
                                and not x[0].startswith("p2:")
                                else x[0]
                            ),
                            x[1],
                        ),
                        order,
                    )
                )
            )

        return cleaned_tuples

    # Gets speed multipliers that apply to a pokemon. If a mon's position in the speed bracket has been affected (eg
    # via quash) we return None so that we know not to account for this mon's speed when deducing others'
    @staticmethod
    def get_speed_multiplier(
        species: Optional[str] = None,
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
        elif item == "quickpowder" and species == "ditto":
            multiplier *= 2

        # Check abilities
        if ability == "quickfeet" and status is not None:
            multiplier *= 1.5
        # Check status; have to do this w/ quickfeet check
        elif status == Status.PAR:
            multiplier *= 0.5

        # Check rest of abilities
        if ability == "unburden" and item is None:
            multiplier *= 2  # Assume mon lost item if this is the case
        elif ability == "sandrush" and Weather.SANDSTORM in weathers:
            multiplier *= 2
        elif ability == "swiftswim" and Weather.RAINDANCE in weathers:
            multiplier *= 2
        elif ability == "slushrush" and Weather.SNOW in weathers:
            multiplier *= 2
        elif ability == "slushrush" and Weather.HAIL in weathers:
            multiplier *= 2
        elif ability == "chlorophyll" and Weather.SUNNYDAY in weathers:
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
