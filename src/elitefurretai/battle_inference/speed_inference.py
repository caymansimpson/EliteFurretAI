# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items to make speed inferences throughout a battle
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

from elitefurretai.battle_inference.inference_utils import (
    _ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    _ITEMS_THAT_ACTIVATE_ON_SWITCH,
    _MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS,
    _PRIORITY_ACTIVATION_ABILITIES,
    FIRST_BLOCK_RESIDUALS,
    SECOND_BLOCK_RESIDUALS,
    THIRD_BLOCK_RESIDUALS,
    get_ability_and_identifier,
    get_pokemon,
    get_pokemon_ident,
    get_priority_and_identifier,
    get_residual_and_identifier,
    get_segments,
    is_ability_event,
    load_battle,
    update_battle,
)


class SpeedInference:

    def __init__(self, battle: Union[Battle, DoubleBattle], opponent_mons: Dict[str, Any]):
        self._opponent_mons: Dict[str, Any] = opponent_mons
        self._battle = load_battle(battle)

    # TODO: redo commentary
    def check_speed(self, obs: Observation):
        """
        The main method of this class. It creates bounds for speed by going through events in Observation

        At the start of the turn:
        - Pre-Abilities on when multiple pokemon switch in at the same time (beginning of turn/battle)
        - Regular Abilities on beginning of turn/battle,
        - Items on beginning of turn/battle

        Then during the turn:
        - Switches,
        - Tera/mega/dynamax,
        - Moves: by priority
        - Residuals

        Theoretically, it should be as simple as storing ability/switch/move priority/residual order with cumulative speed multipliers
        on each interaction (4 unknown variables, 4*turn equations) and then solving the equations with minimum/maximum total opponent
        speed stats? I think the complexity will just be in accurately capturing/identifying all the edge cases
        (like iron ball/prankster/chlorophyll/unburden etc) that affect speed without public messages from the engine
        """
        speed_orders: List[List[Tuple[str, float]]] = []

        segments = get_segments(obs.events)

        if "init" in segments:
            for event in segments["init"]:
                update_battle(self._battle, event)

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

        # TODO: passing obs and segments just for debugging
        self._solve_speeds(self.clean_orders(speed_orders), obs, segments)

    # Uses pulp to solve for speeds. TODO: add more commentary on what this function actually does, and why it's made this way
    def _solve_speeds(self, orders: List[List[Tuple[str, float]]], obs, segments):

        print("entering solver")

        # Create a LP problems
        problems = [
            pulp.LpProblem("MinProblem", pulp.LpMinimize),
            pulp.LpProblem("MaxProblem", pulp.LpMaximize),
        ]

        # Create the variables we're trying to solve for, which are opponent mons
        variables: Dict[str, Union[int, pulp.LpVariable]] = {}
        for mon in self._battle.teampreview_opponent_team:
            key = (
                self._battle.opponent_role + ": " + mon._data.pokedex[mon.species]["name"]
            )
            variables[key] = pulp.LpVariable(
                key,
                lowBound=self._opponent_mons[key]["spe"][0],
                upBound=self._opponent_mons[key]["spe"][1],
                cat="Integer",
            )

        # We record the variables of our mons as their actual speeds, because we know their true value
        for key, mon in self._battle.team.items():

            # TODO: this isnt triggering because I dont have proper stats storage in poke-env
            if mon.stats and mon.stats["spe"]:
                variables[key] = mon.stats["spe"]

            # TODO: remove once we fix stats storage in poke-env ; this is to help the program run; otherwise this will be None
            else:
                variables["p1: Wo-Chien"] = 90
                variables["p1: Tyranitar"] = 81
                variables["p1: Torkoal"] = 40
                variables["p1: Raichu"] = 130
                variables["p1: Smeargle"] = 127

        # TODO: for debugging; should remove after
        ground_truth = {
            "p2: Rillaboom": 150,
            "p2: Pelipper": 93,
            "p2: Pincurchin": 73,
            "p2: Cresselia": 150,
            "p2: Smeargle": 127,
        }

        # For each problem (a minimization and maximization one)
        for problem in problems:

            # This is our objective function; either maximize or minimize total speed stats
            # for opponent mons (this includes our mons; which is an int and invariant of solving)
            problem += pulp.lpSum([variables[key] for key in variables])

            # We add each order that we parsed, only if there's an opponent speed to parse
            for order in orders:
                first_mon, first_mult = order[0]
                second_mon, second_mult = order[1]

                if (
                    first_mon in self._battle.opponent_team
                    or second_mon in self._battle.opponent_team
                ):
                    problem += (
                        variables[first_mon] * first_mult
                        >= variables[second_mon] * second_mult
                    )

            # Solve our set of linear equations (using an arbitrary solver, with no command line output)
            problem.solve(pulp.PULP_CBC_CMD(msg=0))

            # TODO: will eventually replace with explorations on choice_scarf or slow items
            # or check assumptions made in _parse_move with abilities and priorities
            if pulp.LpStatus[problem.status] == "Infeasible":
                print(
                    f"Got Pulp status of {pulp.LpStatus[problem.status]}. Pulp setup: {problem}"
                )
                print("Ground truth:", ground_truth)
                print("Orders:", orders)
                print()
                print(
                    f"My Active Pokemon {[mon.species for mon in obs.active_pokemon if mon]}"
                )
                print(
                    f"Opp Active Pokemon {[mon.species for mon in obs.opponent_active_pokemon if mon]}"
                )
                print("My mons:")
                for s in [
                    f"\t{mon.species}: speed boost ({mon.boosts['spe']}) and effects ({mon.effects}) and status ({mon.status})"
                    for mon in obs.team.values()
                ]:
                    print(s)
                print("Opp mons:")
                for s in [
                    f"\t{mon.species}: speed boost ({mon.boosts['spe']}) and effects ({mon.effects}) and status ({mon.status})"
                    for mon in obs.opponent_team.values()
                ]:
                    print(s)
                print("Side Conditions:", obs.side_conditions)
                print("Opp Side Conditions:", obs.opponent_side_conditions)
                print("Fields:", obs.fields)
                print("Weather:", obs.weather)
                print("Events:", obs.events)
                print(segments)
                raise Exception
            elif pulp.LpStatus[problem.status] == "Optimal":

                # Store solved speeds in _opponent_mons
                for key in variables:
                    if key in self._opponent_mons and variables[key].varValue:
                        if problem.name == "MinProblem":
                            self._opponent_mons[key]["spe"][0] = variables[key].varValue
                        else:
                            self._opponent_mons[key]["spe"][1] = variables[key].varValue
            else:
                raise ValueError(
                    f"Got Pulp status of {pulp.LpStatus[problem.status]}. Pulp setup: {problem}"
                )

        # Just printing to catch any errors
        for key in self._opponent_mons:
            if (
                ground_truth[key] < self._opponent_mons[key]["spe"][0]
                or ground_truth[key] > self._opponent_mons[key]["spe"][1]
            ):
                print()
                print("===== Got a speed that doesnt agree with the ground truth =====")
                print(
                    f"for {key}, we got speed bounds of {self._opponent_mons[key]['spe']}, with true speed being {ground_truth[key]}"
                )
                print("Ground truth:", ground_truth)
                print("Orders:", orders)
                print()
                print(f"My Active Pokemon {[mon.species for mon in obs.active_pokemon]}")
                print(
                    f"Opp Active Pokemon {[mon.species for mon in obs.opponent_active_pokemon]}"
                )
                print(
                    "My mons:",
                    "".join(
                        [
                            f"\n\t{mon.species}: speed boost ({mon.boosts['spe']}) and effects ({mon.effects}) and status ({mon.status})"
                            for mon in obs.team.values()
                        ]
                    ),
                )
                print(
                    "Opp mons:",
                    "".join(
                        [
                            f"\n\t{mon.species}: speed boost ({mon.boosts['spe']}) and effects ({mon.effects}) and status ({mon.status})"
                            for mon in obs.opponent_team.values()
                        ]
                    ),
                )
                print("Side Conditions:", obs.side_conditions)
                print("Opp Side Conditions:", obs.opponent_side_conditions)
                print("Fields:", obs.fields)
                print("Weather:", obs.weather)
                print("Events:", obs.events)
                print()
                print(problems[0])
                print()
                print(problems[1])
                print()
                raise Exception

    # We look at abilities that don't have priority and trigger, found here:
    # https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts
    # If the triggered ability is weather or field, we then look for strings of other
    # abilities/items that activate with that ability as their own order.
    def _parse_preturn_switch(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, float]]]:
        orders: Dict[str, List[Tuple[str, float]]] = {}
        i = 0

        # Go through the switches, since we can't actually anything from switches
        while i < len(events) and events[i][1] == "switch":
            update_battle(self._battle, events[i])
            i += 1

        # Now we look for abilities, both priority abilities and regular abilities
        while i < len(events) and is_ability_event(events[i]):
            update_battle(self._battle, events[i])

            key = (
                "priority_ability"
                if events[i][-1].replace("ability: ", "") in _PRIORITY_ACTIVATION_ABILITIES
                else "ability"
            )
            ability, mon_ident = get_ability_and_identifier(events[i])
            multiplier = None
            if mon_ident:
                multiplier = self._generate_multiplier(mon_ident)

            if multiplier and mon_ident:
                if key not in orders:
                    orders[key] = []
                orders[key].append((mon_ident, multiplier))

            # If the ability triggers a field or a weather, this could trigger both abilities and items
            if ability in _ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
                order, num_traversed = self._get_activations_from_weather_or_terrain(
                    events, i
                )
                if len(order) > 0:
                    orders["activations"] = order
                i += num_traversed

            i += 1

        # Now we look through item events
        while i < len(events) and events[i][-1] in _ITEMS_THAT_ACTIVATE_ON_SWITCH:
            mon_ident = get_pokemon_ident(events[i][2])
            multiplier = self._generate_multiplier(mon_ident)

            if multiplier:
                if "item" not in orders:
                    orders["item"] = []
                orders["item"].append((mon_ident, multiplier))

            i += 1

        return list(orders.values())

    def _parse_battle_mechanic(
        self, events: List[List[str]]
    ) -> List[List[Tuple[str, float]]]:
        speed_orders: List[List[Tuple[str, float]]] = []
        mechanic_order = []
        i = 0

        while i < len(events) and events[i][1] in ["-terastallize", "-dynamax", "-mega"]:
            update_battle(self._battle, events[i])

            if events[i][1] in ["-terastallize", "-dynamax", "-mega"]:
                mon_ident = get_pokemon_ident(events[i][2])
                multiplier = self._generate_multiplier(mon_ident)

                if multiplier:
                    mechanic_order.append((mon_ident, multiplier))

                # If we hit a battle mechanic that can activate abilities or items (eg alakazite -> trace)
                # we go until we find the weather, then we update it and track activations from it
                if (
                    events[i][1] == "-mega"
                    and events[i][-1]
                    in _MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS
                ):
                    order, num_traversed = self._get_activations_from_weather_or_terrain(
                        events, i
                    )
                    if len(order) > 0:
                        speed_orders.append(order)
                    i += num_traversed

            i += 1

        speed_orders.append(mechanic_order)
        return speed_orders

    # Right now, we only look at orders of moves, and not at things that can activate or trigger because of them,
    # just due to the sheer complexity of VGC. Assumes abilities that affect priority
    # Logic to find priority sourced from here: https://bulbapedia.bulbagarden.net/wiki/Priority
    # Examples:
    # ['', '-activate', 'p2a: Oricorio', 'ability: Dancer']
    # ['', 'move', 'p2a: Sableye', 'Quash', 'p1b: Volcarona']
    # ['', 'move', 'p2a: Sableye', 'After You', 'p1b: Volcarona']
    # ['', '-activate', 'p1a: Farigiraf', 'item: Quick Claw'], ['', 'move', 'p1a: Farigiraf', 'Mean Look', 'p2a: Drednaw']
    # ['', 'cant', 'p2a: Ting-Lu', 'par']
    # ['', 'cant', 'p2a: Ting-Lu', 'slp']
    # ['', 'cant', 'p2a: Ting-Lu', 'frz']
    # ['', 'cant', 'p2a: Ting-Lu', 'recharge']
    # ['', -damage', 'p1a: Raichu', '92/120', '[from] confusion']
    def _parse_move(self, events: List[List[str]]) -> List[List[Tuple[str, float]]]:
        priority_orders: List[List[Tuple[str, float]]] = []
        last_moved, last_priority, last_multipliers, temp_order = None, None, {}, None

        for i, event in enumerate(events):

            update_battle(self._battle, event)

            # We only pay attention to moves; we don't consider if a pokemon can't move because
            # we don't know what move they chose. Eventually, we should include if WE moved, since
            # we should know what move we chose, and what the priority would have been
            if event[1] == "move":

                # Now we need to get the priority
                mon_ident, priority = get_priority_and_identifier(event, self._battle)

                multiplier = self._generate_multiplier(mon_ident)

                # We can't find a multiplier or a priority, it means a mon shouldnt be making an order
                # It also means we're going into a new priority bracket, so we should reset tracking
                if multiplier is None or priority is None:
                    last_moved, last_priority, last_multipliers, temp_order = (
                        None,
                        None,
                        {},
                        None,
                    )

                # If we're in a new priority bracket, record everyone's multipliers and this move
                # If we had a temp order, we need to reset it because we don't know if the priority
                # was the slowest in the last priority, or first in this new priority
                elif last_priority != priority:
                    last_priority = priority
                    last_moved = mon_ident
                    temp_order = None
                    last_multipliers = {}

                    # Store them without positional data (eg "p1: Rillaboom" instead of "p1a...")
                    # because of skill swap

                    last_multipliers = self._save_multipliers()

                # We are in the same priority bracket as the last recorded segment, so we can add a tuple
                # We record the last multipliers, since now we know that in the past, these two mons
                # were compared against each other
                else:

                    # This means that our temp_order from a "can't move" is sandwiched between
                    # two orders with same priority, guaranteeing that it can be compared against
                    # the other mons moving before and after it
                    if temp_order is not None:
                        priority_orders.append(temp_order)
                        temp_order = None

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

            # If there's an event where a mon is sleeping or paralysis, if it's an opponent mon,
            # I don't know what move they used, and so what priority they would be acting in. I
            # will only know for sure if this event is sandwiched between two moves that I know
            # are the same priority
            # TODO: right now, I don't store what move I chose, and so can't do the above
            # calculations properly. For now, I treat even my own mon's failure to move as a mystery
            elif event[1] == "cant":

                # If there was a move before this
                if last_priority is not None:

                    # Adapting mon_ident to remove positions, because of ally switch
                    mon_ident = get_pokemon_ident(event[2])
                    mon_ident = mon_ident[:2] + mon_ident[3:]

                    # Record the temp
                    temp_order = [
                        (last_moved, last_multipliers[last_moved]),
                        (mon_ident, last_multipliers[mon_ident]),
                    ]

                    # Now update our tracking
                    last_moved = mon_ident
                    last_multipliers = self._save_multipliers()

        return priority_orders

    # Order that resolutions parse according to: https://www.smogon.com/forums/threads/sword-shield-battle-mechanics-research.3655528/page-64#post-9244179
    # True order can be found in PS code, under onResidualOrder
    # block: Sand damage, Hail, Solar Power, Dry Skin, Ice Body, Rain Dish
    # block: Grassy Terrain, Healer, Hydration, Shed Skin, Black Sludge, Black Sludge, Leftovers
    # each one resolves, in this order: Aqua Ring, Ingrain, Leech Seed, Poison, Burn, Curse,
    # Binding moves, Octolock, Salt Cure, Taunt, Torment, Encore, Disable, Magnet Rise, Yawn, Perish count
    # block: Uproar,CudChew, Harvest, Moody, Slow Start, Speed Boost, Flame Orb, Sticky Barb, Toxic Orb, White Herb
    def _parse_residual(self, events: List[List[str]]) -> List[List[Tuple[str, float]]]:
        orders: Dict[str, List[Tuple[str, float]]] = {}
        last_moved, last_key, last_multipliers = None, None, {}

        # Go through each event
        for i, event in enumerate(events):
            # TODO: fails when a pokemon faints, since we record they fainted and their status becomes fnt instead of par;
            # we then use the wrong multipliers. We should be recording the orders before the update_battle
            update_battle(self._battle, event)

            residual, mon_ident = get_residual_and_identifier(event)
            if residual is None or mon_ident is None:
                continue

            mon_ident = mon_ident[:2] + mon_ident[3:]  # Removing positional data

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

                orders[key].append(
                    [
                        (last_moved, last_multipliers[last_moved]),
                        (mon_ident, last_multipliers[mon_ident]),
                    ]
                )

                last_moved = mon_ident

                # TODO: Need to update; if a mon faints because of a residual, we should remove it
                last_multipliers = self._save_multipliers()

        return [o for order in orders.values() for o in order]

    # Even at beginning of turns, abilities activate on switch
    def _parse_switch(self, events: List[List[str]]) -> List[List[Tuple[str, float]]]:
        last_switched, last_multipliers = None, {}
        speed_orders: List[List[Tuple[str, float]]] = []
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
                        + "This shouldn't happen!"
                    )

                # We grab the mon_ident, but without position information
                mon_ident = (
                    events[i][2][:2] + ": " + mon._data.pokedex[mon.species]["name"]
                )

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
                    ident = (
                        self._battle.player_role
                        + ": "
                        + mon._data.pokedex[mon.species]["name"]
                    )
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
                    ident = (
                        self._battle.opponent_role
                        + ": "
                        + mon._data.pokedex[mon.species]["name"]
                    )
                    last_multipliers[ident] = self._generate_multiplier(
                        ident,
                        override_ability=info[mon]["ability"] if mon in info else None,
                        override_effects=info[mon]["effects"] if mon in info else None,
                        override_speed_boost=(
                            info[mon]["speed_boost"] if mon in info else None
                        ),
                    )

                for mon in self._battle.teampreview_opponent_team:
                    ident = (
                        self._battle.opponent_role
                        + ": "
                        + mon._data.pokedex[mon.species]["name"]
                    )
                    if ident not in last_multipliers:
                        last_multipliers[ident] = 1.0

            # Check if we have an ability event activated by the switch
            elif is_ability_event(events[i]):
                ability, mon_ident = get_ability_and_identifier(events[i])

                # If the ability triggers a field or a weather, this could trigger both abilities and items.
                # First we update our observation class, and then we get activated orders of events.
                # This method also iterates through obs.events, so we continue afterwards to not iterate twice
                if ability in _ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
                    order, num_traversed = self._get_activations_from_weather_or_terrain(
                        events, i
                    )
                    if len(order) > 0:
                        speed_orders.append(order)
                    i += num_traversed

            i += 1

        return speed_orders

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

    def _save_multipliers(self) -> Dict[str, float]:
        multipliers: Dict[str, float] = {}
        for mon in self._battle.team.values():
            if mon:
                key = (
                    self._battle.player_role
                    + ": "
                    + mon._data.pokedex[mon.species]["name"]
                )
                multipliers[key] = self._generate_multiplier(key)

        for mon in self._battle.opponent_team.values():
            if mon:
                key = (
                    self._battle.opponent_role
                    + ": "
                    + mon._data.pokedex[mon.species]["name"]
                )
                multipliers[key] = self._generate_multiplier(key)

        return multipliers

    # Continue iterating through events that may have been triggered by weather_or_terrain
    # The paramter `i` that's passed should be the event that is the activator (eg terrain or weather)
    # It should end exactly on the last event we've found
    def _get_activations_from_weather_or_terrain(
        self, events: List[List[str]], i: int
    ) -> Tuple[List[Tuple[str, float]], int]:

        ability, mon_ident = get_ability_and_identifier(events[i])
        if ability not in _ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
            raise ValueError(
                "_get_activations_from_weather_or_terrain was not passed the right parameter. \
            it should always be passed the index of the event that could possibly activate other abilities \
            or items"
            )

        activated_order = []

        # Save where we started, and take a first step to look at what's next
        start = i
        i += 1

        while i < len(events) and events[i][1] in [
            "-activate",
            "-enditem",
            "-start",
            "-boost",
        ]:
            update_battle(self._battle, events[i])

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
                mon_ident = get_pokemon_ident(events[i][2])
                multiplier = self._generate_multiplier(mon_ident)
                if multiplier:
                    activated_order.append((mon_ident, multiplier))

            i += 1

        # We need to end on the exact last event we looked at. Because we go one further to see when to end,
        # we need to subtract a single additional iteration
        num_traversed = i - start - 1

        return activated_order, num_traversed

    # Takes the list of move orders and prunes non-comparables, converts list of 3
    # or more into pairs, standardizes mon_ident and dedupes
    @staticmethod
    def clean_orders(
        orders: List[List[Tuple[str, float]]]
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
            Effect.AFTER_YOU in effects or Effect.QUASH in effects or item == "fullincense"
        ):
            return None

        multiplier = 1.0

        # Check Tailwind
        if SideCondition.TAILWIND in side_conditions:
            multiplier *= 2

        # Check Grass Pledge (Swamp)
        if SideCondition.GRASS_PLEDGE in side_conditions:
            multiplier *= 0.5

        # Check Trick Room
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
        elif ability == "slowstart" and effects and Effect.SLOW_START in effects:
            multiplier *= 0.5
        elif (
            ability == "protosynthesis" and effects and Effect.PROTOSYNTHESISSPE in effects
        ):
            multiplier *= 2
        elif ability == "quarkdrive" and effects and Effect.QUARKDRIVESPE in effects:
            multiplier *= 2
        elif ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in fields:
            multiplier *= 2

        return multiplier
