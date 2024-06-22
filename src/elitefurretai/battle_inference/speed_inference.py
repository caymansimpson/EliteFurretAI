# -*- coding: utf-8 -*-
"""This module tracks pokemons' moves, stats and items inferences throughout a battle
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
    _PRIORITY_ACTIVATION_ABILITIES,
    get_pokemon_ident,
    is_ability_event,
    parse_ability,
    copy_observation,
)


# TODO: extend class to take into account aggressiveness of assumptions
# TODO: test on actual data to see when/where I make wrong assumptions
# Companion class to BattleInference that handles Speed parsing
class SpeedInference:

    def __init__(self, battle: Union[Battle, DoubleBattle], mons: Dict[str, Any]):
        self._mons: Dict[str, Any] = mons
        self._battle = battle

    # Continue iterating through events that may have been triggered by weather_or_terrain
    def _get_activations_from_weather_or_terrain(
        self, obs: Observation, iterator: Iterator
    ):
        activated_order = []
        event = next(iterator, None)
        while event is not None and event in ["-activate", "-enditem", "-start", "-boost"]:

            # Check to see if we're done with activations. If we see a Booster Energy activation, it means we're at the item
            # activation phase at the beginning of a turn and we should be done. Example activations:
            # ['', '-activate', 'p1b: Iron Hands', 'ability: Quark Drive']
            # ['', '-start', 'p1b: Iron Hands', 'quarkdriveatk']
            # ['', '-enditem', 'p2a: Hawlucha', 'Electric Seed']
            # ['', '-boost', 'p2a: Hawlucha', 'def', '1', '[from] item: Electric Seed']
            if (
                event[1] not in ["-activate", "-enditem", "-start", "-boost"]
                or event[-1] == "Booster Energy"
            ):
                return activated_order

            if event[1] in ["-enditem", "-activate"]:
                mon_ident = get_pokemon_ident(event[2])
                mon = obs.team[mon_ident]
                if mon is None:
                    raise ValueError(f"We couldnt find the mon with ident {mon_ident}")

                sc = (
                    obs.side_conditions
                    if mon_ident.startswith("p1")
                    else obs.opponent_side_conditions
                )

                multiplier = self.get_speed_multiplier(
                    mon_ident=mon_ident,
                    weathers=obs.weather,
                    side_conditions=sc,
                    fields=obs.fields,
                    item=mon.item,
                    ability=mon.ability,
                    status=mon.status,
                    effects=mon.effects,
                )
                activated_order.append((mon_ident, multiplier))
            event = next(iterator, None)

        return activated_order

    # Returns a Tuple of (mon_ident, speed_multiplier, ability) from an ability event
    # eg ['', '-weather', 'Sandstorm', '[from] ability: Sand Stream', '[of] p1b: Tyranitar']
    def _get_ident_and_multiplier_from_ability_event(
        self, event: List[str], obs: Observation
    ) -> Tuple[str, Optional[float], Optional[str]]:

        ability, mon_ident = parse_ability(event)
        if mon_ident is None:
            raise ValueError(f"We couldnt find the mon_ident and ability for {event}")

        mon = (
            obs.team[mon_ident]
            if mon_ident.startswith("p1")
            else obs.opponent_team[mon_ident]
        )
        if mon is None:
            raise ValueError(
                f"We couldnt find the mon for the parsed ability ({ability}) for ident {mon_ident}"
            )

        sc = (
            obs.side_conditions
            if mon_ident.startswith("p1")
            else obs.opponent_side_conditions
        )

        multiplier = self.get_speed_multiplier(
            mon_ident=mon_ident,
            weathers=obs.weather,
            side_conditions=sc,
            fields=obs.fields,
            item=mon.item,
            ability=mon.ability,
            status=mon.status,
            effects=mon.effects,
        )
        return mon_ident, multiplier, ability

    def check_speed(self, observation: Observation):
        """
        Create bounds for speed by going through events in Observation

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

        # So that I can mutate it as the turn goes on
        obs = copy_observation(observation)

        # If first turn, order of priority abilities, abilities and then items tells you things
        # TODO: think about whether this also applies when two mon are ko'd and we have to start the next turn?
        # ^ Theyre similar, but we don't read switch orders at the beginning of a turn
        if any(map(lambda x: x[1] == "start", obs.events)):

            # Go through events to get priority ability activations first
            priority_ability_order = []
            for event in obs.events:
                if event[-1].replace("ability: ", "") in _PRIORITY_ACTIVATION_ABILITIES:
                    mon_ident, multiplier, ability = (
                        self._get_ident_and_multiplier_from_ability_event(event, obs)
                    )
                    priority_ability_order.append((mon_ident, multiplier))

            speed_orders.append(priority_ability_order)

            # Next we look at abilities that don't have priority and trigger, found here:
            # https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts
            # If the triggered ability is weather or field, we then look for strings of other
            # abilities/items that activate with that ability as their own order.
            ability_order = []

            # Go through each event in the turn; assumes there is not a None in obs.events
            iterator = iter(obs.events)
            event = next(iterator, None)
            while event is not None:

                # If we get to item activations at the beginning of the turn, we've gone too far!
                if event[-1] in _ITEMS_THAT_ACTIVATE_ON_SWITCH:
                    break

                # Otherwise, this time ignore priority abilities or non-ability messages
                if (
                    is_ability_event(event)
                    and event[-1].replace("ability: ", "")
                    not in _PRIORITY_ACTIVATION_ABILITIES
                ):
                    mon_ident, multiplier, ability = (
                        self._get_ident_and_multiplier_from_ability_event(event, obs)
                    )
                    ability_order.append((mon_ident, multiplier))

                    # If the ability triggers a field or a weather, this could trigger both abilities and items.
                    # First we update our observation class, and then we get activated orders of events.
                    # This method also iterates through obs.events, so we continue afterwards to not iterate twice
                    # TODO: need to make a copy of observation
                    if ability in _ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS:
                        if (
                            event[1] == "-fieldstart"
                        ):  # TODO: UPDATE TERRAINS; cant just add
                            obs.fields[Field.from_showdown_message(event[2])] = 5
                        elif (
                            event[1] == "-weather"
                        ):  # TODO: UPDATE WEATHers; cant just add
                            obs.weather[Weather.from_showdown_message(event[2])] = 5

                        speed_orders.append(
                            self._get_activations_from_weather_or_terrain(obs, iterator)
                        )
                        continue

                event = next(iterator, None)

            speed_orders.append(ability_order)

            # TODO: need to figure out how to segment these into chapters
            # Now look at items -- only protosynthesis and air balloon for now
            if event and event[-1] in _ITEMS_THAT_ACTIVATE_ON_SWITCH:
                while event:
                    if event[-1] in _ITEMS_THAT_ACTIVATE_ON_SWITCH:
                        mon_ident, multiplier, ability = (
                            self._get_ident_and_multiplier_from_ability_event(event, obs)
                        )
                        ability_order.append((mon_ident, multiplier))

                    event = next(iterator, None)

        else:
            # switch, tera/mega/dynamax, moves by priority, residuals orders
            for event_array in obs.events:
                print("\tnormal battle:  ", event_array)
                # residuals: https://www.smogon.com/forums/threads/ultra-sun-ultra-moon-battle-mechanics-research-read-post-2.3620030/page-11#post-8264465

        # Now I have to clean speed_orders (eg if I have two events from the same mon in a row, or if I have only one event
        # or if I have already stored an equivalent equation), or if there is a None speed multiplier
        # Now I can solve this Linear Programming problem with somewhat of the below: (have to do something about scarf)
        # from pulp import *
        # min_problem = LpProblem("SpeedSolver", LpMinimize)
        # max_problem = LpProblem("SpeedSolver", LpMaximize)
        # mon1_speed = LpVariable(ident, lowBound=low_speed, upBound=max_speed, cat='Integer')
        # max_opt = LpAffineExpression()
        return
        raise NotImplementedError

    # Gets speed multipliers that apply to a pokemon. If a mon's position in the speed bracket has been affected (eg
    # via quash) we return None so that we know not to account for this mon's speed when deducing others'
    @staticmethod
    def get_speed_multiplier(
        mon_ident: str,
        weathers: Dict[Weather, int] = {},
        side_conditions: Dict[SideCondition, int] = {},
        fields: Dict[Field, int] = {},
        speed_boosts: int = 0,
        item: Optional[str] = None,
        ability: Optional[str] = None,
        status: Optional[Status] = None,
        effects: Dict[Effect, int] = {},
    ) -> Optional[float]:

        # We should ignore speed calculations for mons whose place in the speed bracket has been affected
        if Effect.AFTER_YOU in effects or Effect.QUASH in effects or item == "fullincense":
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
        elif item == "quickpowder" and "ditto" in mon_ident:
            multiplier *= 2

        # Check abilities
        if ability == "quickfeet" and status is not None:
            multiplier *= 1.5
        elif ability == "unburden" and item is None:
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
        elif ability == "slowstart" and Effect.SLOW_START in effects:
            multiplier *= 0.5
        elif ability == "protosynthesis" and Effect.PROTOSYNTHESISSPE in effects:
            multiplier *= 2
        elif ability == "quarkdrive" and Effect.QUARKDRIVESPE in effects:
            multiplier *= 2
        elif ability == "surgesurfer" and Field.ELECTRIC_TERRAIN in fields:
            multiplier *= 2

        return multiplier
