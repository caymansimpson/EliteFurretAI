# -*- coding: utf-8 -*-
import re
from typing import Any, List, Optional, Sequence, Union, overload

from poke_env.data import GenData
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.target import Target
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder

"""
This module holds several helper functions to determine the validity of moves.
This is why it's named imprison, which prevents moves from being playable.
"""


@overload
def is_valid_order(order: BattleOrder, battle: Battle) -> bool:
    pass


@overload
def is_valid_order(order: DoubleBattleOrder, battle: DoubleBattle) -> bool:
    pass


# A function that validates whether a move is valid to a battle or not. Note that this isn't truly
# what could be valid to a showdown battle -- it is just used to restrict what our AI considers the
# right "protocol", setting the standard and bounds for the AI. It uses strict "typing"
def is_valid_order(
    order: Union[BattleOrder, DoubleBattleOrder], battle: Union[Battle, DoubleBattle]
) -> bool:

    if isinstance(order, DoubleBattleOrder) and isinstance(battle, DoubleBattle):
        return _is_valid_doubles_order(order, battle)
    elif isinstance(order, BattleOrder) and isinstance(battle, Battle):
        return _is_valid_singles_order(order, battle)
    else:
        return False


def _is_valid_singles_order(order: BattleOrder, battle: Battle) -> bool:

    if order.order is None:
        return False
    elif isinstance(order.order, Move):
        if order.order.id not in map(lambda x: x.id, battle.available_moves):
            return False
        elif battle.force_switch:
            return False
        elif order.terastallize and not battle.can_tera:
            return False
        elif order.mega and not battle.can_mega_evolve:
            return False
        elif order.dynamax and not battle.can_dynamax:
            return False
        elif order.z_move and not battle.can_z_move:
            return False
        else:
            return True
    elif isinstance(order.order, Pokemon):
        if order.order.species not in map(lambda x: x.species, battle.available_switches):
            return False
        elif not battle.force_switch and battle.trapped:
            return False
        elif order.mega:
            return False
        elif order.z_move:
            return False
        elif order.dynamax:
            return False
        elif order.terastallize:
            return False
        else:
            return True
    else:
        return False


# TODO: Sometimes I get ['', 'error', "[Invalid choice] Can't move: You need a switch response"]
# TODO: ['', 'error', "[Invalid choice] There's nothing to choose"]
def _is_valid_doubles_order(double_order: DoubleBattleOrder, battle: DoubleBattle) -> bool:

    # Check each order individually
    for i, order in enumerate([double_order.first_order, double_order.second_order]):

        # Check to see if we should be sending an order in the first place
        if order is not None and not battle.active_pokemon[i]:
            return False

        # The order is only allowed to not exist if we're forced to switch out another pokemon or if one doesn't exist
        if order is None:
            if not (battle.force_switch[1 - i] or not battle.active_pokemon[i]):
                return False
        elif order.order is None:
            if not (battle.force_switch[1 - i] or not battle.active_pokemon[i]):
                return False

        # Check whether a switch is valid
        elif isinstance(order.order, Pokemon):

            if battle.trapped[i] and not battle.force_switch[i]:
                return False

            # Can't be switching out and tera/dynamax/mega/z-move at the same time
            if (
                order.mega
                or order.z_move
                or order.dynamax
                or order.terastallize
                or order.move_target != 0
            ):
                return False

            # Can only switch to the right pokemon
            if order.order.species not in map(
                lambda x: x.species, battle.available_switches[i]
            ):
                return False

        # Check whether a move is valid
        elif isinstance(order.order, Move):

            # Check if we're supposed to switch
            if battle.force_switch[i]:
                return False

            # Check if we can use this move
            if order.order.id not in map(lambda x: x.id, battle.available_moves[i]):
                return False

            # Check to ensure we can tera/dynamax/mega-evolve/z-move if we choose to do so
            if order.dynamax and not battle.can_dynamax[i]:
                return False
            elif order.mega and not battle.can_mega_evolve[i]:
                return False
            elif order.terastallize and not battle.can_tera[i]:
                return False
            elif order.z_move and not battle.can_z_move[i]:
                return False

            # Make sure you're targeting something on the opponent's side of the field with dynamax
            if order.order.target == Target.ADJACENT_FOE or (
                order.dynamax and order.order.base_power > 0
            ):
                if order.move_target in (
                    DoubleBattle.EMPTY_TARGET_POSITION,
                    DoubleBattle.POKEMON_1_POSITION,
                    DoubleBattle.POKEMON_2_POSITION,
                ):
                    return False

                # Trying to target an opponent pokemon that doesn't exist
                if (
                    order.move_target
                    in (
                        DoubleBattle.OPPONENT_1_POSITION,
                        DoubleBattle.OPPONENT_2_POSITION,
                    )
                    and not battle.opponent_active_pokemon[order.move_target - 1]
                ):
                    return False

            # For moves that can target any pokemon on the field:
            elif order.order.target in (Target.ANY, Target.NORMAL):

                # These moves need a target
                if order.move_target == DoubleBattle.EMPTY_TARGET_POSITION:
                    return False

                # Trying to target an opponent pokemon that doesn't exist
                if (
                    order.move_target
                    in (
                        DoubleBattle.OPPONENT_1_POSITION,
                        DoubleBattle.OPPONENT_2_POSITION,
                    )
                    and not battle.opponent_active_pokemon[order.move_target - 1]
                ):
                    return False

                # Trying to target a Pokemon on your own side of the field
                if order.move_target in (
                    DoubleBattle.POKEMON_1_POSITION,
                    DoubleBattle.POKEMON_2_POSITION,
                ):

                    # Can't target yourself
                    if order.move_target == battle.to_showdown_target(
                        order.order, battle.active_pokemon[i]
                    ):
                        return False

                    # Can't target your ally pokemon if they don't exist
                    elif battle.active_pokemon[1 - i] is None:
                        return False

            # Make sure you're're targeting something on your side of the field
            elif order.order.target == Target.ADJACENT_ALLY_OR_SELF:
                if order.move_target not in (
                    DoubleBattle.POKEMON_1_POSITION,
                    DoubleBattle.POKEMON_2_POSITION,
                ):
                    return False

            # We arbitrarily say that self-targeting moves should have no target
            elif order.order.target == Target.SELF:
                if order.move_target != DoubleBattle.EMPTY_TARGET_POSITION:
                    return False

    # Check cases where orders could invalidate each other
    if double_order.first_order and double_order.second_order:

        # Check to make sure we're not dynamaxing two pokemon
        if double_order.first_order.dynamax and double_order.second_order.dynamax:
            return False

        # Check to make sure we're not mega-ing two pokemon
        if double_order.first_order.mega and double_order.second_order.mega:
            return False

        # Check to make sure we're not tera-ing two pokemon
        if (
            double_order.first_order.terastallize
            and double_order.second_order.terastallize
        ):
            return False

        # Check to make sure we're not z-move-ing two pokemon
        if double_order.first_order.z_move and double_order.second_order.z_move:
            return False

        # Check to see we're not switching to the same mon
        if (
            isinstance(double_order.first_order.order, Pokemon)
            and isinstance(double_order.second_order.order, Pokemon)
            and double_order.first_order.order.species
            == double_order.second_order.order.species
        ):
            return False

        # Check to see if we're only supposed to force switch one mon, in which case one order should be None
        if battle.force_switch[0] != battle.force_switch[1]:
            return False

    return True


@overload
def filter_to_reasonable_moves(
    battle: Battle, orders: Sequence[BattleOrder]
) -> List[BattleOrder]:
    pass


@overload
def filter_to_reasonable_moves(
    battle: DoubleBattle, orders: Sequence[DoubleBattleOrder]
) -> List[DoubleBattleOrder]:
    pass


def filter_to_reasonable_moves(
    battle: Union[Battle, DoubleBattle],
    orders: Union[Sequence[BattleOrder], Sequence[DoubleBattleOrder]],
) -> Union[List[BattleOrder], List[DoubleBattleOrder]]:
    battle_format = battle.battle_tag.split("-")[1]
    match = re.match("(gen[0-9])", battle_format)
    if match is None:
        raise ValueError(
            "Could not parse gen from battle json's format: {format}".format(
                format=battle_format
            )
        )
    gen = int(match.groups()[0][-1])

    if len(orders) == 0:
        return []
    elif isinstance(orders[0], BattleOrder) and isinstance(battle, Battle):
        return _filter_to_reasonable_singles_moves(battle, orders, gen)
    elif isinstance(orders[0], DoubleBattleOrder) and isinstance(battle, DoubleBattle):
        return _filter_to_reasonable_doubles_moves(battle, orders, gen)
    else:
        raise ValueError("Invalid order or battle parameter types")


# Filters all orders to reasonable moves; only implemented in Doubles
def _filter_to_reasonable_doubles_moves(
    battle: DoubleBattle, orders: Sequence[Any], gen: int
):
    """
    Filters moves to reasonable moves. This is a helper function to guide AI to not search unlikely moves.

    :param battle: the battle with which to evalute the reasonableness of moves
    :type battle: DoubleBattle
    :param orders: the list of moves to check -- really a List of DoubleBattleOrder's
    :type orders: Sequence[Any]
    :return: A list of tuples that contain reasonable orders
    :rtype: List[DoubleBattleOrder]
    """

    reasonable_moves = []

    for order in orders:
        if (
            order.first_order
            and isinstance(order.first_order.order, Move)
            and order.first_order.order.current_pp == 0
        ):
            continue
        if (
            order.second_order
            and isinstance(order.second_order.order, Move)
            and order.second_order.order.current_pp == 0
        ):
            continue

        if _useless_self_boost(
            order.first_order, battle.active_pokemon[0]
        ) or _useless_self_boost(order.second_order, battle.active_pokemon[1]):
            continue
        if _useless_battle_condition(
            battle, order.first_order
        ) or _useless_battle_condition(battle, order.second_order):
            continue
        if _useless_self_hit(battle, order.first_order, 0, gen) or _useless_self_hit(
            battle, order.second_order, 1, gen
        ):
            continue

        reasonable_moves.append(order)

    return reasonable_moves


# Return if the self-boost is inneffectual
def _useless_self_boost(order: Optional[BattleOrder], mon: Optional[Pokemon]):
    if order is None or mon is None:
        return False

    if order and isinstance(order.order, Move):

        # Only consider self- or ally-boosting moves if you have boosts left, or if you dont, if the other pokemon has sucker punch
        if order.order.boosts and order.order.target == Target.SELF:
            for stat in order.order.boosts:
                if (mon.boosts[stat] < 6 and order.order.boosts[stat] > 0) or (
                    order.order.boosts[stat] > -6 and order.order.boosts[stat] < 0
                ):
                    return False

            return True

    return False


# Return if side condition move is useless. This should eventually return False for everything when we learn better (e.g. Torkoal switch-ins)
def _useless_battle_condition(battle, order: Optional[BattleOrder]):
    if order is None:
        return False

    if not order or not isinstance(order.order, Move):
        return False

    if order.order.side_condition and order.order.side_condition in battle.side_conditions:
        return True
    if order.order.weather and battle.weather and order.order.weather == battle.weather:
        return True
    if order.order.terrain and battle.fields and order.order.terrain in battle.fields:
        return True
    if (
        order.order.pseudo_weather
        and battle.fields
        and order.order.pseudo_weather in battle.fields
    ):
        return True
    return False


# Method to help reduce state space (will eventually default to 0). Here's the cases in which a self-hit is valid:
# Can default to False to eliminate self-hits, and True to not eliminate anything
def _useless_self_hit(battle, order: Optional[BattleOrder], index: int, gen: int):
    if order is None:
        return False

    # Eliminate easy conditions in which this is not a useless self hit
    if not order or not isinstance(order.order, Move):
        return False
    if not (order.order.damage or order.order.base_power > 0):
        return False
    if order.order.self_switch:
        return False

    # If it's a self-hit
    if (
        order.order.target in (Target.ANY, Target.NORMAL) and order.move_target < 0
    ) or order.order.target in (Target.ALL_ADJACENT, Target.ALL):

        # Get the mon who is going to be hit
        target_mon = battle.active_pokemon[1 - index]

        # Only allow this as a potential move under these conditions
        if (
            target_mon.item == "weaknesspolicy"
            and order.order.type.damage_multiplier(
                *target_mon.types, type_chart=GenData.from_gen(gen).type_chart
            )
            >= 2
        ):
            return True
        elif target_mon.ability == "Berserk":
            return False
        elif target_mon.ability == "Justified" and order.order.type == PokemonType.DARK:
            return False
        elif (
            target_mon.ability == "Water Absorb" and order.order.type == PokemonType.WATER
        ):
            return False
        elif (
            target_mon.ability == "Volt Absorb"
            and order.order.type == PokemonType.ELECTRIC
        ):
            return False
        elif target_mon.ability == "Flash Fire" and order.order.type == PokemonType.FIRE:
            return False
        elif target_mon.ability == "Levitate" and order.order.type == PokemonType.GROUND:
            return False
        elif (
            target_mon.ability == "Earth Eater" and order.order.type == PokemonType.GROUND
        ):
            return False
        elif PokemonType.GHOST in target_mon.types and order.order.type in (
            PokemonType.NORMAL,
            PokemonType.FIGHTING,
        ):
            return False
        elif (
            PokemonType.GROUND in target_mon.types
            and order.order.type == PokemonType.ELECTRIC
        ):
            return False
        elif (
            PokemonType.FAIRY in target_mon.types
            and order.order.type == PokemonType.DRAGON
        ):
            return False
        elif (
            PokemonType.STEEL in target_mon.types
            and order.order.type == PokemonType.POISON
        ):
            return False
        elif (
            PokemonType.DARK in target_mon.types
            and order.order.type == PokemonType.PSYCHIC
        ):
            return False
        else:
            return True

    return False


def _filter_to_reasonable_singles_moves(
    battle: Battle, orders: Sequence[BattleOrder], gen: int
) -> List[BattleOrder]:
    raise NotImplementedError("This function is not implemented for singles battles yet")
