# -*- coding: utf-8 -*-
from typing import Optional, Union, overload

from poke_env.battle import Battle, DoubleBattle, Move, Pokemon, PokemonType, Target
from poke_env.data import GenData
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

"""
This module holds several helper functions to determine the validity of moves.
It used to be named Imprison, which prevents moves from being playable.
"""


@overload
def is_valid_order(order: SingleBattleOrder, battle: Battle) -> bool:
    pass


@overload
def is_valid_order(order: DoubleBattleOrder, battle: DoubleBattle) -> bool:
    pass


# A function that validates whether a move is valid to a battle or not. Note that this isn't truly
# what could be valid to a showdown battle -- it is just used to restrict what our AI considers the
# right "protocol", setting the standard and bounds for the AI. It uses strict "typing"
def is_valid_order(
    order: Union[SingleBattleOrder, DoubleBattleOrder, DefaultBattleOrder],
    battle: Union[Battle, DoubleBattle],
) -> bool:
    if isinstance(order, DefaultBattleOrder):
        return True
    elif isinstance(order, DoubleBattleOrder) and isinstance(battle, DoubleBattle):
        return _is_valid_doubles_order(order, battle)
    elif isinstance(order, SingleBattleOrder) and isinstance(battle, Battle):
        return _is_valid_singles_order(order, battle)
    else:
        return False


def _is_valid_singles_order(order: SingleBattleOrder, battle: Battle) -> bool:
    if order.order is None:
        return False
    elif isinstance(order, PassBattleOrder):
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


def _is_valid_doubles_order(double_order: DoubleBattleOrder, battle: DoubleBattle) -> bool:
    # Check each order individually
    for i, order in enumerate([double_order.first_order, double_order.second_order]):
        # Check to see if we should be sending an order in the first place
        # But can send an order when one of your mon faints and you need to send in another
        if (
            (order is not None and not isinstance(order, PassBattleOrder))
            and not battle.active_pokemon[i]
            and not battle.force_switch[i]
        ):
            return False

        # The order is only allowed to not exist (pass) if:
        # 1. This slot does NOT need to force switch
        # 2. AND (the other slot needs to force switch OR this slot has no active pokemon)
        # EXCEPTION: If both slots need to force switch but there aren't enough
        # available switches (e.g., only 1 Pokemon left), pass is valid for one slot.
        if order is None or isinstance(order, PassBattleOrder):
            # If THIS slot must switch, pass is usually not valid
            if battle.force_switch[i]:
                # Exception: both slots need to switch but not enough mons available
                # In this case, one slot switches and the other must pass
                if battle.force_switch[0] and battle.force_switch[1]:
                    # Count unique available switches across both slots
                    all_available = set()
                    for slot_switches in battle.available_switches:
                        for mon in slot_switches:
                            all_available.add(mon.species)
                    # If there's only 1 available switch, one slot MUST pass
                    if len(all_available) <= 1:
                        pass  # Allow pass in this case
                    else:
                        return False
                else:
                    return False
            # Otherwise, pass is valid if other slot is switching or we have no active mon
            if not (battle.force_switch[1 - i] or not battle.active_pokemon[i]):
                return False
        elif order.order is None:
            # Same logic for order.order being None
            if battle.force_switch[i]:
                return False
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
                num_opp_mons = int(battle.opponent_active_pokemon[0] is not None) + int(
                    battle.opponent_active_pokemon[1] is not None
                )

                # These moves need a target, if there isnt just one mon on the other side
                if (
                    order.move_target == DoubleBattle.EMPTY_TARGET_POSITION
                    and num_opp_mons > 1
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
    if (
        double_order.first_order
        and not isinstance(double_order.first_order, PassBattleOrder)
    ) and (
        double_order.second_order
        and not isinstance(double_order.second_order, PassBattleOrder)
    ):
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


def is_reasonable_move(
    order: Union[SingleBattleOrder, DoubleBattleOrder],
    battle: Union[Battle, DoubleBattle],
) -> bool:
    if isinstance(order, SingleBattleOrder) and isinstance(battle, Battle):
        return _is_reasonable_singles_move(battle, order, battle.gen)
    elif isinstance(order, DoubleBattleOrder) and isinstance(battle, DoubleBattle):
        return _is_reasonable_doubles_move(battle, order, battle.gen)
    else:
        raise ValueError(
            "Invalid order or battle parameter types; received "
            + str(type(order))
            + " and "
            + str(type(battle))
        )


# Filters all orders to reasonable moves; only implemented in Doubles
def _is_reasonable_doubles_move(
    battle: DoubleBattle, order: DoubleBattleOrder, gen: int
) -> bool:
    """
    Filters moves to reasonable moves. This is a helper function to guide AI to not search unlikely moves.

    :param battle: the battle with which to evalute the reasonableness of moves
    :type battle: DoubleBattle
    :param order: the move to check
    :type order: DoubleBattleOrder
    :return: A list of tuples that contain reasonable orders
    :rtype: List[DoubleBattleOrder]
    """

    if (
        order.first_order
        and isinstance(order.first_order.order, Move)
        and order.first_order.order.current_pp == 0
    ):
        return False
    if (
        order.second_order
        and isinstance(order.second_order.order, Move)
        and order.second_order.order.current_pp == 0
    ):
        return False

    if _useless_self_boost(
        order.first_order, battle.active_pokemon[0]
    ) or _useless_self_boost(order.second_order, battle.active_pokemon[1]):
        return False
    if _useless_battle_condition(battle, order.first_order) or _useless_battle_condition(
        battle, order.second_order
    ):
        return False
    if _useless_self_hit(battle, order.first_order, 0, gen) or _useless_self_hit(
        battle, order.second_order, 1, gen
    ):
        return False
    if _useless_first_move(battle, order.first_order, 0) or _useless_first_move(
        battle, order.second_order, 1
    ):
        return False

    return True


def _useless_first_move(battle, order: Optional[SingleBattleOrder], index: int):
    if order is None or isinstance(order, PassBattleOrder):
        return False

    if not order or not isinstance(order.order, Move):
        return False

    if not battle.active_pokemon[index].first_turn and order.order.id in [
        "fakeout",
        "firstimpression",
    ]:
        return True

    return False


# Return if the self-boost is inneffectual
def _useless_self_boost(order: Optional[SingleBattleOrder], mon: Optional[Pokemon]):
    if order is None or mon is None or isinstance(order, PassBattleOrder):
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
def _useless_battle_condition(battle, order: Optional[SingleBattleOrder]):
    if order is None or isinstance(order, PassBattleOrder):
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
def _useless_self_hit(battle, order: Optional[SingleBattleOrder], index: int, gen: int):
    if order is None or isinstance(order, PassBattleOrder):
        return False

    # Eliminate easy conditions in which this is not a useless self hit
    if not order or not isinstance(order.order, Move):
        return False
    if not (order.order.damage or order.order.base_power > 0):
        return False
    if order.order.self_switch:
        return False
    if order.order.id == "pollenpuff":
        return False

    # If it's a self-hit
    if (
        order.order.target in (Target.ANY, Target.NORMAL) and order.move_target < 0
    ) or order.order.target in (Target.ALL_ADJACENT, Target.ALL):
        # Get the mon who is going to be hit
        target_mon = battle.active_pokemon[1 - index]

        # Only allow this as a potential move under these conditions
        if target_mon is None:
            return True
        elif (
            target_mon.item == "weaknesspolicy"
            and order.order.type.damage_multiplier(
                *target_mon.types, type_chart=GenData.from_gen(gen).type_chart
            )
            >= 2
        ):
            return False
        elif target_mon.ability == "berserk":
            return False
        elif target_mon.ability == "justified" and order.order.type == PokemonType.DARK:
            return False
        elif target_mon.ability == "waterabsorb" and order.order.type == PokemonType.WATER:
            return False
        elif (
            target_mon.ability == "voltabsorb" and order.order.type == PokemonType.ELECTRIC
        ):
            return False
        elif target_mon.ability == "flashfire" and order.order.type == PokemonType.FIRE:
            return False
        elif target_mon.ability == "levitate" and order.order.type == PokemonType.GROUND:
            return False
        elif target_mon.ability == "eartheater" and order.order.type == PokemonType.GROUND:
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


def _is_reasonable_singles_move(
    battle: Battle, order: SingleBattleOrder, gen: int
) -> bool:
    raise NotImplementedError("This function is not implemented for singles battles yet")
