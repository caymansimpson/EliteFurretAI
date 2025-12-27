#!/usr/bin/env python3
"""
Python implementation of poke-engine-doubles evaluation function.

This implements the evaluation logic from:
https://github.com/pmariglia/poke-engine-doubles/blob/main/src/genx/evaluate.rs

Uses poke-env objects and borrows speed calculation logic from SpeedInference.
"""

from typing import Dict, List, Optional, Union

from poke_env.battle import (
    Battle,
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.inference.speed_inference import SpeedInference

# Constants from poke-engine evaluate.rs
POKEMON_ALIVE = 30.0
POKEMON_HP = 100.0
USED_TERA = -50.0

POKEMON_ATTACK_BOOST = 30.0
POKEMON_DEFENSE_BOOST = 15.0
POKEMON_SPECIAL_ATTACK_BOOST = 30.0
POKEMON_SPECIAL_DEFENSE_BOOST = 15.0
POKEMON_SPEED_BOOST = 30.0

# Boost multipliers
POKEMON_BOOST_MULTIPLIER_6 = 3.3
POKEMON_BOOST_MULTIPLIER_5 = 3.15
POKEMON_BOOST_MULTIPLIER_4 = 3.0
POKEMON_BOOST_MULTIPLIER_3 = 2.5
POKEMON_BOOST_MULTIPLIER_2 = 2.0
POKEMON_BOOST_MULTIPLIER_1 = 1.0
POKEMON_BOOST_MULTIPLIER_0 = 0.0
POKEMON_BOOST_MULTIPLIER_NEG_1 = -1.0
POKEMON_BOOST_MULTIPLIER_NEG_2 = -2.0
POKEMON_BOOST_MULTIPLIER_NEG_3 = -2.5
POKEMON_BOOST_MULTIPLIER_NEG_4 = -3.0
POKEMON_BOOST_MULTIPLIER_NEG_5 = -3.15
POKEMON_BOOST_MULTIPLIER_NEG_6 = -3.3

# Status penalties
POKEMON_FROZEN = -40.0
POKEMON_ASLEEP = -45.0
POKEMON_PARALYZED = -25.0
POKEMON_TOXIC = -30.0
POKEMON_POISONED = -10.0
POKEMON_BURNED = -25.0

# Volatile status scores
LEECH_SEED = -30.0
SUBSTITUTE = 40.0
CONFUSION = -20.0
PERISH3 = -15.0
PERISH2 = -30.0
PERISH1 = -45.0

# Side condition scores
REFLECT = 20.0
LIGHT_SCREEN = 20.0
AURORA_VEIL = 40.0
SAFE_GUARD = 5.0
TAILWIND = 5.0
HEALING_WISH = 30.0

# Hazard scores
STEALTH_ROCK = -10.0
SPIKES = -7.0
TOXIC_SPIKES = -7.0
STICKY_WEB = -25.0

# Speed comparison bonus
FASTER_THAN_OPPONENT = 5.0


def get_boost_multiplier(boost: int) -> float:
    """Get the multiplier for stat boosts [-6, 6]"""
    multipliers = {
        6: POKEMON_BOOST_MULTIPLIER_6,
        5: POKEMON_BOOST_MULTIPLIER_5,
        4: POKEMON_BOOST_MULTIPLIER_4,
        3: POKEMON_BOOST_MULTIPLIER_3,
        2: POKEMON_BOOST_MULTIPLIER_2,
        1: POKEMON_BOOST_MULTIPLIER_1,
        0: POKEMON_BOOST_MULTIPLIER_0,
        -1: POKEMON_BOOST_MULTIPLIER_NEG_1,
        -2: POKEMON_BOOST_MULTIPLIER_NEG_2,
        -3: POKEMON_BOOST_MULTIPLIER_NEG_3,
        -4: POKEMON_BOOST_MULTIPLIER_NEG_4,
        -5: POKEMON_BOOST_MULTIPLIER_NEG_5,
        -6: POKEMON_BOOST_MULTIPLIER_NEG_6,
    }
    return multipliers.get(boost, 0.0)


def evaluate_poison(pokemon: Pokemon, base_score: float) -> float:
    """Evaluate poison damage based on turns remaining"""
    if pokemon.status == Status.TOX:
        # Badly poisoned - gets worse each turn
        # Simplified: assume average of 3 turns remaining
        turns_remaining = 3.0
        return base_score * (1.0 + turns_remaining * 0.1)
    return base_score


def evaluate_burned(pokemon: Pokemon) -> float:
    """Evaluate burn status penalty"""
    # Burn is constant 1/16 HP per turn
    return POKEMON_BURNED


def evaluate_hazards(pokemon: Pokemon, side_conditions: Dict[SideCondition, int]) -> float:
    """Evaluate hazard damage for a Pokemon"""
    if pokemon.fainted:
        return 0.0

    score = 0.0

    # Stealth Rock damage varies by type
    if SideCondition.STEALTH_ROCK in side_conditions:
        score += STEALTH_ROCK

    # Spikes damage (can be stacked 1-3 layers)
    spikes_layers = side_conditions.get(SideCondition.SPIKES, 0)
    score += spikes_layers * SPIKES

    # Toxic Spikes (can be 1-2 layers)
    toxic_spikes_layers = side_conditions.get(SideCondition.TOXIC_SPIKES, 0)
    score += toxic_spikes_layers * TOXIC_SPIKES

    # Sticky Web (speed reduction for grounded Pokemon)
    if SideCondition.STICKY_WEB in side_conditions:
        # Check if Pokemon is grounded (not Flying type, no Levitate, etc.)
        is_grounded = True  # Simplified - assume most Pokemon are grounded
        if "flying" not in [t.name.lower() for t in pokemon.types]:
            # More sophisticated grounding check could be added here
            pass
        if is_grounded:
            score += STICKY_WEB

    return score


def evaluate_pokemon(pokemon: Pokemon) -> float:
    """Evaluate individual Pokemon value"""
    if pokemon.fainted:
        return 0.0

    score = 0.0

    # HP percentage
    score += POKEMON_HP * pokemon.current_hp_fraction

    # Status conditions
    if pokemon.status == Status.BRN:
        score += evaluate_burned(pokemon)
    elif pokemon.status == Status.FRZ:
        score += POKEMON_FROZEN
    elif pokemon.status == Status.SLP:
        score += POKEMON_ASLEEP
    elif pokemon.status == Status.PAR:
        score += POKEMON_PARALYZED
    elif pokemon.status == Status.TOX:
        score += evaluate_poison(pokemon, POKEMON_TOXIC)
    elif pokemon.status == Status.PSN:
        score += evaluate_poison(pokemon, POKEMON_POISONED)

    # Item bonus (simplified)
    if pokemon.item is not None and pokemon.item != "":
        score += 10.0

    # Prevent negative scores
    if score < 0.0:
        score = 0.0

    # Base alive bonus
    score += POKEMON_ALIVE

    return score


def evaluate_slot(pokemon: Pokemon, is_trapped: bool, has_alive_reserve: bool) -> float:
    """Evaluate active slot including boosts and volatile statuses"""
    if pokemon.fainted:
        return 0.0

    score = 0.0

    # Volatile status effects
    if Effect.LEECH_SEED in pokemon.effects:
        score += LEECH_SEED
    if Effect.SUBSTITUTE in pokemon.effects:
        score += SUBSTITUTE
    if Effect.CONFUSION in pokemon.effects:
        score += CONFUSION

    # Perish Song effects (more severe if no reserves or trapped)
    if not has_alive_reserve or is_trapped:
        if Effect.PERISH3 in pokemon.effects:
            score += PERISH3
        elif Effect.PERISH2 in pokemon.effects:
            score += PERISH2
        elif Effect.PERISH1 in pokemon.effects:
            score += PERISH1

    # Stat boost evaluations
    score += get_boost_multiplier(pokemon.boosts.get("atk", 0)) * POKEMON_ATTACK_BOOST
    score += get_boost_multiplier(pokemon.boosts.get("def", 0)) * POKEMON_DEFENSE_BOOST
    score += (
        get_boost_multiplier(pokemon.boosts.get("spa", 0)) * POKEMON_SPECIAL_ATTACK_BOOST
    )
    score += (
        get_boost_multiplier(pokemon.boosts.get("spd", 0)) * POKEMON_SPECIAL_DEFENSE_BOOST
    )
    score += get_boost_multiplier(pokemon.boosts.get("spe", 0)) * POKEMON_SPEED_BOOST

    return score


def get_effective_speed_for_pokemon(
    pokemon: Pokemon, battle: Union[Battle, DoubleBattle], is_player_side: bool
) -> float:
    """
    Calculate effective speed using SpeedInference logic.
    This mirrors the get_effective_speed function from poke-engine.
    """
    if pokemon.fainted or not pokemon.stats or pokemon.stats.get("spe") is None:
        return 0.0

    # Get base speed with boosts
    speed_stat = pokemon.stats["spe"]
    if speed_stat is None:
        if pokemon.base_stats["spe"] >= 75:
            speed_stat = compute_raw_stats(
                pokemon.species, [252] * 6, [31] * 6, pokemon.level, "jolly", pokemon._data
            )[-1]
        else:
            speed_stat = compute_raw_stats(
                pokemon.species,
                [20] * 6,
                [31] * 6,
                pokemon.level,
                "serious",
                pokemon._data,
            )[-1]

    speed_boost = pokemon.boosts.get("spe", 0)

    # Apply boost multipliers
    boost_multipliers = {
        -6: 2.0 / 8,
        -5: 2.0 / 7,
        -4: 2.0 / 6,
        -3: 2.0 / 5,
        -2: 2.0 / 4,
        -1: 2.0 / 3,
        0: 1.0,
        1: 1.5,
        2: 2.0,
        3: 2.5,
        4: 3.0,
        5: 3.5,
        6: 4.0,
    }
    speed = speed_stat * boost_multipliers.get(speed_boost, 1.0)

    # Use SpeedInference.get_speed_multiplier for accuracy
    side_conditions = (
        battle.side_conditions if is_player_side else battle.opponent_side_conditions
    )

    multiplier = SpeedInference.get_speed_multiplier(
        mon=pokemon,
        weathers=battle.weather,
        side_conditions=side_conditions,
        fields=battle.fields,
        speed_boosts=speed_boost,
        item=pokemon.item,
        ability=pokemon.ability,
        status=pokemon.status,
        effects=pokemon.effects,
    )

    if multiplier is None:
        return 0.0  # Speed order affected by special effects

    return speed * abs(multiplier)  # Use abs to handle Trick Room


def evaluate_team_side(
    pokemon_team: Dict[str, Pokemon],
    active_pokemon: List[Optional[Pokemon]],
    side_conditions: Dict[SideCondition, int],
    battle: DoubleBattle,
    used_tera: bool,
) -> float:
    """Evaluate one side of the battle"""
    score = 0.0

    # Count alive reserves
    active_indices = set()
    for i, mon in enumerate(active_pokemon):
        if mon and not mon.fainted:
            # Find the team member that matches this active Pokemon
            for team_key, team_mon in pokemon_team.items():
                if team_mon.species == mon.species:
                    active_indices.add(team_key)
                    break

    has_alive_reserve = any(
        not mon.fainted for key, mon in pokemon_team.items() if key not in active_indices
    )

    # Evaluate all team members
    for pokemon in pokemon_team.values():
        if not pokemon.fainted:
            score += evaluate_pokemon(pokemon)
            score += evaluate_hazards(pokemon, side_conditions)

    # Evaluate active slots
    for i, mon in enumerate(active_pokemon):
        if mon is not None and isinstance(mon, Pokemon) and not mon.fainted:
            score += evaluate_slot(
                mon,
                battle.maybe_trapped[i] or battle.trapped[i],
                has_alive_reserve,
            )

    # Tera usage penalty
    if used_tera:
        score += USED_TERA

    # Side condition bonuses
    score += side_conditions.get(SideCondition.REFLECT, 0) * REFLECT
    score += side_conditions.get(SideCondition.LIGHT_SCREEN, 0) * LIGHT_SCREEN
    score += side_conditions.get(SideCondition.AURORA_VEIL, 0) * AURORA_VEIL
    score += side_conditions.get(SideCondition.SAFEGUARD, 0) * SAFE_GUARD
    score += side_conditions.get(SideCondition.TAILWIND, 0) * TAILWIND

    return score


def evaluate_state(battle: Union[Battle, DoubleBattle]) -> float:
    """
    Main evaluation function - returns score from player's perspective.
    Positive = good for player, negative = good for opponent.

    This implements the core evaluate() function from poke-engine-doubles.
    """
    if not isinstance(battle, DoubleBattle):
        raise ValueError("This evaluation function is designed for double battles")

    # Get active Pokemon lists
    player_active = battle.active_pokemon or [None, None]
    opponent_active = battle.opponent_active_pokemon or [None, None]

    # Check tera usage (simplified - would need battle history)
    player_used_tera = any(mon and mon.is_terastallized for mon in battle.team.values())
    opponent_used_tera = any(
        mon and mon.is_terastallized for mon in battle.opponent_team.values()
    )

    # Evaluate player side (positive contribution)
    player_score = evaluate_team_side(
        battle.team, player_active, battle.side_conditions, battle, player_used_tera
    )

    # Evaluate opponent side (negative contribution)
    opponent_score = evaluate_team_side(
        battle.opponent_team,
        opponent_active,
        battle.opponent_side_conditions,
        battle,
        opponent_used_tera,
    )

    score = player_score - opponent_score

    # Speed comparisons - core feature of poke-engine evaluation
    player_speeds = []
    opponent_speeds = []

    for mon in player_active:
        if mon and not mon.fainted:
            speed = get_effective_speed_for_pokemon(mon, battle, True)
            player_speeds.append(speed)

    for mon in opponent_active:
        if mon and not mon.fainted:
            speed = get_effective_speed_for_pokemon(mon, battle, False)
            opponent_speeds.append(speed)

    # Speed comparison matrix (all vs all)
    trick_room_active = Field.TRICK_ROOM in battle.fields
    faster_than_multiplier = 1.0

    if trick_room_active:
        # In Trick Room, slower Pokemon are favored
        # Value diminishes based on turns remaining (simplified to 4 turns)
        turns_remaining = battle.fields.get(Field.TRICK_ROOM, 4)
        faster_than_multiplier = 4.0 / max(turns_remaining, 1)

    # Compare all player Pokemon speeds vs all opponent Pokemon speeds
    for p_speed in player_speeds:
        for o_speed in opponent_speeds:
            if trick_room_active:
                player_is_faster = p_speed < o_speed  # Slower is better in TR
            else:
                player_is_faster = p_speed > o_speed  # Faster is better normally

            if player_is_faster:
                score += FASTER_THAN_OPPONENT * faster_than_multiplier
            else:
                score -= FASTER_THAN_OPPONENT * faster_than_multiplier

    return score


def evaluate_position_advantage(battle: Union[Battle, DoubleBattle]) -> float:
    """
    Wrapper function that returns the evaluation normalized to [-1, 1] range.
    This is suitable for use as advantage labels in battle datasets.
    """
    raw_score = evaluate_state(battle)

    # Normalize to reasonable range based on typical evaluation scores
    # Typical range is roughly [-500, 500] based on the constants
    max_reasonable_score = 500.0
    normalized = max(-1.0, min(1.0, raw_score / max_reasonable_score))

    return normalized
