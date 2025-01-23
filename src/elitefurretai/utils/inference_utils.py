# -*- coding: utf-8 -*-
"""This module contains general utils that contain categorizations of moves/abilities/items, as well as
    some other utils that are not specific to a particular inference method (eg showdown log parsing)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.weather import Weather
from poke_env.player.player import Player


# In Showdown, Weezing-Galar --> Weezing; Calyrex-Ice --> Calyrex
def get_showdown_identifier(mon: Pokemon, player_role: Optional[str]) -> str:
    if player_role is None:
        raise ValueError(
            "Trying to get identifier when player_role is none; for pokemon "
            + str(Pokemon)
        )

    return player_role + ": " + mon.name


# Converts showdown message into dict key for self._mons
# "[of] p2a: Gardevoir" --> "p2: Gardevoir"
# "p2a: Gardevoir" --> "p2: Gardevoir"
def standardize_pokemon_ident(pokemon_str: str) -> str:
    # Check if event is already standardized
    if pokemon_str.startswith("p1: ") or pokemon_str.startswith("p2: "):
        return pokemon_str

    groups = re.findall(r"(\[of\]\s)?(p[1-2][a-z]):\s(.*)", pokemon_str)
    if len(groups) == 0:
        raise ValueError("Unable to parse pokemon ident from " + pokemon_str)
    elif groups[0][1].endswith("a") or groups[0][1].endswith("b"):
        return groups[0][1][:2] + ": " + groups[0][2]
    else:
        return groups[0][1] + ": " + groups[0][2]


# Gets pokemon from a battle given a pokemon identifier; battle.team stores "p1: Furret"
# But this method will get: "p1a: Furret", "p1: Furret", "furret" from either our team
# or the opponent's
def get_pokemon(
    mon_ident: str, battle: Union[Battle, DoubleBattle, AbstractBattle]
) -> Pokemon:
    if mon_ident in battle.team:
        return battle.team[mon_ident]
    elif mon_ident in battle.opponent_team:
        return battle.opponent_team[mon_ident]
    elif mon_ident[:2] + mon_ident[3:] in battle.team:
        return battle.team[mon_ident[:2] + mon_ident[3:]]
    elif mon_ident[:2] + mon_ident[3:] in battle.opponent_team:
        return battle.opponent_team[mon_ident[:2] + mon_ident[3:]]

    raise ValueError(
        f"Couldn't get a pokemon with ident {mon_ident} from \n"
        + f"\tOur team: {battle.team}"
        + f"\n\tOpponent team: {battle.opponent_team}"
        + f"\n\tOpponent Teampreview team: {battle.teampreview_opponent_team}"
    )


# Note that this doesn't copy objects like Dicts
def copy_pokemon(orig: Pokemon, gen: int) -> Pokemon:
    mon = Pokemon(gen, species=orig.species)
    mon._active = orig._active
    mon._current_hp = orig._current_hp
    mon._effects = orig._effects
    mon._first_turn = orig._first_turn
    mon._gender = orig._gender
    mon._level = orig._level
    mon._max_hp = orig._max_hp
    mon._moves = {k: v for k, v in orig._moves.items()}
    mon._must_recharge = orig._must_recharge
    mon._ability = orig._ability
    mon._preparing_target = orig._preparing_target
    mon._preparing_move = orig._preparing_move
    mon._protect_counter = orig._protect_counter
    mon._revealed = orig._revealed
    mon._shiny = orig._shiny
    mon._status_counter = orig._status_counter
    mon._terastallized_type = orig._terastallized_type

    mon.item = orig.item
    mon._status = orig.status
    mon.stats = {k: v for k, v in orig.stats.items()}
    mon.boosts = {k: v for k, v in orig.boosts.items()}

    return mon


def copy_bare_battle(
    battle: Union[Battle, DoubleBattle, AbstractBattle], turn: Optional[int] = None
) -> Union[Battle, DoubleBattle]:
    b = DoubleBattle(
        battle.battle_tag + "elitefurretai",
        battle.player_username,
        logging.getLogger(battle.player_username),
        gen=battle.gen,
    )

    # Since this can't be replicated via events; it's populated by
    # player
    b.teampreview_team = {mon for mon in battle.teampreview_team}

    # Copy player information we have
    b._player_role = battle._player_role
    b._opponent_username = battle._opponent_username
    b._players = battle._players
    return b


# Updates a battle with an event
def update_battle(battle: Union[Battle, DoubleBattle, AbstractBattle], event: List[str]):
    if len(event) > 1 and event[1] not in ["", "t:"]:
        if event[1] == "win":
            battle.won_by(event[2])
        elif event[1] == "tie":
            battle.tied()
        else:
            battle.parse_message(event)


# Returns true if immunity to Sandstorm
def has_sandstorm_immunity(mon: Pokemon) -> bool:
    if mon.type_1 in SANDSTORM_IMMUNE_TYPES or mon.type_2 in SANDSTORM_IMMUNE_TYPES:
        return True

    elif mon.ability in ["sandforce", "sandrush", "sandveil", "magicguard", "overcoat"]:
        return True

    elif mon.ability is None and (
        "sandforce" in mon.possible_abilities
        or "sandrush" in mon.possible_abilities
        or "sandveil" in mon.possible_abilities
        or "magicguard" in mon.possible_abilities
        or "overcoat" in mon.possible_abilities
    ):
        return True

    elif mon.item == "safetygoggles":
        return True

    else:
        return False


# Returns true if immunity to flinching
def has_flinch_immunity(mon: Pokemon) -> bool:
    if mon.ability in ["shielddust", "innerfocus"]:
        return True
    elif mon.ability is None and (
        "shielddust" in mon.possible_abilities or "innerfocus" in mon.possible_abilities
    ):
        return True
    elif mon.item == "covertcloak":
        return True
    else:
        return False


# Returns whether a Pokemon is immune to psn/brn/par
def has_status_immunity(
    ident: str, status: Status, battle: Union[Battle, DoubleBattle, AbstractBattle]
) -> bool:
    if battle.opponent_role is None:
        return False

    mon = get_pokemon(ident, battle)

    # Already have a status
    if mon.status is not None or Effect.SUBSTITUTE in mon.effects:
        return True

    # Immune to all statuses from field conditions
    elif Field.MISTY_TERRAIN in battle.fields or SideCondition.SAFEGUARD in (
        battle.opponent_side_conditions
        if ident.startswith(battle.opponent_role)
        else battle.side_conditions
    ):
        return True

    # Ability is immune
    elif Weather.SUNNYDAY in battle.weather and mon.ability == "leafguard":
        return True
    elif (
        Weather.SUNNYDAY in battle.weather
        and mon.ability is None
        and "leafguard" in mon.possible_abilities
    ):
        return True

    # Ability is immune
    elif mon.ability in ["comatose", "goodasgold", "purifyingsalt", "shieldsdown"]:
        return True
    elif mon.ability is None and (
        "comatose" in mon.possible_abilities
        or "goodasgold" in mon.possible_abilities
        or "purifyingsalt" in mon.possible_abilities
        or "shieldsdown" in mon.possible_abilities
    ):
        return True

    # Check poison immunities
    elif status in [Status.PSN, Status.TOX]:
        if mon.ability == "immunity":
            return True
        elif mon.ability is None and "immunity" in mon.possible_abilities:
            return True
        elif mon.type_1 in {PokemonType.POISON, PokemonType.STEEL} or mon.type_2 in {
            PokemonType.POISON,
            PokemonType.STEEL,
        }:
            return True

    # Check burn immunities
    elif status == Status.BRN:
        if mon.ability in ["waterveil", "waterbubble"]:
            return True
        elif mon.ability is None and (
            "waterveil" in mon.possible_abilities
            or "waterbubble" in mon.possible_abilities
        ):
            return True
        elif PokemonType.FIRE in [mon.type_1, mon.type_2]:
            return True

    # Check paralysis immunities
    elif status == Status.PAR:
        if PokemonType.ELECTRIC in [mon.type_1, mon.type_2]:
            return True

    # Check Sleep immunities
    elif status == Status.SLP:
        if mon.ability in ["insomnia", "vitalspirit", "naturalcure", "sweetveil"]:
            return True
        elif mon.ability is None and (
            "insomnia" in mon.possible_abilities
            or "vitalspirit" in mon.possible_abilities
            or "sweetveil" in mon.possible_abilities
        ):
            return True
        elif Effect.SWEET_VEIL in mon.effects:
            return True
        elif Field.ELECTRIC_TERRAIN in battle.fields:
            return True
        elif any(
            map(
                lambda x: x is not None and Effect.UPROAR in x.effects,
                battle.all_active_pokemons,
            )
        ):
            return True

    # Check Freeze immunities
    elif status == Status.FRZ:
        if PokemonType.ICE in [mon.type_1, mon.type_2]:
            return True
        elif mon.item == "covertcloak":
            return True
        elif mon.ability in ["shielddust", "magmaarmor"]:
            return True
        elif mon.ability is None and (
            "shielddust" in mon.possible_abilities
            or "magmaarmor" in mon.possible_abilities
        ):
            return True
        elif Weather.SUNNYDAY in battle.weather:
            return True
    return False


# Returns whether a Pokemon is immune to the effects of Rage Powder
def has_rage_powder_immunity(mon: Pokemon) -> bool:
    if PokemonType.GRASS in [mon.type_1, mon.type_2]:
        return True

    elif mon.ability in ["sniper", "overcoat", "stalwart", "propellertail"]:
        return True

    elif mon.ability is None and (
        "sniper" in mon.possible_abilities
        or "overcoat" in mon.possible_abilities
        or "stalwart" in mon.possible_abilities
        or "propellertail" in mon.possible_abilities
    ):
        return True

    elif mon.item == "safetygoggles":
        return True

    return False


def has_unboost_immunity(
    mon_ident: str, boost_type: str, battle: Union[Battle, DoubleBattle, AbstractBattle]
) -> bool:
    mine = battle.player_role and mon_ident.startswith(battle.player_role)

    side_conditions = battle.opponent_side_conditions
    if mine:
        side_conditions = battle.side_conditions

    if SideCondition.MIST in side_conditions:
        return True

    if isinstance(battle, DoubleBattle):
        active_pokemon = battle.opponent_active_pokemon
        team = battle.opponent_team
        if mine:
            active_pokemon = battle.active_pokemon
            team = battle.team

        if (
            any(mon.ability == "flowerveil" for mon in active_pokemon if mon)
            and PokemonType.GRASS in team[mon_ident].types
        ):
            return True

        if (
            any(
                mon.ability is None and "flowerveil" in mon.possible_abilities
                for mon in active_pokemon
                if mon
            )
            and PokemonType.GRASS in team[mon_ident].types
        ):
            return True
    else:
        raise NotImplementedError()

    mon = get_pokemon(mon_ident, battle)
    if mon.item == "clearamulet":
        return True

    if mon.ability in ["clearbody", "fullmetalbody", "whitesmoke", "mirrorarmor"]:
        return True

    if mon.ability is None and (
        "clearbody" in mon.possible_abilities
        or "fullmetalbody" in mon.possible_abilities
        or "whitesmoke" in mon.possible_abilities
        or "mirrorarmor" in mon.possible_abilities
    ):
        return True

    if (
        mon.ability == "bigpeck"
        or (mon.ability is None and "bigpeck" in mon.possible_abilities)
    ) and boost_type == "def":
        return True

    if (
        mon.ability == "illuminate"
        or (mon.ability is None and "illuminate" in mon.possible_abilities)
    ) and boost_type in "accuracy":
        return True

    if (
        mon.ability == "keeneye"
        or (mon.ability is None and "keeneye" in mon.possible_abilities)
    ) and boost_type in "accuracy":
        return True

    if (
        mon.ability == "mindseye"
        or (mon.ability is None and "mindseye" in mon.possible_abilities)
    ) and boost_type in "accuracy":
        return True

    if (
        mon.ability == "hypercutter"
        or (mon.ability is None and "hypercutter" in mon.possible_abilities)
    ) and boost_type in "atk":
        return True

    if mon.boosts[boost_type] == -6:
        return True

    return False


# Assumes a mon that could be levitating has it
def is_grounded(
    mon_ident: str, battle: Union[Battle, DoubleBattle, AbstractBattle]
) -> bool:
    mon = get_pokemon(mon_ident, battle)
    if Field.GRAVITY in battle.fields:
        return True

    elif mon.item == "ironball":
        return True

    elif mon.ability == "levitate":
        return False

    elif mon.ability is None and "levitate" in mon.possible_abilities:
        return False

    elif mon.item == "airballoon":
        return False

    elif mon.type_1 == PokemonType.FLYING or mon.type_2 == PokemonType.FLYING:
        return False

    elif Effect.MAGNET_RISE in mon.effects:
        return False

    return True


# Tells you whether an event is ability-related
def is_ability_event(event: List[str]) -> bool:
    for e in event:
        if (
            e.startswith("[from] ability:")
            or e.startswith("-ability")
            or e.startswith("ability:")
        ):
            return True
    return False


# Takes in an event and returns the abillity and the identifier of the mon who triggered the ability
# See unit_tests for examples
def get_ability_and_identifier(event: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if len(event) > 2 and event[-1] == "ability: Tera Shift":
        return "Tera Shift", standardize_pokemon_ident(event[-2])
    elif len(event) > 1 and "[from] ability: Frisk" in event:
        return "Frisk", standardize_pokemon_ident(event[-1])
    elif len(event) > 2 and "ability: Protosynthesis" in event:
        return "Protosynthesis", standardize_pokemon_ident(event[2])
    elif len(event) > 2 and "ability: Quark Drive" in event:
        return "Quark Drive", standardize_pokemon_ident(event[2])
    elif len(event) > 3 and event[3] == "ability: Forewarn":
        return "Forewarn", standardize_pokemon_ident(event[2])
    elif len(event) > 2 and event[-1] == "[from] ability: Costar":
        return "Costar", standardize_pokemon_ident(event[2])
    elif event[-1] == "[from] ability: Curious Medicine":
        return "Curious Medicine", standardize_pokemon_ident(event[-1])
    elif (
        len(event) > 2
        and event[1] in ["-weather", "-fieldstart"]
        and event[-2].startswith("[from] ability: ")
    ):
        return event[-2].replace("[from] ability: ", ""), standardize_pokemon_ident(
            event[-1]
        )
    elif len(event) > 3 and event[1] == "-ability":
        return event[3], standardize_pokemon_ident(event[2])
    elif event[-1].startswith("ability: "):
        return event[-1].replace("ability: ", ""), standardize_pokemon_ident(event[2])
    else:
        return None, None


# Gets priority and identifier of mon making the move for an event returned by showdown.
# returns None if the priority shouldn't be read (e.g. a move is moved to the first/last
# in its priority bracket)
def get_priority_and_identifier(
    event: List[str], battle: Union[Battle, DoubleBattle, AbstractBattle]
) -> Tuple[str, Optional[int]]:

    mon_ident = standardize_pokemon_ident(event[2])

    mon = get_pokemon(mon_ident, battle)
    move = Move(re.sub("[^a-zA-Z]", "", event[3].lower()), battle.gen)

    # First set our priority to the move's
    priority = move.priority

    # Next, check abilities, assuming that mons will have a priority-affecting ability
    # Also check the actual ability in case of skillswap
    if (
        (
            ("galewings" in mon.possible_abilities and mon.ability is None)
            or "galewings" == mon.ability
        )
        and move.type == PokemonType.FLYING
        and mon.current_hp_fraction == 1.0
    ):
        priority += 1
    elif (
        ("prankster" in mon.possible_abilities and mon.ability is None)
        or "prankster" == mon.ability
    ) and move.category == MoveCategory.STATUS:
        priority += 1
    elif (
        ("triage" in mon.possible_abilities and mon.ability is None)
        or "triage" == mon.ability
    ) and (move.heal + move.drain > 0 or "heal" in move.flags):
        priority += 3
    elif (
        ("myceliummight" in mon.possible_abilities and mon.ability is None)
        or "myceliummight" == mon.ability
    ) and move.category == MoveCategory.STATUS:
        priority = None

    # Next, check edge cases:
    elif move.id == "grassyglide" and Field.GRASSY_TERRAIN in battle.fields:
        priority = 1

    # Override if my place in the priority bracket is overriden
    if (
        Effect.QUASH in mon.effects
        or Effect.AFTER_YOU in mon.effects
        or Effect.QUICK_CLAW in mon.effects
        or Effect.QUICK_DRAW in mon.effects
        or Effect.CUSTAP_BERRY in mon.effects
        or Effect.DANCER in mon.effects
        or mon.item == "laggingtail"
        or mon.item == "fullincense"
        or mon.ability == "stall"
        or move.id == "pursuit"
    ):
        priority = None

    return (mon_ident, priority)


# Takes in an event and returns the residual and the identifier of the mon who triggered the residual
# See unit_tests for examples
def get_residual_and_identifier(
    event: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    if event is None:
        return None, None
    elif event[-2] == "[from] Leech Seed":
        return "Leech Seed", standardize_pokemon_ident(event[2])
    elif event[1] == "-ability":
        return event[3], standardize_pokemon_ident(event[2])
    elif event[1] == "-end" and event[-2] == "Slow Start":
        return "Slow Start", standardize_pokemon_ident(event[2])
    elif event[-1] == "ability: Healer":
        return "Healer", standardize_pokemon_ident(event[2])
    elif event[-1] == "ability: Cud Chew":
        return "Cud Chew", standardize_pokemon_ident(event[2])
    elif event[-1] in ["perish0", "perish1", "perish2", "perish3"]:
        return "Perish Song", standardize_pokemon_ident(event[2])
    elif event[-1].startswith("[from] ability:"):
        return event[-1].replace("[from] ability: ", ""), standardize_pokemon_ident(
            event[2]
        )
    elif event[-1].startswith("[from] item:"):
        return event[-1].replace("[from] item: ", ""), standardize_pokemon_ident(event[2])
    elif event[-1].startswith("[from] "):
        return event[-1].replace("[from] ", ""), standardize_pokemon_ident(event[2])
    elif event[-2].startswith("[from] move: "):
        return event[-2].replace("[from] move: ", ""), standardize_pokemon_ident(event[2])
    else:
        return None, None


# Turns go through phases, and this function will split up the events from showdown logs into different
# segments. This will allow for easier parsing for various aspects of the game. The turns can be split
# into the following segments:
# ======= Start
# Inititalization ("init") <- messages before turn actually starts
# Activations ("activation") <- captures abilities that activate at beginning of turn
# ======== Player Actions ========
# Switches ("switch") <- also includes events for ability/item activations upon switch
# Tera/mega/dynamax/primal ("battle_mechanic")
# Moves, by priority ("move")
# Residuals ("residual")
# ======== Start of Battle or Switching in after Faint ======== ("|upkeep")
# Start of battle / end-of-turn switches for fainted mons (None)
# Preturn switch ("preturn_switch") <- only includes events for ability/item activations upon switch
# Turn ("turn") <- captures event changing turn (if present)
def get_segments(events: List[List[str]], start=0) -> Dict[str, List[List[str]]]:
    indices, segments = {}, {}
    init = []
    last_segment = ""
    i = 0

    # Remove empty events
    events = [event for event in events if len(event) > 1]

    start_of_battle = any(map(lambda x: len(x) >= 2 and x[1] == "start", events))

    # If at the start of the battle, we move directly to the post-turn phase
    if not start_of_battle:

        # We know there has to be a switch or a move; we go until we get there, but if we get a battle_mechanic,
        # we stop
        while (
            i < len(events)
            and events[i][1]
            not in [
                "-terastallize",
                "-dynamax",
                "-mega",
                "-primal",
                "-activate",
                "-end",  # for protosynthesis/quarkdrives end before switch
                "switch",
                "move",
            ]
            and events[i][-1] != "[upkeep]"
            and events[i][-1] not in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            init.append(events[i])
            i += 1

        if i < len(events) and events[i][1] == "-activate":
            last_segment = "activation"
            indices["activation"] = i

        while (
            i < len(events)
            and events[i][1]
            not in [
                "-terastallize",
                "-dynamax",
                "-mega",
                "-primal",
                "switch",
                "move",
                "-end",
            ]
            and events[i][-1] != "[upkeep]"
            and events[i][-1] not in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            i += 1

        # If we stop on a switch, we record the position
        if i < len(events) and (
            events[i][1] == "switch"
            or (
                # for protosynthesis/quarkdrives/neutralizinggas; note that this is a bug in showdown:
                # https://github.com/smogon/pokemon-showdown/blob/c82d1b8433440256f7b75fe1c68deac31a29c09d/data/abilities.ts#L2877
                events[i][1] == "-end"
                and len(events) > 3
            )
        ):
            if last_segment != "" and i >= start:
                segments[last_segment] = events[max(indices[last_segment], start) : i]
            last_segment = "switch"
            indices["switch"] = i

        # We keep on going until we get to battle mechanics or moves (player actions)
        while (
            i < len(events)
            and events[i][1] not in ["-terastallize", "-dynamax", "-mega", "move"]
            and events[i][-1] != "[upkeep]"
            and events[i][-1] not in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            i += 1

        # If we stop on a battle_mechanic, we record the position
        if i < len(events) and events[i][1] in ["-terastallize", "-dynamax", "-mega"]:
            if last_segment != "" and i >= start:
                segments[last_segment] = events[max(indices[last_segment], start) : i]
            last_segment = "battle_mechanic"
            indices["battle_mechanic"] = i

        # Keep going until we get to a move, residuals or upkeep
        while (
            i < len(events)
            and events[i][1] not in ["move", "", "upkeep"]
            and events[i][-1] not in ["[upkeep]", "none"]
            and events[i][-1] not in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            i += 1

        # If we stop on a move, we record the position
        if i < len(events) and (
            events[i][1] == "move" or events[i][-1] in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            if last_segment != "" and i >= start:
                segments[last_segment] = events[max(indices[last_segment], start) : i]
            last_segment = "move"
            indices["move"] = i

        # Keep going until we get to the post-move phase (the empty event)
        # Sometimes there's an empty event because of pivot moves like U-turn. Also
        # there can be an event: ['', '-end', 'p1b: Iron Hands', 'Quark Drive', '[silent]']
        # before it swithces out. So this tries to keep on going until we see an empty string
        # that isnt immediately followed by a switch or has a switch two after
        while i < len(events) and (
            events[i][1] != ""
            or (
                events[i][1] == ""
                and (
                    (len(events) > i + 1 and events[i + 1][1] == "switch")
                    or (
                        len(events) > i + 2
                        and events[i + 2][1] == "switch"
                        and events[i + 1][1] == "-end"
                    )
                )
            )
        ):
            i += 1

        #  Once we hit the empty event, we record what we've gotten to and reset last_segment
        if i < len(events):
            if last_segment != "" and i >= start:
                segments[last_segment] = events[max(indices[last_segment], start) : i]
            last_segment = ""
            i += 1

        # If we are find state upkeep, we should keep going until we're out
        if i < len(events) and (
            events[i][-1] in ["[upkeep]", "none"]
            or events[i][1] in ["-sideend", "-fieldend", "-end"]
        ):
            last_segment = "state_upkeep"
            indices["state_upkeep"] = i

            while i < len(events) and (
                events[i][-1] in ["[upkeep]", "none"]
                or events[i][1] in ["-sideend", "-fieldend", "-end"]
            ):
                i += 1

        # If after going through state upkeep, we're not at upkeep, this means we're at residuals
        if i < len(events) and events[i][1] != "upkeep":
            if last_segment != "" and i >= start:
                segments[last_segment] = events[max(indices[last_segment], start) : i]
            last_segment = "residual"
            indices["residual"] = i

        # Every turn has an upkeep, so we get to this checkpoint
        while i < len(events) and events[i][1] != "upkeep":
            i += 1

        # Reset everything because we're at the checkpoint, ensuring we stop recording
        # whatever we were. If there's no upkeep and we're at the end of events, it means
        # the battle ended, and we shouldnt record last_segment="residual", which triggered above
        if last_segment != "" and indices[last_segment] != i:
            segments[last_segment] = events[indices[last_segment] : i]

        last_segment = "post_upkeep"
        indices["post_upkeep"] = i

    # At this point, we could have a switches from fainted mons or end of turn
    while i < len(events) and events[i][1] not in ["turn", "switch"]:
        if start_of_battle:
            init.append(events[i])
        i += 1

    # If we have post_upkeep
    if last_segment == "post_upkeep" and i > max(indices[last_segment], start):
        segments[last_segment] = events[max(indices[last_segment], start) : -1]

    # If we find a switch, we record it as a preturn_switch
    if i < len(events) and events[i][1] == "switch":
        if last_segment != "" and i >= start and max(indices[last_segment], start) != i:
            segments[last_segment] = events[max(indices[last_segment], start) : i]
        last_segment = "preturn_switch"
        indices["preturn_switch"] = i

    # If we found a preturn switch and there's something to record, record it til the end
    # since this is the last segment
    if events[-1][1] == "turn" and last_segment == "preturn_switch":
        segments["preturn_switch"] = events[max(i, start) : -1]

    # Record the last turn if theres a turn
    if events[-1][1] == "turn" and len(events) > start:
        segments["turn"] = [events[-1]]

    # These are events that we collected that no bearing on the battle; they are initial showdown
    # messages that initialize the battle
    if len(init) > 0:
        segments["init"] = init

    return segments


def observation_to_str(obs):
    message = ""
    message += f"\n\tMy Active Mon:  [{', '.join(map(lambda x: x.species if x else 'None', obs.active_pokemon)) if obs.active_pokemon else ''}]"
    message += f"\n\tOpp Active Mon: [{', '.join(map(lambda x: x.species if x else 'None', obs.opponent_active_pokemon)) if obs.opponent_active_pokemon else ''}]"
    message += f"\n\tWeather: [{', '.join(map(lambda x: x.name, obs.weather))}]"
    message += f"\n\tFields: [{', '.join(map(lambda x: x.name, obs.fields))}]"
    message += f"\n\tMy Side Conditions:  [{', '.join(map(lambda x: x.name, obs.side_conditions))}]"
    message += f"\n\tOpp Side Conditions: [{', '.join(map(lambda x: x.name, obs.opponent_side_conditions))}]"

    message += "\n\tMy Team:"
    for ident, mon in obs.team.items():
        message += f"\n\t\t{ident} => [Speed: {mon.stats['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"

    message += "\n\tOpp Team:"
    for ident, mon in obs.opponent_team.items():
        message += f"\n\t\t{ident} => [Speed: {mon.stats['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"

    message += "\n\n\tEvents:"
    if len(obs.events) == 0:
        message += "\n\t\t(No events yet! You are at the beginning of this turn.)"
    else:
        for event in obs.events:
            message += f"\n\t\t{event}"

    return message


def battle_to_str(battle, opp: Optional[Player] = None) -> str:

    message = f"============= Battle [{battle.battle_tag}] =============\n"
    message += f"The battle is between {battle.player_username} and {battle.opponent_username} from {battle.player_username}'s perspective.\n"

    message += "P1 Teampreview Team (omniscient): ["
    for mon in battle.teampreview_team:
        ident = get_showdown_identifier(mon, battle.player_role)
        mon = battle.team.get(ident, mon)
        message += f"\n\t{mon.name} => "
        message += "[Speed: " + str(mon.stats["spe"])
        message += f" // Item: {mon.item}]"
    message += "]\n"

    opp_teampreview_team = battle.teampreview_opponent_team
    opp_team = battle.opponent_team
    if opp is not None:
        opp_teampreview_team = opp.battles[battle.battle_tag].teampreview_team
        opp_team = opp.battles[battle.battle_tag].team

    message += "P2 Teampreview Team (not omniscient): ["
    for mon in opp_teampreview_team:
        ident = get_showdown_identifier(mon, battle.opponent_role)
        mon = opp_team.get(ident, mon)
        message += f"\n\t{mon.name} => "
        message += "[Speed: " + str(mon.stats["spe"])
        message += f" // Item: {mon.item}]"
    message += "]\n"

    last_obs = None
    for turn, obs in battle.observations.items():
        message += f"\n\nTurn #{turn}:"
        message += observation_to_str(obs)
        last_obs = obs

    if last_obs is not None and battle._current_observation.events != last_obs.events:
        message += f"\n\nCurrent Observation; Turn #{battle.turn}:"
        message += observation_to_str(battle._current_observation)

    return message


DISCERNABLE_ITEMS = set(
    [
        "choiceband",
        "choicescarf",
        "choicespecs",
        "clearamulet",
        "covertcloack",
        "eviolite",
        "expertbelt",
        "gripclaw",
        "heavydutyboots",
        "ironball",
        "kingsrock",
        "laggingtail",
        "lightclaw",
        "machobrace",
        "muscleband",
        "protectivepads",
        "safetygoggles",
        "shedshell",
        "terrainextender",
        "utilityumbrella",
        "wiseglasses",
        "dragonfang",
        "hardstone",
        "magnet",
        "metalcoat",
        "mysticwater" "charcoal",
        "miracleseed",
        "nevermeltice",
        "blackglasses",
        "blackbelt",
        "pixieplate",
        "sharpbeak",
        "spelltag",
        "oddinsence" "silkscarf",
        "silverpowder",
        "softsand",
        "spelltag",
        "twistedspoon",
    ]
)

MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS = {
    "charizarditey",
    "alakazite",
}

# From https://github.com/search?q=repo%3Asmogon%2Fpokemon-showdown+residualorder&type=code
FIRST_BLOCK_RESIDUALS = {
    "Sandstorm",
    "Hail",
    "Solar Power",
    "Dry Skin",
    "Ice Body",
    "Rain Dish",
}

SECOND_BLOCK_RESIDUALS = {
    "Grassy Terrain",
    "Healer",
    "Hydration",
    "Shed Skin",
    "Black Sludge",
    "Leftovers",
}

THIRD_BLOCK_RESIDUALS = {
    "Uproar",
    "CudChew",
    "Harvest",
    "Moody",
    "Slow Start",
    "Speed Boost",
    "Flame Orb",
    "Sticky Barb",
    "Toxic Orb",
}

RESIDUALS_WITH_OWN_PRIORITY = {
    "Aqua Ring",
    "Ingrain",
    "Leech Seed",
    "Poison",
    "brn",
    "Curse",
    "tox",
    "psn",
    "Bind",
    "Octolock",
    "Salt Cure",
    "Taunt",
    "Torment",
    "Encore",
    "Disable",
    "Magnet Rise",
    "Yawn",
    "Perish count",
    "Bind",
    "Whirlpool",
    "Infestation",
    "Clamp",
    "Fire Spin",
    "Thunder Cage",
    "Sand Tomb",
    "Snap Trap",
    "Wrap",
    "Magma Storm",
    "Perish Song",
}

# Denoted by preStart in https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts
PRIORITY_ACTIVATION_ABILITIES = ["As One", "Neutralizing Gas", "Unnerve", "Tera Shift"]

MSGS_THAT_ACTIVATE_BEFORE_ATTACK = [
    "confusion",
    "slp",
    "par",
    "frz",
    "recharge",
    "flinch",
]

ITEMS_THAT_ACTIVATE_ON_SWITCH = [
    "Booster Energy",
    "Air Balloon",
    "Grassy Seed",
    "Electric Seed",
    "Misty Seed",
]

ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS = {
    "Snow Warning",
    "Orichalcum Pulse",
    "Drought",
    "Desolate Land",
    "Hadron Engine",
    "Electric Surge",
    "Grassy Surge",
    "Misty Surge",
    "Pyschic Surge",
}

SANDSTORM_IMMUNE_TYPES = {PokemonType.ROCK, PokemonType.STEEL, PokemonType.GROUND}
