# -*- coding: utf-8 -*-
"""This module contains general utils that contain categorizations of moves/abilities/items, as well as
    some other utils that are not specific to a particular inference method (eg showdown log parsing)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import Move
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory

_DISCERNABLE_ITEMS = set(
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

_MEGASTONES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS = {"charizarditey", "alakazite"}

_flags = {
    "has_status_move": False,
    "last_move": None,
    "num_moves_since_switch": 0,
    "num_moved": 0,
    "1.2atk": 0,
    "1.5atk": 0,
    "1.2spa": 0,
    "1.5spa": 0,
    "1.5def": 0,
    "1.5spd": 0,
    "1.5spe": 0,
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
_PRIORITY_ACTIVATION_ABILITIES = ["As One", "Neutralizing Gas", "Unnerve", "Tera Shift"]

MSGS_THAT_ACTIVATE_BEFORE_ATTACK = ["confusion", "slp", "par", "frz", "recharge", "flinch"]

_ITEMS_THAT_ACTIVATE_ON_SWITCH = [
    "Booster Energy",
    "Air Balloon",
    "Grassy Seed",
    "Electric Seed",
    "Misty Seed",
]

_ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS = {
    "Snow Warning",
    "Orachium Pulse",
    "Drought",
    "Desolate Land",
    "Hadron Engine",
    "Electric Surge",
    "Grassy Surge",
    "Misty Surge",
    "Pyschic Surge",
}


# Converts showdown message into dict key for self._mons
# "[of] p2a: Gardevoir" --> "p2a: Gardevoir"
# "p2a: Gardevoir" --> "p2a: Gardevoir"
def get_pokemon_ident(pokemon_str: str) -> str:
    groups = re.findall(r"(\[of\]\s)?(p[1-2][a-z]):\s(.*)", pokemon_str)
    if len(groups) == 0:
        raise ValueError("Unable to parse pokemon ident from " + pokemon_str)

    return groups[0][1] + ": " + groups[0][2].lower().title()


def get_pokemon(mon_ident: str, battle: Union[Battle, DoubleBattle]) -> Pokemon:
    if mon_ident in battle.team:
        return battle.team[mon_ident]
    elif mon_ident in battle.opponent_team:
        return battle.opponent_team[mon_ident]
    elif mon_ident[:2] + mon_ident[3:] in battle.team:
        return battle.team[mon_ident[:2] + mon_ident[3:]]
    elif mon_ident[:2] + mon_ident[3:] in battle.opponent_team:
        return battle.opponent_team[mon_ident[:2] + mon_ident[3:]]

    for mon in battle.teampreview_opponent_team:
        if (
            mon.species
            == "".join(char for char in mon_ident.split(": ")[1] if char.isalnum()).lower()
        ):
            return mon

    raise ValueError(
        f"Couldn't get a pokemon with ident {mon_ident} from \n"
        + f"\tOur team: {battle.team}"
        + f"\tOpponent team: {battle.opponent_team}"
        + f"\tOpponent Teampreview team: {battle.teampreview_opponent_team}"
    )


# Loads a battle to the specified turn; if turn is None, will initialize the battle
def load_battle(
    battle: Union[Battle, DoubleBattle], turn: Optional[int] = None
) -> Union[Battle, DoubleBattle]:
    b = DoubleBattle(
        battle.battle_tag,
        battle.player_username,
        logging.getLogger(battle.player_username),
        gen=battle.gen,
    )

    if turn:
        for i in range(0, turn):
            for event in battle.observations[i].events:
                update_battle(b, event)

    return b


# Updates a battle with an event
def update_battle(battle: Union[Battle, DoubleBattle], event: List[str]):
    if len(event) > 2 and event[1] not in ["", "t:"]:
        if event[1] == "win":
            battle.won_by(event[2])
        elif event[1] == "tie":
            battle.tied()
        else:
            battle.parse_message(event)


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
    if event[-1] == "ability: Tera Shift":
        return "Tera Shift", get_pokemon_ident(event[-2])
    elif "[from] ability: Frisk" in event:
        return "Frisk", get_pokemon_ident(event[-2])
    elif event[-1] == "ability: Protosynthesis":
        return "Protosynthesis", get_pokemon_ident(event[-2])
    elif event[2] == "ability: Forewarn":
        return "Forewarn", get_pokemon_ident(event[1])
    elif event[-1] == "[from] ability: Costar":
        return "Costar", get_pokemon_ident(event[1])
    elif event[-1] == "[from] ability: Curious Medicine":
        return "Curious Medicine", get_pokemon_ident(event[-1])
    elif event[1] in ["-weather", "-fieldstart"] and event[-2].startswith(
        "[from] ability: "
    ):
        return event[-2].replace("[from] ability: ", ""), get_pokemon_ident(event[-1])
    elif event[1] == "-ability":
        return event[3], get_pokemon_ident(event[2])
    else:
        return None, None


def get_priority_and_identifier(
    event: List[str], battle: Union[Battle, DoubleBattle]
) -> Tuple[str, Optional[str]]:

    mon_ident = get_pokemon_ident(event[2])

    # Removing positional data cuz of skill swap
    mon_ident = mon_ident[:2] + mon_ident[3:]

    mon = get_pokemon(mon_ident, battle)
    move = Move(re.sub("[^a-zA-Z]", "", event[3].lower()), battle.gen)

    # First set our priority to the move's
    priority = move.priority

    # Next, check abilities, assuming that mons will have a priority-affecting ability
    # Also check the actual ability in case of skillswap
    if (
        "galewings" in mon.possible_abilities or "galewings" == mon.ability
    ) and move.type == PokemonType.FLYING:
        priority += 1
    elif (
        "prankster" in mon.possible_abilities or "prankster" == mon.ability
    ) and move.category == MoveCategory.STATUS:
        priority += 1
    elif ("triage" in mon.possible_abilities or "triage" == mon.ability) and move.heal > 0:
        priority += 3
    elif (
        "myceliummight" in mon.possible_abilities or "myceliummight" == mon.ability
    ) and move.category == MoveCategory.STATUS:
        priority = None

    # Next, check edge cases:
    elif (
        len(event) > 3
        and event[3] == "Grassy Glide"
        and Field.GRASSY_TERRAIN in battle.fields
    ):
        priority = 1

    # Override if my place in the priority bracket is overriden
    if (
        Effect.QUASH in mon.effects
        or Effect.AFTER_YOU in mon.effects
        or Effect.QUICK_CLAW in mon.effects
        or Effect.CUSTAP_BERRY in mon.effects
        or Effect.DANCER in mon.effects
    ):
        priority = None

    return (mon_ident, priority)


# Takes in an event and returns the residual and the identifier of the mon who triggered the residual
# See unit_tests for examples
def get_residual_and_identifier(event: List[str]) -> Tuple[Optional[str], Optional[str]]:
    if event is None:
        return None, None
    elif event[-2] == "[from] Leech Seed":
        return "Leech Seed", get_pokemon_ident(event[2])
    elif event[1] == "-ability":
        return event[3], get_pokemon_ident(event[2])
    elif event[1] == "-end" and event[-2] == "Slow Start":
        return "Slow Start", get_pokemon_ident(event[2])
    elif event[-1] == "ability: Healer":
        return "Healer", get_pokemon_ident(event[2])
    elif event[-1] == "ability: Cud Chew":
        return "Cud Chew", get_pokemon_ident(event[2])
    elif event[-1] in ["perish0", "perish1", "perish2", "perish3"]:
        return "Perish Song", get_pokemon_ident(event[2])
    elif event[-1].startswith("[from] ability:"):
        return event[-1].replace("[from] ability: ", ""), get_pokemon_ident(event[2])
    elif event[-1].startswith("[from] item:"):
        return event[-1].replace("[from] item: ", ""), get_pokemon_ident(event[2])
    elif event[-1].startswith("[from] "):
        return event[-1].replace("[from] ", ""), get_pokemon_ident(event[2])
    elif event[-2].startswith("[from] move: "):
        return event[-2].replace("[from] move: ", ""), get_pokemon_ident(event[2])
    else:
        return None, None


# Turns go through phases, and this function will split up the events from showdown logs into different segments
# This will allow for easier parsing for various aspects of the game. The turns can be split
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
def get_segments(events: List[List[str]]) -> Dict[str, List[List[str]]]:
    indices, segments = {}, {}
    init = []
    last_segment = ""
    i = 0

    events = [event for event in events if len(event) > 1]

    # If at the start of the battle, we move directly to the post-turn phase
    if not any(map(lambda x: len(x) >= 2 and x[1] == "start", events)):

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
            not in ["-terastallize", "-dynamax", "-mega", "-primal", "switch", "move"]
            and events[i][-1] != "[upkeep]"
            and events[i][-1] not in MSGS_THAT_ACTIVATE_BEFORE_ATTACK
        ):
            i += 1

        # If we stop on a switch, we record the position
        if i < len(events) and events[i][1] == "switch":
            if last_segment != "":
                segments[last_segment] = events[indices[last_segment] : i]
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
            if last_segment != "":
                segments[last_segment] = events[indices[last_segment] : i]
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
            if last_segment != "":
                segments[last_segment] = events[indices[last_segment] : i]
            last_segment = "move"
            indices["move"] = i

        # Keep going until we get to the post-move phase (the empty event)
        # Sometimes there's an empty event because of pivot moves like U-turn
        while i < len(events) and not (
            events[i][1] == "" and len(events) > i + 1 and events[i + 1][1] != "switch"
        ):
            i += 1

        #  Once we hit the empty event, we record what we've gotten to and reset last_segment
        if i < len(events):
            if last_segment != "":
                segments[last_segment] = events[indices[last_segment] : i]
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
            if last_segment != "":
                segments[last_segment] = events[indices[last_segment] : i]
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

        last_segment = ""

    # At this point, we could have a switches from fainted mons or end of turn
    while i < len(events) and events[i][1] not in ["turn", "switch"]:
        init.append(events[i])
        i += 1

    # If we find a switch, we record it as a preturn_switch
    if i < len(events) and events[i][1] == "switch":
        last_segment = "preturn_switch"
        indices["preturn_switch"] = i

    # If we found a preturn switch and there's something to record, record it til the end
    # since this is the last segment
    if events[-1][1] == "turn" and last_segment == "preturn_switch":
        segments["preturn_switch"] = events[i:-1]

    # Record the last turn if theres a turn
    if events[-1][1] == "turn":
        segments["turn"] = [events[-1]]

    # These are events that we collected that no bearing on the battle; they are initial showdown
    # messages that initialize the battle
    if len(init) > 0:
        segments["init"] = init

    return segments
