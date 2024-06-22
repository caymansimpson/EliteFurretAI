# -*- coding: utf-8 -*-
"""This module contains general utils that contain categorizations of moves/abilities/items, as well as
    some other utils that are not specific to a particular inference method (eg showdown log parsing)
"""

import re
from typing import List, Optional, Tuple

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

# TODO: add from analysis of logs; first we can go through top 50 pokemon and record theirs
_ITEMS = set([])

# Denoted by preStart in https://github.com/smogon/pokemon-showdown/blob/master/data/abilities.ts
_PRIORITY_ACTIVATION_ABILITIES = ["As One", "Neutralizing Gas", "Unnerve", "Tera Shift"]

_ITEMS_THAT_ACTIVATE_ON_SWITCH = ["Booster Energy", "Air Balloon"]

_ABILITIES_THAT_CAN_PUBLICLY_ACTIVATE_ABILITIES_OR_ITEMS = [
    "Snow Warning",
    "Orachium Pulse",
    "Drought",
    "Desolate Land",
    "Hadron Engine",
    "Electric Surge",
    "Grassy Surge",
    "Misty Surge",
    "Pyschic Surge",
]


# Converts showdown message into dict key for self._mons
# "[of] p2a: Gardevoir" --> "p2: Gardevoir"
# "p2a: Gardevoir" --> "p2: Gardevoir"
def get_pokemon_ident(pokemon_str: str) -> str:
    groups = re.findall(r"(\[of\]\s)?(p[1-2])[a-z]:\s(.*)", pokemon_str)
    if len(groups) == 0:
        raise ValueError("Unable to parse pokemon ident from " + pokemon_str)

    return groups[0][1] + ": " + groups[0][2].lower().title()


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
# Example events:
# ['', '-item', 'p2a: Gardevoir', 'Iron Ball', '[from] ability: Frisk', '[of] p1a: Furret', '[identify]']
# ['', '-weather', 'Sandstorm', '[from] ability: Sand Stream', '[of] p1b: Tyranitar']
# ['', '-fieldstart', 'move: Electric Terrain', '[from] ability: Electric Surge', '[of] p1a: Pincurchin']
# ['', '-ability', 'p2a: Gardevoir', 'Sand Stream', '[from] ability: Trace', '[of] p1b: Tyranitar']
# ['', '-activate', 'p2b: Fluttermane', 'ability: Protosynthesis']
# ['-copyboost', 'p2a: Flamigo', 'p2b: Furret', '[from] ability: Costar']
# ['-clearboost', 'p2a: Flamigo', '[from] ability: Curious Medicine', '[of] p2b: Slowking']
# ['-activate', 'p2b: Hypno', 'ability: Forewarn', 'darkpulse, '[of] p1a: Chi-Yu']
# ['', '-activate', 'p1b: Terapagos', 'ability: Tera Shift']
# ['', '-ability', 'p1b: Calyrex', 'As One']
# ['', '-ability', 'p1b: Calyrex', 'Unnerve']
# ['', '-ability', 'p2b: Weezing', 'Neutralizing Gas']
def parse_ability(event: List[str]) -> Tuple[Optional[str], Optional[str]]:
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
