# -*- coding: utf-8 -*-
"""Unit Tests to ensure Speed Inference can detect Choice Scarfs
"""
from unittest.mock import MagicMock

import pytest
from poke_env.environment import (
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.battle_inference.speed_inference import SpeedInference, get_pokemon


def generate_speed_inference():
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    si = SpeedInference(battle=battle, opponent_mons={})
    si._battle.player_role = "p1"
    return si


def test_parse_preturn_switch():
    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Wo-Chien", "Wo-Chien, L50", "160/160"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Rillaboom", "Rillaboom, L50, F", "100/100"],
        ["", "switch", "p2b: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "-item", "p1b: Raichu", "Air Balloon"],
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p2a: Rillaboom",
        ],
        ["", "-enditem", "p2a: Rillaboom", "Grassy Seed"],
        ["", "-boost", "p2a: Rillaboom", "def", "1", "[from] item: Grassy Seed"],
        ["", "-ability", "p1a: Wo-Chien", "Tablets of Ruin"],
    ]

    si._parse_preturn_switch(events)
    assert si._orders == []
    assert Field.GRASSY_TERRAIN in si._battle.fields
    assert get_pokemon("p2a: Rillaboom", si._battle).item is None
    assert get_pokemon("p1b: Raichu", si._battle).item == "airballoon"

    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Wo-Chien", "Wo-Chien, L50", "160/160"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, M", "167/167"],
        ["", "switch", "p2a: Rillaboom", "Rillaboom, L50, M", "100/100"],
        ["", "switch", "p2b: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "-ability", "p1a: Wo-Chien", "Tablets of Ruin"],
        ["", "-item", "p1b: Raichu", "Air Balloon"],
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p2a: Rillaboom",
        ],
        ["", "-enditem", "p2a: Rillaboom", "Grassy Seed"],
        ["", "-boost", "p2a: Rillaboom", "def", "1", "[from] item: Grassy Seed"],
    ]

    orders = si.clean_orders(si._parse_preturn_switch(events))
    assert [("p1: Wo-Chien", 1.0), ("p1: Raichu", 1.0)] in orders
    assert [("p1: Raichu", 1.0), ("p2: Rillaboom", 1.0)] in orders
    assert len(orders) == 2
    assert Field.GRASSY_TERRAIN in si._battle.fields
    assert get_pokemon("p2a: Rillaboom", si._battle).item is None
    assert get_pokemon("p2a: Rillaboom", si._battle).boosts["def"] == 1

    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Wo-Chien", "Wo-Chien, L50", "160/160"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, M", "167/167"],
        ["", "switch", "p2a: Rillaboom", "Rillaboom, L50, F", "100/100"],
        ["", "switch", "p2b: Pincurchin", "Pincurchin, L50, M", "100/100"],
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p2a: Rillaboom",
        ],
        ["", "-enditem", "p1b: Raichu", "Grassy Seed"],
        ["", "-boost", "p1b: Raichu", "def", "1", "[from] item: Grassy Seed"],
        ["", "-enditem", "p2a: Rillaboom", "Grassy Seed"],
        ["", "-boost", "p2a: Rillaboom", "def", "1", "[from] item: Grassy Seed"],
        ["", "-ability", "p1a: Wo-Chien", "Tablets of Ruin"],
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Electric Surge",
            "[of] p2b: Pincurchin",
        ],
        ["", "-enditem", "p1a: Wo-Chien", "Electric Seed"],
        ["", "-boost", "p1a: Wo-Chien", "def", "1", "[from] item: Electric Seed"],
        ["", "-enditem", "p2b: Pincurchin", "Electric Seed"],
        ["", "-boost", "p2b: Pincurchin", "def", "1", "[from] item: Electric Seed"],
    ]
    orders = si.clean_orders(si._parse_preturn_switch(events))

    assert [("p2: Rillaboom", 1.0), ("p1: Wo-Chien", 1.0)] in orders
    assert [("p1: Raichu", 1.0), ("p2: Rillaboom", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Pincurchin", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Pincurchin", 1.0)] in orders
    assert len(orders) == 4
    assert Field.ELECTRIC_TERRAIN in si._battle.fields
    assert get_pokemon("p2a: Rillaboom", si._battle).item is None
    assert get_pokemon("p2b: Pincurchin", si._battle).boosts["def"] == 1

    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Terapagos", "Terapagos, L50, M", "165/165"],
        ["", "switch", "p1b: Weezing", "Weezing-Galar, L50, M", "140/140"],
        ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50", "100/100"],
        ["", "switch", "p2b: Arbok", "Arbok, L50, M", "100/100"],
        ["", "-ability", "p2a: Calyrex", "As One"],
        ["", "-ability", "p2a: Calyrex", "Unnerve"],
        ["", "-activate", "p1a: Terapagos", "ability: Tera Shift"],
        ["", "detailschange", "p1a: Terapagos", "Terapagos-Terastal, L50, M"],
        ["", "-heal", "p1a: Terapagos", "170/170", "[silent]"],
        ["", "-ability", "p1b: Weezing", "Neutralizing Gas"],
    ]

    orders = si.clean_orders(si._parse_preturn_switch(events))
    assert [("p2: Calyrex", 1.0), ("p1: Terapagos", 1.0)] in orders
    assert [("p1: Terapagos", 1.0), ("p1: Weezing", 1.0)] in orders

    # Ensure I parse things right; this was a bug
    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M, tera:Rock", "32/130"],
        ["", "-damage", "p1a: Smeargle", "0 fnt", "[from] Spikes"],
        ["", "faint", "p1a: Smeargle"],
        ["", ""],
        ["", "switch", "p1a: Raichu", "Raichu, L50, F", "60/135"],
        ["", "-damage", "p1a: Raichu", "27/135", "[from] Spikes"],
    ]
    orders = si.clean_orders(si._parse_preturn_switch(events))
    assert not any(map(lambda x: x and x.species == "smeargle", si._battle.active_pokemon))
    assert any(map(lambda x: x and x.species == "raichu", si._battle.active_pokemon))
    assert len(orders) == 0


# Too lazy to test mega that might trigger some other series of abilities
def test_parse_battle_mechanic():
    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p1a: Terapagos", "Terapagos, L50, M", "165/165"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50", "100/100"]
    )
    events = [
        ["", "-terastallize", "p2a: Calyrex", "Psychic"],
        ["", "-terastallize", "p1a: Terapagos", "Stellar"],
        ["", "detailschange", "p1a: Terapagos", "Terapagos-Stellar, L50, M, tera:Stellar"],
        ["", "-heal", "p1a: Terapagos", "235/235", "[silent]"],
    ]
    orders = si.clean_orders(si._parse_battle_mechanic(events))
    assert [("p2: Calyrex", 1.0), ("p1: Terapagos", 1.0)] in orders
    assert len(orders) == 1

    si = generate_speed_inference()
    si._battle._side_conditions = {SideCondition.TAILWIND: 1}
    si._battle.parse_message(
        ["", "switch", "p1a: Terapagos", "Terapagos, L50, M", "165/165"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50", "100/100"]
    )
    events = [
        ["", "-terastallize", "p1a: Terapagos", "Stellar"],
        ["", "detailschange", "p1a: Terapagos", "Terapagos-Stellar, L50, M, tera:Stellar"],
        ["", "-heal", "p1a: Terapagos", "235/235", "[silent]"],
        ["", "-terastallize", "p2a: Calyrex", "Psychic"],
    ]
    orders = si.clean_orders(si._parse_battle_mechanic(events))
    assert [("p1: Terapagos", 2.0), ("p2: Calyrex", 1.0)] in orders
    assert len(orders) == 1


def test_parse_move():
    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Smeargle", "Smeargle, L50", "160/160"])
    si._battle.parse_message(
        ["", "switch", "p1b: Wo-Chien", "Wo-Chien, L50, F", "167/167"]
    )
    si._battle.parse_message(["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"])
    si._battle.parse_message(["", "switch", "p2b: Sentret", "Sentret, L50, M", "100/100"])
    events = [
        ["", "move", "p2a: Furret", "Trailblaze", "p2b: Sentret", "[notarget]"],
        ["", "-fail", "p2a: Furret"],
        ["", "move", "p1a: Smeargle", "Wrap", "p2a: Furret", "[miss]"],
        ["", "-miss", "p1a: Smeargle", "p2a: Furret"],
        ["", "move", "p1b: Wo-Chien", "Leech Seed", "p2a: Furret"],
        ["", "-start", "p2a: Furret", "move: Leech Seed"],
    ]
    orders = si.clean_orders(si._parse_move(events))
    assert [("p2: Furret", 1.0), ("p1: Smeargle", 1.0)] in orders
    assert [("p1: Smeargle", 1.0), ("p1: Wo-Chien", 1.0)] in orders
    assert len(orders) == 2

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50", "160/160 par"])
    si._battle.parse_message(["", "switch", "p1b: Sentret", "Sentret, L50, F", "167/167"])
    si._battle.parse_message(
        ["", "switch", "p2a: Wo-Chien", "Wo-Chien, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2b: Furret", "Furret, L50, M", "100/100"])
    events = [
        ["", "move", "p1b: Sentret", "Double Edge", "p1a: Furret"],
        ["", "-crit", "p1a: Furret"],
        ["", "-damage", "p1a: Furret", "142/160 par"],
        ["", "cant", "p1a: Furret", "par"],
        ["", "move", "p2b: Furret", "U-turn", "p1a: Furret"],
        ["", "-damage", "p1a: Furret", "109/160 par"],
        ["", ""],
        ["", "switch", "p2b: Sentret", "Sentret, L50, M", "100/100", "[from] U-turn"],
        [
            "",
            "-item",
            "p1a: Furret",
            "Leftovers",
            "[from] ability: Frisk",
            "[of] p2b: Sentret",
        ],
        [
            "",
            "-item",
            "p1b: Sentret",
            "Flame Orb",
            "[from] ability: Frisk",
            "[of] p2b: Sentret",
        ],
        ["", "move", "p2a: Wo-Chien", "Stun Spore", "p1a: Furret"],
    ]
    orders = si.clean_orders(si._parse_move(events))
    assert [("p1: Sentret", 1.0), ("p1: Furret", 0.5)] in orders
    assert [("p1: Furret", 0.5), ("p2: Furret", 1.0)] in orders
    assert [("p2: Furret", 1.0), ("p2: Wo-Chien", 1.0)] in orders
    assert len(orders) == 3

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50", "160/160 par"])
    si._battle.parse_message(["", "switch", "p1b: Sentret", "Sentret, L50, F", "167/167"])
    si._battle.parse_message(
        ["", "switch", "p2a: Wo-Chien", "Wo-Chien, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2b: Furret", "Furret, L50, M", "100/100"])
    events = [
        ["", "move", "p1b: Sentret", "Quick Attack", "p1a: Furret"],
        ["", "-crit", "p1a: Furret"],
        ["", "-damage", "p1a: Furret", "142/160 par"],
        ["", "cant", "p1a: Furret", "par"],
        ["", "move", "p2b: Furret", "U-turn", "p1a: Furret"],
        ["", "-damage", "p1a: Furret", "109/160 par"],
        ["", ""],
        ["", "switch", "p2b: Sentret", "Sentret, L50, M", "100/100", "[from] U-turn"],
        [
            "",
            "-item",
            "p1a: Furret",
            "Leftovers",
            "[from] ability: Frisk",
            "[of] p2b: Sentret",
        ],
        [
            "",
            "-item",
            "p1b: Sentret",
            "Flame Orb",
            "[from] ability: Frisk",
            "[of] p2b: Sentret",
        ],
        ["", "move", "p2a: Wo-Chien", "Stun Spore", "p1a: Furret"],
        ["", "-fail", "p1a: Furret", "par"],
    ]
    orders = si.clean_orders(si._parse_move(events))
    assert [("p2: Furret", 1.0), ("p2: Wo-Chien", 1.0)] in orders
    assert len(orders) == 1

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50, F", "167/167"])
    si._battle.parse_message(
        ["", "switch", "p1b: Volcarona", "Volcarona, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Oricorio", "Oricorio, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2b: Sentret", "Sentret, L50, F", "100/100"])
    events = [
        ["", "move", "p1b: Volcarona", "Quiver Dance", "p1b: Volcarona"],
        ["", "-boost", "p1b: Volcarona", "spa", "1"],
        ["", "-boost", "p1b: Volcarona", "spd", "1"],
        ["", "-boost", "p1b: Volcarona", "spe", "1"],
        ["", "-activate", "p2a: Oricorio", "ability: Dancer"],
        [
            "",
            "move",
            "p2a: Oricorio",
            "Quiver Dance",
            "p2a: Oricorio",
            "[from]ability: Dancer",
        ],
        ["", "-boost", "p2a: Oricorio", "spa", "1"],
        ["", "-boost", "p2a: Oricorio", "spd", "1"],
        ["", "-boost", "p2a: Oricorio", "spe", "1"],
        ["", "move", "p1a: Furret", "Agility", "p1a: Furret"],
        ["", "-damage", "p2b: Sentret", "96/100", "[from] confusion"],
        ["", "move", "p2a: Oricorio", "Agility", "p2a: Oricorio"],
    ]
    orders = si.clean_orders(si._parse_move(events))
    assert [("p1: Volcarona", 1.0), ("p1: Furret", 1.0)] in orders
    # TODO: need to fix Oricorio Dancer abilities first
    # assert [('p1: Furret', 1.0), ('p2: Sentret', 1.0)] in orders
    # assert [('p2: Sentret', 1.0), ('p2: Oricorio', 1.5)] in orders
    # assert len(orders) == 3

    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p2b: Smeargle", "Smeargle, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p1a: Tyranitar", "Tyranitar, L50, F", "167/167 par"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50, F", "167/167"]
    )
    events = [
        ["", "move", "p2b: Smeargle", "U-turn", "p1a: Tyranitar"],
        ["", "-supereffective", "p1a: Tyranitar"],
        ["", "-damage", "p1a: Tyranitar", "81/175 par"],
        ["", "cant", "p1a: Tyranitar", "par"],
        ["", "move", "p2a: Tyranitar", "Dragon Tail", "p2b: Smeargle"],
        ["", "-damage", "p2b: Smeargle", "0 fnt"],
        ["", "faint", "p2b: Smeargle"],
    ]
    orders = si.clean_orders(si._parse_move(events))
    assert len(orders) == 0


def test_parse_residual():
    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p2a: Furret", "Furret, L50, M", "165/165"])
    si._battle.parse_message(["", "switch", "p1a: Smeargle", "Smeargle, L50", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p1b: Wo-Chien", "Wo-Chien, L50, M", "165/165"]
    )
    si._battle.parse_message(["", "switch", "p2b: Smeargle", "Smeargle, L50", "100/100"])
    events = [
        ["", "-heal", "p2a: Furret", "27/100", "[from] item: Leftovers"],
        ["", "-damage", "p1a: Smeargle", "15/130", "[from] item: Black Sludge"],
        [
            "",
            "-damage",
            "p2a: Furret",
            "15/100",
            "[from] Leech Seed",
            "[of] p1b: Wo-Chien",
        ],
        ["", "-heal", "p1b: Wo-Chien", "29/191 par", "[silent]"],
        [
            "",
            "-damage",
            "p2b: Smeargle",
            "15/100",
            "[from] Leech Seed",
            "[of] p1b: Wo-Chien",
        ],
        ["", "-heal", "p1b: Wo-Chien", "35/191 par", "[silent]"],
        ["", "-damage", "p1b: Wo-Chien", "0 fnt", "[from] Leech Seed", "[of] p2a: Furret"],
        ["", "-heal", "p2a: Furret", "35/100", "[silent]"],
        ["", "faint", "p1b: Wo-Chien"],
        ["", "-damage", "p2a: Furret", "2/100", "[from] Salt Cure"],
        ["", "-ability", "p1a: Smeargle", "Moody", "boost"],
        ["", "-boost", "p1a: Smeargle", "spe", "2"],
        ["", "-unboost", "p1a: Smeargle", "spa", "1"],
        ["", "-ability", "p2b: Smeargle", "Moody", "boost"],
        ["", "-boost", "p2b: Smeargle", "spe", "2"],
        ["", "-unboost", "p2b: Smeargle", "spa", "1"],
    ]
    orders = si.clean_orders(si._parse_residual(events))
    assert [("p2: Furret", 1.0), ("p1: Smeargle", 1.0)] in orders
    assert [("p2: Furret", 1.0), ("p2: Smeargle", 1.0)] in orders
    assert [("p2: Smeargle", 1.0), ("p1: Wo-Chien", 0.5)] in orders
    assert [("p1: Smeargle", 1.0), ("p2: Smeargle", 1.0)] in orders
    assert len(orders) == 4

    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p1a: Furret", "Furret, L50, M", "165/165 par"]
    )
    si._battle.parse_message(["", "switch", "p1b: Sentret", "Sentret, L50", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p2a: Wo-Chien", "Wo-Chien, L50, M", "165/165"]
    )
    events = [
        ["", "-heal", "p1a: Furret", "119/160 par", "[from] Grassy Terrain"],
        ["", "-heal", "p1a: Furret", "129/160 par", "[from] item: Leftovers"],
        ["", "-heal", "p2a: Wo-Chien", "10/100 tox", "[from] Grassy Terrain"],
        ["", "-heal", "p1b: Sentret", "57/142 brn", "[from] Grassy Terrain"],
        ["", "-damage", "p2a: Wo-Chien", "0 fnt", "[from] psn"],
        ["", "faint", "p2a: Wo-Chien"],
        ["", "-damage", "p1b: Sentret", "49/142 brn", "[from] brn"],
    ]
    orders = si.clean_orders(si._parse_residual(events))
    assert [("p1: Furret", 0.5), ("p1: Furret", 0.5)] in orders
    assert [("p1: Furret", 0.5), ("p2: Wo-Chien", 1.0)] in orders
    assert [("p2: Wo-Chien", 1.0), ("p1: Sentret", 1.0)] in orders
    assert len(orders) == 3

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1b: Sentret", "Sentret, L50, M", "165/165"])
    si._battle.parse_message(["", "switch", "p1a: Smeargle", "Smeargle, L50", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p2a: Wo-Chien", "Wo-Chien, L50, M", "165/165"]
    )
    si._battle.parse_message(
        ["", "switch", "p2b: Smeargle", "Smeargle, L50", "100/100 par"]
    )
    events = [
        ["", "-heal", "p2b: Smeargle", "37/100 par", "[from] Grassy Terrain"],
        ["", "-damage", "p2b: Smeargle", "25/100 par", "[from] item: Black Sludge"],
        ["", "-heal", "p2a: Wo-Chien", "72/100 tox", "[from] Grassy Terrain"],
        ["", "-heal", "p1b: Sentret", "124/142", "[from] Grassy Terrain"],
        [
            "",
            "-damage",
            "p2b: Smeargle",
            "13/100 par",
            "[from] Leech Seed",
            "[of] p2a: Wo-Chien",
        ],
        ["", "-heal", "p2a: Wo-Chien", "81/100 tox", "[silent]"],
        ["", "-damage", "p2a: Wo-Chien", "69/100 tox", "[from] psn"],
        ["", "-damage", "p2a: Wo-Chien", "57/100 tox", "[from] Salt Cure"],
        ["", "-ability", "p2b: Smeargle", "Moody", "boost"],
        ["", "-boost", "p2b: Smeargle", "spe", "2"],
        ["", "-unboost", "p2b: Smeargle", "spa", "1"],
        ["", "-status", "p1b: Sentret", "brn", "[from] item: Flame Orb"],
    ]
    orders = si.clean_orders(si._parse_residual(events))
    assert [("p2: Smeargle", 0.5), ("p2: Smeargle", 0.5)] in orders
    assert [("p2: Smeargle", 0.5), ("p2: Wo-Chien", 1.0)] in orders
    assert [("p2: Wo-Chien", 1.0), ("p1: Sentret", 1.0)] in orders
    assert [("p2: Smeargle", 0.5), ("p1: Sentret", 1.0)] in orders
    assert len(orders) == 4


def test_parse_switch():
    si = generate_speed_inference()
    si._battle._side_conditions = {SideCondition.TAILWIND: 1}
    si._battle.parse_message(["", "switch", "p1a: Smeargle", "Smeargle, L50", "160/160"])
    si._battle.parse_message(
        ["", "switch", "p1b: Wo-Chien", "Wo-Chien, L50, F", "167/167 par"]
    )
    si._battle.parse_message(["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"])
    si._battle.parse_message(["", "switch", "p2b: Sentret", "Sentret, L50, M", "100/100"])
    events = [
        ["", "switch", "p1a: Shuckle", "Shuckle, L50", "160/160"],
        ["", "switch", "p1b: Dragonite", "Dragonite, L50, F", "167/167"],
        ["", "switch", "p2a: Suicune", "Suicune, L50, F", "100/100"],
        ["", "switch", "p2b: Arceus", "Arceus, L50, M", "100/100"],
    ]
    orders = si.clean_orders(si._parse_switch(events))
    assert [("p1: Smeargle", 2.0), ("p1: Wo-Chien", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Furret", 1.0)] in orders
    assert [("p2: Furret", 1.0), ("p2: Sentret", 1.0)] in orders
    assert len(orders) == 3


def test_get_activations_from_weather_or_terrain():
    si = generate_speed_inference()
    events = [
        ["", "-weather", "SunnyDay", "[from] ability: Drought", "[of] p1a: Torkoal"],
        ["", "-terastallize", "p2a: Calyrex", "Psychic"],
    ]
    orders, num = si._get_activations_from_weather_or_terrain(events, 0)
    assert num == 0
    assert len(orders) == 0

    events = [
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p2a: Rillaboom",
        ],
        ["", "-enditem", "p1b: Raichu", "Grassy Seed"],
        ["", "-boost", "p1b: Raichu", "def", "1", "[from] item: Grassy Seed"],
        ["", "-enditem", "p2a: Rillaboom", "Grassy Seed"],
        ["", "-boost", "p2a: Rillaboom", "def", "1", "[from] item: Grassy Seed"],
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Electric Surge",
            "[of] p2b: Pincurchin",
        ],
        ["", "-enditem", "p1a: Wo-Chien", "Electric Seed"],
        ["", "-boost", "p1a: Wo-Chien", "def", "1", "[from] item: Electric Seed"],
        ["", "-enditem", "p2b: Pincurchin", "Electric Seed"],
        ["", "-boost", "p2b: Pincurchin", "def", "1", "[from] item: Electric Seed"],
    ]

    si = generate_speed_inference()
    si._battle._opponent_side_conditions = {SideCondition.TAILWIND: 1}
    si._battle.parse_message(["", "switch", "p1a: Wo-Chien", "Wo-Chien, L50", "160/160"])
    si._battle.parse_message(["", "switch", "p1b: Raichu", "Raichu, L50", "160/160"])
    si._battle.parse_message(["", "switch", "p2a: Rillaboom", "Rillaboom, L50", "160/160"])
    si._battle.parse_message(
        ["", "switch", "p2b: Pincurchin", "Pincurchin, L50", "160/160"]
    )

    orders, num = si._get_activations_from_weather_or_terrain(events, 0)
    assert num == 4
    assert [("p1: Raichu", 1.0), ("p2: Rillaboom", 2.0)] in orders

    si = generate_speed_inference()
    si._battle._side_conditions = {SideCondition.TAILWIND: 1}
    si._battle.parse_message(["", "switch", "p1a: Wo-Chien", "Wo-Chien, L50", "160/160"])
    si._battle.parse_message(["", "switch", "p1b: Raichu", "Raichu, L50", "160/160"])
    si._battle.parse_message(["", "switch", "p2a: Rillaboom", "Rillaboom, L50", "160/160"])
    si._battle.parse_message(
        ["", "switch", "p2b: Pincurchin", "Pincurchin, L50", "160/160"]
    )

    orders, num = si._get_activations_from_weather_or_terrain(events, 5)
    assert num == 4
    assert [("p1: Wo-Chien", 2.0), ("p2: Pincurchin", 1.0)] in orders

    with pytest.raises(ValueError):
        orders = si.clean_orders(si._get_activations_from_weather_or_terrain(events, 1))


def test_save_multipliers():
    si = generate_speed_inference()
    si._battle._side_conditions = {SideCondition.TAILWIND: 1}
    si._battle.parse_message(
        ["", "switch", "p1b: Shuckle", "Shuckle, L50, M", "0/175 fnt"]
    )
    si._battle.parse_message(
        ["", "switch", "p1b: Sentret", "Sentret, L50, M", "142/175 brn"]
    )
    si._battle.parse_message(
        ["", "switch", "p1a: Arceus", "Arceus, L50, M", "142/175 par"]
    )
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50, M", "165/165"])
    si._battle.parse_message(["", "switch", "p2a: Arceus", "Arceus, L50, M", "0/175 fnt"])
    si._battle.parse_message(["", "switch", "p2a: Furret", "Furret, L50", "1/100"])
    si._battle.parse_message(["", "switch", "p2b: Sentret", "Sentret, L50", "100/100 par"])
    si._battle.parse_message(["", "move", "p1a: Furret", "Tailwind"])

    multipliers = si._save_multipliers()

    assert multipliers["p1: Sentret"] == 2.0
    assert multipliers["p1: Furret"] == 2.0
    assert multipliers["p2: Sentret"] == 0.5
    assert multipliers["p2: Furret"] == 1.0


def test_get_speed_multiplier():
    assert SpeedInference.get_speed_multiplier("furret") == 1.0
    assert (
        SpeedInference.get_speed_multiplier(
            species="ditto",
            item="quickpowder",
            side_conditions={SideCondition.TAILWIND: 1},
            speed_boosts=1,
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        == 2 * 2 * 1.5
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="choicescarf",
            ability="slowstart",
            side_conditions={SideCondition.TAILWIND: 1},
            effects={Effect.SLOW_START: 1},
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        == 1.5 * 2 * 0.5
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="laggingtail",
            side_conditions={SideCondition.TAILWIND: 1},
            speed_boosts=1,
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        is None
    )
    assert (
        SpeedInference.get_speed_multiplier(
            species="ditto",
            item="pokeball",
            side_conditions={SideCondition.TAILWIND: 1},
            speed_boosts=1,
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        == 1.5 * 2
    )
    assert (
        SpeedInference.get_speed_multiplier(
            species="ditto",
            item="laggingtail",
            side_conditions={SideCondition.TAILWIND: 1},
            speed_boosts=1,
        )
        is None
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="choicescarf", weathers={Weather.SANDSTORM: 1}, ability="sandrush"
        )
        == 1.5 * 2.0
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="fullincense",
            weathers={Weather.SANDSTORM: 1},
            ability="sandrush",
            effects={Effect.QUASH: 1},
        )
        is None
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="choicescarf",
            weathers={Weather.SANDSTORM: 1},
            ability="sandrush",
            effects={Effect.QUASH: 1},
        )
        is None
    )
    assert (
        SpeedInference.get_speed_multiplier(ability="quickfeet", status=Status.PAR) == 1.5
    )
    assert SpeedInference.get_speed_multiplier(status=Status.PAR) == 0.5
    assert SpeedInference.get_speed_multiplier(speed_boosts=2, status=Status.PAR) == 1.0
    assert SpeedInference.get_speed_multiplier(speed_boosts=-2, status=Status.PAR) == 0.25
    assert (
        SpeedInference.get_speed_multiplier(
            side_conditions={SideCondition.TAILWIND: 1}, status=Status.PAR, speed_boosts=6
        )
        == 4.0
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="powerband",
        )
        == 0.5
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="choicescarf", effects={Effect.PROTOSYNTHESISSPE: 1}
        )
        == 1.5 * 1.5
    )
    assert (
        SpeedInference.get_speed_multiplier(
            item="choicescarf", effects={Effect.QUARKDRIVESPE: 1}
        )
        == 1.5 * 1.5
    )
    assert (
        SpeedInference.get_speed_multiplier(
            ability="surgesurfer", fields={Field.ELECTRIC_TERRAIN: 1, Field.TRICK_ROOM: 1}
        )
        == -2.0
    )
    assert SpeedInference.get_speed_multiplier(
        item="powerbracer",
        speed_boosts=-6,
        status=Status.PAR,
        effects={Effect.SLOW_START: 1},
        side_conditions={SideCondition.GRASS_PLEDGE: 1},
        fields={Field.ELECTRIC_TERRAIN: 1, Field.TRICK_ROOM: 1},
    ) == (0.25 * 0.5 * 0.5 * 0.5 * 0.5 * -1)


def test_clean_orders():
    orders = [
        [("p2: Ting-Lu", 1.0), ("p1: Raichu", 1.0), ("p1: Wo-Chien", 0.67)],
        [("p1b: Raichu", 1.0)],
        [("p1a: Wo-Chien", 1.0), ("p2b: Ting-Lu", 1.0)],
        [("p1a: Wo-Chien", 1.0), ("p2b: Ting-Lu", 1.0), ("p1b: Raichu", 0.25)],
    ]

    cleaned = SpeedInference.clean_orders(orders)
    assert [("p2: Ting-Lu", 1.0), ("p1: Raichu", 1.0)] in cleaned
    assert [("p1: Raichu", 1.0), ("p1: Wo-Chien", 0.67)] in cleaned
    assert [("p1: Wo-Chien", 1.0), ("p2: Ting-Lu", 1.0)] in cleaned
    assert [("p1: Wo-Chien", 1.0), ("p2: Ting-Lu", 1.0)] in cleaned
    assert [("p2: Ting-Lu", 1.0), ("p1: Raichu", 0.25)] in cleaned
    assert len(cleaned) == 5


def test_update_orders():
    si = generate_speed_inference()
    si._orders = [
        [("p1: Furret", 1.0), ("p2: Sentret", 1.0)],
        [("p1: Furret", 1.0), ("p2: Furret", 1.0)],
        [("p1: Furret", -1.0), ("p2: Furret", -1.0)],
        [("p2: Furret", -1.0), ("p1: Furret", -1.0)],
        [("p2: Furret", -1.0), ("p1: Sentret", -1.0)],
    ]
    si._update_orders("p1: Furret", 1.5)
    assert si._orders == [
        [("p1: Furret", 1.5), ("p2: Sentret", 1.0)],
        [("p1: Furret", 1.5), ("p2: Furret", 1.0)],
        [("p1: Furret", -1.5), ("p2: Furret", -1.0)],
        [("p2: Furret", -1.0), ("p1: Furret", -1.5)],
        [("p2: Furret", -1.0), ("p1: Sentret", -1.0)],
    ]


def test_solve_speeds():
    tb = ConstantTeambuilder(
        """
Furret @ Choice Specs
Ability: Frisk
Level: 50
Tera Type: Water
EVs: 252 SpA
Rash Nature
IVs: 0 Atk
- Water Pulse

Smeargle @ Focus Sash
Ability: Moody
Level: 50
Tera Type: Ghost
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Confuse Ray

Calyrex-Shadow
Ability: As One (Spectrier)
Level: 50
Tera Type: Fairy
EVs: 252 SpA / 4 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Confuse Ray

Incineroar @ Choice Scarf
Ability: Blaze
Level: 50
Tera Type: Water
EVs: 84 HP / 4 Atk / 252 SpD / 164 Spe
Adamant Nature
- Parting Shot
    """
    ).team

    # Smeargle and Calyrex speed tie (139) > Furret (110) > Incineroar (101 w/out choicescarf)
    si = generate_speed_inference()
    si._opponent_mons = {
        "p2: Calyrex": {"spe": [139, 222], "can_be_choice": True},
        "p2: Incineroar": {"spe": [58, 111], "can_be_choice": True},
    }
    si._battle.team = {
        "p1: Furret": Pokemon(gen=9, teambuilder=tb[0]),
        "p1: Smeargle": Pokemon(gen=9, teambuilder=tb[1]),
    }
    si._battle._teampreview_opponent_team = {
        Pokemon(gen=9, teambuilder=tb[2]),
        Pokemon(gen=9, teambuilder=tb[3]),
    }
    si._battle._opponent_team = {
        "p2: Calyrex": Pokemon(gen=9, teambuilder=tb[2]),
        "p2: Incineroar": Pokemon(gen=9, teambuilder=tb[3]),
    }
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50", "160/160"])
    si._battle.parse_message(
        ["", "switch", "p1b: Smeargle", "Smeargle, L50, F", "167/167"]
    )
    si._battle.parse_message(["", "switch", "p2a: Calyrex", "Calyrex, L50, F", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p2b: Incineroar", "Incineroar, L50, M", "100/100"]
    )

    events = [
        ["", "-heal", "p2b: Incineroar", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p2a: Calyrex", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p1a: Smeargle", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p2a: Calyrex", "160/160", "[from] Leftovers"],
        ["", "-heal", "p1a: Furret", "160/160", "[from] Grassy Terrain"],
    ]

    si._solve_speeds(si.clean_orders(si._parse_residual(events)))

    assert get_pokemon("p2: Incineroar", si._battle).item == "choicescarf"
    assert si._opponent_mons["p2: Calyrex"]["spe"] == [139.0, 139.0]
    assert si._opponent_mons["p2: Incineroar"]["spe"] == [93.0, 111.0]
