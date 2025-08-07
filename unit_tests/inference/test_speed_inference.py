# -*- coding: utf-8 -*-
"""Unit Tests to ensure Speed Inference can detect Choice Scarfs
"""
from unittest.mock import MagicMock

import pytest
from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.speed_inference import SpeedInference


def generate_speed_inference():
    battle = DoubleBattle("tag", "username", MagicMock(), gen=9)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    bi = BattleInference(battle)
    si = SpeedInference(battle=battle, inferences=bi)
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
    assert si._battle.get_pokemon("p2a: Rillaboom").item is None
    assert si._battle.get_pokemon("p1b: Raichu").item == "airballoon"

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

    orders = si._parse_preturn_switch(events)
    assert [("p1: Wo-Chien", 1.0), ("p1: Raichu", 1.0)] in orders
    assert [("p1: Raichu", 1.0), ("p2: Rillaboom", 1.0)] in orders
    assert len(orders) == 2
    assert Field.GRASSY_TERRAIN in si._battle.fields
    assert si._battle.get_pokemon("p2a: Rillaboom").item is None
    assert si._battle.get_pokemon("p2a: Rillaboom").boosts["def"] == 1

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
    orders = si._parse_preturn_switch(events)

    assert [("p2: Rillaboom", 1.0), ("p1: Wo-Chien", 1.0)] in orders
    assert [("p1: Raichu", 1.0), ("p2: Rillaboom", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Pincurchin", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Pincurchin", 1.0)] in orders
    assert len(orders) == 4
    assert Field.ELECTRIC_TERRAIN in si._battle.fields
    assert si._battle.get_pokemon("p2a: Rillaboom").item is None
    assert si._battle.get_pokemon("p2b: Pincurchin").boosts["def"] == 1

    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Nickname", "Miraidon, L50", "100/100"],
        ["", "switch", "p1b: Iron Hands", "Iron Hands, L50", "100/100"],
        ["", "switch", "p2a: Kommo-o better", "Zamazenta-Crowned, L50", "199/199"],
        ["", "switch", "p2b: Chien-Pao", "Chien-Pao, L50", "155/155"],
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Hadron Engine",
            "[of] p1a: Nickname",
        ],
        ["", "-activate", "p1b: Iron Hands", "ability: Quark Drive"],
        ["", "-start", "p1b: Iron Hands", "quarkdriveatk"],
        ["", "-ability", "p2b: Chien-Pao", "Sword of Ruin"],
        ["", "-ability", "p2a: Kommo-o better", "Dauntless Shield", "boost"],
        ["", "-boost", "p2a: Kommo-o better", "def", "1"],
    ]
    orders = si._parse_preturn_switch(events)
    assert len(orders) == 2
    assert [("p1: Nickname", 1.0), ("p2: Chien-Pao", 1.0)] in orders
    assert [("p2: Chien-Pao", 1.0), ("p2: Kommo-o better", 1.0)] in orders

    # Make sure we account for activations correctly
    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Nickname", "Miraidon, L50", "176/176"],
        ["", "switch", "p1b: Iron Hands", "Iron Hands, L50", "238/238"],
        ["", "switch", "p2a: Miraidon", "Miraidon, L50", "100/100"],
        ["", "switch", "p2b: Iron Hands", "Iron Hands, L50", "100/100"],
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Hadron Engine",
            "[of] p2a: Miraidon",
        ],
        ["", "-activate", "p1b: Iron Hands", "ability: Quark Drive"],
        ["", "-start", "p1b: Iron Hands", "quarkdriveatk"],
        ["", "-activate", "p2b: Iron Hands", "ability: Quark Drive"],
        ["", "-start", "p2b: Iron Hands", "quarkdriveatk"],
        ["", "-activate", "p1a: Nickname", "ability: Hadron Engine"],
    ]
    orders = si._parse_preturn_switch(events)
    assert len(orders) == 2
    assert [("p1: Iron Hands", 1.0), ("p2: Iron Hands", 1.0)] in orders
    assert [("p2: Miraidon", 1.0), ("p1: Nickname", 1.0)] in orders
    assert len(events) == len(si._battle.current_observation.events)

    si = generate_speed_inference()
    events = [
        ["", "switch", "p1a: Calyrex", "Calyrex-Shadow, L50", "176/176"],
        ["", "switch", "p1b: Iron Bundle", "Iron Bundle, L50", "139/139"],
        ["", "switch", "p2a: Groudon", "Groudon, L50", "100/100"],
        ["", "switch", "p2b: Flutter Mane", "Flutter Mane, L50", "100/100"],
        ["", "-ability", "p1a: Calyrex", "As One"],
        ["", "-ability", "p1a: Calyrex", "Unnerve"],
        ["", "-weather", "SunnyDay", "[from] ability: Drought", "[of] p2a: Groudon"],
        ["", "-activate", "p2b: Flutter Mane", "ability: Protosynthesis"],
        ["", "-start", "p2b: Flutter Mane", "protosynthesisspe"],
        ["", "-enditem", "p1b: Iron Bundle", "Booster Energy"],
        ["", "-activate", "p1b: Iron Bundle", "ability: Quark Drive", "[fromitem]"],
        ["", "-start", "p1b: Iron Bundle", "quarkdrivespe"],
    ]
    temp = si._parse_preturn_switch(events)
    orders = list(filter(lambda x: x[0] != x[1], temp))
    assert orders == []

    # # I dont catch this case; I should cuz all the abilities were checked to decide the first one
    # # but I stop checking abilities after a detailschange in case theres a speed change; to get this
    # # case right, I should have to infer that Weezing's Neutralizing Gas was checked against Calyrex's
    # # priority ability and recorded the speed when Calyrex's went off
    # si = generate_speed_inference()
    # events = [
    #     ["", "switch", "p1a: Terapagos", "Terapagos, L50, M", "165/165"],
    #     ["", "switch", "p1b: Weezing", "Weezing-Galar, L50, M", "140/140"],
    #     ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50", "100/100"],
    #     ["", "switch", "p2b: Arbok", "Arbok, L50, M", "100/100"],
    #     ["", "-ability", "p2a: Calyrex", "As One"],
    #     ["", "-ability", "p2a: Calyrex", "Unnerve"],
    #     ["", "-activate", "p1a: Terapagos", "ability: Tera Shift"],
    #     ["", "detailschange", "p1a: Terapagos", "Terapagos-Terastal, L50, M"],
    #     ["", "-heal", "p1a: Terapagos", "170/170", "[silent]"],
    #     ["", "-ability", "p1b: Weezing", "Neutralizing Gas"],
    # ]

    # orders = si._parse_preturn_switch(events)
    # assert [("p2: Terapagos", 1.0), ("p1: Weezing", 1.0)] in orders
    # assert len(orders) == 1  # Can't infer Terapagos because speed may have changed


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
        [
            "",
            "detailschange",
            "p1a: Terapagos",
            "Terapagos-Stellar, L50, M, tera:Stellar",
        ],
        ["", "-heal", "p1a: Terapagos", "235/235", "[silent]"],
    ]
    orders = si._parse_battle_mechanic(events)
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
        [
            "",
            "detailschange",
            "p1a: Terapagos",
            "Terapagos-Stellar, L50, M, tera:Stellar",
        ],
        ["", "-heal", "p1a: Terapagos", "235/235", "[silent]"],
        ["", "-terastallize", "p2a: Calyrex", "Psychic"],
    ]
    orders = si._parse_battle_mechanic(events)
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
    orders = si._parse_move(events)
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
    orders = si._parse_move(events)
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
    orders = si._parse_move(events)
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
    orders = si._parse_move(events)
    assert [("p1: Volcarona", 1.0), ("p1: Furret", 1.0)] in orders

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
    orders = si._parse_move(events)
    assert len(orders) == 0

    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p1b: Delibird", "Delibird, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p1a: Tyranitar", "Tyranitar, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p2b: Bellossom", "Bellossom, L50, F", "167/167"]
    )
    events = [
        ["", "move", "p1b: Delibird", "Spikes", "p2a: Gardevoir"],
        ["", "-sidestart", "p2: speed_ability1", "Spikes"],
        ["", "move", "p2a: Gardevoir", "Future Sight", "p2b: Bellossom"],
        ["", "-start", "p2a: Gardevoir", "move: Future Sight"],
        ["", "move", "p1a: Tyranitar", "Dragon Tail", "p1b: Delibird"],
        ["", "-damage", "p1b: Delibird", "75/100"],
        ["", "drag", "p1b: Raichu", "Raichu, L50, F, tera:Ground", "95/100"],
    ]
    orders = si._parse_move(events)
    assert len(orders) == 1
    assert [("p1: Delibird", 1.0), ("p2: Gardevoir", 1.0)] in orders

    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p1b: Grimmsnarl", "Grimmsnarl, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p1a: Volcarona", "Volcarona, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Whimsicott", "Whimsicott, L50, F", "167/167"]
    )
    si._battle.parse_message(
        ["", "switch", "p2b: Terapagos", "Terapagos, L50, F", "167/167"]
    )
    events = [
        ["", "move", "p1b: Grimmsnarl", "Thunder Wave", "p1a: Volcarona"],
        ["", "-status", "p1a: Volcarona", "par"],
        [
            "",
            "move",
            "p2b: Terapagos",
            "Meteor Beam",
            "p1b: Grimmsnarl",
            "[from]lockedmove",
        ],
        ["", "-damage", "p1b: Grimmsnarl", "41/100"],
        ["", "move", "p2a: Whimsicott", "Endeavor", "p1b: Grimmsnarl"],
        ["", "-damage", "p1b: Grimmsnarl", "1/100"],
        [
            "",
            "move",
            "p1a: Volcarona",
            "Heat Wave",
            "p2a: Whimsicott",
            "[spread] p2a,p2b",
        ],
        ["", "-supereffective", "p2a: Whimsicott"],
        ["", "-damage", "p2a: Whimsicott", "0 fnt"],
        ["", "-damage", "p2b: Terapagos", "135/243"],
        ["", "faint", "p2a: Whimsicott"],
    ]
    orders = si._parse_move(events)
    assert len(orders) == 2

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Nickname", "Miraidon, L50", "167/167"])
    si._battle.parse_message(
        ["", "switch", "p1b: Iron Hands", "Iron Hands, L50, F", "167/167"]
    )
    si._battle.parse_message(["", "switch", "p2a: Terapagos", "Terapagos, L50", "167/167"])
    si._battle.parse_message(
        ["", "switch", "p2b: Incineroar", "Incineroar, L50, F", "167/167"]
    )
    events = [
        ["", "move", "p1a: Nickname", "Draco Meteor", "p2b: Incineroar", "[miss]"],
        ["", "-miss", "p1a: Nickname", "p2b: Incineroar"],
        ["", "move", "p2a: Terapagos", "Tri Attack", "p2b: Incineroar"],
        ["", "-damage", "p2b: Incineroar", "119/202"],
        ["", "move", "p2b: Incineroar", "U-turn", "p2a: Terapagos"],
        ["", "-damage", "p2a: Terapagos", "231/258"],
        ["", ""],
        [
            "",
            "switch",
            "p2b: Rillaboom",
            "Rillaboom, L50, M",
            "176/176",
            "[from] U-turn",
        ],
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p2b: Rillaboom",
        ],
        ["", "move", "p1b: Iron Hands", "Wild Charge", "p1a: Nickname"],
        ["", "-resisted", "p1a: Nickname"],
        ["", "-damage", "p1a: Nickname", "92/100"],
        ["", "-damage", "p1b: Iron Hands", "99/100", "[from] Recoil"],
    ]
    orders = si._parse_move(events)
    assert len(orders) == 3

    si = generate_speed_inference()

    si._battle.parse_message(
        ["", "switch", "p1b: Volcarona", "Volcarona, L50, M", "100/100"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Dragonite", "Dragonite, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2b: Urshifu", "Urshifu, L50, F", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p1a: Grimmsnarl", "Grimmsnarl, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "turn", "5"])
    events = [
        ["", "move", "p2a: Dragonite", "Haze", "p2a: Dragonite"],
        ["", "-clearallboost"],
        ["", "move", "p2b: Urshifu", "Close Combat", "p1a: Grimmsnarl"],
        ["", "-damage", "p1a: Grimmsnarl", "18/100"],
        ["", "-unboost", "p2b: Urshifu", "def", "1"],
        ["", "-unboost", "p2b: Urshifu", "spd", "1"],
    ]

    orders = si._parse_move(events)
    assert orders
    assert si._battle.current_observation.events == events

    si = generate_speed_inference()
    si._battle.parse_message(
        ["", "switch", "p1b: Incineroar", "Incineroar, L50, M", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2a: Calyrex", "Calyrex, L50, F", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p2b: Farigiraf", "Farigiraf, L50, F", "100/100"]
    )
    si._battle.parse_message(
        ["", "switch", "p1a: Rillaboom", "Rillaboom, L50, F", "100/100"]
    )

    events = [
        ["", "move", "p1a: Rillaboom", "Fake Out", "", "[still]"],
        [
            "",
            "cant",
            "p2b: Farigiraf",
            "ability: Armor Tail",
            "Fake Out",
            "[of] p1a: Rillaboom",
        ],
        ["", "move", "p1b: Incineroar", "Fake Out", "", "[still]"],
        ["", "-hint", "Fake Out only works on your first turn out."],
        ["", "-fail", "p1b: Incineroar"],
        ["", "move", "p2b: Farigiraf", "Psychic Noise", "p1a: Rillaboom"],
        ["", "-damage", "p1a: Rillaboom", "0 fnt"],
        ["", "faint", "p1a: Rillaboom"],
        ["", "move", "p2a: Calyrex", "Trick Room", "p2a: Calyrex"],
        ["", "-fieldstart", "move: Trick Room", "[of] p2a: Calyrex"],
    ]

    orders = si._parse_move(events)
    assert orders == [[("p1: Rillaboom", 1.0), ("p1: Incineroar", 1.0)]]


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
        [
            "",
            "-damage",
            "p1b: Wo-Chien",
            "0 fnt",
            "[from] Leech Seed",
            "[of] p2a: Furret",
        ],
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
    orders = si._parse_residual(events)
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
    orders = si._parse_residual(events)
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
    orders = si._parse_residual(events)
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
    orders = si._parse_switch(events)
    assert [("p1: Smeargle", 2.0), ("p1: Wo-Chien", 1.0)] in orders
    assert [("p1: Wo-Chien", 1.0), ("p2: Furret", 1.0)] in orders
    assert [("p2: Furret", 1.0), ("p2: Sentret", 1.0)] in orders
    assert len(orders) == 3

    si = generate_speed_inference()

    events = [
        ["", "switch", "p1a: Nickname", "Miraidon, L50", "176/176"],
        ["", "switch", "p2b: Flutter Mane", "Fluttermane, L50", "100/100"],
        ["", "-activate", "p2b: Flutter Mane", "ability: Protosynthesis"],
        ["", "-start", "p2b: Flutter Mane", "protosynthesisspe"],
    ]
    for event in events:
        si._battle.parse_message(event)

    events = [
        ["", "-end", "p2b: Flutter Mane", "Protosynthesis", "[silent]"],
        ["", "switch", "p2b: Eternatus", "Eternatus, L50", "100/100"],
        ["", "-ability", "p2b: Eternatus", "Pressure"],
        ["", "switch", "p1a: Iron Hands", "Iron Hands, L50, tera:Water", "164/238"],
        ["", "-activate", "p1a: Iron Hands", "ability: Quark Drive"],
        ["", "-start", "p1a: Iron Hands", "quarkdriveatk"],
    ]

    orders = si._parse_switch(events)
    assert orders == [[("p2: Flutter Mane", 1.5), ("p1: Nickname", 1.0)]]

    si = generate_speed_inference()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50", "100/100"],
        ["", "switch", "p1a: Nickname", "Miraidon, L50", "176/176"],
        ["", "switch", "p2b: Flutter Mane", "Fluttermane, L50", "100/100"],
        ["", "-activate", "p2b: Flutter Mane", "ability: Protosynthesis"],
        ["", "-start", "p2b: Flutter Mane", "protosynthesisspe"],
    ]
    for event in events:
        si._battle.parse_message(event)

    events = [
        ["", "-end", "p2b: Flutter Mane", "Protosynthesis", "[silent]"],
        ["", "switch", "p2b: Mandibuzz", "Mandibuzz, L50, F", "100/100"],
        ["", "switch", "p1a: Volcarona", "Volcarona, L50, M", "192/192"],
        ["", "switch", "p2a: Incineroar", "Incineroar, L50, M", "100/100"],
        ["", "-ability", "p2a: Incineroar", "Intimidate", "boost"],
        ["", "-unboost", "p1a: Volcarona", "atk", "1"],
        ["", "-unboost", "p1b: Grimmsnarl", "atk", "1"],
    ]

    orders = si._parse_switch(events)
    assert orders == [
        [("p2: Flutter Mane", 1.5), ("p1: Nickname", 1.0)],
        [("p1: Nickname", 1.0), ("p2: Furret", 1.0)],
    ]

    si = generate_speed_inference()
    si._battle.parse_message(["", "switch", "p1a: Furret", "Furret, L50", "100/100"])
    si._battle.parse_message(
        ["", "switch", "p1b: Flutter Mane", "Fluttermane, L50", "100/100"]
    )
    si._battle.parse_message(
        ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50, F", "100/100"]
    )
    si._battle.parse_message(["", "switch", "p2b: Sentret", "Sentret, L50, F", "100/100"])

    si._battle.team["p1: Flutter Mane"]._effects = {
        Effect.PROTOSYNTHESISSPE: 1,
        Effect.PROTOSYNTHESIS: 1,
    }

    events = [
        ["", "-end", "p1b: Flutter Mane", "Protosynthesis", "[silent]"],
        ["", "switch", "p1b: Rillaboom", "Rillaboom, L50, M", "183/183"],
        [
            "",
            "-fieldstart",
            "move: Grassy Terrain",
            "[from] ability: Grassy Surge",
            "[of] p1b: Rillaboom",
        ],
        ["", "switch", "p2a: Urshifu", "Urshifu-Rapid-Strike, L50, F", "100/100"],
    ]

    orders = si._parse_switch(events)
    assert orders == [
        [("p1: Flutter Mane", 1.5), ("p2: Calyrex", 1.0)],
    ]


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
        orders = si._get_activations_from_weather_or_terrain(events, 1)  # pyright: ignore


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
    assert SpeedInference.get_speed_multiplier(mon=Pokemon(gen=9, species="furret")) == 1.0
    assert (
        SpeedInference.get_speed_multiplier(
            mon=Pokemon(gen=9, species="ditto"),
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
            mon=Pokemon(gen=9, species="ditto"),
            item="pokeball",
            side_conditions={SideCondition.TAILWIND: 1},
            speed_boosts=1,
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        == 1.5 * 2
    )
    assert (
        SpeedInference.get_speed_multiplier(
            mon=Pokemon(gen=9, species="ditto"),
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
            side_conditions={SideCondition.TAILWIND: 1},
            status=Status.PAR,
            speed_boosts=6,
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
            ability="surgesurfer",
            fields={Field.ELECTRIC_TERRAIN: 1, Field.TRICK_ROOM: 1},
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


def test_scrub_orders():
    orders = [
        [("p2: Ting-Lu", 2.0), ("p1: Raichu", 1.0)],
        [("p1: Wo-Chien", 1.0), ("p2: Ting-Lu", 1.0)],
        [("p1: Sentret", 1.0), ("p2: Ting-Lu", 2.0)],
        [("p1: Furret", 1.5), ("p2: Ting-Lu", 2.5)],
        [("p1: Furret", 1.0), ("p1: Sentret", 2.5)],
    ]

    cleaned = SpeedInference.scrub_orders(orders, "p2: Ting-Lu")  # pyright: ignore
    assert [("p1: Sentret", 1.0), ("p1: Raichu", 1.0)] in cleaned
    assert [("p1: Furret", 1.5), ("p1: Raichu", 1.0)] in cleaned
    assert [("p1: Furret", 1.0), ("p1: Sentret", 2.5)] in cleaned
    assert len(cleaned) == 3


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

    si = generate_speed_inference()

    # Load team; Smeargle and Calyrex speed tie (139) > Furret (110) > Incineroar (101 w/out choicescarf)
    for tb_mon in tb:
        mon = Pokemon(gen=9, teambuilder=tb_mon)
        si._battle._team["p1: " + mon.name] = mon

    # Load opponent team
    si._battle._opponent_team = {
        "p2: Calyrex": Pokemon(gen=9, details="Calyrex-Shadow, L50"),
        "p2: Incineroar": Pokemon(gen=9, details="Incineroar, L50, F"),
    }

    si._inferences._battle = si._battle
    for ident, mon in si._battle._opponent_team.items():
        si._inferences._opponent_mons[ident] = BattleInference.load_opponent_set(mon)

    # Set up battle
    events = [
        ["", "switch", "p1a: Furret", "Furret, L50", "160/160"],
        ["", "switch", "p1b: Smeargle", "Smeargle, L50, F", "167/167"],
        ["", "switch", "p2a: Calyrex", "Calyrex-Shadow, L50", "100/100"],
        ["", "switch", "p2b: Incineroar", "Incineroar, L50, M", "100/100"],
    ]
    for event in events:
        si._battle.parse_message(event)

    # Parse events
    # Smeargle and Calyrex speed tie (139) > Furret (110) > Incineroar (101 w/out choicescarf)
    events = [
        ["", "-heal", "p2b: Incineroar", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p2a: Calyrex", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p1a: Smeargle", "160/160", "[from] Grassy Terrain"],
        ["", "-heal", "p2a: Calyrex", "160/160", "[from] Leftovers"],
        ["", "-heal", "p1a: Furret", "160/160", "[from] Grassy Terrain"],
    ]
    orders = si._parse_residual(events)
    si._solve_speeds(orders)  # type: ignore

    assert si._inferences.get_flag("p2: Incineroar", "item") == "choicescarf"
    assert si._inferences.get_flag("p2: Calyrex", "spe") == [139.0, 139.0]
    assert si._inferences.get_flag("p2: Incineroar", "spe") == [93.0, 123.0]
