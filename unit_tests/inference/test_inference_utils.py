# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

from elitefurretai.inference.inference_utils import (
    get_ability_and_identifier,
    get_priority_and_identifier,
    get_residual_and_identifier,
    get_segments,
    has_flinch_immunity,
    has_rage_powder_immunity,
    has_sandstorm_immunity,
    has_status_immunity,
    has_unboost_immunity,
    is_ability_event,
    is_grounded,
    standardize_pokemon_ident,
)


def test_get_segments(residual_logs, edgecase_logs, uturn_logs):
    segments = get_segments(edgecase_logs[0])
    assert segments["preturn_switch"][0] == [
        "",
        "switch",
        "p1a: Indeedee",
        "Indeedee, L50, M",
        "100/100",
    ]
    assert segments["preturn_switch"][-1] == [
        "",
        "-weather",
        "RainDance",
        "[from] ability: Drizzle",
        "[of] p1b: Pelipper",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 1
    )

    segments = get_segments(edgecase_logs[1])
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Seismic Toss",
        "p1a: Indeedee",
    ]
    assert segments["move"][-1] == [
        "",
        "-fieldstart",
        "move: Trick Room",
        "[of] p1a: Indeedee",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 2
    )

    segments = get_segments(edgecase_logs[2])
    assert segments["switch"][0] == [
        "",
        "switch",
        "p2b: Grimmsnarl",
        "Grimmsnarl, L50, M",
        "170/170",
    ]
    assert segments["battle_mechanic"][0] == [
        "",
        "-terastallize",
        "p2a: Heracross",
        "Ghost",
    ]
    assert segments["battle_mechanic"][-1] == [
        "",
        "-terastallize",
        "p1a: Indeedee",
        "Psychic",
    ]
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Endure",
        "p2a: Heracross",
    ]
    assert segments["move"][-1] == ["", "-damage", "p1a: Indeedee", "2/100"]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 4
    )

    segments = get_segments(edgecase_logs[3])
    assert segments["move"][0] == [
        "",
        "move",
        "p2b: Grimmsnarl",
        "Thunder Wave",
        "p1b: Pelipper",
        "[miss]",
    ]
    assert segments["move"][-1] == ["", "-fail", "p1a: Indeedee"]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 2
    )

    segments = get_segments(edgecase_logs[4])
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Endure",
        "p2a: Heracross",
    ]
    assert segments["move"][-1] == ["", "-fail", "p1a: Indeedee"]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 2
    )

    segments = get_segments(edgecase_logs[5])
    assert segments["move"][0] == [
        "",
        "move",
        "p2b: Grimmsnarl",
        "Scary Face",
        "",
        "[still]",
    ]
    assert segments["move"][-1] == ["", "-fail", "p1a: Indeedee"]
    assert segments["state_upkeep"] == [
        ["", "-weather", "Snow", "[upkeep]"],
        ["", "-fieldend", "move: Trick Room"],
        ["", "-fieldend", "move: Psychic Terrain"],
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 2
    )

    segments = get_segments(edgecase_logs[6])
    assert segments["switch"][0] == [
        "",
        "switch",
        "p2a: Drednaw",
        "Drednaw, L50, F",
        "197/197",
    ]
    assert segments["switch"][-1] == [
        "",
        "-weather",
        "RainDance",
        "[from] ability: Drizzle",
        "[of] p1a: Pelipper",
    ]
    assert segments["move"][0] == [
        "",
        "move",
        "p2b: Grimmsnarl",
        "Thunder Wave",
        "p2a: Drednaw",
    ]
    assert segments["move"][-1] == ["", "-activate", "p1a: Pelipper", "trapped"]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 3
    )

    segments = get_segments(edgecase_logs[7])
    assert segments["move"][0] == [
        "",
        "move",
        "p1a: Pelipper",
        "Whirlpool",
        "p1b: Farigiraf",
    ]
    assert segments["move"][-1] == ["", "-miss", "p2a: Drednaw", "p1a: Pelipper"]
    assert segments["residual"][0] == [
        "",
        "-damage",
        "p1b: Farigiraf",
        "67/100",
        "[from] move: Whirlpool",
        "[partiallytrapped]",
    ]
    assert len(segments["residual"]) == 1
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 3
    )

    segments = get_segments(edgecase_logs[8])
    assert segments["move"][0] == ["", "move", "p1a: Pelipper", "U-turn", "p2a: Drednaw"]
    assert segments["move"][-1] == [
        "",
        "-fieldstart",
        "move: Grassy Terrain",
        "[from] ability: Grassy Surge",
        "[of] p1b: Rillaboom",
    ]
    assert segments["residual"][0] == [
        "",
        "-heal",
        "p2a: Drednaw",
        "187/197 par",
        "[from] Grassy Terrain",
    ]
    assert segments["residual"][-1] == [
        "",
        "-heal",
        "p2a: Drednaw",
        "197/197 par",
        "[from] item: Leftovers",
    ]
    assert segments["preturn_switch"] == [
        ["", "switch", "p1a: Farigiraf", "Farigiraf, L50, F", "56/100"]
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 4
    )

    segments = get_segments(edgecase_logs[9])
    assert segments["activation"] == [
        ["", "-activate", "p1a: Farigiraf", "item: Quick Claw"]
    ]
    assert segments["move"][0] == [
        "",
        "move",
        "p1a: Farigiraf",
        "Mean Look",
        "p2a: Drednaw",
    ]
    assert segments["move"][-1] == ["", "-unboost", "p1a: Farigiraf", "spa", "1"]
    assert segments["residual"][0] == [
        "",
        "-heal",
        "p1a: Farigiraf",
        "33/100",
        "[from] Grassy Terrain",
    ]
    assert segments["residual"][-1] == [
        "",
        "-heal",
        "p1a: Farigiraf",
        "33/100",
        "[from] Grassy Terrain",
    ]
    assert segments["preturn_switch"] == [
        ["", "switch", "p2a: Heracross", "Heracross, L50, M, tera:Ghost", "155/155"],
        ["", "-enditem", "p2a: Heracross", "Grassy Seed"],
        ["", "-boost", "p2a: Heracross", "def", "1", "[from] item: Grassy Seed"],
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 4
    )

    segments = get_segments(edgecase_logs[10])
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Vacuum Wave",
        "",
        "[still]",
    ]
    assert segments["move"][-1] == ["", "faint", "p1a: Farigiraf"]
    assert segments["preturn_switch"] == [
        ["", "switch", "p1a: Pelipper", "Pelipper, L50, F", "40/100"],
        ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p1a: Pelipper"],
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 3
    )

    segments = get_segments(edgecase_logs[11])
    assert segments["move"][0] == [
        "",
        "move",
        "p1b: Rillaboom",
        "Frenzy Plant",
        "p1a: Pelipper",
    ]
    assert segments["move"][-1] == ["", "-unboost", "p2a: Heracross", "spa", "1"]
    assert segments["residual"][0] == [
        "",
        "-heal",
        "p1b: Rillaboom",
        "78/100",
        "[from] Grassy Terrain",
    ]
    assert segments["residual"][-1] == [
        "",
        "-heal",
        "p2a: Heracross",
        "106/155",
        "[from] Grassy Terrain",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 3
    )

    segments = get_segments(edgecase_logs[14])
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Endure",
        "p2a: Heracross",
    ]
    assert segments["move"][-1] == [
        "",
        "-damage",
        "p2b: Teddiursa",
        "62/167",
        "[from] item: Life Orb",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 1
    )

    segments = get_segments(edgecase_logs[15])
    assert segments["activation"][0] == [
        "",
        "-activate",
        "p2a: Rillaboom",
        "item: Quick Claw",
    ]
    assert segments["activation"][-1] == ["", ""]
    assert segments["switch"][0] == [
        "",
        "switch",
        "p1b: Wo-Chien",
        "Wo-Chien, L50, tera:Dark",
        "50/160",
    ]
    assert segments["switch"][-1] == ["", "-ability", "p1b: Wo-Chien", "Tablets of Ruin"]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "activation",
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 5
    )

    assert len(get_segments(residual_logs[0])) == 1
    assert len(get_segments(residual_logs[1])) == 1
    assert len(get_segments(residual_logs[2])) == 1
    assert len(get_segments(residual_logs[3])) == 1
    assert len(get_segments(residual_logs[4])) == 1

    segments = get_segments(residual_logs[5])
    assert segments["preturn_switch"][0] == [
        "",
        "switch",
        "p1a: Raichu",
        "Raichu, L50, F",
        "167/167",
    ]
    assert segments["preturn_switch"][-1] == [
        "",
        "-weather",
        "Sandstorm",
        "[from] ability: Sand Stream",
        "[of] p1b: Tyranitar",
    ]
    assert len(segments["preturn_switch"]) == 5
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 1
    )

    segments = get_segments(residual_logs[6])
    assert segments["switch"][0] == [
        "",
        "switch",
        "p1a: Furret",
        "Furret, L50, F",
        "160/160",
    ]
    assert segments["switch"][-1] == [
        "",
        "-item",
        "p2b: Amoonguss",
        "Sitrus Berry",
        "[from] ability: Frisk",
        "[of] p1a: Furret",
        "[identify]",
    ]
    assert segments["battle_mechanic"][0] == ["", "-terastallize", "p2a: Pikachu", "Bug"]
    assert segments["battle_mechanic"][-1] == [
        "",
        "-terastallize",
        "p1b: Tyranitar",
        "Electric",
    ]
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Pikachu",
        "Thunderbolt",
        "p1b: Tyranitar",
    ]
    assert segments["move"][-1] == ["", "-damage", "p2a: Pikachu", "39/100"]
    assert segments["residual"][0] == [
        "",
        "-damage",
        "p1a: Furret",
        "150/160",
        "[from] Sandstorm",
    ]
    assert segments["residual"][-1] == [
        "",
        "-status",
        "p1b: Tyranitar",
        "brn",
        "[from] item: Flame Orb",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 5
    )

    segments = get_segments(residual_logs[12])
    assert segments["switch"][0] == [
        "",
        "switch",
        "p2a: Charizard",
        "Charizard, L50, F",
        "42/100",
    ]
    assert len(segments["switch"]) == 1
    assert segments["move"][0] == [
        "",
        "move",
        "p2b: Amoonguss",
        "Pollen Puff",
        "p1b: Tyranitar",
    ]
    assert segments["move"][-1] == ["", "faint", "p1a: Sentret"]
    assert segments["residual"][0] == [
        "",
        "-heal",
        "p1b: Tyranitar",
        "44/207 brn",
        "[from] Grassy Terrain",
    ]
    assert segments["residual"][-1] == [
        "",
        "-damage",
        "p1b: Tyranitar",
        "32/207 brn",
        "[from] brn",
    ]
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 3
    )

    segments = get_segments(residual_logs[18])
    assert segments["move"][0] == [
        "",
        "move",
        "p1b: Furret",
        "Double-Edge",
        "p2a: Pikachu",
    ]
    assert segments["move"][-1] == ["", "faint", "p1b: Furret"]

    # Now go to uturn logs to test when we have a uturn and an empty log from the request
    segments = get_segments(uturn_logs[0])
    assert segments["move"][0] == ["", "move", "p1a: Tyranitar", "Protect", "", "[still]"]
    assert segments["move"][-1] == [
        "",
        "-sidestart",
        "p2: CustomPlayer 1",
        "move: Tailwind",
    ]
    assert segments["residual"][0] == [
        "",
        "-heal",
        "p1a: Tyranitar",
        "153/175",
        "[from] item: Leftovers",
    ]
    assert segments["residual"][-1] == ["", "-fieldend", "move: Trick Room"]
    assert segments["preturn_switch"] == [
        ["", "switch", "p1b: Grimmsnarl", "Grimmsnarl, L50, M", "202/202"]
    ]
    assert segments["init"]
    assert segments["turn"]
    assert len(segments["turn"]) == 1
    assert len(segments) == 6

    segments = get_segments(uturn_logs[1])
    assert segments["switch"][0] == ["", "switch", "p1b: Chi-Yu", "Chi-Yu, L50", "162/162"]
    assert segments["switch"][-1] == ["", "-item", "p1b: Chi-Yu", "Air Balloon"]
    assert segments["battle_mechanic"] == [
        ["", "-terastallize", "p2a: Rillaboom", "Ghost"]
    ]
    assert segments["move"][0] == ["", "-activate", "p2a: Rillaboom", "confusion"]
    assert segments["move"][-1] == ["", "-fail", "p1a: Wo-Chien"]
    assert segments["residual"][0] == [
        "",
        "-damage",
        "p2a: Rillaboom",
        "6/100 brn",
        "[from] Sandstorm",
    ]
    assert segments["residual"][-1] == [
        "",
        "-damage",
        "p2a: Rillaboom",
        "6/100 brn",
        "[from] brn",
    ]

    logs = [
        ["", "switch", "p2b: Bellossom", "Bellossom, L50, F", "150/150"],
        ["", "-damage", "p2b: Bellossom", "132/150", "[from] Stealth Rock"],
        ["", "move", "p1b: Delibird", "Spikes", "p2a: Gardevoir"],
        ["", "-sidestart", "p2: speed_ability1", "Spikes"],
        ["", "move", "p2a: Gardevoir", "Future Sight", "p2b: Bellossom"],
        ["", "-start", "p2a: Gardevoir", "move: Future Sight"],
        ["", "move", "p1a: Tyranitar", "Dragon Tail", "p1b: Delibird"],
        ["", "-damage", "p1b: Delibird", "75/100"],
        ["", "drag", "p1b: Raichu", "Raichu, L50, F, tera:Ground", "95/100"],
        ["", ""],
        ["", "-weather", "none"],
        ["", "-end", "p1b: Raichu", "move: Future Sight"],
        ["", "-damage", "p1b: Raichu", "19/100"],
        ["", "-enditem", "p1b: Raichu", "Red Card", "[of] p2a: Gardevoir"],
        ["", "-damage", "p1a: Tyranitar", "58/100 brn", "[from] brn"],
        ["", "-damage", "p2a: Gardevoir", "40/143 brn", "[from] brn"],
        ["", "upkeep"],
        ["", "drag", "p2a: Dusclops", "Dusclops, L50, M", "62/147"],
        ["", "-damage", "p2a: Dusclops", "44/147", "[from] Stealth Rock"],
        ["", "-damage", "p2a: Dusclops", "26/147", "[from] Spikes"],
        ["", "turn", "6"],
    ]
    segments = get_segments(logs)
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "activation",
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 4
    )

    logs = [
        ["", "move", "p1b: Iron Hands", "Volt Switch", "p2b: Urshifu"],
        ["", "-supereffective", "p2b: Urshifu"],
        ["", "-damage", "p2b: Urshifu", "134/176"],
        ["", ""],
        ["", "-end", "p1b: Iron Hands", "Quark Drive", "[silent]"],
        [
            "",
            "switch",
            "p1b: Volcarona",
            "Volcarona, L50, M",
            "100/100",
            "[from] Volt Switch",
        ],
        ["", "move", "p2a: Incineroar", "Knock Off", "p1a: Nickname"],
        ["", "-damage", "p1a: Nickname", "4/100"],
        ["", "move", "p1a: Nickname", "Volt Switch", "p2a: Incineroar"],
        ["", "-damage", "p2a: Incineroar", "12/202"],
        ["", ""],
        [
            "",
            "switch",
            "p1a: Iron Hands",
            "Iron Hands, L50, tera:Water",
            "28/100",
            "[from] Volt Switch",
        ],
        ["", "move", "p2b: Urshifu", "Surging Strikes", "p1a: Iron Hands"],
        ["", "-resisted", "p1a: Iron Hands"],
        ["", "-crit", "p1a: Iron Hands"],
        ["", "-damage", "p1a: Iron Hands", "20/100"],
        ["", "-resisted", "p1a: Iron Hands"],
        ["", "-crit", "p1a: Iron Hands"],
        ["", "-damage", "p1a: Iron Hands", "13/100"],
        ["", "-resisted", "p1a: Iron Hands"],
        ["", "-crit", "p1a: Iron Hands"],
        ["", "-damage", "p1a: Iron Hands", "6/100"],
        ["", "-hitcount", "p1a: Iron Hands", "3"],
        ["", ""],
        ["", "-sideend", "p1: bfi24championteam", "Reflect"],
        ["", "-fieldend", "move: Trick Room"],
        ["", "upkeep"],
        ["", "turn", "11"],
    ]
    segments = get_segments(logs)
    assert (
        len(
            set(segments.keys())
            & set(
                [
                    "activation",
                    "switch",
                    "battle_mechanic",
                    "move",
                    "state_upkeep",
                    "residual",
                    "preturn_switch",
                ]
            )
        )
        == 2
    )
    assert "move" in segments
    assert segments["move"][0] == [
        "",
        "move",
        "p1b: Iron Hands",
        "Volt Switch",
        "p2b: Urshifu",
    ]
    assert segments["move"][-1] == ["", "-hitcount", "p1a: Iron Hands", "3"]
    assert "state_upkeep" in segments
    assert segments["state_upkeep"] == [
        ["", "-sideend", "p1: bfi24championteam", "Reflect"],
        ["", "-fieldend", "move: Trick Room"],
    ]

    logs = [
        ["", "-end", "p1b: Iron Hands", "Quark Drive", "[silent]"],
        ["", "switch", "p1b: Grimmsnarl", "Grimmsnarl, L50, M", "100/100"],
        ["", "switch", "p2a: Amoonguss", "Amoonguss, L50, M", "100/100"],
        ["", "move", "p2b: Iron Valiant", "Encore", "", "[still]"],
        ["", "-fail", "p2b: Iron Valiant"],
        ["", "move", "p1a: Nickname", "Draco Meteor", "p2b: Iron Valiant"],
        ["", "-immune", "p2b: Iron Valiant"],
        ["", ""],
        ["", "-fieldend", "move: Electric Terrain"],
        ["", "-end", "p2b: Iron Valiant", "Quark Drive"],
        ["", "upkeep"],
        ["", "-enditem", "p2b: Iron Valiant", "Booster Energy"],
        ["", "-activate", "p2b: Iron Valiant", "ability: Quark Drive", "[fromitem]"],
        ["", "-start", "p2b: Iron Valiant", "quarkdrivespe"],
        ["", "turn", "6"],
    ]
    segments = get_segments(logs)

    assert "switch" in segments
    assert segments["switch"][0] == [
        "",
        "-end",
        "p1b: Iron Hands",
        "Quark Drive",
        "[silent]",
    ]
    assert segments["switch"][-1] == [
        "",
        "switch",
        "p2a: Amoonguss",
        "Amoonguss, L50, M",
        "100/100",
    ]
    assert "move" in segments
    assert segments["move"][0] == [
        "",
        "move",
        "p2b: Iron Valiant",
        "Encore",
        "",
        "[still]",
    ]
    assert segments["move"][-1] == ["", "-immune", "p2b: Iron Valiant"]
    assert segments["state_upkeep"] == [
        ["", "-fieldend", "move: Electric Terrain"],
        ["", "-end", "p2b: Iron Valiant", "Quark Drive"],
    ]

    assert segments["post_upkeep"] == [
        ["", "upkeep"],
        ["", "-enditem", "p2b: Iron Valiant", "Booster Energy"],
        ["", "-activate", "p2b: Iron Valiant", "ability: Quark Drive", "[fromitem]"],
        ["", "-start", "p2b: Iron Valiant", "quarkdrivespe"],
    ]

    events = [
        ["", "-end", "p2b: Weezing", "ability: Neutralizing Gas"],
        ["", "-ability", "p1a: Incineroar", "Intimidate", "boost"],
        ["", "-unboost", "p2a: Baxcalibur", "atk", "1"],
        ["", "-unboost", "p2b: Weezing", "atk", "1"],
        ["", "switch", "p2b: Kyogre", "Kyogre, L50", "100/100"],
        ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2b: Kyogre"],
        ["", "move", "p1b: Calyrex", "Protect", "p1b: Calyrex"],
        ["", "-singleturn", "p1b: Calyrex", "Protect"],
        ["", "move", "p1a: Incineroar", "Fake Out", "", "[still]"],
        ["", "-hint", "Fake Out only works on your first turn out."],
        ["", "-fail", "p1a: Incineroar"],
        ["", "move", "p2a: Baxcalibur", "Scale Shot", "p1b: Calyrex"],
        ["", "-activate", "p1b: Calyrex", "move: Protect"],
        ["", ""],
        ["", "-weather", "RainDance", "[upkeep]"],
        ["", "upkeep"],
        ["", "turn", "7"],
    ]
    segments = get_segments(events)
    assert segments["switch"] == [
        ["", "-end", "p2b: Weezing", "ability: Neutralizing Gas"],
        ["", "-ability", "p1a: Incineroar", "Intimidate", "boost"],
        ["", "-unboost", "p2a: Baxcalibur", "atk", "1"],
        ["", "-unboost", "p2b: Weezing", "atk", "1"],
        ["", "switch", "p2b: Kyogre", "Kyogre, L50", "100/100"],
        ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p2b: Kyogre"],
    ]

    assert segments["move"] == [
        ["", "move", "p1b: Calyrex", "Protect", "p1b: Calyrex"],
        ["", "-singleturn", "p1b: Calyrex", "Protect"],
        ["", "move", "p1a: Incineroar", "Fake Out", "", "[still]"],
        ["", "-hint", "Fake Out only works on your first turn out."],
        ["", "-fail", "p1a: Incineroar"],
        ["", "move", "p2a: Baxcalibur", "Scale Shot", "p1b: Calyrex"],
        ["", "-activate", "p1b: Calyrex", "move: Protect"],
    ]


def test_get_residual_and_identifier():
    assert get_residual_and_identifier(
        [
            "",
            "-damage",
            "p1b: Farigiraf",
            "135/195",
            "[from] move: Bind",
            "[partiallytrapped]",
        ]
    ) == ("Bind", "p1: Farigiraf")
    assert get_residual_and_identifier(
        ["", "-activate", "p1a: Chansey", "ability: Healer"]
    ) == ("Healer", "p1: Chansey")
    assert get_residual_and_identifier(
        ["", "-activate", "p1a: Farigaraf", "ability: Cud Chew"]
    ) == ("Cud Chew", "p1: Farigaraf")
    assert get_residual_and_identifier(
        ["", "-end", "p2a: Regigigas", "Slow Start", "[silent]"]
    ) == ("Slow Start", "p2: Regigigas")
    assert get_residual_and_identifier(
        ["", "-heal", "p1a: Smeargle", "94/162", "[from] Ingrain"]
    ) == ("Ingrain", "p1: Smeargle")
    assert get_residual_and_identifier(
        ["", "-ability", "p1b: Espathra", "Speed Boost", "boost"]
    ) == ("Speed Boost", "p1: Espathra")
    assert get_residual_and_identifier(
        ["", "-ability", "p1a: Smeargle", "Moody", "boost"]
    ) == ("Moody", "p1: Smeargle")
    assert get_residual_and_identifier(
        [
            "",
            "-damage",
            "p2a: Blastoise",
            "88/100 brn",
            "[from] Leech Seed",
            "[of] p1b: Wo-Chien",
        ]
    ) == ("Leech Seed", "p2: Blastoise")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Blastoise", "50/100 psn", "[from] psn"]
    ) == ("psn", "p2: Blastoise")
    assert get_residual_and_identifier(
        ["", "-heal", "p2b: Drednaw", "51/100 par", "[from] Grassy Terrain"]
    ) == ("Grassy Terrain", "p2: Drednaw")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Drednaw", "76/100", "[from] Salt Cure"]
    ) == ("Salt Cure", "p2: Drednaw")
    assert get_residual_and_identifier(
        ["", "-heal", "p1a: Smeargle", "43/162", "[from] Aqua Ring"]
    ) == ("Aqua Ring", "p1: Smeargle")
    assert get_residual_and_identifier(
        ["", "-heal", "p2a: Blastoise", "56/100 brn", "[from] ability: Rain Dish"]
    ) == ("Rain Dish", "p2: Blastoise")
    assert get_residual_and_identifier(
        ["", "-heal", "p1b: Farigiraf", "72/195", "[from] item: Leftovers"]
    ) == ("Leftovers", "p1: Farigiraf")
    assert get_residual_and_identifier(
        ["", "-item", "p1a: Tropius", "Sitrus Berry", "[from] ability: Harvest"]
    ) == ("Harvest", "p1: Tropius")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Grimmsnarl", "24/100", "[from] item: Black Sludge"]
    ) == ("Black Sludge", "p2: Grimmsnarl")
    assert get_residual_and_identifier(
        ["", "-status", "p2a: Blastoise", "brn", "[from] item: Flame Orb"]
    ) == ("Flame Orb", "p2: Blastoise")
    assert get_residual_and_identifier(
        ["", "-damage", "p1a: Smeargle", "64/162", "[from] item: Sticky Barb"]
    ) == ("Sticky Barb", "p1: Smeargle")
    assert get_residual_and_identifier(
        ["", "-status", "p1b: Espathra", "tox", "[from] item: Toxic Orb"]
    ) == ("Toxic Orb", "p1: Espathra")
    assert get_residual_and_identifier(
        ["", "-damage", "p1b: Furret", "25/160", "[from] Sandstorm"]
    ) == ("Sandstorm", "p1: Furret")
    assert get_residual_and_identifier(
        ["", "-damage", "p1b: Tyranitar", "32/207 brn", "[from] brn"]
    ) == ("brn", "p1: Tyranitar")
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish0"]) == (
        "Perish Song",
        "p1: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish1"]) == (
        "Perish Song",
        "p1: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish2"]) == (
        "Perish Song",
        "p1: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1a: Espathra", "perish3"]) == (
        "Perish Song",
        "p1: Espathra",
    )
    assert get_residual_and_identifier(
        ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p1b: Tyranitar"]
    ) == (None, None)


def test_get_ability_and_identifier():
    assert get_ability_and_identifier(
        [
            "",
            "-item",
            "p2a: Gardevoir",
            "Iron Ball",
            "[from] ability: Frisk",
            "[of] p1a: Furret",
        ]
    ) == ("Frisk", "p1: Furret")
    assert get_ability_and_identifier(
        ["", "-weather", "Sandstorm", "[from] ability: Sand Stream", "[of] p1b: Tyranitar"]
    ) == ("Sand Stream", "p1: Tyranitar")
    assert get_ability_and_identifier(
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Electric Surge",
            "[of] p1a: Pincurchin",
        ]
    ) == ("Electric Surge", "p1: Pincurchin")
    assert get_ability_and_identifier(
        [
            "",
            "-ability",
            "p2a: Gardevoir",
            "Sand Stream",
            "[from] ability: Trace",
            "[of] p1b: Tyranitar",
        ]
    ) == ("Sand Stream", "p2: Gardevoir")
    assert get_ability_and_identifier(
        ["", "-activate", "p2b: Fluttermane", "ability: Protosynthesis"]
    ) == ("Protosynthesis", "p2: Fluttermane")
    assert get_ability_and_identifier(
        ["", "-copyboost", "p2a: Flamigo", "p2b: Furret", "[from] ability: Costar"]
    ) == ("Costar", "p2: Flamigo")
    assert get_ability_and_identifier(
        [
            "",
            "-activate",
            "p2b: Hypno",
            "ability: Forewarn",
            "darkpulse",
            "[of] p1a: Chi-Yu",
        ]
    ) == ("Forewarn", "p2: Hypno")
    assert get_ability_and_identifier(["", "-ability", "p1b: Calyrex", "As One"]) == (
        "As One",
        "p1: Calyrex",
    )
    assert get_ability_and_identifier(["", "-ability", "p1b: Calyrex", "Unnerve"]) == (
        "Unnerve",
        "p1: Calyrex",
    )
    assert get_ability_and_identifier(
        ["", "-ability", "p2b: Weezing", "Neutralizing Gas"]
    ) == ("Neutralizing Gas", "p2: Weezing")

    assert get_ability_and_identifier(
        ["", "-activate", "p1a: Nickname", "ability: Hadron Engine"]
    ) == ("Hadron Engine", "p1: Nickname")


def test_standardize_pokemon_ident():
    assert standardize_pokemon_ident("[of] p2a: Gardevoir") == "p2: Gardevoir"
    assert standardize_pokemon_ident("p2a: Gardevoir") == "p2: Gardevoir"
    assert standardize_pokemon_ident("[of] p1b: Wo-Chien") == "p1: Wo-Chien"
    assert standardize_pokemon_ident("p1b: Wo-Chien") == "p1: Wo-Chien"


def test_is_ability_event():
    assert is_ability_event(["", "-ability", "p1b: Aerodactyl", "Unnerve"])
    assert is_ability_event(
        ["", "-activate", "p1a: Iron Valiant", "ability: Quark Drive", "[fromitem]"]
    )
    assert is_ability_event(
        [
            "",
            "-item",
            "p1b: Landorus",
            "Life Orb",
            "[from] ability: Frisk",
            "[of] p2a: Furret",
        ]
    )
    assert is_ability_event(["", "-activate", "p2a: Terapagos", "ability: Tera Shift"])
    assert is_ability_event(["", "-ability", "p1b: Calyrex", "As One"])
    assert is_ability_event(["", "-ability", "p1b: Calyrex", "Unnerve"])
    assert is_ability_event(["", "-ability", "p2b: Weezing", "Neutralizing Gas"])
    assert is_ability_event(
        [
            "",
            "-activate",
            "p2b: Hypno",
            "ability: Forewarn",
            "Dark Pulse",
            "[of] p1a: Furret",
        ]
    )
    assert is_ability_event(
        ["", "-weather", "Sandstorm", "[from] ability: Sand Stream", "[of] p1b: Tyranitar"]
    )
    assert is_ability_event(
        ["", "-copyboost", "p1a: Flamigo", "p1b: Furret", "[from] ability: Costar"]
    )
    assert is_ability_event(
        [
            "",
            "-clearboost",
            "p1a: Furret",
            "[from] ability: Curious Medicine",
            "[of] p2b: Slowking-Galar",
        ]
    )
    assert not is_ability_event(["", "move", "p2b: Furret", "Absolute Destruction"])
    assert not is_ability_event(["", "switch", "p2b: Furret"])
    assert not is_ability_event(["", ""])


def test_get_priority_and_identifier():
    gen = 9
    logger = MagicMock()
    battle = DoubleBattle("tag", "username", logger, gen=gen)

    furret = Pokemon(gen=9, species="furret")
    furret._moves = {
        "tailwind": Move("tailwind", gen),
        "grassyglide": Move("grassyglide", gen),
        "gigadrain": Move("gigadrain", gen),
        "trickroom": Move("trickroom", gen),
        "quickattack": Move("quickattack", gen),
        "gigaimpact": Move("gigaimpact", gen),
        "pursuit": Move("pursuit", gen),
    }

    battle.team = {"p1: Furret": furret}

    # Test regular priorities
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Trick Room"], battle
    ) == ("p1: Furret", -7)
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Quick Attack", "p2b: Arceus"], battle
    ) == ("p1: Furret", 1)
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", 0)

    # Test Pursuit
    assert get_priority_and_identifier(["", "move", "p1a: Furret", "Pursuit"], battle) == (
        "p1: Furret",
        None,
    )

    # Test Gale Wings
    furret.set_hp_status("100/100", store=True)
    furret._ability = "galewings"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", 1)
    furret.set_hp_status("99/100", store=True)
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", 0)

    # Test Prankster
    furret._ability = "prankster"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", 1)

    # Test Mycelium Might
    furret._ability = "myceliummight"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", None)

    # Test Stall
    furret._ability = "stall"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Tailwind"], battle
    ) == ("p1: Furret", None)

    # Test Triage
    furret._ability = "triage"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Drain", "p2b: Arceus"], battle
    ) == ("p1: Furret", 3)

    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Floral Healing", "p2b: Arceus"], battle
    ) == ("p1: Furret", 3)

    # Test Grassy Glide
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Grassy Glide", "p2b: Arceus"], battle
    ) == ("p1: Furret", 0)
    battle._fields = {Field.GRASSY_TERRAIN: 0}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Grassy Glide", "p2b: Arceus"], battle
    ) == ("p1: Furret", 1)

    # Test various effects that should nullify priority predictions
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", 0)
    furret._effects = {Effect.QUASH: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)
    furret._effects = {Effect.AFTER_YOU: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)
    furret._effects = {Effect.QUICK_CLAW: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)
    furret._effects = {Effect.CUSTAP_BERRY: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)
    furret._effects = {Effect.DANCER: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)
    furret._effects = {Effect.QUICK_DRAW: 1}
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)

    # Test Lagging Tail
    furret._effects = {}
    furret._item = "laggingtail"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)

    # Test Full Incense
    furret._item = "fullincense"
    assert get_priority_and_identifier(
        ["", "move", "p1a: Furret", "Giga Impact"], battle
    ) == ("p1: Furret", None)


def test_has_status_immunity():
    gen = 9
    battle = DoubleBattle("tag", "username", MagicMock(), gen=gen)
    battle.player_role = "p1"
    furret = Pokemon(gen=9, species="furret")
    furret._active = True
    battle.team = {"p1: Furret": furret}
    battle._active_pokemon = battle.team
    battle._weather = {}

    # no status ability
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)
    assert not has_status_immunity("p1: Furret", Status.BRN, battle)
    assert not has_status_immunity("p1: Furret", Status.TOX, battle)

    # already has status
    furret._status = Status.PAR
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._status = None
    furret._effects = {Effect.SUBSTITUTE: 0}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    # shields down/comatose/good as gold/purifying salt -> all
    furret._effects = {}
    furret._ability = "shieldsdown"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = "goodasgold"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = "comatose"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = "goodasgold"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = "purifyingsalt"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = None
    furret._possible_abilities = ["goodasgold", "frisk"]
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    furret._ability = "magicguard"
    furret._possible_abilities = ["goodasgold", "magicuard"]
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)

    furret._ability = None
    furret._possible_abilities = ["leafguard", "frisk"]
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)

    battle._weather = {Weather.SUNNYDAY: 1}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)

    furret._ability = "frisk"
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)

    battle._fields = {Field.MISTY_TERRAIN: 1}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)

    battle._fields = {}
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)
    battle._side_conditions = {SideCondition.SAFEGUARD: 1}
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    battle._side_conditions = {}

    furret._type_2 = PokemonType.POISON
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)
    assert not has_status_immunity("p1: Furret", Status.BRN, battle)

    furret._type_2 = PokemonType.STEEL
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)
    assert not has_status_immunity("p1: Furret", Status.BRN, battle)

    furret._type_2 = None
    furret._ability = "immunity"
    assert has_status_immunity("p1: Furret", Status.PSN, battle)
    assert has_status_immunity("p1: Furret", Status.TOX, battle)
    assert not has_status_immunity("p1: Furret", Status.BRN, battle)
    furret._ability = "frisk"

    furret._type_2 = PokemonType.ELECTRIC
    assert has_status_immunity("p1: Furret", Status.PAR, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    furret._type_2 = PokemonType.FIRE
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._type_2 = None
    furret._ability = "waterveil"
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._ability = "waterbubble"
    assert has_status_immunity("p1: Furret", Status.BRN, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._ability = "sweetveil"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._ability = "frisk"
    furret._effects = {Effect.SWEET_VEIL: 1}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._effects = {}
    furret._ability = "insomnia"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._ability = "vitalspirit"
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PAR, battle)

    furret._ability = "frisk"
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)

    battle._fields = {Field.ELECTRIC_TERRAIN: 1}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    battle._fields = {}
    furret._effects = {Effect.UPROAR: 1}
    battle._active_pokemon = {"p1a": furret}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    furret._effects = {}
    sentret = Pokemon(gen=9, species="sentret")
    sentret._active = True
    sentret._effects = {Effect.UPROAR: 1}
    battle._team["p1: Sentret"] = sentret
    battle._active_pokemon = {"p1a": furret, "p1b": sentret}
    assert has_status_immunity("p1: Furret", Status.SLP, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    sentret._effects = {}
    assert not has_status_immunity("p1: Furret", Status.SLP, battle)

    # battle still has sunnyday
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    battle._weather = {}
    assert not has_status_immunity("p1: Furret", Status.FRZ, battle)

    furret._type_2 = PokemonType.ICE
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    furret._type_2 = None
    furret._ability = "magmaarmor"
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    furret._ability = "frisk"
    furret._item = "covertcloak"  # freezing only happens as a secondary effect
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)

    furret._item = None
    furret._ability = "shielddust"
    assert has_status_immunity("p1: Furret", Status.FRZ, battle)
    assert not has_status_immunity("p1: Furret", Status.PSN, battle)


def test_has_flinch_immunity():
    furret = Pokemon(gen=9, species="furret")

    furret._ability = "frisk"
    assert not has_flinch_immunity(furret)

    furret._ability = "innerfocus"
    assert has_flinch_immunity(furret)

    furret._ability = "shielddust"
    assert has_flinch_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "innerfocus"]
    assert has_flinch_immunity(furret)

    furret._ability = "frisk"
    assert not has_flinch_immunity(furret)

    furret.item = "covertcloak"
    assert has_flinch_immunity(furret)


# def has_sandstorm_immunity(mon: Pokemon) -> bool:
def test_has_sandstorm_immunity():
    furret = Pokemon(gen=9, species="furret")

    assert not has_sandstorm_immunity(furret)

    furret._type_2 = PokemonType.ROCK
    assert has_sandstorm_immunity(furret)

    furret._type_2 = PokemonType.GROUND
    assert has_sandstorm_immunity(furret)

    furret._type_2 = PokemonType.STEEL
    assert has_sandstorm_immunity(furret)

    furret._type_2 = PokemonType.FAIRY
    assert not has_sandstorm_immunity(furret)

    furret.item = "safetygoggles"
    assert has_sandstorm_immunity(furret)

    furret.item = "unknown_item"
    assert not has_sandstorm_immunity(furret)

    furret._ability = "magicguard"
    assert has_sandstorm_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "magicguard"]
    assert has_sandstorm_immunity(furret)

    furret._ability = "overcoat"
    assert has_sandstorm_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "overcoat"]
    assert has_sandstorm_immunity(furret)

    furret._ability = "sandforce"
    assert has_sandstorm_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "sandforce"]
    assert has_sandstorm_immunity(furret)

    furret._ability = "frisk"
    assert not has_sandstorm_immunity(furret)

    furret._ability = "sandrush"
    assert has_sandstorm_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "sandrush"]
    assert has_sandstorm_immunity(furret)

    furret._ability = "frisk"
    assert not has_sandstorm_immunity(furret)

    furret._ability = "sandveil"
    assert has_sandstorm_immunity(furret)

    furret._ability = "frisk"
    assert not has_sandstorm_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "sandveil"]
    assert has_sandstorm_immunity(furret)


# def has_rage_powder_immunity(mon: Pokemon) -> bool:
def test_has_rage_powder_immunity():
    furret = Pokemon(gen=9, species="furret")

    assert not has_rage_powder_immunity(furret)

    furret._type_2 = PokemonType.GRASS
    assert has_rage_powder_immunity(furret)

    furret._type_2 = None
    furret.item = "safetygoggles"
    assert has_rage_powder_immunity(furret)

    furret.item = None
    assert not has_rage_powder_immunity(furret)

    furret._ability = "stalwart"
    assert has_rage_powder_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "stalwart"]
    assert has_rage_powder_immunity(furret)

    furret._ability = "frisk"
    assert not has_rage_powder_immunity(furret)

    furret._ability = "sniper"
    assert has_rage_powder_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "sniper"]
    assert has_rage_powder_immunity(furret)

    furret._ability = "frisk"
    assert not has_rage_powder_immunity(furret)

    furret._ability = "overcoat"
    assert has_rage_powder_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "overcoat"]
    assert has_rage_powder_immunity(furret)

    furret._ability = "frisk"
    assert not has_rage_powder_immunity(furret)

    furret._ability = "propellertail"
    assert has_rage_powder_immunity(furret)

    furret._ability = None
    furret._possible_abilities = ["frisk", "propellertail"]
    assert has_rage_powder_immunity(furret)

    furret._ability = "frisk"
    assert not has_rage_powder_immunity(furret)


# def is_grounded(mon_ident: str, battle: Union[Battle, DoubleBattle]) -> bool:
def test_is_grounded():
    gen = 9
    battle = DoubleBattle("tag", "username", MagicMock(), gen=gen)
    furret = Pokemon(gen=9, species="furret")
    battle.team = {"p1: Furret": furret}
    assert is_grounded("p1: Furret", battle)

    furret._type_2 = PokemonType.FLYING
    assert not is_grounded("p1: Furret", battle)

    furret._type_2 = None
    furret._ability = "levitate"
    assert not is_grounded("p1: Furret", battle)

    furret._ability = None
    furret._possible_abilities = ["frisk", "levitate"]
    assert not is_grounded("p1: Furret", battle)

    furret._ability = "frisk"
    assert is_grounded("p1: Furret", battle)

    furret._effects = {Effect.MAGNET_RISE: 1}
    assert not is_grounded("p1: Furret", battle)

    furret._effects = {}
    furret.item = "airballoon"
    assert not is_grounded("p1: Furret", battle)

    battle._fields = {Field.GRAVITY: 1}
    assert is_grounded("p1: Furret", battle)

    furret._ability = "levitate"
    assert is_grounded("p1: Furret", battle)

    battle._fields = {}
    assert not is_grounded("p1: Furret", battle)

    furret.item = "ironball"
    assert is_grounded("p1: Furret", battle)


def test_has_unboost_immunity():
    gen = 9
    battle = DoubleBattle("tag", "username", MagicMock(), gen=gen)
    furret = Pokemon(gen=9, species="furret")
    battle.team = {"p1: Furret": furret}
    battle._player_role = "p1"

    # Because we check Flower Veil (where your other mon can affect
    # boost immunity), we check the whole side. We assign furret twice
    # for convenience
    battle._active_pokemon = {"p1a": furret, "p1b": furret}

    # Furret can have keeneye
    assert not has_unboost_immunity("p1: Furret", "atk", battle)
    assert not has_unboost_immunity("p1: Furret", "def", battle)
    assert not has_unboost_immunity("p1: Furret", "spa", battle)
    assert not has_unboost_immunity("p1: Furret", "spd", battle)
    assert not has_unboost_immunity("p1: Furret", "spe", battle)
    assert not has_unboost_immunity("p1: Furret", "evasion", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)

    furret._possible_abilities = ["frisk", "hypercutter"]
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert not has_unboost_immunity("p1: Furret", "def", battle)

    furret._ability = "hypercutter"
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert not has_unboost_immunity("p1: Furret", "def", battle)

    furret._ability = "frisk"
    assert not has_unboost_immunity("p1: Furret", "atk", battle)
    assert not has_unboost_immunity("p1: Furret", "def", battle)

    furret._ability = "bigpeck"
    assert not has_unboost_immunity("p1: Furret", "atk", battle)
    assert has_unboost_immunity("p1: Furret", "def", battle)

    furret._ability = "mindseye"
    assert not has_unboost_immunity("p1: Furret", "atk", battle)
    assert not has_unboost_immunity("p1: Furret", "def", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)

    furret._ability = "clearbody"
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert has_unboost_immunity("p1: Furret", "def", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)

    furret._ability = "fullmetalbody"
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert has_unboost_immunity("p1: Furret", "def", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)

    furret._ability = None
    furret._boosts["def"] = -6
    assert has_unboost_immunity("p1: Furret", "def", battle)
    assert not has_unboost_immunity("p1: Furret", "accuracy", battle)

    battle._side_conditions = {SideCondition.MIST: 1}
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert has_unboost_immunity("p1: Furret", "def", battle)
    assert has_unboost_immunity("p1: Furret", "spa", battle)
    assert has_unboost_immunity("p1: Furret", "spd", battle)
    assert has_unboost_immunity("p1: Furret", "spe", battle)
    assert has_unboost_immunity("p1: Furret", "evasion", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)

    battle._side_conditions = {}
    furret._item = "clearamulet"
    assert has_unboost_immunity("p1: Furret", "atk", battle)
    assert has_unboost_immunity("p1: Furret", "def", battle)
    assert has_unboost_immunity("p1: Furret", "spa", battle)
    assert has_unboost_immunity("p1: Furret", "spd", battle)
    assert has_unboost_immunity("p1: Furret", "spe", battle)
    assert has_unboost_immunity("p1: Furret", "evasion", battle)
    assert has_unboost_immunity("p1: Furret", "accuracy", battle)
