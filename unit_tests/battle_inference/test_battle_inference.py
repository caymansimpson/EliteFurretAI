# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from elitefurretai.battle_inference.battle_inference import BattleInference
from elitefurretai.battle_inference.inference_utils import (
    get_ability_and_identifier,
    get_pokemon_ident,
    get_residual_and_identifier,
    get_segments,
)


# TODO: test each individual order
# TODO: test each speed range
def test_battle_inference():
    mock = MagicMock()
    mock.gen = 9
    mock.player_username = "elitefurretai"
    battle_inference = BattleInference(mock)
    assert battle_inference
    raise NotImplementedError


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

    # TODO:
    segments = get_segments(edgecase_logs[15])
    print(segments)
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
    assert len(segments) == 5

    segments = get_segments(uturn_logs[1])
    print(segments)
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
    ) == ("Bind", "p1b: Farigiraf")
    assert get_residual_and_identifier(
        ["", "-activate", "p1a: Chansey", "ability: Healer"]
    ) == ("Healer", "p1a: Chansey")
    assert get_residual_and_identifier(
        ["", "-activate", "p1a: Farigaraf", "ability: Cud Chew"]
    ) == ("Cud Chew", "p1a: Farigaraf")
    assert get_residual_and_identifier(
        ["", "-end", "p2a: Regigigas", "Slow Start", "[silent]"]
    ) == ("Slow Start", "p2a: Regigigas")
    assert get_residual_and_identifier(
        ["", "-heal", "p1a: Smeargle", "94/162", "[from] Ingrain"]
    ) == ("Ingrain", "p1a: Smeargle")
    assert get_residual_and_identifier(
        ["", "-ability", "p1b: Espathra", "Speed Boost", "boost"]
    ) == ("Speed Boost", "p1b: Espathra")
    assert get_residual_and_identifier(
        ["", "-ability", "p1a: Smeargle", "Moody", "boost"]
    ) == ("Moody", "p1a: Smeargle")
    assert get_residual_and_identifier(
        [
            "",
            "-damage",
            "p2a: Blastoise",
            "88/100 brn",
            "[from] Leech Seed",
            "[of] p1b: Wo-Chien",
        ]
    ) == ("Leech Seed", "p2a: Blastoise")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Blastoise", "50/100 psn", "[from] psn"]
    ) == ("psn", "p2a: Blastoise")
    assert get_residual_and_identifier(
        ["", "-heal", "p2b: Drednaw", "51/100 par", "[from] Grassy Terrain"]
    ) == ("Grassy Terrain", "p2b: Drednaw")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Drednaw", "76/100", "[from] Salt Cure"]
    ) == ("Salt Cure", "p2a: Drednaw")
    assert get_residual_and_identifier(
        ["", "-heal", "p1a: Smeargle", "43/162", "[from] Aqua Ring"]
    ) == ("Aqua Ring", "p1a: Smeargle")
    assert get_residual_and_identifier(
        ["", "-heal", "p2a: Blastoise", "56/100 brn", "[from] ability: Rain Dish"]
    ) == ("Rain Dish", "p2a: Blastoise")
    assert get_residual_and_identifier(
        ["", "-heal", "p1b: Farigiraf", "72/195", "[from] item: Leftovers"]
    ) == ("Leftovers", "p1b: Farigiraf")
    assert get_residual_and_identifier(
        ["", "-item", "p1a: Tropius", "Sitrus Berry", "[from] ability: Harvest"]
    ) == ("Harvest", "p1a: Tropius")
    assert get_residual_and_identifier(
        ["", "-damage", "p2a: Grimmsnarl", "24/100", "[from] item: Black Sludge"]
    ) == ("Black Sludge", "p2a: Grimmsnarl")
    assert get_residual_and_identifier(
        ["", "-status", "p2a: Blastoise", "brn", "[from] item: Flame Orb"]
    ) == ("Flame Orb", "p2a: Blastoise")
    assert get_residual_and_identifier(
        ["", "-damage", "p1a: Smeargle", "64/162", "[from] item: Sticky Barb"]
    ) == ("Sticky Barb", "p1a: Smeargle")
    assert get_residual_and_identifier(
        ["", "-status", "p1b: Espathra", "tox", "[from] item: Toxic Orb"]
    ) == ("Toxic Orb", "p1b: Espathra")
    assert get_residual_and_identifier(
        ["", "-damage", "p1b: Furret", "25/160", "[from] Sandstorm"]
    ) == ("Sandstorm", "p1b: Furret")
    assert get_residual_and_identifier(
        ["", "-damage", "p1b: Tyranitar", "32/207 brn", "[from] brn"]
    ) == ("brn", "p1b: Tyranitar")
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish0"]) == (
        "Perish Song",
        "p1b: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish1"]) == (
        "Perish Song",
        "p1b: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1b: Wo-Chien", "perish2"]) == (
        "Perish Song",
        "p1b: Wo-Chien",
    )
    assert get_residual_and_identifier(["", "-start", "p1a: Espathra", "perish3"]) == (
        "Perish Song",
        "p1a: Espathra",
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
            "[identify]",
        ]
    ) == ("Frisk", "p1a: Furret")
    assert get_ability_and_identifier(
        ["", "-weather", "Sandstorm", "[from] ability: Sand Stream", "[of] p1b: Tyranitar"]
    ) == ("Sand Stream", "p1b: Tyranitar")
    assert get_ability_and_identifier(
        [
            "",
            "-fieldstart",
            "move: Electric Terrain",
            "[from] ability: Electric Surge",
            "[of] p1a: Pincurchin",
        ]
    ) == ("Electric Surge", "p1a: Pincurchin")
    assert get_ability_and_identifier(
        [
            "",
            "-ability",
            "p2a: Gardevoir",
            "Sand Stream",
            "[from] ability: Trace",
            "[of] p1b: Tyranitar",
        ]
    ) == ("Sand Stream", "p2a: Gardevoir")
    assert get_ability_and_identifier(
        ["", "-activate", "p2b: Fluttermane", "ability: Protosynthesis"]
    ) == ("Protosynthesis", "p2b: Fluttermane")
    assert get_ability_and_identifier(
        ["-copyboost", "p2a: Flamigo", "p2b: Furret", "[from] ability: Costar"]
    ) == ("Costar", "p2a: Flamigo")
    assert get_ability_and_identifier(
        ["-activate", "p2b: Hypno", "ability: Forewarn", "darkpulse", "[of] p1a: Chi-Yu"]
    ) == ("Forewarn", "p2b: Hypno")
    assert get_ability_and_identifier(["", "-ability", "p1b: Calyrex", "As One"]) == (
        "As One",
        "p1b: Calyrex",
    )
    assert get_ability_and_identifier(["", "-ability", "p1b: Calyrex", "Unnerve"]) == (
        "Unnerve",
        "p1b: Calyrex",
    )
    assert get_ability_and_identifier(
        ["", "-ability", "p2b: Weezing", "Neutralizing Gas"]
    ) == ("Neutralizing Gas", "p2b: Weezing")


def test_get_pokemon_ident():
    assert get_pokemon_ident("[of] p2a: Gardevoir") == "p2a: Gardevoir"
    assert get_pokemon_ident("p2a: Gardevoir") == "p2a: Gardevoir"
    assert get_pokemon_ident("[of] p1b: Wo-Chien") == "p1b: Wo-Chien"
    assert get_pokemon_ident("p1b: Wo-Chien") == "p1b: Wo-Chien"
