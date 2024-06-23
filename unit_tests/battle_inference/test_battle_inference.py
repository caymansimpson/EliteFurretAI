# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from elitefurretai.battle_inference.battle_inference import BattleInference
from elitefurretai.battle_inference.inference_utils import get_segments


# TODO: test each individual order
# TODO: test each speed range
def test_battle_inference():
    battle_inference = BattleInference(MagicMock())
    assert battle_inference
    raise NotImplementedError


def test_get_segments(residual_logs, edgecase_logs):
    segments = get_segments(edgecase_logs[0])
    assert segments["preturn_switch"][0] == [
        "",
        "-fieldstart",
        "move: Psychic Terrain",
        "[from] ability: Psychic Surge",
        "[of] p1a: Indeedee",
    ]
    assert segments["preturn_switch"][1] == [
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
        == 1
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
        == 3
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
        == 1
    )

    segments = get_segments(edgecase_logs[4])
    assert segments["move"][0] == [
        "",
        "move",
        "p2a: Heracross",
        "Endure",
        "p2a: Heracross",
    ], ["", "-singleturn", "p2a: Heracross", "move: Endure"]
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
        == 1
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
        == 2
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
        == 2
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

    segments = get_segments(edgecase_logs[9])
    assert segments["move"][0] == ["", "-activate", "p1a: Farigiraf", "item: Quick Claw"]
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
        == 3
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
    assert segments["preturn_switch"][0] == [
        "",
        "-weather",
        "RainDance",
        "[from] ability: Drizzle",
        "[of] p1a: Pelipper",
    ]
    assert len(segments["preturn_switch"]) == 1
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
        == 2
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

    assert len(get_segments(residual_logs[0])) == 0
    assert len(get_segments(residual_logs[1])) == 0
    assert len(get_segments(residual_logs[2])) == 0
    assert len(get_segments(residual_logs[3])) == 0
    assert len(get_segments(residual_logs[4])) == 0

    segments = get_segments(residual_logs[5])
    assert segments["preturn_switch"][0] == [
        "",
        "-weather",
        "Sandstorm",
        "[from] ability: Sand Stream",
        "[of] p1b: Tyranitar",
    ]
    assert len(segments["preturn_switch"]) == 1
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
        == 4
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
