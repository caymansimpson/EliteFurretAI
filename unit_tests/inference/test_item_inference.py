# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from poke_env.battle import DoubleBattle, Pokemon, PokemonType
from poke_env.data.gen_data import GenData

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import copy_pokemon
from elitefurretai.inference.item_inference import ItemInference


def generate_item_inference_and_inferences():
    gen = 9
    battle = DoubleBattle("tag", "username", MagicMock(), gen=gen)
    battle._players = [{"username": "elitefurretai"}, {"username": "joeschmoe"}]
    battle.player_role = "p1"
    tp = {
        Pokemon(gen=gen, species="Furret"),
        Pokemon(gen=gen, species="Raichu"),
        Pokemon(gen=gen, species="Tyranitar"),
        Pokemon(gen=gen, species="Shuckle"),
    }
    battle._teampreview_opponent_team = [copy_pokemon(mon, gen) for mon in tp]
    battle.teampreview_team = [copy_pokemon(mon, gen) for mon in tp]
    battle.team = {mon.identifier("p1"): copy_pokemon(mon, gen) for mon in tp}

    inferences = BattleInference(battle)
    ii = ItemInference(battle=battle, inferences=inferences)

    # I do this because ItemInference copies the battle. For testing purposes, we want these
    # battle states to be synced so that when we update ItemInference's, Inferences also gets updated
    # too (for introducing new opponent pokemon on switch). Normally, we would not do this, because
    # Inferences would be synced to the actual battle object, and ItemInference would be catching up
    inferences._battle = ii._battle

    # Have to fast forward a turn, because Item Inference doesn't try to parse
    # the last turn
    ii._battle._turn += 1

    return ii, inferences


def test_check_items_covertcloak_boost():

    # Icy Wind
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM
    assert ii._battle.get_pokemon("p2: Furret").item == "airballoon"
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is False
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet")
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak")

    # Bulldoze
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Bulldoze", "p2a: Furret", "[spread] p1b,p2a,p2b"],
        ["", "-damage", "p1b: Raichu", "97/100"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p1b: Raichu", "spe", "1"],
        ["", "-unboost", "p2b: Shuckle", "spe", "1"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak")
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet")
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet") is False

    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Bulldoze", "p2a: Furret", "[spread] p1b,p2a,p2b"],
        ["", "-damage", "p1b: Raichu", "97/100"],
        ["", "-damage", "p2a: Furret", "0 fnt"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p1b: Raichu", "spe", "1"],
        ["", "faint", "p2a: Furret"],
    ]

    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak")
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet")
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak") is None
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is None

    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Bulldoze", "p2a: Furret", "[spread] p1b,p2a,p2b"],
        ["", "-damage", "p1b: Raichu", "97/100"],
        ["", "-damage", "p2a: Furret", "0 fnt"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p1b: Raichu", "spe", "1"],
        ["", "-unboost", "p2b: Shuckle", "spe", "1"],
        ["", "faint", "p2a: Furret"],
    ]

    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak") is None
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is None
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet") is False


def test_check_items_covertcloak_flinch():
    # Fake Out
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Fake Out", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "covertcloak"
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak")
    assert not inferences.get_flag("p2: Furret", "can_be_clearamulet")

    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "move", "p1b: Rillaboom", "Fake Out", "p2b: Espeon"],
        ["", "-resisted", "p2b: Espeon"],
        ["", "-damage", "p2b: Espeon", "88/100"],
        ["", "cant", "p2b: Espeon", "flinch"],
        ["", "move", "p1a: Pelipper", "Hurricane", "p2b: Espeon"],
        ["", "-resisted", "p2b: Espeon"],
        ["", "-damage", "p2b: Espeon", "56/100"],
        ["", "-start", "p2b: Espeon", "confusion"],
        ["", "move", "p2a: Muk", "Toxic", "p2b: Espeon"],
        ["", "move", "p2b: Espeon", "Toxic", "p2a: Muk"],
        ["", "-immune", "p2a: Muk"],
    ]
    ii._battle.parse_message(["", "switch", "p1b: Rillaboom", "Rillaboom, L50", "100/100"])
    ii._battle.parse_message(["", "switch", "p2b: Espeon", "Espeon, L50", "100/100"])
    ii._battle.parse_message(["", "switch", "p1a: Pelipper", "Pelipper, L50", "100/100"])
    ii._battle.parse_message(["", "switch", "p2a: Muk", "Muk, L50", "100/100"])
    ii.check_items(events)
    assert inferences.get_flag("p2: Espeon", "item") == GenData.UNKNOWN_ITEM


def test_check_items_covertcloak_status():
    # Nuzzle
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1b: Raichu", "Nuzzle", "p2b: Shuckle"],
        ["", "-damage", "p2b: Shuckle", "98/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "move", "p1b: Raichu", "Nuzzle", "p2b: Raichu"],
        ["", "-damage", "p2b: Raichu", "2/100"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Shuckle", "item") == "covertcloak"
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak")
    assert not inferences.get_flag("p2: Shuckle", "can_be_clearamulet")
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Raichu", "item") == GenData.UNKNOWN_ITEM
    assert not inferences.get_flag("p2: Raichu", "can_be_covertcloak")
    assert not inferences.get_flag("p2: Raichu", "can_be_clearamulet")


def test_check_items_covertcloak_saltcure():
    # Salt Cure
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Salt Cure", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", ""],
        ["", "upkeep"],
        ["", "turn", "11"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "covertcloak"
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak")
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is False
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM
    assert not inferences.get_flag("p2: Shuckle", "can_be_covertcloak")

    # Salt Cure shouldnt lead to inferred covert cloak cuz of miss
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Salt Cure", "p2a: Furret"],
        ["", "-miss", "p1a: Smeagle", "p2a: Furret"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", ""],
        ["", "upkeep"],
        ["", "turn", "11"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM

    # Salt Cure shouldnt lead to inferred covert cloak cuz Furret switched out
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Smeargle", "Salt Cure", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "U-turn", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", ""],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50, F", "100/100"],
        ["", ""],
        ["", "upkeep"],
        ["", "turn", "11"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM


def test_check_items_clearamulet():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-ability", "p1b: Raichu", "Intimidate", "boost"],
        ["", "-unboost", "p2a: Furret", "atk", "1"],
        ["", "-unboost", "p2b: Shuckle", "atk", "1"],
        ["", "move", "p1a: Smeargle", "Salt Cure", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", ""],
        ["", "upkeep"],
        ["", "turn", "11"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "covertcloak"
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak") is True
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet") is False

    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-ability", "p1b: Raichu", "Intimidate", "boost"],
        ["", "-unboost", "p2b: Shuckle", "atk", "1"],
        ["", "turn", "11"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "clearamulet"
    assert inferences.get_flag("p2: Furret", "can_be_clearamulet") is True
    assert inferences.get_flag("p2: Furret", "can_be_covertcloak") is False
    assert inferences.get_flag("p2: Furret", "can_be_choice") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_clearamulet") is False
    assert inferences.get_flag("p2: Shuckle", "can_be_covertcloak") is None
    assert inferences.get_flag("p2: Shuckle", "item") == GenData.UNKNOWN_ITEM


def test_check_items_lightclay():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p1a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "turn", "1"],
        ["", "turn", "2"],
        ["", "turn", "3"],
        ["", "turn", "4"],
        ["", "turn", "5"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == GenData.UNKNOWN_ITEM

    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p2a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "turn", "1"],
    ]
    ii.check_items(events)

    events = [
        ["", "move", "p2a: Furret", "Follow Me", "p2a: Furret"],
        ["", "turn", "6"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "lightclay"

    # Test setting it twice
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "switch", "p2b: Raichu", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p2a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "move", "p2b: Raichu", "Reflect", "[still]"],
        ["", "-fail", "p2b: Raichu"],
        ["", "turn", "1"],
        ["", "turn", "2"],
        ["", "turn", "3"],
        ["", "turn", "4"],
        ["", "move", "p2a: Furret", "Reflect", "[still]"],
        ["", "-fail", "p2a: Furret"],
        ["", "turn", "5"],
        ["", "turn", "6"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "lightclay"
    assert inferences.get_flag("p2: Raichu", "item") == GenData.UNKNOWN_ITEM


def test_check_items_can_be_choice():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "1"],
    ]
    ii.check_items(events)
    assert ii._inferences.get_flag("p2: Furret", "can_be_choice")

    new_events = [
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "2"],
    ]
    ii.check_items(new_events)
    assert ii._inferences.get_flag("p2: Furret", "can_be_choice")

    newer_events = [
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
    ]
    ii.check_items(newer_events)
    assert not ii._inferences.get_flag("p2: Furret", "can_be_choice")


def test_check_items_can_have_assault_vest():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "3/100"],
        ["", "turn", "1"],
    ]
    ii.check_items(events)
    assert not ii._inferences.get_flag("p2: Furret", "has_status_move")

    new_events = [
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "2"],
    ]
    ii.check_items(new_events)
    assert not ii._inferences.get_flag("p2: Furret", "has_status_move")

    newer_events = [
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
        ["", "turn", "3"],
    ]
    ii.check_items(newer_events)
    assert ii._inferences.get_flag("p2: Furret", "has_status_move")

    newest_events = [
        ["", "move", "p2a: Furret", "Quick Attack", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "1/100"],
        ["", "turn", "3"],
    ]
    ii.check_items(newest_events)
    assert ii._inferences.get_flag("p2: Furret", "has_status_move")


def test_check_opponent_switch():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "move", "p1a: Furret", "Spikes"],
        ["", "-sidestart", "p2: joeschmoe", "Spikes"],
        ["", "move", "p2a: Furret", "Volt Switch", "p2b: Shuckle"],
        ["", "-damage", "p2b: Shuckle", "62/100 par"],
        ["", ""],
        [
            "",
            "switch",
            "p2a: Tyranitar",
            "Tyranitar, L50",
            "100/100",
            "[from] Volt Switch",
        ],
        ["", "move", "p2b: Shuckle", "U-turn", "p2a: Tyranitar"],
        ["", "-damage", "p2a: Tyranitar", "97/100"],
        ["", ""],
        ["", "switch", "p2b: Raichu", "Raichu, L50", "100/100", "[from] U-turn"],
        ["", "-damage", "p2b: Raichu", "80/100", "[from] Spikes"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Raichu", "item") == GenData.UNKNOWN_ITEM
    assert inferences.get_flag("p2: Tyranitar", "item") == "heavydutyboots"

    # Just checking that we run through all events
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2a: Gothitelle", "Gothitelle, L50", "100/100"],
        ["", "switch", "p2b: Lapras", "Lapras, L50", "100/100"],
        ["", "switch", "p1a: cloud", "Altaria, L50", "100/100"],
        ["", "switch", "p1b: Goth latina", "Gothitelle, L50", "100/100"],
        ["", "move", "p2b: Lapras", "Protect", "p2b: Lapras"],
        ["", "-singleturn", "p2b: Lapras", "Protect"],
        ["", "move", "p2a: Gothitelle", "Protect", "p2a: Gothitelle"],
        ["", "-singleturn", "p2a: Gothitelle", "Protect"],
        ["", "move", "p1b: Goth latina", "Psychic", "p2b: Lapras"],
        ["", "-activate", "p2b: Lapras", "move: Protect"],
        ["", "move", "p1a: cloud", "Hyper Voice", "p2b: Lapras", "[spread]"],
        ["", "-activate", "p2a: Gothitelle", "move: Protect"],
        ["", "-activate", "p2b: Lapras", "move: Protect"],
        ["", ""],
        ["", "-start", "p1b: Goth latina", "perish0"],
        ["", "-start", "p1a: cloud", "perish0"],
        ["", "-start", "p2b: Lapras", "perish0"],
        ["", "-start", "p2a: Gothitelle", "perish0"],
        ["", "faint", "p1b: Goth latina"],
        ["", "faint", "p1a: cloud"],
        ["", "faint", "p2b: Lapras"],
        ["", "faint", "p2a: Gothitelle"],
        ["", "upkeep"],
        ["", ""],
        ["", "turn", "1"],
    ]
    ii.check_items(events)
    assert ii._battle.turn == 1
    assert ii._battle.active_pokemon == [None, None]
    assert ii._battle.opponent_active_pokemon == [None, None]


def test_check_items_heavy_duty_boots():
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Spikes", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Spikes"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Shuckle", "item") == "heavydutyboots"

    # Should assign Shuckle heavydutyboots if it doesnt take dmg and ttar is immune
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Toxic Spikes", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Toxic Spikes"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._type_2 = PokemonType.POISON
    ii.check_items(events)
    assert inferences.get_flag("p2: Shuckle", "item") == "heavydutyboots"
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Should assign Shuckle heavydutyboots if it doesnt take dmg and ttar is immune
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Toxic Spikes", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Toxic Spikes"],
        ["", "switch", "p2b: Shuckle", "Shuckle, L50", "100/100"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._type_2 = PokemonType.STEEL
    ii.check_items(events)
    assert inferences.get_flag("p2: Shuckle", "item") == "heavydutyboots"
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it has levitate
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Toxic Spikes", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Toxic Spikes"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._ability = "levitate"
    ii.check_items(events)
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it could have levitate
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Toxic Spikes", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Toxic Spikes"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._ability = None
    ii._battle._opponent_team["p2: Tyranitar"]._possible_abilities = ["levitate", "frisk"]
    ii.check_items(events)
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it could have levitate
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Sticky Web", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Sticky Web"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._ability = None
    ii._battle._opponent_team["p2: Tyranitar"]._possible_abilities = ["levitate", "frisk"]
    ii.check_items(events)
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots for stealthrock even if it has levitate
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Stealth Rock", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Stealth Rock"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._ability = "magicguard"
    ii.check_items(events)
    assert inferences.get_flag("p2: Tyranitar", "item") == GenData.UNKNOWN_ITEM

    # Should assign Tyranitar heavydutyboots for stealthrock even if it possibly has levitate
    ii, inferences = generate_item_inference_and_inferences()
    ii._battle._opponent_team = {
        "p2: Tyranitar": Pokemon(gen=9, details="Tyranitar, L50, M")
    }

    events = [
        ["", "switch", "p1a: Smeargle", "Smeargle, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, F", "167/167"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
        ["", "move", "p1a: Smeargle", "Stealth Rock", "p2a: Smeargle"],
        ["", "-sidestart", "p2: joeschmoe", "Stealth Rock"],
        ["", "switch", "p2a: Tyranitar", "Tyranitar, L50", "100/100"],
    ]
    ii._battle._opponent_team["p2: Tyranitar"]._ability = None
    ii.check_items(events)
    assert inferences.get_flag("p2: Tyranitar", "item") == "heavydutyboots"


def test_safety_goggles():
    # Rage Powder
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p1a: Furret", "Furret, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, M", "100/100"],
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p1a: Furret", "Rage Powder", "p1a: Furret"],
        ["", "-singleturn", "p1a: Furret", "move: Rage Powder"],
        ["", "move", "p2a: Furret", "Double Edge", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "97/100"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "safetygoggles"

    # Sandstorm
    ii, inferences = generate_item_inference_and_inferences()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "switch", "p2b: Tyranitar", "Tyranitar, L50, F", "100/100"],
        [
            "",
            "-weather",
            "Sandstorm",
            "[from] ability: Sand Stream",
            "[of] p2b: Tyranitar",
        ],
        ["", "move", "p1a: Furret", "Rage Powder", "p1a: Furret"],
        ["", "-singleturn", "p1a: Furret", "move: Rage Powder"],
        ["", ""],
        ["", "-weather", "Sandstorm", "[upkeep]"],
        ["", "-damage", "p1b: Raichu", "97/100", "[from] Sandstorm"],
        ["", "-damage", "p1a: Furret", "38/130", "[from] Sandstorm"],
    ]
    ii.check_items(events)
    assert inferences.get_flag("p2: Furret", "item") == "safetygoggles"
