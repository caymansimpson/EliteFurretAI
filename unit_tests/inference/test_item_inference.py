# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from poke_env.data.gen_data import GenData
from poke_env.environment import DoubleBattle, Observation, Pokemon, PokemonType

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import (
    copy_pokemon,
    get_pokemon,
    get_showdown_identifier,
)
from elitefurretai.inference.item_inference import ItemInference


def generate_item_inference():
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
    battle._teampreview_opponent_team = {copy_pokemon(mon, gen) for mon in tp}
    battle.teampreview_team = {copy_pokemon(mon, gen) for mon in tp}
    battle.team = {
        get_showdown_identifier(mon, "p1"): copy_pokemon(mon, gen) for mon in tp
    }
    battle._opponent_team = {
        get_showdown_identifier(mon, "p2"): copy_pokemon(mon, gen) for mon in tp
    }

    # Initiate teh battle with what I need
    bi = BattleInference(battle)
    ii = ItemInference(battle=battle, inferences=bi)

    # # Add opponent_mons
    # bi._opponent_mons = {
    #     ident: BattleInference.load_opponent_set(mon)
    #     for ident, mon in ii._battle._opponent_team.items()
    # }
    return ii


def test_check_items_covertcloak():

    # Icy Wind
    ii = generate_item_inference()
    events = [
        ["", "-item", "p2a: Furret", "Air Balloon"],
        ["", "move", "p1a: Smeargle", "Icy Wind", "p2a: Furret", "[spread] p2a,p2b"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p2a: Furret", "spe", "1"],
    ]
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Shuckle", ii._battle).item == "covertcloak"
    assert get_pokemon("p2: Furret", ii._battle).item == "airballoon"

    # Bulldoze
    ii = generate_item_inference()
    events = [
        ["", "move", "p1a: Smeargle", "Bulldoze", "p2a: Furret", "[spread] p1b,p2a,p2b"],
        ["", "-damage", "p1b: Raichu", "97/100"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "-damage", "p2b: Shuckle", "99/100"],
        ["", "-unboost", "p1b: Raichu", "spe", "1"],
        ["", "-unboost", "p2b: Shuckle", "spe", "1"],
    ]
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "covertcloak"

    # Fake Out
    ii = generate_item_inference()
    events = [
        ["", "move", "p1a: Smeargle", "Fake Out", "p2a: Furret"],
        ["", "-damage", "p2a: Furret", "97/100"],
        ["", "move", "p1b: Raichu", "Agility", "p1b: Raichu"],
        ["", "-boost", "p1b: Raichu", "spe", "2"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
    ]
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "covertcloak"

    # Nuzzle
    ii = generate_item_inference()
    events = [
        ["", "move", "p1b: Raichu", "Nuzzle", "p2b: Shuckle"],
        ["", "-damage", "p2b: Shuckle", "98/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "move", "p1b: Raichu", "Nuzzle", "p2b: Raichu"],
        ["", "-damage", "p2b: Raichu", "2/100"],
    ]
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Shuckle", ii._battle).item == "covertcloak"
    assert get_pokemon("p2: Furret", ii._battle).item == GenData.UNKNOWN_ITEM
    assert get_pokemon("p2: Raichu", ii._battle).item == GenData.UNKNOWN_ITEM


def test_check_items_lightclay():
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == GenData.UNKNOWN_ITEM

    ii = generate_item_inference()
    events = [
        ["", "turn", "0"],
        ["", "switch", "p2a: Furret", "Furret, L50, M", "100/100"],
        ["", "move", "p2a: Furret", "Reflect", "p2a: Furret"],
        ["", "-sidestart", "p2: joeschmoe", "move: Reflect"],
        ["", "turn", "1"],
        ["", "turn", "2"],
        ["", "turn", "3"],
        ["", "turn", "4"],
        ["", "turn", "5"],
        ["", "turn", "6"],
    ]
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "lightclay"

    # Test setting it twice
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "lightclay"
    assert get_pokemon("p2: Raichu", ii._battle).item == GenData.UNKNOWN_ITEM


def test_check_items_can_be_choice():
    ii = generate_item_inference()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "1"],
    ]
    ii.check_items(Observation(events=events))
    assert ii._inferences.get_flag("p2: Furret", "can_be_choice")

    new_events = [
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "2"],
    ]
    ii.check_items(Observation(events=new_events))
    assert ii._inferences.get_flag("p2: Furret", "can_be_choice")

    newer_events = [
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
    ]
    ii.check_items(Observation(events=newer_events))
    assert not ii._inferences.get_flag("p2: Furret", "can_be_choice")


def test_check_items_can_have_assault_vest():
    ii = generate_item_inference()
    events = [
        ["", "switch", "p2a: Furret", "Furret, L50, F", "100/100"],
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "3/100"],
        ["", "turn", "1"],
    ]
    ii.check_items(Observation(events=events))
    assert not ii._inferences.get_flag("p2: Furret", "has_status_move")

    new_events = [
        ["", "move", "p2a: Furret", "Last Resort", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "2/100"],
        ["", "turn", "2"],
    ]
    ii.check_items(Observation(events=new_events))
    assert not ii._inferences.get_flag("p2: Furret", "has_status_move")

    newer_events = [
        ["", "move", "p2a: Furret", "Agility", "p2a: Furret"],
        ["", "-boost", "p2a: Furret", "spe", "2"],
        ["", "turn", "3"],
    ]
    ii.check_items(Observation(events=newer_events))
    assert ii._inferences.get_flag("p2: Furret", "has_status_move")

    newest_events = [
        ["", "move", "p2a: Furret", "Quick Attack", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "1/100"],
        ["", "turn", "3"],
    ]
    ii.check_items(Observation(events=newest_events))
    assert ii._inferences.get_flag("p2: Furret", "has_status_move")


def test_check_items_heavy_duty_boots():
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Shuckle", ii._battle).item == "heavydutyboots"
    assert get_pokemon("p2: Furret", ii._battle).item == "airballoon"

    # Should assign Shuckle heavydutyboots if it doesnt take dmg and ttar is immune
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Shuckle", ii._battle).item == "heavydutyboots"
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Should assign Shuckle heavydutyboots if it doesnt take dmg and ttar is immune
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Shuckle", ii._battle).item == "heavydutyboots"
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it has levitate
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it could have levitate
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots if it could have levitate
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Shouldnt assign Tyranitar heavydutyboots for stealthrock even if it has levitate
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Tyranitar", ii._battle).item == GenData.UNKNOWN_ITEM

    # Should assign Tyranitar heavydutyboots for stealthrock even if it possibly has levitate
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Tyranitar", ii._battle).item == "heavydutyboots"


def test_safety_goggles():
    # Rage Powder
    ii = generate_item_inference()
    events = [
        ["", "switch", "p1a: Furret", "Furret, L50, M", "100/100"],
        ["", "switch", "p1b: Raichu", "Raichu, L50, M", "100/100"],
        ["", "move", "p1a: Furret", "Rage Powder", "p1a: Furret"],
        ["", "-singleturn", "p1a: Furret", "move: Rage Powder"],
        ["", "move", "p2a: Furret", "Double Edge", "p1b: Raichu"],
        ["", "-damage", "p1b: Raichu", "97/100"],
    ]
    ii._battle._active_pokemon = {
        "p1a": ii._battle.team["p1: Furret"],
        "p1b": ii._battle.team["p1: Raichu"],
    }
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "safetygoggles"

    # Sandstorm
    ii = generate_item_inference()
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
    ii.check_items(Observation(events=events))
    assert get_pokemon("p2: Furret", ii._battle).item == "safetygoggles"
