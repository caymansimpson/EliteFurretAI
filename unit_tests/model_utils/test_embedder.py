# -*- coding: utf-8 -*-
from unittest.mock import MagicMock
from logging import Logger

from poke_env.data import GenData
from poke_env.environment import DoubleBattle, Move, Pokemon, PokemonType, Status
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.model_utils.embedder import Embedder


def move_generator(gen=9):
    for move in GenData.from_gen(gen).moves:
        yield Move(move, gen=gen)


def mon_generator(gen=9):
    for species in GenData.from_gen(gen).pokedex:
        if species != "missingno":
            yield Pokemon(gen=gen, species=species)


def test_featurize_move():
    embedder = Embedder()

    # Test that we're featurizing None correctly
    none_move = embedder.featurize_move(None)
    assert all(map(lambda x: none_move[x] == -1, none_move))

    # Test that every move has the same length
    len_none_move = len(embedder.feature_dict_to_vector(none_move))
    for move in move_generator():
        featurized_move = tuple(
            embedder.feature_dict_to_vector(embedder.featurize_move(move))
        )
        assert len(featurized_move) == len_none_move

    # Test each implemented feature is working properly
    emb = embedder.featurize_move(Move("icywind", gen=9))
    assert emb["accuracy"] == 0.95
    assert emb["base_power"] == 55 / 100.0
    assert emb["current_pp"] == 1

    emb = embedder.featurize_move(Move("seismictoss", gen=9))
    assert emb["damage"] == 50

    emb = embedder.featurize_move(Move("gigadrain", gen=9))
    assert emb["drain"] == 0.5
    assert emb["force_switch"] == 0
    assert emb["heal"] == 0
    assert emb["is_protect_move"] == 0
    assert emb["is_side_protect_move"] == 0
    assert emb["min_hits"] == 1
    assert emb["max_hits"] == 1
    assert emb["priority"] == 0
    assert emb["recoil"] == 0

    emb = embedder.featurize_move(Move("foulplay", gen=9))
    assert emb["self_switch"] == 0
    assert emb["use_target_offensive"] == 1
    assert emb["OFF_CAT:STATUS"] == 0
    assert emb["OFF_CAT:PHYSICAL"] == 1
    assert emb["OFF_CAT:SPECIAL"] == 0

    emb = embedder.featurize_move(Move("tailwind", gen=9))
    assert emb["TYPE:GHOST"] == 0
    assert emb["TYPE:WATER"] == 0
    assert emb["TYPE:PSYCHIC"] == 0
    assert emb["TYPE:NORMAL"] == 0
    assert emb["TYPE:DARK"] == 0
    assert emb["TYPE:FAIRY"] == 0
    assert emb["TYPE:STEEL"] == 0
    assert emb["TYPE:FIRE"] == 0
    assert emb["TYPE:FLYING"] == 1
    assert emb["TYPE:GROUND"] == 0
    assert emb["TYPE:ICE"] == 0
    assert emb["TYPE:BUG"] == 0
    assert emb["TYPE:ELECTRIC"] == 0
    assert emb["TYPE:GRASS"] == 0
    assert emb["TYPE:POISON"] == 0
    assert emb["TYPE:FIGHTING"] == 0
    assert emb["TYPE:DRAGON"] == 0
    assert emb["TYPE:ROCK"] == 0
    assert emb["SC:LIGHT_SCREEN"] == 0
    assert emb["SC:QUICK_GUARD"] == 0
    assert emb["SC:AURORA_VEIL"] == 0
    assert emb["SC:SAFEGUARD"] == 0
    assert emb["SC:STEALTH_ROCK"] == 0
    assert emb["SC:STICKY_WEB"] == 0
    assert emb["SC:REFLECT"] == 0
    assert emb["SC:TOXIC_SPIKES"] == 0
    assert emb["SC:WIDE_GUARD"] == 0
    assert emb["SC:TAILWIND"] == 1
    assert emb["SC:MIST"] == 0
    assert emb["SC:SPIKES"] == 0
    assert emb["TARGET:ALL"] == 0
    assert emb["TARGET:SELF"] == 0
    assert emb["TARGET:ALL_ADJACENT_FOES"] == 0
    assert emb["TARGET:ANY"] == 0
    assert emb["TARGET:NORMAL"] == 0
    assert emb["TARGET:ADJACENT_ALLY"] == 0
    assert emb["TARGET:ADJACENT_FOE"] == 0
    assert emb["TARGET:ALL_ADJACENT"] == 0
    assert emb["TARGET:ALLY_SIDE"] == 1

    emb = embedder.featurize_move(Move("burningbulwark", gen=9))
    assert emb["EFFECT:SPIKY_SHIELD"] == 0
    assert emb["EFFECT:BANEFUL_BUNKER"] == 0
    assert emb["EFFECT:SUBSTITUTE"] == 0
    assert emb["EFFECT:PROTECT"] == 0
    assert emb["EFFECT:DISABLE"] == 0
    assert emb["EFFECT:SILK_TRAP"] == 0
    assert emb["EFFECT:POWDER"] == 0
    assert emb["EFFECT:SALT_CURE"] == 0
    assert emb["EFFECT:HELPING_HAND"] == 0
    assert emb["EFFECT:IMPRISON"] == 0
    assert emb["EFFECT:SPOTLIGHT"] == 0
    assert emb["EFFECT:DRAGON_CHEER"] == 0
    assert emb["EFFECT:CONFUSION"] == 0
    assert emb["EFFECT:BURNING_BULWARK"] == 1
    assert emb["EFFECT:TAUNT"] == 0
    assert emb["EFFECT:ENCORE"] == 0
    assert emb["EFFECT:GLAIVE_RUSH"] == 0
    assert emb["EFFECT:YAWN"] == 0
    assert emb["EFFECT:FOLLOW_ME"] == 0
    assert emb["EFFECT:HEAL_BLOCK"] == 0
    assert emb["EFFECT:ROOST"] == 0
    assert emb["EFFECT:FLINCH"] == 0
    assert emb["EFFECT:LEECH_SEED"] == 0
    assert emb["EFFECT:ENDURE"] == 0
    assert emb["EFFECT:RAGE_POWDER"] == 0

    emb = embedder.featurize_move(Move("spore", gen=9))
    assert emb["STATUS:FRZ"] == 0
    assert emb["STATUS:BRN"] == 0
    assert emb["STATUS:PAR"] == 0
    assert emb["STATUS:SLP"] == 1
    assert emb["STATUS:PSN"] == 0
    assert emb["STATUS:TOX"] == 0
    assert emb["BOOST:atk"] == 0
    assert emb["BOOST:def"] == 0
    assert emb["BOOST:spa"] == 0
    assert emb["BOOST:spd"] == 0
    assert emb["BOOST:spe"] == 0

    emb = embedder.featurize_move(Move("ominouswind", gen=9))
    assert emb["SELFBOOST:atk"] == 1
    assert emb["SELFBOOST:def"] == 1
    assert emb["SELFBOOST:spa"] == 1
    assert emb["SELFBOOST:spd"] == 1
    assert emb["SELFBOOST:spe"] == 1
    assert emb["chance"] == 0.1


# TODO: just create a furret based on the tests below, from a teambuilder (reverse engineer)
def test_featurize_pokemon():
    embedder = Embedder()

    # Test that we're creating correct features for None
    none_mon = embedder.featurize_pokemon(None)
    for feature in none_mon:
        if not isinstance(none_mon[feature], dict):
            assert none_mon[feature] == -1

    # Test that every mon has the same length
    none_mon_len = len(embedder.feature_dict_to_vector(none_mon))
    for mon in mon_generator():
        assert (
            len(embedder.feature_dict_to_vector(embedder.featurize_pokemon(mon)))
            == none_mon_len
        )

    tb_furret = ConstantTeambuilder(
        """Furret @ Leftovers
        Ability: Run Away
        Level: 50
        Tera Type: Normal
        EVs: 252 Atk / 4 SpA / 252 Spe
        Naive Nature
        - Agility
        - Baton Pass
        - Blizzard
        - Body Slam"""
    )

    furret = Pokemon(gen=9, teambuilder=tb_furret.team[0])
    furret._terastallized_type = PokemonType.DARK
    furret.set_hp_status("21/160 slp")
    furret.set_boost("spe", 2)
    emb = embedder.featurize_pokemon(furret)

    # We have move embeddings
    assert any(map(lambda x: x.startswith("MOVE:0:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:1:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:2:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:3:"), emb))

    # We don't record runaway, which is Furret's ability here
    for key in emb:
        if key.startswith("ABILITY:"):
            assert emb[key] == 0

    assert emb["ITEM:lightclay"] == 0
    assert emb["ITEM:leftovers"] == 1
    assert emb["ITEM:laggingtail"] == 0
    assert emb["ITEM:lifeorb"] == 0
    assert emb["ITEM:heavydutyboots"] == 0
    assert emb["current_hp_fraction"] == 0.13125
    assert emb["level"] == 50 / 100.0
    assert emb["weight"] == 32.5 / 100.0
    assert emb["is_terastallized"] == 0
    assert emb["STAT:hp"] == 160 / 100.0
    assert emb["STAT:atk"] == 128 / 100.0
    assert emb["STAT:def"] == 84 / 100.0
    assert emb["STAT:spa"] == 66 / 100.0
    assert emb["STAT:spd"] == 67 / 100.0
    assert emb["STAT:spe"] == 156 / 100.0
    assert emb["BOOST:accuracy"] == 0
    assert emb["BOOST:atk"] == 0
    assert emb["BOOST:spa"] == 0
    assert emb["BOOST:spd"] == 0
    assert emb["BOOST:spe"] == 2
    assert emb["STATUS: BRN"] == 0
    assert emb["STATUS: TOX"] == 0
    assert emb["STATUS: SLP"] == 1
    assert emb["STATUS: FNT"] == 0
    assert emb["TYPE:ROCK"] == 0
    assert emb["TYPE:FIRE"] == 0
    assert emb["TYPE:ICE"] == 0
    assert emb["TYPE:BUG"] == 0
    assert emb["TYPE:NORMAL"] == 1
    assert emb["TYPE:FLYING"] == 0
    assert emb["TYPE:GRASS"] == 0
    assert emb["TERA_TYPE:ROCK"] == 0
    assert emb["TERA_TYPE:FIRE"] == 0
    assert emb["TERA_TYPE:STELLAR"] == 0
    assert emb["TERA_TYPE:DARK"] == 1
    assert emb["TERA_TYPE:GROUND"] == 0
    assert emb["TERA_TYPE:FIGHTING"] == 0
    assert emb["TERA_TYPE:FLYING"] == 0
    assert emb["TERA_TYPE:GRASS"] == 0


def test_featurize_oponent_pokemon(vgc_battle_p1_logs):
    embedder = Embedder()

    # Test that we featurize none correctly
    none_mon = embedder.featurize_opponent_pokemon(None)
    for feature in none_mon:
        if not isinstance(none_mon[feature], dict):
            assert none_mon[feature] == -1

    # Test that every mon has the same length
    none_mon_len = len(embedder.feature_dict_to_vector(none_mon))
    for mon in mon_generator():
        assert (
            len(embedder.feature_dict_to_vector(embedder.featurize_opponent_pokemon(mon)))
            == none_mon_len
        )

    # Generate battle
    p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

    smeargle = p1_battle.opponent_team["p2: Smeargle"]
    emb = embedder.featurize_opponent_pokemon(smeargle)

    assert any(map(lambda x: x.startswith("MOVE:0:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:1:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:2:"), emb))

    assert emb["MOVE:1:min_hits"] == 3
    assert emb["MOVE:1:max_hits"] == 3
    assert emb["MOVE:1:TYPE:ICE"] == 1
    assert emb["MOVE:2:max_hits"] == -1  # Dont know it
    assert emb["MOVE:1:current_pp"] == 1
    assert emb["ABILITY:thermalexchange"] == 0
    assert emb["ABILITY:immunity"] == 0
    assert emb["ABILITY:punkrock"] == 0
    assert emb["ABILITY:moody"] == 1
    assert emb["ABILITY:stall"] == 0
    assert emb["ABILITY:pixilate"] == 0
    assert emb["ABILITY:powerconstruct"] == 0

    # We don't know Smeargle's item
    for key in emb:
        if key.startswith("ITEM:"):
            assert emb[key] == 0

    # Fainted
    assert emb["current_hp_fraction"] == 0
    assert emb["level"] == 50 / 100.0
    assert emb["weight"] == 58 / 100.0
    assert emb["is_terastallized"] == 0

    # no battle_inference
    assert emb["STAT_MIN:hp"] == 115 / 100.0
    assert emb["STAT_MAX:hp"] == 162 / 100.0
    assert emb["STAT_MIN:atk"] == 22 / 100.0
    assert emb["STAT_MAX:atk"] == 79 / 100.0
    assert emb["STAT_MIN:def"] == 36 / 100.0
    assert emb["STAT_MAX:def"] == 95 / 100.0
    assert emb["STAT_MIN:spa"] == 22 / 100.0
    assert emb["STAT_MAX:spa"] == 79 / 100.0
    assert emb["STAT_MIN:spd"] == 45 / 100.0
    assert emb["STAT_MAX:spd"] == 106 / 100.0
    assert emb["STAT_MIN:spe"] == 72 / 100.0
    assert emb["STAT_MAX:spe"] == 139 / 100.0
    assert emb["BOOST:accuracy"] == 0
    assert emb["BOOST:atk"] == 0
    assert emb["BOOST:def"] == 0
    assert emb["BOOST:spa"] == 0
    assert emb["BOOST:spd"] == 0
    assert emb["BOOST:spe"] == 0
    assert emb["STATUS: FNT"] == 1
    assert emb["STATUS: PSN"] == 0
    assert emb["STATUS: BRN"] == 0
    assert emb["STATUS: PAR"] == 0
    assert emb["STATUS: FRZ"] == 0
    assert emb["STATUS: TOX"] == 0
    assert emb["STATUS: SLP"] == 0
    assert emb["TYPE:NORMAL"] == 1
    assert emb["TYPE:ROCK"] == 0
    assert emb["TYPE:DRAGON"] == 0
    assert emb["TYPE:FLYING"] == 0
    assert emb["TYPE:GHOST"] == 0
    assert emb["TERA_TYPE:NORMAL"] == -1
    assert emb["TERA_TYPE:DRAGON"] == -1
    assert emb["TERA_TYPE:ROCK"] == -1  # Don't know this is the ground truth

    ttar = p1_battle.opponent_team["p2: Tyranitar"]
    emb = embedder.featurize_opponent_pokemon(ttar)

    assert emb["MOVE:0:base_power"] == 0.6
    assert emb["MOVE:0:TYPE:DRAGON"] == 1
    assert emb["MOVE:0:priority"] == -6
    assert emb["MOVE:2:accuracy"] == -1
    assert emb["is_terastallized"] == 1

    assert emb["TERA_TYPE:ROCK"] == 0
    assert emb["TERA_TYPE:PSYCHIC"] == 1
    assert emb["TYPE:ROCK"] == 0
    assert emb["TYPE:PSYCHIC"] == 1


def test_featurize_turn(vgc_battle_p1_logs):

    embedder = Embedder()
    emb_length = None

    # Generate battle
    p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

            # If we just went through a turn
            if len(log) > 1 and log[1] == "-turn" and log[2] == "1":
                p1_battle.parse_message(log)

                emb = embedder.featurize_double_battle(vgc_battle_p1)
                assert emb["MON:0:sent"] == 1
                assert emb["MON:5:TYPE:ROCK"] == -1
                assert emb["MON:5:sent"] == -1
                assert emb["MON:0:active"] == 1
                assert emb["MON:3:active"] == 0
                assert emb["MON:5:active"] == -1

                assert emb["OPP_MON:0:sent"] == 1
                assert emb["OPP_MON:0:active"] == 1

                assert emb["OPP_MON:0:sent"] == 0
                assert emb["OPP_MON:0:active"] == 0

                assert emb["FORCE_SWITCH:0"] == 0
                assert emb["FORCE_SWITCH:1"] == 0

                assert emb["teampreview"] == 0
                assert emb["turn"] == 1
                assert emb["bias"] == 1

                if emb_length is None:
                    emb_length = len(emb.keys())
                else:
                    assert len(emb.keys()) == emb_length


def test_simplify_features(vgc_battle_p1_logs):
    embedder = Embedder()
    emb_length = None

    # Generate battle
    p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

            # If we just went through a turn
            if len(log) > 1 and log[1] == "-turn" and log[2] == "1":
                p1_battle.parse_message(log)

                emb = embedder.featurize_double_battle(vgc_battle_p1)
                simple_emb = embedder.simplify_features(emb)

                assert len(simple_emb) < len(emb)

                if emb_length is None:
                    emb_length = len(emb.keys())
                else:
                    assert len(emb.keys()) == emb_length


def test_featurize_teampreview(example_vgc_teampreview_request):
    logger = MagicMock()
    battle = DoubleBattle("tag", "elitefurretai", logger, gen=9)

    # Initiate battle
    messages = [
        ["", "init", "battle"],
        ["", "title", "elitefurretai vs. CustomPlayer 1"],
        ["", "j", "☆elitefurretai"],
        ["", "j", "☆CustomPlayer 1"],
        ["", "gametype", "doubles"],
        ["", "player", "p1", "elitefurretai", "266", ""],
        ["", "player", "p2", "CustomPlayer 1", "265", ""],
        ["", "teamsize", "p1", "5"],
        ["", "teamsize", "p2", "5"],
        ["", "gen", "9"],
        ["", "tier", "[Gen 9] VGC 2024 Reg G"],
        ["", "rule", "Species Clause: Limit one of each Pokémon"],
        ["", "rule", "Item Clause: Limit 1 of each item"],
        ["", "clearpoke"],
        ["", "poke", "p1", "Delibird, L50, F", ""],
        ["", "poke", "p1", "Raichu, L50, F", ""],
        ["", "poke", "p1", "Tyranitar, L50, F", ""],
        ["", "poke", "p1", "Smeargle, L50, M", ""],
        ["", "poke", "p1", "Furret, L50, F", ""],
        ["", "poke", "p2", "Delibird, L50, F", ""],
        ["", "poke", "p2", "Raichu, L50, F", ""],
        ["", "poke", "p2", "Tyranitar, L50, F", ""],
        ["", "poke", "p2", "Smeargle, L50, F", ""],
        ["", "poke", "p2", "Furret, L50, M", ""],
        ["", "teampreview", "4"],
    ]
    for msg in messages:
        battle.parse_message(msg)

    # Parse Teampreview request
    battle.parse_request(example_vgc_teampreview_request)

    # Need to add teampreview since Teampreview team is usually registered in Player
    battle.teampreview_team = set(battle.team.values())

    embedder = Embedder()
    emb = embedder.featurize_double_battle(battle)

    for key in emb:
        if "active" in key or "sent" in key or "revealed" in key:
            if emb[key] != 0:
                print(key)
            assert emb[key] == 0
        elif "TRAPPED" in key:
            assert emb[key] == 0
        elif "FORCE_SWITCH" in key:
            assert emb[key] == 0
        elif "FIELD" in key:
            assert emb[key] == 0
        elif "SIDE_CONDITION" in key:
            assert emb[key] == 0
        elif "WEATHER" in key:
            assert emb[key] == 0

    assert emb["teampreview"] == 1
    assert emb["turn"] == 0
    assert emb["bias"] == 1
