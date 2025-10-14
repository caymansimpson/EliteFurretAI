# -*- coding: utf-8 -*-
from logging import Logger

from poke_env.battle import DoubleBattle, Move, Pokemon, PokemonType
from poke_env.data import GenData
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.model_utils import BattleData, BattleIterator, Embedder


def move_generator(gen=9):
    for move in GenData.from_gen(gen).moves:
        yield Move(move, gen=gen)


def mon_generator(gen=9):
    for species in GenData.from_gen(gen).pokedex:
        if species != "missingno":
            yield Pokemon(gen=gen, species=species)


def test_embed_move():
    embedder = Embedder()

    # Test that we're featurizing None correctly
    none_move = embedder.generate_move_features(None)
    assert all(map(lambda x: none_move[x] == -1, none_move))

    # Test that every move has the same length
    len_none_move = len(embedder.feature_dict_to_vector(none_move))
    for move in move_generator():
        featurized_move = tuple(
            embedder.feature_dict_to_vector(embedder.generate_move_features(move))
        )
        assert len(featurized_move) == len_none_move

    # Test each implemented feature is working properly
    emb = embedder.generate_move_features(Move("icywind", gen=9))
    assert emb["accuracy"] == 0.95
    assert emb["base_power"] == 55
    assert emb["current_pp"] == 5

    emb = embedder.generate_move_features(Move("seismictoss", gen=9))
    assert emb["damage"] == 50

    emb = embedder.generate_move_features(Move("gigadrain", gen=9))
    assert emb["drain"] == 0.5
    assert emb["force_switch"] == 0
    assert emb["heal"] == 0
    assert emb["is_protect_move"] == 0
    assert emb["is_side_protect_move"] == 0
    assert emb["min_hits"] == 1
    assert emb["max_hits"] == 1
    assert emb["priority"] == 0
    assert emb["recoil"] == 0

    emb = embedder.generate_move_features(Move("foulplay", gen=9))
    assert emb["self_switch"] == 0
    assert emb["use_target_offensive"] == 1
    assert emb["OFF_CAT:STATUS"] == 0
    assert emb["OFF_CAT:PHYSICAL"] == 1
    assert emb["OFF_CAT:SPECIAL"] == 0

    emb = embedder.generate_move_features(Move("tailwind", gen=9))
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

    emb = embedder.generate_move_features(Move("yawn", gen=9))
    assert emb["EFFECT:SUBSTITUTE"] == 0
    assert emb["EFFECT:PROTECT"] == 0
    assert emb["EFFECT:HELPING_HAND"] == 0
    assert emb["EFFECT:TAUNT"] == 0
    assert emb["EFFECT:ENCORE"] == 0
    assert emb["EFFECT:YAWN"] == 1
    assert emb["EFFECT:FOLLOW_ME"] == 0
    assert emb["EFFECT:FLINCH"] == 0

    emb = embedder.generate_move_features(Move("spore", gen=9))
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

    emb = embedder.generate_move_features(Move("ominouswind", gen=9))
    assert emb["SELFBOOST:atk"] == 1
    assert emb["SELFBOOST:def"] == 1
    assert emb["SELFBOOST:spa"] == 1
    assert emb["SELFBOOST:spd"] == 1
    assert emb["SELFBOOST:spe"] == 1
    assert emb["chance"] == 10


def test_generate_pokemon_features():
    embedder = Embedder()
    dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
    dummy_battle._format = embedder.format
    dummy_battle.player_role = "p1"

    # Test that we're creating correct features for None
    none_mon = embedder.generate_pokemon_features(None, dummy_battle)
    assert isinstance(none_mon, dict)
    for feature in none_mon:
        if not isinstance(none_mon[feature], dict):
            assert none_mon[feature] == -1

    # Test that every mon has the same length
    none_mon_len = len(embedder.feature_dict_to_vector(none_mon))
    for mon in mon_generator():
        featurized_mon = embedder.feature_dict_to_vector(
            embedder.generate_pokemon_features(mon, dummy_battle)
        )
        assert len(featurized_mon) == none_mon_len

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
    emb = embedder.generate_pokemon_features(furret, dummy_battle)

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
    assert emb["ITEM:lifeorb"] == 0
    assert emb["current_hp_fraction"] == 0.13125
    assert emb["level"] == 50
    assert emb["weight"] == 32.5
    assert emb["is_terastallized"] == 0
    assert emb["STAT:hp"] == 160
    assert emb["STAT:atk"] == 128
    assert emb["STAT:def"] == 84
    assert emb["STAT:spa"] == 66
    assert emb["STAT:spd"] == 67
    assert emb["STAT:spe"] == 156
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


def test_generate_opponent_pokemon_features(vgc_battle_p1_logs):
    embedder = Embedder()
    dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
    dummy_battle._format = embedder.format
    dummy_battle.player_role = "p1"

    # Test that we featurize none correctly
    none_mon = embedder.generate_opponent_pokemon_features(None, dummy_battle)
    for feature in none_mon:
        if not isinstance(none_mon[feature], dict):
            assert none_mon[feature] == -1

    # Test that every mon has the same length
    none_mon_len = len(embedder.feature_dict_to_vector(none_mon))
    for mon in mon_generator():
        featurized_mon = embedder.feature_dict_to_vector(
            embedder.generate_opponent_pokemon_features(mon, dummy_battle)
        )
        assert len(featurized_mon) == none_mon_len

    # Generate battle
    p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
    for turn in vgc_battle_p1_logs:
        for log in turn:
            if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                p1_battle.parse_message(log)
            elif len(log) > 1 and log[1] == "win":
                p1_battle.won_by(log[2])

    smeargle = p1_battle.opponent_team["p2: Smeargle"]
    emb = embedder.generate_opponent_pokemon_features(smeargle, dummy_battle)

    assert any(map(lambda x: x.startswith("MOVE:0:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:1:"), emb))
    assert any(map(lambda x: x.startswith("MOVE:2:"), emb))

    assert emb["MOVE:1:min_hits"] == 3
    assert emb["MOVE:1:max_hits"] == 3
    assert emb["MOVE:1:TYPE:ICE"] == 1
    assert emb["MOVE:2:max_hits"] == -1  # Dont know it
    assert emb["MOVE:1:current_pp"] == 5
    assert emb["ABILITY:thermalexchange"] == 0
    assert emb["ABILITY:moody"] == 1

    # We don't know Smeargle's item
    for key in emb:
        if key.startswith("ITEM:"):
            assert emb[key] == 0

    # Fainted
    assert emb["current_hp_fraction"] == 0
    assert emb["level"] == 50
    assert emb["weight"] == 58
    assert emb["is_terastallized"] == 0

    # no battle_inference
    assert emb["STAT_MIN:hp"] == 115
    assert emb["STAT_MAX:hp"] == 162
    assert emb["STAT_MIN:atk"] == 22
    assert emb["STAT_MAX:atk"] == 79
    assert emb["STAT_MIN:def"] == 36
    assert emb["STAT_MAX:def"] == 95
    assert emb["STAT_MIN:spa"] == 22
    assert emb["STAT_MAX:spa"] == 79
    assert emb["STAT_MIN:spd"] == 45
    assert emb["STAT_MAX:spd"] == 106
    assert emb["STAT_MIN:spe"] == 72
    assert emb["STAT_MAX:spe"] == 139
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
    emb = embedder.generate_opponent_pokemon_features(ttar, dummy_battle)

    assert emb["MOVE:0:base_power"] == 60
    assert emb["MOVE:0:TYPE:DRAGON"] == 1
    assert emb["MOVE:0:priority"] == -6
    assert emb["MOVE:2:accuracy"] == -1
    assert emb["is_terastallized"] == 1

    assert emb["TERA_TYPE:ROCK"] == 0
    assert emb["TERA_TYPE:PSYCHIC"] == 1
    assert emb["TYPE:ROCK"] == 0
    assert emb["TYPE:PSYCHIC"] == 1


def test_embed_turn(vgc_json_anon):

    embedder = Embedder(feature_set="raw")

    # Generate battle
    bd = BattleData.from_showdown_json(vgc_json_anon)
    iterator = BattleIterator(bd, perspective="p1")

    iterator.next_turn()
    iterator.simulate_request()
    iterator.battle._force_switch = [False, True]  # type: ignore

    emb = embedder.embed(iterator.battle)  # type: ignore

    assert emb["MON:0:sent"] == 1
    assert emb["MON:0:active"] == 0
    assert emb["MON:0:revealed"] == 0
    assert emb["MON:0:is_available_to_switch"] == 1
    assert emb["MON:0:force_switch"] == -1
    assert emb["MON:0:trapped"] == -1

    assert emb["MON:4:sent"] == 1
    assert emb["MON:4:active"] == 1
    assert emb["MON:4:revealed"] == 1
    assert emb["MON:4:is_available_to_switch"] == 0
    assert emb["MON:4:force_switch"] == 1
    assert emb["MON:4:trapped"] == 0

    assert emb["MON:5:sent"] == 0
    assert emb["MON:5:active"] == 0
    assert emb["MON:5:revealed"] == -1

    assert emb["OPP_MON:0:sent"] == 1
    assert emb["OPP_MON:0:active"] == 1

    assert emb["FORMAT:gen9vgc2023regulationc"] == 1
    assert emb["teampreview"] == 0
    assert emb["turn"] == 1

    emb = embedder.generate_feature_engineered_features(iterator.battle)  # type: ignore

    assert emb["NUM_FAINTED"] == 0
    assert emb["PERC_HP_LEFT"] == 1
    assert emb["NUM_STATUSED"] == 0
    assert emb["NUM_MOVES_REVEALED"] == 0
    assert emb["NUM_MONS_REVEALED"] == 2

    assert emb["OPP_NUM_FAINTED"] == 0
    assert emb["OPP_PERC_HP_LEFT"] == 1
    assert emb["OPP_NUM_STATUSED"] == 0
    assert emb["NUM_OPP_MONS_REVEALED"] == 2

    assert emb["TYPE_MATCHUP:OPP_MON:0:MON:0"] == 1
    assert emb["TYPE_MATCHUP:OPP_MON:1:MON:0"] == 1
    assert emb["TYPE_MATCHUP:OPP_MON:2:MON:0"] == 2

    assert emb["EST_DAMAGE_MIN:MON:2:OPP_MON:2:MOVE:0"] == 22
    assert emb["KO:MON:2:OPP_MON:2:MOVE:0"] == 0

    # Dont know move
    assert emb["KO:OPP_MON:3:MON:3:MOVE:2"] == -1


def test_simplify_features(vgc_battle_p1_logs):
    embedder = Embedder()
    simple_embedder = Embedder(feature_set="simple")

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

                emb = embedder.embed(p1_battle)
                simple_emb = simple_embedder.embed(p1_battle)

                assert len(simple_emb) < len(emb)


def test_embed_teampreview(vgc_json_anon):

    embedder = Embedder(feature_set="raw")

    # Generate battle
    bd = BattleData.from_showdown_json(vgc_json_anon)
    iterator = BattleIterator(bd, perspective="p1")

    iterator.next_input()
    iterator.simulate_request()

    emb = embedder.embed(iterator.battle)  # type: ignore

    for key in emb:
        if "active" in key or "sent" in key or "revealed" in key:
            assert emb[key] <= 0
        elif "TRAPPED" in key:
            assert emb[key] == 0
        elif "FORCE_SWITCH" in key:
            assert emb[key] == 0
        elif key.startswith("FIELD"):
            assert emb[key] == 0
        elif key.startswith("SIDE_CONDITION"):
            assert emb[key] == 0
        elif key.startswith("OPP_SIDE_CONDITION"):
            assert emb[key] == 0
        elif key.startswith("WEATHER"):
            assert emb[key] == 0

    assert emb["teampreview"] == 1
    assert emb["turn"] == 0


def test_omniscience(vgc_json_anon):

    e1 = Embedder(feature_set="full", omniscient=True)
    e2 = Embedder(feature_set="full", omniscient=False)

    assert e1.pokemon_embedding_size == e2.pokemon_embedding_size
    assert e1.move_embedding_size == e2.move_embedding_size
    assert e1.battle_embedding_size == e2.battle_embedding_size
    assert e1.opponent_pokemon_embedding_size == e2.opponent_pokemon_embedding_size
    assert e1.embedding_size == e2.embedding_size
    assert e1.group_embedding_sizes == e2.group_embedding_sizes
