# -*- coding: utf-8 -*-
from logging import Logger

import pytest
from poke_env.battle import DoubleBattle, Move, Pokemon, PokemonType
from poke_env.data import GenData
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.etl import BattleData, BattleIterator, Embedder
from elitefurretai.etl.embedder import (
    ABILITY_TO_ID,
    ITEM_TO_ID,
    MOVE_TO_ID,
    NUM_MOVES,
    NUM_SPECIES,
    SPECIES_TO_ID,
)


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
    len_none_move = len(none_move)
    for move in move_generator():
        featurized_move = tuple(embedder.generate_move_features(move).values())
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
    # Move-level SC OHE was pruned; SC is now only at battle-state level
    assert "SC:TAILWIND" not in emb
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
    # Yawn doesn't directly cause a status - it just causes drowsiness
    assert emb["STATUS:FRZ"] == 0
    assert emb["STATUS:BRN"] == 0

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
    none_mon_len = len(none_mon)
    for mon in mon_generator():
        featurized_mon = embedder.generate_pokemon_features(mon, dummy_battle)
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

    # Furret's ability is "runaway" and should be present in Gen 9 mapping
    assert emb["ability_id"] == ABILITY_TO_ID["runaway"]

    assert emb["item_id"] == ITEM_TO_ID["leftovers"]
    assert emb["current_hp_fraction"] == 0.13125
    assert emb["weight"] == 32.5
    assert emb["is_terastallized"] == 0
    assert emb["STAT:hp"] == 160
    assert emb["STAT:atk"] == 128
    assert emb["STAT:def"] == 84
    assert emb["STAT:spa"] == 66
    assert emb["STAT:spd"] == 67
    assert emb["STAT:spe"] == 156
    # One-hot boost encoding: Furret has +2 spe, everything else at 0
    assert emb["BOOST:accuracy:0"] == 1
    assert emb["BOOST:accuracy:1"] == 0
    assert emb["BOOST:atk:0"] == 1
    assert emb["BOOST:spa:0"] == 1
    assert emb["BOOST:spd:0"] == 1
    assert emb["BOOST:spe:0"] == 0
    assert emb["BOOST:spe:2"] == 1
    assert emb["BOOST:spe:1"] == 0
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
    none_mon_len = len(none_mon)
    for mon in mon_generator():
        featurized_mon = embedder.generate_opponent_pokemon_features(mon, dummy_battle)
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
    assert emb["ability_id"] == ABILITY_TO_ID["moody"]

    # We don't know Smeargle's item
    assert emb["item_id"] == 0  # Unknown item

    # Fainted
    assert emb["current_hp_fraction"] == 0
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
    # One-hot boost encoding: Smeargle has all boosts at 0
    assert emb["BOOST:accuracy:0"] == 1
    assert emb["BOOST:atk:0"] == 1
    assert emb["BOOST:def:0"] == 1
    assert emb["BOOST:spa:0"] == 1
    assert emb["BOOST:spd:0"] == 1
    assert emb["BOOST:spe:0"] == 1
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


def test_feature_dict_to_vector_requires_full_embed():
    embedder = Embedder()
    dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
    dummy_battle._format = embedder.format
    dummy_battle.player_role = "p1"

    partial = embedder.generate_move_features(None)
    with pytest.raises(ValueError):
        embedder.feature_dict_to_vector(partial)


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

    # FORMAT feature was removed from embedder
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

    # TYPE_MATCHUP features were pruned (redundant with EST_DAMAGE)
    assert "TYPE_MATCHUP:OPP_MON:0:MON:0" not in emb

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


def test_species_id_feature():
    """Test that species_id is correctly added to pokemon features."""
    embedder = Embedder()
    dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
    dummy_battle._format = embedder.format
    dummy_battle.player_role = "p1"

    # furret should have a valid species_id
    assert "furret" in SPECIES_TO_ID
    furret = Pokemon(gen=9, species="furret")
    emb = embedder.generate_pokemon_features(furret, dummy_battle)
    assert "species_id" in emb
    assert emb["species_id"] == SPECIES_TO_ID["furret"]

    # Unknown/null pokemon should get -1 sentinel
    null_emb = embedder.generate_pokemon_features(None, dummy_battle)
    assert null_emb["species_id"] == -1


def test_move_id_feature():
    """Test that move_id is correctly added to move features."""
    embedder = Embedder()

    # Known moves should get valid IDs
    assert "icywind" in MOVE_TO_ID
    emb = embedder.generate_move_features(Move("icywind", gen=9))
    assert "move_id" in emb
    assert emb["move_id"] == MOVE_TO_ID["icywind"]

    # Null move should get -1 sentinel
    null_emb = embedder.generate_move_features(None)
    assert null_emb["move_id"] == -1


def test_species_to_id_mapping():
    """Test that SPECIES_TO_ID covers Gen 9 Pokemon."""
    assert len(SPECIES_TO_ID) > 100  # Sanity check
    assert NUM_SPECIES == len(SPECIES_TO_ID) + 1  # +1 for unknown at index 0

    # Spot-check common species
    assert "furret" in SPECIES_TO_ID
    assert "pikachu" in SPECIES_TO_ID
    assert "incineroar" in SPECIES_TO_ID

    # All IDs should be positive (0 reserved for unknown)
    for species, sid in SPECIES_TO_ID.items():
        assert sid > 0, f"Species {species} has non-positive ID {sid}"


def test_move_to_id_mapping():
    """Test that MOVE_TO_ID covers Gen 9 moves."""
    assert len(MOVE_TO_ID) > 100  # Sanity check
    assert NUM_MOVES == len(MOVE_TO_ID) + 1  # +1 for unknown at index 0

    # Spot-check common moves
    assert "protect" in MOVE_TO_ID
    assert "earthquake" in MOVE_TO_ID
    assert "icywind" in MOVE_TO_ID

    # All IDs should be positive
    for move, mid in MOVE_TO_ID.items():
        assert mid > 0, f"Move {move} has non-positive ID {mid}"


def test_pruned_features_absent():
    """Verify that pruned features are no longer in embeddings."""
    embedder = Embedder()
    dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
    dummy_battle._format = embedder.format
    dummy_battle.player_role = "p1"

    # Move-level SC, FIELD, WEATHER, EFFECT OHE should be pruned
    emb = embedder.generate_move_features(Move("tailwind", gen=9))
    for key in emb:
        assert not key.startswith("SC:"), f"SC OHE should be pruned: {key}"
        assert not key.startswith("FIELD:"), f"FIELD OHE should be pruned: {key}"
        assert not key.startswith("WEATHER:"), f"WEATHER OHE should be pruned: {key}"
        assert not key.startswith("EFFECT:"), f"EFFECT OHE should be pruned: {key}"

    # Level should be pruned from pokemon features
    furret = Pokemon(gen=9, species="furret")
    pemb = embedder.generate_pokemon_features(furret, dummy_battle)
    assert "level" not in pemb


def test_feature_consistency_across_feature_sets():
    """Verify that species_id and move_id exist in all feature sets."""
    for feature_set in ["raw", "full"]:
        embedder = Embedder(feature_set=feature_set)
        dummy_battle = DoubleBattle("tag", "elitefurretai", None, gen=9)  # type: ignore
        dummy_battle._format = embedder.format
        dummy_battle.player_role = "p1"

        furret = Pokemon(gen=9, species="furret")
        pemb = embedder.generate_pokemon_features(furret, dummy_battle)
        assert "species_id" in pemb, f"species_id missing in {feature_set}"

        memb = embedder.generate_move_features(Move("protect", gen=9))
        assert "move_id" in memb, f"move_id missing in {feature_set}"


def test_embedding_size_consistency():
    """Verify embedding sizes are self-consistent after feature changes."""
    for feature_set in ["raw", "full"]:
        embedder = Embedder(feature_set=feature_set)
        # Total embedding should equal sum of group sizes
        assert embedder.embedding_size == sum(embedder.group_embedding_sizes)


def test_dry_run_battle_embedding(vgc_json_anon):
    """Dry-run through a full battle replay, embedding every decision point.

    Verifies that species_id, move_id, and other features are consistent
    throughout the whole battle, and that no features are missing or extra.
    """
    embedder = Embedder(feature_set="full")
    bd = BattleData.from_showdown_json(vgc_json_anon)
    iterator = BattleIterator(bd, perspective="p1")

    expected_size = embedder.embedding_size
    decision_count = 0

    while iterator.next_input():
        emb = embedder.embed(iterator.battle)  # type: ignore
        vec = embedder.feature_dict_to_vector(emb)

        # Vector size should match embedding_size every decision point
        assert len(vec) == expected_size, (
            f"Decision {decision_count}: vector size {len(vec)} != expected {expected_size}"
        )

        # All species_id features should be valid (not NaN)
        for key, val in emb.items():
            if "species_id" in key:
                assert val == -1 or (isinstance(val, (int, float)) and val >= 0), (
                    f"Decision {decision_count}: invalid species_id value {val} for {key}"
                )
            if "move_id" in key:
                assert val == -1 or (isinstance(val, (int, float)) and val >= 0), (
                    f"Decision {decision_count}: invalid move_id value {val} for {key}"
                )

        decision_count += 1

    assert decision_count > 0, "Battle should have at least one decision point"
