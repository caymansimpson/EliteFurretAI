# -*- coding: utf-8 -*-
"""
Tests for the 4 observation embedding improvements:
1. Duration encoding for weather/fields/side conditions
2. Transition features from previous turn events
3. One-hot boost encoding (-6 to +6)
4. Entity ID embeddings for abilities and items
"""
from logging import Logger

import torch
from poke_env.battle import (
    DoubleBattle,
    Field,
    Move,
    Pokemon,
    SideCondition,
    Weather,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.etl import Embedder
from elitefurretai.etl.embedder import (
    ABILITY_TO_ID,
    BOOST_RANGE,
    FIELD_DURATIONS,
    ITEM_TO_ID,
    NUM_ABILITIES,
    NUM_ITEMS,
    NUM_MOVES,
    NUM_SPECIES,
    SIDE_CONDITION_DURATIONS,
    TRACKED_FIELDS,
    TRACKED_ITEMS,
    TRACKED_WEATHERS,
)
from elitefurretai.supervised.model_archs import EntityIDEncoder, GroupedFeatureEncoder


def _make_battle(turn=0, format_str="gen9vgc2023regc"):
    """Helper to create a dummy battle with a player role."""
    battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
    battle._format = format_str
    battle.player_role = "p1"
    battle._turn = turn
    return battle


def _make_furret():
    """Create a Furret with known ability/item for testing."""
    tb = ConstantTeambuilder(
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
    return Pokemon(gen=9, teambuilder=tb.team[0])


# ─────────────────────────────────────────────────────────────────────
# 1. Duration Encoding Tests
# ─────────────────────────────────────────────────────────────────────


class TestDurationEncoding:
    """Tests for :remaining duration features on weather, fields, and side conditions."""

    def test_duration_features_exist_at_turn_0(self):
        """At turn 0, all :remaining features should be 0 (nothing active)."""
        embedder = Embedder()
        battle = _make_battle(turn=0)
        emb = embedder.generate_battle_features(battle)

        # Weather remaining features
        for weather in TRACKED_WEATHERS:
            key = f"WEATHER:{weather.name}:remaining"
            assert key in emb, f"Missing weather duration feature: {key}"
            assert emb[key] == 0, f"{key} should be 0 at turn 0"

        # Field remaining features
        for field in TRACKED_FIELDS:
            key = f"FIELD:{field.name}:remaining"
            assert key in emb, f"Missing field duration feature: {key}"
            assert emb[key] == 0, f"{key} should be 0 at turn 0"

        # Side condition remaining features (only for duration-based ones)
        for sc in SIDE_CONDITION_DURATIONS:
            for prefix in ["SIDE_CONDITION:", "OPP_SIDE_CONDITION:"]:
                key = f"{prefix}{sc.name}:remaining"
                assert key in emb, f"Missing side condition duration feature: {key}"
                assert emb[key] == 0, f"{key} should be 0 at turn 0"

    def test_weather_duration_just_set(self):
        """Weather just set this turn should have full remaining (remaining/duration)."""
        embedder = Embedder()
        battle = _make_battle(turn=3)
        # Sandstorm set on turn 3, current turn is 3
        battle._weather = {Weather.SANDSTORM: 3}

        emb = embedder.generate_battle_features(battle)

        # remaining = max(0, 5 - (3 - 3)) = 5, normalized = 5/5 = 1.0
        assert emb["WEATHER:SANDSTORM:remaining"] == 1.0
        # Other weathers should be 0
        assert emb["WEATHER:RAINDANCE:remaining"] == 0
        assert emb["WEATHER:SUNNYDAY:remaining"] == 0
        assert emb["WEATHER:SNOWSCAPE:remaining"] == 0

    def test_weather_duration_decays(self):
        """Weather should decay over turns."""
        embedder = Embedder()
        battle = _make_battle(turn=5)
        # Sandstorm set on turn 3, current turn is 5
        battle._weather = {Weather.SANDSTORM: 3}

        emb = embedder.generate_battle_features(battle)

        # remaining = max(0, 5 - (5 - 3)) = 3, normalized = 3/5 = 0.6
        assert emb["WEATHER:SANDSTORM:remaining"] == 0.6

    def test_weather_duration_expired(self):
        """Weather that has expired should have 0 remaining."""
        embedder = Embedder()
        battle = _make_battle(turn=10)
        # Sandstorm set on turn 3, current turn is 10
        battle._weather = {Weather.SANDSTORM: 3}

        emb = embedder.generate_battle_features(battle)

        # remaining = max(0, 5 - (10 - 3)) = max(0, -2) = 0
        assert emb["WEATHER:SANDSTORM:remaining"] == 0

    def test_field_duration_trick_room(self):
        """Trick Room should have correct :remaining values."""
        embedder = Embedder()
        battle = _make_battle(turn=4)
        battle._fields = {Field.TRICK_ROOM: 2}  # Set on turn 2

        emb = embedder.generate_battle_features(battle)

        # Trick Room lasts 5 turns. remaining = max(0, 5 - (4-2)) = 3
        duration = FIELD_DURATIONS[Field.TRICK_ROOM]
        expected = max(0, duration - (4 - 2)) / duration
        assert emb["FIELD:TRICK_ROOM:remaining"] == expected

        # Other fields should be 0
        assert emb["FIELD:GRASSY_TERRAIN:remaining"] == 0

    def test_side_condition_tailwind_duration(self):
        """Tailwind should have correct duration remaining."""
        embedder = Embedder()
        battle = _make_battle(turn=5)
        battle._side_conditions = {SideCondition.TAILWIND: 4}  # Set on turn 4

        emb = embedder.generate_battle_features(battle)

        # Tailwind lasts 4 turns. remaining = max(0, 4 - (5-4)) = 3
        duration = SIDE_CONDITION_DURATIONS[SideCondition.TAILWIND]
        expected = max(0, duration - (5 - 4)) / duration
        assert emb["SIDE_CONDITION:TAILWIND:remaining"] == expected

    def test_opp_side_condition_duration(self):
        """Opponent side condition duration should also work."""
        embedder = Embedder()
        battle = _make_battle(turn=6)
        battle._opponent_side_conditions = {SideCondition.REFLECT: 3}  # Set on turn 3

        emb = embedder.generate_battle_features(battle)

        # Reflect lasts 5 turns. remaining = max(0, 5 - (6-3)) = 2
        duration = SIDE_CONDITION_DURATIONS[SideCondition.REFLECT]
        expected = max(0, duration - (6 - 3)) / duration
        assert emb["OPP_SIDE_CONDITION:REFLECT:remaining"] == expected

    def test_non_duration_side_conditions_no_remaining(self):
        """Side conditions like Stealth Rock should NOT have :remaining features."""
        embedder = Embedder()
        battle = _make_battle(turn=3)
        battle._side_conditions = {SideCondition.STEALTH_ROCK: 2}

        emb = embedder.generate_battle_features(battle)

        # Stealth Rock is presence-only (not in SIDE_CONDITION_DURATIONS)
        assert "SIDE_CONDITION:STEALTH_ROCK:remaining" not in emb
        # But presence should be 1
        assert emb["SIDE_CONDITION:STEALTH_ROCK"] == 1

    def test_duration_feature_count(self):
        """Verify the expected number of :remaining features."""
        embedder = Embedder()
        battle = _make_battle(turn=0)
        emb = embedder.generate_battle_features(battle)

        remaining_keys = [k for k in emb if ":remaining" in k]
        # 4 weathers + 6 fields + 4 duration SCs * 2 sides = 18
        expected = len(TRACKED_WEATHERS) + len(TRACKED_FIELDS) + 2 * len(SIDE_CONDITION_DURATIONS)
        assert len(remaining_keys) == expected


# ─────────────────────────────────────────────────────────────────────
# 2. Transition Feature Tests
# ─────────────────────────────────────────────────────────────────────


class TestTransitionFeatures:
    """Tests for transition features extracted from previous turn events."""

    def test_transition_features_default_turn_0(self):
        """At turn 0, all transition features should default to -1."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=0)

        features = embedder.generate_transition_features(battle)

        assert len(features) > 0
        for key, val in features.items():
            assert val == -1, f"{key} should be -1 at turn 0, got {val}"

    def test_transition_features_default_turn_1(self):
        """At turn 1, prev_turn=0 which has no usable events, so all -1."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=1)

        features = embedder.generate_transition_features(battle)

        for key, val in features.items():
            assert val == -1, f"{key} should be -1 at turn 1 (no prev observations), got {val}"

    def test_transition_feature_keys(self):
        """Verify all expected transition feature keys are present."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=0)

        features = embedder.generate_transition_features(battle)

        # Per-slot features: 4 slots × 8 features = 32
        slots = ["MY:0:", "MY:1:", "OPP:0:", "OPP:1:"]
        per_slot_feats = [
            "used_move", "used_switch", "used_protect",
            "was_crit", "hit_super_effective", "hit_resisted",
            "move_missed", "move_failed",
        ]
        for slot in slots:
            for feat in per_slot_feats:
                key = f"TRANSITION:{slot}{feat}"
                assert key in features, f"Missing feature: {key}"

        # Global features
        assert "TRANSITION:my_slot0_moved_first" in features
        assert "TRANSITION:my_slot1_moved_first" in features
        assert "TRANSITION:any_faint_last_turn" in features
        assert "TRANSITION:my_faint_last_turn" in features
        assert "TRANSITION:opp_faint_last_turn" in features

    def test_transition_feature_count(self):
        """Verify the total number of transition features."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=0)

        features = embedder.generate_transition_features(battle)

        # 4 slots × 8 per-slot + 5 global = 37
        assert len(features) == 37

    def test_transition_from_real_battle(self, vgc_battle_p1_logs):
        """Test transition features from parsing a real battle log.

        Turn 1 events include (from p1 perspective):
        - p2b switches in (OPP:1: used_switch)
        - p1b Smeargle uses U-turn (MY:1: used_move)
        - The U-turn triggers p1a eject button switch
        - p2a Tyranitar uses Dragon Tail (OPP:0: used_move)
        - p1b supereffective on p1a Tyranitar means MY:1: hit_super_effective
        """
        embedder = Embedder(feature_set="full")

        # Parse battle through multiple turns
        p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
        p1_battle._format = "gen9vgc2023regc"
        for turn in vgc_battle_p1_logs:
            for log in turn:
                if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                    p1_battle.parse_message(log)
                elif len(log) > 1 and log[1] == "win":
                    p1_battle.won_by(log[2])

            # Check transition features at turn 2 (should have turn 1 events)
            if p1_battle.turn == 2:
                features = embedder.generate_transition_features(p1_battle)

                # Turn 1 has events, so features should not all be -1
                assert features["TRANSITION:any_faint_last_turn"] == 0  # No faints turn 1
                assert features["TRANSITION:my_faint_last_turn"] == 0
                assert features["TRANSITION:opp_faint_last_turn"] == 0

                # OPP:1: switched in (p2b: gagaga switch)
                assert features["TRANSITION:OPP:1:used_switch"] == 1

                # p1b Smeargle used U-turn (MY:1: used_move)
                assert features["TRANSITION:MY:1:used_move"] == 1

                # p2a Tyranitar used Dragon Tail (OPP:0: used_move)
                assert features["TRANSITION:OPP:0:used_move"] == 1

                # p1b's U-turn was super-effective on p1a Tyranitar
                assert features["TRANSITION:MY:1:hit_super_effective"] == 1

                break

    def test_transition_with_faint(self, vgc_battle_p1_logs):
        """Test transition features at turn 4 where Smeargle faints."""
        embedder = Embedder(feature_set="full")

        p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
        p1_battle._format = "gen9vgc2023regc"
        for turn in vgc_battle_p1_logs:
            for log in turn:
                if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                    p1_battle.parse_message(log)
                elif len(log) > 1 and log[1] == "win":
                    p1_battle.won_by(log[2])

            # Turn 3 events: Smeargle (p1b) faints
            if p1_battle.turn == 4:
                features = embedder.generate_transition_features(p1_battle)

                assert features["TRANSITION:any_faint_last_turn"] == 1
                assert features["TRANSITION:my_faint_last_turn"] == 1  # p1b fainted
                assert features["TRANSITION:opp_faint_last_turn"] == 0

                # p1a Best Bud used Thunder (missed)
                assert features["TRANSITION:MY:0:used_move"] == 1
                assert features["TRANSITION:MY:0:move_missed"] == 1

                # p2b Best Bud also used Thunder (missed)
                assert features["TRANSITION:OPP:1:used_move"] == 1
                assert features["TRANSITION:OPP:1:move_missed"] == 1

                # p1b Smeargle used Spikes
                assert features["TRANSITION:MY:1:used_move"] == 1

                # p2a Tyranitar used Dragon Tail
                assert features["TRANSITION:OPP:0:used_move"] == 1

                break

    def test_transition_with_resisted(self, vgc_battle_p1_logs):
        """Test transition features at turn 5: resisted hits."""
        embedder = Embedder(feature_set="full")

        p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
        p1_battle._format = "gen9vgc2023regc"
        for turn in vgc_battle_p1_logs:
            for log in turn:
                if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                    p1_battle.parse_message(log)
                elif len(log) > 1 and log[1] == "win":
                    p1_battle.won_by(log[2])

            # Turn 4 events: resisted hits
            if p1_battle.turn == 5:
                features = embedder.generate_transition_features(p1_battle)

                # p1b gagaga used Behemoth Bash, resisted by p2b Best Bud
                assert features["TRANSITION:MY:1:used_move"] == 1
                assert features["TRANSITION:MY:1:hit_resisted"] == 1

                # p1a Best Bud fainted this turn
                assert features["TRANSITION:my_faint_last_turn"] >= 1

                break

    def test_transition_included_in_full_embed(self, vgc_battle_p1_logs):
        """Transition features should appear in embed() output when feature_set='full'."""
        embedder = Embedder(feature_set="full")

        p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
        p1_battle._format = "gen9vgc2023regc"
        for turn_idx, turn in enumerate(vgc_battle_p1_logs):
            for log in turn:
                if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                    p1_battle.parse_message(log)
                elif len(log) > 1 and log[1] == "win":
                    p1_battle.won_by(log[2])
            if turn_idx == 0:
                break

        full_emb = embedder.embed(p1_battle)

        # Check transition features exist in the full embedding
        transition_keys = [k for k in full_emb if k.startswith("TRANSITION:")]
        assert len(transition_keys) == 37

    def test_transition_not_in_raw_embed(self):
        """Transition features should NOT appear in embed() when feature_set='raw'."""
        raw_embedder = Embedder(feature_set="raw")
        battle = _make_battle(turn=2)

        emb = raw_embedder.embed(battle)

        transition_keys = [k for k in emb if k.startswith("TRANSITION:")]
        assert len(transition_keys) == 0

    def test_transition_in_group_sizes(self):
        """Transition features should appear in group_embedding_sizes for FULL."""
        full_embedder = Embedder(feature_set="full")

        group_sizes = full_embedder.group_embedding_sizes

        # For FULL: 6 player mons + 6 opp mons + 1 battle + 1 feat_eng + 1 transition = 15
        assert len(group_sizes) == 15

        # Last group should be transition size
        assert group_sizes[-1] == full_embedder.transition_embedding_size
        assert group_sizes[-1] == 37


# ─────────────────────────────────────────────────────────────────────
# 3. One-Hot Boost Encoding Tests
# ─────────────────────────────────────────────────────────────────────


class TestOneHotBoostEncoding:
    """Tests for one-hot encoding of stat boosts."""

    def test_boost_range_constant(self):
        """BOOST_RANGE should be [-6, -5, ..., 5, 6] = 13 values."""
        assert BOOST_RANGE == list(range(-6, 7))
        assert len(BOOST_RANGE) == 13

    def test_boost_keys_format(self):
        """Boost features should use the BOOST:stat:value format."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()

        emb = embedder.generate_pokemon_features(furret, battle)

        # Check that we have boost keys in the new format
        stats = ["accuracy", "atk", "def", "spa", "spd", "spe"]
        for stat in stats:
            for b in BOOST_RANGE:
                key = f"BOOST:{stat}:{b}"
                assert key in emb, f"Missing boost feature: {key}"

        # Old format should NOT exist
        for stat in stats:
            old_key = f"BOOST:{stat}"
            assert old_key not in emb, f"Old boost format still present: {old_key}"

    def test_boost_one_hot_default_zero(self):
        """When boost is 0 (default), only the :0 bin should be 1."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        # Furret starts with all boosts at 0

        emb = embedder.generate_pokemon_features(furret, battle)

        # accuracy boost is 0, so BOOST:accuracy:0 should be 1
        assert emb["BOOST:accuracy:0"] == 1
        for b in BOOST_RANGE:
            if b != 0:
                assert emb[f"BOOST:accuracy:{b}"] == 0

        # atk boost is 0
        assert emb["BOOST:atk:0"] == 1
        for b in BOOST_RANGE:
            if b != 0:
                assert emb[f"BOOST:atk:{b}"] == 0

    def test_boost_one_hot_positive(self):
        """Test positive boost values."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("spe", 2)

        emb = embedder.generate_pokemon_features(furret, battle)

        # spe boost is 2
        assert emb["BOOST:spe:2"] == 1
        assert emb["BOOST:spe:0"] == 0
        assert emb["BOOST:spe:1"] == 0
        assert emb["BOOST:spe:3"] == 0
        assert emb["BOOST:spe:-1"] == 0

    def test_boost_one_hot_negative(self):
        """Test negative boost values."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("atk", -3)

        emb = embedder.generate_pokemon_features(furret, battle)

        assert emb["BOOST:atk:-3"] == 1
        assert emb["BOOST:atk:0"] == 0
        assert emb["BOOST:atk:-2"] == 0

    def test_boost_one_hot_boundary_max(self):
        """Test maximum boost value (+6)."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("spa", 6)

        emb = embedder.generate_pokemon_features(furret, battle)

        assert emb["BOOST:spa:6"] == 1
        assert emb["BOOST:spa:5"] == 0
        assert emb["BOOST:spa:0"] == 0

    def test_boost_one_hot_boundary_min(self):
        """Test minimum boost value (-6)."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("def", -6)

        emb = embedder.generate_pokemon_features(furret, battle)

        assert emb["BOOST:def:-6"] == 1
        assert emb["BOOST:def:-5"] == 0
        assert emb["BOOST:def:0"] == 0

    def test_boost_none_mon_all_minus_one(self):
        """When mon is None, all boost features should be -1."""
        embedder = Embedder()
        battle = _make_battle()

        emb = embedder.generate_pokemon_features(None, battle)

        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            for b in BOOST_RANGE:
                assert emb[f"BOOST:{stat}:{b}"] == -1

    def test_boost_total_features_per_stat(self):
        """Each stat should have 13 one-hot features (BOOST_RANGE has 13 values)."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()

        emb = embedder.generate_pokemon_features(furret, battle)

        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            boost_keys = [k for k in emb if k.startswith(f"BOOST:{stat}:")]
            assert len(boost_keys) == 13, f"Expected 13 boost keys for {stat}, got {len(boost_keys)}"

    def test_boost_one_hot_opponent(self):
        """One-hot boosts should also work for opponent pokemon features."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("spd", 4)

        emb = embedder.generate_opponent_pokemon_features(furret, battle)

        assert emb["BOOST:spd:4"] == 1
        assert emb["BOOST:spd:0"] == 0
        for b in BOOST_RANGE:
            if b != 4:
                assert emb[f"BOOST:spd:{b}"] == 0

    def test_boost_one_hot_exactly_one_hot(self):
        """For a live mon, exactly one bin per stat should be 1, rest 0."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret.set_boost("atk", 3)
        furret.set_boost("spe", -2)
        furret.set_boost("def", 1)

        emb = embedder.generate_pokemon_features(furret, battle)

        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            vals = [emb[f"BOOST:{stat}:{b}"] for b in BOOST_RANGE]
            assert sum(vals) == 1, f"BOOST:{stat} not exactly one-hot: {vals}"

    def test_move_boost_unchanged(self):
        """Move-level BOOST features should still use raw integers, not one-hot.

        The one-hot encoding only applies to Pokemon stat boosts, not move
        effect boosts (like 'BOOST:atk' on a move's secondary effects).
        """
        embedder = Embedder()

        emb = embedder.generate_move_features(Move("spore", gen=9))

        # Move boosts are still raw integers (0, 1, -1, etc.)
        assert emb["BOOST:atk"] == 0
        assert emb["BOOST:def"] == 0
        assert emb["BOOST:spa"] == 0
        assert emb["BOOST:spe"] == 0

        emb2 = embedder.generate_move_features(Move("ominouswind", gen=9))
        assert emb2["SELFBOOST:atk"] == 1


# ─────────────────────────────────────────────────────────────────────
# 4. Entity ID Embedding Tests
# ─────────────────────────────────────────────────────────────────────


class TestEntityIDEmbeddings:
    """Tests for entity ID embeddings (ability_id, item_id)."""

    def test_ability_to_id_mapping(self):
        """ABILITY_TO_ID should map GenData abilities to IDs starting at 1."""
        assert len(ABILITY_TO_ID) > 0
        # IDs should start at 1 (0 is reserved for unknown)
        assert min(ABILITY_TO_ID.values()) == 1
        assert max(ABILITY_TO_ID.values()) == len(ABILITY_TO_ID)
        # Spot-check known Gen 9 abilities
        assert "intimidate" in ABILITY_TO_ID
        assert "runaway" in ABILITY_TO_ID
        assert "moody" in ABILITY_TO_ID

    def test_item_to_id_mapping(self):
        """ITEM_TO_ID should map all tracked items to IDs starting at 1."""
        assert len(ITEM_TO_ID) == len(TRACKED_ITEMS)
        assert min(ITEM_TO_ID.values()) == 1
        assert max(ITEM_TO_ID.values()) == len(TRACKED_ITEMS)
        for item in TRACKED_ITEMS:
            assert item in ITEM_TO_ID

    def test_num_abilities_includes_unknown(self):
        """NUM_ABILITIES should be len(ABILITY_TO_ID) + 1 for the unknown token."""
        assert NUM_ABILITIES == len(ABILITY_TO_ID) + 1

    def test_num_items_includes_unknown(self):
        """NUM_ITEMS should be len(TRACKED_ITEMS) + 1 for the unknown token."""
        assert NUM_ITEMS == len(TRACKED_ITEMS) + 1

    def test_known_ability_id(self):
        """A tracked ability should have its correct ID."""
        embedder = Embedder()
        battle = _make_battle()

        # Create a mon with intimidate (tracked ability)
        tb = ConstantTeambuilder(
            """Incineroar @ Leftovers
            Ability: Intimidate
            Level: 50
            Tera Type: Fire
            EVs: 252 HP
            Adamant Nature
            - Flare Blitz
            - Darkest Lariat
            - Fake Out
            - U-turn"""
        )
        incin = Pokemon(gen=9, teambuilder=tb.team[0])
        emb = embedder.generate_pokemon_features(incin, battle)

        assert emb["ability_id"] == ABILITY_TO_ID["intimidate"]
        assert emb["ability_id"] > 0

    def test_unknown_ability_id(self):
        """An unknown/non-GenData ability should map to 0."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()
        furret._ability = "notarealability"  # type: ignore[attr-defined]

        emb = embedder.generate_pokemon_features(furret, battle)

        assert emb["ability_id"] == 0

    def test_none_mon_ability_minus_one(self):
        """When mon is None, ability_id should be -1."""
        embedder = Embedder()
        battle = _make_battle()

        emb = embedder.generate_pokemon_features(None, battle)
        assert emb["ability_id"] == -1

    def test_known_item_id(self):
        """A tracked item should have its correct ID."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()  # Has Leftovers

        emb = embedder.generate_pokemon_features(furret, battle)

        assert emb["item_id"] == ITEM_TO_ID["leftovers"]
        assert emb["item_id"] > 0

    def test_unknown_item_id(self):
        """An untracked item should map to 0."""
        embedder = Embedder()
        battle = _make_battle()

        tb = ConstantTeambuilder(
            """Furret @ Berry Juice
            Ability: Run Away
            Level: 50
            Tera Type: Normal
            EVs: 252 Atk
            Adamant Nature
            - Body Slam
            - Agility
            - Baton Pass
            - Blizzard"""
        )
        mon = Pokemon(gen=9, teambuilder=tb.team[0])
        emb = embedder.generate_pokemon_features(mon, battle)

        # Berry Juice is not in TRACKED_ITEMS
        assert "berryjuice" not in TRACKED_ITEMS
        assert emb["item_id"] == 0

    def test_none_mon_item_minus_one(self):
        """When mon is None, item_id should be -1."""
        embedder = Embedder()
        battle = _make_battle()

        emb = embedder.generate_pokemon_features(None, battle)
        assert emb["item_id"] == -1

    def test_no_ohe_ability_keys(self):
        """There should be no 'ABILITY:' OHE features anymore."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()

        emb = embedder.generate_pokemon_features(furret, battle)
        ability_keys = [k for k in emb if k.startswith("ABILITY:")]
        assert len(ability_keys) == 0, f"Found OHE ability keys: {ability_keys}"

    def test_no_ohe_item_keys(self):
        """There should be no 'ITEM:' OHE features anymore."""
        embedder = Embedder()
        battle = _make_battle()
        furret = _make_furret()

        emb = embedder.generate_pokemon_features(furret, battle)
        item_keys = [k for k in emb if k.startswith("ITEM:")]
        assert len(item_keys) == 0, f"Found OHE item keys: {item_keys}"

    def test_opponent_known_ability(self, vgc_battle_p1_logs):
        """Opponent with a known ability should get the correct ID."""
        embedder = Embedder()
        battle = _make_battle()

        # Parse battle to get opponent mons
        p1_battle = DoubleBattle("tag", "elitefurretai", Logger("example"), gen=9)
        p1_battle._format = "gen9vgc2023regc"
        for turn in vgc_battle_p1_logs:
            for log in turn:
                if len(log) > 1 and log[1] not in ["", "t:", "win"]:
                    p1_battle.parse_message(log)
                elif len(log) > 1 and log[1] == "win":
                    p1_battle.won_by(log[2])

        smeargle = p1_battle.opponent_team["p2: Smeargle"]
        emb = embedder.generate_opponent_pokemon_features(smeargle, battle)

        # Smeargle's ability "moody" is tracked
        assert emb["ability_id"] == ABILITY_TO_ID["moody"]

    def test_opponent_unknown_ability(self):
        """Opponent with unknown ability should get ID 0."""
        embedder = Embedder()
        battle = _make_battle()

        # Create a mon without revealing ability
        mon = Pokemon(gen=9, species="pikachu")
        # mon.ability is None by default
        emb = embedder.generate_opponent_pokemon_features(mon, battle)

        assert emb["ability_id"] == 0  # Unknown ability

    def test_opponent_unknown_item(self):
        """Opponent with unknown item should get ID 0."""
        embedder = Embedder()
        battle = _make_battle()

        mon = Pokemon(gen=9, species="pikachu")
        emb = embedder.generate_opponent_pokemon_features(mon, battle)

        assert emb["item_id"] == 0  # Unknown item

    def test_embedding_size_consistency(self):
        """Player and opponent pokemon embedding sizes should be consistent."""
        embedder = Embedder()
        battle = _make_battle()

        furret = _make_furret()
        player_emb = embedder.generate_pokemon_features(furret, battle)
        none_player = embedder.generate_pokemon_features(None, battle)

        assert len(player_emb) == len(none_player)

        opp_emb = embedder.generate_opponent_pokemon_features(furret, battle)
        none_opp = embedder.generate_opponent_pokemon_features(None, battle)

        assert len(opp_emb) == len(none_opp)

    def test_ability_id_deterministic_ordering(self):
        """ABILITY_TO_ID should have deterministic ordering (sorted alphabetically)."""
        sorted_abilities = sorted(ABILITY_TO_ID.keys())
        for i, ability in enumerate(sorted_abilities):
            assert ABILITY_TO_ID[ability] == i + 1

    def test_item_id_deterministic_ordering(self):
        """ITEM_TO_ID should have deterministic ordering (sorted alphabetically)."""
        sorted_items = sorted(TRACKED_ITEMS)
        for i, item in enumerate(sorted_items):
            assert ITEM_TO_ID[item] == i + 1


# ─────────────────────────────────────────────────────────────────────
# 5. EntityIDEncoder Model Tests
# ─────────────────────────────────────────────────────────────────────


class TestEntityIDEncoderModel:
    """Tests for the EntityIDEncoder nn.Module integration."""

    def _make_feature_names_and_sizes(self):
        """Create realistic feature names and group sizes from embedder."""
        embedder = Embedder(feature_set="raw")
        return embedder.feature_names, embedder.group_embedding_sizes

    def test_entity_id_encoder_creation(self):
        """EntityIDEncoder should create with correct structure."""
        feature_names, group_sizes = self._make_feature_names_and_sizes()

        encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            ability_embed_dim=16,
            item_embed_dim=16,
        )

        assert encoder.ability_emb.num_embeddings == NUM_ABILITIES
        assert encoder.item_emb.num_embeddings == NUM_ITEMS
        assert encoder.ability_emb.embedding_dim == 16
        assert encoder.item_emb.embedding_dim == 16
        assert encoder.ability_emb.padding_idx == 0
        assert encoder.item_emb.padding_idx == 0

    def test_entity_id_encoder_identifies_id_positions(self):
        """EntityIDEncoder should find ability_id and item_id positions in pokemon groups.

        With group-preserving feature order, groups 0-5 are player Pokemon and
        groups 6-11 are opponent Pokemon. All 12 pokemon groups have entity IDs.
        Group 12 (battle state) does not.
        """
        feature_names, group_sizes = self._make_feature_names_and_sizes()

        encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
        )

        # Groups 0-11 should each have ability_id and item_id
        for gi in range(12):
            gmap = encoder._group_maps[gi]
            entity_types = [etype for _, etype in gmap]
            assert "ability" in entity_types, f"Group {gi} missing ability ID"
            assert "item" in entity_types, f"Group {gi} missing item ID"

        # Group 12 (battle state) should NOT have any entity IDs
        gmap = encoder._group_maps[12]
        assert len(gmap) == 0, "Group 12 (battle state) should not have entity IDs"

    def test_entity_id_encoder_output_size(self):
        """Output size should expand scalar IDs to embedding dimensions."""
        feature_names, group_sizes = self._make_feature_names_and_sizes()

        encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            ability_embed_dim=16,
            item_embed_dim=16,
            species_embed_dim=32,
            move_embed_dim=16,
        )

        # Group 12 (battle state) has no entity IDs
        assert encoder.group_output_sizes[12] == group_sizes[12]

        # Groups 0-11 have entity IDs (ability, item, species, move)
        for gi in range(12):
            n_ids = len(encoder._group_maps[gi])
            assert n_ids > 0, f"Group {gi} should have entity IDs"

    def test_entity_id_encoder_forward(self):
        """Verify forward pass produces correct output shapes."""
        feature_names, group_sizes = self._make_feature_names_and_sizes()

        encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            ability_embed_dim=16,
            item_embed_dim=8,
            species_embed_dim=32,
            move_embed_dim=16,
        )

        batch, seq = 2, 3
        # Create dummy input for each group
        for gi, gsize in enumerate(group_sizes):
            x = torch.randn(batch, seq, gsize)
            # Set entity ID positions to valid integer values
            for local_idx, etype in encoder._group_maps[gi]:
                if etype == "ability":
                    x[:, :, local_idx] = torch.randint(0, NUM_ABILITIES, (batch, seq)).float()
                elif etype == "item":
                    x[:, :, local_idx] = torch.randint(0, NUM_ITEMS, (batch, seq)).float()
                elif etype == "species":
                    x[:, :, local_idx] = torch.randint(0, NUM_SPECIES, (batch, seq)).float()
                else:  # move
                    x[:, :, local_idx] = torch.randint(0, NUM_MOVES, (batch, seq)).float()

            out = encoder.embed_group(x, gi)

            expected_size = encoder.group_output_sizes[gi]
            assert out.shape == (batch, seq, expected_size), (
                f"Group {gi}: expected shape {(batch, seq, expected_size)}, got {out.shape}"
            )

    def test_entity_id_encoder_sentinel_handling(self):
        """Sentinel value -1 should be clamped to 0 (padding index)."""
        feature_names, group_sizes = self._make_feature_names_and_sizes()

        encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
        )

        # Use a pokemon group
        gi = 0
        gsize = group_sizes[gi]
        x = torch.zeros(1, 1, gsize)

        # Set entity IDs to -1 (sentinel)
        for local_idx, etype in encoder._group_maps[gi]:
            x[:, :, local_idx] = -1

        # Should not raise - sentinel should be clamped to 0
        out = encoder.embed_group(x, gi)
        assert out.shape[2] == encoder.group_output_sizes[gi]

    def test_entity_id_padding_idx_gradient(self):
        """Padding index 0 should not receive gradients during training."""
        encoder = EntityIDEncoder(
            feature_names=["ability_id", "item_id"],
            group_sizes=[2],
            num_abilities=5,
            num_items=5,
        )

        # Set input to 0 (unknown) and check that padding_idx blocks gradients
        x = torch.zeros(1, 1, 2)
        out = encoder.embed_group(x, 0)
        loss = out.sum()
        loss.backward()

        # Padding idx weights should have zero gradient
        assert encoder.ability_emb.weight.grad is not None
        assert encoder.item_emb.weight.grad is not None
        assert torch.all(encoder.ability_emb.weight.grad[0] == 0)
        assert torch.all(encoder.item_emb.weight.grad[0] == 0)


class TestGroupedFeatureEncoderWithEntityIDs:
    """Tests for GroupedFeatureEncoder with entity ID and number bank parameters."""

    def test_grouped_encoder_with_entity_ids(self):
        """GroupedFeatureEncoder should accept entity ID params and produce output."""
        embedder = Embedder(feature_set="raw")
        feature_names = embedder.feature_names
        group_sizes = embedder.group_embedding_sizes

        gfe = GroupedFeatureEncoder(
            group_sizes=group_sizes,
            feature_names=feature_names,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            hidden_dim=64,
            aggregated_dim=256,
        )

        batch, seq = 2, 3
        total_features = sum(group_sizes)
        x = torch.randn(batch, seq, total_features)

        # Set entity ID positions to valid integer values
        offset = 0
        for gi, gsize in enumerate(group_sizes):
            for local_idx, etype in gfe.entity_id_encoder._group_maps[gi]:
                if etype == "ability":
                    x[:, :, offset + local_idx] = torch.randint(0, NUM_ABILITIES, (batch, seq)).float()
                elif etype == "item":
                    x[:, :, offset + local_idx] = torch.randint(0, NUM_ITEMS, (batch, seq)).float()
                elif etype == "species":
                    x[:, :, offset + local_idx] = torch.randint(0, NUM_SPECIES, (batch, seq)).float()
                else:  # move
                    x[:, :, offset + local_idx] = torch.randint(0, NUM_MOVES, (batch, seq)).float()
            offset += gsize

        out = gfe(x)
        assert out.shape == (batch, seq, 256)

    def test_grouped_encoder_entity_ids_backprop(self):
        """Verify gradients flow through entity ID embeddings in GroupedFeatureEncoder."""
        embedder = Embedder(feature_set="raw")
        feature_names = embedder.feature_names
        group_sizes = embedder.group_embedding_sizes

        gfe = GroupedFeatureEncoder(
            group_sizes=group_sizes,
            feature_names=feature_names,
            num_abilities=NUM_ABILITIES,
            num_items=NUM_ITEMS,
            num_species=NUM_SPECIES,
            num_moves=NUM_MOVES,
            hidden_dim=64,
            aggregated_dim=256,
        )

        batch, seq = 1, 1
        total_features = sum(group_sizes)
        x = torch.randn(batch, seq, total_features)

        # Set entity IDs to valid non-zero values (to avoid padding_idx)
        offset = 0
        for gi, gsize in enumerate(group_sizes):
            for local_idx, etype in gfe.entity_id_encoder._group_maps[gi]:
                x[:, :, offset + local_idx] = 1.0  # ID=1
            offset += gsize

        out = gfe(x)
        loss = out.sum()
        loss.backward()

        # Entity embedding weights with ID=1 should have non-zero gradients
        assert gfe.entity_id_encoder.ability_emb.weight.grad is not None
        assert torch.any(gfe.entity_id_encoder.ability_emb.weight.grad[1] != 0)


# ─────────────────────────────────────────────────────────────────────
# 6. Integration Tests
# ─────────────────────────────────────────────────────────────────────


class TestEmbedderIntegration:
    """Integration tests verifying all improvements work together."""

    def test_full_embed_size_consistency(self):
        """Full embedding should have consistent size across all omniscient settings."""
        e_full = Embedder(feature_set="full", omniscient=False)
        e_full_omni = Embedder(feature_set="full", omniscient=True)

        assert e_full.embedding_size == e_full_omni.embedding_size
        assert e_full.pokemon_embedding_size == e_full_omni.pokemon_embedding_size
        assert e_full.opponent_pokemon_embedding_size == e_full_omni.opponent_pokemon_embedding_size
        assert e_full.battle_embedding_size == e_full_omni.battle_embedding_size
        assert e_full.transition_embedding_size == e_full_omni.transition_embedding_size

    def test_full_embed_includes_all_improvements(self):
        """Full embedding should include duration, transition, boost, and entity ID features."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=0)
        emb = embedder.embed(battle)

        # Duration features (in battle features)
        assert any(":remaining" in k for k in emb), "Missing duration features"

        # Transition features
        assert any(k.startswith("TRANSITION:") for k in emb), "Missing transition features"

        # One-hot boost features (in pokemon features)
        boost_keys = [k for k in emb if "BOOST:" in k and ":" in k.split("BOOST:")[1]]
        assert len(boost_keys) > 0, "Missing one-hot boost features"

        # Entity ID features
        ability_id_keys = [k for k in emb if "ability_id" in k]
        item_id_keys = [k for k in emb if "item_id" in k]
        species_id_keys = [k for k in emb if "species_id" in k]
        move_id_keys = [k for k in emb if "move_id" in k]
        assert len(ability_id_keys) == 12, f"Expected 12 ability_id keys (6 player + 6 opp), got {len(ability_id_keys)}"
        assert len(item_id_keys) == 12, f"Expected 12 item_id keys, got {len(item_id_keys)}"
        assert len(species_id_keys) == 12, f"Expected 12 species_id keys (6 player + 6 opp), got {len(species_id_keys)}"
        assert len(move_id_keys) > 0, f"Expected move_id keys in move features, got {len(move_id_keys)}"

        # Pruned features should be absent
        assert not any(k.endswith(":level") for k in emb), "level feature should be pruned"
        assert not any("TYPE_MATCHUP:" in k for k in emb), "TYPE_MATCHUP should be pruned"

    def test_raw_embed_includes_boost_and_entity_ids(self):
        """RAW embedding should also include one-hot boosts and entity IDs
        (these are in the per-pokemon features, not FULL-only)."""
        embedder = Embedder(feature_set="raw")
        battle = _make_battle(turn=0)
        emb = embedder.embed(battle)

        # Should have one-hot boosts
        boost_keys = [k for k in emb if "BOOST:" in k and ":" in k.split("BOOST:")[1]]
        assert len(boost_keys) > 0

        # Should have entity IDs
        assert any("ability_id" in k for k in emb)
        assert any("item_id" in k for k in emb)

        # Should NOT have transition features (FULL only)
        assert not any(k.startswith("TRANSITION:") for k in emb)

    def test_vector_conversion_roundtrip(self):
        """feature_dict_to_vector should produce a vector of the correct size."""
        embedder = Embedder(feature_set="full")
        battle = _make_battle(turn=0)
        emb = embedder.embed(battle)

        vec = embedder.feature_dict_to_vector(emb)
        assert len(vec) == embedder.embedding_size

        # All values should be numeric
        for v in vec:
            assert isinstance(v, float)

    def test_group_sizes_sum_to_embedding_size(self):
        """Group embedding sizes should sum to the total embedding size for RAW."""
        raw_embedder = Embedder(feature_set="raw")
        assert sum(raw_embedder.group_embedding_sizes) == raw_embedder.embedding_size

    def test_group_sizes_sum_for_full(self):
        """Group embedding sizes should sum to the total embedding size for FULL."""
        full_embedder = Embedder(feature_set="full")
        assert sum(full_embedder.group_embedding_sizes) == full_embedder.embedding_size

    def test_feature_groups_are_semantically_aligned(self):
        """Feature names in each group should belong to the correct semantic group.

        This is a critical invariant: the group-preserving feature order must ensure
        that GroupedFeatureEncoder receives coherent feature groups. Groups are:
        0-5: MON:0-5 (player Pokemon), 6-11: OPP_MON:0-5 (opponent), 12: battle,
        13: feature-engineered (FULL only), 14: transition (FULL only).
        """
        embedder = Embedder(feature_set="full")
        names = embedder.feature_names
        groups = embedder.group_embedding_sizes

        expected_prefixes = (
            [f"MON:{i}:" for i in range(6)]
            + [f"OPP_MON:{i}:" for i in range(6)]
        )

        start = 0
        # Check player and opponent Pokemon groups
        for g_idx in range(12):
            end = start + groups[g_idx]
            group_names = names[start:end]
            prefix = expected_prefixes[g_idx]
            for feat_name in group_names:
                assert feat_name.startswith(prefix), (
                    f"Group {g_idx}: feature '{feat_name}' does not start with '{prefix}'"
                )
            start = end

        # Group 12: battle state (no MON/OPP_MON/TRANSITION/EST_DAMAGE prefix)
        battle_group = names[start : start + groups[12]]
        for feat_name in battle_group:
            assert not feat_name.startswith("MON:"), f"Battle group has MON feature: {feat_name}"
            assert not feat_name.startswith("OPP_MON:"), f"Battle group has OPP_MON feature: {feat_name}"
            assert not feat_name.startswith("TRANSITION:"), f"Battle group has TRANSITION: {feat_name}"
        start += groups[12]

        # Group 13: feature engineered
        eng_group = names[start : start + groups[13]]
        for feat_name in eng_group:
            assert not feat_name.startswith("TRANSITION:"), (
                f"Feature-engineered group has TRANSITION feature: {feat_name}"
            )
        start += groups[13]

        # Group 14: transition
        trans_group = names[start : start + groups[14]]
        for feat_name in trans_group:
            assert feat_name.startswith("TRANSITION:"), (
                f"Transition group has non-TRANSITION feature: {feat_name}"
            )
