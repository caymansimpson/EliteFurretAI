# -*- coding: utf-8 -*-
"""
This module defines the Embedder class, which is responsible for converting battle state information
from poke-env's DoubleBattle objects into feature vectors suitable for machine learning models.
It supports multiple feature sets (SIMPLE, RAW, FULL) and can generate features for moves, Pokémon,
opponent Pokémon, and the overall battle state. The embedder is designed to be flexible and extensible,
allowing for engineered features and grouping of features for advanced architectures.

Key responsibilities:
- Extract and encode all relevant information from a DoubleBattle into a fixed-length vector.
- Support different levels of feature complexity (SIMPLE, RAW, FULL).
- Provide programmatic access to feature names and group sizes for model design.
- Handle missing or incomplete information gracefully, especially during teampreview.
- Generate engineered features to improve model learning.
- Maintain knowledge of enums and tracked game elements for consistent encoding.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Target,
    Weather,
)
from poke_env.calc import calculate_damage
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.stats import compute_raw_stats

from elitefurretai.inference.battle_inference import BattleInference


class Embedder:
    """
    The Embedder class converts battle state information into feature vectors for ML models.
    It supports multiple feature sets and engineered features, and provides metadata for model design.
    """

    SIMPLE = "simple"
    RAW = "raw"
    FULL = "full"

    def __init__(
        self, format="gen9vgc2023regc", feature_set: str = "raw", omniscient=False
    ):
        """
        Initialize the Embedder with a given format and feature set.
        Sets up knowledge bases for enums and tracked elements, and computes embedding sizes.
        """
        self._knowledge: Dict[str, Any] = {}
        self._format: str = format
        self._omniscient: bool = omniscient
        self._ability_to_id: Dict[str, int] = build_ability_to_id(self._format)
        self._num_abilities: int = len(self._ability_to_id) + 1

        assert feature_set in [self.SIMPLE, self.RAW, self.FULL]
        self._feature_set: str = feature_set

        sets = [
            ("Status", Status),
            ("PokemonType", PokemonType),
            ("MoveCategory", MoveCategory),
            ("Effect", Effect),
        ]

        for key, enum in sets:
            self._knowledge[key] = set(enum)

        # Track all relevant game elements for encoding
        self._knowledge["Pokemon"] = set(GenData.from_gen(int(self._format[3])).pokedex.keys())
        self._knowledge["Effect_VolatileStatus"] = TRACKED_EFFECTS
        self._knowledge["Item"] = TRACKED_ITEMS
        self._knowledge["Target"] = TRACKED_TARGET_TYPES
        self._knowledge["Format"] = TRACKED_FORMATS
        self._knowledge["SideCondition"] = TRACKED_SIDE_CONDITIONS
        self._knowledge["Weather"] = TRACKED_WEATHERS
        self._knowledge["Field"] = TRACKED_FIELDS
        self._knowledge["Ability"] = set(self._ability_to_id.keys())

        # Species and move ID mappings for entity ID encoding
        self._species_to_id: Dict[str, int] = build_species_to_id(self._format)
        self._num_species: int = len(self._species_to_id) + 1
        self._move_to_id: Dict[str, int] = build_move_to_id(self._format)
        self._num_moves: int = len(self._move_to_id) + 1

        # Cache for static move features (move.id -> features dict without prefix)
        # This dramatically speeds up embedding since move features are mostly static
        self._move_cache: Dict[str, Dict[str, float]] = {}

        # Generate embedding size and feature names programmatically upon instantiation
        dummy_battle = self._generate_dummy_battle()
        self._embedding_size = len(self.embed(dummy_battle))
        self._move_embedding_size = len(self.generate_move_features(None))
        self._pokemon_embedding_size = len(
            self.generate_pokemon_features(None, dummy_battle)
        )
        self._opponent_pokemon_embedding_size = len(
            self.generate_opponent_pokemon_features(None, dummy_battle)
        )
        self._battle_embedding_size = len(self.generate_battle_features(dummy_battle))
        self._feature_engineered_embedding_size = len(
            self.generate_feature_engineered_features(dummy_battle)
        )
        self._transition_embedding_size = len(
            self.generate_transition_features(dummy_battle)
        )
        self._feature_names = self._compute_grouped_feature_names(dummy_battle)

    @staticmethod
    def _prep(string) -> str:
        """
        Utility to convert enum names to lowercase strings for consistent encoding.
        """
        return string.lower().replace("_", " ")

    def _compute_grouped_feature_names(
        self, battle: DoubleBattle
    ) -> List[str]:
        """Compute feature names in group order with sorting within each group.

        This preserves semantic group boundaries so that model architectures
        (e.g. GroupedFeatureEncoder) that split the flat vector by
        ``group_embedding_sizes`` receive coherent feature groups.

        Order: MON:0-5, OPP_MON:0-5, battle state, [engineered, transition].
        Within each group, feature names are sorted alphabetically for
        deterministic ordering.

        For the SIMPLE feature set (which doesn't use group_embedding_sizes),
        falls back to sorted order of the simplified features.
        """
        # SIMPLE doesn't use groups; just sort its filtered features
        if self._feature_set == self.SIMPLE:
            return sorted(self.embed(battle).keys())

        grouped_names: List[str] = []

        # Player Pokemon 0-5
        for i in range(6):
            group = self.generate_pokemon_features(None, battle, f"MON:{i}:")
            grouped_names.extend(sorted(group.keys()))

        # Opponent Pokemon 0-5
        for i in range(6):
            group = self.generate_opponent_pokemon_features(
                None, battle, prefix=f"OPP_MON:{i}:"
            )
            grouped_names.extend(sorted(group.keys()))

        # Battle state
        group = self.generate_battle_features(battle)
        grouped_names.extend(sorted(group.keys()))

        # Feature engineered + transition (FULL only)
        if self._feature_set == self.FULL:
            group = self.generate_feature_engineered_features(battle)
            grouped_names.extend(sorted(group.keys()))
            group = self.generate_transition_features(battle)
            grouped_names.extend(sorted(group.keys()))

        return grouped_names

    def _generate_dummy_battle(self) -> DoubleBattle:
        """
        Generates a dummy DoubleBattle for feature size calculation and testing.
        """
        dummy_battle = DoubleBattle(
            "tag", "elitefurretai", logging.Logger("example"), gen=int(self._format[3])
        )
        dummy_battle._format = self._format
        dummy_battle.player_role = "p1"
        return dummy_battle

    @property
    def embedding_size(self) -> int:
        """
        Returns the total embedding size for the current feature set.
        """
        return self._embedding_size

    @property
    def move_embedding_size(self) -> int:
        """
        Returns the embedding size for a single move.
        """
        return self._move_embedding_size

    @property
    def pokemon_embedding_size(self) -> int:
        """
        Returns the embedding size for a single Pokémon.
        """
        return self._pokemon_embedding_size

    @property
    def opponent_pokemon_embedding_size(self) -> int:
        """
        Returns the embedding size for a single opponent Pokémon.
        """
        return self._opponent_pokemon_embedding_size

    @property
    def battle_embedding_size(self) -> int:
        """
        Returns the embedding size for the battle-level features.
        """
        return self._battle_embedding_size

    @property
    def feature_engineered_embedding_size(self) -> int:
        """
        Returns the embedding size for engineered features.
        """
        return self._feature_engineered_embedding_size

    @property
    def transition_embedding_size(self) -> int:
        """
        Returns the embedding size for transition features.
        """
        return self._transition_embedding_size

    @property
    def group_embedding_sizes(self) -> List[int]:
        """
        Returns the sizes of feature groups for advanced model architectures.
        Only available for RAW and FULL feature sets.
        """
        assert self.feature_set != self.SIMPLE, (
            "Group embedding sizes are not available for SIMPLE feature set"
        )
        group_sizes = (
            [self._pokemon_embedding_size] * 6
            + [self._opponent_pokemon_embedding_size] * 6
            + [self._battle_embedding_size]
        )
        if self.feature_set == self.FULL:
            group_sizes += [self._feature_engineered_embedding_size]
            group_sizes += [self._transition_embedding_size]
        return group_sizes

    @property
    def feature_names(self) -> List[str]:
        """
        Returns the sorted list of feature names for the current embedding.
        """
        return self._feature_names

    @property
    def feature_set(self) -> str:
        """
        Returns the feature set type (SIMPLE, RAW, FULL).
        """
        return self._feature_set

    @property
    def format(self) -> str:
        """
        Returns the battle format string.
        """
        return self._format

    @property
    def omniscient(self) -> bool:
        """
        Returns whether the embedder is in omniscient mode (has access to all info).
        """
        return self._omniscient

    @property
    def ability_to_id(self) -> Dict[str, int]:
        """Returns the format-specific ability ID mapping (0 reserved for unknown)."""
        return self._ability_to_id

    @property
    def num_abilities(self) -> int:
        """Returns the ability vocabulary size including unknown token (index 0)."""
        return self._num_abilities

    @property
    def species_to_id(self) -> Dict[str, int]:
        """Returns the format-specific species ID mapping (0 reserved for unknown)."""
        return self._species_to_id

    @property
    def num_species(self) -> int:
        """Returns the species vocabulary size including unknown token (index 0)."""
        return self._num_species

    @property
    def move_to_id(self) -> Dict[str, int]:
        """Returns the format-specific move ID mapping (0 reserved for unknown)."""
        return self._move_to_id

    @property
    def num_moves(self) -> int:
        """Returns the move vocabulary size including unknown token (index 0)."""
        return self._num_moves

    @property
    def num_items(self) -> int:
        """Returns the item vocabulary size including unknown token (index 0)."""
        return len(TRACKED_ITEMS) + 1

    def feature_dict_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """
        Converts a feature dictionary returned by this class into a vector that
        can be used as input into a network. Expects a full embedding dict from
        ``embed()`` and uses the cached group-preserving feature order for
        consistency with ``group_embedding_sizes``.
        """
        if set(features.keys()) != set(self._feature_names):
            raise ValueError(
                "feature_dict_to_vector expects the full embedding output from embed(); "
                "for partial feature dicts, convert explicitly by sorting keys"
            )
        return [float(features[key]) for key in self._feature_names]

    def embed_to_array(
        self, battle: DoubleBattle, bi: Optional["BattleInference"] = None
    ) -> np.ndarray:
        """Embed battle state directly to a NumPy array.

        This avoids an intermediate Python list when the caller immediately
        hands the result back to NumPy or PyTorch.
        """
        features = self.embed(battle, bi)
        return np.fromiter(
            (float(features[key]) for key in self._feature_names),
            dtype=np.float32,
            count=len(self._feature_names),
        )

    def embed_to_vector(
        self, battle: DoubleBattle, bi: Optional["BattleInference"] = None
    ) -> List[float]:
        """
        Embed battle state directly to a vector using cached sorted keys.
        This is faster than embed() + feature_dict_to_vector() because
        the sorted key order is cached.

        Args:
            battle: The battle state to embed
            bi: Optional BattleInference for additional features

        Returns:
            List of floats representing the embedded state
        """
        return self.embed_to_array(battle, bi).tolist()

    def _simplify_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a normal feature dictionary into a heavily simplified one.
        Used for testing and ablation studies.
        """
        simple_features = {}
        for k, v in features.items():
            for feature_str in BASIC_FEATURES:
                if feature_str in k:
                    simple_features[k] = v
        return simple_features

    def generate_feature_engineered_features(
        self, battle: DoubleBattle, bi: Optional[BattleInference] = None
    ) -> Dict[str, float]:
        """
        Adds supplementary engineered features that should help with learning.
        Includes type matchups, fainted counts, status, revealed moves, and estimated damage calculations.
        Handles missing information gracefully, especially during teampreview.
        """
        assert battle.player_role is not None and battle.opponent_role is not None

        features: Dict[str, float] = {}

        # PRUNED: TYPE_MATCHUP — redundant with EST_DAMAGE_* which incorporates
        # type effectiveness + STAB + stats + items
        # # Look at their type matchup against me
        # for i, mon in enumerate(fill_with_none(battle.teampreview_team, 6)):
        #     for j, opp_mon in enumerate(
        #         fill_with_none(battle.teampreview_opponent_team, 6)
        #     ):
        #         if mon is None or opp_mon is None:
        #             features["TYPE_MATCHUP:OPP_MON:" + str(j) + ":MON:" + str(i)] = -1
        #         else:
        #             multiplier = max(
        #                 opp_type.damage_multiplier(
        #                     *mon.types, type_chart=GenData.from_gen(9).type_chart
        #                 )
        #                 for opp_type in opp_mon.types
        #             )
        #             features["TYPE_MATCHUP:OPP_MON:" + str(j) + ":MON:" + str(i)] = (
        #                 multiplier
        #             )

        # Calculate total state
        num_fainted, hp_left, total_hp, num_status, num_revealed = 0, 0, 0, 0, 0
        for mon in battle.team.values():
            num_fainted += int(mon.fainted)
            hp_left += mon.current_hp
            total_hp += mon.max_hp
            num_status += int(mon.status is not None)
            num_revealed += sum(
                map(lambda x: int(x.current_pp < x.max_pp), mon.moves.values())
            )
        features["NUM_FAINTED"] = num_fainted
        features["PERC_HP_LEFT"] = hp_left * 1.0 / total_hp if total_hp > 0 else -1
        features["NUM_STATUSED"] = num_status
        features["NUM_MOVES_REVEALED"] = num_revealed
        features["NUM_MONS_REVEALED"] = sum(
            map(lambda x: x.revealed, battle.team.values())
        )

        # Now do it for the opponent
        num_fainted, hp_frac, total_hp, num_status = 0, 0., 0, 0
        for mon in fill_with_none(list(battle.opponent_team.values()), 4):
            if mon is None:
                hp_frac += 1
            else:
                num_fainted += int(mon.fainted)
                hp_frac += mon.current_hp_fraction
                num_status += int(mon.status is not None)
            total_hp += 1
        features["OPP_NUM_FAINTED"] = num_fainted
        features["OPP_PERC_HP_LEFT"] = hp_frac * 1.0 / total_hp
        features["OPP_NUM_STATUSED"] = num_status
        features["NUM_OPP_MONS_REVEALED"] = sum(
            map(lambda x: x.revealed, battle.opponent_team.values())
        )

        # Need to create temporary opp_team because teampreview doesnt have nicknames, and so normal
        # comparisons via identifiers will fail
        opp_team = {opp_mon.species: opp_mon for opp_mon in battle.opponent_team.values()}

        # Grab list of mons that are either in teampreview or in the team/opponent team
        opp = [
            (i, x if x is None or x.species not in opp_team else opp_team[x.species])
            for i, x in enumerate(
                fill_with_none(list(battle.teampreview_opponent_team), 6)
            )
        ]
        me = [
            (
                i,
                (
                    None
                    if x is None or x.identifier(battle.player_role) not in battle.team
                    else battle.team[x.identifier(battle.player_role)]
                ),
            )
            for i, x in enumerate(fill_with_none(list(battle.team.values()), 6))
        ]

        # Iterate through all mon and opp_mon combos
        for (i, opp_mon), (j, mon) in zip(opp, me):
            opp_moves = (
                fill_with_none(list(opp_mon.moves.keys()), 4)
                if opp_mon is not None
                else [None] * 4
            )
            for k, move in enumerate(opp_moves):
                dmg, ko = (-1, -1), -1
                if mon is not None and opp_mon is not None and move is not None:
                    stats_flag, teampreview_flag = False, False
                    if opp_mon.stats is None or opp_mon.stats.get("hp", None) is None:
                        opp_mon.stats = compute_stats(opp_mon, "max")  # type: ignore
                        stats_flag = True
                    if (
                        opp_mon.identifier(battle.opponent_role)
                        not in battle.opponent_team
                    ):
                        battle._opponent_team[opp_mon.identifier(battle.opponent_role)] = (
                            opp_mon
                        )
                        teampreview_flag = True

                    # This happens when we don't fill in mon's HP because it's not found in a request;
                    # it's stored in another field
                    if mon.stats["hp"] is None:
                        mon.stats["hp"] = mon.max_hp

                    dmg = calculate_damage(
                        opp_mon.identifier(battle.opponent_role),
                        mon.identifier(battle.player_role),
                        opp_mon.moves[move],
                        battle,
                        False,
                    )
                    dmg = (-1, -1) if dmg[0] is None else dmg
                    ko = int(dmg[1] > mon.current_hp)
                    if stats_flag:
                        opp_mon.stats = None
                    if teampreview_flag:
                        battle._opponent_team.pop(opp_mon.identifier(battle.opponent_role))
                features[
                    "EST_DAMAGE_MIN:OPP_MON:"
                    + str(i)
                    + ":MON:"
                    + str(j)
                    + ":MOVE:"
                    + str(k)
                ] = dmg[0]
                features[
                    "EST_DAMAGE_MAX:OPP_MON:"
                    + str(i)
                    + ":MON:"
                    + str(j)
                    + ":MOVE:"
                    + str(k)
                ] = dmg[1]
                features["KO:OPP_MON:" + str(i) + ":MON:" + str(j) + ":MOVE:" + str(k)] = (
                    ko
                )

            moves = (
                fill_with_none(list(mon.moves.keys()), 4)
                if mon is not None
                else [None] * 4
            )
            for k, move in enumerate(moves):
                dmg, ko = (-1, -1), -1
                if (
                    mon is not None
                    and opp_mon is not None
                    and move is not None
                    and mon.identifier(battle.player_role) in battle.team
                ):
                    stats_flag, teampreview_flag = False, False
                    if opp_mon.stats is None or opp_mon.stats.get("hp", None) is None:
                        opp_mon.stats = compute_stats(opp_mon, "max")  # type: ignore
                        stats_flag = True
                    if (
                        opp_mon.identifier(battle.opponent_role)
                        not in battle._opponent_team
                    ):
                        key = opp_mon.identifier(battle.opponent_role)
                        battle._opponent_team[key] = opp_mon
                        teampreview_flag = True

                    if mon.species == "tinglu":
                        pass

                    # This happens when we don't fill in mon's HP because it's not found in a request;
                    # it's stored in another field                    try:
                    if mon.stats["hp"] is None:
                        mon.stats["hp"] = mon.max_hp

                    dmg = calculate_damage(
                        mon.identifier(battle.player_role),
                        opp_mon.identifier(battle.opponent_role),
                        mon.moves[move],
                        battle,
                        False,
                    )
                    dmg = (-1, -1) if dmg[0] is None else dmg

                    # Current HP fraction is 0 if a mon isn't set out to the field
                    current_hp_fraction = (
                        1
                        if opp_mon.current_hp_fraction == 0
                        and opp_mon.status != Status.FNT
                        else opp_mon.current_hp_fraction
                    )
                    ko = int(dmg[1] > opp_mon.stats["hp"] * current_hp_fraction)  # type: ignore

                    if stats_flag:
                        opp_mon.stats = None
                    if teampreview_flag:
                        battle._opponent_team.pop(opp_mon.identifier(battle.opponent_role))
                features[
                    "EST_DAMAGE_MIN:MON:"
                    + str(j)
                    + ":OPP_MON:"
                    + str(i)
                    + ":MOVE:"
                    + str(k)
                ] = dmg[0]
                features[
                    "EST_DAMAGE_MAX:MON:"
                    + str(j)
                    + ":OPP_MON:"
                    + str(i)
                    + ":MOVE:"
                    + str(k)
                ] = dmg[1]
                features["KO:MON:" + str(j) + ":OPP_MON:" + str(i) + ":MOVE:" + str(k)] = (
                    ko
                )

        return features

    def generate_move_features(
        self, move: Optional[Move], prefix: str = ""
    ) -> Dict[str, float]:
        """
        Returns a feature dict representing a Move.

        Uses caching for static move features (everything except current_pp and used).
        This provides significant speedup since moves are embedded many times per battle.
        """
        emb: Dict[str, float] = {}

        if move is None:
            # No move - return all -1s (can't cache this with prefix)
            return self._generate_null_move_features(prefix)

        # Check cache for static features (keyed by move.id, no prefix)
        move_id = move.id
        if move_id not in self._move_cache:
            self._move_cache[move_id] = self._generate_static_move_features(move)

        # Copy cached static features with prefix
        cached = self._move_cache[move_id]
        for key, val in cached.items():
            emb[prefix + key] = val

        # Add dynamic features that depend on battle state
        emb[prefix + "current_pp"] = min(move.current_pp, 5)
        emb[prefix + "used"] = int(move.current_pp < move.max_pp)

        # Entity ID for move (integer index into nn.Embedding)
        emb[prefix + "move_id"] = self._move_to_id.get(move.id, 0)

        return emb

    def _generate_null_move_features(self, prefix: str = "") -> Dict[str, float]:
        """Generate features for a null move (all -1s)."""
        emb: Dict[str, float] = {}

        # Scalar features
        for key in [
            "accuracy", "base_power", "current_pp", "used", "damage", "drain",
            "force_switch", "heal", "is_protect_move", "is_side_protect_move",
            "min_hits", "max_hits", "priority", "recoil", "self_switch",
            "use_target_offensive", "chance"
        ]:
            emb[prefix + key] = -1

        # OHE Category
        for cat in self._knowledge["MoveCategory"]:
            emb[prefix + "OFF_CAT:" + cat.name] = -1
            emb[prefix + "DEF_CAT:" + cat.name] = -1

        # OHE Move Type
        for ptype in self._knowledge["PokemonType"]:
            if ptype not in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]:
                emb[prefix + "TYPE:" + ptype.name] = -1

        # PRUNED: Move SC, FIELD, WEATHER, EFFECT OHE — captured by move_id EntityID
        # # OHE Side Conditions
        # for sc in self._knowledge["SideCondition"]:
        #     emb[prefix + "SC:" + sc.name] = -1
        #
        # # OHE Fields
        # for field in self._knowledge["Field"]:
        #     emb[prefix + "FIELD:" + field.name] = -1
        #
        # # OHE Weathers
        # for weather in self._knowledge["Weather"]:
        #     emb[prefix + "WEATHER:" + weather.name] = -1

        # OHE Targeting Types
        for t in self._knowledge["Target"]:
            emb[prefix + "TARGET:" + t.name] = -1

        # PRUNED: Move EFFECT (volatile status) OHE — captured by move_id EntityID
        # # OHE Volatility Statuses
        # for vs in self._knowledge["Effect_VolatileStatus"]:
        #     emb[prefix + "EFFECT:" + vs.name] = -1

        # OHE Statuses
        for status in self._knowledge["Status"]:
            if status != Status.FNT:
                emb[prefix + "STATUS:" + status.name] = -1

        # Boosts and Self-boosts
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "BOOST:" + stat] = -1
            emb[prefix + "SELFBOOST:" + stat] = -1

        # Entity ID for move
        emb[prefix + "move_id"] = -1

        return emb

    def _generate_static_move_features(self, move: Move) -> Dict[str, float]:
        """
        Generate static move features (without prefix) that can be cached.
        Excludes current_pp and used which are dynamic.
        """
        emb: Dict[str, float] = {}

        # Static scalar features
        emb["accuracy"] = move.accuracy
        emb["base_power"] = move.base_power
        emb["damage"] = move.damage if isinstance(move.damage, int) else 50
        emb["drain"] = move.drain
        emb["force_switch"] = move.force_switch
        emb["heal"] = move.heal
        emb["is_protect_move"] = move.is_protect_move
        emb["is_side_protect_move"] = move.is_side_protect_move
        emb["min_hits"] = move.n_hit[0] if move.n_hit else 1
        emb["max_hits"] = move.n_hit[1] if move.n_hit else 1
        emb["priority"] = move.priority
        emb["recoil"] = move.recoil
        emb["self_switch"] = int(True if move.self_switch else False)
        emb["use_target_offensive"] = int(move.use_target_offensive)

        # Cache secondary list access
        secondary = move.secondary

        # OHE Category
        category = move.category
        defensive_category = move.defensive_category
        for cat in self._knowledge["MoveCategory"]:
            emb["OFF_CAT:" + cat.name] = int(category == cat)
            emb["DEF_CAT:" + cat.name] = int(defensive_category == cat)

        # OHE Move Type
        move_type = move.type
        for ptype in self._knowledge["PokemonType"]:
            if ptype not in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]:
                emb["TYPE:" + ptype.name] = int(ptype == move_type)

        # PRUNED: Move SC, FIELD, WEATHER OHE — captured by move_id EntityID
        # # OHE Side Conditions
        # side_condition = move.side_condition
        # for sc in self._knowledge["SideCondition"]:
        #     emb["SC:" + sc.name] = int(side_condition == sc)
        #
        # # OHE Fields
        # terrain = move.terrain
        # pseudo_weather = move.pseudo_weather
        # for field in self._knowledge["Field"]:
        #     val = -1
        #     if terrain:
        #         val = int(terrain == field)
        #     elif pseudo_weather:
        #         val = int(Field.from_showdown_message(move.entry["name"]) == field)
        #     emb["FIELD:" + field.name] = val
        #
        # # OHE Weathers
        # move_weather = move.weather
        # for weather in self._knowledge["Weather"]:
        #     emb["WEATHER:" + weather.name] = int(move_weather == weather)

        # OHE Targeting Types
        deduced_target = move.deduced_target
        for t in self._knowledge["Target"]:
            emb["TARGET:" + t.name] = int(deduced_target == t)

        # Pre-compute secondary lookups for volatile status and status
        secondary_volatile = set()
        secondary_status = set()
        secondary_boosts: Dict[str, int] = {}
        secondary_self_boosts: Dict[str, int] = {}

        if secondary:
            for info in secondary:
                vs = info.get("volatileStatus", "")
                if vs:
                    secondary_volatile.add(self._prep(vs))
                st = info.get("status", "")
                if st:
                    secondary_status.add(self._prep(st))
                if "boosts" in info:
                    for stat, val in info["boosts"].items():
                        secondary_boosts[stat] = val
                if "self" in info and "boosts" in info["self"]:
                    for stat, val in info["self"]["boosts"].items():
                        secondary_self_boosts[stat] = val

        # PRUNED: Move EFFECT (volatile status) OHE — captured by move_id EntityID
        # # OHE Volatility Statuses
        # volatile_status = move.volatile_status
        # for vs in self._knowledge["Effect_VolatileStatus"]:
        #     val = 0
        #     if vs == volatile_status:
        #         val = 1
        #     elif self._prep(vs.name) in secondary_volatile:
        #         val = 1
        #     emb["EFFECT:" + vs.name] = val

        # OHE Statuses
        move_status = move.status
        for status in self._knowledge["Status"]:
            if status == Status.FNT:
                continue
            val = 0
            if status == move_status:
                val = 1
            elif self._prep(status.name) in secondary_status:
                val = 1
            emb["STATUS:" + status.name] = val

        # Add Boosts
        boosts = move.boosts
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            val = 0
            if boosts and stat in boosts:
                val = boosts[stat]
            elif stat in secondary_boosts:
                val = secondary_boosts[stat]
            emb["BOOST:" + stat] = val

        # Add Self-Boosts
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            val = 0
            if boosts and stat in boosts:
                val = boosts[stat]
            elif stat in secondary_self_boosts:
                val = secondary_self_boosts[stat]
            emb["SELFBOOST:" + stat] = val

        # Chance of secondary effect
        val = 0
        if secondary:
            val = max(info.get("chance", 0) for info in secondary)
        emb["chance"] = val

        return emb

    def generate_pokemon_features(
        self, mon: Optional[Pokemon], battle: DoubleBattle, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Returns a Dict of features representing the pokemon
        """

        emb: Dict[str, float] = {}

        # Add moves to feature dict (and account for the fact that the mon might have <4 moves)
        moves = list(mon.moves.values()) if mon else []
        available_moves = (
            mon.available_moves_from_request(battle.last_request)
            if mon and "moves" in battle.last_request
            else []
        )
        for i, move in enumerate((moves + [None, None, None, None])[:4]):
            move_prefix = prefix + "MOVE:" + str(i) + ":"
            emb.update(self.generate_move_features(move, move_prefix))

            # Record whether a move is available
            available = -1
            if len(available_moves) > 0:
                available = 1 if move in available_moves else 0
            emb[move_prefix + "available"] = available

        # Entity ID for ability (integer index into nn.Embedding)
        emb[prefix + "ability_id"] = (
            self._ability_to_id.get(mon.ability, 0) if mon and mon.ability else -1
        )

        # Entity ID for item (integer index into nn.Embedding)
        emb[prefix + "item_id"] = ITEM_TO_ID.get(mon.item, 0) if mon and mon.item else -1

        # Entity ID for species (integer index into nn.Embedding)
        emb[prefix + "species_id"] = (
            self._species_to_id.get(mon.species, 0) if mon else -1
        )

        # Add various relevant fields for mons
        emb[prefix + "current_hp_fraction"] = mon.current_hp_fraction if mon else -1
        # PRUNED: level is always 50 in VGC — zero information
        # emb[prefix + "level"] = mon.level if mon else -1
        emb[prefix + "weight"] = mon.weight if mon else -1
        emb[prefix + "is_terastallized"] = mon.is_terastallized if mon else -1

        # Add stats
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            val = -1
            if mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                val = mon.stats[stat]  # type:ignore
            emb[prefix + "STAT:" + stat] = val if val is not None else -1

        # One-hot encode boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            boost_val = mon.boosts[stat] if mon else None
            for b in BOOST_RANGE:
                emb[prefix + f"BOOST:{stat}:{b}"] = (
                    -1 if boost_val is None else int(boost_val == b)
                )

        # OHE status
        for status in self._knowledge["Status"]:
            emb[prefix + "STATUS: " + status.name] = (
                int(mon.status == status) if mon else -1
            )

        # OHE types
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]:
                continue
            emb[prefix + "TYPE:" + ptype.name] = (
                int(mon.type_1 == ptype or mon.type_2 == ptype) if mon else -1
            )

        # OHE TeraType
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [PokemonType.THREE_QUESTION_MARKS]:
                continue
            emb[prefix + "TERA_TYPE:" + ptype.name] = (
                int(ptype == mon.tera_type) if mon else -1
            )

        # OHE which switch the pokemon is
        for i, m in enumerate(fill_with_none(list(battle.team.values()), 6)):
            emb[prefix + "SWITCH_NUM:" + str(i)] = (
                -1
                if mon is None
                else int(
                    m is not None
                    and battle.player_role is not None
                    and mon.identifier(battle.player_role)
                    == m.identifier(battle.player_role)
                )
            )

        # Get pokemon battle attributes
        sent = (
            -1
            if mon is None
            else int(
                battle.player_role is not None
                and mon.identifier(battle.player_role) in battle.team
                and not battle.teampreview
            )
        )
        emb[prefix + "sent"] = sent
        active_names = list(map(lambda x: x.name if x else None, battle.active_pokemon))
        name = mon.name if mon else ""
        emb[prefix + "active"] = -1 if mon is None else int(name in active_names)

        trapped, force_switch = -1, -1
        if name in active_names:
            trapped = battle.trapped[active_names.index(name)]
            force_switch = battle.force_switch[active_names.index(name)]

        emb[prefix + "trapped"] = -1 if mon is None else int(trapped)
        emb[prefix + "force_switch"] = -1 if mon is None else int(force_switch)

        # To maintain representation of public belief state
        mon = battle.team.get(
            mon.identifier(battle.player_role) if mon and battle.player_role else "", mon
        )
        emb[prefix + "revealed"] = int(mon.revealed if mon else 0) if sent == 1 else -1

        # Record whether mon is an available switch for active_pokemon1 and 2
        emb[prefix + "is_available_to_switch"] = (
            -1
            if mon is None
            else int(
                mon.name
                in map(lambda x: x.name if x else None, battle.available_switches[0])
                or mon.name
                in map(lambda x: x.name if x else None, battle.available_switches[1])
            )
        )

        return emb

    def generate_opponent_pokemon_features(
        self,
        mon: Optional[Pokemon],
        battle: DoubleBattle,
        inference_flags: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Returns a Dict of features representing the opponents pokemon
        """
        emb: Dict[str, float] = {}

        # Add moves to feature dict (and account for the fact that the mon might have <4 moves)
        moves = list(mon.moves.values()) if mon else []
        for i, move in enumerate((moves + [None, None, None, None])[:4]):
            move_prefix = prefix + "MOVE:" + str(i) + ":"
            emb.update(self.generate_move_features(move, move_prefix))

        # Entity ID for ability (integer index into nn.Embedding)
        # For opponent, use known ability if available; otherwise 0 (unknown)
        if not mon:
            emb[prefix + "ability_id"] = -1
        elif mon.ability:
            emb[prefix + "ability_id"] = self._ability_to_id.get(mon.ability, 0)
        else:
            emb[prefix + "ability_id"] = 0  # Unknown ability

        # Entity ID for item (integer index into nn.Embedding)
        if not mon:
            emb[prefix + "item_id"] = -1
        elif mon.item:
            emb[prefix + "item_id"] = ITEM_TO_ID.get(mon.item, 0)
        elif inference_flags and inference_flags.get("item"):
            emb[prefix + "item_id"] = ITEM_TO_ID.get(inference_flags["item"], 0)
        else:
            emb[prefix + "item_id"] = 0  # Unknown item

        # Entity ID for species (integer index into nn.Embedding)
        emb[prefix + "species_id"] = (
            self._species_to_id.get(mon.species, 0) if mon else -1
        )

        # Add several other fields
        emb[prefix + "current_hp_fraction"] = mon.current_hp_fraction if mon else -1
        # PRUNED: level is always 50 in VGC — zero information
        # emb[prefix + "level"] = mon.level if mon else -1
        emb[prefix + "weight"] = mon.weight if mon else -1
        emb[prefix + "is_terastallized"] = int(mon.is_terastallized) if mon else -1

        # Add stats by calculating
        stats = ["hp", "atk", "def", "spa", "spd", "spe"]
        minstats, maxstats = [-1] * 6, [-1] * 6

        if mon:
            minstats = list(compute_stats(mon, "min").values())
            maxstats = list(compute_stats(mon, "max").values())

        if self._omniscient:
            for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
                val = -1
                if mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                    val = mon.stats[stat]  # type:ignore

                # Do both to preserve embedding size
                emb[prefix + "STAT_MIN:" + stat] = val if val is not None else -1
                emb[prefix + "STAT_MAX:" + stat] = val if val is not None else -1
        else:
            for stat, minstat, maxstat in zip(stats, minstats, maxstats):
                if mon and inference_flags and stat in inference_flags:
                    minstat, maxstat = inference_flags[stat][0], inference_flags[stat][1]
                elif mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                    minstat, maxstat = mon.stats[stat], mon.stats[stat]  # type: ignore
                emb[prefix + "STAT_MIN:" + stat] = (
                    minstat if mon and minstat is not None else -1
                )
                emb[prefix + "STAT_MAX:" + stat] = (
                    maxstat if mon and maxstat is not None else -1
                )

        # One-hot encode boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            boost_val = mon.boosts[stat] if mon else None
            for b in BOOST_RANGE:
                emb[prefix + f"BOOST:{stat}:{b}"] = (
                    -1 if boost_val is None else int(boost_val == b)
                )

        # OHE status
        for status in self._knowledge["Status"]:
            emb[prefix + "STATUS: " + status.name] = (
                int(mon.status == status) if mon else -1
            )

        # OHE types
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]:
                continue
            emb[prefix + "TYPE:" + ptype.name] = (
                int(mon.type_1 == ptype or mon.type_2 == ptype) if mon else -1
            )

        # OHE TeraType
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [PokemonType.THREE_QUESTION_MARKS]:
                continue
            val = -1
            if mon and mon.tera_type is not None:
                val = int(ptype == mon.tera_type)

            emb[prefix + "TERA_TYPE:" + ptype.name] = val

        # Generate features about this pokemon in the context of the battle
        emb[prefix + "sent"] = (
            -1
            if mon is None
            else int(  # battle.opponent_team is basically teampreview_team for teampreview
                battle.opponent_role is not None
                and mon.identifier(battle.opponent_role) in battle.opponent_team
                and not battle.teampreview
            )
        )

        emb[prefix + "active"] = (
            -1
            if mon is None
            else int(
                mon.name
                in map(lambda x: x.name if x else None, battle.opponent_active_pokemon)
            )
        )

        return emb

    def generate_battle_features(self, battle: DoubleBattle) -> Dict[str, float]:
        emb: Dict[str, float] = {}

        # Add Fields
        for field in self._knowledge["Field"]:
            emb["FIELD:" + field.name] = int(field in battle.fields)

        # Add field duration features (normalized remaining turns)
        for field in self._knowledge["Field"]:
            duration = FIELD_DURATIONS.get(field, 5)
            if field in battle.fields:
                set_turn = battle.fields[field]
                remaining = max(0, duration - (battle.turn - set_turn))
                emb["FIELD:" + field.name + ":remaining"] = remaining / duration
            else:
                emb["FIELD:" + field.name + ":remaining"] = 0

        # Add Side Conditions
        for sc in self._knowledge["SideCondition"]:
            emb["SIDE_CONDITION:" + sc.name] = int(sc in battle.side_conditions)

        for sc in self._knowledge["SideCondition"]:
            emb["OPP_SIDE_CONDITION:" + sc.name] = int(
                sc in battle.opponent_side_conditions
            )

        # Add side condition duration features for duration-based (non-stackable) conditions
        for sc in SIDE_CONDITION_DURATIONS:
            duration = SIDE_CONDITION_DURATIONS[sc]
            # Player's side
            if sc in battle.side_conditions:
                set_turn = battle.side_conditions[sc]
                remaining = max(0, duration - (battle.turn - set_turn))
                emb["SIDE_CONDITION:" + sc.name + ":remaining"] = remaining / duration
            else:
                emb["SIDE_CONDITION:" + sc.name + ":remaining"] = 0
            # Opponent's side
            if sc in battle.opponent_side_conditions:
                set_turn = battle.opponent_side_conditions[sc]
                remaining = max(0, duration - (battle.turn - set_turn))
                emb["OPP_SIDE_CONDITION:" + sc.name + ":remaining"] = remaining / duration
            else:
                emb["OPP_SIDE_CONDITION:" + sc.name + ":remaining"] = 0

        # Add Weathers
        for weather in self._knowledge["Weather"]:
            emb["WEATHER:" + weather.name] = int(weather in battle.weather)

        # Add weather duration features
        for weather in self._knowledge["Weather"]:
            if weather in battle.weather:
                set_turn = battle.weather[weather]
                remaining = max(0, WEATHER_DURATION - (battle.turn - set_turn))
                emb["WEATHER:" + weather.name + ":remaining"] = remaining / WEATHER_DURATION
            else:
                emb["WEATHER:" + weather.name + ":remaining"] = 0

        # Add Formats
        for frmt in self._knowledge["Format"]:
            emb["FORMAT:" + frmt] = int(frmt == battle.format)

        emb["teampreview"] = int(battle.teampreview)
        emb["p1rating"] = battle.rating if battle.rating else -1
        emb["p2rating"] = battle.opponent_rating if battle.opponent_rating else -1
        emb["turn"] = battle.turn

        return emb

    def embed(
        self, battle: DoubleBattle, bi: Optional[BattleInference] = None
    ) -> Dict[str, float]:
        """
        Returns a list of integers representing the state of the battle, at the beginning
        of the specified turn. It is from the perspective of the player whose turn it is.
        """
        emb: Dict[str, float] = {}

        # Add each of our mons as features. We want to add even our teampreview pokemon because
        # our opponent may make moves dependent on this information
        for i, mon in enumerate(fill_with_none(battle.teampreview_team, 6)):
            prefix = "MON:" + str(i) + ":"

            # We should featurize the battle copy of the mon, otherwise, we featurize the teampreview mon
            if (
                battle.player_role
                and mon is not None
                and not battle.teampreview
                and mon.identifier(battle.player_role) in battle.team
            ):
                mon = battle.team[mon.identifier(battle.player_role)]

            emb.update(self.generate_pokemon_features(mon, battle, prefix))

        # Featurize each opponent mon
        species = {
            x.species: x for x in battle.opponent_team.values()
        }  # teampreview mons don't have identifiers, so we have to look at species
        for i, mon in enumerate(fill_with_none(battle.teampreview_opponent_team, 6)):
            prefix = "OPP_MON:" + str(i) + ":"

            mon = species.get(mon.species if mon else "", mon)

            flags = (
                bi.get_flags(mon.identifier(battle.opponent_role))
                if bi and mon and battle.opponent_role is not None
                else None
            )

            emb.update(
                self.generate_opponent_pokemon_features(
                    mon, battle, inference_flags=flags, prefix=prefix
                )
            )

        # Add battle features
        emb.update(self.generate_battle_features(battle))

        # Convert embedding to the specified type
        if self._feature_set == self.SIMPLE:
            emb = self._simplify_features(emb)
        elif self._feature_set == self.FULL:
            emb.update(self.generate_feature_engineered_features(battle, bi))
            emb.update(self.generate_transition_features(battle))

        return emb

    def _ident_to_slot(self, ident: str, battle: DoubleBattle) -> Optional[str]:
        """Map a Showdown identifier like 'p1a: Incineroar' to a slot prefix string.

        Returns one of "MY:0:", "MY:1:", "OPP:0:", "OPP:1:" or None if unrecognized.
        """
        if len(ident) < 3:
            return None
        player = ident[:2]  # "p1" or "p2"
        slot_letter = ident[2]  # "a" or "b"
        slot_idx = 0 if slot_letter == "a" else 1

        if player == battle.player_role:
            return f"MY:{slot_idx}:"
        elif player == battle.opponent_role:
            return f"OPP:{slot_idx}:"
        return None

    def generate_transition_features(
        self, battle: DoubleBattle
    ) -> Dict[str, float]:
        """Extract features describing what happened during the previous turn.

        Parses ``battle.observations[prev_turn].events`` to produce per-slot features
        (move, switch, protect, crit, super-effective, resisted, miss, fail) and global
        features (move order, faints).  All values default to -1 when no data is available
        (teampreview, turn 0/1, or missing observations).
        """
        features: Dict[str, float] = {}

        # Initialise every feature to -1 (unknown / not applicable)
        slots = ["MY:0:", "MY:1:", "OPP:0:", "OPP:1:"]
        per_slot_feats = [
            "used_move", "used_switch", "used_protect",
            "was_crit", "hit_super_effective", "hit_resisted",
            "move_missed", "move_failed",
        ]
        for slot in slots:
            for feat in per_slot_feats:
                features[f"TRANSITION:{slot}{feat}"] = -1

        features["TRANSITION:my_slot0_moved_first"] = -1
        features["TRANSITION:my_slot1_moved_first"] = -1
        features["TRANSITION:any_faint_last_turn"] = -1
        features["TRANSITION:my_faint_last_turn"] = -1
        features["TRANSITION:opp_faint_last_turn"] = -1

        # Need a previous turn with observations to extract anything
        prev_turn = battle.turn - 1
        if (
            prev_turn < 1
            or battle.player_role is None
            or not hasattr(battle, "observations")
            or prev_turn not in battle.observations
        ):
            return features

        observation = battle.observations[prev_turn]
        events = observation.events

        # Zero out the counters that will be incremented
        features["TRANSITION:my_faint_last_turn"] = 0
        features["TRANSITION:opp_faint_last_turn"] = 0
        features["TRANSITION:any_faint_last_turn"] = 0

        # Zero out all per-slot features (we now have data to fill them)
        for slot in slots:
            for feat in per_slot_feats:
                features[f"TRANSITION:{slot}{feat}"] = 0

        # Track move order: record (order_index, slot_prefix) for each move event
        move_events_order: List[str] = []  # slot prefixes in order of execution
        last_attacker_slot: Optional[str] = None

        for event in events:
            if len(event) < 2:
                continue
            msg_type = event[1]

            if msg_type == "move":
                mon_ident = event[2]
                slot = self._ident_to_slot(mon_ident, battle)
                if slot is not None:
                    features[f"TRANSITION:{slot}used_move"] = 1
                    last_attacker_slot = slot

                    # Check for protect-type moves
                    if len(event) > 3:
                        move_name = event[3].lower().replace(" ", "").replace("-", "")
                        if move_name in PROTECT_MOVES:
                            features[f"TRANSITION:{slot}used_protect"] = 1

                    move_events_order.append(slot)

            elif msg_type == "switch":
                mon_ident = event[2]
                slot = self._ident_to_slot(mon_ident, battle)
                if slot is not None:
                    features[f"TRANSITION:{slot}used_switch"] = 1

            elif msg_type == "-crit":
                # -crit refers to the target, but we attribute it to the attacker
                if last_attacker_slot is not None:
                    features[f"TRANSITION:{last_attacker_slot}was_crit"] = 1

            elif msg_type == "-supereffective":
                if last_attacker_slot is not None:
                    features[f"TRANSITION:{last_attacker_slot}hit_super_effective"] = 1

            elif msg_type == "-resisted":
                if last_attacker_slot is not None:
                    features[f"TRANSITION:{last_attacker_slot}hit_resisted"] = 1

            elif msg_type == "-miss":
                # -miss event: event[2] is the attacker
                if len(event) > 2:
                    attacker_slot = self._ident_to_slot(event[2], battle)
                    if attacker_slot is not None:
                        features[f"TRANSITION:{attacker_slot}move_missed"] = 1

            elif msg_type == "-fail":
                # -fail can reference the mon whose move failed (event[2])
                if len(event) > 2:
                    fail_slot = self._ident_to_slot(event[2], battle)
                    if fail_slot is not None:
                        features[f"TRANSITION:{fail_slot}move_failed"] = 1

            elif msg_type == "faint":
                fainted_slot = self._ident_to_slot(event[2], battle)
                features["TRANSITION:any_faint_last_turn"] = 1
                if fainted_slot is not None and fainted_slot.startswith("MY:"):
                    features["TRANSITION:my_faint_last_turn"] += 1
                elif fainted_slot is not None and fainted_slot.startswith("OPP:"):
                    features["TRANSITION:opp_faint_last_turn"] += 1

        # Determine move order: did each of my slots move before any opponent?
        my_slots_seen: Dict[str, int] = {}
        opp_first_move_idx: Optional[int] = None
        for idx, slot in enumerate(move_events_order):
            if slot.startswith("OPP:") and opp_first_move_idx is None:
                opp_first_move_idx = idx
            if slot.startswith("MY:") and slot not in my_slots_seen:
                my_slots_seen[slot] = idx

        if "MY:0:" in my_slots_seen:
            if opp_first_move_idx is not None:
                features["TRANSITION:my_slot0_moved_first"] = int(
                    my_slots_seen["MY:0:"] < opp_first_move_idx
                )
            else:
                # No opponent moved (both fainted or switched)
                features["TRANSITION:my_slot0_moved_first"] = 1

        if "MY:1:" in my_slots_seen:
            if opp_first_move_idx is not None:
                features["TRANSITION:my_slot1_moved_first"] = int(
                    my_slots_seen["MY:1:"] < opp_first_move_idx
                )
            else:
                features["TRANSITION:my_slot1_moved_first"] = 1

        return features


def fill_with_none(to_fill: List[Any], n: int) -> List[Any | None]:
    return to_fill + [None] * (n - len(to_fill))


def compute_stats(mon: Pokemon, type="max") -> Dict[str, int]:
    assert type in ["max", "min"]
    stat_types = ["hp", "atk", "def", "spa", "spd", "spe"]

    if type == "min":
        stats = {}
        for k, v in zip(
            stat_types,
            compute_raw_stats(
                mon.species, [0] * 6, [0] * 6, mon.level, "serious", GenData.from_gen(int(mon.gen))
            ),
        ):
            stats[k] = int(0.9 * v) if k != "hp" else v
        return stats
    else:
        stats = {}
        for k, v in zip(
            stat_types,
            compute_raw_stats(
                mon.species, [252] * 6, [31] * 6, mon.level, "serious", GenData.from_gen(int(mon.gen))
            ),
        ):
            stats[k] = int(1.1 * v) if k != "hp" else v
        return stats


BASIC_FEATURES = {
    "base_power",
    "current_pp",
    "TYPE:",
    "current_hp_fraction",
    "STATUS:FAINTEDSTAT:spe",
    "active",
    "sent",
    "STAT_MAX:spe",
    "teampreview",
    "turn",
    "bias",
    "id",
}

# top effects
TRACKED_EFFECTS = {
    Effect.PROTECT,
    Effect.RAGE_POWDER,
    Effect.FLINCH,
    Effect.HELPING_HAND,
    Effect.FOLLOW_ME,
    Effect.ENCORE,
    Effect.SUBSTITUTE,
    Effect.GLAIVE_RUSH,
    Effect.YAWN,
    Effect.LEECH_SEED,
    Effect.PHANTOM_FORCE,
    Effect.ROOST,
    Effect.SPIKY_SHIELD,
    Effect.PERISH0,
    Effect.PERISH1,
    Effect.PERISH2,
    Effect.PERISH3,
    Effect.SALT_CURE,
    Effect.DISABLE,
    Effect.QUICK_GUARD,
    Effect.SAND_TOMB,
    Effect.AFTER_YOU,
    Effect.INSTRUCT,
    Effect.IMPRISON,
    Effect.SAFEGUARD,
    Effect.QUASH,
    Effect.WHIRLPOOL,
    Effect.ENDURE,
    Effect.DESTINY_BOND,
    Effect.BANEFUL_BUNKER,
    Effect.INFESTATION,
    Effect.FOCUS_ENERGY,
    Effect.CONFUSION,
    Effect.PROTOSYNTHESIS,
    Effect.PROTOSYNTHESISATK,
    Effect.PROTOSYNTHESISDEF,
    Effect.PROTOSYNTHESISSPA,
    Effect.PROTOSYNTHESISSPD,
    Effect.PROTOSYNTHESISSPE,
    Effect.QUARK_DRIVE,
    Effect.QUARKDRIVEATK,
    Effect.QUARKDRIVEDEF,
    Effect.QUARKDRIVESPA,
    Effect.QUARKDRIVESPD,
    Effect.QUARKDRIVESPE,
    Effect.ZERO_TO_HERO,
    Effect.TAUNT,
    Effect.TOXIC_DEBRIS,
    Effect.COMMANDER,
    Effect.TYPECHANGE,
    Effect.SUPREME_OVERLORD,
}

TRACKED_TARGET_TYPES = {
    Target.NORMAL,
    Target.ALL_ADJACENT_FOES,
    Target.SELF,
    Target.ANY,
    Target.ADJACENT_ALLY,
    Target.ALLY_SIDE,
    Target.ALL_ADJACENT,
    Target.ALL,
    Target.ADJACENT_FOE,
}

TRACKED_SIDE_CONDITIONS = {
    SideCondition.AURORA_VEIL,
    SideCondition.LIGHT_SCREEN,
    SideCondition.MIST,
    SideCondition.QUICK_GUARD,
    SideCondition.REFLECT,
    SideCondition.SAFEGUARD,
    SideCondition.SPIKES,
    SideCondition.STEALTH_ROCK,
    SideCondition.STICKY_WEB,
    SideCondition.TAILWIND,
    SideCondition.TOXIC_SPIKES,
    SideCondition.WIDE_GUARD,
}

TRACKED_ITEMS = {
    "focussash",
    "assaultvest",
    "boosterenergy",
    "sitrusberry",
    "safetygoggles",
    "choicespecs",
    "leftovers",
    "lifeorb",
    "choicescarf",
    "choiceband",
    "lumberry",
    "clearamulet",
    "mysticwater",
    "rockyhelmet",
    "covertcloak",
    "lightclay",
    "eviolite",
    "loadeddice",
    "weaknesspolicy",
    "charcoal",
    "aguavberry",
    "widelens",
    "mentalherb",
    "psychicseed",
    "blackglasses",
    "figyberry",
    "ejectpack",
    "throatspray",
    "wikiberry",
    "iapapaberry",
    "damprock",
    "mirrorherb",
    "occaberry",
    "sharpbeak",
    "roseliberry",
    "expertbelt",
    "flameorb",
    "airballoon",
    "magoberry",
    "ejectbutton",
}

TRACKED_FORMATS = {
    "gen6doubblesou",
    "gen9vgc2025regulationi",
    "gen9vgc2024regulationf",
    "gen9vgc2024regulationg",
    "gen9vgc2024regulationh",
    "gen9vgc2023regc",
    "gen9vgc2023regulationb",
    "gen9vgc2023regulationa",
}

TRACKED_FIELDS = {
    Field.GRASSY_TERRAIN,
    Field.ELECTRIC_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.TRICK_ROOM,
    Field.PSYCHIC_TERRAIN,
    Field.GRAVITY,
}

TRACKED_WEATHERS = {Weather.SUNNYDAY, Weather.SANDSTORM, Weather.RAINDANCE, Weather.SNOW}

# Duration constants for weather/field/side condition remaining-turn encoding
WEATHER_DURATION = 5
FIELD_DURATIONS = {
    Field.TRICK_ROOM: 5,
    Field.GRASSY_TERRAIN: 5,
    Field.ELECTRIC_TERRAIN: 5,
    Field.MISTY_TERRAIN: 5,
    Field.PSYCHIC_TERRAIN: 5,
    Field.GRAVITY: 5,
}
SIDE_CONDITION_DURATIONS = {
    SideCondition.TAILWIND: 4,
    SideCondition.REFLECT: 5,
    SideCondition.LIGHT_SCREEN: 5,
    SideCondition.AURORA_VEIL: 5,
}

# Protect-type moves for transition feature detection (lowered, no spaces/hyphens)
PROTECT_MOVES = {
    "protect",
    "detect",
    "banefulbunker",
    "kingsshield",
    "spikyshield",
    "obstruct",
    "silktrap",
    "burningbulwark",
    "wideguard",
    "quickguard",
    "matblock",
    "endure",
}

# Boost range for one-hot encoding: -6 to +6 inclusive = 13 bins
BOOST_RANGE = list(range(-6, 7))

# Entity ID mappings: 0 = unknown/unseen, 1-N = known entities
# Sorted for deterministic ordering
ITEM_TO_ID = {item: i + 1 for i, item in enumerate(sorted(TRACKED_ITEMS))}


def build_ability_to_id(format_str: str) -> Dict[str, int]:
    """Build ability ID mapping from poke-env GenData for the given format.

    IDs are 1-indexed; 0 is reserved for unknown/unseen abilities.
    """
    pokedex = GenData.from_gen(int(format_str[3])).pokedex
    abilities: set[str] = set()

    for entry in pokedex.values():
        for ability in entry.get("abilities", {}).values():
            if ability:
                abilities.add(to_id_str(str(ability)))

    return {ability: i + 1 for i, ability in enumerate(sorted(abilities))}


def build_species_to_id(format_str: str) -> Dict[str, int]:
    """Build species ID mapping from poke-env GenData for the given format.

    IDs are 1-indexed; 0 is reserved for unknown/unseen species.
    """
    pokedex = GenData.from_gen(int(format_str[3])).pokedex
    species = sorted(pokedex.keys())
    return {s: i + 1 for i, s in enumerate(species)}


def build_move_to_id(format_str: str) -> Dict[str, int]:
    """Build move ID mapping from poke-env GenData for the given format.

    IDs are 1-indexed; 0 is reserved for unknown/unseen moves.
    """
    movedex = GenData.from_gen(int(format_str[3])).moves
    moves = sorted(movedex.keys())
    return {m: i + 1 for i, m in enumerate(moves)}


DEFAULT_FORMAT = "gen9vgc2023regc"
ABILITY_TO_ID = build_ability_to_id(DEFAULT_FORMAT)
SPECIES_TO_ID = build_species_to_id(DEFAULT_FORMAT)
MOVE_TO_ID = build_move_to_id(DEFAULT_FORMAT)

# Vocabulary sizes (including the 0 = unknown token)
NUM_ABILITIES = len(ABILITY_TO_ID) + 1  # +1 for unknown (index 0)
NUM_ITEMS = len(TRACKED_ITEMS) + 1  # +1 for unknown (index 0)
NUM_SPECIES = len(SPECIES_TO_ID) + 1  # +1 for unknown (index 0)
NUM_MOVES = len(MOVE_TO_ID) + 1  # +1 for unknown (index 0)
