# -*- coding: utf-8 -*-
"""This module defines a class that Embeds objects
"""

import logging
from typing import Any, Dict, List, Optional

from poke_env.calc import calculate_damage
from poke_env.data import GenData
from poke_env.environment import (
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
from poke_env.stats import compute_raw_stats

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.utils.inference_utils import get_showdown_identifier


class Embedder:

    SIMPLE = "simple"
    RAW = "raw"
    FULL = "full"

    def __init__(
        self, format="gen9vgc2023regulationc", feature_set: str = "raw", omniscient=False
    ):
        self._knowledge: Dict[str, Any] = {}
        self._format: str = format
        self._omniscient: bool = omniscient

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

        self._knowledge["Pokemon"] = set(GenData.from_gen(format[3]).pokedex.keys())
        self._knowledge["Effect_VolatileStatus"] = TRACKED_EFFECTS
        self._knowledge["Item"] = TRACKED_ITEMS
        self._knowledge["Target"] = TRACKED_TARGET_TYPES
        self._knowledge["Format"] = TRACKED_FORMATS
        self._knowledge["SideCondition"] = TRACKED_SIDE_CONDITIONS
        self._knowledge["Weather"] = TRACKED_WEATHERS
        self._knowledge["Field"] = TRACKED_FIELDS
        self._knowledge["Ability"] = TRACKED_ABILITIES

        # Generate embedding size and feature names programatically upon instanciation
        # because I muck with Embedder too much to hard code it
        dummy_features = self._generate_dummy_features()
        self._embedding_size = len(dummy_features)
        self._feature_names = list(dummy_features.keys())

    @staticmethod
    def _prep(string) -> str:
        """
        Used to convert names of various enumerations into their string variants
        """
        return string.lower().replace("_", " ")

    def _generate_dummy_features(self) -> Dict[str, float]:
        dummy_battle = DoubleBattle(
            "tag", "elitefurretai", logging.Logger("example"), gen=int(self._format[3])
        )
        dummy_battle._format = self._format
        dummy_battle.player_role = "p1"
        return self.featurize_double_battle(dummy_battle)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def feature_set(self) -> str:
        return self._feature_set

    @property
    def format(self) -> str:
        return self._format

    @property
    def omniscient(self) -> bool:
        return self._omniscient

    def feature_dict_to_vector(self, features: Dict[str, Any]) -> List[float]:
        """
        Converts a feature dictionary returned by this class into a vector that
        can be used as input into a network
        """
        vec = []
        for key in sorted(features.keys()):
            vec.append(float(features[key]))
        return vec

    def _simplify_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a normal feature dictionary into a heavily simplified one.
        Features were chosen to be able to roughly mimic max damage player,
        but should be enough to outperform it, if correct. Primarily used for testing.
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
        """Adds supplementary engineered features that should help with learning"""
        assert battle.player_role is not None and battle.opponent_role is not None

        features: Dict[str, float] = {}

        # Look at their type matchup against me
        for i, mon in enumerate(fill_with_none(battle.teampreview_team, 6)):
            for j, opp_mon in enumerate(
                fill_with_none(battle.teampreview_opponent_team, 6)
            ):
                if mon is None or opp_mon is None:
                    features["TYPE_MATCHUP:OPP_MON:" + str(j) + ":MON:" + str(i)] = -1
                else:
                    multiplier = max(
                        opp_type.damage_multiplier(
                            *mon.types, type_chart=GenData.from_gen(9).type_chart
                        )
                        for opp_type in opp_mon.types
                    )
                    features["TYPE_MATCHUP:OPP_MON:" + str(j) + ":MON:" + str(i)] = (
                        multiplier
                    )

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
        num_fainted, hp_frac, total_hp, num_status = 0, 0, 0, 0
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
        features["NUM_OPP_MONS_REVEALED"] = len(battle.opponent_team)

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
            for i, x in enumerate(fill_with_none(list(battle.team.values()), 4))
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
                    # TODO: guess opponent stats using MetaDB cache; I wonder if I can "warmstart" MetaDB by creating temp tables of possible mons
                    # need these if we don't have an opponent's stats, or if theyre not in the team object. We need to insert them cuz the calculate_damage
                    # function relies on the pokemon being in opponent_team via the get_pokemon function
                    stats_flag, teampreview_flag = False, False
                    if opp_mon.stats is None or opp_mon.stats.get("hp", None) is None:
                        opp_mon.stats = compute_stats(opp_mon, "max")  # type: ignore
                        stats_flag = True
                    if (
                        opp_mon.identifier(battle.opponent_role)
                        not in battle.opponent_team
                    ):
                        battle.opponent_team[opp_mon.identifier(battle.opponent_role)] = (
                            opp_mon
                        )
                        teampreview_flag = True

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
                        battle.opponent_team.pop(opp_mon.identifier(battle.opponent_role))
                features[
                    "EST_DAMAGE:OPP_MON:" + str(i) + ":MON:" + str(j) + ":MOVE:" + str(k)
                ] = (sum(dmg) / 2)
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
                        not in battle.opponent_team
                    ):
                        battle.opponent_team[opp_mon.identifier(battle.opponent_role)] = (
                            opp_mon
                        )
                        teampreview_flag = True

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
                        battle.opponent_team.pop(opp_mon.identifier(battle.opponent_role))
                features[
                    "EST_DAMAGE:MON:" + str(j) + ":OPP_MON:" + str(i) + ":MOVE:" + str(k)
                ] = (sum(dmg) / 2)
                features["KO:MON:" + str(j) + ":OPP_MON:" + str(i) + ":MOVE:" + str(k)] = (
                    ko
                )

        return features

    def featurize_move(self, move: Optional[Move], prefix: str = "") -> Dict[str, float]:
        """
        Returns a feature dict representing a Move
        """

        emb: Dict[str, float] = {}

        emb[prefix + "accuracy"] = move.accuracy if move else -1
        emb[prefix + "base_power"] = move.base_power if move else -1
        emb[prefix + "current_pp"] = min(move.current_pp, 5) if move else -1

        # To maintain representation of public belief state
        emb[prefix + "used"] = int(move.current_pp < move.max_pp) if move else -1

        if move:
            emb[prefix + "damage"] = move.damage if isinstance(move.damage, int) else 50
        else:
            emb[prefix + "damage"] = -1
        emb[prefix + "drain"] = move.drain if move else -1
        emb[prefix + "force_switch"] = move.force_switch if move else -1
        emb[prefix + "heal"] = move.heal if move else -1
        emb[prefix + "is_protect_move"] = move.is_protect_move if move else -1
        emb[prefix + "is_side_protect_move"] = move.is_side_protect_move if move else -1
        if move:
            emb[prefix + "min_hits"] = move.n_hit[0] if move.n_hit else 1
            emb[prefix + "max_hits"] = move.n_hit[1] if move.n_hit else 1
        else:
            emb[prefix + "min_hits"] = -1
            emb[prefix + "max_hits"] = -1
        emb[prefix + "priority"] = move.priority if move else -1
        emb[prefix + "recoil"] = move.recoil if move else -1
        emb[prefix + "self_switch"] = (
            int(True if move.self_switch else False) if move else -1
        )
        emb[prefix + "use_target_offensive"] = (
            int(move.use_target_offensive) if move else -1
        )

        # OHE Category
        for cat in self._knowledge["MoveCategory"]:
            emb[prefix + "OFF_CAT:" + cat.name] = int(move.category == cat) if move else -1

        # OHE Defensive Category
        for cat in self._knowledge["MoveCategory"]:
            emb[prefix + "OFF_CAT:" + cat.name] = (
                int(move.defensive_category == cat) if move else -1
            )

        # OHE Move Type
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [
                PokemonType.THREE_QUESTION_MARKS,
                PokemonType.STELLAR,
            ]:  # not eligible for moves
                continue
            emb[prefix + "TYPE:" + ptype.name] = int(ptype == move.type) if move else -1

        # OHE Side Conditions
        for sc in self._knowledge["SideCondition"]:
            emb[prefix + "SC:" + sc.name] = int(move.side_condition == sc) if move else -1

        # OHE Targeting Types
        for t in self._knowledge["Target"]:
            emb[prefix + "TARGET:" + t.name] = (
                int(move.deduced_target == t) if move else -1
            )

        # OHE Volatility Statuses
        for vs in self._knowledge["Effect_VolatileStatus"]:
            val = 0
            if not move:
                val = -1
            elif vs == move.volatile_status:
                val = 1
            elif move.secondary and self._prep(vs.name) in list(
                map(lambda x: self._prep(x.get("volatileStatus", "")), move.secondary)
            ):
                val = 1
            emb[prefix + "EFFECT:" + vs.name] = val

        # OHE Statuses
        for status in self._knowledge["Status"]:
            val = 0
            if status == Status.FNT:
                continue  # Moves dont cause this
            elif not move:
                val = -1
            elif status == move.status:
                val = 1
            elif move.secondary and self._prep(status.name) in list(
                map(lambda x: self._prep(x.get("status", "")), move.secondary)
            ):
                val = 1
            emb[prefix + "STATUS:" + status.name] = val

        # Add Boosts
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            val = 0
            if not move:
                val = -1
            elif move.boosts and stat in move.boosts:
                val = move.boosts[stat]
            elif move.secondary:
                for info in move.secondary:
                    if "boosts" in info and stat in info["boosts"]:
                        val = info["boosts"][stat]
            emb[prefix + "BOOST:" + stat] = val

        # Add Self-Boosts
        for stat in ["atk", "def", "spa", "spd", "spe"]:
            val = 0
            if not move:
                val = -1
            elif move.boosts and stat in move.boosts:
                val = move.boosts[stat]
            elif move.secondary:
                for info in move.secondary:
                    if (
                        "self" in info
                        and "boosts" in info["self"]
                        and stat in info["self"]["boosts"]
                    ):
                        val = info["self"]["boosts"][stat]
            emb[prefix + "SELFBOOST:" + stat] = val

        # Introduce the chance of a secondary effect happening
        val = 0
        if not move:
            val = -1
        elif move.secondary:
            val = max(map(lambda x: x.get("chance", 0), move.secondary))
        emb[prefix + "chance"] = val

        return emb

    def featurize_pokemon(
        self, mon: Optional[Pokemon], request: Dict[str, Any] = {}, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Returns a Dict of features representing the pokemon
        """

        emb: Dict[str, float] = {}

        # Add moves to feature dict (and account for the fact that the mon might have <4 moves)
        moves = list(mon.moves.values()) if mon else []
        available_moves = (
            mon.available_moves_from_request(request) if mon and "moves" in request else []
        )
        for i, move in enumerate((moves + [None, None, None, None])[:4]):
            move_prefix = prefix + "MOVE:" + str(i) + ":"
            emb.update(self.featurize_move(move, move_prefix))

            # Record whether a move is available
            available = -1
            if len(available_moves) > 0:
                available = 1 if move in available_moves else 0
            emb[move_prefix + "available"] = available

        # OHE abilities
        for ability in self._knowledge["Ability"]:
            emb[prefix + "ABILITY:" + ability] = int(mon.ability == ability) if mon else -1

        # OHE items
        for item in self._knowledge["Item"]:
            emb[prefix + "ITEM:" + item] = int(mon.item == item) if mon else -1

        # Add various relevant fields for mons
        emb[prefix + "current_hp_fraction"] = mon.current_hp_fraction if mon else -1
        emb[prefix + "level"] = mon.level if mon else -1
        emb[prefix + "weight"] = mon.weight if mon else -1
        emb[prefix + "is_terastallized"] = mon.is_terastallized if mon else -1

        # Add stats
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            val = -1
            if mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                val = mon.stats[stat]  # type:ignore
            emb[prefix + "STAT:" + stat] = val if val is not None else -1

        # Add boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "BOOST:" + stat] = mon.boosts[stat] if mon else -1

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

        return emb

    def featurize_opponent_pokemon(
        self,
        mon: Optional[Pokemon],
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
            emb.update(self.featurize_move(move, move_prefix))

        # OHE abilities (and/or possibile abilities if we have them)
        for ability in self._knowledge["Ability"]:
            val = 0
            if not mon:
                val = -1
            elif mon.ability:
                val = int(mon.ability == ability)
            else:
                val = int(ability in mon.possible_abilities)

            emb[prefix + "ABILITY:" + ability] = val

        # OHE items (and look into battle inference)
        for item in self._knowledge["Item"]:
            val = 0
            if not mon:
                val = -1
            elif mon.item:
                val = int(item == mon.item)
            elif inference_flags and inference_flags["item"]:
                val = int(item == inference_flags["item"])
            emb[prefix + "ITEM:" + item] = val

        # Add several other fields
        emb[prefix + "current_hp_fraction"] = mon.current_hp_fraction if mon else -1
        emb[prefix + "level"] = mon.level if mon else -1
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
                emb[prefix + "STAT:" + stat] = val if val is not None else -1
        else:
            for stat, minstat, maxstat in zip(stats, minstats, maxstats):
                if mon and inference_flags and stat in inference_flags:
                    minstat, maxstat = inference_flags[stat][0], inference_flags[stat][1]
                elif mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                    minstat, maxstat = mon.stats[stat], mon.stats[stat]
                emb[prefix + "STAT_MIN:" + stat] = (
                    minstat if mon and minstat is not None else -1
                )
                emb[prefix + "STAT_MAX:" + stat] = (
                    maxstat if mon and maxstat is not None else -1
                )

        # Add boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "BOOST:" + stat] = mon.boosts[stat] if mon else -1

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

        return emb

    def featurize_double_battle(
        self, battle: DoubleBattle, bi: Optional[BattleInference] = None
    ) -> Dict[str, float]:
        """
        Returns a list of integers representing the state of the battle, at the beginning
        of the specified turn. It is from the perspective of the player whose turn it is.
        """
        emb: Dict[str, float] = {}

        # Add each of our mons as features. We want to add even our teampreview pokemon because
        # our opponent may make moves dependent on this information
        for i, mon in enumerate((list(battle.teampreview_team) + [None] * 6)[:6]):
            prefix = "MON:" + str(i) + ":"
            features = {}
            sent = 0

            # We should featurize the battle copy of the mon, otherwise, we featurize the teampreview mon
            if (
                battle.player_role
                and mon
                and get_showdown_identifier(mon, battle.player_role) in battle.team
                and not battle.teampreview
            ):
                features = self.featurize_pokemon(
                    battle.team[get_showdown_identifier(mon, battle.player_role)],
                    battle.last_request,
                    prefix,
                )
                sent = 1
            else:
                features = self.featurize_pokemon(mon, battle.last_request, prefix)

            # OHE which switch the pokemon is
            for j in range(6):
                features[prefix + "SWITCH_NUM" + str(j)] = int(i == j)

            # Get pokemon battle attributes
            features[prefix + "sent"] = sent
            active_names = list(
                map(lambda x: x.name if x else None, battle.active_pokemon)
            )
            name = mon.name if mon else ""
            features[prefix + "active"] = int(name in active_names)

            trapped, force_switch = 0, 0
            if name in active_names:
                trapped = battle.trapped[active_names.index(name)]
                force_switch = battle.force_switch[active_names.index(name)]

            emb[prefix + "trapped"] = int(trapped)
            emb[prefix + "force_switch"] = int(force_switch)

            # To maintain representation of public belief state
            mon = battle.team.get(
                mon.identifier(battle.player_role) if mon and battle.player_role else "",
                mon,
            )
            features[prefix + "revealed"] = (
                int(mon.revealed if mon else 0) if sent == 1 else -1
            )

            # Record whether mon is an available switch for active_pokemon1 and 2
            features[prefix + "is_available_to_switch"] = int(
                (mon.name if mon else "")
                in map(lambda x: x.name if x else None, battle.available_switches[0])
                or (mon.name if mon else "")
                in map(lambda x: x.name if x else None, battle.available_switches[1])
            )

            emb.update(features)

        # Featurize each opponent mon
        species_list = list(
            map(lambda x: x.species if x else "none", battle.opponent_team.values())
        )
        for i, mon in enumerate((list(battle.teampreview_opponent_team) + [None] * 6)[:6]):
            prefix = "OPP_MON:" + str(i) + ":"
            features = {}
            sent = 0

            if battle.opponent_role and mon and mon.species in species_list:

                # Find the mon in our team
                for team_mon in battle.opponent_team.values():
                    if team_mon.species == mon.species:
                        mon = team_mon
                        break

                # Get inferences
                flags = bi.get_flags(mon.identifier(battle.opponent_role)) if bi else None

                # Generate Features
                features = self.featurize_opponent_pokemon(
                    mon,
                    inference_flags=flags,
                    prefix=prefix,
                )

                # battle.opponent_team is basically teampreview_team for teampreview
                if not battle.teampreview:
                    sent = 1

            # We saw this mon in teampreview
            else:
                features = self.featurize_opponent_pokemon(mon, prefix=prefix)

            max_team_size = battle.max_team_size if battle.max_team_size else 6

            # We know a mon has to be sent if their teampreview team is < the amount of mons
            # they need to send to battle
            features[prefix + "sent"] = (
                -1
                if sent == 0
                and len(battle.opponent_team)
                == min(max_team_size, len(battle.teampreview_opponent_team))
                and not battle.teampreview
                else sent
            )
            features[prefix + "active"] = int(
                (mon.name if mon else "")
                in map(lambda x: x.name if x else None, battle.opponent_active_pokemon)
            )

            emb.update(features)

        # Add Fields
        for field in self._knowledge["Field"]:
            emb["FIELD:" + field.name] = int(field in battle.fields)

        # Add Side Conditions
        for sc in self._knowledge["SideCondition"]:
            emb["SIDE_CONDITION:" + sc.name] = int(sc in battle.side_conditions)

        for sc in self._knowledge["SideCondition"]:
            emb["OPP_SIDE_CONDITION:" + sc.name] = int(
                sc in battle.opponent_side_conditions
            )

        # Add Weathers
        for weather in self._knowledge["Weather"]:
            emb["WEATHER:" + weather.name] = int(weather in battle.weather)

        # Add Formats
        for frmt in self._knowledge["Format"]:
            emb["FORMAT:" + frmt] = int(frmt == battle.format)

        emb["teampreview"] = int(battle.teampreview)
        emb["p1rating"] = battle.rating if battle.rating else -1
        emb["p2rating"] = battle.opponent_rating if battle.opponent_rating else -1
        emb["turn"] = battle.turn
        emb["bias"] = 1

        # Convert embedding to the specified type
        if self._feature_set == self.SIMPLE:
            emb = self._simplify_features(emb)
        elif self._feature_set == self.FULL:
            emb.update(self.generate_feature_engineered_features(battle, bi))

        return emb


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
                mon.species, [0] * 6, [0] * 6, mon.level, "serious", mon._data
            ),
        ):
            stats[k] = int(0.9 * v) if k != "hp" else v
        return stats
    else:
        stats = {}
        for k, v in zip(
            stat_types,
            compute_raw_stats(
                mon.species, [252] * 6, [31] * 6, mon.level, "serious", mon._data
            ),
        ):
            stats[k] = int(1.1 * v) if k != "hp" else v
        return stats


BASIC_FEATURES = {
    "base_power",
    "current_pp",
    "TYPE:",
    "current_hp_fraction",
    "STATUS:FAINTED" "STAT:spe",
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

# 99.5% of tracked abilities
TRACKED_ABILITIES = {
    "drizzle",
    "clearbody",
    "steamengine",
    "chlorophyll",
    "commander",
    "innerfocus",
    "shadowtag",
    "sandrush",
    "mirrorarmor",
    "lightningrod",
    "competitive",
    "effectspore",
    "powerspot",
    "unaware",
    "sturdy",
    "slushrush",
    "electricsurge",
    "intimidate",
    "owntempo",
    "waterabsorb",
    "moldbreaker",
    "cursedbody",
    "costar",
    "sapsipper",
    "toxicdebris",
    "swiftswim",
    "guts",
    "zerotohero",
    "windrider",
    "wellbakedbody",
    "beadsofruin",
    "vitalspirit",
    "overgrow",
    "disguise",
    "tabletsofruin",
    "moxie",
    "voltabsorb",
    "sharpness",
    "dazzling",
    "thickfat",
    "electromorphosis",
    "defiant",
    "multiscale",
    "corrosion",
    "pixilate",
    "naturalcure",
    "protosynthesis",
    "goodasgold",
    "flamebody",
    "sandveil",
    "oblivious",
    "galewings",
    "thermalexchange",
    "swordofruin",
    "speedboost",
    "regenerator",
    "stormdrain",
    "vesselofruin",
    "roughskin",
    "flashfire",
    "magicbounce",
    "sandstream",
    "armortail",
    "protean",
    "waterveil",
    "weakarmor",
    "drought",
    "psychicsurge",
    "purifyingsalt",
    "prankster",
    "unburden",
    "snowwarning",
    "technician",
    "quarkdrive",
    "eartheater",
    "queenlymajesty",
    "telepathy",
    "levitate",
    "hugepower",
    "friendguard",
    "moody",
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
    "gen9vgc2023regulationc",
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
