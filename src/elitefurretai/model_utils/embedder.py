# -*- coding: utf-8 -*-
"""This module defines a class that Embeds objects
"""

import logging
import math
from typing import Any, Dict, List, Optional

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
        self, format="gen9vgc2024regc", type: str = "full", max_size: int = 1
    ):
        self._history: List[Dict[str, float]] = []
        self._knowledge: Dict[str, Any] = {}
        self._format: str = format

        assert max_size > 0
        self._max_size: int = max_size

        assert type in [self.SIMPLE, self.RAW, self.FULL]
        self._type: str = type

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

        self._embedding_size = self._get_embedding_size()

    @staticmethod
    def _prep(string) -> str:
        """
        Used to convert names of various enumerations into their string variants
        """
        return string.lower().replace("_", " ")

    def _get_embedding_size(self) -> int:
        dummy_battle = DoubleBattle(
            "tag", "elitefurretai", logging.Logger("example"), gen=int(self._format[3])
        )
        dummy_battle._format = self._format
        dummy_battle.player_role = "p1"
        return len(self.featurize_double_battle(dummy_battle))

    def history(self, last_n: Optional[int] = None) -> List[List[float]]:
        if last_n is None:
            return list(map(lambda x: self.feature_dict_to_vector(x), self._history))
        else:
            to_return = []

            for i in range(last_n):
                if len(self._history) > i:
                    to_return.append(self.feature_dict_to_vector(self._history[i]))
                else:
                    to_return.insert(0, [-1] * self.embedding_size)
            return to_return

    @property
    def embedding_size(self):
        return self._embedding_size

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

    def _add_full_features(self, battle: DoubleBattle, bi: Optional[BattleInference] = None) -> Dict[str, float]:
        """Adds supplementary engineered features that should help with learning"""
        features: Dict[str, float] = {}
        """
        Features to consider:
        - Move orders for each mon
        - Type matchups for each mon
        - num non-fainted mons on my side
        - num non-fainted mons on their side
        - can they knock me out?
        - can I knock them out?
        - do i have a move that can change the number of move orders (speed)
        - do they have a move that can change the number of move orders (speed)
        - number of moves i have revealed
        - number of moves they have revealed
        - number of mons i have revealed
        - number of moves they have revealed
        - % hp I have left
        - % hp they have left
        """
        return {}

        # TODO: implement
        perc_hp = 1.0
        for i, mon in enumerate((list(battle.teampreview_team) + [None] * 6)[:6]):
            prefix = "MON:" + str(i) + ":"

            # TODO: does mon in teampreview and batte.team stay the same? no.

            if (mon.name if mon else "") in map(lambda x: x.name if x else None, battle.active_pokemon):
                pass

            for move in mon.moves:
                pass

        opp_perc_hp = 1.0
        for i, mon in enumerate((list(battle.teampreview_opponent_team) + [None] * 6)[:6]):
            prefix = "OPP_MON:" + str(i) + ":"

            for move in mon.moves:
                pass

        # TODO: add
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
            minstats = map(
                lambda x: math.floor(x[1] * 0.9) if x[0] != 0 else x[1],
                enumerate(
                    compute_raw_stats(
                        mon.species, [0] * 6, [0] * 6, mon.level, "serious", mon._data
                    )
                ),
            )
            maxstats = map(
                lambda x: math.floor(x[1] * 1.1) if x[0] != 0 else x[1],
                enumerate(
                    compute_raw_stats(
                        mon.species, [252] * 6, [31] * 6, mon.level, "serious", mon._data
                    )
                ),
            )

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

            features[prefix + "sent"] = sent
            features[prefix + "active"] = int(
                (mon.name if mon else "")
                in map(lambda x: x.name if x else None, battle.active_pokemon)
            )

            # To maintain representation of public belief state
            features[prefix + "revealed"] = int(mon.revealed if mon else 0)

            # Record whether mon is an available switch for active_pokemon1 and 2
            for j, switches in enumerate(battle.available_switches):
                val = -1
                if battle.active_pokemon[j] is not None:
                    val = 1 if mon in switches else 0
                features[prefix + "available_switch:" + str(j)] = val

            # Record the number that corresponds to the mon switch (eg #3, #4)
            # We don't look at the request if there isn't one
            val = -1
            mons = battle.last_request.get("side", {}).get("pokemon", [])
            for j, json_mon in enumerate((mons + [None] * 6)[:6]):
                if (
                    mon is not None
                    and json_mon is not None
                    and json_mon.get("identifier", "")
                    == get_showdown_identifier(mon, battle.player_role)
                ):
                    val = j + 1
            features[prefix + "switch_number"] = val

            emb.update(features)

        # Featurize each opponent mon
        for i, mon in enumerate((list(battle.teampreview_opponent_team) + [None] * 6)[:6]):
            prefix = "OPP_MON:" + str(i) + ":"
            features = {}
            sent = 0

            # Meaning we have seen this mon on the field
            if (
                battle.opponent_role
                and mon
                and get_showdown_identifier(mon, battle.opponent_role)
                in battle.opponent_team
            ):
                flags = (
                    bi.get_flags(get_showdown_identifier(mon, battle.opponent_role))
                    if bi
                    else None
                )
                features = self.featurize_opponent_pokemon(
                    battle.opponent_team[
                        get_showdown_identifier(mon, battle.opponent_role)
                    ],
                    inference_flags=flags,
                    prefix=prefix,
                )
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

        # Add additional things about the battle state
        for i, trapped in enumerate(battle.trapped):
            emb["TRAPPED:" + str(i)] = int(trapped)

        for i, fs in enumerate(battle.force_switch):
            emb["FORCE_SWITCH:" + str(i)] = int(fs)

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
        if self._type == self.SIMPLE:
            emb = self._simplify_features(emb)
        elif self._type == self.FULL:
            emb.update(self._add_full_features(battle, bi))

        # Store the embedding in our history
        if len(self._history) == self._max_size:
            self._history.pop(0)
        self._history.append(emb)

        return emb


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

TRACKED_EFFECTS = {
    Effect.from_showdown_message("ability: Protosynthesis"),
    Effect.from_showdown_message("ability: Protosynthesis"),
    Effect.from_showdown_message("ability: Quark Drive"),
    Effect.from_showdown_message("protosynthesisspe"),
    Effect.from_showdown_message("quarkdrivespe"),
    Effect.from_showdown_message("ability: Zero to Hero"),
    Effect.from_showdown_message("protosynthesisspa"),
    Effect.from_showdown_message("perish3"),
    Effect.from_showdown_message("move: Taunt"),
    Effect.from_showdown_message("protosynthesisatk"),
    Effect.from_showdown_message("Substitute"),
    Effect.from_showdown_message("ability: Toxic Debris"),
    Effect.from_showdown_message("move: Leech Seed"),
    Effect.from_showdown_message("perish2"),
    Effect.from_showdown_message("Salt Cure"),
    Effect.from_showdown_message("ability: Commander"),
    Effect.from_showdown_message("perish1"),
    Effect.from_showdown_message("Encore"),
    Effect.from_showdown_message("move: Yawn"),
    Effect.from_showdown_message("confusion"),
    Effect.from_showdown_message("Throat Chop"),
    Effect.from_showdown_message("Disable"),
    Effect.from_showdown_message("perish0"),
    Effect.from_showdown_message("move: Substitute"),
    Effect.from_showdown_message("quarkdriveatk"),
    Effect.from_showdown_message("move: Trick"),
    Effect.from_showdown_message("ability: Flash Fire"),
    Effect.from_showdown_message("quarkdrivespa"),
    Effect.from_showdown_message("typechange"),
    Effect.from_showdown_message("Charge"),
    Effect.from_showdown_message("move: After You"),
    Effect.from_showdown_message("move: Sand Tomb"),
    Effect.from_showdown_message("trapped"),
    Effect.from_showdown_message("ability: Storm Drain"),
    Effect.from_showdown_message("move: Struggle"),
    Effect.from_showdown_message("ability: Dancer"),
    Effect.from_showdown_message("protosynthesisspd"),
    Effect.from_showdown_message("move: Quash"),
    Effect.from_showdown_message("move: Endure"),
    Effect.from_showdown_message("ability: Supreme Overlord"),
    Effect.from_showdown_message("move: Imprison"),
    Effect.from_showdown_message("move: Focus Energy"),
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

TRACKED_ABILITIES = {
    "protosynthesis",
    "quarkdrive",
    "intimidate",
    "beadsofruin",
    "regenerator",
    "vesselofruin",
    "swordofruin",
    "defiant",
    "multiscale",
    "zerotohero",
    "prankster",
    "goodasgold",
    "commander",
    "unaware",
    "galewings",
    "friendguard",
    "toxicdebris",
    "thermalexchange",
    "drought",
    "innerfocus",
    "snowwarning",
    "technician",
    "roughskin",
    "psychicsurge",
    "levitate",
    "mirrorarmor",
    "flashfire",
    "sandstream",
    "tabletsofruin",
    "oblivious",
    "hugepower",
    "drizzle",
    "clearbody",
    "eartheater",
    "purifyingsalt",
    "vitalspirit",
    "chlorophyll",
    "sandrush",
    "shadowtag",
    "armortail",
    "magicbounce",
    "stormdrain",
    "flamebody",
    "swiftswim",
    "corrosion",
    "pixilate",
    "disguise",
    "telepathy",
    "weakarmor",
    "wellbakedbody",
    "unburden",
    "sharpness",
    "sandveil",
    "guts",
}

TRACKED_ITEMS = {
    "focussash",
    "assaultvest",
    "boosterenergy",
    "safetygoggles",
    "sitrusberry",
    "leftovers",
    "choicespecs",
    "lifeorb",
    "choicescarf",
    "choiceband",
    "clearamulet",
    "mysticwater",
    "lumberry",
    "covertcloak",
    "rockyhelmet",
    "lightclay",
    "eviolite",
    "weaknesspolicy",
    "charcoal",
    "loadeddice",
    "aguavberry",
    "mentalherb",
    "widelens",
    "psychicseed",
    "ejectpack",
    "throatspray",
    "damprock",
    "figyberry",
    "expertbelt",
    "blackglasses",
    "iapapaberry",
    "mirrorherb",
    "blacksludge",
    "wikiberry",
    "airballoon",
    "flameorb",
    "ejectbutton",
}

TRACKED_FORMATS = {
    "gen6doubblesou",
    "gen9vgc2024regf",
    "gen9vgc2024regg",
    "gen9vgc2024regh",
    "gen9vgc2024regc",
    "gen9vgc2024regb",
    "gen9vgc2024rega",
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
