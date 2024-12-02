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
from elitefurretai.inference.inference_utils import get_showdown_identifier


class Embedder:

    def __init__(self, format="gen9vgc2024regh", simple: bool = True):
        self._history: List[Dict[str, float]] = []
        self._knowledge: Dict[str, Any] = {}
        self._format = format
        self._gen_data = GenData.from_gen(format[3])
        self._simple: bool = simple

        sets = [
            ("Status", Status),
            ("PokemonType", PokemonType),
            ("MoveCategory", MoveCategory),
            ("Effect", Effect),
        ]

        for key, enum in sets:
            self._knowledge[key] = set(enum)

        self._knowledge["Pokemon"] = set(self._gen_data.pokedex.keys())
        self._knowledge["Effect_VolatileStatus"] = TRACKED_EFFECTS
        self._knowledge["Item"] = TRACKED_ITEMS
        self._knowledge["Target"] = TRACKED_TARGET_TYPES
        self._knowledge["Format"] = TRACKED_FORMATS
        self._knowledge["SideCondition"] = TRACKED_SIDE_CONDITIONS
        self._knowledge["Weather"] = TRACKED_WEATHERS
        self._knowledge["Field"] = TRACKED_FIELDS

        # Sourced from https://github.com/smogon/pokemon-showdown/blob/67354c8f3d285b52a2e4cd4c6aa194a1bfd19c1c/data/abilities.ts
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
        if self._simple:
            return len(self._simplify_features(self.featurize_double_battle(dummy_battle)))
        else:
            return len(self.featurize_double_battle(dummy_battle))

    def history(self, last_n: Optional[int] = None) -> List[List[float]]:
        if last_n is None:
            return list(map(lambda x: self.feature_dict_to_vector(x), self._history))
        else:
            to_return = []

            # Need to construct dummy battle
            dummy_battle = DoubleBattle(
                "tag", "elitefurretai", logging.Logger("example"), gen=int(self._format[3])
            )
            dummy_battle._format = self._format

            for i in range(last_n):
                if len(self._history) > i:
                    to_return.append(self.feature_dict_to_vector(self._history[i]))
                else:
                    emb = self.featurize_double_battle(dummy_battle)
                    to_return.insert(0, self.feature_dict_to_vector(emb))
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

    # As of writing, this is 1449 features, mostly due to OHE of pokemon types
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

    def featurize_move(self, move: Optional[Move], prefix: str = "") -> Dict[str, float]:
        """
        Returns a feature dict representing a Move
        """

        emb: Dict[str, float] = {}

        emb[prefix + "accuracy"] = move.accuracy if move else -1  # TODO: normalize
        emb[prefix + "base_power"] = (
            move.base_power / 100.0 if move else -1
        )  # TODO: normalize
        emb[prefix + "current_pp"] = min(move.current_pp, 5) / 5.0 if move else -1

        # To maintain representation of public belief state
        emb[prefix + "used"] = int(move.current_pp < move.max_pp) if move else -1

        # TODO: is this right?
        if move:
            emb[prefix + "damage"] = (
                move.damage if isinstance(move.damage, int) else 50
            ) / 100.0  # TODO: normalize
        else:
            emb[prefix + "damage"] = -1
        emb[prefix + "drain"] = move.drain if move else -1  # TODO: normalize
        emb[prefix + "force_switch"] = move.force_switch if move else -1
        emb[prefix + "heal"] = move.heal if move else -1  # TODO: normalize
        emb[prefix + "is_protect_move"] = move.is_protect_move if move else -1
        emb[prefix + "is_side_protect_move"] = move.is_side_protect_move if move else -1
        if move:
            emb[prefix + "min_hits"] = (
                move.n_hit[0] if move.n_hit else 1
            )  # TODO: normalize
            emb[prefix + "max_hits"] = (
                move.n_hit[1] if move.n_hit else 1
            )  # TODO: normalize
        else:
            emb[prefix + "min_hits"] = -1
            emb[prefix + "max_hits"] = -1
        emb[prefix + "priority"] = move.priority if move else -1  # TODO: normalize
        emb[prefix + "recoil"] = move.recoil if move else -1  # TODO: normalize
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
            emb[prefix + "BOOST:" + stat] = val  # TODO: normalize

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
            emb[prefix + "SELFBOOST:" + stat] = val  # TODO: normalize

        # Introduce the chance of a secondary effect happening
        val = 0
        if not move:
            val = -1
        elif move.secondary:
            val = max(map(lambda x: x.get("chance", 0), move.secondary)) / 100.0
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
        emb[prefix + "level"] = mon.level / 100.0 if mon else -1
        emb[prefix + "weight"] = mon.weight / 100.0 if mon else -1
        emb[prefix + "is_terastallized"] = mon.is_terastallized if mon else -1

        # Add stats
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            val = -1
            if mon and mon.stats and stat in mon.stats and mon.stats[stat]:
                val = mon.stats[stat] / 100.0  # pyright:ignore
            emb[prefix + "STAT:" + stat] = val  # TODO: normalize

        # Add boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "BOOST:" + stat] = (
                mon.boosts[stat] if mon else -1
            )  # TODO: normalize

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
        emb[prefix + "level"] = mon.level / 100.0 if mon else -1
        emb[prefix + "weight"] = mon.weight / 100.0 if mon else -1  # TODO: normalize
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
            emb[prefix + "STAT_MIN:" + stat] = (  # TODO: normalize
                minstat / 100.0 if mon and minstat is not None else -1
            )
            emb[prefix + "STAT_MAX:" + stat] = (  # TODO: normalize
                maxstat / 100.0 if mon and maxstat is not None else -1
            )

        # Add boosts; don't add evasion
        for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "BOOST:" + stat] = (
                mon.boosts[stat] if mon else -1
            )  # TODO: normalize

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
                (mon.species if mon else "")
                in map(lambda x: x.species if x else None, battle.active_pokemon)
            )

            # To maintain representation of public belief state
            features[prefix + "revealed"] = int(mon.revealed if mon else 0)

            # Record whether mon is an available switch for active_pokemon1 and 2
            for j, switches in enumerate(battle.available_switches):
                val = -1
                if battle.active_pokemon[j] is not None:
                    val = 1 if mon in switches else 0
                features[prefix + "available_switch:" + str(j)] = val

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
                (mon.species if mon else "")
                in map(lambda x: x.species if x else None, battle.opponent_active_pokemon)
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
        emb["p1rating"] = battle.rating if battle.rating else -1  # TODO: normalize
        emb["p2rating"] = (
            battle.opponent_rating if battle.opponent_rating else -1
        )  # TODO: normalize
        emb["turn"] = battle.turn  # TODO: normalize
        emb["bias"] = 1

        self._history.append(emb)

        if self._simple:
            emb = self._simplify_features(emb)

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

# TODO: need to revisit once I have showdown data to understand usage
TRACKED_EFFECTS = {
    Effect.DISABLE,
    Effect.ENCORE,
    Effect.SPOTLIGHT,
    Effect.SILK_TRAP,
    Effect.HEAL_BLOCK,
    Effect.TAUNT,
    Effect.YAWN,
    Effect.IMPRISON,
    Effect.FOLLOW_ME,
    Effect.GLAIVE_RUSH,
    Effect.CONFUSION,
    Effect.BANEFUL_BUNKER,
    Effect.PROTECT,
    Effect.BURNING_BULWARK,
    Effect.DRAGON_CHEER,
    Effect.HELPING_HAND,
    Effect.FLINCH,
    Effect.SUBSTITUTE,
    Effect.RAGE_POWDER,
    Effect.ROOST,
    Effect.SALT_CURE,
    Effect.LEECH_SEED,
    Effect.POWDER,
    Effect.SPIKY_SHIELD,
    Effect.ENDURE,
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
    "embodyaspecthearthflame",
    "trace",
    "waterveil",
    "gooey",
    "mountaineer",
    "static",
    "schooling",
    "icescales",
    "thickfat",
    "ripen",
    "liquidooze",
    "imposter",
    "simple",
    "sandforce",
    "desolateland",
    "quickfeet",
    "goodasgold",
    "poisonpuppeteer",
    "roughskin",
    "whitesmoke",
    "comatose",
    "scrappy",
    "tintedlens",
    "stakeout",
    "overgrow",
    "moxie",
    "solidrock",
    "stancechange",
    "innardsout",
    "stamina",
    "thermalexchange",
    "sapsipper",
    "moody",
    "chillingneigh",
    "eartheater",
    "flashfire",
    "myceliummight",
    "magnetpull",
    "prismarmor",
    "illusion",
    "opportunist",
    "slushrush",
    "wonderguard",
    "motordrive",
    "forecast",
    "turboblaze",
    "toxicboost",
    "waterbubble",
    "fluffy",
    "guts",
    "lightningrod",
    "primordialsea",
    "asoneglastrier",
    "mistysurge",
    "teravolt",
    "harvest",
    "toxicdebris",
    "wonderskin",
    "poisontouch",
    "psychicsurge",
    "teraformzero",
    "pixilate",
    "aerilate",
    "tanglinghair",
    "swordofruin",
    "shieldsdown",
    "electricsurge",
    "mummy",
    "cursedbody",
    "magicguard",
    "sniper",
    "serenegrace",
    "clearbody",
    "shadowshield",
    "marvelscale",
    "sturdy",
    "guarddog",
    "skilllink",
    "limber",
    "regenerator",
    "toughclaws",
    "ironbarbs",
    "filter",
    "embodyaspectwellspring",
    "electromorphosis",
    "fairyaura",
    "heatproof",
    "rebound",
    "chlorophyll",
    "unseenfist",
    "persistent",
    "magicbounce",
    "gulpmissile",
    "embodyaspectteal",
    "reckless",
    "transistor",
    "voltabsorb",
    "shielddust",
    "angershell",
    "hustle",
    "quarkdrive",
    "fullmetalbody",
    "cudchew",
    "adaptability",
    "tabletsofruin",
    "noguard",
    "drought",
    "aromaveil",
    "download",
    "hadronengine",
    "moldbreaker",
    "asonespectrier",
    "steelworker",
    "galvanize",
    "competitive",
    "contrary",
    "dragonsmaw",
    "cottondown",
    "wanderingspirit",
    "hugepower",
    "intimidate",
    "defiant",
    "ironfist",
    "shedskin",
    "waterabsorb",
    "flareboost",
    "sandstream",
    "bulletproof",
    "pastelveil",
    "sweetveil",
    "dauntlessshield",
    "refrigerate",
    "unburden",
    "synchronize",
    "darkaura",
    "shadowtag",
    "furcoat",
    "cheekpouch",
    "blaze",
    "mirrorarmor",
    "solarpower",
    "rockypayload",
    "surgesurfer",
    "zerotohero",
    "iceface",
    "battlebond",
    "queenlymajesty",
    "lingeringaroma",
    "purifyingsalt",
    "supremeoverlord",
    "analytic",
    "windrider",
    "libero",
    "arenatrap",
    "snowwarning",
    "technician",
    "immunity",
    "prankster",
    "strongjaw",
    "unaware",
    "toxicchain",
    "multitype",
    "terashift",
    "torrent",
    "pressure",
    "poisonheal",
    "punkrock",
    "rkssystem",
    "disguise",
    "seedsower",
    "multiscale",
    "orichalcumpulse",
    "sharpness",
    "protean",
    "steamengine",
    "levitate",
    "purepower",
    "screencleaner",
    "infiltrator",
    "parentalbond",
    "effectspore",
    "aftermath",
    "soundproof",
    "triage",
    "deltastream",
    "corrosion",
    "wellbakedbody",
    "flamebody",
    "megalauncher",
    "terashell",
    "soulheart",
    "vesselofruin",
    "beadsofruin",
    "swarm",
    "beastboost",
    "stormdrain",
    "rockhead",
    "naturalcure",
    "dryskin",
    "dazzling",
    "steelyspirit",
    "victorystar",
    "grimneigh",
    "protosynthesis",
    "sheerforce",
    "swiftswim",
    "armortail",
    "overcoat",
    "berserk",
    "neutralizinggas",
    "intrepidsword",
    "gorillatactics",
    "speedboost",
    "quickdraw",
    "justified",
    "compoundeyes",
    "powerconstruct",
    "neuroforce",
    "drizzle",
    "grassysurge",
    "sandrush",
    "embodyaspectcornerstone",
    "frisk",
    "stall",
}

TRACKED_ITEMS = {
    "aguavberry",
    "assaultvest",
    "safetygoggles",
    "clearamulet",
    "covertcloak",
    "heavydutyboots",
    "leftovers",
    "sitrusberry",
    "focussash",
    "choicescarf",
    "lifeorb",
    "choiceband",
    "choicespecs",
    "wikiberry",
    "elecrticseed",
    "grassyseed",
    "psychicseed",
    "rockyhelmet",
    "boosterenergy",
    "flameorb",
    "lightclay",
    "laggingtail",
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
