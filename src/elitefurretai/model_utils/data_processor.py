# -*- coding: utf-8 -*-
"""This module defines a random players baseline
"""

import datetime
import re

# -*- coding: utf-8 -*-
from logging import Logger
from typing import Any, Dict, Iterator, List, Union

import orjson
from poke_env.data import GenData
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.observed_pokemon import ObservedPokemon
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon_type import PokemonType
from poke_env.stats import compute_raw_stats

from elitefurretai.model_utils.battle_data import BattleData


class DataProcessor:
    _omniscient: bool = False
    _double_data: bool = False
    _gen_data: Dict[int, GenData] = {}

    def __init__(self, omniscient: bool = False, double_data: bool = False):
        self._omniscient = omniscient
        self._double_data = double_data

    def stream_data(self, files: List[str]) -> Iterator[BattleData]:
        for filepath in files:
            with open(filepath, "r") as f:
                json = orjson.loads(f.read())
                yield self._process_battle(json, perspective="p1")
                if self._double_data:
                    yield self._process_battle(json, perspective="p2")

    def load_data(self, files: List[str]) -> Dict[str, BattleData]:
        data = {}
        for bd in self.stream_data(files):
            data[bd.roomid] = bd

        return data

    def _process_battle(
        self, battle_json: Dict[str, Any], perspective: str = "p1"
    ) -> BattleData:
        match = re.match("(gen[0-9])", str(battle_json.get("format")))
        if match is None:
            raise ValueError(
                "Could not parse gen from battle json's format: {format}".format(
                    format=battle_json.get("format")
                )
            )
        gen = int(match.groups()[0][-1])

        if gen not in self._gen_data:
            self._gen_data[gen] = GenData(gen)

        battle = None
        if "vgc" in str(battle_json.get("format")) or "doubles" in str(
            battle_json.get("format")
        ):
            battle = DoubleBattle(
                "tag", battle_json[perspective], Logger("elitefurretai"), gen=gen
            )
        else:
            battle = Battle(
                "tag", battle_json[perspective], Logger("elitefurretai"), gen=gen
            )

        battle.player_role = perspective
        battle.opponent_username = battle_json["p2" if perspective == "p1" else "p1"]

        logs = battle_json["log"]
        for log in logs:
            split_message = log.split("|")

            # Implement parts that parse_message can't deal with
            if (
                len(split_message) == 1
                or split_message == ["", ""]
                or split_message[1] == "t:"
            ):
                continue
            elif split_message[1] == "win":
                battle.won_by(split_message[2])
            elif split_message[1] == "tie":
                battle.tied()
            else:
                battle.parse_message(split_message)

        hashed_roomid = "battle-{format}{num}".format(
            format=battle_json["format"], num=hash(str(battle_json) + perspective)
        )

        can_see_p1_team = perspective == "p1" or self._omniscient
        can_see_p2_team = perspective == "p2" or self._omniscient

        return BattleData(
            roomid=battle_json.get("roomid", hashed_roomid),
            format=battle_json["format"],
            p1=battle_json["p1"],
            p2=battle_json["p2"],
            p1rating=battle_json["p1rating"],
            p2rating=battle_json["p2rating"],
            p1_team=self._prepare_team(
                team_list=battle_json["p1team"], gen=gen, omniscient=can_see_p1_team
            ),
            p2_team=self._prepare_team(
                team_list=battle_json["p2team"], gen=gen, omniscient=can_see_p2_team
            ),
            score=battle_json["score"],
            winner=battle_json["winner"],
            end_type=battle_json["endType"],
            observations=battle.observations,
        )

    def _prepare_team(
        self, team_list: List[Dict[str, Any]], gen: int, omniscient: bool = False
    ) -> List[ObservedPokemon]:
        team = []

        for mon_info in team_list:
            species = mon_info["species"].lower().replace("-", "").replace(" ", "")

            ability = None
            tera = None
            item = None
            moves = {}
            stats = ObservedPokemon.initial_stats()

            if omniscient:
                stats = dict(
                    zip(
                        ["hp", "atk", "def", "spa", "spd", "spe"],
                        compute_raw_stats(
                            species,
                            list(mon_info["evs"].values()),
                            list(mon_info["ivs"].values()),
                            mon_info["level"],
                            mon_info.get("nature", "serious").lower(),
                            self._gen_data[gen],
                        ),
                    )
                )
                stats.pop("hp")  # Remove for now, since pokemon.stats doesnt have hp

                # These are things we wouldn't know if we don't have omniscience
                ability = mon_info["ability"]
                if "teraType" in mon_info:
                    tera = PokemonType.from_name(str(mon_info.get("teraType")))
                item = mon_info["item"]
                moves = {str(m): Move(m, gen=gen) for m in mon_info["moves"]}

            gender = None
            if mon_info.get("gender", None) == "M":
                gender = PokemonGender.MALE
            elif mon_info.get("gender", None) == "F":
                gender = PokemonGender.FEMALE
            else:
                gender = PokemonGender.NEUTRAL

            team.append(
                ObservedPokemon(
                    species=species,
                    stats=stats,
                    moves=moves,
                    ability=ability,
                    item=item,
                    gender=gender,
                    tera_type=tera,
                    shiny=mon_info.get("shiny", None),
                    level=mon_info["level"],
                )
            )

        return team

    @staticmethod
    def pokemon_to_json(pokemon: Pokemon) -> Dict[str, Any]:
        return {
            "name": pokemon.species,
            "species": pokemon.species,
            "item": pokemon.item,
            "ability": pokemon.ability,
            "moves": list(pokemon.moves.keys()),
            # TODO: reverse engineer
            "nature": "serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
        }

    @staticmethod
    def battle_to_json(battle: Union[Battle, DoubleBattle]) -> bytes:
        p1team = battle.team if battle.player_role == "p1" else battle.opponent_team
        p2team = battle.opponent_team if battle.player_role == "p1" else battle.team

        json = {
            "winner": battle.player_username if battle.won else battle.opponent_username,
            "turns": battle.turn,
            "p1": (
                battle.player_username
                if battle.player_role == "p1"
                else battle.opponent_username
            ),
            "p2": (
                battle.opponent_username
                if battle.player_role == "p1"
                else battle.player_username
            ),
            "p1team": [
                DataProcessor.pokemon_to_json(mon) for species, mon in p1team.items()
            ],
            "p2team": [
                DataProcessor.pokemon_to_json(mon) for species, mon in p2team.items()
            ],
            "score": [
                len(list(filter(lambda x: x[1].fainted, p1team.items()))),
                len(list(filter(lambda x: x[1].fainted, p2team.items()))),
            ],
            "inputLog": None,
            "log": [
                "|".join(split_message)
                for turn in sorted(battle.observations.keys())
                for split_message in battle.observations[turn].events
            ],
            "p1rating": None,
            "p2rating": None,
            "endType": (  # Don't yet have ability to tell if forfeited
                "normal" if battle.won else "forced"
            ),
            "ladderError": False,
            "timestamp": datetime.datetime.now().strftime("%a %b %d %Y %H:%M:%S GMT%z"),
        }

        return orjson.dumps(json)

    @property
    def omniscient(self) -> bool:
        """
        :return: Whether the BattleData object should know hidden information
            from the opponents (eg has omniscient view). This only applies to data
            that has omniscient perspective to begin with.
        :rtype: bool
        """
        return self._omniscient

    @property
    def double_data(self) -> bool:
        """
        :return: Whether we should return two BattleData objects for each battle.
            One for each player's perspective (to get double the data)
        :rtype: bool
        """
        return self._double_data
