# -*- coding: utf-8 -*-
"""This class stores the data of a battle, from a json file
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import orjson
from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment import (
    AbstractBattle,
    DoubleBattle,
    Move,
    ObservedPokemon,
    PokemonGender,
    PokemonType,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.utils.inference_utils import get_showdown_identifier


@dataclass
class BattleData:
    battle_tag: str
    format: str
    end_type: str
    turns: int
    score: List[int]
    p1_rating: int
    p2_rating: int
    p1_team: List[ObservedPokemon]
    p2_team: List[ObservedPokemon]
    p1: str
    p2: str
    winner: str
    logs: List[str]
    input_logs: List[str]

    @staticmethod
    def from_elite_furret_ai_json(json: Dict[str, Any]):
        return BattleData(
            battle_tag=json["battle_tag"],
            format=json["format"],
            end_type=json["end_type"],
            turns=json["turns"],
            score=json["score"],
            p1_rating=json["p1_rating"],
            p2_rating=json["p2_rating"],
            p1_team=team_from_json(json["p1_team"]),
            p2_team=team_from_json(json["p2_team"]),
            p1=json["p1"],
            p2=json["p2"],
            winner=json["winner"],
            logs=json["logs"],
            input_logs=json["input_logs"],
        )

    @staticmethod
    def from_showdown_json(json: Dict[str, Any]):
        return BattleData(
            battle_tag=str(hash("".join(json["log"]))),
            format=json["format"],
            end_type=json["endType"],
            turns=int(json["turns"]),
            score=json["score"],
            p1_rating=int(json["p1rating"]["rpr"]) if isinstance(json["p2rating"], dict) else None,  # type: ignore
            p2_rating=int(json["p2rating"]["rpr"]) if isinstance(json["p2rating"], dict) else None,  # type: ignore
            p1_team=team_from_json(json["p1team"]),
            p2_team=team_from_json(json["p2team"]),
            p1=json["p1"],
            p2=json["p2"],
            winner=json["winner"],
            logs=json["log"],
            input_logs=json["inputLog"],
        )

    @staticmethod
    def from_self_play(
        p1_battle: AbstractBattle, p2_battle: AbstractBattle, input_logs: List[str]
    ):

        # Construct teams
        p1_team: List[ObservedPokemon] = []
        for tp_mon in p1_battle.teampreview_team:
            ident = get_showdown_identifier(tp_mon, "p1")
            mon = (
                p1_battle.team[ident] if ident in p1_battle.team else tp_mon
            )  # To handle transformed mons
            mon._terastallized_type = tp_mon.tera_type
            mon.stats = tp_mon.stats

            p1_team.append(ObservedPokemon.from_pokemon(mon))  # type: ignore

        p2_team: List[ObservedPokemon] = []
        for tp_mon in p2_battle.teampreview_team:
            ident = get_showdown_identifier(tp_mon, "p2")
            mon = (
                p2_battle.team[ident] if ident in p2_battle.team else tp_mon
            )  # To handle transformed mons
            mon._terastallized_type = tp_mon.tera_type
            mon.stats = tp_mon.stats
            p2_team.append(ObservedPokemon.from_pokemon(mon))  # type: ignore

        # Reconstruct logs
        logs = []
        for turn in range(len(p1_battle.observations)):
            logs.extend(p1_battle.observations[turn].events)

        # According to: https://github.com/smogon/pokemon-showdown/blob/b719e950f1166406a1cbf225c2a263e0848e4b0f/server/room-battle.ts#L526
        # endType can be # 'forfeit' | 'forced' | 'normal' = 'normal'
        end_type, winner = "forced", ""
        if p1_battle.won is not None:
            end_type = "normal"
            winner = p1_battle.player_role if p1_battle.won else p1_battle.opponent_role

        return BattleData(
            battle_tag=p1_battle.battle_tag,
            format=p1_battle.format,  # pyright: ignore
            end_type=end_type,
            turns=p1_battle.turn,
            score=[
                len(list(filter(lambda x: not x.fainted, p1_battle.team.values()))),
                len(list(filter(lambda x: not x.fainted, p2_battle.team.values()))),
            ],
            p1_rating=0,
            p2_rating=0,
            p1_team=p1_team,
            p2_team=p2_team,
            p1=p1_battle.player_username,  # pyright: ignore
            p2=p2_battle.player_username,  # pyright: ignore
            winner=winner,  # pyright: ignore
            logs=logs,
            input_logs=input_logs,
        )

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "battle_tag": self.battle_tag,
            "format": self.format,
            "end_type": self.end_type,
            "turns": self.turns,
            "score": self.score,
            "p1_rating": self.p1_rating,
            "p2_rating": self.p2_rating,
            "p1_team": list(map(lambda x: observed_pokemon_to_dict(x), self.p1_team)),
            "p2_team": list(map(lambda x: observed_pokemon_to_dict(x), self.p2_team)),
            "p1": self.p1,
            "p2": self.p2,
            "winner": self.winner,
            "logs": self.logs,
            "input_logs": self.input_logs,
        }

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            serialized = self.to_dict()
            f.write(orjson.dumps(serialized))

    def to_battle(self, perspective: str) -> DoubleBattle:
        assert perspective in ["p1", "p2"]

        player = self.p1 if perspective == "p1" else self.p2
        team = self.p1_team if perspective == "p1" else self.p2_team
        opp_player = self.p2 if perspective == "p1" else self.p1
        opp_team = self.p2_team if perspective == "p1" else self.p1_team

        battle = DoubleBattle(
            self.battle_tag, player, logging.getLogger(player), gen=int(self.format[3])
        )
        battle.player_role = perspective

        battle.player_username = player
        battle.teampreview_team = {mon.to_pokemon() for mon in team}

        # Assumes an order in inputLog
        index = 0 if perspective == "p1" else 1
        choices = map(
            lambda x: int(x) - 1,
            self.input_logs[index].replace(f">{perspective} team ", "").split(", "),
        )
        sent_team = {}
        for omon in [team[choice] for choice in choices]:
            mon = omon.to_pokemon()
            sent_team[get_showdown_identifier(mon, perspective)] = mon
        battle.team = sent_team

        battle.opponent_username = opp_player
        battle._teampreview_opponent_team = {mon.to_pokemon() for mon in opp_team}

        return battle

    # To help with some corrupted logs that we have
    @property
    def is_valid_for_supervised_learning(self) -> bool:
        # Old showdown protool
        if any(map(lambda x: "[ability2] " in x, self.logs)):
            return False

        # Player error
        elif self.p1_rating is None or self.p2_rating is None:
            return False

        # Battle didnt start
        elif self.input_logs == []:
            return False

        # Edge-case where an active mon just faints in the same turn and you can revival blessing it.
        # Adding this to model_battle_order will increase the size and inaccuracy of the model
        elif any(map(lambda x: "switch 1" in x or "switch 2" in x, self.input_logs)):
            return False

        # Creates a bunch of edge-cases that technically shouldn't be supported
        elif any(map(lambda x: "Metronome" in x, self.logs)):
            return False

        # Old showdown protocol
        elif any(map(lambda x: "|-ability||Zero to Hero" in x, self.logs)):
            return False

        # Can't do these battles properly without requests
        elif any(map(lambda x: "-transform" in x, self.logs)):
            return False

        # Too lazy to implement this edge-case
        elif any(
            map(
                lambda x: "0 fnt|[from] Stealth Rock" in x or "0 fnt|[from] Spikes" in x,
                self.logs,
            )
        ):
            return False

        # ef this noise
        elif any(map(lambda x: "Zoroark" in x or "Zorua" in x, self.logs)):
            return False

        return True

    # From anonymous logs, we need to make the following changes to ensure compatability
    @staticmethod
    def showdown_translation(msg):
        if msg[1] == "move" and msg[-1].startswith("[from] "):
            msg[-1] = msg[-1].replace("[from] ", "[from]")
        elif msg[1] == "move" and msg[-2].startswith("[from] "):
            msg[-2] = msg[-2].replace("[from] ", "[from]")
        return msg


def team_from_json(team: List[Dict[str, Any]]) -> List[ObservedPokemon]:
    return [pokemon_from_json(pokemon_json) for pokemon_json in team]


def pokemon_from_json(mon_info: Dict[str, Any], gen=9) -> ObservedPokemon:
    stats = {}
    if "stats" in mon_info:
        stats = mon_info["stats"]
    else:
        nature = "serious"
        if "nature" in mon_info and len(mon_info["nature"]) > 1:
            nature = mon_info["nature"].lower()

        stats = mon_info.get(
            "stats",
            dict(
                zip(
                    ["hp", "atk", "def", "spa", "spd", "spe"],
                    compute_raw_stats(
                        to_id_str(mon_info["species"]),
                        list(mon_info["evs"].values()),
                        list(mon_info["ivs"].values()),
                        mon_info["level"],
                        nature,
                        GenData.from_gen(gen),
                    ),
                )
            ),
        )

    tera = None
    if mon_info.get("teraType", "null") != "null":
        tera = PokemonType.from_name(str(mon_info.get("teraType")))

    moves = OrderedDict()
    for m in mon_info["moves"]:
        moves[str(m)] = Move(to_id_str(m), gen=gen)

    gender = None
    if mon_info.get("gender", None) == "M":
        gender = PokemonGender.MALE
    elif mon_info.get("gender", None) == "F":
        gender = PokemonGender.FEMALE
    else:
        gender = PokemonGender.NEUTRAL

    return ObservedPokemon(
        species=to_id_str(mon_info["species"]),
        name=mon_info.get("name", to_id_str(mon_info["species"])),
        stats=stats,
        moves=moves,
        ability=to_id_str(mon_info["ability"]),
        item=mon_info.get("item", None),
        gender=gender,
        tera_type=tera,
        shiny=mon_info.get("shiny", None),
        level=mon_info["level"],
    )


# Converts an ObservedPokemon into a dictionary for storage
def observed_pokemon_to_dict(omon: ObservedPokemon) -> Dict[str, Any]:
    gender = "N"
    if omon.gender == PokemonGender.MALE:
        gender = "M"
    elif omon.gender == PokemonGender.FEMALE:
        gender = "F"

    tera = omon.tera_type
    if tera is not None:
        tera = tera.name.title()
    else:
        tera = "null"

    # Default to false
    shiny = omon.shiny
    if shiny is None:
        shiny = False

    return {
        "species": to_id_str(omon.species),
        "ability": to_id_str(omon.ability),
        "name": omon.name,
        "moves": list(omon.moves.keys()),
        "item": to_id_str(omon.item) if omon.item else "null",
        "gender": gender,
        "teraType": tera,
        "shiny": shiny,
        "level": omon.level,
        "stats": omon.stats,
    }
