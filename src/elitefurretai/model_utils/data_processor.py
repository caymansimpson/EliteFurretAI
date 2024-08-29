# -*- coding: utf-8 -*-
"""This module defines a random players baseline
"""

import pickle
import re

# -*- coding: utf-8 -*-
from logging import Logger
from typing import Any, Dict, Iterator, List, Union

from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.observed_pokemon import ObservedPokemon
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon_type import PokemonType
from poke_env.player import Player
from poke_env.stats import compute_raw_stats

from elitefurretai.battle_inference.inference_utils import get_showdown_identifier
from elitefurretai.model_utils.battle_data import BattleData


class DataProcessor:
    _gen_data: Dict[int, GenData] = {}

    def __init__(self):
        pass

    def stream_data(self, files: List[str]) -> Iterator[BattleData]:
        for filepath in files:
            with open(filepath, "rb") as f:
                yield pickle.loads(f.read())

    # Writes BattleData to filepath
    def write_battle_data_to_file(self, bd: BattleData, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(bd, f)

    # A one time thing to convert showdown anonymous data to BattleData
    def convert_anonymous_showdown_log_to_battledata(
        self, showdown_log: Dict[str, Any]
    ) -> BattleData:

        # Load a poke-env battle object
        battle = self.anonymous_json_to_battle(showdown_log)

        # Prepare variables
        p1_team: List[ObservedPokemon] = []
        p2_team: List[ObservedPokemon] = []
        p1_teampreview_team: List[ObservedPokemon] = []
        p2_teampreview_team: List[ObservedPokemon] = []

        p1_teampreview_team = self._json_to_observed_team(
            showdown_log["p1team"], battle.gen
        )
        p2_teampreview_team = self._json_to_observed_team(
            showdown_log["p2team"], battle.gen
        )

        # Prepare sent teams; Assumes first log is teampreview choice
        choices = map(
            lambda x: int(x) - 1,
            showdown_log["inputLog"][0].replace(">p1 team ", "").split(", "),
        )
        team = [showdown_log["p1team"][choice] for choice in choices]
        p1_team = self._json_to_observed_team(team, battle.gen)

        # Assumes second log is teampreview choice
        choices = map(
            lambda x: int(x) - 1,
            showdown_log["inputLog"][1].replace(">p2 team ", "").split(", "),
        )
        team = [showdown_log["p2team"][choice] for choice in choices]
        p2_team = self._json_to_observed_team(team, battle.gen)

        return BattleData(
            roomid=str(hash("".join(showdown_log["log"]))),
            format=showdown_log["format"],
            p1=showdown_log["p1"],
            p2=showdown_log["p2"],
            p1rating=showdown_log["p1rating"],
            p2rating=showdown_log["p2rating"],
            p1_teampreview_team=p1_teampreview_team,
            p2_teampreview_team=p2_teampreview_team,
            p1_team=p1_team,
            p2_team=p2_team,
            score=showdown_log["score"],
            winner=showdown_log["winner"],
            end_type="forced" if showdown_log["endType"] == "forced" else "normal",
            observations=battle.observations,
            inputs=showdown_log["inputLog"],
            source=BattleData.SOURCE_SHOWDOWN_ANON,
        )

    # Converts two players and a battle tag into a battle_data log
    def self_play_to_battle_data(
        self, p1: Player, p2: Player, battle_tag: str
    ) -> BattleData:

        # Get battles
        p1_battle = p1._battles[battle_tag]
        p2_battle = p2._battles[battle_tag]

        # Load any necessary data
        if p1_battle.gen not in self._gen_data:
            self._gen_data[p1_battle.gen] = GenData.from_gen(p1_battle.gen)

        # Add Teampreview information; not recorded in battle.team
        for tp_mon in p1_battle.teampreview_team:
            mon = p1_battle.team[get_showdown_identifier(tp_mon, "p1")]
            mon._terastallized_type = tp_mon.tera_type

        for tp_mon in p2_battle.teampreview_team:
            mon = p2_battle.team[get_showdown_identifier(tp_mon, "p2")]
            mon._terastallized_type = tp_mon.tera_type

        # Prepare variables
        p1_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for ident, mon in p1_battle.team.items()
        ]
        p2_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for ident, mon in p2_battle.team.items()
        ]
        p1_teampreview_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for mon in p1_battle.teampreview_team
        ]
        p2_teampreview_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for mon in p2_battle.teampreview_team
        ]

        # According to: https://github.com/smogon/pokemon-showdown/blob/b719e950f1166406a1cbf225c2a263e0848e4b0f/server/room-battle.ts#L526
        # endType can be # 'forfeit' | 'forced' | 'normal' = 'normal'
        end_type, winner = "forced", ""
        if p1_battle.won is not None:
            end_type = "normal"
            winner = p1_battle.player_role if p1_battle.won else p1_battle.opponent_role

        return BattleData(
            roomid=battle_tag,
            format=p1_battle.format,  # pyright: ignore
            p1=p1_battle.player_username,  # pyright: ignore
            p2=p2_battle.player_username,  # pyright: ignore
            p1rating=p1.prestige if hasattr(p1, "prestige") else 0,  # pyright: ignore
            p2rating=p2.prestige if hasattr(p2, "prestige") else 0,  # pyright: ignore
            p1_teampreview_team=p1_teampreview_team,
            p2_teampreview_team=p2_teampreview_team,
            p1_team=p1_team,
            p2_team=p2_team,
            score=[
                len(list(filter(lambda x: not x.fainted, p1_battle.team.values()))),
                len(list(filter(lambda x: not x.fainted, p2_battle.team.values()))),
            ],
            winner=winner,  # pyright: ignore
            end_type=end_type,
            observations=p1_battle.observations,
            inputs=[],
            source=BattleData.SOURCE_EFAI,
        )

    # Imperfect information
    def online_play_to_battle_data(self, p1: Player, battle_tag: str) -> BattleData:

        battle = p1._battles[battle_tag]

        # Load any necessary data
        if battle.gen not in self._gen_data:
            self._gen_data[battle.gen] = GenData.from_gen(battle.gen)

        # Prepare variables
        p1_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for ident, mon in battle.team.items()
        ]
        p2_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon)
            for ident, mon in battle.opponent_team.items()
        ]
        p1_teampreview_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for mon in battle.teampreview_team
        ]
        p2_teampreview_team: List[ObservedPokemon] = [
            ObservedPokemon.from_pokemon(mon) for mon in battle.teampreview_opponent_team
        ]

        # According to: https://github.com/smogon/pokemon-showdown/blob/b719e950f1166406a1cbf225c2a263e0848e4b0f/server/room-battle.ts#L526
        # endType can be # 'forfeit' | 'forced' | 'normal' = 'normal'
        end_type, winner = "forced", ""
        if battle.won is not None:
            end_type = "normal"
            winner = battle.player_role if battle.won else battle.opponent_role

        return BattleData(
            roomid=battle_tag,
            format=battle.format,  # pyright: ignore
            p1=battle.player_username,  # pyright: ignore
            p2=battle.opponent_username,  # pyright: ignore
            p1rating=p1.prestige if hasattr(p1, "prestige") else 0,  # pyright: ignore
            p2rating=battle.opponent_rating if battle.opponent_rating else 0,
            p1_teampreview_team=p1_teampreview_team,
            p2_teampreview_team=p2_teampreview_team,
            p1_team=p1_team,
            p2_team=p2_team,
            score=[
                len(list(filter(lambda x: not x.fainted, battle.team.values()))),
                len(list(filter(lambda x: not x.fainted, battle.opponent_team.values()))),
            ],
            winner=winner,  # pyright: ignore
            end_type=end_type,
            observations=battle.observations,
            inputs=[],
            source=BattleData.SOURCE_SHOWDOWN,
        )

    def anonymous_json_to_battle(self, showdown_log) -> Union[Battle, DoubleBattle]:
        match = re.match("(gen[0-9])", str(showdown_log.get("format")))
        if match is None:
            raise ValueError(
                "Could not parse gen from battle json's format: {format}".format(
                    format=showdown_log.get("format")
                )
            )
        gen = int(match.groups()[0][-1])

        # Load any necessary data
        if gen not in self._gen_data:
            self._gen_data[gen] = GenData.from_gen(gen)

        # Create battle
        battle = None
        if "vgc" in str(showdown_log.get("format")) or "doubles" in str(
            showdown_log.get("format")
        ):
            battle = DoubleBattle(
                "tag", showdown_log["p1"], Logger("elitefurretai"), gen=gen
            )
        else:
            battle = Battle("tag", showdown_log["p1"], Logger("elitefurretai"), gen=gen)

        # Set roles and usernames
        battle.player_role = "p1"
        battle._player_username = showdown_log["p1"]
        battle._opponent_username = showdown_log["p2"]

        # Prepare teampreview teams
        obs_teampreview_team = self._json_to_observed_team(
            showdown_log["p1team"], battle.gen
        )
        teampreview_team = set()
        for omon in obs_teampreview_team:
            teampreview_team.add(self._observed_mon_to_mon(omon, gen))
        battle.teampreview_team = teampreview_team

        obs_opp_teampreview_team = self._json_to_observed_team(
            showdown_log["p2team"], battle.gen
        )
        opp_teampreview_team = set()
        for omon in obs_opp_teampreview_team:
            opp_teampreview_team.add(self._observed_mon_to_mon(omon, gen))
        battle._teampreview_opponent_team = opp_teampreview_team

        # Prepare sent teams; assumes p1 team choice is first log
        choices = map(
            lambda x: int(x) - 1,
            showdown_log["inputLog"][0].replace(">p1 team ", "").split(", "),
        )
        team = {}
        for omon in [obs_teampreview_team[choice] for choice in choices]:
            mon = self._observed_mon_to_mon(omon, gen)
            team[get_showdown_identifier(mon, "p1")] = mon
        battle.team = team

        # Assumes p2 team choice is second log
        choices = map(
            lambda x: int(x) - 1,
            showdown_log["inputLog"][1].replace(">p2 team ", "").split(", "),
        )
        opp_team = {}
        for omon in [obs_opp_teampreview_team[choice] for choice in choices]:
            mon = self._observed_mon_to_mon(omon, gen)
            opp_team[get_showdown_identifier(mon, "p2")] = mon
        battle._opponent_team = opp_team

        # Go through events to reconstruct battle state and Observations
        logs = showdown_log["log"]
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

        return battle

    def _json_to_observed_team(
        self, team_list: List[Dict[str, Any]], gen: int
    ) -> List[ObservedPokemon]:
        return [self._json_to_observed_mon(mon_info, gen) for mon_info in team_list]

    def _json_to_observed_mon(self, mon_info: Dict[str, Any], gen: int) -> ObservedPokemon:
        species = to_id_str(mon_info["species"])

        ability = None
        tera = None
        item = None
        moves = {}

        # Generate stats
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

        ability = to_id_str(mon_info["ability"])
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

        return ObservedPokemon(
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

    def _observed_mon_to_mon(self, omon: ObservedPokemon, gen: int) -> Pokemon:
        mon = Pokemon(gen=gen, species=omon.species)
        mon._moves = {k: v for k, v in omon.moves.items()}
        mon._item = omon.item
        mon._gender = omon.gender
        mon._terastallized_type = omon.tera_type
        mon._shiny = omon.shiny
        mon._level = omon.level
        if omon.stats and "hp" in omon.stats and isinstance(omon.stats["hp"], int):
            mon.stats = {k: v for k, v in omon.stats.items()}  # pyright: ignore
        return mon
