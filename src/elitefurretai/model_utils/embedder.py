# -*- coding: utf-8 -*-
"""This module defines a class that Embeds objects
"""

# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

from poke_env.data import GenData
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.effect import Effect
from poke_env.environment.field import Field
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.environment.target import Target
from poke_env.environment.weather import Weather


class Embedder:

    def __init__(self, gen=9):
        self._knowledge: Dict[str, Any] = {}
        self._gen_data = GenData.from_gen(gen)
        sets = [
            ("Field", Field),
            ("SideCondition", SideCondition),
            ("Status", Status),
            ("Weather", Weather),
            ("PokemonType", PokemonType),
            ("PokemonGender", PokemonGender),
            ("MoveCategory", MoveCategory),
            ("Target", Target),
            ("Effect", Effect),
            ("Effect_VolatileStatus", Effect),
        ]

        self._FORMATS: List[str] = [
            "gen6doubblesou",
            "gen9vgc2024regf",
            "gen9vgc2024regg",
        ]

        for key, enum in sets:
            if key == "Effect_VolatileStatus":
                values = list(filter(lambda x: x.is_volatile_status, list(enum)))
                self._knowledge[key] = set(values)
            else:
                self._knowledge[key] = set(enum)

        self._knowledge["Pokemon"] = set(self._gen_data.pokedex.keys())

        abilities = set()
        for mon in self._gen_data.pokedex.values():
            abilities.update(mon["abilities"].values())
        self._knowledge["Ability"] = abilities

        self._MOVE_LEN = 197
        self._POKEMON_LEN = 2715
        self._OPPONENT_POKEMON_LEN = 2715
        self._TEAMPREVIEW_LEN = 100
        self._DOUBLEBATTLE_LEN = 100

    def _prep(self, string) -> str:
        return string.lower().replace("_", " ")

    def embed_move(self, move: Optional[Move]) -> List[int]:
        """
        Returns a list of integers representing the move
        """

        # If the move is None or empty, return a negative array (filled w/ -1's)
        if move is None or move.is_empty:
            return [-1] * self._MOVE_LEN

        embeddings = []

        embeddings.append(
            [
                move.accuracy,
                move.base_power,
                int(move.breaks_protect),
                move.crit_ratio,
                move.current_pp,
                move.damage,
                move.drain,
                move.dynamaxed,
                move.expected_hits,
                int(move.force_switch),
                move.heal,
                int(move.ignore_ability),
                int(move.ignore_defensive),
                int(move.ignore_evasion),
                1 if move.ignore_immunity else 0,
                int(move.is_protect_counter),
                int(move.is_protect_move),
                int(move.is_side_protect_move),
                move.n_hit[0] if move.n_hit else 1,  # minimum times the move hits
                move.n_hit[1] if move.n_hit else 1,  # maximum times the move hits
                move.priority,
                move.recoil,
                int(move.self_destruct is not None),
                int(move.self_switch is not None),
                int(move.steals_boosts),
                int(move.thaws_target),
                int(move.use_target_offensive),
            ]
        )

        # Add Category
        embeddings.append(
            [
                int(move.category == category)
                for category in self._knowledge["MoveCategory"]
            ]
        )

        # Add Defensive Category
        embeddings.append(
            [
                int(move.defensive_category == category)
                for category in self._knowledge["MoveCategory"]
            ]
        )

        # Add Move Type
        embeddings.append(
            [
                int(move.type == pokemon_type)
                for pokemon_type in self._knowledge["PokemonType"]
            ]
        )

        # Add Fields
        embeddings.append(
            [int(move.terrain == field) for field in self._knowledge["Field"]]
        )

        # Add Side Conditions
        embeddings.append(
            [int(move.side_condition == sc) for sc in self._knowledge["SideCondition"]]
        )

        # Add Weathers
        embeddings.append(
            [int(move.weather == weather) for weather in self._knowledge["Weather"]]
        )

        # Add Targeting Types; cardinality is 14
        embeddings.append(
            [int(move.deduced_target == t) for t in self._knowledge["Target"]]
        )

        # Add Volatility Statuses; cardinality is 57
        volatility_status_embeddings = []
        for vs in self._knowledge["Effect_VolatileStatus"]:
            if vs == move.volatile_status:
                volatility_status_embeddings.append(1)
            elif move.secondary and self._prep(vs.name) in list(
                map(lambda x: self._prep(x.get("volatileStatus", "")), move.secondary)
            ):
                volatility_status_embeddings.append(1)
            else:
                volatility_status_embeddings.append(0)
        embeddings.append(volatility_status_embeddings)

        # Add Statuses
        status_embeddings = []
        for status in self._knowledge["Status"]:
            if status == move.status:
                status_embeddings.append(1)
            elif move.secondary and self._prep(status.name) in list(
                map(lambda x: self._prep(x.get("status", "")), move.secondary)
            ):
                status_embeddings.append(1)
            else:
                status_embeddings.append(0)
        embeddings.append(status_embeddings)

        # Add Boosts
        boost_embeddings = {
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "evasion": 0,
            "accuracy": 0,
        }
        if move.boosts:
            for stat in move.boosts:
                boost_embeddings[stat] += move.boosts[stat]
        elif move.secondary:
            for x in move.secondary:
                for stat in x.get("boosts", {}):
                    boost_embeddings[stat] += x["boosts"][stat]
        embeddings.append(boost_embeddings.values())

        # Add Self-Boosts
        self_boost_embeddings = {
            "atk": 0,
            "def": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "evasion": 0,
            "accuracy": 0,
        }
        if move.self_boost:
            for stat in move.self_boost:
                self_boost_embeddings[stat] += move.self_boost[stat]
        elif move.secondary:
            for x in move.secondary:
                for stat in x.get("self", {}).get("boosts", {}):
                    self_boost_embeddings[stat] += x["self"]["boosts"][stat]
        embeddings.append(self_boost_embeddings.values())

        # Introduce the chance of a secondary effect happening
        chance = 0
        for x in move.secondary:
            chance = max(chance, x.get("chance", 0))
        embeddings.append([chance])

        return [item for sublist in embeddings for item in sublist]

    # TODO: add item
    def embed_pokemon(self, mon: Pokemon) -> List[int]:
        """
        Returns a list of integers representing the pokemon
        """

        # If the mon is None, return a negative array (filled w/ -1's)
        if mon is None:
            return [-1] * self._POKEMON_LEN

        embeddings = []

        # Append moves to embedding (and account for the fact that the mon might have <4 moves)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self.embed_move(move))

        # OHE mons
        embeddings.append(
            [int(mon.species == pokemon) for pokemon in self._knowledge["Pokemon"]]
        )

        # OHE abilities
        embeddings.append(
            [int(mon.ability == ability) for ability in self._knowledge["Ability"]]
        )

        # Add the current hp, whether its fainted, its level, its weight and whether its recharging or preparing
        embeddings.append(
            [
                mon.current_hp,
                int(mon.fainted),
                mon.level,
                mon.weight,
                int(mon.must_recharge),
                1 if mon.preparing else 0,
                int(mon.is_dynamaxed),
                int(mon.is_terastallized),
            ]
        )

        # Add stats and boosts
        embeddings.append(mon.stats.values() if mon.stats else [-1] * 6)
        embeddings.append(mon.boosts.values())

        # Add Gender
        embeddings.append(
            [int(mon.gender == gender) for gender in self._knowledge["PokemonGender"]]
        )

        # Add status (one-hot encoded)
        embeddings.append(
            [int(mon.status == status) for status in self._knowledge["Status"]]
        )

        # Add Types (one-hot encoded)
        embeddings.append(
            [
                int(mon.type_1 == pokemon_type)
                for pokemon_type in self._knowledge["PokemonType"]
            ]
        )
        embeddings.append(
            [
                int(mon.type_2 is not None and mon.type_2 == pokemon_type)
                for pokemon_type in self._knowledge["PokemonType"]
            ]
        )

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    # TODO: add inferred stat ranges, item
    def embed_opponent_pokemon(self, mon: Pokemon) -> List[int]:
        """
        Returns a list of integers representing the opponents pokemon
        """
        embeddings = []

        # Append moves to embedding (and account for the fact that the mon might have <4 moves, or we don't know of them)
        for move in (list(mon.moves.values()) + [None, None, None, None])[:4]:
            embeddings.append(self.embed_move(move))

        # OHE mons
        embeddings.append(
            [int(mon.species == pokemon) for pokemon in self._knowledge["Pokemon"]]
        )

        # OHE possible abilities
        embeddings.append(
            [
                int(ability in mon.possible_abilities)
                for ability in self._knowledge["Ability"]
            ]
        )

        # Add whether the mon is active, the current hp, whether its fainted, its level, its weight and whether its recharging or preparing
        embeddings.append(
            [
                mon.current_hp,
                int(mon.fainted),
                mon.level,
                mon.weight,
                int(mon.must_recharge),
                1 if mon.preparing else 0,
                int(mon.is_dynamaxed),
            ]
        )

        # Add stats and boosts
        embeddings.append(mon.base_stats.values())
        embeddings.append(mon.boosts.values())

        # Add status (one-hot encoded)
        embeddings.append(
            [int(mon.status == status) for status in self._knowledge["Status"]]
        )

        # Add Types (one-hot encoded)
        embeddings.append(
            [
                int(mon.type_1 == pokemon_type)
                for pokemon_type in self._knowledge["PokemonType"]
            ]
        )
        embeddings.append(
            [
                int(mon.type_2 is not None and mon.type_2 == pokemon_type)
                for pokemon_type in self._knowledge["PokemonType"]
            ]
        )

        # Flatten all the lists into a Nx1 list
        return [item for sublist in embeddings for item in sublist]

    # TODO: account for when opponent has <6 mons
    # TODO: might need to pull in player information so that we have the ability to remember
    # our teampreview team so AI might learn how to fake out other players
    def embed_double_battle(self, battle: DoubleBattle) -> List[int]:
        """
        Returns a list of integers representing the state of the battle, at the beginning
        of the specified turn. It is from the perspective of the player whose turn it is.
        """
        embeddings = []

        # Add each of our mons to embeddings. We want to add even our teampreview pokemon because
        # our opponent may make moves dependent on this information
        for mon in battle.teampreview_team:
            embeddings.append(self.embed_pokemon(mon))

            # Record whether a pokemon is active, or whether it has been brought
            sent = int(mon.species in battle.team.keys()) if mon else 0
            active = int(
                mon.species
                in map(lambda x: x.species if x else None, battle.active_pokemon)
            )
            embeddings.append([sent, active])

        # Embed each opponent mon
        for mon in battle.teampreview_opponent_team:
            embeddings.append(self.embed_opponent_pokemon(mon))

            # Record whether a pokemon is active, or whether it has been brought, could have been brought or isnt brought
            # This is VGC specific; can implement using battle.format to make a variable w/ required # of pokemon length
            seen = int(
                mon.species
                in set(map(lambda x: x.species, battle.teampreview_opponent_team))
            )
            if seen == 0 and len(battle.opponent_team) == 4:
                seen = -1
            active = int(
                mon.species
                in map(lambda x: x.species if x else None, battle.opponent_active_pokemon)
            )
            embeddings.append([seen, active])

        embeddings.append(battle.trapped)
        embeddings.append(battle.force_switch)
        embeddings.append(battle.can_mega_evolve)
        embeddings.append(battle.can_z_move)
        embeddings.append(battle.can_dynamax)
        embeddings.append(battle.can_tera)
        embeddings.append(battle.maybe_trapped)

        embeddings.append(battle.opponent_can_dynamax)
        embeddings.append(battle.opponent_can_mega_evolve)
        embeddings.append(battle.opponent_can_z_move)
        embeddings.append(battle.opponent_can_z_move)
        embeddings.append([battle.opponent_dynamax_turns_left, int(battle.teampreview)])

        # Add Fields;
        embeddings.append(
            [int(field in battle.fields) for field in self._knowledge["Field"]]
        )

        # Add Side Conditions
        embeddings.append(
            [int(sc in battle.side_conditions) for sc in self._knowledge["SideCondition"]]
        )
        embeddings.append(
            [
                int(sc in battle.opponent_side_conditions)
                for sc in self._knowledge["SideCondition"]
            ]
        )

        # Add Weathers
        embeddings.append(
            [int(weather == battle.weather) for weather in self._knowledge["Weather"]]
        )

        # Add Formats
        embeddings.append([int(frmt == battle.format) for frmt in self._FORMATS])

        # Add Player Ratings, the battle's turn and a bias term
        embeddings.append(
            list(
                map(
                    lambda x: x if x else -1,
                    [battle.rating, battle.opponent_rating, battle.turn, 1],
                )
            )
        )

        # Flatten all the lists into a 7814-dim list
        return [item for sublist in embeddings for item in sublist]
