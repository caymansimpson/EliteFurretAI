# -*- coding: utf-8 -*-
"""This module returns probabilities of infostates, from information learned in battle logs.
"""

import os.path
import re
import sqlite3
from typing import Any, List, Optional, Tuple, Union

from poke_env.environment import (
    DoubleBattle,
    Move,
    ObservedPokemon,
    PokemonGender,
    PokemonType,
)


class MetaDB:

    _DATABASE_DIR: str = "data/database"

    def __init__(self, db: str = "frisk.db", write: bool = False):
        self._db = db
        self._write = write  # Just as a safeguard for scripting
        self._conn = None
        self.open()

    def query(self, query) -> Optional[List[Any]]:
        if self._conn is None:
            raise AttributeError("connection not initialized; _conn is None")
        elif "insert" in query.lower():
            if not self._write:
                raise ValueError("Cannot insert into database")
            else:
                self._conn.cursor().execute(query)
                self._conn.commit()
        else:
            return self._conn.cursor().execute(query).fetchall()

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def open(self):
        self._conn = sqlite3.connect(os.path.join(self._DATABASE_DIR, self._db))

    # Given a teampreview, predicts the spreads of the team.
    def predict_vgc_team(
        self, battle: DoubleBattle, battle_format: str, probabilities=False
    ) -> Union[List[ObservedPokemon], List[Tuple[ObservedPokemon, float]]]:

        # Get all information about the pokemon and store them in ObservedPokemon
        observed = []
        for mon in battle._teampreview_opponent_team:
            observed.append(
                ObservedPokemon.from_pokemon(battle.opponent_team.get(mon.species, mon))
            )

        # Get generation
        match = re.match("(gen[0-9])", battle_format)
        if match is None:
            raise ValueError(
                "Could not parse gen from battle json's format: {format}".format(
                    format=battle_format
                )
            )
        gen = int(match.groups()[0][-1])

        query = """
        SELECT
            teams.team_id,
            counts.num,
            pokemon.mon_id,
            pokemon.species,
            pokemon.level,
            pokemon.ability,
            pokemon.item,
            pokemon.gender,
            pokemon.shiny,
            pokemon.tera_type,
            pokemon.move1_id,
            pokemon.move2_id,
            pokemon.move3_id,
            pokemon.move4_id,
            pokemon.hp,
            pokemon.atk,
            pokemon.def,
            pokemon.spa,
            pokemon.spd,
            pokemon.spe
        FROM (
            SELECT team_id
            FROM team_gen{gen}
            INNER JOIN pokemon_gen{gen} pokemon
            ON team_gen{gen}.mon_id = pokemon.mon_id
            WHERE
                {team_clause}
            GROUP BY team_id
            HAVING COUNT(*) == {team_length}
        ) teams
        INNER JOIN team_gen{gen}
        ON teams.team_id = team_gen{gen}.team_id
        INNER JOIN pokemon_gen{gen} pokemon
        ON team_gen{gen}.mon_id = pokemon.mon_id
        INNER JOIN (
            SELECT team_id, num
            FROM team_counts_gen{gen}
            WHERE format = '{format}'
        ) counts
        ON counts.team_id = teams.team_id
        ORDER BY counts.num DESC
        """.format(
            gen=gen,
            team_clause=self._construct_team_clause(observed),
            team_length=len(observed),
            format=battle_format,
        )

        results = self.query(query)

        if results is None:
            raise ValueError("Query did not even return empty results")

        # No matches, so we return no matches
        if len(results) == 0:
            return []

        elif probabilities:
            print(results)
            teams = {}
            counts = {}

            # Go through every row (pokemon) and construct teams
            for row in results:
                counts[row[0]] = float(row[1])

                team = teams.get(row[0], [])
                team.append(self._create_observed_pokemon(row, gen))
                teams[row[0]] = team

            # Normalize the team counts
            total = sum(counts.values())
            for team_id in counts:
                counts[team_id] = counts[team_id] * 1.0 / total

            return [(teams[team_id], counts[team_id]) for team_id in teams]

        else:
            # Filter results to the most common team (it's the first row cuz of the ORDER BY)
            relevant_results = filter(lambda row: row[1] == results[0][1], results)

            # Return the team
            return [self._create_observed_pokemon(row, gen) for row in relevant_results]

    def _construct_team_clause(self, mons: List[ObservedPokemon]) -> str:
        """
        Given a list of ObservedPokemon, constructs a SQL clause that will match
        the team.
        """
        return "\n\t\tOR ".join([self._construct_pokemon_clause(mon) for mon in mons])

    def _construct_pokemon_clause(self, mon: ObservedPokemon) -> str:
        """
        Given an ObservedPokemon, constructs a SQL clause that will match
        the pokemon.
        """
        ability = mon.ability.lower() if mon.ability else "NULL"
        item = mon.item.lower() if mon.item else "NULL"
        gender = mon.gender.name.lower() if mon.gender else "NULL"
        tera_type = mon.tera_type.name.lower() if mon.tera_type else "NULL"
        shiny = str(mon.shiny).upper() if mon.shiny else "NULL"

        # Because shiny isn't yet a battle-dependent characteristic, will leave this out of the
        # query for now. Will add it back in future generations if interactions depend on this
        # shiny = str(mon.shiny).upper() if mon.shiny is not None else "NULL"

        clause = "\t\t\tpokemon.species = '{species}'".format(species=mon.species.lower())
        if mon.level:
            clause += "\n\t\t\tAND pokemon.level = {level}".format(level=mon.level)
        if ability != "NULL":
            clause += "\n\t\t\tAND pokemon.ability = '{ability}'".format(ability=ability)
        if item != "NULL":
            clause += "\n\t\t\tAND pokemon.item = '{item}'".format(item=item)
        if gender != "NULL":
            clause += "\n\t\t\tAND pokemon.gender = '{gender}'".format(gender=gender)
        if tera_type != "NULL":
            clause += "\n\t\t\tAND pokemon.tera_type = '{teratype}'".format(
                teratype=tera_type
            )
        if shiny != "NULL":
            clause += "\n\t\t\tAND pokemon.shiny = {shiny}".format(shiny=shiny)

        # Have to compare every move against itself
        move_clause = []
        if len(mon.moves) > 0:
            for move in mon.moves.keys():
                move_clause.append(
                    "("
                    + " OR ".join(
                        "pokemon.move{i}_id = '{move}'".format(i=i, move=move)
                        for i in range(1, 5)
                    )
                    + ")"
                )

            clause += (
                "\n\t\t\tAND (\n\t\t\t\t"
                + "\n\t\t\t\tAND ".join(move_clause)
                + "\n\t\t\t)"
            )

        if mon.stats:
            for k, v in mon.stats.items():
                if isinstance(v, list):
                    clause += "\n\t\t\tAND pokemon.{stat} BETWEEN {min} AND {max}".format(
                        stat=k, min=v[0], max=v[1]
                    )
                else:
                    clause += "\n\t\t\tAND pokemon.{stat} = {value}".format(
                        stat=k, value=str(v)
                    )

        return "(\n" + clause + "\n\t\t)"

    def _create_observed_pokemon(self, row: List[Any], gen: int) -> ObservedPokemon:
        stats = {
            "hp": row[14],
            "atk": row[15],
            "def": row[16],
            "spa": row[17],
            "spd": row[18],
            "spe": row[19],
        }
        moves = {
            row[10]: Move(row[10], gen=gen),
            row[11]: Move(row[11], gen=gen),
            row[12]: Move(row[12], gen=gen),
            row[13]: Move(row[13], gen=gen),
        }

        return ObservedPokemon(
            species=row[3],
            level=row[4],
            stats=stats,
            moves=moves,
            ability=row[5],
            item=row[6],
            gender=PokemonGender[row[7].upper()],
            tera_type=PokemonType.from_name(row[9]) if row[9] else None,
            shiny=row[8] == 1,
        )
