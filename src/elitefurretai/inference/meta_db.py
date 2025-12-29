# -*- coding: utf-8 -*-
"""This module returns probabilities of infostates, from information learned in battle logs."""

import os.path
import re
import sqlite3
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from poke_env.battle import Move, ObservedPokemon, PokemonGender, PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

_DATABASE_REL_PATH: str = "data/database"


class MetaDB:
    POKEMON_SCHEMA = [
        "mon_id",
        "species",
        "gender",
        "tera_type",
        "item",
        "hp",
        "atk",
        "def",
        "spa",
        "spd",
        "spe",
        "ability",
        "level",
        "move1",
        "move2",
        "move3",
        "move4",
    ]

    TEAM_SCHEMA = ["team_id", "mon_id"]

    BATTLE_SCHEMA = [
        "battle_id",
        "p1_rating",
        "p2_rating",
        "team1_id",
        "team2_id",
        "format",
    ]

    TEAM_COUNTS_SCHEMA = ["team_id", "format", "num"]

    def __init__(self, db: str = "frisk.db", write: bool = False, v: bool = False):
        self._db = db
        self._write = write  # Just as a safeguard for scripting
        self._conn: Optional[sqlite3.Connection] = None
        self._v = v

        current_file = os.path.dirname(os.path.abspath(__file__))  # Gets current directory
        three_up = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        self._DB_ABS_PATH = os.path.join(three_up, _DATABASE_REL_PATH)

        self.open()

    def drop_all_tables(self, gen=9):
        if not self._db.startswith("test_"):
            response = input(
                f"Are you sure you want to drop all tables of gen {gen}? If you do, answer 'hell yea'?\n"
            )
            if response != "hell yea":
                raise RuntimeError(
                    f"User responded '{response}' instead of 'hell yea' which means they didnt want to drop all tables"
                )

        tables = self.query("SELECT name FROM sqlite_schema WHERE type='table';")
        for table in tables if tables else []:
            if f"gen{gen}" in table[0]:
                self.query(f"DROP TABLE IF EXISTS {table[0]};")

    def create_new_tables(self, gen=9):
        self.query(
            f"""
        CREATE TABLE IF NOT EXISTS team_gen{gen} (
            team_id INTEGER NOT NULL,
            mon_id INTEGER NOT NULL,
            UNIQUE(team_id, mon_id)
        );"""
        )

        self.query(
            f"""
        CREATE TABLE IF NOT EXISTS pokemon_gen{gen} (
            mon_id INTEGER PRIMARY KEY,
            species TEXT NOT NULL,
            gender TEXT,
            tera_type TEXT,
            item TEXT,
            hp INTEGER NOT NULL,
            atk INTEGER NOT NULL,
            def INTEGER NOT NULL,
            spa INTEGER NOT NULL,
            spd INTEGER NOT NULL,
            spe INTEGER NOT NULL,
            ability TEXT NOT NULL,
            level INTEGER NOT NULL,
            move1 TEXT,
            move2 TEXT,
            move3 TEXT,
            move4 TEXT,
            UNIQUE(mon_id)
        );"""
        )

        self.query(
            f"""
        CREATE TABLE IF NOT EXISTS team_counts_gen{gen} (
            team_id INTEGER PRIMARY KEY,
            format TEXT NOT NULL,
            num INTEGER NOT NULL,
            UNIQUE(team_id, format)
        );"""
        )

        self.query(
            f"""
        CREATE TABLE IF NOT EXISTS battle_gen{gen} (
            battle_id INTEGER PRIMARY KEY,
            p1_rating INTEGER,
            p2_rating INTEGER,
            team1_id INTEGER,
            team2_id INTEGER,
            format TEXT,
            UNIQUE(battle_id)
        );"""
        )

        self.query(
            f"""
        CREATE INDEX team_id
        ON team_gen{gen}(team_id);
        """
        )

    def query(self, query) -> Optional[List[Any]]:
        if self._v:
            print("query:", query)

        try:
            if self._conn is None:
                raise AttributeError("connection not initialized; _conn is None")
            elif "insert" in query.lower() or "create" in query.lower():
                if not self._write:
                    raise ValueError("Cannot insert into database")
                else:
                    self._conn.cursor().execute(query)
                    self._conn.commit()
            else:
                return self._conn.cursor().execute(query).fetchall()
        except Exception as e:
            print("Error! Query:", query)
            print("Error:", e)
        return None

    def write(self, table: str, data: List[Any]):
        assert (
            table.startswith("team_gen")
            or table.startswith("pokemon_gen")
            or table.startswith("battle_gen")
            or table.startswith("team_counts_gen")
        )

        if "battle_gen" in table:
            schema = self.BATTLE_SCHEMA
        elif "team_gen" in table:
            schema = self.TEAM_SCHEMA
        elif "pokemon_gen" in table:
            schema = self.POKEMON_SCHEMA
        else:
            schema = self.TEAM_COUNTS_SCHEMA

        values = ""
        for val in data:
            if isinstance(val, str):
                values += f'"{val}", '
            elif isinstance(val, PokemonType):
                values += f'"{str(val.name)}", '
            else:
                values += f"{str(val)}, "
        values = values[:-2]

        self.query(
            f"INSERT INTO {table} ({', '.join(schema)}) VALUES ({values}) ON CONFLICT DO NOTHING;"
        )

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def open(self):
        self._conn = sqlite3.connect(os.path.join(self._DB_ABS_PATH, self._db))

    # Given a teampreview, predicts the spreads of the team.
    def predict_vgc_team(
        self, observed: List[ObservedPokemon], battle_format: str, probabilities=False
    ) -> Union[List[ObservedPokemon], List[Tuple[List[ObservedPokemon], float]]]:
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
        WITH pokemon as (
            SELECT *
            FROM pokemon_gen{gen}
            WHERE
                {team_clause}
        ),
        team_mon as (
            SELECT team_id, pokemon.*
            FROM team_gen{gen}
            INNER JOIN pokemon
            ON team_gen{gen}.mon_id = pokemon.mon_id
        )

        SELECT
            teams.team_id,
            counts.num,
            team_mon.mon_id,
            team_mon.species,
            team_mon.gender,
            team_mon.tera_type,
            team_mon.item,
            team_mon.hp,
            team_mon.atk,
            team_mon.def,
            team_mon.spa,
            team_mon.spd,
            team_mon.spe,
            team_mon.ability,
            team_mon.level,
            team_mon.move1,
            team_mon.move2,
            team_mon.move3,
            team_mon.move4
        FROM (
            SELECT team_id
            FROM team_mon
            GROUP BY team_id
            HAVING COUNT(*) == {team_length}
        ) teams
        INNER JOIN (
            SELECT team_id, num
            FROM team_counts_gen{gen}
            WHERE format = '{format}'
        ) counts
        ON counts.team_id = teams.team_id
        INNER JOIN team_mon
        ON teams.team_id = team_mon.team_id
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
            teams: Dict[str, List[ObservedPokemon]] = {}
            counts = {}

            # Go through every row (pokemon) and construct teams
            for row in results:
                counts[row[0]] = float(row[1])

                team = teams.get(row[0], [])
                team.append(self.create_observed_pokemon(row[2:], gen))
                teams[row[0]] = team

            # Normalize the team counts
            total = sum(counts.values())
            for team_id in counts:
                counts[team_id] = counts[team_id] * 1.0 / total

            return [(teams[team_id], counts[team_id]) for team_id in teams]

        else:
            # Filter results to the most common team
            team_id = sorted(results, key=lambda row: -1 * row[1])[0][0]
            relevant_results = filter(lambda row: row[0] == team_id, results)

            # Return the team
            return [self.create_observed_pokemon(row[2:], gen) for row in relevant_results]

    def _construct_team_clause(self, mons: List[ObservedPokemon]) -> str:
        """
        Given a list of ObservedPokemon, constructs a SQL clause that will match
        the team.
        """
        return "\n\t\t\tOR ".join([self._construct_pokemon_clause(mon) for mon in mons])

    def _construct_pokemon_clause(self, mon: ObservedPokemon) -> str:
        """
        Given an ObservedPokemon, constructs a SQL clause that will match
        the pokemon.
        """
        ability = to_id_str(mon.ability) if mon.ability else "NULL"
        item = (
            to_id_str(mon.item)
            if mon.item and mon.item != GenData.UNKNOWN_ITEM
            else "NULL"
        )
        # gender = mon.gender.name.lower() if mon.gender else "NULL" TODO: changed so i dont store gender
        tera_type = mon.tera_type.name if mon.tera_type else "NULL"

        clause = "\t\t\tspecies = '{species}'".format(species=mon.species.lower())
        if mon.level:
            clause += "\n\t\t\tAND level = {level}".format(level=mon.level)
        if ability != "NULL":
            clause += "\n\t\t\tAND ability = '{ability}'".format(ability=ability)
        if item != "NULL":
            clause += "\n\t\t\tAND item = '{item}'".format(item=item)
        # if gender != "NULL":
        #     clause += "\n\t\t\tAND gender = '{gender}'".format(gender=gender)
        if tera_type != "NULL":
            clause += "\n\t\t\tAND tera_type = '{teratype}'".format(teratype=tera_type)

        # Have to compare every move against itself
        move_clause = []
        if len(mon.moves) > 0:
            for move in mon.moves.keys():
                move_clause.append(
                    "("
                    + " OR ".join(
                        "move{i} = '{move}'".format(i=i, move=move) for i in range(1, 5)
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
                    clause += "\n\t\t\tAND {stat} BETWEEN {min} AND {max}".format(
                        stat=k, min=v[0], max=v[1]
                    )
                elif v is not None:
                    clause += "\n\t\t\tAND {stat} = {value}".format(stat=k, value=str(v))

        return "(\n" + clause + "\n\t\t\t)"

    @staticmethod
    def create_observed_pokemon(row: List[Any], gen: int) -> ObservedPokemon:
        stats = {
            "hp": row[5],
            "atk": row[6],
            "def": row[7],
            "spa": row[8],
            "spd": row[9],
            "spe": row[10],
        }

        moves = OrderedDict()
        moves[row[13]] = Move(row[13], gen=gen)
        if len(row) > 14 and row[14] != "NULL":
            moves[row[14]] = Move(row[14], gen=gen)
        if len(row) > 15 and row[15] != "NULL":
            moves[row[15]] = Move(row[15], gen=gen)
        if len(row) > 16 and row[16] != "NULL":
            moves[row[16]] = Move(row[16], gen=gen)

        return ObservedPokemon(
            species=row[1],
            level=row[12],
            name=row[1],
            stats=stats,
            moves=moves,
            ability=row[11],
            item=row[4],
            gender=PokemonGender.FEMALE,
            tera_type=PokemonType.from_name(row[3]) if row[3] else None,
            shiny=False,
        )
