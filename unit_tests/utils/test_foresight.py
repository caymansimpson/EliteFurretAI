# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

from poke_env.environment import Move, ObservedPokemon, PokemonGender, PokemonType

from elitefurretai.utils.foresight import Foresight


def test_query():
    f = Foresight("test_frisk.db", write=True)

    # Set up database
    f.query("""DROP TABLE IF EXISTS team_gen9;""")
    f.query("""DROP TABLE IF EXISTS pokemon_gen9;""")

    f.query(
        """
    CREATE TABLE IF NOT EXISTS team_gen9 (
        team_id INTEGER NOT NULL,
        mon_id INTEGER NOT NULL
    );"""
    )

    f.query(
        """
    CREATE TABLE IF NOT EXISTS pokemon_gen9 (
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
        shiny BOOLEAN NOT NULL,
        move1_id TEXT,
        move2_id TEXT,
        move3_id TEXT,
        move4_id TEXT
    );"""
    )

    # Insert into database
    f.query("INSERT INTO team_gen9 (team_id, mon_id) VALUES(1, 1);")
    f.query(
        """
    INSERT INTO pokemon_gen9 (mon_id, species, gender, tera_type, item, hp, atk, def, spa, spd, spe, ability, level, shiny, move1_id, move2_id, move3_id, move4_id)
    VALUES(1, "furret", "f", "normal", "focussash", 100, 100, 100, 100, 100, 100, "frisk", 100, TRUE, "endeavor", "followme", "protect", "superfang"
    );"""
    )

    # Query database
    results = f.query(
        """
    SELECT team_gen9.team_id, species
    FROM team_gen9
    JOIN pokemon_gen9
    ON team_gen9.mon_id = pokemon_gen9.mon_id
    """
    )

    # Verify that the query worked
    assert results[0][0] == 1
    assert results[0][1] == "furret"

    f.close()


def test_predict_vgc_team():
    f = Foresight("test_frisk.db", write=True)

    # Create database and data structures
    f.query("""DROP TABLE IF EXISTS team_gen9;""")
    f.query("""DROP TABLE IF EXISTS pokemon_gen9;""")
    f.query("""DROP TABLE IF EXISTS team_counts_gen9;""")

    f.query(
        """
    CREATE TABLE IF NOT EXISTS team_gen9 (
        team_id INTEGER NOT NULL,
        mon_id INTEGER NOT NULL
    );"""
    )

    f.query(
        """
    CREATE TABLE IF NOT EXISTS pokemon_gen9 (
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
        shiny BOOLEAN NOT NULL,
        move1_id TEXT,
        move2_id TEXT,
        move3_id TEXT,
        move4_id TEXT
    );"""
    )

    f.query(
        """
    CREATE TABLE IF NOT EXISTS team_counts_gen9 (
        team_id INTEGER PRIMARY KEY,
        format TEXT NOT NULL,
        num INTEGER NOT NULL
    );"""
    )

    f.query(
        """
    INSERT INTO team_gen9 (team_id, mon_id)
    VALUES
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2);
    """
    )

    f.query(
        """
    INSERT INTO pokemon_gen9 (mon_id, species, gender, tera_type, item, hp, atk, def, spa, spd, spe, ability, level, shiny, move1_id, move2_id, move3_id, move4_id)
    VALUES
        (1, "furret", "female", "normal", "focussash", 100, 100, 100, 100, 100, 100, "frisk", 100, TRUE, "endeavor", "followme", "protect", "superfang"),
        (2, "sentret", "female", "normal", "sitrusberry", 100, 100, 100, 100, 100, 100, "runaway", 100, FALSE, "endeavor", "followme", "protect", "superfang"),
        (3, "shuckle", "male", "steel", "leftovers", 100, 100, 100, 100, 100, 100, "contrary", 100, FALSE, "shellsmash", "powertrick", "toxic", "infestation");
    """
    )

    f.query(
        """
    INSERT INTO team_counts_gen9 (team_id, format, num)
    VALUES
        (1, "gen9vgc2024regg", 200),
        (2, "gen9vgc2024regg", 50);
    """
    )

    # Simulate what we observed in battle
    m = MagicMock()
    moves = {
        "followme": Move("followme", gen=9),
        "superfang": Move("superfang", gen=9),
    }
    m._teampreview_opponent_team = [
        ObservedPokemon(species="furret", level=100),
        ObservedPokemon(species="sentret", level=100),
    ]
    m.opponent_team = {"furret": ObservedPokemon(species="furret", level=100, moves=moves)}
    team = f.predict_vgc_team(m, battle_format="gen9vgc2024regg")

    # Verify that all properties and entries in the database are right
    assert team[0].species == "furret"
    assert team[1].species == "sentret"
    assert team[2].species == "shuckle"
    assert team[2].gender == PokemonGender.MALE
    assert team[2].tera_type == PokemonType.STEEL
    assert team[2].item == "leftovers"
    assert team[2].ability == "contrary"
    assert team[2].level == 100
    assert team[2].shiny is False
    assert "shellsmash" in team[2].moves.keys()
    assert "powertrick" in team[2].moves.keys()
    assert "toxic" in team[2].moves.keys()
    assert "infestation" in team[2].moves.keys()
    assert team[2].stats["hp"] == 100
    assert team[2].stats["atk"] == 100
    assert team[2].stats["def"] == 100
    assert team[2].stats["spa"] == 100
    assert team[2].stats["spd"] == 100
    assert team[2].stats["spe"] == 100

    teams = f.predict_vgc_team(m, battle_format="gen9vgc2024regg", probabilities=True)

    team, probability = teams[0]
    assert probability == 0.8
    assert teams[1][1] == 0.2
