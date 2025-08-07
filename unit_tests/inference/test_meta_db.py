# -*- coding: utf-8 -*-
from collections import OrderedDict

from poke_env.battle import Move, ObservedPokemon, PokemonType

from elitefurretai.inference.meta_db import MetaDB


def test_query():
    f = MetaDB("test_frisk.db", write=True)

    # Set up database
    f.drop_all_tables()
    f.create_new_tables()

    # Insert into database via multiple methods
    f.write("team_gen9", [1, 1])
    f.query(
        """
    INSERT INTO pokemon_gen9 (mon_id, species, gender, tera_type, item, hp, atk, def, spa, spd, spe, ability, level, move1, move2, move3, move4)
    VALUES(1, "furret", "f", "NORMAL", "focussash", 100, 100, 100, 100, 100, 100, "frisk", 100, "endeavor", "followme", "protect", "superfang"
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

    print(results)

    # Verify that the query worked
    assert results is not None
    assert results[0][0] == 1
    assert results[0][1] == "furret"

    f.close()


def test_predict_vgc_team():
    f = MetaDB("test_frisk.db", write=True)

    # Create database and data structures
    f.drop_all_tables()
    f.create_new_tables()

    f.query(
        """
    INSERT INTO team_gen9 (team_id, mon_id)
    VALUES
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 3);
    """
    )

    f.query(
        """
    INSERT INTO pokemon_gen9 (mon_id, species, gender, tera_type, item, hp, atk, def, spa, spd, spe, ability, level, move1, move2, move3, move4)
    VALUES
        (1, "furret", "female", "normal", "focussash", 100, 100, 100, 100, 100, 100, "frisk", 100, "endeavor", "followme", "protect", "superfang"),
        (2, "sentret", "female", "normal", "sitrusberry", 100, 100, 100, 100, 100, 100, "runaway", 100, "endeavor", "followme", "protect", "superfang"),
        (3, "shuckle", "male", "steel", "leftovers", 100, 100, 100, 100, 100, 100, "contrary", 100, "shellsmash", "powertrick", "toxic", "infestation"),
        (4, "shuckle", "male", "steel", "leftovers", 100, 100, 100, 100, 100, 100, "sturdy", 100, "shellsmash", "powertrick", "toxic", "infestation");
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
    moves = OrderedDict(
        {
            "followme": Move("followme", gen=9),
            "superfang": Move("superfang", gen=9),
        }
    )
    observed_mons = [
        ObservedPokemon(species="furret", level=100, name="elitefurretai", moves=moves),
        ObservedPokemon(species="sentret", level=100, name="elitesentretai", moves=moves),
        ObservedPokemon(species="shuckle", level=100, name="eliteshuckleai"),
    ]
    team = f.predict_vgc_team(observed_mons, battle_format="gen9vgc2024regg")

    # Verify that all properties and entries in the database are right
    assert isinstance(team[0], ObservedPokemon)
    assert isinstance(team[1], ObservedPokemon)
    assert isinstance(team[2], ObservedPokemon)
    assert team[0].species == "furret"
    assert team[1].species == "sentret"
    assert team[2].species == "shuckle"
    assert team[2].tera_type == PokemonType.STEEL
    assert team[2].item == "leftovers"
    assert team[2].ability == "sturdy"
    assert team[2].level == 100
    assert "shellsmash" in team[2].moves.keys()
    assert "powertrick" in team[2].moves.keys()
    assert "toxic" in team[2].moves.keys()
    assert "infestation" in team[2].moves.keys()
    assert team[2].stats
    assert team[2].stats["hp"] == 100
    assert team[2].stats["atk"] == 100
    assert team[2].stats["def"] == 100
    assert team[2].stats["spa"] == 100
    assert team[2].stats["spd"] == 100
    assert team[2].stats["spe"] == 100

    teams = f.predict_vgc_team(
        observed_mons, battle_format="gen9vgc2024regg", probabilities=True
    )
    team, probability = teams[0]  # type: ignore
    assert team[2].ability == "sturdy"  # type: ignore
    assert probability == 0.8
    assert teams[1][1] == 0.2  # type: ignore
