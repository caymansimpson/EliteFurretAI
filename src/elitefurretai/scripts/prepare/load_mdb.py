# -*- coding: utf-8 -*-
"""This script analyzes VGC logs
"""
import sys
import time

import orjson
from poke_env.data.normalize import to_id_str

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.utils.meta_db import MetaDB


def load_files(files):
    for file in files:
        with open(file, "r") as f:
            yield BattleData.from_showdown_json(orjson.loads(f.read()))


# The filename should point to a json of a list of files to read from
def main(filename):
    files = []
    with open(filename, "rb") as f:
        files = orjson.loads(f.read())

    mdb = MetaDB(write=True, v=False)
    print("Files loaded and Meta DB up and running!")
    mdb.drop_all_tables()
    mdb.create_new_tables()

    # For tracking
    start_time = time.time()
    count = 0

    # Go through each file
    for bd in load_files(files):

        battle_id = str(hash("".join(bd.logs)))
        team_ids = {}

        # Go through each perspective
        for perspective in ["p1", "p2"]:
            battle = bd.to_battle(perspective)
            mon_ids = []

            # Construct and write each mon
            for mon in battle.teampreview_team:
                moves = sorted(list(mon.moves.values()), key=lambda x: x.id)
                mon_info = [
                    mon.species,
                    "F",
                    mon.tera_type,
                    to_id_str(mon.item),
                    mon.stats["hp"],
                    mon.stats["atk"],
                    mon.stats["def"],
                    mon.stats["spa"],
                    mon.stats["spd"],
                    mon.stats["spe"],
                    mon.ability,
                    50,
                    moves[0].id,
                    moves[1].id if len(moves) > 1 else "NULL",
                    moves[2].id if len(moves) > 2 else "NULL",
                    moves[3].id if len(moves) > 3 else "NULL",
                ]
                mon_id = hash("".join(map(lambda x: str(x), mon_info)))

                mdb.write("pokemon_gen9", [mon_id] + mon_info)
                mon_ids.append(mon_id)

            # Write each team for each perspective
            team_id = hash("".join(map(lambda x: str(x), sorted(mon_ids))))
            for mon_id in mon_ids:
                mdb.write("team_gen9", [team_id, mon_id])

            team_ids[perspective] = team_id

        # Write each battle
        mdb.write(
            "battle_gen9",
            [
                battle_id,
                bd.p1_rating,
                bd.p2_rating,
                team_ids["p1"],
                team_ids["p2"],
                bd.format,
            ],
        )

        count += 1
        if count % 10 == 0:
            hours = int(time.time() - start_time) // 3600
            minutes = int(time.time() - start_time) // 60
            seconds = int(time.time() - start_time) % 60
            print(f"\rProcessed {count} battles in {hours}h {minutes}m {seconds}s", end="")

    # Construct team_counts_gen9
    print("Done writing teams... Generating team_counts now...")
    team1s = mdb.query("SELECT team1_id, COUNT(*) FROM battle_gen9 GROUP BY team1_id")
    team2s = mdb.query("SELECT team2_id, COUNT(*) FROM battle_gen9 GROUP BY team2_id")
    teams = {}
    for row in team1s if team1s else []:
        teams[row[0]] = row[1]
    for row in team2s if team2s else []:
        teams[row[0]] = teams.get(row[0], 0) + row[1]

    for team_id, num in teams.items():
        mdb.write("team_counts_gen9", [team_id, "gen9vgc", num])

    print()
    print(f"Done reading {count} battles in {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    main(sys.argv[1])
