import random
import sys
import time

import orjson
from poke_env.data.normalize import to_id_str
from poke_env.environment import ObservedPokemon

from elitefurretai.model_utils import BattleData, BattleIterator
from elitefurretai.utils import MetaDB


def load_files(files):
    for file in files:
        with open(file, "r") as f:
            yield BattleData.from_showdown_json(orjson.loads(f.read()))


def main(files):

    mdb = MetaDB()
    battle_num, num_right, num_wrong, wrong_aspects, num_mons, start = (
        0,
        {},
        {},
        {},
        {},
        time.time(),
    )
    print(
        f"Done reading {len(files)} files! Starting to predict teams for {len(files) * 2} battles!\n"
    )

    # For each battle
    for bd in load_files(files):

        # For each perspective
        for perspective in ["p1", "p2"]:
            battle_num += 1
            iter = BattleIterator(bd, perspective=perspective)

            true_team = bd.p2_team if perspective == "p1" else bd.p1_team

            # For each turn, construct what we know about the team
            for turn in range(bd.turns):
                iter.next_turn()
                team = []

                # Construct observed team
                for mon in iter.battle.opponent_team.values():
                    team.append(ObservedPokemon.from_pokemon(mon))

                for mon in iter.battle.teampreview_opponent_team:
                    if not any(map(lambda x: x.species == mon.species, team)):
                        team.append(
                            ObservedPokemon(
                                mon.species,
                                level=mon.level,
                                name=mon.species,
                            )
                        )

                # Predicted team returning empty
                predicted_team = mdb.predict_vgc_team(team, "gen9vgc", probabilities=False)

                # Calculate various dimensions of accuracy
                num_aspects_wrong = 0
                for observed_mon in predicted_team:
                    assert isinstance(observed_mon, ObservedPokemon)
                    observed_mon_moves = list(
                        map(lambda x: to_id_str(x), observed_mon.moves.keys())
                    )
                    for mon in true_team:
                        if to_id_str(mon.species) == to_id_str(observed_mon.species):
                            if to_id_str(mon.item) != to_id_str(observed_mon.item):
                                num_aspects_wrong += 1
                            if to_id_str(mon.ability) != to_id_str(observed_mon.ability):
                                num_aspects_wrong += 1
                            if mon.tera_type != observed_mon.tera_type:
                                num_aspects_wrong += 1

                            if mon.stats["hp"] != observed_mon.stats["hp"]:  # type: ignore
                                num_aspects_wrong += 1
                            if mon.stats["atk"] != observed_mon.stats["atk"]:  # type: ignore
                                num_aspects_wrong += 1
                            if mon.stats["def"] != observed_mon.stats["def"]:  # type: ignore
                                num_aspects_wrong += 1
                            if mon.stats["spa"] != observed_mon.stats["spa"]:  # type: ignore
                                num_aspects_wrong += 1
                            if mon.stats["spd"] != observed_mon.stats["spd"]:  # type: ignore
                                num_aspects_wrong += 1
                            if mon.stats["spe"] != observed_mon.stats["spe"]:  # type: ignore
                                num_aspects_wrong += 1
                            num_aspects_wrong += sum(
                                map(
                                    lambda x: int(to_id_str(x) not in observed_mon_moves),
                                    mon.moves,
                                )
                            )

                            num_mons[turn] = num_mons.get(turn, 0) + 1

                if num_aspects_wrong == 0:
                    num_right[turn] = num_right.get(turn, 0) + 1
                else:
                    num_wrong[turn] = num_wrong.get(turn, 0) + 1
                    wrong_aspects[turn] = wrong_aspects.get(turn, 0) + num_aspects_wrong

                # Print progress
                time_left = ((len(files) * 2) + 1 - battle_num) * (
                    (time.time() - start) / battle_num
                )
                hours_left = int((time_left) // 3600)
                minutes_left = int((time_left % 3600) // 60)
                seconds_left = int((time_left) % 60)

                hours = int((time.time() - start) // 3600)
                minutes = int(((time.time() - start) % 3600) // 60)
                seconds = int((time.time() - start) % 60)
                print(
                    f"""\r\033[KBattle #{battle_num} // Right: {sum(num_right.values())} // """
                    + f"""Wrong: {sum(num_wrong.values())} // """
                    + f"""% Aspects Right: {round(100.0 * (sum(num_mons.values()) * 12.0 - sum(wrong_aspects.values())) / sum(num_mons.values()) / 12.0, 2)} // """
                    + f"""Total time: {hours}h {minutes}m {seconds}s // """
                    + f"""Total turns predicted: {sum(num_right.values()) + sum(num_wrong.values())} // """
                    + f"""Estimated time left is {hours_left}h {minutes_left}m {seconds_left}s""",
                    end="",
                )

    # Print final statistics!
    if sum(num_right.values()) + sum(num_wrong.values()) == 0:
        print("No battles or turns were processed!")
        sys.exit()

    print()
    print("Finished!")
    print(
        "Percent Accuracy:",
        100
        * round(
            sum(num_right.values()) / (sum(num_right.values()) + sum(num_wrong.values())),
            2,
        ),
    )
    for turn in range(max(len(num_right), len(num_wrong))):
        print(
            f"""\tTurn #{turn} // % Perfect Prediction: {round(100.0 * num_right.get(turn, 0) / (num_right.get(turn, 0) + num_wrong.get(turn, 0)), 2)}"""
            + f""" // % Aspects Right: {round(100.0 * (num_mons.get(turn, 0) * 12.0 - wrong_aspects.get(turn, 0)) / (num_mons.get(turn, 0) * 12.0), 2)}"""
        )

    hours = int((time.time() - start) // 3600)
    minutes = int(((time.time() - start) % 3600) // 60)
    seconds = int((time.time() - start) % 60)
    per_battle = round((time.time() - start) / (len(files * 2)), 2)
    per_turn = round(
        (time.time() - start) / (sum(num_right.values()) + sum(num_wrong.values())), 2
    )
    print(
        f"Total time taken for {(len(files) * 2)} iterations: {hours}h {minutes}m {seconds}s at {per_battle}s per battle and {per_turn}s per turn"
    )
    print()


# Parameters are a filepath to a list of filenames storing BattleData and the number of battles to process
if __name__ == "__main__":
    files = []
    with open(sys.argv[1], "rb") as f:
        files = sorted(orjson.loads(f.read()), key=lambda x: random.random())[
            : int(sys.argv[2])
        ]

    main(files)
