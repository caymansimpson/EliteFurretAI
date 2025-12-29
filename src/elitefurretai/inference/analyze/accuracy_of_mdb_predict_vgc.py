# -*- coding: utf-8 -*-
"""
accuracy_of_predict_vgc.py

This script evaluates the accuracy of a team prediction model (MetaDB.predict_vgc_team) for Pokémon VGC battles.
It loads a list of battle files, iterates through each battle and both player perspectives, and for each turn,
compares the predicted opponent team (species, item, ability, tera type, stats, and moves) to the true team.
It tracks the number of perfectly predicted teams and the number of correct aspects per Pokémon, reporting
overall and per-turn accuracy statistics.

Key features:
- Loads and parses battle logs for both player perspectives.
- For each turn, constructs the observed opponent team and predicts the full team using MetaDB.
- Compares predicted and true teams on all relevant aspects (species, item, ability, tera type, stats, moves).
- Tracks and prints accuracy metrics per turn and overall, including percent of perfect predictions and aspects correct.

Currently, it gets the following stats:
Percent Accuracy: 52.0%
Turn #0 // % Perfect Prediction: 42.3% // % Aspects Right: 83.2%
Turn #1 // % Perfect Prediction: 48.08% // % Aspects Right: 87.05%
Turn #2 // % Perfect Prediction: 49.57% // % Aspects Right: 87.78%
Turn #3 // % Perfect Prediction: 52.75% // % Aspects Right: 88.99%
Turn #4 // % Perfect Prediction: 55.9% // % Aspects Right: 90.04%
Turn #5 // % Perfect Prediction: 56.92% // % Aspects Right: 90.65%
Turn #6 // % Perfect Prediction: 56.25% // % Aspects Right: 90.6%
Turn #7 // % Perfect Prediction: 58.15% // % Aspects Right: 90.84
Turn #8 // % Perfect Prediction: 58.2% // % Aspects Right: 90.86%
Turn #9 // % Perfect Prediction: 59.79% // % Aspects Right: 91.81%
Turn #10 // % Perfect Prediction: 57.5% // % Aspects Right: 90.77%
Turn #11 // % Perfect Prediction: 58.75% // % Aspects Right: 90.85%
Turn #12 // % Perfect Prediction: 61.67% // % Aspects Right: 91.34%
Turn #13 // % Perfect Prediction: 64.71% // % Aspects Right: 93.71%
Turn #14 // % Perfect Prediction: 67.86% // % Aspects Right: 95.14%
Turn #15 // % Perfect Prediction: 77.78% // % Aspects Right: 97.38%
Turn #16 // % Perfect Prediction: 66.67% // % Aspects Right: 96.06%
Turn #17 // % Perfect Prediction: 60.0% // % Aspects Right: 95.28%
Turn #18 // % Perfect Prediction: 50.0% // % Aspects Right: 96.88%
Turn #19 // % Perfect Prediction: 50.0% // % Aspects Right: 96.88%
Turn #20 // % Perfect Prediction: 50.0% // % Aspects Right: 96.88%
Turn #21 // % Perfect Prediction: 50.0% // % Aspects Right: 96.88%
Turn #22 // % Aspects Right: 96.53%

Total time taken for 1000 iterations: 3h 31m 0s at 12.66s per battle and 1.83s per turn
"""

import random
import sys
import time
from typing import Dict

import orjson
from poke_env.battle import ObservedPokemon
from poke_env.data.normalize import to_id_str

from elitefurretai.etl import BattleData, BattleIterator
from elitefurretai.inference import MetaDB


def load_files(files):
    """
    Generator that loads BattleData objects from a list of file paths.
    """
    for file in files:
        with open(file, "r") as f:
            yield BattleData.from_showdown_json(orjson.loads(f.read()))


def main(files):
    """
    Main evaluation loop. For each battle and each player perspective, iterates through all turns,
    predicts the opponent's team, and compares it to the true team. Tracks and prints accuracy statistics.
    """
    mdb = MetaDB()
    battle_num = 0
    num_right: Dict[int, int] = {}  # Tracks number of perfectly predicted teams per turn
    num_wrong: Dict[int, int] = {}  # Tracks number of imperfect predictions per turn
    wrong_aspects: Dict[int, int] = {}  # Tracks number of incorrect aspects per turn
    num_mons: Dict[int, int] = {}  # Tracks number of Pokémon compared per turn
    start = time.time()
    print(
        f"Done reading {len(files)} files! Starting to predict teams for {len(files) * 2} battles!\n"
    )

    # For each battle
    for bd in load_files(files):
        # For each player perspective (p1 and p2)
        for perspective in ["p1", "p2"]:
            battle_num += 1
            iter = BattleIterator(bd, perspective=perspective)

            # Get the true team for the opponent from this perspective
            true_team = bd.p2_team if perspective == "p1" else bd.p1_team

            # For each turn, construct what we know about the team
            for turn in range(bd.turns):
                iter.next_turn()
                team = []

                # Construct observed team from opponent's revealed Pokémon
                for mon in iter.battle.opponent_team.values():
                    team.append(ObservedPokemon.from_pokemon(mon))

                # Add any unrevealed Pokémon from teampreview (with minimal info)
                for mon in iter.battle.teampreview_opponent_team:
                    if not any(map(lambda x: x.species == mon.species, team)):
                        team.append(
                            ObservedPokemon(
                                mon.species,
                                level=mon.level,
                                name=mon.species,
                            )
                        )

                # Predict the full team using MetaDB
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
                            # Compare item, ability, tera type, and all stats
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
                            # Count number of moves in true mon not present in predicted mon
                            num_aspects_wrong += sum(
                                map(
                                    lambda x: int(to_id_str(x) not in observed_mon_moves),
                                    mon.moves,
                                )
                            )

                            num_mons[turn] = num_mons.get(turn, 0) + 1

                # Track perfect and imperfect predictions for this turn
                if num_aspects_wrong == 0:
                    num_right[turn] = num_right.get(turn, 0) + 1
                else:
                    num_wrong[turn] = num_wrong.get(turn, 0) + 1
                    wrong_aspects[turn] = wrong_aspects.get(turn, 0) + num_aspects_wrong

                # Print progress and estimated time left
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
    accuracy = sum(num_right.values()) / (
        sum(num_right.values()) + sum(num_wrong.values())
    )
    print(f"Percent Accuracy: {100 * round(accuracy, 2)}")
    for turn in range(max(len(num_right), len(num_wrong))):
        perfect_prediction = num_right.get(turn, 0) / (
            num_right.get(turn, 0) + num_wrong.get(turn, 0)
        )
        aspects_right = (num_mons.get(turn, 0) * 12.0 - wrong_aspects.get(turn, 0)) / (
            num_mons.get(turn, 0) * 12.0
        )
        print(
            f"""\tTurn #{turn} // % Perfect Prediction: {round(100 * perfect_prediction, 2)}%"""
            + f""" // % Aspects Right: {round(100.0 * aspects_right, 2)}%"""
        )

    hours = int((time.time() - start) // 3600)
    minutes = int(((time.time() - start) % 3600) // 60)
    seconds = int((time.time() - start) % 60)
    per_battle = round((time.time() - start) / (len(files * 2)), 2)
    per_turn = round(
        (time.time() - start) / (sum(num_right.values()) + sum(num_wrong.values())), 2
    )
    print(
        f"Total time taken for {(len(files) * 2)} iterations: "
        + f"{hours}h {minutes}m {seconds}s at {per_battle}s per battle and {per_turn}s per turn"
    )
    print()


# Parameters are a filepath to a list of filenames storing BattleData and the number of battles to process
# Example usage: python3 src/elitefurretai/scripts/analyze/accuracy_of_predict_vgc.py <filepath> <number of battles>
if __name__ == "__main__":
    # Load the list of files to process, shuffle, and subsample if needed
    files = []
    with open(sys.argv[1], "rb") as f:
        files = sorted(orjson.loads(f.read()), key=lambda x: random.random())[
            : int(sys.argv[2])
        ]

    main(files)
