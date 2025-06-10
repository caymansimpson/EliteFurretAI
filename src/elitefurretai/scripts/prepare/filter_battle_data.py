# -*- coding: utf-8 -*-
"""This script holds logic for which files to train on for supervised learning
"""
import os
import sys
import time

import orjson

from elitefurretai.model_utils.battle_data import BattleData


def load_files(files):
    for file in files:
        bd = None
        with open(file, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

            yield file, bd


# To help with some corrupted logs that we have
def is_valid_for_supervised_learning(bd: BattleData) -> bool:

    # Old showdown protool
    if any(map(lambda x: "[ability2] " in x, bd.logs)):
        return False

    # Player error
    elif bd.p1_rating is None or bd.p2_rating is None:
        return False

    # We should only train on high elo battles
    elif bd.p1_rating < 1500 or bd.p2_rating < 1500:
        return False

    # Battle didnt start
    elif bd.input_logs == []:
        return False

    # Edge-case where an active mon just faints in the same turn and you can revival blessing it.
    # Adding this to model_battle_order will increase the size and inaccuracy of the model
    elif any(map(lambda x: "switch 1" in x or "switch 2" in x, bd.input_logs)):
        return False

    # Creates a bunch of edge-cases that technically shouldn't be supported
    elif any(map(lambda x: "Metronome" in x, bd.logs)):
        return False

    #  Eject Pack proc after Moody gets merged into a preturn switch; bad showdown logic
    elif any(
        map(
            lambda x: bd.logs[x].endswith("Eject Pack")
            and bd.logs[max(0, x - 3)].endswith("|Moody|boost"),
            range(len(bd.logs)),
        )
    ):
        return False

    # There is an edge-case that has Dancer activating before Eject Button activates
    elif any(map(lambda x: x == "|-enditem|p2b: 780b3dada7|Eject Button", bd.logs)):
        return False

    # Old showdown protocol
    elif any(map(lambda x: "|-ability||Zero to Hero" in x, bd.logs)):
        return False

    # Can't do these battles properly without requests
    elif any(map(lambda x: "-transform" in x, bd.logs)):
        return False

    # Too lazy to implement this edge-case
    elif any(
        map(
            lambda x: "0 fnt|[from] Stealth Rock" in x or "0 fnt|[from] Spikes" in x,
            bd.logs,
        )
    ):
        return False

    # ef this noise
    elif any(map(lambda x: "Zoroark" in x or "Zorua" in x, bd.logs)):
        return False

    return True


# Argument should be directory with all the raw files
def main(dir):
    files = sorted(map(lambda x: os.path.join(dir, x), os.listdir(dir)))
    print(f"Finished loading {len(files)} files! Will start processing...")

    # For tracking
    start_time = time.time()
    last = start_time
    count = 0

    new_files = []

    # Go through each file
    for filename, bd in load_files(files):

        if is_valid_for_supervised_learning(bd):
            new_files.append(filename)

        count += 1
        if time.time() - last > 1:
            hours = int(time.time() - start_time) // 3600
            minutes = int(time.time() - start_time) // 60
            seconds = int(time.time() - start_time) % 60

            time_per_battle = (time.time() - start_time) * 1.0 / count
            est_time_left = (len(files) - count) * time_per_battle
            hours_left = int(est_time_left // 3600)
            minutes_left = int((est_time_left % 3600) // 60)
            seconds_left = int(est_time_left % 60)

            processed = f"Processed {count} battles ({round(count * 100.0 / len(files), 2)}%) in {hours}h {minutes}m {seconds}s"
            left = f" with an estimated {hours_left}h {minutes_left}m {seconds_left}s left      "
            print("\r" + processed + left, end="")
            last = time.time()

    print()
    print(f"Done reading {count} battles in {round(time.time() - start_time, 2)} seconds")

    prepare_dir = os.path.dirname(__file__)
    scripts_dir = os.path.dirname(prepare_dir)
    src_dir = os.path.dirname(os.path.dirname(scripts_dir))
    elitefurretai_dir = os.path.dirname(src_dir)
    filename = os.path.join(elitefurretai_dir, "data/battles/supervised_battle_files.json")

    with open(filename, "wb") as f:
        f.write(orjson.dumps(new_files))


if __name__ == "__main__":
    main(sys.argv[1])
