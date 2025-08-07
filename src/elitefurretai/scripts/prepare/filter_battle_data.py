# -*- coding: utf-8 -*-
"""
filter_battle_data.py

This script filters Pokémon Showdown battle logs for supervised learning.
It loads all battle files from a directory, applies a series of quality and edge-case filters,
and saves a list of valid file paths for downstream model training.

Key features:
- Removes battles with protocol errors, low ELO, missing data, or problematic edge-cases.
- Excludes battles with certain moves, abilities, or Pokémon that are hard to model.
- Uses multithreading for fast file loading and filtering.
- Outputs a JSON list of valid file paths for further processing.
"""

import concurrent.futures
import os
import sys
import time

import orjson

from elitefurretai.model_utils.battle_data import BattleData


def load_file(file):
    """
    Loads a single battle file and parses it into a BattleData object.
    Returns (file, BattleData) if successful, else None.
    """
    try:
        with open(file, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))
            return file, bd
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return None


# To help with some corrupted logs that we have
def is_valid_for_supervised_learning(bd: BattleData) -> bool:
    """
    Applies a series of filters to determine if a battle is suitable for supervised learning.
    Returns True if valid, False otherwise.
    """

    # Filter out logs with old protocol artifacts
    if any(map(lambda x: "[ability2] " in x, bd.logs)):
        return False

    # Filter out battles with missing player ratings
    elif bd.p1_rating is None or bd.p2_rating is None:
        return False

    # Only keep high ELO battles (both players >= 1500)
    elif bd.p1_rating < 1500 or bd.p2_rating < 1500:
        return False

    # Filter out battles that never started (no input logs)
    elif bd.input_logs == []:
        return False

    # Remove edge-case where an active mon faints and is revived in the same turn
    elif any(map(lambda x: "switch 1" in x or "switch 2" in x, bd.input_logs)):
        return False

    # Remove battles with Metronome (too random)
    elif any(map(lambda x: "Metronome" in x, bd.logs)):
        return False

    # Remove battles with Eject Pack proc after Moody boost (bad showdown logic)
    elif any(
        map(
            lambda x: bd.logs[x].endswith("Eject Pack")
            and bd.logs[max(0, x - 3)].endswith("|Moody|boost"),
            range(len(bd.logs)),
        )
    ):
        return False

    # Remove battles with Dancer activating before Eject Button (edge-case)
    elif any(map(lambda x: x == "|-enditem|p2b: 780b3dada7|Eject Button", bd.logs)):
        return False

    # Filter out logs with old protocol (Zero to Hero ability)
    elif any(map(lambda x: "|-ability||Zero to Hero" in x, bd.logs)):
        return False

    # Remove battles without requests (can't process properly)
    elif any(map(lambda x: "-transform" in x, bd.logs)):
        return False

    # Remove battles with Revival Blessing (too niche/complex)
    elif any(map(lambda x: "Revival Blessing" in x, bd.logs)):
        return False

    # Remove battles with fainted from hazards edge-case (too niche)
    elif any(
        map(
            lambda x: "0 fnt|[from] Stealth Rock" in x or "0 fnt|[from] Spikes" in x,
            bd.logs,
        )
    ):
        return False

    # Remove battles with Zoroark/Zorua (illusion is hard to model)
    elif any(map(lambda x: "Zoroark" in x or "Zorua" in x, bd.logs)):
        return False

    return True


def main(read_dir, save_file, num_threads):
    """
    Loads all battle files from read_dir, filters them using is_valid_for_supervised_learning,
    and saves the list of valid file paths to save_file.
    """
    files = sorted(map(lambda x: os.path.join(read_dir, x), os.listdir(read_dir)))
    print(f"Finished loading {len(files)} files! Will start processing...")

    start_time = time.time()
    valid_files = []

    # Use ThreadPoolExecutor for parallel file loading and filtering
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(load_file, file): file for file in files}
        count, last = 0, 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                file, bd = result
                if is_valid_for_supervised_learning(bd):
                    valid_files.append(file)

                count += 1
                # Print progress every second
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
                    left = f" with an estimated {hours_left}h {minutes_left}m {seconds_left}s left "
                    print("\r" + processed + left, end="")
                    last = int(time.time())

    print(
        f"Done reading {len(files)} battles in {round(time.time() - start_time, 2)} seconds. Ended with {len(valid_files)} valid battles!"
    )

    # Save the list of valid files as JSON
    prepare_dir = os.path.dirname(__file__)
    scripts_dir = os.path.dirname(prepare_dir)
    src_dir = os.path.dirname(os.path.dirname(scripts_dir))
    elitefurretai_dir = os.path.dirname(src_dir)
    filename = os.path.join(elitefurretai_dir, save_file)

    with open(filename, "wb") as f:
        f.write(orjson.dumps(valid_files))


if __name__ == "__main__":
    # Parse command-line arguments and run main
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python script.py <read_dir> <save_file> [num_threads]")
        sys.exit(1)

    read_dir = sys.argv[1]
    save_file = sys.argv[2]
    num_threads = int(sys.argv[3]) if len(sys.argv) == 4 else os.cpu_count()

    main(read_dir, save_file, num_threads)
