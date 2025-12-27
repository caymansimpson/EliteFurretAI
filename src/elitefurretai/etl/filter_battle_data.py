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
import argparse

import orjson

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.supervised.train_utils import format_time


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
        print(f"\nError loading file {file}: {e}\n")
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

    # Filter out logs with old protocol (Symbiosis)
    elif any(map(lambda x: "|ability: Symbiosis" in x, bd.logs)):
        return False

    # Remove battles without requests (can't process properly)
    elif any(map(lambda x: "-transform" in x, bd.logs)):
        return False

    # Remove battles with Revival Blessing (too niche/complex)
    elif any(map(lambda x: "Pawmot" in x, bd.logs)):
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

    # Remove battles with Commander ability (not generalized enough to model)
    # elif any(map(lambda x: "Commander" in x, bd.logs)):
    #     return False

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

    # Process in batches to avoid memory explosion
    batch_size = 10000  # Process 10k files at a time
    total_count = 0

    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(load_file, file): file for file in batch_files}

            try:
                # Timeout of 5m per future, which is running 10k files at one time
                for future in concurrent.futures.as_completed(futures, timeout=300):
                    try:
                        result = future.result(timeout=10)
                        if result is not None:
                            file, bd = result
                            if is_valid_for_supervised_learning(bd):
                                valid_files.append(file)
                    except concurrent.futures.TimeoutError:
                        print("\nTimeout processing file", file=sys.stderr)
                    except Exception as e:
                        print(f"\nError processing future: {e}", file=sys.stderr)

                    total_count += 1

                    # Print progress
                    time_taken = format_time(time.time() - start_time)
                    time_per_battle = (time.time() - start_time) / total_count
                    time_left = format_time((len(files) - total_count) * time_per_battle)

                    print(f"\rProcessed {total_count}/{len(files)} battles ({round(total_count * 100.0 / len(files), 2)}%) in {time_taken} with an estimated {time_left} left    ", end="")

            except concurrent.futures.TimeoutError:
                print("\n\nBatch timeout reached!", file=sys.stderr)

    print(
        f"\nDone reading {len(files)} battles in {format_time(time.time() - start_time)}. Ended with {len(valid_files)} valid battles!"
    )

    # Save the list of valid files as JSON
    with open(save_file, "wb") as f:
        f.write(orjson.dumps(valid_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter Pokémon Showdown battle logs for supervised learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage:
            python filter_battle_data.py <read_dir> <save_file> [--num-threads N]
        """
    )
    parser.add_argument("read_dir", type=str, help="Directory containing raw battle files")
    parser.add_argument("save_file", type=str, help="Output JSON file for valid battle file paths")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count(), help="Number of threads for parallel filtering (default: number of CPUs)")
    args = parser.parse_args()

    main(args.read_dir, args.save_file, args.num_threads)
