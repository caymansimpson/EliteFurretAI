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

import argparse
import concurrent.futures
import os
import time

import orjson

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.battle_iterator import BattleIterator
from elitefurretai.supervised.utils import format_time


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


def load_and_validate(file):
    """Worker entrypoint: load + run full filter in one shot, return file path or None.

    Used by `main` with ProcessPoolExecutor so the smoke parse runs in
    parallel across cores (it's CPU-bound; the old structure ran it
    serially in the main thread).
    """
    result = load_file(file)
    if result is None:
        return None
    fp, bd = result
    try:
        ok = is_valid_for_supervised_learning(bd)
    except Exception:
        return None
    return fp if ok else None


def battle_parses_cleanly(bd: BattleData) -> bool:
    """Smoke-parse the battle from both perspectives via BattleIterator.

    Real-world Showdown replays occasionally trip poke-env's parser
    (e.g. nickname-as-species when `move: Trick` references an
    unrevealed opponent — caught at preprocess time as KeyError on the
    pokedex lookup). Static log-substring filters can't catch these,
    so we actually replay the messages here and reject if either
    perspective throws.
    """
    for perspective in ("p1", "p2"):
        try:
            it = BattleIterator(bd, perspective=perspective, omniscient=False)
            while it.next_input():
                pass
        except Exception:
            return False
    return True


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
            lambda x: (
                bd.logs[x].endswith("Eject Pack")
                and bd.logs[max(0, x - 3)].endswith("|Moody|boost")
            ),
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

    # Final gate: actually replay through BattleIterator both perspectives
    # to catch poke-env parsing failures that static log scans miss.
    elif not battle_parses_cleanly(bd):
        return False

    return True


def main(read_dir, save_file, num_threads, input_json=None):
    """
    Loads battle files and filters them using is_valid_for_supervised_learning.
    File list comes from `input_json` if provided (re-validates an existing
    pre-filtered list, useful when a smoke-parse check is added after the
    fact), otherwise from `read_dir`.
    """
    if input_json is not None:
        with open(input_json, "rb") as f:
            files = orjson.loads(f.read())
        print(
            f"Loaded {len(files)} files from input JSON {input_json}; will re-validate..."
        )
    else:
        files = sorted(map(lambda x: os.path.join(read_dir, x), os.listdir(read_dir)))
        print(f"Finished loading {len(files)} files! Will start processing...")

    start_time = time.time()
    valid_files = []
    total_count = 0
    # Larger chunksize amortizes per-task IPC overhead for cheap (~5ms) per-battle work
    chunksize = 256

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        for result in executor.map(load_and_validate, files, chunksize=chunksize):
            if result is not None:
                valid_files.append(result)
            total_count += 1
            if total_count % 2000 == 0 or total_count == len(files):
                time_taken = time.time() - start_time
                per_battle = time_taken / total_count
                time_left = (len(files) - total_count) * per_battle
                print(
                    f"\rProcessed {total_count}/{len(files)} ({total_count * 100.0 / len(files):.2f}%) "
                    f"valid={len(valid_files)} in {format_time(time_taken)}, ~{format_time(time_left)} left   ",
                    end="",
                    flush=True,
                )

    print(
        f"\nDone reading {len(files)} battles in {format_time(time.time() - start_time)}. "
        f"Ended with {len(valid_files)} valid battles!"
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
        """,
    )
    parser.add_argument(
        "read_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing raw battle files (ignored if --input-json is set)",
    )
    parser.add_argument(
        "save_file", type=str, help="Output JSON file for valid battle file paths"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes for parallel filtering (default: number of CPUs)",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Optional: re-validate an existing JSON list of file paths instead of scanning read_dir",
    )
    args = parser.parse_args()

    if args.read_dir is None and args.input_json is None:
        parser.error("either read_dir or --input-json must be provided")

    main(args.read_dir, args.save_file, args.num_threads, input_json=args.input_json)
