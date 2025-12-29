# pyright: reportArgumentType=false
"""This file goes through a bunch of battle files and looks at how effective action pruning would be.
Analysis reveals:
1. 2% of chosen actions by players are not valid by is_valid_order
2. 4% of chosen actions by players are not valid or are not reasonable
3. When we look at the possible actions (41^2 as defined by MDBO), 12% are valid by is_valid_order and 8% are reasonable by is_reasonable_move

This means that by using this heuristic, we can eliminate 33% of states in the action space while only
losing 2% of prediction accuracy. However, we compare how it empirically performs in practice with an
action prediction model in reasonable_filter_on_action_model.py
"""

import sys
import time
from typing import Dict, Tuple

import orjson

from elitefurretai.etl import MDBO, BattleData, BattleIterator
from elitefurretai.etl.battle_order_validator import (
    is_reasonable_move,
    is_valid_order,
)


# battle_filepath points to a JSON file that is a list of filepaths that contain BattleData
def get_iterators(battle_filepath, num_battles):
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())

    files = files[: int(num_battles / 2)]

    for file in files:
        with open(file, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))
            yield BattleIterator(bd, perspective="p1", omniscient=True)
            yield BattleIterator(bd, perspective="p2", omniscient=True)


def main(battle_filepath, write_filepath, num_battles):
    start = time.time()
    labels_data: Dict[Tuple[int, bool, bool], int] = {
        (i, valid, is_reasonable): 0
        for i in range(MDBO.action_space())
        for valid in [True, False]
        for is_reasonable in [True, False]
    }
    num_steps, num_valid, num_reasonable, num_both, num = 0, 0, 0, 0, 0
    for iter in get_iterators(battle_filepath, num_battles):
        # Iterate through the battle and get the player's input commands until ended or we exceed the steps per battle parameter
        while not iter.battle.finished and iter.next_input():
            # Get the last input command found by the iterator by the player, and stop if there's no more
            input = iter.last_input
            if (
                input is None
                or iter.last_input_type == MDBO.TEAMPREVIEW
                or iter.last_input_type is None
            ):
                continue

            # Collect Labels Dataset
            dbo = iter.last_order().to_double_battle_order(iter.battle)  # type: ignore[arg-type]
            is_valid = is_valid_order(dbo, iter.battle)  # type: ignore[arg-type]
            is_reasonable = is_reasonable_move(iter.battle, dbo)  # type: ignore[arg-type]
            labels_data[(iter.last_order().to_int(), is_valid, is_reasonable)] += 1

            # Collect overall action space pruning stats
            for possible_action_int in range(MDBO.action_space()):
                try:
                    dbo = MDBO.from_int(
                        possible_action_int, iter.last_input_type
                    ).to_double_battle_order(iter.battle)  # type: ignore[arg-type]
                    v = is_valid_order(dbo, iter.battle)  # type: ignore
                    r = is_reasonable_move(iter.battle, dbo)  # type: ignore
                    if v and r:
                        num_both += 1
                        num_valid += 1
                        num_reasonable += 1
                    elif v:
                        num_valid += 1
                    elif r:
                        num_reasonable += 1
                except Exception:
                    continue  # Skip invalid orders (if the int creates an invalid order)

            num_steps += 1
        num += 1

        time_left = (time.time() - start) / num * (num_battles - num)
        hours_left = int(time_left // 3600)
        minutes_left = int((time_left % 3600) // 60)
        seconds_left = int(time_left % 60)
        print(
            f"\033[2K\rProcessed {num} battles in {round(time.time() - start, 2)}s; estimated {hours_left}h {minutes_left}m {seconds_left}s left",
            end="",
        )

    # Print action pruning stats
    print("\nDone!")
    total_possible = num_steps * MDBO.action_space()
    print(
        f"Overall action space pruning stats: {num_steps} steps processed, "
        + f"{total_possible} total possible actions, "
        + f"{num_valid} ({round(num_valid / total_possible, 2) * 100}%) valid orders, "
        + f"{num_reasonable} ({round(num_reasonable / total_possible, 2) * 100}%) reasonable orders "
        + f"{num_both} ({round(num_both / total_possible, 2) * 100}%) both orders"
    )

    total_reasonable, total_valid, total_both, actions = 0, 0, 0, [0] * MDBO.action_space()
    for (action_idx, is_valid, is_reasonable), num in labels_data.items():
        total_reasonable += num * int(is_reasonable)
        total_valid += num * int(is_valid)
        total_both += num * int(is_reasonable and is_valid)
        actions[action_idx] += num

    print("For labels:")
    print(
        f"\tTotal Valid: {total_valid} ({round(total_valid / num_steps, 2) * 100}%) (should be close to 100%)"
    )
    print(
        f"\tTotal Reasonable & Valuable: {total_reasonable} ({round(total_reasonable / num_steps, 2) * 100}%) (should be as close to 100%)"
    )
    print(
        f"\tTotal Both: {total_both} ({round(total_both / num_steps, 2) * 100}%) (should be as close to 100%)"
    )

    print("Now writing to file...")
    with open(write_filepath, "w") as f:
        f.write("action,is_valid,is_reasonable,count\n")
        for k, v in labels_data.items():  # type: ignore
            f.write(f"{k[0]},{k[1]},{k[2]},{v}\n")


# Example usage :
# python3 src/elitefurretai/scripts/analyze/action_pruning.py data/battles/supervised_battle_files_w_commander.json . 10000
# python3 src/elitefurretai/scripts/analyze/action_pruning.py battle_filepath write_filepath num_battles
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[2]))
