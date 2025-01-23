# -*- coding: utf-8 -*-
"""This script goes through training data to make sure there are no surprises
"""

import os.path
import sys
import time

from elitefurretai.model_utils import (
    BattleData,
    BattleIterator,
    Embedder,
    ModelBattleOrder,
)
from elitefurretai.model_utils.training_generator import file_generator


def main(directory, frmt):

    # For tracking progress through iterations
    total_battles = int(sum(1 for entry in os.scandir(directory) if entry.is_file()))
    print(f"{total_battles} battles to process...\n")
    count, start, last, step = 0, time.time(), 0, 0

    embedder = Embedder(format=frmt, simple=False)

    print("Processed 0 battles (0% done)...")

    # Open up battle file
    for bd, filename in file_generator(directory):

        # Look at battle from each player's perspective
        for perspective in ["p1", "p2"]:

            # Wrap the battle processing in a try/except block since BattleIterator can throw errors
            try:

                # Create battle from the file, with a battle iterator
                battle = bd.to_battle(perspective)
                iter = BattleIterator(
                    battle,
                    bd,
                    perspective=perspective,
                    custom_parse=BattleData.showdown_translation,
                )

                # Iterate through the battle and get the player's input commands
                while not battle.finished and iter.next_input():

                    # Get the last input command found by the iterator by the player, and stop if there's no more
                    input = iter.last_input
                    if input is None:
                        continue

                    request = iter.simulate_request()
                    battle.parse_request(request)

                    x = embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle))  # type: ignore

                    # Standardize the input (remove identifiers names and convert to ints)
                    y_order = ModelBattleOrder.from_battle_data(
                        input, battle, bd
                    ).to_int()  # Human order
                    y_win = int(
                        bd.winner == (bd.p1 if perspective == "p1" else bd.p2)
                    )  # Win prediction

                    assert x is not None and y_order is not None and y_win is not None

                    step += 1

            except AssertionError:
                print("Error reading:", filename)
            except RecursionError:
                print("Error reading:", filename)
            except IndexError:
                print("Error reading:", filename)

            count += 1
            if time.time() - start > last + 10:  # Print every 10 seconds
                print(
                    f"Processed {count} battles and {step} steps in {round(time.time() - start, 2)} secs ({round(count / total_battles * 100, 2)}% battles done)..."
                )
                last += 10

    print(
        f"Done! Read {count} battles with {step} steps in {round(time.time() - start, 2)} seconds"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
