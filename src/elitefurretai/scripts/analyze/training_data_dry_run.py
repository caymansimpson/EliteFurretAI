# -*- coding: utf-8 -*-
"""This script goes through training data to make sure there are no surprises. It also helps us understand performance
"""

import os
import sys
import time

import orjson
import torch
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset


# This function takes in a list of filepaths for BattleData files
def main(files):

    total_battles = len(files) * 2
    print(f"Starting dry run to go through {total_battles} battles in a training loop...")

    # Prepare data
    dataset = BattleDataset(files)
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
    )

    # For tracking progress through iterations
    battle_count, start, last, step_count = 0, time.time(), 0, 0

    # Iterate through batches of battles with data_loader
    for states, actions, action_masks, wins, masks in data_loader:
        batch_size, _, _ = states.shape
        battle_count += batch_size

        masks_reshaped = masks.view(-1, 1)  # (batch_size*steps, 1)
        step_count += int(
            torch.sum(masks_reshaped)
        )  # Only count valid timesteps (eg turns)

        # Do training things
        pass

        # Print every second
        if time.time() > last + 1:
            hours = int(time.time() - start) // 3600
            minutes = (int(time.time() - start) % 3600) // 60
            seconds = int(time.time() - start) % 60

            time_per_battle = (time.time() - start) * 1.0 / battle_count
            est_time_left = (total_battles - battle_count) * time_per_battle
            hours_left = int(est_time_left // 3600)
            minutes_left = (int(est_time_left) % 3600) // 60
            seconds_left = int(est_time_left % 60)

            processed = f"Processed {battle_count} battles ({round(battle_count * 100.0 / total_battles, 2)}%) and {step_count} steps in {hours}h {minutes}m {seconds}s"
            left = (
                f" // Estimated {hours_left}h {minutes_left}m {seconds_left}s left      "
            )
            print("\r" + processed + left, end="")
            last = time.time()

    hours = int((time.time() - start) // 3600)
    minutes = (int(time.time() - start) % 3600) // 60
    seconds = int((time.time() - start) % 60)
    print(
        f"\nDone! Read {battle_count} battles for a total of {int(step_count)} steps in {hours}h {minutes}m {seconds}s"
    )


if __name__ == "__main__":
    files = []
    with open(sys.argv[1], "rb") as f:
        files = orjson.loads(f.read())[233165:]

    main(files)
