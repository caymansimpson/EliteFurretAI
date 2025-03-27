# -*- coding: utf-8 -*-
"""This script goes through training data to make sure there are no surprises
"""

import sys
import os.path
import random
import time
from torch.utils.data import DataLoader

import torch

from elitefurretai.model_utils import BattleDataset, BattleData


def main(files, frmt):

    # For tracking progress through iterations
    count, start, last, step, total_battles, batch_size = 0, time.time(), 0, 0, len(files), 10

    print(f"Starting dry run for {total_battles} battles...")

    dataset = BattleDataset(
        files=files,
        format=frmt,
        label_type="filename",
        bd_eligibility_func=BattleData.is_valid_for_supervised_learning
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(os.cpu_count() or 1, 4),
    )

    print("Dataset and Dataloader created... starting to load data...")
    for batch_X, batch_Y, batch_mask in data_loader:
        count += batch_size
        step += int(sum(map(lambda x: torch.sum(x, dim=0), batch_mask)))

        if time.time() - start > last + 10:  # Print every 10 seconds
            hours = int((time.time() - start) // 3600)
            minutes = int((time.time() - start) // 60)
            seconds = int((time.time() - start) % 60)
            print(
                f"Processed {count} battles and {step} steps in {hours}h {minutes}m {seconds}s"
                + f" ({round(count / total_battles * 100, 3)}% battles done)..."
            )
            last += 10

    hours = int((time.time() - start) // 3600)
    minutes = int((time.time() - start) // 60)
    seconds = int((time.time() - start) % 60)
    print(f"Done! Read {count} battles with {step} steps in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":

    # Get Battles, sort randomly and take 1000
    files = sorted(map(lambda x: os.path.join(sys.argv[1], x), os.listdir(sys.argv[1])), key=lambda x: random.random())
    files = files[:1000]

    # Profile code too
    main(files, "gen9vgc2023regulationc")
