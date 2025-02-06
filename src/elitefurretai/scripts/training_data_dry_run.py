# -*- coding: utf-8 -*-
"""This script goes through training data to make sure there are no surprises
"""

import sys
import os.path
import random
import time
import cProfile
import pstats
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset, BattleData


def main(files, frmt):

    # For tracking progress through iterations
    count, start, last, step, total_battles, batch_size = 0, time.time(), 0, 0, len(files), 32

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
        num_workers=1  # min(os.cpu_count() or 1, 4),
    )

    print("Dataset and Dataloader created... starting to load data...")
    for batch_X, batch_Y in data_loader:
        count += batch_size
        step += len(batch_X)

        if time.time() - start > last + 10:  # Print every 10 seconds
            hours = (time.time() - start) // 3600
            minutes = (time.time() - start) // 60
            seconds = (time.time() - start) % 60
            print(
                f"Processed {batch_size} battles and {step} steps in {hours}h {minutes}m {seconds}s"
                + f" ({round(count / total_battles * 100, 3)}% battles done)..."
            )
            last += 10

    hours = (time.time() - start) // 3600
    minutes = (time.time() - start) // 60
    seconds = (time.time() - start) % 60
    print(f"Done! Read {count} battles with {step} steps in {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":

    # Get Battles, sort randomly and take 1000
    files = sorted(map(lambda x: os.path.join(sys.argv[1], x), os.listdir(sys.argv[1])), key=lambda x: random.random())
    files = files[:1000]

    # Profile code too
    with cProfile.Profile() as pr:
        main(files, "gen9vgc2023regulationc")

        print("\nResults of profiling:")
        stats = pstats.Stats(pr)
        stats.sort_stats('cumtime')  # Sort by cumulative time
        stats.print_stats(100)  # Print top 100 lines
