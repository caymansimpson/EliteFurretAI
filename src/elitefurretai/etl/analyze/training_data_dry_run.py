# -*- coding: utf-8 -*-
"""
This script goes through training data to make sure there are no surprises.
It also helps us understand performance; can be run with a profiler.
"""

import os
import sys
import time

import orjson
from torch.utils.data import DataLoader

from elitefurretai.etl import BattleDataset


# This function takes in a list of filepaths for BattleData files
def main(files):
    dataset = BattleDataset(files)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
    )
    start, num_batches = time.time(), 0

    # Iterate through batches of battles with data_loader
    for states, actions, action_masks, wins, masks in dataloader:
        # Do training things
        pass

        # Print progress
        now = time.time()
        h, m, s = int(now - start) // 3600, int(now - start) // 60, int(now - start) % 60
        time_per_batch = (now - start) * 1.0 / (num_batches + 1)
        t_left = (len(dataloader) - num_batches) * time_per_batch
        h_left, m_left, s_left = (
            int(t_left // 3600),
            int((t_left % 3600) // 60),
            int(t_left % 60),
        )

        assert dataloader.batch_size is not None
        processed = f"Processed {num_batches * dataloader.batch_size} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {h}h {m}m {s}s"
        left = f" with an estimated {h_left}h {m_left}m {s_left}s left in this epoch"
        print("\033[2K\r" + processed + left, end="")
        num_batches += 1

    print("Done with training loop!")


# Usage: python3 src/elitefurretai/scripts/analyze/training_data_dry_run.py <filepath> <num_battles>
if __name__ == "__main__":
    files = []
    with open(sys.argv[1], "rb") as f:
        files = orjson.loads(f.read())[: int(sys.argv[2])]
    main(files)
