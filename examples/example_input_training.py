# -*- coding: utf-8 -*-
import os
import time

import orjson
from torch.utils.data import DataLoader

from elitefurretai.model_utils import (
    BattleData,
    BattleDataset,
    BattleIterator,
    Embedder,
)


# This function reads through a single BattleData file and generates training data from that file
def one_file_example(filename):
    for perspective in ["p1", "p2"]:

        bd = None
        with open(filename, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        iter = BattleIterator(bd, perspective=perspective)
        assert iter.battle.format is not None

        embedder = Embedder(format=iter.battle.format, feature_set="full", omniscient=True)

        while iter.next_input() and not iter.battle.finished:

            # Get the last input command found by the iterator
            input = iter.last_input
            if input is None:
                break

            request = iter.simulate_request()
            if request is not None:
                iter.battle.parse_request(request)

            features = embedder.featurize_double_battle(iter.battle)  # type: ignore
            order = iter.last_order()

            print(
                f"Got {len(features)} features -> {order.message} on {perspective}'s turn #{iter.battle.turn}"
            )

        print(f"Done with perspective {perspective}!\n")

    print("Done with both perspectives!")


# This function takes a list of filepaths and generates training data in batches from them
def list_of_files(files):
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

        processed = f"Processed {num_batches * dataloader.batch_size} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {h}h {m}m {s}s"
        left = f" with an estimated {h_left}h {m_left}m {s_left}s left in this epoch"
        print("\033[2K\r" + processed + left, end="")
        num_batches += 1

    print("Done with training loop!")
