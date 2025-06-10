# -*- coding: utf-8 -*-
import os

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
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
    )

    # Iterate through batches of battles with data_loader
    for states, actions, action_masks, wins, masks in data_loader:

        # Do training things
        pass

    print("Done with training loop!")
