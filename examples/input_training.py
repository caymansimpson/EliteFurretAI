# -*- coding: utf-8 -*-
"""Example: turning BattleData files into per-step training tensors.

Two illustrative entry points:

* ``one_file_example`` — walks a single BattleData JSON turn by turn, runs the
  Embedder on each state, and prints the chosen action.
* ``list_of_files`` — wraps a list of files in ``BattleDataset`` and iterates
  via a PyTorch ``DataLoader``, the shape used by the supervised trainers in
  ``src/elitefurretai/supervised``.

Run directly to exercise both against a small bundled fixture::

    python examples/input_training.py
"""

import glob
import os
import time

import orjson
from torch.utils.data import DataLoader

from elitefurretai.etl import (
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

        # Embedder is gen-keyed; battle.gen exposes the int (e.g. 9).
        embedder = Embedder(gen=iter.battle.gen, feature_set="raw", omniscient=True)

        while iter.next_input() and not iter.battle.finished:
            # Get the last input command found by the iterator
            input = iter.last_input
            if input is None:
                break

            request = iter.simulate_request()
            if request is not None:
                iter.battle.parse_request(request)

            features = embedder.embed_to_vector(iter.battle)  # type: ignore
            order = iter.last_order()

            print(
                f"Got {len(features)} features -> {order.message} on {perspective}'s turn #{iter.battle.turn}"
            )

        print(f"Done with perspective {perspective}!\n")

    print("Done with both perspectives!")


# This function takes a list of filepaths and generates training data in batches from them
def list_of_files(files):
    # BattleDataset now requires an embedder so each step is vectorized at load time.
    embedder = Embedder(gen=9, feature_set="raw", omniscient=True)
    dataset = BattleDataset(files, embedder=embedder)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
    )
    start, num_batches = time.time(), 0

    # Iterate through batches of battles with data_loader
    for batch in dataloader:
        # batch is a dict with keys: states, actions, action_masks, wins,
        # move_orders, kos, switches, masks. See BattleDataset.__getitem__.
        _ = batch["states"]

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

    print("\nDone with training loop!")


if __name__ == "__main__":
    fixture_dir = "data/fixture/gen9vgc2023regc_logs"
    files = sorted(glob.glob(os.path.join(fixture_dir, "*.json")))
    assert files, f"No fixtures found in {fixture_dir}"

    print(f"--- one_file_example on {files[0]} ---")
    one_file_example(files[0])

    print(f"\n--- list_of_files on {len(files)} fixture(s) ---")
    list_of_files(files)
