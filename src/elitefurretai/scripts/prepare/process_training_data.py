# -*- coding: utf-8 -*-
"""
This script generates training data from a list of BattleData files by processing them.
It supports:
- Teampreview only data
- Full trajectories
"""

import os
import sys
import time

import orjson
import torch
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset, Embedder


# This function takes in a list of filepaths for BattleData files and saves preprocessed trajectories in chunks
def trajectories(battle_filepath, save_dir, beginning_pct, end_pct, chunk_size=1024):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks.
    Each output file will contain up to `chunk_size` trajectories to avoid huge files.
    """

    # Load the list of battle file paths from a JSON file
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())
    files = files[int(len(files) * beginning_pct) : int(len(files) * end_pct)]
    print(f"Loaded {len(files)} files!")
    emb = Embedder(format="gen9vgc2025regi", feature_set="full", omniscient=False)

    # Create a BattleDataset that yields full trajectories (one per __getitem__)
    dataset = BattleDataset(files, embedder=emb, steps_per_battle=40)
    # Use a DataLoader to iterate through the dataset one trajectory at a time
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    start, num_batches = time.time(), 0

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)
    chunk = []  # Holds the current chunk of trajectories to be saved
    file_idx = 0  # Index for naming output files

    # Iterate through all trajectories in the dataset
    for idx, (states, actions, action_masks, wins, masks) in enumerate(dataloader):
        # Remove batch dimension (since batch_size=1)
        traj = (
            states.squeeze(0),
            actions.squeeze(0),
            action_masks.squeeze(0),
            wins.squeeze(0),
            masks.squeeze(0),
        )
        chunk.append(traj)
        # If chunk is full, save it to disk and start a new chunk
        if len(chunk) >= chunk_size:
            torch.save(chunk, os.path.join(save_dir, f"trajectories_{file_idx:05d}.pt"))
            file_idx += 1
            chunk = []

        # Print progress information
        now = time.time()
        h, m, s = int(now - start) // 3600, int(now - start) // 60, int(now - start) % 60
        time_per_batch = (now - start) * 1.0 / (num_batches + 1)
        t_left = (len(dataloader) - num_batches) * time_per_batch
        h_left, m_left, s_left = (
            int(t_left // 3600),
            int((t_left % 3600) // 60),
            int(t_left % 60),
        )

        processed = f"Processed {num_batches * int(dataloader.batch_size or 1)} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {h}h {m}m {s}s"
        left = f"{h_left}h {m_left}m {s_left}s left"
        print(f"\033[2K\r {processed} with an estimated {left}", end="")
        num_batches += 1

    # Save any remaining trajectories in the last chunk
    if chunk:
        torch.save(chunk, os.path.join(save_dir, f"trajectories_{file_idx:05d}.pt"))

    print(f"\nAll batches saved to {save_dir}")


def teampreview(battle_filepath, save_dir, beginning_pct, end_pct, chunk_size=8096):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks,
    but saving only the teampreview turns.

    Each output file will contain up to `chunk_size` trajectories to avoid huge files.
    """

    # Load the list of battle file paths from a JSON file
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())

    files = files[int(len(files) * beginning_pct) : int(len(files) * end_pct)]

    print(f"Loaded {len(files)} files!")

    dataset = BattleDataset(files, steps_per_battle=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    start, num_batches = time.time(), 0

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)
    chunk = []  # Holds the current chunk of trajectories to be saved
    file_idx = 0  # Index for naming output files

    # Iterate through all trajectories in the dataset
    for idx, (states, actions, action_masks, wins, masks) in enumerate(dataloader):
        # Remove batch dimension (since batch_size=1)
        chunk.append(
            (
                states.squeeze(0),
                actions.squeeze(0),
                action_masks.squeeze(0),
                wins.squeeze(0),
                masks.squeeze(0),
            )
        )
        # If chunk is full, save it to disk and start a new chunk
        if len(chunk) >= chunk_size:
            torch.save(chunk, os.path.join(save_dir, f"teampreview_{file_idx:05d}.pt"))
            file_idx += 1
            chunk = []

        # Print progress information
        now = time.time()
        h, m, s = int(now - start) // 3600, int(now - start) // 60, int(now - start) % 60
        time_per_batch = (now - start) * 1.0 / (num_batches + 1)
        t_left = (len(dataloader) - num_batches) * time_per_batch
        h_left, m_left, s_left = (
            int(t_left // 3600),
            int((t_left % 3600) // 60),
            int(t_left % 60),
        )

        processed = f"Processed {num_batches * int(dataloader.batch_size or 1)} battles ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {h}h {m}m {s}s"
        left = f"{h_left}h {m_left}m {s_left}s left"
        print(f"\033[2K\r {processed} with an estimated {left}", end="")
        num_batches += 1

    # Save any remaining trajectories in the last chunk
    if chunk:
        torch.save(chunk, os.path.join(save_dir, f"trajectories_{file_idx:05d}.pt"))

    print(f"\nAll batches saved to {save_dir}")


# Usage: python process_training_data.py <battle_filepath> <save_dir> <beginning_pct> <end_pct>
if __name__ == "__main__":
    trajectories(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    # teampreview(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
