# -*- coding: utf-8 -*-
"""
This script generates training data from a list of BattleData files by processing them.
It supports:
- Teampreview only data
- Full trajectories
"""

import argparse
import os
import time

import orjson
import torch
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset, Embedder


# This function takes in a list of filepaths for BattleData files and saves preprocessed trajectories in chunks
def trajectories(
    battle_filepath, save_dir, beginning_pct, end_pct, chunk_size=2048, batch_size=512
):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks.
    Each output file will contain up to `chunk_size` trajectories to avoid huge files.

    Args:
        battle_filepath: Path to JSON file containing list of battle files
        save_dir: Directory where processed trajectories will be saved
        beginning_pct: Starting percentage of battles to process (0.0-1.0)
        end_pct: Ending percentage of battles to process (0.0-1.0)
        chunk_size: Number of trajectories per output file
        batch_size: Batch size for data loader
    """

    # Load the list of battle file paths from a JSON file
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())
    files = files[int(len(files) * beginning_pct) : int(len(files) * end_pct)]
    print(f"Loaded {len(files)} files!")
    emb = Embedder(format="gen9vgc2025regh", feature_set="full", omniscient=False)

    # Create a BattleDataset that yields full trajectories (one per __getitem__)
    dataset = BattleDataset(files, embedder=emb, steps_per_battle=40)
    # Use a DataLoader to iterate through the dataset one trajectory at a time
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    start, num_batches = time.time(), 0

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)
    chunk = []  # Holds the current chunk of trajectories to be saved
    file_idx = 0  # Index for naming output files

    # Iterate through all trajectories in the dataset
    for idx, (states, actions, action_masks, wins, masks) in enumerate(dataloader):
        batch_len = states.shape[0]
        for i in range(batch_len):
            traj = (
                states[i],
                actions[i],
                action_masks[i],
                wins[i],
                masks[i],
            )
            chunk.append(traj)
            if len(chunk) >= chunk_size:
                torch.save(
                    chunk, os.path.join(save_dir, f"trajectories_{file_idx:05d}.pt")
                )
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


def teampreview(
    battle_filepath, save_dir, beginning_pct, end_pct, chunk_size=8096, batch_size=1
):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks,
    but saving only the teampreview turns.

    Each output file will contain up to `chunk_size` trajectories to avoid huge files.

    Args:
        battle_filepath: Path to JSON file containing list of battle files
        save_dir: Directory where processed teampreview data will be saved
        beginning_pct: Starting percentage of battles to process (0.0-1.0)
        end_pct: Ending percentage of battles to process (0.0-1.0)
        chunk_size: Number of trajectories per output file
        batch_size: Batch size for data loader
    """

    # Load the list of battle file paths from a JSON file
    files = []
    with open(battle_filepath, "rb") as f:
        files = orjson.loads(f.read())

    files = files[int(len(files) * beginning_pct) : int(len(files) * end_pct)]

    print(f"Loaded {len(files)} files!")

    dataset = BattleDataset(files, steps_per_battle=1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    start, num_batches = time.time(), 0

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)
    chunk = []  # Holds the current chunk of trajectories to be saved
    file_idx = 0  # Index for naming output files

    # Iterate through all trajectories in the dataset
    for idx, (states, actions, action_masks, wins, masks) in enumerate(dataloader):
        # Remove batch dimension (since batch_size=1)
        for i in range(states.shape[0]):
            chunk.append(
                (
                    states[i],
                    actions[i],
                    action_masks[i],
                    wins[i],
                    masks[i],
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
        torch.save(chunk, os.path.join(save_dir, f"teampreview_{file_idx:05d}.pt"))

    print(f"\nAll batches saved to {save_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Pokémon battle data into training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full trajectories (all turns)
  python process_training_data.py battles.json data/processed_data 0.0 1.0 --mode trajectories --chunk-size 2048 --batch-size 512

  # Process only teampreview data
  python process_training_data.py battles.json data/teampreview_data 0.0 1.0 --mode teampreview --chunk-size 8096

  # Process only the first 10% of battles for trajectories
  python process_training_data.py battles.json data/processed_data 0.0 0.1 --mode trajectories

  # Process the second half of battles for teampreview
  python process_training_data.py battles.json data/processed_data 0.5 1.0 --mode teampreview
        """,
    )

    # Required arguments
    parser.add_argument(
        "battle_filepath",
        type=str,
        help="Path to JSON file containing list of battle files",
    )
    parser.add_argument(
        "save_dir", type=str, help="Directory where processed data will be saved"
    )
    parser.add_argument(
        "beginning_pct",
        type=float,
        help="Starting percentage of battles to process (0.0-1.0)",
    )
    parser.add_argument(
        "end_pct", type=float, help="Ending percentage of battles to process (0.0-1.0)"
    )

    # Optional arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trajectories", "teampreview"],
        default="trajectories",
        help="Processing mode: full trajectories or teampreview only (default: trajectories)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Number of trajectories per output file (default: 2048 for trajectories, 8096 for teampreview)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for data loading (default: 512 for trajectories, 1 for teampreview)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set default chunk sizes and batch sizes based on mode if not specified
    if args.chunk_size is None:
        args.chunk_size = 2048 if args.mode == "trajectories" else 8096

    if args.batch_size is None:
        args.batch_size = 512 if args.mode == "trajectories" else 1

    # Call the appropriate function based on the selected mode
    if args.mode == "trajectories":
        print(
            f"Processing full trajectories with chunk size {args.chunk_size} and batch size {args.batch_size}"
        )
        trajectories(
            args.battle_filepath,
            args.save_dir,
            args.beginning_pct,
            args.end_pct,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
    else:  # teampreview mode
        print(
            f"Processing teampreview data with chunk size {args.chunk_size} and batch size {args.batch_size}"
        )
        teampreview(
            args.battle_filepath,
            args.save_dir,
            args.beginning_pct,
            args.end_pct,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
