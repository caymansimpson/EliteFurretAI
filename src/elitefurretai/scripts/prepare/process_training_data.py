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
from elitefurretai.scripts.train.train_utils import format_time


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)
    # Process batches and save trajectories
    trajectories = []
    batch_count = 0
    file_count = 0
    start = time.time()

    for data in dataloader:
        # data is a dictionary of tensors
        batch_size = data["states"].size(0)

        for i in range(batch_size):
            # Get the valid steps for this trajectory
            valid_steps = int(data["masks"][i].sum().item())
            if valid_steps == 0:
                continue

            # Extract data for valid steps only
            trajectories.append(
                {
                    "states": data["states"][i].clone(),
                    "actions": data["actions"][i].clone(),
                    "action_masks": data["action_masks"][i].clone(),
                    "wins": data["wins"][i].clone(),
                    "move_orders": data["move_orders"][i].clone(),
                    "kos": data["kos"][i].clone(),
                    "switches": data["switches"][i].clone(),
                    "masks": data["masks"][i].clone(),
                }
            )

            time_taken = format_time(time.time() - start)
            time_left = format_time(
                (time.time() - start) / (file_count + 1) * (len(files) - file_count)
            )
            print(
                f"\033[2K\rProcessed trajectory #{file_count} in {time_taken}. Estimated time left: {time_left}",
                end="",
            )

        batch_count += 1

        # Save trajectories in chunks
        if len(trajectories) >= chunk_size:
            save_path = os.path.join(save_dir, f"trajectories_{file_count}.pt")
            torch.save(trajectories, save_path)
            trajectories = []
            file_count += 1

    # Save any remaining trajectories
    if trajectories:
        save_path = os.path.join(save_dir, f"trajectories_{file_count}.pt")
        torch.save(trajectories, save_path)

    print(
        f"Processing complete in {format_time(time.time() - start)}. {file_count + 1} trajectory files saved to {save_dir}"
    )


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

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)

    # Process batches and save teampreview data
    teampreview_data = []
    batch_count = 0
    file_count = 0
    start = time.time()

    for data in dataloader:
        # data is a dictionary of tensors
        batch_size = data["states"].size(0)

        for i in range(batch_size):
            # Get only the first step (teampreview)
            if data["masks"][i, 0].item() == 0:
                continue

            # Extract teampreview data
            teampreview_data.append(
                {
                    "states": data["states"][i, 0].clone(),
                    "actions": data["actions"][i, 0].clone(),
                    "action_masks": data["action_masks"][i, 0].clone(),
                    "wins": data["wins"][i, 0].clone(),
                    "move_orders": data["move_orders"][i, 0].clone(),
                    "kos": data["kos"][i, 0].clone(),
                    "switches": data["switches"][i, 0].clone(),
                    "masks": data["masks"][i, 0].clone(),
                }
            )

            time_taken = format_time(time.time() - start)
            time_left = format_time(
                (time.time() - start) / (file_count + 1) * (len(files) - file_count)
            )
            print(
                f"\033[2K\rProcessed trajectory #{file_count} in {time_taken}. Estimated time left: {time_left}",
                end="",
            )

        batch_count += 1

        # Save teampreview data in chunks
        if len(teampreview_data) >= chunk_size:
            save_path = os.path.join(save_dir, f"teampreview_{file_count}.pt")
            torch.save(teampreview_data, save_path)
            teampreview_data = []
            file_count += 1

    # Save any remaining teampreview data
    if teampreview_data:
        save_path = os.path.join(save_dir, f"teampreview_{file_count}.pt")
        torch.save(teampreview_data, save_path)

    print(
        f"Processing complete in {format_time(time.time() - start)}. {file_count + 1} trajectory files saved to {save_dir}"
    )


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
