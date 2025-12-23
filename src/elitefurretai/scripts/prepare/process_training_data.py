# -*- coding: utf-8 -*-
"""
This script generates training data from a list of BattleData files by processing them.
It supports:
- Teampreview only data
- Full trajectories

IMPORTANT: As of the teampreview leakage fix (Nov 2024), BattleDataset now applies
team order randomization augmentation automatically. This means:
  - Each battle is randomly augmented during preprocessing
  - Preprocessed data will NOT be identical across runs (randomness from shuffling)
  - To reproduce exact datasets, set random seed before running this script

If you need to regenerate preprocessed data with the augmentation fix, you MUST
re-run this script. Simply loading old preprocessed data will contain the memorization
issue where 88.6% of team patterns deterministically map to one action.
"""

import argparse
import os
import random
import time
import platform
import warnings

import orjson
import torch

from elitefurretai.model_utils import BattleDataset, Embedder, MDBO
from elitefurretai.model_utils.train_utils import format_time, save_compressed


def save_metadata(save_dir, file_trajectory_counts):
    """
    Save metadata file containing trajectory counts for each .pt file.
    This allows OptimizedPreprocessedTrajectoryDataset to skip loading files during initialization.

    Args:
        save_dir: Directory where metadata will be saved
        file_trajectory_counts: List of trajectory counts per file
    """
    metadata_path = os.path.join(save_dir, '_metadata.json')
    metadata = {'file_trajectory_counts': file_trajectory_counts}

    with open(metadata_path, 'wb') as f:
        f.write(orjson.dumps(metadata))

    print(f"\nMetadata saved to {metadata_path}")


# This function takes in a list of filepaths for BattleData files and saves preprocessed trajectories in chunks
def trajectories(
    files,
    save_dir,
    chunk_size=256,
    batch_size=32,
    auxiliary_objectives=False,
    num_workers=4,
    compressed=True,
    augment_teampreview=True,
):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks.
    Each output file will contain up to `chunk_size` trajectories to avoid huge files.

    Args:
        files: List of battle file paths to process
        save_dir: Directory where processed trajectories will be saved
        chunk_size: Number of trajectories per output file
        batch_size: Batch size for data loader
        auxiliary_objectives: Whether to include auxiliary objectives like move order, KO order, and switch order
        num_workers: Number of worker processes for data loading
        augment_teampreview: If True, randomly shuffle team order to prevent memorization (default True)
    """

    print(f"Processing {len(files)} battle files into trajectories...")

    # Create an Embedder instance for feature extraction
    emb = Embedder(format="gen9vgc2023regulationc", feature_set="full", omniscient=False)

    # Create a BattleDataset that yields full trajectories (one per __getitem__)
    dataset = BattleDataset(files, embedder=emb, steps_per_battle=17, augment_teampreview=augment_teampreview)

    # Use a DataLoader to iterate through the dataset one trajectory at a time
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Starting data loading loop")
    os.makedirs(save_dir, exist_ok=True)

    # Process batches and save trajectories
    trajectories = []
    file_trajectory_counts = []  # Track count per file for metadata
    file_count = 0
    start = time.time()
    num_trajectories_processed = 0

    for data in dataloader:
        # data is a dictionary of tensors
        batch_size = data["states"].size(0)

        for i in range(batch_size):
            # Get the valid steps for this trajectory
            valid_steps = int(data["masks"][i].sum().item())
            if valid_steps == 0:
                continue

            to_store = {
                "states": data["states"][i].to(torch.float16),
                "actions": data["actions"][i],
                "action_masks": data["action_masks"][i].to(torch.bool),
                "wins": data["wins"][i].to(torch.float16),
                "masks": data["masks"][i].to(torch.bool),
            }

            if auxiliary_objectives:
                to_store["move_orders"] = data["move_orders"][i]
                to_store["kos"] = data["kos"][i]
                to_store["switches"] = data["switches"][i]

            trajectories.append(to_store)
            num_trajectories_processed += 1

            # Save trajectories in chunks
            if len(trajectories) >= chunk_size:
                save_path = os.path.join(save_dir, f"trajectories_{file_count:06d}.pt")
                if compressed:
                    save_compressed(trajectories, save_path + ".zst")
                else:
                    torch.save(trajectories, save_path)
                file_trajectory_counts.append(len(trajectories))  # Record count
                trajectories = []
                file_count += 1

        time_taken = format_time(time.time() - start)
        time_left = format_time(
            (time.time() - start) / num_trajectories_processed * (len(files) * 2 - num_trajectories_processed)
        )
        perc = (num_trajectories_processed) / len(files) / 2 * 100  # two trajectories per battle
        print(
            f"\033[2K\rProcessed trajectory #{num_trajectories_processed} ({perc:.2f}%) in {time_taken}. Estimated time left: {time_left}",
            end="",
        )

    # Save any remaining trajectories
    if trajectories:
        save_path = os.path.join(save_dir, f"trajectories_{file_count:06d}.pt")
        if compressed:
            save_compressed(trajectories, save_path + ".zst")
        else:
            torch.save(trajectories, save_path)
        file_trajectory_counts.append(len(trajectories))  # Record count

    # Save metadata file
    save_metadata(save_dir, file_trajectory_counts)

    print(
        f"\nProcessing complete in {format_time(time.time() - start)}. {file_count + 1} trajectory files saved to {save_dir}"
    )


def teampreview(
    files,
    save_dir,
    chunk_size=8096,
    batch_size=1,
    auxiliary_objectives=False,
    num_workers=4,
    compressed=True,
    augment_teampreview=True,
):
    """
    Loads battles, processes them into trajectories, and saves them in manageable chunks,
    but saving only the teampreview turns.

    Each output file will contain up to `chunk_size` trajectories to avoid huge files.

    Args:
        files: List of battle file paths to process
        save_dir: Directory where processed teampreview data will be saved
        chunk_size: Number of trajectories per output file
        batch_size: Batch size for data loader
        auxiliary_objectives: Whether to include auxiliary objectives like move order, KO order, and switch order
        augment_teampreview: If True, randomly shuffle team order to prevent memorization (default True)
    """

    print(f"Processing {len(files)} battle files into teampreview data...")

    dataset = BattleDataset(files, steps_per_battle=1, augment_teampreview=augment_teampreview)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Starting training loop")
    os.makedirs(save_dir, exist_ok=True)

    # Process batches and save teampreview data
    teampreview_data = []
    file_trajectory_counts = []  # Track count per file for metadata
    file_count = 0
    num_trajectories_processed = 0
    start = time.time()

    for data in dataloader:

        # data is a dictionary of tensors
        batch_size = data["states"].size(0)

        for i in range(batch_size):
            # Get only the first step (teampreview)
            if data["masks"][i, 0].item() == 0:
                continue

            # Extract teampreview data
            to_store = {
                "states": data["states"][i, 0],
                "actions": data["actions"][i, 0],
                "action_masks": data["action_masks"][i, 0, :MDBO.teampreview_space()],
                "wins": data["wins"][i, 0],
                "masks": data["masks"][i, 0],
            }

            if auxiliary_objectives:
                to_store["move_orders"] = data["move_orders"][i, 0]
                to_store["kos"] = data["kos"][i, 0]
                to_store["switches"] = data["switches"][i, 0]

            teampreview_data.append(to_store)

            num_trajectories_processed += 1

            # Save teampreview data in chunks
            if len(teampreview_data) >= chunk_size:
                save_path = os.path.join(save_dir, f"teampreview_{file_count:06d}.pt")
                if compressed:
                    save_compressed(teampreview_data, save_path + ".zst")
                else:
                    torch.save(teampreview_data, save_path)
                file_trajectory_counts.append(len(teampreview_data))  # Record count
                teampreview_data = []
                file_count += 1

        time_taken = format_time(time.time() - start)
        time_left = format_time(
            (time.time() - start) / (num_trajectories_processed) * (len(files) * 2 - num_trajectories_processed)
        )
        perc = (num_trajectories_processed) / len(files) / 2 * 100  # two trajectories per battle
        print(
            f"\033[2K\rProcessed trajectory #{num_trajectories_processed} ({perc:.2f}%) in {time_taken}. Estimated time left: {time_left}",
            end="",
        )

    # Save any remaining teampreview data
    if teampreview_data:
        save_path = os.path.join(save_dir, f"teampreview_{file_count:06d}.pt")
        if compressed:
            save_compressed(teampreview_data, save_path + ".zst")
        else:
            torch.save(teampreview_data, save_path)
        file_trajectory_counts.append(len(teampreview_data))  # Record count

    # Save metadata file
    save_metadata(save_dir, file_trajectory_counts)

    print(
        f"\nProcessing complete in {format_time(time.time() - start)}. {file_count + 1} trajectory files saved to {save_dir}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Pokémon battle data into train/test/val datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full trajectories with default 90/5/5 split
  python process_training_data.py battles.json output_folder --mode trajectories

  # Process teampreview data with custom 80/10/10 split
  python process_training_data.py battles.json output_folder --mode teampreview --train-pct 0.8 --test-pct 0.1 --val-pct 0.1

  # Process with custom seed and chunk size
  python process_training_data.py battles.json output_folder --mode trajectories --seed 42 --chunk-size 512

  # Process with auxiliary objectives and no compression
  python process_training_data.py battles.json output_folder --mode trajectories --auxiliary-objectives --no-compressed
        """,
    )

    # Required arguments
    parser.add_argument(
        "battle_filepath",
        type=str,
        help="Path to JSON file containing list of battle files",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Output folder where train/test/val subdirectories will be created",
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
        "--train-pct",
        type=float,
        default=0.9,
        help="Percentage of battles for training set (default: 0.9)",
    )
    parser.add_argument(
        "--test-pct",
        type=float,
        default=0.05,
        help="Percentage of battles for test set (default: 0.05)",
    )
    parser.add_argument(
        "--val-pct",
        type=float,
        default=0.05,
        help="Percentage of battles for validation set (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="Random seed for reproducible train/test/val split (default: 21)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Number of trajectories per output file (default: 8192)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading (default: 32)",
    )
    parser.add_argument(
        "--auxiliary-objectives",
        action="store_true",
        help="Include auxiliary objectives like move order, KO order, and switch order",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, (os.cpu_count() or 0)),
        help="Number of worker processes for data loading (default: min(4, number of CPUs))",
    )
    parser.add_argument(
        "--no-compressed",
        action="store_true",
        help="Disable compression of output files (default: files are compressed with zst)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate split percentages
    total_pct = args.train_pct + args.test_pct + args.val_pct
    if abs(total_pct - 1.0) > 0.001:
        raise ValueError(f"Train/test/val percentages must sum to 1.0, got {total_pct}")

    # On Windows, PyTorch's default shared memory strategy can hit limits with large datasets
    # on specifially Windows, that happens with multiprocessing DataLoader
    # See https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
    if platform.system().lower() == 'windows' or "microsoft" in platform.uname()[2].lower():
        # Use file_system sharing to avoid Windows shared memory limits
        torch.multiprocessing.set_sharing_strategy('file_system')
        warnings.filterwarnings('ignore', message='.*socket.send.*')
        print("Heads Up! Using 'file_system' sharing strategy for PyTorch multiprocessing on Windows")

    # Load all battle files
    print(f"Loading battle files from {args.battle_filepath}...")
    with open(args.battle_filepath, "rb") as f:
        all_files = orjson.loads(f.read())
    print(f"Loaded {len(all_files)} battle files")

    # Set random seed for reproducibility
    random.seed(args.seed)
    random.shuffle(all_files)
    print(f"Shuffled battles with seed={args.seed}")

    # Split into train/test/val
    n_train = int(len(all_files) * args.train_pct)
    n_test = int(len(all_files) * args.test_pct)
    n_val = int(len(all_files) * args.val_pct)

    train_files = all_files[:n_train]
    test_files = all_files[n_train:n_train + n_test]
    val_files = all_files[n_train + n_test:n_train + n_test + n_val]

    print(f"\nSplit summary (seed={args.seed}):")
    print(f"  Train: {len(train_files)} battles ({len(train_files) / len(all_files) * 100:.1f}%)")
    print(f"  Test:  {len(test_files)} battles ({len(test_files) / len(all_files) * 100:.1f}%)")
    print(f"  Val:   {len(val_files)} battles ({len(val_files) / len(all_files) * 100:.1f}%)")

    # Create output directories
    train_dir = os.path.join(args.output_folder, "train")
    test_dir = os.path.join(args.output_folder, "test")
    val_dir = os.path.join(args.output_folder, "val")

    # Determine processing function
    process_fn = trajectories if args.mode == "trajectories" else teampreview
    compressed = not args.no_compressed

    print(f"\nProcessing mode: {args.mode}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Auxiliary objectives: {args.auxiliary_objectives}")
    print(f"Compressed: {compressed}")

    # Process each split
    print("\n" + "=" * 60)
    print("PROCESSING TRAINING SET")
    print("=" * 60)
    process_fn(
        train_files,
        train_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        auxiliary_objectives=args.auxiliary_objectives,
        num_workers=args.num_workers,
        compressed=compressed,
    )

    print("\n" + "=" * 60)
    print("PROCESSING TEST SET")
    print("=" * 60)
    process_fn(
        test_files,
        test_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        auxiliary_objectives=args.auxiliary_objectives,
        num_workers=args.num_workers,
        compressed=compressed,
    )

    print("\n" + "=" * 60)
    print("PROCESSING VALIDATION SET")
    print("=" * 60)
    process_fn(
        val_files,
        val_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        auxiliary_objectives=args.auxiliary_objectives,
        num_workers=args.num_workers,
        compressed=compressed,
    )

    print("\n" + "=" * 60)
    print("ALL PROCESSING COMPLETE!")
    print("=" * 60)
    print("Output saved to:")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    print(f"  Val:   {val_dir}")
