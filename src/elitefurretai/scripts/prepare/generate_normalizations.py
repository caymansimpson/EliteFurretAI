import os
import random
import sys
import time

import orjson
import torch
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset, Embedder

"""
=============== How to call via terminal ===============
python src/elitefurretai/scripts/prepare/generate_normalizations.py /Users/username/Desktop/battle_file.jsons 1000 /Users/username/Desktop/

=============== How to use the generated normalizations ===============
# Load them
mean = torch.load("mean.pt")
std = torch.load("std.pt")

# During training loop
for batch in train_loader:
    states, actions, action_mask, wins, masks = batch

    # Reshape mean/std for broadcasting (add temporal dimension)
    # states.shape = (batch_size, steps, features)
    # mean.shape = (1, 1, features) -> broadcasts across batch and steps
    normalized_states = (states - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)

    # Rest of training logic...
"""


# battle_files is a file that contains a list of filenames that contain BattleData to read from
def main(
    battle_files,
    num_battles,
    write_dir,
    format="gen9vgc2023regulationc",
    feature_set="full",
    omniscient=True,
):
    files = []
    with open(battle_files, "rb") as f:
        files = sorted(orjson.loads(f.read()), key=lambda x: random.random())

    # Load data
    embedder = Embedder(format=format, feature_set=feature_set, omniscient=omniscient)
    dataset = BattleDataset(files, embedder=embedder)
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
    )

    # Track progress
    start, last = time.time(), 0
    print(
        f"Starting to calculate means and stds for {num_battles} battles to normalize input data..."
    )

    # Calculate means and stds across all batches
    sum_x = torch.zeros(embedder.embedding_size)
    sum_x2 = torch.zeros(embedder.embedding_size)
    battle_count, step_count = 0, 0

    # Iterate through batches of battles with data_loader
    for states, _, _, _, masks in data_loader:

        # Reshape to 2D with matching dimensions
        batch_size, _, _ = states.shape
        battle_count += batch_size
        states_reshaped = states.view(
            -1, embedder.embedding_size
        )  # (batch_size*steps, embedding_size)
        masks_reshaped = masks.view(-1, 1)  # (batch_size*steps, 1)

        # Apply masks to zero out invalid timesteps
        masked_states = states_reshaped * masks_reshaped

        sum_x += torch.sum(masked_states, dim=0)
        sum_x2 += torch.sum(masked_states**2, dim=0)
        step_count += torch.sum(masks_reshaped)  # Only count valid timesteps (eg turns)

        # Print progress
        if time.time() - start > last + 1:
            hours = int(time.time() - start) // 3600
            minutes = int(time.time() - start) // 60
            seconds = int(time.time() - start) % 60

            time_per_battle = (time.time() - start) * 1.0 / battle_count
            est_time_left = (len(files) - battle_count) * time_per_battle
            hours_left = int(est_time_left // 3600)
            minutes_left = int((est_time_left % 3600) // 60)
            seconds_left = int(est_time_left % 60)

            processed = f"Processed {battle_count} battles ({round(battle_count * 100.0 / num_battles, 2)}%) in {hours}h {minutes}m {seconds}s"
            left = f" with an estimated {hours_left}h {minutes_left}m {seconds_left}s left      "
            print("\r" + processed + left, end="")
            last = time.time()

        # Stop after we read enough battles; we will go slightly over because of batches
        if battle_count >= num_battles:
            break

    # Print progress
    hours = int((time.time() - start) // 3600)
    minutes = int((time.time() - start) // 60)
    seconds = int((time.time() - start) % 60)
    print(
        f"Finished going through data for {battle_count} battles in {hours}h {minutes}m {seconds}s!"
    )
    print("Now generating normalizations...")

    # Calculating means and stds
    mean = sum_x / step_count
    std = torch.sqrt((sum_x2 / step_count) - mean**2)

    # Save them
    mean_file = os.path.join(
        write_dir,
        f"mean_{feature_set}_{format}_{'omniscient' if omniscient else 'nonomniscient'}.pt",
    )
    std_file = os.path.join(
        write_dir,
        f"std_{feature_set}_{format}_{'omniscient' if omniscient else 'nonomniscient'}.pt",
    )
    torch.save(mean, mean_file)
    torch.save(std, std_file)

    # Print stats
    hours = int((time.time() - start) // 3600)
    minutes = int((time.time() - start) // 60)
    seconds = int((time.time() - start) % 60)
    print(
        f"Finished generating feature means and standard devaitions for {battle_count} battles in {hours}h {minutes}m {seconds}s!"
    )
    print(f"Saved them to {mean_file} and {std_file}")


if __name__ == "__main__":
    main(sys.argv[1:], sys.argv[2], sys.argv[3])
