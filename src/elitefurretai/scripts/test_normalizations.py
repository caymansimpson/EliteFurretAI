import time
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EasyDataset(Dataset):
    def __init__(self, num=100, emb_length=10):
        self.num = num
        self.emb_length = emb_length

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return torch.normal(torch.arange(self.emb_length).float(), torch.ones(self.emb_length).float())


def main(num_samples, batch_size):

    print(f"starting testing with {num_samples} samples and {batch_size} batch size")

    # Caculate progress
    start, last, total = time.time(), 0, num_samples

    # Calculate means and stds across all batches
    total_sum = None
    total_sq_sum = None
    total_count = 0

    dataset = EasyDataset(num_samples)

    # Generate Data Loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(os.cpu_count() or 1, 4),
        shuffle=True
    )

    print("Done loading dataset and dataloader")

    for batch_X in data_loader:
        batch_sum = torch.sum(batch_X, dim=0)
        batch_sq_sum = torch.sum(batch_X ** 2, dim=0)
        batch_count = batch_X.shape[0]

        if total_sum is None and total_sq_sum is None:
            total_sum = batch_sum
            total_sq_sum = batch_sq_sum
        elif total_sum is not None and total_sq_sum is not None:
            total_sum += batch_sum
            total_sq_sum += batch_sq_sum

        total_count += batch_count

        if time.time() - start > last + 5:  # Print every 5 seconds
            hours = int((time.time() - start) // 3600)
            minutes = int((time.time() - start) // 60)
            seconds = int((time.time() - start) % 60)
            print(
                f"Calculated means and stds for {total_count}/{total} battles "
                f"({round(total_count / total * 100, 2)}% complete) in "
                f"{hours}h {minutes}m {seconds}s"
            )
            last += 5

    assert total_sum is not None and total_sq_sum is not None
    means = (total_sum / total_count).tolist()
    variances = (total_sq_sum / total_count - (total_sum / total_count) ** 2).tolist()
    stds = [np.maximum(np.sqrt(var), 1e-6) for var in variances]  # Avoid division by zero

    hours = int((time.time() - start) // 3600)
    minutes = int((time.time() - start) // 60)
    seconds = int((time.time() - start) % 60)
    print(f"Done generating normalizations for {total} battles in {hours}h {minutes}m {seconds}s!")

    print(means)
    print(stds)


if __name__ == "__main__":
    main(1000, 32)
