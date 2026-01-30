import platform
import random
from typing import Any, Dict, List, Optional, Union

import torch

from elitefurretai.etl.battle_dataset import OptimizedPreprocessedTrajectoryDataset
from elitefurretai.etl.embedder import Embedder


def _trajectory_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that efficiently stacks trajectory dictionaries.

    This function is more memory-efficient than the default collate because it:
    1. Pre-allocates the output tensors
    2. Uses contiguous memory to avoid shared memory fragmentation
    3. Converts to the target dtype explicitly to avoid dtype inference issues
    """
    if not batch:
        return {}

    # Get the keys from the first element
    keys = batch[0].keys()
    result = {}

    for key in keys:
        # Stack tensors for this key
        tensors = [item[key] for item in batch]
        stacked = torch.stack(tensors, dim=0)
        # Make contiguous to ensure clean memory layout for sharing
        result[key] = stacked.contiguous()

    return result


class OptimizedPreprocessedSampler(torch.utils.data.Sampler):
    """
    Sampler that ensures each worker only gets indices for its assigned files.
    This maximizes cache hit rates.
    """

    def __init__(
        self,
        dataset: OptimizedPreprocessedTrajectoryDataset,
        shuffle: bool = False,
        shuffle_files: bool = False,
    ):
        """
        Args:
            dataset: WorkerAwareTrajectoryDataset instance
            shuffle: Whether to shuffle trajectories within each file
            shuffle_files: Whether to shuffle file order (maintains locality)
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.shuffle_files = shuffle_files

        # Group indices by file for locality
        self.file_groups: Dict[int, List[int]] = {}
        for global_idx, (file_idx, local_idx) in enumerate(dataset._idx_map):
            if file_idx not in self.file_groups:
                self.file_groups[file_idx] = []
            self.file_groups[file_idx].append(global_idx)

    def __iter__(self):
        # Get file order
        file_indices = list(self.file_groups.keys())
        if self.shuffle_files:
            random.shuffle(file_indices)

        # Yield indices file by file (maintains locality)
        for file_idx in file_indices:
            trajectory_indices = self.file_groups[file_idx].copy()
            if self.shuffle:
                random.shuffle(trajectory_indices)

            for idx in trajectory_indices:
                yield idx

    def __len__(self):
        return len(self.dataset)


class OptimizedBattleDataLoader(torch.utils.data.DataLoader):
    """
    All-in-one DataLoader for OptimizedPreprocessedSampler that uses
    OptimizedPreprocessedTrajectorySampler to maximize cache hit rates.

    The parameters that are set here were ones that were deemed optimal from explorations
    in src/elitefurretai/scripts/analyze/training_profiler.py. The overall speedup was 3x
    compared to native and naive multiprocessed loading of preprocessed files.

    Completely abstracts everything away from the user so they only have to use this dataloader.

    Note on shared memory issues (Bus errors):
    - On WSL2/Linux with many workers, PyTorch's default tensor sharing can exhaust
      shared memory or file descriptors, causing "Bus error" crashes
    - This class uses a custom collate function and 'forkserver' multiprocessing
      context to mitigate these issues
    - If you still encounter bus errors, reduce num_workers, files_per_worker,
      or prefetch_factor
    """

    def __init__(
        self,
        # OptimizedPreprocessedTrajectoryDataset parameters
        folder_path: str,
        metadata_filename: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        files_per_worker: Union[int, str, None] = "default",
        # Optimal DataLoader parameters, from scripts/analyze/training_profiler.py
        batch_size: int = 512,
        shuffle: bool = False,
        num_workers: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = 1,
    ):
        # Ensure file_system sharing strategy is set for WSL/Windows compatibility
        # This must be done before creating workers
        if platform.system() == "Windows" or "microsoft" in platform.uname()[2].lower():
            torch.multiprocessing.set_sharing_strategy("file_system")

        # Build dataset kwargs conditionally, and build dataset with them
        dataset_kwargs: Dict[str, Any] = {"folder_path": folder_path}
        if metadata_filename is not None:
            dataset_kwargs["metadata_filename"] = metadata_filename
        if embedder is not None:
            dataset_kwargs["embedder"] = embedder
        if files_per_worker is not None:
            dataset_kwargs["files_per_worker"] = files_per_worker
        dataset = OptimizedPreprocessedTrajectoryDataset(**dataset_kwargs)

        # Create sampler with shuffle capability
        sampler = OptimizedPreprocessedSampler(
            dataset,
            shuffle=shuffle,
            shuffle_files=shuffle,
        )

        # Determine multiprocessing context
        # 'forkserver' is more robust than 'fork' for complex objects and avoids
        # shared memory issues that can cause bus errors
        mp_context = None
        if num_workers > 0 and platform.system() != "Windows":
            try:
                mp_context = torch.multiprocessing.get_context("forkserver")
            except ValueError:
                # forkserver not available, fall back to default
                mp_context = None

        # Initialize DataLoader WITHOUT shuffle parameter (since we use sampler)
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            collate_fn=_trajectory_collate_fn,
            multiprocessing_context=mp_context,
        )
