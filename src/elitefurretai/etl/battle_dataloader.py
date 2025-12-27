from typing import Dict, List, Union, Optional, Any
import random
import torch

from elitefurretai.etl.battle_dataset import OptimizedPreprocessedTrajectoryDataset
from elitefurretai.etl.embedder import Embedder


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

        # Initialize DataLoader WITHOUT shuffle parameter (since we use sampler)
        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
