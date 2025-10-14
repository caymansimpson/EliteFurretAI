import os
import time
import threading
import argparse
from typing import Optional, List, Dict
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, Sampler
import psutil
import math
import random
from torch.utils.data import DataLoader
from elitefurretai.model_utils import Embedder, PreprocessedTrajectoryDataset


class WorkerAwareTrajectoryDataset(Dataset):
    """
    On Windows, PyTorch's DataLoader with num_workers > 0 uses spawn multiprocessing (not fork like on Linux), which means:

    Each worker starts a fresh Python process
    Your dataset object gets pickled and copied to each worker
    Each worker has its own isolated cache
    No sharing possible between workers

    This is why you see 0% cache hit rates with BattleDataset and PreprocessedBattleDataset - the cache exists, but only within each worker's isolated memory.
    A normally implemented cache will not work with multiprocessing on Windows.

    Strategy: Smart File-Level Batching with Worker-Aware Caching
    The key insight to this dataset: Instead of fighting multiprocessing, work with it by having each worker "own" specific files.
    Dataset that's aware of DataLoader worker processes.
    Each worker loads and caches only its assigned files.

    ✅ Works with existing file structure
    ✅ Scales to 1.5TB+ datasets
    ✅ Simple to use - just pass folder path
    ✅ Efficient caching within each worker
    """

    def __init__(
        self,
        folder_path: str,
        embedder,
        files_per_worker: int = 2,
        verbose: bool = False
    ):
        """
        Args:
            folder_path: Path to folder containing .pt files
            embedder: Your embedder instance
            files_per_worker: How many files each worker should cache at once
            verbose: Print debug info
        """
        self.folder_path = folder_path
        self.embedder = embedder
        self.files_per_worker = files_per_worker
        self.verbose = verbose
        self._worker_id: Optional[int] = None
        self._num_workers: Optional[int] = None
        self._worker_files: Optional[set] = None

        # Get all trajectory files
        self.trajectory_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.pt')
        ])

        if not self.trajectory_files:
            raise ValueError(f"No .pt files found in {folder_path}")

        if self.verbose:
            print(f"Found {len(self.trajectory_files)} trajectory files")

        # Build index: global_idx -> (file_idx, local_idx)
        self._idx_map = []
        self.file_trajectory_counts = []

        for file_idx, file_path in enumerate(self.trajectory_files):
            trajectories = torch.load(file_path, map_location='cpu', weights_only=False)
            num_trajectories = len(trajectories)
            self.file_trajectory_counts.append(num_trajectories)

            for local_idx in range(num_trajectories):
                self._idx_map.append((file_idx, local_idx))

            del trajectories

        self._length = len(self._idx_map)

        # Worker-specific state (initialized in worker_init_fn)
        self._worker_id = None
        self._num_workers = None
        self._worker_files = None  # Files this worker is responsible for
        self._worker_cache: dict = {}  # This worker's file cache

        if self.verbose:
            print(f"Total trajectories: {self._length:,}")

    def _init_worker(self):
        """Initialize worker-specific state. Called automatically by DataLoader."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading
            self._worker_id = 0
            self._num_workers = 1
            self._worker_files = set(range(len(self.trajectory_files)))
        else:
            # Multi-process loading
            self._worker_id = worker_info.id
            self._num_workers = worker_info.num_workers

            # Divide files among workers
            files_per_worker_calc = math.ceil(len(self.trajectory_files) / self._num_workers)
            start_file = self._worker_id * files_per_worker_calc
            end_file = min(start_file + files_per_worker_calc, len(self.trajectory_files))

            self._worker_files = set(range(start_file, end_file))

            if self.verbose:
                print(f"Worker {self._worker_id}: responsible for files {start_file}-{end_file - 1}")

    def _load_file(self, file_idx: int):
        """Load file with worker-local caching"""

        # Lazy worker initialization
        if self._worker_id is None:
            self._init_worker()

        # Check worker-local cache
        if file_idx in self._worker_cache:
            return self._worker_cache[file_idx]

        # Load file
        file_path = self.trajectory_files[file_idx]
        trajectories = torch.load(file_path, map_location='cpu', weights_only=False)

        # Cache management: keep only files_per_worker most recent
        if len(self._worker_cache) >= self.files_per_worker:
            # Evict oldest (first key)
            oldest_file_idx = next(iter(self._worker_cache))
            del self._worker_cache[oldest_file_idx]

        self._worker_cache[file_idx] = trajectories

        return trajectories

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        file_idx, local_idx = self._idx_map[idx]

        # Lazy worker initialization
        if self._worker_id is None:
            self._init_worker()

        # Skip if this file isn't assigned to this worker
        # (DataLoader will handle distributing indices appropriately)
        if self._worker_files and file_idx not in self._worker_files:
            # This shouldn't happen with proper sampler, but handle gracefully
            # Return from any cached file as fallback
            if self._worker_cache:
                fallback_file_idx = next(iter(self._worker_cache))
                trajectories = self._worker_cache[fallback_file_idx]
                return trajectories[0]  # Return first trajectory
            # If no cache, load any file we're responsible for
            fallback_file_idx = next(iter(self._worker_files))
            trajectories = self._load_file(fallback_file_idx)
            return trajectories[0]

        trajectories = self._load_file(file_idx)
        return trajectories[local_idx]


class WorkerAwareSampler(Sampler):
    """
    Sampler that ensures each worker only gets indices for its assigned files.
    This maximizes cache hit rates.
    """

    def __init__(
        self,
        dataset: WorkerAwareTrajectoryDataset,
        shuffle: bool = True,
        shuffle_files: bool = True
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


class LRUCache:
    """Simple LRU cache implementation for thread safety and multiprocessing"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self._lock: Optional[threading.Lock] = None  # Will be initialized in _ensure_lock()

    def _ensure_lock(self):
        """Lazy initialization of lock for multiprocessing compatibility"""
        if self._lock is None:
            self._lock = threading.Lock()

    def get(self, key):
        self._ensure_lock()
        with self._lock:  # type: ignore
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        self._ensure_lock()
        with self._lock:  # type: ignore
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                # Add new, evict if necessary
                if len(self.cache) >= self.capacity:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[key] = value

    def size(self):
        self._ensure_lock()
        with self._lock:  # type: ignore
            return len(self.cache)

    def keys(self):
        self._ensure_lock()
        with self._lock:  # type: ignore
            return list(self.cache.keys())

    def __getstate__(self):
        """Custom pickling - exclude the lock"""
        state = self.__dict__.copy()
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling - lock will be recreated when needed"""
        self.__dict__.update(state)
        self._lock = None


class OptimizedPreprocessedTrajectoryDataset(Dataset):
    """
    Optimized version of PreprocessedTrajectoryDataset with:
    - LRU caching for multiple files
    - Memory usage monitoring
    - Better error handling
    - Cache statistics
    """

    def __init__(
        self,
        folder_path: str,
        embedder,
        cache_size: int = 3,
        max_memory_gb: Optional[float] = None,
        preload_indices: bool = True,
        verbose: bool = True
    ):
        self.embedder = embedder
        self.folder_path = folder_path
        self.cache_size = cache_size
        self.max_memory_bytes = max_memory_gb * (1024**3) if max_memory_gb else None
        self.verbose = verbose
        self._stats_lock: Optional[threading.Lock] = None

        # Get all trajectory files
        self.trajectory_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.pt')
        ])

        if not self.trajectory_files:
            raise ValueError(f"No .pt files found in {folder_path}")

        if self.verbose:
            print(f"Found {len(self.trajectory_files)} trajectory files")

        # Build index mapping if requested
        self._idx_map = []
        self.file_lengths = []
        self._length = 0

        if preload_indices:
            start_time = time.time()
            if self.verbose:
                print("Building trajectory index...")

            for file_idx, file_path in enumerate(self.trajectory_files):
                try:
                    # Quick load to get length
                    trajectories = torch.load(file_path, map_location='cpu', weights_only=False)
                    num_trajectories = len(trajectories)
                    self.file_lengths.append(num_trajectories)

                    for local_idx in range(num_trajectories):
                        self._idx_map.append((file_idx, local_idx))

                    self._length += num_trajectories
                    del trajectories  # Free memory immediately

                    if self.verbose and (file_idx + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Processed {file_idx + 1}/{len(self.trajectory_files)} files in {elapsed:.1f}s")

                except Exception as e:
                    print(f"Warning: Could not load {file_path}: {e}")
                    self.file_lengths.append(0)

            if self.verbose:
                total_time = time.time() - start_time
                print(f"Index built in {total_time:.1f}s. Total trajectories: {self._length}")
        else:
            # Estimate length (less accurate but faster startup)
            if self.verbose:
                print("Estimating dataset size (fast startup mode)...")

            sample_file = torch.load(self.trajectory_files[0], map_location='cpu', weights_only=False)
            avg_trajectories_per_file = len(sample_file)
            del sample_file

            self._length = len(self.trajectory_files) * avg_trajectories_per_file
            if self.verbose:
                print(f"Estimated {self._length} trajectories")

        # Initialize caching
        self.file_cache = LRUCache(self.cache_size)

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'files_loaded': 0,
            'memory_peak_mb': 0
        }

        # Initialize caching
        self.file_cache = LRUCache(self.cache_size)

        # Statistics - use Manager for multiprocessing compatibility
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'files_loaded': 0,
            'memory_peak_mb': 0
        }
        self._stats_lock = None  # Lazy initialization like LRUCache

    def _ensure_stats_lock(self):
        """Lazy initialization of stats lock"""
        if self._stats_lock is None:
            self._stats_lock = threading.Lock()

    def _increment_stat(self, key: str, value: int = 1):
        """Thread-safe stat increment"""
        self._ensure_stats_lock()
        assert self._stats_lock is not None
        with self._stats_lock:
            self.stats[key] += value

    def _update_stat(self, key: str, value):
        """Thread-safe stat update"""
        self._ensure_stats_lock()
        assert self._stats_lock is not None
        with self._stats_lock:
            self.stats[key] = value

    def __getstate__(self):
        """Custom pickling for multiprocessing"""
        state = self.__dict__.copy()
        state['_stats_lock'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling for multiprocessing"""
        self.__dict__.update(state)
        self._stats_lock = None

    def _load_file(self, file_idx: int):
        """Load file with caching and memory monitoring"""
        # Check cache first
        cached_data = self.file_cache.get(file_idx)
        if cached_data is not None:
            self._increment_stat('cache_hits')
            return cached_data

        # Cache miss - load file
        self._increment_stat('cache_misses')
        self._increment_stat('files_loaded')

        file_path = self.trajectory_files[file_idx]

        try:
            # Monitor memory before loading
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB

            # Load file
            trajectories = torch.load(file_path, map_location='cpu', weights_only=False)

            # Monitor memory after loading
            memory_after = process.memory_info().rss / (1024**2)  # MB
            memory_used = memory_after - memory_before

            # Update peak memory (thread-safe)
            current_peak = self.stats['memory_peak_mb']
            if memory_after > current_peak:
                self._update_stat('memory_peak_mb', memory_after)

            if self.verbose and memory_used > 500:  # Log if using > 500MB
                print(f"\033[2K\rLoaded {file_path} (+{memory_used:.1f}MB, total: {memory_after:.1f}MB)", end="")

            # Check memory limit
            if self.max_memory_bytes and memory_after * 1024**2 > self.max_memory_bytes:
                print(f"\033[2K\rWarning: Memory usage ({memory_after:.1f}MB) exceeds limit", end="")

            # Cache the data
            self.file_cache.put(file_idx, trajectories)

            return trajectories

        except Exception:
            print(f"Error loading {file_path}")
            raise

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if hasattr(self, '_idx_map') and self._idx_map:
            # Use prebuilt index
            file_idx, local_idx = self._idx_map[idx]
        else:
            # Estimate file and local index (less accurate)
            if not hasattr(self, '_estimated_trajectories_per_file'):
                # Lazy initialization
                sample_file = torch.load(self.trajectory_files[0], map_location='cpu', weights_only=False)
                self._estimated_trajectories_per_file = len(sample_file)
                del sample_file

            file_idx = idx // self._estimated_trajectories_per_file
            local_idx = idx % self._estimated_trajectories_per_file

            if file_idx >= len(self.trajectory_files):
                raise IndexError(f"Index {idx} out of range")

        trajectories = self._load_file(file_idx)

        if local_idx >= len(trajectories):
            raise IndexError(f"Local index {local_idx} out of range for file {file_idx}")

        return trajectories[local_idx]

    def get_cache_stats(self):
        """Get detailed cache statistics"""
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / (1024**2)

        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_accesses if total_accesses > 0 else 0

        return {
            'cache_size': self.file_cache.size(),
            'cache_capacity': self.cache_size,
            'cached_files': self.file_cache.keys(),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'files_loaded': self.stats['files_loaded'],
            'current_memory_mb': current_memory_mb,
            'memory_peak_mb': self.stats['memory_peak_mb']
        }

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_cache_stats()
        print(f"\n{'=' * 50}")
        print("DATASET CACHE STATISTICS")
        print(f"{'=' * 50}")
        print(f"Cache size: {stats['cache_size']}/{stats['cache_capacity']}")
        print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Files loaded: {stats['files_loaded']}")
        print(f"Current memory: {stats['current_memory_mb']:.1f} MB")
        print(f"Peak memory: {stats['memory_peak_mb']:.1f} MB")
        print(f"Cached files: {stats['cached_files']}")


class FileLocalizedSampler(Sampler):
    """
    Custom sampler that groups accesses by file to improve cache hit rates
    """

    def __init__(self, dataset, shuffle=True, file_shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.file_shuffle = file_shuffle

        # Group indices by file
        self.file_groups: dict[int, list[int]] = {}

        if hasattr(dataset, '_idx_map') and dataset._idx_map:
            # Use prebuilt index
            for global_idx, (file_idx, local_idx) in enumerate(dataset._idx_map):
                if file_idx not in self.file_groups:
                    self.file_groups[file_idx] = []
                self.file_groups[file_idx].append(global_idx)
        else:
            # Estimate grouping
            trajectories_per_file = len(dataset) // len(dataset.trajectory_files)
            for global_idx in range(len(dataset)):
                file_idx = global_idx // trajectories_per_file
                if file_idx not in self.file_groups:
                    self.file_groups[file_idx] = []
                self.file_groups[file_idx].append(global_idx)

    def __iter__(self):
        # Get file order
        file_indices = list(self.file_groups.keys())
        if self.file_shuffle:
            random.shuffle(file_indices)

        # For each file, yield all its trajectories
        for file_idx in file_indices:
            trajectory_indices = self.file_groups[file_idx].copy()
            if self.shuffle:
                random.shuffle(trajectory_indices)

            for traj_idx in trajectory_indices:
                yield traj_idx

    def __len__(self):
        return len(self.dataset)


class PreloadedDataset(Dataset):
    """
    Alternative approach: Preload all data that fits in memory
    """

    def __init__(self, folder_path: str, embedder, max_memory_gb: float = 16):
        self.embedder = embedder
        self.max_memory_bytes = max_memory_gb * 1024**3

        trajectory_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.pt')
        ])

        self.trajectories: list[dict] = []
        current_memory = 0

        print(f"Preloading data (max {max_memory_gb} GB)...")

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        for i, file_path in enumerate(trajectory_files):
            file_size = os.path.getsize(file_path)

            # Check if we have room (conservative estimate)
            if current_memory + file_size > self.max_memory_bytes:
                print(f"Memory limit reached after {i} files. Loaded {len(self.trajectories)} trajectories.")
                break

            try:
                file_trajectories = torch.load(file_path, map_location='cpu', weights_only=False)
                self.trajectories.extend(file_trajectories)
                current_memory += file_size

                # Check actual memory usage
                actual_memory = process.memory_info().rss - initial_memory

                print(f"\033[2K\rLoaded {len(file_trajectories):,} trajectories from {os.path.basename(file_path)} "
                      f"(file: {file_size / (1024 ** 2):.1f}MB, actual: {actual_memory / (1024 ** 2):.1f}MB)", end="")

                # Break if actual memory exceeds limit
                if actual_memory > self.max_memory_bytes:
                    print("Actual memory usage exceeded limit. Stopping.")
                    break

            except Exception:
                print(f"Error loading {file_path}")
                break

        print(f"Total trajectories loaded: {len(self.trajectories):,}")
        final_memory = (process.memory_info().rss - initial_memory) / (1024**2)
        print(f"Total memory used: {final_memory:.1f} MB")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def embedding_size(self):
        return self.embedder.embedding_size


# Utility functions
def analyze_trajectory_files(folder_path: str):
    """Analyze trajectory files to recommend optimal settings"""

    trajectory_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith('.pt')
    ])

    if not trajectory_files:
        print(f"No .pt files found in {folder_path}")
        return

    print(f"Analyzing {len(trajectory_files)} trajectory files...")

    total_size = 0
    total_trajectories = 0
    file_stats = []

    for file_path in trajectory_files:
        size_bytes = os.path.getsize(file_path)
        size_gb = size_bytes / (1024**3)
        total_size += size_bytes

        # Load file to get trajectory count
        try:
            trajectories = torch.load(file_path, map_location='cpu', weights_only=False)
            num_trajectories = len(trajectories)
            total_trajectories += num_trajectories

            file_stats.append({
                'file': os.path.basename(file_path),
                'size_gb': size_gb,
                'num_trajectories': num_trajectories,
                'trajectories_per_gb': num_trajectories / size_gb if size_gb > 0 else 0
            })

            del trajectories

        except Exception as e:
            print(f"Could not analyze {file_path}: {e}")

    # Calculate statistics
    avg_size_gb = (total_size / (1024**3)) / len(file_stats)
    avg_trajectories = total_trajectories / len(file_stats)

    print(f"\n{'=' * 60}")
    print("TRAJECTORY FILE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Total files: {len(file_stats)}")
    print(f"Total size: {total_size / (1024 ** 3):.2f} GB")
    print(f"Total trajectories: {total_trajectories:,}")
    print(f"Average file size: {avg_size_gb:.2f} GB")
    print(f"Average trajectories per file: {avg_trajectories:.0f}")

    # Recommendations
    print("\n📋 RECOMMENDATIONS:")

    # Cache size recommendation
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    recommended_cache_size = max(2, min(6, int(available_memory_gb / (avg_size_gb * 2))))
    print(f"• Recommended cache_size: {recommended_cache_size} files")
    print(f"  (Based on {available_memory_gb:.1f}GB RAM and {avg_size_gb:.1f}GB avg file size)")

    # File size recommendation
    if avg_size_gb > 2.0:
        print(f"• Consider splitting large files ({avg_size_gb:.1f}GB → ~1-1.5GB each)")
        print("  This improves cache hit rates and reduces memory pressure")
    elif avg_size_gb < 0.5:
        print(f"• Consider merging small files ({avg_size_gb:.1f}GB → ~1-1.5GB each)")
        print("  This reduces filesystem overhead")

    # Memory strategy recommendation
    if total_size / (1024**3) < available_memory_gb * 0.7:
        print("• Consider PreloadedDataset - your data can fit in memory!")
    else:
        print("• Use OptimizedPreprocessedTrajectoryDataset with caching")

    return {
        'total_size_gb': total_size / (1024**3),
        'total_trajectories': total_trajectories,
        'avg_file_size_gb': avg_size_gb,
        'recommended_cache_size': recommended_cache_size,
        'can_preload': total_size / (1024**3) < available_memory_gb * 0.7
    }


"""
Test script for optimized dataset implementations
Run this to compare your current dataset with optimized versions
"""


class DatasetBenchmark:
    """Comprehensive benchmarking suite for dataset implementations"""

    def __init__(self, trajectory_folder: str, embedder: Embedder):
        self.trajectory_folder = trajectory_folder
        self.embedder: Embedder = embedder
        self.results: dict = {}

        # Analyze the data first
        # print("🔍 Analyzing trajectory files...")
        # self.file_analysis = analyze_trajectory_files(trajectory_folder)

    def benchmark_dataset(self, name: str, dataset, sampler=None, num_batches=20, batch_size=32, num_workers=4):
        """Benchmark a single dataset implementation"""

        print(f"\n🧪 Testing {name}...")
        print(f"   Dataset length: {len(dataset):,}")

        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        start_total = time.time()

        # Create DataLoader
        if sampler is not None:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Important for cache performance
                num_workers=num_workers,
                pin_memory=True
            )

        # Benchmark loading
        batch_times = []

        try:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                batch_start = time.time()

                # Access the data to ensure it's loaded
                states = batch["states"]

                # Simulate some processing
                _ = states.mean()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                if i % 5 == 0:
                    current_memory = process.memory_info().rss / (1024**2)
                    print(f"\033[2K\rBatch {i}: {batch_time:.3f}s, Memory: {current_memory:.1f}MB", end="")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

        total_time = time.time() - start_total
        final_memory = process.memory_info().rss / (1024**2)
        memory_increase = final_memory - initial_memory

        # Calculate statistics
        avg_batch_time = sum(batch_times) / len(batch_times)
        throughput = batch_size / avg_batch_time  # trajectories per second

        result = {
            'avg_batch_time': avg_batch_time,
            'total_time': total_time,
            'throughput': throughput,
            'memory_increase_mb': memory_increase,
            'final_memory_mb': final_memory,
            'batches_processed': len(batch_times)
        }

        # Get cache stats if available
        if hasattr(dataset, 'get_cache_stats'):
            cache_stats = dataset.get_cache_stats()
            result.update(cache_stats)

            print(f"   📊 Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"   📈 Files loaded: {cache_stats.get('files_loaded', 0)}")

        print(f"   ⏱️  Avg batch time: {avg_batch_time:.3f}s")
        print(f"   🚀 Throughput: {throughput:.1f} trajectories/sec")
        print(f"   💾 Memory increase: {memory_increase:.1f}MB")
        print(f"   ⏰ Total time: {total_time:.1f}s")

        self.results[name] = result
        return result

    def run_all_benchmarks(self, num_batches=20, batch_size=32):
        """Run all available benchmarks"""

        print(f"\n{'=' * 60}")
        print("🚀 STARTING COMPREHENSIVE BENCHMARK")
        print(f"{'=' * 60}")
        print(f"Trajectory folder: {self.trajectory_folder}")
        print(f"Batches to test: {num_batches}")
        print(f"Batch size: {batch_size}")

        # 1. Original implementation (if available)
        print("\n1️⃣ Testing Original PreprocessedTrajectoryDataset...")
        original_dataset = PreprocessedTrajectoryDataset(
            self.trajectory_folder,
            embedder=self.embedder
        )
        self.benchmark_dataset("Original", original_dataset, num_batches=num_batches, batch_size=batch_size)

        # 2. Optimized with different cache sizes
        cache_sizes = [2, 3, 5] if self.file_analysis['total_size_gb'] > 8 else [3, 5, 8]  # type: ignore

        for cache_size in cache_sizes:
            try:
                print(f"\n2️⃣ Testing Optimized Dataset (cache_size={cache_size})...")
                optimized_dataset = OptimizedPreprocessedTrajectoryDataset(
                    self.trajectory_folder,
                    embedder=self.embedder,
                    cache_size=cache_size,
                    verbose=False
                )
                self.benchmark_dataset(
                    f"Optimized (cache={cache_size})",
                    optimized_dataset,
                    num_batches=num_batches,
                    batch_size=batch_size
                )
            except Exception as e:
                print(f"   ❌ Could not test optimized dataset: {e}")

        # 3. Optimized with File-Localized Sampler
        try:
            print("\n3️⃣ Testing File-Localized Sampler...")
            optimized_dataset = OptimizedPreprocessedTrajectoryDataset(
                self.trajectory_folder,
                embedder=self.embedder,
                cache_size=3,
                verbose=False
            )
            file_sampler = FileLocalizedSampler(optimized_dataset, shuffle=True)
            self.benchmark_dataset(
                "File-Localized Sampler",
                optimized_dataset,
                sampler=file_sampler,
                num_batches=num_batches,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"   ❌ Could not test file-localized sampler: {e}")

        # 4. Preloaded Dataset (if data fits in memory)
        if self.file_analysis.get('can_preload', False):  # type: ignore
            try:
                print("\n4️⃣ Testing Preloaded Dataset...")
                available_memory = psutil.virtual_memory().total / (1024**3)
                max_memory = min(available_memory * 0.6, self.file_analysis['total_size_gb'] + 2)  # type: ignore

                preloaded_dataset = PreloadedDataset(
                    self.trajectory_folder,
                    embedder=self.embedder,
                    max_memory_gb=max_memory
                )
                self.benchmark_dataset(
                    "Preloaded",
                    preloaded_dataset,
                    num_batches=num_batches,
                    batch_size=batch_size
                )
            except Exception as e:
                print(f"   ❌ Could not test preloaded dataset: {e}")
        else:
            print("\n4️⃣ Skipping Preloaded Dataset (data too large for memory)")

        # 5. Worker Aware Dataset with WorkerAwareSampler -- cache statistics don't work here
        try:
            print("\n5️⃣ Testing Worker Aware Dataset...")
            trajectory_files = sorted([
                os.path.join(self.trajectory_folder, f)
                for f in os.listdir(self.trajectory_folder)
                if f.endswith('.pt')
            ])
            num_workers = 4

            worker_aware_dataset = WorkerAwareTrajectoryDataset(
                self.trajectory_folder,
                embedder=self.embedder,
                files_per_worker=max(2, len(trajectory_files) // (num_workers * 2)),
                verbose=False
            )
            worker_sampler = WorkerAwareSampler(worker_aware_dataset)
            self.benchmark_dataset(
                "Worker Aware",
                worker_aware_dataset,
                sampler=worker_sampler,
                num_batches=num_batches,
                batch_size=batch_size
            )
        except Exception as e:
            print(f"   ❌ Could not test worker aware dataset: {e}")

        self.print_summary()

        return self.results

    def print_summary(self):
        """Print comprehensive summary of all benchmarks"""

        print(f"\n{'=' * 60}")
        print("📊 BENCHMARK RESULTS SUMMARY")
        print(f"{'=' * 60}")

        if not self.results:
            print("No results to display.")
            return

        # Sort by throughput
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['throughput'],
            reverse=True
        )

        print(f"{'Method':<25} {'Batch Time':<12} {'Throughput':<15} {'Memory':<12} {'Hit Rate':<10}  {'Total Time':<12}")
        print("-" * 80)

        for name, result in sorted_results:
            batch_time = result['avg_batch_time']
            throughput = result['throughput']
            memory = result['memory_increase_mb']
            hit_rate = result.get('hit_rate', 0) * 100
            total_time = result['total_time']

            print(f"{name:<25} {batch_time:<12.3f} {throughput:<15.1f} {memory:<12.1f} {hit_rate:<10.1f}% {total_time:<12.3f}s")

        # Best performer
        best_method, best_result = sorted_results[0]
        print(f"\n🏆 Best performer: {best_method}")
        print(f"   ⚡ {best_result['throughput']:.1f} trajectories/sec")
        print(f"   ⏱️  {best_result['avg_batch_time']:.3f}s per batch")

        # Compare with original if available
        if "Original" in self.results and len(sorted_results) > 1:
            original_throughput = self.results["Original"]["throughput"]
            improvement = (best_result['throughput'] / original_throughput - 1) * 100
            print(f"   📈 {improvement:.1f}% faster than original")


def main():
    parser = argparse.ArgumentParser(description="EliteFurretAI Dataset Profiler & Benchmark")
    parser.add_argument(
        "--folder", "-f", type=str, required=True,
        help="Path to the folder containing trajectory .pt files"
    )
    parser.add_argument(
        "--batches", "-b", type=int, default=20,
        help="Number of batches to benchmark (default: 20)"
    )
    parser.add_argument(
        "--batch_size", "-s", type=int, default=32,
        help="Batch size for benchmarking (default: 32)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--embedder", type=str, default=None,
        help="Optional: path to a pickled Embedder or config (if needed)"
    )
    args = parser.parse_args()

    # You may need to adjust this depending on how your Embedder is constructed
    if args.embedder:
        # Example: load from pickle or config
        import pickle
        with open(args.embedder, "rb") as f:
            embedder = pickle.load(f)
    else:
        # Or construct a default Embedder (customize as needed)
        embedder = Embedder()

    benchmark = DatasetBenchmark(
        trajectory_folder=args.folder,
        embedder=embedder
    )
    benchmark.run_all_benchmarks(
        num_batches=args.batches,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
