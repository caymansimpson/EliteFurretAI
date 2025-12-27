# -*- coding: utf-8 -*-
"""
This script profiles training and data loading performance
"""
import os
import psutil
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time

from elitefurretai.etl import OptimizedBattleDataLoader, MDBO, Embedder
from elitefurretai.supervised.train_utils import (
    flatten_and_filter,
    topk_cross_entropy_loss,
    load_compressed,
)
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel
from torch.utils.data import Dataset, DataLoader


def profile_training_step(model, dataloader, optimizer, num_batches=150):
    process = psutil.Process(os.getpid())

    # Get baseline memory before training
    baseline_ram = process.memory_info().rss / 1024**3  # GB
    baseline_system = psutil.virtual_memory().percent

    print("\n" + "=" * 80)
    print("BASELINE MEMORY (before training)")
    print("=" * 80)
    print(f"Process RAM usage: {baseline_ram:.2f} GB")
    print(f"System RAM usage: {baseline_system:.1f}%")
    print(f"System RAM available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print("=" * 80)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True
    ) as prof:
        model.train()
        scaler = torch.amp.GradScaler("cuda")  # type: ignore

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            # Get data from dictionary
            with record_function("data loading"):
                states = batch["states"].cuda()
                actions = batch["actions"].cuda()
                action_masks = batch["action_masks"].cuda()
                wins = batch["wins"].cuda()
                masks = batch["masks"].cuda()

            # Mixed precision context
            with torch.amp.autocast("cuda"):  # type: ignore

                # Forward pass
                with record_function("forward pass"):
                    turn_action_logits, teampreview_logits, win_logits = model(states, masks)
                    # Use turn action logits for profiling
                    action_logits = turn_action_logits

                # Processing data and logits
                with record_function("data processing"):
                    masked_action_logits = action_logits.masked_fill(
                        ~action_masks.bool(), float("-inf")
                    )

                    # Use helper for flattening and filtering
                    flat_data = flatten_and_filter(
                        states=states,
                        action_logits=masked_action_logits,
                        actions=actions,
                        win_logits=win_logits,
                        wins=wins,
                        action_masks=action_masks,
                        masks=masks,
                    )
                    if flat_data is None:
                        continue

                    valid_states, valid_action_logits, valid_actions, valid_win_logits, valid_wins = (
                        flat_data
                    )

                # Calculate Losses
                with record_function("loss calculation"):
                    action_loss = topk_cross_entropy_loss(
                        valid_action_logits, valid_actions, weights=None, k=3
                    )
                    win_loss = torch.nn.functional.mse_loss(valid_win_logits, valid_wins.float())
                    loss = action_loss + win_loss

            # Backward pass and optimization step
            with record_function("backwards pass"):
                optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

    # Print summary
    # ===== DATALOADER WORKER ANALYSIS =====
    print(f"\n{'=' * 80}")
    print("DATALOADER WORKER MEMORY")
    print(f"{'=' * 80}")

    # Get all Python processes (including DataLoader workers)
    current_pid = os.getpid()
    parent_process = psutil.Process(current_pid)

    try:
        children = parent_process.children(recursive=True)
        total_worker_mem = 0
        print(f"Main process PID: {current_pid}")
        print(f"Main process RAM: {parent_process.memory_info().rss / 1024**3:.2f} GB")

        if children:
            print(f"\nDataLoader worker processes ({len(children)}):")
            for child in children:
                try:
                    child_mem = child.memory_info().rss / 1024**3
                    total_worker_mem += child_mem
                    print(f"  Worker PID {child.pid}: {child_mem:.2f} GB")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            print(f"\nTotal worker memory: {total_worker_mem:.2f} GB")
            print(f"Total process tree memory: {(parent_process.memory_info().rss / 1024**3 + total_worker_mem):.2f} GB")
        else:
            print("No worker processes found (num_workers=0 or workers already terminated)")
    except Exception as e:
        print(f"Could not analyze worker memory: {e}")

    # ==================== NEW: MEMORY ANALYSIS ====================
    print("\n" + "=" * 80)
    print("CUDA MEMORY BREAKDOWN (Self Memory)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",  # Memory allocated by this operation itself
        row_limit=20,
        max_name_column_width=60
    ))

    print("\n" + "=" * 80)
    print("CUDA MEMORY BREAKDOWN (Total Memory)")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",  # Total memory including children
        row_limit=20,
        max_name_column_width=60
    ))

    print("\n" + "=" * 80)
    print("CPU MEMORY BREAKDOWN")
    print("=" * 80)
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage",
        row_limit=20,
        max_name_column_width=60
    ))

    # ==================== DETAILED MEMORY SUMMARY ====================
    print("\n" + "=" * 80)
    print("TOP MEMORY ALLOCATIONS (by operation)")
    print("=" * 80)

    # Get all events sorted by memory
    events = prof.key_averages()
    memory_events = [(e.key, e.self_cuda_memory_usage, e.cuda_memory_usage, e.count)
                     for e in events if e.self_cuda_memory_usage > 0]
    memory_events.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Operation':<50} {'Self Memory':>15} {'Total Memory':>15} {'Count':>10}")
    print("-" * 90)
    for name, self_mem, total_mem, count in memory_events[:20]:
        print(f"{name[:50]:<50} {self_mem / 1024**2:>13.2f} MB {total_mem / 1024**2:>13.2f} MB {count:>10}")

    # ==================== PEAK MEMORY USAGE ====================
    print("\n" + "=" * 80)
    print("PEAK MEMORY USAGE")
    print("=" * 80)

    total_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
    total_reserved = torch.cuda.max_memory_reserved(0) / 1024**3

    print(f"Peak allocated: {total_allocated:.2f} GB")
    print(f"Peak reserved:  {total_reserved:.2f} GB")
    print(f"Current allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Current reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")


# Profiles dataloader performance
def profile_dataloader(dataloader, num_batches=150):
    """
    Profile your DataLoader to find bottlenecks
    """
    print(f"Profiling DataLoader with {dataloader.num_workers} workers...")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Pin memory: {dataloader.pin_memory}")
    print(f"Prefetch factor: {dataloader.prefetch_factor}")

    if torch.cuda.is_available():
        print(f"\nGPU memory before creating iterator: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")

    # Warmup; let prefetch buffers fill
    print("\nWarmup (filling prefetch buffers)...")
    iterator = iter(dataloader)
    for i in range(min(5, num_batches)):
        batch = next(iterator)
        if i == 0 and torch.cuda.is_available():
            # Move first batch to GPU to initialize CUDA context
            _ = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    print("Warmup complete!\n")

    # Now measure actual performance (after prefetch is warmed up)
    load_times = []
    transfer_times = []
    total_times = []

    # Create CUDA events for precise timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for i in range(num_batches):
        try:
            # Measure end-to-end time per batch (includes load + transfer)
            batch_start = time.time()

            # Get batch (may need to wait for loading)
            batch = next(iterator)

            # Measure transfer time only
            transfer_start = time.time()
            if torch.cuda.is_available():
                _ = {k: v.cuda() for k, v in batch.items()}

            transfer_end = time.time()
            batch_end = transfer_end

            # Calculate times
            total_time = batch_end - batch_start
            transfer_time = transfer_end - transfer_start
            load_time = total_time - transfer_time  # Load is the difference

            total_times.append(total_time)
            load_times.append(load_time)
            transfer_times.append(transfer_time)

            # Record Memory usage and times
            if i % 10 == 0:
                gpu_mem = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
                print(f"Batch {i}: Total={total_time:.3f}s, Load={load_time:.3f}s, Transfer={transfer_time:.3f}s, GPU={gpu_mem:.2f}GB")

        except StopIteration:
            print(f"DataLoader exhausted at batch {i}")
            break

    # Analysis
    avg_total = sum(total_times) / len(total_times)
    avg_load = sum(load_times) / len(load_times)
    avg_transfer = sum(transfer_times) / len(transfer_times)

    print(f"\n{'=' * 50}")
    print("DATALOADER ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Average total time per batch: {avg_total:.3f}s")
    print(f"  - Data loading: {avg_load:.3f}s ({avg_load / avg_total * 100:.1f}%)")
    print(f"  - GPU transfer: {avg_transfer:.3f}s ({avg_transfer / avg_total * 100:.1f}%)")
    print(f"Estimated throughput: {1 / avg_total:.1f} batches/sec")
    print(f"                      {dataloader.batch_size / avg_total:.1f} samples/sec")

    return {
        'avg_total_time': avg_total,
        'avg_load_time': avg_load,
        'avg_transfer_time': avg_transfer,
        'throughput': 1 / avg_total
    }


# Estimates GPU Memory usage and availability; important on Windows
def check_gpu_memory():
    """Check available GPU memory and suggest settings"""
    if not torch.cuda.is_available():
        print("⚠️  No CUDA available!")
        return None

    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
    free_memory = total_memory - reserved_memory

    print(f"\n{'=' * 60}")
    print("GPU MEMORY STATUS")
    print(f"{'=' * 60}")
    print(f"Total GPU memory: {total_memory:.2f} GB")
    print(f"Reserved: {reserved_memory:.2f} GB")
    print(f"Allocated: {allocated_memory:.2f} GB")
    print(f"Free: {free_memory:.2f} GB")

    batch_size = 512
    seq_len = 40
    feature_size = 9223
    num_workers = 2
    prefetch_factor = 1

    # Total batches in memory: num_workers × prefetch_factor
    total_batches = num_workers * prefetch_factor

    # Plus we need to account for ALL tensors in the batch dict:
    # - states: (batch, seq_len, features)
    # - actions: (batch, seq_len) - int64 = 8 bytes
    # - action_masks: (batch, seq_len, num_actions) - bool = 1 byte
    # - wins: (batch,) - float32 = 4 bytes
    # - masks: (batch, seq_len) - float32 = 4 bytes
    states_size = batch_size * seq_len * feature_size * 4
    actions_size = batch_size * seq_len * 8
    action_masks_size = batch_size * seq_len * 2338 * 1  # MDBO.action_space()
    wins_size = batch_size * 4
    masks_size = batch_size * seq_len * 4

    total_per_batch = (states_size + actions_size + action_masks_size + wins_size + masks_size) / (1024**3)
    estimated_pinned = total_per_batch * total_batches

    print("\nDetailed memory calculation:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Feature size: {feature_size}")
    print(f"  Workers: {num_workers}, Prefetch: {prefetch_factor}")
    print(f"  Memory per batch: {total_per_batch:.2f} GB")
    print(f"  Total prefetched batches: {total_batches}")
    print(f"  Estimated pinned memory: {estimated_pinned:.2f} GB")

    if estimated_pinned < free_memory * 0.5:
        print("✅ pin_memory=True should work (if not on WSL)")
        return True
    else:
        print("❌ pin_memory=True will likely cause OOM")
        print(f"   Recommended max batch_size: {int(batch_size * free_memory * 0.5 / estimated_pinned)}")
        return False


# Tests whether we can actually allocate memory on GPU. This is necessary on
# Windows because pin_memory doesn't work with WSL; we needed to ensure that this
# was due to pin_memory and not memory allocation on GPU
def test_raw_gpu_allocation():
    """Test if we can actually allocate what pin_memory needs"""
    print("\n" + "=" * 60)
    print("TESTING RAW GPU ALLOCATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    # Clear any existing allocations
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"Starting GPU memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
    print(f"Starting GPU reserved: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")

    # Test: Can we allocate 1.72GB as a single tensor?
    test_sizes = [0.5, 1.0, 1.5, 1.72, 2.0, 5.0, 10.0]  # GB

    for size_gb in test_sizes:
        try:
            print(f"\n Testing {size_gb:.2f} GB allocation...", end=" ")
            num_elements = int(size_gb * (1024**3) / 4)  # 4 bytes per float32

            # Allocate on GPU
            test_tensor = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
            torch.cuda.synchronize()

            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print("✅ Success!")
            print(f"    Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ OOM at {size_gb:.2f} GB")
                print(f"    Error: {e}")
                return False
            else:
                raise

    print("\n✅ All allocations succeeded!")
    return True


# Tests whether we can allocate memory on CPU and transfer to GPU
# This is a problem on Windows and WSL, where there is a known bug.
# Right now, we're testing whether we can lock memory manually and
# transfer from CPU to GPU
def test_pinned_memory_allocation():
    """Test if we can allocate pinned (page-locked) host memory"""
    print("\n" + "=" * 60)
    print("TESTING PINNED MEMORY ALLOCATION")
    print("=" * 60)

    test_sizes = [0.5, 1.0, 1.5, 1.72, 2.0]  # GB

    for size_gb in test_sizes:
        try:
            print(f"\nTesting {size_gb:.2f} GB pinned memory...", end=" ")
            num_elements = int(size_gb * (1024**3) / 4)  # 4 bytes per float32

            # Allocate pinned memory on CPU
            test_tensor = torch.zeros(num_elements, dtype=torch.float32, pin_memory=True)
            print("✅ Success!")

            # Try to transfer to GPU
            print("  Transferring to GPU...", end=" ")
            gpu_tensor = test_tensor.cuda()
            torch.cuda.synchronize()
            print("✅ Success!")

            # Clean up
            del test_tensor, gpu_tensor
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print("❌ Failed")
            print(f"    Error: {e}")
            return False

    print("\n✅ All pinned allocations succeeded!")
    return True


# Tests whether we can use pin_memory itself explicitly
# This is a problem on Windows and WSL, where there is a known bug. This method
# confirms whether the bug is isolated to pin_memory if all other tests pass
def test_dataloader_pin_memory_simple():
    """Test pin_memory with a minimal dataset"""
    print("\n" + "=" * 60)
    print("TESTING DATALOADER PIN_MEMORY (SIMPLE)")
    print("=" * 60)

    class SimpleDataset(Dataset):
        def __init__(self, size, tensor_size_mb):
            self.size = size
            self.tensor_size = int(tensor_size_mb * (1024**2) / 4)  # elements

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Return a dict like your real dataset
            return {
                'data': torch.randn(self.tensor_size, dtype=torch.float32)
            }

    # Test with increasing sizes
    test_configs = [
        (10, 100),   # 10 samples, 100MB each = 1GB total
        (10, 172),   # 10 samples, 172MB each = 1.72GB total
        (20, 172),   # 20 samples, 172MB each = 3.44GB total
    ]

    for num_samples, tensor_mb in test_configs:
        print(f"\nTesting {num_samples} samples × {tensor_mb}MB = {num_samples * tensor_mb / 1024:.2f}GB")

        dataset = SimpleDataset(num_samples, tensor_mb)

        # Test with pin_memory=True
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=1,
            )

            print("  DataLoader created, fetching batches...", end=" ")
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Just test first 5 batches
                    break
                _ = batch['data'].cuda()

            print("✅ Success!")

        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    print("\n✅ All simple dataloader tests passed!")
    return True


def analyze_trajectory_file(file_path: str, target_steps: int = 17):
    """
    Analyze a trajectory file's memory usage.

    Args:
        file_path: Path to .pt.zst trajectory file
        target_steps: Number of steps to simulate trajectory truncation to

    Returns:
        Dictionary with size analysis
    """

    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {os.path.basename(file_path)}")
    print(f"{'=' * 70}")
    reduction_pct = 0.0

    # 1. Compressed file size on disk
    compressed_size = os.path.getsize(file_path)
    print(f"\n📁 Compressed file size: {compressed_size / (1024**2):.2f} MB")

    # 2. Load and analyze trajectories
    trajectories = load_compressed(file_path, map_location='cpu')
    num_trajectories = len(trajectories)
    print(f"📊 Number of trajectories: {num_trajectories}")

    # 3. Calculate memory for one trajectory
    first_traj = trajectories[0]
    traj_memory = 0

    print(f"\n{'─' * 70}")
    print("SINGLE TRAJECTORY BREAKDOWN:")
    print(f"{'─' * 70}")

    for key, tensor in first_traj.items():
        tensor_size = tensor.element_size() * tensor.numel()
        traj_memory += tensor_size

        print(f"  {key:15s}: shape={str(tuple(tensor.shape)):20s} "
              f"dtype={str(tensor.dtype):15s} size={tensor_size / (1024**2):8.2f} MB")

    print(f"{'─' * 70}")
    print(f"  {'TOTAL':15s}: {traj_memory / (1024**2):53.2f} MB")

    # 4. Calculate batch size (512 trajectories)
    batch_size = 512
    batch_memory = traj_memory * batch_size
    print(f"\n📦 Batch size ({batch_size} trajectories): {batch_memory / (1024**2):.2f} MB")
    print(f"                                   = {batch_memory / (1024**3):.2f} GB")

    # 5. Calculate full file memory when decompressed
    total_memory = traj_memory * num_trajectories
    print(f"\n💾 Total decompressed file in RAM: {total_memory / (1024**2):.2f} MB")
    print(f"                                  = {total_memory / (1024**3):.2f} GB")

    compression_ratio = compressed_size / total_memory
    print(f"\n📉 Compression ratio: {compression_ratio:.2%} "
          f"({1.0 / compression_ratio:.1f}x compression)")

    # 6. Simulate truncation to target_steps
    print(f"\n{'─' * 70}")
    print(f"SIMULATING TRUNCATION TO {target_steps} STEPS:")
    print(f"{'─' * 70}")

    actual_steps = first_traj['states'].shape[0]
    print(f"  Current steps: {actual_steps}")
    print(f"  Target steps:  {target_steps}")

    if actual_steps <= target_steps:
        print(f"  ⚠️  Trajectory already ≤ {target_steps} steps, no truncation needed")
        truncated_traj_memory = traj_memory
        truncated_batch_memory = batch_memory
        truncated_total_memory = total_memory
    else:
        # Calculate size if we truncate to target_steps
        truncated_traj_memory = 0

        for key, tensor in first_traj.items():
            if tensor.dim() > 0 and tensor.shape[0] == actual_steps:
                # This tensor has a time dimension
                truncated_size = tensor.element_size() * (tensor.numel() // actual_steps) * target_steps
            else:
                # No time dimension (shouldn't happen in your data)
                truncated_size = tensor.element_size() * tensor.numel()

            truncated_traj_memory += truncated_size

        truncated_batch_memory = truncated_traj_memory * batch_size
        truncated_total_memory = truncated_traj_memory * num_trajectories

        reduction_pct = (1 - truncated_traj_memory / traj_memory) * 100

        print(f"\n  Truncated trajectory size: {truncated_traj_memory / (1024**2):.2f} MB "
              f"({reduction_pct:.1f}% reduction)")
        print(f"  Truncated batch size:      {truncated_batch_memory / (1024**2):.2f} MB "
              f"({truncated_batch_memory / (1024**3):.2f} GB)")
        print(f"  Truncated file in RAM:     {truncated_total_memory / (1024**2):.2f} MB "
              f"({truncated_total_memory / (1024**3):.2f} GB)")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Compressed on disk:        {compressed_size / (1024**2):8.2f} MB")
    print(f"  Decompressed in RAM:       {total_memory / (1024**3):8.2f} GB")
    print(f"  Per trajectory:            {traj_memory / (1024**2):8.2f} MB")
    print(f"  Batch (512 trajectories):  {batch_memory / (1024**3):8.2f} GB")

    if actual_steps > target_steps:
        print(f"\n  If truncated to {target_steps} steps:")
        print(f"    File in RAM:             {truncated_total_memory / (1024**3):8.2f} GB")
        print(f"    Batch (512 trajectories): {truncated_batch_memory / (1024**3):8.2f} GB")
        print(f"    Memory savings:          {(total_memory - truncated_total_memory) / (1024**3):8.2f} GB "
              f"({reduction_pct:.1f}%)")

    return {
        'compressed_size_mb': compressed_size / (1024**2),
        'num_trajectories': num_trajectories,
        'trajectory_size_mb': traj_memory / (1024**2),
        'batch_size_gb': batch_memory / (1024**3),
        'total_decompressed_gb': total_memory / (1024**3),
        'compression_ratio': compression_ratio,
        'actual_steps': actual_steps,
        'target_steps': target_steps,
        'truncated_trajectory_size_mb': truncated_traj_memory / (1024**2),
        'truncated_batch_size_gb': truncated_batch_memory / (1024**3),
        'truncated_total_gb': truncated_total_memory / (1024**3),
    }


def main(data_path):

    torch.multiprocessing.set_sharing_strategy('file_system')

    print("Starting!")

    # Initialize CUDA context first to see baseline memory
    if torch.cuda.is_available():
        print("Initializing CUDA context...")
        _ = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        print(f"CUDA initialized. Memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB\n")

    # =============================== FIRST, TEST GPU MEMORY ALLOCATION ===============================
    # Run comprehensive diagnostics
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE PIN_MEMORY DIAGNOSTICS")
    print("=" * 60)

    raw_alloc_ok = test_raw_gpu_allocation()
    pinned_alloc_ok = test_pinned_memory_allocation()
    simple_dl_ok = test_dataloader_pin_memory_simple()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Raw GPU allocation: {'✅ PASS' if raw_alloc_ok else '❌ FAIL'}")
    print(f"Pinned memory allocation: {'✅ PASS' if pinned_alloc_ok else '❌ FAIL'}")
    print(f"Simple DataLoader pin_memory: {'✅ PASS' if simple_dl_ok else '❌ FAIL'}")

    # Explicit debugging and recommendations
    if not (raw_alloc_ok and pinned_alloc_ok and simple_dl_ok):
        print("\n⚠️  Some tests failed. This suggests a system-level issue:")
        if not raw_alloc_ok:
            print("  - GPU memory allocation failing - check nvidia-smi")
        if not pinned_alloc_ok:
            print("  - Pinned memory allocation failing - check system RAM")
        if not simple_dl_ok:
            print("  - DataLoader pin_memory failing - likely WSL2/driver issue")
        print("\nSkipping pin_memory for your dataset...")
    else:
        print("\n✅ All basic tests passed - issue is specific to your dataset")

    # ============================== SECOND, PROFILE DATA FILES ===============================
    analyze_trajectory_file(os.path.join(data_path, "trajectories_000000.pt.zst"))

    # =============================== THIRD, PROFILE DATALOADER ===============================
    # Initialize dataset and dataloader
    start = time.time()

    # Windows memory sharing optimization that needs to be done
    torch.multiprocessing.set_sharing_strategy('file_system')
    successful_loader = None

    # You can play around with the parameters here to find the right balance
    embedder, train_loader = None, None
    try:
        embedder = Embedder(format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False)
        print(f"\nEmbedder initialized. Size: {embedder.embedding_size}! Loading datasets...")
        train_loader = OptimizedBattleDataLoader(data_path, embedder=embedder, batch_size=512)
        print(f"DataLoader created in {time.time() - start:.2f}s! Size: {len(train_loader)} batches.")

        # Try to profile
        result = profile_dataloader(train_loader)

        # If we get here, it worked!
        successful_loader = train_loader
        print("\n✅ Successfully profiled")
        print("Results:", result)

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # Catch RuntimeError too!
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "acceleratorerror" in error_msg.lower():
            print("\n❌ OOM")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # Some other error
            raise
    except Exception as e:
        # Check if it's an OOM wrapped in another exception
        error_msg = str(e)
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise

    if successful_loader is None:
        print("\n❌ Could not create a working DataLoader!")
        return

    # =============================== THIRD, PROFILE TRAINING ===============================
    # Initialize model
    print("\n" + "=" * 60)
    print("Now onto training profiling... Initializing model...")
    print("=" * 60)
    assert embedder is not None, "Embedder must be initialized"

    # Use a simplified config for profiling (similar to three_headed_transformer.py)
    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=[1024, 512],
        late_layers=[512, 256],
        lstm_layers=2,
        lstm_hidden_size=256,
        dropout=0.1,
        gated_residuals=False,
        early_attention_heads=4,
        late_attention_heads=4,
        use_grouped_encoder=False,
        group_sizes=None,
        teampreview_head_layers=[256],
        teampreview_head_dropout=0.1,
        teampreview_attention_heads=4,
        turn_head_layers=[256],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=17,
    ).to("cuda")
    print(f"Finished loading data and model! for a total of {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
    )

    print("Initialized model! Starting training profiling...")
    assert train_loader is not None, "train_loader must be initialized"
    profile_training_step(model, train_loader, optimizer)


if __name__ == "__main__":
    main(sys.argv[1])
