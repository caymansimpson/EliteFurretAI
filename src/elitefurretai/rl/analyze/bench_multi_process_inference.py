# -*- coding: utf-8 -*-
"""Microbenchmark: 2-thread vs 2-process inference parallelism on the same GPU.

Direct evidence for the Plan C decision in
planning/stage2/2026-05-15-00-12-compile-lock-serialization-fix-plan.md.

Hypothesis under test: Run C plateaued at 3.5 traj/s on full curriculum
because Python-side encode/decode/sampling work in each InferenceService
thread serializes on the GIL (the explicit `_COMPILE_LOCK` was reverted
in production — see Updates section of that plan doc). If true,
moving services to separate processes should give close to linear
parallelism. If false (e.g. CUDA serialization without MPS is the real
limit), multi-process won't help and Plan C is the wrong investment.

What this measures:
  * Mode A (2 threads in 1 process, lock held):  forwards serialize on
    `_COMPILE_LOCK` exactly as production does today.
  * Mode B (2 threads in 1 process, no lock):     forwards serialize on
    the GIL alone (Layer-2-style — kept for reference even though
    reverted in production).
  * Mode C (2 separate subprocesses):             each process has its
    own GIL and CUDA context. Full parallelism, modulo GPU compute
    serialization (which we measure by extrapolation).

Workload: a "production-shaped" transformer (4 layers, 8 heads, FF=1024,
d_model=512) wrapped under `torch.compile(mode='default', dynamic=True)`.
Each "agent" runs N forwards on (B=8, T=1, E=512) input — representative
of the post-vectorization production batch shape.

If C/A >= 1.5x and C/B >= 1.2x, Plan C is validated for this hardware
and we should commit to the integration. If C/A < 1.2x, multi-process
inference doesn't deliver — the bottleneck is CUDA compute / driver
serialization and we should pivot to Plan D (distillation).

Run:
  cd /home/cayman/Repositories/EliteFurretAI
  source ../venv/bin/activate
  python -u src/elitefurretai/rl/analyze/bench_multi_process_inference.py
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as torch_mp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Model — production-shaped (4-layer transformer @ d=512, h=8, ff=1024)
# ─────────────────────────────────────────────────────────────────────


class ProductionShapeTransformer(nn.Module):
    """Strip-down of the sep_arch trunk with similar Python+CUDA proportions.

    Not a faithful clone of TransformerThreeHeadedModel — just the
    transformer trunk + a couple of linear projections to approximate
    the per-forward CPU/GPU breakdown. Tuned to match the production
    backbone enough that GIL/CUDA ratios are representative.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        ff_dim: int = 1024,
        num_layers: int = 4,
    ):
        super().__init__()
        # Input projection — represents the encoder's final ff stack
        self.input_proj = nn.Linear(d_model, d_model)
        # Trunk — matches sep_arch.yaml transformer_layers=4, heads=8, ff=1024
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output heads — represents turn_head + value_head
        self.turn_head = nn.Linear(d_model, 256)
        self.value_head = nn.Linear(d_model, 51)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d_model)
        h = self.input_proj(x)
        h = self.transformer(h)
        # Take the last position's output (matches single-turn inference)
        last = h[:, -1, :]
        return self.turn_head(last), self.value_head(last)


def build_compiled_model(device: str) -> nn.Module:
    """Build + compile a production-shaped model."""
    model = ProductionShapeTransformer().to(device).eval()
    compiled = torch.compile(model, mode="default", dynamic=True)
    return compiled


def warmup(model: nn.Module, device: str, batch_sizes: List[int]) -> None:
    """Trigger dynamo compilation across the production batch envelope."""
    with torch.no_grad():
        for B in batch_sizes:
            x = torch.zeros(B, 1, 512, device=device)
            for _ in range(2):
                _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────
# Workload — N forwards on a single agent, with realistic Python wrapper
# ─────────────────────────────────────────────────────────────────────
#
# Each forward simulates the per-request work in
# RealModelBatchHandler.__call__:
#   1. Decode N request `state` arrays into a (B, 1, E) tensor (np.stack
#      + torch.from_numpy + .to(device))
#   2. Look up prior hidden state per request (dict get)
#   3. Pad-and-concat hidden tensor + mask (matches _pad_transformer_context)
#   4. Forward through compiled model
#   5. Move logits to CPU + per-request sampling (categorical)
#   6. Build per-request response objects
#
# Without these steps, the bench measures only model compute and
# underestimates the GIL contention in production. We add them so the
# Mode-A vs Mode-B vs Mode-C comparison reflects real throughput pressure.


E = 512  # d_model — must match ProductionShapeTransformer


def run_forwards(
    model: nn.Module,
    device: str,
    n_forwards: int,
    batch_size: int,
    lock: threading.Lock | None = None,
) -> float:
    """Run n_forwards realistic per-batch ops; return total wall time.

    Each iteration does:
      - np.stack + tensor conversion (GIL-bound Python work)
      - prior-hidden lookup from a dict (GIL-bound)
      - model forward (CUDA, releases GIL)
      - logits.cpu() + multinomial sample (mix of GIL + CUDA)
      - response construction (GIL-bound)
    """
    rng = np.random.default_rng(42)
    # Pre-build a pool of "states" — simulates worker-sent inference requests.
    state_pool = [rng.standard_normal((1, E), dtype=np.float32) for _ in range(batch_size)]
    hidden_dict: dict = {}  # simulates prior_hiddens lookups
    start = time.perf_counter()
    for i in range(n_forwards):
        # 1. Decode requests
        stacked = np.stack(state_pool, axis=0)  # (B, 1, E)
        x = torch.from_numpy(stacked).to(device).unsqueeze(1).float().squeeze(2)
        # 2. Prior-hidden lookups (cold dict; ~O(B))
        _priors = [hidden_dict.get(f"k{j}_{i % 11}") for j in range(batch_size)]
        # 3. (Pad/concat hidden — omitted; not load-bearing without history)
        # 4. Forward
        if lock is not None:
            with torch.no_grad(), lock:
                turn_logits, value_logits = model(x)
        else:
            with torch.no_grad():
                turn_logits, value_logits = model(x)
        # 5. CPU transfer + per-request sampling
        turn_cpu = turn_logits.cpu()
        for j in range(batch_size):
            _ = int(torch.multinomial(torch.softmax(turn_cpu[j], dim=0), 1).item())
        # 6. Build "responses"
        _ = [{"sample": j, "value": float(value_logits[j, 0].item())} for j in range(batch_size)]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter() - start


# ─────────────────────────────────────────────────────────────────────
# Mode A & B — both threads, with vs without a shared lock
# ─────────────────────────────────────────────────────────────────────


def run_threaded(
    n_threads: int,
    n_forwards_per_thread: int,
    batch_size: int,
    device: str,
    use_lock: bool,
) -> Tuple[float, List[float]]:
    """Spawn n_threads, each with its own compiled model copy; run forwards."""
    models = [build_compiled_model(device) for _ in range(n_threads)]
    for m in models:
        warmup(m, device, [1, 4, 16, 32])

    shared_lock = threading.Lock() if use_lock else None
    per_thread_times: List[float] = [0.0] * n_threads

    def thread_worker(i: int) -> None:
        per_thread_times[i] = run_forwards(
            models[i], device, n_forwards_per_thread, batch_size, shared_lock
        )

    threads = [
        threading.Thread(target=thread_worker, args=(i,)) for i in range(n_threads)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    return wall, per_thread_times


# ─────────────────────────────────────────────────────────────────────
# Mode C — subprocesses
# ─────────────────────────────────────────────────────────────────────


def subprocess_entrypoint(
    proc_idx: int,
    n_forwards: int,
    batch_size: int,
    device: str,
    out_q: "torch_mp.Queue",
) -> None:
    """Subprocess body: build, warm, run forwards, report timing + RSS."""
    torch.set_num_threads(1)  # avoid CPU oversubscription across processes
    model = build_compiled_model(device)
    warmup(model, device, [1, 4, 16, 32])

    forward_time = run_forwards(model, device, n_forwards, batch_size, lock=None)

    # Capture RSS + GPU memory for this subprocess
    import resource

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    gpu_alloc_mb = 0.0
    if device.startswith("cuda"):
        gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    out_q.put(
        {
            "proc_idx": proc_idx,
            "forward_time_s": forward_time,
            "rss_mb": rss_kb / 1024.0,
            "gpu_alloc_mb": gpu_alloc_mb,
        }
    )


def run_multiprocess(
    n_procs: int, n_forwards_per_proc: int, batch_size: int, device: str
) -> Tuple[float, List[dict]]:
    """Spawn n_procs subprocesses, each runs n_forwards_per_proc forwards."""
    ctx = torch_mp.get_context("spawn")
    out_q: "torch_mp.Queue" = ctx.Queue()
    processes = []
    start = time.perf_counter()
    for i in range(n_procs):
        p = ctx.Process(
            target=subprocess_entrypoint,
            args=(i, n_forwards_per_proc, batch_size, device, out_q),
        )
        p.start()
        processes.append(p)
    results = []
    for _ in range(n_procs):
        results.append(out_q.get())
    for p in processes:
        p.join()
    wall = time.perf_counter() - start
    return wall, results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def _steady_state_throughput(wall_or_max_forward: float, total_forwards: int) -> float:
    """Throughput from steady-state per-unit forward time, ignoring startup."""
    if wall_or_max_forward <= 0:
        return 0.0
    return total_forwards / wall_or_max_forward


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_forwards_per_unit = 100  # per thread/process; smaller to keep total runtime bounded
    batch_size = 32  # production batch_size cap; pessimistic GIL load per forward
    scaling = [1, 2, 4]

    print(f"device={device} batch_size={batch_size} forwards_per_unit={n_forwards_per_unit}")
    print(f"scaling sweep: N ∈ {scaling}")
    print("Per-forward Python wrapper includes np.stack + dict lookups + sampling + response build")
    print("(matches RealModelBatchHandler.__call__ overhead)\n")

    rows: List[dict] = []

    for N in scaling:
        total_forwards = N * n_forwards_per_unit
        print(f"────── N = {N} ──────")

        # Mode A: N threads, shared lock
        wall_A, times_A = run_threaded(N, n_forwards_per_unit, batch_size, device, use_lock=True)
        steady_A = _steady_state_throughput(max(times_A), total_forwards)
        print(f"  A (threads + lock):   wall={wall_A:.2f}s  steady={steady_A:.1f} fwd/s  "
              f"per-unit_max={max(times_A):.2f}s")

        # Mode B: N threads, no lock
        wall_B, times_B = run_threaded(N, n_forwards_per_unit, batch_size, device, use_lock=False)
        steady_B = _steady_state_throughput(max(times_B), total_forwards)
        print(f"  B (threads, no lock): wall={wall_B:.2f}s  steady={steady_B:.1f} fwd/s  "
              f"per-unit_max={max(times_B):.2f}s")

        # Mode C: N subprocesses
        wall_C, results_C = run_multiprocess(N, n_forwards_per_unit, batch_size, device)
        forward_times_C = [r["forward_time_s"] for r in results_C]
        steady_C = _steady_state_throughput(max(forward_times_C), total_forwards)
        rss_mean = statistics.mean(r["rss_mb"] for r in results_C)
        gpu_mean = statistics.mean(r["gpu_alloc_mb"] for r in results_C)
        print(f"  C (subprocesses):     wall={wall_C:.2f}s  steady={steady_C:.1f} fwd/s  "
              f"per-unit_max={max(forward_times_C):.2f}s  "
              f"(includes ~{wall_C - max(forward_times_C):.1f}s one-time spawn/import/compile)")
        print(f"     per-subprocess RSS={rss_mean:.0f} MB  GPU_alloc={gpu_mean:.0f} MB")

        rows.append(
            {
                "N": N,
                "A": steady_A,
                "B": steady_B,
                "C": steady_C,
                "rss_mb": rss_mean,
                "gpu_mb": gpu_mean,
            }
        )

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("SCALING SUMMARY (steady-state fwd/s, ignoring subprocess startup)")
    print("═" * 80)
    print(f"{'N':>3} | {'A (lock)':>10} | {'B (no lock)':>11} | {'C (procs)':>10} | "
          f"{'B/A':>5} | {'C/A':>5} | {'C/B':>5} | {'RSS(MB)':>8}")
    print("─" * 80)
    for r in rows:
        ratio_BA = r["B"] / r["A"] if r["A"] > 0 else 0
        ratio_CA = r["C"] / r["A"] if r["A"] > 0 else 0
        ratio_CB = r["C"] / r["B"] if r["B"] > 0 else 0
        print(f"{r['N']:>3} | {r['A']:>10.1f} | {r['B']:>11.1f} | {r['C']:>10.1f} | "
              f"{ratio_BA:>5.2f} | {ratio_CA:>5.2f} | {ratio_CB:>5.2f} | {r['rss_mb']:>8.0f}")

    print("\nDecision rule from 2026-05-15-00-12 plan:")
    print("  C/A >= 1.5x AND C/B >= 1.2x at N=2  →  Plan C is validated")
    print("  C/A scales roughly linearly with N  →  Plan C scales (good)")
    print("  C/A flat or sub-linear across N     →  GPU/CUDA serialization is the limit")


if __name__ == "__main__":
    main()
