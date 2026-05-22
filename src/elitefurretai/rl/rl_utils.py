"""rl_utils.py — Shared utility helpers for the RL package.

This module collects small, broadly-reusable helpers that were previously
duplicated or off-purpose for their host file. Pulled out so train.py and
worker.py can stay focused on their stated purpose (orchestration / actor
loop) rather than carrying Linux memory accounting and logging boilerplate.

Contents
--------
setup_logging
    The standard root-WARNING + elitefurretai/__main__-INFO basicConfig
    used by both the trainer and worker entrypoints. Was duplicated
    between train.py and worker.py (the worker variant passes
    ``force=True`` since third-party imports in a subprocess may have
    already attached a root handler).

list_pt_files
    Directory listing filtered to ``.pt`` files. Was duplicated inside
    opponents.py (twice in the same file).

normalize_curriculum
    Rescale a curriculum dict to sum to 1.0, with a self-play fallback
    when input sums to 0. Used by both ``OpponentPool`` and
    ``_ShowdownBackend`` (vgcbench-zeroing renormalization).

read_pss_bytes / format_memory_bytes / sum_process_tree_pss_bytes
    Linux memory-accounting helpers built on ``/proc/<pid>/smaps_rollup``.
    Lived in train.py and worker.py purely as system instrumentation;
    they have nothing to do with training logic or the actor loop.

start_memory_watchdog
    Background thread that watches combined Pss across the training
    process tree and requests graceful shutdown when a threshold is
    exceeded. Was in train.py but is pure infrastructure.

is_cuda_device
    Predicate over a torch device string. Replaces three inline
    ``startswith("cuda")`` checks across learners.py and
    inference_subprocess.py.

timestamp_iso
    ISO-format wall-clock timestamp for checkpoint metadata. Replaces
    three inline ``datetime.now().isoformat()`` calls.

collate_trajectories
    Pad/batch variable-length worker trajectories into fixed-shape
    tensors and compute GAE advantages/returns. Lives here so both
    train.py (main learner) and exploiters.py (exploiter learner) can
    use it without a circular import between them.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

from elitefurretai.etl.encoder import MDBO

logger = logging.getLogger(__name__)


def setup_logging(force: bool = False) -> None:
    """Configure the root logger and raise elitefurretai/__main__ to INFO.

    Root stays at WARNING so poke-env's per-player loggers (named by the
    randomly-generated Showdown username, e.g. ``"M00OP00E08ED700"``) don't
    echo every websocket request/response at INFO — those add up to
    ~1.5 GB/hr of log spam and can OOM/disk-fill WSL2 on a long run.

    Our own modules sit under ``elitefurretai`` and ``__main__`` and stay
    at INFO so training progress, checkpoint events, and watchdog
    notifications are preserved.

    ``force=True`` is needed in worker subprocesses because third-party
    imports may have already attached a root handler before this runs;
    ``basicConfig`` is a no-op in that case unless force is set.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=force,
    )
    logging.getLogger("elitefurretai").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)


def list_pt_files(directory: Optional[str]) -> list[str]:
    """Return absolute paths of every ``.pt`` file directly under ``directory``.

    Returns ``[]`` for a ``None`` or non-existent directory so callers can
    treat "no checkpoint dir yet" and "empty dir" identically.
    """
    if not directory or not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".pt")
    ]


def normalize_curriculum(curriculum: Dict[str, float]) -> Dict[str, float]:
    """Return ``curriculum`` rescaled to sum to 1.0.

    Falls back to ``{"self_play": 1.0}`` when input sums to 0 (no opponents
    configured). Preserves 0-valued keys — callers that need to drop them
    should do so explicitly. Used both by ``OpponentPool`` (after PFSP
    rebalance and Showdown-side vgcbench-zeroing) and by
    ``WorkerOpponentFactory`` (each control-broadcast tick).
    """
    total = float(sum(curriculum.values()))
    if total <= 0:
        return {"self_play": 1.0}
    return {key: value / total for key, value in curriculum.items()}


def read_pss_bytes(pid: int) -> Optional[int]:
    """Read Pss (proportional set size) from ``/proc/<pid>/smaps_rollup``.

    Pss splits each shared page across the processes sharing it, so
    summing Pss across the process tree equals the *physical* RAM the
    tree is holding (no double-counting). Returns ``None`` if
    smaps_rollup is unreadable (process exited / permission denied / not
    Linux).
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    return None


def format_memory_bytes(mem_bytes: int) -> str:
    """Format a byte count as a short human-readable string (KB/MB/GB)."""
    if mem_bytes < 1024 * 1024:
        return f"{mem_bytes / 1024:.0f}KB"
    if mem_bytes < 1024 * 1024 * 1024:
        return f"{mem_bytes / (1024 * 1024):.0f}MB"
    return f"{mem_bytes / (1024 * 1024 * 1024):.0f}GB"


def sum_process_tree_pss_bytes() -> Tuple[int, Dict[str, int]]:
    """Sum Pss across this process and all recursive children.

    Returns ``(total_bytes, breakdown_by_role_bytes)`` where the breakdown
    keys are ``"trainer"``, ``"showdown"``, ``"vgcbench"``, ``"workers"``,
    and ``"other"`` — classified by cmdline so we can see at a glance
    which subprocess family is responsible for memory pressure.

    Why Pss, not Rss
    ----------------
    Linux RSS counts each shared page in full for every process mapping
    it, so summing child RSS double-counts shared libraries and shared
    mmaps. On a typical RL training tree (1 trainer + 4 workers + frozen
    subprocess + ~20 inductor compile workers + 4 vgcbench runners + 4
    showdown servers with ~7 helper procs each), sum(RSS) overstates
    physical RAM by ~8-10 GB because every Python interpreter shares the
    same libpython, libcuda, libstdc++, etc.

    Pss (proportional set size) from ``/proc/<pid>/smaps_rollup`` splits
    each shared page fairly across its sharers. sum(Pss) across a
    process tree equals the physical RAM the tree actually occupies.

    Empirical comparison from a may15-profile snapshot (4 showdown, 4
    vgcbench, 4 workers, frozen subprocess, 20+ inductor workers):
        sum(RSS) = 24.2 GB   sum(Pss) = 14.2 GB
    sum(RSS) had been tripping the 22-24 GB watchdog while real WSL2
    RAM usage stayed comfortably below the 23 GB physical ceiling.

    Falls back to RSS on a per-process basis only when smaps_rollup is
    unreadable (rare on Linux; possible if /proc isn't mounted or the
    process exited mid-read).
    """
    me = psutil.Process(os.getpid())
    pss = read_pss_bytes(me.pid)
    total = pss if pss is not None else me.memory_info().rss
    breakdown: Dict[str, int] = {"trainer": total}
    for child in me.children(recursive=True):
        try:
            cmdline = " ".join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        child_pss = read_pss_bytes(child.pid)
        if child_pss is None:
            try:
                child_pss = child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if "node" in cmdline or "pokemon-showdown" in cmdline:
            role = "showdown"
        elif "vgc" in cmdline.lower() or "vgcbench" in cmdline.lower():
            role = "vgcbench"
        elif "mp_worker_process" in cmdline or "multiprocessing" in cmdline:
            role = "workers"
        else:
            role = "other"
        breakdown[role] = breakdown.get(role, 0) + child_pss
        total += child_pss
    return total, breakdown


def start_memory_watchdog(
    shutdown_requested: threading.Event,
    threshold_gb: Optional[float],
    poll_interval_s: float = 30.0,
) -> Optional[threading.Thread]:
    """Watch combined Pss and request shutdown if it exceeds the threshold.

    Returns the daemon thread (or ``None`` if disabled) so callers can
    join in tests. The thread exits on its own once ``shutdown_requested``
    fires — the main loop's existing ``finally`` block handles
    checkpoint + cleanup.
    """
    if threshold_gb is None or threshold_gb <= 0:
        logger.info("Memory watchdog disabled (threshold_gb=%r)", threshold_gb)
        return None

    threshold_bytes = int(threshold_gb * 1024**3)
    logger.info(
        "Memory watchdog armed at %.1f GB combined Pss (poll every %.0fs)",
        threshold_gb,
        poll_interval_s,
    )

    def _loop() -> None:
        while not shutdown_requested.is_set():
            try:
                total, breakdown = sum_process_tree_pss_bytes()
            except psutil.NoSuchProcess:
                return
            if total >= threshold_bytes:
                parts = ", ".join(
                    f"{role}={rss / 1024**3:.2f}GB" for role, rss in breakdown.items()
                )
                logger.critical(
                    "Memory watchdog: combined Pss %.2f GB >= %.1f GB threshold "
                    "(%s). Requesting graceful shutdown.",
                    total / 1024**3,
                    threshold_gb,
                    parts,
                )
                shutdown_requested.set()
                return
            # Use Event.wait so a shutdown from another path wakes us
            # immediately and the thread exits without a stale 30s delay.
            if shutdown_requested.wait(timeout=poll_interval_s):
                return

    thread = threading.Thread(target=_loop, name="memory-watchdog", daemon=True)
    thread.start()
    return thread


def is_cuda_device(device: Any) -> bool:
    """True if ``device`` (str or torch.device) names a CUDA device."""
    return str(device).startswith("cuda")


def timestamp_iso() -> str:
    """Wall-clock timestamp in ISO 8601 form for checkpoint metadata."""
    return datetime.now().isoformat()


def collate_trajectories(trajectories, device, gamma, gae_lambda, max_seq_len=40):
    """Collate list of trajectories into batched tensors with padding.

    What this is for
    ----------------
    Workers ship variable-length trajectories — battle 1 might be 12 turns,
    battle 2 might be 38 turns. The learner expects fixed-shape padded
    tensors. This function does that conversion AND computes the GAE
    advantages and returns that PPO needs.

    Steps performed:
      1. Truncate trajectories longer than max_seq_len. We keep the LAST N
         steps because late-game decisions tend to matter more for outcomes.
      2. Pre-allocate (B, T, ...) tensors and copy fields in.
      3. Compute GAE advantages backwards through each trajectory:
              δ_t   = r_t + γ V(s_{t+1}) - V(s_t)
              A_t   = δ_t + γ λ A_{t+1}
              R_t   = V(s_t) + A_t
         where δ is the TD error, A is the advantage, R is the return target.
      4. Move everything to the learner device.

    Args:
        trajectories: List of trajectory dicts ({"steps": [...], ...}).
        device: torch device for the final batch.
        gamma: Discount factor γ.
        gae_lambda: GAE λ.
        max_seq_len: Truncate trajectories longer than this (keeps the tail).

    Returns:
        Dict of padded tensors keyed by name (states, actions, …).
    """
    # Keep the TAIL (not the head) so the right edge of every trajectory is the
    # real end of the battle. The GAE bootstrap below treats the position past
    # the last real step as terminal (next_val=0); that's only valid because
    # this slice preserves the natural terminal at index seq_len-1.
    trajectories = [
        traj["steps"][-max_seq_len:] if len(traj["steps"]) > max_seq_len else traj["steps"]
        for traj in trajectories
    ]

    batch_size = len(trajectories)
    max_len = max(len(t) for t in trajectories)
    dim = len(trajectories[0][0]["state"])
    action_space = MDBO.action_space()

    # ── Pre-allocate numpy buffers and fill them per-trajectory ──────────────
    # Avoids the per-step `torch.tensor(...)` calls in the inner loop, each of
    # which is a separate allocation + dtype check. Numpy slice-assign is
    # ~10× cheaper for the dimensions we care about. Final torch.from_numpy
    # is zero-copy.
    np_states = np.zeros((batch_size, max_len, dim), dtype=np.float32)
    np_actions = np.zeros((batch_size, max_len), dtype=np.int64)
    np_rewards = np.zeros((batch_size, max_len), dtype=np.float32)
    np_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
    np_values = np.zeros((batch_size, max_len), dtype=np.float32)
    np_is_tp = np.zeros((batch_size, max_len), dtype=bool)
    np_padding = np.zeros((batch_size, max_len), dtype=bool)
    # Default to all-1s; turn steps overwrite with the real mask. Teampreview
    # and padded positions stay all-1s (the learner ignores them via flat_is_tp
    # and padding_mask anyway).
    np_masks = np.ones((batch_size, max_len, action_space), dtype=np.float32)

    # Hoisted: teampreview steps have mask=None, so substitute this all-1s
    # placeholder. np.stack copies the row, so aliasing across steps is safe.
    # The learner ignores teampreview mask positions via flat_is_tp anyway.
    tp_mask_placeholder = np.ones(action_space, dtype=np.float32)

    for i, traj in enumerate(trajectories):
        seq_len = len(traj)

        # Single pass over `traj` to extract every field. Cuts dict-lookup and
        # iteration overhead by ~7× vs one comprehension per field.
        states_l: List[Any] = []
        actions_l: List[int] = []
        rewards_l: List[float] = []
        log_probs_l: List[float] = []
        values_l: List[float] = []
        is_tp_l: List[bool] = []
        masks_l: List[Any] = []
        for step in traj:
            states_l.append(step["state"])
            actions_l.append(step["action"])
            rewards_l.append(step["reward"])
            log_probs_l.append(step["log_prob"])
            values_l.append(step["value"])
            is_tp_l.append(step["is_teampreview"])
            masks_l.append(tp_mask_placeholder if step["is_teampreview"] else step["mask"])

        np_states[i, :seq_len] = np.stack(states_l)
        np_actions[i, :seq_len] = actions_l
        np_rewards[i, :seq_len] = rewards_l
        np_log_probs[i, :seq_len] = log_probs_l
        np_values[i, :seq_len] = values_l
        np_is_tp[i, :seq_len] = is_tp_l
        np_padding[i, :seq_len] = True
        np_masks[i, :seq_len] = np.stack(masks_l)

    # Convert to torch (zero-copy on CPU). Move to device once at the end.
    states = torch.from_numpy(np_states)
    actions = torch.from_numpy(np_actions)
    rewards = torch.from_numpy(np_rewards)
    log_probs = torch.from_numpy(np_log_probs)
    values = torch.from_numpy(np_values)
    is_tp = torch.from_numpy(np_is_tp)
    padding_mask = torch.from_numpy(np_padding)
    masks = torch.from_numpy(np_masks)

    # ── Vectorized GAE ───────────────────────────────────────────────────────
    # Single reverse-T loop with (B,) vector ops instead of B*T Python
    # iterations. Padding handled by masking the gae update (gae stays at 0
    # for batches whose t is past their seq_len, since the initial value is 0
    # and we never updated it through the all-padded suffix).
    advantages = torch.zeros(batch_size, max_len, dtype=torch.float32)
    gae = torch.zeros(batch_size, dtype=torch.float32)
    pad_float = padding_mask.float()
    for t in reversed(range(max_len)):
        if t + 1 < max_len:
            # When t+1 is padded, treat next_val as 0 (terminal at end-of-traj).
            # NOTE: this conflates "padded position" with "terminal state" — safe
            # only because trajectories above are truncated to keep the tail, so
            # the slot past any real step is always the natural terminal.
            next_val = values[:, t + 1] * pad_float[:, t + 1]
        else:
            next_val = torch.zeros(batch_size, dtype=torch.float32)
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        gae_new = delta + gamma * gae_lambda * gae
        gae = torch.where(padding_mask[:, t], gae_new, gae)
        advantages[:, t] = gae
    returns = (advantages + values) * pad_float

    return {
        "states": states.to(device, non_blocking=True),
        "actions": actions.to(device, non_blocking=True),
        "rewards": rewards.to(device, non_blocking=True),
        "log_probs": log_probs.to(device, non_blocking=True),
        "values": values.to(device, non_blocking=True),
        "is_teampreview": is_tp.to(device, non_blocking=True),
        "advantages": advantages.to(device, non_blocking=True),
        "returns": returns.to(device, non_blocking=True),
        "padding_mask": padding_mask.to(device, non_blocking=True),
        "masks": masks.to(device, non_blocking=True),
    }
