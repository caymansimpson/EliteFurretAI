"""train.py — Top-level RNaD training entrypoint.

What this file does
-------------------
This is the orchestrator. When you run
    python -m elitefurretai.rl.train --config <path.yaml>
the function below is what executes. It owns:

  1. Config loading + Showdown server launch + (optional) external runners.
  2. Model + learner construction (or resume from a checkpoint).
  3. Spawning worker subprocesses (worker.py) via mp.Process.
  4. The main training loop that pulls trajectories from workers, batches
     them, calls learner.update(), and broadcasts new weights.
  5. Periodic checkpointing, exploiter training subprocess launching, and
     opponent curriculum updates.
  6. Graceful shutdown (SIGTERM/SIGINT) so checkpoints are saved on Ctrl-C.

Where this file fits
--------------------
            ┌────────────── train.py (this file) ──────────────┐
            │   Loop:                                          │
            │     traj = mp_traj_queue.get()                   │
            │     if buffer >= train_batch_size:               │
            │         batch  = collate_trajectories(buffer)    │
            │         metrics = learner.update(batch)          │
            │         if checkpoint_interval: broadcast_weights│
            │                                                  │
            └────┬──────────────────────────────────────┬──────┘
                 │ trajectories                  weights│
                 │ (mp_traj_queue)       (weight_queues)│
        ┌────────┴────────┐                ┌────────────┴───────┐
        │   worker.py     │                │   worker.py        │
        │   (xN workers)  │  ←──────────── │   (back-channel)   │
        └────────┬────────┘                └────────────────────┘
                 │ websockets
        ┌────────┴────────┐
        │ Showdown servers│
        │  (xM servers)   │
        └─────────────────┘

Key design notes for new readers
--------------------------------
- Workers are SEPARATE PROCESSES. The trainer can't share Python objects with
  them; everything goes through mp.Queue (pickled tensors, dicts, etc.).
- Trajectories live in a list buffer here until `train_batch_size` of them
  are collected, at which point they're collated into one big padded batch.
- Weight broadcasts happen at `checkpoint_interval` cadence — not per update.
  The trade-off: more frequent broadcasts = workers play more on-policy data
  (better PPO ratios) but more wasted bandwidth. Less frequent = the opposite.
- We use mp.set_start_method("spawn") because CUDA + fork is unsafe.
"""

import argparse
import copy
import gc
import logging
import multiprocessing as mp
import os
import queue
import random
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

import wandb
from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.system_utils import (
    configure_torch_multiprocessing,
    suppress_third_party_warnings,
)
from elitefurretai.rl.config import RUST_ENGINE_BACKEND, RNaDConfig
from elitefurretai.rl.learners import (
    PortfolioRNaDLearner,
    build_model_from_config,
    load_checkpoint,
    save_checkpoint,
)
from elitefurretai.rl.model_registry import ModelRegistry
from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.players import RNaDAgent, cleanup_worker_executors
from elitefurretai.rl.worker import mp_worker_process
from elitefurretai.supervised import format_time

logger = logging.getLogger(__name__)


def generate_shutdown_signal():
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info("%s received. Initiating graceful shutdown...", sig_name)
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    return shutdown_requested


def _read_pss_bytes(pid: int) -> Optional[int]:
    """Read Pss (proportional set size) from /proc/<pid>/smaps_rollup.

    Pss splits each shared page across the processes sharing it, so summing
    Pss across the process tree equals the *physical* RAM the tree is
    holding (no double-counting). Returns None if smaps_rollup is
    unreadable (process exited / permission denied / not Linux).
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("Pss:"):
                    # "Pss:           12345 kB\n"
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    return None


def _sum_process_tree_rss_bytes() -> Tuple[int, Dict[str, int]]:
    """Sum proportional RSS (Pss) of the current process and all recursive
    children. Returns (total_bytes, breakdown_by_role_bytes).

    Why Pss, not Rss
    ----------------
    Linux RSS counts each shared page in full for every process mapping it,
    so summing child RSS double-counts shared libraries and shared mmaps.
    On a typical RL training tree (1 trainer + 4 workers + frozen
    subprocess + ~20 inductor compile workers + 4 vgcbench runners + 4
    showdown servers with ~7 helper procs each), sum(RSS) overstates
    physical RAM by ~8-10 GB because every Python interpreter shares the
    same libpython, libcuda, libstdc++, etc.

    Pss (proportional set size) from /proc/<pid>/smaps_rollup splits each
    shared page fairly across its sharers. sum(Pss) across a process tree
    equals the physical RAM the tree actually occupies.

    Empirical comparison from a may15-profile snapshot (4 showdown, 4
    vgcbench, 4 workers, frozen subprocess, 20+ inductor workers):
        sum(RSS) = 24.2 GB   sum(Pss) = 14.2 GB
    sum(RSS) had been tripping the 22-24 GB watchdog while real WSL2 RAM
    usage stayed comfortably below the 23 GB physical ceiling.

    Fallback to RSS only happens on a per-process basis if smaps_rollup
    is unreadable (rare on Linux; possible if /proc isn't mounted or if
    the process exited mid-read).
    """
    me = psutil.Process(os.getpid())
    pss = _read_pss_bytes(me.pid)
    total = pss if pss is not None else me.memory_info().rss
    breakdown: Dict[str, int] = {"trainer": total}
    for child in me.children(recursive=True):
        try:
            cmdline = " ".join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        child_pss = _read_pss_bytes(child.pid)
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
    """Watch combined RSS and request shutdown if it exceeds the threshold.

    Returns the daemon thread (or None if disabled) so callers can join in
    tests. The thread exits on its own once `shutdown_requested` fires —
    the main loop's existing `finally` block handles checkpoint + cleanup.
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
                total, breakdown = _sum_process_tree_rss_bytes()
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


def initialize_learner(
    config: RNaDConfig,
    agent: RNaDAgent,
    base_model: Any,
) -> PortfolioRNaDLearner:
    """Create the portfolio learner with one initial reference snapshot."""
    ref_model = copy.deepcopy(base_model)
    ref_agent = RNaDAgent(ref_model)
    return PortfolioRNaDLearner(
        agent,
        [ref_agent],
        config=config,
    )


def resolve_worker_model_source(config: RNaDConfig, agent: RNaDAgent) -> str:
    """Persist a bootstrap checkpoint for worker startup and return its path."""
    assert config.training.run_dir, "run_dir must be set before bootstrap"
    bootstrap_path = os.path.join(config.training.run_dir, "worker_bootstrap_initial.pt")
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "config": config.to_dict(),
            "step": 0,
            "timestamp": datetime.now().isoformat(),
        },
        bootstrap_path,
    )
    logger.info("Saved worker bootstrap checkpoint to %s", bootstrap_path)
    return bootstrap_path


def initialize_training_state(
    config: RNaDConfig,
) -> Tuple[
    RNaDAgent,
    PortfolioRNaDLearner,
    str,
    Dict[str, Any],
    int,
    Optional[Dict[str, float]],
]:
    """Initialize model/learner and worker bootstrap state in one place.

    Returns:
        agent: Main training agent
        learner: Initialized learner
        worker_model_path: Checkpoint path workers should bootstrap from
        worker_model_config: Model config dict for worker model construction
        start_step: Starting update step (restored for resume)
        resume_curriculum: Curriculum restored from checkpoint (if any)
    """
    cfg = config.to_dict()
    resume_curriculum: Optional[Dict[str, float]] = None
    start_step = 0

    if config.training.resume_from:
        logger.info("Resuming model+optimizer from %s...", config.training.resume_from)
        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=config.training.embedder_feature_set,
            omniscient=False,
        )
        # Load checkpoint contents BEFORE constructing the learner. Order
        # matters: `initialize_learner` deepcopies `base_model` as the
        # initial RNaD reference, so the model weights must already be the
        # trained ones at that point. If we built the learner first and
        # loaded weights after, the ref would freeze at random init and
        # rnad_alpha * KL(curr || ref) would pull the policy toward a
        # random anchor on every resume (see
        # planning/stage2/2026-05-14-17-00-resume-state-bugs.md).
        checkpoint = load_checkpoint(config.training.resume_from, config.hardware.device)
        base_model = build_model_from_config(cfg, embedder, config.hardware.device, None)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)
        learner.load_resume_state(checkpoint)
        start_step = int(checkpoint["step"])
        old_config = RNaDConfig.from_dict(checkpoint["config"])
        if old_config.curriculum.curriculum_weights:
            resume_curriculum = old_config.curriculum.curriculum_weights

        worker_model_path = config.training.resume_from
        worker_model_config = old_config.to_dict()
    elif config.training.initialize_path:
        logger.info(
            "Initializing model weights from %s...", config.training.initialize_path
        )
        init_checkpoint = torch.load(
            config.training.initialize_path,
            map_location=config.hardware.device,
            weights_only=False,
        )
        checkpoint_cfg = init_checkpoint.get("config", cfg)
        # BC checkpoints store config in flat form (e.g. optimizer is the
        # string "adamw"); RL checkpoints use the nested RNaDConfig shape.
        # We only need embedder_feature_set here — read it from either layout
        # and fall back to the runtime config if absent.
        if (
            isinstance(checkpoint_cfg.get("training"), dict)
            and "embedder_feature_set" in checkpoint_cfg["training"]
        ):
            ckpt_feature_set = checkpoint_cfg["training"]["embedder_feature_set"]
        else:
            ckpt_feature_set = checkpoint_cfg.get(
                "embedder_feature_set", config.training.embedder_feature_set
            )
        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=ckpt_feature_set,
            omniscient=False,
        )
        # Build the model from the *runtime* config (so architecture knobs
        # like value_head_layers take effect) and partial-load the checkpoint
        # weights via strict=False. Mismatched keys (e.g. an old 2-layer
        # win_head when the runtime config asks for a deep value head) are
        # skipped on both sides and logged. When the runtime architecture
        # matches the checkpoint's, strict=False is a no-op behaviorally.
        base_model = build_model_from_config(
            cfg,
            embedder,
            config.hardware.device,
            init_checkpoint["model_state_dict"],
            strict=False,
        )
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)

        worker_model_path = config.training.initialize_path
        worker_model_config = cfg
    else:
        logger.info("Initializing fresh model from config...")
        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=config.training.embedder_feature_set,
            omniscient=False,
        )
        base_model = build_model_from_config(cfg, embedder, config.hardware.device, None)
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)

        worker_model_path = resolve_worker_model_source(config, agent)
        worker_model_config = cfg

    return (
        agent,
        learner,
        worker_model_path,
        worker_model_config,
        start_step,
        resume_curriculum,
    )


def get_dead_workers(
    processes: List[mp.Process], error_queue: MPQueue, verbose: bool = True
) -> List[mp.Process]:
    dead_procs = [p for p in processes if not p.is_alive()]

    if verbose and len(dead_procs) > 0:
        logger.error("=" * 60)
        logger.error("%d worker process(es) died:", len(dead_procs))
        for p in dead_procs:
            exit_code = p.exitcode
            exit_reason = {
                None: "still running (race condition?)",
                0: "normal exit",
                1: "general error",
                -9: "SIGKILL (out of memory?)",
                -11: "SIGSEGV (segmentation fault)",
                -15: "SIGTERM (terminated)",
            }.get(exit_code, f"exit code {exit_code}")
            logger.error("  - %s (PID: %s): %s", p.name, p.pid, exit_reason)

        # Check error queue for detailed error messages from workers
        logger.error("Checking for error reports from workers...")
        error_found = False
        while True:
            try:
                error_info = error_queue.get_nowait()
                error_found = True
                logger.error(
                    "Error from Worker %s: %s",
                    error_info["worker_id"],
                    error_info["error"],
                )
                logger.error("Traceback:\n%s", error_info["traceback"])
            except Exception:
                break
        if not error_found:
            logger.error(
                "No error reports in queue (worker may have crashed before reporting)"
            )

        logger.error("Training cannot continue. Saving checkpoint and exiting...")
        logger.error("=" * 60)

    return dead_procs


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

    for i, traj in enumerate(trajectories):
        seq_len = len(traj)

        np_states[i, :seq_len] = np.stack([step["state"] for step in traj])
        np_actions[i, :seq_len] = [step["action"] for step in traj]
        np_rewards[i, :seq_len] = [step["reward"] for step in traj]
        np_log_probs[i, :seq_len] = [step["log_prob"] for step in traj]
        np_values[i, :seq_len] = [step["value"] for step in traj]
        np_is_tp[i, :seq_len] = [step["is_teampreview"] for step in traj]
        np_padding[i, :seq_len] = True

        for t, step in enumerate(traj):
            if not step["is_teampreview"]:
                np_masks[i, t] = step["mask"]

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


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ In-process exploiter co-training helpers                                    ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║ The exploiter pipeline is a SECOND learner that lives in this same process. ║
# ║ It optimizes a pure exploitation objective (no RNaD regularization) against ║
# ║ a frozen copy of the main agent ("the victim"). When it sustains a win-rate ║
# ║ ≥ threshold over the rolling window, it "graduates" — gets snapshotted to   ║
# ║ disk (entering the EXPLOITERS curriculum slot for main to defend against)   ║
# ║ and a fresh generation is initialized from the BC checkpoint.               ║
# ║                                                                             ║
# ║ Why in-process and not a separate script (the OLD subprocess pipeline       ║
# ║ removed in this change):                                                    ║
# ║   - Reuses the existing worker pool — no extra CPU pressure.                ║
# ║   - Uses spare GPU capacity for the second learner's gradients.             ║
# ║   - One run, one config, one wandb stream — no subprocess orchestration.    ║
# ║                                                                             ║
# ║ Why graduation instead of fixed-interval snapshots:                         ║
# ║   - Quality control: every entry in the curriculum is a validated weakness. ║
# ║   - Diversity: fresh-init each generation finds DIFFERENT exploits (each    ║
# ║     graduation pushes main to defend against the previous exploit, so the   ║
# ║     next generation can't rely on the same trick).                          ║
# ║   - Implicit difficulty curriculum: when main is robust, no exploit can     ║
# ║     graduate; the pipeline naturally winds down rather than churning the    ║
# ║     curriculum with weak checkpoints.                                       ║
# ║                                                                             ║
# ║ See planning/stage2/2026-05-07-07-58-exploiter-data-plane.md and            ║
# ║ src/elitefurretai/rl/exploiter_implementation_plan.md for full design.      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝


def _train_exploiter_weight(config: RNaDConfig) -> float:
    """Read the curriculum weight that gates the entire pipeline.

    `train_exploiter > 0` is the SINGLE source of truth — there is no
    separate `exploiter_enabled` flag. A weight-only gate prevents the
    inconsistent-state bug where a flag and a weight could disagree
    (e.g., flag off but weight 0.20 → workers sample exploiter battles
    that get silently dropped because no learner exists to consume them).
    """
    return float(config.curriculum.curriculum_weights.get("train_exploiter", 0.0))


def _build_exploiter_config(main_config: RNaDConfig) -> RNaDConfig:
    """Construct a config for the exploiter learner from the main config.

    The exploiter shares almost everything with main (architecture, value
    head, hardware) but needs algorithmic overrides:
      - rnad_alpha = 0: the exploiter has no Nash regularizer. Its goal
        is pure exploitation, not equilibrium. (PortfolioRNaDLearner
        reduces to plain PPO when rnad_alpha=0 and max_portfolio_size=1.)
      - ent_coef = ent_coef_end = exploiter_ent_coef: no annealing —
        exploiter trains for relatively short generations and we want a
        constant entropy bonus to prevent premature mode collapse onto
        a single exploit. (Main anneals from ent_coef → ent_coef_end
        because main trains for 100K+ updates; the exploiter's 5K-cap
        per generation doesn't justify a schedule.)
      - backbone_lr = heads_lr = exploiter_lr: a single LR override
        applied to both parameter groups. The plan calls for one number
        (`exploiter_lr=1e-4`); the cleanest interpretation is to use it
        for both groups. Could be split if we later observe one group
        needs different treatment.
      - max_portfolio_size = 1: no rolling reference portfolio for the
        exploiter. With rnad_alpha=0 the references are unused anyway,
        but reducing portfolio size avoids needless deepcopies.

    Everything else (architecture, value head, hardware) is inherited
    from main so the exploiter's model has the same shape and the same
    inference path on workers.
    """
    cfg = copy.deepcopy(main_config)
    cfg.algorithm.rnad_alpha = 0.0
    cfg.algorithm.ent_coef = main_config.exploiter.ent_coef
    cfg.algorithm.ent_coef_end = main_config.exploiter.ent_coef
    cfg.optimizer.backbone_lr = main_config.exploiter.lr
    cfg.optimizer.heads_lr = main_config.exploiter.lr
    cfg.optimizer.lr = main_config.exploiter.lr
    cfg.portfolio.max_portfolio_size = 1
    return cfg


def _load_bc_state_dict(
    bc_model_path: Optional[str], device: str
) -> Optional[Dict[str, Any]]:
    """Load and cache the BC checkpoint's state_dict for fresh-init each generation.

    Cached so we don't re-read from disk every graduation event (which
    happens up to once per ~15 minutes of wall-clock; not a bottleneck
    but unnecessary I/O if the file is huge). Returns None when no BC
    path is configured — caller should validate before reaching here.
    """
    if not bc_model_path:
        return None
    bc_checkpoint = torch.load(bc_model_path, map_location=device, weights_only=False)
    return bc_checkpoint["model_state_dict"]


def _initialize_exploiter_pipeline(
    config: RNaDConfig,
    agent: RNaDAgent,
    worker_model_config: Dict[str, Any],
) -> Tuple[
    Optional[RNaDAgent],
    Optional[RNaDAgent],
    Optional[PortfolioRNaDLearner],
    Optional[Dict[str, Any]],
]:
    """Build the exploiter learner stack in the main process.

    Returns (exploiter_agent, victim_agent, exploiter_learner, bc_state_dict).
    All four are None when `train_exploiter == 0` (the pipeline is dormant
    and the four state slots aren't allocated, saving GPU memory).

    Generation 0 init source = deepcopy of main_agent (per design decisions
    confirmed 2026-05-07). The first generation gets a competent starting
    point that's already trained on the same task. Subsequent generations
    re-initialize from BC (`bc_state_dict`) to force a *different* basin —
    if we re-deepcopied main, the exploiter would just chase main's current
    policy and find the same exploit repeatedly.

    The victim_agent is constructed from BC initially; it'll be overwritten
    by the first victim_refresh broadcast (which copies main's current
    weights). Doing it this way keeps the victim_agent shape-compatible
    with the model architecture even if the warmup gate fires before the
    first refresh.

    The exploiter_ref_agent is a frozen deepcopy of the initial exploiter.
    With rnad_alpha=0 it's mathematically unused (the KL-to-ref term zeroes
    out), but PortfolioRNaDLearner's __init__ requires a non-empty ref list
    and tracks per-ref selection counts — easier to satisfy that interface
    than to special-case it.
    """
    if _train_exploiter_weight(config) <= 0:
        return None, None, None, None

    if not config.curriculum.bc_model_path:
        raise ValueError(
            "exploiter co-training requires `curriculum.bc_model_path` to be "
            "set — each new generation re-initializes from BC to find a "
            "different exploit basin. Set bc_model_path or set "
            "train_exploiter to 0 in curriculum_weights."
        )

    device = config.hardware.device

    # ── Generation 0: deepcopy of main ──────────────────────────────────
    # The first generation inherits main's weights. Main has been trained
    # via supervised learning + RNaD, so it's a strong starting point.
    # Using a fresh BC init for gen 0 would waste 200 warmup updates
    # learning basics the exploiter already knows from main.
    exploiter_model = copy.deepcopy(agent.model)
    exploiter_agent = RNaDAgent(exploiter_model)

    # ── Victim: BC-init, will be overwritten by first refresh ───────────
    # The victim is the frozen target. We init from BC so it has a real
    # policy (not random) in case warmup ends and a train_exploiter battle
    # samples before the first victim_refresh fires. Once main process
    # broadcasts victim_weights, this gets replaced with current main.
    bc_state_dict = _load_bc_state_dict(config.curriculum.bc_model_path, device)
    embedder = Embedder(
        format=config.curriculum.battle_format,
        feature_set=config.training.embedder_feature_set,
        omniscient=False,
    )
    # strict=False mirrors the main-agent BC load (train.py:303-312): the BC
    # checkpoint may have a different value/win head shape than the runtime
    # architecture (e.g. sep_arch's deep value head). Trunk + policy load fine;
    # mismatched heads get fresh-initialized — they'll be overwritten on the
    # first victim_refresh anyway.
    victim_model = build_model_from_config(
        worker_model_config, embedder, device, bc_state_dict, strict=False
    )
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False
    victim_agent = RNaDAgent(victim_model)

    # ── Reference for the exploiter learner ──────────────────────────────
    # Frozen copy of the initial exploiter. Mathematically unused under
    # rnad_alpha=0 but PortfolioRNaDLearner's interface requires it.
    exploiter_ref_agent = RNaDAgent(copy.deepcopy(exploiter_model))

    exploiter_config = _build_exploiter_config(config)
    exploiter_learner = PortfolioRNaDLearner(
        exploiter_agent,
        [exploiter_ref_agent],
        config=exploiter_config,
    )

    logger.info(
        "Exploiter pipeline initialized | gen 0 from main | victim from BC | "
        "lr=%g, ent_coef=%g, batch=%d, threshold=%.2f over %d battles, "
        "max_updates_per_gen=%d, victim_refresh=%d",
        config.exploiter.lr,
        config.exploiter.ent_coef,
        config.exploiter.batch_size,
        config.exploiter.graduation_threshold,
        config.exploiter.graduation_window,
        config.exploiter.max_updates_per_generation,
        config.exploiter.victim_refresh_interval,
    )
    return exploiter_agent, victim_agent, exploiter_learner, bc_state_dict


def _mask_curriculum_during_warmup(
    curriculum: Dict[str, float], in_warmup: bool
) -> Dict[str, float]:
    """Zero `train_exploiter` and redistribute its weight proportionally to
    other slots while in warmup.

    Why warmup masks the curriculum (and doesn't just drop trajectories
    server-side): if workers sample train_exploiter battles during warmup,
    we'd burn 20% of CPU on battles whose trajectories we'd then discard.
    Better to redistribute so all workers spend warmup time generating
    useful main-learner data.

    Why proportional redistribution rather than dumping into self_play:
    keeps the BC/ghost/baseline mix during warmup the same as steady
    state. If the user configured 0.4 self_play / 0.4 train_exploiter
    / 0.2 ghosts, masking-then-self-play would yield 0.8/0.0/0.2 during
    warmup (over-weighting self-play). Proportional gives 0.667/0.0/0.333,
    preserving the relative mix.
    """
    if not in_warmup:
        return dict(curriculum)
    if "train_exploiter" not in curriculum:
        return dict(curriculum)
    masked_weight = curriculum["train_exploiter"]
    if masked_weight <= 0:
        return dict(curriculum)
    remainder_total = sum(v for k, v in curriculum.items() if k != "train_exploiter")
    if remainder_total <= 0:
        # Pathological case: only train_exploiter in the curriculum. Fall
        # back to self_play during warmup so workers still produce data.
        return {"self_play": 1.0}
    scale = (remainder_total + masked_weight) / remainder_total
    masked = {
        k: (v * scale if k != "train_exploiter" else 0.0) for k, v in curriculum.items()
    }
    return masked


def _maybe_run_exploiter_update(
    config: RNaDConfig,
    updates: int,
    agent: RNaDAgent,
    exploiter_learner: Optional[PortfolioRNaDLearner],
    exploiter_agent: Optional[RNaDAgent],
    victim_agent: Optional[RNaDAgent],
    bc_state_dict: Optional[Dict[str, Any]],
    opponent_pool: OpponentPool,
    exploiter_trajectories: List[Dict[str, Any]],
    exploiter_win_buffer: "deque[float]",
    exploiter_updates_total: int,
    exploiter_updates_in_generation: int,
    exploiter_generation: int,
    registry: Optional[ModelRegistry] = None,
) -> Dict[str, Any]:
    """Maybe run one exploiter learner update and handle graduation/reset.

    Fires when (a) pipeline is on, (b) past warmup, and (c) the exploiter
    buffer has ≥ `exploiter.batch_size` trajectories. Otherwise returns the
    existing counters unchanged.

    Mutates `exploiter_trajectories` (cleared after a fired update) and
    `exploiter_win_buffer` (cleared on graduation/timeout) in place so the
    caller's references see the update without reassignment.

    On graduation or timeout, saves a snapshot under `<run_dir>/exploiters/`,
    registers it with the opponent pool, reinitializes the exploiter from BC,
    refreshes the victim from current main, and increments the generation
    counter. The returned `victim_needs_broadcast` flag tells the caller to
    include `victim_weights` in the next checkpoint broadcast.

    Returns a dict with keys: `updates_total`, `updates_in_generation`,
    `generation`, `victim_needs_broadcast`.
    """
    # Guard: pipeline off, in warmup, or buffer underfull → no-op. Fires when:
    #   1. Pipeline is on (exploiter_learner is not None — gated by
    #      train_exploiter > 0 at startup).
    #   2. Past warmup (main has settled into RNaD basin; first exploits
    #      found will target real weaknesses, not BC artifacts).
    #   3. Exploiter buffer has ≥ exploiter_train_batch_size trajectories.
    # Independent of main learner update — both can fire in the same outer
    # iteration, or just one, depending on the 80/20 trajectory split and
    # the buffer fill rates.
    if (
        exploiter_learner is None
        or updates < config.exploiter.warmup_updates
        or len(exploiter_trajectories) < config.exploiter.batch_size
    ):
        return {
            "updates_total": exploiter_updates_total,
            "updates_in_generation": exploiter_updates_in_generation,
            "generation": exploiter_generation,
            "victim_needs_broadcast": False,
        }

    exp_batch = collate_trajectories(
        exploiter_trajectories,
        config.hardware.device,
        config.algorithm.gamma,
        config.algorithm.gae_lambda,
        max_seq_len=config.architecture.max_seq_len,
    )
    exp_metrics = exploiter_learner.update(exp_batch)
    exploiter_updates_total += 1
    exploiter_updates_in_generation += 1
    exploiter_trajectories.clear()

    # Win rate over the rolling window (only meaningful once the buffer has
    # filled; before that we'd be reading from too few samples for a robust gate).
    buffer_full = len(exploiter_win_buffer) >= config.exploiter.graduation_window
    cur_win_rate = (
        sum(exploiter_win_buffer) / len(exploiter_win_buffer)
        if len(exploiter_win_buffer) > 0
        else 0.0
    )

    # Log under `exploiter/` namespace to keep the wandb UI uncluttered
    # (main learner metrics keep the bare names).
    if config.training.use_wandb:
        exp_log = {f"exploiter/{k}": v for k, v in exp_metrics.items()}
        exp_log["exploiter/generation"] = exploiter_generation
        exp_log["exploiter/updates_total"] = exploiter_updates_total
        exp_log["exploiter/updates_in_generation"] = exploiter_updates_in_generation
        exp_log["exploiter/win_rate_rolling"] = cur_win_rate
        exp_log["exploiter/win_buffer_size"] = len(exploiter_win_buffer)
        wandb.log(exp_log)

    # ===== GRADUATION CHECK =====
    # Two paths:
    #   - graduated: buffer full AND win rate ≥ threshold. The exploiter has
    #     demonstrated a real exploit and deserves a place in the curriculum.
    #   - timed_out: this generation hit the per-gen update cap without
    #     graduating. Either main is robust to this basin OR the exploiter
    #     is stuck in a local optimum. Either way, save the best-effort
    #     weights and try a different basin via BC re-init.
    graduated = buffer_full and cur_win_rate >= config.exploiter.graduation_threshold
    timed_out = (
        exploiter_updates_in_generation >= config.exploiter.max_updates_per_generation
    )

    victim_needs_broadcast = False
    if graduated or timed_out:
        snapshot_filename = f"exploiter_gen_{exploiter_generation}_step_{updates}.pt"
        snapshot_path = os.path.join(
            str(config.training.run_dir),
            "exploiters",
            snapshot_filename,
        )
        # Save in the same format checkpoints use (so
        # is_checkpoint_compatible_with_model_config can validate it for the
        # curriculum loader).
        torch.save(
            {
                "model_state_dict": exploiter_agent.model.state_dict()
                if exploiter_agent is not None
                else {},
                "config": config.to_dict(),
                "step": updates,
                "exploiter_generation": exploiter_generation,
                "exploiter_win_rate": cur_win_rate,
                "graduated": graduated,
                "timed_out": timed_out,
                "timestamp": datetime.now().isoformat(),
            },
            snapshot_path,
        )
        opponent_pool.add_exploiter(snapshot_path)
        if registry is not None:
            new_slot = opponent_pool.slot_for_exploiter_path[snapshot_path]
            checkpoint = torch.load(
                snapshot_path,
                map_location=registry.device,
            )
            registry.sync_weights(
                f"exploiter_snap_{new_slot}",
                checkpoint["model_state_dict"],
            )

        logger.info(
            "[Update %d] Exploiter generation %d %s | "
            "win_rate=%.3f over %d battles | "
            "exploiter_updates_in_gen=%d | snapshot=%s",
            updates,
            exploiter_generation,
            "GRADUATED" if graduated else "TIMED OUT (force-reset)",
            cur_win_rate,
            len(exploiter_win_buffer),
            exploiter_updates_in_generation,
            snapshot_path,
        )
        if config.training.use_wandb:
            wandb.log(
                {
                    "exploiter/graduation_event": 1.0,
                    "exploiter/graduation_win_rate": cur_win_rate,
                    "exploiter/generation_completed": exploiter_generation,
                    "exploiter/generation_was_timeout": float(timed_out),
                }
            )

        # ── Reinitialize for a fresh generation ─────────────────────────────
        # BC init forces a different starting basin so the next generation
        # can't just re-learn the exploit main has already defended against
        # (it would converge to the same weights from the same start).
        # Refresh victim from CURRENT main so the new exploiter immediately
        # faces the latest defender rather than a snapshot from the previous
        # victim refresh tick.
        if bc_state_dict is not None and exploiter_agent is not None:
            # strict=False: BC checkpoint may have a different value/win head
            # shape than the runtime architecture; trunk + policy load, heads
            # fresh-initialize. Same rationale as the victim init above.
            exploiter_agent.model.load_state_dict(bc_state_dict, strict=False)
        if victim_agent is not None:
            victim_agent.model.load_state_dict(agent.model.state_dict())
            # Workers must sync to the new victim before the next generation's
            # first train_exploiter battle.
            victim_needs_broadcast = True

        exploiter_generation += 1
        exploiter_updates_in_generation = 0
        exploiter_win_buffer.clear()

    return {
        "updates_total": exploiter_updates_total,
        "updates_in_generation": exploiter_updates_in_generation,
        "generation": exploiter_generation,
        "victim_needs_broadcast": victim_needs_broadcast,
    }


def _build_update_metrics(
    metrics: Dict[str, Any],
    config: RNaDConfig,
    opponent_pool: OpponentPool,
    updates: int,
    total_time: float,
    time_per_update: float,
    total_battles: int,
    battles_this_update: int,
    total_received_trajectories: int,
    total_received_steps: int,
    total_learner_trajectories: int,
    total_learner_steps: int,
    learner_steps_this_update: int,
    learner_trajectories_this_update: int,
) -> Dict[str, float]:
    """Enrich `metrics` in place with throughput/timing/temperature/opponent fields.

    Mutates `metrics` to add all the per-second and total-counter fields the
    wandb log expects. Returns the three "recent" rates separately because the
    console `logger.info` call wants them as positional args (they're also
    stored in `metrics` under `*_per_second_recent` keys for wandb).

    Returns a dict with keys: `battles_per_second`, `learner_steps_per_second`,
    `learner_trajectories_per_second` (all "recent" — i.e., over the last
    update interval rather than cumulative since start).
    """
    metrics["update_step"] = updates
    metrics["total_battles"] = total_battles
    metrics["battles_per_second"] = total_battles / total_time
    metrics["received_trajectories_per_second"] = total_received_trajectories / total_time
    metrics["received_steps_per_second"] = total_received_steps / total_time
    metrics["learner_trajectories_per_second"] = total_learner_trajectories / total_time
    metrics["learner_steps_per_second"] = total_learner_steps / total_time
    metrics["learner_steps_this_update"] = learner_steps_this_update
    metrics["time_per_update_seconds"] = time_per_update

    recent_battles_per_second = (
        battles_this_update / time_per_update if time_per_update > 0 else 0.0
    )
    recent_learner_steps_per_second = (
        learner_steps_this_update / time_per_update if time_per_update > 0 else 0.0
    )
    recent_learner_trajectories_per_second = (
        learner_trajectories_this_update / time_per_update if time_per_update > 0 else 0.0
    )
    metrics["battles_per_second_recent"] = recent_battles_per_second
    metrics["learner_steps_per_second_recent"] = recent_learner_steps_per_second
    metrics["learner_trajectories_per_second_recent"] = (
        recent_learner_trajectories_per_second
    )
    metrics["temperature"] = config.temperature_at_step(updates)
    metrics.update(opponent_pool.get_training_metrics())

    return {
        "battles_per_second": recent_battles_per_second,
        "learner_steps_per_second": recent_learner_steps_per_second,
        "learner_trajectories_per_second": recent_learner_trajectories_per_second,
    }


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--log-debug",
        type=str,
        default="",
        help=(
            "Comma-separated logger names to elevate to DEBUG for this run "
            "(e.g. 'elitefurretai.rl.inference_trainer' to see [batch-fill] "
            "lines that were dropped to DEBUG in commit f7fd34d)."
        ),
    )
    args = parser.parse_args()

    if args.log_debug:
        for name in args.log_debug.split(","):
            name = name.strip()
            if name:
                logging.getLogger(name).setLevel(logging.DEBUG)

    # Set up signal handlers for graceful shutdown (e.g., when killed via nohup)
    shutdown_requested = generate_shutdown_signal()

    # Load config
    config = RNaDConfig.load(args.config)
    config.verify()

    # Memory watchdog: request graceful shutdown if combined RSS approaches
    # the WSL2 VM ceiling, so we get a clean checkpoint instead of a
    # Hyper-V supervisor kill. See
    # planning/stage2/2026-05-12-09-30-second-wsl-crash-and-watchdog.md.
    start_memory_watchdog(shutdown_requested, config.training.memory_watchdog_threshold_gb)

    # ── Rust backend + exploiter co-training are mutually exclusive ───────
    # Co-training requires async multi-agent inference: each worker holds
    # 3 model copies (main, exploiter, victim) driven by independent
    # BatchInferencePlayers, with weights synced via separate broadcast
    # keys. The Rust backend uses a synchronous single-model driver and
    # has no path for this; adding one would duplicate substantial
    # machinery for what is currently a fallback execution path. This
    # guard surfaces the requirement loudly rather than silently falling
    # back to broken behavior (e.g., the Rust backend's update_*_weights
    # methods are no-ops by design — exploiter weights would never reach
    # workers, and train_exploiter battles would silently use stale
    # random-init exploiter/victim weights).
    if (
        config.hardware.battle_backend == RUST_ENGINE_BACKEND
        and _train_exploiter_weight(config) > 0
    ):
        raise ValueError(
            "In-process exploiter co-training (curriculum_weights["
            "'train_exploiter'] > 0) requires the showdown_websocket "
            "backend. The Rust backend does not implement the multi-agent "
            "inference path needed for exploiter/victim. Either set "
            "battle_backend=showdown_websocket or set train_exploiter=0."
        )

    server_processes: List[subprocess.Popen] = []
    if config.hardware.battle_backend != RUST_ENGINE_BACKEND:
        server_processes = launch_showdown_servers(
            config.hardware.num_servers, config.hardware.showdown_start_port
        )

    # Auto-launch external vgc-bench runners based on curriculum: when
    # vgc_bench_baseline has positive curriculum weight the runners
    # come up; otherwise they're skipped to avoid ~3.7 GB host RAM for
    # opponents nobody is asking for.
    vgcbench_manager: Optional[VGCBenchManager] = None
    vgc_bench_curriculum_weight = config.curriculum.curriculum_weights.get(
        OpponentPool.VGC_BENCH_BASELINE, 0.0
    )
    if (
        vgc_bench_curriculum_weight > 0
        and config.hardware.battle_backend != RUST_ENGINE_BACKEND
    ):
        server_ports = [
            config.hardware.showdown_start_port + i
            for i in range(config.hardware.num_servers)
        ]
        vgcbench_manager = VGCBenchManager(config, server_ports)
        vgcbench_manager.launch()

    # Generate unique run ID to avoid stale-account collisions on Showdown server.
    # Include date + high-resolution random bits so rapid restarts don't reuse IDs.
    run_id = f"{datetime.now().strftime('%m%d%H%M%S')}{random.getrandbits(16):04x}"

    # Initialize wandb. If wandb_run_name is None, wandb auto-assigns a random
    # name (e.g. "spring-darkness-42"); we read it back via wandb.run.name.
    if config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            config=config.to_dict(),
            tags=config.training.wandb_tags,
        )
        run_name = wandb.run.name if wandb.run and wandb.run.name else run_id
    else:
        run_name = config.training.wandb_run_name or run_id

    # Resolve the per-run directory. Each wandb run gets its own directory
    # under save_dir to keep checkpoints traceable to their run. On resume,
    # we copy the source run's ghosts/ and exploiters/ snapshots into the new
    # directory so the curriculum (which samples opponents from those
    # subdirs) keeps continuity — new snapshots produced this run accumulate
    # alongside the copies in the new dir, leaving the source untouched.
    run_dir = os.path.join(config.training.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ghosts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "exploiters"), exist_ok=True)
    config.training.run_dir = run_dir
    logger.info("Run directory: %s", run_dir)

    if config.training.resume_from:
        resume_parent = os.path.dirname(os.path.abspath(config.training.resume_from))
        # If resuming from a ghost or exploiter file, walk up one level.
        if os.path.basename(resume_parent) in ("ghosts", "exploiters"):
            resume_parent = os.path.dirname(resume_parent)
        if os.path.abspath(resume_parent) != os.path.abspath(run_dir):
            for subdir in ("ghosts", "exploiters"):
                src_dir = os.path.join(resume_parent, subdir)
                dst_dir = os.path.join(run_dir, subdir)
                if not os.path.isdir(src_dir):
                    logger.info(
                        "Resume copy: %s not present in source run, skipping", src_dir
                    )
                    continue
                files = sorted(
                    f
                    for f in os.listdir(src_dir)
                    if os.path.isfile(os.path.join(src_dir, f))
                )
                if not files:
                    logger.info("Resume copy: %s is empty, nothing to copy", src_dir)
                    continue
                total_bytes = 0
                for fname in files:
                    src_path = os.path.join(src_dir, fname)
                    dst_path = os.path.join(dst_dir, fname)
                    size = os.path.getsize(src_path)
                    total_bytes += size
                    logger.info(
                        "Resume copy: %s/%s (%.1f MB) → %s",
                        subdir,
                        fname,
                        size / 1e6,
                        dst_dir,
                    )
                    shutil.copy2(src_path, dst_path)
                logger.info(
                    "Resume copy: %d file(s), %.1f MB total copied into %s",
                    len(files),
                    total_bytes / 1e6,
                    dst_dir,
                )
        else:
            logger.info(
                "Resume copy: resume source matches new run_dir (%s) — no copy needed",
                run_dir,
            )

    # Initialize model and learner given the config (handles fresh start, resume, and weight initialization cases)
    (
        agent,
        learner,
        worker_model_path,
        worker_model_config,
        start_step,
        resume_curriculum,
    ) = initialize_training_state(config)

    # Create opponent pool
    logger.info("Initializing opponent pool...")
    # The yaml's curriculum wins over the resumed-checkpoint curriculum so
    # users can switch matchup mixes (e.g. enable train_exploiter mid-curve)
    # without re-running from scratch. Log loudly when the yaml differs from
    # the checkpoint, since the change also affects what the registry
    # registers (BC/exploiter/victim) at startup — silent mismatches
    # previously left the registry holding services the opponent pool would
    # never sample from.
    active_curriculum = config.curriculum.curriculum_weights
    if resume_curriculum and resume_curriculum != active_curriculum:
        logger.warning(
            "Curriculum override on resume: yaml weights differ from "
            "checkpoint's saved weights. Using yaml.\n"
            "  yaml:       %s\n"
            "  checkpoint: %s",
            active_curriculum,
            resume_curriculum,
        )
    opponent_pool = OpponentPool(
        main_model=agent,
        device=config.hardware.device,
        battle_format=config.curriculum.battle_format,
        bc_model_path=config.curriculum.bc_model_path,
        exploiter_models_dir=os.path.join(run_dir, "exploiters"),
        ghosts_dir=os.path.join(run_dir, "ghosts"),
        vgc_bench_checkpoint_path=config.curriculum.vgc_bench_checkpoint_path,
        max_ghosts=config.curriculum.max_ghosts,
        max_exploiter_models=config.curriculum.max_exploiter_models,
        curriculum=active_curriculum,
    )

    # ── Initialize the exploiter co-training pipeline ──────────────────────
    # All four returned values are None when train_exploiter == 0 (default).
    # Live state lives in this main process (GPU); workers construct their
    # own CPU model copies and receive weight updates via broadcasts (the
    # `exploiter_weights` / `victim_weights` keys). There is no need to
    # pass agent references to workers — their copies are independent
    # nn.Module instances synchronized by state_dict broadcasts only.
    (
        exploiter_agent,
        victim_agent,
        exploiter_learner,
        bc_state_dict,
    ) = _initialize_exploiter_pipeline(config, agent, worker_model_config)
    exploiter_pipeline_on = exploiter_learner is not None

    start = time.time()

    # ==================== START WORKERS ====================
    # Each worker is a separate OS process running worker.py:mp_worker_process.
    # We give each worker:
    #   - a unique server_port (or 0 for Rust backend)
    #   - the model checkpoint path to bootstrap from (loaded fresh per worker)
    #   - shared mp queues for trajectories (in) and weight updates (out)
    #   - an mp.Event for graceful shutdown
    # The trainer process itself does NOT play battles; it only collects
    # trajectories and trains. All Showdown websocket activity lives in workers.
    if config.hardware.battle_backend == RUST_ENGINE_BACKEND:
        worker_ports = [0 for _ in range(config.hardware.num_workers)]
        server_loads: List[int] = []
    else:
        worker_ports, server_loads = allocate_server_ports(
            config.hardware.num_workers,
            config.hardware.players_per_worker,  # Number of concurrent players each worker runs
            config.hardware.num_servers,
            config.hardware.max_players_per_server,
            config.hardware.showdown_start_port,
        )

    # ── Multiprocessing queues for inter-process communication ──────────────
    #   mp_traj_queue (workers → trainer): completed trajectory dicts
    #     maxsize=1024 prevents unbounded memory growth if the trainer falls
    #     behind. If the queue fills, workers block until the trainer drains.
    #   weight_queues (trainer → each worker): one queue per worker so we can
    #     broadcast new weights without serializing through a shared queue.
    #     maxsize=2 with "drain before put" semantics prevents stale weight
    #     payloads from piling up.
    #   mp_error_queue: workers push tracebacks here on crash so the trainer
    #     can surface them in logs before exiting.
    #   mp_stop_event: trainer sets this on shutdown to ask workers to exit.
    # ────────────────────────────────────────────────────────────────────────
    mp_traj_queue: MPQueue = MPQueue(maxsize=1024)
    weight_queues: List[MPQueue] = [
        MPQueue(maxsize=2) for _ in range(config.hardware.num_workers)
    ]
    mp_error_queue: MPQueue = MPQueue(maxsize=100)
    mp_stop_event: MPEvent = mp.Event()

    # ── Centralized inference ────────────────────────────────────────────
    # Trainer process owns a ModelRegistry of InferenceServices (main
    # always, BC if curriculum uses it, exploiter/victim if
    # train_exploiter is on). Workers submit inference requests via
    # mp.Queues. Each service holds an independent copy of its model so
    # the learner can update without racing with inference; we sync
    # weights into them at the same cadence as the per-worker broadcast.
    registry: Optional[ModelRegistry] = None
    main_request_queue: Optional[MPQueue] = None
    main_response_queues: List[Optional[MPQueue]] = [None] * config.hardware.num_workers
    inference_device = config.hardware.device
    inference_embedder = Embedder(
        format=config.curriculum.battle_format,
        feature_set=config.training.embedder_feature_set,
        omniscient=False,
    )
    registry = ModelRegistry(
        num_workers=config.hardware.num_workers,
        batch_size=config.hardware.batch_size,
        batch_timeout=config.hardware.batch_timeout,
        device=inference_device,
        compile_mode=config.hardware.compile_inference_model,
        embedding_size=inference_embedder.embedding_size,
    )

    # Build a CPU-or-GPU copy of the main model for inference (kept
    # independent of the learner's model so backward+forward don't
    # race; weights sync at broadcast cadence via registry.sync_weights).
    main_inference_base = build_model_from_config(
        worker_model_config,
        inference_embedder,
        inference_device,
        None,
        strict=False,
    )
    main_inference_base.eval()
    main_inference_base.load_state_dict(agent.model.state_dict())
    registry.register("main", RNaDAgent(main_inference_base))

    # Step 3: register BC if a BC checkpoint is configured AND the
    # curriculum will actually use it. Wasting a service slot on an
    # unused model would still cost compile time + GPU memory; the
    # curriculum check keeps that gated.
    bc_checkpoint_path = config.curriculum.bc_model_path
    bc_curriculum_weight = config.curriculum.curriculum_weights.get(
        OpponentPool.BC_PLAYER, 0.0
    )
    if bc_checkpoint_path and bc_curriculum_weight > 0:
        logger.info(
            "Step 3: registering BC model from %s (curriculum weight %.3f)",
            bc_checkpoint_path,
            bc_curriculum_weight,
        )
        bc_checkpoint = torch.load(
            bc_checkpoint_path, map_location="cpu", weights_only=False
        )
        bc_inference_base = build_model_from_config(
            worker_model_config,
            inference_embedder,
            inference_device,
            bc_checkpoint["model_state_dict"],
            strict=False,
        )
        bc_inference_base.eval()
        for param in bc_inference_base.parameters():
            param.requires_grad = False
        # compile=True: safe now that RealModelBatchHandler holds
        # _COMPILE_LOCK (a global process-wide threading.Lock) around
        # every compiled forward call. The lock serializes dynamo's
        # global trace state across concurrent InferenceService threads,
        # eliminating the "FX symbolic trace of dynamo-optimized function"
        # race. See unit_tests/rl/test_compile_race_reproducer.py.
        # bc moves to the "frozen" subprocess: never syncs after
        # startup (BC is frozen forever), so subprocess co-location
        # costs nothing — and frees one GIL-contender from the trainer
        # process. The trainer's GIL ceiling proved binding when 4
        # services contend; moving bc out brings the count to 3
        # (main + exploiter + victim).
        bc_inference_base.cpu()
        registry.register(
            "bc",
            RNaDAgent(bc_inference_base),
            compile=True,
            process_group="frozen",
        )
        del bc_checkpoint

    # Step 4: register exploiter (live-trained adversary) + victim
    # (frozen periodically-refreshed copy of main) if the curriculum
    # gates the train_exploiter pipeline ON. exploiter is sync'd from
    # the exploiter learner each update; victim is sync'd from main
    # at victim_refresh_interval. Both registered with compile=True
    # now that the global _COMPILE_LOCK in RealModelBatchHandler
    # serializes dynamo's trace state across concurrent threads.
    #
    # Plan C grouping for live-trained pair:
    #   - exploiter STAYS IN-PROCESS. The exploiter learner runs in the
    #     trainer and updates exploiter weights at high frequency;
    #     cross-process sync (~24ms per IPC) can't keep up.
    #   - victim MOVES TO "frozen". It refreshes every
    #     victim_refresh_interval (1000 battles ≈ 3 min at 6 traj/s),
    #     well below the IPC throughput ceiling — and getting it out of
    #     trainer keeps the trainer GIL contention bounded.
    train_exploiter_weight = config.curriculum.curriculum_weights.get(
        OpponentPool.TRAIN_EXPLOITER, 0.0
    )
    if train_exploiter_weight > 0:
        logger.info(
            "Step 4: registering exploiter (in-process) + victim (frozen subprocess) "
            "(curriculum weight %.3f)",
            train_exploiter_weight,
        )
        # exploiter: in-trainer (sync-rate constraint)
        exploiter_base = build_model_from_config(
            worker_model_config,
            inference_embedder,
            inference_device,
            None,
            strict=False,
        )
        exploiter_base.eval()
        # exploiter starts with whatever build_model_from_config gave
        # us (BC init if `initialize_path` is set, else fresh). It'll
        # be sync'd as the exploiter learner produces updates.
        registry.register("exploiter", RNaDAgent(exploiter_base), compile=True)
        # victim: subprocess (rare refresh, no sync-rate constraint)
        victim_base = build_model_from_config(
            worker_model_config,
            inference_embedder,
            "cpu",
            None,
            strict=False,
        )
        victim_base.eval()
        # Victim starts as a copy of main; weights refresh periodically
        # via registry.sync_weights("victim", ...).
        victim_base.load_state_dict(agent.model.state_dict())
        for param in victim_base.parameters():
            param.requires_grad = False
        registry.register(
            "victim",
            RNaDAgent(victim_base),
            compile=True,
            process_group="frozen",
        )

    # Ghost slots: pre-register max_ghosts services so the slot pool is
    # fixed-size and the registration plumbing never happens mid-run.
    # Slots start with main-agent weights as placeholders; only slots
    # listed in `opponent_pool.active_ghost_slots()` are valid routing
    # targets (workers filter on that set). compile=True now that the
    # global _COMPILE_LOCK in RealModelBatchHandler eliminates the
    # torch.compile multi-thread race (see registry plan).
    #
    # 2026-05-15 Plan C step 5: ghosts (LRU swaps on checkpoint events)
    # AND exploiter_snaps (LRU swaps on graduation events) share a
    # single "frozen" subprocess. They have homogeneous low-frequency
    # weight-sync semantics. Earlier the plan called for two separate
    # subprocesses ("ghosts" + "snaps"), but the compile-peak memory
    # footprint (~4.6 GB per subprocess) tripped the watchdog at 3-
    # subprocess fanout. Merging halves per-subprocess fixed overhead
    # (Python interpreter, PyTorch import, CUDA context) at the cost of
    # serializing forwards within the merged subprocess — acceptable
    # because both groups are low-traffic vs main/bc in trainer.
    # Checkpoints load to CPU because the subprocess does its own
    # device move + compile at startup; loading directly to cuda would
    # leak transient GPU allocations during trainer-side staging.
    for slot in range(config.curriculum.max_ghosts):
        ghost_agent = copy.deepcopy(registry._raw_agents["main"])
        ghost_agent.model.cpu()
        registry.register(
            f"ghost_{slot}", ghost_agent, compile=True, process_group="frozen"
        )
    # Load weights for any pre-existing ghost checkpoints onto their
    # assigned slots. `slot_for_ghost_path` was populated by
    # OpponentPool._load_ghosts. sync_weights pre-start_all() updates
    # the trainer-side CPU shadow only; the subprocess picks up the
    # latest weights via the shadow at spawn time.
    for path, slot in opponent_pool.slot_for_ghost_path.items():
        checkpoint = torch.load(path, map_location="cpu")
        registry.sync_weights(f"ghost_{slot}", checkpoint["model_state_dict"])

    # Exploiter snapshot slots: pre-register max_exploiter_models services
    # (parallel to ghosts). Each holds an independent agent; sync_weights
    # populates real exploiter snapshot weights from disk for any slot
    # OpponentPool's _load_exploiter_models pre-assigned at startup.
    #
    # Plan C step 5: snaps SHARE the "frozen" subprocess with ghosts
    # (see ghost loop above for rationale).
    for slot in range(config.curriculum.max_exploiter_models):
        exploiter_snap_agent = copy.deepcopy(registry._raw_agents["main"])
        exploiter_snap_agent.model.cpu()
        registry.register(
            f"exploiter_snap_{slot}",
            exploiter_snap_agent,
            compile=True,
            process_group="frozen",
        )
    for path, slot in opponent_pool.slot_for_exploiter_path.items():
        checkpoint = torch.load(path, map_location="cpu")
        registry.sync_weights(f"exploiter_snap_{slot}", checkpoint["model_state_dict"])

    # All services registered — start in-process service threads and
    # spawn subprocesses for any process_group set above. Must happen
    # before workers connect because workers will start sending requests
    # immediately after spawn.
    registry.start_all()

    # Pull out main's queues for the back-compat per-worker
    # spawn-args interface. The full bundle is also passed below so
    # workers can construct WorkerInferenceClients with one client
    # per registered model.
    all_queues = registry.queues_for_workers()
    main_request_queue, main_response_qs = all_queues["main"]
    main_response_queues = list(main_response_qs)

    logger.info(
        "ModelRegistry initialized; %d models registered (%s). "
        "device=%s batch_size=%d batch_timeout=%.4f compile=%s",
        len(registry.names()),
        ", ".join(registry.names()),
        inference_device,
        config.hardware.batch_size,
        config.hardware.batch_timeout,
        config.hardware.compile_inference_model,
    )

    # Per-worker bundle of (request_q, response_q) pairs keyed by model
    # name. Each worker gets ITS slice of the response queues; the
    # request queue is shared across workers per model. Workers wrap
    # their slice in WorkerInferenceClients on the worker side.
    queues_by_worker: List[Optional[Dict[str, Any]]] = [None] * config.hardware.num_workers
    for w in range(config.hardware.num_workers):
        queues_by_worker[w] = {
            name: (req_q, resp_qs[w]) for name, (req_q, resp_qs) in all_queues.items()
        }

    # Create Processes
    processes: List[mp.Process] = []
    for i, server_port in enumerate(worker_ports):
        assert mp_traj_queue and mp_error_queue and mp_stop_event
        p = mp.Process(
            target=mp_worker_process,
            args=(
                i,  # worker_id
                server_port,
                worker_model_path,
                worker_model_config,
                mp_traj_queue,
                weight_queues[i],
                mp_error_queue,
                mp_stop_event,
                run_id,
                config,
                False,  # verbose
                main_request_queue,
                main_response_queues[i],
                queues_by_worker[i],
                sorted(opponent_pool.active_ghost_slots()),
                sorted(opponent_pool.active_exploiter_slots()),
            ),
            daemon=True,
            name=f"MPWorker-{i}",
        )
        p.start()
        processes.append(p)
        logger.info(
            "Started multiprocessing worker %d (PID: %s) on port %d", i, p.pid, server_port
        )

    if server_loads:
        logger.info("Server allocation (players per server):")
        for idx, load in enumerate(server_loads):
            port = config.hardware.showdown_start_port + idx
            cap = config.hardware.max_players_per_server
            logger.info("  Port %d: %d/%d players", port, load, cap)

    # ==================== MAIN TRAINING LOOP ====================
    # The trainer's hot loop. On each iteration:
    #   1. Check for shutdown signal or dead workers.
    #   2. Pop one trajectory from the worker → trainer queue (blocking up to 1s).
    #   3. Route by opponent_type: train_exploiter trajectories feed the
    #      exploiter learner; everything else feeds main. Both buffers
    #      record win/loss with the opponent pool for unified logging.
    #   4. Once a buffer ≥ its batch size, collate and call .update() on
    #      the corresponding learner. Two learners run independently with
    #      different cadences (main fires every train_batch_size=256
    #      trajectories; exploiter every exploiter_train_batch_size=64).
    #   5. Periodically (per-checkpoint_interval): update reference model,
    #      save main checkpoint, broadcast new weights + curriculum +
    #      exploiter/victim weights to workers.
    #   6. After every exploiter update, check graduation criteria: if
    #      win-rate ≥ threshold OR per-generation update cap hit, save
    #      exploiter snapshot to <run_dir>/exploiters/ (entering the
    #      EXPLOITERS curriculum slot for main to defend against),
    #      reinitialize the exploiter from BC, refresh victim from current
    #      main, and increment the generation counter.
    # ─────────────────────────────────────────────────────────────────────────
    trajectories = []  # Main learner buffer
    updates = start_step  # Main update step (may be >0 if resumed from checkpoint)
    total_battles = 0  # Total battles completed across all workers
    total_received_trajectories = 0
    total_received_steps = 0
    total_learner_trajectories = 0
    total_learner_steps = 0
    last_update_time = time.time()
    prev_total_battles = 0  # snapshot at last log for recent b/s

    # ── Exploiter pipeline state (only meaningful when on) ───────────────────
    # `exploiter_trajectories` is the second buffer for the exploiter learner.
    # `exploiter_updates_total` counts gradient updates since startup; used
    #   for global logging.
    # `exploiter_updates_in_generation` resets on graduation; used to enforce
    #   the `exploiter_max_updates_per_generation` stall safeguard.
    # `exploiter_generation` is incremented every time an exploiter graduates
    #   (or hits the stall cap and is force-reset). 0-indexed, so generation
    #   0 is the deepcopy-of-main start.
    # `exploiter_win_buffer` is a rolling deque of 1.0/0.0 wins over the last
    #   `graduation_window` train_exploiter battles. We don't reuse
    #   opponent_pool.win_rate_tracking because that uses a fixed
    #   tracking_window (default 100) sized for live logging — too small
    #   to be a statistically robust graduation gate.
    # `victim_needs_broadcast` flags the next broadcast to include refreshed
    #   victim weights. Cleared after the broadcast fires.
    exploiter_trajectories: List[Dict[str, Any]] = []
    exploiter_updates_total = 0
    exploiter_updates_in_generation = 0
    exploiter_generation = 0
    exploiter_win_buffer: "deque[float]" = deque(maxlen=config.exploiter.graduation_window)
    victim_needs_broadcast = False

    try:
        while updates < config.training.max_updates:
            # Check for shutdown signal (from SIGTERM/SIGINT)
            if shutdown_requested.is_set():
                logger.info("Shutdown requested. Signaling workers to stop...")
                mp_stop_event.set()
                break

            # Check if workers have died
            dead_workers: List[mp.Process] = get_dead_workers(processes, mp_error_queue)
            if len(dead_workers) > 0:
                logger.error(
                    "Training cannot continue due to worker failure. Saving checkpoint and exiting..."
                )
                break

            # ===== COLLECT TRAJECTORIES FROM WORKERS =====
            # Workers push completed battle trajectories to the queue asynchronously.
            # Each trajectory carries an `opponent_type` tag set at battle setup time
            # (see WorkerOpponentFactory.configure_opponent_for_batch). The tag tells
            # us which learner the trajectory belongs to:
            #   - "train_exploiter" → exploiter learner (player WAS the exploiter,
            #     opponent WAS the frozen victim; trajectory captures exploiter's
            #     actions, rewards, log probs).
            #   - everything else → main learner (self_play, bc_player, ghosts,
            #     exploiters [vs frozen snapshots], baselines).
            # Routing happens AT INGRESS rather than at update time so the buffers
            # never get cross-contaminated.
            try:
                traj = mp_traj_queue.get(timeout=1.0)

                total_battles += 1  # Each trajectory represents one completed battle
                total_received_trajectories += 1
                total_received_steps += len(traj["steps"])

                # Track win rate by opponent type. These rolling results are consumed by
                # OpponentPool.update_curriculum(). Smoothing/noise handling lives there.
                # Done for ALL trajectories so logging shows train_exploiter win rates
                # alongside the others.
                opponent_pool.record_battle_result(
                    opponent_type=traj["opponent_type"],
                    won=traj["won"],
                    battle_length=traj["battle_length"],
                    forfeited=traj["forfeited"],
                )

                # Route by opponent_type. The exploiter learner only consumes
                # battles where the player was the live exploiter; main learner
                # gets everything else (its existing trajectory shape doesn't
                # change). Forfeited train_exploiter battles still count for
                # the win-rate gate (a forfeit is a 0.0, treated like any
                # other loss).
                if (
                    exploiter_pipeline_on
                    and traj["opponent_type"] == OpponentPool.TRAIN_EXPLOITER
                ):
                    exploiter_trajectories.append(traj)
                    exploiter_win_buffer.append(1.0 if traj["won"] else 0.0)
                else:
                    trajectories.append(traj)
            except queue.Empty:
                # No new trajectories yet, continue waiting
                continue

            # ===== PERFORM TRAINING UPDATE =====
            # Once we've collected enough trajectories, train the model
            if len(trajectories) >= config.training.train_batch_size:
                # Collate trajectories into padded batches (handles variable length sequences)
                batch = collate_trajectories(
                    trajectories,
                    config.hardware.device,
                    config.algorithm.gamma,
                    config.algorithm.gae_lambda,
                    max_seq_len=config.architecture.max_seq_len,
                )
                learner_steps_this_update = int(batch["padding_mask"].sum().item())
                learner_trajectories_this_update = len(trajectories)
                total_learner_steps += learner_steps_this_update
                total_learner_trajectories += learner_trajectories_this_update
                battles_this_update = total_battles - prev_total_battles
                prev_total_battles = total_battles

                # Execute one RNaD policy update (PPO + KL regularization vs reference)
                metrics = learner.update(batch)
                updates += 1

                # Calculate time since last update
                current_time = time.time()
                time_per_update = current_time - last_update_time
                last_update_time = current_time

                # ===== LOG TRAINING METRICS =====
                total_time = time.time() - start
                recent_rates = _build_update_metrics(
                    metrics=metrics,
                    config=config,
                    opponent_pool=opponent_pool,
                    updates=updates,
                    total_time=total_time,
                    time_per_update=time_per_update,
                    total_battles=total_battles,
                    battles_this_update=battles_this_update,
                    total_received_trajectories=total_received_trajectories,
                    total_received_steps=total_received_steps,
                    total_learner_trajectories=total_learner_trajectories,
                    total_learner_steps=total_learner_steps,
                    learner_steps_this_update=learner_steps_this_update,
                    learner_trajectories_this_update=learner_trajectories_this_update,
                )

                # Build win rate string for console output
                win_rate_str = " | ".join(
                    [
                        f"{opp_type}: {wr * 100:.1f}%"
                        for opp_type, wr in opponent_pool.get_win_rate_stats().items()
                        if len(opponent_pool.win_rate_tracking.get(opp_type, [])) > 0
                    ]
                )
                if win_rate_str:
                    win_rate_str = f" | Win rates: {win_rate_str}"

                logger.info(
                    "Update %d: Loss=%.4f, Policy=%.4f, Value=%.4f, RNaD=%.4f | "
                    "Total Battles=%d in %s (%.2f b/s; %.2f overall) | "
                    "Learner Steps=%d (%.2f steps/s; %.2f overall) | "
                    "Learner Trajectories=%d (%.2f traj/s; %.2f overall)%s",
                    updates,
                    metrics["loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["rnad_loss"],
                    total_battles,
                    format_time(total_time),
                    recent_rates["battles_per_second"],
                    total_battles / total_time,
                    total_learner_steps,
                    recent_rates["learner_steps_per_second"],
                    total_learner_steps / total_time,
                    total_learner_trajectories,
                    recent_rates["learner_trajectories_per_second"],
                    total_learner_trajectories / total_time,
                    win_rate_str,
                )

                if (
                    config.training.use_wandb
                    and updates % config.training.log_interval == 0
                ):
                    metrics["portfolio_size"] = len(learner.ref_models)

                    # Reference selection counts
                    total_selections = sum(learner.portfolio_selection_counts)
                    if total_selections > 0:
                        for ref_idx, count in enumerate(
                            learner.portfolio_selection_counts
                        ):
                            metrics[f"portfolio_selection_pct_ref_{ref_idx}"] = (
                                count / total_selections
                            ) * 100

                    wandb.log(metrics)

                # ===== ADD NEW REFERENCE TO PORTFOLIO =====
                # The reference set is the "anchors" the KL penalty pulls the current
                # policy toward (min-KL across them). Every `portfolio_add_interval`
                # updates we append a frozen snapshot of the current policy and prune
                # the oldest if we exceed `max_portfolio_size`. With size=1 this is
                # standard RNaD's moving anchor; with size>1 we keep a rolling set of
                # historical anchors to prevent cyclic strategy drift. The portfolio
                # is different from the ghosts in the curriculum, since the models
                # in the portfolio are used for loss, while models in the curriculum
                # are used to battle against and create trajectories
                if updates % config.portfolio.portfolio_add_interval == 0:
                    logger.info(
                        "[Update %d] Adding new reference to portfolio...", updates
                    )
                    _t_add_ref = time.perf_counter()
                    learner.add_reference_model(
                        RNaDAgent(copy.deepcopy(agent.model))
                    )  # Snapshot current policy
                    logger.info(
                        "[Update %d] add_reference_model: %.1fms (portfolio_size=%d)",
                        updates,
                        (time.perf_counter() - _t_add_ref) * 1000.0,
                        len(learner.ref_models),
                    )

                # ===== SAVE CHECKPOINT =====
                # Periodically save model, optimizer state, and training progress
                # NOTE: this saves ghosts in an unbounded way as training goes
                # on; this happens with all the models saved below
                if updates % config.training.checkpoint_interval == 0:
                    logger.info(
                        "[Update %d] Saving checkpoint and updating curriculum...", updates
                    )
                    _t_block_start = time.perf_counter()

                    # Save model, for safety and to use to battle against
                    _t_save = time.perf_counter()
                    ghost_checkpoint_path = save_checkpoint(
                        agent,
                        learner,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        os.path.join(str(config.training.run_dir), "ghosts"),
                    )
                    logger.info(
                        "[Update %d] save_checkpoint: %.1fms",
                        updates,
                        (time.perf_counter() - _t_save) * 1000.0,
                    )

                    # Add checkpoint to ghosts pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_ghost(updates, ghost_checkpoint_path)
                    # Sync new weights into the registry slot. OpponentPool
                    # already assigned the slot in add_ghost; look it up.
                    if registry is not None:
                        new_slot = opponent_pool.slot_for_ghost_path[ghost_checkpoint_path]
                        checkpoint = torch.load(
                            ghost_checkpoint_path,
                            map_location=registry.device,
                        )
                        registry.sync_weights(
                            f"ghost_{new_slot}", checkpoint["model_state_dict"]
                        )

                    # Recompute curriculum in the learner/main process only,
                    # if we want to update it. Algorithm to update the
                    # curriculum is TODO: define and describe
                    if config.curriculum.adaptive_curriculum:
                        opponent_pool.update_curriculum()

                    # ===== BROADCAST WEIGHTS TO WORKERS =====
                    # We send a single dict per worker containing weights AND
                    # the new curriculum AND sampling knobs. Doing them in one
                    # payload guarantees workers transition together instead
                    # of one updating curriculum first and another updating
                    # weights first (which could produce inconsistent data).
                    #
                    # When the exploiter pipeline is on, the payload is also
                    # the carrier for `exploiter_weights` (live exploiter,
                    # included EVERY broadcast — workers' exploiter copies
                    # need to track gradient updates closely) and
                    # `victim_weights` (frozen victim, included ONLY when
                    # refreshed — sending it every broadcast would waste
                    # bandwidth since the victim is intentionally stale).
                    #
                    # The curriculum we send is also masked during warmup so
                    # workers don't sample train_exploiter battles before the
                    # exploiter learner has seen any updates (see
                    # `_mask_curriculum_during_warmup` for the rationale).
                    #
                    # Note: state_dicts are moved to CPU before pickling.
                    # Workers are CPU-only so this both fits their device and
                    # is the only way to transfer GPU tensors via mp.Queue.
                    logger.info(
                        "[Update %d] Broadcasting weights to worker processes...", updates
                    )
                    _t_cpu_move = time.perf_counter()
                    cpu_weights = {k: v.cpu() for k, v in agent.model.state_dict().items()}
                    logger.info(
                        "[Update %d] main->cpu state_dict: %.1fms",
                        updates,
                        (time.perf_counter() - _t_cpu_move) * 1000.0,
                    )
                    # Centralized inference: sync the trainer-side
                    # InferenceService model(s) from the learner. Same
                    # cadence as the worker broadcast — workers in
                    # centralized mode ignore the broadcasted weights
                    # (their VGCEnvironment.update_weights drops on
                    # None model) but the service copy MUST be kept
                    # current or workers will play increasingly
                    # off-policy. Step 2 syncs only "main"; step 4 will
                    # also sync "exploiter" + "victim" here.
                    if registry is not None and "main" in registry.names():
                        registry.sync_weights("main", cpu_weights)
                    in_warmup = updates < config.exploiter.warmup_updates
                    update_payload: Dict[str, Any] = {
                        "weights": cpu_weights,
                        "curriculum": _mask_curriculum_during_warmup(
                            opponent_pool.curriculum, in_warmup
                        ),
                        "temperature": config.temperature_at_step(updates),
                        "top_p": config.exploration.top_p,
                        "active_ghost_slots": sorted(opponent_pool.active_ghost_slots()),
                        "active_exploiter_slots": sorted(
                            opponent_pool.active_exploiter_slots()
                        ),
                    }
                    if exploiter_pipeline_on and exploiter_agent is not None:
                        exploiter_cpu = {
                            k: v.cpu()
                            for k, v in exploiter_agent.model.state_dict().items()
                        }
                        update_payload["exploiter_weights"] = exploiter_cpu
                        # Step 4: sync the trainer-side exploiter
                        # InferenceService at the same cadence as the
                        # worker broadcast.
                        if registry is not None and "exploiter" in registry.names():
                            registry.sync_weights("exploiter", exploiter_cpu)
                    if (
                        exploiter_pipeline_on
                        and victim_agent is not None
                        and victim_needs_broadcast
                    ):
                        victim_cpu = {
                            k: v.cpu() for k, v in victim_agent.model.state_dict().items()
                        }
                        update_payload["victim_weights"] = victim_cpu
                        if registry is not None and "victim" in registry.names():
                            registry.sync_weights("victim", victim_cpu)
                        # Cleared after queueing: each refresh is broadcast
                        # exactly once, then we wait for the next refresh tick.
                        victim_needs_broadcast = False
                    _t_wq = time.perf_counter()
                    for i, wq in enumerate(weight_queues):
                        try:
                            # Clear old weights to avoid queue overflow
                            # Keep-at-most-latest semantics prevents workers from replaying stale
                            # curriculum/weight snapshots when learner is faster than consumers.
                            while not wq.empty():
                                try:
                                    wq.get_nowait()
                                except Exception:
                                    break
                            wq.put_nowait(update_payload)
                        except Exception as e:
                            logger.warning("Failed to broadcast to worker %d: %s", i, e)
                    logger.info(
                        "[Update %d] worker queue broadcast (%d workers): %.1fms | "
                        "checkpoint+broadcast block total=%.1fms",
                        updates,
                        len(weight_queues),
                        (time.perf_counter() - _t_wq) * 1000.0,
                        (time.perf_counter() - _t_block_start) * 1000.0,
                    )

                # ===== VICTIM REFRESH =====
                # The exploiter trains against a stationary target (the
                # victim) so its gradient signal is stable. But if we never
                # refreshed, the exploiter would eventually master a stale
                # main and graduate snapshots irrelevant to the *current*
                # main. So every `victim_refresh_interval` main updates,
                # copy main's current state_dict into the victim and queue
                # a `victim_weights` broadcast so workers sync up.
                #
                # Indexed on MAIN updates (not exploiter updates) so the
                # cadence is independent of `exploiter_train_batch_size`.
                # Skip update 0 — there's nothing fresh to refresh from at
                # initialization.
                if (
                    exploiter_pipeline_on
                    and victim_agent is not None
                    and updates > 0
                    and updates % config.exploiter.victim_refresh_interval == 0
                ):
                    victim_agent.model.load_state_dict(agent.model.state_dict())
                    # Flag for the NEXT broadcast (which may be this
                    # iteration if the schedules align, or the next
                    # checkpoint_interval otherwise).
                    victim_needs_broadcast = True
                    logger.info("[Update %d] Refreshed victim from current main", updates)

                # ===== EXPLOITER LEARNER UPDATE + GRADUATION =====
                # All exploiter-side logic lives in `_maybe_run_exploiter_update`:
                # gating on warmup/buffer-fill, the learner step, win-rate
                # graduation, snapshot save, and BC re-init. Helper mutates the
                # trajectory/win buffers in place and returns the updated counters.
                exploiter_result = _maybe_run_exploiter_update(
                    config=config,
                    updates=updates,
                    agent=agent,
                    exploiter_learner=exploiter_learner,
                    exploiter_agent=exploiter_agent,
                    victim_agent=victim_agent,
                    bc_state_dict=bc_state_dict,
                    opponent_pool=opponent_pool,
                    exploiter_trajectories=exploiter_trajectories,
                    exploiter_win_buffer=exploiter_win_buffer,
                    exploiter_updates_total=exploiter_updates_total,
                    exploiter_updates_in_generation=exploiter_updates_in_generation,
                    exploiter_generation=exploiter_generation,
                    registry=registry,
                )
                exploiter_updates_total = exploiter_result["updates_total"]
                exploiter_updates_in_generation = exploiter_result["updates_in_generation"]
                exploiter_generation = exploiter_result["generation"]
                # OR with current flag: a pending victim refresh from earlier this
                # iteration (the `victim_refresh_interval` block above) must not be
                # cleared by a no-op exploiter call.
                victim_needs_broadcast = (
                    victim_needs_broadcast or exploiter_result["victim_needs_broadcast"]
                )

                # Clear trajectory buffer after successful update
                trajectories = []

                # Periodic memory cleanup to prevent gradual OOM
                if updates % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected...")
        mp_stop_event.set()
    finally:
        # ===== GRACEFUL SHUTDOWN =====
        logger.info("Waiting for workers to finish (max 5 seconds)...")

        # Signal multiprocessing workers to stop
        mp_stop_event.set()

        # Wait for processes to finish
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                logger.warning(
                    "Worker %s (PID: %s) still running, terminating...", p.name, p.pid
                )
                p.terminate()
                p.join(timeout=1.0)

        # Save progress before exiting (handles Ctrl+C, kill, or normal completion)
        logger.info("Shutting down training...")

        # Save final checkpoint so training can be resumed later
        final_path = save_checkpoint(
            agent,
            learner,
            updates,
            config,
            opponent_pool.curriculum,
            str(config.training.run_dir),
        )
        logger.info("Final model saved to %s", final_path)

        # Close W&B run properly to ensure all logs are synced
        if config.training.use_wandb:
            wandb.finish()

        # Final cleanup: Shutdown workers and Showdown servers
        cleanup_worker_executors()
        # Stop every InferenceService thread the registry started. Drain
        # nothing — workers have exited so no new requests will arrive;
        # any already-buffered ones get dropped.
        if registry is not None:
            registry.stop_all()
            logger.info("ModelRegistry stopped (services: %s)", registry.names())
        if vgcbench_manager is not None:
            vgcbench_manager.shutdown()
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization or mp.Process creation
    mp.set_start_method("spawn", force=True)

    # Configure platform-specific multiprocessing behavior
    configure_torch_multiprocessing(use_file_system_sharing=True)
    suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

    # Root logger at WARNING silences poke-env's per-player loggers (named
    # by random username e.g. "M00OP00E08ED700") which otherwise echo every
    # raw Showdown websocket request/response at INFO. On a 200-update run
    # those add up to ~10+ GB of log — enough to OOM/disk-fill WSL2.
    # Our own modules sit under "elitefurretai" and stay at INFO so training
    # progress (Update N: ..., curriculum, checkpoint events) is preserved.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("elitefurretai").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)

    main()
