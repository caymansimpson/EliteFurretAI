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
        │   (×N workers)  │  ←──────────── │   (back-channel)   │
        └────────┬────────┘                └────────────────────┘
                 │ websockets
        ┌────────┴────────┐
        │ Showdown servers│
        │  (×M servers)   │
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
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch

import wandb
from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_external_vgcbench_runners,
    launch_showdown_servers,
    shutdown_external_vgcbench_runners,
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
        base_model = build_model_from_config(cfg, embedder, config.hardware.device, None)
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)

        start_step, old_config = load_checkpoint(
            config.training.resume_from, agent, learner.optimizer, config.hardware.device
        )
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
        checkpoint_rnad = RNaDConfig.from_dict(checkpoint_cfg)
        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=checkpoint_rnad.training.embedder_feature_set,
            omniscient=False,
        )
        base_model = build_model_from_config(
            checkpoint_cfg,
            embedder,
            config.hardware.device,
            init_checkpoint["model_state_dict"],
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


def train_exploiter_subprocess(victim_checkpoint: str, config: RNaDConfig):
    """Launch exploiter training as subprocess."""
    logger.info("EXPLOITER TRAINING TRIGGERED - Victim: %s", victim_checkpoint)

    # Launch exploiter_train.py as subprocess
    cmd = [
        sys.executable,
        "src/elitefurretai/rl/exploiter_train.py",
        "--victim",
        victim_checkpoint,
        "--steps",
        str(config.training.exploiter_updates),
        "--eval-games",
        str(config.training.exploiter_eval_games),
        "--threshold",
        str(config.training.exploiter_min_win_rate),
        "--output-dir",
        os.path.join(str(config.training.run_dir), "exploiters"),
        "--team-pool",
        config.training.exploiter_team_pool_path
        if config.training.exploiter_team_pool_path
        else "",
        "--learning-rate",
        str(config.training.exploiter_lr),
        "--ent-coef",
        str(config.training.exploiter_ent_coef),
    ]

    subprocess.run(cmd, check=True)
    logger.info("Exploiter training complete")


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown (e.g., when killed via nohup)
    shutdown_requested = generate_shutdown_signal()

    # Load config
    config = RNaDConfig.load(args.config)
    config.verify()

    server_processes: List[subprocess.Popen] = []
    if config.hardware.battle_backend != RUST_ENGINE_BACKEND:
        server_processes = launch_showdown_servers(
            config.hardware.num_servers, config.hardware.showdown_start_port
        )

    # Optionally auto-launch external vgc-bench runners (isolated environment)
    external_runner_processes: List[subprocess.Popen] = []
    external_runner_log_files: List[TextIO] = []
    if (
        config.curriculum.auto_launch_external_vgcbench
        and config.hardware.battle_backend != RUST_ENGINE_BACKEND
    ):
        server_ports = [
            config.hardware.showdown_start_port + i
            for i in range(config.hardware.num_servers)
        ]
        (
            external_runner_processes,
            external_runner_log_files,
        ) = launch_external_vgcbench_runners(config, server_ports)

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

    # Resolve the per-run directory. On resume, we continue writing into the
    # original run's directory (inferred from resume_from's parent) so ghosts
    # and exploiters keep accumulating in one place across resumes.
    if config.training.resume_from:
        resume_parent = os.path.dirname(os.path.abspath(config.training.resume_from))
        # If resuming from a ghost or exploiter file, walk up one level.
        if os.path.basename(resume_parent) in ("ghosts", "exploiters"):
            resume_parent = os.path.dirname(resume_parent)
        run_dir = resume_parent
    else:
        run_dir = os.path.join(config.training.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ghosts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "exploiters"), exist_ok=True)
    config.training.run_dir = run_dir
    logger.info("Run directory: %s", run_dir)

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
    opponent_pool = OpponentPool(
        main_model=agent,
        device=config.hardware.device,
        battle_format=config.curriculum.battle_format,
        bc_model_path=config.curriculum.bc_model_path,
        exploiter_models_dir=os.path.join(run_dir, "exploiters"),
        ghosts_dir=os.path.join(run_dir, "ghosts"),
        vgc_bench_checkpoint_path=config.curriculum.vgc_bench_checkpoint_path,
        max_ghosts=config.curriculum.max_ghosts,
        curriculum=resume_curriculum or config.curriculum.curriculum_weights,
    )

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
    #   3. Append it to a buffer; record win/loss with the opponent pool.
    #   4. Once buffer ≥ train_batch_size, collate it into one big padded batch
    #      and call learner.update(). This is one optimizer step.
    #   5. Log metrics, periodically: update reference model, snapshot to portfolio,
    #      save checkpoint, broadcast new weights to workers, maybe spawn an
    #      exploiter training subprocess.
    # ─────────────────────────────────────────────────────────────────────────
    trajectories = []  # Buffer to accumulate trajectories before training
    updates = start_step  # Current training step (may be >0 if resumed from checkpoint)
    total_battles = 0  # Track total number of battles completed across all workers
    total_received_trajectories = 0
    total_received_steps = 0
    total_learner_trajectories = 0
    total_learner_steps = 0
    last_update_time = time.time()

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
            # Workers push completed battle trajectories to the queue asynchronously
            # We collect them here until we have enough for a training batch
            try:
                traj = mp_traj_queue.get(timeout=1.0)

                trajectories.append(traj)
                total_battles += 1  # Each trajectory represents one completed battle
                total_received_trajectories += 1
                total_received_steps += len(traj["steps"])

                # Track win rate by opponent type. These rolling results are consumed by
                # OpponentPool.update_curriculum(). Smoothing/noise handling lives there.
                opponent_pool.record_battle_result(
                    opponent_type=traj["opponent_type"],
                    won=traj["won"],
                    battle_length=traj["battle_length"],
                    forfeited=traj["forfeited"],
                )
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
                total_learner_steps += learner_steps_this_update
                total_learner_trajectories += len(trajectories)

                # Execute one RNaD policy update (PPO + KL regularization vs reference)
                metrics = learner.update(batch)
                updates += 1

                # Calculate time since last update
                current_time = time.time()
                time_per_update = current_time - last_update_time
                last_update_time = current_time

                # ===== LOG TRAINING METRICS =====
                total_time = time.time() - start
                metrics["update_step"] = updates
                metrics["total_battles"] = total_battles
                metrics["battles_per_second"] = total_battles / total_time
                metrics["received_trajectories_per_second"] = (
                    total_received_trajectories / total_time
                )
                metrics["received_steps_per_second"] = total_received_steps / total_time
                metrics["learner_trajectories_per_second"] = (
                    total_learner_trajectories / total_time
                )
                metrics["learner_steps_per_second"] = total_learner_steps / total_time
                metrics["learner_steps_this_update"] = learner_steps_this_update
                metrics["time_per_update_seconds"] = time_per_update
                metrics["temperature"] = config.temperature_at_step(updates)
                metrics.update(opponent_pool.get_training_metrics())

                # Build win rate string for console output
                win_rate_stats = opponent_pool.get_win_rate_stats()
                win_rate_str = " | ".join(
                    [
                        f"{opp_type}: {wr * 100:.1f}%"
                        for opp_type, wr in win_rate_stats.items()
                        if len(opponent_pool.win_rate_tracking.get(opp_type, [])) > 0
                    ]
                )
                if win_rate_str:
                    win_rate_str = f" | Win rates: {win_rate_str}"

                logger.info(
                    "Update %d: Loss=%.4f, Policy=%.4f, Value=%.4f, RNaD=%.4f | "
                    "Total Battles=%d in %s (%.2f b/s) | Learner Steps=%d (%.2f steps/s) | Learner Trajectories=%d (%.2f traj/s)%s",
                    updates,
                    metrics["loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["rnad_loss"],
                    total_battles,
                    format_time(total_time),
                    total_battles / total_time,
                    total_learner_steps,
                    total_learner_steps / total_time,
                    total_learner_trajectories,
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
                    learner.add_reference_model(
                        RNaDAgent(copy.deepcopy(agent.model))
                    )  # Snapshot current policy

                # ===== SAVE CHECKPOINT =====
                # Periodically save model, optimizer state, and training progress
                # NOTE: this saves ghosts in an unbounded way as training goes
                # on; this happens with all the models saved below
                if updates % config.training.checkpoint_interval == 0:
                    logger.info(
                        "[Update %d] Saving checkpoint and updating curriculum...", updates
                    )

                    # Save model, for safety and to use to battle against
                    ghost_checkpoint_path = save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        os.path.join(str(config.training.run_dir), "ghosts"),
                    )

                    # Add checkpoint to ghosts pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_ghost(updates, ghost_checkpoint_path)

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
                    # Note: state_dict is moved to CPU before pickling. Workers
                    # are CPU-only so this both fits their device and is the
                    # only way to transfer GPU tensors via mp.Queue.
                    logger.info(
                        "[Update %d] Broadcasting weights to worker processes...", updates
                    )
                    cpu_weights = {k: v.cpu() for k, v in agent.model.state_dict().items()}
                    update_payload = {
                        "weights": cpu_weights,
                        "curriculum": opponent_pool.curriculum.copy(),
                        "temperature": config.temperature_at_step(updates),
                        "top_p": config.exploration.top_p,
                        # Option C: broadcast explicit paths so workers don't scan directories
                        "exploiter_paths": [p for _, p in opponent_pool.exploiter_models],
                        "ghost_paths": [p for _, p in opponent_pool.ghosts],
                    }
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

                # ===== TRAIN EXPLOITER (FIND WEAKNESSES) =====
                # Periodically train a new exploiter agent to beat current policy
                # Exploiters are added to opponent pool to patch discovered weaknesses
                if (
                    config.training.train_exploiters
                    and updates % config.training.exploiter_interval == 0
                ):
                    # Save current policy as "victim" for exploiter to train against
                    victim_path = os.path.join(
                        str(config.training.run_dir), f"victim_step_{updates}.pt"
                    )
                    save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        str(config.training.run_dir),
                    )

                    # Launch exploiter training in subprocess (runs independently)
                    try:
                        train_exploiter_subprocess(victim_path, config)
                    except Exception as e:
                        logger.warning("Exploiter training failed: %s", e)

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
            learner.optimizer,
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
        shutdown_external_vgcbench_runners(
            external_runner_processes,
            external_runner_log_files,
        )
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization or mp.Process creation
    mp.set_start_method("spawn", force=True)

    # Configure platform-specific multiprocessing behavior
    configure_torch_multiprocessing(use_file_system_sharing=True)
    suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

    # Configure root logger so our logger.info() calls are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
