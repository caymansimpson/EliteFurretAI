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
                 │ (mp_traj_queue)      (control_queues)│
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
from datetime import datetime
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, Optional, Tuple

import torch

import wandb
from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.system_utils import (
    configure_torch_multiprocessing,
    suppress_third_party_warnings,
)
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.exploiters import (
    ExploiterPipelineState,
    initialize_exploiter_pipeline,
    mask_curriculum_during_warmup,
    maybe_run_exploiter_update,
)
from elitefurretai.rl.learners import (
    PortfolioRNaDLearner,
    build_model_from_config,
    load_checkpoint,
    save_checkpoint,
)
from elitefurretai.rl.model_registry import ModelRegistry
from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.rl_utils import (
    collate_trajectories,
    setup_logging,
    start_memory_watchdog,
)
from elitefurretai.rl.rnad_model import RNaDModel
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
    config: RNaDConfig, agent: RNaDModel, base_model: Any
) -> PortfolioRNaDLearner:
    """Create the portfolio learner with one initial reference snapshot."""
    ref_model = copy.deepcopy(base_model)
    ref_agent = RNaDModel(ref_model)
    return PortfolioRNaDLearner(agent, [ref_agent], config=config)


def initialize_training_state(
    config: RNaDConfig,
) -> Tuple[
    RNaDModel,
    PortfolioRNaDLearner,
    Dict[str, Any],
    Optional[Dict[str, float]],
]:
    """Initialize model/learner and worker bootstrap state in one place.
    Handles resuming from another run, initializing off of a BC model
    or doing a fresh init.

    The outer-loop `updates` counter always starts at 0 each run; learner
    internals (LR scheduler, rnad_alpha schedule) restore from the
    checkpoint via `learner.load_resume_state`, so schedules stay
    continuous even though the displayed step resets.

    Returns:
        agent: Main training agent
        learner: Initialized learner
        worker_model_config: Model config dict for worker model construction
        resume_curriculum: Curriculum restored from checkpoint (if any)
    """
    cfg = config.to_dict()
    resume_curriculum: Optional[Dict[str, float]] = None

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
        # random anchor on every resume
        checkpoint = load_checkpoint(config.training.resume_from, config.hardware.device)
        base_model = build_model_from_config(cfg, embedder, config.hardware.device, None)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        agent = RNaDModel(base_model)
        learner = initialize_learner(config, agent, base_model)
        learner.load_resume_state(checkpoint)
        old_config = RNaDConfig.from_dict(checkpoint["config"])
        if old_config.curriculum.curriculum_weights:
            resume_curriculum = old_config.curriculum.curriculum_weights

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
        agent = RNaDModel(base_model)
        learner = initialize_learner(config, agent, base_model)

        worker_model_config = cfg
    else:
        logger.info("Initializing fresh model from config...")
        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=config.training.embedder_feature_set,
            omniscient=False,
        )
        base_model = build_model_from_config(cfg, embedder, config.hardware.device, None)
        agent = RNaDModel(base_model)
        learner = initialize_learner(config, agent, base_model)

        worker_model_config = cfg

    return (
        agent,
        learner,
        worker_model_config,
        resume_curriculum,
    )


def setup_run_directory(config: RNaDConfig, run_name: str) -> str:
    """Create <save_dir>/<run_name>/{ghosts,exploiters} and copy resume snapshots.

    On resume, the source run's ghosts/ and exploiters/ contents are copied
    into the new dir so the curriculum keeps continuity — new snapshots
    accumulate alongside the copies, leaving the source untouched.

    Returns the absolute run_dir path; also sets config.training.run_dir
    as a side effect since downstream code reads it from there.
    """
    run_dir = os.path.join(config.training.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ghosts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "exploiters"), exist_ok=True)
    config.training.run_dir = run_dir
    logger.info("Run directory: %s", run_dir)

    if not config.training.resume_from:
        return run_dir

    resume_parent = os.path.dirname(os.path.abspath(config.training.resume_from))
    # If resuming from a ghost or exploiter file, walk up one level.
    if os.path.basename(resume_parent) in ("ghosts", "exploiters"):
        resume_parent = os.path.dirname(resume_parent)
    if os.path.abspath(resume_parent) == os.path.abspath(run_dir):
        logger.info(
            "Resume copy: source matches new run_dir (%s) — no copy needed", run_dir
        )
        return run_dir

    for subdir in ("ghosts", "exploiters"):
        src_dir = os.path.join(resume_parent, subdir)
        dst_dir = os.path.join(run_dir, subdir)
        if not os.path.isdir(src_dir):
            logger.info("Resume copy: %s not present in source run, skipping", src_dir)
            continue
        files = sorted(
            f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))
        )
        if not files:
            logger.info("Resume copy: %s is empty, nothing to copy", src_dir)
            continue
        total_bytes = 0
        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            total_bytes += os.path.getsize(src_path)
            shutil.copy2(src_path, dst_path)
        logger.info(
            "Resume copy: %d file(s), %.1f MB total copied into %s",
            len(files),
            total_bytes / 1e6,
            dst_dir,
        )
    return run_dir


def setup_model_registry(
    config: RNaDConfig,
    agent: RNaDModel,
    worker_model_config: Dict[str, Any],
    opponent_pool: OpponentPool,
) -> ModelRegistry:
    """Build and start the ModelRegistry with all inference services.

    Trainer process owns a ModelRegistry of InferenceServices that workers
    submit inference requests to. Each service holds an independent copy
    of its model so the learner can update without racing with inference;
    weights sync at broadcast cadence via registry.sync_weights.

    Registers:
      - `main` always (initialized from `agent`).
      - `bc` in the "frozen" subprocess if a BC checkpoint is configured
        AND its curriculum weight > 0. BC never re-syncs after startup,
        so subprocess co-location costs nothing and frees a GIL contender.
      - `exploiter` (in-process) + `victim` (frozen subprocess) if
        `train_exploiter` curriculum weight > 0. Exploiter must stay
        in-process because its sync rate is too high for IPC; victim's
        rare refresh fits the subprocess path comfortably.
      - `ghost_{0..max_ghosts}` and `exploiter_snap_{0..max_exploiter_models}`
        slots in the "frozen" subprocess. Slots are fixed-size; LRU swaps
        happen mid-run via sync_weights, never re-registration. Weights
        are loaded from disk for any snapshots OpponentPool pre-assigned
        at startup.

    Calls `registry.start_all()` before returning so the services are
    ready to accept inference requests from workers.
    """
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

    # Build a CPU-or-GPU copy of the main model for inference, kept
    # independent of the learner's model so backward+forward don't race.
    main_inference_base = build_model_from_config(
        worker_model_config, inference_embedder, inference_device, None, strict=False
    )
    main_inference_base.eval()
    main_inference_base.load_state_dict(agent.model.state_dict())
    registry.register("main", RNaDModel(main_inference_base))

    bc_checkpoint_path = config.curriculum.bc_model_path
    bc_curriculum_weight = config.curriculum.curriculum_weights.get(
        OpponentPool.BC_PLAYER, 0.0
    )
    if bc_checkpoint_path and bc_curriculum_weight > 0:
        logger.info(
            "Registering BC model from %s (curriculum weight %.3f)",
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
        bc_inference_base.cpu()
        registry.register(
            "bc",
            RNaDModel(bc_inference_base),
            compile=True,
            process_group="frozen",
        )
        del bc_checkpoint

    exploiter_curriculum_weight = config.curriculum.curriculum_weights.get(
        OpponentPool.TRAIN_EXPLOITER, 0.0
    )
    if exploiter_curriculum_weight > 0:
        logger.info(
            "Registering exploiter (in-process) + victim (frozen subprocess) "
            "(curriculum weight %.3f)",
            exploiter_curriculum_weight,
        )
        exploiter_base = build_model_from_config(
            worker_model_config, inference_embedder, inference_device, None, strict=False
        )
        exploiter_base.eval()
        # Exploiter starts with whatever build_model_from_config gave us
        # (BC init if `initialize_path` is set, else fresh). It'll be
        # sync'd as the exploiter learner produces updates.
        registry.register("exploiter", RNaDModel(exploiter_base), compile=True)

        victim_base = build_model_from_config(
            worker_model_config, inference_embedder, "cpu", None, strict=False
        )
        victim_base.eval()
        # Victim starts as a copy of main; weights refresh periodically
        # via registry.sync_weights("victim", ...).
        victim_base.load_state_dict(agent.model.state_dict())
        for param in victim_base.parameters():
            param.requires_grad = False
        registry.register(
            "victim",
            RNaDModel(victim_base),
            compile=True,
            process_group="frozen",
        )

    # Ghost + exploiter-snapshot slot pools: fixed-size at registration,
    # LRU-swapped mid-run via sync_weights. Both live in the "frozen"
    # subprocess (low sync rate). Workers route to a slot only if it's
    # in opponent_pool.active_*_slots().
    registry.register_snapshot_slots(
        "ghost_", config.curriculum.max_ghosts, source_name="main"
    )
    for path, slot in opponent_pool.slot_for_ghost_path.items():
        checkpoint = torch.load(path, map_location="cpu")
        registry.sync_weights(f"ghost_{slot}", checkpoint["model_state_dict"])

    registry.register_snapshot_slots(
        "exploiter_snap_", config.curriculum.max_exploiter_models, source_name="main"
    )
    for path, slot in opponent_pool.slot_for_exploiter_path.items():
        checkpoint = torch.load(path, map_location="cpu")
        registry.sync_weights(f"exploiter_snap_{slot}", checkpoint["model_state_dict"])

    # Start all services before workers spawn — workers send requests
    # immediately after spawn.
    registry.start_all()
    return registry


def broadcast_weights_to_workers(
    config: RNaDConfig,
    updates: int,
    agent: RNaDModel,
    exploiter_agent: Optional[RNaDModel],
    victim_agent: Optional[RNaDModel],
    opponent_pool: OpponentPool,
    registry: Optional[ModelRegistry],
    control_queues: List[MPQueue],
    exploiter_state: ExploiterPipelineState,
    exploiter_pipeline_on: bool,
) -> None:
    """Sync new weights into the inference services and broadcast a control payload.

    Two distinct things happen here at the same cadence:

      1. WEIGHTS — synced trainer-side via `registry.sync_weights(...)` for
         every registered service whose weights changed (main always; live
         exploiter if the pipeline is on; victim if a refresh tick fired
         since the last broadcast). This is how policy refreshes actually
         reach inference under centralized inference. Workers never see
         these weights — they call into the InferenceService via mp.Queue.

      2. CONTROL PAYLOAD — a small dict (curriculum, sampling knobs,
         active ghost/exploiter slot lists) shipped to each worker's
         control_queue. Bundling these into one dict guarantees workers
         transition together rather than racing between curriculum and
         sampling updates.

    The function name is historical: under centralized inference the
    control payload no longer carries weights (workers don't hold a policy).
    It still owns the "do everything that needs to happen at broadcast
    cadence" responsibility, including the registry weight syncs.

    Curriculum is masked during warmup so workers don't sample
    train_exploiter battles before the exploiter learner has seen any
    updates (see `mask_curriculum_during_warmup`).

    State_dicts are moved to CPU before handing to `registry.sync_weights`
    so subprocess-hosted services can pickle them across the control queue.

    Side effects:
      - `registry.sync_weights` for main / exploiter / victim as applicable.
      - Clears `exploiter_state.victim_needs_broadcast` after a victim sync
        (so each refresh is processed exactly once).
      - Drain-then-put on each worker's control_queue (keep-at-most-latest).
    """
    logger.info("[Update %d] Broadcasting weights + control payload...", updates)

    # ── Weight syncs into the trainer-side InferenceServices ───────────
    # The service copy MUST be kept current or workers play increasingly
    # off-policy. Moves to CPU first so subprocess services can receive
    # the state_dict across their control queue.
    cpu_weights = {k: v.cpu() for k, v in agent.model.state_dict().items()}
    if registry is not None and "main" in registry.names():
        registry.sync_weights("main", cpu_weights)

    if exploiter_pipeline_on and exploiter_agent is not None:
        exploiter_cpu = {k: v.cpu() for k, v in exploiter_agent.model.state_dict().items()}
        if registry is not None and "exploiter" in registry.names():
            registry.sync_weights("exploiter", exploiter_cpu)
    if (
        exploiter_pipeline_on
        and victim_agent is not None
        and exploiter_state.victim_needs_broadcast
    ):
        victim_cpu = {k: v.cpu() for k, v in victim_agent.model.state_dict().items()}
        if registry is not None and "victim" in registry.names():
            registry.sync_weights("victim", victim_cpu)
        # Cleared after the sync queues: each refresh is processed exactly
        # once, then we wait for the next refresh tick.
        exploiter_state.victim_needs_broadcast = False

    # ── Control payload broadcast to each worker ───────────────────────
    in_warmup = updates < config.exploiter.warmup_updates
    control_payload: Dict[str, Any] = {
        "curriculum": mask_curriculum_during_warmup(opponent_pool.curriculum, in_warmup),
        "temperature": config.temperature_at_step(updates),
        "top_p": config.exploration.top_p,
        "active_ghost_slots": sorted(opponent_pool.active_ghost_slots()),
        "active_exploiter_slots": sorted(opponent_pool.active_exploiter_slots()),
    }
    for i, q in enumerate(control_queues):
        try:
            # Keep-at-most-latest semantics prevents workers from replaying
            # stale control snapshots when learner is faster than consumers.
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
            q.put_nowait(control_payload)
        except Exception as e:
            logger.warning("Failed to broadcast to worker %d: %s", i, e)


def get_dead_workers(
    processes: List[mp.Process], error_queue: MPQueue
) -> List[mp.Process]:
    dead_procs = [p for p in processes if not p.is_alive()]

    if len(dead_procs) == 0:
        return []

    logger.error("=" * 60)
    logger.error("%d worker process(es) died:", len(dead_procs))
    for p in dead_procs:
        exit_reason = {
            None: "still running (race condition?)",
            0: "normal exit",
            1: "general error",
            -9: "SIGKILL (out of memory?)",
            -11: "SIGSEGV (segmentation fault)",
            -15: "SIGTERM (terminated)",
        }.get(p.exitcode, f"exit code {p.exitcode}")
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

    # Memory watchdog: request graceful shutdown if we approach dangerous
    # memory territory so we get a clean checkpoint before we break
    start_memory_watchdog(shutdown_requested, config.training.memory_watchdog_threshold_gb)

    server_processes: List[subprocess.Popen] = launch_showdown_servers(
        config.hardware.num_servers, config.hardware.showdown_start_port
    )

    # Auto-launch external vgc-bench runners based on curriculum
    vgcbench_manager: Optional[VGCBenchManager] = None
    if config.curriculum.curriculum_weights.get(OpponentPool.VGC_BENCH_BASELINE, 0.0) > 0:
        server_ports = [
            config.hardware.showdown_start_port + i
            for i in range(config.hardware.num_servers)
        ]
        vgcbench_manager = VGCBenchManager(config, server_ports)
        vgcbench_manager.launch()

    # Generate unique run ID to avoid stale-account collisions on Showdown server.
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

    run_dir = setup_run_directory(config, run_name)

    # Initialize model and learner given the config (handles fresh start, resume, and weight initialization cases)
    (
        agent,
        learner,
        worker_model_config,
        resume_curriculum,
    ) = initialize_training_state(config)

    # Create opponent pool
    logger.info("Initializing opponent pool...")
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
        bc_model_path=config.curriculum.bc_model_path,
        exploiter_models_dir=os.path.join(run_dir, "exploiters"),
        ghosts_dir=os.path.join(run_dir, "ghosts"),
        max_ghosts=config.curriculum.max_ghosts,
        max_exploiter_models=config.curriculum.max_exploiter_models,
        curriculum=active_curriculum,
    )

    # ── Initialize the exploiter co-training pipeline ──
    # All four returned values are None when train_exploiter == 0 (default).
    (
        exploiter_agent,
        victim_agent,
        exploiter_learner,
        bc_state_dict,
    ) = initialize_exploiter_pipeline(config, agent, worker_model_config)
    exploiter_pipeline_on = exploiter_learner is not None

    start = time.time()

    # ==================== START WORKERS ====================
    # Each worker is a separate OS process running worker.py:mp_worker_process.
    # We give each worker:
    #   - a unique server_port for its Showdown connection
    #   - the model checkpoint path to bootstrap from (loaded fresh per worker)
    #   - shared mp queues for trajectories (in) and weight updates (out)
    #   - an mp.Event for graceful shutdown
    worker_ports, server_loads = allocate_server_ports(
        config.hardware.num_workers,
        config.hardware.players_per_worker,
        config.hardware.num_servers,
        config.hardware.max_players_per_server,
        config.hardware.showdown_start_port,
    )

    # ── Multiprocessing queues for inter-process communication ──────────────
    #   mp_traj_queue (workers → trainer): completed trajectory dicts.
    #   control_queues (trainer → each worker): trainer-to-worker CONTROL
    #     broadcasts — curriculum, sampling knobs (temperature, top_p),
    #     and active ghost/exploiter slot lists. One queue per worker so
    #     we can fan out without contending on a shared queue.
    #     NOTE: this is NOT how policy weights reach inference under
    #     centralized inference. Workers don't hold a policy; the trainer
    #     syncs weights directly into trainer-side InferenceServices via
    #     `registry.sync_weights(...)`. These queues only carry the
    #     control plane that workers use to choose opponents and sample.
    #   mp_error_queue: workers push tracebacks here on crash so the trainer
    #     can surface them in logs before exiting.
    #   mp_stop_event: trainer sets this on shutdown to ask workers to exit.
    # ────────────────────────────────────────────────────────────────────────
    mp_traj_queue: MPQueue = MPQueue(maxsize=1024)
    control_queues: List[MPQueue] = [
        MPQueue(maxsize=2) for _ in range(config.hardware.num_workers)
    ]
    mp_error_queue: MPQueue = MPQueue(maxsize=100)
    mp_stop_event: MPEvent = mp.Event()

    # Initialize model registry, which holds all models in GPU
    registry = setup_model_registry(config, agent, worker_model_config, opponent_pool)

    logger.info(
        "ModelRegistry initialized; %d models registered (%s). "
        "device=%s batch_size=%d batch_timeout=%.4f compile=%s",
        len(registry.names()),
        ", ".join(registry.names()),
        config.hardware.device,
        config.hardware.batch_size,
        config.hardware.batch_timeout,
        config.hardware.compile_inference_model,
    )

    # Per-worker bundle of (request_q, response_q) pairs keyed by model
    # name. Each worker gets its slice of the response queues; the
    # request queue is shared across workers per model. Workers wrap
    # their slice in WorkerInferenceClients on the worker side.
    queues_by_worker: List[Optional[Dict[str, Any]]] = [None] * config.hardware.num_workers
    for w in range(config.hardware.num_workers):
        queues_by_worker[w] = {
            name: (req_q, resp_qs[w])
            for name, (req_q, resp_qs) in registry.queues_for_workers().items()
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
                worker_model_config,
                mp_traj_queue,
                control_queues[i],
                mp_error_queue,
                mp_stop_event,
                run_id,
                config,
                queues_by_worker[i],
                sorted(opponent_pool.active_ghost_slots()),
                sorted(opponent_pool.active_exploiter_slots()),
            ),
            daemon=True,
            name=f"Worker-{i}",
        )
        p.start()
        processes.append(p)
        logger.info(
            "Started multiprocessing worker %d (PID: %s) on port %d", i, p.pid, server_port
        )

    # Print state of workers, players and servers
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
    updates = 0  # Main update step; resets to 0 each run (learner schedules keep continuity via load_resume_state)
    total_battles = 0  # Total battles completed across all workers
    total_received_trajectories = 0
    total_received_steps = 0
    total_learner_trajectories = 0
    total_learner_steps = 0
    last_update_time = time.time()
    prev_total_battles = 0  # snapshot at last log for recent b/s

    # Exploiter pipeline loop-state (only meaningful when the pipeline is on).
    exploiter_state = ExploiterPipelineState.new(config)

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
            # (see WorkerOpponentFactory.sample_opp_type_for / apply_opp_type_to_pair).
            # The tag tells
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
                    exploiter_state.trajectories.append(traj)
                    exploiter_state.win_buffer.append(1.0 if traj["won"] else 0.0)
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
                        RNaDModel(copy.deepcopy(agent.model))
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

                    # Save model, for safety and to use to battle against
                    ghost_checkpoint_path = save_checkpoint(
                        agent,
                        learner,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        os.path.join(str(config.training.run_dir), "ghosts"),
                    )

                    # Add checkpoint to ghosts pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_ghost(ghost_checkpoint_path)
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

                    # Broadcast all new model weights to workers
                    broadcast_weights_to_workers(
                        config=config,
                        updates=updates,
                        agent=agent,
                        exploiter_agent=exploiter_agent,
                        victim_agent=victim_agent,
                        opponent_pool=opponent_pool,
                        registry=registry,
                        control_queues=control_queues,
                        exploiter_state=exploiter_state,
                        exploiter_pipeline_on=exploiter_pipeline_on,
                    )

                # ===== VICTIM REFRESH =====
                # The exploiter trains against a stationary target (the
                # victim) so its gradient signal is stable. But if we never
                # refreshed, the exploiter would eventually master a stale
                # main and graduate snapshots irrelevant to the *current*
                # main. So every `victim_refresh_interval` main updates,
                # copy main's current state_dict into the victim and flag
                # the next broadcast to push victim weights into the
                # registry's victim InferenceService.
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
                    exploiter_state.victim_needs_broadcast = True
                    logger.info("[Update %d] Refreshed victim from current main", updates)

                # ===== EXPLOITER LEARNER UPDATE + GRADUATION =====
                # All exploiter-side logic lives in `maybe_run_exploiter_update`:
                # gating on warmup/buffer-fill, the learner step, win-rate
                # graduation, snapshot save, and BC re-init. Helper mutates
                # `exploiter_state` in place — both the buffers and the counters
                # (including `victim_needs_broadcast` on graduation, which the
                # broadcast block above will clear after queueing).
                maybe_run_exploiter_update(
                    config=config,
                    updates=updates,
                    agent=agent,
                    exploiter_learner=exploiter_learner,
                    exploiter_agent=exploiter_agent,
                    victim_agent=victim_agent,
                    bc_state_dict=bc_state_dict,
                    opponent_pool=opponent_pool,
                    state=exploiter_state,
                    registry=registry,
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

    setup_logging()

    main()
