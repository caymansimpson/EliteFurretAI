"""
Enhanced RNaD training with:
- Config-driven training
- Resume from checkpoint
- Integrated exploiter training
- Team pool randomization
- Comprehensive wandb tracking
- Option 1: True multiprocessing (separate processes, bypasses GIL)
- Option 2a: Per-worker executors (threading with reduced contention)
"""

import argparse
import asyncio
import copy
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import psutil
import torch
from poke_env import ServerConfiguration

import wandb
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.etl.system_utils import (
    configure_torch_multiprocessing,
    suppress_third_party_warnings,
)
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import PortfolioRNaDLearner, RNaDLearner
from elitefurretai.rl.model_io import (
    build_model_from_config,
    load_checkpoint,
    save_checkpoint,
)
from elitefurretai.rl.opponents import (
    OpponentPool,
    WorkerOpponentFactory,
)
from elitefurretai.rl.players import RNaDAgent, cleanup_worker_executors
from elitefurretai.rl.server_manager import (
    allocate_server_ports,
    derive_external_vgcbench_username,
    launch_external_vgcbench_runners,
    launch_showdown_servers,
    shutdown_external_vgcbench_runners,
    shutdown_showdown_servers,
)
from elitefurretai.supervised import format_time
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


def initialize_learner(
    config: RNaDConfig,
    agent: RNaDAgent,
    base_model: FlexibleThreeHeadedModel,
) -> Union[RNaDLearner, PortfolioRNaDLearner]:
    """Create either standard or portfolio learner using shared config knobs."""
    if config.use_portfolio_regularization:
        ref_model = copy.deepcopy(base_model)
        ref_agent = RNaDAgent(ref_model)
        return PortfolioRNaDLearner(
            agent,
            [ref_agent],
            lr=config.lr,
            rnad_alpha=config.rnad_alpha,
            gradient_clip=config.max_grad_norm,
            device=config.device,
            gamma=config.gamma,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            use_mixed_precision=config.use_mixed_precision,
            max_portfolio_size=config.max_portfolio_size,
            portfolio_update_strategy=config.portfolio_update_strategy,
        )

    ref_model = copy.deepcopy(base_model)
    ref_agent = RNaDAgent(ref_model)
    return RNaDLearner(
        agent,
        ref_agent,
        lr=config.lr,
        rnad_alpha=config.rnad_alpha,
        device=config.device,
        gamma=config.gamma,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        use_mixed_precision=config.use_mixed_precision,
        gradient_clip=config.max_grad_norm,
    )


def resolve_worker_model_source(config: RNaDConfig, agent: RNaDAgent) -> str:
    """Persist a bootstrap checkpoint for worker startup and return its path."""
    os.makedirs(config.save_dir, exist_ok=True)
    bootstrap_path = os.path.join(config.save_dir, "worker_bootstrap_initial.pt")
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "config": config.to_dict(),
            "step": 0,
            "timestamp": datetime.now().isoformat(),
        },
        bootstrap_path,
    )
    print(f"Saved worker bootstrap checkpoint to {bootstrap_path}")
    return bootstrap_path


def initialize_training_state(
    config: RNaDConfig,
) -> Tuple[
    RNaDAgent,
    Union[RNaDLearner, PortfolioRNaDLearner],
    str,
    Dict[str, Any],
    int,
    Optional[Dict[str, float]],
]:
    """Initialize model/learner and worker bootstrap state in one place.

    Returns:
        agent: Main training agent
        learner: Initialized learner (standard or portfolio)
        worker_model_path: Checkpoint path workers should bootstrap from
        worker_model_config: Model config dict for worker model construction
        start_step: Starting update step (restored for resume)
        resume_curriculum: Curriculum restored from checkpoint (if any)
    """
    cfg = config.to_dict()
    resume_curriculum: Optional[Dict[str, float]] = None
    start_step = 0

    if config.resume_from:
        print(f"Resuming model+optimizer from {config.resume_from}...")
        embedder = Embedder(
            format=config.battle_format,
            feature_set=cfg["embedder_feature_set"],
            omniscient=False,
        )
        base_model = build_model_from_config(cfg, embedder, config.device, None)
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)

        start_step, old_config = load_checkpoint(
            config.resume_from, agent, learner.optimizer, config.device
        )
        if old_config.curriculum:
            resume_curriculum = old_config.curriculum

        worker_model_path = config.resume_from
        worker_model_config = old_config.to_dict()
    elif config.initialize_path:
        print(f"Initializing model weights from {config.initialize_path}...")
        init_checkpoint = torch.load(
            config.initialize_path,
            map_location=config.device,
            weights_only=False,
        )
        checkpoint_cfg = init_checkpoint.get("config", cfg)
        embedder = Embedder(
            format=config.battle_format,
            feature_set=checkpoint_cfg.get("embedder_feature_set", cfg["embedder_feature_set"]),
            omniscient=False,
        )
        base_model = build_model_from_config(
            checkpoint_cfg,
            embedder,
            config.device,
            init_checkpoint["model_state_dict"],
        )
        agent = RNaDAgent(base_model)
        learner = initialize_learner(config, agent, base_model)

        worker_model_path = config.initialize_path
        worker_model_config = checkpoint_cfg
    else:
        print("Initializing fresh model from config...")
        embedder = Embedder(
            format=config.battle_format,
            feature_set=cfg["embedder_feature_set"],
            omniscient=False,
        )
        base_model = build_model_from_config(cfg, embedder, config.device, None)
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


def get_dead_workers(processes: List[mp.Process], error_queue: MPQueue, verbose: bool = True) -> List[mp.Process]:
    dead_procs = [p for p in processes if not p.is_alive()]

    if verbose and len(dead_procs) > 0:
        print(f"\n{'='*60}")
        print(f"ERROR: {len(dead_procs)} worker process(es) died:")
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
            print(f"  - {p.name} (PID: {p.pid}): {exit_reason}")

        # Check error queue for detailed error messages from workers
        print("\nChecking for error reports from workers...")
        error_found = False
        while True:
            try:
                error_info = error_queue.get_nowait()
                error_found = True
                print(f"\n--- Error from Worker {error_info['worker_id']} ---")
                print(f"Exception: {error_info['error']}")
                print(f"Traceback:\n{error_info['traceback']}")
            except Exception:
                break
        if not error_found:
            print("No error reports in queue (worker may have crashed before reporting)")

        print("\nTraining cannot continue. Saving checkpoint and exiting...")
        print(f"{'='*60}\n")

    return dead_procs


def mp_worker_process(
    worker_id: int,
    server_port: int,
    model_path: str,
    model_config: Dict[str, Any],
    traj_queue: MPQueue,
    weight_queue: MPQueue,
    error_queue: MPQueue,
    stop_event: MPEvent,
    run_id: str,
    config: RNaDConfig,
    verbose: bool = False,
):
    """
    Multiprocessing worker process for true parallel RL data collection.

    Each worker process has:
    - Its own Python GIL (true parallelism)
    - Its own model copy (no contention)
    - Its own GPU memory allocation
    - Its own asyncio event loop

    Args:
        worker_id: Unique worker identifier
        server_port: Showdown server port
        model_path: Path to model checkpoint
        model_config: Model configuration dict
        traj_queue: Multiprocessing queue for sending trajectories to learner
        weight_queue: Multiprocessing queue for receiving weight updates
        error_queue: Multiprocessing queue for reporting errors to main process
        stop_event: Multiprocessing event to signal shutdown
        run_id: Unique run identifier
        config: Worker/runtime configuration
        verbose: Whether to print detailed logs
    """
    def get_memory_usage_mb():
        """Get current process memory usage as a human-readable string (KB, MB, GB)."""
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        if mem_bytes < 1024 * 1024:
            return f"{mem_bytes / 1024:.0f}KB"
        elif mem_bytes < 1024 * 1024 * 1024:
            return f"{mem_bytes / (1024 * 1024):.0f}MB"
        else:
            return f"{mem_bytes / (1024 * 1024 * 1024):.0f}GB"

    try:
        suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

        battle_format = config.battle_format
        team_pool_path = config.base_team_path
        team_subdirectory = config.team_pool_path
        num_players = config.players_per_worker
        device = config.device
        batch_size = config.batch_size
        batch_timeout = config.batch_timeout
        max_battle_steps = config.max_battle_steps
        num_battles_per_pair = config.num_battles_per_pair
        curriculum = config.curriculum
        bc_model_path = config.bc_action_path
        exploiter_models_dir = config.exploiter_models_dir
        max_exploiter_models = config.max_exploiter_models
        max_loaded_exploiter_models = config.max_loaded_exploiter_models
        max_past_models = config.max_past_models
        past_models_dir = config.past_models_dir
        vgc_bench_checkpoint_path = config.vgc_bench_checkpoint_path
        external_vgcbench_usernames = config.external_vgcbench_usernames
        external_vgcbench_startup_wait_s = config.external_vgcbench_startup_wait_s
        max_loaded_ghost_models = config.max_loaded_ghost_models

        if external_vgcbench_usernames and config.auto_launch_external_vgcbench:
            external_vgcbench_usernames = [
                derive_external_vgcbench_username(username, server_port)
                for username in external_vgcbench_usernames
            ]

        if verbose:
            print(f"[MPWorker {worker_id}] Starting (PID: {os.getpid()})... Initial memory: {get_memory_usage_mb()}", flush=True)

        # Set thread count to avoid contention within process
        torch.set_num_threads(2)

        # IMPORTANT: Load model to CPU first, then move to device
        # This avoids CUDA re-initialization issues with forked processes
        if verbose:
            print(f"[MPWorker {worker_id}] Loading model from {model_path}... Memory: {get_memory_usage_mb()}", flush=True)
        # Load checkpoint and extract config
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"[MPWorker {worker_id}] Checkpoint loaded... Memory: {get_memory_usage_mb()}", flush=True)

        # Create embedder from checkpoint config
        embedder_feature_set = checkpoint["config"].get("embedder_feature_set", "full")
        embedder = Embedder(
            format=battle_format,
            feature_set=embedder_feature_set,
            omniscient=False,
        )

        # Build main model using consolidated builder
        if verbose:
            print(f"[MPWorker {worker_id}] Building model on device={device}... Memory: {get_memory_usage_mb()}", flush=True)
        model = build_model_from_config(
            model_config, embedder, device, checkpoint["model_state_dict"]
        )
        model.eval()
        agent = RNaDAgent(model)
        del checkpoint  # Free memory
        if verbose:
            print(f"[MPWorker {worker_id}] Model built... Memory: {get_memory_usage_mb()}", flush=True)

        # Create BC agent if needed (frozen, never receives weight updates)
        bc_agent = None
        if curriculum and curriculum.get("bc_player", 0) > 0 and bc_model_path:
            if verbose:
                print(f"[MPWorker {worker_id}] Loading BC model... Memory: {get_memory_usage_mb()}", flush=True)
            bc_checkpoint = torch.load(bc_model_path, map_location="cpu", weights_only=False)
            bc_model = build_model_from_config(
                model_config, embedder, device, bc_checkpoint["model_state_dict"]
            )
            bc_model.eval()
            for param in bc_model.parameters():
                param.requires_grad = False
            bc_agent = RNaDAgent(bc_model)
            del bc_checkpoint
            if verbose:
                print(f"[MPWorker {worker_id}] BC model loaded and frozen... Memory: {get_memory_usage_mb()}", flush=True)

        # Load team repo
        if verbose:
            print(f"[MPWorker {worker_id}] Loading teams from {team_pool_path}... Memory: {get_memory_usage_mb()}", flush=True)
        team_repo = TeamRepo(team_pool_path)

        # Set up asyncio event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server_config = ServerConfiguration(
            f"ws://localhost:{server_port}/showdown/websocket", ""
        )

        # Create trajectory queue for this worker
        local_traj_queue: queue.Queue = queue.Queue()

        # Use WorkerOpponentFactory to manage all opponent creation and sampling
        if verbose:
            print(f"[MPWorker {worker_id}] Creating opponent factory... Memory: {get_memory_usage_mb()}", flush=True)
        factory = WorkerOpponentFactory(
            team_repo=team_repo,
            battle_format=battle_format,
            team_subdirectory=team_subdirectory,
            server_config=server_config,
            main_agent=agent,
            bc_agent=bc_agent,
            curriculum=curriculum,
            embedder=embedder,
            worker_id=worker_id,
            run_id=run_id,
            device=device,
            batch_size=batch_size,
            batch_timeout=batch_timeout,
            max_battle_steps=max_battle_steps,
            exploiter_models_dir=exploiter_models_dir,
            max_exploiter_models=max_exploiter_models,
            max_loaded_exploiter_models=max_loaded_exploiter_models,
            ghost_models_dir=past_models_dir,
            max_ghost_models=max_past_models,
            max_loaded_ghost_models=max_loaded_ghost_models,
            vgc_bench_checkpoint_path=vgc_bench_checkpoint_path,
            external_vgcbench_usernames=external_vgcbench_usernames,
            model_config=model_config,
        )

        # Create all worker-local agents via factory
        num_pairs = num_players // 2
        factory.create_agents(num_pairs, local_traj_queue)
        if verbose:
            print(f"[MPWorker {worker_id}] Created {num_pairs} player pairs... Memory: {get_memory_usage_mb()}", flush=True)

        # Start inference loops
        factory.start_inference_loops()
        if verbose:
            print(f"[MPWorker {worker_id}] Inference loops started... Memory: {get_memory_usage_mb()}", flush=True)

        async def run_battles(num_battles_per_pair: int):
            """Main battle loop with team randomization and curriculum-based opponent selection."""
            await asyncio.sleep(0.2)  # Let inference loops warm up
            if external_vgcbench_usernames:
                await asyncio.sleep(external_vgcbench_startup_wait_s)

            battle_batch = 0
            last_weight_check = time.time()

            while not stop_event.is_set():
                battle_batch += 1
                start = time.time()

                # Check for weight updates (from learner process)
                if time.time() - last_weight_check > 1.0:
                    try:
                        while not weight_queue.empty():
                            new_weights = weight_queue.get_nowait()
                            model.load_state_dict(new_weights)
                            if verbose:
                                print(f"[MPWorker {worker_id}] Updated weights")
                    except Exception:
                        pass
                    last_weight_check = time.time()

                # Prepare and execute curriculum-driven battles for this batch
                tasks, _batch_opponent_types = factory.prepare_batch_tasks(
                    num_battles_per_pair
                )
                await asyncio.gather(*tasks)

                # Transfer trajectories to main process
                total_time = time.time() - start
                battles_completed = sum(p.n_finished_battles for p in factory.players)
                transferred = 0
                while not local_traj_queue.empty():
                    try:
                        traj = local_traj_queue.get_nowait()
                        traj_queue.put(traj)
                        transferred += 1
                    except Exception:
                        break

                if verbose:
                    print(f"[MPWorker {worker_id}] Batch {battle_batch}: {battles_completed} battles in {total_time:.2f}s ({battles_completed / total_time:.2f} b/s) | Sent {transferred} trajectories... Memory: {get_memory_usage_mb()}", flush=True)

                # Clean up battle state for next batch
                factory.reset_all_battles()

        loop.run_until_complete(run_battles(num_battles_per_pair))
        if verbose:
            print(f"[MPWorker {worker_id}] Finished gracefully", flush=True)

    except Exception as e:
        import traceback
        error_msg = f"[MPWorker {worker_id}] FATAL ERROR: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        try:
            error_queue.put_nowait({"worker_id": worker_id, "error": str(e), "traceback": traceback.format_exc()})
        except Exception:
            pass  # Queue might be full or closed
        # Re-raise so exitcode is non-zero
        raise


def collate_trajectories(trajectories, device, gamma, gae_lambda, max_seq_len=40):
    """Collate list of trajectories into batched tensors with padding.

    Args:
        trajectories: List of trajectory dicts
        device: torch device
        max_seq_len: Maximum sequence length (matches model's positional embedding size)
                     Trajectories longer than this are truncated (keeping last N steps
                     since later game decisions are typically more impactful).

    Returns:
        batch_dict: Dict of tensors for training
        metadata: List of dicts with {"opponent_type": str, "won": bool} for each trajectory
    """
    # Extract metadata and steps from new trajectory format
    metadata = []
    raw_trajectories = []
    for traj in trajectories:
        if isinstance(traj, dict) and "steps" in traj:
            # New format: {"steps": [...], "opponent_type": str, "won": bool}
            metadata.append({
                "opponent_type": traj.get("opponent_type", "self_play"),
                "won": traj.get("won", False),
            })
            raw_trajectories.append(traj["steps"])
        else:
            # Legacy format: list of steps directly
            metadata.append({"opponent_type": "self_play", "won": False})
            raw_trajectories.append(traj)

    trajectories = raw_trajectories

    # Truncate long trajectories to max_seq_len
    truncated_trajectories = []
    for traj in trajectories:
        if len(traj) > max_seq_len:
            # Keep last max_seq_len steps (later decisions matter more)
            truncated_trajectories.append(traj[-max_seq_len:])
        else:
            truncated_trajectories.append(traj)
    trajectories = truncated_trajectories

    batch_size = len(trajectories)
    max_len = max(len(t) for t in trajectories)

    # Get dim from first state
    dim = len(trajectories[0][0]["state"])

    states = torch.zeros(batch_size, max_len, dim)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    rewards = torch.zeros(batch_size, max_len)
    log_probs = torch.zeros(batch_size, max_len)
    values = torch.zeros(batch_size, max_len)
    is_tp = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    processed_advantages = []
    processed_returns = []

    for i, traj in enumerate(trajectories):
        seq_len = len(traj)

        # Extract fields
        traj_states = torch.tensor(np.array([step["state"] for step in traj]))
        traj_actions = torch.tensor([step["action"] for step in traj])
        traj_rewards = torch.tensor([step["reward"] for step in traj])
        traj_log_probs = torch.tensor([step["log_prob"] for step in traj])
        traj_values = torch.tensor([step["value"] for step in traj])
        traj_is_tp = torch.tensor([step["is_teampreview"] for step in traj])

        states[i, :seq_len] = traj_states
        actions[i, :seq_len] = traj_actions
        rewards[i, :seq_len] = traj_rewards
        log_probs[i, :seq_len] = traj_log_probs
        values[i, :seq_len] = traj_values
        is_tp[i, :seq_len] = traj_is_tp
        padding_mask[i, :seq_len] = 1

        # Compute GAE
        advs = torch.zeros(seq_len)
        rets = torch.zeros(seq_len)
        last_gae_lam = torch.tensor(0.0)

        next_val = torch.tensor(0.0)

        for t in reversed(range(seq_len)):
            delta = traj_rewards[t] + gamma * next_val - traj_values[t]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            advs[t] = last_gae_lam
            rets[t] = traj_values[t] + last_gae_lam
            next_val = traj_values[t]

        processed_advantages.append(advs)
        processed_returns.append(rets)

    # Pad advantages and returns
    advantages = torch.zeros(batch_size, max_len)
    returns = torch.zeros(batch_size, max_len)

    for i in range(batch_size):
        seq_len = len(processed_advantages[i])
        advantages[i, :seq_len] = processed_advantages[i]
        returns[i, :seq_len] = processed_returns[i]

    return {
        "states": states.to(device),
        "actions": actions.to(device),
        "rewards": rewards.to(device),
        "log_probs": log_probs.to(device),
        "values": values.to(device),
        "is_teampreview": is_tp.to(device),
        "advantages": advantages.to(device),
        "returns": returns.to(device),
        "padding_mask": padding_mask.to(device),
    }, metadata


def train_exploiter_subprocess(victim_checkpoint: str, config: RNaDConfig):
    """Launch exploiter training as subprocess."""
    print(f"\n{'=' * 60}")
    print("EXPLOITER TRAINING TRIGGERED")
    print(f"Victim: {victim_checkpoint}")
    print(f"{'=' * 60}\n")

    # Launch exploiter_train.py as subprocess
    cmd = [
        sys.executable,
        "src/elitefurretai/rl/exploiter_train.py",
        "--victim",
        victim_checkpoint,
        "--steps",
        str(config.exploiter_updates),
        "--eval-games",
        str(config.exploiter_eval_games),
        "--threshold",
        str(config.exploiter_min_win_rate),
        "--output-dir",
        (
            str(config.exploiter_models_dir)
            if config.exploiter_models_dir
            else "data/models/exploiters"
        ),
        "--team-pool",
        config.exploiter_team_pool_path if config.exploiter_team_pool_path else "",
        "--learning-rate",
        str(config.exploiter_lr),
        "--ent-coef",
        str(config.exploiter_ent_coef),
    ]

    subprocess.run(cmd, check=True)
    print("\nExploiter training complete!")


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown (e.g., when killed via nohup)
    shutdown_requested = threading.Event()
    def signal_handler(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n{sig_name} received. Initiating graceful shutdown...")
        shutdown_requested.set()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Load config
    config = RNaDConfig.load(args.config)
    config.verify()
    print(f"Loaded config from {args.config}, using device: {config.device}... Loading servers")

    # Launch Showdown servers
    server_processes = launch_showdown_servers(config.num_servers, config.showdown_start_port)

    # Optionally auto-launch external vgc-bench runners (isolated environment)
    external_runner_processes: List[subprocess.Popen] = []
    external_runner_log_files: List[TextIO] = []
    if config.auto_launch_external_vgcbench:
        server_ports = [
            config.showdown_start_port + i for i in range(config.num_servers)
        ]
        (
            external_runner_processes,
            external_runner_log_files,
        ) = launch_external_vgcbench_runners(config, server_ports)

    # Generate unique run ID to avoid stale state on Showdown server
    run_id = datetime.now().strftime("%H%M")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"rnad_{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
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
    print("Initializing opponent pool...")
    opponent_pool = OpponentPool(
        main_model=agent,
        device=config.device,
        battle_format=config.battle_format,
        bc_teampreview_path=config.bc_teampreview_path,
        bc_action_path=config.bc_action_path,
        bc_win_path=config.bc_win_path,
        exploiter_models_dir=config.exploiter_models_dir,
        past_models_dir=config.past_models_dir,
        vgc_bench_checkpoint_path=config.vgc_bench_checkpoint_path,
        max_past_models=config.max_past_models,
    )
    if resume_curriculum:
        opponent_pool.curriculum = resume_curriculum

    start = time.time()

    # ==================== START WORKERS ====================
    # Workers generate training data by playing battles against opponent pool
    # Allocate ports for all workers
    worker_ports, server_loads = allocate_server_ports(
        config.num_workers,
        config.players_per_worker,  # Number of concurrent players each worker runs
        config.num_servers,
        config.max_players_per_server,
        config.showdown_start_port,
    )

    # Create multiprocessing queues
    mp_traj_queue: MPQueue = MPQueue(maxsize=1024)
    weight_queues: List[MPQueue] = [MPQueue(maxsize=5) for _ in range(config.num_workers)]
    mp_error_queue: MPQueue = MPQueue(maxsize=100)  # For workers to report errors
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
                worker_model_path,  # model_path
                worker_model_config,
                mp_traj_queue,
                weight_queues[i],
                mp_error_queue,  # error reporting queue
                mp_stop_event,
                run_id,
                config,
            ),
            daemon=True,
            name=f"MPWorker-{i}",
        )
        p.start()
        processes.append(p)
        print(f"✓ Started multiprocessing worker {i} (PID: {p.pid}) on port {server_port}")

    print("Server allocation (players per server):")
    for idx, load in enumerate(server_loads):
        port = config.showdown_start_port + idx
        cap = config.max_players_per_server
        print(f"  - Port {port}: {load}/{cap} players")

    # ==================== MAIN TRAINING LOOP ====================
    # Collect trajectories from workers, batch them, and perform policy updates
    trajectories = []  # Buffer to accumulate trajectories before training
    updates = start_step  # Current training step (may be >0 if resumed from checkpoint)
    total_battles = 0  # Track total number of battles completed across all workers

    # Timing breakdown
    time_collecting_battles = 0.0
    time_training = 0.0
    time_broadcasting = 0.0
    last_update_time = time.time()

    try:
        while updates < config.max_updates:
            # Check for shutdown signal (from SIGTERM/SIGINT)
            if shutdown_requested.is_set():
                print("\nShutdown requested. Signaling workers to stop...")
                mp_stop_event.set()
                break

            # Check if workers have died
            dead_workers: List[mp.Process] = get_dead_workers(processes, mp_error_queue)
            if len(dead_workers) > 0:
                print("\nTraining cannot continue due to worker failure. Saving checkpoint and exiting...")
                print(f"{'='*60}\n")
                break

            # ===== COLLECT TRAJECTORIES FROM WORKERS =====
            # Workers push completed battle trajectories to the queue asynchronously
            # We collect them here until we have enough for a training batch
            try:
                collection_start = time.time()
                traj = mp_traj_queue.get(timeout=1.0)
                time_collecting_battles += time.time() - collection_start

                trajectories.append(traj)
                total_battles += 1  # Each trajectory represents one completed battle

                # Track win rate by opponent type (new trajectory format includes metadata)
                if isinstance(traj, dict) and "opponent_type" in traj:
                    opp_type = traj["opponent_type"]
                    won = traj.get("won", False)
                    battle_len = traj.get("battle_length", 0)
                    forfeited = traj.get("forfeited", False)
                    opponent_pool.record_battle_result(
                        opponent_type=opp_type,
                        won=won,
                        battle_length=battle_len,
                        forfeited=forfeited,
                    )
            except (queue.Empty, Exception):
                # No new trajectories yet, continue waiting
                continue

            # ===== PERFORM TRAINING UPDATE =====
            # Once we've collected enough trajectories, train the model
            if len(trajectories) >= config.train_batch_size:
                # Collate trajectories into padded batches (handles variable length sequences)
                batch, _ = collate_trajectories(trajectories, config.device, config.gamma, config.gae_lambda, max_seq_len=40)

                # Execute one RNaD policy update (PPO + KL regularization vs reference)
                training_start = time.time()
                metrics = learner.update(batch)
                time_training += time.time() - training_start
                updates += 1

                # Calculate time since last update
                current_time = time.time()
                time_per_update = current_time - last_update_time
                last_update_time = current_time

                # ===== LOG TRAINING METRICS =====
                if config.use_wandb and updates % config.log_interval == 0:
                    metrics['update_step'] = updates
                    metrics['total_battles'] = total_battles
                    metrics['battles_per_second'] = total_battles / (time.time() - start)
                    metrics['time_per_update_seconds'] = time_per_update
                    metrics.update(opponent_pool.get_training_metrics())

                    # Timing breakdown
                    total_time = time.time() - start
                    if total_time > 0:
                        metrics['time_collecting_battles_pct'] = (time_collecting_battles / total_time) * 100
                        metrics['time_training_pct'] = (time_training / total_time) * 100
                        metrics['time_broadcasting_pct'] = (time_broadcasting / total_time) * 100

                    # Portfolio-specific metrics (if using portfolio learner)
                    if isinstance(learner, PortfolioRNaDLearner):
                        metrics['portfolio_size'] = len(learner.ref_models)

                        # KL divergence per reference (recent average)
                        for ref_idx, kl_hist in enumerate(learner.portfolio_kl_history):
                            if len(kl_hist) > 0:
                                metrics[f'portfolio_kl_ref_{ref_idx}'] = sum(kl_hist[-10:]) / len(kl_hist[-10:])

                        # Reference selection counts
                        total_selections = sum(learner.portfolio_selection_counts)
                        if total_selections > 0:
                            for ref_idx, count in enumerate(learner.portfolio_selection_counts):
                                metrics[f'portfolio_selection_pct_ref_{ref_idx}'] = (count / total_selections) * 100

                    wandb.log(metrics)

                    # Build win rate string for console output
                    win_rate_stats = opponent_pool.get_win_rate_stats()
                    win_rate_str = " | ".join([
                        f"{opp_type}: {wr * 100:.1f}%"
                        for opp_type, wr in win_rate_stats.items()
                        if len(opponent_pool.win_rate_tracking.get(opp_type, [])) > 0
                    ])
                    if win_rate_str:
                        win_rate_str = f" | Win rates: {win_rate_str}"

                    print(f"Update {updates}: Loss={metrics['loss']:.4f}, Policy={metrics['policy_loss']:.4f}, Value={metrics['value_loss']:.4f}, RNaD={metrics['rnad_loss']:.4f} | Total Battles={total_battles} in {format_time(time.time() - start)} ({total_battles / (time.time() - start):.2f} b/s)", flush=True)
                    print(f"\t{win_rate_str}")

                # ===== UPDATE REFERENCE MODEL =====
                # Reference model is used for KL regularization in RNaD
                # Periodically sync it with current policy to prevent policy collapse
                if updates % config.ref_update_interval == 0:
                    print(f"[Update {updates}] Updating reference model...")
                    if isinstance(learner, PortfolioRNaDLearner):
                        learner.update_main_reference()  # Update primary reference in portfolio
                    elif isinstance(learner, RNaDLearner):
                        learner.update_ref_model()  # Update single reference model

                # ===== ADD NEW REFERENCE TO PORTFOLIO =====
                # Portfolio regularization: maintain multiple reference snapshots
                # Helps prevent cyclic behavior by regularizing against diverse past policies
                if isinstance(learner, PortfolioRNaDLearner) and updates % config.portfolio_add_interval == 0:
                    print(f"[Update {updates}] Adding new reference to portfolio...")
                    learner.add_reference_model( RNaDAgent(copy.deepcopy(agent.model)))  # Snapshot current policy

                # ===== SAVE CHECKPOINT =====
                # Periodically save model, optimizer state, and training progress
                if updates % config.checkpoint_interval == 0:
                    print(f"[Update {updates}] Saving checkpoint and updating curriculum...")
                    checkpoint_path = save_checkpoint(agent, learner.optimizer, updates, config, opponent_pool.curriculum, config.save_dir)

                    ghost_checkpoint_path = checkpoint_path
                    if os.path.abspath(config.past_models_dir) != os.path.abspath(config.save_dir):
                        ghost_checkpoint_path = save_checkpoint(
                            agent,
                            learner.optimizer,
                            updates,
                            config,
                            opponent_pool.curriculum,
                            config.past_models_dir,
                        )

                    # Add checkpoint to past models pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_past_model(updates, ghost_checkpoint_path)
                    opponent_pool.update_curriculum(adaptive=config.adaptive_curriculum)

                    # ===== BROADCAST WEIGHTS TO WORKERS (MULTIPROCESSING MODE) =====
                    print(f"[Update {updates}] Broadcasting weights to worker processes...")
                    broadcast_start = time.time()
                    cpu_weights = {k: v.cpu() for k, v in agent.model.state_dict().items()}
                    for i, wq in enumerate(weight_queues):
                        try:
                            # Clear old weights to avoid queue overflow
                            while not wq.empty():
                                try:
                                    wq.get_nowait()
                                except Exception:
                                    break
                            wq.put_nowait(cpu_weights)
                        except Exception as e:
                            print(f"  Warning: Failed to broadcast to worker {i}: {e}")
                    time_broadcasting += time.time() - broadcast_start

                # ===== TRAIN EXPLOITER (FIND WEAKNESSES) =====
                # Periodically train a new exploiter agent to beat current policy
                # Exploiters are added to opponent pool to patch discovered weaknesses
                if config.train_exploiters and updates % config.exploiter_interval == 0:

                    # Save current policy as "victim" for exploiter to train against
                    victim_path = os.path.join(config.save_dir, f"victim_step_{updates}.pt")
                    save_checkpoint(agent, learner.optimizer, updates, config, opponent_pool.curriculum, config.save_dir)

                    # Launch exploiter training in subprocess (runs independently)
                    try:
                        train_exploiter_subprocess(victim_path, config)
                    except Exception as e:
                        print(f"Exploiter training failed: {e}")

                # Clear trajectory buffer after successful update
                trajectories = []

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected...")
        mp_stop_event.set()
    finally:
        # ===== GRACEFUL SHUTDOWN =====
        print("\nWaiting for workers to finish (max 5 seconds)...")

        # Signal multiprocessing workers to stop
        mp_stop_event.set()

        # Wait for processes to finish
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                print(f"  Worker {p.name} (PID: {p.pid}) still running, terminating...")
                p.terminate()
                p.join(timeout=1.0)

        # Save progress before exiting (handles Ctrl+C, kill, or normal completion)
        print("\nShutting down training...")

        # Save final checkpoint so training can be resumed later
        final_path = save_checkpoint(agent, learner.optimizer, updates, config, opponent_pool.curriculum, config.save_dir)
        print(f"Final model saved to {final_path}")

        # Close W&B run properly to ensure all logs are synced
        if config.use_wandb:
            wandb.finish()

        # Final cleanup: Shutdown workers and Showdown servers
        cleanup_worker_executors()
        shutdown_external_vgcbench_runners(
            external_runner_processes,
            external_runner_log_files,
        )
        shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization or mp.Process creation
    mp.set_start_method("spawn", force=True)

    # Configure platform-specific multiprocessing behavior
    configure_torch_multiprocessing(use_file_system_sharing=True)
    suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

    main()
