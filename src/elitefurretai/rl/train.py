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
import warnings
from collections import deque
from datetime import datetime
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
from poke_env import ServerConfiguration

import wandb
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.checkpoint import load_checkpoint as _load_checkpoint
from elitefurretai.rl.checkpoint import save_checkpoint as _save_checkpoint
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import PortfolioRNaDLearner, RNaDLearner
from elitefurretai.rl.model_builder import build_model_from_config
from elitefurretai.rl.opponents import (
    ExploiterRegistry,
    OpponentPool,
    WorkerOpponentFactory,
)
from elitefurretai.rl.players import RNaDAgent, cleanup_worker_executors
from elitefurretai.rl.server_manager import allocate_server_ports as _allocate_server_ports
from elitefurretai.rl.server_manager import (
    launch_showdown_servers as _launch_showdown_servers,
)
from elitefurretai.rl.server_manager import (
    shutdown_showdown_servers as _shutdown_showdown_servers,
)
from elitefurretai.supervised import FlexibleThreeHeadedModel, format_time

# Filter out Pydantic field attribute warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")


def load_model(filepath: Optional[str], device: str, config: Optional[RNaDConfig] = None) -> FlexibleThreeHeadedModel:
    """Load model from checkpoint or create new one.

    Uses build_model_from_config for consistent model construction.
    """
    if filepath is not None:
        checkpoint = torch.load(filepath, map_location=device)
        cfg: Dict[str, Any] = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    elif config is not None:
        cfg = config.to_dict()
        state_dict = None
    else:
        raise ValueError("Either filepath or cfg must be provided.")

    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=cfg["embedder_feature_set"],
        omniscient=False,
    )

    return build_model_from_config(cfg, embedder, device, state_dict)


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
    battle_format: str,
    team_pool_path: str,
    team_subdirectory: Optional[str],
    num_players: int,
    run_id: str,
    device: str = "cuda",
    batch_size: int = 16,
    batch_timeout: float = 0.01,
    max_battle_steps: int = 40,
    num_battles_per_pair: int = 20,
    curriculum: Optional[Dict[str, float]] = None,
    bc_model_path: Optional[str] = None,
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
        battle_format: Pokemon Showdown format string
        team_pool_path: Base path for team repository
        team_subdirectory: Subdirectory within format for teams
        num_players: Number of concurrent battles per worker
        run_id: Unique run identifier
        device: Device for model inference
        batch_size: Batch size for inference
        batch_timeout: Timeout for batching inference requests
        max_battle_steps: Maximum steps before forfeiting battle
        num_battles_per_pair: Number of battles to run per player pair per batch
        curriculum: Dict of opponent type -> sampling probability
        bc_model_path: Path to BC model for opponent sampling
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
        server_config = ServerConfiguration(f"ws://localhost:{server_port}/showdown/websocket", None)  # type: ignore[arg-type]

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
        )

        # Create player pairs and MaxDamagePlayer opponents via factory
        num_pairs = num_players // 2
        players, opponents = factory.create_player_pairs(num_pairs, local_traj_queue)
        factory.create_max_damage_opponents(num_pairs)
        if verbose:
            print(f"[MPWorker {worker_id}] Created {num_pairs} player pairs... Memory: {get_memory_usage_mb()}", flush=True)

        # Start inference loops
        factory.start_inference_loops()
        if verbose:
            print(f"[MPWorker {worker_id}] Inference loops started... Memory: {get_memory_usage_mb()}", flush=True)

        async def run_battles(num_battles_per_pair: int):
            """Main battle loop with team randomization and curriculum-based opponent selection."""
            await asyncio.sleep(0.2)  # Let inference loops warm up

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

                # IMPORTANT: Randomize ALL teams before each batch for better generalization
                factory.randomize_all_teams()

                # Sample opponent types and configure opponents for this batch
                batch_opponent_types: List[str] = []
                for player, opponent in zip(factory.players, factory.opponents):
                    opp_type = factory.configure_opponent_for_batch(player, opponent)
                    batch_opponent_types.append(opp_type)

                # Run battles - MaxDamagePlayer pairs use dedicated opponents
                tasks = []
                for i, player in enumerate(factory.players):
                    if batch_opponent_types[i] == WorkerOpponentFactory.MAX_DAMAGE and factory.max_damage_opponents:
                        md_opp = factory.max_damage_opponents[i % len(factory.max_damage_opponents)]
                        tasks.append(player.battle_against(md_opp, n_battles=num_battles_per_pair))
                    else:
                        tasks.append(player.battle_against(factory.opponents[i], n_battles=num_battles_per_pair))
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


def save_checkpoint(
    model: RNaDAgent,
    optimizer,
    step: int,
    config: RNaDConfig,
    curriculum: Dict,
    save_dir: str = "data/models",
):
    """Backward-compatible wrapper to rl.checkpoint.save_checkpoint."""
    return _save_checkpoint(model, optimizer, step, config, curriculum, save_dir)


def load_checkpoint(
    filepath: str, model: RNaDAgent, optimizer, device: str
) -> Tuple[int, RNaDConfig]:
    """Backward-compatible wrapper to rl.checkpoint.load_checkpoint."""
    return _load_checkpoint(filepath, model, optimizer, device)


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
            str(config.exploiter_output_dir)
            if config.exploiter_output_dir
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


def launch_showdown_servers(num_servers: int, start_port: int = 8000) -> List[subprocess.Popen]:
    """Backward-compatible wrapper to rl.server_manager.launch_showdown_servers."""
    return _launch_showdown_servers(num_servers, start_port)


def shutdown_showdown_servers(server_processes: List[subprocess.Popen]) -> None:
    """Backward-compatible wrapper to rl.server_manager.shutdown_showdown_servers."""
    _shutdown_showdown_servers(server_processes)


def allocate_server_ports(
    num_workers: int,
    players_per_worker: int,
    num_showdown_servers: int,
    max_players_per_server: int,
    showdown_start_port: int,
) -> Tuple[List[int], List[int]]:
    """Backward-compatible wrapper to rl.server_manager.allocate_server_ports."""
    return _allocate_server_ports(
        num_workers,
        players_per_worker,
        num_showdown_servers,
        max_players_per_server,
        showdown_start_port,
    )


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    # Set up signal handlers for graceful shutdown (e.g., when killed via nohup)
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        """Handle SIGTERM and SIGINT for graceful shutdown."""
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n{sig_name} received. Initiating graceful shutdown...")
        shutdown_requested.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Load or create config
    config = RNaDConfig.load(args.config)
    print(f"Loaded config from {args.config}")

    # Check the config's paths
    assert os.path.exists(config.bc_teampreview_path)
    assert os.path.exists(config.bc_action_path)
    assert os.path.exists(config.bc_win_path)
    assert os.path.exists(config.base_team_path)
    if config.team_pool_path:
        path = os.path.join(config.base_team_path, config.battle_format, config.team_pool_path)
        assert os.path.exists(path)
    if config.resume_from:
        assert os.path.exists(config.resume_from)
    if config.initialize_path:
        assert os.path.exists(config.initialize_path)

    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Launch Showdown servers
    server_processes = launch_showdown_servers(
        config.num_showdown_servers, config.showdown_start_port
    )

    # Generate unique run ID to avoid stale state on Showdown server
    # Use last 4 digits only (HHMM) for compact usernames that survive Showdown's char stripping
    run_id = datetime.now().strftime("%H%M")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"rnad_{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
        )

    # Determine team sampling parameters
    assert isinstance(config.base_team_path, str)
    team_subdirectory = config.team_pool_path
    msg = f"{config.battle_format}/{team_subdirectory})" if team_subdirectory else config.battle_format
    print(f"Loaded team repository from {config.base_team_path} (sampling from {msg})")

    # Determine model path for initialization
    print(f"Loading main model from {config.initialize_path}...")
    base_model = load_model(config.initialize_path, device, config=config)
    agent = RNaDAgent(base_model)

    # Create learner (portfolio or standard)
    learner: Union[RNaDLearner, PortfolioRNaDLearner]
    if config.use_portfolio_regularization:
        print(f"Using Portfolio RNaD with {config.max_portfolio_size} references")
        # Create initial portfolio with current model
        ref_base_model = copy.deepcopy(base_model)
        ref_agent = RNaDAgent(ref_base_model)
        ref_models = [ref_agent]  # Start with one reference

        learner = PortfolioRNaDLearner(
            agent,
            ref_models,
            lr=config.lr,
            rnad_alpha=config.rnad_alpha,
            gradient_clip=config.max_grad_norm,
            device=device,
            gamma=config.gamma,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            use_mixed_precision=config.use_mixed_precision,
            max_portfolio_size=config.max_portfolio_size,
            portfolio_update_strategy=config.portfolio_update_strategy,
        )
    else:
        print("Using standard RNaD with single reference")
        # Create ref model (deep copy)
        ref_base_model = copy.deepcopy(base_model)
        ref_agent = RNaDAgent(ref_base_model)

        learner = RNaDLearner(
            agent,
            ref_agent,
            lr=config.lr,
            rnad_alpha=config.rnad_alpha,
            device=device,
            gamma=config.gamma,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            use_mixed_precision=config.use_mixed_precision,
            gradient_clip=config.max_grad_norm,
        )

    # Create opponent pool
    print("Initializing opponent pool...")
    opponent_pool = OpponentPool(
        main_model=agent,
        device=device,
        battle_format=config.battle_format,
        bc_teampreview_path=config.bc_teampreview_path,
        bc_action_path=config.bc_action_path,
        bc_win_path=config.bc_win_path,
        exploiter_registry_path=config.exploiter_registry_path,
        past_models_dir=config.past_models_dir,
        max_past_models=config.max_past_models,
    )

    # Resume from checkpoint if requested (restore optimizer + step state + curriculum)
    start_step = 0
    start = time.time()
    if config.resume_from:
        assert isinstance(config.resume_from, str)
        start_step, old_config = load_checkpoint(
            config.resume_from, agent, learner.optimizer, device
        )
        if old_config.curriculum:
            opponent_pool.curriculum = old_config.curriculum

    # ==================== START WORKERS ====================
    # Workers generate training data by playing battles against opponent pool

    # Display allocation strategy
    print("\nAllocation Strategy:")
    print(f"  Total players: {config.num_players}")
    print(f"  Workers: {config.num_workers} (each runs {config.players_per_worker} player(s) on {device})")
    print(f"  Servers: {config.num_servers} (max {config.max_players_per_server} connections each)")
    print(f"  Total connections: {config.num_players * 2} (player + opponent pairs)")

    # Allocate ports for all workers
    worker_ports, server_loads = allocate_server_ports(
        config.num_workers,
        config.players_per_worker,  # Number of concurrent players each worker runs
        config.num_showdown_servers,
        config.max_players_per_server,
        config.showdown_start_port,
    )

    # Create multiprocessing queues
    mp_traj_queue: MPQueue = MPQueue(maxsize=1024)
    weight_queues: List[MPQueue] = [MPQueue(maxsize=5) for _ in range(config.num_workers)]
    mp_error_queue: MPQueue = MPQueue(maxsize=100)  # For workers to report errors
    mp_stop_event: MPEvent = mp.Event()

    # Get model config for worker processes from checkpoint
    # TODO: I think this is wrong. sometimes initialize_path is going to be null. Also, we need to handle resume_from?
    assert isinstance(config.initialize_path, str)
    checkpoint = torch.load(config.initialize_path, map_location="cpu", weights_only=False)
    model_config: Dict[str, Any] = checkpoint.get("config", {})

    processes: List[mp.Process] = []
    for i, server_port in enumerate(worker_ports):
        assert mp_traj_queue and mp_error_queue and mp_stop_event
        p = mp.Process(
            target=mp_worker_process,
            args=(
                i,  # worker_id
                server_port,
                config.initialize_path,  # model_path
                model_config,
                mp_traj_queue,
                weight_queues[i],
                mp_error_queue,  # error reporting queue
                mp_stop_event,
                config.battle_format,
                config.base_team_path,
                team_subdirectory,
                config.players_per_worker,  # Number of players this worker should run
                run_id,
                device,  # Use same device as learner (GPU has more memory: 32GB vs 24GB CPU RAM)
                config.batch_size,
                config.batch_timeout,
                config.max_battle_steps,
                config.num_battles_per_pair,
                config.curriculum,  # Curriculum for opponent sampling
                config.bc_action_path,  # BC model path for opponent sampling
            ),
            daemon=True,
            name=f"MPWorker-{i}",
        )
        p.start()
        processes.append(p)
        print(f"✓ Started multiprocessing worker {i} (PID: {p.pid}) on port {server_port}")

    # For main loop, we'll use mp_traj_queue
    print(
        f"\nStarted {config.num_workers} workers with {config.players_per_worker} player(s) each."
    )
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

    # Win rate tracking with rolling window of last 100 battles per opponent type
    WIN_RATE_WINDOW = 100
    win_tracking: Dict[str, deque] = {
        OpponentPool.SELF_PLAY: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.BC_PLAYER: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.EXPLOITERS: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.GHOSTS: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.MAX_DAMAGE: deque(maxlen=WIN_RATE_WINDOW),
    }

    # Battle length tracking (steps per game) by opponent type
    battle_length_tracking: Dict[str, deque] = {
        OpponentPool.SELF_PLAY: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.BC_PLAYER: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.EXPLOITERS: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.GHOSTS: deque(maxlen=WIN_RATE_WINDOW),
        OpponentPool.MAX_DAMAGE: deque(maxlen=WIN_RATE_WINDOW),
    }

    # Forfeit tracking
    forfeit_count = 0

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

                    if opp_type in win_tracking:
                        win_tracking[opp_type].append(1.0 if won else 0.0)
                    if opp_type in battle_length_tracking and battle_len > 0:
                        battle_length_tracking[opp_type].append(battle_len)
                    if forfeited:
                        forfeit_count += 1
            except (queue.Empty, Exception):
                # No new trajectories yet, continue waiting
                continue

            # ===== PERFORM TRAINING UPDATE =====
            # Once we've collected enough trajectories, train the model
            if len(trajectories) >= config.train_batch_size:
                # Collate trajectories into padded batches (handles variable length sequences)
                batch, _traj_metadata = collate_trajectories(trajectories, device, config.gamma, config.gae_lambda, max_seq_len=40)

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

                    # Add win rates by opponent type to metrics
                    for opp_type, wins in win_tracking.items():
                        if len(wins) > 0:
                            metrics[f'win_rate_{opp_type}'] = sum(wins) / len(wins)

                    # Add average battle lengths by opponent type
                    for opp_type, lengths in battle_length_tracking.items():
                        if len(lengths) > 0:
                            metrics[f'avg_battle_length_{opp_type}'] = sum(lengths) / len(lengths)

                    # Overall average battle length
                    all_lengths = [length for lengths in battle_length_tracking.values() for length in lengths]
                    if all_lengths:
                        metrics['avg_battle_length_overall'] = sum(all_lengths) / len(all_lengths)

                    # Forfeit rate
                    if total_battles > 0:
                        metrics['forfeit_rate'] = forfeit_count * 1. / total_battles

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
                    win_rate_str = " | ".join([
                        f"{opp_type}: {sum(wins)/len(wins)*100:.1f}%"
                        for opp_type, wins in win_tracking.items()
                        if len(wins) > 0
                    ])
                    if win_rate_str:
                        win_rate_str = f" | Win rates: {win_rate_str}"

                    print(f"Update {updates}: Loss={metrics['loss']:.4f}, Policy={metrics['policy_loss']:.4f}, Value={metrics['value_loss']:.4f}, RNaD={metrics['rnad_loss']:.4f}{win_rate_str} | Total Battles={total_battles} in {format_time(time.time() - start)} ({total_battles / (time.time() - start):.2f} b/s)", flush=True)

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

                    # Add checkpoint to past models pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_past_model(updates, checkpoint_path)
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

                        # Reload exploiter registry to include newly trained exploiter
                        opponent_pool.exploiter_registry = ExploiterRegistry(config.exploiter_registry_path)
                        print(f"Reloaded exploiter registry: {len(opponent_pool.exploiter_registry.exploiters)} exploiters")
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
        shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization or mp.Process creation
    mp.set_start_method("spawn", force=True)

    # Set sharing strategy for WSL compatibility
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Ignore the specific warning in the specified file
    warnings.filterwarnings("ignore", message=r'Field')

    main()
