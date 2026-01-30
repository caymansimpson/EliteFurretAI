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
import gc
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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration

import wandb
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learner import RNaDLearner
from elitefurretai.rl.multiprocess_actor import (
    BatchInferencePlayer,
    cleanup_worker_executors,
)
from elitefurretai.rl.opponent_pool import ExploiterRegistry, OpponentPool
from elitefurretai.rl.portfolio_learner import PortfolioRNaDLearner
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel
from elitefurretai.supervised.train_utils import format_time


def load_model(filepath, device):
    """Load model from checkpoint or create new one."""

    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}. Initializing random model.")
        config: Dict[str, Any] = {
            "dropout": 0.1,
            "teampreview_head_dropout": 0.3,
            "gated_residuals": False,
            "use_grouped_encoder": True,
            "grouped_encoder_hidden_dim": 512,
            "grouped_encoder_aggregated_dim": 4096,
            "pokemon_attention_heads": 16,
            "early_layers": [4096, 2048, 2048, 1024],
            "early_attention_heads": 16,
            "lstm_layers": 4,
            "lstm_hidden_size": 512,
            "late_layers": [2048, 2048, 1024, 1024],
            "late_attention_heads": 32,
            "teampreview_head_layers": [512, 256],
            "teampreview_attention_heads": 8,
            "turn_head_layers": [2048, 1024, 1024, 1024],
        }
        state_dict = None
    else:
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=config["embedder_feature_set"],
        omniscient=False,
    )
    input_size = embedder.embedding_size

    model = FlexibleThreeHeadedModel(
        input_size=input_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config.get("lstm_layers", 2),
        lstm_hidden_size=config.get("lstm_hidden_size", 512),
        dropout=config.get("dropout", 0.1),
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 8),
        late_attention_heads=config.get("late_attention_heads", 8),
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=(
            embedder.group_embedding_sizes
            if config.get("use_grouped_encoder", False)
            else None
        ),
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        turn_head_layers=config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=config["max_seq_len"],
    ).to(device)

    if state_dict:
        model.load_state_dict(state_dict)

    return model


def worker_loop(
    opponent_pool: OpponentPool,
    traj_queue: queue.Queue,
    num_players: int,
    worker_id: int,
    server_port: int,
    team_repo: TeamRepo,
    battle_format: str,
    run_id: str,
    stop_event: threading.Event,
    team_subdirectory: Optional[str] = None,
    batch_size: int = 16,
):
    """
    Runs an asyncio loop for a set of players with dynamic opponent selection.

    Args:
        opponent_pool: OpponentPool for sampling opponents
        traj_queue: Queue for pushing completed trajectories
        num_players: Number of concurrent players in this worker
        worker_id: Unique worker ID
        server_port: Absolute port number of the assigned showdown server
        team_repo: TeamRepo for random team sampling
        battle_format: Game format string for battle initialization and team sampling (e.g., gen9vgc2023regc)
        run_id: Unique identifier for this training run (prevents stale server state issues)
        stop_event: Event to signal this worker to stop
        team_subdirectory: Optional subdirectory within format to sample teams from (e.g., "easy", "rental_teams")
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_config = ServerConfiguration(
        f"ws://localhost:{server_port}/showdown/websocket",
        None,  # type: ignore[arg-type]
    )

    players = []
    for i in range(num_players):
        # Sample random team if repo provided
        team = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)

        # Compact format without separators (Showdown strips all special chars)
        # Format: W{worker}{pair}R{run_id} e.g. W01R2324 (9 chars)
        # Option 2a: Pass worker_id so each player uses worker-specific executor
        player = BatchInferencePlayer(
            model=opponent_pool.main_model,
            device=opponent_pool.device,
            batch_size=batch_size,
            account_configuration=AccountConfiguration(f"W{worker_id}{i}R{run_id}", None),
            server_configuration=server_config,
            trajectory_queue=traj_queue,
            battle_format=battle_format,
            team=team,
            worker_id=worker_id,  # Option 2a: worker-specific executor
        )
        players.append(player)

    # Start inference loops for main players in POKE_LOOP (poke-env's event loop)
    # This is done synchronously BEFORE starting battles because:
    # 1. The queue and battle handlers run in POKE_LOOP
    # 2. start_inference_loop() schedules the loop in POKE_LOOP via run_coroutine_threadsafe
    for p in players:
        p.start_inference_loop()

    async def run_battles():
        # Give inference loops time to start in POKE_LOOP
        await asyncio.sleep(0.2)

        battle_batch = 0
        while not stop_event.is_set():
            battle_batch += 1
            start = time.time()
            print(f"[Worker {worker_id}] Starting battle batch {battle_batch}...")

            # Sample opponents for each player (with random teams if available)
            # Compact alphanumeric format: {worker}{batch:05d}{pair}R{run_id}
            # Example: 00000100R2324 = worker 0, batch 00001, pair 0, run 2324
            # After opponent_pool prefix (e.g. "Self"): Self00000100R2324 (16 chars)
            # Supports up to 99,999 batches (batch_batch:05d)
            opponents = []
            for i in range(num_players):
                opp_config = AccountConfiguration(f"{worker_id}{battle_batch:05d}{i}R{run_id}", None)
                opponent_team = team_repo.sample_team(
                    battle_format, subdirectory=team_subdirectory
                )
                # Option 2a: Pass worker_id for worker-specific executor
                opponent = opponent_pool.sample_opponent(
                    opp_config, server_config, team=opponent_team, worker_id=worker_id
                )
                opponents.append(opponent)

            # CRITICAL: Start inference loops for BatchInferencePlayer opponents
            # This is now synchronous - schedules in POKE_LOOP via run_coroutine_threadsafe
            for opponent in opponents:
                if isinstance(opponent, BatchInferencePlayer):
                    opponent.start_inference_loop()

            print(
                f"[Worker {worker_id}] Running {len(players)} pairs × 20 battles = {len(players) * 20} total battles..."
            )

            # Run a batch of battles
            tasks = []
            for player, opponent in zip(players, opponents):
                tasks.append(player.battle_against(opponent, n_battles=20))
            await asyncio.gather(*tasks)

            total_time = time.time() - start
            print(f"[Worker {worker_id}] Batch {battle_batch} completed in {total_time:.2f}s ({len(players) * 20 / total_time:.2f} battles/s)!")

            # Clean up: Stop opponent inference loops AND disconnect from server
            # CRITICAL: Must await stop_listening() to properly close websocket connections
            # Otherwise old players remain connected and new ones with same names get rejected
            for opponent in opponents:
                try:
                    # Cancel inference future to stop the inference loop
                    if isinstance(opponent, BatchInferencePlayer) and hasattr(
                        opponent, "_inference_future"
                    ):
                        opponent._inference_future.cancel()
                    # Disconnect from Showdown server to free the username
                    await opponent.stop_listening()  # type: ignore
                except Exception as e:
                    print(f"[Worker {worker_id}] Warning: Error stopping opponent: {e}")

            # MEMORY CLEANUP: Clear completed battles from poke-env's internal dict
            # poke-env stores all battles in player._battles which grows unboundedly
            for player in players:
                player.reset_battles()

            # Give server time to process disconnections before creating new opponents
            await asyncio.sleep(0.5)

        print(f"[Worker {worker_id}] Stop requested, exiting battle loop...")

    try:
        loop.run_until_complete(run_battles())
        print(f"[Worker {worker_id}] Worker finished gracefully.")
    except Exception as e:
        print(f"[Worker {worker_id}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise to ensure main thread knows worker died
        raise


# =============================================================================
# Option 1: True Multiprocessing Worker
# Each worker runs in a separate Python process with its own GIL, model copy,
# and can run truly in parallel with other workers.
# =============================================================================


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
    max_battle_steps: int = 40,
    num_battles_per_pair: int = 20,
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
    """
    def get_memory_usage_mb():
        """Get current process memory usage in MB."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    try:
        print(f"[MPWorker {worker_id}] Starting (PID: {os.getpid()})...", flush=True)
        print(f"[MPWorker {worker_id}] Initial memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Set thread count to avoid contention within process
        torch.set_num_threads(2)

        # IMPORTANT: Load model to CPU first, then move to device
        # This avoids CUDA re-initialization issues with forked processes
        print(f"[MPWorker {worker_id}] Loading model from {model_path}...", flush=True)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"[MPWorker {worker_id}] Checkpoint loaded, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Extract needed data from checkpoint before building model
        embedder_feature_set = checkpoint["config"].get("embedder_feature_set", "full")
        model_state_dict = checkpoint["model_state_dict"]

        # Create embedder for this process
        print(f"[MPWorker {worker_id}] Creating embedder (feature_set={embedder_feature_set})...", flush=True)
        embedder = Embedder(
            format=battle_format,
            feature_set=embedder_feature_set,
            omniscient=False,
        )
        print(f"[MPWorker {worker_id}] Embedder created, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Free checkpoint to save memory (we've extracted what we need)
        del checkpoint
        gc.collect()
        print(f"[MPWorker {worker_id}] After checkpoint cleanup, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Build model
        print(f"[MPWorker {worker_id}] Building model on device={device}...", flush=True)
        model = FlexibleThreeHeadedModel(
            input_size=embedder.embedding_size,
            early_layers=model_config["early_layers"],
            late_layers=model_config["late_layers"],
            lstm_layers=model_config.get("lstm_layers", 2),
            lstm_hidden_size=model_config.get("lstm_hidden_size", 512),
            dropout=model_config.get("dropout", 0.1),
            gated_residuals=model_config.get("gated_residuals", False),
            early_attention_heads=model_config.get("early_attention_heads", 8),
            late_attention_heads=model_config.get("late_attention_heads", 8),
            use_grouped_encoder=model_config.get("use_grouped_encoder", False),
            group_sizes=(
                embedder.group_embedding_sizes
                if model_config.get("use_grouped_encoder", False)
                else None
            ),
            grouped_encoder_hidden_dim=model_config.get("grouped_encoder_hidden_dim", 128),
            grouped_encoder_aggregated_dim=model_config.get("grouped_encoder_aggregated_dim", 1024),
            pokemon_attention_heads=model_config.get("pokemon_attention_heads", 2),
            teampreview_head_layers=model_config.get("teampreview_head_layers", []),
            teampreview_head_dropout=model_config.get("teampreview_head_dropout", 0.1),
            teampreview_attention_heads=model_config.get("teampreview_attention_heads", 4),
            turn_head_layers=model_config.get("turn_head_layers", []),
            num_actions=MDBO.action_space(),
            num_teampreview_actions=MDBO.teampreview_space(),
            max_seq_len=model_config.get("max_seq_len", 17),
        ).to(device)
        print(f"[MPWorker {worker_id}] Model built and moved to {device}, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        model.load_state_dict(model_state_dict)

        # Clean up state dict now that it's loaded
        del model_state_dict
        gc.collect()
        model.eval()
        print(f"[MPWorker {worker_id}] Model weights loaded, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Wrap in RNaDAgent
        agent = RNaDAgent(model)

        # Load team repo
        print(f"[MPWorker {worker_id}] Loading team repo from {team_pool_path}...", flush=True)
        team_repo = TeamRepo(team_pool_path)
        print(f"[MPWorker {worker_id}] Team repo loaded, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        print(f"[MPWorker {worker_id}] Setting up players...", flush=True)

        # Create a simple opponent pool that only does self-play
        # (In multiprocessing mode, we keep it simple - just self-play with own model copy)

        # Set up asyncio event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        server_config = ServerConfiguration(
            f"ws://localhost:{server_port}/showdown/websocket",
            None,  # type: ignore[arg-type]
        )

        # Create local trajectory queue (thread-safe within process)
        local_traj_queue: queue.Queue = queue.Queue()

        # Create players
        players = []
        opponents = []
        for i in range(num_players):
            team = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)
            opponent_team = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)

            player = BatchInferencePlayer(
                model=agent,
                device=device,
                batch_size=batch_size,
                account_configuration=AccountConfiguration(f"MP{worker_id}P{i}R{run_id}", None),
                server_configuration=server_config,
                trajectory_queue=local_traj_queue,
                battle_format=battle_format,
                team=team,
                worker_id=worker_id,
                embedder=embedder,  # Share embedder to save memory
                max_battle_steps=max_battle_steps,
            )
            players.append(player)

            # Self-play opponent (uses same model copy - no contention since same process)
            opponent = BatchInferencePlayer(
                model=agent,
                device=device,
                batch_size=batch_size,
                account_configuration=AccountConfiguration(f"MP{worker_id}O{i}R{run_id}", None),
                server_configuration=server_config,
                trajectory_queue=None,  # Opponent doesn't collect trajectories
                battle_format=battle_format,
                team=opponent_team,
                worker_id=worker_id,
                embedder=embedder,  # Share embedder to save memory
                max_battle_steps=max_battle_steps,
            )
            opponents.append(opponent)
            print(f"[MPWorker {worker_id}] Created player pair {i}, memory: {get_memory_usage_mb():.1f} MB", flush=True)

        # Start inference loops
        print(f"[MPWorker {worker_id}] Starting inference loops for {len(players)} players and {len(opponents)} opponents...", flush=True)
        for p in players:
            p.start_inference_loop()
        for o in opponents:
            o.start_inference_loop()
        print(f"[MPWorker {worker_id}] Inference loops started, memory: {get_memory_usage_mb():.1f} MB", flush=True)
        print(f"[MPWorker {worker_id}] Beginning battle loop...", flush=True)

        async def run_battles(num_battles_per_pair):
            # Give inference loops time to start
            await asyncio.sleep(0.2)

            battle_batch = 0
            last_weight_check = time.time()

            while not stop_event.is_set():
                battle_batch += 1
                start = time.time()

                # Check for weight updates every second
                if time.time() - last_weight_check > 1.0:
                    try:
                        while not weight_queue.empty():
                            new_weights = weight_queue.get_nowait()
                            model.load_state_dict(new_weights)
                            print(f"[MPWorker {worker_id}] Updated weights")
                    except Exception:
                        pass
                    last_weight_check = time.time()

                # Track memory before battles for leak detection
                mem_before_battles = get_memory_usage_mb()

                # Run battles
                tasks = []
                for player, opponent in zip(players, opponents):
                    tasks.append(player.battle_against(opponent, n_battles=num_battles_per_pair))
                await asyncio.gather(*tasks)

                total_time = time.time() - start
                battles_completed = num_players * num_battles_per_pair
                mem_mb = get_memory_usage_mb()

                # Log memory delta to detect leaks (only if significant growth)
                mem_delta = mem_mb - mem_before_battles
                if mem_delta > 500:  # More than 500MB growth in one batch
                    print(f"[MPWorker {worker_id}] WARNING: Large memory growth in batch {battle_batch}: +{mem_delta:.0f}MB", flush=True)

                # Transfer trajectories from local queue to multiprocessing queue
                transferred = 0
                while not local_traj_queue.empty():
                    try:
                        traj = local_traj_queue.get_nowait()
                        traj_queue.put(traj)
                        transferred += 1
                    except Exception:
                        break

                print(f"[MPWorker {worker_id}] Batch {battle_batch}: {battles_completed} battles in {total_time:.2f}s ({battles_completed / total_time:.2f} b/s), mem={mem_mb:.0f}MB | Sent {transferred} trajectories to learner", flush=True)

                # Report memory stats BEFORE cleanup to diagnose leaks
                if battle_batch % 5 == 0:
                    num_hidden = sum(len(p.hidden_states) for p in players) + sum(len(o.hidden_states) for o in opponents)
                    num_traj = sum(len(p.current_trajectories) for p in players)
                    num_completed = sum(len(p.completed_trajectories) for p in players) + sum(len(o.completed_trajectories) for o in opponents)
                    num_battles_p = sum(len(p._battles) for p in players)
                    num_battles_o = sum(len(o._battles) for o in opponents)

                    # === MEMORY LEAK DIAGNOSTIC ===
                    # Check Battle._observations sizes (suspected leak source)
                    import sys
                    total_obs_size = 0
                    total_obs_count = 0
                    for p in players + opponents:
                        for battle in list(p._battles.values()):
                            if hasattr(battle, '_observations'):
                                obs_dict = battle._observations
                                total_obs_count += len(obs_dict)
                                total_obs_size += sys.getsizeof(obs_dict)
                                for obs in list(obs_dict.values()):
                                    total_obs_size += sys.getsizeof(obs)
                                    # Observation contains dicts - estimate their size
                                    if hasattr(obs, '__dict__'):
                                        total_obs_size += sys.getsizeof(obs.__dict__)

                    # Check hidden state tensor sizes
                    hidden_tensor_size = 0
                    hidden_tensor_count = 0
                    for p in players + opponents:
                        for tag, (h, c) in list(p.hidden_states.items()):
                            hidden_tensor_count += 1
                            hidden_tensor_size += h.nelement() * h.element_size()
                            hidden_tensor_size += c.nelement() * c.element_size()

                    # Check trajectory data sizes
                    traj_data_size = 0
                    for p in players:
                        for tag, traj_list in list(p.current_trajectories.items()):
                            traj_data_size += sys.getsizeof(traj_list)
                            for step in traj_list:
                                if step is not None:
                                    traj_data_size += sys.getsizeof(step)
                                    if isinstance(step, dict):
                                        for v in step.values():
                                            traj_data_size += sys.getsizeof(v)

                    print(f"[MPWorker {worker_id}] LEAK DIAG: obs={total_obs_size/1024/1024:.2f}MB ({total_obs_count} obs), hidden={hidden_tensor_size/1024/1024:.2f}MB ({hidden_tensor_count} states), traj={traj_data_size/1024/1024:.2f}MB", flush=True)
                    # === END MEMORY LEAK DIAGNOSTIC ===

                    # Detailed per-player memory diagnostics
                    for pi, p in enumerate(players):
                        if len(p.hidden_states) > 10 or len(p.current_trajectories) > 10:
                            print(f"[MPWorker {worker_id}] Player {pi}: hidden={len(p.hidden_states)}, current_traj={len(p.current_trajectories)}, completed_traj={len(p.completed_trajectories)}, battles={len(p._battles)}", flush=True)
                    for oi, o in enumerate(opponents):
                        if len(o.hidden_states) > 10 or len(o.completed_trajectories) > 100:
                            print(f"[MPWorker {worker_id}] Opponent {oi}: hidden={len(o.hidden_states)}, completed_traj={len(o.completed_trajectories)}, battles={len(o._battles)}", flush=True)

                    print(f"[MPWorker {worker_id}] PRE-cleanup: hidden={num_hidden}, active_traj={num_traj}, completed_traj={num_completed}, battles_p={num_battles_p}, battles_o={num_battles_o}", flush=True)

                # MEMORY CLEANUP: Clear completed battles from poke-env's internal dict
                # poke-env stores all battles in player._battles which grows unboundedly
                # This caused OOM after ~40 updates with hundreds of completed battles
                for player in players:
                    player.reset_battles()
                    player.clear_completed_trajectories()
                    # Also clear hidden states for battles that ended
                    # (they should be auto-cleaned but belt-and-suspenders)
                    player.hidden_states.clear()
                    player.current_trajectories.clear()
                for opponent in opponents:
                    opponent.reset_battles()
                    opponent.clear_completed_trajectories()  # Critical for opponents!
                    opponent.hidden_states.clear()

                # Force garbage collection every batch and report memory
                gc.collect()
                mem_after_gc = get_memory_usage_mb()
                if battle_batch % 5 == 0:
                    print(f"[MPWorker {worker_id}] Memory after GC: {mem_after_gc:.0f}MB", flush=True)

                # MEMORY THRESHOLD KILL: If worker exceeds 10GB, something is very wrong
                # Kill this worker so main process can restart or continue without it
                MEMORY_LIMIT_MB = 10000  # 10GB
                if mem_after_gc > MEMORY_LIMIT_MB:
                    print(f"[MPWorker {worker_id}] CRITICAL: Memory {mem_after_gc:.0f}MB exceeds limit {MEMORY_LIMIT_MB}MB. Terminating worker.", flush=True)

                    # Try to diagnose what's using memory before dying
                    import sys
                    try:
                        # Get top 20 objects by size
                        all_objects = gc.get_objects()
                        type_counts: Dict[str, int] = {}
                        type_sizes: Dict[str, int] = {}
                        for obj in all_objects:
                            t = type(obj).__name__
                            type_counts[t] = type_counts.get(t, 0) + 1
                            try:
                                type_sizes[t] = type_sizes.get(t, 0) + sys.getsizeof(obj)
                            except Exception:
                                pass

                        print(f"[MPWorker {worker_id}] Top object types by count:", flush=True)
                        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
                            print(f"  {t}: {c} objects, ~{type_sizes.get(t, 0) / 1024 / 1024:.1f}MB", flush=True)
                    except Exception as e:
                        print(f"[MPWorker {worker_id}] Failed to get memory breakdown: {e}", flush=True)

                    # Signal error and exit
                    try:
                        error_queue.put_nowait({
                            "worker_id": worker_id,
                            "error": f"Memory limit exceeded: {mem_after_gc:.0f}MB > {MEMORY_LIMIT_MB}MB",
                            "traceback": ""
                        })
                    except Exception:
                        pass
                    return  # Exit the async function, which will exit the worker

        loop.run_until_complete(run_battles(num_battles_per_pair))
        print(f"[MPWorker {worker_id}] Finished gracefully", flush=True)

    except Exception as e:
        import traceback
        error_msg = f"[MPWorker {worker_id}] FATAL ERROR: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        # Send error to main process for visibility
        try:
            error_queue.put_nowait({"worker_id": worker_id, "error": str(e), "traceback": traceback.format_exc()})
        except Exception:
            pass  # Queue might be full or closed
        # Re-raise so exitcode is non-zero
        raise


def collate_trajectories(trajectories, device, max_seq_len=17):
    """Collate list of trajectories into batched tensors with padding.

    Args:
        trajectories: List of trajectory dicts
        device: torch device
        max_seq_len: Maximum sequence length (matches model's positional embedding size)
                     Trajectories longer than this are truncated (keeping last N steps
                     since later game decisions are typically more impactful).
    """
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
        gamma = 0.99
        lam = 0.95

        next_val = torch.tensor(0.0)

        for t in reversed(range(seq_len)):
            delta = traj_rewards[t] + gamma * next_val - traj_values[t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
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
    }


def save_checkpoint(
    model: RNaDAgent,
    optimizer,
    step: int,
    config: RNaDConfig,
    curriculum: Dict,
    save_dir: str = "data/models",
):
    """Save complete training state for resuming."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"main_model_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "curriculum": curriculum,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str, model: RNaDAgent, optimizer, config: RNaDConfig, device: str
):
    """Load training state from checkpoint."""
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return 0, {}

    print(f"Resuming from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    curriculum = checkpoint.get("curriculum", {})

    print(f"Resumed from step {step}")
    return step, curriculum


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


def launch_showdown_servers(
    num_servers: int, start_port: int = 8000
) -> List[subprocess.Popen]:
    """Launch Showdown servers on consecutive ports.

    Args:
        num_servers: Number of servers to launch
        start_port: Starting port number (default 8000)

    Returns:
        List of subprocess.Popen objects for each server
    """
    print(f"\n{'=' * 60}")
    print(f"LAUNCHING {num_servers} SHOWDOWN SERVERS")
    print(f"Ports: {start_port}-{start_port + num_servers - 1}")
    print(f"{'=' * 60}\n")

    # Determine path to pokemon-showdown (relative to repo root)
    # Script is in src/elitefurretai/rl/train.py, so go up 4 levels to repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    showdown_dir = os.path.join(repo_root, "..", "pokemon-showdown")

    if not os.path.exists(showdown_dir):
        raise FileNotFoundError(f"Pokemon Showdown not found at {showdown_dir}")

    server_processes = []
    for i in range(num_servers):
        port = start_port + i
        try:
            # Launch server with stdout/stderr redirected to suppress logs
            process = subprocess.Popen(
                [
                    "node",
                    "pokemon-showdown",
                    "start",
                    "--no-security",
                    "--port",
                    str(port),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=showdown_dir,  # Run from pokemon-showdown directory
                preexec_fn=os.setsid
                if os.name != "nt"
                else None,  # Create process group on Unix
            )
            server_processes.append(process)
            print(f"✓ Launched Showdown server on port {port} (PID: {process.pid})")
            time.sleep(0.5)  # Small delay between launches
        except FileNotFoundError:
            print(f"ERROR: 'node' or 'pokemon-showdown' not found at {showdown_dir}")
            # Clean up already launched servers
            shutdown_showdown_servers(server_processes)
            raise
        except Exception as e:
            print(f"ERROR launching server on port {port}: {e}")
            shutdown_showdown_servers(server_processes)
            raise

    # Give servers time to fully start
    print("\nWaiting for servers to initialize...")
    time.sleep(2)
    print(f"All {num_servers} servers ready!\n")
    return server_processes


def shutdown_showdown_servers(server_processes: List[subprocess.Popen]) -> None:
    """Gracefully shut down all Showdown server processes.

    Args:
        server_processes: List of subprocess.Popen objects to terminate
    """
    if not server_processes:
        return

    print(f"\n{'=' * 60}")
    print(f"SHUTTING DOWN {len(server_processes)} SHOWDOWN SERVERS")
    print(f"{'=' * 60}\n")

    for i, process in enumerate(server_processes):
        if process.poll() is None:  # Process is still running
            try:
                print(
                    f"Terminating server {i + 1}/{len(server_processes)} (PID: {process.pid})..."
                )
                if os.name != "nt":  # Unix
                    # Kill entire process group to catch child processes
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:  # Windows
                    process.terminate()

                # Wait up to 3 seconds for graceful shutdown
                try:
                    process.wait(timeout=3)
                    print(f"✓ Server on PID {process.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    print(
                        f"⚠ Server on PID {process.pid} didn't respond, force killing..."
                    )
                    if os.name != "nt":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            except ProcessLookupError:
                print(f"Server on PID {process.pid} already terminated")
            except Exception as e:
                print(f"Error terminating server PID {process.pid}: {e}")

    print("\nAll servers shut down.\n")


def allocate_server_ports(
    num_workers: int,
    players_per_worker: int,
    num_showdown_servers: int,
    max_players_per_server: int,
    showdown_start_port: int,
) -> Tuple[List[int], List[int]]:
    """Distribute workers across servers while respecting per-server player caps."""

    total_players = num_workers * players_per_worker
    total_capacity = num_showdown_servers * max_players_per_server
    if total_players > total_capacity:
        raise ValueError(
            "Not enough Showdown server capacity. "
            f"Need {total_players} concurrent players but only have capacity for {total_capacity}. "
            "Increase num_showdown_servers or max_players_per_server (see OPTIMIZATIONS.md)."
        )

    server_loads = [0 for _ in range(num_showdown_servers)]
    worker_ports: List[int] = []

    for _ in range(num_workers):
        selected_idx: Optional[int] = None
        for idx in range(num_showdown_servers):
            projected = server_loads[idx] + players_per_worker
            if projected <= max_players_per_server:
                if selected_idx is None or server_loads[idx] < server_loads[selected_idx]:
                    selected_idx = idx

        if selected_idx is None:
            raise ValueError(
                "Unable to allocate worker to a Showdown server without exceeding max_players_per_server. "
                "Try increasing num_showdown_servers or relaxing the per-server cap."
            )

        server_loads[selected_idx] += players_per_worker
        worker_ports.append(showdown_start_port + selected_idx)

    return worker_ports, server_loads


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--initialize",
        type=str,
        default=None,
        help="Initialize model weights from checkpoint (starts fresh training)",
    )
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
    if args.config:
        config = RNaDConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = RNaDConfig()
        print("Using default config")

    # Save config for this run
    os.makedirs(config.save_dir, exist_ok=True)
    config_path = os.path.join(
        config.save_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    config.save(config_path)
    print(f"Config saved to {config_path}")

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
            name=config.wandb_run_name
            or f"rnad_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
        )
        # Save config as artifact
        artifact = wandb.Artifact(name="config", type="config")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)

    # Load team repo (required)
    # TeamRepo expects base_team_path to contain format folders (e.g., data/teams/gen9vgc2023regc/)
    team_repo = TeamRepo(config.base_team_path)

    # Determine team sampling parameters
    if config.team_pool_path:
        # Sample only from specific subdirectory within the format
        team_subdirectory = config.team_pool_path
        print(
            f"Loaded team repository from {config.base_team_path} (sampling from {config.battle_format}/{team_subdirectory})"
        )
    else:
        # Sample from all teams in the format
        team_subdirectory = None
        print(
            f"Loaded team repository from {config.base_team_path} (sampling from all {config.battle_format} teams)"
        )

    # Determine model path for initialization
    # Priority: config.initialize_path > args.initialize
    model_init_path = config.initialize_path or args.initialize
    print(f"Loading main model from {model_init_path}...")
    base_model = load_model(model_init_path, device)
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

    # Resume from checkpoint if requested (restore optimizer + step state)
    # Note: --checkpoint is for model initialization only, not resuming training state
    start_step = 0
    start = time.time()
    curriculum = {}
    if config.resume_from or args.resume:
        # Priority: CLI flag --resume > config.resume_from
        if args.resume:
            checkpoint_path = os.path.join(
                config.save_dir,
                sorted(
                    [
                        f
                        for f in os.listdir(config.save_dir)
                        if f.startswith("main_model_step_")
                    ]
                )[-1],
            )
        else:
            assert isinstance(config.resume_from, str)
            checkpoint_path = config.resume_from
        assert isinstance(checkpoint_path, str)
        start_step, curriculum = load_checkpoint(
            checkpoint_path, agent, learner.optimizer, config, device
        )

    # Create opponent pool
    print("Initializing opponent pool...")
    opponent_pool = OpponentPool(
        main_model=agent,
        device=device,
        battle_format=config.battle_format,
        bc_teampreview_path=(
            config.bc_teampreview_path
            if os.path.exists(config.bc_teampreview_path)
            else None
        ),
        bc_action_path=(
            config.bc_action_path if os.path.exists(config.bc_action_path) else None
        ),
        bc_win_path=config.bc_win_path if os.path.exists(config.bc_win_path) else None,
        exploiter_registry_path=config.exploiter_registry_path,
        past_models_dir=config.past_models_dir,
        max_past_models=config.max_past_models,
    )

    # Restore curriculum if resuming
    if curriculum:
        opponent_pool.curriculum = curriculum
        print(f"Restored curriculum: {curriculum}")

    # ==================== START WORKERS ====================
    # Workers generate training data by playing battles against opponent pool

    # Allocate ports for all workers
    worker_ports, server_loads = allocate_server_ports(
        config.num_workers,
        config.players_per_worker,
        config.num_showdown_servers,
        getattr(config, "max_players_per_server", config.players_per_worker),
        config.showdown_start_port,
    )

    # Choose between multiprocessing (Option 1) and threading (Option 2a)
    if config.use_multiprocessing:
        # ==================== OPTION 1: MULTIPROCESSING ====================
        print("\n=== Using TRUE MULTIPROCESSING mode (Option 1) ===")
        print("Each worker runs in a separate process with its own GIL and model copy.")
        print(f"  - Learner: {device}")
        print(f"  - Workers: {device}")

        # Create multiprocessing queues
        mp_traj_queue: Optional[MPQueue] = MPQueue(maxsize=1000)
        weight_queues: List[MPQueue] = [MPQueue(maxsize=5) for _ in range(config.num_workers)]
        mp_error_queue: Optional[MPQueue] = MPQueue(maxsize=100)  # For workers to report errors
        mp_stop_event: Optional[MPEvent] = mp.Event()

        # Get model config for worker processes
        model_config = base_model.config if hasattr(base_model, 'config') else {}
        # If config is not on model, extract from checkpoint
        checkpoint = torch.load(model_init_path, map_location="cpu", weights_only=False)
        model_config = checkpoint.get("config", {})

        processes: List[mp.Process] = []
        for i, server_port in enumerate(worker_ports):
            assert mp_traj_queue is not None
            assert mp_error_queue is not None
            assert mp_stop_event is not None
            p = mp.Process(
                target=mp_worker_process,
                args=(
                    i,  # worker_id
                    server_port,
                    model_init_path,  # model_path
                    model_config,
                    mp_traj_queue,
                    weight_queues[i],
                    mp_error_queue,  # error reporting queue
                    mp_stop_event,
                    config.battle_format,
                    config.base_team_path,
                    team_subdirectory,
                    config.players_per_worker,
                    run_id,
                    device,  # Use same device as learner (GPU has more memory: 32GB vs 24GB CPU RAM)
                    config.batch_size,
                    config.max_battle_steps,
                    config.num_battles_per_pair,
                ),
                daemon=True,
                name=f"MPWorker-{i}",
            )
            p.start()
            processes.append(p)
            print(f"✓ Started multiprocessing worker {i} (PID: {p.pid}) on port {server_port}")

        # For main loop, we'll use mp_traj_queue
        # Threading variables set to None for compatibility
        threads: List[threading.Thread] = []  # Empty list
        thread_traj_queue: Optional[queue.Queue] = None  # Will use mp_traj_queue instead
        thread_stop_event: Optional[threading.Event] = None  # Will use mp_stop_event instead

    else:
        # ==================== OPTION 2a: THREADING ====================
        print("\n=== Using THREADING mode with per-worker executors (Option 2a) ===")
        print("Workers share memory but have separate inference executors to reduce contention.")

        # Create threading queue
        thread_traj_queue = queue.Queue()
        thread_stop_event = threading.Event()

        # Multiprocessing variables set to None/empty
        mp_traj_queue = None
        mp_error_queue = None
        mp_stop_event = None
        weight_queues = []
        processes = []

        threads = []
        for i, server_port in enumerate(worker_ports):
            t = threading.Thread(
                target=worker_loop,
                args=(
                    opponent_pool,
                    thread_traj_queue,
                    config.players_per_worker,
                    i,
                    server_port,
                    team_repo,
                    config.battle_format,
                    run_id,
                    thread_stop_event,  # Fixed: was worker_stop_event
                    team_subdirectory,
                    config.batch_size,
                ),
                daemon=True,  # Daemon threads die when main exits
                name=f"Worker-{i}",
            )
            t.start()
            threads.append(t)
            print(f"✓ Started threading worker {i} on port {server_port}")

    print(
        f"\nStarted {config.num_workers} workers with {config.players_per_worker} players each."
    )
    if config.num_showdown_servers > 1:
        print("Server allocation (players per server):")
        for idx, load in enumerate(server_loads):
            port = config.showdown_start_port + idx
            cap = getattr(config, "max_players_per_server", config.players_per_worker)
            print(f"  - Port {port}: {load}/{cap} players")
    else:
        print(f"All workers using single showdown server on port {config.showdown_start_port}")

    # ==================== MAIN TRAINING LOOP ====================
    # Collect trajectories from workers, batch them, and perform policy updates
    trajectories = []  # Buffer to accumulate trajectories before training
    updates = start_step  # Current training step (may be >0 if resumed from checkpoint)
    last_exploiter_check = start_step  # Track when we last checked for exploiter training
    total_battles = 0  # Track total number of battles completed across all workers

    try:
        while updates < config.max_updates:
            # Check for shutdown signal (from SIGTERM/SIGINT)
            if shutdown_requested.is_set():
                print("\nShutdown requested. Signaling workers to stop...")
                if config.use_multiprocessing and mp_stop_event is not None:
                    mp_stop_event.set()
                elif thread_stop_event is not None:
                    thread_stop_event.set()
                break

            # Check if workers have died
            if config.use_multiprocessing:
                dead_procs = [p for p in processes if not p.is_alive()]
                if dead_procs:
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
                    if mp_error_queue is not None:
                        print("\nChecking for error reports from workers...")
                        error_found = False
                        while True:
                            try:
                                error_info = mp_error_queue.get_nowait()
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
                    break
            else:
                dead_threads = [t for t in threads if not t.is_alive()]
                if dead_threads:
                    print(f"\n{'='*60}")
                    print(f"ERROR: {len(dead_threads)} worker thread(s) died:")
                    for t in dead_threads:
                        print(f"  - {t.name}")
                    print("Training cannot continue. Saving checkpoint and exiting...")
                    print(f"{'='*60}\n")
                    break

            # ===== COLLECT TRAJECTORIES FROM WORKERS =====
            # Workers push completed battle trajectories to the queue asynchronously
            # We collect them here until we have enough for a training batch
            try:
                if config.use_multiprocessing and mp_traj_queue is not None:
                    traj = mp_traj_queue.get(timeout=1.0)
                elif thread_traj_queue is not None:
                    traj = thread_traj_queue.get(timeout=1.0)
                else:
                    time.sleep(0.1)
                    continue
                trajectories.append(traj)
                total_battles += 1  # Each trajectory represents one completed battle
            except (queue.Empty, Exception):
                # No new trajectories yet, continue waiting
                continue

            # ===== PERFORM TRAINING UPDATE =====
            # Once we've collected enough trajectories, train the model
            if len(trajectories) >= config.train_batch_size:
                # Collate trajectories into padded batches (handles variable length sequences)
                batch = collate_trajectories(trajectories, device)

                # Execute one RNaD policy update (PPO + KL regularization vs reference)
                metrics = learner.update(batch)
                updates += 1

                # ===== LOG TRAINING METRICS =====
                if config.use_wandb and updates % config.log_interval == 0:
                    # Core training metrics from the RNaD update
                    log_dict = {
                        "loss": metrics["loss"],
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "entropy": metrics["entropy"],
                        "rnad_loss": metrics["rnad_loss"],
                        "update_step": updates,
                        "total_battles": total_battles,
                    }

                    # Portfolio-specific metrics
                    if config.use_portfolio_regularization and isinstance(
                        learner, PortfolioRNaDLearner
                    ):
                        log_dict["portfolio_size"] = metrics.get("portfolio_size", 1)
                        portfolio_stats = learner.get_portfolio_stats()
                        # Track how often each reference model is selected for regularization
                        for i, count in enumerate(portfolio_stats["selection_counts"]):
                            log_dict[f"portfolio/ref_{i}_selections"] = count
                        # Track average KL divergence from each reference model
                        for i, avg_kl in enumerate(portfolio_stats["avg_kl_per_ref"]):
                            log_dict[f"portfolio/ref_{i}_avg_kl"] = avg_kl

                    # Log win rates against different opponent types
                    # Helps track progress against BC models, past versions, exploiters, etc.
                    win_rate_stats = opponent_pool.get_win_rate_stats()
                    for opp_type, win_rate in win_rate_stats.items():
                        log_dict[f"win_rate/{opp_type}"] = win_rate

                    # Log curriculum weights (how often we sample each opponent type)
                    for opp_type, weight in opponent_pool.curriculum.items():
                        log_dict[f"curriculum/{opp_type}"] = weight

                    wandb.log(log_dict)

                # Print progress to console periodically
                if updates % 1 == 0:
                    print(
                        f"Update {updates}: Loss={metrics['loss']:.4f}, "
                        f"Policy={metrics['policy_loss']:.4f}, "
                        f"Value={metrics['value_loss']:.4f}, "
                        f"Total Battles={total_battles} in {format_time(time.time() - start)}",
                        flush=True
                    )

                # ===== UPDATE REFERENCE MODEL =====
                # Reference model is used for KL regularization in RNaD
                # Periodically sync it with current policy to prevent policy collapse
                if updates % config.ref_update_interval == 0:
                    print(f"[Update {updates}] Updating reference model...")
                    if config.use_portfolio_regularization and isinstance(
                        learner, PortfolioRNaDLearner
                    ):
                        learner.update_main_reference()  # Update primary reference in portfolio
                    else:
                        assert isinstance(learner, RNaDLearner)
                        learner.update_ref_model()  # Update single reference model

                # ===== ADD NEW REFERENCE TO PORTFOLIO =====
                # Portfolio regularization: maintain multiple reference snapshots
                # Helps prevent cyclic behavior by regularizing against diverse past policies
                if config.use_portfolio_regularization and (
                    updates % config.portfolio_add_interval == 0
                ):
                    print(f"[Update {updates}] Adding new reference to portfolio...")
                    new_ref = RNaDAgent(
                        copy.deepcopy(agent.model)
                    )  # Snapshot current policy
                    if isinstance(learner, PortfolioRNaDLearner):
                        learner.add_reference_model(new_ref)
                        portfolio_stats = learner.get_portfolio_stats()
                        print(f"  Portfolio size: {portfolio_stats['portfolio_size']}")
                        print(f"  Selection counts: {portfolio_stats['selection_counts']}")

                # ===== SAVE CHECKPOINT =====
                # Periodically save model, optimizer state, and training progress
                # Allows resuming training if interrupted
                if updates % config.checkpoint_interval == 0:
                    print(f"[Update {updates}] Saving checkpoint...")
                    checkpoint_path = save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        config.save_dir,
                    )

                    # Add checkpoint to past models pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_past_model(updates, checkpoint_path)

                    if config.use_wandb:
                        # Upload model to W&B for versioning and easy access
                        artifact = wandb.Artifact(f"model_step_{updates}", type="model")
                        artifact.add_file(checkpoint_path)
                        wandb.log_artifact(artifact)

                    # ===== BROADCAST WEIGHTS TO WORKERS (MULTIPROCESSING MODE) =====
                    # In multiprocessing mode, workers have their own model copies
                    # We need to periodically sync them with the updated main model
                    if config.use_multiprocessing:
                        print(f"[Update {updates}] Broadcasting weights to worker processes...")
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

                # ===== UPDATE CURRICULUM (ADAPTIVE OPPONENT SAMPLING) =====
                # Adjust weights for sampling different opponent types based on win rates
                # E.g., if we're beating BC players too easily, reduce their sampling weight
                if updates % config.curriculum_update_interval == 0:
                    opponent_pool.update_curriculum(adaptive=True)
                    print(
                        f"[Update {updates}] Curriculum updated: {opponent_pool.curriculum}"
                    )

                # ===== TRAIN EXPLOITER (FIND WEAKNESSES) =====
                # Periodically train a new exploiter agent to beat current policy
                # Exploiters are added to opponent pool to patch discovered weaknesses
                if config.train_exploiters and (
                    updates - last_exploiter_check >= config.exploiter_interval
                ):
                    last_exploiter_check = updates

                    # Save current policy as "victim" for exploiter to train against
                    victim_path = os.path.join(
                        config.save_dir, f"victim_step_{updates}.pt"
                    )
                    save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        config.save_dir,
                    )

                    # Launch exploiter training in subprocess (runs independently)
                    try:
                        train_exploiter_subprocess(victim_path, config)

                        # Reload exploiter registry to include newly trained exploiter
                        opponent_pool.exploiter_registry = ExploiterRegistry(
                            config.exploiter_registry_path
                        )
                        print(
                            f"Reloaded exploiter registry: {len(opponent_pool.exploiter_registry.exploiters)} exploiters"
                        )
                    except Exception as e:
                        print(f"Exploiter training failed: {e}")

                # Clear trajectory buffer after successful update
                trajectories = []

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected...")
        if config.use_multiprocessing and mp_stop_event is not None:
            mp_stop_event.set()
        elif thread_stop_event is not None:
            thread_stop_event.set()
    finally:
        # ===== GRACEFUL SHUTDOWN =====
        print("\nWaiting for workers to finish (max 5 seconds)...")

        if config.use_multiprocessing:
            # Signal multiprocessing workers to stop
            if mp_stop_event is not None:
                mp_stop_event.set()

            # Wait for processes to finish
            for p in processes:
                p.join(timeout=5.0)
                if p.is_alive():
                    print(f"  Worker {p.name} (PID: {p.pid}) still running, terminating...")
                    p.terminate()
                    p.join(timeout=1.0)
        else:
            # Signal threading workers to stop
            if thread_stop_event is not None:
                thread_stop_event.set()

            # Wait for threads to finish
            for t in threads:
                t.join(timeout=5.0)
                if t.is_alive():
                    print(f"  Worker {t.name} still running, will be terminated.")

        # Save progress before exiting (handles Ctrl+C, kill, or normal completion)
        print("\nShutting down training...")

        # Save final checkpoint so training can be resumed later
        final_path = save_checkpoint(
            agent,
            learner.optimizer,
            updates,
            config,
            opponent_pool.curriculum,
            config.save_dir,
        )
        print(f"Final model saved to {final_path}")

        # Close W&B run properly to ensure all logs are synced
        if config.use_wandb:
            wandb.finish()

        # Option 2a: Clean up worker-specific executors (threading mode)
        if not config.use_multiprocessing:
            cleanup_worker_executors()

        # Shutdown Showdown servers
        shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # Must be done before any CUDA initialization or mp.Process creation
    mp.set_start_method("spawn", force=True)

    # Set sharing strategy for WSL compatibility
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
