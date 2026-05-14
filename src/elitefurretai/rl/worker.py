"""
worker.py — Actor process for IMPALA-style distributed RL

What this file is
-----------------
The body of each worker subprocess. The trainer spawns one of these per
config.hardware.num_workers; each runs a long-lived loop that plays battles
and ships completed trajectories back to the learner.

In EliteFurretAI's training loop, computation is split between:

  LEARNER (train.py main process, GPU)
    - Batches completed trajectories
    - Runs gradient updates
    - Broadcasts updated weights every checkpoint_interval steps

  ACTORS (this file, N worker processes, CPU)
    - Each worker maintains its own copy of the policy (CPU, no gradients)
    - Plays battles against various opponents
    - Sends completed trajectories to the learner via mp_traj_queue
    - Periodically polls weight_queue for updated weights from the learner

This is an IMPALA-style architecture (Espeholt et al. 2018). The key property:
actors are always slightly off-policy (they play with an older checkpoint while
the learner is already one or more updates ahead). This is intentional and
handled by importance-sampling corrections in the learner.

Why multiprocessing instead of threading
----------------------------------------
Python's GIL makes threaded inference effectively serial. With one OS process
per worker, each has its own GIL and Python interpreter, so we get true
parallelism on a multi-core machine. The tradeoff is that interprocess data
exchange (weight broadcasts, trajectory shipping) goes through pickled
mp.Queue payloads, which is slower than sharing memory. We pay that cost
because it buys us 2-3× higher actor throughput.

Process lifecycle
-----------------
The worker has three phases:
  PHASE 1: Model loading. Read checkpoint from disk, build the model on CPU,
           optionally load the BC opponent model.
  PHASE 2: Environment setup. Build a VGCEnvironment which wraps either
           Showdown or Rust battle execution behind a uniform interface.
  PHASE 3: Battle loop. Repeatedly: poll for new weights → play a batch of
           battles → forward completed trajectories to the learner.

BACKEND SELECTION
  The worker supports two battle backends, selected by config.hardware.battle_backend:
  - "showdown_websocket": Async websocket battles against a local Showdown server.
      Uses WorkerOpponentFactory + BatchInferencePlayer. Higher throughput.
  - "rust_engine": Synchronous in-process Rust battles.
      Uses _RustPolicyOpponentPool + SyncRustBattleDriver. More deterministic.

  Both paths are fully encapsulated by VGCEnvironment (engine/vgc_environment.py),
  which owns all backend-specific code. mp_worker_process creates one VGCEnvironment,
  calls run_battle_batch() in a loop, and receives BatchResult objects — identical
  regardless of which backend is active.
"""

import asyncio
import gc
import logging
import os
import time
from collections import deque
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, Optional, cast

import psutil
import torch

from elitefurretai.engine.vgc_environment import VGCEnvironment
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.etl.system_utils import suppress_third_party_warnings
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import build_model_from_config
from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.players import RNaDAgent

logger = logging.getLogger(__name__)


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
    # M4 centralized inference: when both queues are provided, the worker
    # skips loading the main model into its process and submits inference
    # requests to the trainer-side InferenceService instead. `model_path`
    # is then unused for the main agent (still used for ghost loading
    # etc.). `main_is_transformer` is required so BatchInferencePlayer
    # knows hidden-state shapes without a model to introspect.
    main_inference_request_queue: Optional[MPQueue] = None,
    main_inference_response_queue: Optional[MPQueue] = None,
    main_is_transformer: Optional[bool] = None,
    # Step 3+: bundle of (req_q, resp_q) keyed by model name for ALL
    # registered models in the trainer's ModelRegistry. Workers wrap
    # this in WorkerInferenceClients so the factory can hot-swap
    # opponent.inference_client between main / bc / ghost slots / etc.
    # When None (legacy mode or step-2 config), only the singular
    # main_inference_request_queue path is used.
    queues_by_model: Optional[Dict[str, Any]] = None,
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
        server_port: Showdown server port (ignored for Rust backend)
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
        # Match the trainer's setup at train.py:1653-1659. Default root to
        # WARNING so poke-env's per-player loggers (named by Showdown
        # username — top-level, not under "elitefurretai") stay quiet —
        # they echo every `<<<` received and `>>>` sent websocket message
        # at INFO, including big `request` JSON payloads, which fills the
        # run.log at ~1.5 GB/hr. Our own modules under "elitefurretai" and
        # "__main__" stay at INFO so training progress, worker memory
        # reports, and watchdog events are preserved. `force=True` ensures
        # this takes effect even if a third-party import already attached
        # a handler to the root logger (basicConfig is otherwise a no-op
        # in that case).
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.getLogger("elitefurretai").setLevel(logging.INFO)
        logging.getLogger("__main__").setLevel(logging.INFO)
        suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

        battle_format = config.curriculum.battle_format
        team_pool_path = config.curriculum.base_team_path
        num_battles_per_pair = config.hardware.num_battles_per_pair
        curriculum = config.curriculum.curriculum_weights
        bc_model_path = config.curriculum.bc_model_path
        external_vgcbench_usernames = config.curriculum.external_vgcbench_usernames
        external_vgcbench_startup_wait_s = (
            config.curriculum.external_vgcbench_startup_wait_s
        )

        # Apply dedicated worker restrictions to initial curriculum before environment setup.
        # Dedicated workers always face VGCBench; remaining workers never do.
        if config.curriculum.dedicated_vgcbench_workers > 0:
            if worker_id < config.curriculum.dedicated_vgcbench_workers:
                curriculum = {OpponentPool.VGC_BENCH_BASELINE: 1.0}
            else:
                curriculum = {
                    k: v
                    for k, v in curriculum.items()
                    if k != OpponentPool.VGC_BENCH_BASELINE
                }

        if verbose:
            logger.debug(
                "[MPWorker %d] Starting (PID: %d)... Initial memory: %s",
                worker_id,
                os.getpid(),
                get_memory_usage_mb(),
            )

        # ── CPU thread budgeting per worker ───────────────────────────────────
        # PyTorch on CPU spawns intra-op threads (controlled here) for matmul,
        # convolutions, etc. With 6 worker processes and 8 CPU cores total, we
        # MUST limit each process or they'll oversubscribe and fight for cores.
        # Empirically 2 threads per worker beats 1 (lets one matmul use 2 cores)
        # and beats 4 (oversubscribes). This is the hidden scaling knob behind
        # the topology choices in HardwareConfig.
        # ─────────────────────────────────────────────────────────────────────
        torch.set_num_threads(2)

        # ── PHASE 1: MODEL LOADING ────────────────────────────────────────────────────
        # Load to CPU first to avoid CUDA re-initialization issues in forked processes,
        # which can cause deadlocks on WSL2. The parent serialized a checkpoint to disk
        # at resolve_worker_model_source(); we read from there.
        if verbose:
            logger.debug(
                "[MPWorker %d] Loading model from %s... Memory: %s",
                worker_id,
                model_path,
                get_memory_usage_mb(),
            )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if verbose:
            logger.debug(
                "[MPWorker %d] Checkpoint loaded... Memory: %s",
                worker_id,
                get_memory_usage_mb(),
            )

        # BC checkpoints store config flat (embedder_feature_set at top level);
        # RL checkpoints store it nested under `training`. Read either layout —
        # mirrors the trainer-side logic in train.py:206-213. The previous code
        # only read the flat layout, which silently fell back to the default
        # "full" on RL-checkpoint resume and produced a 15-encoder model that
        # couldn't load the 13-encoder weights.
        ckpt_cfg = checkpoint["config"]
        if (
            isinstance(ckpt_cfg.get("training"), dict)
            and "embedder_feature_set" in ckpt_cfg["training"]
        ):
            embedder_feature_set = ckpt_cfg["training"]["embedder_feature_set"]
        else:
            embedder_feature_set = ckpt_cfg.get("embedder_feature_set", "full")
        embedder = Embedder(
            format=battle_format,
            feature_set=embedder_feature_set,
            omniscient=False,
        )

        device = "cpu"
        if verbose:
            logger.debug(
                "[MPWorker %d] Building model on device=%s... Memory: %s",
                worker_id,
                device,
                get_memory_usage_mb(),
            )
        # strict=False matches the trainer side (train.py:306-312) so the
        # worker tolerates the same partial-load scenarios: e.g. sep_arch's
        # BC init where the new value_ff_stack and reshaped win_head don't
        # exist in the checkpoint. Without this, workers crash at startup
        # with `Missing/Unexpected key(s) in state_dict` while the trainer
        # itself loads fine — a silent asymmetry.
        # Centralized-inference (M4) skips loading the main model into
        # this worker's process — the trainer-side InferenceService owns
        # it. We still need `embedder` (built above) for featurization
        # and `model_config` for ghost loading paths. Detected from
        # spawn args: when both inference queues are present, build a
        # client and skip the model build.
        centralized_main = (
            main_inference_request_queue is not None
            and main_inference_response_queue is not None
        )

        model: Optional[Any] = None
        agent: Optional[RNaDAgent] = None
        main_inference_client = None
        worker_inference_clients = None  # set in centralized mode (step 3+)
        if centralized_main:
            from elitefurretai.rl.inference_client import InferenceClient
            from elitefurretai.rl.worker_inference_clients import (
                WorkerInferenceClients,
            )

            assert main_is_transformer is not None, (
                "main_is_transformer must be set in centralized mode"
            )
            from poke_env.concurrency import POKE_LOOP

            # Step 3+: when the trainer passed a per-model queues bundle,
            # wrap it in WorkerInferenceClients (one InferenceClient per
            # registered model). When only legacy main queues were passed,
            # construct a single main client.
            if queues_by_model is not None:
                worker_inference_clients = WorkerInferenceClients(
                    worker_id=worker_id,
                    queues_by_model=cast(Any, queues_by_model),
                    loop=POKE_LOOP,
                )
                main_inference_client = worker_inference_clients.get("main")
                if verbose:
                    logger.debug(
                        "[MPWorker %d] Centralized inference (bundle) enabled; "
                        "models=%s",
                        worker_id,
                        worker_inference_clients.names(),
                    )
            else:
                main_inference_client = InferenceClient(
                    worker_id=worker_id,
                    request_queue=cast(Any, main_inference_request_queue),
                    response_queue=cast(Any, main_inference_response_queue),
                    loop=POKE_LOOP,
                )
                main_inference_client.start()
                if verbose:
                    logger.debug(
                        "[MPWorker %d] Centralized inference (legacy single) "
                        "enabled; skipping main model load",
                        worker_id,
                    )
        else:
            model = build_model_from_config(
                model_config,
                embedder,
                device,
                checkpoint["model_state_dict"],
                strict=False,
            )
            model.eval()
            agent = RNaDAgent(model)
        # Optional: torch.compile the inference agent for kernel fusion +
        # dispatch-overhead removal. dynamic=True so the transformer's
        # growing context tensor doesn't trigger recompilation each turn.
        # First inference call after launch pays the compile cost.
        # Skipped in centralized mode (no agent here; trainer owns the
        # compile decision).
        compile_mode = (
            config.hardware.compile_inference_model if not centralized_main else None
        )
        if compile_mode:
            if verbose:
                logger.debug(
                    "[MPWorker %d] torch.compile(agent, mode=%s, dynamic=True)",
                    worker_id,
                    compile_mode,
                )
            # torch.compile returns an OptimizedModule that delegates
            # __call__ and attribute access to the wrapped module; the
            # downstream code only invokes agent(x, ...) and reads
            # agent.model. Cast preserves that runtime contract for
            # the type checker. Guarded by compile_mode (None in
            # centralized mode) so agent is always set here.
            assert agent is not None
            agent = cast(RNaDAgent, torch.compile(agent, mode=compile_mode, dynamic=True))

            # Warm the compile cache synchronously BEFORE workers start
            # accepting battle requests. Without this, the first real
            # inference call pays the 30-60s compile cost while battles'
            # 8s inference_request_timeout fires repeatedly, causing a
            # cascade of "Invalid choice" errors as fallback choices
            # arrive after Showdown has moved on. Two warmup shapes
            # cover the two transformer code paths actually hit in
            # production: turn 0 (no context) and turn 1+ (with context).
            warmup_start = time.time()
            embedding_size = embedder.embedding_size
            with torch.no_grad():
                # Turn 0: no hidden context.
                x_warm = torch.zeros(1, 1, embedding_size, device=device)
                _, _, _, _, ctx_warm = agent(x_warm, None)
                # Turn 1: with prior context.
                agent(x_warm, ctx_warm)
            if verbose:
                logger.debug(
                    "[MPWorker %d] compile warmup complete in %.1fs",
                    worker_id,
                    time.time() - warmup_start,
                )
        del checkpoint
        if verbose:
            logger.debug(
                "[MPWorker %d] Model built... Memory: %s", worker_id, get_memory_usage_mb()
            )

        bc_agent = None
        if curriculum and curriculum.get("bc_player", 0) > 0 and bc_model_path:
            if verbose:
                logger.debug(
                    "[MPWorker %d] Loading BC model... Memory: %s",
                    worker_id,
                    get_memory_usage_mb(),
                )
            bc_checkpoint = torch.load(
                bc_model_path, map_location="cpu", weights_only=False
            )
            # strict=False for the same partial-load reason as the main
            # model above. BC opponents drive action selection through the
            # policy head only, so a partially-loaded value/win head doesn't
            # affect their behavior — only their (unused-as-opponent) value
            # predictions are fresh-initialized.
            bc_model = build_model_from_config(
                model_config,
                embedder,
                device,
                bc_checkpoint["model_state_dict"],
                strict=False,
            )
            bc_model.eval()
            for param in bc_model.parameters():
                param.requires_grad = False
            bc_agent = RNaDAgent(bc_model)
            del bc_checkpoint
            if verbose:
                logger.debug(
                    "[MPWorker %d] BC model loaded and frozen... Memory: %s",
                    worker_id,
                    get_memory_usage_mb(),
                )

        # ── Co-training agents (in-process exploiter pipeline) ────────────
        # When `train_exploiter > 0` in the configured curriculum, build CPU
        # model copies for the exploiter (live, weights synced from main via
        # broadcast) and victim (frozen periodically-refreshed copy of main).
        # Initial weights come from the first broadcast; until then we hold
        # randomly-initialized models. The warmup gate in train.py prevents
        # train_exploiter battles from being sampled before the first
        # broadcast lands.
        exploiter_agent = None
        victim_agent = None
        if curriculum and curriculum.get("train_exploiter", 0) > 0:
            if verbose:
                logger.debug(
                    "[MPWorker %d] Building exploiter and victim agents... Memory: %s",
                    worker_id,
                    get_memory_usage_mb(),
                )
            exploiter_model = build_model_from_config(model_config, embedder, device)
            exploiter_model.eval()
            exploiter_agent = RNaDAgent(exploiter_model)
            victim_model = build_model_from_config(model_config, embedder, device)
            victim_model.eval()
            for param in victim_model.parameters():
                param.requires_grad = False
            victim_agent = RNaDAgent(victim_model)

        if verbose:
            logger.debug(
                "[MPWorker %d] Loading teams from %s... Memory: %s",
                worker_id,
                team_pool_path,
                get_memory_usage_mb(),
            )
        team_repo = TeamRepo(team_pool_path)

        # ── PHASE 2: ENVIRONMENT SETUP ────────────────────────────────────────────────
        # VGCEnvironment wraps both backends (Showdown websocket and Rust in-process)
        # behind a single interface. The backend is chosen by config.hardware.battle_backend.
        # initial_curriculum reflects any dedicated-worker overrides applied above.
        if verbose:
            logger.debug(
                "[MPWorker %d] Creating VGCEnvironment... Memory: %s",
                worker_id,
                get_memory_usage_mb(),
            )
        env = VGCEnvironment.from_config(
            config=config,
            worker_id=worker_id,
            agent=agent,
            model=model,
            model_config=model_config,
            embedder=embedder,
            team_repo=team_repo,
            bc_agent=bc_agent,
            server_port=server_port,
            run_id=run_id,
            initial_curriculum=curriculum,
            exploiter_agent=exploiter_agent,
            victim_agent=victim_agent,
            main_inference_client=main_inference_client,
            main_is_transformer=main_is_transformer,
            worker_inference_clients=worker_inference_clients,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # ── PHASE 3: BATTLE LOOP ──────────────────────────────────────────────────────
        # Each iteration:
        #   1. Poll weight_queue for a learner broadcast. If found, propagate to env:
        #      - model weights (env.update_weights)
        #      - curriculum distribution (env.update_curriculum)
        #      - temperature / top_p (env.update_sampling)
        #      - exploiter_paths / ghost_paths (Option C: explicit model file lists)
        #   2. env.run_battle_batch(battles_per_task) → BatchResult
        #   3. Forward trajectories to traj_queue (Phase 4)
        #   4. Handle timeout recovery (env.rebuild()) or reset (env.reset_battles())

        async def run_battles(num_battles_per_pair: int):
            await env.setup()
            if external_vgcbench_usernames:
                await asyncio.sleep(external_vgcbench_startup_wait_s)

            battle_batch = 0
            last_weight_check = time.time()
            consecutive_vgcbench_timeouts = 0
            vgcbench_disabled_locally = False
            consecutive_zero_completion_batches = 0
            # Raised from 5 to 15 after the 2026-05-06 overnight run crashed at
            # update 137 from clustered "not in that room" popups in worker 0.
            # Real fix is the room-state race in poke-env (see planning doc
            # 2026-05-06-04-50-zero-completion-room-state-race.md); this is the
            # interim guard so transient clustering doesn't kill multi-hour runs.
            zero_completion_failover_threshold = 15

            # Rolling windows for slowdown diagnostics before hard failures occur.
            batch_duration_window: deque[float] = deque(maxlen=50)
            timeout_window: deque[int] = deque(maxlen=50)
            zero_completion_window: deque[int] = deque(maxlen=50)

            battles_per_task = max(1, min(4, num_battles_per_pair))

            try:
                while not stop_event.is_set():
                    battle_batch += 1
                    start = time.time()

                    # ── Weight broadcast polling ─────────────────────────────
                    # The learner periodically pushes a payload onto each
                    # worker's weight_queue containing:
                    #   - "weights": the new model state_dict (CPU tensors)
                    #   - "curriculum": new opponent-mix probabilities
                    #   - "temperature" / "top_p": new exploration knobs
                    #   - "exploiter_paths" / "ghost_paths": disk paths the
                    #     worker may need to load if curriculum mentions them
                    #   - "exploiter_weights" / "victim_weights" (optional):
                    #     in-process exploiter co-training. Exploiter is
                    #     pushed every broadcast; victim only on refresh
                    #     (every victim_refresh_interval main updates).
                    #
                    # We only check once per second (not every iteration) to
                    # avoid lock-contention overhead. Stale weights for ~1 sec
                    # is fine — IMPALA already accepts off-policy data.
                    # ─────────────────────────────────────────────────────────
                    if time.time() - last_weight_check > 1.0:
                        try:
                            while not weight_queue.empty():
                                incoming_payload = weight_queue.get_nowait()
                                if (
                                    isinstance(incoming_payload, dict)
                                    and "weights" in incoming_payload
                                ):
                                    env.update_weights(incoming_payload["weights"])
                                    new_curriculum = incoming_payload.get("curriculum")
                                    if isinstance(new_curriculum, dict):
                                        # Reapply dedicated-worker overrides on every broadcast
                                        if (
                                            config.curriculum.dedicated_vgcbench_workers
                                            > 0
                                        ):
                                            if (
                                                worker_id
                                                < config.curriculum.dedicated_vgcbench_workers
                                            ):
                                                new_curriculum = {
                                                    OpponentPool.VGC_BENCH_BASELINE: 1.0
                                                }
                                            else:
                                                new_curriculum = {
                                                    k: v
                                                    for k, v in new_curriculum.items()
                                                    if k != OpponentPool.VGC_BENCH_BASELINE
                                                }
                                        env.update_curriculum(
                                            new_curriculum,
                                            exploiter_paths=incoming_payload.get(
                                                "exploiter_paths"
                                            ),
                                            ghost_paths=incoming_payload.get(
                                                "ghost_paths"
                                            ),
                                        )
                                    env.update_sampling(
                                        temperature=incoming_payload.get("temperature"),
                                        top_p=incoming_payload.get("top_p"),
                                    )
                                    exploiter_weights = incoming_payload.get(
                                        "exploiter_weights"
                                    )
                                    if exploiter_weights is not None:
                                        env.update_exploiter_weights(exploiter_weights)
                                    victim_weights = incoming_payload.get("victim_weights")
                                    if victim_weights is not None:
                                        env.update_victim_weights(victim_weights)
                                else:
                                    env.update_weights(incoming_payload)
                                if verbose:
                                    logger.debug(
                                        "[MPWorker %d] Updated weights", worker_id
                                    )
                        except Exception:
                            pass
                        last_weight_check = time.time()

                    result = await env.run_battle_batch(battles_per_task)

                    # VGCBench failover: if external challenges repeatedly hang, disable
                    # that opponent type locally so training keeps making progress.
                    if result.had_timeout:
                        timed_out_set = set(result.timed_out_types)
                        if OpponentPool.VGC_BENCH_BASELINE in timed_out_set:
                            consecutive_vgcbench_timeouts += 1
                        else:
                            consecutive_vgcbench_timeouts = 0
                    else:
                        consecutive_vgcbench_timeouts = 0

                    if (
                        not vgcbench_disabled_locally
                        and external_vgcbench_usernames
                        and consecutive_vgcbench_timeouts >= 2
                    ):
                        updated_curriculum = env.get_curriculum()
                        if (
                            updated_curriculum.get(OpponentPool.VGC_BENCH_BASELINE, 0.0)
                            > 0.0
                        ):
                            updated_curriculum[OpponentPool.VGC_BENCH_BASELINE] = 0.0
                            env.update_curriculum(updated_curriculum)
                            vgcbench_disabled_locally = True

                    # ── PHASE 4: TRAJECTORY FORWARDING ───────────────────────────────────────────
                    # Bulk-transfer from the environment's local queue to mp_traj_queue (learner).

                    total_time = time.time() - start
                    transferred = 0
                    for traj in result.trajectories:
                        traj_queue.put(traj)
                        transferred += 1

                    if verbose:
                        logger.debug(
                            "[MPWorker %d] Batch %d: %d battles in %.2fs (%.2f b/s) | Sent %d trajectories... Memory: %s",
                            worker_id,
                            battle_batch,
                            result.battles_completed,
                            total_time,
                            result.battles_completed / total_time if total_time > 0 else 0,
                            transferred,
                            get_memory_usage_mb(),
                        )

                    # Hard failover: if we sampled tasks but produced zero trajectories for
                    # many consecutive batches, crash the worker so the main process can
                    # checkpoint and exit instead of stalling indefinitely.
                    if result.sampled_types and not result.completed_counts:
                        consecutive_zero_completion_batches += 1
                    else:
                        consecutive_zero_completion_batches = 0

                    batch_duration_s = time.time() - start
                    batch_duration_window.append(batch_duration_s)
                    timeout_window.append(1 if result.had_timeout else 0)
                    zero_completion_window.append(
                        1 if (result.sampled_types and not result.completed_counts) else 0
                    )

                    # ── Hard failover ────────────────────────────────────────
                    # If we sampled tasks but produced zero trajectories for
                    # many consecutive batches, something is seriously wrong
                    # (Showdown server died, network meltdown, etc.). Crash
                    # the worker on purpose so the trainer notices, saves a
                    # checkpoint, and exits cleanly — much better than
                    # silently stalling forever.
                    # ─────────────────────────────────────────────────────────
                    if (
                        consecutive_zero_completion_batches
                        >= zero_completion_failover_threshold
                    ):
                        raise RuntimeError(
                            "Worker entered sustained zero-completion state "
                            f"({consecutive_zero_completion_batches} consecutive batches)."
                        )

                    if battle_batch % 10 == 0:
                        gc.collect()
                        if battle_batch % 50 == 0:
                            logger.info(
                                "[MPWorker %d] Batch %d memory: %s",
                                worker_id,
                                battle_batch,
                                get_memory_usage_mb(),
                            )

                    # Rebuild on timeout or lingering unfinished battles; otherwise reset cleanly.
                    if (
                        result.had_timeout
                        or env.get_unfinished_summary()["total_unfinished"] > 0
                    ):
                        await env.rebuild()
                        continue

                    env.reset_battles()
            finally:
                await env.teardown()

        try:
            loop.run_until_complete(run_battles(num_battles_per_pair))
        finally:
            # Stop the centralized inference client cleanly so its
            # response-dispatcher thread exits and pending submits get
            # cancelled rather than hanging on shutdown.
            if main_inference_client is not None:
                main_inference_client.stop()
        if verbose:
            logger.debug("[MPWorker %d] Finished gracefully", worker_id)

    except Exception as e:
        import traceback

        error_msg = f"[MPWorker {worker_id}] FATAL ERROR: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        try:
            error_queue.put_nowait(
                {
                    "worker_id": worker_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        raise
