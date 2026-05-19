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
from typing import Any, Dict, List, Optional, cast

import psutil
import torch

from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.engine.vgc_environment import VGCEnvironment
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.etl.system_utils import suppress_third_party_warnings
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.rnad_model import RNaDAgent

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
    # etc.).
    main_inference_request_queue: Optional[MPQueue] = None,
    main_inference_response_queue: Optional[MPQueue] = None,
    # Bundle of (req_q, resp_q) keyed by model name for ALL registered
    # models in the trainer's ModelRegistry. Workers wrap this in
    # WorkerInferenceClients so the factory can hot-swap
    # opponent.inference_client between main / bc / ghost slots / etc.
    # When None (legacy mode), only the singular
    # main_inference_request_queue path is used (back-compat shim).
    queues_by_model: Optional[Dict[str, Any]] = None,
    initial_active_ghost_slots: List[int] = [],
    initial_active_exploiter_slots: List[int] = [],
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
        base_team_path = config.curriculum.base_team_path
        num_battles_per_pair = config.hardware.num_battles_per_pair
        curriculum = config.curriculum.curriculum_weights
        external_vgcbench_usernames = VGCBenchManager.USERNAMES
        external_vgcbench_startup_wait_s = VGCBenchManager.STARTUP_WAIT_S

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
        # Centralized inference skips loading the main model into
        # this worker's process — the trainer-side InferenceService owns
        # it. We still need `embedder` (built above) for featurization
        # and `model_config` for ghost loading paths. Detected from
        # spawn args: when both inference queues are present, build a
        # client and skip the model build.
        assert (
            main_inference_request_queue is not None
            and main_inference_response_queue is not None
        ), "Centralized inference required; main inference queues missing from spawn args"

        model: Optional[Any] = None
        agent: Optional[RNaDAgent] = None
        main_inference_client = None
        worker_inference_clients = None  # set in centralized mode (step 3+)

        from poke_env.concurrency import POKE_LOOP

        from elitefurretai.rl.inference_worker import (
            InferenceClient,
            WorkerInferenceClients,
        )

        # When the trainer passed a per-model queues bundle,
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
                    "[MPWorker %d] Centralized inference (bundle) enabled; models=%s",
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
        del checkpoint
        if verbose:
            logger.debug(
                "[MPWorker %d] Centralized inference ready... Memory: %s",
                worker_id,
                get_memory_usage_mb(),
            )

        bc_agent = None  # BC inference goes through registry now
        exploiter_agent = None  # exploiter inference goes through registry now
        victim_agent = None  # victim inference goes through registry now

        if verbose:
            logger.debug(
                "[MPWorker %d] Loading teams from %s... Memory: %s",
                worker_id,
                base_team_path,
                get_memory_usage_mb(),
            )
        team_repo = TeamRepo(base_team_path)

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
            worker_inference_clients=worker_inference_clients,
        )

        # Close the blind window between spawn and first broadcast: seed
        # the factory with whatever ghost/exploiter slots were active at
        # spawn time.
        if initial_active_ghost_slots or initial_active_exploiter_slots:
            _backend = getattr(env, "_backend", None)
            _wfactory = getattr(_backend, "_factory", None)
            if _wfactory is not None:
                if initial_active_ghost_slots:
                    _wfactory.set_active_ghost_slots(initial_active_ghost_slots)
                if initial_active_exploiter_slots:
                    _wfactory.set_active_exploiter_slots(initial_active_exploiter_slots)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # ── PHASE 3: BATTLE LOOP ──────────────────────────────────────────────────────
        # Each iteration:
        #   1. Poll weight_queue for a learner broadcast. If found, propagate to env:
        #      - model weights (env.update_weights)
        #      - curriculum distribution (env.update_curriculum)
        #      - temperature / top_p (env.update_sampling)
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
                                        env.update_curriculum(new_curriculum)
                                    if (
                                        "active_ghost_slots" in incoming_payload
                                        or "active_exploiter_slots" in incoming_payload
                                    ):
                                        _backend = getattr(env, "_backend", None)
                                        _wfactory = getattr(_backend, "_factory", None)
                                        if _wfactory is not None:
                                            if "active_ghost_slots" in incoming_payload:
                                                _wfactory.set_active_ghost_slots(
                                                    incoming_payload["active_ghost_slots"]
                                                )
                                            if (
                                                "active_exploiter_slots"
                                                in incoming_payload
                                            ):
                                                _wfactory.set_active_exploiter_slots(
                                                    incoming_payload[
                                                        "active_exploiter_slots"
                                                    ]
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
