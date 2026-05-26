"""
worker.py — Actor process for IMPALA-style distributed RL

What this file is
-----------------
The body of each worker subprocess. The trainer spawns one of these per
config.hardware.num_workers; each runs a long-lived loop that plays battles
and ships completed trajectories back to the learner.

In EliteFurretAI's training loop, computation is split between:

  LEARNER (train.py main process, GPU)
    - Owns the policy weights and the ModelRegistry of InferenceServices.
    - Batches completed trajectories and runs gradient updates.
    - Each checkpoint tick: syncs new weights into the InferenceServices
      via registry.sync_weights AND broadcasts a control payload
      (curriculum, sampling knobs, active ghost/exploiter slot lists) to
      every worker over control_queue.

  ACTORS (this file, N worker processes, CPU)
    - Hold NO policy of their own. Each inference call routes through a
      WorkerInferenceClients bundle whose mp.Queues hit the trainer-side
      InferenceService for the relevant model (main / bc / ghost_N /
      exploiter / victim / exploiter_snap_N).
    - Play battles against various opponents.
    - Send completed trajectories to the learner via mp_traj_queue.
    - Drain control_queue each iteration to pick up new curriculum,
      sampling knobs (temperature, top_p), and active ghost/exploiter
      slot lists. The control queue is the trainer→worker CONTROL plane
      only — policy weight refreshes don't flow through it; they happen
      trainer-side via registry.sync_weights into the InferenceServices
      workers route requests through.

This is an IMPALA-style architecture (Espeholt et al. 2018). The key property:
actors are always slightly off-policy (they sample from a snapshot of the
inference service while the learner is one or more updates ahead). This is
intentional and handled by importance-sampling corrections in the learner.

Why multiprocessing instead of threading
----------------------------------------
Python's GIL would serialize per-actor work under threads. One OS process
per worker gives each its own GIL and interpreter, so we get true CPU
parallelism. The cost is that interprocess data (trajectory shipping,
broadcast payloads, inference request/response) goes through pickled
mp.Queue traffic instead of shared memory — worth it for the ~2-3×
throughput vs threads.

Process lifecycle
-----------------
The worker has three phases:
  PHASE 1: Bootstrap. Read embedder feature_set from the `model_config`
           spawn arg and build the Embedder. Skip constructing the policy
           itself — centralized inference owns it in the trainer process.
           Ghost / exploiter-snapshot weight loads happen trainer-side
           via registry.sync_weights, not in the worker.
  PHASE 2: Environment setup. Build a VGCEnvironment (over the Showdown
           websocket backend) and wire it to the WorkerInferenceClients
           bundle handed in via spawn args.
  PHASE 3: Battle loop. Each iteration: drain control_queue and apply any
           new curriculum/temperature/active-slot updates → run one
           batch of battles via env.run_battle_batch() → push completed
           trajectories to the learner.
"""

import asyncio
import gc
import logging
import os
import time
from collections import deque
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Dict, List, cast

import psutil
import torch
from poke_env.concurrency import POKE_LOOP

from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.engine.vgc_environment import VGCEnvironment
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.etl.system_utils import suppress_third_party_warnings
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.inference_worker import WorkerInferenceClients
from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.rl_utils import format_memory_bytes, setup_logging

logger = logging.getLogger(__name__)


def mp_worker_process(
    worker_id: int,
    server_port: int,
    model_config: Dict[str, Any],
    traj_queue: MPQueue,
    control_queue: MPQueue,
    error_queue: MPQueue,
    stop_event: MPEvent,
    run_id: str,
    config: RNaDConfig,
    # Bundle of (req_q, resp_q) keyed by model name for ALL registered
    # models in the trainer's ModelRegistry. Workers wrap this in
    # WorkerInferenceClients so the factory can hot-swap
    # opponent.inference_client between main / bc / ghost slots / etc.
    queues_by_model: Dict[str, Any] = {},
    initial_active_ghost_slots: List[int] = [],
    initial_active_exploiter_slots: List[int] = [],
):
    """
    Multiprocessing worker process for true parallel RL data collection.

    Each worker process has:
    - Its own Python GIL (true parallelism)
    - Its own asyncio event loop
    - No policy of its own — inference routes through WorkerInferenceClients
      to the trainer-side InferenceService (see `queues_by_model`).

    Detailed per-iteration logs are emitted at DEBUG level on the
    `elitefurretai.rl.worker` logger; raise that logger's level to surface
    them at runtime.

    Args:
        worker_id: Unique worker identifier
        server_port: Showdown server port
        model_config: Model configuration dict (used to derive Embedder feature set)
        traj_queue: Multiprocessing queue for sending trajectories to learner
        control_queue: Multiprocessing queue carrying trainer→worker control
            updates: curriculum, sampling knobs (temperature, top_p), and
            active ghost/exploiter slot lists.
        error_queue: Multiprocessing queue for reporting errors to main process
        stop_event: Multiprocessing event to signal shutdown
        run_id: Unique run identifier
        config: Worker/runtime configuration
    """

    def get_memory_usage_mb():
        return format_memory_bytes(psutil.Process(os.getpid()).memory_info().rss)

    try:
        setup_logging(force=True)
        suppress_third_party_warnings(suppress_pydantic_field_warnings=True)

        gen = config.curriculum.gen
        base_team_path = config.curriculum.base_team_path
        num_battles_per_pair = config.hardware.num_battles_per_pair
        logger.debug(
            "[Worker %d] Starting (PID: %d)... Initial memory: %s",
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

        # ── PHASE 1: BOOTSTRAP ───────────────────────────────────────────────
        # Under centralized inference the worker never builds a policy — the
        # trainer-side InferenceService owns it. All we need locally is
        # `embedder_feature_set` so featurization matches what the service's
        # model was trained on. Read it from the in-memory `model_config`
        # dict the trainer handed us via spawn args.
        #
        # BC checkpoints store the field flat (embedder_feature_set at top
        # level); RL checkpoints store it nested under `training`. Read
        # either layout, matching trainer-side logic in initialize_training_state.
        if (
            isinstance(model_config.get("training"), dict)
            and "embedder_feature_set" in model_config["training"]
        ):
            embedder_feature_set = model_config["training"]["embedder_feature_set"]
        else:
            embedder_feature_set = model_config.get("embedder_feature_set", "full")
        embedder = Embedder(
            gen=gen,
            feature_set=embedder_feature_set,
            omniscient=False,
        )

        # Bundle one InferenceClient per registered model name (main / bc /
        # ghost_N / exploiter / victim / exploiter_snap_N). Each client is a
        # thin RPC layer over an (request_q, response_q) mp.Queue pair that
        # hits the trainer-side InferenceService.
        #   - worker_id: tags outgoing inference requests so the service
        #     routes each response back to THIS worker's response_q.
        #   - queues_by_model: the per-worker slice of the trainer's
        #     {name: (req_q, resp_qs)} bundle — each entry has this
        #     worker's specific response_q.
        #   - loop=POKE_LOOP: response callbacks must resolve their futures
        #     on the same asyncio loop that runs battles (poke-env's
        #     POKE_LOOP). Pinning here lets battle code `await
        #     client.submit(...)` without cross-loop deadlocks.
        worker_inference_clients = WorkerInferenceClients(
            worker_id=worker_id,
            queues_by_model=cast(Any, queues_by_model),
            loop=POKE_LOOP,
        )
        logger.debug(
            "[Worker %d] Centralized inference enabled and ready... Memory: %s",
            worker_id,
            get_memory_usage_mb(),
        )
        logger.debug(
            "[Worker %d] Loading teams from %s... Memory: %s",
            worker_id,
            base_team_path,
            get_memory_usage_mb(),
        )
        team_repo = TeamRepo(base_team_path)

        # ── PHASE 2: ENVIRONMENT SETUP ────────────────────────────────────────────────
        # VGCEnvironment wraps the Showdown websocket backend behind a uniform
        # interface. The curriculum is read from `config.curriculum.curriculum_weights`
        # inside the backend; subsequent control_queue broadcasts update it
        # via env.update_curriculum().
        logger.debug(
            "[Worker %d] Creating VGCEnvironment... Memory: %s",
            worker_id,
            get_memory_usage_mb(),
        )
        env = VGCEnvironment.from_config(
            config=config,
            worker_id=worker_id,
            embedder=embedder,
            team_repo=team_repo,
            server_port=server_port,
            run_id=run_id,
            worker_inference_clients=worker_inference_clients,
        )

        # Close the blind window between spawn and first broadcast: seed
        # the env with whatever ghost/exploiter slots were active at
        # spawn time. (env.set_active_*_slots no-ops if setup() hasn't
        # built the factory yet — but it has, two lines above.)
        env.set_active_ghost_slots(initial_active_ghost_slots)
        env.set_active_exploiter_slots(initial_active_exploiter_slots)

        # ── PHASE 3: BATTLE LOOP ──────────────────────────────────────────────────────
        # Each iteration:
        #   1. Poll control_queue for a learner broadcast. If found, propagate to env:
        #      - curriculum distribution (env.update_curriculum)
        #      - temperature / top_p (env.update_sampling)
        #      - active ghost / exploiter slot lists (factory.set_active_*_slots)
        #   2. env.run_battle_batch(battles_per_task) → BatchResult
        #   3. Forward trajectories to traj_queue (Phase 4)
        #   4. Handle timeout recovery (env.rebuild()) or reset (env.reset_battles())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run_battles(num_battles_per_pair: int):
            await env.setup()
            if VGCBenchManager.USERNAMES:
                await asyncio.sleep(VGCBenchManager.STARTUP_WAIT_S)

            battle_batch = 0
            last_control_check = time.time()
            consecutive_vgcbench_timeouts = 0
            vgcbench_disabled_locally = False
            consecutive_zero_completion_batches = 0
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

                    # ── Control broadcast polling ────────────────────────────
                    # The learner periodically pushes a control payload onto
                    # each worker's control_queue containing:
                    #   - "curriculum": new opponent-mix probabilities
                    #   - "temperature" / "top_p": new exploration knobs
                    #   - "active_ghost_slots" / "active_exploiter_slots":
                    #     which slots in the registry's snapshot pools are
                    #     currently valid routing targets.
                    #
                    # We check once per second (not every iteration) to avoid
                    # lock-contention overhead. ~1s staleness on the control
                    # plane is fine; IMPALA already accepts off-policy data.
                    # ─────────────────────────────────────────────────────────
                    if time.time() - last_control_check > 1.0:
                        try:
                            while not control_queue.empty():
                                payload = control_queue.get_nowait()
                                if not isinstance(payload, dict):
                                    continue
                                new_curriculum = payload.get("curriculum")
                                if isinstance(new_curriculum, dict):
                                    # Change 7: forward team_distribution_by_format
                                    # alongside curriculum so workers transition
                                    # atomically. Key is absent on broadcasts that
                                    # don't recompute team distribution; pass None
                                    # in that case (factory retains existing state).
                                    env.update_curriculum(
                                        new_curriculum,
                                        team_distribution_by_format=payload.get(
                                            "team_distribution_by_format"
                                        ),
                                    )
                                env.update_sampling(
                                    temperature=payload.get("temperature"),
                                    top_p=payload.get("top_p"),
                                )
                                if "active_ghost_slots" in payload:
                                    env.set_active_ghost_slots(
                                        payload["active_ghost_slots"]
                                    )
                                if "active_exploiter_slots" in payload:
                                    env.set_active_exploiter_slots(
                                        payload["active_exploiter_slots"]
                                    )
                                logger.debug(
                                    "[Worker %d] Applied control payload", worker_id
                                )
                        except Exception:
                            pass
                        last_control_check = time.time()

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
                        and VGCBenchManager.USERNAMES
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

                    logger.debug(
                        "[Worker %d] Batch %d: %d battles in %.2fs (%.2f b/s) | Sent %d trajectories... Memory: %s",
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
                                "[Worker %d] Batch %d memory: %s",
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
            # Stop every centralized inference client cleanly so their
            # response-dispatcher threads exit and pending submits get
            # cancelled rather than hanging on shutdown.
            worker_inference_clients.stop_all()
        logger.debug("[Worker %d] Finished gracefully", worker_id)

    except Exception as e:
        import traceback

        error_msg = f"[Worker {worker_id}] FATAL ERROR: {e}\n{traceback.format_exc()}"
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
