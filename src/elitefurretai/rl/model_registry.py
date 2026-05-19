# -*- coding: utf-8 -*-
"""Trainer-side registry of named inference models.

`ModelRegistry` is the centralized-inference equivalent of "the trainer
knows about all the models it owns." For each registered model it stands
up a `RealModelBatchHandler` + `InferenceService` + per-worker mp.Queues,
optionally compiles + warms up the inference path, and exposes a uniform
API for adding / weight-syncing / shutting down model-driven services.

Why this exists
---------------
Centralized inference ships per-model services (one for main, one for
BC, optionally one per exploiter/victim/ghost). Doing this manually
would mean threading a `<name>_inference_client` parameter through
several files per model type. The registry collapses that into:

    registry.register("main", main_agent)
    registry.register("bc", bc_agent)
    for slot in range(max_ghosts):
        registry.register(f"ghost_{slot}", initial_ghost_agent)
    registry.start_all()

Adding a new model type post-registration is `registry.register(name,
agent)` plus a one-line hot-swap branch in `assign_opponent_role`.

What this owns
--------------
- A `Dict[str, InferenceService]` keyed by model name (in-process only).
- A `Dict[str, InferenceSubprocessHandle]` keyed by group name (Plan C).
- A `Dict[str, mp.Queue]` of request queues (one shared per model).
- A `Dict[str, List[mp.Queue]]` of response queues (one per worker per
  model).
- A `Dict[str, RNaDAgent]` of raw agent references (so `sync_weights`
  can update model parameters in place for in-process services; for
  subprocess services, the trainer keeps a CPU shadow copy so future
  `sync_weights` calls have the latest weights to ship over).

Lifecycle
---------
1. Trainer constructs the registry once at startup.
2. Trainer registers each active model name → agent (optionally with a
   `process_group` if the service should live in a subprocess).
3. Trainer calls `registry.start_all()` once. This starts in-process
   service threads AND spawns subprocesses for any non-default groups.
   No service serves traffic before this call (fixes the registration
   race that bit us in Run C).
4. Trainer calls `queues_for_workers()` to get the bundle to pass into
   each `mp.Process` spawn args. Workers wrap their slice in
   `WorkerInferenceClients`.
5. Each broadcast tick, trainer calls `sync_weights(name, state_dict)`
   for any model whose weights have changed. The registry routes the
   sync to either the in-process model or the right subprocess.
6. On shutdown, trainer calls `stop_all()` — stops in-process services
   AND shuts down subprocesses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import InferenceResponse
from elitefurretai.rl.inference_subprocess import (
    InferenceSubprocessHandle,
    ServiceSpec,
    SubprocessSpec,
)
from elitefurretai.rl.inference_trainer import InferenceService, RealModelBatchHandler
from elitefurretai.rl.rnad_model import RNaDAgent

logger = logging.getLogger(__name__)


@dataclass
class _PendingInProcessService:
    """An in-process service that's been registered but not yet started.

    Held by `_pending_in_process` between `register()` and `start_all()`.
    """

    name: str
    agent: RNaDAgent
    service: InferenceService
    handler: RealModelBatchHandler


@dataclass
class _PendingSubprocessService:
    """A subprocess-bound service waiting to be packaged into a SubprocessSpec.

    Held by `_pending_subprocess[group]` between `register()` and `start_all()`.
    """

    name: str
    agent: RNaDAgent  # kept on CPU; pickled into the subprocess at spawn
    compile: bool
    probabilistic: bool
    request_queue: "torch_mp.Queue"
    response_queues: Dict[int, "torch_mp.Queue[InferenceResponse]"]


class ModelRegistry:
    """Owns trainer-side InferenceServices keyed by model name.

    Supports two service backends:
      - In-process: an `InferenceService` thread inside the trainer process.
        Lowest latency, easiest to debug; shares the trainer's GIL.
      - Subprocess: an `InferenceSubprocessHandle` spawning a dedicated
        Python process that hosts one or more services. Higher fixed
        cost (~1 GB RSS, ~150 MB GPU per subprocess) but escapes the
        trainer's GIL — used by Plan C to recover throughput when many
        services run concurrently.

    Pick the backend via `register(..., process_group=...)`. `None`
    (default) keeps the service in-process. Any other string groups
    services into a shared subprocess (e.g. `process_group='ghosts'`).
    """

    def __init__(
        self,
        num_workers: int,
        batch_size: int = 32,
        batch_timeout: float = 0.005,
        device: str = "cpu",
        compile_mode: Optional[str] = None,
        embedding_size: Optional[int] = None,
    ):
        if compile_mode and embedding_size is None:
            raise ValueError(
                "embedding_size is required when compile_mode is set "
                "(needed for warmup forward passes)"
            )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.device = device
        self.compile_mode = compile_mode
        self.embedding_size = embedding_size

        # Shared across backends: every registered service has a request
        # queue and N response queues, regardless of where it runs.
        self._request_queues: Dict[str, "torch_mp.Queue"] = {}
        self._response_queues: Dict[str, List["torch_mp.Queue"]] = {}

        # In-process backend: built at register(), started at start_all().
        self._pending_in_process: Dict[str, _PendingInProcessService] = {}
        self._services: Dict[str, InferenceService] = {}
        self._handlers: Dict[str, RealModelBatchHandler] = {}

        # Subprocess backend: ServiceSpecs accumulate per group between
        # register() and start_all(). At start_all(), each group is
        # packaged into one SubprocessSpec + InferenceSubprocessHandle.
        self._pending_subprocess: Dict[str, List[_PendingSubprocessService]] = {}
        self._subprocesses: Dict[str, InferenceSubprocessHandle] = {}
        # Control queue per group, created at register() so spec captures it.
        self._control_queues: Dict[str, "torch_mp.Queue"] = {}

        # Shared: name → owning group ("__in_process__" or group name).
        # Set at register() so sync_weights / has_name etc. can route.
        self._service_to_group: Dict[str, str] = {}

        # CPU shadow copy of agents — needed for sync_weights on
        # subprocess services (we keep weights here so we can ship updated
        # state_dicts over the control queue).
        self._raw_agents: Dict[str, RNaDAgent] = {}

        # Multiprocessing context — used for queues and Process spawning.
        # `spawn` is required for CUDA-safe forks; pinned here to keep
        # queue ↔ process compatibility.
        self._mp_ctx = torch_mp.get_context("spawn")

        self._started = False

    # Sentinel for in-process group
    _IN_PROCESS = "__in_process__"

    # ── lifecycle ─────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        agent: RNaDAgent,
        *,
        probabilistic: bool = True,
        compile: bool = True,
        process_group: Optional[str] = None,
    ) -> None:
        """Register a model. Idempotent NOT supported — re-registering
        the same name raises. Use `sync_weights(name, ...)` to update
        weights of an already-registered model.

        Parameters
        ----------
        name : str
            Unique service name (e.g. "main", "ghost_0").
        agent : RNaDAgent
            The model wrapped as an RNaDAgent. For in-process services,
            moved to the registry's device. For subprocess services,
            kept on CPU; the subprocess moves it to its device at
            startup.
        probabilistic : bool
            Forwarded to RealModelBatchHandler.
        compile : bool
            Whether to apply torch.compile (subject to registry's
            compile_mode being non-None).
        process_group : Optional[str]
            If None (default), the service runs in-process. Any other
            string groups this service into a subprocess of that name
            (e.g. "ghosts" or "snaps"). All services sharing a
            process_group land in the same subprocess.
        """
        if self._started:
            raise RuntimeError(
                f"Cannot register '{name}' after start_all() has been called"
            )
        if name in self._service_to_group:
            raise ValueError(
                f"Model '{name}' already registered. Use sync_weights() "
                "to update weights without re-registering."
            )

        # Allocate queues common to both backends.
        request_q: "torch_mp.Queue" = self._mp_ctx.Queue()
        response_qs: List["torch_mp.Queue"] = [
            self._mp_ctx.Queue() for _ in range(self.num_workers)
        ]
        response_dict: Dict[int, "torch_mp.Queue[InferenceResponse]"] = {
            i: q for i, q in enumerate(response_qs)
        }
        self._request_queues[name] = request_q
        self._response_queues[name] = response_qs
        self._raw_agents[name] = agent

        if process_group is None:
            self._register_in_process(
                name, agent, probabilistic, compile, request_q, response_dict
            )
            self._service_to_group[name] = self._IN_PROCESS
        else:
            self._register_subprocess(
                name,
                agent,
                probabilistic,
                compile,
                request_q,
                response_dict,
                process_group,
            )
            self._service_to_group[name] = process_group

        logger.info(
            "Registry: registered '%s' (group=%s, compile=%s, device=%s)",
            name,
            process_group or "in-process",
            compile and self.compile_mode is not None,
            self.device,
        )

    def _register_in_process(
        self,
        name: str,
        agent: RNaDAgent,
        probabilistic: bool,
        compile: bool,
        request_q: "torch_mp.Queue",
        response_dict: Dict[int, "torch_mp.Queue[InferenceResponse]"],
    ) -> None:
        """Build a compiled+warmed agent + handler + service in this
        process; defer service.start() to start_all()."""
        agent.model.to(self.device)
        agent.model.eval()

        agent_for_handler: RNaDAgent
        if self.compile_mode and compile:
            logger.info(
                "Registry: compiling model '%s' with mode=%s dynamic=True device=%s",
                name,
                self.compile_mode,
                self.device,
            )
            compiled = cast(
                RNaDAgent,
                torch.compile(agent, mode=self.compile_mode, dynamic=True),
            )
            assert self.embedding_size is not None
            # Warm dynamo across the production batch-size envelope so
            # first real requests don't trigger an in-band recompile
            # (which would hold _COMPILE_LOCK and freeze all other
            # services for 30-90s — see 2026-05-15-00-12 plan doc).
            # x is always (B, 1, E): a single new turn's input.
            # Sequence growth lives in the ctx hidden tensor; only B
            # varies across production batches.
            with torch.no_grad():
                for B in (1, 4, 16, 32):
                    x = torch.zeros(B, 1, self.embedding_size, device=self.device)
                    _, _, _, _, ctx = compiled(x, None)
                    compiled(x, ctx)
            agent_for_handler = compiled
        else:
            agent_for_handler = agent

        handler = RealModelBatchHandler(
            agent_for_handler,
            device=self.device,
            probabilistic=probabilistic,
        )
        service = InferenceService(
            name=name,
            batch_handler=handler,
            request_queue=request_q,
            response_queues=response_dict,
            batch_size=self.batch_size,
            batch_timeout=self.batch_timeout,
        )
        self._pending_in_process[name] = _PendingInProcessService(
            name=name, agent=agent, service=service, handler=handler
        )

    def _register_subprocess(
        self,
        name: str,
        agent: RNaDAgent,
        probabilistic: bool,
        compile: bool,
        request_q: "torch_mp.Queue",
        response_dict: Dict[int, "torch_mp.Queue[InferenceResponse]"],
        process_group: str,
    ) -> None:
        """Queue this service for inclusion in `process_group`'s subprocess.

        The agent stays on CPU here — the subprocess will move it to
        its device + compile at startup.
        """
        agent.model.cpu()
        agent.model.eval()

        # Create the control queue lazily for new groups.
        if process_group not in self._control_queues:
            self._control_queues[process_group] = self._mp_ctx.Queue()
        self._pending_subprocess.setdefault(process_group, []).append(
            _PendingSubprocessService(
                name=name,
                agent=agent,
                compile=compile,
                probabilistic=probabilistic,
                request_queue=request_q,
                response_queues=response_dict,
            )
        )

    def start_all(self) -> None:
        """Start every registered service.

        - In-process services: `InferenceService.start()` on each.
        - Subprocess groups: package the per-group pending list into a
          `SubprocessSpec`, spawn an `InferenceSubprocessHandle`.

        After this call no further `register()` is permitted.
        """
        if self._started:
            raise RuntimeError("start_all() already called")

        # Start in-process services. They're already compiled + warmed;
        # this only fires the thread that polls the queue.
        for name, pending in self._pending_in_process.items():
            pending.service.start()
            self._services[name] = pending.service
            self._handlers[name] = pending.handler
        self._pending_in_process.clear()

        # Spawn one subprocess per group.
        for group_name, pending_list in self._pending_subprocess.items():
            service_specs = [
                ServiceSpec(
                    name=p.name,
                    agent=p.agent,
                    request_queue=p.request_queue,
                    response_queues=p.response_queues,
                    compile=p.compile,
                    probabilistic=p.probabilistic,
                )
                for p in pending_list
            ]
            spec = SubprocessSpec(
                group_name=group_name,
                services=service_specs,
                control_queue=self._control_queues[group_name],
                device=self.device,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                compile_mode=self.compile_mode,
                embedding_size=self.embedding_size,
            )
            handle = InferenceSubprocessHandle(spec, ctx=self._mp_ctx)
            handle.start()
            self._subprocesses[group_name] = handle
            logger.info(
                "Registry: started subprocess '%s' with %d services: %s",
                group_name,
                len(service_specs),
                [s.name for s in service_specs],
            )
        self._pending_subprocess.clear()

        self._started = True

    def sync_weights(self, name: str, state_dict: Dict[str, Any]) -> None:
        """Update an already-registered model's weights in place.

        Routes to either the in-process agent (`load_state_dict` direct)
        or the owning subprocess via `SyncWeightsMsg`. Safe to call while
        inference is running — load_state_dict copies values in place.

        Side effect: the registry's CPU shadow agent is also updated, so
        future syncs can build off the latest weights and any subprocess
        re-spawn would start from the right place.

        May be called either before or after `start_all()`:
          - Before: the trainer-side shadow is updated. For subprocess
            services, the spawn-time spec's agent reference is the
            same object as `_raw_agents[name]`, so the subprocess
            inherits the latest weights when it starts. Used by train.py
            to pre-fill ghost slot weights from disk at startup.
          - After: in-process services get an in-place load; subprocess
            services additionally get a `SyncWeightsMsg` over the group's
            control queue.
        """
        if name not in self._service_to_group:
            raise KeyError(f"Model '{name}' is not registered")

        t0 = time.perf_counter()
        # Always keep the trainer-side shadow up to date. For in-process
        # services this is the same underlying module the handler holds.
        # For subprocess services this is the CPU shadow that gets
        # pickled into the subprocess at spawn time.
        self._raw_agents[name].model.load_state_dict(state_dict)
        t_shadow = time.perf_counter()

        group = self._service_to_group[name]
        if group == self._IN_PROCESS:
            logger.info(
                "sync_weights[%s]: in-process load=%.1fms",
                name,
                (t_shadow - t0) * 1000.0,
            )
            return
        if not self._started:
            # Subprocess hasn't been spawned yet; the shadow update
            # above will be picked up at spawn time when start_all()
            # pickles the spec's agent reference into the subprocess.
            logger.info(
                "sync_weights[%s]: pre-start shadow update load=%.1fms",
                name,
                (t_shadow - t0) * 1000.0,
            )
            return
        handle = self._subprocesses.get(group)
        if handle is None:
            raise RuntimeError(f"Subprocess group '{group}' not registered")
        # Ship CPU tensors over the control queue, NEVER CUDA tensors.
        # torch.multiprocessing's CUDA-tensor sharing reduction calls
        # `_new_shared_cuda` on the receiving side, which on WSL2 raises
        # `cudaErrorInvalidResourceHandle` because the trainer's CUDA
        # context can't be shared with the subprocess's context. The
        # trainer-side shadow we just updated above is on CPU anyway
        # (subprocess-bound agents are placed there by
        # `_register_subprocess`), so we can pull a clean CPU state_dict
        # from it. This avoids materializing redundant CPU copies of
        # `state_dict`'s CUDA tensors at the call site.
        cpu_state_dict = self._raw_agents[name].model.state_dict()
        t_dict = time.perf_counter()
        handle.sync_weights(name, cpu_state_dict)
        t_enq = time.perf_counter()
        logger.info(
            "sync_weights[%s->%s]: load=%.1fms get_state_dict=%.1fms "
            "enqueue=%.1fms total=%.1fms",
            name,
            group,
            (t_shadow - t0) * 1000.0,
            (t_dict - t_shadow) * 1000.0,
            (t_enq - t_dict) * 1000.0,
            (t_enq - t0) * 1000.0,
        )

    def queues_for_workers(
        self,
    ) -> Dict[str, Tuple["torch_mp.Queue", List["torch_mp.Queue"]]]:
        """Bundle queue handles for spawning workers.

        Returns `{name: (request_queue, [response_queue per worker])}`
        for every registered service (both in-process and subprocess).
        Workers route to per-name queues identically regardless of
        backend.
        """
        return {
            name: (self._request_queues[name], self._response_queues[name])
            for name in self._service_to_group
        }

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Snapshot of per-model service diagnostics (in-process only).

        Subprocess services don't yet surface diagnostics to the trainer
        — return only in-process names. If diagnostics from subprocess
        services become important, the control_queue protocol can grow
        a `DiagnosticsRequestMsg` round-trip; left as future work.
        """
        return {
            name: svc.get_diagnostics_snapshot() for name, svc in self._services.items()
        }

    def names(self) -> List[str]:
        """All registered service names (both backends)."""
        return list(self._service_to_group.keys())

    def stop_all(self, timeout_s: float = 5.0) -> None:
        """Stop every registered service. Idempotent."""
        # In-process services first (cheap, predictable).
        for name, svc in list(self._services.items()):
            svc.stop(timeout_s=timeout_s)
        self._services.clear()

        # Subprocesses: each owns multiple services, shutdown is a
        # control-queue message + join.
        for handle in list(self._subprocesses.values()):
            handle.shutdown(timeout_s=max(timeout_s, 15.0))
        self._subprocesses.clear()

        self._handlers.clear()
        self._request_queues.clear()
        self._response_queues.clear()
        self._raw_agents.clear()
        self._service_to_group.clear()
        self._control_queues.clear()
        self._pending_in_process.clear()
        self._pending_subprocess.clear()
        self._started = False
