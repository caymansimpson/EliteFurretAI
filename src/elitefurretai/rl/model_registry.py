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

Adding a new model type post-registration is `registry.register(name,
agent)` plus a one-line hot-swap branch in `assign_opponent_role`.

What this owns
--------------
- A `Dict[str, InferenceService]` keyed by model name.
- A `Dict[str, mp.Queue]` of request queues (one shared per model).
- A `Dict[str, List[mp.Queue]]` of response queues (one per worker per
  model).
- A `Dict[str, RNaDAgent]` of raw agent references (so `sync_weights`
  can update model parameters in place between training updates).

Lifecycle
---------
1. Trainer constructs the registry once at startup.
2. Trainer registers each active model name → agent.
3. Trainer calls `queues_for_workers()` to get the bundle to pass into
   each `mp.Process` spawn args. Workers wrap their slice in
   `WorkerInferenceClients`.
4. Each broadcast tick, trainer calls `sync_weights(name, state_dict)`
   for any model whose weights have changed (typically just the main
   live agent and any live exploiter).
5. On shutdown, trainer calls `stop_all()`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import InferenceResponse
from elitefurretai.rl.inference_trainer import InferenceService, RealModelBatchHandler
from elitefurretai.rl.players import RNaDAgent

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Owns trainer-side InferenceServices keyed by model name."""

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

        # name → resources
        self._services: Dict[str, InferenceService] = {}
        self._request_queues: Dict[str, "torch_mp.Queue"] = {}
        self._response_queues: Dict[str, List["torch_mp.Queue"]] = {}
        self._handlers: Dict[str, RealModelBatchHandler] = {}
        self._raw_agents: Dict[str, RNaDAgent] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        agent: RNaDAgent,
        *,
        probabilistic: bool = True,
        compile: bool = True,
    ) -> None:
        """Register a model. Builds + starts an InferenceService for it.

        Idempotent NOT supported — re-registering the same name raises.
        Use `sync_weights(name, ...)` to update weights of an
        already-registered model.

        `compile=False` skips compile for this specific model even when
        the registry's `compile_mode` is set. Use this to work around the
        torch.compile + concurrent-service dynamo race observed when
        multiple compiled models run in parallel inference threads —
        compile only the highest-traffic model (typically "main"); leave
        secondary models eager. Per-call latency is higher but per-model
        traffic share usually makes that acceptable.
        """
        if name in self._services:
            raise ValueError(
                f"Model '{name}' already registered. Use sync_weights() "
                "to update weights without re-registering."
            )

        # Move agent to the registry's device + eval mode.
        agent.model.to(self.device)
        agent.model.eval()

        # Optionally compile + warm up. The compiled wrapper feeds the
        # handler; the raw agent stays in `_raw_agents` for sync_weights.
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
            with torch.no_grad():
                x = torch.zeros(1, 1, self.embedding_size, device=self.device)
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
        request_q: "torch_mp.Queue" = torch_mp.Queue()
        response_qs: List["torch_mp.Queue"] = [
            torch_mp.Queue() for _ in range(self.num_workers)
        ]
        response_dict: Dict[int, "torch_mp.Queue[InferenceResponse]"] = {
            i: q for i, q in enumerate(response_qs)
        }
        service = InferenceService(
            name=name,
            batch_handler=handler,
            request_queue=request_q,
            response_queues=response_dict,
            batch_size=self.batch_size,
            batch_timeout=self.batch_timeout,
        )
        service.start()

        self._services[name] = service
        self._request_queues[name] = request_q
        self._response_queues[name] = response_qs
        self._handlers[name] = handler
        self._raw_agents[name] = agent
        logger.info("Registry: registered '%s' on device=%s", name, self.device)

    def sync_weights(self, name: str, state_dict: Dict[str, Any]) -> None:
        """Update an already-registered model's weights in place.

        Called by the trainer at broadcast cadence — for the main live
        agent every checkpoint_interval, for exploiter every exploiter
        update, for victim on refresh, etc.

        Safe to call while inference is running: load_state_dict copies
        values in place. A mid-flight forward may briefly use a mix of
        old/new params — same off-policy semantics as the legacy
        per-worker broadcast. PPO importance ratios tolerate this.
        """
        if name not in self._raw_agents:
            raise KeyError(f"Model '{name}' is not registered")
        self._raw_agents[name].model.load_state_dict(state_dict)

    def queues_for_workers(
        self,
    ) -> Dict[str, Tuple["torch_mp.Queue", List["torch_mp.Queue"]]]:
        """Bundle queue handles for spawning workers.

        Returns `{name: (request_queue, [response_queue per worker])}`.
        Trainer slices per worker_id when constructing spawn args:

            all_queues = registry.queues_for_workers()
            for worker_id in range(num_workers):
                per_worker = {
                    name: (req_q, resp_qs[worker_id])
                    for name, (req_q, resp_qs) in all_queues.items()
                }
                mp.Process(target=..., args=(..., per_worker, ...))
        """
        return {
            name: (self._request_queues[name], self._response_queues[name])
            for name in self._services
        }

    def get_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """Snapshot of per-model service diagnostics. Returns
        {model_name: {counter_name: value}}."""
        return {
            name: svc.get_diagnostics_snapshot()
            for name, svc in self._services.items()
        }

    def names(self) -> List[str]:
        return list(self._services.keys())

    def stop_all(self, timeout_s: float = 5.0) -> None:
        """Stop every registered service. Idempotent."""
        for name, svc in list(self._services.items()):
            svc.stop(timeout_s=timeout_s)
        self._services.clear()
        self._request_queues.clear()
        self._response_queues.clear()
        self._handlers.clear()
        self._raw_agents.clear()
