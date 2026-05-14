# -*- coding: utf-8 -*-
"""Worker-side bundle of per-model inference clients.

Counterpart to `ModelRegistry`: the trainer constructs the registry,
slices its queues per worker, and spawns each worker with a per-model
queues bundle. Inside the worker, `WorkerInferenceClients` wraps that
bundle into one `InferenceClient` per model name.

Players hot-swap between models by reading from this bundle:

    clients.get("main")     # main agent
    clients.get("bc")       # BC opponent
    clients.get("ghost_2")  # ghost slot 2

The bundle owns lifecycle of the underlying `InferenceClient`s — start
them up at construction, stop them all on teardown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Tuple

from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_client import InferenceClient

logger = logging.getLogger(__name__)


class WorkerInferenceClients:
    """One per worker. Holds an InferenceClient per registered model name."""

    def __init__(
        self,
        worker_id: int,
        queues_by_model: Dict[
            str, Tuple["torch_mp.Queue", "torch_mp.Queue"]
        ],
        loop: asyncio.AbstractEventLoop,
    ):
        """`queues_by_model` should be the per-worker slice produced by
        the trainer:

            all_queues = registry.queues_for_workers()
            per_worker = {
                name: (req_q, resp_qs[worker_id])
                for name, (req_q, resp_qs) in all_queues.items()
            }
        """
        self.worker_id = worker_id
        self._clients: Dict[str, InferenceClient] = {}
        for name, (req_q, resp_q) in queues_by_model.items():
            client = InferenceClient(
                worker_id=worker_id,
                request_queue=req_q,
                response_queue=resp_q,
                loop=loop,
            )
            client.start()
            self._clients[name] = client
        logger.debug(
            "WorkerInferenceClients[w=%d]: %d clients (%s)",
            worker_id,
            len(self._clients),
            ", ".join(self._clients.keys()),
        )

    def get(self, model_name: str) -> InferenceClient:
        """Return the InferenceClient for `model_name`. Raises KeyError
        if the model wasn't registered (catches typos and missing
        opt-in registrations early)."""
        if model_name not in self._clients:
            raise KeyError(
                f"No inference client for '{model_name}' in worker "
                f"{self.worker_id}. Registered: {sorted(self._clients)}"
            )
        return self._clients[model_name]

    def has(self, model_name: str) -> bool:
        return model_name in self._clients

    def names(self) -> List[str]:
        return list(self._clients.keys())

    def stop_all(self, timeout_s: float = 2.0) -> None:
        """Stop every client's dispatcher thread. Idempotent."""
        for name, client in list(self._clients.items()):
            client.stop(timeout_s=timeout_s)
        self._clients.clear()
