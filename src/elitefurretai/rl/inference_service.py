# -*- coding: utf-8 -*-
"""Trainer-side inference service.

Owns a model (or fake echo callable for tests), a request mp.Queue shared
across workers, and a per-worker response mp.Queue dict. Runs a daemon
thread that:

  1. drains as many requests from the request queue as it can (up to
     batch_size, or until batch_timeout elapses since the first item)
  2. runs ONE batched forward over those requests
  3. samples actions on GPU
  4. dispatches per-request InferenceResponse objects to the right
     worker's response queue

This file ships the M1 skeleton: real queue plumbing + thread lifecycle,
but uses a callable hook for "model" so tests can supply a fake echo. M2
swaps in the real model + sampling.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Union

from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import (
    EvictRequest,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)


# Type alias for the "model" hook the service uses to produce responses
# from a list of requests. Swappable for testing (echo, deterministic
# stub) and for real-model wiring (M2).
BatchHandler = Callable[[List[InferenceRequest]], List[InferenceResponse]]


class InferenceService:
    """Centralized batched inference, served from the trainer process.

    One instance per model. Spawn workers with this service's
    `request_queue` and the `response_queues` dict so they can send
    requests in and receive responses back.
    """

    def __init__(
        self,
        name: str,
        batch_handler: BatchHandler,
        request_queue: "torch_mp.Queue[Union[InferenceRequest, EvictRequest]]",
        response_queues: Dict[int, "torch_mp.Queue[InferenceResponse]"],
        batch_size: int = 32,
        batch_timeout: float = 0.005,
    ):
        self.name = name
        self._handler = batch_handler
        self._request_queue = request_queue
        self._response_queues = response_queues
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Diagnostics — same fields the legacy per-player batcher logged,
        # so existing aggregation/observability rolls forward unchanged.
        self._diagnostics: Dict[str, float] = {
            "inference_batches": 0.0,
            "inference_batch_items": 0.0,
            "inference_batch_size_max": 0.0,
            "inference_batches_filled_to_max": 0.0,
            "inference_batches_flushed_timeout": 0.0,
            "inference_handler_seconds": 0.0,
            "inference_evictions": 0.0,
        }
        self._diag_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"InferenceService[{self.name}] already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"InferenceService-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and join. Does NOT drain pending
        requests; callers are responsible for ensuring no producer
        will submit after stop() returns."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    "InferenceService[%s] thread did not exit within %.1fs",
                    self.name,
                    timeout_s,
                )
            self._thread = None

    # ── main loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        # Mirrors BatchInferencePlayer._inference_loop: block for the
        # first request, then opportunistically gather more up to either
        # batch_size or batch_timeout since the first arrived.
        #
        # The queue carries InferenceRequest AND EvictRequest. We
        # process EvictRequests inline (they're cheap dict-pops) and
        # only pack InferenceRequests into batches.
        while not self._stop.is_set():
            try:
                first = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(first, EvictRequest):
                self._handle_evict(first)
                continue
            batch: List[InferenceRequest] = [first]
            deadline = time.monotonic() + self.batch_timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, EvictRequest):
                    # Don't waste the batch; handle and keep gathering.
                    self._handle_evict(item)
                    continue
                batch.append(item)

            self._update_batch_diagnostics(len(batch))
            handler_start = time.monotonic()
            try:
                responses = self._handler(batch)
            except Exception:
                logger.exception(
                    "InferenceService[%s] handler raised; dropping batch of %d",
                    self.name,
                    len(batch),
                )
                continue
            handler_s = time.monotonic() - handler_start
            with self._diag_lock:
                self._diagnostics["inference_handler_seconds"] += handler_s

            self._dispatch_responses(batch, responses)

    def _handle_evict(self, req: EvictRequest) -> None:
        """Forward eviction to the handler if it exposes evict()."""
        evict_fn = getattr(self._handler, "evict", None)
        if callable(evict_fn):
            evict_fn(req.worker_id, req.player_id, req.battle_tag)
        with self._diag_lock:
            self._diagnostics["inference_evictions"] += 1

    def _dispatch_responses(
        self,
        batch: List[InferenceRequest],
        responses: List[InferenceResponse],
    ) -> None:
        if len(responses) != len(batch):
            logger.error(
                "InferenceService[%s] handler returned %d responses for %d "
                "requests; dropping mismatched batch",
                self.name,
                len(responses),
                len(batch),
            )
            return
        for req, resp in zip(batch, responses):
            target = self._response_queues.get(req.worker_id)
            if target is None:
                logger.error(
                    "InferenceService[%s] no response queue for worker_id=%d "
                    "(request_id=%d); response dropped",
                    self.name,
                    req.worker_id,
                    req.request_id,
                )
                continue
            target.put(resp)

    # ── diagnostics ───────────────────────────────────────────────────

    def _update_batch_diagnostics(self, batch_len: int) -> None:
        with self._diag_lock:
            self._diagnostics["inference_batches"] += 1
            self._diagnostics["inference_batch_items"] += batch_len
            if batch_len > self._diagnostics["inference_batch_size_max"]:
                self._diagnostics["inference_batch_size_max"] = float(batch_len)
            if batch_len >= self.batch_size:
                self._diagnostics["inference_batches_filled_to_max"] += 1
            else:
                self._diagnostics["inference_batches_flushed_timeout"] += 1
            n = self._diagnostics["inference_batches"]
            if n % 500 == 0:
                items = self._diagnostics["inference_batch_items"]
                filled = self._diagnostics["inference_batches_filled_to_max"]
                timeout = self._diagnostics["inference_batches_flushed_timeout"]
                logger.warning(
                    "[batch-fill svc=%s] n=%d avg=%.2f max=%d filled%%=%.1f "
                    "timeout%%=%.1f cap=%d",
                    self.name,
                    int(n),
                    items / n,
                    int(self._diagnostics["inference_batch_size_max"]),
                    100.0 * filled / n,
                    100.0 * timeout / n,
                    self.batch_size,
                )

    def get_diagnostics_snapshot(self) -> Dict[str, float]:
        with self._diag_lock:
            return dict(self._diagnostics)


# ─────────────────────────────────────────────────────────────────────
# M1 helper: deterministic echo handler used by the round-trip test.
# Returns action_idx=0 for every request. Doesn't manage hidden state
# (RealModelBatchHandler manages it); tests using this echo handler
# don't exercise hidden-state correctness.
# ─────────────────────────────────────────────────────────────────────


def echo_batch_handler(batch: List[InferenceRequest]) -> List[InferenceResponse]:
    return [
        InferenceResponse(
            request_id=req.request_id,
            action_idx=0,
            log_prob=0.0,
            value=0.0,
        )
        for req in batch
    ]
