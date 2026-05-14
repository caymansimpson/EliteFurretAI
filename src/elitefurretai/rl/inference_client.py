# -*- coding: utf-8 -*-
"""Worker-side inference client.

One per InferenceService. The client owns:

  - the request mp.Queue (shared with trainer's service)
  - this worker's response mp.Queue (drained by a daemon thread)
  - a request_id → asyncio.Future mapping (in-flight requests)

Players call `await client.submit(state=..., battle_tag=..., ...)` and
get back the response. Hidden state is NOT shipped on the wire (D3-alt);
the trainer-side handler keeps it keyed by (worker_id, battle_tag) and
the worker just sends the lightweight battle_tag.

Players also call `client.evict(battle_tag)` when a battle ends so the
trainer can free its hidden-state slot.

Design notes
------------
- We keep request_id allocation per-client (per worker × per service).
  Trainer is oblivious to id collisions across workers because it just
  echoes back whatever id arrived.
- We store the asyncio loop at construction time and use
  call_soon_threadsafe — this assumes the client is created on the same
  loop that players will run on (POKE_LOOP).
- On worker teardown, `stop()` cancels every pending Future with
  `CancelledError` so awaiting battle coroutines unblock cleanly.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading
from typing import Dict, Optional, Union

import numpy as np
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import (
    EvictRequest,
    InferenceRequest,
    InferenceResponse,
)

logger = logging.getLogger(__name__)


class InferenceClient:
    """Submits requests to one InferenceService and awaits responses."""

    def __init__(
        self,
        worker_id: int,
        request_queue: "torch_mp.Queue[Union[InferenceRequest, EvictRequest]]",
        response_queue: "torch_mp.Queue[InferenceResponse]",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.worker_id = worker_id
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._loop = loop or asyncio.get_event_loop()
        self._next_id = itertools.count()
        self._pending: Dict[int, asyncio.Future] = {}
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._dispatcher: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._dispatcher is not None:
            raise RuntimeError(f"InferenceClient[w={self.worker_id}] already started")
        self._stop.clear()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name=f"InferenceClient-w{self.worker_id}",
            daemon=True,
        )
        self._dispatcher.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop dispatching. Cancels all pending futures so awaiting
        coroutines unblock with CancelledError."""
        self._stop.set()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=timeout_s)
            self._dispatcher = None
        with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    self._loop.call_soon_threadsafe(
                        lambda f=fut: f.cancel() if not f.done() else None
                    )
            self._pending.clear()

    # ── submit / await ────────────────────────────────────────────────

    async def submit(
        self,
        *,
        state: np.ndarray,
        mask: Optional[np.ndarray],
        is_teampreview: bool,
        player_id: str,
        battle_tag: str,
        temperature: float,
        top_p: float,
    ) -> InferenceResponse:
        """Send a request and await its response.

        Caller is responsible for any timeout (use asyncio.wait_for).
        Hidden state is NOT shipped here; the trainer-side handler keeps
        it keyed by (worker_id, player_id, battle_tag). `player_id`
        distinguishes the two sides of a self-play battle that share the
        same battle_tag.
        """
        request_id = next(self._next_id)
        future: asyncio.Future = self._loop.create_future()
        with self._pending_lock:
            self._pending[request_id] = future
        request = InferenceRequest(
            request_id=request_id,
            worker_id=self.worker_id,
            player_id=player_id,
            battle_tag=battle_tag,
            state=state,
            mask=mask,
            is_teampreview=is_teampreview,
            temperature=temperature,
            top_p=top_p,
        )
        # mp.Queue.put can block; in steady state the trainer drains
        # faster than workers produce. If the queue fills, blocking is
        # the right back-pressure signal.
        self._request_queue.put(request)
        return await future

    def evict(self, player_id: str, battle_tag: str) -> None:
        """Tell the trainer to free hidden state for a finished battle.

        Sync (not async) because it's fire-and-forget — no response. The
        worker calls this from any context (asyncio loop, callback,
        sync teardown) when a battle ends or is dropped. Each side of a
        self-play battle evicts independently.
        """
        self._request_queue.put(
            EvictRequest(
                worker_id=self.worker_id,
                player_id=player_id,
                battle_tag=battle_tag,
            )
        )

    # ── dispatch ──────────────────────────────────────────────────────

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                resp = self._response_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._pending_lock:
                future = self._pending.pop(resp.request_id, None)
            if future is None:
                # Could happen if the request was cancelled before the
                # response arrived (e.g. battle ended, stop() was called).
                # Drop silently — the response is moot.
                continue
            self._loop.call_soon_threadsafe(self._resolve_future, future, resp)

    @staticmethod
    def _resolve_future(future: asyncio.Future, resp: InferenceResponse) -> None:
        if not future.done():
            future.set_result(resp)
