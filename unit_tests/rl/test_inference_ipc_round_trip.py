# -*- coding: utf-8 -*-
"""Milestone 1: prove the IPC layer (request → service → response → client
→ future) actually works end-to-end with a fake echo "model".

This test runs entirely in-process (single Python interpreter) but uses
real torch.multiprocessing queues, real threads, and a real asyncio loop
on the client side.

D3-alt note
-----------
The wire payload no longer carries hidden state — the trainer-side
handler manages it. The echo handler doesn't manage hidden either, so
these tests don't validate hidden-state correctness; they only validate
the request/response plumbing. The handler tests
(`test_inference_handler_real_model.py`) cover hidden-state behavior.
"""

from __future__ import annotations

import asyncio
from typing import Dict

import numpy as np
import pytest
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_trainer import (
    InferenceService,
    echo_batch_handler,
)
from elitefurretai.rl.inference_worker import InferenceClient


@pytest.fixture
def ipc_setup():
    """Spin up: 1 service + 2 clients sharing it (simulating 2 workers)."""
    request_queue: torch_mp.Queue = torch_mp.Queue()
    response_queues: Dict[int, torch_mp.Queue] = {
        0: torch_mp.Queue(),
        1: torch_mp.Queue(),
    }
    service = InferenceService(
        name="test",
        batch_handler=echo_batch_handler,
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size=8,
        batch_timeout=0.005,
    )
    service.start()

    loop = asyncio.new_event_loop()
    clients = {
        worker_id: InferenceClient(
            worker_id=worker_id,
            request_queue=request_queue,
            response_queue=response_queues[worker_id],
            loop=loop,
        )
        for worker_id in (0, 1)
    }
    for c in clients.values():
        c.start()

    yield service, clients, loop

    for c in clients.values():
        c.stop()
    service.stop()
    loop.close()


def _run_async(loop, coro):
    return loop.run_until_complete(coro)


def test_round_trip_single_request(ipc_setup):
    """One request, one response — the simplest case."""
    _service, clients, loop = ipc_setup
    state = np.zeros(16, dtype=np.float32)
    resp = _run_async(
        loop,
        clients[0].submit(
            state=state,
            mask=None,
            is_teampreview=True,
            player_id="p",
            battle_tag="battle-1",
            temperature=1.0,
            top_p=1.0,
        ),
    )
    assert resp.action_idx == 0
    assert resp.value == 0.0


def test_round_trip_concurrent_requests(ipc_setup):
    """Two clients each fire 100 requests; all 200 resolve correctly with
    no cross-talk between clients."""
    _service, clients, loop = ipc_setup

    async def fire(client, n):
        states = [np.full(16, i, dtype=np.float32) for i in range(n)]
        results = await asyncio.gather(
            *[
                client.submit(
                    state=s,
                    mask=None,
                    is_teampreview=False,
                    player_id="p",
                    battle_tag=f"battle-{i}",
                    temperature=1.0,
                    top_p=1.0,
                )
                for i, s in enumerate(states)
            ]
        )
        # Each worker's request_ids are monotonic and unique.
        ids = [r.request_id for r in results]
        assert ids == sorted(ids), f"Out-of-order ids: {ids[:10]}..."
        assert len(set(ids)) == n
        return results

    async def both():
        r0, r1 = await asyncio.gather(fire(clients[0], 100), fire(clients[1], 100))
        return r0, r1

    r0, r1 = _run_async(loop, both())
    assert len(r0) == 100
    assert len(r1) == 100


def test_evict_request_round_trip():
    """EvictRequest goes through the same queue and is dispatched to the
    handler's evict() method. Uses a tracking handler since echo_batch_handler
    doesn't implement evict."""
    import threading

    from elitefurretai.rl.inference_ipc import InferenceResponse

    evict_calls: list[tuple[int, str, str]] = []
    evict_event = threading.Event()

    class TrackingHandler:
        def __call__(self, batch):
            return [
                InferenceResponse(
                    request_id=req.request_id, action_idx=0, log_prob=0.0, value=0.0
                )
                for req in batch
            ]

        def evict(self, worker_id: int, player_id: str, battle_tag: str) -> None:
            evict_calls.append((worker_id, player_id, battle_tag))
            evict_event.set()

    request_queue: torch_mp.Queue = torch_mp.Queue()
    response_queues: Dict[int, torch_mp.Queue] = {0: torch_mp.Queue()}
    service = InferenceService(
        name="evict-test",
        batch_handler=TrackingHandler(),
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size=8,
        batch_timeout=0.005,
    )
    service.start()

    loop = asyncio.new_event_loop()
    client = InferenceClient(
        worker_id=0,
        request_queue=request_queue,
        response_queue=response_queues[0],
        loop=loop,
    )
    client.start()
    try:
        # Warmup: a normal inference request should NOT trigger evict.
        _run_async(
            loop,
            client.submit(
                state=np.zeros(16, dtype=np.float32),
                mask=None,
                is_teampreview=False,
                player_id="p",
                battle_tag="battle-evict",
                temperature=1.0,
                top_p=1.0,
            ),
        )
        assert evict_calls == []

        # Send an evict and wait for the service thread to dispatch it.
        client.evict("p", "battle-evict")
        assert evict_event.wait(timeout=1.0), "evict not processed within 1s"
        assert evict_calls == [(0, "p", "battle-evict")]
    finally:
        client.stop()
        service.stop()
        loop.close()


def test_clean_shutdown_cancels_pending(ipc_setup):
    """Calling client.stop() must cancel still-awaiting submit calls.

    To remove race against the service draining the queue, we stop the
    service FIRST so requests submitted afterwards can never be served.
    """
    service, clients, loop = ipc_setup
    service.stop()  # No response will ever come.

    async def submit_then_stop_client():
        future = asyncio.create_task(
            clients[0].submit(
                state=np.zeros(16, dtype=np.float32),
                mask=None,
                is_teampreview=False,
                player_id="p",
                battle_tag="battle-stop",
                temperature=1.0,
                top_p=1.0,
            )
        )
        # Let the submit reach `await future` before we stop the client.
        await asyncio.sleep(0.05)
        clients[0].stop()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(future, timeout=1.0)

    _run_async(loop, submit_then_stop_client())
