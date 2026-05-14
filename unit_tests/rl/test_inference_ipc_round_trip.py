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

from elitefurretai.rl.inference_client import InferenceClient
from elitefurretai.rl.inference_service import (
    InferenceService,
    echo_batch_handler,
)


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


def test_diagnostics_record_batches(ipc_setup):
    """Service should accumulate batch counts and at least one batch
    should have been dispatched."""
    _service, clients, loop = ipc_setup

    async def fire():
        await asyncio.gather(
            *[
                clients[0].submit(
                    state=np.zeros(16, dtype=np.float32),
                    mask=None,
                    is_teampreview=False,
                    player_id="p",
                    battle_tag="battle-x",
                    temperature=1.0,
                    top_p=1.0,
                )
                for _ in range(20)
            ]
        )

    _run_async(loop, fire())
    snap = _service.get_diagnostics_snapshot()
    assert snap["inference_batches"] >= 1
    assert snap["inference_batch_items"] == 20
    assert snap["inference_batch_size_max"] >= 1
    assert (
        snap["inference_batches_filled_to_max"]
        + snap["inference_batches_flushed_timeout"]
        == snap["inference_batches"]
    )


def test_evict_request_round_trip(ipc_setup):
    """EvictRequest goes through the same queue, doesn't appear in
    inference batches, increments evictions counter."""
    service, clients, loop = ipc_setup

    # Drive a few inference requests first to establish a baseline.
    async def warmup():
        await clients[0].submit(
            state=np.zeros(16, dtype=np.float32),
            mask=None,
            is_teampreview=False,
            player_id="p",
            battle_tag="battle-evict",
            temperature=1.0,
            top_p=1.0,
        )

    _run_async(loop, warmup())
    pre = service.get_diagnostics_snapshot()
    pre_batches = pre["inference_batches"]
    pre_evictions = pre["inference_evictions"]

    # Send an evict; loop briefly so service drains it.
    clients[0].evict("p", "battle-evict")

    # Service drains in its background thread; busy-wait briefly.
    import time as _time

    deadline = _time.monotonic() + 1.0
    while service.get_diagnostics_snapshot()["inference_evictions"] == pre_evictions:
        if _time.monotonic() > deadline:
            raise AssertionError("evict not processed within 1s")
        _time.sleep(0.005)

    post = service.get_diagnostics_snapshot()
    assert post["inference_evictions"] == pre_evictions + 1
    # Eviction must NOT count as an inference batch.
    assert post["inference_batches"] == pre_batches


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
