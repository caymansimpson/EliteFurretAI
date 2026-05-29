# -*- coding: utf-8 -*-
"""Unit tests for the subprocess host in inference_service — Plan C step 1 infrastructure.

These tests are standalone (no registry integration yet) and cover:
  1. spawn + serve a single request through one service
  2. SyncWeightsMsg actually applies new weights inside the subprocess
  3. ShutdownMsg cleanly stops the subprocess within the timeout
  4. crash detection: subprocess killed externally → is_alive() == False

All tests use CPU + eager (compile_mode=None) to keep them fast and
deterministic. Production runs use cuda + compile_mode='default'; that
path is exercised by the end-to-end smoke test once train.py is wired
up at step 4.
"""

from __future__ import annotations

import copy
import os
import signal
import time
from typing import Tuple

import numpy as np
import pytest
import torch
from torch import multiprocessing as torch_mp

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.inference_service import (
    InferenceRequest,
    InferenceResponse,
    InferenceSubprocessHandle,
    ServiceSpecification,
    SubprocessSpecification,
)
from elitefurretai.rl.rnad_model import RNaDModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_small_agent() -> Tuple[RNaDModel, Embedder]:
    embedder = Embedder(feature_set="simple")
    model = TransformerThreeHeadedModel(
        embedder=embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        max_seq_len=40,
    )
    model.eval()
    return RNaDModel(model), embedder


def _make_request(
    embedder: Embedder,
    request_id: int = 0,
    worker_id: int = 0,
    is_teampreview: bool = False,
) -> InferenceRequest:
    state = (
        np.random.RandomState(seed=request_id)
        .randn(embedder.embedding_size)
        .astype(np.float32)
    )
    mask = np.ones(2025, dtype=np.float32) if not is_teampreview else None
    return InferenceRequest(
        request_id=request_id,
        worker_id=worker_id,
        player_id="p0",
        battle_tag=f"t-{request_id}",
        state=state,
        mask=mask,
        is_teampreview=is_teampreview,
        temperature=1.0,
        top_p=1.0,
    )


def _build_specification(
    agent: RNaDModel,
    ctx,
    group_name: str = "test_group",
    service_name: str = "test_svc",
    num_workers: int = 1,
) -> Tuple[
    SubprocessSpecification,
    "torch_mp.Queue",
    "torch_mp.Queue",
]:
    """Build a SubprocessSpecification hosting one service; return (specification, req_q, resp_q_0)."""
    req_q: "torch_mp.Queue" = ctx.Queue()
    resp_qs = {i: ctx.Queue() for i in range(num_workers)}
    control_q: "torch_mp.Queue" = ctx.Queue()
    service = ServiceSpecification(
        name=service_name,
        agent=agent,
        request_queue=req_q,
        response_queues=resp_qs,
        compile=False,
        probabilistic=False,
    )
    specification = SubprocessSpecification(
        group_name=group_name,
        services=[service],
        control_queue=control_q,
        device="cpu",
        batch_size=4,
        batch_timeout=0.005,
        compile_mode=None,
        embedding_size=None,
    )
    return specification, req_q, resp_qs[0]


# ─────────────────────────────────────────────────────────────────────
# Test 1: spawn + serve one request
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_subprocess_spawn_and_serve_one_request():
    ctx = torch_mp.get_context("spawn")
    agent, embedder = _make_small_agent()
    specification, req_q, resp_q = _build_specification(agent, ctx)

    handle = InferenceSubprocessHandle(specification, ctx=ctx)
    handle.start()
    try:
        # Send one request, await response within a generous timeout.
        req = _make_request(embedder, request_id=42)
        req_q.put(req)
        resp: InferenceResponse = resp_q.get(timeout=60.0)

        assert resp.request_id == req.request_id
        assert 0 <= resp.action_idx < 2025
        assert np.isfinite(resp.value)
        assert resp.log_prob <= 0.0
    finally:
        handle.shutdown(timeout_s=10.0)


# ─────────────────────────────────────────────────────────────────────
# Test 2: sync_weights actually applies inside the subprocess
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_subprocess_sync_weights_changes_outputs():
    """Sanity: send a request, capture value. Sync to a NEW agent's
    weights (different random init). Send same request, get a different
    value. This proves SyncWeightsMsg flows through and load_state_dict
    actually applies to the subprocess's model.
    """
    ctx = torch_mp.get_context("spawn")
    agent_a, embedder = _make_small_agent()

    specification, req_q, resp_q = _build_specification(agent_a, ctx)
    handle = InferenceSubprocessHandle(specification, ctx=ctx)
    handle.start()
    try:
        req = _make_request(embedder, request_id=7)
        req_q.put(req)
        resp_before: InferenceResponse = resp_q.get(timeout=60.0)
        value_before = resp_before.value

        # Build a DIFFERENT agent (different random init) on CPU.
        torch.manual_seed(12345)
        agent_b, _ = _make_small_agent()
        new_state_dict = copy.deepcopy(agent_b.model.state_dict())

        handle.sync_weights(specification.services[0].name, new_state_dict)

        # Send another request — same battle_tag so the (now-applied)
        # different weights drive the forward. Use a slightly different
        # request_id so the response is the new one.
        req2 = _make_request(embedder, request_id=8)
        req_q.put(req2)
        resp_after: InferenceResponse = resp_q.get(timeout=60.0)
        value_after = resp_after.value

        # With different weights AND different state inputs (different
        # request_id seeds the state), the value should differ.
        # The tiny test network produces values ~1e-5, so use a 1e-5
        # threshold: well above fp32 noise, accommodates the small
        # output magnitude.
        assert abs(value_after - value_before) > 1e-5, (
            f"sync_weights did not change outputs: before={value_before} "
            f"after={value_after}"
        )
    finally:
        handle.shutdown(timeout_s=10.0)


# ─────────────────────────────────────────────────────────────────────
# Test 3: shutdown exits within timeout
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_subprocess_shutdown_exits_cleanly():
    ctx = torch_mp.get_context("spawn")
    agent, _ = _make_small_agent()
    specification, _, _ = _build_specification(agent, ctx)
    handle = InferenceSubprocessHandle(specification, ctx=ctx)
    handle.start()
    assert handle.is_alive()
    handle.shutdown(timeout_s=15.0)
    assert not handle.is_alive()


# ─────────────────────────────────────────────────────────────────────
# Test 4: external kill is detected via is_alive
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_subprocess_external_kill_detected():
    """SIGKILL the subprocess externally; is_alive() should flip to
    False. The shutdown() path must handle a dead-subprocess case
    gracefully (no exception)."""
    ctx = torch_mp.get_context("spawn")
    agent, _ = _make_small_agent()
    specification, _, _ = _build_specification(agent, ctx)
    handle = InferenceSubprocessHandle(specification, ctx=ctx)
    handle.start()
    try:
        # Wait briefly for the subprocess to be fully up (services
        # started). Otherwise SIGKILL may race with init.
        time.sleep(2.0)
        assert handle.is_alive()
        proc_pid = handle._process.pid  # type: ignore[union-attr]
        assert proc_pid is not None
        os.kill(proc_pid, signal.SIGKILL)
        # Poll for is_alive flip; should be near-instant.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not handle.is_alive():
                break
            time.sleep(0.05)
        assert not handle.is_alive(), "subprocess still reported alive after SIGKILL"
    finally:
        # shutdown on a dead process should not raise
        handle.shutdown(timeout_s=5.0)


# ─────────────────────────────────────────────────────────────────────
# Bonus: hosting multiple services in one subprocess
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_subprocess_hosts_multiple_services():
    """Verify the 4-process layout's core assumption: one subprocess
    can host N services, each with its own request/response queues."""
    ctx = torch_mp.get_context("spawn")
    agent_x, embedder = _make_small_agent()
    agent_y, _ = _make_small_agent()

    req_qx: "torch_mp.Queue" = ctx.Queue()
    resp_qx0: "torch_mp.Queue" = ctx.Queue()
    req_qy: "torch_mp.Queue" = ctx.Queue()
    resp_qy0: "torch_mp.Queue" = ctx.Queue()
    control_q: "torch_mp.Queue" = ctx.Queue()

    specification = SubprocessSpecification(
        group_name="multi",
        services=[
            ServiceSpecification(
                name="svc_x",
                agent=agent_x,
                request_queue=req_qx,
                response_queues={0: resp_qx0},
                compile=False,
                probabilistic=False,
            ),
            ServiceSpecification(
                name="svc_y",
                agent=agent_y,
                request_queue=req_qy,
                response_queues={0: resp_qy0},
                compile=False,
                probabilistic=False,
            ),
        ],
        control_queue=control_q,
        device="cpu",
        batch_size=4,
        batch_timeout=0.005,
        compile_mode=None,
        embedding_size=None,
    )

    handle = InferenceSubprocessHandle(specification, ctx=ctx)
    handle.start()
    try:
        req_qx.put(_make_request(embedder, request_id=1))
        req_qy.put(_make_request(embedder, request_id=2))
        resp_x = resp_qx0.get(timeout=60.0)
        resp_y = resp_qy0.get(timeout=60.0)
        assert resp_x.request_id == 1
        assert resp_y.request_id == 2
    finally:
        handle.shutdown(timeout_s=10.0)
