# -*- coding: utf-8 -*-
"""Milestone 3: two complementary correctness tests.

1. **Multi-step equivalence** (in-process). Drive 10 sequential turns
   through both the centralized handler and the legacy-style direct
   model call. Under D3-alt the centralized hidden state is owned by
   `RealModelBatchHandler.hidden_states[(worker_id, battle_tag)]`; the
   legacy comparison threads its own `legacy_hidden` through direct
   model calls. Assert action/log_prob/value match step-by-step.

2. **Multi-process round-trip** (real `mp.Process`). Spawns a child
   process that uses `InferenceClient` to submit requests to the main
   process's `InferenceService`. Validates pickling, `torch.mp.Queue`
   shared-memory tensor transit across the OS process boundary, asyncio
   loop in subprocess, and clean shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as std_mp
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch import multiprocessing as torch_mp

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.inference_ipc import InferenceRequest
from elitefurretai.rl.inference_trainer import (
    InferenceService,
    RealModelBatchHandler,
    echo_batch_handler,
)
from elitefurretai.rl.inference_worker import InferenceClient
from elitefurretai.rl.rnad_model import RNaDModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Multi-step equivalence (in-process)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def small_transformer_agent():
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


def test_multistep_equivalence_with_growing_context(small_transformer_agent):
    """Drive N sequential turns through both paths; assert
    action/log_prob/value agree step-by-step. Catches accumulated drift."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=False)

    n_steps = 10
    embedding_size = embedder.embedding_size
    states = [
        np.random.RandomState(seed=42 + t).randn(embedding_size).astype(np.float32)
        for t in range(n_steps)
    ]
    mask = np.ones(2025, dtype=np.float32)

    # Centralized: handler owns hidden state, keyed by (worker, battle).
    centralized_results: List[Tuple[int, float, float]] = []
    for t in range(n_steps):
        req = InferenceRequest(
            request_id=t,
            worker_id=0,
            player_id="p",
            battle_tag="multistep",
            state=states[t],
            mask=mask,
            is_teampreview=False,
            temperature=1.0,
            top_p=1.0,
        )
        [resp] = handler([req])
        centralized_results.append((resp.action_idx, resp.log_prob, resp.value))

    # Legacy-equivalent path: drive RNaDModel directly the way
    # `_gpu_inference_sync` + `_run_batch` would.
    legacy_results: List[Tuple[int, float, float]] = []
    legacy_hidden = None
    for t in range(n_steps):
        state_t = torch.from_numpy(states[t]).to("cpu").unsqueeze(0).unsqueeze(1).float()
        if legacy_hidden is None:
            h_mask = None
        else:
            h_mask = torch.ones(1, legacy_hidden.shape[1], dtype=torch.bool)
        with torch.no_grad():
            turn_logits, _, values, _, next_hidden = agent(
                state_t, legacy_hidden, mask=None, hidden_mask=h_mask
            )
        log_probs = torch.log_softmax(turn_logits[0, 0], dim=-1).numpy()
        valid_mask = mask.astype(bool)
        log_valid_mass = float(np.log(np.exp(log_probs[valid_mask]).sum()))
        action = int(np.argmax(log_probs))
        log_prob = float(log_probs[action] - log_valid_mass)
        value = float(values[0, 0].cpu().item())
        legacy_results.append((action, log_prob, value))
        legacy_hidden = next_hidden

    for t, (cent, leg) in enumerate(zip(centralized_results, legacy_results)):
        assert cent[0] == leg[0], f"step {t}: action centralized={cent[0]} legacy={leg[0]}"
        assert cent[1] == pytest.approx(leg[1], abs=1e-5), (
            f"step {t}: log_prob diff {abs(cent[1] - leg[1])}"
        )
        assert cent[2] == pytest.approx(leg[2], abs=1e-5), (
            f"step {t}: value diff {abs(cent[2] - leg[2])}"
        )


def test_multistep_equivalence_batched_agrees_with_per_request(
    small_transformer_agent,
):
    """Submitting N requests in ONE batch must produce close outputs
    (modulo float32 noise from batched matmul) to one-at-a-time."""
    agent, embedder = small_transformer_agent
    handler_batch = RealModelBatchHandler(agent, device="cpu", probabilistic=False)
    handler_seq = RealModelBatchHandler(agent, device="cpu", probabilistic=False)

    n = 6
    requests = []
    for i in range(n):
        state = (
            np.random.RandomState(seed=i).randn(embedder.embedding_size).astype(np.float32)
        )
        # Pre-populate prior hidden (length i+1) for batches index >0.
        if i > 0:
            ctx = torch.randn(1, i + 1, 32)
            handler_batch.hidden_states[(0, "p", f"b-{i}")] = ctx.clone()
            handler_seq.hidden_states[(0, "p", f"b-{i}")] = ctx.clone()
        requests.append(
            InferenceRequest(
                request_id=i,
                worker_id=0,
                player_id="p",
                battle_tag=f"b-{i}",
                state=state,
                mask=np.ones(2025, dtype=np.float32),
                is_teampreview=False,
                temperature=1.0,
                top_p=1.0,
            )
        )

    batched = handler_batch(requests)
    sequential = [handler_seq([req])[0] for req in requests]

    for b, s in zip(batched, sequential):
        # Compare action/log_prob/value within tolerance. action_idx is
        # NOT asserted: argmax is discontinuous and batching noise can
        # flip ties (legacy has the same nondeterminism).
        assert b.value == pytest.approx(s.value, abs=1e-3)
        assert b.log_prob == pytest.approx(s.log_prob, abs=1e-3)
    # Stored hidden in both handlers should match in shape and (closely)
    # in values per-key.
    for i in range(n):
        bk = handler_batch.hidden_states[(0, "p", f"b-{i}")]
        sk = handler_seq.hidden_states[(0, "p", f"b-{i}")]
        assert bk.shape == sk.shape
        assert torch.allclose(bk, sk, atol=1e-4), (
            f"b-{i}: hidden max abs diff {(bk - sk).abs().max().item():.2e}"
        )


# ─────────────────────────────────────────────────────────────────────
# Cross-process round-trip (real mp.Process)
# ─────────────────────────────────────────────────────────────────────


def _client_subprocess_main(
    request_q: "torch_mp.Queue",
    response_q: "torch_mp.Queue",
    result_q: "torch_mp.Queue",
    n_requests: int,
) -> None:
    """Top-level function for spawn (must be importable, no closures).

    Subprocess sets up its own asyncio loop + InferenceClient, fires N
    requests concurrently, collects responses, posts a structured result
    onto `result_q` for the parent to verify.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = InferenceClient(
            worker_id=0,
            request_queue=request_q,
            response_queue=response_q,
            loop=loop,
        )
        client.start()

        async def submit_all():
            results = await asyncio.gather(
                *[
                    client.submit(
                        state=np.full(8, i, dtype=np.float32),
                        mask=None,
                        is_teampreview=False,
                        player_id="p",
                        battle_tag=f"battle-{i}",
                        temperature=1.0,
                        top_p=1.0,
                    )
                    for i in range(n_requests)
                ]
            )
            return [(r.request_id, r.action_idx) for r in results]

        try:
            payload = loop.run_until_complete(submit_all())
            result_q.put(("ok", payload))
        finally:
            client.stop()
            loop.close()
    except Exception as exc:  # pragma: no cover — only on test failure
        result_q.put(("error", repr(exc)))


@pytest.mark.slow
def test_cross_process_round_trip():
    """Real subprocess → trainer-process service → response queue →
    subprocess. First test that exercises pickling + tensor IPC across
    a real OS process boundary.
    """
    try:
        std_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    request_q: "torch_mp.Queue" = torch_mp.Queue()
    response_q: "torch_mp.Queue" = torch_mp.Queue()
    result_q: "torch_mp.Queue" = torch_mp.Queue()

    service = InferenceService(
        name="mp-round-trip",
        batch_handler=echo_batch_handler,
        request_queue=request_q,
        response_queues={0: response_q},
        batch_size=4,
        batch_timeout=0.005,
    )
    service.start()

    proc = torch_mp.Process(
        target=_client_subprocess_main,
        args=(request_q, response_q, result_q, 10),
    )
    proc.start()
    proc.join(timeout=60)

    try:
        assert proc.exitcode == 0, (
            f"subprocess exited with {proc.exitcode} (None means timeout)"
        )

        status, payload = result_q.get(timeout=5)
        assert status == "ok", f"subprocess raised: {payload}"
        assert len(payload) == 10
        ids = [item[0] for item in payload]
        assert ids == sorted(ids)
        assert len(set(ids)) == 10
        for _id, action in payload:
            assert action == 0  # echo
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        service.stop()
