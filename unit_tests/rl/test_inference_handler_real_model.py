# -*- coding: utf-8 -*-
"""Milestone 2: real `RNaDAgent`-backed `BatchHandler` produces sensible
outputs and matches the legacy per-player path numerically.

D3-alt
------
Hidden state lives inside `RealModelBatchHandler.hidden_states`, keyed
by (worker_id, battle_tag). Tests pre-populate that dict to simulate
"this battle has a turn-T context" rather than passing hidden through
the request payload.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.inference_handlers import RealModelBatchHandler
from elitefurretai.rl.inference_ipc import InferenceRequest
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel


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
    return RNaDAgent(model), embedder


def _make_request(
    embedder,
    request_id=0,
    worker_id=0,
    player_id="p",
    battle_tag="t-0",
    is_teampreview=False,
):
    state = np.random.randn(embedder.embedding_size).astype(np.float32)
    mask = np.ones(2025, dtype=np.float32) if not is_teampreview else None
    return InferenceRequest(
        request_id=request_id,
        worker_id=worker_id,
        player_id=player_id,
        battle_tag=battle_tag,
        state=state,
        mask=mask,
        is_teampreview=is_teampreview,
        temperature=1.0,
        top_p=1.0,
    )


def test_handler_single_turn0_request(small_transformer_agent):
    """Turn 0 (no prior context). Handler stores a length-1 context after."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=True)
    np.random.seed(0)

    req = _make_request(embedder, battle_tag="t-0", is_teampreview=False)
    [resp] = handler([req])

    assert resp.request_id == req.request_id
    assert 0 <= resp.action_idx < 2025
    assert np.isfinite(resp.value)
    assert resp.log_prob <= 0.0
    # Hidden state landed in the handler's dict, length 1 (turn 0 → 1).
    stored = handler.hidden_states[(req.worker_id, req.player_id, req.battle_tag)]
    assert stored.shape == (1, 1, 32)


def test_handler_turn1_grows_context_by_one(small_transformer_agent):
    """Pre-populate a 4-step prior; expect length-5 stored after."""
    agent, _embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=True)
    np.random.seed(0)

    prior_ctx = torch.randn(1, 4, 32)
    handler.hidden_states[(0, "p", "t-1")] = prior_ctx
    req = _make_request(_embedder, battle_tag="t-1", is_teampreview=False)
    handler([req])
    assert handler.hidden_states[(0, "p", "t-1")].shape == (1, 5, 32)


def test_handler_teampreview_request(small_transformer_agent):
    """Teampreview decision: action range is 0..89."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=True)
    np.random.seed(0)

    req = _make_request(embedder, battle_tag="tp", is_teampreview=True)
    [resp] = handler([req])
    assert 0 <= resp.action_idx < 90


def test_handler_batch_of_mixed_shapes(small_transformer_agent):
    """Batch of 4 requests with mixed prior context lengths. Each gets a
    correctly-sized stored context (each grew by one row)."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=True)
    np.random.seed(0)

    priors = [None, torch.randn(1, 2, 32), torch.randn(1, 5, 32), None]
    reqs = [
        _make_request(embedder, request_id=i, battle_tag=f"b-{i}")
        for i in range(4)
    ]
    for i, prior in enumerate(priors):
        if prior is not None:
            handler.hidden_states[(0, "p", f"b-{i}")] = prior

    responses = handler(reqs)
    assert len(responses) == 4
    expected_next_lens = [1, 3, 6, 1]
    for i, expected_len in enumerate(expected_next_lens):
        assert handler.hidden_states[(0, "p", f"b-{i}")].shape == (1, expected_len, 32)


def test_handler_argmax_matches_legacy_inference_path(small_transformer_agent):
    """Strong M2 check: build the same agent the legacy player would
    build, run RNaDAgent directly, and compare the chosen action to the
    centralized handler's choice. probabilistic=False for determinism.
    """
    agent, embedder = small_transformer_agent
    centralized = RealModelBatchHandler(agent, device="cpu", probabilistic=False)

    np.random.seed(42)
    state = np.random.randn(embedder.embedding_size).astype(np.float32)
    mask = np.ones(2025, dtype=np.float32)
    prior_ctx = torch.randn(1, 3, 32)

    centralized.hidden_states[(0, "p", "argmax")] = prior_ctx
    req = InferenceRequest(
        request_id=0,
        worker_id=0,
        player_id="p",
        battle_tag="argmax",
        state=state,
        mask=mask,
        is_teampreview=False,
        temperature=1.0,
        top_p=1.0,
    )
    [centralized_resp] = centralized([req])

    # Legacy: drive the model directly the same way `_gpu_inference_sync` does.
    state_t = torch.from_numpy(state).to("cpu").unsqueeze(0).unsqueeze(1).float()
    h_mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        turn_logits, _, values, _, _ = agent(
            state_t, prior_ctx, mask=None, hidden_mask=h_mask
        )
    log_probs = torch.log_softmax(turn_logits[0, 0], dim=-1).numpy()
    valid_mask = mask.astype(bool)
    log_valid_mass = float(np.log(np.exp(log_probs[valid_mask]).sum()))
    legacy_action = int(np.argmax(log_probs))
    legacy_log_prob = float(log_probs[legacy_action] - log_valid_mass)
    legacy_value = float(values[0, 0].cpu().item())

    assert centralized_resp.action_idx == legacy_action
    assert centralized_resp.log_prob == pytest.approx(legacy_log_prob, abs=1e-5)
    assert centralized_resp.value == pytest.approx(legacy_value, abs=1e-5)
    # Stored context grew by 1.
    assert centralized.hidden_states[(0, "p", "argmax")].shape == (1, 4, 32)


def test_handler_evict_frees_state(small_transformer_agent):
    """`evict` removes the (worker, battle) entry; subsequent requests
    on the same battle_tag start from no context."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=False)
    np.random.seed(0)

    req = _make_request(embedder, battle_tag="evict-me", is_teampreview=False)
    handler([req])
    assert (0, "p", "evict-me") in handler.hidden_states

    handler.evict(0, "p", "evict-me")
    assert (0, "p", "evict-me") not in handler.hidden_states

    # Idempotent (no-op on missing).
    handler.evict(0, "p", "evict-me")


def test_handler_through_ipc_layer(small_transformer_agent):
    """Wire RealModelBatchHandler into InferenceService + InferenceClient
    and submit through the full IPC layer. Real model + queues +
    dispatch + futures all together in-process.
    """
    import asyncio
    from typing import Dict

    from torch import multiprocessing as torch_mp

    from elitefurretai.rl.inference_client import InferenceClient
    from elitefurretai.rl.inference_service import InferenceService

    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=False)

    request_queue: torch_mp.Queue = torch_mp.Queue()
    response_queues: Dict[int, torch_mp.Queue] = {0: torch_mp.Queue()}
    service = InferenceService(
        name="real-test",
        batch_handler=handler,
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size=4,
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
        np.random.seed(0)

        async def submit_one():
            return await client.submit(
                state=np.random.randn(embedder.embedding_size).astype(np.float32),
                mask=np.ones(2025, dtype=np.float32),
                is_teampreview=False,
                player_id="p",
                battle_tag="ipc-1",
                temperature=1.0,
                top_p=1.0,
            )

        resp = loop.run_until_complete(submit_one())
        assert 0 <= resp.action_idx < 2025
        assert np.isfinite(resp.value)
        # Handler stored the context.
        assert (0, "p", "ipc-1") in handler.hidden_states

        # And evict() works through the client → service → handler path.
        client.evict("p", "ipc-1")

        import time as _time

        deadline = _time.monotonic() + 1.0
        while (0, "p", "ipc-1") in handler.hidden_states:
            if _time.monotonic() > deadline:
                raise AssertionError("evict did not clear handler state")
            _time.sleep(0.005)
    finally:
        client.stop()
        service.stop()
        loop.close()


def test_handler_top_p_zeros_tail(small_transformer_agent):
    """top_p < 1.0 doesn't break argmax (sanity)."""
    agent, embedder = small_transformer_agent
    handler = RealModelBatchHandler(agent, device="cpu", probabilistic=False)
    np.random.seed(0)

    req = _make_request(embedder, battle_tag="topp", is_teampreview=False)
    req_top = InferenceRequest(
        request_id=req.request_id,
        worker_id=req.worker_id,
        player_id=req.player_id,
        battle_tag=req.battle_tag,
        state=req.state,
        mask=req.mask,
        is_teampreview=req.is_teampreview,
        temperature=req.temperature,
        top_p=0.01,
    )
    [resp_top] = handler([req_top])
    handler.evict(0, "p", "topp")
    [resp_full] = handler([req])
    assert resp_top.action_idx == resp_full.action_idx
