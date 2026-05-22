# -*- coding: utf-8 -*-
"""Unit tests for ModelRegistry + WorkerInferenceClients.

These run entirely in-process (single Python interpreter) and exercise
the registry / worker-bundle plumbing without spawning subprocesses or
loading real models. Real-model + cross-process behavior is already
covered by:
  - test_inference_handler_real_model.py (M2 — real RNaDModel in handler)
  - test_inference_multistep_and_mp.py (M3 — multistep + cross-process)

The registry tests focus on:
  - register / sync_weights / queues_for_workers / stop_all behavior
  - duplicate-register and missing-name errors
  - WorkerInferenceClients construction from queues bundle, get/has/names,
    clean shutdown
  - Smoke end-to-end: registry → workers (in-process) → submit + receive
"""

from __future__ import annotations

import asyncio
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.inference_worker import WorkerInferenceClients
from elitefurretai.rl.model_registry import ModelRegistry
from elitefurretai.rl.rnad_model import RNaDModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel


@pytest.fixture
def small_agent_factory():
    """Returns a callable that builds a fresh small agent each call.

    Multiple agents needed to exercise multi-model registration without
    sharing weights between names.
    """
    embedder = Embedder(feature_set="simple")

    def make() -> RNaDModel:
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
        return RNaDModel(model)

    return make, embedder


# ─────────────────────────────────────────────────────────────────────
# ModelRegistry
# ─────────────────────────────────────────────────────────────────────


def test_registry_register_then_start(small_agent_factory):
    """After register() + start_all(), the service is running and exposed
    via names(). (Plan C split register/start_all to fix Run C's
    registration race.)"""
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=2, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        registry.start_all()
        assert "main" in registry.names()
    finally:
        registry.stop_all()


def test_registry_register_duplicate_raises(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        with pytest.raises(ValueError, match="already registered"):
            registry.register("main", make())
    finally:
        registry.stop_all()


def test_registry_register_three_models(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=3, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        registry.register("bc", make())
        registry.register("ghost_0", make())
        assert sorted(registry.names()) == ["bc", "ghost_0", "main"]
        bundles = registry.queues_for_workers()
        assert sorted(bundles.keys()) == ["bc", "ghost_0", "main"]
        for name, (req_q, resp_qs) in bundles.items():
            assert req_q is not None
            assert len(resp_qs) == 3  # one per worker
    finally:
        registry.stop_all()


def test_registry_sync_weights_updates_in_place(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        agent = make()
        registry.register("main", agent)

        # Build a different state_dict by zeroing one parameter.
        new_sd = {k: v.clone() for k, v in agent.model.state_dict().items()}
        first_key = next(iter(new_sd))
        new_sd[first_key] = new_sd[first_key].zero_()

        registry.sync_weights("main", new_sd)
        # Verify the agent's parameter is now zeroed.
        assert agent.model.state_dict()[first_key].abs().sum().item() == 0.0
    finally:
        registry.stop_all()


def test_registry_sync_weights_unknown_name_raises():
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        with pytest.raises(KeyError, match="not registered"):
            registry.sync_weights("does_not_exist", {})
    finally:
        registry.stop_all()


def test_registry_compile_requires_embedding_size():
    with pytest.raises(ValueError, match="embedding_size"):
        ModelRegistry(
            num_workers=1,
            batch_size=4,
            batch_timeout=0.005,
            compile_mode="default",
            # embedding_size deliberately omitted
        )


def test_registry_stop_all_idempotent(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    registry.register("main", make())
    registry.stop_all()
    # Second call shouldn't raise.
    registry.stop_all()
    assert registry.names() == []


# ─────────────────────────────────────────────────────────────────────
# WorkerInferenceClients
# ─────────────────────────────────────────────────────────────────────


def _per_worker_slice(bundles: Dict[str, Tuple], worker_id: int) -> Dict[str, Tuple]:
    return {
        name: (req_q, resp_qs[worker_id]) for name, (req_q, resp_qs) in bundles.items()
    }


def test_worker_clients_constructs_one_per_model(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=2, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        registry.register("bc", make())
        loop = asyncio.new_event_loop()
        try:
            worker_slice = _per_worker_slice(registry.queues_for_workers(), 0)
            clients = WorkerInferenceClients(0, worker_slice, loop=loop)
            try:
                assert sorted(clients.names()) == ["bc", "main"]
                assert "main" in clients
                assert "bc" in clients
                assert "ghost_0" not in clients
                assert clients["main"].worker_id == 0
                assert clients["bc"].worker_id == 0
            finally:
                clients.stop_all()
        finally:
            loop.close()
    finally:
        registry.stop_all()


def test_worker_clients_get_unknown_raises(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        loop = asyncio.new_event_loop()
        try:
            slice_ = _per_worker_slice(registry.queues_for_workers(), 0)
            clients = WorkerInferenceClients(0, slice_, loop=loop)
            try:
                assert clients.get("does_not_exist") is None
                with pytest.raises(KeyError, match="No inference client"):
                    _ = clients["does_not_exist"]
            finally:
                clients.stop_all()
        finally:
            loop.close()
    finally:
        registry.stop_all()


# ─────────────────────────────────────────────────────────────────────
# Smoke end-to-end (in-process)
# ─────────────────────────────────────────────────────────────────────


def test_register_multiple_ghost_slots(small_agent_factory):
    """Registry accepts N ghost_<i> registrations with distinct services."""
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=2, batch_size=4, batch_timeout=0.005)
    try:
        for slot in range(3):
            registry.register(f"ghost_{slot}", make(), compile=False)
        registry.start_all()
        assert set(registry.names()) == {"ghost_0", "ghost_1", "ghost_2"}
    finally:
        registry.stop_all()


def test_registry_to_worker_round_trip(small_agent_factory):
    """Register two models, build worker clients, submit one request to
    each, verify both come back with valid action indices and the
    handler stored hidden state for both."""
    make, embedder = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make())
        registry.register("bc", make())
        registry.start_all()

        loop = asyncio.new_event_loop()
        try:
            slice_ = _per_worker_slice(registry.queues_for_workers(), 0)
            clients = WorkerInferenceClients(0, slice_, loop=loop)
            try:

                async def submit_to(name: str) -> int:
                    resp = await clients[name].submit(
                        state=np.random.randn(embedder.embedding_size).astype(np.float32),
                        mask=np.ones(2025, dtype=np.float32),
                        is_teampreview=False,
                        player_id="p",
                        battle_tag=f"tag-{name}",
                        temperature=1.0,
                        top_p=1.0,
                    )
                    return resp.action_idx

                np.random.seed(0)

                async def both():
                    return await asyncio.gather(submit_to("main"), submit_to("bc"))

                main_action, bc_action = loop.run_until_complete(both())
                assert 0 <= main_action < 2025
                assert 0 <= bc_action < 2025

                # Handlers stored state per-model.
                main_handler = registry._handlers["main"]
                bc_handler = registry._handlers["bc"]
                assert (0, "p", "tag-main") in main_handler.hidden_states
                assert (0, "p", "tag-bc") in bc_handler.hidden_states
                # Cross-isolation: main's tag isn't in bc's handler.
                assert (0, "p", "tag-main") not in bc_handler.hidden_states
                assert (0, "p", "tag-bc") not in main_handler.hidden_states
            finally:
                clients.stop_all()
        finally:
            loop.close()
    finally:
        registry.stop_all()


# ─────────────────────────────────────────────────────────────────────
# Plan C — process_group routing
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_registry_mixed_process_groups(small_agent_factory):
    """Register two in-process services and one subprocess service;
    start_all() builds both backends; a request to each comes back."""
    import copy as _copy

    from elitefurretai.rl.inference_ipc import InferenceRequest

    make, embedder = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make(), compile=False)
        registry.register("bc", make(), compile=False)
        registry.register("ghost_0", make(), compile=False, process_group="ghosts")

        # Before start_all(): names visible but services not running.
        assert set(registry.names()) == {"main", "bc", "ghost_0"}

        registry.start_all()

        # Send a request to each via its queue; receive a response.
        bundles = registry.queues_for_workers()
        for name in ("main", "bc", "ghost_0"):
            req_q, resp_qs = bundles[name]
            state = np.random.randn(embedder.embedding_size).astype(np.float32)
            req = InferenceRequest(
                request_id=hash(name) & 0xFFFF,
                worker_id=0,
                player_id="p",
                battle_tag=f"tag-{name}",
                state=state,
                mask=np.ones(2025, dtype=np.float32),
                is_teampreview=False,
                temperature=1.0,
                top_p=1.0,
            )
            req_q.put(req)
            resp = resp_qs[0].get(timeout=60.0)
            assert resp.request_id == req.request_id
            assert 0 <= resp.action_idx < 2025

        _ = _copy  # silence unused import (kept for symmetry with sync test)
    finally:
        registry.stop_all()


@pytest.mark.slow
def test_registry_subprocess_sync_weights(small_agent_factory):
    """sync_weights for a process_group service routes via the
    subprocess's control queue; subsequent requests reflect new weights.
    """
    import copy

    from elitefurretai.rl.inference_ipc import InferenceRequest

    make, embedder = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        agent_a = make()
        registry.register("ghost_0", agent_a, compile=False, process_group="ghosts")
        registry.start_all()

        bundles = registry.queues_for_workers()
        req_q, resp_qs = bundles["ghost_0"]

        def submit(req_id: int):
            req = InferenceRequest(
                request_id=req_id,
                worker_id=0,
                player_id="p",
                battle_tag=f"tag-{req_id}",
                state=np.random.RandomState(seed=req_id)
                .randn(embedder.embedding_size)
                .astype(np.float32),
                mask=np.ones(2025, dtype=np.float32),
                is_teampreview=False,
                temperature=1.0,
                top_p=1.0,
            )
            req_q.put(req)
            return resp_qs[0].get(timeout=60.0)

        resp_before = submit(1)

        # Build a NEW agent (different random init) and sync its weights.
        torch.manual_seed(98765)
        agent_b = make()
        registry.sync_weights("ghost_0", copy.deepcopy(agent_b.model.state_dict()))

        resp_after = submit(2)
        assert abs(resp_after.value - resp_before.value) > 1e-5, (
            f"cross-process sync_weights did not change outputs: "
            f"before={resp_before.value} after={resp_after.value}"
        )
    finally:
        registry.stop_all()


def test_registry_register_after_start_all_raises(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make(), compile=False)
        registry.start_all()
        with pytest.raises(RuntimeError, match="after start_all"):
            registry.register("bc", make(), compile=False)
    finally:
        registry.stop_all()


def test_registry_start_all_twice_raises(small_agent_factory):
    make, _ = small_agent_factory
    registry = ModelRegistry(num_workers=1, batch_size=4, batch_timeout=0.005)
    try:
        registry.register("main", make(), compile=False)
        registry.start_all()
        with pytest.raises(RuntimeError, match="already called"):
            registry.start_all()
    finally:
        registry.stop_all()
