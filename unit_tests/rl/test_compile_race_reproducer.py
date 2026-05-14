# -*- coding: utf-8 -*-
"""Reproducer for the torch.compile + concurrent-service dynamo race
documented in the model_registry_plan.

The race manifests as RuntimeError("Detected that you are using FX to
symbolically trace a dynamo-optimized function") when multiple compiled
models are invoked concurrently from separate threads.

This test is expected to FAIL on plain torch + concurrent calls; the
goal of Task 2.2-2.4 is to find a wrapper that makes it pass.
"""
from __future__ import annotations

import random
import threading
from typing import Any, List, Optional

import pytest
import torch
import torch.nn as nn

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel


class TinyAgent(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


@pytest.mark.timeout(120)
def test_two_compiled_models_concurrent_calls_no_race():
    """Two distinct compiled models called from two threads should not
    raise. Fails today; the fix from Task 2.2/2.3/2.4 should make it pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = torch.compile(
        TinyAgent().to(device).eval(), mode="default", dynamic=True
    )
    m2 = torch.compile(
        TinyAgent().to(device).eval(), mode="default", dynamic=True
    )
    # Warm up each on a fixed shape (single-threaded, no race)
    with torch.no_grad():
        m1(torch.zeros(1, 64, device=device))
        m2(torch.zeros(1, 64, device=device))

    errors: List[BaseException] = []

    def loop(m: nn.Module) -> None:
        try:
            with torch.no_grad():
                for _ in range(500):
                    m(torch.randn(4, 64, device=device))
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=loop, args=(m1,))
    t2 = threading.Thread(target=loop, args=(m2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == [], f"compile race triggered: {errors[0]!r}"


def _make_small_rnad_agent(device: str) -> tuple[nn.Module, int, int]:
    """Construct a small but real TransformerThreeHeadedModel-backed RNaDAgent.

    Uses transformer_layers=2 (vs production's 4) to keep the test fast,
    but still exercises the same Transformer + growing hidden-state path
    that triggers the dynamo race in production.

    Returns (agent_module, embedding_size, hidden_size).
    """
    embedder = Embedder(feature_set="simple")
    # early_layers[-1] == hidden_size for context tensors
    early_layers = [64, 32]
    late_layers = [64, 32]
    model = TransformerThreeHeadedModel(
        embedder=embedder,
        early_layers=early_layers,
        late_layers=late_layers,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        max_seq_len=40,
    )
    model.eval().to(device)
    agent: nn.Module = RNaDAgent(model)
    return agent, embedder.embedding_size, early_layers[-1]


@pytest.mark.timeout(180)
def test_two_real_rnad_agents_concurrent_calls_no_race():
    """Two distinct compiled RNaDAgent instances called from two threads
    should not raise.

    Mirrors the production scenario: sep_arch compiles BOTH main and bc
    RNaDAgent instances (TransformerThreeHeadedModel backbone). Each
    InferenceService daemon thread calls one compiled agent; growing
    hidden-state context (turn 0 → turn 1 → ...) forces dynamo to
    recompile on shape changes, which is exactly when the race fires.

    Expected to FAIL before Task 2.2/2.3/2.4 fix; the fix should make
    it pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent1_raw, emb_size, hidden_size = _make_small_rnad_agent(device)
    agent2_raw, _, _ = _make_small_rnad_agent(device)

    agent1 = torch.compile(agent1_raw, mode="default", dynamic=True)
    agent2 = torch.compile(agent2_raw, mode="default", dynamic=True)

    # Warm up each agent single-threaded on (turn 0, no context) and
    # (turn 1, 1-step context) to seed the dynamo cache before racing.
    def _warmup(agent: Any) -> None:
        with torch.no_grad():
            # Turn 0: no hidden state
            x0 = torch.zeros(1, 1, emb_size, device=device)
            _, _, _, _, h0 = agent(x0, None, mask=None, hidden_mask=None)
            # Turn 1: 1-step context (shape (1, 1, hidden_size))
            ctx1 = h0[:1, :1, :].clone()
            hmask1 = torch.ones(1, 1, dtype=torch.bool, device=device)
            x1 = torch.zeros(1, 1, emb_size, device=device)
            agent(x1, ctx1, mask=None, hidden_mask=hmask1)

    _warmup(agent1)
    _warmup(agent2)

    errors: List[BaseException] = []

    def loop(agent: Any) -> None:
        rng = random.Random()
        try:
            with torch.no_grad():
                for _ in range(200):
                    ctx_len = rng.randint(0, 10)
                    batch_size = rng.randint(1, 8)
                    x = torch.randn(batch_size, 1, emb_size, device=device)
                    if ctx_len == 0:
                        hidden: Optional[torch.Tensor] = None
                        hmask: Optional[torch.Tensor] = None
                    else:
                        hidden = torch.randn(
                            batch_size, ctx_len, hidden_size, device=device
                        )
                        hmask = torch.ones(
                            batch_size, ctx_len, dtype=torch.bool, device=device
                        )
                    agent(x, hidden, mask=None, hidden_mask=hmask)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=loop, args=(agent1,))
    t2 = threading.Thread(target=loop, args=(agent2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == [], f"compile race triggered: {errors[0]!r}"


@pytest.mark.timeout(180)
def test_two_real_rnad_agents_with_per_model_lock():
    """Same as test_two_real_rnad_agents_concurrent_calls_no_race but
    each compiled model has its own threading.Lock serializing entry.

    Tests whether serializing dynamo trace-entry per-model bypasses the
    race. Locks are per-model (lock1 for agent1, lock2 for agent2) so
    the two models can still run concurrently with each other — only
    simultaneous re-entries into the SAME model are blocked.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent1_raw, emb_size, hidden_size = _make_small_rnad_agent(device)
    agent2_raw, _, _ = _make_small_rnad_agent(device)

    agent1 = torch.compile(agent1_raw, mode="default", dynamic=True)
    agent2 = torch.compile(agent2_raw, mode="default", dynamic=True)

    def _warmup(agent: Any) -> None:
        with torch.no_grad():
            x0 = torch.zeros(1, 1, emb_size, device=device)
            _, _, _, _, h0 = agent(x0, None, mask=None, hidden_mask=None)
            ctx1 = h0[:1, :1, :].clone()
            hmask1 = torch.ones(1, 1, dtype=torch.bool, device=device)
            x1 = torch.zeros(1, 1, emb_size, device=device)
            agent(x1, ctx1, mask=None, hidden_mask=hmask1)

    _warmup(agent1)
    _warmup(agent2)

    lock1 = threading.Lock()
    lock2 = threading.Lock()

    errors: List[BaseException] = []

    def loop_locked(agent: Any, lock: threading.Lock) -> None:
        rng = random.Random()
        try:
            with torch.no_grad():
                for _ in range(200):
                    ctx_len = rng.randint(0, 10)
                    batch_size = rng.randint(1, 8)
                    x = torch.randn(batch_size, 1, emb_size, device=device)
                    if ctx_len == 0:
                        hidden: Optional[torch.Tensor] = None
                        hmask: Optional[torch.Tensor] = None
                    else:
                        hidden = torch.randn(
                            batch_size, ctx_len, hidden_size, device=device
                        )
                        hmask = torch.ones(
                            batch_size, ctx_len, dtype=torch.bool, device=device
                        )
                    with lock:
                        agent(x, hidden, mask=None, hidden_mask=hmask)
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=loop_locked, args=(agent1, lock1))
    t2 = threading.Thread(target=loop_locked, args=(agent2, lock2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == [], f"compile race triggered with per-model lock: {errors[0]!r}"
