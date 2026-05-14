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

import threading
from typing import List

import pytest
import torch
import torch.nn as nn


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
