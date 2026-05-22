# -*- coding: utf-8 -*-
"""Subprocess host for inference services (Plan C — multi-process registry).

This module lets the trainer spawn one or more subprocesses that each
host a group of `InferenceService`s. Each subprocess gets its own
Python interpreter (escaping GIL contention with the trainer process)
and its own CUDA context. Trainer ↔ subprocess communication is via
two queue types:

  - Per-service `request_queue` / `response_queues`; workers send
    `InferenceRequest`s in and get `InferenceResponse`s back.
  - One `control_queue` per subprocess — carries `SyncWeightsMsg` (apply
    a new state_dict to a named service's model) and `ShutdownMsg`
    (stop services and exit).

See planning/stage2/2026-05-15-01-30-plan-c-4-process-implementation.md
for the full design rationale.

Why a separate module (not inside `model_registry.py`):

- The subprocess entrypoint must be importable as a module-level
  callable for `mp.Process(target=run_subprocess, ...)`. Keeping it
  here keeps the registry small and gives the subprocess code a
  natural test boundary.
- The `SubprocessSpecification` dataclass is what crosses the pickle boundary;
  having it sit next to the entrypoint makes that contract explicit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, assert_never, cast

import torch
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import InferenceRequest, InferenceResponse
from elitefurretai.rl.inference_trainer import (
    EvictRequest,
    InferenceService,
    RealModelBatchHandler,
)
from elitefurretai.rl.rl_utils import is_cuda_device
from elitefurretai.rl.rnad_model import RNaDModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Control messages — sent over control_queue from trainer to subprocess
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncWeightsMsg:
    """Apply a state_dict to a named model's raw agent in-place.

    `state_dict` values may be CPU tensors; the subprocess moves them
    to its device on load.
    """

    model_name: str
    state_dict: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class ShutdownMsg:
    """Stop all services in this subprocess and return from run_subprocess."""


ControlMsg = Union[SyncWeightsMsg, ShutdownMsg]


# ─────────────────────────────────────────────────────────────────────
# Specifications — sent across the spawn pickle boundary at process start
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ServiceSpecification:
    """Specification for one service that should run inside a subprocess.

    `agent` is constructed on CPU in the trainer process and crosses
    the pickle boundary; the subprocess moves it to its device and
    optionally compiles it.
    """

    name: str
    agent: RNaDModel
    request_queue: "torch_mp.Queue[Union[InferenceRequest, EvictRequest]]"
    response_queues: Dict[int, "torch_mp.Queue[InferenceResponse]"]
    compile: bool = True
    probabilistic: bool = True


@dataclass
class SubprocessSpecification:
    """Specification passed to `run_subprocess` via `mp.Process(args=(specification,))`."""

    group_name: str
    services: List[ServiceSpecification]
    control_queue: "torch_mp.Queue[ControlMsg]"
    device: str = "cpu"
    batch_size: int = 32
    batch_timeout: float = 0.005
    compile_mode: Optional[str] = None
    embedding_size: Optional[int] = None


# ─────────────────────────────────────────────────────────────────────
# Subprocess entrypoint — runs inside the spawned process
# ─────────────────────────────────────────────────────────────────────


def run_subprocess(specification: SubprocessSpecification) -> None:
    """Main entrypoint for an inference subprocess.

    Builds an `InferenceService` for each entry in `specification.services`,
    starts them, then listens on `specification.control_queue` for
    `SyncWeightsMsg` / `ShutdownMsg`. Returns cleanly on `ShutdownMsg`.
    """
    torch.set_num_threads(1)
    # Note on CUDA device selection: we deliberately do NOT call
    # `torch.cuda.set_device(...)` here. That call eagerly initializes
    # this subprocess's CUDA context, which on a heavily-loaded shared
    # GPU (trainer process holds many GB of model + compile artifacts)
    # can OOM at exactly the wrong moment — `set_device` allocates ~300
    # MB of fixed context overhead upfront. Letting the first CUDA op
    # (the model `.to(device)` calls below) create the context lazily
    # gives the allocator a chance to size to actual need. Single-GPU
    # setups (the only ones we support) get cuda:0 by default.
    # If multi-GPU support is added, we'd want to parse specification.device
    # and use CUDA_VISIBLE_DEVICES at process spawn time instead.
    if is_cuda_device(specification.device) and not torch.cuda.is_available():
        raise RuntimeError(
            f"InferenceSubprocess[{specification.group_name}] requested device "
            f"{specification.device!r} but CUDA is not available"
        )

    services: Dict[str, InferenceService] = {}
    raw_agents: Dict[str, RNaDModel] = {}

    # PASS 1: build (compile + warmup + handler) for every service.
    # No service threads start yet — if we started thread N now, while
    # service N+1 is in its dynamo compile, an early request to service N
    # would race with the FX trace of service N+1 (dynamo's tracer state
    # is process-global) and raise "Detected that you are using FX to
    # symbolically trace a dynamo-optimized function". ModelRegistry on
    # the trainer side uses the same split (register → start_all) for
    # exactly this reason; the subprocess needs it too.
    for service_specification in specification.services:
        agent = service_specification.agent
        agent.model.to(specification.device).eval()
        raw_agents[service_specification.name] = agent

        agent_for_handler: RNaDModel
        if specification.compile_mode and service_specification.compile:
            logger.info(
                "InferenceSubprocess[%s]: compiling '%s' mode=%s",
                specification.group_name,
                service_specification.name,
                specification.compile_mode,
            )
            compiled = cast(
                RNaDModel,
                torch.compile(agent, mode=specification.compile_mode, dynamic=True),
            )
            assert specification.embedding_size is not None, (
                "embedding_size required when compile_mode is set"
            )
            with torch.no_grad():
                for B in (1, 4, 16, 32):
                    x = torch.zeros(
                        B, 1, specification.embedding_size, device=specification.device
                    )
                    _, _, _, _, ctx = compiled(x, None)
                    compiled(x, ctx)
            agent_for_handler = compiled
        else:
            agent_for_handler = agent

        handler = RealModelBatchHandler(
            agent_for_handler,
            device=specification.device,
            probabilistic=service_specification.probabilistic,
        )
        service = InferenceService(
            name=service_specification.name,
            batch_handler=handler,
            request_queue=service_specification.request_queue,
            response_queues=service_specification.response_queues,
            batch_size=specification.batch_size,
            batch_timeout=specification.batch_timeout,
        )
        services[service_specification.name] = service

    # PASS 2: start every service thread. All compiles are now complete,
    # so no in-flight forward can collide with a fresh dynamo trace.
    for service in services.values():
        service.start()

    logger.info(
        "InferenceSubprocess[%s] up with %d services: %s",
        specification.group_name,
        len(services),
        sorted(services.keys()),
    )

    while True:
        msg = specification.control_queue.get()
        if isinstance(msg, ShutdownMsg):
            logger.info(
                "InferenceSubprocess[%s] received shutdown", specification.group_name
            )
            break
        elif isinstance(msg, SyncWeightsMsg):
            if msg.model_name not in raw_agents:
                logger.error(
                    "InferenceSubprocess[%s] sync_weights: unknown model '%s' (known: %s)",
                    specification.group_name,
                    msg.model_name,
                    sorted(raw_agents.keys()),
                )
                continue
            t_start = time.perf_counter()
            raw_agents[msg.model_name].model.load_state_dict(msg.state_dict)
            t_loaded = time.perf_counter()
            logger.info(
                "InferenceSubprocess[%s] synced weights for '%s' load=%.1fms",
                specification.group_name,
                msg.model_name,
                (t_loaded - t_start) * 1000.0,
            )
        else:
            assert_never(msg)

    for service in services.values():
        service.stop()


# ─────────────────────────────────────────────────────────────────────
# Trainer-side handle — wraps mp.Process + control queue
# ─────────────────────────────────────────────────────────────────────


class InferenceSubprocessHandle:
    """Trainer-side handle for one inference subprocess.

    Owns the `mp.Process` and the control queue. Use:

        handle = InferenceSubprocessHandle(specification)
        handle.start()
        ...
        handle.sync_weights("ghost_0", new_state_dict)
        ...
        handle.shutdown()
    """

    def __init__(
        self,
        specification: SubprocessSpecification,
        ctx: Optional[Any] = None,
    ):
        self.group_name = specification.group_name
        self.specification = specification
        self.control_queue = specification.control_queue
        self._ctx = ctx if ctx is not None else torch_mp.get_context("spawn")
        self._process: Optional[Any] = None  # mp.Process

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError(f"InferenceSubprocess[{self.group_name}] already started")
        proc = self._ctx.Process(
            target=run_subprocess,
            args=(self.specification,),
            name=f"InferenceSubprocess-{self.group_name}",
            daemon=False,
        )
        proc.start()
        self._process = proc

    def sync_weights(self, model_name: str, state_dict: Dict[str, torch.Tensor]) -> None:
        if self._process is None or not self._process.is_alive():
            raise RuntimeError(
                f"InferenceSubprocess[{self.group_name}] not running; cannot sync"
            )
        self.control_queue.put(
            SyncWeightsMsg(model_name=model_name, state_dict=state_dict)
        )

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def shutdown(self, timeout_s: float = 30.0) -> None:
        """Send ShutdownMsg and join. Falls back to terminate() on timeout."""
        if self._process is None:
            return
        try:
            self.control_queue.put(ShutdownMsg())
        except (BrokenPipeError, EOFError):
            # Subprocess already gone; just join.
            pass
        deadline = time.monotonic() + timeout_s
        self._process.join(timeout=timeout_s)
        if self._process.is_alive():
            remaining = max(0.0, deadline - time.monotonic())
            logger.warning(
                "InferenceSubprocess[%s] did not exit within %.1fs (remaining=%.1fs); "
                "terminating",
                self.group_name,
                timeout_s,
                remaining,
            )
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._process = None
