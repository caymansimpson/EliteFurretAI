# -*- coding: utf-8 -*-
"""Trainer-side inference: wire types, service loop, batch handler, and subprocess host.

Trainer process owns one `InferenceService` per active model (main, bc, ...).
Workers own one `InferenceClient` per service (see `inference_worker.py`).
Requests flow worker → trainer through `inference_request_queue`; responses
flow trainer → worker through the per-worker `response_queue[worker_id]`.

The trainer can run services either in-process (the `InferenceService` thread)
or out-of-process via `run_subprocess` (Plan C — multi-process registry,
escaping GIL contention with the trainer process and getting its own CUDA
context). See planning/stage2/2026-05-15-01-30-plan-c-4-process-implementation.md
for the full design rationale.

Wire format
-----------
The dataclasses below are the wire format — kept small and pickle-clean.
torch.multiprocessing auto-shares `torch.Tensor` payloads via /dev/shm;
numpy arrays and primitives go through pickle.

Shape conventions:

- `state`:      (embedding_size,) float32
- `mask`:       (action_space,) bool — None for teampreview
- `battle_tag`: str — identifies which battle's hidden state to use; the
                trainer-side handler initializes to None on first sight
                of a (worker_id, battle_tag) pair.

Hidden state lives in the trainer, not on the wire
--------------------------------------------------
The handler keeps a `hidden_states` dict keyed by
(worker_id, player_id, battle_tag) and looks it up per request. The wire
only carries the small battle_tag string instead of a bulky
(1, T, hidden_size) tensor — this shrinks per-request IPC payload by ~40x.
This is the design that survived measurement after the initial "ship hidden
in every request" version proved to be IPC-bound (~120 KB per request).

Memory hygiene: callers MUST send an `EvictRequest` when a battle ends or
the dict grows monotonically over the run. The handler exposes
`evict(worker_id, player_id, battle_tag)` for the service to call.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, assert_never, cast

import numpy as np
import torch
from torch import multiprocessing as torch_mp

from elitefurretai.rl.rl_utils import is_cuda_device
from elitefurretai.rl.rnad_model import RNaDModel

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Wire format — IPC dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class InferenceRequest:
    """Worker → trainer. One per inference call (one per battle decision).

    `player_id` distinguishes the two sides of a self-play battle that
    share the same `battle_tag`. Without it, both sides write to the
    same hidden_states slot, the trainer-side hidden tensor grows by
    2 per turn (once per side), and the model overflows max_seq_len.
    """

    request_id: int
    worker_id: int
    player_id: str
    battle_tag: str
    state: np.ndarray
    mask: Optional[np.ndarray]
    is_teampreview: bool
    temperature: float
    top_p: float


@dataclass
class InferenceResponse:
    """Trainer → worker. One per resolved request."""

    request_id: int
    action_idx: int
    log_prob: float
    value: float


@dataclass
class EvictRequest:
    """Worker → trainer. Free the hidden state for a finished battle.

    Without this, the trainer's `hidden_states` dict grows monotonically
    over the run. EvictRequest can travel through the same request
    queue as InferenceRequest (the service's loop discriminates by type).

    `player_id` matches the same field on InferenceRequest so we evict
    only the side that finished (the opposite side is the same battle
    instance from poke-env's perspective and may still need its
    own hidden state).
    """

    worker_id: int
    player_id: str
    battle_tag: str


# Single process-wide lock around compiled-model forward calls. torch.dynamo's
# trace state is global across model instances of the same class, so a
# per-model lock doesn't help — only this global lock prevents the
# "Detected that you are using FX to symbolically trace a dynamo-optimized
# function" race when multiple compiled models run on concurrent service
# threads. Lock contention is negligible because (a) compiled forwards
# release the GIL during GPU work, and (b) the GIL already serializes
# Python-level dynamo code. See reproducer at
# unit_tests/rl/test_compile_race_reproducer.py.
_COMPILE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────


class InferenceService:
    """Centralized batched inference, served from the trainer process.

    One instance per model. Spawn workers with this service's
    `request_queue` and the `response_queues` dict so they can send
    requests in and receive responses back.
    """

    def __init__(
        self,
        name: str,
        batch_handler: Callable[[List[InferenceRequest]], List[InferenceResponse]],
        request_queue: "torch_mp.Queue[Union[InferenceRequest, EvictRequest]]",
        response_queues: Dict[int, "torch_mp.Queue[InferenceResponse]"],
        batch_size: int = 32,
        batch_timeout: float = 0.005,
    ):
        self.name = name
        self._handler = batch_handler
        self._request_queue = request_queue
        self._response_queues = response_queues
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"InferenceService[{self.name}] already started")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"InferenceService-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and join. Does NOT drain pending
        requests; callers are responsible for ensuring no producer
        will submit after stop() returns."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    "InferenceService[%s] thread did not exit within %.1fs",
                    self.name,
                    timeout_s,
                )
            self._thread = None

    # ── main loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        # Mirrors RLTrajectoryPlayer._inference_loop: block for the
        # first request, then opportunistically gather more up to either
        # batch_size or batch_timeout since the first arrived.
        #
        # The queue carries InferenceRequest AND EvictRequest. We
        # process EvictRequests inline (they're cheap dict-pops) and
        # only pack InferenceRequests into batches.
        while not self._stop.is_set():
            try:
                first = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(first, EvictRequest):
                self._handle_evict(first)
                continue
            batch: List[InferenceRequest] = [first]
            deadline = time.monotonic() + self.batch_timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(item, EvictRequest):
                    # Don't waste the batch; handle and keep gathering.
                    self._handle_evict(item)
                    continue
                batch.append(item)

            try:
                responses = self._handler(batch)
            except Exception:
                logger.exception(
                    "InferenceService[%s] handler raised; dropping batch of %d",
                    self.name,
                    len(batch),
                )
                continue

            self._dispatch_responses(batch, responses)

    def _handle_evict(self, req: EvictRequest) -> None:
        """Forward eviction to the handler if it exposes evict()."""
        evict_fn = getattr(self._handler, "evict", None)
        if callable(evict_fn):
            evict_fn(req.worker_id, req.player_id, req.battle_tag)

    def _dispatch_responses(
        self,
        batch: List[InferenceRequest],
        responses: List[InferenceResponse],
    ) -> None:
        if len(responses) != len(batch):
            logger.error(
                "InferenceService[%s] handler returned %d responses for %d "
                "requests; dropping mismatched batch",
                self.name,
                len(responses),
                len(batch),
            )
            return
        for req, resp in zip(batch, responses):
            target = self._response_queues.get(req.worker_id)
            if target is None:
                logger.error(
                    "InferenceService[%s] no response queue for worker_id=%d "
                    "(request_id=%d); response dropped",
                    self.name,
                    req.worker_id,
                    req.request_id,
                )
                continue
            target.put(resp)


# ─────────────────────────────────────────────────────────────────────
# Real-model batch handler
# ─────────────────────────────────────────────────────────────────────


class RealModelBatchHandler:
    """Callable that runs one batched forward + sampling pass.

    Use as the `batch_handler` argument to `InferenceService`.
    """

    def __init__(
        self,
        agent: RNaDModel,
        device: str = "cpu",
        probabilistic: bool = True,
    ):
        self.agent = agent
        self.device = device
        self.probabilistic = probabilistic
        # Hidden state lives here, keyed by (worker_id, player_id,
        # battle_tag). The service calls __call__ on a single thread,
        # so no lock is needed for normal request processing. `evict`
        # is also called from that thread (the service drains evict
        # requests off the same queue), so still no lock needed.
        #
        # player_id is required in the key because both sides of a
        # self-play battle share the same battle_tag — keying without
        # it would collide and double-grow the hidden tensor.
        self.hidden_states: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def __call__(self, batch: List[InferenceRequest]) -> List[InferenceResponse]:
        if not batch:
            return []

        states_np = np.stack([r.state for r in batch], axis=0)
        states = torch.from_numpy(states_np).to(self.device).unsqueeze(1).float()

        # Look up each request's prior hidden from our local dict.
        prior_hiddens: List[Optional[torch.Tensor]] = [
            self.hidden_states.get((r.worker_id, r.player_id, r.battle_tag)) for r in batch
        ]

        hidden_batch, hidden_mask = self._pad_transformer_context(prior_hiddens)
        with torch.no_grad(), _COMPILE_LOCK:
            turn_logits, tp_logits, values, _, next_hidden = self.agent(
                states, hidden_batch, mask=None, hidden_mask=hidden_mask
            )

        turn_logits_cpu = turn_logits.cpu()
        tp_logits_cpu = tp_logits.cpu()
        values_cpu = values.cpu().numpy()

        responses: List[InferenceResponse] = []
        for i, req in enumerate(batch):
            if req.is_teampreview:
                logits = tp_logits_cpu[i, 0]
            else:
                logits = turn_logits_cpu[i, 0]

            mask_np = req.mask
            action_idx, log_prob = self._sample_action(
                logits=logits,
                mask=mask_np,
                temperature=req.temperature,
                top_p=req.top_p,
                is_teampreview=req.is_teampreview,
            )

            # Update the trainer-side hidden_states dict in place. The
            # response no longer carries this — workers don't need it.
            self.hidden_states[(req.worker_id, req.player_id, req.battle_tag)] = (
                self._slice_next_hidden(next_hidden, i, prior_hiddens[i])
            )
            responses.append(
                InferenceResponse(
                    request_id=req.request_id,
                    action_idx=action_idx,
                    log_prob=log_prob,
                    value=float(values_cpu[i, 0]),
                )
            )
        return responses

    def evict(self, worker_id: int, player_id: str, battle_tag: str) -> None:
        """Free the hidden state for one side of a finished battle.

        Called by the service when an EvictRequest arrives. No-op if
        missing. Each side of a self-play battle evicts independently."""
        self.hidden_states.pop((worker_id, player_id, battle_tag), None)

    # ── hidden-state batching ─────────────────────────────────────────

    def _pad_transformer_context(
        self, prior_hiddens: List[Optional[torch.Tensor]]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Pad each request's growing context to the max in this batch.

        Mirrors the legacy `_run_batch` transformer padding logic, but
        reads from the trainer-side `hidden_states` dict instead of
        per-request `hidden` fields.
        """
        context_lengths: List[int] = []
        hidden_size: Optional[int] = None
        for ctx in prior_hiddens:
            if ctx is None:
                context_lengths.append(0)
                continue
            if ctx.ndim != 3 or ctx.shape[0] != 1:
                raise ValueError(
                    f"Expected transformer context shape (1, T, H), got {tuple(ctx.shape)}"
                )
            context_lengths.append(int(ctx.shape[1]))
            if hidden_size is None:
                hidden_size = int(ctx.shape[2])

        max_context_len = max(context_lengths, default=0)
        if max_context_len == 0:
            return None, None
        assert hidden_size is not None

        B = len(prior_hiddens)
        hidden = torch.zeros(B, max_context_len, hidden_size, dtype=torch.float32)
        hmask = torch.zeros(B, max_context_len, dtype=torch.bool)
        for i, ctx in enumerate(prior_hiddens):
            length = context_lengths[i]
            if ctx is None or length == 0:
                continue
            hidden[i, :length, :] = ctx[0, :length, :]
            hmask[i, :length] = True
        return hidden.to(self.device), hmask.to(self.device)

    def _slice_next_hidden(
        self,
        next_hidden_batch: object,
        index: int,
        prev_hidden: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the new hidden state for request[index] in a form the
        handler stores back into `hidden_states[(worker_id, battle_tag)]`.

        Transformer notes (correctness fix vs legacy):
            The model's forward_with_hidden concatenates `hidden_state`
            (padded to max_T) + `encoded` (length 1), so the new
            encoded state sits at position max_T regardless of this
            request's actual prior length L_i. The legacy
            RLTrajectoryPlayer._run_batch sliced [:L_i+1], which
            included a padding-derived position at L_i instead of the
            real new state at position max_T. With mixed-length batches
            this corrupted the next-turn hidden state and propagated
            indefinitely. We fix by explicitly concatenating the prior
            slice with the new-state slice at max_T.
        """
        ctx_batch = cast(torch.Tensor, next_hidden_batch).cpu()
        prev_len = 0 if prev_hidden is None else prev_hidden.shape[1]
        max_T = ctx_batch.shape[1] - 1  # output length is max_T + 1
        prior = ctx_batch[index : index + 1, :prev_len, :]
        new_state = ctx_batch[index : index + 1, max_T : max_T + 1, :]
        return torch.cat([prior, new_state], dim=1).clone()

    # ── per-request sampling ──────────────────────────────────────────

    def _sample_action(
        self,
        logits: torch.Tensor,
        mask: Optional[np.ndarray],
        temperature: float,
        top_p: float,
        is_teampreview: bool,
    ) -> tuple[int, float]:
        """Mirror RLTrajectoryPlayer's per-request sampling math:
        temperature softmax → mask + renormalize → top-p filter →
        multinomial; PPO old_log_prob from the masked T=1 distribution.
        """
        temp = max(temperature, 1e-6)
        probs = torch.softmax(logits / temp, dim=-1).numpy()
        unscaled_log_probs = torch.log_softmax(logits, dim=-1).numpy()

        if not is_teampreview and mask is not None:
            probs = probs * mask
            total = probs.sum()
            if total == 0.0:
                probs = mask / mask.sum()
            else:
                probs = probs / total
            if top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                cum = np.cumsum(probs[sorted_idx])
                cutoff = int(np.searchsorted(cum, top_p)) + 1
                keep = sorted_idx[:cutoff]
                filtered = np.zeros_like(probs)
                filtered[keep] = probs[keep]
                probs = filtered / filtered.sum()

        n = probs.shape[0]
        if self.probabilistic:
            action = int(np.random.choice(n, p=probs))
        else:
            action = int(np.argmax(probs))

        if mask is not None and not is_teampreview:
            valid_mask = mask.astype(bool)
            log_valid_mass = float(np.log(np.exp(unscaled_log_probs[valid_mask]).sum()))
            log_prob = float(unscaled_log_probs[action] - log_valid_mass)
        else:
            log_prob = float(unscaled_log_probs[action])
        return action, log_prob


# ─────────────────────────────────────────────────────────────────────
# Deterministic echo handler used by IPC plumbing tests. Returns
# action_idx=0 for every request. Doesn't manage hidden state
# (RealModelBatchHandler manages it); tests using this echo handler
# don't exercise hidden-state correctness.
# ─────────────────────────────────────────────────────────────────────


def echo_batch_handler(batch: List[InferenceRequest]) -> List[InferenceResponse]:
    return [
        InferenceResponse(
            request_id=req.request_id,
            action_idx=0,
            log_prob=0.0,
            value=0.0,
        )
        for req in batch
    ]


# ─────────────────────────────────────────────────────────────────────
# Subprocess host (Plan C — multi-process registry)
#
# Lets the trainer spawn one or more subprocesses that each host a group
# of `InferenceService`s. Each subprocess gets its own Python interpreter
# (escaping GIL contention with the trainer) and its own CUDA context.
# Trainer ↔ subprocess communication is via two queue types:
#
#   - Per-service `request_queue` / `response_queues`; workers send
#     `InferenceRequest`s in and get `InferenceResponse`s back.
#   - One `control_queue` per subprocess — carries `SyncWeightsMsg`
#     (apply a new state_dict to a named service's model) and
#     `ShutdownMsg` (stop services and exit).
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
