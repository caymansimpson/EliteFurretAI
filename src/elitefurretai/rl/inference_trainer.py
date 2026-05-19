# -*- coding: utf-8 -*-
"""Trainer-side inference: service loop + real-model batch handler.

Two units live here because they are always co-located in the trainer
process and the service is constructed by passing a handler to it:

  - `InferenceService`: owns the request mp.Queue (shared across workers)
      and the per-worker response mp.Queue dict. A daemon thread drains
      as many requests as it can (up to batch_size or until batch_timeout
      elapses since the first item), invokes the handler once, and
      dispatches per-request responses to each worker's response queue.
      Handles `EvictRequest` inline.
  - `RealModelBatchHandler`: callable that takes a list of
      `InferenceRequest` and returns a list of `InferenceResponse`. Owns
      the model (`RNaDAgent`), device, and the trainer-side
      `hidden_states` dict.
  - `echo_batch_handler`: deterministic stub used by IPC plumbing tests.

Hidden state lives here (not on the wire)
-----------------------------------------
The handler owns a `hidden_states` dict keyed by
(worker_id, player_id, battle_tag). The wire payload only carries the
lightweight battle_tag; the handler looks up the prior context, runs
the forward, and stores the updated context back in its own dict.
The wire response carries action / log_prob / value but NOT
next_hidden. This is the design that survived measurement after the
initial "ship hidden in every request" version proved to be IPC-bound
(~120 KB per request; the trainer-side dict shrinks payloads ~40x).

Memory hygiene: callers MUST send an `EvictRequest` when a battle ends
or the dict grows monotonically over the run. The handler exposes
`evict(worker_id, player_id, battle_tag)` for the service to call.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from torch import multiprocessing as torch_mp

from elitefurretai.rl.inference_ipc import (
    EvictRequest,
    InferenceRequest,
    InferenceResponse,
)
from elitefurretai.rl.rnad_model import RNaDAgent

logger = logging.getLogger(__name__)

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


# Type alias for the "model" hook the service uses to produce responses
# from a list of requests. Swappable for testing (echo, deterministic
# stub) and for real-model wiring.
BatchHandler = Callable[[List[InferenceRequest]], List[InferenceResponse]]


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
        batch_handler: BatchHandler,
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

        # Diagnostics — same fields the legacy per-player batcher logged,
        # so existing aggregation/observability rolls forward unchanged.
        self._diagnostics: Dict[str, float] = {
            "inference_batches": 0.0,
            "inference_batch_items": 0.0,
            "inference_batch_size_max": 0.0,
            "inference_batches_filled_to_max": 0.0,
            "inference_batches_flushed_timeout": 0.0,
            "inference_handler_seconds": 0.0,
            "inference_evictions": 0.0,
        }
        self._diag_lock = threading.Lock()

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
        # Mirrors BatchInferencePlayer._inference_loop: block for the
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

            self._update_batch_diagnostics(len(batch))
            handler_start = time.monotonic()
            try:
                responses = self._handler(batch)
            except Exception:
                logger.exception(
                    "InferenceService[%s] handler raised; dropping batch of %d",
                    self.name,
                    len(batch),
                )
                continue
            handler_s = time.monotonic() - handler_start
            with self._diag_lock:
                self._diagnostics["inference_handler_seconds"] += handler_s

            self._dispatch_responses(batch, responses)

    def _handle_evict(self, req: EvictRequest) -> None:
        """Forward eviction to the handler if it exposes evict()."""
        evict_fn = getattr(self._handler, "evict", None)
        if callable(evict_fn):
            evict_fn(req.worker_id, req.player_id, req.battle_tag)
        with self._diag_lock:
            self._diagnostics["inference_evictions"] += 1

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

    # ── diagnostics ───────────────────────────────────────────────────

    def _update_batch_diagnostics(self, batch_len: int) -> None:
        with self._diag_lock:
            self._diagnostics["inference_batches"] += 1
            self._diagnostics["inference_batch_items"] += batch_len
            if batch_len > self._diagnostics["inference_batch_size_max"]:
                self._diagnostics["inference_batch_size_max"] = float(batch_len)
            if batch_len >= self.batch_size:
                self._diagnostics["inference_batches_filled_to_max"] += 1
            else:
                self._diagnostics["inference_batches_flushed_timeout"] += 1
            n = self._diagnostics["inference_batches"]
            if n % 500 == 0:
                items = self._diagnostics["inference_batch_items"]
                filled = self._diagnostics["inference_batches_filled_to_max"]
                timeout = self._diagnostics["inference_batches_flushed_timeout"]
                logger.debug(
                    "[batch-fill svc=%s] n=%d avg=%.2f max=%d filled%%=%.1f "
                    "timeout%%=%.1f cap=%d",
                    self.name,
                    int(n),
                    items / n,
                    int(self._diagnostics["inference_batch_size_max"]),
                    100.0 * filled / n,
                    100.0 * timeout / n,
                    self.batch_size,
                )

    def get_diagnostics_snapshot(self) -> Dict[str, float]:
        with self._diag_lock:
            return dict(self._diagnostics)


# ─────────────────────────────────────────────────────────────────────
# Real-model batch handler
# ─────────────────────────────────────────────────────────────────────


class RealModelBatchHandler:
    """Callable that runs one batched forward + sampling pass.

    Use as the `batch_handler` argument to `InferenceService`.
    """

    def __init__(
        self,
        agent: RNaDAgent,
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
        # Diagnostic: catch overgrown contexts before they crash inside
        # the model's positional encoder (max_seq_len bound). This fires
        # if eviction-on-stale (BatchInferencePlayer._reset_battle_hidden_state)
        # somehow misses a cleanup site, since each in-flight request
        # for the same battle would otherwise grow hidden by 1.
        for ph, req in zip(prior_hiddens, batch):
            if ph is not None and ph.shape[1] > 35:
                logger.warning(
                    "Long context: worker=%d player=%s battle=%s shape=%s",
                    req.worker_id,
                    req.player_id,
                    req.battle_tag,
                    tuple(ph.shape),
                )

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
            BatchInferencePlayer._run_batch sliced [:L_i+1], which
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
        """Mirror BatchInferencePlayer's per-request sampling math:
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
