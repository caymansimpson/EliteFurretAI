# -*- coding: utf-8 -*-
"""IPC data types for centralized inference.

Trainer process owns one InferenceService per active model (main, bc, ...).
Workers own one InferenceClient per service. Requests flow worker → trainer
through `inference_request_queue`; responses flow trainer → worker through
the per-worker `response_queue[worker_id]`.

These dataclasses are the wire format. Keep them small and pickle-clean.
torch.multiprocessing auto-shares torch.Tensor payloads via /dev/shm; numpy
arrays and primitives go through pickle. Dataclasses are picklable as long
as all fields are picklable.

Hidden state lives in trainer, not on the wire
----------------------------------------------
The trainer-side handler keeps a `hidden_states` dict keyed by
(worker_id, player_id, battle_tag) and looks it up per request. The
wire only carries the small battle_tag string instead of a bulky
(1, T, hidden_size) tensor — this shrinks per-request IPC payload by
~40x and was the design that beat the initial "ship hidden in every
request" version (~120 KB/request, ~14 MB/sec total, IPC-bound).

Shape conventions
-----------------
- `state`:      (embedding_size,) float32
- `mask`:       (action_space,) bool — None for teampreview
- `battle_tag`: str — identifies which battle's hidden state to use; the
                trainer-side handler initializes to None on first sight
                of a (worker_id, battle_tag) pair.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    """Trainer → worker. One per resolved request.

    Fields match what the legacy `_run_batch` puts in its future result.
    The legacy `probs` field is dropped — nothing downstream consumes it.
    The model's distributional `win_dist_logits` output is also dropped
    — the learner recomputes it during training and the trajectory format
    doesn't store it. `next_hidden` was previously here for the worker
    to store back into its `hidden_states` dict; the trainer-side
    handler keeps the dict now, so the worker no longer needs it on
    the wire.
    """

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
