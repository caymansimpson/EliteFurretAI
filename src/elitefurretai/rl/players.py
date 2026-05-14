"""players.py — RL battle participants (the "actors" in the IMPALA picture).

What this file is
-----------------
Three classes that play Pokemon battles for the RL pipeline:

1. RNaDAgent              — A thin nn.Module wrapper around the trained model.
                            Hides the LSTM-vs-Transformer difference behind a
                            uniform `forward()` interface so callers don't care.

2. BatchInferencePlayer   — The high-throughput, async player used during RL
                            training. Gathers per-turn decisions from many
                            concurrent battles, *batches* them into one model
                            forward pass for efficiency, and pushes finished
                            trajectories to a queue for the learner to consume.

3. MaxDamagePlayer        — A non-learning heuristic player used as a curriculum
                            opponent. Picks the move with highest damage estimate
                            via poke-env's calculate_damage. Cheap and consistent.

How this fits the bigger picture
--------------------------------
The trainer (train.py) spawns N worker processes (worker.py). Each worker runs
many battles in parallel. In each battle, every turn is a "decision request":
the worker needs the model's policy/value for a particular state.

Naive approach: each decision = its own model forward pass. But model forwards
have meaningful per-call overhead (Python ↔ C++ trampoline, kernel launch,
cache misses), so doing 100 separate single-state forwards is far slower than
one batched forward over 100 states.

BatchInferencePlayer solves this by maintaining an asyncio queue of pending
decisions, running an inference loop that gathers up to `batch_size` requests
(or waits at most `batch_timeout` seconds) and dispatches them together.
This is a classic "dynamic batching" pattern.

Why so much state-tracking machinery
------------------------------------
Pokemon Showdown is async over websockets. By the time a model decision comes
back, the battle state may have changed (e.g. opponent forfeited, a force
switch was triggered, etc.). The fingerprinting + request-generation tracking
exists to detect those drifts and *drop* stale decisions rather than send
invalid commands. Without it, we'd see waves of "invalid choice" errors.

Trajectory collection
---------------------
Every step that produces a real decision is appended to that battle's trajectory
buffer. When the battle ends, rewards are filled in (terminal reward = +1 win /
-1 loss, plus per-step shaping) and the full trajectory is shipped to the
trajectory_queue for the learner to train on.
"""

import asyncio
import concurrent.futures
import logging
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

if TYPE_CHECKING:
    from elitefurretai.rl.inference_client import InferenceClient

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon
from poke_env.calc import calculate_damage
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.data import GenData
from poke_env.player import BattleOrder, DoubleBattleOrder, Player
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.masking import (
    fast_get_action_mask,
    get_valid_targets,
    slot_is_commanding,
)
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

logger = logging.getLogger("MaxDamagePlayer")


def _request_fingerprint(request: Optional[Dict[str, Any]]) -> Optional[tuple]:
    # ── Why this exists ──────────────────────────────────────────────────────
    # Showdown sends us a fresh "request" payload before every decision point.
    # The request describes which moves are available, who's active, whether
    # we must force-switch, whether terastallization is allowed, etc.
    #
    # We use this payload to (a) build the action mask and (b) decode the
    # network's chosen action back into a Showdown command. Both must agree
    # on what the request looked like.
    #
    # But because inference is async, by the time the model answers, Showdown
    # may have sent a *new* request (e.g. a force-switch was triggered after
    # an opponent KO'd us). If we mask & decode against different requests,
    # we'll send invalid commands.
    #
    # This function condenses a request into a hashable tuple ("fingerprint").
    # Compare fingerprints before sending to detect drift; if they don't match,
    # discard the decision and let the next request handler re-decide.
    # ─────────────────────────────────────────────────────────────────────────
    if request is None:
        return None

    active_entries = []
    for active in request.get("active", []) or []:
        if not isinstance(active, dict):
            active_entries.append(None)
            continue
        moves = []
        for move in active.get("moves", []) or []:
            if not isinstance(move, dict):
                continue
            moves.append(
                (
                    move.get("id"),
                    move.get("target"),
                    move.get("disabled"),
                    move.get("pp"),
                )
            )
        active_entries.append(
            (
                tuple(moves),
                active.get("canTerastallize"),
                active.get("trapped"),
                active.get("maybeTrapped"),
                active.get("commanding"),
            )
        )

    side_entries = []
    side = request.get("side")
    side_pokemon = side.get("pokemon", []) if isinstance(side, dict) else []
    for mon in side_pokemon:
        if not isinstance(mon, dict) or not mon.get("active", False):
            continue
        side_entries.append(
            (
                mon.get("ident"),
                mon.get("condition"),
                mon.get("active"),
                mon.get("commanding"),
                mon.get("terastallized"),
            )
        )

    force_switch = request.get("forceSwitch")
    force_switch_fingerprint = (
        tuple(bool(value) for value in force_switch)
        if isinstance(force_switch, list)
        else None
    )

    return (
        request.get("rqid"),
        bool(request.get("teamPreview")),
        bool(request.get("wait")),
        force_switch_fingerprint,
        tuple(active_entries),
        tuple(side_entries),
    )


# ── Inference executor management ────────────────────────────────────────────
# We run model.forward() in a background thread (via ThreadPoolExecutor) so the
# main asyncio event loop (which handles all the websocket I/O for poke-env)
# isn't blocked while a forward pass runs. One executor per worker keeps
# inference work isolated and makes thread accounting tidy.
#
# Why max_workers=1: we WANT serialization here. If we let two threads run
# forward passes concurrently inside the same worker process, they'd compete
# for the same model state, the same Python GIL, and the same CPU. Better to
# queue them and run one at a time.
# ─────────────────────────────────────────────────────────────────────────────
_WORKER_EXECUTORS: Dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = Lock()

# Global fallback executor for non-worker contexts (e.g., testing or scripts)
_FALLBACK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="fallback_inference"
)


def get_worker_executor(worker_id: Optional[int] = None) -> ThreadPoolExecutor:
    """Get or create a ThreadPoolExecutor for a specific worker."""
    if worker_id is None:
        return _FALLBACK_EXECUTOR

    with _EXECUTOR_LOCK:
        if worker_id not in _WORKER_EXECUTORS:
            _WORKER_EXECUTORS[worker_id] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"worker{worker_id}_inference",
            )
        return _WORKER_EXECUTORS[worker_id]


def cleanup_worker_executors() -> None:
    """Shutdown all worker executors (call at training end)."""
    with _EXECUTOR_LOCK:
        for executor in _WORKER_EXECUTORS.values():
            executor.shutdown(wait=False)
        _WORKER_EXECUTORS.clear()


class BatchInferencePlayer(Player):
    """High-performance player with async batched inference.

    This is the workhorse of the RL data-collection pipeline. It subclasses
    poke-env's `Player` (which handles the websocket protocol with Showdown)
    and adds three things:

      1. **Batched inference**: an asyncio queue + background loop that gathers
         many simultaneous decision requests, runs ONE batched model forward
         pass over them, and dispatches the answers. Drastically reduces
         per-decision overhead.

      2. **Trajectory collection**: every (state, action, log_prob, value, mask)
         tuple is stashed in a per-battle buffer; when the battle finishes,
         rewards are filled in and the trajectory is shipped to the learner.

      3. **Stale-decision detection**: because inference is async, the battle
         state may change while we're waiting. We snapshot the request before
         queueing, fingerprint it, and drop decisions that no longer match
         when the answer comes back. Without this, we get cascading invalid
         choice errors from Showdown.

    Three layers of defense against stale decisions:
      - request_generation counter: a monotonic per-battle id; if a newer
        request was issued while we were waiting, drop the old answer.
      - turn/teampreview/force_switch tuple: detect coarse state changes.
      - request fingerprint: detect fine-grained legality changes (e.g. a move
        becoming disabled, terastallization availability changing).
    """

    def __init__(
        self,
        model: Optional["RNaDAgent"] = None,
        device="cpu",
        batch_size=16,
        batch_timeout=0.01,
        probabilistic=True,
        trajectory_queue=None,
        accept_open_team_sheet=False,
        worker_id: Optional[int] = None,
        embedder: Optional[Embedder] = None,
        max_battle_steps: int = 40,
        opponent_type: str = "self_play",
        # Centralized-inference: when set, the player submits requests
        # to a shared trainer-side InferenceService instead of running
        # a per-player inference loop. Mutually exclusive with `model`.
        inference_client: Optional["InferenceClient"] = None,
        **kwargs,
    ):
        battle_format = kwargs.get("battle_format", "gen9vgc2023regc")

        if (model is None) == (inference_client is None):
            raise ValueError(
                "BatchInferencePlayer requires exactly one of `model` "
                "(legacy per-player inference) or `inference_client` "
                "(centralized inference via InferenceService)"
            )

        self.model = model
        self.inference_client = inference_client
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.probabilistic = probabilistic
        self.trajectory_queue = trajectory_queue
        self.worker_id = worker_id
        self.opponent_type = opponent_type
        self.embedder = (
            embedder
            if embedder is not None
            else Embedder(
                format=battle_format, feature_set=Embedder.FULL, omniscient=False
            )
        )
        # Legacy mode owns a per-player asyncio queue + inference loop.
        # Centralized mode submits via `self.inference_client` and never
        # touches `self.queue`. Type kept as Optional so attribute always
        # exists for callers that defensively check it.
        self.queue: Optional[asyncio.Queue] = (
            create_in_poke_loop(asyncio.Queue, POKE_LOOP) if model is not None else None
        )
        # context tensor accumulated across turns by TransformerThreeHeadedModel
        self.hidden_states: Dict[str, Any] = {}
        self._inference_task: Optional[asyncio.Task] = None
        self._inference_future: Optional[concurrent.futures.Future] = None
        self.temperature: float = 1.0  # Sampling temperature (set by trainer)
        self.top_p: float = 1.0  # Nucleus sampling threshold (set by trainer)
        self.current_trajectories: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        self.completed_trajectories: List[Dict[str, Any]] = []
        self._discarded_battles: set[str] = set()
        # Per-battle request generation counter. Incremented on each battle
        # request handler invocation to suppress stale async sends.
        self._request_generation: Dict[str, int] = {}
        self.max_battle_steps = max_battle_steps
        # Timeout for waiting on a batched inference response.
        # Why: if the inference loop stalls, we prefer fallback behavior over hanging
        # a battle coroutine indefinitely.
        self.inference_request_timeout_s = 8.0
        self._diagnostics: Dict[str, float] = {
            "requests_total": 0.0,
            "default_choice_requests": 0.0,
            "embed_calls": 0.0,
            "embed_seconds": 0.0,
            "inference_batches": 0.0,
            "inference_batch_items": 0.0,
            "inference_batch_size_max": 0.0,
            "inference_batches_filled_to_max": 0.0,
            "inference_batches_flushed_timeout": 0.0,
            "inference_executor_seconds": 0.0,
            "inference_wait_calls": 0.0,
            "inference_wait_seconds": 0.0,
            "inference_timeouts": 0.0,
            "transformer_batched_calls": 0.0,
            "transformer_context_items": 0.0,
            "transformer_context_tokens_real": 0.0,
            "transformer_context_tokens_padded": 0.0,
            "transformer_context_len_max": 0.0,
            "stale_request_generation_drops": 0.0,
            "stale_inference_drops": 0.0,
            "stale_request_snapshot_drops": 0.0,
            "mask_mismatches": 0.0,
            "send_failures": 0.0,
            "default_send_failures": 0.0,
            "discarded_battles": 0.0,
            "trajectory_steps_buffered": 0.0,
            "completed_trajectories": 0.0,
            "completed_trajectory_steps": 0.0,
            "room_lost_recoveries": 0.0,
            "message_handler_timeouts": 0.0,
            "battle_lock_tasks_cancelled": 0.0,
        }

        # Battles finalized via the "not in that room" popup recovery path
        # (see _recover_room_lost_battle). Read by _battle_finished_callback so
        # those trajectories are shipped with forfeited=True and dropped from
        # opponent win-rate tracking.
        self._room_lost_battles: set[str] = set()

        super().__init__(accept_open_team_sheet=accept_open_team_sheet, **kwargs)

        # Install popup-recovery hook on the ps_client. Showdown emits
        # |popup|...not in that room... when we send /choose to a battle whose
        # room has already been closed server-side (because |win| was processed
        # but our local state hadn't flipped yet). poke-env's default popup
        # handling is just a logger.warning — the affected battle stays
        # "in progress" forever locally and stalls the worker. This hook
        # detects the popup, parses the battle tag, and finalizes the battle.
        self._original_handle_message = self.ps_client._handle_message
        self.ps_client._handle_message = self._handle_message_with_popup_recovery

    def _embed_battle_state(self, battle: Any) -> np.ndarray:
        embed_start = asyncio.get_running_loop().time()
        embed_to_array = getattr(self.embedder, "embed_to_array", None)
        if callable(embed_to_array):
            result = cast(np.ndarray, embed_to_array(cast(DoubleBattle, battle)))
        else:
            result = np.asarray(
                self.embedder.embed_to_vector(cast(DoubleBattle, battle)),
                dtype=np.float32,
            )
        self._diagnostics["embed_calls"] += 1
        self._diagnostics["embed_seconds"] += (
            asyncio.get_running_loop().time() - embed_start
        )
        return result

    def clear_completed_trajectories(self) -> None:
        self.completed_trajectories.clear()

    def get_diagnostics_snapshot(self) -> Dict[str, float]:
        return dict(self._diagnostics)

    async def stop_listening(self):
        await self.ps_client.stop_listening()

    def _reset_battle_hidden_state(self, battle_tag: str) -> None:
        """Drop hidden state for a battle from BOTH the player-local dict
        AND the trainer-side InferenceService (when in centralized mode).

        The legacy code resets hidden on every stale-request / timeout /
        error path so the next request for that battle starts from scratch.
        Without this, multiple in-flight requests' hidden updates pile up
        and the context tensor exceeds `max_seq_len`, crashing the
        positional encoder. Centralized mode hits the same race; the
        trainer-side dict needs the same reset.
        """
        self.hidden_states.pop(battle_tag, None)
        if self.inference_client is not None:
            self.inference_client.evict(self.username, battle_tag)

    def stop_inference_loop(self, timeout_s: float = 1.0) -> None:
        """Stop the background inference loop if it is running.

        Why: during worker-side rebuilds, we must stop old loop tasks before creating
        fresh players, or stale loops can keep references alive and leak pending work.
        """
        if self._inference_future is None:
            return

        # First request cooperative cancellation.
        self._inference_future.cancel()

        # Then drain completion briefly so the loop has a chance to unwind cleanly.
        try:
            self._inference_future.result(timeout=timeout_s)
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            pass
        except Exception:
            pass
        finally:
            self._inference_future = None

    def teardown_runtime(self, timeout_s: float = 1.5) -> None:
        """Best-effort teardown of inference + websocket listener state.

        Why: this is the explicit teardown step needed before rebuilding agents.
        Without it, old clients can remain logged in, causing `|nametaken|` collisions
        and repeated timeout loops after a desync event.
        """
        # Stop inference first so we don't enqueue decisions while disconnecting.
        self.stop_inference_loop(timeout_s=timeout_s)

        # Ask poke-env to stop websocket listening on this player.
        try:
            fut = asyncio.run_coroutine_threadsafe(self.stop_listening(), POKE_LOOP)
            fut.result(timeout=timeout_s)
        except Exception:
            pass

    def start_inference_loop(self) -> None:
        # Centralized mode: the trainer-side InferenceService owns its own
        # loop; players have no per-instance loop to start. No-op.
        if self.inference_client is not None:
            return
        self._inference_future = asyncio.run_coroutine_threadsafe(
            self._inference_loop(), POKE_LOOP
        )

    async def _inference_loop(self):
        # ── The dynamic-batching heart of this class ─────────────────────────
        # Loop forever:
        #   1. Block until at least ONE decision request arrives.
        #   2. Greedily pull more requests off the queue, but never wait longer
        #      than `batch_timeout` total and never gather more than `batch_size`.
        #   3. Run ONE model forward pass over the gathered batch.
        #   4. Set the result on each pending future so the awaiting coroutines
        #      can resume.
        #
        # The trade-off is throughput vs latency:
        #   - Big batch_timeout / big batch_size = larger batches, less wasted
        #     model overhead, but each individual decision waits longer.
        #   - Small values = snappier per-decision but more wasted forwards.
        # ─────────────────────────────────────────────────────────────────────
        # This loop only runs in legacy mode (model + per-player queue).
        # Centralized mode never starts the inference loop; the trainer-side
        # InferenceService owns its equivalent.
        assert self.queue is not None, (
            "_inference_loop entered without a queue — centralized-mode "
            "players should not start the legacy inference loop"
        )
        while True:
            batch: List[Any] = []
            futures: List[Any] = []
            battle_tags: List[str] = []
            is_tps: List[bool] = []
            masks: List[Any] = []
            try:
                # Step 1: wait for at least one request — no point batching
                # nothing.
                item = await self.queue.get()
                self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)

                # Step 2: opportunistically gather more, capped by both batch
                # size and elapsed time since the first item arrived.
                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        self._add_to_batch(
                            batch, futures, battle_tags, is_tps, masks, item
                        )
                    except asyncio.TimeoutError:
                        # No more requests arrived within the window — flush.
                        break
            except asyncio.CancelledError:
                # Worker shutting down; exit loop cleanly.
                break

            if batch:
                # Diagnostics: track batch fill quality so we can tell whether
                # batch_size and batch_timeout are well-tuned.
                self._diagnostics["inference_batches"] += 1
                self._diagnostics["inference_batch_items"] += len(batch)
                self._diagnostics["inference_batch_size_max"] = max(
                    self._diagnostics["inference_batch_size_max"],
                    float(len(batch)),
                )
                if len(batch) >= self.batch_size:
                    self._diagnostics["inference_batches_filled_to_max"] += 1
                else:
                    self._diagnostics["inference_batches_flushed_timeout"] += 1
                # Periodic batch-fill log so we can tell at a glance whether
                # batch_size is well-tuned. Emits every 500 batches per player.
                if self._diagnostics["inference_batches"] % 500 == 0:
                    n = self._diagnostics["inference_batches"]
                    items = self._diagnostics["inference_batch_items"]
                    filled = self._diagnostics["inference_batches_filled_to_max"]
                    timeout = self._diagnostics["inference_batches_flushed_timeout"]
                    # WARNING level chosen so the line surfaces past the
                    # worker's INFO-suppression for poke-env loggers. This is
                    # diagnostic and should be lowered or removed once
                    # batch_size has been tuned (2026-05-13 throughput work).
                    self.logger.warning(
                        "[batch-fill] n=%d avg=%.2f max=%d filled%%=%.1f timeout%%=%.1f cap=%d",
                        int(n),
                        items / max(n, 1),
                        int(self._diagnostics["inference_batch_size_max"]),
                        100.0 * filled / max(n, 1),
                        100.0 * timeout / max(n, 1),
                        self.batch_size,
                    )
                # Step 3+4: run the batched forward and resolve futures.
                await self._run_batch(batch, futures, battle_tags, is_tps, masks)

    def _add_to_batch(self, batch, futures, battle_tags, is_tps, masks, item):
        batch.append(item[0])
        futures.append(item[1])
        battle_tags.append(item[2])
        is_tps.append(item[3])
        masks.append(item[4])

    def _gpu_inference_sync(self, states_np, hidden_cpu, hidden_mask_cpu=None):
        # Legacy-mode-only: only called from _run_batch which only runs in
        # legacy mode. Centralized mode routes inference through the
        # trainer's RealModelBatchHandler.
        assert self.model is not None, (
            "_gpu_inference_sync called without a model — centralized-mode "
            "players don't own a model and shouldn't enter this path"
        )
        states_tensor = (
            torch.tensor(states_np, dtype=torch.float32).to(self.device).unsqueeze(1)
        )

        # Transformer: hidden_cpu is the accumulated context tensor or None
        hidden = hidden_cpu.to(self.device) if hidden_cpu is not None else None
        hidden_mask = (
            hidden_mask_cpu.to(self.device) if hidden_mask_cpu is not None else None
        )

        with torch.no_grad():
            turn_logits, tp_logits, values, _, next_hidden = self.model(
                states_tensor,
                hidden,
                hidden_mask=hidden_mask,
            )

        # ── Two probability distributions, one for sampling, one for PPO ────
        # We compute TWO different softmaxes here, and this distinction is
        # subtle but important.
        #
        # `turn_probs` / `tp_probs` (temperature-scaled):
        #   Used to actually SAMPLE the action. Higher temperature flattens
        #   the distribution → more exploration. Lower temperature sharpens
        #   it → more exploitation. We anneal temperature down over training.
        #
        # `turn_log_probs` / `tp_log_probs` (T=1, unscaled):
        #   Recorded into the trajectory for later use as `old_log_prob` in
        #   PPO's importance ratio. PPO assumes these come from the *true*
        #   policy distribution. If we used the temperature-scaled log-probs
        #   here, the importance ratio would be biased and the gradient
        #   estimate would be wrong.
        #
        # In short: temperature is a sampling-time exploration knob; PPO math
        # always uses the underlying T=1 distribution.
        # ─────────────────────────────────────────────────────────────────────
        temp = max(self.temperature, 1e-6)
        turn_probs = torch.softmax(turn_logits / temp, dim=-1).cpu().numpy()
        tp_probs = torch.softmax(tp_logits / temp, dim=-1).cpu().numpy()

        turn_log_probs = torch.log_softmax(turn_logits, dim=-1).cpu().numpy()
        tp_log_probs = torch.log_softmax(tp_logits, dim=-1).cpu().numpy()

        values_np = values.cpu().numpy()

        # next_hidden is the accumulated context tensor (batch, T, H)
        next_hidden_cpu = next_hidden.cpu()

        return (
            turn_probs,
            tp_probs,
            turn_log_probs,
            tp_log_probs,
            values_np,
            next_hidden_cpu,
        )

    async def _run_batch(self, states, futures, battle_tags, is_tps, masks):
        # ── Where the batched forward pass actually runs ─────────────────────
        # Inputs are lists, one element per gathered request:
        #   states       — the embedded battle state vectors
        #   futures      — the asyncio futures awaited by each requesting battle
        #   battle_tags  — id of the battle each state came from (used to look
        #                  up that battle's accumulated transformer context)
        #   is_tps       — booleans: is this a teampreview decision (90 actions)
        #                  or a turn decision (2025 actions)?
        #   masks        — per-state legality masks (for turn decisions only)
        # ─────────────────────────────────────────────────────────────────────
        states_np = np.array(states)

        context_lengths: List[int] = []
        hidden_size: Optional[int] = None
        context_tensors: List[Optional[torch.Tensor]] = []
        for tag in battle_tags:
            ctx = cast(Optional[torch.Tensor], self.hidden_states.get(tag, None))
            context_tensors.append(ctx)
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
        hidden_batch_cpu: Optional[torch.Tensor] = None
        hidden_mask_cpu: Optional[torch.Tensor] = None
        batch_size = len(battle_tags)
        if max_context_len > 0:
            if hidden_size is None:
                for ctx in context_tensors:
                    if ctx is not None:
                        hidden_size = int(ctx.shape[2])
                        break
            assert hidden_size is not None
            hidden_batch_cpu = torch.zeros(
                batch_size,
                max_context_len,
                hidden_size,
                dtype=torch.float32,
            )
            hidden_mask_cpu = torch.zeros(
                batch_size,
                max_context_len,
                dtype=torch.bool,
            )
            for index, ctx in enumerate(context_tensors):
                if ctx is None:
                    continue
                length = context_lengths[index]
                if length == 0:
                    continue
                hidden_batch_cpu[index, :length, :] = ctx[0, :length, :]
                hidden_mask_cpu[index, :length] = True

        self._diagnostics["transformer_batched_calls"] += 1
        self._diagnostics["transformer_context_items"] += batch_size
        self._diagnostics["transformer_context_tokens_real"] += float(
            sum(context_lengths)
        )
        self._diagnostics["transformer_context_tokens_padded"] += float(
            batch_size * max_context_len
        )
        self._diagnostics["transformer_context_len_max"] = max(
            self._diagnostics["transformer_context_len_max"],
            float(max_context_len),
        )

        loop = asyncio.get_running_loop()
        executor = get_worker_executor(self.worker_id)
        inference_start = loop.time()
        (
            turn_probs,
            tp_probs,
            turn_log_probs,
            tp_log_probs,
            values,
            next_ctx_batch,
        ) = await loop.run_in_executor(
            executor,
            self._gpu_inference_sync,
            states_np,
            hidden_batch_cpu,
            hidden_mask_cpu,
        )
        next_ctx_batch = cast(torch.Tensor, next_ctx_batch)
        self._diagnostics["inference_executor_seconds"] += (
            asyncio.get_running_loop().time() - inference_start
        )

        next_lengths = [length + 1 for length in context_lengths]
        all_results: List[Dict[str, Any]] = []
        for i, tag in enumerate(battle_tags):
            next_len = next_lengths[i]
            self.hidden_states[tag] = next_ctx_batch[i : i + 1, :next_len, :].clone()
            all_results.append(
                {
                    "turn_probs": turn_probs[i, 0],
                    "tp_probs": tp_probs[i, 0],
                    "turn_log_probs": turn_log_probs[i, 0],
                    "tp_log_probs": tp_log_probs[i, 0],
                    "value": values[i, 0],
                }
            )

        for i, future in enumerate(futures):
            is_tp = is_tps[i]
            mask = masks[i]
            r = all_results[i]

            if is_tp:
                probs = r["tp_probs"]
                unscaled_log_probs = r["tp_log_probs"]
                valid_actions = list(range(len(probs)))
            else:
                probs = r["turn_probs"]
                unscaled_log_probs = r["turn_log_probs"]
                if mask is not None:
                    probs = probs * mask
                    if probs.sum() == 0:
                        probs = mask / mask.sum()
                    else:
                        probs = probs / probs.sum()

                    if self.top_p < 1.0:
                        sorted_idx = np.argsort(-probs)
                        cum = np.cumsum(probs[sorted_idx])
                        cutoff = np.searchsorted(cum, self.top_p) + 1
                        keep = sorted_idx[:cutoff]
                        filtered = np.zeros_like(probs)
                        filtered[keep] = probs[keep]
                        probs = filtered / filtered.sum()

                valid_actions = list(range(len(probs)))

            action = (
                np.random.choice(valid_actions, p=probs)
                if self.probabilistic
                else np.argmax(probs)
            )

            # Compute log_prob from the MASKED distribution so PPO
            # ratios are consistent with the learner (which also masks).
            if mask is not None:
                valid_mask = mask.astype(bool)
                log_valid_mass = np.log(np.exp(unscaled_log_probs[valid_mask]).sum())
                log_prob = float(unscaled_log_probs[action] - log_valid_mass)
            else:
                log_prob = float(unscaled_log_probs[action])

            future.set_result(
                {
                    "action": action,
                    "log_prob": log_prob,
                    "value": r["value"],
                    "probs": probs,
                }
            )

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        self._diagnostics["requests_total"] += 1

        # Defensive guard: do not attempt to send orders for battles that are already
        # marked finished locally.
        if getattr(battle, "finished", False):
            return

        request_generation = self._request_generation.get(battle.battle_tag, 0) + 1
        self._request_generation[battle.battle_tag] = request_generation

        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            self._diagnostics["default_choice_requests"] += 1
            message = self.choose_default_move().message
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
                self._diagnostics["default_send_failures"] += 1
                logger.warning(
                    "DEFAULT_SEND_FAILURE "
                    f"tag={battle.battle_tag} turn={getattr(battle, 'turn', '?')} "
                    f"message={message} err={repr(exc)}"
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self._reset_battle_hidden_state(battle.battle_tag)
            return

        choice = await self._choose_move_async(
            battle,
            request_generation=request_generation,
        )

        # Drop if a newer request superseded this one while inference/order
        # computation was in flight.
        if self._request_generation.get(battle.battle_tag, -1) != request_generation:
            return

        # Drop if battle finished before send.
        if getattr(battle, "finished", False):
            return

        if isinstance(choice, str):
            message = choice
        elif hasattr(choice, "message"):
            message = choice.message
        else:
            message = str(choice)

        # Root-cause guardrail #1: TODO fix
        # If battle requires a forced switch but the chosen message is a move command,
        # rewrite to default order to avoid guaranteed invalid-choice errors.
        if (
            isinstance(battle, DoubleBattle)
            and any(battle.force_switch)
            and isinstance(message, str)
            and message.startswith("/choose move")
        ):
            message = self.choose_default_move().message

        # Root-cause guardrail #2:
        # If no active slot can tera, strip accidental tera directive from message.
        if (
            isinstance(battle, DoubleBattle)
            and isinstance(message, str)
            and "terastallize" in message
            and hasattr(battle, "can_tera")
            and not any(getattr(battle, "can_tera", []))
        ):
            message = message.replace(" terastallize", "")

        if message:
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
                self._diagnostics["send_failures"] += 1
                # Rich context to diagnose first trigger root causes:
                # - which battle tag failed
                # - what message we attempted
                # - legal move/switch context where available
                active_names = []
                legal_move_names = []
                legal_switch_names = []
                if isinstance(battle, DoubleBattle):
                    active_names = [
                        p.species for p in battle.active_pokemon if p is not None
                    ]
                    for moves in battle.available_moves:
                        legal_move_names.append([m.id for m in moves])
                    for switches in battle.available_switches:
                        legal_switch_names.append([p.species for p in switches])

                logger.warning(
                    "SEND_FAILURE "
                    f"tag={battle.battle_tag} turn={getattr(battle, 'turn', '?')} "
                    f"teampreview={getattr(battle, 'teampreview', False)} "
                    f"force_switch={getattr(battle, 'force_switch', None)} "
                    f"message={message} active={active_names} legal_moves={legal_move_names} "
                    f"legal_switches={legal_switch_names} err={repr(exc)}"
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self._reset_battle_hidden_state(battle.battle_tag)

    def choose_move(self, battle: AbstractBattle) -> Any:
        return self._choose_move_async(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        raise RuntimeError(
            "teampreview() should not be called; decisions are routed through choose_move()"
        )

    async def _choose_move_async(
        self,
        battle: AbstractBattle,
        request_generation: Optional[int] = None,
    ):
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()

        current_steps = len(self.current_trajectories.get(battle.battle_tag, []))
        if current_steps >= self.max_battle_steps:
            self._diagnostics["discarded_battles"] += 1
            self._discarded_battles.add(battle.battle_tag)
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return (
                "/forfeit" if self.trajectory_queue is not None else DefaultBattleOrder()
            )

        try:
            state = self._embed_battle_state(battle)
        except Exception:
            if battle.teampreview:
                return "/team 1234"
            try:
                return self.choose_random_doubles_move(battle)
            except Exception:
                return DefaultBattleOrder()
        # The mask and final MDBO decode must use the same request payload. In the
        # async batched path, battle.last_request can change mid-turn while inference
        # is in flight; without a snapshot we can sample under one button layout and
        # serialize under another, producing invalid Showdown commands.
        request_snapshot = (
            deepcopy(battle.last_request) if battle.last_request is not None else None
        )
        # This is cheaper than recomputing the full order twice and lets us drop the
        # result if the live request has drifted before we send it.
        request_fingerprint = _request_fingerprint(request_snapshot)
        mask = (
            None
            if battle.teampreview
            else fast_get_action_mask(battle, request_override=request_snapshot)
        )

        # Snapshot request-time battle state so we can detect stale inference outputs.
        request_turn = getattr(battle, "turn", -1)
        request_teampreview = battle.teampreview
        request_force_switch = (
            tuple(bool(x) for x in battle.force_switch)
            if isinstance(battle, DoubleBattle)
            else tuple()
        )

        wait_start = asyncio.get_running_loop().time()
        try:
            if self.inference_client is not None:
                # Centralized path: trainer-side InferenceService runs the
                # forward + sampling AND owns the hidden state, keyed by
                # (worker_id, battle_tag). The wire payload only carries
                # the lightweight battle_tag. Player no longer tracks
                # hidden_states locally in this mode.
                response = await asyncio.wait_for(
                    self.inference_client.submit(
                        state=state,
                        mask=mask,
                        is_teampreview=battle.teampreview,
                        player_id=self.username,
                        battle_tag=battle.battle_tag,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    ),
                    timeout=self.inference_request_timeout_s,
                )
                result: Dict[str, Any] = {
                    "action": response.action_idx,
                    "log_prob": response.log_prob,
                    "value": response.value,
                }
            else:
                # Legacy path: per-player asyncio queue + inference loop.
                assert self.queue is not None
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                await self.queue.put(
                    (state, future, battle.battle_tag, battle.teampreview, mask)
                )
                result = await asyncio.wait_for(
                    future, timeout=self.inference_request_timeout_s
                )
            self._diagnostics["inference_wait_calls"] += 1
            self._diagnostics["inference_wait_seconds"] += (
                asyncio.get_running_loop().time() - wait_start
            )
        except asyncio.TimeoutError:
            self._diagnostics["inference_wait_calls"] += 1
            self._diagnostics["inference_wait_seconds"] += (
                asyncio.get_running_loop().time() - wait_start
            )
            self._diagnostics["inference_timeouts"] += 1
            queue_repr = (
                self.queue.qsize()
                if self.queue is not None and hasattr(self.queue, "qsize")
                else "centralized"
            )
            logger.debug(
                "INFERENCE_TIMEOUT tag=%s turn=%s teampreview=%s queue_size=%s",
                battle.battle_tag,
                getattr(battle, "turn", "?"),
                battle.teampreview,
                queue_repr,
            )
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return DefaultBattleOrder()

        # Request-generation guard: if a newer request is already active for this
        # battle tag, drop this stale result before decoding/sending.
        if request_generation is not None:
            latest_generation = self._request_generation.get(battle.battle_tag, -1)
            if latest_generation != request_generation:
                self._diagnostics["stale_request_generation_drops"] += 1
                logger.debug(
                    "STALE_REQUEST_GENERATION_DROP tag=%s request_gen=%d latest_gen=%d",
                    battle.battle_tag,
                    request_generation,
                    latest_generation,
                )
                self.current_trajectories.pop(battle.battle_tag, None)
                self._reset_battle_hidden_state(battle.battle_tag)
                return DefaultBattleOrder()

        action_idx = result["action"]

        # Root-cause guardrail #3:
        # If battle state changed while waiting on batched inference, discard this
        # decision instead of sending a potentially invalid/stale command.
        current_turn = getattr(battle, "turn", -1)
        current_teampreview = battle.teampreview
        current_force_switch = (
            tuple(bool(x) for x in battle.force_switch)
            if isinstance(battle, DoubleBattle)
            else tuple()
        )
        if (
            current_turn != request_turn
            or current_teampreview != request_teampreview
            or current_force_switch != request_force_switch
        ):
            self._diagnostics["stale_inference_drops"] += 1
            logger.debug(
                "STALE_INFERENCE_DROP tag=%s turn=%d->%d tp=%s->%s fs=%s->%s",
                battle.battle_tag,
                request_turn,
                current_turn,
                request_teampreview,
                current_teampreview,
                request_force_switch,
                current_force_switch,
            )
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return DefaultBattleOrder()

        current_request_fingerprint = _request_fingerprint(battle.last_request)
        if current_request_fingerprint != request_fingerprint:
            self._diagnostics["stale_request_snapshot_drops"] += 1
            logger.debug(
                "STALE_REQUEST_SNAPSHOT_DROP tag=%s turn=%d rqid=%s->%s",
                battle.battle_tag,
                current_turn,
                None if request_snapshot is None else request_snapshot.get("rqid"),
                None if battle.last_request is None else battle.last_request.get("rqid"),
            )
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return DefaultBattleOrder()

        if mask is not None and action_idx < len(mask) and mask[action_idx] == 0:
            self._diagnostics["mask_mismatches"] += 1
            logger.warning(
                "MASK_MISMATCH tag=%s turn=%s action_idx=%d mask_value=%s",
                battle.battle_tag,
                getattr(battle, "turn", "?"),
                action_idx,
                mask[action_idx],
            )

        if self.trajectory_queue is not None:
            self._diagnostics["trajectory_steps_buffered"] += 1
            self.current_trajectories.setdefault(battle.battle_tag, []).append(
                {
                    "state": state,
                    "action": action_idx,
                    "log_prob": result["log_prob"],
                    "value": result["value"],
                    "reward": 0,
                    "is_teampreview": battle.teampreview,
                    "mask": mask,
                    "opponent_fainted": sum(
                        1 for m in battle.opponent_team.values() if m.fainted
                    ),
                }
            )
        else:
            self.current_trajectories.setdefault(battle.battle_tag, []).append(None)

        if battle.teampreview:
            try:
                return MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW).message
            except (ValueError, AssertionError):
                return "/team 123456"

        try:
            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            mdbo = MDBO.from_int(action_idx, type=action_type)
            return mdbo.to_double_battle_order(battle, request=request_snapshot)
        except (ValueError, KeyError, AttributeError, IndexError, AssertionError):
            return DefaultBattleOrder()

    # Compiled once: matches the body of the popup Showdown emits when a
    # /choose lands on a closed room, e.g.
    #   |popup|You tried to send "/choose move ..." to the room
    #   "battle-gen9vgc2024regg-12345" but it failed because you were not in
    #   that room.
    _ROOM_LOST_POPUP_RE = re.compile(
        r'\|popup\|.*to the room "(battle-[^"]+)".*not in that room'
    )

    # Hard cap on time spent in any single ps_client message handler. Used to
    # bound the per-message wrapper task lifetime so a hung `_handle_battle_message`
    # cannot pile up indefinite waiters on the per-battle lock in ps_client.
    # No legitimate handler runs longer than a few seconds; 60s is conservative.
    _MESSAGE_HANDLER_TIMEOUT_S: float = 60.0

    async def _handle_message_with_popup_recovery(self, message: str) -> None:
        """Wrap the underlying ps_client._handle_message to (1) recover from
        "not in that room" popups and (2) bound handler runtime.

        Without (1), the popup is a logger warning only — the affected battle
        keeps `finished=False` locally even though the room is gone server-side,
        so its `send_challenges` task hangs in `_battle_count_queue.join()` and
        the worker eventually trips the zero-completion safety guard. See
        `planning/stage2/2026-05-06-04-50-zero-completion-room-state-race.md`.

        Without (2), a single hung `_handle_battle_message` call (holding the
        per-battle lock in ps_client._battle_locks) causes every subsequent
        message for the same battle to stack up as a lock waiter, leaking
        ~30 tasks per stuck battle and forcing a WSL OOM after ~12 hours.
        """
        try:
            await asyncio.wait_for(
                self._original_handle_message(message),
                timeout=self._MESSAGE_HANDLER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            self._diagnostics["message_handler_timeouts"] += 1
            self.logger.warning(
                "Message handler exceeded %.0fs timeout; preview=%s",
                self._MESSAGE_HANDLER_TIMEOUT_S,
                message[:200],
            )
            return
        match = self._ROOM_LOST_POPUP_RE.search(message)
        if match is not None:
            await self._recover_room_lost_battle(match.group(1))

    async def _recover_room_lost_battle(self, battle_tag: str) -> None:
        """Mark a battle as forfeit-by-us and run the standard finished path.

        Pairs with the `if not battle.finished` guard added in poke-env's
        Player._handle_battle_message |win|/|tie| branch, which prevents a
        later |win| from double-decrementing _battle_count_queue.
        """
        battle = self._battles.get(battle_tag)
        if battle is None or battle.finished:
            return

        self._diagnostics["room_lost_recoveries"] += 1
        self._room_lost_battles.add(battle_tag)

        # Outcome is unknown (we lost the room); record as a loss for reward
        # shaping but flag forfeited=True so the curriculum drops it from
        # opponent win-rate tracking (per the adaptive-curriculum overhaul
        # plan, Change 1).
        battle._won = False
        battle._finish_battle()

        # Balance _battle_count_queue so send_challenges()'s join() can
        # release. The poke-env-side guard skips a second decrement if |win|
        # later arrives for this same battle.
        try:
            self._battle_count_queue.get_nowait()
            self._battle_count_queue.task_done()
        except asyncio.QueueEmpty:
            # Already balanced (|win| beat us to it). Nothing to do.
            pass

        # Free the per-battle lock in ps_client. If `_handle_battle_message`
        # for this battle is hung, every subsequent message for the same
        # battle is queued as a lock waiter and pinned forever — observed
        # leaking ~30 tasks per stuck battle. Cancelling all ps_client
        # tasks whose repr mentions this battle tag releases both the holder
        # (via `async with` finalizer) and the waiters (via their acquire()
        # raising CancelledError). Then drop the lock entry so future
        # messages for the same tag get a fresh, unowned lock.
        ps_client = self.ps_client
        battle_lock_map = getattr(ps_client, "_battle_locks", None)
        if battle_lock_map is not None and battle_tag in battle_lock_map:
            cancelled = 0
            for task in list(getattr(ps_client, "_active_tasks", ())):
                if not task.done() and battle_tag in repr(task):
                    task.cancel()
                    cancelled += 1
            if cancelled:
                self._diagnostics["battle_lock_tasks_cancelled"] += cancelled
            battle_lock_map.pop(battle_tag, None)

        self._battle_finished_callback(battle)

        async with self._battle_end_condition:
            self._battle_end_condition.notify_all()

    def _battle_finished_callback(self, battle: AbstractBattle):
        # ── End-of-battle: assign rewards and ship the trajectory ────────────
        # poke-env calls this hook once per battle when it finishes (win, loss,
        # draw, or forfeit). This is where we:
        #   1. Walk the per-step trajectory we've been collecting.
        #   2. Fill in rewards (we deferred this until the outcome is known).
        #   3. Push the completed trajectory to the trajectory_queue, where the
        #      worker will eventually forward it to the learner via mp.Queue.
        #
        # Reward shaping (a small but important design choice):
        #   - per-step penalty   = -0.005   (encourages winning quickly)
        #   - terminal bonus     = +1 win / -1 loss
        #   - KO bonus           = +0.05 per opponent fainted *this step*
        #
        # The KO bonus is computed by diffing this step's opponent_fainted
        # count against the previous step's. Why per-step delta and not
        # "did anyone faint just now": cleanly handles double-KOs and is
        # robust to multi-turn effects.
        # ─────────────────────────────────────────────────────────────────────
        self._request_generation.pop(battle.battle_tag, None)

        # Centralized inference: trainer-side handler keeps a
        # hidden_states dict keyed by (worker_id, player_id,
        # battle_tag). Tell it to free this side's slot so the dict
        # doesn't grow monotonically. Cheap fire-and-forget IPC message.
        if self.inference_client is not None:
            self.inference_client.evict(self.username, battle.battle_tag)

        # If the battle was discarded mid-flight (e.g. trajectory exceeded
        # max_battle_steps), drop everything — we don't want to train on the
        # truncated trajectory because the terminal reward is undefined.
        if battle.battle_tag in self._discarded_battles:
            self._discarded_battles.discard(battle.battle_tag)
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return

        # If trajectory_queue is None this is an opponent-only player (we're
        # not collecting from this side); just clean up state.
        if self.trajectory_queue is None:
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            return

        if battle.battle_tag in self.current_trajectories:
            traj = self.current_trajectories.pop(battle.battle_tag)
            for t, step in enumerate(traj):
                if step is None:
                    continue
                step_reward = -0.005
                if t == len(traj) - 1:
                    step_reward += 1.0 if battle.won else -1.0
                # KO reward: +0.05 for each opponent pokemon KO'd this step
                prev_fainted = 0
                prev_step = traj[t - 1] if t > 0 else None
                if prev_step is not None:
                    prev_fainted = prev_step["opponent_fainted"]
                ko_delta = step["opponent_fainted"] - prev_fainted
                if ko_delta > 0:
                    step_reward += 0.05 * ko_delta
                step["reward"] = step_reward

            filtered_traj = [step for step in traj if step is not None]
            self._diagnostics["completed_trajectories"] += 1
            self._diagnostics["completed_trajectory_steps"] += len(filtered_traj)
            # Battles finalized via the room-lost popup recovery path have an
            # unknown true outcome, so flag forfeited=True; the curriculum
            # drops these from opponent win-rate tracking.
            forfeited = battle.battle_tag in self._room_lost_battles
            self._room_lost_battles.discard(battle.battle_tag)
            # Ship to the queue. The worker process picks it up and forwards it
            # to the learner over mp.Queue (with metadata for opponent tracking).
            self.trajectory_queue.put(
                {
                    "steps": filtered_traj,
                    "opponent_type": self.opponent_type,
                    "won": battle.won,
                    "battle_length": len(filtered_traj),
                    "forfeited": forfeited,
                }
            )
            self._reset_battle_hidden_state(battle.battle_tag)


class RNaDAgent(torch.nn.Module):
    """RL Agent wrapper around TransformerThreeHeadedModel.

    Why this exists
    ---------------
    The supervised (BC) model maintains a growing context tensor across turns.
    RNaDAgent presents a uniform `forward(x, hidden_state)` API that callers
    use without caring about the underlying architecture details.

    `get_initial_state(batch_size, device)` returns None (empty context) to
    start a fresh battle.

    This is a wrapper, not a model — it has no parameters of its own beyond
    those of the wrapped model.
    """

    def __init__(self, model: TransformerThreeHeadedModel):
        super().__init__()
        self.model = model

    def get_initial_state(self, batch_size: int, device: str):
        # Transformer has no initial hidden state — context starts as None.
        return None

    def forward(self, x, hidden_state=None, mask=None, hidden_mask=None):
        assert isinstance(self.model, TransformerThreeHeadedModel)
        turn_logits, tp_logits, value, win_dist_logits, next_hidden = (
            self.model.forward_with_hidden(x, hidden_state, mask, hidden_mask)
        )
        return turn_logits, tp_logits, value, win_dist_logits, next_hidden


class MaxDamagePlayer(Player):
    """A non-learning, heuristic-only opponent for the curriculum.

    Why we have this
    ----------------
    Pure self-play is unstable: the agent can fall into degenerate cycles where
    it only learns to beat its current self. A diverse curriculum of opponents
    helps. MaxDamage is one of the simplest useful baselines:
        - Estimates damage of every legal move against every opponent.
        - Picks (softmax-sampled, with `temperature`) the highest-damage move.
        - For switch decisions, scores the switch by how much damage the new
          mon could do next turn, scaled by `switch_threshold`.
        - Uses poke-env's calculate_damage, which handles type effectiveness,
          STAB, abilities, items, and stat boosts.

    This player has *no learnable parameters*. It's just a fixed policy.
    The RL agent learning to beat MaxDamage at >50% is a basic sanity check.
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2023regc",
        switch_threshold: float = 1.5,
        temperature: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, battle_format=battle_format, **kwargs)
        self.switch_threshold = switch_threshold
        self.temperature = temperature

    @staticmethod
    def _estimate_evs_and_nature(
        base_stats: Dict[str, int],
    ) -> Tuple[List[int], str]:
        hp, atk, dfn, spa, spd, spe = (
            base_stats["hp"],
            base_stats["atk"],
            base_stats["def"],
            base_stats["spa"],
            base_stats["spd"],
            base_stats["spe"],
        )

        stat_values = {
            "hp": hp,
            "atk": atk,
            "def": dfn,
            "spa": spa,
            "spd": spd,
            "spe": spe,
        }
        highest = max(stat_values, key=lambda k: stat_values[k])

        evs = [0, 0, 0, 0, 0, 0]

        if highest == "hp":
            evs[0] = 252
            evs[2] = 128
            evs[4] = 128
            nature = "calm"
        elif highest == "atk":
            evs[1] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "adamant"
        elif highest == "def":
            evs[2] = 252
            evs[0] = 252
            nature = "bold"
        elif highest == "spa":
            evs[3] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "modest"
        elif highest == "spd":
            evs[4] = 252
            evs[0] = 252
            nature = "calm"
        else:
            evs[5] = 252
            if atk >= spa:
                evs[1] = 252
                nature = "adamant"
            else:
                evs[3] = 252
                nature = "modest"

        return evs, nature

    def _estimate_opponent_stats(self, battle) -> None:
        gen_data = GenData.from_gen(battle.gen)

        for mon in battle.opponent_team.values():
            if mon.stats.get("hp") is not None:
                continue

            base_stats = mon.base_stats
            if not base_stats:
                continue

            evs, nature = self._estimate_evs_and_nature(base_stats)
            ivs = [31, 31, 31, 31, 31, 31]
            level = mon.level if mon.level else 50

            try:
                raw_stats = compute_raw_stats(
                    mon.species, evs, ivs, level, nature, gen_data
                )
                mon.stats = {
                    "hp": raw_stats[0],
                    "atk": raw_stats[1],
                    "def": raw_stats[2],
                    "spa": raw_stats[3],
                    "spd": raw_stats[4],
                    "spe": raw_stats[5],
                }
            except Exception:
                continue

    def teampreview(self, battle) -> str:  # type: ignore
        self._estimate_opponent_stats(battle)
        return self._select_teampreview_by_damage(battle)

    def _select_teampreview_by_damage(self, battle) -> str:
        my_team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())

        if not my_team or not opponent_team:
            return self.random_teampreview(battle)

        pokemon_total_damages: List[Tuple[Pokemon, float]] = []
        for mon in my_team:
            total_damage = 0.0

            if mon.moves:
                for move in mon.moves.values():
                    if not move or move.base_power == 0:
                        continue

                    for opp_mon in opponent_team:
                        try:
                            damage_range = calculate_damage(
                                mon.identifier(battle.player_role),
                                opp_mon.identifier(battle.opponent_role),
                                move,
                                battle,
                            )
                            if damage_range and damage_range[0] is not None:
                                total_damage += (damage_range[0] + damage_range[1]) / 2.0
                        except Exception:
                            continue

            pokemon_total_damages.append((mon, total_damage))

        pokemon_total_damages.sort(key=lambda x: x[1], reverse=True)

        team_list = list(battle.team.values())
        selected_indices: List[int] = []
        for selected_mon, _ in pokemon_total_damages[:4]:
            for idx, mon in enumerate(team_list):
                if (
                    mon.species == selected_mon.species
                    and (idx + 1) not in selected_indices
                ):
                    selected_indices.append(idx + 1)
                    break

        if len(selected_indices) >= 4:
            return "/team " + "".join(str(i) for i in selected_indices)

        return self.random_teampreview(battle)

    def choose_move(self, battle) -> BattleOrder:  # type: ignore
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._estimate_opponent_stats(battle)

        used_switches: Set[str] = set()
        slot_orders: List[BattleOrder] = []

        for slot in range(2):
            active_mon = (
                battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            )

            if slot < len(battle.force_switch) and battle.force_switch[slot]:
                switches = [
                    s
                    for s in battle.available_switches[slot]
                    if s.species not in used_switches
                ]
                if switches:
                    switch_candidates = [
                        (
                            cast(BattleOrder, self.create_order(s)),
                            self._get_best_move_damage(battle, s)[0],
                        )
                        for s in switches
                    ]
                    chosen_order, _ = self._softmax_sample(
                        switch_candidates, self.temperature
                    )
                    slot_orders.append(chosen_order)
                    chosen_payload = self._get_order_payload(chosen_order)
                    if isinstance(chosen_payload, Pokemon):
                        used_switches.add(chosen_payload.species)
                else:
                    slot_orders.append(DefaultBattleOrder())
                continue

            if any(battle.force_switch):
                slot_orders.append(PassBattleOrder())
                continue

            if active_mon is None:
                slot_orders.append(PassBattleOrder())
                continue

            if slot_is_commanding(battle, slot, battle.last_request):
                slot_orders.append(PassBattleOrder())
                continue

            candidates = self._score_available_actions(battle, slot, used_switches)

            if candidates:
                chosen_order, chosen_score = self._softmax_sample(
                    candidates, self.temperature
                )

                chosen_payload = self._get_order_payload(chosen_order)
                if isinstance(chosen_payload, Pokemon):
                    used_switches.add(chosen_payload.species)

                slot_orders.append(chosen_order)
            else:
                slot_orders.append(DefaultBattleOrder())

        if len(slot_orders) == 2:
            return DoubleBattleOrder(
                first_order=cast(SingleBattleOrder, slot_orders[0]),
                second_order=cast(SingleBattleOrder, slot_orders[1]),
            )
        elif len(slot_orders) == 1:
            return DoubleBattleOrder(first_order=cast(SingleBattleOrder, slot_orders[0]))

        return self.choose_random_doubles_move(battle)

    @staticmethod
    def _softmax_sample(
        candidates: Sequence[Tuple[BattleOrder, float]], temperature: float
    ) -> Tuple[BattleOrder, float]:
        if not candidates:
            raise ValueError("Cannot sample from empty candidates list")
        if len(candidates) == 1:
            return candidates[0]

        scores = [s for _, s in candidates]

        if temperature <= 0:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx]

        max_score = max(scores)
        if max_score > 0:
            normalized = [s / max_score for s in scores]
        else:
            idx = random.randrange(len(candidates))
            return candidates[idx]

        scaled = [s / temperature for s in normalized]
        max_scaled = max(scaled)
        exp_scores = [math.exp(s - max_scaled) for s in scaled]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return candidates[i]

        return candidates[-1]

    def _score_available_actions(
        self, battle: DoubleBattle, slot: int, used_switches: Set[str]
    ) -> List[Tuple[BattleOrder, float]]:
        if slot_is_commanding(battle, slot, battle.last_request):
            return []

        available_moves = (
            battle.available_moves[slot] if slot < len(battle.available_moves) else []
        )
        active_mon = (
            battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        )
        candidates: List[Tuple[BattleOrder, float]] = []

        request_moves: List[Dict[str, Any]] = []
        if battle.last_request and slot < len(battle.last_request.get("active", [])):
            raw_request_moves = battle.last_request["active"][slot].get("moves", [])
            if isinstance(raw_request_moves, list):
                request_moves = [
                    move for move in raw_request_moves if isinstance(move, dict)
                ]

        request_move_by_id = {
            move["id"]: move
            for move in request_moves
            if isinstance(move.get("id"), str)
            and not move.get("disabled", False)
            and move.get("pp", 1) > 0
        }

        if request_move_by_id:
            available_moves = [
                move for move in available_moves if move.id in request_move_by_id
            ]

        if available_moves and active_mon is not None:
            for move in available_moves:
                request_move = request_move_by_id.get(move.id)
                targets = get_valid_targets(
                    battle,
                    slot,
                    request_move=request_move,
                    move=move,
                    active_mon=active_mon,
                )

                for target in targets:
                    if target < 0:
                        continue

                    target_mon = None
                    if target == 1:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[0] if len(opp_active) > 0 else None
                    elif target == 2:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[1] if len(opp_active) > 1 else None
                    elif target == 0:
                        for opp in battle.opponent_active_pokemon:
                            if opp is not None:
                                target_mon = opp
                                break

                    if target_mon is None:
                        continue

                    try:
                        player_role = battle.player_role or "p1"
                        opponent_role = battle.opponent_role or "p2"
                        damage_range = calculate_damage(
                            active_mon.identifier(player_role),
                            target_mon.identifier(opponent_role),
                            move,
                            battle,
                        )
                        if damage_range and damage_range[0] is not None:
                            avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                            candidates.append(
                                (
                                    cast(
                                        BattleOrder,
                                        self.create_order(move, move_target=target),
                                    ),
                                    avg_damage,
                                )
                            )
                    except Exception:
                        continue

            if not candidates and available_moves:
                move = available_moves[0]
                request_move = request_move_by_id.get(move.id)
                targets = get_valid_targets(
                    battle,
                    slot,
                    request_move=request_move,
                    move=move,
                    active_mon=active_mon,
                )
                opp_targets = [t for t in targets if t > 0]
                target = (
                    opp_targets[0]
                    if opp_targets
                    else (0 if 0 in targets else targets[0] if targets else 0)
                )
                candidates.append(
                    (cast(BattleOrder, self.create_order(move, move_target=target)), 0.0)
                )

        available_switches = [
            s for s in battle.available_switches[slot] if s.species not in used_switches
        ]
        for switch_mon in available_switches:
            switch_damage = self._get_best_move_damage(battle, switch_mon)[0]
            switch_score = switch_damage / self.switch_threshold
            candidates.append(
                (cast(BattleOrder, self.create_order(switch_mon)), switch_score)
            )

        return candidates

    def _get_best_available_move(
        self, battle: DoubleBattle, slot: int
    ) -> Tuple[Optional[BattleOrder], float]:
        candidates = self._score_available_actions(battle, slot, set())
        if not candidates:
            return (None, 0.0)
        move_candidates = [
            (order, score)
            for order, score in candidates
            if not isinstance(self._get_order_payload(order), Pokemon)
        ]
        if not move_candidates:
            return (None, 0.0)
        best = max(move_candidates, key=lambda x: x[1])
        return best

    def _get_best_move_damage(
        self, battle, attacker: Pokemon
    ) -> Tuple[float, Optional[str], Optional[int]]:
        if not attacker.moves:
            return (0.0, None, None)

        max_damage = 0.0
        best_move_id: Optional[str] = None
        best_target_idx: Optional[int] = None

        targets = [
            (idx, mon)
            for idx, mon in enumerate(battle.opponent_active_pokemon)
            if mon is not None
        ]
        if not targets:
            return (0.0, None, None)

        for move_id, move in attacker.moves.items():
            if not move or move.current_pp == 0:
                continue

            for idx, target in targets:
                try:
                    damage_range = calculate_damage(
                        attacker.identifier(battle.player_role or "p1"),
                        target.identifier(battle.opponent_role or "p2"),
                        move,
                        battle,
                    )
                    if damage_range and damage_range[0] is not None:
                        avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                        if avg_damage >= max_damage:
                            max_damage = avg_damage
                            best_move_id = move_id
                            best_target_idx = idx
                except Exception:
                    continue

        return (max_damage, best_move_id, best_target_idx)

    @staticmethod
    def _get_order_payload(order: BattleOrder):
        return getattr(order, "order", None)


__all__ = [
    "RNaDAgent",
    "MaxDamagePlayer",
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
