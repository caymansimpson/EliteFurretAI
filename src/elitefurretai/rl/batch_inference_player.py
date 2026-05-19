"""batch_inference_player.py — high-throughput async Player for RL training.

BatchInferencePlayer gathers per-turn decisions from many concurrent battles
and *batches* them into one model forward pass for efficiency, then pushes
finished trajectories to a queue for the learner to consume.

This is training-time plumbing: coupled to the trajectory queue, the
InferenceClient IPC layer, and worker-process orchestration. It is NOT a
user-facing agent — see ``elitefurretai/agents/`` for those (eval Players,
heuristic baselines, subprocess managers).

Moved here from the former ``rl/players.py`` on 2026-05-19 as part of the
agents/ directory reorganization (see planning/stage2/2026-05-19-09-30-agents-directory-reorg.md).

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
-1 loss; 0 on all other steps) and the full trajectory is shipped to the
trajectory_queue for the learner to train on.
"""

import asyncio
import concurrent.futures
import logging
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
    cast,
)

if TYPE_CHECKING:
    from elitefurretai.rl.inference_worker import InferenceClient

import numpy as np
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.concurrency import POKE_LOOP
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.masking import fast_get_action_mask

logger = logging.getLogger(__name__)


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
        inference_client: "InferenceClient",
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
        **kwargs,
    ):
        battle_format = kwargs.get("battle_format", "gen9vgc2023regc")

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
        # context tensor accumulated across turns by TransformerThreeHeadedModel
        self.hidden_states: Dict[str, Any] = {}
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
            "server_leavebattle_sent": 0.0,
            "server_leavebattle_send_failed": 0.0,
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
        # The trainer-side InferenceService owns its own loop.
        # Players have no per-instance loop to start. No-op kept for
        # backward compatibility with callers in opponents.py, evaluate.py,
        # showdown_benchmark.py, and vgc_environment.py.
        pass

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
            # Centralized path: trainer-side InferenceService runs the
            # forward + sampling AND owns the hidden state, keyed by
            # (worker_id, battle_tag).
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
            logger.debug(
                "INFERENCE_TIMEOUT tag=%s turn=%s teampreview=%s",
                battle.battle_tag,
                getattr(battle, "turn", "?"),
                battle.teampreview,
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

        # Server-side cleanup hint. The "not in that room" popup means
        # showdown booted us from the battle's user list, but the battle
        # itself stays alive on the server until the other side ends or
        # showdown's ~15-min timeout fires. With ~6 popup-recoveries/min
        # (measured in dulcet-sun-59), 1.5 abandoned battles per real
        # battle accumulate × ~5 MB of battle history each ≈ 1.3 GB/hr
        # of showdown server growth (matches measured rate). Sending
        # `/leavebattle` targeted at the battle room tells the server to
        # treat us as having forfeited from that room; the other side's
        # next turn (or its own timeout) then ends the battle, freeing
        # the server-side state immediately instead of waiting 15 min.
        #
        # Best-effort, deliberately last in the recovery sequence: the
        # client-side cleanup above is the load-bearing work; this is a
        # hint to the server. Wrapping in try/except is justified here
        # (not a "hide errors" anti-pattern) because send may legitimately
        # fail mid-shutdown — the websocket can be closing as the
        # popup-recovery fires. Any failure is counted as a diagnostic
        # so we can spot regressions, and logged at DEBUG so the steady
        # stream during a healthy run doesn't clog WARNING-level output.
        try:
            await ps_client.send_message("/leavebattle", room=battle_tag)
            self._diagnostics["server_leavebattle_sent"] += 1
        except Exception as exc:  # noqa: BLE001
            self._diagnostics["server_leavebattle_send_failed"] += 1
            self.logger.debug("leavebattle send failed for %s: %s", battle_tag, exc)

    def _battle_finished_callback(self, battle: AbstractBattle):
        # ── End-of-battle: assign rewards and ship the trajectory ────────────
        # poke-env calls this hook once per battle when it finishes (win, loss,
        # draw, or forfeit). This is where we:
        #   1. Walk the per-step trajectory we've been collecting.
        #   2. Fill in rewards (we deferred this until the outcome is known).
        #   3. Push the completed trajectory to the trajectory_queue, where the
        #      worker will eventually forward it to the learner via mp.Queue.
        #
        # Reward: terminal step gets +1 on win, -1 otherwise (loss/tie/forfeit).
        # All other steps get 0. battle.won is False for both losses and ties.
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
                if t == len(traj) - 1:
                    step["reward"] = 1.0 if battle.won else -1.0
                else:
                    step["reward"] = 0.0

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


__all__ = [
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
