"""rl_trajectory_player.py — async Player that produces RL training trajectories.

RLTrajectoryPlayer is the poke-env Player used by every RL worker process.
On each decision point it embeds the current battle state, submits the
embedding to the trainer-side InferenceService via `InferenceClient.submit()`,
and decodes the returned action into a Showdown command. Training-time
trajectories are accumulated per battle and shipped to the learner once the
battle finishes — that trajectory output (not how the policy is served) is the
class's reason for existing.

This is training-time plumbing: coupled to the trajectory queue, the
InferenceClient IPC layer, and worker-process orchestration. It is NOT a
user-facing agent — see ``elitefurretai/agents/`` for those (eval Players,
heuristic baselines, subprocess managers). This agent is purely for training
and optimizing for speed during training.

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
import logging
import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

import numpy as np
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.concurrency import POKE_LOOP
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder

from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.inference_worker import InferenceClient
from elitefurretai.rl.masking import fast_get_action_mask

logger = logging.getLogger(__name__)


class RLTrajectoryPlayer(Player):
    """Async Player that routes every decision through a centralized InferenceService.

    Subclasses poke-env's `Player` (which handles the websocket protocol with
    Showdown) and adds three things:

      1. **Centralized inference submission**: each decision embeds the battle
         state and `await`s `InferenceClient.submit()`. The trainer-side
         InferenceService gathers submissions across workers/players and runs
         ONE batched forward pass per service tick.

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

    # ── Change 7: stamped by WorkerOpponentFactory at battle setup ──────
    current_team_name: Optional[str] = None

    def __init__(
        self,
        inference_client: InferenceClient,
        worker_id: int,
        embedder: Embedder,
        trajectory_queue=None,
        accept_open_team_sheet=False,
        max_battle_steps: int = 40,
        opponent_type: str = "self_play",
        **kwargs,
    ):
        self.inference_client = inference_client
        self.trajectory_queue = trajectory_queue
        self.worker_id = worker_id
        self.opponent_type = opponent_type
        self.embedder = embedder
        # context tensor accumulated across turns by TransformerThreeHeadedModel
        self.hidden_states: Dict[str, Any] = {}
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
        embed_to_array = getattr(self.embedder, "embed_to_array", None)
        if callable(embed_to_array):
            return cast(np.ndarray, embed_to_array(cast(DoubleBattle, battle)))
        return np.asarray(
            self.embedder.embed_to_vector(cast(DoubleBattle, battle)),
            dtype=np.float32,
        )

    def clear_completed_trajectories(self) -> None:
        self.completed_trajectories.clear()

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

    def _abort_decision(self, battle_tag: str) -> DefaultBattleOrder:
        """Drop the partial trajectory + hidden state for this battle and
        return a default order. Use whenever inference timed out or its
        result became stale before send."""
        self.current_trajectories.pop(battle_tag, None)
        self._reset_battle_hidden_state(battle_tag)
        return DefaultBattleOrder()

    def teardown_runtime(self, timeout_s: float = 1.5) -> None:
        """Best-effort teardown of websocket listener state.

        Why: this is the explicit teardown step needed before rebuilding agents.
        Without it, old clients can remain logged in, causing `|nametaken|` collisions
        and repeated timeout loops after a desync event.
        """
        try:
            fut = asyncio.run_coroutine_threadsafe(self.stop_listening(), POKE_LOOP)
            fut.result(timeout=timeout_s)
        except Exception:
            pass

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        # Defensive guard: do not attempt to send orders for battles that are already
        # marked finished locally.
        if getattr(battle, "finished", False):
            return

        request_generation = self._request_generation.get(battle.battle_tag, 0) + 1
        self._request_generation[battle.battle_tag] = request_generation

        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
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

        # Guardrail: if a force-switch is active but the chosen message is a
        # move command, the request and decode disagree (a known race we still
        # see in production). Falling back to default order is strictly safer
        # than letting Showdown reject the choice.
        if (
            isinstance(battle, DoubleBattle)
            and any(battle.force_switch)
            and isinstance(message, str)
            and message.startswith("/choose move")
        ):
            message = self.choose_default_move().message

        if message:
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception as exc:
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
        # Capture the request's rqid so we can detect drift (Showdown bumps rqid
        # on every |request| emission; same rqid ⇒ same request state). This is
        # cheaper than recomputing the full order twice and lets us drop the
        # result if the live request has drifted before we send it. Sufficient
        # only on the showdown_websocket backend — direct-sim use omits rqid.
        request_rqid = request_snapshot.get("rqid") if request_snapshot else None
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
        except asyncio.TimeoutError:
            logger.debug(
                "INFERENCE_TIMEOUT tag=%s turn=%s teampreview=%s",
                battle.battle_tag,
                getattr(battle, "turn", "?"),
                battle.teampreview,
            )
            return self._abort_decision(battle.battle_tag)

        # Request-generation guard: if a newer request is already active for this
        # battle tag, drop this stale result before decoding/sending.
        if request_generation is not None:
            latest_generation = self._request_generation.get(battle.battle_tag, -1)
            if latest_generation != request_generation:
                logger.debug(
                    "STALE_REQUEST_GENERATION_DROP tag=%s request_gen=%d latest_gen=%d",
                    battle.battle_tag,
                    request_generation,
                    latest_generation,
                )
                return self._abort_decision(battle.battle_tag)

        action_idx = result["action"]

        # If battle state changed while waiting on inference, discard this
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
            return self._abort_decision(battle.battle_tag)

        current_rqid = battle.last_request.get("rqid") if battle.last_request else None
        if current_rqid != request_rqid:
            logger.debug(
                "STALE_REQUEST_SNAPSHOT_DROP tag=%s turn=%d rqid=%s->%s",
                battle.battle_tag,
                current_turn,
                request_rqid,
                current_rqid,
            )
            return self._abort_decision(battle.battle_tag)

        if mask is not None and action_idx < len(mask) and mask[action_idx] == 0:
            logger.warning(
                "MASK_MISMATCH tag=%s turn=%s action_idx=%d mask_value=%s",
                battle.battle_tag,
                getattr(battle, "turn", "?"),
                action_idx,
                mask[action_idx],
            )

        if self.trajectory_queue is not None:
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
            for task in list(getattr(ps_client, "_active_tasks", ())):
                if not task.done() and battle_tag in repr(task):
                    task.cancel()
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
        # popup-recovery fires. Failures are logged at DEBUG so the steady
        # stream during a healthy run doesn't clog WARNING-level output.
        try:
            await ps_client.send_message("/leavebattle", room=battle_tag)
        except Exception as exc:  # noqa: BLE001
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
                    "team_name": self.current_team_name,
                    "battle_format": battle.format,
                }
            )
            self._reset_battle_hidden_state(battle.battle_tag)


__all__ = [
    "RLTrajectoryPlayer",
]
