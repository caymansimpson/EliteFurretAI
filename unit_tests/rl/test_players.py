import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from poke_env.battle import AbstractBattle, DoubleBattle

from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer


class _Recorder:
    """Async callable recording whether it was called — replacement for the
    `.called` attribute trick that pyright complains about on plain functions."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, _msg: str) -> None:
        self.called = True


def test_score_available_actions_filters_moves_not_in_request():
    player = MaxDamagePlayer.__new__(MaxDamagePlayer)
    player.debug = False
    player.switch_threshold = 100.0
    player.create_order = lambda move, move_target=None: SimpleNamespace(
        order=move, move_target=move_target
    )
    player._get_best_move_damage = lambda battle, switch_mon: (0.0, None)

    battle = MagicMock(spec=DoubleBattle)
    battle.last_request = {
        "active": [
            {
                "moves": [
                    {"id": "meteorbeam", "target": "normal", "pp": 8, "disabled": False}
                ]
            },
            {"moves": []},
        ]
    }

    stale_move = SimpleNamespace(id="terastarstorm")
    legal_move = SimpleNamespace(id="meteorbeam")
    battle.available_moves = [[stale_move, legal_move], []]

    active_mon = MagicMock()
    active_mon.species = "Terapagos"
    active_mon.identifier.return_value = "p1: Terapagos"
    battle.active_pokemon = [active_mon, None]

    opponent = MagicMock()
    opponent.species = "Volcarona"
    opponent.identifier.return_value = "p2: Volcarona"
    battle.opponent_active_pokemon = [opponent, None]
    battle.available_switches = [[], []]
    battle.player_role = "p1"
    battle.opponent_role = "p2"

    with patch("elitefurretai.rl.players.calculate_damage", return_value=(10, 10)):
        candidates = player._score_available_actions(battle, 0, set())

    assert candidates
    assert all(candidate[0].order.id == "meteorbeam" for candidate in candidates)


# ── Popup-recovery tests ────────────────────────────────────────────────────
# Cover the "not in that room" race documented in
# planning/stage2/2026-05-06-04-50-zero-completion-room-state-race.md.
#
# We bypass __init__ via __new__ because BatchInferencePlayer.__init__ would
# spin up a real ps_client + asyncio loop, which we don't want in unit tests.


def test_start_inference_loop_is_noop_in_centralized_mode():
    """In centralized mode the trainer-side InferenceService runs the
    loop; the player must not start its own (it has no queue)."""
    player = BatchInferencePlayer.__new__(BatchInferencePlayer)
    player.inference_client = MagicMock()  # truthy, simulates centralized mode
    player._inference_future = None
    # Should not error and should not set _inference_future.
    player.start_inference_loop()
    assert player._inference_future is None


def _make_player_for_popup_tests():
    """Construct a minimal BatchInferencePlayer with just the attributes the
    popup-recovery code path touches."""
    player = BatchInferencePlayer.__new__(BatchInferencePlayer)
    player._diagnostics = {
        "room_lost_recoveries": 0.0,
        "completed_trajectories": 0.0,
        "completed_trajectory_steps": 0.0,
        "message_handler_timeouts": 0.0,
        "battle_lock_tasks_cancelled": 0.0,
    }
    player._room_lost_battles = set()
    player._battles = {}
    player._battle_count_queue = asyncio.Queue()
    player._battle_end_condition = asyncio.Condition()
    player._discarded_battles = set()
    player._request_generation = {}
    player.current_trajectories = {}
    player.hidden_states = {}
    player.trajectory_queue = None  # opponent-only mode (no trajectory ship)
    player.opponent_type = "self_play"
    # Centralized-inference attribute: tests bypass __init__ via __new__,
    # so we set it explicitly. None = legacy mode (no inference client).
    player.inference_client = None
    player._original_handle_message = MagicMock()
    # Minimal ps_client stand-in. `_battle_locks` and `_active_tasks` are read
    # by _recover_room_lost_battle to free queued lock waiters. `logger`
    # is used by the wrapper's timeout-warning path.
    player.ps_client = SimpleNamespace(  # type: ignore[misc]
        _battle_locks={}, _active_tasks=set(), logger=MagicMock()
    )
    return player


class _FakeBattle:
    """Minimal stand-in for AbstractBattle for popup-recovery tests."""

    def __init__(self, tag: str, finished: bool = False):
        self.battle_tag = tag
        self._finished = finished
        self._won = None

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def won(self):
        return self._won

    def _finish_battle(self) -> None:
        self._finished = True


def _make_battle(tag: str, finished: bool = False) -> _FakeBattle:
    return _FakeBattle(tag, finished)


def test_popup_regex_matches_not_in_that_room():
    msg = (
        '|popup|You tried to send "/choose move surgingstrikes 1, switch ironhands"'
        ' to the room "battle-gen9vgc2024regg-647827"'
        " but it failed because you were not in that room."
    )
    m = BatchInferencePlayer._ROOM_LOST_POPUP_RE.search(msg)
    assert m is not None
    assert m.group(1) == "battle-gen9vgc2024regg-647827"


def test_popup_regex_ignores_other_popups():
    other = "|popup|The user 'VGCBENCH' was not found."
    assert BatchInferencePlayer._ROOM_LOST_POPUP_RE.search(other) is None


@pytest.mark.asyncio
async def test_recover_room_lost_battle_marks_finished_and_records_forfeit():
    """The recovery path must (a) flip battle.finished, (b) set _won=False,
    (c) decrement _battle_count_queue so send_challenges.join() releases,
    (d) add the tag to _room_lost_battles for trajectory shipping."""
    player = _make_player_for_popup_tests()
    tag = "battle-gen9vgc2024regg-1"
    battle = _make_battle(tag)
    player._battles[tag] = cast(AbstractBattle, battle)

    # Simulate the queue state when a battle is in-flight: _battle_count_queue
    # has one pending item put by send_challenges.
    await player._battle_count_queue.put(None)
    assert player._battle_count_queue.qsize() == 1

    await player._recover_room_lost_battle(tag)

    assert battle.finished is True
    assert battle.won is False
    assert tag in player._room_lost_battles
    assert player._diagnostics["room_lost_recoveries"] == 1.0
    # Queue emptied via get_nowait + task_done.
    assert player._battle_count_queue.qsize() == 0


@pytest.mark.asyncio
async def test_recover_room_lost_battle_is_idempotent_for_already_finished():
    """Already-finished battles must be skipped (the |win| path beat us)."""
    player = _make_player_for_popup_tests()
    tag = "battle-gen9vgc2024regg-2"
    battle = _make_battle(tag, finished=True)
    player._battles[tag] = cast(AbstractBattle, battle)
    # Queue is empty: _handle_battle_message |win| branch already drained it.
    await player._recover_room_lost_battle(tag)

    # No counter bump, no _room_lost_battles add.
    assert tag not in player._room_lost_battles
    assert player._diagnostics["room_lost_recoveries"] == 0.0


@pytest.mark.asyncio
async def test_recover_room_lost_battle_is_idempotent_for_unknown_tag():
    """Tags we don't track must be silently ignored (no KeyError)."""
    player = _make_player_for_popup_tests()
    await player._recover_room_lost_battle("battle-gen9vgc2024regg-999")
    assert player._diagnostics["room_lost_recoveries"] == 0.0


@pytest.mark.asyncio
async def test_recover_room_lost_battle_survives_empty_queue():
    """If |win| beat us to the count_queue decrement, get_nowait raises
    QueueEmpty and we must not crash — the rest of finalization still runs."""
    player = _make_player_for_popup_tests()
    tag = "battle-gen9vgc2024regg-3"
    battle = _make_battle(tag)
    player._battles[tag] = cast(AbstractBattle, battle)
    # Queue is empty (|win| already decremented).

    await player._recover_room_lost_battle(tag)

    assert battle.finished is True
    assert tag in player._room_lost_battles
    assert player._diagnostics["room_lost_recoveries"] == 1.0


@pytest.mark.asyncio
async def test_handle_message_with_popup_recovery_calls_recovery_on_match():
    """The wrapped _handle_message must call the original first AND fire
    recovery when the popup matches."""
    player = _make_player_for_popup_tests()
    tag = "battle-gen9vgc2024regg-4"
    battle = _make_battle(tag)
    player._battles[tag] = cast(AbstractBattle, battle)
    await player._battle_count_queue.put(None)

    recorder = _Recorder()
    player._original_handle_message = cast(Any, recorder)

    msg = (
        '|popup|You tried to send "/choose move flamethrower 1"'
        f' to the room "{tag}"'
        " but it failed because you were not in that room."
    )
    await player._handle_message_with_popup_recovery(msg)

    assert recorder.called, "original ps_client._handle_message must run"
    assert battle.finished is True
    assert tag in player._room_lost_battles


@pytest.mark.asyncio
async def test_handle_message_with_popup_recovery_no_op_for_unrelated_messages():
    """Unrelated messages (battle updates, other popups) must not trigger
    recovery."""
    player = _make_player_for_popup_tests()

    recorder = _Recorder()
    player._original_handle_message = cast(Any, recorder)

    for msg in [
        ">battle-gen9vgc2024regg-5\n|turn|3",
        "|popup|The user 'VGCBENCH' was not found.",
        "|updateuser| TestUser",
    ]:
        await player._handle_message_with_popup_recovery(msg)

    assert recorder.called
    assert player._diagnostics["room_lost_recoveries"] == 0.0


@pytest.mark.asyncio
async def test_handle_message_with_popup_recovery_times_out_hung_handler(
    monkeypatch,
):
    """A hung _original_handle_message must trip the per-message timeout
    rather than pin the wrapper task forever (the leak that crashed WSL
    after ~12 hours)."""
    player = _make_player_for_popup_tests()
    # Shrink the timeout from the production 60s to keep the test fast.
    monkeypatch.setattr(BatchInferencePlayer, "_MESSAGE_HANDLER_TIMEOUT_S", 0.05)

    async def never_returns(_msg: str) -> None:
        await asyncio.Event().wait()  # never set

    player._original_handle_message = cast(Any, never_returns)

    await player._handle_message_with_popup_recovery(">battle-stuck|turn|1")

    assert player._diagnostics["message_handler_timeouts"] == 1.0


@pytest.mark.asyncio
async def test_recover_room_lost_battle_frees_lock_and_cancels_waiters():
    """Popup recovery must drop the per-battle lock from ps_client and
    cancel queued lock-waiter tasks. Without this, ~30 tasks per stuck
    battle accumulate as lock waiters and pin memory."""
    player = _make_player_for_popup_tests()
    tag = "battle-gen9vgc2024regg-7"
    battle = _make_battle(tag)
    player._battles[tag] = cast(AbstractBattle, battle)

    # Seed ps_client with a lock for this battle plus three pending
    # "waiter" tasks. Two of them name the battle tag in their repr (so
    # they should be cancelled); the third belongs to a different battle.
    lock = asyncio.Lock()
    await lock.acquire()
    player.ps_client._battle_locks[tag] = lock

    async def waiter() -> None:
        await asyncio.Event().wait()

    task_a = asyncio.create_task(waiter(), name=f"waiter-{tag}-a")
    task_b = asyncio.create_task(waiter(), name=f"waiter-{tag}-b")
    task_other = asyncio.create_task(waiter(), name="waiter-battle-gen9vgc2024regg-other")
    player.ps_client._active_tasks = {task_a, task_b, task_other}

    try:
        await player._recover_room_lost_battle(tag)
        # Yield once so cancelled tasks transition to done.
        await asyncio.sleep(0)

        # Lock entry dropped, two matching tasks cancelled, the unrelated
        # one untouched.
        assert tag not in player.ps_client._battle_locks
        assert task_a.cancelled()
        assert task_b.cancelled()
        assert not task_other.done()
        assert player._diagnostics["battle_lock_tasks_cancelled"] == 2.0
    finally:
        if not task_other.done():
            task_other.cancel()
        # Drain cancellations so pytest-asyncio doesn't complain.
        await asyncio.gather(task_a, task_b, task_other, return_exceptions=True)
