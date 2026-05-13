# Popup-Recovery Lock-Waiter Leak Fix + Slow-Handler Diagnostic

## Context

The 2026-05-07 09:22 run reached update 800 over ~12 hours, then the entire
WSL VM crashed (Hyper-V supervisor-killed) due to host memory exhaustion.
No SIGTERM was delivered, so no `main_model_step_800.pt` was saved — only
the ghost-pool weights (no optimizer state).

## Before State

Yesterday's Layer-2 popup recovery
([planning/stage2/2026-05-06-04-50-zero-completion-room-state-race.md](2026-05-06-04-50-zero-completion-room-state-race.md))
finalised the affected battle's local state on the "not in that room" popup
but did **not** touch poke-env's `ps_client._battle_locks[battle_tag]`. As a
result:

- ~32 popups/hour produced 988 asyncio `Task was destroyed but it is pending!`
  errors per hour, all pinned at
  [players.py:1145](../../src/elitefurretai/rl/players.py#L1145) inside the
  wrapper.
- Worker RSS grew from 1 GB at batch 50 to 3 GB at batch 5500 (≈1 GB net per
  worker, six workers).
- Total host memory: 6 × 3 GB workers + 6 × 0.9 GB VGCBench + 6 × 0.2 GB
  Showdown + trainer ≈ 25.6 GB vs WSL2's 23 GiB → swap thrash → VM kill.

## Problem

In [poke_env/ps_client/ps_client.py:168-176](../../../poke-env/src/poke_env/ps_client/ps_client.py#L168-L176),
each battle message is dispatched under a per-battle `asyncio.Lock`:

```python
async with self._battle_locks[battle_tag]:
    await self._handle_battle_message(split_messages)
```

When `_handle_battle_message` hangs for any reason (some message type
awaits an internal Future that never resolves), the lock is held forever.
Subsequent messages for the same battle stack up as waiters on the
`acquire()` call. Each waiter is a wrapper task pinned by the PS client's
`_active_tasks` set, holding references to message text, battle state, and
embedder caches.

The popup recovery cleaned up the battle but left the lock — and its
waiters — intact. Result: a slow but unbounded RSS climb.

## Solution

### A. Per-message timeout + waiter cancellation in popup recovery

Two coordinated edits in [src/elitefurretai/rl/players.py](../../src/elitefurretai/rl/players.py):

1. **Per-message timeout** ([players.py:1158-1170](../../src/elitefurretai/rl/players.py#L1158-L1170)).
   Wrap `_original_handle_message(message)` in
   `asyncio.wait_for(..., timeout=60.0)`. On timeout, the inner coroutine is
   cancelled (releasing the held lock via the `async with` finalizer and
   freeing all references the wrapper task held), and the
   `message_handler_timeouts` diagnostic is bumped. No legitimate handler
   runs more than a few seconds, so 60 s is a conservative safety net.

2. **Lock-entry release + waiter cancellation in `_recover_room_lost_battle`**
   ([players.py:1206-1223](../../src/elitefurretai/rl/players.py#L1206-L1223)).
   When the recovery fires:
   - Iterate `ps_client._active_tasks`, cancel any task whose `repr()`
     mentions the battle tag (releases the holder via `async with` finalizer
     *and* causes all waiters' `acquire()` to raise `CancelledError`).
   - Pop `ps_client._battle_locks[battle_tag]` so future messages for that
     tag get a fresh, unowned lock.
   - New diagnostic counter `battle_lock_tasks_cancelled` tracks how many
     waiters got freed per recovery.

### B. Slow-handler instrumentation in poke-env

[poke_env/ps_client/ps_client.py:168-197](../../../poke-env/src/poke_env/ps_client/ps_client.py#L168-L197).
Around the `async with self._battle_locks[...]` block, measure
`lock_wait_s` (acquisition time) and `handler_s` (handler runtime)
separately. If either exceeds 5 s, log a warning with the battle tag, both
durations, the first 5 message types, and a 200-char preview. This is
purely diagnostic — it doesn't *fix* the upstream hang but identifies which
poke-env handler is hanging so we can fix it permanently rather than rely
on the timeout workaround.

### Tests

Two new tests in [unit_tests/rl/test_players.py](../../unit_tests/rl/test_players.py):

- `test_handle_message_with_popup_recovery_times_out_hung_handler` —
  verifies the per-message timeout fires when the inner handler hangs and
  bumps the `message_handler_timeouts` diagnostic.
- `test_recover_room_lost_battle_frees_lock_and_cancels_waiters` —
  verifies the lock entry is popped, matching tasks in `_active_tasks` are
  cancelled, unrelated tasks are untouched, and the
  `battle_lock_tasks_cancelled` diagnostic increments.

All 197 RL tests pass. Ruff + ruff-format clean. Pyright produces no new
errors (the 4 pre-existing errors in unrelated MaxDamagePlayer tests
remain).

## Reasoning

- **Why a timeout *and* lock cleanup, not just one?** The timeout
  (Solution A.1) catches *any* future hang regardless of cause — defense
  in depth. The lock cleanup (Solution A.2) specifically frees the waiters
  that have already accumulated by the time the popup fires, which is the
  observed dominant contributor (~30 waiters per popup × ~32 popups/hr).
  Together they collapse the leak rate to roughly zero.
- **Why also instrument poke-env (B)?** The timeout in A.1 is a workaround,
  not a root-cause fix. We don't know *which* poke-env handler is hanging.
  Instrumentation in the next run will produce slow-handler warnings that
  identify the offending message type so a proper upstream fix can be made.
- **Why repr-matching to identify waiter tasks?** asyncio doesn't expose
  "tasks waiting on this Lock". `task.__repr__()` includes the coroutine's
  source location and frame locals (which include `battle_tag` for any
  task currently inside `_handle_message`). Brittle? Yes — but the only
  alternative is monkey-patching poke-env's task tracking, which is worse.
- **Why pop the lock entry?** A lock with a cancelled holder is still
  technically usable, but cancelled waiters may have left the lock in an
  inconsistent state. Replacing it (by dropping the entry; the next
  message creates a fresh one) is safer than reusing.

## Hard-constraint compliance

- WSL2 `pin_memory=False`: untouched.
- Both backends preserved: changes are in `BatchInferencePlayer` (Showdown
  side) and poke-env (Showdown-only dependency). Rust backend unaffected.
- No try/except hiding errors: the only `try/except` added catches
  `asyncio.TimeoutError`, which is a documented control-flow signal from
  `wait_for`, not error suppression.
- `src/elitefurretai/scripts/` lint exclusion: untouched.

## Planned Next Steps

1. Launch the next training run, resuming from
   `main_model_step_144.pt` (the most recent main+optimizer checkpoint —
   `step_800` survived only as a ghost-pool weights-only file).
2. After 30 min of training, sample the log for:
   - `message_handler_timeouts` count (should be ≤ a handful).
   - `battle_lock_tasks_cancelled` count (should track popup rate × ~30).
   - `Slow battle message: ...` warnings from poke-env (these identify the
     hanging handler type).
3. After 12 h:
   - Worker RSS should stay flat at ~1-1.5 GB (vs. yesterday's 3 GB climb).
   - No WSL crash.
4. Once the slow-handler warnings identify the upstream hang, file a
   targeted poke-env fix (replacing the 60s timeout with a real bug fix).
5. Consider adding a memory watchdog (Option C from the conversation)
   that triggers SIGTERM if combined RSS exceeds 18 GB, as a safety net
   independent of the leak fix.

## Correction (added 2026-05-12)

The claims in this doc that ghost-pool files are "weights only / no
optimizer state" (Context paragraph, Planned Next Step 1) are
**wrong**. Verified on 2026-05-12 by `torch.load`-ing
`data/models/rl/bright-pine-189/ghosts/main_model_step_875.pt`:

```
keys: ['model_state_dict', 'optimizer_state_dict', 'step',
       'curriculum', 'config', 'timestamp']
step: 875                       optimizer state count: 196
model_state_dict tensors: 199
```

`save_checkpoint`
([learners.py:820-843](../../src/elitefurretai/rl/learners.py#L820-L843))
is the same function for both the periodic save at
[train.py:1436](../../src/elitefurretai/rl/train.py#L1436) (writing to
`<run_dir>/ghosts/`) and the shutdown save at
[train.py:1613](../../src/elitefurretai/rl/train.py#L1613) (writing to
`<run_dir>/`). The two paths produce structurally identical files; the
`ghosts/` prefix only marks opponent-pool membership. So step_800.pt
from the May 7 run was a resumable checkpoint after all, and the May 11
relaunch could have continued from step 800 instead of step 144 (~656
updates of progress that did not need to be re-traced). See
[2026-05-12-09-30-second-wsl-crash-and-watchdog.md](2026-05-12-09-30-second-wsl-crash-and-watchdog.md)
for the full follow-up.
