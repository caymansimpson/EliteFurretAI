# Zero-Completion Room-State Race — Diagnosis and Fix Plan

## Context

The 2026-05-06 overnight `easy_test.yaml` K3-topology run crashed at update 137
(4h 19m wall-clock) with:

```
RuntimeError: Worker entered sustained zero-completion state (5 consecutive batches).
```

emitted from worker 0. Last forward progress on worker 0 was at batch 700; the
healthy workers (2 and 5) were at batch 2450/2550 by the same time, so worker 0
was already running ~3.5× slow for hours and the trip was just the failure
mode. Run wrote `main_model_step_137.pt` cleanly before exit; the rest of the
data pipeline was unharmed.

Throughout the run, 70 instances of:

```
|popup|You tried to send "/choose ..." to the room "battle-..." but it failed
because you were not in that room.
```

were logged across all 6 workers. They cluster on individual workers — when
multiple players in one worker hit the popup near-simultaneously, the worker's
batch returns zero completed battles. Five consecutive zero-completion batches
trip the safety guard in
[src/elitefurretai/rl/worker.py:309](../../src/elitefurretai/rl/worker.py#L309).

This is a different bug class from the residual `Invalid choice` errors
([planning/stage2/2026-04-26-22-00-two-residual-bugs.md](2026-04-26-22-00-two-residual-bugs.md))
and from the recently-fixed VGCBench username plumbing — it is a state-tracking
*race* between battle termination and the player's next `/choose` send, not a
mis-filtered legal action set.

Interim mitigation (landed `worker.py` change, this commit):
`zero_completion_failover_threshold` bumped 5 → 15. Buys ~3× headroom against
the bursty failure mode but does not fix the underlying race.

## Before State

### Code paths

- [src/elitefurretai/rl/players.py](../../src/elitefurretai/rl/players.py) —
  `BatchInferencePlayer._run_batch` is the per-turn driver that emits `/choose`
  via the underlying poke-env `Player.send_message`. It currently does not gate
  on whether the battle is still alive at the moment of send.
- [poke-env Player base class] — handles `|popup|` messages via
  `_handle_battle_message`; emits warnings but does not flip the battle to a
  finished state on the specific "not in that room" popup. The battle's
  `battle.finished` flag relies on a separate `|win|`/`|tie|` message arrival.
- [src/elitefurretai/engine/sync_battle_driver.py](../../src/elitefurretai/engine/sync_battle_driver.py) —
  sequences turn-by-turn execution; treats unfinished battles as "still owe a
  decision" until `battle.finished` flips.
- [src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py](../../src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py) —
  fuzz harness, currently halts on `Invalid choice` popups and empty masks
  only. Does not catch the "not in that room" family.

### Failure shape (from the 04:38 crash log)

| Time | Worker 0 batch | Healthy peer batch | Δ |
|------|---|---|---|
| 00:25 | 50 | (workers 2/5 ~50) | 0 |
| 01:13 | 250 | (workers 2/5 ~600) | -350 |
| 02:30 | 500 | (workers 2/5 ~1000) | -500 |
| 03:35 | 600 | (workers 2/5 ~1900) | -1300 |
| 04:28 | **700** | (workers 2/5 ~2450) | -1750 |
| 04:38 | 700 (5 zero batches) | 2550 | crash |

Worker 0 was always slower; it accumulated stuck players and never recovered.
Slowdown was monotonic, not sudden. Healthy workers continued at ~2.6 b/s
locally throughout.

## Problem

The race specifically: at the end of a battle, `Showdown → poke-env` sends a
terminal message (`|win|<player>` or `|tie`), but at the same time the
`BatchInferencePlayer` may have already prepared and submitted a `/choose` for
the next turn (or for an in-flight forced-switch / teampreview). Showdown
processes the terminal first, removes the player from the room, then rejects
the now-orphaned `/choose` with the popup. From the worker's perspective, the
local battle is still "active" — it never received a clean `battle.finished`
flip, so the batch driver waits for a decision that will never come, and the
batch closes with one fewer completed battle than expected.

When this happens to N>1 players in the same worker concurrently, batches
return zero completions and the failover guard trips after 5 such batches.

## Solution

Three layers, in order of cost vs payoff. **Recommend implementing 1 and 2
together; 3 only if 1+2 don't fully eliminate the popup volume.**

### Layer 1 — Pre-send finished check (defensive, cheapest)

In `BatchInferencePlayer._run_batch` (or whatever code path emits the
`/choose` message), check `battle.finished` immediately before
`send_message(battle_tag, ...)`. If finished, skip the send and ensure the
batch driver records a "done" state for that battle.

**Files touched:** `src/elitefurretai/rl/players.py` only.

**Estimated effort:** ~30 minutes including a unit test using
`MagicMock(spec=DoubleBattle)` with `battle.finished = True`, asserting no
send is attempted.

**Limitation:** does not catch the case where the terminal `|win|` message
hasn't yet been *processed* on poke-env's side at the moment of the pre-check
— the battle may not be marked finished yet locally even though the room is
already gone server-side. So this layer alone won't drop the popup count to
zero, only reduce it. That's fine; combined with layer 2 it closes the gap.

### Layer 2 — Popup-handler completion (root-cause for the orphan-choose case)

Extend the player's `_handle_battle_message` (or the popup-specific override)
to recognize the `"not in that room"` popup as a terminal signal for that
specific battle and call `battle._won_by(...)` or
`battle._tied()` (whichever poke-env supports for "I don't know who won, just
mark it ended") to flip `battle.finished = True`. Then notify the batch driver
so the in-progress decision is cancelled and the batch's pending count is
decremented.

**Files touched:**
- `src/elitefurretai/rl/players.py` — handler for `not in that room`. The
  battle outcome is unknown so we mark it as a *forfeit by us* (consistent with
  current handling of timeouts) so curriculum bookkeeping
  (`record_battle_result(forfeited=True)`) drops it from win-rate tracking
  cleanly per the [adaptive curriculum overhaul plan](2026-05-03-23-37-adaptive-curriculum-overhaul.md)
  Change 1.
- Investigate whether poke-env's base `Player` already has a hook we should
  override; if so, use that. If not, a sub-class hook in our player.

**Estimated effort:** ~1 hour. Most of this is reading poke-env to confirm the
right state-flip API. Test alongside layer 1 above.

### Layer 3 — Fuzz harness extension (regression prevention)

Extend [showdown_invalid_choice_diagnostics.py](../../src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py)
to also halt-and-report on the `not in that room` popup family. Same artifact
schema as the existing Invalid-choice diagnostics. This isn't a fix per se but
gives us a regression-test surface for layer 1 and 2.

**Files touched:**
- `src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py` —
  extend `_handle_battle_error` (or equivalent) to recognize this popup family
  and write the same kind of failure report. The `MaskedRandomPlayer` is a
  natural fuzz target since masked-random play has high room-state churn from
  fast-ending battles.

**Estimated effort:** ~30 minutes. Mostly mechanical — copying the existing
Invalid-choice halt path and adapting the regex.

### Sequencing

1. Layer 1 first, in isolation (gated by a unit test) — measure the popup
   count drop on a fresh 1h smoke training run before continuing.
2. Layer 2 next, in isolation — measure popup count drop again. If layer 1 +
   layer 2 together drive popup count below ~0.05% of battles (effective zero
   for our throughput), stop here.
3. Layer 3 last, as guardrail. Run the fuzz harness for 30 min in masked-random
   mode and confirm it can flush new popup variants if any.
4. Once all three layers are in, revert the
   `zero_completion_failover_threshold` from 15 back to 5 (or even tighter,
   like 3) — at that point any zero-completion cluster represents a real
   blocker that should fail loud.

## Reasoning

**Why three layers instead of one root-cause fix?**
Because the room-state mismatch can happen at multiple lifecycle moments:
between turn N's request and the `/choose` response, between teampreview
selection and the first turn, between a forced-switch decision and the next
request. A single point-fix at "the" race is fragile. Layer 1 is a defensive
no-op when things are working and a save when they aren't; layer 2 closes the
loop after the failure has happened. Together they cover both
"prevent-the-send" and "recover-after-the-popup," which is the cheapest way to
get to zero popups without rewriting poke-env.

**Why mark as forfeit-by-us in layer 2?**
The room-state mismatch could happen because (a) we lost cleanly and the
`|win|` was just slow, (b) we won and the same, (c) the opponent disconnected,
or (d) the server forced the room closed. We can't reliably distinguish (a)
from (b). Marking as forfeit by us is conservative for win-rate signal and
combined with the recently-landed forfeit-drop logic
(Change 1 in the adaptive curriculum overhaul plan) means these don't pollute
opponent win-rate tracking.

**Why a fuzz-harness extension instead of just unit tests?**
Synthetic unit tests will catch the immediate regression for the specific
trigger we know about. The fuzz harness will surface the *next* lifecycle
moment where the same race exists (forced-switch, Tera, Commander cleanup
events), which is the actual long-tail risk. The harness already has the
infrastructure — it's a 30-minute extension.

**Why bump the threshold to 15 in the meantime instead of disabling the
guard?**
The guard is real protection against degenerate states (network partition,
Showdown server crash). Disabling it would mask actual bugs. 15 is enough
headroom for the known clustering pattern from this run (worst observed:
~10 stuck battles in ~5 minutes on one worker), with ~50% margin.

## Hard constraints to respect

- WSL2: `pin_memory=False` always; no change here.
- Both backends preserved — Layer 1 and 2 are in
  `BatchInferencePlayer` which is Showdown-specific. The Rust backend has its
  own state-tracking and is not affected by this race, but it should not
  regress; verify by running a short Rust smoke after layer 2 lands.
- No try/except hiding errors — layer 2 explicitly converts the popup into a
  *known-good* terminal state, not a swallow.
- `src/elitefurretai/scripts/` lint exclusion unchanged.

## Planned Next Steps

When this plan is picked up (likely a future Claude session sitting down
cold):

1. **Read first**:
   - This doc (you're here).
   - [src/elitefurretai/rl/players.py](../../src/elitefurretai/rl/players.py) —
     `BatchInferencePlayer`, especially `_run_batch` and the `_handle_*` paths
     it inherits from `poke_env.Player`.
   - The most recent run log for an example of the popup:
     `data/benchmarks/2026-05-05-easy-test/run.log` (search "not in that room").
   - poke-env's `Player` source for terminal-state APIs (`_won_by`, `_tied`,
     `_handle_battle_message`).

2. **Layer 1**:
   - Add the pre-send `battle.finished` check to `BatchInferencePlayer`.
   - Unit test in `unit_tests/rl/test_players.py` (create or extend) using a
     `MagicMock(spec=DoubleBattle)` with `finished=True`; assert no send.
   - Quality gates.
   - Run a 30-min smoke at the K3 topology; record popup count from the log.
     Compare to the 70/4h baseline (≈17.5/hr).

3. **Layer 2**:
   - Implement the popup handler. Decide whether it's a sub-class override or
     a base-class extension (poke-env upstream contribution if minimal).
   - Mark battle as forfeit-by-us; ensure `record_battle_result(forfeited=True)`
     is called downstream so curriculum stays clean.
   - Tests + quality gates + 30-min smoke. Target popup count < 1/hr.

4. **Layer 3**:
   - Extend
     [showdown_invalid_choice_diagnostics.py](../../src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py)
     to halt-and-dump on the new popup family. The artifact schema and report
     structure is unchanged; just a new regex match in the error handler.
   - Run a 30-min masked-random fuzz with the harness; confirm 0 captures.

5. **Revert the threshold bump**:
   - Once layers 1+2+3 are landed and a 4-hour smoke training run survives
     without zero-completion stalls, change `zero_completion_failover_threshold`
     back to 5 (or to 3 for tighter coverage of any new latent issues).

6. **Update RL.md** §X (the "Worker failure modes" section, if it exists, or
   add one) with the room-state-race diagnosis and how the layered fix works.

## Updates

### 2026-05-06 — Layers 1 and 2 implemented; Layer 3 deferred

Implemented in this session at Cayman's direction. Layer 3 (fuzz harness
extension) intentionally skipped — the issue has not surfaced via the existing
Invalid-choice fuzz harness, and the popup family is sufficiently distinct
that the existing diagnostics path doesn't catch it cheaply. Will revisit if a
future incident shows the same race in a *new* lifecycle moment.

**Layer 1: pre-send `battle.finished` check** — *already existed*. Confirmed
in [src/elitefurretai/rl/players.py:818](../../src/elitefurretai/rl/players.py#L818)
and [:850](../../src/elitefurretai/rl/players.py#L850). Both gate the
`/choose` send on `getattr(battle, "finished", False)`. The popup still fires
because at the moment of send, `|win|` has been received but not yet
*processed* locally — the local `battle.finished` is still False under the
race. Pre-send check is a real save in some cases but not the dominant fix.

**Layer 2 — three coordinated changes:**

1. **poke-env guard against double-decrement of `_battle_count_queue`** —
   modified `Player._handle_battle_message` in
   [poke-env/src/poke_env/player/player.py:306-325](../../../poke-env/src/poke_env/player/player.py#L306-L325).
   The `|win|`/`|tie|` branch is now wrapped in `if not battle.finished:`. Without
   this, when our popup-recovery decrements the queue first, a later `|win|`
   would block forever in `await self._battle_count_queue.get()`, leaving an
   orphaned coroutine holding the per-battle lock.

2. **Popup detection hook on `BatchInferencePlayer.ps_client._handle_message`** —
   added in [src/elitefurretai/rl/players.py:333-339](../../src/elitefurretai/rl/players.py#L333-L339).
   The hook wraps the original `ps_client._handle_message` (preserves all
   existing behavior) and additionally calls a recovery routine when the
   "not in that room" popup matches a tracked battle.

3. **`_recover_room_lost_battle(battle_tag)`** — added at
   [src/elitefurretai/rl/players.py:1144-1183](../../src/elitefurretai/rl/players.py#L1144-L1183).
   The recovery:
   - bumps a new `room_lost_recoveries` diagnostic counter
   - adds the tag to a new `_room_lost_battles: Set[str]` instance attribute
   - sets `battle._won = False` and calls `battle._finish_battle()`
   - calls `self._battle_count_queue.get_nowait()` + `task_done()` (caught with
     `asyncio.QueueEmpty` if the `|win|` path beat us)
   - calls `self._battle_finished_callback(battle)` to ship the trajectory
   - notifies `_battle_end_condition` for waiters

   `_battle_finished_callback` was extended at the trajectory-ship site
   ([players.py:1226-1232](../../src/elitefurretai/rl/players.py#L1226-L1232))
   to read `_room_lost_battles` and ship `forfeited=True` for these — so the
   adaptive curriculum drops them from opponent win-rate tracking per Change 1
   of [the curriculum overhaul plan](2026-05-03-23-37-adaptive-curriculum-overhaul.md).

**Tests:** 7 new tests in
[unit_tests/rl/test_players.py](../../unit_tests/rl/test_players.py) cover the
regex, the recovery routine's idempotence (already-finished, unknown-tag,
empty-queue paths), and the wrapped `_handle_message` hook (positive and
negative). All 188 RL tests pass; pyright/ruff clean on touched files.

**Layer 3 (fuzz harness) deferred** — per Cayman, the issue hasn't surfaced
in fuzz-harness runs. The `MaskedRandomPlayer` ends battles too quickly
(mostly forfeit-by-no-progress) to reliably reproduce the close-mid-inference
race that produces this popup. Skip.

**Threshold revert (next step):** the
`zero_completion_failover_threshold` was bumped 5 → 15 as a safety net
([worker.py:309-314](../../src/elitefurretai/rl/worker.py#L309-L314)). Once
the next overnight run survives 200+ updates without zero-completion stalls,
revert to 5 (or tighten to 3) so any new latent stall fails loud as a real bug.
Track this revert as a separate todo on the next session sit-down.

**Hard-constraint compliance:**

- WSL2: no `pin_memory` change.
- Both backends: Layer 2 changes touch `BatchInferencePlayer` and poke-env,
  both of which are Showdown-side only. The Rust backend uses a separate
  player and is unaffected.
- No try/except hiding: the only `try/except` added is `asyncio.QueueEmpty` on
  `get_nowait()`, which is part of the documented idempotency contract for
  the count queue (the `|win|` path may have beat us). Not error-hiding.
- Lint exclusion for `src/elitefurretai/scripts/`: untouched.
