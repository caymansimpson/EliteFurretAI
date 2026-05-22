# Second WSL VM Crash + Memory Watchdog + Slow-Handler Lead

## Context

The 2026-05-11 13:05 training run
([data/benchmarks/2026-05-11-easy-test/run.log](../../data/benchmarks/2026-05-11-easy-test/run.log))
reached update 886 over ~13 hours, then died at 02:06:35 on 2026-05-12 with
no Python traceback, no SIGTERM, no `wandb` shutdown. `who -b` confirms the
WSL2 VM next booted at 09:26 on 2026-05-12 (~7 h gap). This is the same
Hyper-V supervisor-kill pattern documented in
[2026-05-11-13-04-popup-recovery-lock-leak-fix.md](2026-05-11-13-04-popup-recovery-lock-leak-fix.md)
for the 2026-05-07 run.

## Before State

The popup-recovery lock-leak fix from
[2026-05-11-13-04](2026-05-11-13-04-popup-recovery-lock-leak-fix.md) was in
effect for this run. Plan was:

- Worker RSS should stay flat at ~1–1.5 GB (vs. the May 7 run's 3 GB climb).
- No WSL crash.
- `message_handler_timeouts` ≤ a handful; `battle_lock_tasks_cancelled`
  tracking popup rate × ~30.

Last good checkpoint:
[data/models/rl/bright-pine-189/ghosts/main_model_step_875.pt](../../data/models/rl/bright-pine-189/ghosts/main_model_step_875.pt)
at 01:53. Eleven updates (886 − 875) were lost when the VM died.

## Problem

The lock-leak fix worked partially — but not enough to keep the VM alive.

### Crash fingerprint

| Signal | May 7 run | May 11 run |
|---|---|---|
| Updates reached | 800 (~12 h) | 886 (~13 h) |
| Worker RSS at end | ~3 GB | **~2 GB** ← improved |
| Combined RSS estimate | ~25.6 GB vs 23 GiB | ~19–21 GB vs 23 GiB |
| Python traceback | none | none |
| SIGTERM delivered | no | no |
| WSL boot after death | yes | yes (09:26 next morning) |

Per-worker leak rate dropped roughly 33 % (1 → 2 GB over 13 h vs. 1 → 3 GB
over 12 h), but the system still ended up close enough to the 23 GiB ceiling
that one allocation spike (likely a checkpoint save + weight broadcast + a
batch of waiter accumulation) tipped it over.

### What the new diagnostics told us

The popup-recovery fix added three counters / signals. Their behaviour in
this run is the load-bearing evidence for where to look next:

| Diagnostic | Hits in 13 h | Reading |
|---|---|---|
| `battle_lock_tasks_cancelled` | 0 | Popups (May 7's cause) did **not** materially happen this run. The lock-cleanup branch of the fix was a defensive no-op. |
| `message_handler_timeouts` | 0 | No handler hung past the 60 s ceiling. |
| `Slow battle message` warning | **68 074** | A different, ongoing handler pathology — and the actual remaining leak source. |

### Slow-handler signature

Categorising the 68 074 warnings:

**By `handler_s` (the work-time bucket):**
- 43 380 events with `handler=0.00s` — these are **waiters** that ran instantly once they got the lock.
- ~12 400 events with `handler=0.06–0.15s` — fast handlers that waited 5 s on someone slow.
- **~1 400 events with `handler=5.0–5.10s`** — the offending slow handlers.

**By `types=...`:** every slow event is a battle-startup message —
`init,title,j` (most common), `request`, `player`, `player,gen,tier,rule,rule`,
`t:,gametype`, `j`.

**Suspicious clustering:** the slow `handler_s` values bunch tightly around
5 s (228 × 5.03s, 208 × 5.01s, 198 × 5.04s, 188 × 5.10s…). That's the
fingerprint of a fixed 5-second `wait_for` / `sleep` somewhere on the
battle-init code path, not a generic compute hotspot.

**Per-battle pattern:** for every new battle, the first message (`init,title,j`)
takes ~5 s of handler work; while the lock is held, the next 5–6 setup
messages for the same battle queue as waiters in `ps_client._active_tasks`,
each holding references to message text + battle state until they drain.

That backlog is the residual leak source the popup-recovery fix didn't
catch: a per-battle waiter burst, not a per-popup one. With ~95 000 battles
over 13 h, even small per-battle residual retention compounds.

### Other notable signals

- **Worker desync at end**: Worker 2 at batch 5300, Worker 4 at batch 3750
  — a 1500-batch spread (vs. < 1-minute spread at startup). Workers
  accumulate slow-handler debt at very different rates.
- **wandb network blip at 19:08:52** (`api: retrying error`) — incidental;
  recovered within a minute. Not the cause.

## Solution

Three independent tracks. Tracks A and B can land before the next launch;
Track C is a research dive.

### A. Memory watchdog (new safety net)

The trainer already has `generate_shutdown_signal()`
([train.py:103-113](../../src/elitefurretai/rl/train.py#L103-L113)) wiring
SIGTERM/SIGINT to a `shutdown_requested` `threading.Event`, which the main
loop checks each iteration
([train.py:917-918](../../src/elitefurretai/rl/train.py#L917-L918)) and
which causes the `finally` block to checkpoint and clean up
([train.py:1402-1447](../../src/elitefurretai/rl/train.py#L1402-L1447)).
What's missing is the trigger.

**Design (proposed — pending sign-off in this doc's Updates section):**

A new daemon thread launched from `main()` immediately after
`shutdown_requested = generate_shutdown_signal()` (around
[train.py:659](../../src/elitefurretai/rl/train.py#L659)):

1. Every 30 s, sample combined RSS = self + all recursive children via
   `psutil.Process(os.getpid()).children(recursive=True)`. mp.Process
   workers, Showdown subprocesses, and external VGCBench runners are all
   children of the trainer so a single tree walk captures the full
   footprint.
2. If combined RSS exceeds `hardware.memory_watchdog_threshold_gb` (new
   config field, default 18.0 GB):
   - Log `CRITICAL` with the breakdown (trainer / workers / showdown /
     vgcbench / other).
   - Call `shutdown_requested.set()`. The main loop sees this on the next
     iteration (≤ ~1 min later — a learner step's worth) and breaks into
     the existing `finally`, which saves a checkpoint and cleans up.
3. If `memory_watchdog_threshold_gb` is `None` or `0`, watchdog is
   disabled.

**Threshold rationale:** WSL2 has 23 GiB total. May 7 crashed at ~25.6 GB
estimated, May 11 at ~19–21 GB estimated. 18 GB gives ~5 GB headroom for
the in-flight learner step + checkpoint save + W&B flush before any
hard ceiling.

**Why set the event, not send SIGTERM:** the existing SIGTERM handler
already sets the same event, so going through the event is one less hop
and avoids any signal-delivery race. The trainer's main loop becomes the
single source of "we are shutting down."

**Files touched:**
- [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py) —
  add `memory_watchdog_threshold_gb: Optional[float] = 18.0` to
  `HardwareConfig`.
- [src/elitefurretai/rl/train.py](../../src/elitefurretai/rl/train.py) —
  new `start_memory_watchdog(shutdown_requested, threshold_gb)` helper +
  call site in `main()`.
- [src/elitefurretai/rl/configs/single_team.yaml](../../src/elitefurretai/rl/configs/single_team.yaml)
  — set `memory_watchdog_threshold_gb: 18.0` (and document in adjacent
  benchmark configs).
- [unit_tests/rl/test_config.py](../../unit_tests/rl/test_config.py) — add
  a test that asserts the default and round-trips a custom value.
- Either a new `unit_tests/rl/test_watchdog.py` or a focused test in
  [unit_tests/rl/test_train.py](../../unit_tests/rl/test_train.py) that
  spawns the watchdog with a mocked `psutil.Process.memory_info()` and
  asserts the event fires above threshold and stays clear below.

### B. Resume strategy

Next launch should resume from `main_model_step_875.pt` once Track A is in.
This is a config change in the active YAML; no code work.

### C. Slow-handler investigation (next leak source)

The 5-second handler cluster on battle-init messages is the next thing to
chase. Concrete next steps:

1. Read [poke-env/src/poke_env/ps_client/ps_client.py](../../../poke-env/src/poke_env/ps_client/ps_client.py)
   and [poke-env/src/poke_env/player/player.py](../../../poke-env/src/poke_env/player/player.py)
   along the `_handle_message` / `_handle_battle_message` path for
   `init`/`title`/`j`/`player`/`request`/`gametype`/`t:` handlers. Look
   specifically for `wait_for`, `sleep`, `asyncio.Event.wait`, or any
   `await` with a 5-second deadline.
2. Cross-check against `RLTrajectoryPlayer._battle_started_callback` /
   `_create_battle` and any team-validation or teampreview-prefetch path
   in [src/elitefurretai/rl/players.py](../../src/elitefurretai/rl/players.py).
3. Once identified, either (a) remove the synchronous wait, or (b) move
   it off the lock so subsequent messages for the same battle don't queue
   as waiters during the 5 s window. (b) preserves whatever invariant the
   wait was guarding while collapsing the waiter accumulation that's the
   actual leak driver.

This is a poke-env-side fix, so it needs to flow upstream once verified.

## Reasoning

- **Why a watchdog *and* a slow-handler fix?** They address different
  failure modes. The watchdog is a strict safety net so the VM never
  dies again regardless of which leak source is dominant — we get clean
  checkpoints rather than 7-hour outages. The slow-handler fix removes
  the underlying growth source so the watchdog isn't tripping every 13 h.
  Watchdog without slow-handler fix = clean shutdowns but still ~12 h
  cycles. Slow-handler fix without watchdog = still vulnerable to the
  next unknown leak.
- **Why 18 GB and not lower?** Tighter thresholds give more headroom but
  trip earlier, costing throughput. 18 GB is the lowest value that's
  comfortably above the steady-state-without-leak working set (~12 GB:
  6 × 1 GB workers + 5.4 GB VGCBench + 1.2 GB Showdown + trainer) so a
  clean run never triggers it.
- **Why poll every 30 s, not faster?** A poll costs ~1 ms × ~25 processes
  = trivial, but more frequent polling doesn't help: the trainer can't
  act on a watchdog signal mid-learner-step anyway (event is checked at
  loop top). 30 s is enough resolution to catch the climb.
- **Why not also bound per-worker RSS individually?** Possible follow-up
  but more invasive — workers don't have a clean checkpoint path. The
  trainer-side watchdog gets us 95 % of the benefit (no VM death) at
  ~30 lines of code.
- **Why pin the slow handler to the lock-held window, not the handler
  itself?** Because the actual leak isn't the handler's 5 s of work —
  it's the 5-6 waiters that accumulate during that 5 s, each retaining
  message + battle refs. Moving the wait off the lock (Track C step 3b)
  collapses the waiter count even if we never fix the underlying wait.

## Hard-constraint compliance

- WSL2 `pin_memory=False`: untouched.
- Both backends preserved: watchdog is backend-agnostic; the slow-handler
  investigation only affects Showdown.
- No try/except hiding errors: watchdog's `set()` is the same flow
  SIGTERM uses, no exception handling needed.
- `src/elitefurretai/scripts/` lint exclusion: untouched.

## Planned Next Steps

1. Confirm Track A design (threshold, knob name, configurable disable) —
   capture answer in Updates section below.
2. Implement Track A. Run `ruff` / `pyright` / `pytest unit_tests -q`.
3. Set up resume from `step_875.pt` for the next launch (Track B).
4. Launch next run.
5. After 30 min: confirm watchdog reports normal RSS (~12 GB) and is not
   tripping.
6. After 12 h: confirm no WSL crash; if RSS approaches 18 GB, watchdog
   should trip and produce `step_NNN.pt` cleanly.
7. In parallel, do Track C reading and identify the 5 s wait source.

## Updates

### 2026-05-12 — Track A implementation landed

Threshold set to **20 GB** (per discussion — 3 GiB headroom below WSL2's
23 GiB rather than my initial 18 GB recommendation, on the grounds that
the steady-state working set is well under that and 20 GB delays the trip
into the genuine danger zone).

Changes:
- New `training.memory_watchdog_threshold_gb: Optional[float] = 20.0` in
  [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py).
- `_sum_process_tree_rss_bytes` + `start_memory_watchdog` helpers in
  [src/elitefurretai/rl/train.py](../../src/elitefurretai/rl/train.py),
  wired into `main()` immediately after the SIGTERM/SIGINT handler
  setup. The watchdog daemon thread polls every 30 s, classifies child
  RSS as `trainer / workers / showdown / vgcbench / other`, and on
  breach logs a CRITICAL line with the per-role breakdown and calls
  `shutdown_requested.set()`.
- [single_team.yaml](../../src/elitefurretai/rl/configs/single_team.yaml)
  now sets `memory_watchdog_threshold_gb: 20.0` explicitly.
- New tests in
  [unit_tests/rl/test_memory_watchdog.py](../../unit_tests/rl/test_memory_watchdog.py):
  disabled-when-None, disabled-when-zero, fires-on-breach (with mocked
  RSS sampler), no-trip-below-threshold-plus-wakes-on-external-shutdown.
- New round-trip test for the config field in
  [unit_tests/rl/test_config.py](../../unit_tests/rl/test_config.py).

Quality gates: ruff clean, ruff-format clean, pyright clean (`0 errors,
0 warnings, 0 informations` on the changed files), `pytest unit_tests/rl
-q` 202 passed / 1 skipped / 2 warnings (was 197 / 1 / 2 before — 5 new
tests).

### 2026-05-12 — Track C lead: `_battle_count_queue` saturation

The "5-second fixed timeout" hypothesis from the Problem section is
**wrong**. Re-checking the handler-time distribution with 0.1 s buckets:

| Bucket | Count |
|---|---|
| 0.0 s (waiters) | 43 380 |
| 0.1 s | 12 112 |
| 5.0–5.5 s | ~5 750 |
| 5.5–6.0 s | ~3 350 |
| 6.0–7.0 s | ~1 000 |
| 7.0 s + tail to **29.19 s** | ~1 500 |

The cluster at 5.0+ is just the warning threshold (`_SLOW_HANDLER_S = 5.0`
in [poke-env/src/poke_env/ps_client/ps_client.py:183](../../../poke-env/src/poke_env/ps_client/ps_client.py#L183));
the underlying distribution is a long right tail, not a fixed deadline.

Strong suspect: **`_battle_count_queue` saturation in poke-env's
Player**. The default `max_concurrent_battles = 1` (poke-env
[player.py:62](../../../poke-env/src/poke_env/player/player.py#L62));
EFA never overrides it for training (only `analyze/play_model.py` sets
it). The queue is created with that as its `maxsize`
([player.py:134-135](../../../poke-env/src/poke_env/player/player.py#L134-L135))
and:

- `_create_battle` `await`s `self._battle_count_queue.put(None)` at
  [player.py:221](../../../poke-env/src/poke_env/player/player.py#L221) —
  **inside the per-battle lock held by `_handle_battle_message`**.
- When a player has battle X active and an `init` message for battle Y
  arrives, `_create_battle` blocks on `put(None)` until battle X ends
  ([player.py:317](../../../poke-env/src/poke_env/player/player.py#L317),
  `_battle_count_queue.get()` on `win`/`tie`).
- The wait equals the previous battle's remaining duration. VGC battles
  in this run typically last 5–30 s wall-clock — **exactly the observed
  tail**.
- While `init` is stuck, every subsequent message for battle Y
  (`t:,gametype`, `player`, `request`, …) queues as a waiter on
  battle Y's lock — exactly the 4–6 waiter-per-slow-holder ratio in
  the data.

The run.log shows challenges getting accepted back-to-back within ~1 s
(e.g. M02MD0097ACEE00 accepts battle 716792 at 02:06:24 then 716793 at
02:06:25), confirming the trainer dispatches concurrent challenges
faster than the player can drain them under `max_concurrent_battles=1`.

**Proposed fix** (deferred to a separate change, not in this commit):
pass `max_concurrent_battles=num_battles_per_pair` (or some sensible
upper bound like `num_players * num_battles_per_pair / num_players` =
`num_battles_per_pair`) when constructing the
`RLTrajectoryPlayer`/`RNaDModel` players in the training path. This
should collapse the slow-handler warnings to ~zero, eliminate the
waiter accumulation that's the residual leak source, and remove the
worker-desync caused by per-worker queue-debt skew.

**Why deferred:** raising the cap changes concurrency semantics — the
inference batcher's queue size, per-player memory footprint, and
inference-loop fairness all change with N concurrent battles per
player. That deserves a small focused PR with its own test plan, not a
ride-along on the watchdog change.

## Planned Next Steps (revised)

1. ✅ Track A landed.
2. Resume strategy for the next launch: use `resume_from:
   data/models/rl/bright-pine-189/ghosts/main_model_step_875.pt`.
   **Correction to the original 2026-05-11-13-04 doc's claim that
   ghost-pool files are "weights only":** `save_checkpoint`
   ([learners.py:820-843](../../src/elitefurretai/rl/learners.py#L820-L843))
   is the *same* function for the periodic save (`<run_dir>/ghosts/...pt`)
   and the shutdown save (`<run_dir>/...pt`). Both write
   `model_state_dict + optimizer_state_dict + step + curriculum + config`.
   Verified by `torch.load`-ing `step_875.pt`: 199 model tensors, 196
   optimizer state entries, step=875. Adam moments survive — no warm-up
   tax.
3. Launch next run with the watchdog active.
4. After 30 min: verify the watchdog log line `Memory watchdog armed
   at 20.0 GB combined RSS (poll every 30s)` appears at startup and
   does not trip on a healthy run.
5. After 12 h: confirm no WSL crash; if RSS approaches 20 GB, watchdog
   should trip and produce a full `step_NNN.pt` cleanly.
6. Open a small focused change for the `max_concurrent_battles` fix
   (Track C). Test plan: rerun for 1 h, confirm `Slow battle message`
   warning count drops by >90 %, and that per-worker batch progress
   stays in sync (< 100-batch spread end-to-end).
