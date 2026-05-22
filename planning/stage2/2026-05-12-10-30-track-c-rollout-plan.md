# Track C Rollout Plan — `max_concurrent_battles` Concurrency Fix

## Context

Track C from
[2026-05-12-09-30-second-wsl-crash-and-watchdog.md](2026-05-12-09-30-second-wsl-crash-and-watchdog.md)
identified a candidate fix for the residual memory-growth source surfaced by
the slow-handler diagnostic: poke-env's default `max_concurrent_battles = 1`
serialises battle setup behind the previous battle's full duration, producing
the 5–30 s slow-handler distribution and ~80 MB/h per-worker RSS climb.

The proposed code change (configurable `hardware.max_concurrent_battles_per_player`
threaded through `WorkerOpponentFactory.create_player_pairs`) is mechanical.
The risk is **not** in the code — it's in tangling unverified variables.
This doc sequences the experiments so each effect can be attributed.

## Before State

After the 2026-05-12 watchdog landing
([2026-05-12-09-30-second-wsl-crash-and-watchdog.md](2026-05-12-09-30-second-wsl-crash-and-watchdog.md)):

- `single_team.yaml` resumes from
  `data/models/rl/bright-pine-189/ghosts/main_model_step_875.pt` via
  `resume_from`. The original 2026-05-11-13-04 doc's claim that
  ghost-pool files are "weights only" is **wrong** — verified by
  `torch.load`: `step_875.pt` has full optimizer state (196 entries)
  and `step=875`. `save_checkpoint`
  ([learners.py:820-843](../../src/elitefurretai/rl/learners.py#L820-L843))
  is the same function for periodic ghost saves and shutdown saves.
- `training.memory_watchdog_threshold_gb = 20.0` is set. The watchdog
  daemon will request graceful shutdown at 20 GB combined RSS, so the
  next memory event ends with a full `main_model_step_NNN.pt` checkpoint
  on disk instead of a Hyper-V VM kill.
- `hardware.max_concurrent_battles_per_player` does **not** exist yet
  — poke-env's default of 1 is in force.

## Problem

If we bundle the concurrency change with the next launch, the experiment
mixes:

1. New checkpoint init point (`step_875.pt` warm-start, fresh Adam).
2. New safety net (memory watchdog).
3. New concurrency semantics (raised `max_concurrent_battles`).

Any outcome — crash, no crash, faster, slower, more leaky, less leaky —
becomes hard to attribute. Worse, the slow-handler → memory-leak causal
link is still **hypothesis**, not measured fact. Shipping a fix for an
unverified diagnosis on a 13 h experiment costs a full training cycle if
we're wrong.

## Solution — Sequenced Experiments

### Stage 0 (today): Baseline run with watchdog only

- **Config**: `single_team.yaml` as it stands right now (watchdog at 20 GB,
  warm-start from `step_875.pt`, no concurrency change).
- **Duration**: long-run (until crash, clean SIGTERM, or 24 h).
- **What this measures**:
  - Watchdog wiring works end-to-end. Specifically: the startup log line
    `Memory watchdog armed at 20.0 GB combined RSS (poll every 30s)`
    appears and the daemon thread is alive.
  - When RSS approaches 20 GB (expected around hours 10–14 given May 11's
    ~1 GB/worker climb over 13 h), the watchdog trips, the trainer's
    `finally` block runs cleanly, and `main_model_step_NNN.pt` lands on
    disk with optimizer state.
  - If the run never trips the watchdog (i.e., RSS stays below 20 GB for
    the full duration), we learn the May 11 leak was a tail event, not
    deterministic — useful counter-evidence.
- **Pass criteria**: a full `main_model_step_NNN.pt` exists at the end,
  wandb closed cleanly, no WSL VM reset (`who -b` unchanged from launch).

### Stage 1 (after Stage 0): Land the Track C PR (default off)

Mechanical wiring — should not affect anything in production yet:

- New `hardware.max_concurrent_battles_per_player: Optional[int] = None`
  in [config.py](../../src/elitefurretai/rl/config.py).
- `WorkerOpponentFactory.__init__` reads it; `create_player_pairs`
  passes it through to both `RLTrajectoryPlayer` constructions
  ([opponents.py:1188, 1207](../../src/elitefurretai/rl/opponents.py#L1188-L1207))
  via a kwargs-only path so the None case leaves poke-env's default
  untouched.
- Commented-out `# max_concurrent_battles_per_player: 16` in
  [single_team.yaml](../../src/elitefurretai/rl/configs/single_team.yaml)'s
  `hardware:` block.
- Tests in
  [unit_tests/rl/test_worker_opponent_factory.py](../../unit_tests/rl/test_worker_opponent_factory.py):
  None → `Player._max_concurrent_battles == 1`; 16 → 16 on every
  constructed player.
- Round-trip test in
  [unit_tests/rl/test_config.py](../../unit_tests/rl/test_config.py).

**Pass criteria**: full `pytest unit_tests/rl -q` green, no behaviour
change with the YAML untouched (the YAML doesn't enable it yet).

### Stage 2 (short measurement run): Hypothesis test

- **Config**: Stage 1's YAML + uncomment
  `max_concurrent_battles_per_player: 16` (matches `num_battles_per_pair`).
- **Duration**: 1 h.
- **What this measures**: whether the slow-handler hypothesis is right.
  - Count `Slow battle message` lines in `run.log`. Baseline rate from
    May 11 was ~5 200/h. Expected drop: > 90 % (i.e., < 500/h, ideally
    < 50/h). If it doesn't drop, the `_battle_count_queue` hypothesis is
    wrong and the slow handlers come from somewhere else.
  - Worker batch-progress spread at t+1h. Baseline May 11 (≈ batch 280
    per worker at the 1 h mark, < 30-batch spread early on). New
    expected: similar or tighter. Bigger spread would be a regression.
  - Worker RSS at t+1h. Baseline ~1.0–1.2 GB per worker at 1 h. New
    expected: same or lower. Higher would mean concurrency raised
    in-flight battle state faster than the lock-waiter retention shrank
    — i.e., a wash or net negative.
- **Pass criteria**: slow-handler rate drops > 90 %, batch spread ≤ 50,
  per-worker RSS within ±200 MB of baseline.

### Stage 3 (long validation): Memory effect

Only run if Stage 2 passes.

- **Config**: same as Stage 2.
- **Duration**: 13 h, matching May 11's wall-clock.
- **What this measures**: whether the residual leak collapses with the
  slow-handler source removed.
  - Worker RSS at t+13h. Baseline: 2 GB. Target: flat near 1 GB. Anything
    above 1.5 GB means the leak has a non-`_battle_count_queue` source.
  - Watchdog trip: should not fire if memory stays bounded.
- **Pass criteria**: per-worker RSS < 1.5 GB sustained through t+13h
  and watchdog never trips.

### Stage 4: Decide on default

- If Stage 3 passes: leave the in-code default at `None` (least surprise
  for callers without an explicit YAML — keeps poke-env's default in
  legacy paths) but set
  `max_concurrent_battles_per_player: 16` as the active value in
  [single_team.yaml](../../src/elitefurretai/rl/configs/single_team.yaml)
  and the other training configs (`easy_test.yaml`, `benchmark_K1.yaml`,
  `benchmark_K3.yaml`). Update
  [src/elitefurretai/rl/RL.md](../../src/elitefurretai/rl/RL.md) with a
  one-paragraph note on the topology knob.
- If Stage 3 shows RSS still climbing: the leak is elsewhere. Stage 2's
  throughput win is still worth keeping (faster battle setup, tighter
  worker sync) but the memory-leak investigation moves to next candidates:
  embedder cache growth, inference-queue task retention, gradient buffer
  accumulation. The watchdog catches the next event cleanly so we have
  better data to start from.

## Reasoning

- **Why baseline first, not the bundled change?** The May 11 run is
  currently our only data point with this codebase shape, and it ended
  in a VM kill that destroyed observability. We need a baseline where
  the trainer survives to its own checkpoint so we have clean
  per-stage / per-worker / per-time RSS curves to compare against.
- **Why the in-code default stays None even after Stage 3 passes?**
  Analysis scripts (`analyze/play_model.py`, `analyze/evaluate.py`,
  `engine/analyze/showdown_benchmark.py`,
  `engine/analyze/showdown_invalid_choice_diagnostics.py`) construct
  `RLTrajectoryPlayer` directly without `OpponentPool` /
  `WorkerOpponentFactory`. A non-None code default would silently change
  their behaviour. Per-config opt-in via YAML keeps the surface area
  scoped to training.
- **Why 16 and not "unlimited"?** With `num_battles_per_pair = 16`, the
  trainer dispatches at most 16 challenges per pair at a time. Setting
  the cap to 16 means the queue never blocks under normal scheduling but
  also can't unboundedly accumulate if a future change accidentally
  enqueues more.
- **Why not also instrument poke-env to verify the hypothesis directly?**
  We could add a one-line log inside `_create_battle` around
  `_battle_count_queue.put(None)` to prove the wait time matches the
  observed handler distribution. Worth doing if Stage 2 fails — the
  log would tell us whether the bottleneck is even where I claimed.
  Skipped at the start because Stage 2's 1-hour rerun is itself a
  direct hypothesis test.

## Hard-constraint compliance

- WSL2 `pin_memory=False`: untouched at every stage.
- Both backends preserved: change is in the `RLTrajectoryPlayer`
  (Showdown-only) construction path; Rust backend's path doesn't go
  through `_battle_count_queue`.
- No try/except hiding errors: the change is a kwarg pass-through, no
  exception handling involved.
- `src/elitefurretai/scripts/` lint exclusion: untouched.

## Planned Next Steps

1. **Now**: kick off Stage 0 baseline run with the current
   [single_team.yaml](../../src/elitefurretai/rl/configs/single_team.yaml).
2. While Stage 0 runs: implement Stage 1 (the Track C PR) in a separate
   working tree — no behaviour change from a YAML perspective until
   Stage 2.
3. After Stage 0 reaches a clean terminal state (watchdog trip with
   checkpoint, or graceful 24 h completion): Stage 2.
4. After Stage 2 passes: Stage 3.
5. After Stage 3: Stage 4 decision (update YAMLs + RL.md or pivot to
   next leak candidate).

## Updates

_(empty — to be filled as stages land)_
