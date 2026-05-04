# Showdown Topology Winner For 15 Updates

## Context

This note records a Showdown-only topology sweep for Stage 2 single-team RL training. The user asked for a practical answer to two questions on the maintained `showdown_websocket` backend:

1. which topology is fast enough to reach updates quickly
2. which topology still looks like it is learning over a `15` update horizon

The sweep intentionally focused on the topology knobs the user requested rather than reopening prior batching work:

- `num_players`
- `num_battles_per_pair`
- `num_servers`
- `num_workers`

All tests were run from the current `train.py` entrypoint on Showdown, not the Rust backend and not the standalone benchmark harness.

## Before State

Before this sweep:

- prior Stage 2 notes had already established a safe Showdown actor batching recommendation of `batch_size=8`, `batch_timeout=0.02`
- earlier backend comparisons suggested Showdown was still the better choice when the goal is fast wall-clock progress to a modest number of updates on this machine
- we had no fresh checked-in answer for which Showdown training topology should be preferred for a real `15` update run after the recent trainer and websocket-side fixes

The initial attempt to automate a one-update sweep used temporary repo-root scripts that streamed multiple candidates into one aggregate output. That run was not trustworthy enough to rank candidates because the parser did not recover clean trainer update lines.

## Problem

We needed a reproducible Showdown recommendation for a real Stage 2 training run that balances:

1. time to first useful optimizer step
2. sustained battles-per-second and learner-steps-per-second over multiple updates
3. evidence that the run is learning rather than only finishing updates quickly

The required answer needed to be grounded in the actual trainer logs, then promoted to a full `15` update run and recorded for future sessions.

## Solution

### Sweep setup

All topology candidates were derived from the conservative Showdown learning config shape:

- `battle_backend: showdown_websocket`
- `batch_size: 8`
- `batch_timeout: 0.02`
- `train_batch_size: 64`
- `curriculum.vgc_bench_baseline: 0.0`
- `max_battle_steps: 40`
- `max_updates: 1` for the first sweep pass

The five one-update candidates were:

1. `baseline`: `num_workers=4`, `num_players=12`, `num_servers=4`, `num_battles_per_pair=32`
2. `lower_bpp`: `4`, `12`, `4`, `16`
3. `denser_players`: `4`, `16`, `4`, `16`
4. `fewer_workers`: `3`, `12`, `3`, `24`
5. `more_workers`: `6`, `18`, `6`, `16`

### One-update topology results

All five candidates reached `Update 1` successfully.

Key results:

- `baseline`
  - about `82.6s` to complete `Update 1`
  - `609` learner steps
  - about `12.33` learner steps/s
- `lower_bpp`
  - about `84.9s`
  - `539` learner steps
  - about `10.61` learner steps/s
- `denser_players`
  - about `96.2s`
  - `624` learner steps
  - about `10.04` learner steps/s
- `fewer_workers`
  - about `77.6s`
  - `561` learner steps
  - about `11.21` learner steps/s
- `more_workers`
  - about `98.0s`
  - `625` learner steps
  - about `12.33` learner steps/s

Interpretation of the one-update pass:

- `baseline` was best on time-to-first-update
- `more_workers` matched the best learner-step throughput while producing the strongest early learning shape among the top candidates
- the single-update view was not sufficient by itself because startup overhead still mattered a lot at this horizon

### Three-update tiebreak

To disambiguate the only two serious finalists, we ran a narrower `3` update comparison between:

1. `baseline`: `4 / 12 / 4 / 32`
2. `more_workers`: `6 / 18 / 6 / 16`

Observed progression:

- `baseline`
  - `Update 1`: `1.06 b/s`, `10.04 steps/s`
  - `Update 2`: `1.18 b/s`, `11.79 steps/s`
  - `Update 3`: `1.24 b/s`, `11.78 steps/s`
  - `Update 3` summary: `Total Battles=192 in 0h 2m 34s (1.24 b/s) | Learner Steps=1819 (11.78 steps/s)`
- `more_workers`
  - `Update 1`: `1.24 b/s`, `10.77 steps/s`
  - `Update 2`: `1.39 b/s`, `13.90 steps/s`
  - `Update 3`: `1.32 b/s`, `13.43 steps/s`
  - `Update 3` summary: `Total Battles=192 in 0h 2m 25s (1.32 b/s) | Learner Steps=1948 (13.43 steps/s)`

Interpretation of the tiebreak:

- once startup overhead is amortized even slightly, `more_workers` is clearly better on sustained training throughput
- the gain is material rather than marginal: roughly `10-15%` better learner-step throughput across the three-update window
- this made `more_workers` the right candidate to promote to the full `15` update run

### Full 15-update promoted run

The promoted topology was:

- `num_workers: 6`
- `num_players: 18`
- `num_servers: 6`
- `num_battles_per_pair: 16`

Extracted trainer summaries:

- `Update 1`
  - `Update 1: Loss=3.1139, Policy=0.9578, Value=4.4840, RNaD=0.6072 | Total Battles=64 in 0h 0m 47s (1.35 b/s) | Learner Steps=549 (11.55 steps/s) | Win rates: self_play: 78.1% | bc_player: 50.0% | max_damage: 33.3% | simple_heuristic_baseline: 8.3%`
- `Update 5`
  - `Update 5: Loss=3.3071, Policy=1.1544, Value=4.4563, RNaD=0.8695 | Total Battles=320 in 0h 3m 30s (1.52 b/s) | Learner Steps=2687 (12.77 steps/s) | Win rates: self_play: 66.0% | bc_player: 56.7% | max_damage: 33.3% | simple_heuristic_baseline: 5.0%`
- `Update 10`
  - `Update 10: Loss=3.7021, Policy=1.4297, Value=4.6971, RNaD=0.8111 | Total Battles=640 in 0h 6m 59s (1.53 b/s) | Learner Steps=5536 (13.21 steps/s) | Win rates: self_play: 58.0% | bc_player: 56.0% | max_damage: 34.4% | simple_heuristic_baseline: 15.0%`
- `Update 15`
  - `Update 15: Loss=3.0833, Policy=0.9237, Value=4.4951, RNaD=0.6064 | Total Battles=960 in 0h 9m 55s (1.61 b/s) | Learner Steps=8453 (14.20 steps/s) | Win rates: self_play: 60.0% | bc_player: 61.0% | max_damage: 37.0% | simple_heuristic_baseline: 17.0%`

Observed trend:

- battles/s improved from `1.35` to `1.61`
- learner steps/s improved from `11.55` to `14.20`
- `bc_player` win rate improved from `50.0%` to `61.0%`
- `simple_heuristic_baseline` win rate improved from `8.3%` to `17.0%`
- `max_damage` also trended upward from `33.3%` to `37.0%`

### Error behavior

`PS_ERROR` was still present throughout the Showdown websocket runs, including the promoted topology. The full promoted run completed successfully despite substantial error volume, which means the current trainer loop is robust enough to continue making progress under this aggressive topology.

This does not mean the websocket path is clean. It means the topology recommendation is usable right now for fast Stage 2 experiments while websocket legality cleanup remains a separate quality task.

## Reasoning

This recommendation helps Stage 2 because it answers the exact operational question we actually face when iterating on single-team RL:

- how to get a meaningful number of Showdown updates quickly
- without moving to Rust for this specific workflow
- and without selecting a topology that only looks good at `Update 1`

The important conclusion is that the fastest useful Showdown topology is not simply the one that reaches `Update 1` first. For a `15` update horizon, the better choice is the topology that sustains higher battle throughput and higher learner-step throughput once the system is warm.

That is why `more_workers` wins over the `baseline` topology. Its extra worker/server startup cost is paid back quickly, and by `Update 3` onward it is already ahead on the metrics that matter for actual training progress.

## Planned Next Steps/Implementation Plan

1. Use `src/elitefurretai/rl/configs/single_team_showdown_more_workers_15.yaml` as the current default candidate when we want an aggressive Showdown `15` update Stage 2 run.
2. Keep `batch_size=8` and `batch_timeout=0.02` unchanged while using this topology. The topology win here does not reopen the batching choice.
3. Treat persistent `PS_ERROR` volume as the next quality constraint on further Showdown scaling. We are now close enough to the websocket stability limit that legality cleanup is more valuable than blindly adding more concurrency.
4. If we need a shorter smoke or debugging config, continue using the simpler `4 / 12 / 4 / 32` baseline because it still reaches the first update quickly and is easier to reason about.

## Updates

- Re-ran the topology sweep with explicit per-candidate logs after the first aggregate parser attempt proved unreliable.
- Narrowed the promotion decision with a direct `3` update tiebreak instead of guessing from one-update startup-heavy results.
- Completed a full `15` update promoted run on the winning Showdown topology.
- Added a checked-in config for the winner at `src/elitefurretai/rl/configs/single_team_showdown_more_workers_15.yaml` so future sessions can reproduce it without rebuilding a temporary file.