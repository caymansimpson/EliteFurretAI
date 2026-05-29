# Engine Runtime Guide

This document is the durable reference for EliteFurretAI's battle-execution layer. The engine package owns battle-execution concerns; RL algorithm concerns live in [`src/elitefurretai/rl/`](../rl/) (see [RL.md](../rl/RL.md)).

## Current Backend

`showdown_websocket` — local Pokemon Showdown servers plus poke-env websocket players — is the sole supported backend. The earlier `rust_engine` in-process simulator was removed once Showdown optimization made it the clear winner end-to-end.

## What Stage 2 Taught Us (and why we kept Showdown)

These lessons drove the decision and still inform how we evaluate future engine changes.

### 1. Random battle throughput is a weak decision metric

Random-policy engine stepping overstated raw simulator throughput and missed the parts of the loop that actually dominated training time. The meaningful comparison had to include:

- embedding
- action masking
- policy inference
- legality handling
- actual training-loop update cadence

The most decision-relevant numbers came from model-backed benchmarks and paired `train.py` runs, not from pure environment stepping.

### 2. Showdown won the latest learner-facing wall-clock comparison

In the paired 10-update comparison on the Transformer + `full` feature path:

- Rust: `640` battles in `35m 23s` — `0.30 b/s`, `19011` learner steps, `8.95` steps/s
- Showdown: `640` battles in `13m 27s` — `0.79 b/s`, `6447` learner steps, `7.98` steps/s

Showdown reached the same 10 learner updates ~2.6× faster in wall-clock time, even though Rust produced slightly more learner steps per second. The runs weren't semantically identical — Rust produced more learner steps per trajectory than Showdown — but the wall-clock gap was enough to commit.

### 3. End-to-end legality and trajectory validity matter more than raw step speed

The dominant Rust losses came from long/truncated trajectories, legality drift and retries, and Python-side synchronization costs — not from the simulator core. The same logic applies to ongoing Showdown work: validity and usable training signal dominate over step throughput.

### 4. Showdown is not "solved"

The Showdown run still logged a large number of invalid websocket choices. Choosing Showdown means it's the right place to spend optimization effort — not that it's already clean.

## Engine Package Layout

### [`vgc_environment.py`](vgc_environment.py)

Worker-side environment over the Showdown websocket backend. Workers call `VGCEnvironment.from_config(...)` once, then `run_battle_batch(n)` in a loop. Internally it wraps `_ShowdownBackend` + [`WorkerOpponentFactory`](../rl/opponents.py) + [`RLTrajectoryPlayer`](../rl/rl_trajectory_player.py)s.

### [`showdown_server_manager.py`](showdown_server_manager.py)

Owns local Showdown server lifecycle (`launch_showdown_servers`, `shutdown_showdown_servers`) and port allocation across workers. Also coordinates external vgc-bench runner processes when the curriculum uses them.

### [`battle_renderer.py`](battle_renderer.py)

Pure functions that render `DoubleBattle` / `Observation` / `Pokemon` / `Move` into human-readable strings. Used by [`HumanPlayer`](../agents/human_player.py) for the interactive CLI (state snapshots, action reference, teampreview) and by the inference debug paths (`format_observation`, `format_battle_log` — successors to the broken `observation_to_str` / `battle_to_str` once in `inference/inference_utils.py`).

The module imports nothing from `inference/` or `rl/` and does no I/O. To preserve that boundary `engine/__init__.py` loads `VGCEnvironment` lazily via `__getattr__`, so importing `engine.battle_renderer` from `inference/` doesn't pull in the RL training stack.

### [`analyze/`](analyze/)

Engine-level diagnostic and benchmark scripts:

- [`showdown_benchmark.py`](analyze/showdown_benchmark.py) — websocket comparison harness used historically for backend comparison; still useful for sanity-checking Showdown throughput regressions
- [`showdown_embedder_profile.py`](analyze/showdown_embedder_profile.py) — profiles the embedder hot path on real Showdown trajectories
- [`showdown_invalid_choice_diagnostics.py`](analyze/showdown_invalid_choice_diagnostics.py) — diagnoses invalid-choice patterns (the dominant remaining Showdown failure mode)

## Ownership Guide

Use this when deciding where work belongs.

### Put work in [`battle_renderer.py`](battle_renderer.py) when

- you need to render any battle structure as a string for a human reader or a debug log
- the rendering rules diverge between the human CLI and the inference debug dumps (split the leaf primitive vs. the composite view, not the file)

### Put work in [`vgc_environment.py`](vgc_environment.py) (or `_ShowdownBackend`) when

- the worker-side battle loop needs to change
- batching or task-management behavior is wrong
- trajectory formatting at the env boundary is wrong

### Put work in [`showdown_server_manager.py`](showdown_server_manager.py) when

- server launch, shutdown, or port allocation is wrong
- external vgc-bench process management is wrong

### Put work in RL modules when

- legality / masking logic is changing
- policy or learner logic is changing
- opponent-pool / curriculum / centralized-inference logic is changing (these live in [`rl/opponents.py`](../rl/opponents.py), [`rl/model_registry.py`](../rl/model_registry.py), [`rl/rl_trajectory_player.py`](../rl/rl_trajectory_player.py))

## Related Docs

- [`src/elitefurretai/rl/RL.md`](../rl/RL.md)
- [`planning/stage2/2026-04-13-11-20-showdown-benchmark-and-backend-comparison.md`](../../../planning/stage2/2026-04-13-11-20-showdown-benchmark-and-backend-comparison.md)
- [`planning/stage2/2026-04-13-14-00-training-speed-recommendations.md`](../../../planning/stage2/2026-04-13-14-00-training-speed-recommendations.md)
