# Rust Backend Operationalization

## Context

Stage 2 Rust backend work has moved beyond pure prototyping. We now have enough benchmark and parity evidence to make an operational recommendation about whether the Rust path should be used for training, how Rust-specific work should be organized going forward, and where durable documentation should live.

## Before State

Before this note:

- the repo had detailed Stage 2 investigation notes and benchmark writeups
- the RL subsystem already had a primary overview doc at [src/elitefurretai/rl/RL.md](src/elitefurretai/rl/RL.md)
- there was no durable Rust-backend-specific reference doc in the RL doc set
- the benchmark folders still contained a large volume of superseded intermediate diagnostics and temporary smoke logs

## Problem

We needed to make three things explicit:

1. Whether the Rust backend is ready to use for real training.
2. How future Rust backend work should be organized so runtime, legality, and benchmark changes do not blur together.
3. Where durable backend knowledge should live, separate from dated investigation notes.

We also wanted to prune superseded generated artifacts so the benchmark directories stop growing around old intermediate states.

## Solution

Make the Rust backend operational recommendation explicit, add a durable backend reference under the RL primary docs, link it from [src/elitefurretai/rl/RL.md](src/elitefurretai/rl/RL.md), and remove clearly superseded benchmark artifacts.

## Reasoning

The benchmark evidence now supports a practical training recommendation.

In the latest paired long backend comparison after the commander and transformed-target fixes:

- Rust ended at `6.24 learner steps/s` with `24.2 learner steps per battle`
- Showdown ended at `3.97 learner steps/s` with `11.8 learner steps per battle`
- Rust had `0` invalid-choice lines in the paired run
- Showdown still had `308` invalid-choice lines in the paired run

That is enough to justify using Rust for training now, while still preserving the websocket backend as a supported fallback and comparison path.

The documentation split also matters:

- dated planning notes are the right place for evolving investigation history
- the RL primary docs are the right place for durable system knowledge, ownership boundaries, and extension guidance

## Planned Next Steps

1. Keep Rust as the default training backend for new Stage 2 runs.
2. Preserve websocket support for backend parity checks and fallback.
3. Continue landing legality fixes in shared masking code when they should affect both backends.
4. Prefer updating the durable Rust backend doc when ownership boundaries or core invariants change.

## Updates

- 2026-04-10 14:45: Landed the durable documentation split.
    - [src/elitefurretai/rl/RL.md](src/elitefurretai/rl/RL.md) now has an explicit Battle Backends section that states both supported backends and documents the current operational recommendation.
    - Added [src/elitefurretai/rl/RUST_ENGINE.md](src/elitefurretai/rl/RUST_ENGINE.md) as the durable backend reference covering:
        - backend boundary and ownership
        - core Rust runtime classes
        - source-of-truth rules
        - request sanitization expectations
        - testing and benchmarking rules
        - practical operational guidance

- 2026-04-10 14:45: Operational recommendation recorded.
    - The Rust backend is now considered ready for real training use.
    - The websocket backend should remain supported and should not be deleted by default.
    - The current policy is:
        - use `rust_engine` as the primary training runtime
        - keep `showdown_websocket` for fallback and comparison

- 2026-04-10 14:45: Work-organization recommendation recorded.
    - Runtime contract and request/state synchronization work belongs primarily in:
        - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py)
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py)
    - Shared legality and backend parity work belongs primarily in:
        - [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py)
        - [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py)
    - Throughput and backend-comparison work belongs primarily in:
        - [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py)
        - [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py)
        - backend compare configs under [src/elitefurretai/rl/configs](src/elitefurretai/rl/configs)

- 2026-04-10 14:45: Benchmark cleanup performed.
    - Removed clearly superseded intermediate JSONL diagnostics, trace directories, and temporary smoke logs from the benchmark folders.
    - Kept the final paired post-target-fix backend comparison artifacts and the Stage 2 planning narrative.

- 2026-04-10 15:10: Added durable battle-debugging instructions to the Rust backend reference.
    - [src/elitefurretai/rl/RUST_ENGINE.md](src/elitefurretai/rl/RUST_ENGINE.md) now documents how to:
        - capture rejected-choice JSONL diagnostics
        - record full per-battle trace files
        - render readable markdown battle reviews
        - choose between `rust_model_benchmark.py` and `train.py` depending on whether the question is trace-level debugging or operational throughput

- 2026-04-10 15:35: Moved engine runtime modules into a dedicated package.
    - Introduced [src/elitefurretai/engine](src/elitefurretai/engine) as the package for battle-execution concerns rather than keeping those files flat under RL.
    - Moved these modules into the new package:
        - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py)
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py)
        - [src/elitefurretai/engine/battle_snapshot.py](src/elitefurretai/engine/battle_snapshot.py)
        - [src/elitefurretai/engine/team_converter.py](src/elitefurretai/engine/team_converter.py)
        - [src/elitefurretai/engine/showdown_server_manager.py](src/elitefurretai/engine/showdown_server_manager.py)
        - [src/elitefurretai/engine/rust_engine_benchmark.py](src/elitefurretai/engine/rust_engine_benchmark.py)
        - [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py)
    - Reasoning:
        - these files are runtime and engine-boundary concerns, not core RL algorithm definitions
        - the benchmark entrypoints stay in the engine package because they compare execution backends rather than modeling choices
        - `team_converter.py` belongs with the engine package because it currently converts team text into the Rust-engine binding shape rather than exposing a shared backend-agnostic schema

- 2026-04-10 15:40: Validated the package move through the real training entrypoint.
    - Re-ran a short Rust smoke through [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py) with [src/elitefurretai/rl/configs/rust_smoke.yaml](src/elitefurretai/rl/configs/rust_smoke.yaml).
    - Smoke log artifact: [data/models/rl/rust_smoke/engine_package_rust_smoke.txt](data/models/rl/rust_smoke/engine_package_rust_smoke.txt)
    - Result highlights:
        - Rust backend selected successfully and Showdown launch skipped
        - `Update 1` completed with `58 learner steps` at `4.92 learner steps/s`
        - final checkpoint saved to `data/models/rl/rust_smoke/main_model_step_1.pt`
    - Interpretation:
        - the module move did not just preserve unit tests; the real worker/learner Rust path still runs after the package split

- 2026-04-10 15:55: Mirrored the engine package split in test layout and Showdown naming.
    - Moved engine-focused tests into [unit_tests/engine](unit_tests/engine) so runtime ownership and regression coverage are grouped the same way.
    - Renamed the Showdown process helper module to [src/elitefurretai/engine/showdown_server_manager.py](src/elitefurretai/engine/showdown_server_manager.py) to make its websocket-only responsibility explicit.
    - Updated imports in RL entrypoints and durable docs to point at the renamed module and new engine test paths.