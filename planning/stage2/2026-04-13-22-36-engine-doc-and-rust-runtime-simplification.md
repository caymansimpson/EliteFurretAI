# Engine Documentation And Rust Runtime Simplification

## Context

Stage 2 work spent multiple sessions benchmarking the two supported RL execution backends:

- `showdown_websocket`
- `rust_engine`

The most recent paired comparison used the checked-in 10-update configs:

- `src/elitefurretai/rl/configs/single_team_compare_rust_10_updates.yaml`
- `src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml`

Those runs were used to decide which backend should receive near-term optimization effort.

## Before State

Before this pass:

- backend learnings were spread across multiple Stage 2 notes and older durable docs
- `src/elitefurretai/rl/RUST_ENGINE.md` still read like the primary recommendation path
- the maintained Rust runtime still carried benchmark-era optimization branches that were no longer aligned with the project decision
- the Rust path still exposed centralized GPU inference and other tuning surfaces that increased maintenance cost without being the preferred production path

The repo therefore had a documentation mismatch:

- planning notes increasingly pointed toward Showdown-first optimization
- runtime code and durable docs still looked Rust-primary in several places

## Problem

Once Cayman decided to move forward with Showdown, the codebase needed to reflect that decision clearly.

The concrete problems were:

1. there was no single durable engine reference consolidating Stage 2 backend learnings
2. the maintained Rust runtime was more complex than needed for a supported fallback backend
3. tests still covered public optimization toggles and behavior that we no longer wanted to preserve
4. future sessions would likely re-open deprecated Rust tuning paths unless the simplification and recommendation were recorded explicitly

## Solution

This pass made four coordinated changes.

### 1. Created a single durable engine reference

Added:

- `src/elitefurretai/engine/ENGINE.md`

This document now consolidates:

- the current backend decision
- the paired learner-facing Rust vs Showdown comparison
- the key Stage 2 Rust learnings worth keeping
- the optimizations intentionally removed or de-prioritized
- module-level ownership for all engine runtime files

### 2. Updated durable docs to match the Showdown-first decision

Updated:

- `src/elitefurretai/rl/RL.md`
- `src/elitefurretai/rl/RUST_ENGINE.md`
- `src/elitefurretai/engine/__init__.py`

Net result:

- Showdown is now documented as the primary optimization and training path
- Rust remains documented as a supported fallback/comparison backend
- old Rust-specific docs are preserved as historical context instead of current guidance

### 3. Simplified the maintained Rust runtime

Updated runtime modules:

- `src/elitefurretai/engine/rust_battle_engine.py`
- `src/elitefurretai/engine/sync_battle_driver.py`
- `src/elitefurretai/engine/rust_model_benchmark.py`
- `src/elitefurretai/engine/rust_engine_benchmark.py`
- `src/elitefurretai/engine/battle_snapshot.py`
- `src/elitefurretai/engine/team_converter.py`
- `src/elitefurretai/rl/train.py`
- `src/elitefurretai/rl/config.py`

Kept in the maintained Rust path:

- cached request access around the binding
- request sanitization
- standalone `DoubleBattle` synchronization
- local worker inference
- local batched LSTM inference when natural batching exists
- automatic `embed_to_array()` fallback/use when available
- rejection diagnostics and trace capture

Removed or intentionally stopped preserving as maintained runtime behavior:

- centralized GPU inference for Rust workers
- remote model-sync plumbing tied to centralized inference
- extra public runtime toggles for fast-embed and batched-inference ablations
- alternate uncached binding wrapper surface
- additional Transformer batching complexity in the sync policy path

Compatibility note:

- deprecated central-inference config fields were left in `RNaDConfig` for YAML compatibility, but are now documented as ignored for the simplified Rust runtime

### 4. Updated tests to match the simplified contract

Updated:

- `unit_tests/engine/test_rust_battle_engine.py`
- `unit_tests/engine/test_sync_battle_driver.py`

The tests now assert the simplified behavior instead of preserving the old optimization surface.

## Reasoning

This helps build the best VGC bot because backend complexity only pays for itself if it improves learner-facing outcomes.

The latest paired Stage 2 comparison showed:

- Rust: `640` battles in `35m 23s`, `19011` learner steps, `8.95` learner steps/s
- Showdown: `640` battles in `13m 27s`, `6447` learner steps, `7.98` learner steps/s

Interpretation:

- Showdown reached the same learner update count much faster in wall-clock time
- Rust still provided useful technical lessons, but those lessons were mostly about correctness and local hot-path design rather than proving the entire Rust optimization stack should remain primary

So the right engineering move is:

- optimize Showdown first
- keep Rust healthy enough to compare, debug, and validate parity questions
- remove Rust-only maintenance burden that no longer advances the primary training path

This simplification also reduces the chance that future work accidentally optimizes stale infrastructure instead of the backend now giving the better near-term wall-clock training path.

## Planned Next Steps/Implementation Plan

1. Treat `showdown_websocket` as the default target for Stage 2 RL throughput and legality improvements.
2. Keep `rust_engine` runnable and regression-tested as a fallback/comparison backend.
3. Use `src/elitefurretai/engine/ENGINE.md` as the primary durable engine reference going forward.
4. If Rust is revisited as a primary optimization target later, reintroduce only the specific optimizations justified by fresh learner-facing evidence rather than restoring the old benchmark-era surface wholesale.

## Validation

Focused validation completed after the simplification pass:

- file error checks on the edited runtime and test files returned clean
- `pytest unit_tests/engine/test_rust_battle_engine.py unit_tests/engine/test_sync_battle_driver.py unit_tests/rl/test_config.py -q`
- result: `53` tests passed

During validation, an indentation/reward-loop regression in `sync_battle_driver.py` was introduced and then fixed in the same session before final test reruns. The final validated state includes the repaired reward logic.