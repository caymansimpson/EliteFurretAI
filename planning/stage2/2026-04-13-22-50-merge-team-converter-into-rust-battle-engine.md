# Merge Team Conversion Into Rust Battle Engine

## Context

The Rust fallback runtime had already been simplified so that the maintained engine path favored clarity over preserving benchmark-only module boundaries.

One remaining split was:

- `src/elitefurretai/engine/rust_battle_engine.py`
- `src/elitefurretai/engine/team_converter.py`

Even though both files existed only to support the Rust binding boundary.

## Before State

Before this change:

- `rust_battle_engine.py` owned the Rust binding adapter, request cache, request sanitization, and standalone `DoubleBattle` synchronization
- `team_converter.py` separately owned `pokepaste_to_rust_team()` and `pokepaste_to_rust_team_json()`
- `sync_battle_driver.py` imported from both modules
- the converter tests imported from the standalone converter module

This kept one extra engine module alive even though it was not shared by other backends.

## Problem

The team-conversion helpers are not a general engine abstraction. They are specific to the Rust binding's expected team schema.

Keeping them in a separate module made the current layout more fragmented than necessary:

1. the Rust binding boundary was split across two files
2. the sync driver needed two engine imports for one backend concern
3. durable docs still had to describe an extra runtime module that did not represent an independent subsystem

## Solution

This pass merged the Rust team-conversion helpers directly into:

- `src/elitefurretai/engine/rust_battle_engine.py`

and removed:

- `src/elitefurretai/engine/team_converter.py`

Updated files:

- `src/elitefurretai/engine/rust_battle_engine.py`
- `src/elitefurretai/engine/sync_battle_driver.py`
- `unit_tests/engine/test_rust_battle_engine.py`
- `src/elitefurretai/engine/ENGINE.md`
- `src/elitefurretai/rl/RL.md`

Net result:

- the Rust binding adapter and Rust-team schema conversion now live in one module
- the sync driver imports the helper functions from `rust_battle_engine.py`
- durable docs now describe a consolidated Rust boundary instead of a separate converter module

## Reasoning

This helps the project because the current goal is not to preserve a maximally granular Rust module graph. The goal is to keep the fallback backend understandable and cheap to maintain while Showdown receives primary optimization effort.

The converter helpers are tightly coupled to the Rust binding shape, so merging them into `rust_battle_engine.py` improves cohesion:

- one place defines the Rust boundary
- one place is updated when the binding schema changes
- one place is documented as owning Rust-specific preprocessing

That is the right tradeoff for the simplified fallback runtime.

## Planned Next Steps/Implementation Plan

1. Keep converter-focused tests, but continue treating them as coverage for `rust_battle_engine.py` rather than for a separate subsystem.
2. If team conversion later becomes shared across multiple backends, split it back out only then.
3. Continue simplifying Rust-only surfaces when they do not improve learner-facing outcomes or maintainability.

## Validation

Post-merge file diagnostics returned clean for:

- `src/elitefurretai/engine/rust_battle_engine.py`
- `src/elitefurretai/engine/sync_battle_driver.py`
- `unit_tests/engine/test_rust_battle_engine.py`
- `src/elitefurretai/engine/ENGINE.md`
- `src/elitefurretai/rl/RL.md`

Focused engine tests were then rerun after the merge.