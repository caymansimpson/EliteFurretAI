# Showdown Request Snapshot Fix

## Context

After the earlier request-slot and Commander fixes, the next 500-battle Reg G Showdown random-team diagnostic still produced invalid websocket choices.

The current-loop artifacts showed:

- `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b500_requestfix5`
- `completed_battles=500`
- `invalid_choice_count=1742`
- every failure came from a single repeated family: `Uproar needs a target`

## Before State

The Showdown action path had already become request-aware in two separate places:

1. `src/elitefurretai/rl/fast_action_mask.py` built legality from `battle.last_request`
2. `src/elitefurretai/etl/encoder.py` decoded MDBO turn actions against request slots when a live request was present

However, `BatchInferencePlayer._choose_move_async()` still used those request-aware components at different times against the mutable live battle object:

- the mask was generated early from the current `battle.last_request`
- after batched inference completed, `MDBO.to_double_battle_order()` re-read `battle.last_request`

That meant a same-turn request mutation could still let the player sample an action under one request and decode/send it under another.

## Problem

The remaining `Uproar` family did not reproduce when replaying the stored request through the current mask logic in isolation.

That was the key clue that the broken invariant was no longer a stable target-rule bug. The real problem was that the mask and decoder were not operating on a single immutable request snapshot.

In the websocket path, that is enough to create illegal commands even when each individual component is correct, because request updates can arrive while batched inference is in flight.

## Solution

Implemented a request-snapshot fix across the Showdown action path.

### 1. Snapshot the request inside `BatchInferencePlayer`

In `src/elitefurretai/rl/players.py`:

- deep-copy `battle.last_request` before enqueuing inference
- compute a compact request fingerprint from `rqid`, active move payloads, force-switch state, and active side metadata
- after inference returns, drop the action if the live request fingerprint no longer matches the original snapshot

### 2. Use the same snapshot for masking and decoding

In `src/elitefurretai/rl/fast_action_mask.py`:

- `fast_get_action_mask()` now accepts an optional request override

In `src/elitefurretai/etl/encoder.py`:

- `MDBO.to_double_battle_order()` now accepts an optional request override
- request-slot move decoding can be performed against the exact request snapshot used for masking

This keeps legality enumeration and final Showdown serialization on the same button-space snapshot.

### 3. Add focused regressions

Added regressions for:

- request-override decoding in `unit_tests/etl/test_encoder_edge_cases.py`
- same-turn request mutation drop behavior in `unit_tests/rl/test_multiprocess_actor.py`

## Reasoning

This is the right abstraction boundary because the websocket request is the real source of truth for legal move buttons, targets, and same-turn battle actions.

Once the player is using batched async inference, “request-aware” is not sufficient by itself. The whole action path must be request-consistent.

Using a single request snapshot is both cleaner and more performant than adding more move-specific patches, because it fixes the class of race-like request-mismatch bugs rather than only the last visible symptom.

## Verification

Focused regressions:

- `pytest unit_tests/etl/test_encoder_edge_cases.py unit_tests/rl/test_multiprocess_actor.py -q`
- result: passed

Short smoke validation after the fix:

- `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b100_requestfix6_smoke`
- `completed_battles=100`
- `invalid_choice_count=0`

Full requested validation:

- `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b500_requestfix6`
- `completed_battles=500`
- `p1_wins=241`
- `invalid_choice_count=0`
- `p1_invalid_choice_count=0`
- `p1_turn_le_3_count=0`

## Planned Next Steps/Implementation Plan

1. Per the Showdown debugging loop, scale the clean random-team sweep upward by roughly 10x next if we want the stronger no-invalid-confidence target.
2. If a new family appears at larger scale, start from grouped artifacts again rather than assuming it is related to this fixed request-snapshot class.
3. Keep future Showdown runtime fixes request-snapshot-based whenever batched inference or async timing is involved.