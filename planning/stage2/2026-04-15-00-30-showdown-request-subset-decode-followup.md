# Showdown Request-Subset Decode Follow-Up

## Context

After the move-order invariant fix in `poke-env` and the force-switch mask fix in `fast_action_mask`, a 150-battle Reg G Showdown diagnostic with random teams per battle still reproduced invalid choices.

The failing family in `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b150` was:

- `invalid_choice_count=38`
- all failures were `request_type=turn`
- all failures were `Can't move` errors where Dragonite was choice-locked to `Outrage` but the submitted command used `ironhead`

## Before State

`src/elitefurretai/etl/encoder.py` decoded `move 1`, `move 2`, etc. by indexing directly into `list(moving_mon.moves.keys())`.

That behavior was correct only when the live request exposed the same move slots as the Pokemon's full known moveset.

In the Showdown websocket path, the action mask is built from `battle.last_request["active"][slot]["moves"]`, which can be a strict legal subset when choice-locked, disabled, or otherwise restricted.

## Problem

The Showdown path was still mixing two different move-slot spaces:

- mask generation used the live request subset
- action decoding used the full known moveset

That mismatch let the model sample an action that was legal in request-slot space and then decode it into the wrong concrete move when constructing the Showdown command.

## Solution

Updated `src/elitefurretai/etl/encoder.py` so `MDBO.to_double_battle_order()` prefers the live request move list for turn decoding when an active request exists for that slot.

Implementation details:

1. added `_get_request_ordered_moves(...)` to read the slot-local active request and call `moving_mon.available_moves_from_request(...)`
2. kept the existing `struggle` / `recharge` special case ahead of request-based decoding
3. kept the old full-moveset fallback for contexts without a live request

Added focused regressions in `unit_tests/etl/test_encoder_edge_cases.py` for:

- single legal request move decoding to the correct move under a restricted request
- partial request subsets preserving request order for move-slot indexing

## Reasoning

This keeps the Showdown websocket path internally consistent without redefining `Pokemon.moves` to mean "currently legal moves".

That matters because the rest of the codebase still needs `Pokemon.moves` to represent the revealed moveset, while request decoding needs the server's current legal button order.

The decoder is the right boundary for this fix because it is where action-space integers become concrete Showdown commands.

## Verification

Focused regression run:

- `python -m pytest unit_tests/etl/test_encoder_edge_cases.py -q`
- result: all tests passed

Showdown rerun after the decoder change:

- artifact dir: `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b150_requestfix`
- `completed_battles=150`
- `invalid_choice_count=90`

The original Dragonite / `ironhead` choice-lock family no longer appeared, which indicates the request-subset move decode mismatch was fixed.

## Residual Problem Families

The rerun exposed separate Showdown command-construction issues:

1. `Can't pass` when the second slot still has legal moves
   - dominant family: 77 failures
   - representative message: `/choose move drainpunch 2, pass`
   - representative error: `Your Tatsugiri must make a move (or switch)`
2. bad target serialization for ally-only and single-target moves
   - `Helping Hand` emitted with opponent targets or no target
   - `Draco Meteor` emitted without a required target
   - `Protect` emitted with an explicit target

These failures point to remaining mismatches between the action encoding and Showdown target/pass semantics, not the request-subset move-index bug fixed in this step.

## Planned Next Steps/Implementation Plan

1. Trace why the Showdown path can still emit `pass` during normal turn requests when both slots have legal actions.
2. Audit `MDBO` target decoding against request target types so ally-only, self-target, and single-target moves serialize with the correct target syntax.
3. Add focused regressions for:
   - ally-only moves like `Helping Hand`
   - self-target moves like `Protect`
   - single-target moves with exactly one live foe still requiring explicit targets
4. Re-run the same 150-battle random-team Showdown diagnostic after each family is fixed.