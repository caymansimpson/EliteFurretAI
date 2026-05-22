# Showdown Target Legality Centralization

## Context

This note records a focused Stage 2 Showdown websocket legality cleanup after the 15-update topology winner run surfaced three dominant invalid-choice families:

1. commanding slots still admitted move actions in the fast mask path
2. force-switch legality in the fast mask was looser than the sync driver path
3. target-required move handling still drifted across the fast mask, heuristic player path, and final action serialization, especially for request-target-driven moves like `Uproar`

The goal was to centralize target legality so there is one maintained owner for request-driven target semantics, then use that centralization to remove the top three Showdown legality leaks.

## Before State

Before this change:

- `src/elitefurretai/rl/fast_action_mask.py` owned one copy of request-target legality and move-target mapping
- `src/elitefurretai/rl/players.py` sometimes used the request-aware helper and sometimes fell back directly to `battle.get_possible_showdown_targets(...)`
- `src/elitefurretai/engine/sync_battle_driver.py` already behaved more strictly for force-switch handling, but its target handling still depended on helpers exposed from the fast mask module
- the fast mask did not short-circuit commanding slots before move enumeration
- the fast mask force-switch path admitted `pass` too broadly compared with the stricter sync-driver behavior
- request target types like `randomNormal` were not centralized, so target-required moves could still fall back to no-target behavior in one path while another path required an explicit target

## Problem

This split ownership made Showdown legality fragile in exactly the ways the PS_ERROR review predicted:

- commanding slots could still generate illegal move orders in one path
- force-switch requests could still generate illegal `pass` combinations in one path
- single-target and ally-target semantics could still drift across the fast mask, heuristic action generation, and final request-driven order serialization

As long as these paths were maintained separately, each websocket cleanup pass risked fixing only one caller while leaving the others behind.

## Solution

### Centralized request-target owner

Added `src/elitefurretai/rl/request_targeting.py` as the shared owner for request-driven target legality.

It now centralizes:

- target-type to legal-target mapping on the current board
- request-move target resolution
- the common RL action-generation entry point used by the heuristic player path

Important kept behavior:

- request target types are treated as the source of truth when present
- `randomNormal` is now treated as a target-required single-target request shape
- `adjacentAlly` now returns no legal targets when the ally slot is gone instead of silently falling back to a no-target move

### Commander fix in the fast mask

Updated `src/elitefurretai/rl/fast_action_mask.py` so `get_valid_slot_actions(...)` short-circuits commanding slots to `pass` only before move enumeration.

This keeps the fast legality path aligned with the already stricter Showdown player and sync-driver behavior.

### Force-switch fix in the fast mask

Updated `src/elitefurretai/rl/fast_action_mask.py` force-switch handling to match the stricter sync-driver rule:

- non-forced slots remain `pass` only when any slot is in force switch
- forced slots only get `pass` when the number of available switch targets is smaller than the number of forced slots
- healthy forced-switch slots no longer receive a spurious `pass` option just because the slot is part of a force-switch phase

### Shared adoption in the Showdown player and sync driver

- `src/elitefurretai/rl/players.py` now routes target legality through the centralized resolver instead of maintaining a separate request-vs-poke-env split at the call site
- `src/elitefurretai/engine/sync_battle_driver.py` now imports request-target legality from the shared module rather than through the fast mask module

## Reasoning

This helps Stage 2 because websocket invalid-choice cleanup is now anchored on one maintained legality owner instead of three partially overlapping ones.

That reduces two real risks:

1. fixing one legality family in only one caller
2. reintroducing drift later when the request format or target semantics change

It also directly addresses the three highest-value PS_ERROR families from the 15-update Showdown review without reopening unrelated training or model code.

## Validation

Focused executable validation passed on the touched slice:

- `pytest unit_tests/rl/test_fast_action_mask.py unit_tests/etl/test_encoder_edge_cases.py unit_tests/engine/test_sync_battle_driver.py`
  - result: `59` passed, `1` skipped

Added or expanded focused fast-mask regressions for:

- commanding-slot pass-only behavior
- force-switch no-pass behavior when enough replacements exist
- `randomNormal` explicit-target handling
- `adjacentAlly` with no partner returning no legal targets

One adjacent async-player regression probe hit a pre-existing test-harness issue instead of a legality regression:

- `unit_tests/rl/test_multiprocess_actor.py -k 'Uproar or request_mutation or same_turn'`
- failure surfaced: partially constructed `RLTrajectoryPlayer` test double missing `_diagnostics`

That issue was not part of the requested legality slice and was left unchanged.

## Planned Next Steps/Implementation Plan

1. Re-run a short Showdown training or invalid-choice diagnostic pass to measure whether the dominant websocket rejection families materially shrink after this legality centralization.
2. Decide whether the remaining target fallback in the non-request heuristic path should also be eliminated entirely or kept as a non-Showdown fallback.
3. Fix the separate `RLTrajectoryPlayer` test-harness setup issue in `unit_tests/rl/test_multiprocess_actor.py` before relying on that slice for future websocket regression probes.