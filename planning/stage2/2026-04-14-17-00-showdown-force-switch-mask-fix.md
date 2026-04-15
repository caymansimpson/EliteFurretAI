# Showdown Force-Switch Mask Fix

## Context

After the move-order invariant fix removed the early `Protect` and `Behemoth Bash` invalid choices, the remaining Showdown websocket failures all came from force-switch requests during high-concurrency self-play diagnostics.

The residual diagnostic run at `data/benchmarks/2026_04_14_showdown_next10_anyturn_seed5_b150` showed:

- `invalid_choice_count=24`
- all errors were `request_type=force_switch`
- every failure was a `/choose ... pass` command where both slots still needed a replacement

## Before State

Before this fix, `src/elitefurretai/rl/fast_action_mask.py` treated `pass` as a per-slot force-switch option whenever the active Pokemon in that slot was fainted.

That meant the mask could admit:

- `switch, pass`
- `pass, switch`

even when both slots were force-switched and there were two distinct legal replacement targets available.

## Problem

`pass` legality in doubles force-switch is not a slot-local property.

It depends on the joint replacement state of both slots:

- if both slots need to switch and there are enough distinct legal replacements, both slots must switch
- if both slots need to switch and there are fewer distinct replacements than required slots, one slot may pass as a shortage fallback

The old mask encoded the wrong abstraction boundary, so the policy could sample commands that the Showdown server correctly rejected.

## Solution

Updated `src/elitefurretai/rl/fast_action_mask.py` to move `pass` legality to the joint force-switch pairing stage:

1. `get_valid_slot_actions()` now returns real switch targets during force switch and only falls back to `pass` when a slot has no switch targets at all
2. added `_allow_force_switch_pass()` which computes whether pass is legal from the joint state of both forced slots
3. `pass` is admitted only when both slots are force-switched and the number of distinct replacement targets is smaller than the number of forced slots

Added focused regressions in `unit_tests/rl/test_fast_action_mask.py` for:

- double force-switch with two replacements: pass must be masked out
- double force-switch with one replacement: exactly one slot may pass

## Reasoning

This helps build the best VGC bot because the model’s valid-action set now matches game legality more closely during a high-impact decision type.

The fix is also efficient:

- no inference-path redesign
- no websocket retry dependence
- only a tiny constant-time check over the distinct switch targets already enumerated by the mask builder

By removing illegal `switch, pass` combinations at the masking layer, we avoid wasting policy mass on actions that can never be executed on the Showdown server.

## Planned Next Steps/Implementation Plan

1. Keep this force-switch rule aligned with `unit_tests/etl/test_battle_order_validator.py` so the mask and legality validator continue to share the same doubles semantics.
2. Re-run larger Showdown diagnostic sweeps if needed to check whether any non-force-switch invalid-choice families remain under heavier concurrency.
3. If more websocket invalid choices appear later, classify them separately from this resolved force-switch issue.