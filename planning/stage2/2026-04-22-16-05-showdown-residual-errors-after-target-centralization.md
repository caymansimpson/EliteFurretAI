# Showdown Residual Errors After Target Centralization

## Context

This note records the first post-fix Showdown websocket invalid-choice diagnostic run after centralizing request-target legality and tightening the fast-mask commander and force-switch logic.

The goal of this pass was not to implement another fix immediately. It was to follow the established Showdown debugging loop and answer four questions first:

1. which invalid-choice families still survive under a training-like pressure profile
2. whether the previous top families actually shrank or disappeared
3. which surviving family is now the dominant one
4. what the smallest next source-level fix should be

This run used the maintained websocket diagnostic harness rather than ad hoc log scraping.

## Before State

Before this rerun:

- request-target legality had been centralized into `src/elitefurretai/rl/request_targeting.py`
- the fast mask had been tightened for commander and force-switch handling
- shared target resolution had been wired into the Showdown player and sync-driver paths
- the prior PS_ERROR review had identified three high-value families to attack:
  - commander-related move misuse
  - force-switch pass misuse
  - `Uproar` target mismatch

The open question was whether those families had actually moved in a fresh websocket diagnostic run.

## Problem

The raw invalid-choice total alone is misleading because Showdown retries the same bad request many times.

We needed to follow the Ralph Wiggum loop and group the remaining failures by:

- exact error message
- attempted `/choose` shape
- request type
- deduped request windows rather than raw retry count

Only then could we decide what to fix next.

## Solution

### Reproduction profile

Ran:

- `python -m elitefurretai.engine.showdown_invalid_choice_diagnostics`
- config: `src/elitefurretai/rl/configs/single_team_showdown_more_workers_15.yaml`
- checkpoint: `data/models/supervised/curious-darkness-77_best.pt`
- random teams on both sides
- `200` battles
- `max_concurrent_battles=4`
- `device=cpu`
- `batch_size=8`
- `batch_timeout=0.02`
- `max_battle_steps=40`

Artifacts:

- summary: `/tmp/showdown_invalid_choice_post_target_centralization/summary.json`
- raw records: `/tmp/showdown_invalid_choice_post_target_centralization/invalid_choices.jsonl`

### High-level result

The run completed successfully but still produced many websocket invalid choices.

Summary:

- `completed_battles=200`
- `invalid_choice_count=9286`
- `p1_invalid_choice_count=9286`
- `p1_turn_le_3_count=2`
- request-type distribution: all `turn`

Important deduping result:

- raw invalid-choice records: `9286`
- deduped request windows: `318`

So the surviving problem is not thousands of unrelated bugs. It is a much smaller set of repeated bad request windows.

### Grouped families

Top exact messages by raw count:

- `4528` x `[Invalid choice] Can't move: Your Dondozo doesn't have a move matching dracometeor`
- `1547` x `[Invalid choice] Can't move: Your Dondozo doesn't have a move matching icywind`
- `1515` x `[Invalid choice] Can't move: Your Dondozo doesn't have a move matching muddywater`
- `100` x `[Invalid choice] Can't move: You can't choose a target for Uproar`

Top attempted messages included:

- `/choose move dracometeor 2, move wavecrash 1`
- `/choose move muddywater, move orderup -1`

Top normalized choice shapes included:

- `/choose move muddywater, move orderup X`
- `/choose move icywind, move orderup X`

Top deduped groups were still commander-shaped request windows, not force-switch windows.

### What disappeared versus what survived

What appears materially improved:

- the old force-switch pass family did not show up in the grouped residual results
- the current run showed only `turn` request failures, not `force_switch`

What still survives:

- commander-family move misuse is still the dominant residual family by a large margin
- `Uproar` target handling still survives as a smaller but real family

## Analysis

### Biggest surviving family

The biggest grouped family is still commander-related move misuse, specifically Tatsugiri move payloads being turned into active-slot commands that Showdown evaluates against Dondozo.

Representative record:

- attempted: `/choose move dracometeor 1, move orderup 1`
- error: `Can't move: Your Dondozo doesn't have a move matching dracometeor`

Representative request facts from the same record:

- battle-state active order: Tatsugiri, Dondozo
- request-side active order: Tatsugiri, Dondozo
- slot `0` request moves: `muddywater`, `icywind`, `dracometeor`, `sleeptalk`
- slot `1` request moves: `orderup`, `protect`, `wavecrash`, `earthquake`
- active payload `commanding` flags: `[None, None]`
- side active-pokemon `commanding` flags: `[True, False]`

Important local check:

- `slot_is_commanding(...)` returns `True` for slot `0` and `False` for slot `1` on this representative request

That means the helper itself is not the direct failure point anymore.

### Shared invariant that is still broken

The surviving invariant break is:

> when a side-pokemon entry marks the active slot as `commanding=True`, the policy path must never be able to sample or serialize a move action for that slot.

The request clearly carries enough information to know this.

Because `slot_is_commanding(...)` returns the correct answer on the representative request, the remaining commander-family bug is now most likely one of these:

1. the async batched policy path is no longer using the same commander-aware legality that we believe it is
2. a slot-order mismatch still exists between the mask-building view and the final serialized order for commander states
3. a correct commander-aware mask is being built, but the sampled action is not actually being constrained by that mask in the failing path

The evidence is now against the simpler explanation that the request lacks commander metadata altogether.

### Secondary surviving family

`Uproar` still survives as a smaller family:

- representative attempted choice: `/choose move uproar 1, move woodhammer 1`
- Showdown error: `You can't choose a target for Uproar`

This indicates the current `randomNormal` handling in `src/elitefurretai/rl/request_targeting.py` is still wrong for at least this Showdown request shape.

The current centralization treated `randomNormal` as a target-required single-target family. The diagnostic evidence says that is not correct for websocket serialization of `Uproar`.

## Reasoning

This diagnostic pass materially narrowed the problem.

The important progress is not the raw count. It is that the residual error surface is now concentrated:

- force-switch misuse is no longer the dominant problem
- the commander family is still the dominant bug
- `Uproar` is still a smaller target-shape bug

That is a better state because the next fix can now be selected based on one dominant grouped family instead of a broad mixed bag.

## Suggested Next Change

Following the debugging loop, the next source-level fix should target the commander family first and should stop for explicit approval before implementation.

### Recommended next fix

Add a focused commander invariant check in the async Showdown player path around the exact sampled action and the exact request snapshot used to decode it.

Specifically:

- in `src/elitefurretai/rl/players.py`, immediately after sampling `action_idx` and before serializing through `MDBO.to_double_battle_order(...)`, decode the per-slot action choice against the same request snapshot and assert that any `commanding=True` slot resolves only to `pass`
- if the invariant is violated, log the exact action index, decoded per-slot choices, request snapshot, and mask summary for that battle tag and fall back to a safe pass/default order for that request
- use that instrumentation to determine whether the source bug is:
  - a bad commander-aware mask
  - mask misuse during sampling
  - or slot-order drift between mask construction and final decode

### Why this is the right boundary

- the helper already answers the commander question correctly on the failing request
- the surviving family lives in the actual async Showdown policy path, not in the heuristic helpers
- this is the narrowest place where the sampled action, the mask, and the final request-backed decode are all visible together

### Tradeoff

- this adds a small amount of debugging logic in the hot path while the family is still live
- but it should quickly convert the remaining commander ambiguity into one precise failing invariant instead of forcing another blind fix attempt

### Follow-on after commander

If the commander family is removed, the next smallest fix should be to change `randomNormal` handling for `Uproar` so that Showdown serialization omits the explicit target.

## Planned Next Steps/Implementation Plan

1. Stop here for approval before implementing the next commander-path fix.
2. After approval, add the focused commander invariant instrumentation and the smallest regression test at the async Showdown player layer.
3. Re-run the same diagnostic harness and confirm whether the commander family disappears.
4. If it does, repeat the loop for the smaller surviving `Uproar` family.