# Showdown Request-Slot Decode And Commander Request Sync

## Context

After the earlier request-subset decoder fix, a 150-battle Reg G Showdown random-team diagnostic still produced 90 invalid choices in `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b150_requestfix`.

The remaining families were:

- 77 normal-turn `Can't pass` failures, mostly with Tatsugiri still treated as commanding after the live request said otherwise
- 13 target/move serialization failures caused by decoding turn actions against a compact legal-move list instead of the raw request button slots

## Before State

`src/elitefurretai/etl/encoder.py` still used `available_moves_from_request(...)` as a compacted legal move list during request-aware decoding.

That was enough to fix strict request subsets like choice lock, but it still lost the original request slot numbering whenever disabled moves appeared before enabled ones.

`src/elitefurretai/rl/fast_action_mask.py` still preferred stale battle-state `Effect.COMMANDER` data over the current request's `commanding` flags.

In poke-env, Commander cleanup was mostly message-driven, with an extra Dondozo faint special case in `abstract_battle.parse_message()`, but requests themselves were not reconciling Commander state back onto the `Pokemon.effects` map.

## Problem

The Showdown path still had two independent drift points:

1. move decoding could map a legal raw request slot onto the wrong concrete move when disabled buttons preceded enabled ones
2. the legality mask could continue allowing `pass` for a Tatsugiri that was no longer commanding because cached battle effects lagged behind request truth

These were both source-of-truth problems between the live Showdown request and local cached state.

## Solution

Implemented three focused fixes:

1. `src/elitefurretai/etl/encoder.py`
   - replaced compact request-move decoding with raw request-slot decoding
   - `MDBO.to_double_battle_order()` now resolves `move N` against the exact `request["active"][slot]["moves"][N-1]` entry
   - decoding reuses `moving_mon.available_moves_from_request(...)` on a single synthetic move request so special move handling stays centralized

2. `src/elitefurretai/rl/fast_action_mask.py`
   - changed `slot_is_commanding(...)` to trust request metadata first
   - battle-state `Effect.COMMANDER` is now only a fallback when the request does not specify commanding status

3. `poke-env/src/poke_env/battle/pokemon.py`
   - synchronized `Effect.COMMANDER` during `Pokemon.update_from_request(...)`
   - if request data says `commanding: true`, Commander is started when missing
   - if request data says `commanding: false`, Commander is ended when stale

## Reasoning

The decoder and the mask must use the same button space. In Showdown, that space is the raw request payload, not the compact list of currently enabled moves.

Commander truth also belongs to the latest request when it is available. Using stale effects ahead of request metadata is exactly how normal-turn `pass` became legal after Commander had already ended.

Syncing Commander inside poke-env request updates is the cleanest source fix because it keeps local battle state aligned with Showdown without relying on special-case downstream compensation.

## Verification

Focused regressions passed:

- `python -m pytest unit_tests/etl/test_encoder_edge_cases.py unit_tests/rl/test_fast_action_mask.py -q`
- `python -m pytest unit_tests/environment/test_double_battle.py -q`
- result: 13 passed in poke-env, EliteFurret targeted suite passed with one existing skipped test

Added regressions for:

- disabled request slots preserving raw move-slot indexing
- ally-target move decoding preserving explicit ally targets
- request `commanding: false` overriding stale Commander effects in the mask
- poke-env request updates clearing stale Commander effects

Live rerun:

- artifact dir: `data/benchmarks/2026_04_15_showdown_random_teams_seed8_b150_requestfix2`
- `completed_battles=150`
- `invalid_choice_count=1`
- `p1_invalid_choice_count=1`
- remaining family: `Expanding Force needs a target`

## Residual Problem

One invalid choice remains:

- attempted message: `/choose move expandingforce, move drainpunch -1`
- server error: `Expanding Force needs a target`

This indicates our local target heuristic for `Expanding Force` under Psychic Terrain is still diverging from Showdown request truth. The live request explicitly exposed `target: normal`, and Showdown required an explicit target even though the local helper rewrote it to spread behavior.

That residual is separate from the two fixes in this step and is now the next isolated family to clean up.

## Planned Next Steps/Implementation Plan

1. Remove or narrow local target-type rewrites when the live Showdown request already provides a concrete `target` value.
2. Add a focused regression for `Expanding Force` under Psychic Terrain using the exact request shape from `battle-gen9vgc2024regg-616833`.
3. Re-run the same 150-battle random-team Showdown diagnostic to confirm `invalid_choice_count=0`.