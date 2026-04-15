# Showdown Request-Order Decode Fix

## Context

Stage 2 Showdown runtime debugging produced a reproducible set of websocket invalid choices within the first three turns. The dominant early failures were:

- `Protect` emitted with an explicit target token
- `Behemoth Bash` emitted without a required target token

These errors were captured from live websocket self-play under the same general concurrency profile used for Showdown RL diagnostics.

## Before State

Before this fix:

- the Showdown action mask in `src/elitefurretai/rl/fast_action_mask.py` enumerated valid actions from the live request payload
- poke-env legal move generation in `poke-env/src/poke_env/battle/double_battle.py` also used request-derived move ordering via `available_moves_from_request`
- but `src/elitefurretai/etl/encoder.py` decoded MDBO turn actions back into `SingleBattleOrder`s by indexing `moving_mon.moves`

That meant the action mask and legality logic were using one move ordering source, while the final decode path used another.

## Problem

`moving_mon.moves` is a persistent dictionary built from teambuilder order plus later move discovery order. It is not guaranteed to match the move-button order in `battle.last_request['active'][slot]['moves']`.

When those orders diverged, a legal action index could be decoded into the wrong move:

- a no-target action intended for `Protect` or `Iron Defense` could decode into `Behemoth Bash` or `Body Press`
- a targeted action intended for `Behemoth Bash` or `Body Press` could decode into `Protect` or `Iron Defense`

That is the exact shape observed in the Showdown invalid-choice captures.

## Solution

Final fix moved the invariant to the source of truth in poke-env:

1. updated `poke-env/src/poke_env/battle/pokemon.py` so `Pokemon.update_from_request()` reorders `self._moves` to match the request move order every time a fresh request arrives
2. kept MDBO turn decoding in `src/elitefurretai/etl/encoder.py` aligned with the project’s original assumption that move slot `N` corresponds to the `N`th entry in `moving_mon.moves`
3. retained legality validation in MDBO decoding so a broken invariant still fails loudly against `battle.valid_orders`

Also added regressions for both layers:

- `poke-env/unit_tests/environment/test_pokemon.py` verifies request parsing reorders `Pokemon.moves`
- `unit_tests/etl/test_encoder_edge_cases.py` verifies EliteFurret decoding behaves correctly once the source invariant is maintained

## Reasoning

This helps build the best VGC bot because the policy’s action index now has a single consistent interpretation across:

- action masking
- legal order generation
- final websocket serialization

Without that consistency, the model can appear to choose a legal action while the transport layer sends an illegal one. Fixing the `Pokemon.moves` ordering invariant is the real root-cause correction, because the embedder, legality checks, and decoder all depend on the same move-slot semantics.

## Planned Next Steps/Implementation Plan

1. Re-run the Showdown invalid-choice diagnostic capture with the same high-concurrency profile used for the original reproduction.
2. Confirm that the two Zamazenta error families (`Protect` target suffixes and untargeted `Behemoth Bash`) disappear.
3. If residual invalid choices remain, classify whether they come from stale request handling, force-switch handling, or another state/encoding mismatch.