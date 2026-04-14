# Disable Poke-Env Observation Logging By Default

## Context

Stage 2 work is currently focused on improving self-play and simulator throughput for RL training runs.

One avoidable cost in the Showdown-backed runtime was that `poke-env`'s `AbstractBattle` recorded full per-turn `Observation` snapshots and appended every parsed event into observation history even when those observations were never consumed.

This cost sits directly in the per-message battle update path used by actor processes.

## Before State

Before this change:

- every `AbstractBattle` allocated and maintained observation history by default
- every parsed showdown event appended into `current_observation.events`
- every `turn` message deep-copied battle state into a fresh `Observation`
- finished battles always stored the final observation state
- replay saving reused the same observation history, but non-replay RL runs still paid the same bookkeeping cost

## Problem

Observation logging was enabled on the default battle path despite not being needed for standard RL self-play training.

That added avoidable Python overhead in the hottest part of the websocket backend:

1. extra list appends for every parsed message
2. extra object construction and dictionary copying every turn
3. extra retained Python objects for battle-long observation history

For actor-heavy RL runs, this slows simulation throughput for no learner-facing benefit.

## Solution

Updated `poke-env` battle construction so observation logging is now opt-in instead of default-on.

Implementation details:

- added `log_observations: bool = False` to `AbstractBattle`, `Battle`, and `DoubleBattle`
- enabled observation bookkeeping only when `log_observations=True` or replay saving is enabled
- guarded event appends, per-turn observation snapshots, and final battle observation storage behind that flag
- kept replay generation working by auto-enabling observation logging when `save_replays` is truthy
- updated the observation tests so observation-specific coverage opts in explicitly and added a default-path check that no observation history is recorded

## Reasoning

This helps build the best VGC bot because Stage 2 training quality depends on how many useful battles we can simulate per unit time.

Removing unnecessary Python bookkeeping from the actor battle loop improves the Showdown backend's cost profile without changing battle semantics, action selection, or replay behavior when replay capture is requested.

This is the right tradeoff for RL training infrastructure:

- fast by default for training
- still available for debugging and replay workflows
- no backend removal or divergence from the supported websocket path

## Planned Next Steps/Implementation Plan

1. Re-run throughput comparisons on the websocket backend after this change once the intended Python environment is active.
2. If any debugging workflow needs observation history without replay saving, thread `log_observations=True` through that caller explicitly.
3. Continue auditing other per-message battle bookkeeping in `poke-env` and the actor stack for default-on costs that do not help learner outcomes.

## Validation

Code diagnostics were checked for the touched files after the edit.

Focused runtime validation was partially blocked in this session because the available interpreter lacked workspace test dependencies such as `pytest` and `orjson`, so only static/tooling validation completed here.