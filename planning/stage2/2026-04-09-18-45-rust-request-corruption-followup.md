# Rust Request Corruption Follow-Up

## Context
The Rust self-play throughput work in Stage 2 is now limited less by raw stepping speed and more by rejected-choice loops that cause stall-limit truncations. We already kept one proven legality fix in the sync driver for force-switch handling, but a second investigation pass showed that the remaining rejections are not explained by one more obvious local Python legality bug.

This document is the follow-up handoff focused specifically on the remaining request-corruption behavior: what concrete examples we observed, what battle situations those examples cluster around, and what the next debugging and implementation steps should be.

## Before State
Before this follow-up:
- the best kept short diagnostic state was the force-switch legality fix plus JSONL rejection diagnostics
- that kept state was measured with `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_final_kept_fix_200.jsonl`
- a deeper experimental patch set was tried and then reverted because it regressed measured throughput despite reducing some old rejection clusters

The reverted experimental patch set attempted three ideas:
- synchronize poke-env active-slot cache directly from request parsing
- filter switch actions against `battle.available_switches`
- infer move target types from move metadata when the Rust request left `target` blank

## Problem
We need to understand the remaining rejected choices at the level of specific battle states and request payloads, not just aggregate counters.

The key questions are:
1. What individual corrupted requests actually look like.
2. In what battle situations they occur.
3. What source-of-truth mismatch is causing them.
4. What we should instrument and fix next.

## Solution
Summarize the surviving evidence from the kept diagnostic state and from the reverted experimental pass, then turn that into a concrete request-corruption debugging plan.

The core conclusion from this pass is:

The next bug is most likely a deeper Rust request contract mismatch where `request["active"]` can describe an acting mon while the paired `side.pokemon` entries still carry stale active or faint state. Because of that, Python legality generation cannot safely treat raw `side.pokemon` as authoritative during those turns.

## Reasoning
The important distinction is between:
- a local legality bug, where Python simply computes the wrong legal action set from a correct request
- a request-corruption bug, where the request itself contains internally inconsistent views of the current battle state

The retained evidence points to the second case.

In the kept state, the dominant rejection shapes were `move, pass` and `switch, pass`, which suggested stale slot identity or stale force-switch visibility. In the experimental state, once those obvious local symptoms were patched, the remaining dominant rejection shape became `move, move`, and those move requests still failed despite apparently reasonable local legality logic. That pattern is much more consistent with an upstream request inconsistency than with one more missing local rule.

## Planned Next Steps
1. Build a filtered corruption corpus from the existing JSONL diagnostics so we can review only the first corrupted step per battle tag and request shape.
2. Add adapter-side sanity checks that compare `request["active"]`, `side.pokemon`, protocol-tracked active slots, and poke-env active slots on every rejected choice.
3. Preserve the last `N` protocol lines per side around the first corruption so we can see exactly which switch, faint, or replace messages preceded the bad request.
4. Implement request sanitization at the Rust adapter boundary by rebuilding active-slot identity and condition from protocol-tracked slot occupancy plus `active[]`, rather than trusting raw `side.pokemon` for the front slots.
5. Re-run the same 100-battle diagnostic benchmark and keep the change only if it improves both rejection counts and `non_truncated_battles_per_second` relative to the current kept baseline.

## Updates

- 2026-04-10 00:55: Kept the combined adapter reconciliation and force-switch legality fix after it materially improved both the short diagnostic run and the longer model-backed benchmark.
    - Root cause confirmed from the diagnostics:
        - `DoubleBattle.parse_request()` does not reliably overwrite `battle._active_pokemon` when stale slot mappings already exist, so even a sanitized request can leave poke-env believing one front slot is missing or mapped to the wrong mon.
        - once that stale active-slot state was corrected, the dominant remaining rejection family shifted almost entirely into force-switch windows, where the sync driver was still admitting `pass` even on double replacement requests with enough healthy replacement targets.
    - Code changes kept:
        - [src/elitefurretai/rl/rust_battle_engine.py](src/elitefurretai/rl/rust_battle_engine.py) now reconciles `battle._active_pokemon` directly from the sanitized request front pair immediately after request application, including the fallback path used when `parse_request()` rejects an edge-case shape.
        - [src/elitefurretai/rl/sync_battle_driver.py](src/elitefurretai/rl/sync_battle_driver.py) now only admits `pass` during force-switch handling when the number of healthy replacement targets is smaller than the number of forced slots. This removes the invalid `pass, switch X` and `switch X, pass` choices that were still being generated on `forceSwitch=[True, True]` requests with multiple legal replacements.
        - [unit_tests/rl/test_rust_battle_engine.py](unit_tests/rl/test_rust_battle_engine.py) gained a regression test for active-slot reconciliation from sanitized request front-pair state.
        - [unit_tests/rl/test_sync_battle_driver.py](unit_tests/rl/test_sync_battle_driver.py) gained a regression test covering double force-switch requests with enough replacements, asserting that no spurious `pass` options are offered.
    - Validation:
        - `pytest unit_tests/rl/test_sync_battle_driver.py unit_tests/rl/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed after the kept fix landed, with the same one skipped trapped-scenario test as before.
    - Short diagnostic benchmark artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_active_slot_reconcile_force_switch_fix_200.txt`
        - paired rejection stream: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_active_slot_reconcile_force_switch_fix_200.jsonl`
    - 200-battle result:
        - `completed_battles=200`
        - `truncated_battles=3`
        - `non_truncated_battles=197`
        - `duration_seconds=156.508`
        - `battles_per_second=1.278`
        - `non_truncated_battles_per_second=1.259`
        - `p1_rejected_choices=79`
        - `p2_rejected_choices=78`
        - `p1_unrecovered_rejections=79`
        - `p2_unrecovered_rejections=78`
    - Comparison versus the prior kept front-pair-normalization baseline (`rejection_diagnostics_front_pair_normalization_200`):
        - stall truncations improved from `43 -> 3`
        - non-truncated battles improved from `157 -> 197`
        - non-truncated throughput improved from `1.201 -> 1.259`
        - `p1` rejected choices improved from `1634 -> 79`
        - `p2` rejected choices improved from `1665 -> 78`
        - the remaining rejection surface is now small and move-heavy (`157` move requests in the diagnostic JSONL, versus thousands previously)
    - Longer 1000-battle benchmark artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rust_model_benchmark_active_slot_reconcile_force_switch_fix_1000.txt`
    - 1000-battle result:
        - `completed_battles=1000`
        - `truncated_battles=90`
        - `non_truncated_battles=910`
        - `duration_seconds=1644.460`
        - `battles_per_second=0.608`
        - `non_truncated_battles_per_second=0.553`
        - `turn_limit_truncations=64`
        - `stalled_limit_truncations=26`
        - `p1_rejected_choices=786`
        - `p2_rejected_choices=681`
        - `p1_unrecovered_rejections=785`
        - `p2_unrecovered_rejections=677`
    - Comparison versus the previous long-horizon best-path benchmark (`rust_model_benchmark_best_with_causes.txt`):
        - truncations improved from `307 -> 90`
        - non-truncated battles improved from `693 -> 910`
        - `p1` rejected choices improved from `9943 -> 786`
        - `p2` rejected choices improved from `10253 -> 681`
        - stall-limit truncations improved from `307 -> 26`
        - throughput shifted from a stall-heavy short-battle regime (`1.385 battles/s`) to a much cleaner retained-data regime (`0.553 non-truncated battles/s` versus `0.960` previously). The important change is that the actor is now finishing far more battles naturally instead of cycling through rejected-choice loops and stall truncations.
    - Revisited earlier experiments after this result:
        - the active-order-only sanitizer experiment stays reverted because it reduced move-pass errors only by shifting the failure surface into switch phases and collapsing fallback recovery.
        - the pivot-slot tracking experiment also stays reverted because it regressed both throughput and rejection counts; the stronger signal was stale `battle._active_pokemon`, not unslotted pivot replacement inference.
        - the earlier sync-driver-side request-slot recovery experiment stays reverted because the correct place to repair the move-side state mismatch was the adapter boundary, not broader legality overrides inside the driver.
    - Updated interpretation:
        - the earlier request-corruption hypothesis was correct, but the decisive bug was not just malformed front-pair ordering by itself. The decisive bug was that the sanitized request was not being propagated into poke-env's authoritative active-slot cache, and once that was fixed, the remaining invalid choices were mostly over-admitted force-switch passes.
        - The remaining work is now much narrower: a small residual set of move-phase rejections that are no longer the dominant throughput bottleneck.

- 2026-04-09 23:55: Added a first-error battle recorder so one failing Rust self-play battle can be debugged end to end instead of only through per-rejection JSONL events.
    - Code change kept:
        - [src/elitefurretai/rl/rust_battle_engine.py](src/elitefurretai/rl/rust_battle_engine.py) now preserves the full raw and normalized protocol transcript per side in addition to the bounded recent protocol history.
        - [src/elitefurretai/rl/sync_battle_driver.py](src/elitefurretai/rl/sync_battle_driver.py) now supports an optional `error_battle_record_path` that records the first battle to hit a rejected choice.
        - [src/elitefurretai/rl/rust_model_benchmark.py](src/elitefurretai/rl/rust_model_benchmark.py) now exposes that recorder via `--error-battle-record-path`.
        - [unit_tests/rl/test_sync_battle_driver.py](src/elitefurretai/rl/test_sync_battle_driver.py) gained focused coverage for the new recording path.
    - Recorder contents:
        - final request types and final sanitized requests for both sides
        - a per-step trace with request snapshots, legal-choice previews, submitted choices, fallback choices, and acceptance results
        - the full raw and normalized protocol transcript for both sides
    - Validation:
        - `pytest unit_tests/rl/test_sync_battle_driver.py unit_tests/rl/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed after landing the recorder.
    - Captured artifact:
        - `data/benchmarks/training_throughput_2026_04_09/error_battle_full_debug_80.json`
        - paired summary: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_full_debug_80.txt`
        - paired rejection stream: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_full_debug_80.jsonl`
    - Immediate outcome:
        - this gives a single battle-sized debugging unit with enough context to manually walk from protocol lines to request shape to choice generation and rejection handling without stitching together multiple artifacts.

- 2026-04-09 23:32: Tried a sync-driver legality repair that trusted the sanitized request front pair even when poke-env still had a missing active slot, then reverted it after the benchmark regressed.
    - Hypothesis tested:
        - the remaining `move X, pass` loop might be caused by `_get_slot_actions()` collapsing slot 1 to `pass` whenever `battle.active_pokemon[slot]` was `None`, even if the sanitized request still exposed a healthy active front pair.
    - Experimental code change:
        - [src/elitefurretai/rl/sync_battle_driver.py](src/elitefurretai/rl/sync_battle_driver.py) was temporarily changed so move-slot availability could be inferred from the sanitized request front pair instead of only from poke-env active-slot state.
        - a focused regression test was added and then removed with the revert once the benchmark showed the change should not land.
    - Validation:
        - focused RL tests passed both before and after the experiment, so the issue was not local correctness of the code edit itself.
    - Benchmark artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_request_slot_recovery_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=53`
        - `non_truncated_battles=147`
        - `duration_seconds=176.743`
        - `battles_per_second=1.132`
        - `non_truncated_battles_per_second=0.832`
        - `p1_rejected_choices=699`
        - `p2_rejected_choices=812`
        - `p1_fallback_recoveries=0`
        - `p2_fallback_recoveries=0`
        - `p1_unrecovered_rejections=699`
        - `p2_unrecovered_rejections=812`
    - Comparison versus the current kept front-pair-normalization baseline (`rejection_diagnostics_front_pair_normalization_200`):
        - the targeted `move X, pass` rejection family dropped from `1450` events to `0`
        - but raw throughput regressed (`1.529 -> 1.132`)
        - retained throughput regressed badly (`1.201 -> 0.832`)
        - stall truncations regressed (`43 -> 53`)
        - fallback recovery collapsed from `1692` recovered rejections to `0`
    - What changed in the rejection surface:
        - the move-phase `move X, pass` loop disappeared
        - but the dominant rejection family shifted into switch-phase failures like `pass, switch 2`, `switch 2, pass`, and `switch 2, switch 3`
        - request-type counts shifted from `move=2160, switch=1139` in the kept baseline to `move=216, switch=1295` in the experiment
    - Conclusion:
        - the missing-active-slot symptom is real, but directly trusting the sanitized request for move-slot availability in the sync driver is not the right fix
        - it over-corrects the move-phase symptom and breaks the recovery behavior that was still containing many switch-phase errors
        - this experiment was reverted so the repository remains on the best proven state
    - Updated next target:
        - the remaining durable fix likely belongs closer to the adapter/state-sync boundary, not in a broader sync-driver override of poke-env active-slot presence

- 2026-04-09 23:01: Implemented front-pair request normalization at the adapter boundary and reran the 200-battle diagnostic benchmark.
    - Code change kept:
        - [src/elitefurretai/rl/rust_battle_engine.py](src/elitefurretai/rl/rust_battle_engine.py) now sanitizes the cached Rust request itself, not just the poke-env ingestion path.
        - For move requests, the adapter now:
            - annotates each roster entry with its original request index,
            - matches `active[]` entries back to roster mons by move IDs,
            - reorders both `active[]` and the leading `side.pokemon` entries from protocol-tracked `a` / `b` slot occupancy,
            - and exposes that sanitized request through both `request_json()` and `side_snapshot()`.
        - [src/elitefurretai/rl/sync_battle_driver.py](src/elitefurretai/rl/sync_battle_driver.py) now preserves original switch numbering from `_request_index` even after the sanitized roster is reordered for move-phase coherence.
        - Focused regression coverage was added in [unit_tests/rl/test_rust_battle_engine.py](unit_tests/rl/test_rust_battle_engine.py) and [unit_tests/rl/test_sync_battle_driver.py](unit_tests/rl/test_sync_battle_driver.py).
    - Validation:
        - `pytest unit_tests/rl/test_rust_battle_engine.py unit_tests/rl/test_sync_battle_driver.py unit_tests/rl/test_fast_action_mask.py -q` passed, with the same one skipped trapped-scenario test as before.
    - Benchmark artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_front_pair_normalization_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=43`
        - `non_truncated_battles=157`
        - `duration_seconds=130.765`
        - `battles_per_second=1.529`
        - `non_truncated_battles_per_second=1.201`
        - `p1_rejected_choices=1634`
        - `p2_rejected_choices=1665`
        - `p1_fallback_recoveries=863`
        - `p2_fallback_recoveries=829`
        - `p1_unrecovered_rejections=771`
        - `p2_unrecovered_rejections=836`
    - Comparison versus the prior kept baseline (`rejection_diagnostics_final_kept_fix_200`):
        - raw throughput improved (`1.329 -> 1.529`)
        - non-truncated throughput improved (`1.023 -> 1.201`)
        - stall truncations improved (`46 -> 43`)
        - `p1` rejected choices improved (`1949 -> 1634`)
        - `p1` unrecovered rejections improved (`924 -> 771`)
    - Comparison versus the kept target-sign fix run (`rejection_diagnostics_target_sign_fix_200`):
        - raw throughput improved (`1.123 -> 1.529`)
        - non-truncated throughput improved (`0.955 -> 1.201`)
        - `p1` rejected choices improved (`2405 -> 1634`)
        - `p1` unrecovered rejections improved (`803 -> 771`)
        - stall truncations regressed slightly (`30 -> 43`), so the root cause is not fully solved
    - Updated interpretation:
        - this is the first change in this loop that improves both rejection counts and retained-throughput in the same benchmark shape, which strongly supports the adapter-boundary request-mismatch hypothesis
        - however, the remaining dominant rejection surface is still move-phase `move X, pass`, so front-pair normalization is a meaningful fix, not a complete fix
    - Revisit of earlier kept fixes after this result:
        - the force-switch legality fix still looks independently correct and should stay because it constrains request phases the adapter normalization does not change
        - the target-sign correction should also stay because it is protocol-faithful regardless of the request-shape bug
        - the older narrower active-entry reorder experiment remains superseded and should stay reverted because the new improvement came from normalizing the request consumed by legality generation, not from blindly promoting raw `active: true` roster entries

- 2026-04-09 20:15: Fixed the Python-side target-sign interpretation, reran diagnostics, and kept that fix.
    - Code change kept:
        - [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py) now treats poke-env / Showdown target signs correctly: positive values for foes, negative values for allies.
        - The sync-driver fast legality path now emits target strings consistent with the rest of the codebase and with poke-env.
        - Focused regression coverage was added in [unit_tests/rl/test_fast_action_mask.py](unit_tests/rl/test_fast_action_mask.py) and [unit_tests/rl/test_sync_battle_driver.py](unit_tests/rl/test_sync_battle_driver.py).
    - Benchmark artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_fix_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=30`
        - `non_truncated_battles=170`
        - `duration_seconds=178.047`
        - `battles_per_second=1.123`
        - `non_truncated_battles_per_second=0.955`
        - `p1_rejected_choices=2405`
        - `p1_fallback_recoveries=1602`
        - `p1_unrecovered_rejections=803`
    - Comparison versus the previous kept baseline:
        - stall truncations improved (`46 -> 30`)
        - unrecovered rejections improved (`924 -> 803`)
        - raw throughput regressed (`1.329 -> 1.123`)
        - non-truncated throughput also regressed (`1.023 -> 0.955`)
    - Interpretation:
        - this confirms the earlier protocol review mattered; the Python legality path really was wrong
        - however, fixing target signs did not remove the dominant rejected-choice loop, so it is not the main throughput bottleneck by itself

- 2026-04-09 20:15: The dominant remaining move rejections still point at malformed request structure rather than at target-sign semantics.
    - In the new kept diagnostic artifact, the largest remaining clusters are still `move, pass` with `12` or `11` legal choices.
    - Representative raw request shape from unrecovered `move 1, pass` events:
        - `len(active[]) == 2`
        - both `active[*].moves[*].target` values are blank strings rather than protocol target types like `normal` or `adjacentFoe`
        - `side.pokemon[*].active` still marks the two active mons in non-contiguous positions such as indices `0` and `2`
    - Why this matters:
        - even after the target fix, the request still does not present a clean poke-env-style pairing between `active[]` and the leading `side.pokemon` entries
        - this keeps the request-corruption / adapter-boundary hypothesis alive, but with a more specific shape: blank move-target metadata plus roster ordering that does not match poke-env's expectations

- 2026-04-09 22:39: Captured a concrete protocol-history review corpus for manual inspection.
    - New example note:
        - `planning/stage2/2026-04-09-22-39-rust-protocol-corruption-examples.md`
    - New diagnostic artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_protocol_examples_80.jsonl`
    - Short-run summary:
        - `completed_battles=80`
        - `truncated_battles=41`
        - `stalled_limit_truncations=41`
        - `p1_rejected_choices=1009`
        - `p2_rejected_choices=1177`
    - The four documented examples all share the same shape:
        - move request with `len(active[]) == 2`
        - blank move target metadata for both actives
        - non-contiguous `side.pokemon[*].active` indices like `[2, 5]`, `[0, 4]`, or `[0, 2]`
        - protocol history that shows the malformed request can happen both early and late in the battle, not only after long faint chains
    - Why this helps:
        - we now have a stable hand-inspection corpus that can be compared before and after any adapter-boundary sanitization change

- 2026-04-09 20:15: Tried a narrower active-entry reorder repair and explicitly reverted it.
    - Experimental idea:
        - in [src/elitefurretai/rl/rust_battle_engine.py](src/elitefurretai/rl/rust_battle_engine.py), after move-ID and protocol-slot matching, promote any remaining `active: true` side entries ahead of the bench before `parse_request()`.
    - Why it seemed promising:
        - the post-target-fix diagnostics still showed raw requests with active mons at positions like `0` and `2`, so making active entries contiguous looked like a plausible local repair.
    - Benchmark artifact from the experiment:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_reorder_fix_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=44`
        - `non_truncated_battles=156`
        - `duration_seconds=182.913`
        - `battles_per_second=1.093`
        - `non_truncated_battles_per_second=0.853`
        - `p1_rejected_choices=2860`
        - `p1_fallback_recoveries=1850`
        - `p1_unrecovered_rejections=1010`
    - Outcome:
        - the experiment made both rejection counts and retained-throughput worse, so it was reverted
    - Updated interpretation:
        - the remaining problem is not fixed by blindly forcing all `active: true` roster entries to the front
        - any future adapter repair needs more context than raw `active` flags alone, likely incorporating protocol-tracked slot identity and more explicit request invariants

- 2026-04-09 19:05: Re-checked the Showdown decision protocol and poke-env parsing assumptions before blaming the Rust engine.
    - Confirmed protocol points from Showdown:
        - in doubles, choices are comma-delimited and explicit `pass` is valid for slots that do not need a decision
        - move targets use positive numbers for foes and negative numbers for allies
        - `REQUEST.active` is the per-active decision payload, while `REQUEST.side` is team-wide state
    - Confirmed poke-env assumption from `DoubleBattle.parse_request`:
        - it directly pairs `request["active"][i]` with `request["side"]["pokemon"][i]`
        - therefore poke-env expects the leading `side.pokemon` entries to line up with the active request entries
    - Important correction about our own code:
        - the current fast action mask and sync-driver legality helpers interpret target signs in the opposite direction from Showdown and poke-env
        - our code currently treats negative targets as opponents and positive targets as allies, but Showdown uses positive targets for foes and negative targets for allies
    - Consequence:
        - we have not yet proved that the Rust battle engine itself is broken
        - what we have proved is that the current Python integration layer is not protocol-faithful in at least one important way, and that this alone can generate invalid move requests and misleading rejection diagnostics
    - Updated interpretation:
        - the force-switch / `side.pokemon` mismatches are still suspicious and still worth investigating
        - however, before asserting that the Rust engine emits corrupted requests, we should first fix the Python-side target-sign interpretation and then re-run the same diagnostics

- 2026-04-09 18:45: Kept-state rejection clusters versus experimental-state clusters.
    - Kept diagnostic artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_final_kept_fix_200.jsonl`
    - Experimental diagnostic artifact from the reverted target-inference pass:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_core_fix2_100.jsonl`
    - Aggregate shape change:
        - kept state was dominated by `move, pass` and `switch, pass` / `pass, switch`
        - experimental state was dominated by `move, move`
    - Interpretation:
        - the local fixes changed the rejection surface, but did not solve the underlying state mismatch

- 2026-04-09 18:45: Representative corrupted-request examples worth inspecting first.
    - Example A: early move request with two `active[]` entries but only one front-slot `active` flag.
        - Source artifact:
            - `rejection_diagnostics_final_kept_fix_200.jsonl`
        - Representative event:
            - battle tag `rust-sync-0`, side `p1`, turn `2`
            - rejected choice: `move 2 terastallize, pass`
            - legal preview began with only `move X, pass` options
            - request shape summary:
                - `active[]` length was `2`
                - `side.pokemon` still marked only slot 1 and slot 4 as active, while slot 2 was healthy but inactive
        - Why it matters:
            - this is the cleanest symptom that the request exposes two acting mons while raw side ordering and active flags disagree about who is actually on the field

    - Example B: single force-switch request with only one visible active mon and over-approximated switch targets.
        - Source artifact:
            - `rejection_diagnostics_final_kept_fix_200.jsonl`
        - Representative event:
            - battle tag `rust-sync-1`, side `p2`, turn `4`
            - rejected choice: `pass, switch 3`
            - fallback: `pass, switch 1`
            - `forceSwitch=[false, true]`
            - legal preview included `pass, switch 1`, `pass, switch 3`, `pass, switch 5`, `pass, switch 6`
            - request-side active flags only showed one active mon: Ogerpon
        - Why it matters:
            - this suggests the request-side `side.pokemon` list is not a reliable source of which non-fainted mons are actually legal replacement targets for that force-switch phase

    - Example C: experimental move-phase state where `move, move` still rejects after patching the earlier `move, pass` symptom.
        - Source artifact:
            - `rejection_diagnostics_core_fix2_100.jsonl`
        - Representative event:
            - battle tag `rust-sync-4`, side `p1`, turn `11`
            - rejected choice: `move 3 -2, move 3 -1 terastallize`
            - fallback: `move 1 -2, move 1 -2`
            - front request entries contained acting move lists, but `side.pokemon` front slots still included fainted mons in the leading positions
        - Why it matters:
            - after the local patches, the dominant remaining failures were not pass-related anymore; they were full move-pair failures caused by deeper disagreement between `active[]`, `side.pokemon`, and protocol-tracked state

- 2026-04-09 18:45: Coarse counts for when corruption shows up.
    - From the kept-state JSONL:
        - `double_force_switch` rejection events: `967`
        - `switch_active_flags_lt_forced_slots`: `967`
        - move rejections where the front side entries already included a `fnt` condition: `823`
        - move rejections with `len(active[]) == 2` but fewer than two `active=true` side entries: `46`
    - From the reverted experimental-state JSONL:
        - move rejections where the front side entries included a `fnt` condition: `2751`
        - move rejections with `len(active[]) == 2` but fewer than two `active=true` side entries: `1456`
    - Interpretation:
        - corruption is not confined to one exotic corner case
        - the clearest clusters are:
            - post-KO or post-replacement phases involving single or double force-switch
            - move phases where a mon is acting according to `active[]` but the paired `side.pokemon` front entries still describe stale or fainted mons

- 2026-04-09 18:45: Battle situations where the Rust request appears to be corrupted.
    - Situation 1: immediately after early-game switches or replacements.
        - The kept examples often appear on turn `2` to `4`, which means this is not only a late-game exhaustion issue.
    - Situation 2: single force-switch requests.
        - The request may expose only one current active flag while also implying a forced replacement for the missing slot.
    - Situation 3: double force-switch requests.
        - The request frequently has fewer visible active flags than forced slots, which makes raw request-side active ordering unreliable.
    - Situation 4: move phases after prior faint or replace churn.
        - These are the most concerning cases because they can still surface after local legality fixes, implying the front-slot request state itself is stale.

- 2026-04-09 18:45: Recommended debugging path from here.
    - First, do not start with another broad legality patch in the sync driver.
        - The evidence now points to a request consistency problem at the adapter boundary.
    - Second, create a small filtered artifact for human review.
        - Preferred contents per line:
            - battle tag
            - side
            - turn
            - request type
            - rejected choice
            - fallback choice
            - `forceSwitch`
            - `active[]` summary
            - front `side.pokemon` summary
            - protocol-tracked active slots
            - last 5 protocol lines for that side
        - This should be emitted only for the first corrupted step per battle tag and corruption class so the output stays readable.
    - Third, add an adapter-side invariant checker.
        - Suggested invariants:
            - if `len(active[]) == 2` on a move request, there should be two coherent front-slot identities after sanitization
            - if a slot is acting according to `active[]`, the sanitized front-slot entry must not still say `0 fnt`
            - if `forceSwitch` requires `k` slots, the sanitized request should expose exactly `k` missing or forced front slots in a coherent way
    - Fourth, fix at the adapter boundary.
        - Most likely fix:
            - rebuild the leading `side.pokemon` entries and front-slot active flags from protocol-tracked slot occupancy plus `active[]`
            - treat raw `side.pokemon` as a backing roster, not as the trusted source for current front-slot state during corrupted turns
    - Fifth, only keep the fix if it improves the current kept baseline:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_final_kept_fix_200.txt`
        - target metrics to improve:
            - fewer unrecovered rejections
            - fewer stall-limit truncations
            - higher `non_truncated_battles_per_second`
