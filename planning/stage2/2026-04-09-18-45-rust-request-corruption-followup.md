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

- 2026-04-10 12:50: Fixed the Tatsugiri commander legality leak in the Showdown websocket path.
    - Problem confirmed:
        - the legacy Showdown backend still produced invalid choices of the form `Can't move: Your Dondozo doesn't have a move matching dracometeor` even after the Rust sync-driver commander fix.
        - root cause was path split: the websocket backend uses [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py) and heuristic/player logic in [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py), so the Rust sync-driver fix could not affect it.
    - Code changes kept:
        - [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py) now exposes commander detection through battle-state `Effect.COMMANDER` plus request fallbacks, and commanding slots are forced to `pass` only.
        - [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py) now short-circuits commanding slots to `PassBattleOrder()` and suppresses heuristic move scoring for those slots.
        - [unit_tests/rl/test_fast_action_mask.py](unit_tests/rl/test_fast_action_mask.py) now includes websocket-side commander regressions covering commander detection, per-slot legality, and full action-pair masking.
    - Validation:
        - `/home/cayman/Repositories/venv/bin/python -m pytest unit_tests/rl/test_fast_action_mask.py -q` passed.
        - post-fix Showdown smoke artifact: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_post_commander_fix_smoke.txt`
        - result: the old Dondozo/Draco Meteor commander error no longer appears.
    - Important remaining limitation:
        - the Showdown backend is not rejection-free yet.
        - dominant remaining invalid choices in the longer three-update run were:
            - `180` x `Can't move: Your Terapagos doesn't have a move matching terastarstorm`
            - forced-switch `Can't pass` errors on several mons
            - occasional targeting issues like `Protect` target selection and `Behemoth Bash needs a target`
        - so the commander leak was real and fixed, but it was not the only remaining websocket legality mismatch.

- 2026-04-10 13:50: Investigated dynamic target-shape edge cases for `Expanding Force` and `Tera Starstorm`, then patched the shared legality resolver plus the Showdown heuristic path.
    - Rust-side probe findings:
        - direct binding requests from `pokemon_showdown_py.RustBattle` expose blank move targets (`target: ""`) for both `Tera Starstorm` and `Expanding Force`, even in states where the move is mechanically spread.
        - the Rust binding accepts index-based side-choice strings for `Tera Starstorm` before and during tera activation, including:
            - `move 1 terastallize, move 1 1`
            - `move 1 terastallize 1, move 1 1`
            - `move 1 1 terastallize, move 1 1`
        - for `Expanding Force` under Psychic Terrain, the Rust binding accepted every tested target suffix form in the probe (`no target`, opponent targets, and even ally targets), which means the raw Rust command layer is effectively target-agnostic there.
        - practical consequence:
            - the Rust path cannot rely on request `target` metadata for these transformed moves and must infer spread-vs-single behavior from battle state.
    - Showdown-side probe findings:
        - the real Showdown websocket request for `Tera Starstorm` starts as `target="normal"` before Stellar activation and flips to `target="allAdjacentFoes"` after Stellar is active.
        - this confirmed that the websocket path should follow request metadata when it exists, but the Rust path still needs a state-derived fallback.
    - Code changes kept:
        - [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py)
            - added battle-state-aware target overrides for:
                - `terastarstorm` when the user is Stellar-terastallized / effectively Stellar
                - `expandingforce` when Psychic Terrain is active and the user is grounded
            - exported request-aware target resolution so both masking and other legality generators use the same rule.
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py)
            - now uses the shared request-aware target resolver for Rust-side legal choice generation.
        - [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py)
            - heuristic action scoring now filters `battle.available_moves` against the current request move list before scoring.
            - this removes stale move candidates that survived in poke-env state after the request had already narrowed legal options.
            - when a request move exists, heuristic target generation now follows the shared request-aware resolver instead of relying only on poke-env's static move target metadata.
        - new regression tests:
            - [unit_tests/rl/test_fast_action_mask.py](unit_tests/rl/test_fast_action_mask.py)
            - [unit_tests/rl/test_players.py](unit_tests/rl/test_players.py)
    - Validation:
        - `/home/cayman/Repositories/venv/bin/python -m pytest unit_tests/rl/test_fast_action_mask.py unit_tests/rl/test_players.py -q` passed.
        - Showdown smoke artifact:
            - `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_post_target_fix_smoke.txt`
        - result:
            - the targeted transformed-move signatures were `0` in the smoke:
                - `Can't move: Your Terapagos doesn't have a move matching terastarstorm`
                - `Can't move: Your .*expandingforce`
    - Remaining limitation after the fix:
        - the websocket backend still has unrelated request/move mismatches.
        - in the new longer run the dominant remaining invalid-choice families became:
            - `172` x `Can't move: Your Farigiraf doesn't have a move matching psychic`
            - `112` x `Can't move: Your Dragapult doesn't have a move matching dragondarts`
        - so the transformed-target fixes are real and kept, but the next websocket cleanup pass should focus on stale or rewritten move ids, not target shape.

- 2026-04-10 12:10: Generated a concise post-fix rejection review corpus for manual legality inspection.
    - Artifact:
        - `data/benchmarks/training_throughput_2026_04_10/error_battles_first_rejection_commander_fix_v3.md`
    - Source set:
        - trace directory: `data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp_commander_fix_v3`
        - summary: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp_commander_fix_v3.txt`
    - Rendering mode:
        - p1-only
        - no first-turn filter
        - `--stop-after-first-rejection` enabled so each battle ends at its first rejected decision window
    - Result:
        - `Distinct battle tags included: 8`
        - battle tags included: `rust-sync-14`, `rust-sync-140`, `rust-sync-159`, `rust-sync-199`, `rust-sync-200`, `rust-sync-323`, `rust-sync-348`, `rust-sync-369`
    - Why this corpus exists:
        - the earlier first-3-turn view is now empty after the commander fix because the earliest first rejection is turn 16.
        - this new artifact gives a compact review set from the same post-fix run without requiring another long benchmark just to recover a few remaining legality mismatches.

- 2026-04-10 11:40: Fixed commander legality in the Python sync-driver action generator.
    - Problem confirmed:
        - the readable battle doc exposed that when Tatsugiri was commanding Dondozo, the `Legal Choices` section still listed Tatsugiri move and terastallize actions.
        - in this state Tatsugiri should be pass-only, so the Python legality generator was over-admitting actions.
    - Code change kept:
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now returns only `pass` for an active slot whose sanitized request-side pokemon entry has `commanding=True`.
        - this keeps commander behavior aligned with battle mechanics even when the synchronized request still exposes move payloads for the commanded slot.
        - [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) now includes a regression test covering a commanding Tatsugiri slot and asserting that its only legal action is `pass`.
    - Validation:
        - `source ../venv/bin/activate && pytest unit_tests/engine/test_sync_battle_driver.py -q` passed.
    - Documentation update:
        - the readable renderer now explicitly notes that `Legal Choices` are generated by the Python sync driver from synchronized poke-env state and sanitized requests, not emitted directly by Rust.
    - Fresh 400-battle benchmark after the fix:
        - summary artifact: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp_commander_fix_v3.txt`
        - rejection stream: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp_commander_fix_v3.jsonl`
        - trace directory: `data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp_commander_fix_v3`
        - `completed_battles=400`
        - `truncated_battles=4`
        - `non_truncated_battles=396`
        - `p1_rejected_choices=108`
        - `p2_rejected_choices=104`
        - `stalled_limit_truncations=4`
    - Comparison versus the pre-fix teampreview compatibility benchmark (`rejection_diagnostics_doc_source_400_p1_tp.txt`):
        - `p1_rejected_choices` improved from `8425 -> 108` (`-98.72%`)
        - `p2_rejected_choices` improved from `6277 -> 104` (`-98.34%`)
        - `truncated_battles` improved from `248 -> 4` (`-98.39%`)
        - `stalled_limit_truncations` improved from `241 -> 4` (`-98.34%`)
        - `non_truncated_battles` improved from `152 -> 396` (`+160.53%`)
    - Early-rejection implication:
        - the earliest first rejection in the fresh JSONL is now turn `16`
        - there are no battles whose first rejection occurs within the first `3` turns in this post-fix run
        - the first-3-turn readable artifact therefore renders zero matching battle tags, which is the expected result rather than a renderer failure

- 2026-04-10 08:20: Added human-readable rejected-choice state dumps and generated a 10-battle review document.
    - Code change kept:
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now writes richer rejected-choice diagnostics containing:
            - full `legal_choices`
            - `battle_state_poke_env`
            - `battle_state_request`
            - `protocol_log_before_step`
            - `protocol_log_after_step`
        - the battle-state formatter was hardened so lightweight test doubles used in RL unit tests do not crash the diagnostic path.
        - the human-readable state formatting was narrowed for manual review:
            - battle-state active and team listings now show only `active` and `fainted`
            - request active-payload moves now show only move id plus target
            - request side-pokemon entries now omit condition, details, and moves
    - Validation:
        - `pytest unit_tests/engine/test_sync_battle_driver.py unit_tests/engine/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed, with the same one skipped trapped-scenario test as before.
    - Readable diagnostics source artifacts:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400.jsonl`
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400.txt`
    - 400-battle source-run result:
        - `completed_battles=400`
        - `truncated_battles=10`
        - `non_truncated_battles=390`
        - `duration_seconds=395.923`
        - `battles_per_second=1.010`
        - `non_truncated_battles_per_second=0.985`
        - `p1_rejected_choices=262`
        - `p2_rejected_choices=256`
        - `stalled_limit_truncations=8`
    - Human-readable review artifact:
        - `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md`
        - built from the first 10 battle tags in the readable JSONL that produced rejected-choice events
        - for each rejected step, the document includes the full protocol-log prefix before the rejected choice, the protocol-log suffix emitted after that step, poke-env battle state, request-defined battle state, the rejected choice, and the full legal choice list
        - the regenerated document currently includes `10` distinct battle tags and `233` rejected steps
    - Interpretation note:
        - this rerun was still generated to guarantee at least 10 distinct error battles for manual review, but unlike the first document-source attempt it remained consistent with the cleaner post-fix benchmark behavior instead of collapsing into a stall-heavy corpus.

- 2026-04-10 09:05: Switched the human-readable review doc from rejection-event JSONL to full trace records so each step can be rendered in chronological request order.
    - Code change kept:
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now enriches per-step trace-side state with:
            - full rendered battle state
            - full rendered request state
            - full legal choice list
            - the protocol-log length at the moment the request was issued
        - the same driver now records multiple error battles when `error_battle_record_path` is a directory, writing one JSON file per selected error battle instead of only recording the first error battle to a single file.
        - [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) gained coverage for directory-based multi-battle trace recording.
    - Validation:
        - `pytest unit_tests/engine/test_sync_battle_driver.py unit_tests/engine/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed, with the same one skipped trapped-scenario test as before.
    - Trace-record source artifacts:
        - directory: `data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400`
        - summary: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400.txt`
    - 400-battle trace-run result:
        - `completed_battles=400`
        - `truncated_battles=8`
        - `non_truncated_battles=392`
        - `duration_seconds=509.796`
        - `battles_per_second=0.785`
        - `non_truncated_battles_per_second=0.769`
        - `p1_rejected_choices=177`
        - `p2_rejected_choices=130`
    - Regenerated review artifact:
        - `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md`
        - structure is now one chronological request step at a time:
            - logs until request
            - battle state
            - request
            - legal choices
            - submitted choice and whether it was rejected
    - Important backend finding:
        - probing the Rust backend directly showed that its initial request is currently exposed to Python as `move`, not `teamPreview`.
        - because of that, the regenerated document cannot include explicit teampreview request sections for this backend path yet; the traces start at the first move request window instead.

- 2026-04-10 09:45: Added a synthetic teampreview compatibility layer to the Rust sync driver and regenerated the doc from p1 only.
    - Scope of the fix:
        - the native `pokemon_showdown_py.RustBattle` object still exposes its initial request as `move` and rejects `team ...` choices directly.
        - to surface teampreview in the RL path we control, [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now:
            - constructs a synthetic poke-env teampreview battle for each side before instantiating the Rust battle
            - samples or selects a real `team ....` choice for each side using the existing teampreview policy path
            - reorders the six-mon rosters according to those teampreview choices before passing them into `RustBattle`
            - records that teampreview request/choice pair as the first trace step in each error-battle record
        - the battle-state renderer now prints teampreview rosters when `battle.teampreview` is true.
    - Direct probe result for the native binding:
        - initial request keys are `['active', 'noCancel', 'side']`
        - `teamPreview` is absent
        - `team 1234` is rejected immediately while `move ...` is accepted
        - this confirms the teampreview exposure bug exists below the Python adapter layer.
    - Validation:
        - `pytest unit_tests/engine/test_sync_battle_driver.py unit_tests/engine/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed after landing the compatibility layer.
    - New teampreview-enabled source artifacts:
        - trace directory: `data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp`
        - summary: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.txt`
        - regenerated p1-only doc: `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md`
    - 400-battle result for the compatibility-layer run:
        - `completed_battles=400`
        - `truncated_battles=248`
        - `non_truncated_battles=152`
        - `p1_rejected_choices=8425`
        - `p2_rejected_choices=6277`
    - Interpretation:
        - the compatibility layer succeeds at surfacing the first teampreview request and the players' `team ...` choices for inspection.
        - it is not yet a clean behavioral replacement for the old path, because this first implementation materially regressed benchmark quality and likely still disagrees with native Rust battle initialization in some cases.
    - Canonical rendering spec:
        - future sessions should use `planning/stage2/2026-04-10-10-15-rust-battle-readout-spec.md` as the source of truth for the trace-record schema and the p1-only markdown rendering rules.

- 2026-04-10 00:55: Kept the combined adapter reconciliation and force-switch legality fix after it materially improved both the short diagnostic run and the longer model-backed benchmark.
    - Root cause confirmed from the diagnostics:
        - `DoubleBattle.parse_request()` does not reliably overwrite `battle._active_pokemon` when stale slot mappings already exist, so even a sanitized request can leave poke-env believing one front slot is missing or mapped to the wrong mon.
        - once that stale active-slot state was corrected, the dominant remaining rejection family shifted almost entirely into force-switch windows, where the sync driver was still admitting `pass` even on double replacement requests with enough healthy replacement targets.
    - Code changes kept:
        - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py) now reconciles `battle._active_pokemon` directly from the sanitized request front pair immediately after request application, including the fallback path used when `parse_request()` rejects an edge-case shape.
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now only admits `pass` during force-switch handling when the number of healthy replacement targets is smaller than the number of forced slots. This removes the invalid `pass, switch X` and `switch X, pass` choices that were still being generated on `forceSwitch=[True, True]` requests with multiple legal replacements.
        - [unit_tests/engine/test_rust_battle_engine.py](unit_tests/engine/test_rust_battle_engine.py) gained a regression test for active-slot reconciliation from sanitized request front-pair state.
        - [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) gained a regression test covering double force-switch requests with enough replacements, asserting that no spurious `pass` options are offered.
    - Validation:
        - `pytest unit_tests/engine/test_sync_battle_driver.py unit_tests/engine/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed after the kept fix landed, with the same one skipped trapped-scenario test as before.
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
        - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py) now preserves the full raw and normalized protocol transcript per side in addition to the bounded recent protocol history.
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now supports an optional `error_battle_record_path` that records the first battle to hit a rejected choice.
        - [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py) now exposes that recorder via `--error-battle-record-path`.
        - [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) gained focused coverage for the new recording path.
    - Recorder contents:
        - final request types and final sanitized requests for both sides
        - a per-step trace with request snapshots, legal-choice previews, submitted choices, fallback choices, and acceptance results
        - the full raw and normalized protocol transcript for both sides
    - Validation:
        - `pytest unit_tests/engine/test_sync_battle_driver.py unit_tests/engine/test_rust_battle_engine.py unit_tests/rl/test_fast_action_mask.py -q` passed after landing the recorder.
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
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) was temporarily changed so move-slot availability could be inferred from the sanitized request front pair instead of only from poke-env active-slot state.
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
        - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py) now sanitizes the cached Rust request itself, not just the poke-env ingestion path.
        - For move requests, the adapter now:
            - annotates each roster entry with its original request index,
            - matches `active[]` entries back to roster mons by move IDs,
            - reorders both `active[]` and the leading `side.pokemon` entries from protocol-tracked `a` / `b` slot occupancy,
            - and exposes that sanitized request through both `request_json()` and `side_snapshot()`.
        - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) now preserves original switch numbering from `_request_index` even after the sanitized roster is reordered for move-phase coherence.
        - Focused regression coverage was added in [unit_tests/engine/test_rust_battle_engine.py](unit_tests/engine/test_rust_battle_engine.py) and [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py).
    - Validation:
        - `pytest unit_tests/engine/test_rust_battle_engine.py unit_tests/engine/test_sync_battle_driver.py unit_tests/rl/test_fast_action_mask.py -q` passed, with the same one skipped trapped-scenario test as before.
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
        - Focused regression coverage was added in [unit_tests/rl/test_fast_action_mask.py](unit_tests/rl/test_fast_action_mask.py) and [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py).
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
        - in [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py), after move-ID and protocol-slot matching, promote any remaining `active: true` side entries ahead of the bench before `parse_request()`.
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
