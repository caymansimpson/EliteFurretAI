# Rust Battle Readout Spec

## Context
We now have a human-readable Rust error-battle document at `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md` that is generated from per-battle trace records rather than from rejection-only JSONL events.

This spec exists so future AI sessions can regenerate that document deterministically from the source artifacts without reverse-engineering the trace layout again.

## Before State
Before this spec existed:
- there were multiple generations of the readable document
- older versions were built from rejected-choice JSONL events only
- those older versions could not render full chronological request steps or teampreview
- the latest version uses p1-only per-battle trace records with a synthetic teampreview compatibility layer in the Python Rust driver

## Problem
Future sessions need an exact description of:
1. which source artifacts to read
2. what JSON schema those trace records use
3. how to render the readable markdown document from them
4. which parts are native Rust behavior versus Python compatibility-layer behavior

Without that, a later session may regenerate the wrong document shape, accidentally include p2 again, or fail to understand why teampreview appears in the trace despite the native binding not exposing it.

## Solution
Define the source artifacts, record schema, rendering rules, caveats for the current p1-only Rust battle readout, and the checked-in command used to regenerate the markdown document.

## Reasoning
The important thing to preserve is not just the final markdown file, but the contract between:
- the trace recorder in `src/elitefurretai/engine/sync_battle_driver.py`
- the per-battle JSON files written under `error_battle_record_path`
- the renderer that turns those records into the readable markdown doc

This lets future sessions regenerate the doc, audit the trace data directly, or intentionally change the layout without conflating source-data changes with presentation changes.

## Renderer Implementation

The canonical renderer now lives at:
- `src/elitefurretai/scripts/render_error_battle_readable.py`

Use the repository's normal venv activation pattern before running it:

```bash
source ../venv/bin/activate && /home/cayman/Repositories/venv/bin/python \
  src/elitefurretai/scripts/render_error_battle_readable.py \
  --records-dir data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp \
  --summary-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.txt \
  --output-path data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md \
  --limit 10 \
  --side p1
```

This replaces the earlier one-off inline terminal snippet that produced an inconsistent markdown layout.

Optional filtering flags now supported by the renderer:
- `--max-first-rejection-turn N`: only include battles whose first recorded rejection happens on or before turn `N`
- `--stop-after-first-rejection`: stop rendering a battle immediately after the first rejected decision window
- `--rejection-jsonl-path <path>`: use the rejection diagnostics JSONL as the source of first-rejection turn selection when the trace files themselves do not preserve rejection result metadata

Example for the early-rejection view:

```bash
source ../venv/bin/activate && /home/cayman/Repositories/venv/bin/python \
  src/elitefurretai/scripts/render_error_battle_readable.py \
  --records-dir data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp \
  --summary-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.txt \
  --output-path data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md \
  --limit 10 \
  --side p1 \
  --max-first-rejection-turn 5 \
  --rejection-jsonl-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.jsonl \
  --stop-after-first-rejection
```

## Source Artifacts

### Current p1-only teampreview-enabled source set
- Trace directory:
  `data/benchmarks/training_throughput_2026_04_09/error_battle_records_trace_400_p1_tp`
- Run summary:
  `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.txt`
- Rendered markdown output:
  `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md`

### Older source sets
- `error_battle_records_trace_400`:
  full trace records, but no explicit teampreview step because the native Rust binding only exposed the first move request
- `rejection_diagnostics_doc_source_400.jsonl`:
  rejection-event stream, not suitable for the current chronological request-step document

## Native vs Compatibility Behavior

### Native Rust binding behavior
The native `pokemon_showdown_py.RustBattle` object currently:
- exposes the initial request as `move`
- does not expose `teamPreview` in its first request
- rejects `team ....` choices if submitted directly before the first move request

### Python compatibility-layer behavior
`src/elitefurretai/engine/sync_battle_driver.py` now inserts a synthetic teampreview stage before constructing the Rust battle:
- it builds a poke-env teampreview battle for each side
- it samples or selects a legal `team ....` choice for each side
- it reorders the six-mon team lists according to those choices
- it passes the reordered teams into `RustBattle`
- it records the teampreview request and chosen `team ....` action as the first trace step

Because of this, the current readable document can include teampreview even though the native binding cannot.

## Per-Battle Record Schema

Each file in `error_battle_records_trace_400_p1_tp/` is a single JSON object with these top-level keys:
- `battle_tag`: string
- `format_id`: string
- `completed`: boolean
- `truncated`: boolean
- `winner`: string or null
- `p1_won`: boolean
- `final_turn`: integer
- `final_request_types`: object with `p1` and `p2`
- `final_requests`: object with `p1` and `p2`
- `trace`: array of chronological trace events
- `protocol_log`: object with `p1` and `p2`, each a full array of protocol log entries

### `protocol_log` entry format
Each protocol-log entry is either:
- `{ "raw": <string>, "normalized": <string> }`
or, in degenerate cases, a plain string

For rendering:
- if the raw and normalized sequences for a slice are identical, print the shared lines once
- otherwise print the raw sequence first and the normalized sequence afterwards
- indent the actual protocol lines by two spaces within each section so the protocol text starts in the same column

## Trace Event Schema

The `trace` array alternates between:
- `decision_window`
- `decision_result`

The renderer should treat them as pairs in order.

### `decision_window`
Important keys:
- `event_type`: always `decision_window`
- `turn_before`: integer
- `stalled_steps_before`: integer
- `p1`: side-state object
- `p2`: side-state object
- `submitted_choices`: object with `p1` and `p2`
- `fallback_choices`: object with `p1` and `p2`

### Side-state object (`p1` or `p2`)
Important keys:
- `battle_tag`: string
- `side`: `p1` or `p2`
- `request_type`: `teamPreview`, `move`, `switch`, `wait`, or null-ish
- `sanitized_request`: the request dict used by Python
- `raw_request`: raw binding request text or null
- `legal_choice_count`: integer
- `legal_choices`: full list of legal choice strings
- `battle_state_poke_env`: rendered battle-state string
- `battle_state_request`: rendered request-state string
- `protocol_log_length_before`: integer offset into `record["protocol_log"][side]`
- `protocol_history`: short recent binding snapshot messages

### `decision_result`
Important keys:
- `event_type`: always `decision_result`
- `turn_before`: integer
- `turn_after`: integer
- `initial_acceptance`: object with `p1` and `p2`
- `final_acceptance`: object with `p1` and `p2`
- `fallback_recovered`: object with `p1` and `p2`
- `protocol_log_lengths`: object with `p1` and `p2`

## Current Markdown Rendering Rules

### Record selection
- Read the first 10 files from `error_battle_records_trace_400_p1_tp/` using path-sorted order.
- Render only p1's perspective.
- Optional mode: filter to records whose first rejection turn is at or below a caller-specified threshold.

### Battle header
For each record, render:
- `## Battle N: <battle_tag>`
- `Winner: ...`
- `Truncated: ...`
- `Final turn: ...`

### Step selection
Iterate through `trace` in order.
For each `decision_window`, pair it with the next `decision_result` if present.
Only render the `p1` side-state from the window.

Optional truncation mode:
- if `--stop-after-first-rejection` is enabled, stop rendering that battle immediately after the first `decision_window` whose paired `decision_result.initial_acceptance` contains a rejection for either side.

### Step header
Render:
- `### Step X (turn <turn_before>, <request_type>)`

### Logs Until Request
Use `protocol_log.p1` plus the side-state's `protocol_log_length_before`.

Algorithm:
1. Keep a running `previous_log_index`, initially `0`.
2. Read `before_len = p1.protocol_log_length_before`.
3. Slice `protocol_log.p1[previous_log_index:before_len]`.
4. Render that slice as the step's `Logs Until Request` block.
5. Set `previous_log_index = before_len`.

This produces non-overlapping cumulative log segments for p1.

### Log rendering layout
Render the log slice in chronological order, but group it by representation:
1. If the raw and normalized sequences are identical, render the shared lines once.
2. Otherwise render a `raw:` subsection first.
3. Then render a `normalized:` subsection afterwards.
4. Within each subsection, indent every log line by two spaces so the actual protocol text starts in the same column.

This is intentional: it makes it easier to scan the raw protocol first and compare the normalized version afterwards, instead of interleaving `raw` and `normalized` line by line.

Example shape:

```text
raw:
  |switch|p2: Dondozo|dondozo, L50, M|225/225
  |move|1a|Protect|1a
normalized:
  |switch|p2a: Dondozo|dondozo, L50, M|225/225
  |move|p1a: Dondozo|Protect|p1a: Dondozo
```

### Battle State block
Render `p1.battle_state_poke_env` verbatim inside a `text` fenced block.

### Request block
Render `p1.battle_state_request` verbatim inside a `text` fenced block.

### Legal Choices block
Default rule:
- render every entry from `p1.legal_choices` on its own line

Provenance note:
- these legal choices are generated by the Python sync driver from the synchronized poke-env-style battle state plus the sanitized request
- they are not emitted as a native legal-action list by the Rust engine itself

Current exception for teampreview:
- do not spell out all 90 teampreview choices in the markdown
- instead render exactly:
  - `Omitted here because the teampreview choice set is global and unchanged across battles.`
  - `Total teampreview legal choices: 90`
  - `Reference examples: team 1234, team 4615, team 5634`

### Choice line
Use `decision_window.submitted_choices.p1` and `decision_result.initial_acceptance.p1`.

Render format:
- `Choice: <choice> | rejected=<true_or_false> | fallback_recovered=<true_or_false>`

Where:
- `rejected = not decision_result.initial_acceptance.p1`
- `fallback_recovered = decision_result.fallback_recovered.p1`

If there is no paired `decision_result`, omit the derived flags and print only `Choice: <choice>`.

## Teampreview Formatting Rules

The current teampreview battle-state string is intentionally different from move/switch states.

For teampreview, `battle_state_poke_env` should show:
- battle tag and turn
- perspective line
- `My Teampreview Team:`
- six numbered preview slots
- `Opp Teampreview Team:`
- six numbered preview slots

For teampreview, `battle_state_request` should show:
- `Request type: teamPreview`
- `wait=False teamPreview=True forceSwitch=None`
- empty `Active payloads:` section
- `Side pokemon:` entries with `ident`, `active=False`, and `_request_index`

## Reproduction Checklist

To regenerate the current doc in a future session:
1. Use the teampreview-enabled trace directory `error_battle_records_trace_400_p1_tp`.
2. Sort the JSON files and take the first 10.
3. Render only `p1` from each `decision_window` / `decision_result` pair.
4. Use cumulative `protocol_log_length_before` slicing for `Logs Until Request`.
5. Render every recorded `decision_window`; do not skip intermediate turns when consecutive move windows differ only by a small protocol slice.
6. Render log slices as either shared lines or grouped `raw:` then `normalized:` sections as described above.
7. Render battle state and request strings verbatim.
8. Collapse teampreview legal choices into the short three-line summary above.
9. Write the result to `data/benchmarks/training_throughput_2026_04_09/error_battles_top_10_readable.md`.

For the early-rejection-first-pass view requested during debugging:
1. Add `--max-first-rejection-turn 5`.
2. Add `--stop-after-first-rejection`.
3. Add `--rejection-jsonl-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_doc_source_400_p1_tp.jsonl` if the directory-based trace files do not contain usable rejection result metadata.
4. Expect fewer than 10 battles if fewer than 10 trace records satisfy that filter.

## Renderer Audit Note

An earlier markdown generation pass incorrectly jumped from the turn 1 request window to the turn 3 request window in `rust-sync-0`.

That was a renderer/documentation bug, not a native Rust trace omission.

The source trace file `error_battle_records_trace_400_p1_tp/rust-sync-0.json` contains a real p1 `decision_window` for turn 2 with:
- `turn_before = 2`
- `request_type = move`
- `protocol_log_length_before = 33`

If a future regenerated document appears to skip that window again, the renderer logic is wrong.

## Planned Next Steps
1. Reconcile the synthetic teampreview compatibility layer with native Rust opening-lineup behavior so the compatibility path no longer regresses benchmark quality.
2. If the native binding is updated later to expose true teampreview requests, remove the compatibility note and regenerate the doc from native trace data.
3. If future readers want a rawer debugging view, add an optional rendering mode that spells out all 90 teampreview choices instead of collapsing them.