# Rust Protocol Corruption Examples

## Context
This note captures a small review corpus from the fresh diagnostic run at `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_protocol_examples_80.jsonl`.

The goal is not to explain every rejection. The goal is to preserve a few representative malformed move requests that can be inspected by hand while debugging the Rust adapter boundary.

## Before State
Before this capture:
- the Python-side target-sign fix had already been kept
- rejected-choice diagnostics already recorded raw requests, fallback choices, and legal-choice previews
- we still did not retain the last protocol lines leading into a malformed request

This pass added bounded per-side protocol history to the adapter and emitted that history with rejected-choice events.

## Problem
We need concrete examples where a move request is internally inconsistent enough that `move 1, pass` is offered by our legality layer but rejected by the simulator.

The recurring shape we wanted to isolate was:
- request type `move`
- rejected choice `move 1, pass`
- both `active[*].moves[*].target` values blank
- `side.pokemon[*].active` marks non-contiguous roster indices as active

## Solution
Run a short diagnostic benchmark after the protocol-history instrumentation and extract the first representative malformed move requests with exact protocol context.

### Benchmark summary
- artifact: `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_protocol_examples_80.txt`
- `completed_battles=80`
- `truncated_battles=41`
- `non_truncated_battles=39`
- `stalled_limit_truncations=41`
- `p1_rejected_choices=1009`
- `p2_rejected_choices=1177`
- `p1_unrecovered_rejections=581`
- `p2_unrecovered_rejections=763`

## Reasoning
These examples are useful because they keep showing the same contract mismatch:
- `active[]` clearly describes two acting mons
- move metadata for both active mons has empty target strings
- `side.pokemon` still represents the active pair as arbitrary roster indices instead of a coherent front-pair view

That is consistent with an adapter-boundary request-shaping problem, not just one more legality bug inside the sync driver.

## Examples

### Example 1: turn-5 p2 move request with active indices `[2, 5]`
- battle tag: `rust-sync-0`
- side: `p2`
- turn: `5`
- rejected choice: `move 1, pass`
- legal preview: `move 1, pass`, `move 1 terastallize, pass`, `move 2, pass`, `move 2 terastallize, pass`, `move 3, pass`, `move 3 terastallize, pass`, `move 4, pass`, `move 4 terastallize, pass`
- side active indices: `[2, 5]`
- side active flags: `[false, false, true, false, false, true]`
- front move lists described by `active[]`:
  - slot A: `Spore`, `Pollen Puff`, `Rage Powder`, `Clear Smog`
  - slot B: `Wood Hammer`, `High Horsepower`, `U-turn`, `Fake Out`
- protocol history:
  - `|-heal|p2a|160/219|[from] grassyterrain|[of] p2a`
  - `|-heal|p1a|171/207|[from] grassyterrain|[of] p1a`
  - `|-heal|p2a|133/207|[from] grassyterrain|[of] p2a`
  - `|upkeep`
  - `|turn|5`
- why it matters:
  - both active entries expose normal move menus, but the roster still says the active pair lives in positions `2` and `5`
  - every move target string in both active menus is `""`

### Example 2: turn-10 p1 move request with two fainted mons leading the roster
- battle tag: `rust-sync-4`
- side: `p1`
- turn: `10`
- rejected choice: `move 1, pass`
- legal choice count: `10`
- side active indices: `[2, 5]`
- side active flags: `[false, false, true, false, false, true]`
- side conditions: `['0 fnt', '0 fnt', '219/219 100', '130/167 100', '32/181 100', '158/207 100']`
- front move lists described by `active[]`:
  - slot A: `Wood Hammer`, `High Horsepower`, `U-turn`, `Fake Out`
  - slot B: `Spore`, `Pollen Puff`, `Rage Powder`, `Clear Smog`
- protocol history:
  - `|-weather|RainDance|[upkeep]`
  - `|-heal|p1a|158/207|[from] grassyterrain|[of] p1a`
  - `|-end|p2b|move: Taunt`
  - `|upkeep`
  - `|turn|10`
- why it matters:
  - the move request is built for Rillaboom plus Amoonguss, but the first two roster entries are both fainted
  - this is the clearest manual example that `active[]` and leading `side.pokemon` entries are not describing the same front pair

### Example 3: late-game p2 move request with active indices `[0, 4]`
- battle tag: `rust-sync-4`
- side: `p2`
- turn: `22`
- rejected choice: `move 1, pass`
- legal choice count: `9`
- side active indices: `[0, 4]`
- side active flags: `[true, false, false, false, true, false]`
- side conditions: `['120/207 100', '0 fnt', '219/219 100', '0 fnt', '29/181 100', '0 fnt']`
- front move lists described by `active[]`:
  - slot A: `Glacial Lance`, `High Horsepower`, `Trick Room`, `Encore`
  - slot B: `Thunderbolt`, `Taunt`, `Eerie Impulse`, `Sunny Day`
- protocol history:
  - `|-weather|RainDance|[upkeep]`
  - `|-heal|p1a|103/207|[from] grassyterrain|[of] p1a`
  - `|-heal|p2a|120/207|[from] grassyterrain|[of] p2a`
  - `|upkeep`
  - `|turn|22`
- why it matters:
  - even when one active mon really is roster index `0`, the partner is still represented as roster index `4` instead of a contiguous front-slot pair
  - this suggests the corruption is not limited to one specific early-game replace path

### Example 4: early turn-3 p1 move request immediately after a `Spore` interaction
- battle tag: `rust-sync-7`
- side: `p1`
- turn: `3`
- rejected choice: `move 1, pass`
- legal choice count: `12`
- side active indices: `[0, 2]`
- side active flags: `[true, false, true, false, false, false]`
- side conditions: `['207/207 100', '159/159 100', '219/219 100', '167/167 100', '181/181 100', '207/207 100']`
- front move lists described by `active[]`:
  - slot A: `Spore`, `Pollen Puff`, `Rage Powder`, `Clear Smog`
  - slot B: `Glacial Lance`, `High Horsepower`, `Trick Room`, `Encore`
- protocol history:
  - `|move|2a|Spore|1a`
  - `|-immune|p1a: Amoonguss`
  - `|`
  - `|upkeep`
  - `|turn|3`
- why it matters:
  - this shows the same malformed shape can appear very early, before long replacement chains or endgame churn
  - the issue is not limited to turns where the roster already contains multiple fainted mons

## Cross-example pattern
Across all four examples:
- `request['active']` has length `2`
- both active move menus expose only blank `target` metadata
- the active pair is represented by non-contiguous roster indices: `[2, 5]`, `[2, 5]`, `[0, 4]`, `[0, 2]`
- the sync driver still thinks `move 1, pass` is legal, but the simulator rejects it

That makes the next implementation target more concrete: sanitize the request into a coherent front-pair representation before poke-env legality and choice generation consume it.

## Planned Next Steps
1. Add invariant logging that records which protocol-tracked slots the adapter believes are `a` and `b` at the moment these requests are parsed.
2. Rebuild leading `side.pokemon` ordering from protocol-tracked slot occupancy for move requests before calling `parse_request()`.
3. Preserve the current JSONL shape so the same four-example extraction can be rerun after any adapter fix.