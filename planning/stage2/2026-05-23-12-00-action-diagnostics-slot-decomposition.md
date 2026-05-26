# 2026-05-23 12:00 — Action diagnostics: per-slot decomposition + per-move favor

## Context

Following the BC-action-quality investigation in
[2026-05-23-01-00-bc-action-quality-fixes.md](2026-05-23-01-00-bc-action-quality-fixes.md),
we observed regressions that hinted at *substructural* failures — e.g.
sandy-firefly's "untargeted move" mode (`move 3, move 3`) and a SWITCH
collapse separate from MOVE accuracy. The existing diagnostic at
[src/elitefurretai/supervised/analyze/action_model_diagnostics.py](../../src/elitefurretai/supervised/analyze/action_model_diagnostics.py)
treated each action as one opaque int and could not split:

* "model picked the right move slot but wrong target"
* "model picked the right target but wrong move slot"
* "model swapped a switch for a move (or vice versa) at one slot"
* "model disproportionately favors/disfavors specific single-mon orders"

## Before state

Diagnostics output:

* Action-type distribution (MOVE / SWITCH / BOTH / FORCE_SWITCH) with
  prediction bias — already answered "switches more than it should"
  at the *whole-action* granularity.
* Top-k accuracy by action type.
* Confidence / loss / invalid-prediction stats.
* `move_analysis`: top-N most-common actual vs predicted full action
  indices (no per-move bias).

## Problem

The whole-action int collapses two independent per-mon decisions
(which slot, which target, switch vs move) into a single label. A
"wrong" prediction could be off by a target only, or by an entire
kind — the report could not tell the difference, which made it hard
to direct loss-weight or featureset changes.

## Solution

Extended `ActionDiagnostics` with three new pieces of state (no new
batch fields, no embedder spelunking):

1. `_parse_order` / `_decompose_action` / `_order_to_canonical` —
   parse each MDBO `/choose <order1>, <order2>` into two
   per-active-slot dicts (`kind`, `slot`, `target`, `tera`).
2. `_classify_slot_pair` — bucket each slot's (actual, predicted)
   pair into one of:
   * `move_correct`
   * `move_slot_correct_target_wrong`  ← question #2
   * `move_target_correct_slot_wrong`  ← question #3
   * `move_both_wrong`
   * `switch_correct` / `switch_wrong_slot`
   * `pass_match`
   * `kind_mismatch_actual_<a>_pred_<p>` (e.g. actual move, pred switch)
3. Per-canonical-order counters (actual + predicted) restricted to
   `move ...` orders when building the favor table, with a
   `min_joint_count_threshold=10` filter to suppress rare-noise rows.

Two new report sections + corresponding print blocks:

* `per_slot_decomposition` — counts and pct per bucket across all
  decomposable timesteps (~2× total_predictions).
* `per_move_favor` — top-15 over- and under-predicted move-kind
  single-slot orders by `predicted_pct − actual_pct`.

Question #4 (status vs damaging move) is deferred — it needs a move
category lookup (either embedder per-slot feature index, or batch
move-id passthrough). Punted to a follow-up once the slot-decomp
report tells us whether category errors are a meaningful share of
`move_both_wrong`.

## Reasoning — why purely action-index parsing

MDBO encodes orders as `<slot>×<slot>` pairs from
`_ORDER_MAPPINGS_TO_INT` (see
[src/elitefurretai/etl/encoder.py:160-173](../../src/elitefurretai/etl/encoder.py#L160-L173)),
so the substructure can be recovered from the int alone without
re-running the embedder, the dataloader, or anything stateful. The
parser round-trips against `MDBO.from_int(i, MDBO.TURN).message` for
arbitrary `i`, including tera and pass cases (verified in the
smoke-test below).

## Verification

Smoke test (run from venv):

```
i=     0 msg='/choose move 1 -2, move 1 -2'  -> ['move 1 -2', 'move 1 -2']
i=   100 msg='/choose move 1, move 2 -2'      -> ['move 1', 'move 2 -2']
i=   500 msg='/choose move 2 -1, move 1 -2 terastallize'
                                              -> ['move 2 -1', 'move 1 -2 tera']
i=  1500 msg='/choose move 4 1, move 2 -2 terastallize'
                                              -> ['move 4 1', 'move 2 -2 tera']
i=  2024 msg='/choose pass, pass'             -> ['pass', 'pass']
all smoke checks passed
```

Quality gates: `ruff check`, `ruff format`, `pyright` all clean on the
modified file.

## Planned Next Steps

1. Run the extended diagnostic on `cool-bee-85-finetune_best.pt` and
   on `lyric-feather-99_best.pt` (once it finishes) against
   `data/battles/regc_final_v4/test/`, 500 batches each. Compare
   per-slot bucket shares — especially `move_slot_correct_target_wrong`
   share, which is the direct measurement of the "untargeted move"
   regression sandy-firefly displayed.
2. Re-evaluate whether question #4 (status vs damaging move) is worth
   wiring up — depends on whether `move_both_wrong` dominates or
   whether category mistakes are a separable subbucket.
3. If `move_slot_correct_target_wrong` is large, consider per-slot
   target supervision (separate target head) — but only after we
   confirm the failure mode is widespread, not isolated to one model.

## Updates

* 2026-05-23 12:00 — implementation landed, smoke-tested. Not yet
  run on a real checkpoint.
* 2026-05-23 14:00 — ran on `cool-bee-85-finetune_best.pt` (200
  batches, CPU, safe-mode). Headline: Top-1 30.17%, MOVE Top-1
  15.92%. 0 invalid predictions. Action-type bias near zero across
  the board (SWITCH +0.5pp). Per-slot decomposition:
  `move_target_correct_slot_wrong` 15.96% > `move_slot_correct_target_wrong`
  10.18% — model anchors targets more reliably than slots, the
  *inverse* of sandy-firefly's untargeted-move regression. Per-move
  favor (slot-int): `move 2 2` over-predicted by +7.80pp; `move 4`
  and `move 1 1` are the strongest under-predicted patterns.

* 2026-05-23 16:00 — **shipped #4 (status vs damaging) and
  reworked #5 to be move-name-based** (was slot-int based).

  ### What changed

  Added in [action_model_diagnostics.py](
  ../../src/elitefurretai/supervised/analyze/action_model_diagnostics.py):

  * `_lookup_category(state, mon_idx, move_slot, off_cat_indices)` —
    reads `MON:i:MOVE:j:OFF_CAT:{PHYSICAL,SPECIAL,STATUS}` one-hots
    and returns the category string.
  * `_lookup_move_name(state, mon_idx, move_slot, move_id_indices)` —
    reads `MON:i:MOVE:j:move_id` (int) and resolves through the
    inverse of `etl.embedder.MOVE_TO_ID` to a human name like
    `"protect"`, `"fakeout"`.
  * `_active_mon_indices(state, mon_active_indices)` — returns the
    list of MON:i with `active=1`.

  New analyze_batch tracking (only applied when both sides of a
  slot-pair chose a move):
  * `category_confusion: {(actual_cat, pred_cat): float}` — counts
    of category transitions, with uniform-split weight for ambiguous
    timesteps (see caveat below).
  * `per_move_name_actual / per_move_name_predicted` — replacing
    the slot-int `per_move_favor` from the earlier iteration.

  New report sections:
  * `category_confusion`: full 4×4 matrix (P/S/STATUS/UNKNOWN) +
    collapsed 2×2 (DAMAGING vs STATUS) + `off_diagonal_pct` summary.
  * `per_move_name_favor`: top-15 over- and under-predicted moves by
    name (e.g. `protect +3.2pp`), with `ambiguous_pct` reported.

  Also refactored `main()`:
  * CLI is now argparse-based with `--device {cpu,cuda,auto}`,
    `--batch-size`, `--num-workers`, `--prefetch-factor`,
    `--files-per-worker`, `--torch-num-threads`, `--max-batches`.
  * `--safe` shortcut forces `device=cpu, batch_size=32,
    num_workers=2, prefetch_factor=1, files_per_worker=1,
    torch_num_threads=2` — drop-in for "co-existing with a training
    run on the same machine".

  ### Caveat: ambiguity from missing active-slot positional feature

  The embedder emits `MON:i:active` (which 2 of 6 mons are currently
  on the field) but **no `active_slot_0` / `active_slot_1`
  positional marker**. MDBO slot 0 corresponds to
  `battle.active_pokemon[0]`, but the state tensor doesn't tell us
  which MON:i that is. So when both active mons have *different*
  categories or move IDs at the chosen move-slot, we cannot
  attribute the chosen move to a specific mon from the state alone.

  We handle this by uniform-weight splitting: if active mons A and B
  have categories {PHYSICAL, STATUS} at the chosen slot, the chosen
  move contributes 0.5 weight to each. This preserves *aggregate*
  trends but blurs specific attribution — a Protect from mon A and
  a Detect from mon B at slot 2 will both gain weight when the
  model predicts move 2. Reports include `ambiguous_pct` so the
  reader knows how often this splitting kicked in.

  Cleaner alternatives, all deferred as costlier:
  1. Add `MON:i:active_slot_0` / `active_slot_1` to the embedder —
     would force a re-preprocessing pass on all stored data.
  2. Have the dataloader emit per-timestep active-slot mapping
     alongside `states`.
  3. Re-derive mappings from raw battle logs (heavy I/O).

  ### Safe-run CLI (one-liner)

  ```bash
  source ../venv/bin/activate && \
  nice -n 19 python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
      data/models/supervised/<checkpoint>.pt \
      data/battles/regc_final_v4/test/ \
      --safe --max-batches 200
  ```

  `--safe` is co-runnable with a training job on this machine (8
  cores, 24 GB RAM, RTX 3090). Without `--safe`, defaults are tuned
  for a dedicated machine.

## Interpretation guide (use for every run summary)

Whenever this diagnostic is run, the assistant should produce a
summary structured to these six sections, in this order. The point
is that two runs against different checkpoints should be directly
comparable at a glance.

### 1. Headline (one paragraph)

State: checkpoint name, max_batches used, total predictions, Top-1
overall, **and** MOVE Top-1, BOTH Top-1, SWITCH Top-1, FORCE_SWITCH
Top-1 — these four are the comparable cross-model anchors. Note any
extreme value (>2× difference from a reference model). Always
mention `invalid_predictions.rate`; non-zero is a red flag.

### 2. Action-type bias (question #1)

Read `action_type_distribution[*].prediction_bias`. Apply these
thresholds:

| Bias magnitude | Read it as |
|---|---|
| ≤ 1 pp | Calibrated; nothing to flag |
| 1–3 pp | Mild lean; mention but don't alarm |
| 3–6 pp | Meaningful skew — investigate |
| > 6 pp | Strong skew, likely action-head pathology |

Explicitly answer "is the model over/under-switching?" using the
SWITCH bias.

### 3. Per-slot decomposition (questions #2 and #3)

From `per_slot_decomposition.buckets`, report these in this order:

1. `move_correct` pct (the positive signal).
2. `move_slot_correct_target_wrong` pct → **question #2** (right
   move, wrong target).
3. `move_target_correct_slot_wrong` pct → **question #3** (right
   target, wrong move).
4. `move_both_wrong` pct (residual failure).
5. Sum of `kind_mismatch_*` rows → **flag if > 1%**; this means the
   model swaps moves for switches at the slot level.

Then state the *ratio* of #3 to #2 — >1 means the model anchors
target better than slot (cool-bee pattern); <1 means it anchors
slot better than target (sandy-firefly pattern). The ratio is more
diagnostic than the absolute numbers.

### 4. Category confusion (question #4)

From `category_confusion`:

* Report `off_diagonal_pct` — the overall mis-categorization rate.
* Report the four collapsed cells (DAMAGING↔STATUS). Specifically
  flag `actual_STATUS_pred_DAMAGING` as a fraction of all STATUS
  actuals — this is "model swapped a status move for damage", the
  most expensive class of error for VGC strategy.
* Always print `ambiguous_pct`. If > 50%, downweight the category
  finding's confidence.

### 5. Per-move-name favor/disfavor (question #5)

From `per_move_name_favor`, print the top 5 over- and under-
predicted moves with their bias. Thresholds:

| Bias | Read it as |
|---|---|
| < 1 pp | Noise / not actionable |
| 1–3 pp | Mild preference; note if a pattern (e.g. all status moves under-favored) |
| > 3 pp | Strong preference — name it as a "smoking gun" |

Always also note `ambiguous_pct` — if > 50%, prefix with "names
are noisy due to active-slot ambiguity".

### 6. Where the loss lives + suggested next steps

Print `loss_contribution` top entry's `pct_of_total`. Whichever
action type contributes >80% of loss is the head most worth
improving. End with 1–3 concrete suggestions tied to the findings
above (never generic).

### Section presentation rules

* Use tables for any list of >3 items.
* Cite specific cell values, not adjectives ("`move 2 2` +7.80pp",
  not "model strongly favors targeted slot-2 moves").
* If a finding might be confounded by `ambiguous_pct > 30%`, label
  it `(weighted)` to be honest.

## Addendum 2026-05-23 17:00 — per-mon kind matrix + decomposition trees + ambiguous_pct definition

After the first test run, four refinements landed in the diagnostic
and need to be reflected in every summary:

### What's `ambiguous_pct`

Same definition for #4 (categories) and #5 (move names):

> The fraction of move-vs-move slot pairs where the **two active
> mons would give different answers** at the chosen move slot —
> different categories for #4, different `move_id`s for #5. Because
> the state tensor does not tell us which active mon owns MDBO
> slot 0 vs slot 1, we cannot attribute the chosen move to a
> single mon on those timesteps. We split unit weight uniformly
> across all combinations consistent with the two candidates.

For categories, `ambiguous_pct` typically ~70–85% (only ~3 possible
categories, so two random mons agree by chance ~1/3 of the time).
For move names, `ambiguous_pct` is typically ~95–99% (hundreds of
possible names → agreement is rare).

This means **per-move-name favors reflect population-level patterns,
not specific-mon attribution.** A "tailslap +4pp" tells you that
tailslap-bearing slots are over-represented in predictions, but not
that any particular mon is over-using tailslap. The user has
accepted this tradeoff; the cleaner fix (active-slot embedder
feature) is deferred.

### New section: per-mon kind confusion (replaces vague "switching" lens)

`per_mon_kind_confusion` is a 4×4 matrix (MOVE/SWITCH/PASS/OTHER on
each axis) derived from `slot_buckets`. Each datapoint is one
active mon on one turn, so 2 per turn under TURN order_type.

Interpretation:
* Diagonal entries (MOVE→MOVE, SWITCH→SWITCH, PASS→PASS) sum to
  the kind-agreement rate. Healthy models are >95% diagonal.
* Off-diagonal MOVE↔SWITCH entries reveal a deeper failure than
  the per-slot `kind_mismatch_*` rows: "the model thinks this mon
  should switch when the BC label says it should move".

### New section: move-vs-move trees (#3 reframed)

Two trees rooted on the same 4-cell 2×2. Throughout this section
"**move id**" means "the move slot 1-4 within the mon's moveset"
(unambiguous from the MDBO action); we use that wording rather than
"slot" to avoid collision with "active slot" (MDBO position 0 vs 1).

* Tree 1 (move-id first): `P(target ✓ | move id ✓)` vs
  `P(target ✓ | move id ✗)`. If these are similar, move-id and
  target errors are independent. If
  `P(target ✓ | move id ✓) >> P(target ✓ | move id ✗)`, the model
  uses move-id info to infer target (good).
* Tree 2 (target-first): symmetric.
  `P(move id ✓ | target ✓)` vs `P(move id ✓ | target ✗)`.

Both trees are presented because they answer different questions:
"is move-id a useful predictor of target?" vs "is target a useful
predictor of move-id?". A well-targeted model has both conditionals
elevated.

### Updated interpretation-guide sections (replaces sections 2-5 above)

#### 2 (revised). Per-mon kind confusion (Q1 reframed)

Report total datapoints, then read the diagonal sum as "kind
agreement %". Read MOVE→SWITCH and SWITCH→MOVE counts directly —
both as raw count and as fraction of the actual-MOVE / actual-SWITCH
row totals. Flag if either off-diagonal share is > 2%.

The previous SWITCH-bias number (from `action_type_distribution`)
remains useful but is now secondary: the per-mon matrix gives the
direct answer to "does the model substitute switches for moves at
the mon level".

#### 3 (revised). Move-vs-move trees (Q2 + Q3)

Print Tree 1 and Tree 2 each in compact form. Then report the
**conditional gap**:

| Quantity | What it tells you |
|---|---|
| `P(target ✓ \| move id ✓) − P(target ✓ \| move id ✗)` | How much getting the move id right helps the target. Large positive = the heads correlate / share representation. |
| `P(move id ✓ \| target ✓) − P(move id ✓ \| target ✗)` | Symmetric — how much target right helps move id. |

If both gaps are <5 pp, the heads are effectively independent — a
target-side change won't help move-id accuracy and vice versa.

#### 4 (unchanged). Category confusion

Same as before. Always also print the one-line definition of
`ambiguous_pct` for the reader.

#### 5 (revised). Per-move-name favor (per-mon framing)

The section now bills itself as "per-mon": each active mon per
turn is one datapoint (≤2 per turn). Aggregate move-name
distributions are identical to the previous per-MDBO-slot framing
(by symmetry), but the framing matches how the user reasons.
Still print the one-line `ambiguous_pct` definition.
