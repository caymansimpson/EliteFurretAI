# 2026-05-23 01:00 — BC action-quality fixes (may23.yaml + train.py save-best change)

## Context

`sandy-firefly-98` (may22.yaml, finished 2026-05-22) regressed sharply
against `classic-firebrand-90` (also may22.yaml, same dataset, earlier
commit) on the action-prediction metrics that actually feed downstream
RL. This doc records the diagnostic comparison, the cause analysis, and
the targeted fixes shipped as `may23.yaml` + a small `train.py` change.

## Before state — diagnostic comparison (test split, 500 batches)

Both checkpoints evaluated against `data/battles/regc_final_v4/test/`
via [src/elitefurretai/supervised/analyze/action_model_diagnostics.py](
../../src/elitefurretai/supervised/analyze/action_model_diagnostics.py)
(patched to pass through `value_head_layers` and the number-bank knobs
embedded in the checkpoint config).

| Metric | classic-firebrand-90 | sandy-firefly-98 | Δ |
|---|---|---|---|
| OVERALL Top-1 | 29.04% | 22.15% | −6.9 pp |
| MOVE Top-1 | 9.66% | 8.80% | −0.9 pp |
| MOVE Top-3 | 23.05% | 22.28% | −0.8 pp |
| MOVE Top-5 | 36.74% | 34.99% | −1.8 pp |
| MOVE Top-10 | 47.33% | **56.67%** | **+9.3 pp** |
| BOTH Top-1 | 19.23% | 4.48% | **−14.8 pp** |
| BOTH Top-3 | 42.03% | 22.06% | −20 pp |
| BOTH Top-5 | **72.58%** | **23.01%** | **−50 pp** |
| SWITCH Top-1 | 98.97% | 75.53% | **−23 pp** |
| FORCE_SWITCH Top-1 | 98.66% | 95.20% | −3.5 pp |
| MOVE prediction bias | −1.1% | **+6.4%** (over) | — |
| BOTH prediction bias | −0.8% | **−6.5%** (under) | — |
| Sandy's top predicted action | — | `move 3, move 3` (untargeted) | — |
| Classic's top predicted action | `move 1 2, move 1 1` (targeted) | — | — |

The full per-head trajectories also tell the same story: sandy-firefly's
MOVE Top-1 *peaked at eval #5 (epoch ~10, 19.7%)* and decayed to 10.9%
by the saved-best checkpoint at eval #17, while Test Loss kept falling.
SWITCH Top-1 followed the same peak-and-decay shape (73.6% → 63.2%).

## Problem

Three coupled failure modes, ranked by severity:

1. **BOTH collapse.** Sandy-firefly predicts BOTH actions only 5.6%
   of the time (actual 12.0%) — substitutes plain MOVEs instead.
   BOTH Top-5 dropped 73% → 23%. This is the trunk losing its
   joint move+switch representation.
2. **Target collapse.** Sandy-firefly's top predicted actions are
   *untargeted* (`move 3, move 3`, `move 1, move 2`). The model
   hedges by dropping target indices. In doubles, this is a
   degenerate fallback that likely produces invalid actions at RL
   time unless the target is implicit.
3. **SWITCH degraded.** 99% → 75% Top-1 — direct consequence of
   `switch_loss_weight: 0.1` (was 1.0 in classic-firebrand). The
   easiest, highest-base-rate head was starved of gradient.

The mechanism: may22.yaml dropped `switch_loss_weight` and
`teampreview_loss_weight` from 1.0 to 0.1 and added
`value_label_smoothing: 0.05`. SWITCH and BOTH share the trunk with
MOVE — under-weighting SWITCH starved the trunk's gradient signal
on switch-side tokens, which then degraded the joint MOVE+SWITCH
(BOTH) predictions.

## Reasoning — why only `switch_loss_weight` goes back to 1.0

A simpler "revert everything to classic-firebrand" config would also
fix the regression, but it would discard the legitimate
improvements in may22 (wider transformer FF, longer training,
deeper value head).

Weights judged through an RL-importance lens:

| Head | RL importance | Decisions / battle | Current | Diag regression | Action |
|---|---|---|---|---|---|
| turn | critical | ~25-30 | 1.0 | — | keep 1.0 |
| move_loss_weight | critical | ~17 (67%) | 1.0 | small (−1pp Top-1) | keep 1.0 |
| switch_loss_weight | high | ~5 (19%) | 0.1 | huge (−23pp Top-1) | **raise to 1.0** |
| win_loss_weight | high (becomes RL critic) | — | 0.35 | Win Corr unstable | keep 0.35 |
| teampreview_loss_weight | low | 1 | 0.1 | tiny (−2pp on near-saturated metric) | **keep at 0.1** |

Teampreview is a single decision per battle. The +2pp drop from
99.7% → 97.7% is irrelevant to downstream RL.

## Solution — may23.yaml diff (vs may22.yaml)

```yaml
# Loss Weights
switch_loss_weight: 1.0            # was 0.1

# Distributional Value Head
value_label_smoothing: 0.0         # was 0.05 — double-smoothing on C51

# Regularization (compensate for FF dim 2048 × 50 epochs)
dropout: 0.25                      # was 0.2
weight_decay: 3.0e-4               # was 1.44e-4

# Save-best (new field)
save_best_metric: action_score     # new — see train.py change below
```

No architecture change. No epoch reduction (kept at 50, per user
preference — the save-best change carries the burden of picking the
right checkpoint).

## Solution — train.py change

`--save-best` previously keyed on Test Loss, which is dominated by
`turn_top3_loss * turn_loss_weight` (top-3 *probability sharpness*).
This metric did not track MOVE/BOTH Top-1 argmax accuracy — sandy-
firefly's saved-best checkpoint sits at the epoch where Test Loss
bottomed out *but the action heads had already regressed by ~50%*.

The fix at [src/elitefurretai/supervised/train.py:743-790](
../../src/elitefurretai/supervised/train.py#L743-L790):

```python
action_score = (
    metrics.get("move_top3_acc", 0) * 0.5
    + metrics.get("both_top3_acc", 0) * 0.3
    + metrics.get("switch_top1_acc", 0) * 0.1
    + metrics.get("win_corr", 0) * 0.1
)
```

And `save_best_metric: action_score` in the config switches the
save-best comparator from `min(test_loss)` to `max(action_score)`.

Weights chosen by RL importance:

* MOVE Top-3 (0.5) — most decisions, hardest task
* BOTH Top-3 (0.3) — joint reasoning, where MODEL_EVALUATION.md's
  "develop a second strategy" requirement lives
* SWITCH Top-1 (0.1) — high base rate, easy task; included to flag
  collapses
* Win Corr (0.1) — becomes the RL critic

Default behaviour (`save_best_metric: test_loss`) preserves backwards
compatibility with prior configs that don't set the field.

## Planned Next Steps

1. `lyric-feather-99` is training now (50 epochs, ~6h ETA).
2. When it finishes:
   * Re-run [action_model_diagnostics.py](
     ../../src/elitefurretai/supervised/analyze/action_model_diagnostics.py)
     on `lyric-feather-99_best.pt`.
   * Goal: BOTH Top-5 back to ≥60% (classic was 73%) and SWITCH
     Top-1 back to ≥95% (classic was 99%), without sacrificing
     MOVE Top-1 (~9-10%).
3. If the action-score save-best works as designed, the saved
   checkpoint should sit near the *MOVE Top-1 peak*, not the Test
   Loss minimum — verify by checking which eval# produced the saved
   checkpoint.
4. If `lyric-feather-99` improves on `classic-firebrand-90`'s
   action metrics, promote it as the new BC reference checkpoint
   for the next RL warm-start (replacing `cool-bee-85-finetune`).

## Updates

* 2026-05-23 01:00 — config + train.py change landed, training
  kicked off (PID 27187, log at `/tmp/may23_training.log`).
