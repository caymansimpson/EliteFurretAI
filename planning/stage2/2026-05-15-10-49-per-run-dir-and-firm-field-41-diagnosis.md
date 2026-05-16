# Per-run-dir on resume; firm-field-41 win-rate regression diagnosis

**Date**: 2026-05-15 10:49
**Trigger**: firm-field-41 (12h run on `sep_arch.yaml`) showed value_loss
and policy_loss steadily improving while win rates against fixed baselines
*declined*. User asked for a diagnosis and a structural cleanup of the
resume-checkpoint directory behavior.

## Context

`firm-field-41` resumed from `pretty-jazz-4/main_model_step_365.pt` and
ran 674 wandb steps (~12h wall-clock) on `sep_arch.yaml`. Loss curves
moved in the expected direction. Win rates against fixed opponents
collapsed:

| Metric | start (0–30) | end (640–673) | Δ |
|---|---|---|---|
| `win_rate_vgc_bench_baseline` | 0.111 | **0.044** | **−61%** |
| `win_rate_max_damage` | 0.381 | 0.328 | −14% |
| `win_rate_simple_heuristic_baseline` | 0.184 | 0.152 | −17% |
| `win_rate_bc_player` | 0.499 | 0.405 | −19% |
| `entropy` | 5.43 | 5.29 | flat-high |
| `loss` (total) | 1.49 | 0.81 | ↓ |

Two structural problems were also surfaced during the investigation:

1. The `data/models/rl/pretty-jazz-4/` directory contained checkpoints
   from at least three distinct wandb runs (`pretty-jazz-4` itself,
   `celestial-bush-39`, `firm-field-41`) because `train.py:1108-1113`
   forced every resume to reuse the original run's directory. Step
   numbers in filenames were non-monotonic with mtime, making it
   genuinely hard to attribute a `.pt` to a wandb run.
2. The user's `sep_arch.yaml` was being modified mid-discussion to
   address findings (rnad_alpha bump, head depth bump) — the doc
   trail should capture both the *what* and the *why*.

## Before State

### `train.py:1106-1121`

```python
if config.training.resume_from:
    resume_parent = os.path.dirname(os.path.abspath(config.training.resume_from))
    if os.path.basename(resume_parent) in ("ghosts", "exploiters"):
        resume_parent = os.path.dirname(resume_parent)
    run_dir = resume_parent   # <-- writes into source run's dir
else:
    run_dir = os.path.join(config.training.save_dir, run_name)
```

### `sep_arch.yaml` (pre-discussion)

```yaml
algorithm:
  rnad_alpha: 0.05      # <-- contribution α*rnad ≈ 0.005 of total loss
  vf_coef: 0.5
architecture:
  turn_head_layers: [512, 256, 256]
  value_head_layers: [512, 512, 256]
```

## Problem

### 1. Loss-vs-capability divergence (firm-field-41)

Measured `vf_coef × value_loss / |policy_loss|` ratio across the run:

| Window | \|policy_loss\| | vf × value | **Ratio** | α × rnad |
|---|---|---|---|---|
| start (0–30) | 0.146 | 1.488 | **10.18×** | 0.0052 |
| q1 (150–180) | 0.244 | 1.386 | 5.68× | 0.0037 |
| mid (330–360) | 0.270 | 1.263 | 4.68× | 0.0028 |
| q3 (480–510) | 0.260 | 1.210 | 4.65× | 0.0032 |
| end (640–673) | 0.240 | 1.075 | 4.48× | 0.0055 |

Two distinct pathologies stacking:

- **Value-gradient dominance through the shared trunk**. The value
  cross-entropy gradient backpropagating into the trunk is 4–10×
  the policy gradient magnitude. With `backbone_lr=1e-5` per-step
  drift is small but consistent over 674 updates, biasing trunk
  features toward minimizing observed-return cross-entropy on a
  self-play distribution that's getting *worse* against fixed
  opponents.
- **R-NaD regularization is functionally off**. `rnad_alpha=0.05`
  × `rnad_loss≈0.1` = ~0.005, i.e. 0.5% of total loss. The "tether
  to portfolio references" that's the *entire point* of RNaD is
  contributing essentially zero. Nothing prevents the policy from
  drifting away from BC-anchored behaviors.

The pretty-jazz-4 resume source was checked and found *not* to be
regressed — its own wandb history (151 steps) shows improving win
rates (`vgc 0.08→0.18, bc 0.33→0.55`). The damage is being inflicted
by firm-field-41's training itself, not inherited.

Note: the file `pretty-jazz-4/main_model_step_365.pt` was actually
written by a different wandb run than `pretty-jazz-4` (the directory
had collided across resumes — see problem #2).

### 2. Run-directory collision

`pretty-jazz-4/` accumulated checkpoints at steps 157, 206, 208, 223,
224, 365, 373, 418, 427, 746, 1039 with non-monotonic timestamps —
the result of at least three distinct runs (`pretty-jazz-4`,
`celestial-bush-39`, `firm-field-41`) all resuming from each other
and writing into the same directory because of the deliberate
"reuse parent dir on resume" behavior. Auditing which run produced
which checkpoint required cross-referencing mtimes against wandb
`created_at` timestamps.

## Solution

### Part A: Per-run directory on resume (this commit)

Changed `train.py` to always set `run_dir = save_dir/wandb_run_name`
regardless of resume status. On resume, copy `ghosts/` and
`exploiters/` snapshots from the resume source into the new run_dir
so the curriculum (which reads opponents from these subdirs) keeps
continuity. New snapshots produced this run accumulate alongside the
copies, leaving the source directory untouched.

Implemented with `shutil.copy2` (preserves mtimes for auditability),
with per-file log lines showing source filename, size, and target
dir, plus a summary line. Skips cleanly when source subdir is absent
or empty, and short-circuits when source == destination.

Disk cost: up to (max_ghosts + max_exploiter_models) × ~300 MB per
resume, currently 10 × 300 MB ≈ 3 GB. Acceptable on the 24 GB WSL2.

### Part B: Config adjustments to `sep_arch.yaml` (user-applied)

- `rnad_alpha: 0.05 → 0.3` (6× bump). With current `rnad_loss≈0.1`,
  contribution rises from 0.5% to ~3% of total loss. Restores a
  meaningful tether toward portfolio references.
- `turn_head_layers: [512, 256, 256] → [768, 512, 256, 256]` —
  deepens policy turn head. Lets the head absorb more
  policy-specific reasoning so the trunk doesn't have to encode it.
- `value_head_layers: [512, 512, 256] → [768, 512, 512, 256]` —
  same logic for value head.

Trunk shape (`transformer_layers=4`, `early_layers`, `late_layers`)
was deliberately left unchanged so the `cool-bee-85-finetune` BC
checkpoint still partial-loads cleanly (only the fresh-init heads
are replaced).

## Reasoning

- **Why not lower vf_coef**: value_loss is still ~2.0 (cross-entropy
  over 51 bins, well above zero). Value head has genuine headroom.
  Cutting `vf_coef` would slow real value learning without
  addressing the routing problem.
- **Why not Path C (stop-gradient on value→trunk)**: user wants to
  preserve "one model, one forward pass." Path A (deeper heads) +
  raised RNaD α was chosen as the first move because it preserves
  every architectural property and pays no BC retraining cost.
- **Why per-run-dir over symlinks**: explicitly requested copy with
  logging — disk cost is small (~3 GB max per resume), and copies
  are robust to future deletion of the source.

## Planned Next Steps

1. **Launch a fresh run on the updated `sep_arch.yaml`**. With
   per-run-dir live, checkpoints land cleanly under
   `data/models/rl/<new-wandb-name>/` from the start.
2. **Monitor for**: win rate vs fixed baselines stabilizing or
   climbing (vgc_bench ≥ 0.10 and trending up; bc_player ≥ 0.50);
   entropy beginning to decline; gradient-dominance ratio shrinking
   as the deeper heads absorb specialization.
3. **If after ~200 updates the gradient ratio is still ≥4× and
   win rates are still declining**, that's the signal that head
   capacity wasn't enough and the next experiment is either:
   - Gradient scaling on the value→trunk path (~10 LOC, preserves
     single-model architecture)
   - Architecturally separate per-head transformers above a thin
     `feature_encoder + early_ff_stack` trunk (significant change,
     ~50% more transformer params, requires partial-load strategy
     for BC trunk)
4. **Verify the resume-copy code path** with a small smoke test
   (resume a config and confirm the log lines fire and files land in
   the new dir) before committing this code path to the main RL
   loop.

## Risks

- **Stale ghost/exploiter files accumulating across many resumes**:
  every resume copies all snapshots from source. If a run resumes
  from a long-lived source, then is itself resumed from, copies
  multiply. Mitigation: ghosts are LRU-capped by `max_ghosts=5` so
  the curriculum doesn't sample stale files even if more accumulate;
  exploiters similarly capped at `max_exploiter_models=5`. Operator
  can prune manually.
- **rnad_alpha 0.3 may over-regularize early**: if the bump pushes
  the policy too aggressively toward references, policy learning
  stalls. Watch `rnad_loss` trajectory — if it climbs and stays
  high, alpha is too strong.

## Updates

_None yet — fresh run not launched._
