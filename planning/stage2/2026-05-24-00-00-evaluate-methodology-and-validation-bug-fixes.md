# 2026-05-24 00:00 — evaluate() per-action-type methodology + post-training validation bf16 fix

## Context

Two issues surfaced by the may23 bs × lr sweep results
([2026-05-23-01-00-bc-action-quality-fixes.md](2026-05-23-01-00-bc-action-quality-fixes.md)):

1. **Methodology mismatch.** `utils.py:evaluate()` and
   `action_model_diagnostics.py` reported different per-action-type
   Top-K accuracy for the same checkpoint on the same data. Some cells
   had >15 pp gaps (e.g. S2 MOVE Top-1: 19.0% per evaluate(), 2.35% per
   diagnostic). Since `--save-best`'s `action_score` is derived from
   evaluate()'s numbers, this meant save-best was selecting on a
   metric that didn't match the inference-time methodology used to
   measure production checkpoints (SUPERVISED.md's numbers come from
   the diagnostic).

2. **Post-training validation pass crashed.** Every may22+ training run
   ended with a `RuntimeError: mat1 and mat2 must have the same dtype,
   got BFloat16 and Float` from an inductor-compiled `extern_kernels.
   addmm` call. This was the dual_expand bf16 issue documented in
   [2026-05-22-11-51-dual-expand-dtype-flex-fix.md](2026-05-22-11-51-dual-expand-dtype-flex-fix.md)'s
   "adjacent items" — never fixed because we hadn't run into it.
   Training still completed (both checkpoints saved before the crash)
   but the wandb run exited non-zero and skipped the final validation
   metrics.

## Before state

### Methodology gap (verified by direct comparison)

| Metric | evaluate() output | diagnostic output | gap |
|---|---|---|---|
| S5 MOVE Top-1 | 0.2041 | 0.0809 | 12.3 pp |
| S5 SWITCH Top-1 | 0.6385 | 0.8384 | -20 pp |
| S5 BOTH Top-1 | 0.0333 | 0.0144 | +1.9 pp |

The diagnostic excludes force-switch turns from MOVE/SWITCH/BOTH
classification (it uses the state's force_switch indicator), while
`evaluate()` bucketed force-switch actions into SWITCH or BOTH based
on MDBO classification. This shifted samples between buckets and
distorted every per-type Top-K accuracy.

### Validation crash trace

```
File ".../torchinductor_cayman/.../call.py", line 8647, in call
    extern_kernels.addmm(arg26_1,
        reinterpret_tensor(buf4, (40*s77, 704), (704, 1), 0),
        reinterpret_tensor(arg25_1, (704, 256), (1, 704), 0),
        alpha=1, beta=1, out=buf12)
RuntimeError: mat1 and mat2 must have the same dtype,
              but got BFloat16 and Float
```

Cause: `train.py:811` passed the compiled `model` (not
`model._orig_mod`) to `evaluate()`. The compiled artifact's
inductor-generated addmm doesn't tolerate the bf16 dataloader cast
on `states`. The in-loop eval at `train.py:666` was already routed
through `_orig_mod` for the same reason; the post-training
validation pass had been overlooked.

## Solution

### 1. evaluate() — bucket force_switch separately

[utils.py:615-712](../../src/elitefurretai/supervised/utils.py#L615-L712):

```python
fs_indices = config.get("force_switch_indices", []) if config else []
if fs_indices and valid_states is not None:
    turn_states = valid_states[turn_mask]
    fs_per_sample = torch.zeros(
        turn_states.shape[0], dtype=torch.bool, device=turn_states.device,
    )
    for fs_idx in fs_indices:
        fs_per_sample = fs_per_sample | (turn_states[:, fs_idx] > 0.5)
else:
    fs_per_sample = ...  # all-False fallback

for action_idx, action in enumerate(turn_actions):
    if bool(fs_per_sample[action_idx].item()):
        action_type = "force_switch"
    else:
        # ... existing MDBO-based MOVE/SWITCH/BOTH classification
```

Added "force_switch" to the metric initialization and normalization
loops so the bucket reports `force_switch_steps` and
`force_switch_top{k}_acc` cleanly.

Verified by direct re-evaluation of `noble-firebrand-104_best.pt`
(sweep cell S5): the new evaluate() output matches the diagnostic
JSON to 4 decimal places on every metric (MOVE/SWITCH/BOTH/FS Top-1
and Top-3, sample counts identical).

### 2. train.py — route post-training validation through _orig_mod

[train.py:810-823](../../src/elitefurretai/supervised/train.py#L810-L823):

```python
val_eval_model = getattr(model, "_orig_mod", model)
metrics = evaluate(
    val_eval_model,
    val_loader,
    ...
)
```

One-line behavioral fix. The in-loop eval was already correct; this
mirrors that path.

## Reasoning

For (1), the alternative would have been to update the diagnostic to
match evaluate()'s methodology (don't separate force_switch). That
would make the existing SUPERVISED.md numbers and downstream RL
inference behave inconsistently — RL uses masked-and-argmax-over-
valid-actions, which is what the diagnostic measures. The diagnostic
methodology is closer to inference behavior, so evaluate() should
align to it, not the other way around.

For (2), the alternative would have been to fix the underlying
dtype-flexibility in the compiled forward path (the "adjacent items"
audit from the 2026-05-22 doc). That's a much bigger change touching
every Linear/Attention/LayerNorm site and risks regressing
throughput. The eager-path workaround already exists for in-loop
eval; using it for post-training validation is a 1-line consistency
fix with zero risk.

## Planned Next Steps

1. `feasible-night-106` is now training (50 epochs, bs=512,
   lr=1.09e-4, all may24 defaults). PID 19056, log at
   `/tmp/may24_training.log`. ETA ~7 h.
2. When complete: re-run action diagnostics, compare against
   `cool-bee-85-finetune_best` (the production checkpoint).
3. If diagnostic numbers approach or beat cool-bee-85-finetune's
   (MOVE T3 52.6%, BOTH T1 47.7%): proceed to 15-epoch cosine
   finetune with `win_loss_weight=0.5` to match cool-bee's procedure,
   then promote as the new RL initializer.
4. If still below cool-bee-85-finetune: bisect git history between
   2026-05-03 (cool-bee training) and 2026-05-22 (classic-firebrand /
   sandy-firefly) to find what regressed. Likely suspects: bf16
   autocast change (a79d962), bf16 dataloader cast (b3cdb46), CUDA
   stream prefetcher (562c76c).

## Updates

- 2026-05-24 00:00 — both fixes landed (this doc). Sanity-check
  passes (evaluate output == diagnostic output bit-for-bit). may24
  training kicked off.
