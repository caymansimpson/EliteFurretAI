# may23 bs × lr × wd sweep (10 epochs / cell)

Purpose: figure out why `cool-bee-85-finetune_best` (bs=128, lr=5.44e-5)
is still ~17 pp better on MOVE Top-1 than the may22/may23 family
(bs=512, lr=5.44e-5). Six cells isolate `(batch_size, learning_rate)`
while controlling AdamW's `lr × wd` per-step decay (held constant
across cells at 7.84e-9 — cool-bee-85's value).

See `planning/stage2/2026-05-23-01-00-bc-action-quality-fixes.md`
for the full diagnostic that motivated this sweep.

| Cell | bs  | lr       | wd       | Hypothesis tested                       |
|------|-----|----------|----------|-----------------------------------------|
| S1   | 128 | 5.44e-5  | 1.44e-4  | cool-bee baseline (reference at ff=2048)|
| S2   | 128 | 1.09e-4  | 7.20e-5  | does higher lr alone help at bs=128?    |
| S3   | 128 | 2.18e-4  | 3.60e-5  | aggressive lr; diverge or accelerate?   |
| S4   | 512 | 5.44e-5  | 1.44e-4  | current state (confirms regression)     |
| S5   | 512 | 1.09e-4  | 7.20e-5  | sqrt-scaled lr at bs=512                |
| S6   | 512 | 2.18e-4  | 3.60e-5  | linear-scaled lr at bs=512              |

**Fixed across all cells** (cool-bee-85 defaults except ff_dim):
- `transformer_ff_dim: 2048` (user-pinned for this sweep)
- `dropout: 0.20`, `transformer_dropout: 0.10`
- `value_label_smoothing: 0.0`, `switch_loss_weight: 1.0`, `teampreview_loss_weight: 1.0`
- `num_epochs: 10`, `eval_every: 10` (single eval at the end)
- `worker_batch_size` equals `batch_size` (no gradient accumulation)
- `save_best_metric: action_score` — only one eval, so best == final

**Per-cell expected runtime:** ~75 min (sequential).
**Total sweep cost:** ~7.5 h.

## How to run

```bash
cd /home/cayman/Repositories/EliteFurretAI
bash src/elitefurretai/supervised/sweep_configs/may23_bs_lr/run_sweep.sh
```

Each cell's wandb run name auto-generated; checkpoint saved as
`data/models/supervised/<wandb_name>.pt`. Look up the wandb run names
in the per-cell logs at `/tmp/may23_bs_lr_S{1..6}.log`.

## How to read the results

After all 6 cells complete:

```bash
source ../venv/bin/activate
for ckpt in $(ls data/models/supervised/*.pt | grep -v _best | tail -6); do
  python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
    "$ckpt" data/battles/regc_final_v4/test/ 500 \
    > /tmp/diag_$(basename $ckpt .pt).log 2>&1
done
```

Then compare MOVE Top-1 (the headline metric) across all 6 cells.
Decision rules in `2026-05-23-01-00-bc-action-quality-fixes.md`.
