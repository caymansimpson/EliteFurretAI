# Fix: NaN Training Loss in Supervised Sweep

**Date**: 2026-03-23
**Status**: Completed

## Context
All wandb sweep runs (sweep `cc2j9589`, project `elitefurretai-hydreigon`) produced NaN training loss. 7 runs completed/crashed — 2 crashed with backward() exceptions, 5 completed with completely static metrics (model never updated because GradScaler silently skipped every optimizer step).

## Root Cause
**`_build_causal_mask` in `TransformerThreeHeadedModel` produced NaN values via `0 * float("-inf") = NaN`** (IEEE 754 arithmetic).

The original code:
```python
causal = torch.triu(torch.ones(turn_len, turn_len, device=device), diagonal=1)
mask[n_dt:, n_dt:] = causal * float("-inf")  # BUG: 0 * -inf = NaN
```

This created a mask where positions that should be 0 (attend) were NaN, and positions that should be -inf (don't attend) were correctly -inf.

The mask "worked" in `model.eval() + torch.no_grad()` because PyTorch's SDPA uses a fused kernel in that mode that happens to handle NaN gracefully. In any other mode (train, gradient tracking), the NaN propagated through attention → all turn logits → loss → gradients, causing universal NaN.

## Fix
One-line change in `_build_causal_mask` — construct the mask by applying `torch.triu` to a `-inf`-filled matrix (triu sets lower triangle to 0 via replacement, not multiplication):

```python
mask[n_dt:, n_dt:] = torch.triu(
    torch.full((turn_len, turn_len), float("-inf"), device=device),
    diagonal=1,
)
```

**File**: `src/elitefurretai/supervised/model_archs.py`, `_build_causal_mask` method.

## Verification
- Forward pass produces no NaN in all 8 combinations of {eval, train} × {no_grad, grad} × {FP32, autocast}
- Training loss steadily decreases: 4.26 → 3.76 over 10 steps on a single batch
- Gradient norms are finite and stable (~1.5-1.8)
- Mixed precision (autocast + GradScaler) works correctly with scale=65536

## New Sweep (Attempt 1 — `j13ld3pr`)
Started new sweep with 50 runs using `data/battles/regc_final_v4/`. First run (`stilted-sweep-1`) completed 2 epochs with improving metrics (Turn Top3: 0.305→0.312, MOVE Top3: 0.176→0.185), then was OOM-killed mid-epoch 3.

### OOM Root Cause
`dmesg` confirmed the Linux OOM killer killed the process. Peak concurrent DataLoader workers was 7 (3 persistent train + 4 test), each a `forkserver`-spawned process holding PyTorch runtime + decompressed 244MB `.pt.zst` files. Combined with VS Code/Node processes (~5-10GB), total memory exceeded 24GB RAM + 16GB swap.

### OOM Fix
In `train_sweep.py`:
- Train workers: `num_workers` 3→2, `files_per_worker` 3→2
- Test workers: `num_workers` 4→2
- Peak concurrent workers: 7→4, saving ~6-8GB RAM

## New Sweep (Attempt 2 — `soft-sweep-1`)
Restarted with `nohup` for terminal persistence. Epoch 1 completed in 26min. Memory stable at 14GB used with 9GB available during eval phase (when all 4 workers are alive). Swap stable at 5GB.

## Planned Next Steps
- Monitor sweep for successful completion (50 runs × ~40 min/run ≈ 33 hours)
- Analyze results when sweep finishes
- Clean up `diagnose_nan.py` from repo root
