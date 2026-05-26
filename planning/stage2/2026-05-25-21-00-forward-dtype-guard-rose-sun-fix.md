# 2026-05-25 21:00 — `forward()` dtype guard (rose-sun-108 analyze crash fix)

## Context

`rose-sun-108` supervised training (`run-20260525_113522-nudeo4mv`,
`train.py`, may24.yaml) completed 50 epochs and validation cleanly, then
crashed inside the post-validation `analyze(...)` call. The crash did not
corrupt the saved checkpoint at
`data/models/supervised/rose-sun-108.pt`, but did suppress the action
diagnostics output.

## Before State

After the 2026-05-22 `_dual_expand` dtype-flex patch, dtype handling for
bf16-collated states was split across three places:

1. `OptimizedBattleDataLoader` collate downcast states to bf16 for H2D
   bandwidth (b3cdb46).
2. `_dual_expand` / `_dual_expand_legacy` cast embedding outputs to
   `out.dtype` so eager `index_put` / `cat` would accept the result.
3. `evaluate(...)` in utils.py promoted `states` back to fp32 at the
   batch boundary before calling `model(states, masks)`.

The contract was implicit: *every eager caller of forward() is responsible
for promoting bf16 input to fp32*. `analyze(...)` was added/kept without
that promotion, so it inherited the same latent crash the May 22 patch
was supposed to solve.

## Problem

`analyze(...)` (utils.py:908-913) reads `batch["states"]` straight from
the dataloader without dtype promotion, then calls `model(states, masks)`
under `@torch.no_grad()`. The first `addmm` against fp32 weights raised:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

Validation succeeded earlier in the same run because `evaluate(...)` does
have the promotion. Two callers, two opposite contracts.

## Solution

Move the promotion into `TransformerThreeHeadedModel.forward` itself, so
every eager caller (analyze, evaluate, RL inference, future tooling) sees
one contract: bf16 in, fp32 path, fp32 out. Then delete the per-caller
promotion from `evaluate(...)` so there's a single source of truth.

```python
# model_archs.py TransformerThreeHeadedModel.forward (entry)
if x.dtype != torch.float32:
    x = x.float()
```

On the training hot path the input is bf16 (from collate) and gets cast
to fp32 once at entry; the next op runs under `autocast(bf16)` and
recasts. The wasted up-then-down cast is one elementwise op on a
~(B, T, 706) tensor — negligible vs. the encoder's 30M-param compute.

H2D bandwidth optimization is preserved: states still cross PCIe as bf16;
the promotion happens GPU-side after the transfer.

## Reasoning

Three options considered:

1. **Patch `analyze()` only** — one-line `.float()`. Fixes the immediate
   bug but leaves the implicit "each caller promotes" contract that
   already bit us twice (May 22, today). Rejected.
2. **Promote at `forward()` entry (chosen).** Single source of truth.
   Zero cost on RL inference (which always passes fp32). One unavoidable
   no-op cast on the training path (autocast recasts immediately after).
3. **Drop bf16 from the collate.** Undoes b3cdb46's H2D bandwidth
   halving. Bad tradeoff — supervised throughput still matters because
   we keep producing BC candidates that feed RL warm starts.

A guard variant (`if not torch.is_grad_enabled() and ...`) was considered
to avoid the wasted training cast, but rejected per user preference for
the simpler unconditional form.

## What Was Kept From Prior Patches

- `_dual_expand` / `_dual_expand_legacy` `.to(out.dtype)` casts in
  model_archs.py — defense-in-depth, no-op on fp32 inputs, removes a
  hidden assumption from the encoder.
- `test_dual_expand_bf16_input_matches_legacy` unit test — exercises
  encoder dtype robustness independent of any caller guard.
- `inductor_config.shape_padding = False` workaround in train.py:600-608
  — unrelated; still needed for inductor's pre-flight autotuner bug.

## Verification

- All 31 `test_model_archs.py` tests pass, including new
  `test_transformer_model_forward_accepts_bf16_input` which directly
  exercises the rose-sun crash path (bf16 input through eager forward
  with no autocast).
- `ruff check` / `ruff format --check` / `pyright` clean on the three
  edited files.

## Planned Next Steps

- Rerun `analyze(...)` against `rose-sun-108.pt` standalone (no need to
  retrain) to surface the diagnostics that the crash suppressed. This
  is a few minutes of work; not blocking.
- Decide whether `rose-sun-108` is a better RL warm start than the
  current `cool-bee-85-finetune`. Compare action scores from the analyze
  output once available.

## Updates

- 2026-05-25 21:00 — fix landed (this doc). Tests green.
