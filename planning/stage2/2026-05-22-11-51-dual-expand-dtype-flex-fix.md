# 2026-05-22 11:51 — `_dual_expand` made dtype-flexible (root fix for the bf16-states eval crash)

## Context

Three back-to-back BC training restarts (may22.yaml) crashed at the first
`evaluate(...)` call after epoch 1, each at a different point in the stack:

1. `pad_mm.should_pad_bench` (inductor pre-flight autotuner).
2. `extern_kernels.addmm` in the inductor-generated artifact.
3. `_dual_expand` index_put, after we switched eval to the eager path via
   `model._orig_mod`.

All three were symptoms of one thing: the model's input `states` tensor is
bf16, but the rest of the model has fp32 parameters and embedding outputs.

## Before State

- Commit `b3cdb46` (2026-05-22 10:17) added a `states.to(torch.bfloat16)`
  cast in `OptimizedBattleDataLoader`'s collate (battle_dataloader.py:37-42)
  to halve H2D bandwidth. Training tolerates this because the train forward
  runs under `torch.amp.autocast(bf16)` + `torch.compile`, both of which
  silently promote mismatched dtypes inside index_put / addmm.
- `evaluate(...)` runs under `@torch.no_grad()` with no autocast wrapper.
  When eval went through the compiled wrapper, inductor's codegen tolerated
  the mismatch (until it didn't — see (1) and (2) above). When eval was
  rerouted to the uncompiled module, eager strictness rejected the same
  mismatched index_put in `_dual_expand`.

The implicit dependency was: **the model only worked because of silent dtype
promotion inside torch.compile / autocast**. Any path that bypassed those
two — i.e., a normal eager forward with bf16 input — would crash.

## Problem

`GroupedFeatureEncoder._dual_expand` (model_archs.py:405) allocates its
output buffer with `dtype=x.dtype`, then writes embedding-lookup results
(always fp32) into it via index_put:

```python
out = torch.empty(B, T, layout["output_size"], dtype=x.dtype, device=x.device)
...
out[:, :, dst_idx] = emb.reshape(B, T, info["k"] * info["embed_dim"])
```

When `x.dtype == bf16`, eager PyTorch errors with:

```
RuntimeError: Index put requires the source and destination dtypes match,
got BFloat16 for the destination and Float for the source.
```

The legacy reference implementation `_dual_expand_legacy` has the same
flaw — it uses `torch.cat` over a mix of bf16 passthrough slices and fp32
embedding outputs, which is rejected by eager PyTorch identically.

## Solution

Cast embedding outputs to `out.dtype` (the input's dtype) before the
assignment in both the eid and number-bank loops. Same cast applied to
`_dual_expand_legacy`'s `parts.append(...)` calls so the two stay
bit-identical.

```python
out[:, :, dst_idx] = emb.reshape(B, T, info["k"] * info["embed_dim"]).to(out.dtype)
```

Net effect:
- fp32 input: out is fp32, emb is fp32 — the `.to(out.dtype)` is a no-op.
  Numerically identical to the previous behavior.
- bf16 input: out is bf16, emb (fp32 from embedding lookup) is downcast to
  bf16 before assignment. Same downcast happens implicitly under
  autocast(bf16) when the embedding output flows into the next Linear, so
  this matches training-time semantics.
- compiled and eager paths now both succeed without relying on
  inductor's silent promotion.

Test coverage added at
`unit_tests/supervised/test_model_archs.py::test_dual_expand_bf16_input_matches_legacy`:
runs both implementations against bf16 inputs, asserts dtype preservation,
and confirms exact equivalence between `_dual_expand` and
`_dual_expand_legacy` on bf16 inputs.

## Reasoning

Three alternatives considered:

1. **Always allocate `out` as fp32.** Forces extra cast on passthrough
   (bf16 → fp32) and doubles memory of the encoder's intermediate buffer.
   Rejected — adds bandwidth cost to the hot training path solely to
   benefit the rare eval path.
2. **Wrap `evaluate(...)` in `torch.amp.autocast(bf16)`.** Doesn't fix the
   bug: autocast doesn't intercept index_put / cat, so the dtype mismatch
   still hits eager strictness. Rejected.
3. **Cast emb to `out.dtype` (chosen).** Zero hot-path bandwidth cost,
   bit-identical for fp32 inputs, matches autocast-equivalent semantics
   for bf16 inputs. Removes the implicit dependency on compile-mode
   silent promotion.

The fp32 cast in `evaluate(...)` (utils.py:374-385) is kept as the
immediate functional fix and primary contract: eval still runs entirely in
fp32. The `_dual_expand` fix is defense-in-depth — it removes one of the
several assumptions the model had about its input dtype, so future
dataloader changes or autocast-mode shifts don't surface the same kind of
bug elsewhere.

## Planned Next Steps

None for this fix specifically — it's complete. Adjacent items that the
investigation surfaced and may be worth touching later:

- The rest of `TransformerThreeHeadedModel.forward` still assumes fp32
  weights × fp32 input or autocast bf16 promotion. A complete
  dtype-flexibility audit would check every Linear / Attention / LayerNorm
  call site. Not urgent — the dataloader cast only affects `states`, not
  parameters, and eval's explicit fp32 cast contains the blast radius.
- The `inductor_config.shape_padding = False` workaround in
  `train.py:600-608` is kept as belt-and-suspenders for the inductor
  pre-flight benchmark bug. It costs nothing on tensor-core-aligned
  shapes; once the upstream pad_mm autotuner is patched it can be
  removed.

## Updates

- 2026-05-22 11:51 — fix landed (this doc). All eight `dual_expand` unit
  tests pass (five existing fp32-equivalence cases + the new bf16 case).
  Ruff and pyright clean. Currently-running BC training (`sandy-firefly-98`,
  PID 15399) does not need a restart — it carries the eval-side fp32
  cast that already prevents the crash; the model_archs change is
  future-correctness.
