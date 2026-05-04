# Fine-Tune Script Modernization (2026-05-03)

## Context
`src/elitefurretai/supervised/fine_tune.py` was last touched in the LSTM era. It still imported `FlexibleThreeHeadedModel` and constructed it with LSTM-only kwargs (`lstm_layers`, `lstm_hidden_size`, `early_attention_heads`, `late_attention_heads`), which means it could not load any modern transformer checkpoint (including the Stage I handoff `curious-darkness-77_best.pt`).

## Before State
- `fine_tune.py` instantiated `FlexibleThreeHeadedModel` and would `KeyError` on a current config that lacks `lstm_*` keys.
- DataLoader hardcoded to `num_workers=7, prefetch=8` — slower than the optimal `nw=3, pf=2, wb=128` profile documented in `SUPERVISED.md`.
- No `state_input_dim` propagation → would crash silently when feature_set < FULL.
- Trusted saved `teampreview_idx` / `force_switch_indices` without recomputing — stale if feature schema evolves.
- No `torch.compile`, no `lr_schedule` config (always plateau), no numeric-type coercion for YAML strings, no `--save-best`, no post-training `analyze()`.
- `num_epochs` handling was visibly broken: `config["num_epochs"] = config.get("num_epochs", 10)` is a no-op; the docstring referenced an arg that no longer exists.
- `SUPERVISED.md` claimed fine-tune supports "freezes early layers (optional)" — never implemented.

## Problem
Running `fine_tune.py` against `curious-darkness-77_best.pt` would fail at construction time. Even if patched minimally, it would silently use a 2–3x slower DataLoader and ignore the modern training infra (compile, save-best, schedule choice).

## Solution
Rewrote `fine_tune.py` to be a thin wrapper around the existing `train.py` infrastructure:

1. **Model**: Construct `TransformerThreeHeadedModel` with `_build_transformer_kwargs(config)`. Load weights with `strict=False` to tolerate older checkpoints missing the distributional head.
2. **Reuse**: Imports `train_epoch` from `train.py`, `evaluate`/`analyze` from `utils.py` — no duplicated training loop logic.
3. **Hygiene**: Forces runtime device, recomputes embedder-derived indices, sets `state_input_dim`, computes `accumulation_steps` from `batch_size // worker_batch_size`.
4. **Type coercion**: Mirrors `train.py`'s `_FLOAT_KEYS`/`_INT_KEYS` so YAML overrides like `learning_rate: 5e-5` don't pass through as strings.
5. **Modern training niceties**: `torch.compile` on CUDA, `lr_schedule` choice (`cosine`/`plateau`), `--save-best`, post-training `analyze()`.
6. **Save-state correctness**: All `torch.save` calls use `raw_model.state_dict()` (not the compiled module) to avoid `_orig_mod.` prefixes in the saved keys.

Also updated `SUPERVISED.md` to drop the unimplemented "freezes early layers" claim and reflect the actual current behavior.

## Reasoning
- **Why `strict=False` on load**: matches the previous behavior and is necessary for upgrading older checkpoints to architectures with new buffers/parameters (e.g. distributional value head). Missing/unexpected keys are logged.
- **Why recompute indices**: Saved configs from older runs may have indices that no longer match the current `Embedder.feature_names` order. The model's weight shapes are stable across these schema tweaks (the embedder slices to a fixed `embedding_size`), but the *indices* used for training masks must come from the live embedder.
- **Why save `raw_model` not `model`**: `torch.compile` wraps the module so its `state_dict()` keys are prefixed with `_orig_mod.`. Saving the raw module keeps checkpoint format byte-compatible with `train.py`'s output, so `behavior_clone_player.py`, `analyze/*`, and RL initialization all continue to work without changes.
- **Out of scope (B, not C)**: The "freeze early layers" feature was kept out — it wasn't actually anywhere in the codebase before. Doc was corrected instead. If we want it later, it's a focused follow-up.

## Verification
- `ruff check`, `ruff format --check`, `pyright` all pass on the modified file.
- No call sites of `fine_tune.py`'s public symbols exist outside the script itself, so no downstream changes were needed.

## Planned Next Steps
- (When needed) Run a fine-tune of `curious-darkness-77_best.pt` against fresh data to validate end-to-end. Suggested config-override: small `num_epochs` (e.g. 5–10), reduced `learning_rate` (e.g. 1e-5–2e-5), keep architecture keys untouched.
- (Optional follow-up) Implement the freeze-early-layers feature mentioned in the original `SUPERVISED.md` if a use case arises (e.g. fine-tuning to a single team where early features are stable but heads need adaptation).

## Updates
*(none yet)*
