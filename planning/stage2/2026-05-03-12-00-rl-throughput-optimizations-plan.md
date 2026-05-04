# RL Throughput Optimizations — Implementation Plan

## Context

This plan implements five RL training speedups identified in a code+profile audit on
2026-05-03. The plan is staged from lowest-risk/highest-confidence to highest-effort,
with a gated benchmark protocol so each change is independently verified, committed
on success, and reverted on failure. The goal is months of training time saved.

**Baseline topology (from `2026-04-22-11-30-showdown-topology-winner-for-15-updates.md`):**
- `battle_backend: showdown_websocket`
- `num_workers: 6`, `num_players: 18`, `num_servers: 6`, `num_battles_per_pair: 16`
- `batch_size: 8`, `batch_timeout: 0.02`, `train_batch_size: 64`
- `max_battle_steps: 40`, transformer (7 layers, 16 heads, ff=2048), full featureset
- Observed: **1.61 b/s, 14.20 learner steps/s, ~40s/update steady state** at update 15.

**Changes in this plan, in execution order:**

| # | Change | Confidence | Expected speedup | Risk | Benchmark gate? |
|---|---|---|---|---|---|
| 1 | H10 — remove redundant `clip_grad_norm_` call | High | <1% | None | No (smoke only) |
| 2 | H8  — `AdamW(fused=True)` | High | <1% | None (CUDA only) | No (smoke only) |
| 3 | H7+H11 — vectorize collate + GAE | High | 5–10% | None | **Yes (b/s)** |
| 4 | H4 — PPO mini-epochs (K=2–4) | High | **2–3× updates/hr** | Off-policy drift if K too high | **Yes (quality only)** |
| 5 | H1 — KV-cache transformer context in actor | Medium-high | 1.5–2× actor inference | Algorithmic correctness — needs parity check | **Yes (b/s)** |

**Why the gate selection**:

- Items 1–2 are bundled, correctness-neutral, and any gain is below benchmark
  variance. Smoke run only; commit if smoke run is healthy.
- Item 3 is expected to move `b/s` measurably (collate runs every update on
  CPU). Throughput-gated.
- Item 4 will *not* move `b/s` — actor collection rate is unchanged. Its win
  is `updates/hour` at fixed `b/s`, which is predictable arithmetic
  (≈K× cheaper updates per battle, modulo per-epoch GPU time). The real risk is
  *learning quality*, so we gate on the 200-update quality run and skip the
  throughput benchmark.
- Item 5 directly attacks actor inference time, which is in the b/s hot path.
  Throughput-gated. Also requires a parity test before benchmarking.

## Before State

- `learners.py`: AdamW built without `fused=`, computes grad-norm twice per step
  (once with `inf` for monitoring, once with the real clip).
- `train.py:collate_trajectories`: Python loop over trajectories + Python GAE loop;
  many small `torch.tensor(...)` calls, then `.to(device)` on each.
- `learners.py:update`: single forward+backward per batch; no PPO inner-loop.
- `BatchInferencePlayer._run_batch` (transformer path): pads variable-length contexts
  and runs the **whole sequence** through the encoder every turn — O(T²) work per step.

## Problem

Actor data collection is the bottleneck (~95% of wall-clock per update). The learner
GPU is heavily idle, the transformer re-encodes its full history every turn during
actor play, and small CPU inefficiencies stack up across millions of steps over a
months-long run.

## Solution Overview

This plan does five things in sequence, each with a benchmark gate:

0. **Establish a reproducible benchmark and measure baseline variance.**
1. Apply trivial fixes (H10, H8). Verify they don't regress.
2. Vectorize `collate_trajectories` (H7+H11). Verify speedup.
3. Add PPO mini-epochs (H4). Verify learning quality preserved.
4. Implement transformer KV-cache (H1). Verify parity then speedup.

After every change: **run the benchmark protocol below; if statistically better,
commit; otherwise revert and document.**

---

## Phase 0 — Baseline & Variance (run before any code changes)

**Purpose**: produce a single trustworthy `b/s_mean` and `b/s_std` to compare
Phase 2 and Phase 4 against. Phases 1 and 3 do not gate against this.

### 0.1 Benchmark configuration

Create a frozen benchmark config at
`src/elitefurretai/rl/configs/benchmark_throughput_baseline.yaml` derived from the
current topology winner. Keep these locked across all phases:

- `battle_backend: showdown_websocket`
- `num_workers: 6`, `num_players: 18`, `num_servers: 6`, `num_battles_per_pair: 16`
- `batch_size: 8`, `batch_timeout: 0.02`, `train_batch_size: 64`
- `max_battle_steps: 40`
- `max_updates: 30` (long enough that startup is amortized; short enough for ~5 reruns)
- `use_wandb: false` (avoid network noise; we'll parse train.py logs directly)
- `train_exploiters: false`, `auto_launch_external_vgcbench: false`
- Same `initialize_path` (BC checkpoint) every run for identical model init
- Same RNG seed if available (add a `seed` field if not yet present, or accept observed variance)
- Curriculum: same as winner doc (no VGCBench since it's noisy on networking)

### 0.2 Benchmark harness

Add `scripts/run_throughput_benchmark.sh` (excluded from lint per CLAUDE.md):

```bash
#!/usr/bin/env bash
# run_throughput_benchmark.sh <run_label> <n_repeats>
# Captures per-update b/s, learner steps/s, time/update, and a single summary
# row at the end. Writes to data/benchmarks/<date>_<label>/run_<i>.log
```

Behavior:
- Run `python -m elitefurretai.rl.train --config benchmark_throughput_baseline.yaml`
- Parse trainer's `Update N: ...` lines. Skip first 3 updates (warmup).
- Compute mean + std of: `b/s` from "Total Battles=X in Ys (Z b/s)" and `learner steps/s`
- Repeat `n_repeats` times (default 5).
- Emit `summary.json` with per-run and aggregate stats.

### 0.3 Statistical decision rule (applies to Phase 2 and Phase 4 only)

For Phase 2 and Phase 4, compute aggregate `b/s_mean` and `b/s_std` over warm
updates, across N≥5 reruns. Decision rule for "real improvement":

- **Promote (commit)** if `(new_mean - old_mean) > 1.5 × max(old_std, new_std)` AND new_mean ≥ old_mean × 1.05 (i.e., ≥5% improvement *and* outside variance band).
- **Reject (revert)** if new_mean < old_mean − 1.0 × max(old_std, new_std).
- **Inconclusive** (between thresholds): document, keep change only if it has zero downside (e.g., correctness improvement), otherwise revert.

The 1.5σ threshold is deliberately conservative for a months-long training cost.

Phase 1 commits on a smoke run only (no benchmark — gains are below noise).
Phase 3 uses a *quality* gate (200-update win-rate trajectory), not a throughput
gate, since K> 1 doesn't change `b/s`.

### 0.4 Run baseline

```
./scripts/run_throughput_benchmark.sh baseline 5
```

Expected output: `data/benchmarks/2026-05-DD_baseline/summary.json` containing
`b/s_mean`, `b/s_std`, `steps_per_s_mean`, `steps_per_s_std`, `time_per_update_mean`.
Record these as **the** baseline numbers. All future phases compare against this.

---

## Phase 1 — H10 + H8: Trivial wins, bundled

### 1.1 Changes

**`src/elitefurretai/rl/learners.py`**

- In `_build_optimizer`, when `opt.type == "adamw"` and `device == "cuda"`,
  pass `fused=True` to `optim.AdamW(...)`. Skip on CPU (fused needs CUDA).
- In `update()`, replace the double `clip_grad_norm_` pattern:
  ```python
  grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")).item()
  grad_norm_after  = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip).item()
  ```
  with a single call:
  ```python
  grad_norm_before = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip).item()
  grad_norm_after  = min(grad_norm_before, self.gradient_clip)
  ```
  Note: `clip_grad_norm_` returns the **pre-clip** norm and clips in place; the
  post-clip norm is mathematically `min(pre, max_norm)`. Apply this to both the
  AMP and non-AMP branches.

### 1.2 Verify (smoke only, no benchmark)

- `ruff check src unit_tests`
- `ruff format src unit_tests --check`
- `pyright src unit_tests`
- `pytest unit_tests/rl -q` (smoke that learner update still runs)
- Run a **3-update** sanity training run; confirm:
  - `grad_norm_before_clip` and `grad_norm_after_clip` are sensible
    (`grad_norm_after_clip == min(grad_norm_before_clip, max_grad_norm)`).
  - Losses are finite and in the same ballpark as a recent run.
  - No new warnings about `fused` not being applied (CUDA path).

### 1.3 Decision

- Phase 1 fixes are correctness-neutral. **Commit if smoke run is healthy.**
- No throughput benchmark — expected gain is below variance band, running it
  would only generate noise.
- If smoke run shows anything anomalous (NaN losses, wrong grad-norm values,
  fused warnings), revert and document.

### 1.4 Commit message (on success)

```
H8+H10: fused AdamW on CUDA + single grad-norm clip call

- AdamW(fused=True) when device=cuda; foreach path on CPU.
- Replaced double clip_grad_norm_ (one for monitoring, one for clipping)
  with a single in-place call; post-clip norm is min(pre-clip, max_norm).

No measurable throughput change expected; included for cleanliness and for
future learner-side scaling work.
```

---

## Phase 2 — H7+H11: Vectorize collate + GAE

### 2.1 Changes

**`src/elitefurretai/rl/train.py:collate_trajectories`**

Replace the current implementation with:

1. **Pre-allocate one numpy buffer** for states: `np.zeros((B, T, dim), dtype=np.float32)`.
   Fill via `np.stack` per trajectory → assign slice. (Avoids `torch.tensor(np.array([...]))`
   inside the inner loop.)
2. **Vectorize GAE**: build `rewards`, `values`, `padding_mask` as `(B, T)` tensors first;
   then compute deltas and GAE in a single reverse-time loop over `T` (not over `B × T`):

   ```python
   advantages = torch.zeros_like(rewards)
   gae = torch.zeros(B)
   for t in reversed(range(T)):
       not_done = padding_mask[:, t].float()
       next_val = values[:, t+1] if t+1 < T else torch.zeros(B)
       delta = rewards[:, t] + gamma * next_val * not_done - values[:, t]
       gae   = delta + gamma * gae_lambda * gae * not_done
       advantages[:, t] = gae
   returns = advantages + values
   ```

   This is `O(T)` Python loop with `O(B)` vector ops per step instead of `O(B*T)`
   Python iterations.
3. **Move to device once** at the end: `result = {k: v.to(device, non_blocking=True) for k, v in cpu_dict.items()}`.
4. **Action masks**: build a single `(B, T, mask_dim)` numpy buffer rather than
   per-step `torch.tensor`. Default to `1.0` and overwrite where masks exist.

Keep the existing truncation logic (`max_seq_len`) and metadata extraction unchanged.

### 2.2 Correctness check

Add `unit_tests/rl/test_collate_trajectories.py` (or extend if it exists):

- Build a fixed seed list of 8 trajectories with known rewards/values.
- Run new `collate_trajectories` and a frozen reference implementation that
  uses the old per-trajectory loop on CPU.
- Assert tensor-by-tensor equality on `states`, `actions`, `advantages`, `returns`,
  `padding_mask`, `masks` (with `torch.allclose(..., atol=1e-5)` for floats).

This is the gate for Phase 2: **the test must pass before benchmarking**. If
the new code produces different numbers, GAE in the learner has different
gradients and downstream comparisons become invalid.

### 2.3 Verify

- `pytest unit_tests/rl/test_collate_trajectories.py -q`
- Quality gates: `ruff check`, `ruff format --check`, `pyright`, `pytest unit_tests -q`

### 2.4 Benchmark

```
./scripts/run_throughput_benchmark.sh phase2_collate 5
```

### 2.5 Decision

- **Promote** if `b/s_mean` improves by ≥5% and ≥1.5σ.
- **Reject and revert** otherwise. Document in this file under "Updates" why
  collate vectorization didn't help (e.g., GAE was not a bottleneck on this
  batch size).

### 2.6 Commit message (on success)

```
H7+H11: vectorize collate_trajectories and GAE

- Pre-allocate (B, T, dim) numpy buffer for states; one .to(device) per tensor.
- GAE replaced with single reverse-T loop using (B,) vector ops.
- Added test_collate_trajectories parity test against frozen reference.

Benchmark: <X% b/s improvement, <Y% time/update reduction>, σ=<Z>.
```

---

## Phase 3 — H4: PPO mini-epochs

### 3.1 Design notes

The current `learner.update()` does one forward+backward per batch. Standard PPO does
4–10. Each extra epoch increases the **policy ratio drift** from old log-probs, so the
clip range matters more. We add a config knob `ppo_epochs` (default 1 → identical
behavior) with a recommended starting value of 3.

We do **not** mini-batch within the batch (`train_batch_size=64` is already small).
Each epoch processes the full batch once.

The **reference forward passes are not repeated per epoch** (they are frozen — the
KL target stays fixed). This requires caching `ref_outputs` once before the inner loop.

### 3.2 Changes

**`src/elitefurretai/rl/config.py`**

- Add `ppo_epochs: int = 1` to algorithm section.
- Add `ppo_kl_early_stop: float | None = None` (optional safety: stop epochs early
  if approximate KL between current and old policy exceeds this).

**`src/elitefurretai/rl/learners.py:update`**

Restructure the body of `update()`:

```
# Once per update (not per epoch):
- compute initial_hidden_state
- forward main model and ref models
- store ref_logits_per_head (frozen)

# Per epoch:
for epoch in range(self.config.algorithm.ppo_epochs):
    forward main model only  # NOT ref models
    compute losses (PPO ratio uses original old_log_probs from batch)
    backward + step
    optionally early-stop on approx KL
```

Critical:
- Old log probs (`flat_old_log_probs`) come from the actor at collection time and
  do **not** update across epochs. This is correct PPO behavior.
- Reference logits are computed once and reused.
- Mixed-precision GradScaler step is per-epoch.
- LR scheduler steps **once per update**, not per epoch.
- The `_step` counter and `temperature` annealing track *updates*, not epochs.

### 3.3 Correctness check

- With `ppo_epochs=1`, the new code path must be bit-equivalent to the old. Add
  a test that runs both old and new paths against a fixed batch and asserts
  `torch.allclose` on output logits and grad norms.
- With `ppo_epochs > 1`, validate that `total_loss` and `policy_loss` are
  finite, that ratio doesn't NaN out, and grad-norm stays bounded. Run a 5-update
  sanity run.

### 3.4 Verify

- `pytest unit_tests/rl -q`
- Quality gates as above.
- Smoke run: 5 updates with `ppo_epochs=3`, confirm losses and win rates look reasonable.

### 3.5 Benchmark protocol (quality only, no throughput gate)

PPO mini-epochs does not change `b/s` (collection rate is unchanged); it
multiplies *learning per battle*. The arithmetic is predictable:

```
time/update_K = time/update_K=1 + (K-1) * forward_backward_time
updates/hour ≈ K × baseline_updates/hour  (for K up to ~5, since learner is idle)
```

So the throughput benchmark would only confirm something we already know.
Skip it. The real risk is **learning quality**: too-large K causes off-policy
drift and can collapse training.

**Learning quality benchmark (200 updates × 3 reruns each):**

```
./scripts/run_throughput_benchmark.sh phase3_quality_K1 3 --max-updates 200
./scripts/run_throughput_benchmark.sh phase3_quality_K3 3 --max-updates 200
./scripts/run_throughput_benchmark.sh phase3_quality_K4 3 --max-updates 200
```

For each run, record at update 50, 100, 150, 200:
- `bc_player` win rate (from `Update N: ... Win rates:` lines)
- `simple_heuristic_baseline` win rate
- `policy_loss`, `rnad_loss`, `value_loss`
- wall-clock seconds elapsed

Plot win rate vs **wall-clock time** (not vs update number). K=3 should reach
the same win rate in <1/2 the wall-clock time of K=1.

### 3.6 Decision

- **Promote K=3** as the new default if at every checkpoint (50/100/150/200
  updates) its win rate vs wall-clock is ≥ K=1's, and losses don't diverge.
- **Promote K=4** instead if it dominates K=3 on both axes.
- **Reject** if win-rate-vs-wall-clock regresses for any K, or if losses go
  unstable. Keep `ppo_epochs` config knob (default 1) and document the failure.

### 3.7 Commit message (on success)

```
H4: PPO mini-epochs (ppo_epochs=K)

- Added algorithm.ppo_epochs (default 1; recommended K=<3|4>).
- Reference model forward passes computed once per update; main model
  recomputed per epoch. Optional ppo_kl_early_stop guard added.
- Old log probs from actor reused across epochs (standard PPO).

Quality benchmark (200 updates × 3 reruns):
- bc_player win rate at update 200: <Y%> @ <T> minutes wall-clock
  (vs K=1 baseline: <Y'%> @ <T'> minutes).
- simple_heuristic at update 200: <Z%> @ <T> minutes (vs <Z'%> @ <T'>).
- Effective wall-clock speedup at equivalent quality: <X×>.
```

---

## Phase 4 — H1: KV-cache transformer context in actor

### 4.1 Design

The transformer currently re-encodes the full context every turn:
`forward_with_hidden(x, context) → cat(context, x) → encoder(full_seq)`. With T=30
turns, total work over a battle is O(T³) for the action selection path.

KV-cache approach: per battle, keep the cached **key/value tensors per layer** for
all past tokens. On each turn, run only the new token through the encoder using
those cached K/V (O(T) per step, O(T²) per battle).

PyTorch's `nn.TransformerEncoder` does **not** natively support KV-cache. Two options:

- **Option A**: Replace the encoder with a custom MultiHeadAttention path that
  accepts and returns past_kv. More code, more correct, fastest.
- **Option B**: Keep the encoder but reuse outputs by feeding only the new token's
  query against the *cached* key/value sequence at each layer (still O(T) per layer
  but with simpler glue). Almost as fast, less invasive.

This plan picks **Option B** because it preserves the trained weights exactly and is
easier to reason about. If profiling after Option B shows there's still meaningful
gain on the table, escalate to Option A in a separate phase.

### 4.2 Changes

**`src/elitefurretai/supervised/model_archs.py:TransformerThreeHeadedModel`**

- Add `forward_with_kv_cache(self, new_token_features, past_kv) -> (turn_logits,
  tp_logits, value, win_dist_logits, new_past_kv)`:
  - For each transformer layer, manually call self-attention with `past_kv[layer]`
    appended to keys and values.
  - Append the new K/V to `past_kv[layer]` to return.
  - The decision-token mechanics (`[ACTOR]`, `[CRITIC]`, `[FIELD]`) need extra
    care: those tokens currently sit at the front of the context. With KV-cache,
    we can keep them in `past_kv[0]` cached as well, since they're learned and
    constant — but their **output** at each turn depends on the new token. Keep
    them as part of the cached sequence; just include them in K/V.
- Keep the existing `forward_with_hidden` path intact (used by training where the
  full sequence is needed for the gradient).

**`src/elitefurretai/rl/players.py:BatchInferencePlayer`**

- For transformer path, replace the `hidden_states[tag]` semantics: store
  `kv_cache_per_battle[tag] = list-of-(K, V)-per-layer` instead of the growing
  context tensor. On each `_run_batch`, batch the *current step* only and call the
  new `forward_with_kv_cache`.
- The current code already pads variable-length contexts (the slow path); replace
  this with a per-battle cache that grows independently. Padding only needed if
  we batch across battles for the encoder layers — which we still want.

  Implementation note: Different battles have different cache lengths in the same
  worker. Pad keys/values to the max-cached-length in the batch, mask attention
  to the real length. This is the same masking idiom already used in
  `forward_with_hidden` (`hidden_mask`), just applied at each layer.

### 4.3 Parity test

This is the riskiest phase. Add `unit_tests/rl/test_kv_cache_parity.py`:

- Generate a synthetic 10-turn trajectory.
- Run the model two ways:
  1. The original `forward_with_hidden` once at the end with the full context.
  2. The new `forward_with_kv_cache` 10 times, threading `past_kv`.
- For the output at the *final* turn, assert
  `torch.allclose(out_full, out_kv, atol=1e-4)` for each of `turn_logits`, `tp_logits`, `value`, `win_dist_logits`.

This must pass before benchmarking.

Also add a small batched parity test: 4 battles each at different turn counts (3, 5, 7, 10), batched into one `forward_with_kv_cache` call vs four sequential `forward_with_hidden` calls, all close.

### 4.4 Verify

- `pytest unit_tests/rl/test_kv_cache_parity.py -q`
- Quality gates as above.
- 5-update smoke run; confirm losses, win rates look identical to baseline.

### 4.5 Benchmark

```
./scripts/run_throughput_benchmark.sh phase4_kvcache 5
```

Compare against the latest committed baseline (post-Phase 3).

### 4.6 Decision

- **Promote** if `b/s_mean` improves by ≥10% and ≥1.5σ AND the parity tests are
  green AND the 5-update smoke run shows identical losses to baseline.
- **Reject and revert** otherwise. Particularly: if win rates differ from
  baseline at update 5 by more than noise, this is a correctness issue, not a
  speed issue — revert immediately and document the bug.

### 4.7 Commit message (on success)

```
H1: KV-cache transformer context in actor inference

- Added TransformerThreeHeadedModel.forward_with_kv_cache for online actors.
- BatchInferencePlayer keeps per-battle past_kv instead of growing context.
- Training path (forward_with_hidden) unchanged; gradients still flow through
  the full sequence.
- Added test_kv_cache_parity to assert outputs match the recompute path.

Benchmark: <X%> b/s improvement, transformer actor forward time <Y%>
of previous; learner-side throughput unchanged.
```

---

## Reasoning

- **Why this order**: each change strictly multiplies on the previous, and earlier
  changes are lower-risk. Trivial fixes go first to clear the deck. Vectorized
  collate is pure infrastructure cleanup. PPO mini-epochs is the single biggest
  win and must be tested with quality, not just throughput. KV-cache is the
  riskiest because it touches model code; saving it for last keeps prior gains
  in the bag if it goes sideways.
- **Why benchmark gates**: this is a multi-month training run. Committing a
  "looks-fine" change that actually regresses by 3% costs hours per day. The
  1.5σ rule is conservative on purpose.
- **Why parity tests for 2 and 4**: they change *math* the learner depends on.
  Sanity tests are cheaper than chasing a "loss looks different" bug 30k updates in.
- **Why we don't do H6 (drop engineered features) or H3 (smaller model) here**:
  those are blocked on the BC ablation study and don't need a separate plan
  beyond switching the BC checkpoint when A2/A3 results land.

## Planned Next Steps / Implementation Plan

1. Phase 0: write the benchmark harness, run baseline 5×, record `b/s_mean` and
   `b/s_std`. (Used by Phase 2 and Phase 4 only.)
2. Phase 1: trivial fixes, smoke run, commit if healthy. **No benchmark.**
3. Phase 2: vectorize collate, run parity test, run benchmark, commit or revert.
4. Phase 3: PPO mini-epochs — 200-update quality benchmark for K∈{1,3,4}.
   **Quality gate, no throughput gate.**
5. Phase 4: KV-cache implementation, parity tests, run benchmark, commit or revert.
6. Update RL.md "Performance & Optimization" section with the final realized
   speedups from this run.
7. After all phases: run the full benchmark harness once more on the final
   stack and compare against the Phase 0 baseline for a single "post-optimization"
   number to record.

## Updates

*(Filled in as each phase completes — record `b/s_mean`, `b/s_std`, decision,
commit hash or revert note.)*

- **Phase 0 baseline** (b/s_mean, b/s_std): `TBD`
- **Phase 1 (H10+H8)** (smoke result, commit hash): `TBD`
- **Phase 2 (H7+H11)** (b/s delta, parity test, commit hash): `TBD`
- **Phase 3 (H4)** (chosen K, win-rate@200 vs wall-clock, commit hash): `TBD`
- **Phase 4 (H1)** (b/s delta, parity test, commit hash): `TBD`
- **Final stack vs. Phase 0 baseline** (b/s_mean delta, updates/hour delta): `TBD`
