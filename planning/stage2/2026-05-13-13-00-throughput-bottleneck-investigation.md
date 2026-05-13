# Throughput Bottleneck Investigation — Living Document

**Started**: 2026-05-13
**Status**: Active investigation
**Owner**: Cayman + Claude

This is the central source of truth for everything we've learned about RL training
throughput on EliteFurretAI. Append updates as new data arrives; do not delete
disproven hypotheses (mark them resolved). When in doubt about the current
training state, this doc trumps memory — verify against the live system first.

---

## TL;DR

- Current throughput: **3 traj/s, 51 learner-steps/s** (sep_arch run, PID 22810).
- VGCBench (peer reference): **15–20 traj/s** (~5–7x ahead).
- We are **rollout-bound**, GIL-limited within each worker, and the hot path
  is a Python-loop-over-tensor-positions inside the model encoder
  (`_dual_expand` at `model_archs.py:381`). Featurization in
  `etl/embedder.py` is a secondary ~30% cost. Showdown protocol round-trips
  are essentially free.
- The single highest-leverage optimization is **vectorizing `_dual_expand`**
  (32% own-time per worker).

---

## Confirmed findings (with evidence)

### F1. Pipeline is rollout-bound, not learner-bound
**Evidence**: `nvidia-smi dmon -s u` shows SM% mostly 0–25% with bursts to 88%.
The bursts are learner steps; the long flat stretches are inference + idle.
Memory bandwidth column is 1–3% — small batches, not big tensor moves.

**Implication**: GPU has huge headroom. Optimizing the learner step itself
would yield ~zero throughput gain. All wins come from rollout side.

### F2. Each worker process is single-core GIL-bound
**Evidence**: `ps`/htop shows 4 worker processes (PIDs 23469–23472 in current
sep_arch run) each at 99% CPU. `py-spy top --pid 23469` reports
`GIL: 44%, Active: 138%, Threads: 5`. Of 5 threads, only one is active at
any moment.

**Implication**: Adding threads to a worker won't help. Multi-processing
already exploited (4 workers); further parallelism must either (a) reduce
per-worker GIL pressure or (b) add more processes (CPU-bound).

### F3. Inference runs *inside each worker process*, not centrally
**Evidence**: py-spy dump shows hot thread is `worker0_inference_0` running
`_gpu_inference_sync` (`players.py:506`). Each of the 4 workers has its own
model copy and its own batcher.

**Implication**: 4× model copies in GPU memory. Each worker's batcher only
sees its own assigned battles (~96 of 384 max). Batch fill is structurally
limited per-worker. This is the key architectural difference from
VGCBench's `SubprocVecEnv` (one centralized model, one big batched forward).

### F4. `_dual_expand` is the single hottest function (32% own time)
**Evidence**: py-spy top output ranks `_dual_expand` (`model_archs.py:381`)
first by OwnTime at 32% (28.01s out of ~88s observed).

**Code inspection** (read 2026-05-13): the function loops over all
"replacement positions" in a feature tensor and, per position, slices the
input, looks up the right embedding bank, computes a bucket index from
clamped float values, and appends to a Python list. Then `torch.cat` at the
end. Per-position Python overhead, ad-hoc tensor slicing, and serial
embedding calls all hit the GIL.

**Implication**: Vectorizing this — precompute the position layout at init,
batch embeddings of the same bank into one lookup, use `torch.bucketize`
instead of per-position arithmetic — should significantly reduce the
worker's CPU cost.

### F5. Featurization (`etl/embedder.py`) costs ~30% own time
**Evidence**: Sum of OwnTime for `_generate_null_move_features` (7%),
`<genexpr>` (6%), `generate_opponent_pokemon_features` (5%),
`generate_pokemon_features` (4%), `embed` (8%) = ~30%.

**Implication**: Even if model inference were free, featurization alone
would cap throughput. Vectorization with numpy (or eventual Rust port) is
on the table.

### F6. Showdown protocol layer is essentially free
**Evidence**: `_handle_battle_message` (poke-env) shows 34% **TotalTime**
but **0% OwnTime**. All cost is dispatched into featurization + inference.

**Implication**: Running 4 separate Showdown servers (one per port) is
likely unnecessary CPU/memory cost. A single server with concurrent battles
should be sufficient. Cheap to test (drop `num_servers` to 1).

### F7. `max_concurrent_battles_per_player=32` was a major win
**Evidence**: Removed poke-env's default `_battle_count_queue` saturation
(max=1) which serialized battle setup phases. Throughput jumped from
~0.5 traj/s pre-change to 3 traj/s after.

**Implication**: Already shipped; this lever is exhausted at the current
value. Could check whether even higher (64+) helps, but unlikely to be
significant.

---

## Architectural comparison: us vs VGCBench

| | VGCBench | EliteFurretAI |
|---|---|---|
| Model width | `d_model=256` (~2–5M params) | 27M params (cool-bee-85 derived) |
| Inference pipeline | SB3 `SubprocVecEnv` — one shared model, batched across all envs | Per-worker `BatchInferencePlayer` with async batcher |
| Model copies | 1 | `num_workers` (=4) |
| Update batch | `batch_size=512` | `train_batch_size=256` |
| Showdown servers | 1 | 4 |
| Action head | Flat masked action distribution | Autoregressive turn head with decision tokens |
| Hardware | H100 + likely 16–32 cores | RTX 3090 + 8 cores WSL2 |

The 5–7x gap is plausibly explained by: smaller model (3–5x), centralized
inference (1.5–2x), with hardware/cores worth a small remainder. None of
these are individually impossible to close.

---

## Hypotheses (open and resolved)

### Open

| ID | Hypothesis | Confidence | How we'd test |
|---|---|---|---|
| H1 | Vectorizing `_dual_expand` cuts worker CPU by 15–25% | High | Implement + benchmark single worker |
| H2 | `batch_size: 4 → 32` improves throughput 1.5–2x | High | Add fill-size logging, then bump and measure |
| H3 | `num_servers: 4 → 1` is throughput-neutral; reclaims CPU for ~1.1x | Medium | Single-knob ablation run |
| H4 | `torch.compile(model, mode='reduce-overhead')` on inference cuts GIL hold by 30%+ | Medium | One-knob run after vectorizing dual_expand |
| H5 | Vectorizing `embedder.py` (esp. the `<genexpr>` hot spot) cuts featurization by 2x | Medium | Profile-guided rewrite of the generator |
| H6 | Centralized inference (à la `SubprocVecEnv`) closes ~half the gap to VGCBench | High | Significant refactor — defer until cheap wins exhausted |
| H7 | Distilling a small (2–5M param) "rollout-only" model gives 2–4x on rollouts with no learner impact | Medium | Train distillation, A/B with current architecture |
| H8 | Rust backend for self-play closes most of the remaining gap | High (but unverified for RL training) | Wire up Rust backend to `worker.py`'s battle stepping path |

### Resolved (do not remove)

| ID | Hypothesis | Resolution |
|---|---|---|
| R1 | "GPU compute is the bottleneck" | **Disproven** (F1). SM% mostly idle. |
| R2 | "Showdown JS round-trips are slow" | **Disproven** (F6). `_handle_battle_message` is 0% OwnTime. |
| R3 | "More threads per worker will help" | **Disproven** (F2). GIL caps Active at ~138% regardless of thread count. |
| R4 | "Inference is centrally batched across workers" | **Disproven** (F3). Per-worker batchers. |

---

## Validated changes already shipped

| Date | Change | Effect |
|---|---|---|
| 2026-05-12 | `max_concurrent_battles_per_player: 1 → 32` (configurable) | 6x throughput jump (~0.5 → 3 traj/s) |
| 2026-05-12 | Memory watchdog (`memory_watchdog_threshold_gb` in TrainingConfig) | Prevents WSL2 OOM crashes; no perf impact |
| 2026-05-12 | Log level WARNING in workers (was INFO) | Cuts log file size 100x, minor CPU saving |
| 2026-05-13 | Worker model loads use `strict=False` (sep_arch) | Enables partial-load ablations |

---

## Queued experiments (in priority order)

| # | Experiment | Expected gain | Effort | Status |
|---|---|---|---|---|
| 1 | Vectorize `_dual_expand` (precompute position layout, batch per-bank embeddings) | 1.5–3x worker throughput | 1–2 days | Not started |
| 2 | Add inference batch-size logging (avg, p50, p95, timeout rate) | 0x (diagnostic) | 1 hour | Not started |
| 3 | Bump `batch_size: 4 → 32` after (2) confirms batches fill | 1.5–2x | 1 hour | Pending (2) |
| 4 | `num_servers: 1` ablation | 1.1–1.3x | 5 min | Not started |
| 5 | `torch.compile` inference model | 1.5–2.5x | 1 day | Pending (1) |
| 6 | Vectorize `embedder.py` `<genexpr>` and `_generate_null_move_features` | 1.5–2x featurization | half day | Not started |
| 7 | Centralized inference refactor (à la VGCBench) | 2–3x overall | 1–2 weeks | Not started |
| 8 | Rust featurization (PyO3 binding for `etl/embedder.py` hot paths) | 1.5–2x | 2–4 weeks | Not started |
| 9 | Distilled rollout model (2–5M params) | 2–4x | 1 week training + integration | Not started |
| 10 | Switch self-play battle execution to Rust backend | 3–10x | Backend already exists; integration work unknown | Not started |

---

## How to use this doc

**When starting a session on throughput**: read this top-to-bottom. The TL;DR
+ Confirmed findings is the authoritative current state.

**After running an experiment**: append an entry to "Validated changes" or
mark a hypothesis resolved. Update the TL;DR throughput number if it
moved. Record the wandb run name and PID for traceability.

**When learning something new**: add a Confirmed finding (with evidence) or
an Open hypothesis. Don't rewrite history — append.

**When a hypothesis is disproven**: move it to Resolved with a one-line
note. Don't delete it; future-us needs to know what we already ruled out.

---

## Updates log

### 2026-05-13 13:00 — Initial diagnosis
- Profiled sep_arch run (wandb `ovcxrhy9`, PID 22810/23469).
- Ran `nvidia-smi dmon` → confirmed rollout-bound (F1).
- Ran `py-spy dump --pid 23469` → confirmed per-worker inference (F3) and
  `_dual_expand` as hot frame.
- Ran `py-spy top --pid 23469` 100s sample → ranked F4, F5, F6 by OwnTime;
  measured GIL=44%, Active=138% (F2).
- Read `_dual_expand` at `model_archs.py:381` → confirmed Python loop
  pattern, candidate for vectorization (H1).
- Created this doc.

