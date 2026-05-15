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

- **Current "exploiters off" throughput: ~5.0 traj/s** (natural-experiment
  observation during 2026-05-14 session; equivalent to post-merge 4.98
  baseline). With exploiters on, throughput drops to ~3.8 traj/s — see F11.
- Run B (topology bump with exploiters off) launching from
  [sep_arch.yaml](../../src/elitefurretai/rl/configs/sep_arch.yaml):
  `num_players: 12→16`, `max_concurrent_battles_per_player: 32→48`,
  `num_battles_per_pair: 32→48`. Hypothesis: more in-flight battles
  per existing worker grow svc=main batch fill above the 5-7 average,
  lifting traj/s above 5.0.
- Earlier baseline data point: **3.78 traj/s** (wandb `glad-lake-19`,
  PID 19534, centralized inference + torch.compile, batch avg 7.2 / max
  32 / cap 32) — superseded by the post-cleanup 4.98 measurement and
  the 5.0 natural experiment.
- VGCBench (peer reference): **15–20 traj/s** (~4–5x ahead).
- **Bottleneck has shifted to the inference service** (post-M4). Workers
  are essentially idle (Active 0% in py-spy) waiting on inference
  responses; the inference service runs at 90% Active inside the main
  train.py process and is **CPU/Python-bound, not GPU-bound** (GPU avg
  SM ≈ 22–25%, plenty of headroom). New top frame: `_slice_next_hidden`
  at 13% OwnTime in `inference_handlers.py` (the F9 hidden-state fix
  paying a real per-request Python cost).
- **Batch fill is real but underwhelming**: avg ≈ 7.2 (vs pre-M4
  per-player cap of 4 — improvement), but well below the predicted
  18–48. 99.8% of batches flush via 5ms timeout; only 0.2% reach cap.
  The service finishes batches faster than requests arrive — bumping
  `batch_timeout` (currently 5ms) toward 10–15ms is a cheap candidate.
- **Engine choice (Rust vs Showdown) is deprioritized.** Worker idle
  data confirms workers are not engine-bound — they're inference-
  response-bound. A faster engine just enqueues into the same Python
  ceiling. H8 moved to resolved/deprioritized.
- **Highest-ROI next moves**: (a) `batch_timeout` bump, (b) vectorize
  `_slice_next_hidden`, (c) distillation (Plan D — smaller model directly
  reduces per-call Python cost in the service), (d) Plan C registry
  parallelism (multiple service processes sidestep single-GIL ceiling).

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

### F4. `_dual_expand` is the single hottest function (32% own time) — RESOLVED 2026-05-13
**Evidence**: py-spy top output ranks `_dual_expand` (`model_archs.py:381`)
first by OwnTime at 32% (28.01s out of ~88s observed).

**Code inspection** (read 2026-05-13): the function loops over all
"replacement positions" in a feature tensor and, per position, slices the
input, looks up the right embedding bank, computes a bucket index from
clamped float values, and appends to a Python list. Then `torch.cat` at the
end. Per-position Python overhead, ad-hoc tensor slicing, and serial
embedding calls all hit the GIL.

**Resolution (2026-05-13)**: vectorized version shipped. Post-vectorization
profile shows `_dual_expand` at 6% OwnTime (down from 32%); in-training
throughput up 13%. See "Updates log" 2026-05-13 14:38 for the full
re-profile.

### F5. Featurization (`etl/embedder.py`) costs ~30% own time
**Evidence**: Sum of OwnTime for `_generate_null_move_features` (7%),
`<genexpr>` (6%), `generate_opponent_pokemon_features` (5%),
`generate_pokemon_features` (4%), `embed` (8%) = ~30%.

**Implication**: Even if model inference were free, featurization alone
would cap throughput. Vectorization with numpy (or eventual Rust port) is
on the table.

### F6. Showdown protocol layer is essentially CPU-free, but has finite per-second QPS
**Evidence**: `_handle_battle_message` (poke-env) shows 34% **TotalTime**
but **0% OwnTime**. All cost is dispatched into featurization + inference.

**Implication (initial, partly wrong)**: Running 4 separate Showdown
servers (one per port) is likely unnecessary CPU/memory cost. A single
server with concurrent battles should be sufficient.

**Update 2026-05-13 16:01 — `num_servers: 1` killed the run.** Tested
`num_servers: 1` while keeping `num_players: 12` and
`max_concurrent_battles_per_player: 32` (= 384 concurrent battle streams
funneled through a single node Showdown server). The server saturated
on per-second message capacity; workers stalled out and the
zero-completion watchdog correctly aborted after 15 consecutive empty
batches. `_handle_battle_message`'s 0% OwnTime was per-WORKER CPU
cost, **not** Showdown's per-second throughput capacity. **Reverted to
`num_servers: 4`.** Could test 2 or 3 later if reclaiming CPU matters,
but 1 is structurally too low at this concurrency.

### F9. Latent correctness bug in legacy hidden-state slicing
**Discovered 2026-05-13 during M3 of centralized-inference work.**

`BatchInferencePlayer._run_batch` (transformer path) slices
`next_ctx_batch[i:i+1, :L_i+1, :]` as the new hidden state for
request i with prior length L_i. The model's `forward_with_hidden`
puts the new encoded state at position `max_T` regardless of L_i, so
the slice `[:L_i+1]` picks a padding-derived position at index L_i
instead of the real new state when the batch is mixed-length.

**Effect**: battles that started in heterogeneous batches (L_0 = 0
batched with longer contexts) had their hidden state set to
padding-derived garbage on the very first turn, and the corruption
propagated indefinitely.

**Fix**: shipped in `RealModelBatchHandler._slice_next_hidden` on
the `centralized-inference` branch. Concatenates the prior slice
with the explicit max_T slice.

**Implication**: training to date has been somewhat undermined for
the affected battles. Post-merge model behavior will be slightly
different (more correct); brief regression possible as the model
re-adapts. Worth keeping an eye on training curves after the
centralized-inference branch merges.

### F11. Enabling exploiter co-training regresses traj/s ~25% via dual-batcher GIL contention (NATURAL EXPERIMENT)
**Discovered 2026-05-14 22:14**, after commit `9ac7f0d` ("Session bundle:
enable exploiter curriculum") shipped earlier the same day.

**Evidence (live sep_arch run with exploiters on)**:
- Throughput: ~3.8 traj/s, down from post-merge 4.98 baseline
- Two batchers active in the log: `[batch-fill svc=main] avg=3.75 max=32`
  and `[batch-fill svc=exploiter] avg=4.61 max=14`. Both 100%
  timeout-flushed, both well under cap=32. Pre-exploiter single
  `svc=main` was avg 7.2 (per F10).
- Sum of the two avgs (~8.4) is roughly the prior single-batcher load,
  but split inefficiently — neither sub-batcher accumulates enough
  density during the 5 ms timeout to fill, so per-request Python
  overhead grows as a share of total work.

**Natural experiment confirming causation**: during the same session,
the live exploiter service stopped firing requests mid-run (cause not
yet root-caused, but the symptom is documented). With no other change,
traj/s **jumped 3.8 → 5.0** while exploiters were quiet, then dropped
back when exploiters resumed. This is a same-process, same-checkpoint,
same-worker A/B that no planned experiment could match. 5.0 ≈ the
4.98 post-merge baseline, so the "exploiters off" path fully recovers
prior throughput.

**Architectural mechanism**: each `InferenceService` runs in its own
Python thread inside the trainer process
([inference_trainer.py:125](../../src/elitefurretai/rl/inference_trainer.py)).
With full curriculum + exploiter co-training, the trainer hosts 9+
service threads (`main`, `bc`, `exploiter`, `victim`, up to 5 ghost
slots). They all share one GIL. Cross-model batching is explicitly
forbidden for correctness (different weights → different forward
passes; see Plan B rejection in
[2026-05-14-00-15-model-registry-plan.md](2026-05-14-00-15-model-registry-plan.md)).
So enabling exploiters multiplies service threads without raising the
single-GIL ceiling, and splits request density across the new services.

**Implications**:
- The "exploiters off" path is the realistic high-throughput regime
  unless we change the inference service architecture.
- **Plan C registry parallelism (multi-process services) becomes the
  natural fix**: putting each service in its own process bypasses the
  single-GIL ceiling and lets exploiter co-training coexist with main
  at full throughput.
- F11 also strengthens the case for **Plan D (distillation)**: a
  smaller rollout model means each service's per-call Python cost
  drops, so the same GIL window covers more work, partially absorbing
  multi-service overhead.

### F10. Post-M4 bottleneck has moved to the inference service (CPU/Python-bound, not GPU)
**Discovered 2026-05-14 during post-M4 profiling session** (wandb `glad-lake-19`,
PID 19534, sep_arch with centralized inference + torch.compile).

**Evidence**:
- **Worker py-spy** (PID 21154, 4800 samples): GIL **0%**, Active **0%**.
  Workers are essentially idle during sampling — virtually all time spent
  blocked, waiting on inference responses. The OwnTime that does appear
  is featurization (`_generate_null_move_features` 1.67s,
  `<genexpr>` 1.27s, `generate_opponent_pokemon_features` 0.91s,
  `generate_pokemon_features` 0.73s) plus asyncio loop dispatch.
- **Inference service py-spy** (in main train.py process PID 19534,
  3900 samples): GIL 36%, Active 90%. 87% of TotalTime in
  `__call__ (inference_handlers.py)`. New top OwnTime frames:
  `_slice_next_hidden` (13%, 4.15s — F9 fix paying real per-request Python
  cost), `__call__` dispatch (10%), `static_cuda_launcher.run` (12%),
  transformer forward (8%), activation (6%). Featurization-to-tensor
  inside model forward (`_encode_features`) is 33% TotalTime / 0% OwnTime.
- **`nvidia-smi dmon -s u`** over ~40s: SM% mean ≈ 22–25%, with one
  85% spike (learner backward) and a 67% peak. GPU has ~75% headroom.
- **Batch-fill logs** (from inference service): avg batch 7.2, max 32,
  filled% 0.2, timeout% 99.8 (cap=32, timeout=5ms). The service
  finishes batches faster than requests arrive.

**Implications**:
- The inference service is the new bottleneck, but it's **not GPU-bound**.
  It's CPU/Python-bound on a single process (GIL ceiling + per-request
  slicing/padding/sampling overhead in `inference_handlers.py`).
- A faster engine (Rust) would only enqueue requests into a queue that
  is already the bottleneck. **H8 deprioritized.**
- M4 unlocked ~2x larger batches (4 → 7.2 avg) — real win, but well
  below the predicted 18–48 ceiling. The 5ms timeout is too aggressive
  given the request arrival pattern; bumping it should grow batches.
- Highest-leverage moves are inference-service-side: distillation (D)
  to shrink per-call Python cost, `_slice_next_hidden` vectorization,
  `batch_timeout` bump, and eventually Plan C registry parallelism to
  break the single-process Python ceiling.

### F8. Per-player batchers are structurally limited to ~4 concurrent requests
**Evidence**: After adding batch-fill logging (2026-05-13), thousands of
batches per player consistently show:
- avg batch size: 1.4–2.2
- max EVER observed: **4** (across 4500+ batches per player, even with
  `cap=32`)
- 100% of batches flush via timeout (none ever fill to cap)

**Why**: each `BatchInferencePlayer` has its own per-player batcher
serving its own ~32 concurrent battles. Battles for one player
multiplex over ONE websocket per player, so requests arrive serially.
Even with 32 concurrent battles, only 1–4 are typically in
"waiting-for-action" state during the 5ms timeout window. The bottleneck
isn't batch capacity — it's request arrival rate per player.

**Implication**:
- Bumping `batch_size` further from 32 won't help (we've already proven
  it can't fill).
- The next batching lever is **centralized inference** (share a batcher
  across all players within a worker, or across all workers in the
  process). With 12 players, that gives a nominal 12x more request
  density per timeout window — reaching real batch sizes of 18–48.
- This is exactly what VGCBench's `SubprocVecEnv` provides for free.

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
| ~~H1~~ | ~~Vectorizing `_dual_expand` cuts worker CPU by 15–25%~~ | **Partially resolved**: 1.88x microbenchmark speedup on CUDA. In-training impact pending next sep_arch launch. See Validated changes 2026-05-13. |
| H2 | `batch_size: 4 → 32` improves throughput 1.5–2x | High | Add fill-size logging, then bump and measure |
| H3 | `num_servers: 4 → 1` is throughput-neutral; reclaims CPU for ~1.1x | Medium | Single-knob ablation run |
| ~~H4~~ | ~~`torch.compile(model, mode='reduce-overhead')` on inference cuts GIL hold by 30%+~~ | **Blocked on architecture change** (2026-05-13). Tested with mode='default' + dynamic=True + warmup; CPU inference path can't tolerate first-call compile latency vs 8s `inference_request_timeout_s`. Re-enable after worker inference moves to GPU or centralized inference ships. |
| H5 | Vectorizing `embedder.py` (esp. the `<genexpr>` hot spot) cuts featurization by 2x | Medium | Profile-guided rewrite of the generator |
| H6 | Centralized inference (à la `SubprocVecEnv`) closes ~half the gap to VGCBench | High | Significant refactor — defer until cheap wins exhausted |
| H7 | Distilling a small (2–5M param) "rollout-only" model gives 2–4x on rollouts with no learner impact | High (raised after F10) | Train distillation, A/B with current architecture |
| ~~H8~~ | ~~Rust backend for self-play closes most of the remaining gap~~ | **Deprioritized 2026-05-14**: F10 shows workers are Active 0% post-M4 — they're inference-response-bound, not engine-bound. A faster engine just queues into the same Python ceiling. Re-evaluate only if Plan D + Plan C + inference-service optimizations don't close the gap. |
| H9 | Bumping `batch_timeout: 5ms → 10–15ms` grows avg batch from 7.2 toward 18+, raising throughput proportionally | High | Single-knob ablation; risk: turn latency up but well under 8s SLA |
| H10 | Vectorizing `_slice_next_hidden` (13% OwnTime in inference service) cuts per-request Python cost ~10% | High | Refactor + microbenchmark, same playbook as `_dual_expand` (F4) |

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
| 2026-05-13 | Vectorized `_dual_expand` (H1 shipped) | Microbenchmark: **1.88x on CUDA**, 1.11x on CPU. In-training: dark-sea-5 (vec only) → **3.43 traj/s** (+13% vs pre-vec 3.0). |
| 2026-05-13 | `batch_size: 4 → 32` + batch-fill logging | Cleaned, apples-to-apples vs dark-sea-5 (resumed from same step 151). New run: **3.69 traj/s** (+8% over dark-sea-5). Modest gain because batches structurally cap at 4 per player (F8). |
| 2026-05-13 | `num_servers: 1` (REVERTED) | Killed run via zero-completion watchdog. Reverted to 4. See F6 update. |

---

## Queued experiments (in priority order)

Ordering reflects the **cheap-first protocol** in "Sequencing rationale" above:
finish all cheap experiments before committing to an architectural change.

### Tier 1 — Cheap wins (do all of these, in this order)

| # | Experiment | Expected gain | Effort | Status |
|---|---|---|---|---|
| ~~1~~ | ~~Vectorize `_dual_expand`~~ | ~~1.5–3x worker CPU~~ | ~~1–2 days~~ | **Shipped 2026-05-13** — 1.88x microbench, +13% in-training |
| ~~1b~~ | ~~Re-profile after vectorization~~ | ~~diagnostic~~ | ~~15 min~~ | **Done 2026-05-13** — confirmed `_dual_expand` 32% → 6% OwnTime |
| ~~2~~ | ~~Add inference batch-size logging~~ | ~~diagnostic~~ | ~~1 hour~~ | **Done 2026-05-13** — revealed F8 (per-player cap of 4) |
| ~~3~~ | ~~Bump `batch_size: 4 → 32`~~ | ~~1.5–2x~~ | ~~1 hour~~ | **Done 2026-05-13** — only +8% (gain limited by F8) |
| ~~4~~ | ~~`num_servers: 1` ablation~~ | ~~1.1–1.3x~~ | ~~5 min~~ | **Done 2026-05-13 — REVERTED, killed run via zero-completion watchdog** |
| ~~5~~ | ~~`torch.compile(mode='reduce-overhead')` inference model~~ | ~~1.5–2.5x~~ | ~~1 day~~ | **2026-05-13: blocked on per-worker path, but later re-enabled in M4 centralized inference (one-process compile).** See "Updates log" 17:10 and the 2026-05-14 entry. |
| ~~5b~~ | ~~After (5), re-profile~~ | ~~diagnostic~~ | ~~15 min~~ | Implicit in 2026-05-14 post-M4 profile (F10). |
| **7** | Bump inference service `batch_timeout: 5ms → 10–15ms` (H9) | 1.3–2.0x (avg batch 7.2 → 18+) | 5 min | **Pending** — highest-ROI cheap win now. Risk: turn latency up; well under 8s SLA. |
| **8** | Vectorize `_slice_next_hidden` (H10, inference service) | 1.05–1.10x | half day | **Pending** — same playbook as `_dual_expand`. 13% OwnTime is the new top frame. |
| **6** | Vectorize `embedder.py` (`<genexpr>`, `_generate_null_move_features`, `generate_*_features`) | 1.05–1.15x worker side | half day | **Un-deferred 2026-05-14** — F10 shows these are the bulk of worker OwnTime during brief active windows. Lower priority than #7 and #8 but cheap. |

### Tier 2 — Decision point

After Tier 1, re-measure throughput and apply the decision protocol from
"Sequencing rationale" to pick at most one architectural change.

### Tier 3 — Architectural options (pick one based on Tier 1 data)

| # | Experiment | Expected gain | Effort | Notes |
|---|---|---|---|---|
| D | Distilled rollout model (2–5M params) | 2–4x | 1 week | **Likely best near-term arch option** — leverages existing BC pipeline |
| A | Centralized inference refactor (à la VGCBench) | 2–3x | 1–2 weeks | Cleanest match to VGCBench; obsoletes #3 |
| C | Switch self-play battle execution to Rust backend | 3–10x | unknown | Highest ceiling; obsoletes #4 |
| B | Rust featurization (PyO3 binding for `etl/embedder.py`) | 1.5–2x | 2–4 weeks | Only if Tier 1 + chosen Tier 3 leave featurization dominant |

### Deferred

| # | Experiment | Reason deferred |
|---|---|---|
| ~~6~~ | ~~Vectorize `embedder.py` `<genexpr>` and `_generate_null_move_features`~~ | **Un-deferred 2026-05-14** — moved to Tier 1 under #6. F10 shows it's the bulk of worker OwnTime, and the Rust-featurization argument (B) for deferring no longer applies since the active workers are doing exactly these calls during their brief Active windows. |

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

## Sequencing rationale (why we do cheap experiments first)

Several queued experiments will be made obsolete by larger architectural
changes (centralized inference, Rust backend, Rust featurization). The
naive instinct is to skip the cheap ones and jump to the architectural
change. We deliberately don't do that. Reasons:

1. **ROI math favors cheap-first.** 1 hour for 1.5–2x is a great trade
   even if the work gets thrown away in two weeks. We get the speedup
   *during the interim*, which often means weeks of faster training
   runs while planning the larger refactor.
2. **Cheap experiments produce the data that decides whether the
   architectural change is necessary.** If `batch_size=32` + `num_servers=1`
   + `torch.compile` together close the gap to VGCBench, we may not need
   centralized inference at all. A 1–2 week refactor is not worth
   committing to without knowing the cheap path didn't already get us
   there.
3. **Each cheap fix shifts the bottleneck.** After every change, re-profile
   to see what's now hot. Picking the next architectural target without
   fresh profile data is guessing.

### Obsolescence table

| Queued experiment | Obsoleted by | Throw-away risk | Verdict |
|---|---|---|---|
| 1b py-spy diagnostic | – | None | Always do |
| 2 batch-size logging | partial: A | Low (logging still useful under A) | Do |
| 3 `batch_size: 4 → 32` | **A** centralized inference | Throw-away under A | Do (1 hour, big gain in interim) |
| 4 `num_servers: 1` | **C** Rust backend | Throw-away under C | Do (5 min, near-free) |
| 5 `torch.compile` | – | None — synergistic with A and D | Do (survives every refactor) |
| 6 Vectorize `embedder.py` | **B** Rust featurization | High (half day of Python work replaced by Rust) | **Skip unless profile mandates** |
| A Centralized inference | – | Architectural | Decide after cheap experiments |
| B Rust featurization | – | Architectural | Decide after cheap experiments |
| C Rust backend (self-play) | – | Architectural | Highest ceiling, biggest unknown |
| D Distilled rollout model | – | Architectural | Best near-term arch option |

### Decision protocol

After completing the cheap experiments (1b, 2, 3, 4, 5):

- If throughput ≥ 10 traj/s → keep tuning. The gap to VGCBench is mostly
  closed; further cheap wins (e.g. higher `batch_size`, more workers)
  may be enough.
- If throughput 5–10 traj/s → commit to one architectural change.
  Recommended order of ROI/effort: **D → A → B → C**.
  - D (distilled rollout model) is fastest because the BC pipeline
    already exists; we'd train a smaller model and swap it in.
  - A (centralized inference) is the biggest structural win and the
    cleanest match to VGCBench's approach.
  - B (Rust featurization) only if profile data shows featurization
    remains the dominant cost after A.
  - C (Rust backend) has the best ceiling but the biggest integration
    unknown.
- If throughput still < 5 traj/s after all cheap experiments → re-profile
  and identify what the cheap path *failed* to remove. The remaining
  bottleneck may point to a different architectural change than we'd
  expected.

---

## Updates log

### 2026-05-14 00:38 — Post-M4 profiling session: workers idle, inference service is new bottleneck, engine choice deprioritized

**Run profiled**: wandb `glad-lake-19` (PID 19534), sep_arch with centralized
inference + torch.compile. Run started 2026-05-13 23:31:32, received external
SIGTERM at 2026-05-14 00:32:28 (graceful shutdown — NOT a crash; no OOM, no
watchdog firing, no fatal exception. Likely user-initiated kill; worth
checking shell history if origin matters).

**Throughput**: **3.78 traj/s** (`battles_per_second` from wandb summary).
Note: this is a **regression from 4.05 traj/s** reported for D3-alt in
[centralized inference implementation plan](./2026-05-13-18-00-centralized-inference-implementation-plan.md)
and [model registry plan](./2026-05-14-00-15-model-registry-plan.md). Possible
causes: different starting checkpoint, additional features in this build,
or natural variance. Worth a separate apples-to-apples comparison run
before declaring confident progress.

**Batch-fill data** (centralized inference, log line at 23:39:43):
```
[batch-fill svc=main] n=8000 avg=7.18 max=32 filled%=0.2 timeout%=99.8 cap=32
```
M4 unlocked avg batch ~7.2 (vs pre-M4 per-player cap of 4 — real ~1.8x
win) but well below the 18–48 ceiling predicted in F8. 99.8% of batches
flush via 5ms timeout; only 0.2% reach cap. The service is finishing
batches faster than requests arrive.

**Inference service py-spy** (in main train.py process, 3900 samples):
- GIL **36%**, Active **90%**, 87% TotalTime in `__call__ (inference_handlers.py)`
- Top OwnTime: `_slice_next_hidden` 13% (4.15s) [the F9 fix],
  `static_cuda_launcher.run` 12%, `__call__` dispatch 10%, transformer
  forward 8%, `activation.forward` 6%, `_detect_is_causal_mask` 4%,
  `_as_tensor_fullprec` 4%, `_sample_action` 3%, `_pad_transformer_context` 2%
- `_encode_features` is 33% TotalTime / 0% OwnTime (feature→tensor pack
  inside model forward; cost is in callees, not the function itself)
- ~28% of OwnTime is pure Python inside `inference_handlers.py` —
  slicing, padding, sampling, dispatch
- ~33% is PyTorch internals (compiled kernels + tensor ops)

**Worker py-spy** (PID 21154, 4800 samples):
- GIL **0%**, Active **0%** — workers are essentially idle during sampling,
  blocked waiting on inference responses
- Top OwnTime (the small chunk of active time): `_generate_null_move_features`
  1.67s, `<genexpr>` 1.27s, `generate_opponent_pokemon_features` 0.91s,
  `generate_pokemon_features` 0.73s, `embed` 0.58s, `embed_to_array` 0.41s
- Top TotalTime: `_run (asyncio/events.py)` 8.46s, `_handle_message
  (ps_client)` 7.98s, `_handle_battle_message (poke_env)` 7.92s,
  `_choose_move_async` 7.37s, `_embed_battle_state` 6.80s

**`nvidia-smi dmon -s u`** (~40 samples / ~40s window):
- SM% mean ≈ 22–25%, median ≈ 24%, mode in low-20s
- One 85% spike (paired with 76% mem bandwidth — learner backward)
- One 67% peak
- Long stretches at 6–8% (inference idle gaps between request bursts)
- Memory bandwidth mostly 1–15%, one 76% spike

**Architectural finding (F10 added)**: bottleneck has moved to inference
service. Workers are inference-response-bound, not engine-bound. GPU has
~75% headroom. The service is CPU/Python-bound on a single process
(GIL ceiling + per-request slicing/padding/sampling overhead).

**Implications and ranking shifts**:
- **H8 (Rust backend) deprioritized**. Workers are Active 0%; making the
  engine faster just enqueues into the same Python ceiling. Re-evaluate
  only if other levers fail.
- **H9 (`batch_timeout: 5ms → 10–15ms`) added as Tier 1 #7** — highest-ROI
  cheap win. The service finishes batches faster than requests arrive,
  so loosening the timeout should grow avg batch from 7.2 toward 18+.
  Risk: turn latency up, but well under 8s SLA.
- **H10 (vectorize `_slice_next_hidden`) added as Tier 1 #8** — 13%
  OwnTime is the new top frame. Same playbook as `_dual_expand`.
- **Tier 1 #6 (vectorize `embedder.py`) un-deferred** — F10 shows these
  are the bulk of worker OwnTime in active windows. Lower priority than
  #7 and #8 but cheap.
- **H7 (distillation) raised to High confidence** — GPU is 75% idle, so
  the model isn't too big for the hardware. But a smaller model directly
  reduces per-call Python encode/dispatch cost in the inference service,
  which IS the bottleneck. Plan D becomes the highest-impact arch move.
- **Plan C (model registry) gains a second motivation** — beyond
  curriculum unblock, it naturally distributes inference work across
  multiple service processes, sidestepping the single-process GIL ceiling.

**Open questions** (for next session):
- Why did throughput regress 4.05 → 3.78 vs the registry doc baseline?
  Run a controlled comparison.
- Does `batch_timeout` bump actually grow batches as predicted, or is
  there a deeper structural reason batches stay small? (E.g. workers
  blocking on the previous batch's response before firing the next
  request.)
- Is `_encode_features` (33% TotalTime in the service) a worthwhile
  target after `_slice_next_hidden`? Its 0% OwnTime hides the real
  cost — need to look one frame deeper.

### 2026-05-13 17:10 — torch.compile experiment — NEGATIVE RESULT, rolled back

**Hypothesis (H4)**: `torch.compile(agent, mode='default', dynamic=True)`
on the worker inference path will give 1.5–2.5x via kernel fusion +
dispatch reduction, since model forward (linear + transformer + layer_norm)
is the largest useful category in py-spy at ~36% OwnTime.

**Implementation shipped**:
- `HardwareConfig.compile_inference_model: Optional[str]` knob added
  (config.py + roundtrip test in test_config.py).
- `worker.py:259` — conditional `torch.compile(agent, mode=..., dynamic=True)`
  after building the main inference agent.
- Config-cast wrapper for pyright (compiled module's `OptimizedModule`
  type doesn't structurally match `RNaDAgent` even though attribute
  delegation works at runtime).
- Equivalence test in `test_model_archs.py` passes — compiled and eager
  outputs match within 1e-4 across multiple batch sizes and turn-0/turn-1
  context paths.

**First training-run attempt** (sep_arch with `compile_inference_model: default`):
all battles cascaded into 8s timeouts (matching `inference_request_timeout_s
= 8.0`). Workers ran for 12+ minutes without producing a single learner
update. Per-worker first-call compile latency on CPU (TorchInductor
generates and compiles C++ code; 27M-param transformer model) is
~30-90s; meanwhile every queued inference request hits the 8s timeout,
sends a fallback choice, and Showdown rejects it as "Invalid choice"
because game state has moved on.

**Second attempt** (added synchronous compile warmup before accepting
battle requests, two-shape warmup covering turn-0 and turn-1 paths):
same failure pattern. Real-traffic shapes apparently differ from
warmup shapes enough to trigger recompile, OR the per-process compile
contention across 4 workers compounds the latency, OR both. Run again
produced 0 updates and started cascading into websocket disconnects
(`ConnectionClosedOK`).

**Diagnosis**: torch.compile on CPU is the wrong tool for this online
inference path. The mechanism is sound (kernel fusion + dispatch
overhead reduction would help) but the deployment environment doesn't
support it: workers run inference on CPU (worker.py:232 hard-codes
`device = "cpu"`), CPU TorchInductor compile times are too long, and
the inference path has a hard 8s SLA via `inference_request_timeout_s`.

**Rolled back**: `compile_inference_model: null` in sep_arch.yaml. Kept
the code path (worker.py + config knob + tests) so we can re-enable
under either of:
- (a) Worker inference moved to GPU — much faster compile (~5-15s)
  AND the speedup is much larger on GPU.
- (b) Centralized inference architecture — one process compiles once,
  serves all workers' requests. Same as (a) effectively.

**Strategic implication**: this is the second cheap-win that hit a
wall (after `num_servers: 1` killed the run). H4 stays open but moves
to "blocked on architectural change". The next genuine cheap win
isn't in our queue — Tier 1 is exhausted. **We're at the decision
point earlier than expected.**

Updated throughput baseline: **3.69 traj/s** (vec + batch_size=32,
unchanged from 16:30 measurement).

### 2026-05-13 16:30 — Tier 1 wave 2 results (batch_size + num_servers + logging)

**Run**: relaunched sep_arch from `pretty-jazz-4/main_model_step_151.pt`
(PID 13128 = log `sep_arch_b32s4_*.log`). Same starting checkpoint as
dark-sea-5, so apples-to-apples on the post-vectorization baseline.

**Changes shipped**:
1. `hardware.batch_size: 4 → 32` (sep_arch.yaml)
2. `hardware.num_servers: 4 → 1` then **REVERTED to 4** after the first
   attempt died via zero-completion watchdog
3. Batch-fill logging in `BatchInferencePlayer._inference_loop`
   (counters: `inference_batches_filled_to_max`,
   `inference_batches_flushed_timeout`; periodic WARNING log every 500
   batches with avg/max/filled%/timeout%/cap)

**Failure mode discovered** (first attempt with `num_servers: 1`): the
single Showdown server saturated under 12 players × 32 concurrent
battles = 384 streams. Zero-completion watchdog fired after 15
consecutive empty batches at 16:01. F6 updated to capture: 0% OwnTime
in py-spy means CPU cost is free, NOT that throughput capacity is
infinite. Reverted to `num_servers: 4`.

**Throughput** (clean run after revert, updates 153–155 average):

| Metric | dark-sea-5 (vec only) | This run (vec + batch_size=32) | Δ |
|---|---|---|---|
| Throughput | 3.43 traj/s | **3.69 traj/s** | +8% |
| Learner steps/s | 59 | 63 | +7% |

**Batch-fill data** (the most important finding):

| Per-player batcher (4500+ batches sampled per player) | |
|---|---|
| avg batch size | 1.43 – 2.16 |
| max EVER observed | **4** (across thousands of batches) |
| % of batches that filled to cap | **0.0%** (zero) |
| % of batches that flushed via timeout | **100%** |
| Configured cap | 32 |

This is the structural insight (now F8): per-player batchers cap at ~4
not because of `batch_size` but because each Player has ONE websocket
to Showdown, and battles' requests arrive serially. With 32 concurrent
battles per player, only 1–4 are typically in "waiting-for-action"
state during the 5ms timeout window. **Bumping `batch_size` further
cannot help.**

**Implication for next steps**:
- `torch.compile` (Tier 1 #5) is now the highest-leverage cheap win:
  the model forward is the largest *useful* category and `torch.compile`
  should fuse linear → norm → activation kernels.
- The next architectural lever is **centralized inference** (sharing
  one batcher across all players within a worker, or across workers):
  combining 12 players' request streams gives ~12x request density per
  timeout window, unlocking real batch sizes of 18–48. This is exactly
  what VGCBench's `SubprocVecEnv` provides for free.

### 2026-05-13 14:38 — Re-profile after vectorization (experiment 1b)

**Run**: relaunched sep_arch from `pretty-jazz-4/main_model_step_151.pt`
(PID 5386, workers 6079–6082). Profiled worker 6079 with `py-spy top`
for 30s once warm.

**Headline stats**:
- GIL: 44% → **37%** (down 7 pp ✓)
- Active: 138% → **110%** (DOWN 28 pp — see interpretation below)
- Throughput: 3.0 → **3.4 traj/s** (+13%); 51 → 59 learner-steps/s (+16%)
- Worker CPU (per ps): 99% → **143–145%** (more thread-level parallelism)

**Top OwnTime frames — before vs after** (only meaningful changes shown):

| Function | Before %Own | After %Own | Δ |
|---|---|---|---|
| `_dual_expand` (model_archs) | **32%** | **6%** | **−26 pp** ✓✓ |
| `embedding` (torch functional) | 5% | ~0% | −5 pp ✓ |
| `<genexpr>` (etl/embedder) | 6% | 2% | −4 pp |
| `_generate_null_move_features` | 7% | 5% | −2 pp |
| `embed` (etl/embedder) | 8% | 1% | −7 pp |
| `forward` (linear) | 9% | **14%** | +5 pp (relatively bigger) |
| `forward` (transformer) | 15% | **17%** | +2 pp |
| `forward` (model_archs) | 3% | 1% | −2 pp |
| `layer_norm` | 2% | 4% | +2 pp |
| `embed_to_array` | 1% | 3% | +2 pp |
| `_worker` (concurrent.futures) | 20% | 34% | +14 pp (queue-wait, see below) |

**Re-categorized**:

| Category | Before %Own | After %Own | Notes |
|---|---|---|---|
| Model compute (linear + transformer + layer_norm + activation) | ~30% | **~36%** | Now the largest genuine-compute category |
| Featurization (`etl/embedder.py`) | ~30% | **~16%** | Roughly halved |
| `_dual_expand` (residual) | 32% | **6%** | Mostly fixed |
| Thread / queue-wait overhead | small | ~30% | `_worker` rose because inference thread waits more |

**Interpretation**:
- **F4 RESOLVED**: `_dual_expand` is no longer the dominant frame.
  Vectorization fully validated.
- **The new bottleneck is the model forward itself** — `linear.forward`
  + `transformer.forward` together are now 31% of OwnTime, the largest
  single chunk of *useful* work. This is exactly the regime where
  `torch.compile` shines (kernel fusion across linear → norm → activation).
- **Featurization (F5) is meaningfully smaller now** (~16% vs ~30%
  before). This isn't because we changed featurization — it's because
  the same wall-clock featurization cost is now a smaller fraction of a
  shorter total worker step. Confirms the F5 "do not vectorize Python
  embedder.py" deferral.
- **Active% dropped 138% → 110%** is *good news*: the worker now does
  the same logical work using less CPU per unit time. The inference
  thread blocks more on `concurrent.futures` queue (the `_worker` 34%
  OwnTime is mostly queue-wait overhead, not real work).
- **`_worker` 34% OwnTime is largely queue-blocking**, not active
  computation. This means the inference thread is *finishing batches
  faster than they arrive*, which is the strongest signal yet that
  bumping `batch_size` will help — it'll fill each forward call with
  more concurrent battles.

**Updated next-step ranking** (from this profile):

1. **Bump `batch_size: 4 → 32`** (Tier 1 #3) — was already pending; the
   queue-wait dominance in `_worker` is the smoking gun that batches
   are under-filled. Highest-confidence cheap win now.
2. **`torch.compile` the inference model** (Tier 1 #5) — model forward
   is now the largest *useful* category. Kernel fusion should bite hard.
3. **`num_servers: 1`** (Tier 1 #4) — still essentially free; do
   alongside #1 since it also frees CPU.
4. **Add batch-size logging** (Tier 1 #2) — should still be done to
   validate the assumption #1 rests on.

### 2026-05-13 16:00 — `_dual_expand` vectorized and shipped
- Replaced the per-position Python loop with: layout precomputed at
  `GroupedFeatureEncoder.__init__`, batched embedding lookup per type/bank
  at forward time, scatter into pre-allocated output tensor via flat
  `dst_indices` advanced indexing. Source/destination index tensors live
  as non-persistent buffers so they move with `.to(device)` and stay out
  of `state_dict` (so checkpoint loads aren't disturbed).
- Kept `_dual_expand_legacy` on the class strictly for regression testing.
  Do not remove without porting equivalent coverage.
- Added 7 equivalence tests in `unit_tests/supervised/test_model_archs.py`
  covering passthrough-only groups, mixed eid+nb groups across all
  replacement-position patterns, and layout-size invariants.
- All 43 model_archs tests + 35 embedder_improvements tests still pass.
- Quality gates clean (ruff check, ruff format, pyright).
- **Microbenchmark** (realistic 14-group encoder, batch=4, seq=30, 100 iters):
  - CPU: legacy 1503 µs/call → vectorized 1352 µs/call (1.11x)
  - **CUDA: legacy 3301 µs/call → vectorized 1755 µs/call (1.88x)**
  - CUDA gain is what matters (workers run on CUDA). The ~46% latency
    cut on `_dual_expand` translates to roughly 15% worker-CPU savings
    given its 32% OwnTime in the original profile — modest but real.
- Next: restart sep_arch with new code, re-run py-spy top to confirm
  `_dual_expand` is no longer #1 by OwnTime and identify the new top
  frame for the next optimization (likely `_generate_null_move_features`
  or `<genexpr>` in `etl/embedder.py`, per F5).

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

