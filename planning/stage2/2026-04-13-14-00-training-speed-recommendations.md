# Training Speed Implementation Plan: GPU Inference, Worker Scaling, Step Cap, and Faster Full Embedding

## Executive Summary

Current sustained training throughput (Rust backend, 4 workers, `full` feature set, ~100M param Transformer):
- **9.3 learner steps/s** (steady state after warmup)
- **0.35 battles/s** (256 trajectories per update, ~27 steps/battle)
- **~13 min per learner update** (256 × train_batch_size)

The breakdown of where wall-clock time goes during RL training:

| Component | % of wall time | Evidence |
|-----------|---------------|----------|
| Actor-side model inference (CPU) | ~60-70% | Benchmark profiling: `policy_inference_seconds=22.87/25.46s` in Rust benchmark; 208 inference calls consuming 15.97s/16.96s in Showdown cProfile |
| Embedding (Python) | ~10-15% | `embed()` + `feature_dict_to_vector()`: 1.43s/16.96s in cProfile, plus `_dual_expand` at 2.90s |
| Learner GPU update | ~5-10% | GPU update itself is fast (mixed precision on RTX 3090). Bottleneck is waiting for trajectories |
| Battle simulation (Rust engine) | ~5% | Random engine: 12 battles/s vs model-backed 0.76 battles/s → engine itself is <5% of model-backed time |
| Queue / IPC overhead | ~5% | Multiprocessing queue serialization, weight broadcast |
| Trajectory collation | <2% | `collate_trajectories()` is pure tensor ops, negligible |

The fundamental bottleneck is **actor-side CPU model inference**. Every battle decision requires a full forward pass through a ~100M parameter model on CPU, and there are ~27 decisions per battle across ~256 battles per update.

Session note (2026-04-13): the fast full-embedding pass was validated on the Rust model benchmark and improved throughput from **0.320 battles/s** to **0.738 battles/s** on the same 12-battle, 3-concurrent CPU benchmark shape. Profiled embed time dropped from **2.559s** to **1.141s**, and total profiled time dropped from **71.425s** to **31.322s**.

This document is now the concrete implementation handoff for the next session. The selected scope is:

1. **Implement a centralized GPU inference server for Rust training actors**
2. **Scale worker count after GPU offload removes the current CPU inference bottleneck**
3. **Raise the battle step cap from 30 to 40 and add explicit truncation visibility**
4. **Implement a faster embedding path while preserving the `full` feature set**

Explicit non-goals for this implementation pass:

1. **Do not implement a smaller actor model / distillation path in this pass**
2. **Do not switch RL actors from `full` to `raw` features**
3. **Do not do CPU quantization work in this pass**
4. **Do not remove the websocket backend; preserve both `rust_engine` and `showdown_websocket` as supported backends**

The intended order is not arbitrary. GPU actor inference is the enabling architectural change. Once that exists, worker scaling becomes worthwhile, embedding becomes the next material actor-side hotspot, and raising the step cap improves the value of the trajectories we are already paying to generate.

---

## Selected Path For The Next AI Session

### Goal

Increase learner-facing RL throughput by attacking the dominant actor-side bottlenecks without changing the supervised model family or dropping the `full` feature set.

### Target Outcome

Starting point:
- **9.3 learner steps/s**
- **0.35 battles/s**
- **4 Rust workers**
- **CPU actor inference**
- **`full` feature set**
- **`max_battle_steps=30`**

Target after the selected work:
- **GPU-backed actor inference server in the Rust backend path**
- **worker count increased beyond 4 after validation**
- **`max_battle_steps=40`**
- **faster embed path that still emits the same `full` feature vector semantics**
- **new metrics and benchmarks proving end-to-end training improvement**

### Why These Four Together

1. **GPU inference server** removes the largest single bottleneck: CPU forward passes through the ~100M actor policy.
2. **Worker scaling** only becomes attractive after GPU offload; before that it mostly amplifies CPU contention.
3. **Faster full embedding** becomes more important after inference is accelerated, because embedding becomes a larger share of actor wall time.
4. **Higher step cap** improves effective learning yield by reducing premature truncation in long or messy battles.

### Revised Priority Order

1. **GPU inference server**
2. **Raise step cap to 40 and add truncation metrics**
3. **Scale worker count and tune concurrency**
4. **Implement faster full embedding path**

The ordering above is the recommended execution order for implementation, validation, and rollback safety. The first two items are the most important to land cleanly. Worker scaling and faster embedding should be tuned on top of the new GPU-backed baseline rather than guessed in advance.

---

## Recommendation 1: Move Actor Inference to GPU (Centralized Inference Server)

### Estimated Speedup: 3-5x on learner steps/s

### The Problem
Each of the 4 worker processes independently runs model inference on CPU. A single forward pass through the ~100M param Transformer takes ~77ms on CPU (15.97s ÷ 208 calls from the cProfile). This dominates 60-70% of wall time.

The RTX 3090 can execute the same forward pass in ~2-5ms (batch of 16), but the GPU sits idle during trajectory collection because the learner only uses it for the brief gradient update step.

### Evidence
- Benchmark: Rust model-backed at `max_concurrent=1` → 0.393 battles/s. The profiled breakdown showed 22.87s of 25.46s (89.8%) was policy inference
- cProfile: `_gpu_inference_sync` + `torch.nn.modules.module._call_impl` accounted for 15.97s of 16.96s total in Showdown maxed profile
- `torch._C._nn.linear` alone consumed 6.19s (36.4% of total) — this is pure matrix multiplication, ~20-50x faster on GPU
- The 3090 is currently used only for the learner gradient step, which is a small fraction of wall time. GPU utilization during trajectory collection is near 0%

### Why This Fixes the Core Problem
CPU inference is fundamentally the wrong hardware for matrix-heavy forward passes. Moving actors to request inference from the GPU changes the bottleneck from "CPU mat-mul speed" to "GPU batch throughput + network/IPC latency." A single GPU can serve many actors simultaneously because batched GPU inference amortizes the kernel launch overhead.

### How the Estimate Was Derived
- Current: 208 inference calls × ~77ms/call on CPU = 16s wall time
- GPU at batch=16: same 208 calls / 16 per batch = 13 batches × ~5ms = 65ms total compute
- Even with IPC round-trip overhead (~5ms per request), GPU throughput would be ~10-20x faster per inference call
- But actors still need to embed and prepare requests, so the **system-level speedup** is bounded by embedding cost + IPC → estimated 3-5x on overall steps/s
- Conservative end (3x): accounts for IPC serialization overhead and GPU contention with learner updates
- Optimistic end (5x): assumes embedding is also optimized (see Rec #3) and IPC is cheap (shared memory)

### What Would Need to Change
1. **Create a GPU inference server** (a dedicated thread or process on the GPU) that accepts batched state tensors from workers and returns action logits
2. **Decouple embedding from inference** in the actor loop: workers embed states (CPU), send state tensors to the GPU server, receive logits back
3. **Use shared memory** (e.g., `torch.multiprocessing` shared tensors or a ring buffer) for zero-copy transfer between workers and the GPU inference process
4. **Time-share the GPU** between inference server and learner: inference runs between gradient steps, or use CUDA streams to overlap

Key files to modify:
- `src/elitefurretai/rl/train.py`: Add GPU inference server process, modify worker dispatch
- `src/elitefurretai/engine/sync_battle_driver.py`: Replace local model forward with remote inference call
- `src/elitefurretai/rl/players.py` (`SyncPolicyPlayer`): Send embed vector to inference server instead of calling `model.forward()` locally

---

## Deferred Recommendation 2: Smaller Actor Policy Network (Distillation)

### Estimated Speedup: 2-4x on actor inference, 1.5-3x on overall steps/s

### The Problem
The policy network used by actors is the same ~100M parameter Transformer used by the learner. Actors don't need the value head or the full model capacity — they only need to produce reasonable action distributions for trajectory collection. A smaller "actor model" would be much faster on CPU.

### Evidence
- `torch._C._nn.linear` consumed 6.19s (36.4% of profiled time). Linear layer cost scales linearly with parameter count
- `_dual_expand` consumed 2.90s — the entity-ID and number-bank expansion produces large intermediate tensors that scale with model width
- LSTM forward consumed 2.68s — LSTM cost scales with `hidden_size²`; a half-sized LSTM would be ~4x faster
- The model architecture uses layers of `[2048, 1024, 1024]` → `[1024, 1024]` with an aggregated dim of 4096. A 4x smaller actor (e.g., `[512, 256, 256]` → `[256, 256]`, aggregated dim 1024) would be ~16x fewer FLOPs in the dense layers
- DeepMind's Ataraxos (the RNaD reference) uses an asymmetric actor-learner architecture for exactly this reason

### Why This Fixes the Core Problem
Cutting the actor model from ~100M to ~10-25M params makes CPU inference proportionally cheaper. The actor doesn't need to produce perfect action distributions — it needs to produce *good enough* distributions for trajectory collection. The learner corrects for the importance-sampling ratio anyway (PPO clip objective).

### How the Estimate Was Derived
- If actor model is 4x fewer parameters: linear layer time drops from 6.19s to ~1.55s
- LSTM drops from 2.68s to ~0.67s (quadratic scaling with hidden_size)
- `_dual_expand` drops proportionally to embedding dim: ~1.45s
- New total inference time: ~4-5s vs current 16s → ~3-4x faster inference
- System-level: embedding still takes ~1.5s, so overall 1.5-3x depending on batch dynamics
- Combinable with Rec #1: small actor on GPU would be ~10-20x faster at inference

### What Would Need to Change
1. **Define a slimmer architecture** (e.g., halve all layer widths, use 2 Transformer layers instead of 7, smaller LSTM). Keep input dim and output dim the same
2. **Periodic distillation**: After each learner checkpoint, distill the large model into the small actor model (soft targets from teacher logits)
3. **Weight broadcast changes**: Instead of broadcasting full model weights to actors, broadcast small actor weights. Add a distillation step to the checkpoint loop
4. **Config additions**: `actor_model_config` separate from `model_config` in `RNaDConfig`

Key files to modify:
- `src/elitefurretai/supervised/model_archs.py`: Add a `SmallActor` or parametrize existing arch with smaller defaults
- `src/elitefurretai/rl/train.py`: Add distillation step at weight broadcast intervals
- `src/elitefurretai/rl/config.py`: Add `actor_early_layers`, `actor_late_layers`, etc.
- `src/elitefurretai/engine/sync_battle_driver.py`: Load actor-specific model

---

## Recommendation 3: Faster Embedding via Precomputed Tensor Operations

### Estimated Speedup: 1.5-2x on embedding, 1.1-1.3x on overall steps/s

### The Problem
The Python-side `Embedder.embed()` function is called per-decision, per-battle. It iterates through Pokemon features, move features, battle features etc. using pure Python dict operations, list comprehensions, and getattr calls. The cProfile shows:

- `generate_opponent_pokemon_features()`: 0.73s (called 2652 times)
- `generate_pokemon_features()`: 0.58s (called 2652 times)
- `feature_dict_to_vector()`: 0.51s (called 442 times)
- Total embedding cost: ~1.94s / 16.96s = 11.4% of profiled time

### Evidence
- 442 embed calls across 12 battles × ~2 sides × ~18 turns = 442 (matches expected decision count)
- Each embed call produces a large feature vector (9000+ features for `full` set) via Python dict manipulation
- `feature_dict_to_vector` alone costs 1.15ms per call — this is purely converting a dict to a list in sorted key order
- The "fast embed" path (`embed_to_vector`) already caches key order, reducing this from ~2ms to ~1.15ms. Further gains require bypassing the dict entirely

### Why This Fixes the Core Problem
Embedding is the second-largest CPU cost after model inference. Unlike inference (which benefits from GPU offload), embedding inherently runs on CPU. Making it faster directly reduces actor wall time.

For the selected implementation path, this recommendation must preserve the `full` feature set. The optimization target is **how the features are produced**, not **which features exist**. The output vector shape, ordering, and meaning must remain compatible with the existing trained model.

### How the Estimate Was Derived
- Current: 1.94s for 442 calls = 4.4ms/call
- If we replace Python dict operations with direct tensor indexing (precomputed feature extraction into a fixed numpy buffer): ~1-2ms/call (2-4x faster)
- System-level: 1.94s → ~0.7-1.0s, saving ~1s out of ~15s per 12-battle batch → 1.1x overall
- The existing sweep showed dropping fast embed increased cost by 11.3% (1.514 → 1.343 battles/s), confirming embedding is a meaningful fraction

### What Would Need to Change
1. **Convert `embed()` to write into a pre-allocated numpy array** instead of building a Python dict. Use known fixed indices for each feature position
2. **Eliminate the dict intermediary**: Go directly from `DoubleBattle` attributes → float array positions using index maps computed at init
3. **Preserve `full` feature semantics exactly** — faster embedding must emit the same feature layout and values (within normal floating-point tolerance) as the current `embed_to_vector()` path
4. **Split feature generation into stable write regions** (player Pokemon, opponent Pokemon, battle state, engineered features, transition features) so each region can be optimized independently without changing model compatibility
5. **Cythonize or Numba-JIT the hottest loops** (`generate_pokemon_features`, `generate_opponent_pokemon_features`) if pure-Python optimization plateaus

Key files to modify:
- `src/elitefurretai/etl/embedder.py`: Add `embed_to_array()` method that writes to pre-allocated buffer
- `src/elitefurretai/engine/sync_battle_driver.py`: Switch to `embed_to_array()` in the hot loop

---

## Recommendation 4: Increase Worker Count with Async Trajectory Pipeline

### Estimated Speedup: 1.5-2x on battles/s, 1.5-2x on learner steps/s

### The Problem
Currently using 4 workers with `num_battles_per_pair=32` and `train_batch_size=256`. The learner blocks waiting for 256 trajectories before performing an update. With 4 workers at ~0.09 battles/s each, the learner waits ~14 min between updates.

The machine has 8 CPU cores, but only 4 workers are running. Adding more workers would increase trajectory generation throughput, but is currently limited by CPU core count (each worker needs a core for inference).

### Evidence
- Training log: 256 battles in 14m 8s = 0.30 b/s → 768 battles in 37m 7s = 0.35 b/s (throughput increases as more battles overlap)
- 4 workers × ~0.09 b/s = 0.36 b/s theoretical, close to observed 0.35 b/s
- With GPU inference offload (Rec #1), workers would no longer be CPU-bound → can run 8-16 workers on 8 cores since they'd mostly be waiting on GPU responses and game simulation
- The Rust engine already supports arbitrary concurrency within a single worker via `max_concurrent_battles`

### Why This Fixes the Core Problem
More workers generating trajectories in parallel means the learner spends less time idle waiting for data. The trajectory queue (`maxsize=1024`) has headroom. The key constraint today is that 4 CPU-bound workers saturate 4 of 8 cores.

### How the Estimate Was Derived
- If Rec #1 removes CPU inference cost from workers, each worker becomes I/O + embed bound (~30% of current cost)
- This means 8-12 workers could fit in the same CPU budget currently used by 4
- 2-3x more workers × same per-worker throughput = 2-3x more trajectories/s
- But: learner GPU must now handle more inference requests (bounded by GPU throughput) → net 1.5-2x
- Without Rec #1: simply adding workers hits CPU contention. This recommendation is most valuable *in combination* with GPU inference offload

### What Would Need to Change
1. **Increase `num_workers` in config** from 4 to 8-12 (after Rec #1 is implemented)
2. **Reduce `num_battles_per_pair`** proportionally (e.g., from 32 to 16) to keep each worker's memory footprint reasonable
3. **Async learner updates**: Instead of blocking until `train_batch_size` trajectories arrive, continuously update in a sliding-window fashion (process trajectories as they arrive, trigger updates when batch is ready)
4. **Profile memory usage**: Each worker holds a model copy (~400MB for a 100M param model in FP32). With GPU inference, workers only need the embedder, so memory per worker drops to ~50MB

Key files to modify:
- `src/elitefurretai/rl/config.py`: Increase `num_workers` default
- `src/elitefurretai/rl/train.py`: Modify main loop to support async trajectory consumption

---

## Phased Implementation Plan

### Phase 1: GPU Inference Server

#### Objective
Replace per-worker CPU model execution in the Rust training path with a centralized GPU-backed batched inference service while preserving the existing policy outputs, hidden-state semantics, and backend split.

#### Required Behavior
1. **Workers still own battle simulation and embedding**
2. **Workers no longer run the actor model forward locally for the Rust path**
3. **A centralized inference component batches requests across workers**
4. **Returned outputs must preserve the existing action-selection contract**:
    - turn logits / probabilities
    - teampreview logits / probabilities
    - value outputs needed for trajectory storage
    - next hidden state / context for recurrent or transformer inference
5. **Backend support must remain config-driven**:
    - `rust_engine` should use the new GPU inference server when enabled
    - `showdown_websocket` must remain functional and should not be broken by the Rust work

#### Main Design Constraints
1. **Do not duplicate a full GPU model copy per worker**
2. **Do not serialize entire Python objects for requests if a tensor payload is sufficient**
3. **Do not regress action legality or hidden-state bookkeeping**
4. **Do not block learner updates indefinitely on inference traffic**

#### Likely Files
1. `src/elitefurretai/rl/train.py`
2. `src/elitefurretai/engine/sync_battle_driver.py`
3. `src/elitefurretai/rl/players.py`
4. Possibly a new helper module under `src/elitefurretai/rl/` or `src/elitefurretai/engine/` for the inference server transport

#### Acceptance Criteria
1. Rust training still runs end to end with the `full` feature set
2. Actor model inference for the Rust backend is no longer executed on CPU in each worker
3. Rust model-backed benchmark throughput improves materially over the current `0.756 battles/s` maxed benchmark baseline
4. No regression in battle validity, request handling, or trajectory collection

### Phase 2: Raise Step Cap And Add Truncation Visibility

#### Objective
Raise `max_battle_steps` from 30 to 40 and make truncation visible as a first-class runtime signal.

Status check: the current codebase already appears to satisfy this phase in the Rust path (`max_battle_steps=40` and truncation metrics are present), so treat this as validated rather than pending reimplementation.

#### Required Behavior
1. `max_battle_steps` defaults align with `max_seq_len=40`
2. Training logs report how many completed trajectories were truncated by the step cap
3. Benchmarks or training summaries expose truncation rate clearly enough to compare before and after

#### Likely Files
1. `src/elitefurretai/rl/config.py`
2. `src/elitefurretai/rl/train.py`
3. `src/elitefurretai/engine/sync_battle_driver.py`

#### Acceptance Criteria
1. Training and benchmark configs default to a 40-step cap unless intentionally overridden
2. Per-update training logs include truncation counts or truncation rate
3. The new cap does not introduce obvious stalled-battle regressions

### Phase 3: Scale Workers After GPU Offload

#### Objective
Increase worker count only after GPU inference is stable, then tune worker count, battles-per-pair, and queue behavior for higher trajectory throughput.

#### Required Behavior
1. Start from the first stable GPU-inference baseline
2. Increase `num_workers` from 4 to at least 8 in a controlled sweep
3. Re-tune `num_battles_per_pair`, actor batch sizes, and queue buffering based on observed throughput and memory
4. Measure learner steps/s rather than only battles/s

#### Notes
This phase should not be merged blind. Worker scaling is only a win if the GPU server can absorb the increased inference load without starving the learner or causing inference backlog collapse.

#### Acceptance Criteria
1. Worker count above 4 is stable for Rust training
2. Learner steps/s improves relative to the first GPU-inference baseline
3. Queue growth remains bounded and shutdown behavior remains clean

### Phase 4: Faster Full-Feature Embedding

#### Objective
Reduce CPU time spent producing the `full` feature vector without changing its semantics.

#### Required Behavior
1. Preserve output shape, ordering, and meaning of the existing `full` vector
2. Replace dict-heavy assembly in hot paths with pre-indexed writes into a reusable array or tensor buffer
3. Maintain compatibility with the current supervised checkpoint and RL training path

#### Validation Requirement
The new fast `full` embed path should be checked against the existing path on representative battle states. If the output differs beyond normal floating-point tolerance, that is a bug.

#### Acceptance Criteria
1. `full` vector compatibility is validated against the old path
2. Rust actor-side embedding time drops materially in profiling or benchmark instrumentation
3. End-to-end learner steps/s improves on top of the GPU-inference baseline

Session result: the Rust model-backed benchmark moved from **0.320 battles/s** / **17.192 decisions/s** to **0.738 battles/s** / **16.363 decisions/s** after routing the hot path through `embed_to_array()`. The benchmark also eliminated truncation in this run (`truncated_battles=0` vs `1` in the baseline run).

Session result: the centralized GPU inference server for Rust actors is implemented and validated on the Rust model-backed benchmark path. The server uses a spawn-backed child process so CUDA can initialize safely, and workers now register their model weights through a pipe-backed client instead of running the forward pass locally.

- 12 battles, `max_concurrent=3`, centralized GPU inference enabled (`--use-central-gpu-inference --inference-device cuda`): **0.789 battles/s** / **21.183 decisions/s** / `28.937` profiled seconds
- 12 battles, `max_concurrent=6`, centralized GPU inference enabled: **1.540 battles/s** / **29.908 decisions/s** / `14.454` profiled seconds
- Relative to the earlier 12-battle, 3-concurrent CPU benchmark on the same shape, the GPU-backed path is modestly faster at the same concurrency and scales much better when concurrency is raised
- The higher-concurrency run is a useful scaling smoke, but a fuller `num_workers` sweep in the main training loop is still the next validation step if we want to measure learner-facing throughput rather than only driver concurrency

---

## Non-Goals For This Pass

1. **No smaller actor / distilled actor implementation yet**
2. **No actor-side switch from `full` to `raw` features**
3. **No CPU quantization work**
4. **No websocket-backend removal or backend unification that breaks the current fallback path**
5. **No broad model-architecture redesign unrelated to the four selected items**

---

## Validation And Benchmark Expectations

The next session should validate each phase rather than land everything and only test at the end.

### Minimum Validation Sequence

1. **GPU inference server smoke test**
    - Rust model-backed benchmark runs successfully
    - action selection and hidden-state updates remain correct
2. **Post-GPU benchmark**
    - compare against the current Rust model-backed maxed baseline: `0.756 battles/s`
    - compare training learner steps/s against the current sustained baseline: `~9.3 steps/s`
3. **Post-step-cap validation**
    - truncation rate is visible
    - long battles no longer die at 30 by default
4. **Worker scaling sweep**
    - test at least worker counts 4, 6, 8, and 10 or 12 depending on memory
5. **Fast full-embed validation**
    - compare old and new `full` feature vectors for equivalence
    - rerun the Rust model-backed benchmark and at least one real training run

### Primary Success Metrics

1. **Learner steps/s**
2. **Battles/s**
3. **Steps per battle**
4. **Truncation rate**
5. **GPU inference batch size distribution / backlog behavior**
6. **Actor-side embedding time**

---

## Deferred Recommendation 5: Mixed Precision / Quantized Actor Inference

### Estimated Speedup: 1.5-2x on actor inference alone, or 1.2-1.5x overall

### The Problem
Actors run the full FP32 model on CPU. The learner already uses mixed precision (FP16 on GPU), but actors don't benefit from this. Modern CPUs (including your setup) support BF16/FP16 SIMD operations, and INT8 quantization can provide 2-4x throughput improvement.

### Evidence
- `torch._C._nn.linear` is 6.19s / 16.96s = 36.5% of profiled time. This is pure matrix multiplication, which directly benefits from reduced precision
- PyTorch's `torch.quantization.quantize_dynamic` can INT8-quantize Linear and LSTM layers with minimal accuracy loss
- LSTM forward is 2.68s — INT8 LSTM is typically 2-3x faster on CPU
- The actor model doesn't need training-level precision; it just needs to produce reasonable action distributions

### Why This Fixes the Core Problem
Quantization reduces the memory bandwidth and compute cost of the dominant operation (linear/LSTM forward pass) on CPU. It's one of the lowest-effort, highest-impact changes — often a few lines of code.

### How the Estimate Was Derived
- INT8 dynamic quantization typically provides 1.5-2x speedup on CPU for transformer-like models (Intel benchmarks, PyTorch docs)
- Linear layer time: 6.19s → ~3.1-4.1s with INT8
- LSTM: 2.68s → ~1.3-1.8s with INT8
- Net inference time: ~16s → ~10-12s → 1.3-1.6x faster per inference batch
- Overall system: 1.2-1.5x given embedding and other costs are unchanged
- Risk: quantization quality depends on weight distribution. Needs validation that action distributions remain close to FP32

### What Would Need to Change
1. **Apply dynamic quantization to actor model copies** before broadcasting to workers:
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
   )
   ```
2. **Validate action distribution quality**: Compare KL divergence between FP32 and INT8 model on a held-out set of battle states
3. **Benchmark before/after** with the existing `rust_model_benchmark.py` harness
4. **Consider ONNX Runtime** for CPU inference: ONNX Runtime's INT8 path is often faster than PyTorch's native quantization

Key files to modify:
- `src/elitefurretai/rl/train.py`: Add quantization step to weight broadcast
- `src/elitefurretai/engine/sync_battle_driver.py`: Accept quantized model
- `src/elitefurretai/rl/model_io.py`: Support serializing/loading quantized models

---

## Recommendation 6: Reduce Wasted Trajectories (Truncation + Invalid Choices)

### Estimated Speedup: 1.2-1.5x on effective learner steps/s

### The Problem
Not all generated trajectories are equally valuable. Two sources of waste:

1. **Truncated battles** (hit `max_battle_steps=30`): The throughput sweep showed 29.2% truncation at optimal settings. Truncated trajectories provide partial learning signal with biased advantage estimates (missing terminal reward)
2. **Showdown invalid choices**: In the 3-update comparison, Showdown had 308 invalid-choice lines vs Rust's 0. Each invalid choice wastes an embedding + inference cycle

### Evidence
- Throughput sweep: `max_concurrent=6` had 292/1000 truncations (29.2%). Non-truncated battles/s was 1.072 vs raw 1.514
- When truncation was lowest (17.2% at `max_concurrent=2`), throughput collapsed to 0.253 b/s — there's a tradeoff
- The 100-update training run used `max_battle_steps=30`, meaning trajectories are killed at 30 turns. Average VGC battle is ~12-18 turns, so 30 should rarely bind — but battles with stalled loops or complex endgames get cut off
- Steps per battle in the Rust training: 27 steps/battle (close to the 30-step cap). This suggests many battles are running long — possibly due to the model being imperfect in endgame scenarios

### Why This Fixes the Core Problem
Truncated trajectories consume the same inference/embedding cost as complete ones but provide lower-quality learning signal. Reducing the truncation rate means more of the compute spent on trajectory generation produces useful gradient steps.

### How the Estimate Was Derived
- If 29% of trajectories are truncated and contribute ~50% of the learning signal of complete ones, effective utilization is: `0.71 × 1.0 + 0.29 × 0.5 = 0.855`
- Reducing truncation to 10% (via better stall detection or raising the cap to 40): `0.90 × 1.0 + 0.10 × 0.5 = 0.95` → effective 1.11x
- Additionally, completing more battles means the terminal reward (win/loss) is incorporated in GAE, improving value function accuracy → faster convergence (hard to quantify, but meaningful)
- Combined impact: 1.2-1.5x on *effective* steps/s (same compute, better learning per step)

### What Would Need to Change
1. **Increase `max_battle_steps` from 30 to 40** (matches `max_seq_len=40` in config) and monitor if truncation rate drops significantly
2. **Improve stall detection**: Rather than killing at a fixed step count, detect when both sides are repeating the same action and fast-forward or terminate with a draw reward
3. **Log truncation rate** as a first-class training metric (currently tracked in benchmark but not in the main training loop with per-update granularity)
4. **Prioritized trajectory sampling**: When the queue is full, prefer non-truncated trajectories over truncated ones

Key files to modify:
- `src/elitefurretai/rl/config.py`: Increase `max_battle_steps` default
- `src/elitefurretai/engine/sync_battle_driver.py`: Add stall detection logic
- `src/elitefurretai/rl/train.py`: Add per-update truncation rate logging

---

## Combined Impact Estimate

The recommendations are not all independent. Here's the expected impact when applying them in combination:

| Combination | Expected Speedup | Notes |
|-------------|-----------------|-------|
| Rec #1 alone (GPU inference) | 3-5x | Highest single-item impact |
| Rec #1 + #4 (GPU inference + more workers) | 4-8x | Workers no longer CPU-bound, can scale to 8-12 |
| Rec #1 + #4 + #3 (GPU inference + more workers + faster full embed) | 5-9x | Embedding becomes the next meaningful actor hotspot |
| Rec #6 alone (reduce waste) | 1.2-1.5x | Orthogonal, always worth doing |
| Selected scope (#1 + #4 + #6 + #3) | 5-10x | Intended implementation bundle for the next session |

### Recommended Execution Order

1. **Medium effort (3-5 days)**: Rec #1 (GPU inference server) — the enabling architectural change and highest-value item
2. **Quick win (1 day)**: Rec #6 (raise `max_battle_steps` to 40, add truncation logging) — low risk, immediate quality improvement
3. **Medium effort (2-3 days)**: Rec #4 (more workers) — tune only after the GPU path is stable
4. **Medium effort (1-2 days)**: Rec #3 (faster `full` embedding) — becomes more important after GPU inference removes the dominant CPU forward-pass cost
5. **Deferred**: Rec #2 (small actor model) — useful only if GPU contention remains a major blocker after the selected scope lands
6. **Deferred**: Rec #5 (CPU quantization) — no longer a priority once actor inference moves to GPU

### Target: Current → 10x

Starting from 9.3 learner steps/s:
- After Rec #1: ~35-60 steps/s (GPU-backed actor inference)
- After Rec #6: better effective sample quality and lower truncation waste on top of that baseline
- After Rec #4: ~55-90 steps/s (more workers once CPU inference is gone)
- After Rec #3: ~60-100+ steps/s (faster `full` embedding reducing the next actor-side hotspot)

Final target for the selected scope: **~60-100+ learner steps/s** (roughly 6-10x over current)

### Current Worker Sweep

The cautious worker-count sweep is in progress on the real learner loop.

Completed datapoint:
- `num_workers=4`
- `num_players=12`
- `num_battles_per_pair=32`
- `max_battle_steps=40`
- `device=cuda`
- `use_mixed_precision=true`
- `max_updates=1`
- `completed_battles=256`
- `duration_seconds=1275`
- `battles_per_second=0.20`
- `learner_steps_per_second=6.04`
- `truncated_battles=0`
- worker RSS held around `2.42 GB`
- inference server RSS held around `1.68 GB`
- main trainer RSS held around `1.13-1.14 GB`

Observed shutdown caveat:
- the 4-worker run finished the learner update cleanly, but the server process emitted a `ConnectionResetError` when the main process terminated remaining worker connections during shutdown
- this did not affect the collected throughput numbers, but it is worth tightening the shutdown path separately if we want quieter long-run exits

Current status:
- `num_workers=6` is running as the next sweep point
- `num_workers=8` will only be run if the 6-worker memory and CPU profile still leaves enough WSL headroom

Updated datapoint:
- `num_workers=6`
- `num_players=18`
- `num_battles_per_pair=32`
- `max_battle_steps=40`
- `device=cuda`
- `use_mixed_precision=true`
- `max_updates=1`
- `completed_battles=256`
- `duration_seconds=1141`
- `battles_per_second=0.22`
- `learner_steps_per_second=6.66`
- `truncated_battles=0`
- worker RSS held around `2.42 GB`
- inference server RSS held around `1.68 GB`
- main trainer RSS held around `1.13 GB`
- shutdown still triggered the same `ConnectionResetError` trace in the central inference server when the parent began terminating the remaining workers

Interpretation:
- 6 workers improved learner steps/s over the 4-worker run (`6.66` vs `6.04`), but the wall-clock gain was modest and the first batch stretched much longer than the 4-worker baseline
- that suggests 6 workers is still stable on this machine, but it is already close to the point where extra concurrency mostly converts into longer batch latency rather than a strong throughput jump

Final datapoint:
- `num_workers=8`
- `num_players=24`
- `num_battles_per_pair=32`
- `max_battle_steps=40`
- `device=cuda`
- `use_mixed_precision=true`
- `max_updates=1`
- `completed_battles=256`
- `duration_seconds=1469`
- `battles_per_second=0.17`
- `learner_steps_per_second=5.18`
- `truncated_battles=0`
- worker RSS held around `2.47 GB`
- inference server RSS held around `1.62 GB`
- main trainer RSS held around `1.13 GB`

Observed behavior:
- the 8-worker run remained memory-stable and completed the first learner update, but throughput fell below both the 4-worker and 6-worker runs
- CPU pressure shifted heavily onto the centralized inference server, which sat near full utilization while per-worker CPU stayed low
- the practical ceiling on this WSL box appears to be 6 workers for the current centralized-inference setup; 8 workers is stable but not the best throughput point

Shutdown caveat:
- the same `ConnectionResetError` appeared in the central inference server during parent-driven worker teardown
- as with the smaller sweep points, this did not corrupt the collected metrics, but it is still worth cleaning up separately if we want quieter long-running exits

Sweep conclusion:
- the worker sweep is now complete enough to answer the hardware question
- 6 workers is the best throughput point seen so far, and 8 workers is not worth adopting on this machine for week-long runs

---

## Appendix: Raw Evidence Sources

### Benchmark results

| Test | Backend | Config | Battles/s |
|------|---------|--------|-----------|
| Random, simple | Rust | max_concurrent=1 | 12.053 |
| Random, maxed | Showdown | max_concurrent=4 | 15.096 |
| Model, simple | Rust | max_concurrent=1 | 0.393 |
| Model, simple | Showdown | max_concurrent=1, batch=4, timeout=0.005 | 0.413 |
| Model, maxed | Rust | max_concurrent=3 | 0.756 |
| Model, maxed | Showdown | max_concurrent=4, batch=8, timeout=0.01 | 0.508 |

### cProfile top functions (Showdown model, maxed, 12 battles)

| Function | tottime | cumtime | calls |
|----------|---------|---------|-------|
| `torch._C._nn.linear` | 6.185s | 6.185s | 9984 |
| `model_archs._dual_expand` | 1.308s | 2.904s | 2704 |
| `{torch.lstm}` | 2.676s | 2.676s | 208 |
| `embedder.embed` | 0.024s | 1.434s | 442 |
| `embedder.generate_opponent_pokemon_features` | 0.227s | 0.733s | 2652 |
| `embedder.generate_pokemon_features` | 0.272s | 0.580s | 2652 |
| `embedder.feature_dict_to_vector` | 0.407s | 0.510s | 442 |
| `ps_client._handle_message` | 0.008s | 2.728s | 1957 |

### Training run metrics (4 workers, Rust, 100-update config)

| Update | Elapsed | Battles | Battles/s | Steps/s | Steps/Battle |
|--------|---------|---------|-----------|---------|--------------|
| 1 | 14m 8s | 256 | 0.30 | 8.19 | 27.1 |
| 2 | 27m 53s | 512 | 0.31 | 8.25 | 26.9 |
| 3 | 37m 7s | 768 | 0.34 | 9.32 | 27.0 |
| 7 | 85m 52s | 1792 | 0.35 | 9.34 | 26.9 |

### Training config (100-update run)

- Model: ~100M param Transformer (curious-darkness-77 checkpoint)
- Workers: 4 (multiprocessing)
- Inference batch: 4, timeout: 5ms
- Train batch: 256 trajectories/update
- Max battle steps: 40
- Device: CUDA (learner), CPU (actors)
- Feature set: full
- Mixed precision: enabled (learner only)
