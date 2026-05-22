# `_COMPILE_LOCK` serialization fix plan

**Date**: 2026-05-15 00:12
**Predecessor**: [2026-05-14-22-14-exploiter-regression-and-run-b-topology-bump.md](2026-05-14-22-14-exploiter-regression-and-run-b-topology-bump.md)
**Throughput investigation parent**: [2026-05-13-13-00-throughput-bottleneck-investigation.md](2026-05-13-13-00-throughput-bottleneck-investigation.md)

## Context

Yesterday's session (continued into today) traced a sustained throughput
regression that fires every time additional centralized inference
services come online (exploiters → −24%; ghosts → −39%). The mechanism
turned out to be different from the GIL-contention story we initially
proposed; it's `_COMPILE_LOCK` serialization. Run B (ghosts on) and
Run B' (ghosts off) bracket the effect cleanly.

## Before State

`InferenceService` runs in a Python thread inside the trainer process,
with each registered model getting its own service. Every forward goes
through [inference_trainer.py:325](../../src/elitefurretai/rl/inference_trainer.py#L325):

```python
with torch.no_grad(), _COMPILE_LOCK:
    turn_logits, tp_logits, values, _, next_hidden = self.agent(...)
```

`_COMPILE_LOCK` is a process-wide `threading.Lock()` introduced to fix
the dynamo "FX symbolically traced of dynamo-optimized function" race
when multiple services compile concurrently
([model-registry-plan §future-work-closed](2026-05-14-00-15-model-registry-plan.md#L242)).
The lock works correctly — but it is acquired **on every forward**,
not just during compile events. With N active services, all forwards
serialize through this single lock.

`ModelRegistry.register` already pre-warms each compiled service at
registration with two synthetic forwards on shape `(1, 1, embedding_size)`
([model_registry.py:142-145](../../src/elitefurretai/rl/model_registry.py#L142-L145)).
This trains dynamo on the warmup shape, but doesn't cover production
shapes `(batch_size, sequence_length, embedding_size)` — `dynamic=True`
generalizes some, but not always. Result: services pay an additional
recompile on their first production-shape request, which holds the lock
for many seconds and starves all other services.

## Problem

The combined effect of (a) lock-held-on-every-forward and (b) first-real-call
recompile causes two distinct throughput costs:

1. **Transient**: when a new service starts firing traffic, its first
   production-shape forward triggers a recompile. The lock is held for
   tens of seconds; every other service's forward stalls. Visible as
   the wave of `Slow battle message ... handler=8.00s` warnings during
   startup. Eventually clears as services warm.
2. **Sustained**: even fully-compiled services hold the lock for the
   wall-clock duration of every forward (~10-50 ms each). With multiple
   active services, forwards are strictly serialized. The dominant
   service (main, ~88% of traffic at full curriculum) loses lock-time
   share proportional to other services' aggregate load.

The sustained cost matches the data:

| Run | Curriculum | Steady traj/s | Δ vs Run B' |
|---|---|---|---|
| Run B' (ghosts=0, ghost services idle) | self_play 0.43, no ghosts | **5.4** | baseline |
| Run B (ghosts=0.125, 5 ghost services serving) | self_play 0.375, ghosts 0.125 | **3.72** | −31% |
| Run B pre-update-400 (ghosts fall back to self_play) | self_play eff. ~0.5 | 6.1 | +13% (more self-play = more double-side main load) |

Removing ghost service traffic recovers throughput by ~31%. Run B'
confirms throughput does NOT drop after update 400 when ghosts are
zeroed — the prior drop was entirely caused by ghost services serving
traffic and contending for `_COMPILE_LOCK`.

The `py-spy dump` taken during Run B' startup ([discussion above](#))
caught bc in a full first-call compile holding the lock; main was idle
at line 325 acquiring it. Direct visual evidence of the mechanism.

## Solution

Three-layer fix, ordered cheap → architectural. We propose committing
to layer 1 (warmup) and layer 2 (narrow lock) immediately; layer 3
(multi-process) stays in the architectural backlog as the further
upside path if needed after measuring 1+2.

### Layer 1 — Pre-warm services with realistic shapes

`ModelRegistry.register` already pre-warms with `(1, 1, embedding_size)`.
Change to warm a small set of representative production shapes so dynamo
doesn't recompile on first real request:

```python
# In ModelRegistry.register, after building `compiled`:
with torch.no_grad():
    # warm the (1, 1) "synthetic" shape (existing call)
    x_small = torch.zeros(1, 1, self.embedding_size, device=self.device)
    _, _, _, _, ctx = compiled(x_small, None)
    compiled(x_small, ctx)
    # warm representative production shapes: small batch + various seq lens
    for B in (1, 4, 16, 32):           # batch_size range
        for T in (1, 8, 20, 40):       # seq_len range (under max_seq_len)
            x = torch.zeros(B, T, self.embedding_size, device=self.device)
            compiled(x, None)
```

This pays compile time **at trainer startup, before workers connect**,
when no traffic is in flight. Subsequent first-real-call requests hit
the dynamo cache and skip the recompile. Eliminates the transient cost
and the wave of Slow-battle-message warnings during ramp-up.

Cost: ~30-90 s extra startup time per service × number of services, but
paid once and not visible in throughput steady-state. Lazy alternative:
warm only after first weight sync (so we don't waste startup on services
the curriculum may never route to), but probably not worth the
complexity.

### Layer 2 — Narrow `_COMPILE_LOCK` to compilation events only

After the warmup, mark the service as `is_compiled = True`. The forward
path then skips lock acquisition unless an unexpected recompile is
detected. Concretely:

```python
# inference_trainer.py:325 becomes:
if self._is_compiled.is_set():
    with torch.no_grad():
        outputs = self.agent(...)
else:
    with torch.no_grad(), _COMPILE_LOCK:
        outputs = self.agent(...)
        self._is_compiled.set()
```

`_is_compiled` is a per-service `threading.Event`. The registry's
warmup sets it after the production-shape warmup completes. The lock
is held only during the initial unmark→mark transition (i.e., genuinely
during compile) or in defensive failure modes.

The remaining concern is **steady-state recompile**: a request shape
outside the warmed-up envelope could trigger dynamo to recompile. With
`dynamic=True` and the wider warmup shapes from Layer 1, this should
not happen in practice, but it would be a correctness regression
(silent race) if it did. Two mitigations:

1. **Detect compile events via dynamo hook** and re-take the lock when
   compile fires. `torch._dynamo` exposes guards/hooks for "about to
   compile." If hookable, wrap with that.
2. **Guard with `torch.compiler.is_compiling()` check** if available,
   or fall back to "lock on every forward for the first N batches after
   any sync_weights" as a conservative compromise.

For the first pass we recommend Layer 1 (richer warmup) + Layer 2a
(simple `is_compiled` flag), and add an explicit assertion or
diagnostic log when a service trips back to "is_compiled = False" so
we'd see recompile events if they occur.

### Layer 3 — Multi-process services (deferred)

Each `InferenceService` runs in its own subprocess. Workers route via
`mp.Queue` to per-service subprocesses. Independent dynamo state per
process → no global race → no need for `_COMPILE_LOCK` at all.
Architecturally cleanest; also frees the GIL beyond just lock
contention. Cost: real refactor (~3-5 days of work, plus IPC overhead
that needs measurement). Defer until Layer 1+2 numbers are in.

## Reasoning

- Layer 1 alone makes the **transient** startup wave disappear — fewer
  `Slow battle` warnings, faster ramp to steady state. Easy win.
- Layer 2 is what unlocks the **sustained** throughput recovery: it
  lets multiple services run forwards in parallel (modulo GIL, which
  still serializes Python-side execution but releases during CUDA
  syncs).
- After Layer 1+2, expected steady traj/s with full curriculum: ~5+
  (Run B' baseline) plus whatever the additional ghost-side traffic
  contributes beyond what main saturates. Hard to predict; would
  measure.
- Layer 3 is the only fix that escapes the GIL for actual CPU-bound
  Python overhead in the service threads. We don't have evidence yet
  that GIL contention (separate from the lock) is the next bottleneck;
  measure first.

## Planned Next Steps

1. **Confirm Run B' steady-state numbers** as the new baseline (≥10
   updates past update 500's checkpoint event to ensure post-checkpoint
   behavior matches pre-checkpoint).
2. **Implement Layer 1** (richer-shape warmup in `ModelRegistry.register`).
   Measure: number of `Slow battle` warnings during startup; time to
   first Update line. Compare against current Run B' which has 1500+
   Slow battles in warmup.
3. **Implement Layer 2** (`is_compiled` flag in `InferenceService`).
   Run full curriculum (re-enable `ghosts: 0.125`). Measure steady-state
   traj/s. Goal: recover to ≥5 traj/s.
4. **If Layer 1+2 insufficient**: commit to Layer 3 (multi-process
   services). Otherwise mark Plan C as deprecated since the cheaper
   fix sufficed.
5. **Re-enable exploiter co-training** (`train_exploiter: 0.1`,
   `exploiters: 0.1`) and confirm no regression.

## Risks

- **Layer 2 correctness**: if a recompile fires without re-taking the
  lock, we hit the dynamo race that motivated the lock originally.
  Symptom: `RuntimeError: Detected that you are using FX to symbolically
  trace a dynamo-optimized function`. Mitigation: assertion + diagnostic
  log, and the richer warmup in Layer 1 should make this case rare.
- **Startup time**: Layer 1 grows trainer warmup. Estimate: 5-15 s per
  shape combination × 12 services × maybe 4-8 unique shapes after
  dynamic-shape generalization = bounded by a couple of minutes added
  to startup. Acceptable; recover ~30% throughput in exchange.
- **Inductor cache thrash**: pre-warming with many shapes may evict
  some cached artifacts. Probably fine since the architecture is the
  same across services (same cache keys), but worth monitoring the
  first run after Layer 1 ships.

## Updates

### 2026-05-15 ~00:50 — Layer 1+2 implemented; Layer 2 did not deliver; reverted

**What was shipped**:
- Layer 1: broadened the registration-time warmup in
  `ModelRegistry.register` from a single `(1,1,E)` forward to
  `B ∈ {1,4,16,32}` with `T = 1` (x is always single-turn input;
  ctx is internally padded to max_seq_len, so the only production
  shape that varies is B). Two earlier attempts to vary T crashed at
  the model's transformer mask-reshape boundary, motivating the
  B-only envelope.
- Layer 2: added `RealModelBatchHandler._compile_complete` Event +
  `mark_compile_complete()` method; modified the per-forward path in
  `__call__` to skip `_COMPILE_LOCK` when the event was set; registry
  set it after each model's warmup completed.

**Test run (Run C, wandb `celestial-bush-39`)**: full curriculum
restored (ghosts 0.1, exploiters 0.1, train_exploiter 0.1), same
topology as Run B (`num_players=16, max_concurrent=48,
num_battles_per_pair=48`). Resumed from
`pretty-jazz-4/main_model_step_365.pt`.

| Run | Curriculum | Steady traj/s |
|---|---|---|
| Run B' (ghosts off, no fix) | self_play heavy | ~5.4 |
| Run B pre-400 (ghost fallback to self_play) | effectively self_play heavy | 6.1 |
| Run B post-400 (no fix, ghosts active) | full curriculum | 3.72 |
| **Run C (Layer 1+2 fix, full curriculum)** | **full curriculum** | **~3.5** |

Plateau confirmed across updates 368-373: 2.88, 3.32, 3.46, 3.45,
3.75, 3.56, 3.56 → no climb past Run B post-400. The Layer 2
lock-bypass did NOT recover throughput.

**Two issues uncovered during the test**:

1. **Correctness bug in Layer 2's design**. Dynamo's eval_frame has
   global state across services. When service A is mid-compile (FX
   tracing) and service B's already-warmed handler runs a forward
   without acquiring the lock, dynamo raises `RuntimeError: Detected
   that you are using FX to symbolically trace a dynamo-optimized
   function`. This fires during the registration overlap window: as
   soon as `register("victim")` returns, victim's service thread
   starts polling its queue and serving any traffic that arrives,
   while later registers (`register("ghost_0")` etc.) are still
   doing warmup compiles. The original `_COMPILE_LOCK` guarded
   against this; Layer 2's bypass broke the guarantee. Observed as
   ~10 "handler raised; dropping batch" errors in a 30 s window
   during startup before all services were warmed.

2. **`_COMPILE_LOCK` is not the dominant serializer**. With the
   bypass active and services running concurrent forwards, throughput
   matched the pre-fix Run B post-400 baseline (~3.7 traj/s). The
   real bottleneck appears to be the GIL itself: per-forward Python
   work (request decode, tensor packing, hidden-state slicing in
   `_pad_transformer_context` and `_slice_next_hidden`, sampling,
   response encode) runs while holding the GIL. CUDA work releases
   the GIL during compute, but that's a smaller fraction of total
   forward time. With 12 service threads doing this work, they
   serialize on the GIL whether or not the explicit
   `_COMPILE_LOCK` is held.

**Decision**: revert Layer 2; keep Layer 1.

- Layer 1 stays. The broader B-shape warmup is harmless (a few extra
  startup seconds) and plausibly reduces in-band recompiles when
  first production-shape requests arrive. No measured negative impact.
- Layer 2 reverted. Restored `with torch.no_grad(), _COMPILE_LOCK:`
  on every forward at
  [inference_trainer.py:325](../../src/elitefurretai/rl/inference_trainer.py#L325).
  Removed `_compile_complete` event + `mark_compile_complete()` from
  `RealModelBatchHandler` and the post-warmup call in
  `ModelRegistry.register`. Quality gates clean (ruff, pyright,
  27/27 tests pass).

**Next step**: the throughput recovery requires Plan C (multi-process
inference services). See "What data would prove Plan C is the right
fix" below.

## What data would prove Plan C is the right fix

Plan C is a real refactor (~3-5 days) and we shouldn't commit to it
without conviction. Before starting:

1. **py-spy with GIL accounting on the trainer.** Run Run B (or a
   reproducer with full curriculum) and capture
   `py-spy record --pid <trainer> --duration 120 --output trainer.svg`
   plus `py-spy top --pid <trainer>` for a snapshot. We want two
   measurements:
   - **Per-thread Active% in steady state**. The hypothesis predicts
     the sum across InferenceService-* threads is close to 100%
     (one CPU's worth of work, GIL-limited). If instead the sum is
     several hundred percent, we're not GIL-bound and Plan C
     wouldn't help much.
   - **GIL-held time per thread**. If service threads spend ≥70% of
     their Active time holding the GIL, multi-process is the right
     fix. If they spend most of their time waiting on CUDA syncs
     (`cudaStreamSynchronize` etc.), the bottleneck is GPU compute
     and Plan C wouldn't help — distillation (Plan D) would.

2. **One-service vs all-services throughput delta with the same
   topology.** Re-run with `bc_player=0, ghosts=0, exploiters=0,
   train_exploiter=0` (collapsed to a single active service) and
   measure traj/s. Compare against full curriculum (12 active
   services). We already have most of this comparison:
   - Run B' (~5.4) — main + bc + idle services
   - Run C (~3.5) — main + bc + ghosts + exploiters + idle exploiter_snaps

   A cleaner "main only" baseline would tighten the comparison.
   Expected: main-only matches single-process ceiling (~6 traj/s);
   full multi-service degrades by ~40% from GIL contention. That
   gap IS what Plan C recovers.

3. **Microbenchmark of multi-process inference.** Before committing
   to Plan C's full integration cost, write a 100-line script that:
   - Builds two `RNaDModel` instances on the same GPU
   - Runs N concurrent inference forwards via:
     (a) two threads in one process (`threading.Thread`)
     (b) two subprocesses via `torch.multiprocessing`
   - Measures total throughput for each config

   If (b) achieves ≥1.5x (a)'s throughput, multi-process gives real
   parallelism on this workload and Plan C is validated. If (b) ≤
   (a), there's a deeper shared-resource limit (CUDA driver,
   PCIe bandwidth) and Plan C wouldn't help; that points at Plan D
   (distillation) instead.

4. **Per-service forward latency under load.** Add timing around
   `self.agent(...)` in `RealModelBatchHandler.__call__`. If
   per-forward latency for the same shape increases 2-3x going from
   1 service active to 12, that's GIL+lock contention quantified.
   If latency is constant but the services-per-second rate drops,
   the bottleneck is elsewhere (queue throughput, dispatch).

**Cheapest first**: do #1 (py-spy GIL accounting) — ~15 minutes of
work, gives direct evidence. If GIL-held % is high, do #3 (microbench)
to confirm multi-process delivers parallelism on this exact GPU and
model. Commit to Plan C only after both data points are in the green.

If py-spy shows the threads are mostly CUDA-waiting (not GIL-held),
re-route: Plan D (distillation to a smaller model) becomes the
right path, since smaller per-forward Python+CUDA time means less
serialization tax per request even with the existing architecture.

### 2026-05-15 ~01:30 — Microbench results: Plan C validated by scaling, not at N=2

Ran [bench_multi_process_inference.py](../../src/elitefurretai/rl/analyze/bench_multi_process_inference.py)
with a production-shape transformer (4 layers, 8 heads, d_model=512,
ff_dim=1024), production batch_size=32, and realistic per-forward
Python wrapper (np.stack + dict lookups + sampling + response build,
mimicking `RealModelBatchHandler.__call__`). Scaling sweep across
N ∈ {1, 2, 4}:

```
  N |   A (lock) | B (no lock) |  C (procs) |   B/A |   C/A |   C/B
  1 |      218.0 |       239.1 |      197.3 |  1.10 |  0.91 |  0.83
  2 |      240.4 |       241.6 |      287.2 |  1.00 |  1.19 |  1.19
  4 |      159.0 |       153.6 |      335.7 |  0.97 |  2.11 |  2.19
```

Per-subprocess RSS measured at ~1000 MB; GPU alloc ~44 MB on this
tiny test model (production-sized model would push GPU alloc to
~150 MB per subprocess).

**Key signals**:

1. **Threads regress past N=2.** A goes 218 → 240 → 159 across N=1,2,4.
   Adding more threads beyond 2 makes things *worse*. Direct evidence
   that GIL is the binding constraint on per-forward Python work — not
   the lock, not CUDA. Lock vs no-lock difference (A vs B) is noise.
2. **Processes scale sub-linearly but positively.** C goes 197 → 287
   → 336. Sub-linear because CUDA still time-slices contexts on the
   GPU (no MPS on WSL2 — see below), but the Python wrapper work runs
   in genuine parallel.
3. **Decision threshold at N=2 ("C/A ≥ 1.5x") fails.** C/A = 1.19x at
   N=2. The threshold was too narrow; the actual signal is the
   *scaling trajectory*. At N=4: C/A = 2.11x, well above the 1.5x bar.

**MPS on WSL2 — not available**. NVIDIA's Multi-Process Service
requires the MPS daemon on a Linux host with direct GPU access. WSL2's
CUDA stack uses a stub `libcuda.so` that forwards to the Windows host
driver and does not expose the kernel interfaces MPS needs. Without
MPS, multi-process inference shares the GPU via time-slicing — what
my microbench measured. Native Linux dual-boot would unlock MPS;
otherwise Plan C's GPU-side gain is bounded.

**Conclusions**:

- **Skip the 1-extra-process variant.** N=2 gives only +19% — not worth
  a 1-2 day implementation.
- **Commit to a ≥3-subprocess (≥4 process group) design.** That's where
  multi-process actually pays for itself.
- Translating microbench scaling to production (currently 3.5 traj/s
  with 12 services in 1 process): a 4-process-group design projects
  to roughly 2x → ~7 traj/s on full curriculum.
- Memory cost: 3 extra subprocesses × ~1 GB host = +3 GB; +0.5-1 GB
  GPU. Fits with margin under the 22 GB watchdog.

See [2026-05-15-01-30-plan-c-4-process-implementation.md](2026-05-15-01-30-plan-c-4-process-implementation.md)
for the implementation plan.
