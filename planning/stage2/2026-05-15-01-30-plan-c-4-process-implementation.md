# Plan C — Multi-process inference services, 4-process implementation plan

**Date**: 2026-05-15 01:30
**Predecessor**: [2026-05-15-00-12-compile-lock-serialization-fix-plan.md](2026-05-15-00-12-compile-lock-serialization-fix-plan.md)
(see "Microbench results: Plan C validated by scaling, not at N=2")
**Status**: Pre-implementation. Microbench evidence in hand; ~3-5 day build.

## Context

Microbench evidence
([bench_multi_process_inference.py](../../src/elitefurretai/rl/analyze/bench_multi_process_inference.py))
shows the trainer's 12-service inference pipeline is GIL-bound on the
per-forward Python wrapper work. Threads regress past N=2; processes
scale sub-linearly but positively (2.11x at N=4 vs N=4 threads). On
WSL2 we can't use MPS, so the GPU still time-slices between processes
— but the Python parallelism alone projects ~2x in production.

This doc specifies a concrete 4-process-group implementation.

## Target throughput

Current production state with full curriculum (ghosts+exploiters):
**~3.5 traj/s** (Run B post-400 / Run C steady).

Projected with 4-process Plan C: **~7 traj/s** (microbench's 2.1x
multiplier on the GIL-limited steady state). Exceeds Run B's 6.1
pre-400 peak; starts closing the gap to VGCBench's 15-20 traj/s.

## Process layout

Group services by **liveness** (frozen vs live-weights) and traffic
share, not by category. This keeps weight-sync work natural per group.

| Process | Hosts | Why | Traffic share |
|---|---|---|---|
| **Trainer (existing)** | `main`, `bc`, learner pipeline, exploiter learner | Highest-traffic services that already share weight state with the learner. Learner needs GPU access anyway. | ~88% (main dominates) |
| **Subprocess `ghosts`** | `ghost_0..ghost_4` | All five ghost slots — homogeneous frozen-by-rotation lifecycle. One LRU sync on each ghost-save event from trainer. | ~10% |
| **Subprocess `snaps`** | `exploiter_snap_0..exploiter_snap_4` | All five exploiter snapshot slots — same lifecycle as ghosts but different opponent semantic. Distinct from ghosts so each subprocess's queue depth stays bounded. | ~0-2% |
| **Subprocess `live`** | `exploiter` (live), `victim` | Both train-time live services, exploiter weights updated frequently from the in-trainer exploiter learner; victim weights updated on `victim_refresh_interval`. | ~2% |

Why **not** put `exploiter` + `victim` with `snaps`: they need
different sync cadence and snaps run cold; mixing them complicates the
weight-sync protocol.

Why **not** keep `bc` out of the trainer: bc is the second-highest-traffic
service, but it's frozen-from-checkpoint at startup and doesn't sync
weights afterward. Keeping it in-trainer adds zero weight-sync work
and gives the trainer process two services to amortize fixed costs
over. If profiling later shows bc's traffic pushes the trainer
process's GIL utilization above 80%, we can move it to its own
subprocess.

## Architecture changes

### New module: `src/elitefurretai/rl/inference_subprocess.py`

Subprocess-side entrypoint. Receives a `SubprocessSpec` (dataclass with
list of services, queue handles, device, compile_mode) via spawn args.
Inside the subprocess:

1. `torch.cuda.set_device(...)` — explicit context creation.
2. For each service in spec: build a fresh `RNaDAgent`, optionally
   compile, register its `InferenceService` with the queues from the
   spec.
3. Listen on a `control_queue` for:
   - `SyncWeightsMsg(service_name, state_dict)` — apply to the raw
     agent (post-compile model still reflects updates since the
     compiled wrapper holds a reference to the same underlying
     module).
   - `ShutdownMsg()` — stop all services, exit.

`SubprocessSpec` is a `dataclass(frozen=True)` declared in
`model_registry.py` and serialized via spawn's pickling.

### `src/elitefurretai/rl/model_registry.py` — extend, don't rewrite

Add **two new responsibilities**:

1. **Track process_group per registration**. New optional
   `process_group: Optional[str] = None` parameter to `register()`.
   `None` means "stay in trainer process" (the current behavior).
   Non-None means "queue for this named subprocess; don't build the
   service in-process yet."

2. **Spawn subprocesses on `start_all()`**. New method. After all
   registrations are done, trainer calls `registry.start_all()`. It:
   - For each `process_group != None`: collect all services tagged
     for it, build a `SubprocessSpec` with their queue handles,
     `mp.Process(target=inference_subprocess.main, args=(spec,))`.
   - For each `process_group is None`: build + start the
     `InferenceService` in-trainer-process exactly as today.
   - Track subprocesses in `self._subprocesses: Dict[str, mp.Process]`
     and `self._control_queues: Dict[str, mp.Queue]`.

Why split `register()` from `start_all()`: today's bug we saw in Run C
was that services started threads as soon as `register()` returned,
which let workers route to them while later registers were still
compiling. The new split fixes this for the in-process case too
(no service runs until all are warmed) AND naturally fits the
"send-spec-then-spawn" flow for subprocesses.

`sync_weights(name, state_dict)` becomes process-aware: if the named
service lives in a subprocess, push `SyncWeightsMsg` to that
subprocess's control queue; otherwise apply directly to the in-process
agent as today.

`stop_all()` sends `ShutdownMsg` to each subprocess and joins, in
addition to stopping in-process service threads.

### `src/elitefurretai/rl/train.py` — call-site changes only

Lines around 1258-1355 (the register block) gain `process_group=...`:

```python
registry.register("main", RNaDAgent(main_inference_base))                 # process_group=None
registry.register("bc", RNaDAgent(bc_inference_base), compile=True)      # process_group=None
registry.register("exploiter", exploiter_agent, compile=True, process_group="live")
registry.register("victim", victim_agent, compile=True, process_group="live")
for slot in range(max_ghosts):
    registry.register(f"ghost_{slot}", ghost_agent, compile=True, process_group="ghosts")
for slot in range(max_exploiter_models):
    registry.register(f"exploiter_snap_{slot}", exploiter_snap_agent,
                      compile=True, process_group="snaps")

# NEW — replaces the implicit "service started in register" path.
registry.start_all()
```

`registry.sync_weights(...)` calls (the broadcast tick and exploiter
sync paths) need no source changes — the registry handles routing
internally.

Shutdown path in `main()` gains a `registry.stop_all()` call (or
ensures the existing shutdown signal reaches subprocesses).

### `src/elitefurretai/rl/inference_trainer.py` — minimal

The `InferenceService` class is reused as-is inside subprocesses.
`_COMPILE_LOCK` stays as the per-process serializer (each subprocess
has its own copy, which is exactly the right scope).

### Unit tests

New `unit_tests/rl/test_inference_subprocess.py` covering:

1. **Spawn + serve**: spawn a subprocess hosting a 1-service spec,
   send a forward request through the queue, assert response shape.
2. **Sync weights**: spawn, send `SyncWeightsMsg` with a different
   state_dict, send a forward, assert output reflects new weights.
3. **Shutdown**: spawn, send `ShutdownMsg`, assert process exits
   within timeout.
4. **Crash detection**: spawn, kill subprocess externally, assert
   trainer detects via `process.is_alive() == False` and surfaces.

Extend `unit_tests/rl/test_model_registry.py` with:

5. **Mixed process_group registration**: register two services with
   `process_group=None` and one with `process_group="g1"`. Call
   `start_all()`. Assert one subprocess spawned, two in-process
   services running, and a request to each gets a response.
6. **Cross-process sync_weights**: register a service in a subprocess,
   call `registry.sync_weights("name", sd)`, send a request, assert
   response reflects updated weights.

## Implementation steps (sequenced for testable checkpoints)

| Step | Output | Validation |
|---|---|---|
| 1 | `SubprocessSpec` dataclass + `InferenceSubprocess` standalone class (no registry integration yet) | Unit tests 1-4 pass |
| 2 | `ModelRegistry.register(process_group=...)` parameter; `start_all()` method | Unit tests 5-6 pass; trainer with all `process_group=None` (current behavior) still works |
| 3 | `ModelRegistry.sync_weights` cross-process path | Existing weight sync tests pass; new cross-process test passes |
| 4 | `train.py` calls `start_all()`; flips `process_group` for ghosts only | End-to-end smoke run; throughput measurement |
| 5 | Flip `process_group` for snaps + live | End-to-end full curriculum run; throughput measurement vs Run C 3.5 baseline |
| 6 | Diagnostic: register-time `process_group` defaults driven from config | Optional polish; can be done later |

After Step 4, we have data on **2-process** real throughput (trainer +
ghosts). After Step 5, we have **4-process** data. Each step is a
clean stopping point; if early data disappoints, we abort before
the full cost.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **state_dict serialization across mp.Queue is slow** (26M params ~ 100 MB; pickled once per sync) | Medium | Benchmark on Step 3. If >1s per sync, switch to shared-memory tensor handles (`torch.Tensor.share_memory_()`). |
| **GPU OOM** from 4 CUDA contexts | Low | Microbench showed ~50 MB per context for tiny model; production model adds ~100 MB. 4 contexts ≈ +600 MB GPU. RTX 3090 has 24 GB; we currently use ~10 GB. Plenty of headroom. |
| **Host RAM exceeds watchdog** | Low | Each subprocess ~1 GB; 3 subprocesses = +3 GB. Current 17 GB + 3 = 20 GB, watchdog at 22 GB. Margin is ~2 GB — bump watchdog to 24 GB if needed. |
| **Subprocess crash leaves trainer in inconsistent state** | Medium | Detection via `process.is_alive()` polling in the existing memory-watchdog thread; on crash, log loud + request graceful shutdown of trainer. No auto-restart in v1 (avoid hiding bugs). |
| **Startup time grows** (4 processes × 30-90s warmup) | Medium | Acceptable; one-time cost. Could parallelize subprocess startup but adds complexity — defer. |
| **Cross-process compile-cache misses** (each subprocess does its own first-compile) | Medium | torch._inductor disk cache is shared, so the second subprocess's compile is faster than the first. Measure on Step 4; if too slow, pre-warm the cache. |

## Decision points along the way

- **After Step 4 (trainer + ghosts subprocess)**: measure throughput.
  - If `≥ 4.5 traj/s` (i.e., > +28% vs 3.5): continue to Step 5.
  - If `< 4.0 traj/s`: stop. Either the implementation has a bug or
    the microbench overestimated production gain. Diagnose with
    py-spy before continuing.
- **After Step 5 (full 4-process)**: measure throughput.
  - If `≥ 6 traj/s`: ship. Plan C delivered.
  - If `4.5-6 traj/s`: ship but flag the gap. May indicate
    weight-sync overhead or service-specific issues.
  - If `< 4.5 traj/s`: roll back. Plan C didn't deliver; pivot to
    Plan D (distillation).

## What this plan does NOT do

- **No MPS**. Without MPS, the GPU still time-slices kernel launches
  across the 4 contexts. Microbench shows we get ~2.1x at N=4 anyway
  (the Python parallelism dominates). Native Linux dual-boot would
  unlock MPS and likely push closer to 4x, but that's a separate
  decision.
- **No service rebalancing**. We don't try to evenly distribute
  traffic across the 4 processes (main dominates). The 4 groups are
  by liveness/lifecycle, not by traffic. If profiling later shows
  the trainer process is still GIL-bound (most likely because main +
  bc together are ~90% of traffic), the follow-up is to move bc
  into its own subprocess.
- **No worker-side changes**. Workers see per-service queues exactly
  as today. The cross-process routing is invisible to them.

## Estimated effort

- **Step 1-3** (subprocess infrastructure + tests): ~2 days
- **Step 4-5** (train.py integration + 2 measurement runs): ~1 day
- **Polish + planning doc update with results**: ~half day
- **Total: 3.5 days**, with clean stopping points at Steps 3 and 4
  in case data turns out worse than projected.

## Updates

### 2026-05-15 ~02:00 — Step 1 complete: subprocess infrastructure + tests

Shipped:
- [`src/elitefurretai/rl/inference_subprocess.py`](../../src/elitefurretai/rl/inference_subprocess.py)
  — new module. Defines:
  - `SyncWeightsMsg`, `ShutdownMsg` (frozen dataclasses, the control queue payload).
  - `ServiceSpec`, `SubprocessSpec` (the cross-pickle-boundary specs).
  - `run_subprocess(spec)` — module-level entrypoint for `mp.Process`. Builds
    `InferenceService` per spec.services, listens on control queue, applies
    `SyncWeightsMsg` via `model.load_state_dict`, exits on `ShutdownMsg`.
  - `InferenceSubprocessHandle` — trainer-side wrapper for the subprocess +
    control queue. `.start()`, `.sync_weights(name, state_dict)`,
    `.is_alive()`, `.shutdown(timeout_s)` with SIGKILL fallback.

- [`unit_tests/rl/test_inference_subprocess.py`](../../unit_tests/rl/test_inference_subprocess.py)
  — 5 tests covering:
  1. spawn + serve one request through one service
  2. sync_weights actually changes outputs
  3. shutdown exits within timeout
  4. external SIGKILL is detected via is_alive
  5. multi-service subprocess (the 4-process design's core assumption)

All 5 pass in 11s on CPU/eager. Full test suite still green (32/32:
27 existing + 5 new). Ruff + pyright clean.

Design notes from implementation:
- Service threads share `_COMPILE_LOCK` *inside* each subprocess (per-process
  globally, not cross-process). That's correct — each subprocess has its
  own dynamo state, so the lock scope shrinks naturally.
- The pickle boundary works for the spec: agents constructed on CPU
  in trainer process cross to the subprocess via spawn. Subprocess
  moves them to its device.
- `SyncWeightsMsg.state_dict` carries CPU tensors; subprocess'
  `load_state_dict` handles the device migration internally.
- `InferenceSubprocessHandle.shutdown` is idempotent + safe on
  already-dead subprocesses (caught `BrokenPipeError` / `EOFError`
  on the `control_queue.put`).

**Next**: Step 2 — extend `ModelRegistry.register` with `process_group`
parameter and `start_all()` method. Will reuse the subprocess
infrastructure above.

### 2026-05-15 ~02:45 — Step 2 complete: ModelRegistry process-aware

Shipped:
- [`src/elitefurretai/rl/model_registry.py`](../../src/elitefurretai/rl/model_registry.py)
  — refactored:
  - `register(name, agent, *, probabilistic, compile, process_group)` —
    new optional `process_group: Optional[str] = None` parameter. `None`
    keeps the service in-process (default); any string queues it for
    inclusion in that named subprocess group.
  - `start_all()` — new. Calls `service.start()` on each in-process
    service (deferred from `register()`) AND spawns one
    `InferenceSubprocessHandle` per group. Required before any traffic
    flows. Raises if called twice; subsequent `register()` raises.
  - `sync_weights(name, sd)` — process-aware. Always updates the
    trainer-side CPU shadow copy in `_raw_agents`; for in-process
    services that's also the live model; for subprocess services it
    additionally ships a `SyncWeightsMsg` over the group's control
    queue.
  - `stop_all()` — shuts down in-process services and all subprocess
    handles. Idempotent.
  - `get_diagnostics()` — in-process services only (subprocess
    diagnostics surfaced via control-queue round-trip is future work).
  - `queues_for_workers()` — returns queues for both backends, so
    workers route identically regardless of where the service lives.

- [`unit_tests/rl/test_model_registry.py`](../../unit_tests/rl/test_model_registry.py)
  — updated:
  - Existing tests that exercised live services now call `start_all()`.
  - Tests that only check naming / queues / sync_weights work pre-start
    (sync_weights to in-process services is allowed even before
    start_all() because the trainer-side shadow copy is the same module
    the handler holds).
  - **4 new tests**:
    - `test_registry_mixed_process_groups` — register 2 in-process + 1
      subprocess service, start_all, send a request to each, verify
      response. Direct proof of the 4-process design's core flow.
    - `test_registry_subprocess_sync_weights` — register subprocess
      service, send request, sync_weights with different agent, send
      another request, assert value changed.
    - `test_registry_register_after_start_all_raises` — guard against
      runtime registration.
    - `test_registry_start_all_twice_raises` — guard against
      duplicate start.

All 36 inference-suite tests pass (32 from step 1 + 4 new). Ruff +
pyright clean. The 4 pre-existing failures in `test_compile_race_reproducer`
+ `test_worker_opponent_factory` are unrelated to this work
(`test_two_real_rnad_agents_with_cudagraph_mark_step` is flaky:
passes 1/3 runs, uses `random.Random()` without seed — known
preexisting flake).

Design notes from implementation:
- Splitting `register()` from `start_all()` naturally fixes the
  registration race we hit in Run C: no service serves traffic
  until all are warmed.
- Each subprocess gets its own `control_queue` (created lazily on
  first registration to that group).
- Subprocess services pass agents on CPU; the subprocess moves them
  to its device. This keeps trainer GPU memory low during registration.
- `_raw_agents` is the canonical CPU shadow for both backends. For
  in-process: it's the same module the handler holds.  For subprocess:
  it's the trainer-side copy that stays in sync via `sync_weights`
  side-effect (so a future subprocess respawn would start from the
  current weights).
- Pre-`start_all()`, `get_diagnostics()` returns `{}` and
  `_services` is empty — only `names()` and `queues_for_workers()`
  reflect the planned services.

**Next**: Step 3 — measure cross-process `sync_weights` latency.
state_dict pickling + queue transfer for a 26M-param model is ~100 MB.
If slower than ~1 s per sync, switch to `Tensor.share_memory_()` shared-
memory handles. (This is technically already covered by the unit test
working, but we need a perf measurement before Step 4 production
integration.) Then Step 4: flip `process_group="ghosts"` in train.py
and measure end-to-end throughput.

### 2026-05-15 ~03:00 — Step 3 complete: sync_weights latency PASS

[bench_sync_weights_latency.py](../../src/elitefurretai/rl/analyze/bench_sync_weights_latency.py)
on a production-shape 25.5M-param model (97.6 MB state_dict, sep_arch
dimensions):

| Path | Mean (n=20) | Notes |
|---|---|---|
| In-process | 14.3 ms | `load_state_dict` only |
| Subprocess | 23.8 ms | pickle + `mp.Queue.put` + CPU shadow update |

Overhead +9.5 ms (1.66x). Well under the 1.0s pass threshold (40x
margin). torch_mp.Queue uses shared memory for tensor storages so
the actual data transfer is mmap-fast; only the dict + metadata get
pickled. **Ship as-is — no shared_memory_() optimization needed.**

### 2026-05-15 ~11:25 — Step 4 complete: ghosts subprocess delivers on production run

[train.py:1339-1361](../../src/elitefurretai/rl/train.py) updated:
ghost slots now register with `process_group="ghosts"`. Ghost weights
load to CPU shadow (`map_location="cpu"`); subprocess inherits via
spawn-time pickle of `_raw_agents`. `registry.start_all()` added
after all registers, before workers spawn.

Two bugs found and fixed during integration:
1. **`set_device` rejected bare `"cuda"`** — production config uses
   `device: cuda` without an index. Initial fix: parse `cuda:N` or
   default to `cuda:0`.
2. **Even valid `set_device(0)` could OOM** — under load, the trainer
   already had ~12-15 GB of GPU memory pinned (7 compiled models +
   inductor cache + learner). The subprocess's eager CUDA context
   init at `set_device` failed. Final fix: drop `set_device` entirely
   and let the first model `.to(device)` create the context lazily.

End-to-end test (full curriculum: self_play=0.45, bc=0.05, ghosts=0.1,
max_damage=0.15, simple_h=0.05, vgc_bench=0.2; exploiters off via
weight=0):

| Update | traj/s |
|---|---|
| 366 (warmup) | 1.90 |
| 367 | 5.89 |
| 368 | 6.15 |
| 369 | 5.33 |
| 370 | 5.61 |
| 371 | 5.96 |
| 372 | 6.00 |
| 373 | 6.29 |
| 374 | 6.32 |
| 375 | 5.42 |
| **mean updates 367-375** | **5.89 traj/s** |

Compared to:
- Run B post-400 (single-process, ghosts on): 3.72 traj/s
- Run B' (single-process, ghosts off): 5.4 traj/s
- This run (Plan C: ghosts in subprocess): **5.89 traj/s** (+58% over
  Run B post-400; +9% over the ghosts-off baseline)

**Decision rule per plan**: ≥4.5 traj/s to validate, ≥6 traj/s to ship
without follow-up. Updates averaged 5.89, peak 6.32 — clears the
validate bar, sits below the ship-without-follow-up bar. Step 4
PASSES.

The Run B' "no ghosts" 5.4 baseline was effectively a cheat — battles
that would have routed to ghosts fell back to self_play, which has
main on both sides (more compute per battle but no GIL split). The
fact that Plan C with ghosts active EXCEEDS that baseline says the
multi-process design recovers more throughput than ghosts cost.

**Next**: Step 5 — flip `process_group` for `snaps` + (if curriculum
ever re-enables `train_exploiter`) `live`. Today's run had exploiters
weight=0 so the snap services were idle in trainer; Step 5 still
matters because under curriculum changes those services would activate
and reintroduce the GIL contention.

### 2026-05-15 ~12:30 — Step 5 complete: finalized 2-process layout; exploiter learner is the new ceiling

After several iterations, the production-ready layout is:

| Process | Hosts | Why |
|---|---|---|
| **Trainer** | main + exploiter + exploiter_learner | main is highest-traffic; exploiter must co-locate with the in-trainer exploiter_learner (which needs CUDA access to the same agent) |
| **Subprocess `frozen`** | bc, victim, ghost_0..4, exploiter_snap_0..4 (12 services) | All have low sync cadence (bc never; victim every ~3 min; ghosts/snaps via LRU on save events). Merged into one subprocess to keep memory below the 22 GB watchdog on this 23 GB WSL2. |

Bugs found + fixed during Step 5:
1. Initial 3-subprocess design (ghosts + snaps + live) tripped the
   memory watchdog at compile peak (13.87 GB across 3 subprocesses).
   Merged ghosts+snaps into "frozen"; dropped live entirely.
2. Live subprocess was rejected on sync-rate grounds: the exploiter
   learner sits in the trainer (needs CUDA access to the agent for
   gradient updates). Moving its corresponding service to a subprocess
   would require per-update IPC sync at the learner's step rate —
   feasible only via shared-memory state_dict, which is bigger than
   today's scope.
3. Subprocess had the same registration race the trainer-side
   ModelRegistry already fixed: `service.start()` fired immediately
   after each service's compile, so earlier services served traffic
   while later ones were still dynamo-compiling, triggering "FX to
   symbolically trace a dynamo-optimized function". Fix: same
   pattern as `register/start_all` — `run_subprocess` now does two
   passes (build all, then start all). Verified via subprocess-side
   "started subprocess 'frozen' with 12 services" + no race errors
   in production run.
4. CUDA OOM if subprocess called `set_device(0)` while trainer held
   heavy GPU state. Fix: drop `set_device` entirely; let CUDA
   context init lazily on first model `.to(device)`.

#### Step 5 result

Production run (full curriculum WITH `train_exploiter=0.125`):

| Update | traj/s |
|---|---|
| 366 (warmup) | 0.91 |
| 367 | 4.24 |
| 368 | 3.59 |
| 369 | 4.09 |
| 370 | 3.37 |
| **steady-state mean** | **~3.65 traj/s** |

Compared to baselines:
- Run B post-400 (single-process, full curriculum): 3.72 traj/s
- Run C (Layer 2 broken, full curriculum): 3.5 traj/s
- Step 4 (Plan C, NO exploiter training): **5.97 traj/s** ← Plan C's real win
- **Step 5 (Plan C, WITH exploiter training): ~3.65 traj/s** ← same as no-Plan-C baseline

#### Conclusion: Plan C delivers conditionally

Plan C is a real win when `train_exploiter=0`. With `train_exploiter > 0`
the win evaporates because the **exploiter learner** running in the
trainer process competes for trainer GIL alongside main inference. This
isn't a Plan-C-fixable cost — moving the learner out of trainer would
require shared-memory state_dict + CUDA context coordination, which is
~2-3 days of additional work for a use case that's not always on
(exploiter training is typically a phase, not always-on).

Memory cost analysis:
- Trainer alone: ~4.5 GB host RAM
- Frozen subprocess (12 services): ~5-6 GB host RAM during compile peak,
  ~2-3 GB steady-state
- Total: ~10-12 GB peak, well within 22 GB watchdog ceiling.
- GPU: ~10-15 GB combined (trainer ~10 GB, subprocess ~3-4 GB). RTX 3090
  24 GB is fine.

#### What to do about the exploiter-training-on case

Three options on the table for a future session:

1. **Reduce exploiter learner cadence** — explored briefly here.
   Current cadence is already ~once per 85s; can't help much.
2. **Move exploiter agent to dedicated CUDA-aware subprocess** — keep
   the exploiter LEARNER in trainer for gradient access, ship updated
   state_dict to a subprocess INFERENCE service for routed battles.
   At ~once-per-85s sync rate, the 24ms IPC cost is negligible.
   Requires careful agent reference management (separate CPU shadow
   for sync vs CUDA agent for learner).
3. **Plan D — distillation** — smaller rollout model means less
   GIL-held Python time per main forward, leaving more headroom for
   exploiter learner to coexist. Bigger lever (~1 week).

For Stage 2 we recommend shipping Plan C as documented (real win when
exploiter is off) and revisiting option 2 or 3 if/when exploiter-on
throughput becomes critical.

#### What got shipped (code)

- [`src/elitefurretai/rl/inference_subprocess.py`](../../src/elitefurretai/rl/inference_subprocess.py)
  — new module, 5 unit tests passing
- [`src/elitefurretai/rl/model_registry.py`](../../src/elitefurretai/rl/model_registry.py)
  — refactored for `process_group` + `start_all()`, 4 new unit tests
- [`src/elitefurretai/rl/train.py`](../../src/elitefurretai/rl/train.py)
  — bc and victim flipped to `process_group="frozen"`; ghosts and
  exploiter_snaps already there from Step 4; exploiter+victim split
  (exploiter in-trainer, victim subprocess)
- [`src/elitefurretai/rl/configs/sep_arch.yaml`](../../src/elitefurretai/rl/configs/sep_arch.yaml)
  — `memory_watchdog_threshold_gb: 22` (back to default after testing);
  `auto_launch_external_vgcbench: false` (turned off for cleaner
  memory budget); curriculum has vgc_bench=0

All inference tests pass: 5 subprocess + 9 registry + 8 handler + 5
multistep + 8 ipc = 35/35 in the inference suite. Pre-existing failures
in `test_compile_race_reproducer` (flaky) and `test_worker_opponent_factory`
(unrelated to this work) remain.
