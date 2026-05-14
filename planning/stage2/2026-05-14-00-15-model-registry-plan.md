# Model Registry Plan — Centralizing All Model Types

**Date**: 2026-05-14
**Branch**: `centralized-inference` (continues the work from 2026-05-13)
**Predecessor**: [centralized-inference implementation plan](./2026-05-13-18-00-centralized-inference-implementation-plan.md)
**Status**: Pre-implementation. Plan agreed; building registry classes next.

This doc captures the design choice for extending centralized inference
beyond the main agent to ALL model-driven opponents (BC, ghost,
exploiter, victim) before merging the branch.

---

## Context

After the D3-alt + torch.compile iteration on 2026-05-13, the centralized
inference architecture beats the per-player baseline by ~9% on
throughput (4.05 vs 3.7 traj/s) and 37% on learner steps. But it's
limited to the **main agent only** — the curriculum is forced to
`self_play: 1.0` because BC / ghost / exploiter / victim still use the
legacy per-worker model copies + `opponent.model = X` hot-swap pattern.

To merge cleanly, we want the full curriculum back. That requires
centralizing the other model types too — but doing so naively means a
LOT of repeated wiring (one `inference_client` parameter per model,
through 4-5 files each). We want a pattern that automates this.

---

## Plan options considered

### Plan A — Manual per-model wiring

Add named services + clients explicitly: `bc_inference_client`,
`exploiter_inference_client`, etc. Thread each through factory →
worker.py → train.py.

- Adding model type N+1 = touch ~6 files. Lots of boilerplate.
- Ghosts (multiple checkpoints) need per-ghost services or special-case.

### Plan B — Single multiplexed service with `model_id` field

One InferenceService holds `Dict[model_id, agent]`. Each request carries
a `model_id`. Service dispatches.

- **Critical perf cost**: a batch can't be batched ACROSS model_ids.
  `[main, main, bc, main]` runs two forwards (3 + 1), not one of 4.
  Loses ~30–50% of the batching win on heterogeneous workloads.

### Plan C — Model registry with auto-wiring

A `ModelRegistry` class in the trainer is the source of truth for "which
models exist." Trainer code does `registry.register("name", agent)`. The
registry stands up a per-model `InferenceService` and per-worker
`InferenceClient` for each. Workers receive a single
`WorkerInferenceClients` bundle that exposes `clients.get("name")`.

Adding model type N+1 = `registry.register(...)` + 1 hot-swap branch =
**2 lines**.

### Plan D — Hybrid (centralize live models, leave ghosts per-worker)

Centralize main + BC + exploiter + victim. Keep ghosts per-worker
(loaded from disk, slow path).

- Avoids "many ghost services" complexity.
- Bakes in dual-mode logic forever.

---

## Comparison

| Dimension | A (manual) | B (multiplexed) | **C (registry)** | D (hybrid) |
|---|---|---|---|---|
| **Speed** | Same as current centralized per type | **Worse** (no cross-model batching) | Same as A | Centralized types same; ghosts on slow legacy path |
| **Precision** | Same | Same | Same | Same |
| **Conceptual simplicity** | Verbose, spread across files | Clean abstraction *until* the no-cross-batch gotcha bites | One pattern uniformly | Two patterns coexist — readers must know which to use |
| **Code (first build)** | Medium | Smallest | Largest (~250 LOC for registry + bundle) | Medium |
| **Code (adding model N+1)** | High — ~6 files per type | Low (`registry.register(...)`, modulo perf cost) | **Low — 1 line + 1 hot-swap branch** | Mixed: easy for centralized types, awkward for ghost-likes |
| **Ghost handling** | Awkward (many services or special case) | Awkward (same) | **Clean (slots in registry)** | Forced legacy path |
| **Diagnostics per-model** | Easy (one service each) | Hard (need internal per-model counters) | Easy (per-service diagnostics naturally per name) | Mixed |

---

## Decision: Plan C

**Why C over the others**:

1. **Speed-equivalent to A** but ~5x less code per added model.
2. **Beats B on speed.** The "can't batch across model_ids" cost is
   real and unrecoverable without redesign.
3. **Beats D on conceptual simplicity.** D bakes in two coexisting
   patterns; C uses one pattern for everything.
4. **Matches the stated requirement**: automate adding new models,
   keep the model concept clean, wrap with helpers.
5. **Future-proof.** Distilled rollout models, evaluator models,
   second-policy-head experiments — all are `registry.register("name",
   agent)`.

**Cost**: ~1 day focused work. Two new classes (~150 LOC each), edits
to ~4 existing files.

**Risk**: longer warmup (~3–5 min if all 13 model slots compile
sequentially). Mitigation: torch.compile's per-shape cache should
share most work across same-architecture models. Lazy-compile (compile
on first request, not at startup) is the fallback.

---

## Architecture sketch

```python
# src/elitefurretai/rl/model_registry.py (new)
class ModelRegistry:
    """Trainer-side registry of named models.

    For each registered model, owns an InferenceService + per-worker
    response queues. Stand up at trainer startup; the worker spawn
    args carry the queue handles, and workers wrap them in a
    WorkerInferenceClients bundle.
    """
    def __init__(self, num_workers, batch_size, batch_timeout,
                 device, compile_mode=None): ...
    def register(self, name: str, agent: RNaDAgent) -> None: ...
    def sync_weights(self, name: str, state_dict) -> None: ...
    def queues_for_workers(self) -> Dict[str, Tuple[MPQueue, List[MPQueue]]]: ...
    def get_diagnostics(self) -> Dict[str, Dict[str, float]]: ...
    def stop_all(self) -> None: ...

# src/elitefurretai/rl/worker_inference_clients.py (new)
class WorkerInferenceClients:
    """Worker-side bundle of per-model inference clients.

    Constructed once per worker from spawn-time queues. Players hot-swap
    by calling clients.get(model_name) and assigning the result to
    player.inference_client.
    """
    def __init__(self, worker_id: int,
                 queues_by_model: Dict[str, Tuple[MPQueue, MPQueue]],
                 loop): ...
    def get(self, model_name: str) -> InferenceClient: ...
    def stop_all(self): ...
```

Hot-swap in `assign_opponent_role`:
```python
def assign_opponent_role(self, player, opponent):
    opp_type = self.sample_opponent_type()
    if opp_type == BC_PLAYER and self._has_client("bc"):
        opponent.inference_client = self.clients.get("bc")
    elif opp_type == GHOSTS:
        slot = random.choice(self._active_ghost_slots)
        opponent.inference_client = self.clients.get(f"ghost_{slot}")
    elif opp_type == EXPLOITERS and self._has_client("exploiter_snap"):
        ...
    else:
        opponent.inference_client = self.clients.get("main")
    player.opponent_type = opp_type
```

---

## Implementation order

1. **Standalone `ModelRegistry` + `WorkerInferenceClients` classes**
   with unit tests, no integration yet. (~half day)
2. **Replace existing `inference_service` / `main_inference_client`
   plumbing in train.py and worker.py with `registry.register("main",
   ...)`.** Should produce identical throughput as the current
   centralized run. Validation = re-run sep_arch and confirm 4.05 traj/s.
3. **Add BC**: `registry.register("bc", bc_agent)` + factory hot-swap
   path for `BC_PLAYER`. Re-enable `bc_player: 0.1` in curriculum and
   validate.
4. **Add victim + exploiter**: same pattern for the live-trained
   exploiter and frozen victim.
5. **Add ghost slots**: pre-register `ghost_0` ... `ghost_<max-1>`. On
   ghost save events, `registry.sync_weights(f"ghost_{slot}", new_sd)`.
   Re-enable `ghosts: 0.1`.
6. **Re-enable full sep_arch curriculum**, measure end-to-end throughput.
7. **Commit + merge to main.**

Each step is testable in isolation; if any step regresses throughput
or breaks something, we pause and diagnose before proceeding.

---

## Open questions for later

- **Compile-cache sharing**: does torch.compile actually share compiled
  artifacts across same-arch model copies? Need to check at step 5
  (when 5+ ghost services compile). If not, lazy-compile is the
  fallback.
- **Ghost slot eviction policy**: when a 6th ghost saves and we have
  only 5 slots, which slot's weights get replaced? Current legacy code
  uses LRU (oldest checkpoint loaded first); registry should preserve
  that behavior via `sync_weights("ghost_<oldest_slot>", new_sd)`.
- **Eval-time inference**: OpponentPool's main-process eval battles
  also do inference. Should they share the registry? For now: no — eval
  is rare enough that legacy per-call inference is fine. Revisit if
  eval becomes a bottleneck.

---

## Implementation results (2026-05-14)

### Step-by-step measurements

| Step | What changed | Curriculum | Throughput | Learner steps/s | Notes |
|---|---|---|---|---|---|
| 1 | Built `ModelRegistry` + `WorkerInferenceClients` + 10 unit tests | n/a | n/a | n/a | 297 tests pass |
| 2 | Replaced `inference_service` plumbing with `registry.register("main", ...)` | `self_play=1.0` | **4.22 traj/s** | ~83 | Parity with prior 4.05 traj/s (no regression from refactor) |
| 3a | Added `bc` registration with compile=True | `self_play=0.9, bc=0.1` | 3.88 traj/s | ~76 | **70 errors / 15500 batches (0.45%)**: torch.compile + multi-threaded service calls trigger dynamo "FX symbolic trace of dynamo-optimized function" race |
| 3b | Same, but `compile=False` for BC | `self_play=0.9, bc=0.1` | **3.78 traj/s** | ~76 | **0 errors**. Workaround documented in `register(compile=...)` |
| 4 | Conditional registration of `exploiter` + `victim` (gated by `train_exploiter` curriculum weight) | unchanged from step 3 | n/a (not exercised) | n/a | Wiring + state-sync ready; activates if curriculum re-enables |
| 5 | DEFERRED: ghost slot rotation | n/a | n/a | n/a | Multiple-slot machinery + dynamic loading is post-merge work; legacy on-demand per-worker ghost loading retained for now |
| 6 | Full original curriculum re-enabled | `self_play=0.3, bc=0.1, ghosts=0.1, max_damage=0.1, simple_h=0.1, vgc_bench=0.3` | **4.98 traj/s** | ~87 | **+34% vs main-branch baseline 3.7 traj/s, +45% on learner steps**. Bumped `memory_watchdog_threshold_gb` 20→22 for VGCBench runners. |

### Notable findings

1. **Step 2 regression risk avoided**: pure refactor, 4.22 traj/s = no regression vs 4.05. The registry abstraction is essentially free at runtime.
2. **Critical bug found in step 3**: `torch.compile` with `mode='default', dynamic=True` is NOT thread-safe across multiple compiled models in concurrent service threads. Symptom: `RuntimeError: Detected that you are using FX to symbolically trace a dynamo-optimized function`. Workaround: only compile the highest-traffic model; mark secondaries `compile=False`. Per-call eager inference is fine for low-traffic models (BC handles ~10% of curriculum).
3. **Full curriculum is FASTER than self-play-only.** Counterintuitive but consistent: vgc_bench (avg battle length 14) and simple_heuristic (11) finish faster than self-play (20). The opponent mix shortens average trajectory length, so traj/s rises despite more inference traffic per battle.
4. **Memory watchdog needed +2 GB headroom** for the full curriculum (VGCBench runners ~5.5 GB). On 24 GB WSL2, 22 GB threshold leaves comfortable buffer.

### What got merged

- `src/elitefurretai/rl/model_registry.py` (new, ~210 LOC)
- `src/elitefurretai/rl/worker_inference_clients.py` (new, ~70 LOC)
- `unit_tests/rl/test_model_registry.py` (new, 10 tests)
- `train.py`: replaced inference_service plumbing with registry; conditional bc + exploiter + victim registration; state-sync via `registry.sync_weights`
- `worker.py`: accepts `queues_by_model` spawn arg; constructs `WorkerInferenceClients`
- `vgc_environment.py` + `_ShowdownBackend`: thread `worker_inference_clients` parameter through
- `opponents.py`: factory accepts bundle; `_resolve_centralized_client` + `_swap_to` helpers; refactored `configure_opponent_for_batch` to use them; legacy ghost path retained
- `sep_arch.yaml`: full curriculum re-enabled, watchdog 20→22 GB

### Future work

- **Ghost centralization** (step 5 deferred): pre-register `max_ghosts` slot services; rotate via `sync_weights`. Requires worker-side slot tracking (broadcast active slots).
- **Resolve torch.compile multi-thread race**: would let us compile bc/exploiter/victim too. Possibly fixable with `torch.compiler.cudagraph_mark_step_begin()` or per-thread compile contexts. Worth investigating if secondary models become high-traffic.
- **Eval-time inference**: OpponentPool's main-process eval battles still use legacy inference. Re-route through registry if eval becomes a bottleneck.
