# Ghost Centralization + Compile-Race Investigation + Legacy Cleanup — Design

**Date**: 2026-05-14
**Predecessor**: [model registry plan](./2026-05-14-00-15-model-registry-plan.md)
**Status**: Pre-implementation. Design agreed; ready for plan writing.

This doc covers closing out the registry plan's "Future work" section. Three
items were originally listed (ghost centralization, torch.compile race,
eval-time inference); during brainstorming we dropped eval (no production
caller exists) and added a fourth: deleting the legacy per-worker inference
path now that centralized has been the only production mode for two weeks.

---

## Context

The 2026-05-14 ModelRegistry merge (`3ead76c`) centralized inference for
main + bc + exploiter + victim through the registry pattern, lifting
throughput from 3.7 to 4.98 traj/s on `sep_arch.yaml`'s full curriculum.
Three items were left as "Future work":

1. **Ghost centralization** — ghosts (~10% of curriculum) still run the
   legacy per-worker `loaded_ghosts` disk-cache path.
2. **torch.compile multi-thread race** — `torch.compile(dynamic=True)` on
   multiple concurrent service threads triggers a dynamo FX trace race; we
   compile only `main` and run bc/exploiter/victim eager as a workaround.
3. **Eval-time inference** — `OpponentPool` was noted as having
   main-process eval battles that still use legacy inference.

## Before state

- Branch: `main` at `1944eda`.
- Current throughput baseline (from registry plan): 4.98 traj/s, ~87
  learner steps/s on `sep_arch.yaml` full curriculum. (To be re-verified
  as Measurement 1.)
- Ghost path: each worker holds a `loaded_ghosts: Dict[str, RNaDAgent]`
  cache and loads checkpoints on demand from `data/models/ghosts/`.
- Dual-mode `BatchInferencePlayer`: accepts either `model` (legacy) OR
  `inference_client` (centralized). Centralized is selected in production
  via `enable_centralized_inference: true` in sep_arch.yaml.
- `OpponentPool.sample_opponent` and the family of `_create_*_opponent`
  methods exist on the main-process pool but have no production callers
  (verified via grep across `src/`, `unit_tests/`).

## Problem

1. Ghosts on the legacy path don't benefit from centralized batching
   (capped at 4 per-player) and each worker carries ghost model copies
   in memory. Three inference code paths coexist (centralized worker
   path, legacy ghost path, hypothetical eval path), which is a
   maintenance and reasoning burden.
2. Without the compile-race fix, ~30–40% of inference traffic
   (bc + future-compiled secondaries) runs eager.
3. The dual-mode `BatchInferencePlayer` and the `enable_centralized_inference`
   config flag are now pure tech debt: centralized has been the only
   production mode for two weeks. Deleting them removes a class of
   future bugs ("what if someone flips the flag and the legacy path is
   stale?").

## Solution

### Scope

| Item | In | Out |
|---|---|---|
| Ghost centralization | ✅ | |
| torch.compile race (4hr time-box) | ✅ | |
| Legacy inference path deletion | ✅ | |
| Eval-time inference | | ✅ (no production caller exists) |

### 1. Ghost centralization

**Approach**: pre-register `max_ghosts` ghost slot services at registry
startup (each `compile=False` initially). Maintain `active_ghost_slots:
Set[int]` reflecting which slots hold real ghost weights vs placeholder
weights. Broadcast active slots to workers; workers pick from active
slots when sampling GHOSTS.

**Trainer startup** (in `train.py` registry setup):

```python
# After registering main/bc/exploiter/victim
for slot in range(max_ghosts):
    registry.register(f"ghost_{slot}", clone_of_main_agent, compile=False)
active_ghost_slots: Set[int] = set()

# Scan disk; load up to max_ghosts existing checkpoints (oldest first by step)
for slot, (step, path) in enumerate(existing_ghosts_sorted):
    state_dict = torch.load(path, map_location=device)
    registry.sync_weights(f"ghost_{slot}", state_dict)
    active_ghost_slots.add(slot)
```

**Trainer during training** (when a new ghost saves, currently the
`add_ghost` path in `OpponentPool` + the broadcast site at train.py:1613):

```python
if len(active_ghost_slots) < max_ghosts:
    slot = first unused slot
else:
    slot = LRU slot (oldest by insertion order — match legacy LRU eviction)
registry.sync_weights(f"ghost_{slot}", new_state_dict)
active_ghost_slots.add(slot)
# Track LRU via slot_insertion_order: List[int]
```

**Broadcast**: extend the existing weight-broadcast payload with
`active_ghost_slots: List[int]`. Workers update their local copy on
receive. Initial spawn args carry the startup-time set so workers
have valid state before the first broadcast.

**Worker hot-swap** (in `WorkerOpponentFactory._swap_to` ghost branch):

```python
slot = random.choice(tuple(self._active_ghost_slots))  # IndexError if empty
opponent.inference_client = self.clients.get(f"ghost_{slot}")
```

No fallback: `OpponentPool._opponent_available(GHOSTS)` already gates
sampling on ghosts existing; the curriculum guarantees `active_ghost_slots`
is non-empty when this code runs. If it isn't, that's a bug worth crashing
on, not papering over.

**Memory cost**: 5 slots × ~300 MB = ~1.5 GB additional GPU memory.
On the 24 GB 3090 with ~6.5 GB currently held by trainer model copies,
this leaves comfortable headroom.

### 2. torch.compile race investigation (4-hour time-box)

Investigation proceeds in order, stopping at the first attempt that
survives a 5-minute multi-model load test (main + bc both `compile=True`,
~500 calls each, zero `RuntimeError`s):

| Step | Try | Notes |
|---|---|---|
| 1 | Wrap each compiled-model call in a per-model `threading.Lock` | Tests whether the race is at trace-entry. Lock contention on already-traced fast-path is microseconds. |
| 2 | `torch.compiler.cudagraph_mark_step_begin()` between compiled calls | Tests CUDA graph capture state hypothesis. |
| 3 | One dedicated daemon thread per compiled service; the service's compiled model is only ever called from its own thread | Architectural — eliminates concurrency on a single compiled model entirely. |
| 4 | None work in 4 hours | Document findings, file an upstream issue if reproducer is minimal, ship ghost-only. |

**On success**: enable `compile=True` for bc, exploiter, victim, and all
`ghost_*` slots in `train.py` registry setup. Re-run sep_arch full
curriculum for 200 batches; verify zero compile-race errors AND
throughput rises.

### 3. Legacy inference path cleanup

Once Sections 1 + 2 are done (regardless of whether 2 succeeded), strip
the legacy inference path.

**`config.py`** — remove `enable_centralized_inference: bool`. Centralized
is the only mode.

**`players.py` — `BatchInferencePlayer`**:
- Constructor: drop `model` parameter and dual-mode validation.
- Remove: `self.queue`, `self.model`, `_inference_loop`, `_inference_future`,
  `_run_batch`, `_gpu_inference_sync`, `_add_to_batch`, `start_inference_loop`.
- `_choose_move_async`: collapse to the `inference_client` path only.

**`worker.py`**:
- Drop the legacy `if not centralized: build model + bc_model + ...` branch.
- Drop `update_weights` (currently no-op for centralized workers).
- Remove spawn args conditional on legacy mode.

**`opponents.py` — `WorkerOpponentFactory`**:
- Remove `loaded_ghosts: Dict[str, RNaDAgent]`, `loaded_exploiters`,
  `_get_cached_model`, `_get_ghost_agent`, `_get_exploiter_agent`,
  disk-scanning `_load_ghosts`.
- `set_ghost_paths` → replaced by `set_active_ghost_slots(slots: List[int])`
  broadcast handler.
- `set_exploiter_paths` is dead (exploiter is registry-routed); remove.

**`opponents.py` — main-process `OpponentPool`**:
- Remove `sample_opponent` + the family of `_create_self_play_opponent`,
  `_create_bc_opponent`, `_create_exploiter_opponent`,
  `_create_ghost_opponent` methods. **Verify-before-delete**: grep
  `test_opponent_pool.py` and any other test files for callers; if
  tests exercise them, the tests are dead too (no production code uses
  the methods being tested).

**Tests**:
- `test_players.py`: drop test cases that exercise legacy-mode
  `BatchInferencePlayer`.
- `test_worker_opponent_factory.py`: drop test cases that pass model
  objects to the factory.

**Docs**:
- `sep_arch.yaml`: remove `enable_centralized_inference` line; replace
  the multi-line ghost-loading comment with a one-liner that ghosts
  are registry slots.
- `src/elitefurretai/rl/RL.md`: remove "legacy ghost path retained" and
  "still use legacy inference" mentions; update section 8b architecture
  description.
- This doc + `2026-05-14-00-15-model-registry-plan.md`: append "Updates"
  marking ghost done, compile-race outcome, eval-dropped, cleanup
  shipped.

**Verify dead before deletion**: grep every removed symbol across `src/`,
`unit_tests/`, `planning/` for residual references. Anything that turns
up either gets the caller fixed or scope-creeps the cleanup explicitly.

### 4. Measurement protocol

Each measurement run:

- **Config**: `sep_arch.yaml` unchanged (full curriculum, num_workers=4,
  num_servers=4, batch_size=32, batch_timeout=0.005, `compile_inference_model=default`).
- **Warmup**: skip first 30 updates (compile + Showdown server warmup +
  curriculum steady state).
- **Window**: 20 consecutive post-warmup updates, contiguous. If anomaly,
  restart the window.
- **Metrics**: `traj/s` (mean ± stddev), `learner steps/s` (mean),
  batch-fill avg + max, errors per 1k batches.
- **Wall time**: ~12–15 min per run.

### 5. Sequencing & gates

All work committed directly to `main`, one commit per logical step.

```
Step 0  Re-baseline on main @ 1944eda             → MEASUREMENT 1
        Reference for everything that follows.

Step 1  Implement ghost centralization
        Tests: new unit tests for slot lifecycle. 297+ existing pass.
        Run sep_arch.                              → MEASUREMENT 2
        GATE: throughput ≥ baseline. If regressed, diagnose before
              continuing.
        COMMIT.

Step 2  torch.compile race investigation (4hr time-box per §2)
        If fix lands:
          Enable compile=True on bc / exploiter / victim / ghost_*.
          Run sep_arch.                            → MEASUREMENT 3
          GATE: throughput ≥ step 1 result AND zero compile-race errors.
          COMMIT.
        If no fix in 4hrs:
          Document findings inline in this doc's Updates.
          Skip Measurement 3.
          No commit (no code change).

Step 3  Cleanup pass per §3.
        Grep-verify nothing references removed symbols.
        Run sep_arch.                              → MEASUREMENT 4
        GATE: throughput ≥ previous measurement (cleanup shouldn't
              change perf).
        COMMIT.

Step 4  Append final comparison table to this doc's Updates section.
        Mark the registry plan's "Future work" items closed.
        COMMIT (doc-only).
```

Total expected compute: ~1 hour across 4 measurement runs. Total
implementation time: ~1 day (ghost) + up to 4 hours (compile race) +
~2–3 hours (cleanup) = roughly 2 working days.

## Reasoning

**Why Approach A for ghost slots (pre-register all `max_ghosts`)** over
lazy registration: mid-run service registration would require new code
paths in `ModelRegistry`, new error modes ("worker requests a slot the
trainer doesn't have"), and complicates worker-side state. Pre-registering
costs 1.5 GB on a system with 17 GB headroom — premature optimization to
save it. One uniform pattern beats two.

**Why no worker-side fallback** when `active_ghost_slots` is empty: the
curriculum already gates GHOSTS sampling on availability; if a worker
gets here with empty slots, that's a desync bug in broadcast plumbing
worth crashing on per the project's "no try/catch hiding errors" rule.

**Why time-box the compile race instead of committing to solve it**: it's
research-shaped with uncertain upper-bound time cost. The race may not
be fixable from user space at all. The other two items deliver value
independently; we shouldn't gate them on a maybe-impossible fix. 4 hours
is enough to try the three credible attempts; if none work, we ship the
deterministic wins and file upstream.

**Why delete the legacy path now** rather than carrying it as a fallback
flag: the project's "no backwards-compat hacks" rule applies, the
centralized path has been the only production mode for two weeks, and
the legacy code blocks future refactoring without offering safety
(stale-untested code isn't a fallback, it's a liability).

**Why measure after cleanup**: trust-but-verify. Cleanup is "remove
dead code" but the rule is "evidence before assertions" — confirm
throughput holds.

## Planned next steps

1. User reviews and approves this design doc.
2. Invoke `writing-plans` to produce a step-by-step implementation
   plan with explicit commit boundaries, test gates, and measurement
   command lines.
3. Execute the plan per the plan's review checkpoints (typically
   via `executing-plans` skill).

## Open risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Compile race not fixable in 4 hrs | Medium | Low (just ships ghost-only) | Time-box is the mitigation. |
| Ghost broadcast desync (worker has stale `active_ghost_slots`) | Low | Medium (workers route to wrong slot) | Initial spawn args carry startup state; broadcast extends existing weight-broadcast path which already has ordering guarantees. Tests for the spawn-args path. |
| Pre-registered ghost slot placeholders eat too much GPU memory under unusual configs (e.g., `max_ghosts=20`) | Low | Medium | `sep_arch.yaml` uses 5 slots. Add a watchdog warning if total trainer GPU memory after registry init exceeds a threshold; document in `RL.md` that `max_ghosts` × per-model-size budget is a real cost. |
| Cleanup removes something a test silently depends on | Low | Low | Run full test suite after every cleanup commit, not just at end. |
| Compile race fix works in test but flakes under production load | Low | Medium | Measurement 3's gate ("zero compile-race errors over 20 update window") is the catch. If errors > 0, treat compile fix as failed and revert that commit. |

## Updates

### 2026-05-14 13:00 — Design written

Brainstorming completed with Cayman. Scope finalized: ghost
centralization + 4-hour compile-race investigation + full legacy
inference path cleanup. Eval-time inference dropped (no production
caller). All work commits to `main` directly (no feature branch).
Ready for plan writing.
