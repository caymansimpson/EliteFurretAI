# RL Codebase Refactoring: Tasks 3B and 3C

## Context

Continuation of the RL codebase refactoring (see `2026-04-30-22-27-rl-codebase-refactoring-3A-4-7.md`). Tasks 3B (worker.py extraction) and 3C (VGCEnvironment wrapper) complete the structural cleanup.

## Task 3B — Create worker.py

### What was extracted from train.py

Moved these five units from `src/elitefurretai/rl/train.py` to the new `src/elitefurretai/rl/worker.py`:
1. `_load_agent_team_sources()` — helper for loading agent team files from path or directory
2. `_build_team_supplier()` — factory that returns a callable team sampler
3. `_RustPolicyOpponentPool` — manages opponent sampling for the Rust backend
4. `_run_rust_worker_loop()` — battle loop for the Rust in-process backend
5. `mp_worker_process()` — full multiprocessing worker entry point (both backends)

`train.py` now only contains the learner-side logic: `initialize_learner`, `resolve_worker_model_source`, `initialize_training_state`, `get_dead_workers`, `collate_trajectories`, `train_exploiter_subprocess`, `main`.

### Module docstring

`worker.py` has a module-level docstring explaining the IMPALA actor-learner split, the two backend paths (Showdown websocket vs Rust in-process), and the weight broadcast/trajectory forwarding flow.

### Phase comments in mp_worker_process

Four structured block comments in `mp_worker_process` marking:
- Phase 1: Model loading (CPU-first to avoid CUDA fork issues)
- Phase 2: Opponent factory setup (curricula, opponent categories)
- Phase 3: Battle loop (weight polling, task execution, timeout handling)
- Phase 4: Trajectory forwarding (local queue → learner mp_traj_queue)

### Option C broadcast extension (from plan section 3D)

The weight_queue broadcast payload in `train.py` was extended from:
```python
{"weights": ..., "curriculum": ..., "temperature": ..., "top_p": ...}
```
to:
```python
{"weights": ..., "curriculum": ..., "temperature": ..., "top_p": ...,
 "exploiter_paths": [...],  # explicit file paths from OpponentPool.exploiter_models
 "ghost_paths": [...]}      # explicit file paths from OpponentPool.past_models
```

Workers now receive exact file lists instead of scanning directories every 10 batches. This eliminates O(N_batches × N_workers) directory scans.

Changes to `_RustPolicyOpponentPool` (worker.py):
- Added `set_exploiter_paths(paths)` — applies explicit list from broadcast
- Added `set_ghost_paths(paths)` — applies explicit list from broadcast
- Both are called in the Rust weight-polling loop when keys are present

Changes to `WorkerOpponentFactory` (opponents.py):
- Added `set_exploiter_paths(paths)` — same as above for Showdown path
- Added `set_ghost_paths(paths)` — same
- Removed `_refresh_exploiters_if_needed()` method (called every 10 batches)
- Removed `_refresh_past_models_if_needed()` method
- Removed both calls from `prepare_batch_tasks()`
- Initial load on construction unchanged

### Import cleanup in train.py

16 imports removed from `train.py` that were only used by the extracted code: `asyncio`, `deque`, `AccountConfiguration`, `ServerConfiguration`, `MaxBasePowerPlayer`, `RandomPlayer`, `derive_external_vgcbench_username`, `SyncBaselineController`, `SyncPolicyPlayer`, `SyncRustBattleDriver`, `TeamRepo`, `is_checkpoint_compatible_with_model_config`, `load_agent_from_checkpoint`, `SimpleHeuristicBaselineCls`, `WorkerOpponentFactory`, `MaxDamagePlayer`.

## Task 3C — VGCEnvironment wrapper

Created `src/elitefurretai/engine/vgc_environment.py` with `VGCEnvironment` — a unified facade over both battle backends.

### Public interface

```python
class VGCEnvironment:
    @classmethod
    def from_config(cls, config, worker_id, agent, model, model_config,
                    embedder, team_repo, bc_agent, server_port, run_id) -> VGCEnvironment
    async def setup(self) -> None
    async def run_battle_batch(self, n_battles) -> Tuple[List[Dict], List[str]]
    def update_weights(self, state_dict) -> None
    def update_curriculum(self, curriculum, exploiter_paths, ghost_paths) -> None
    def update_sampling(self, temperature, top_p) -> None
    def get_diagnostics(self) -> Dict[str, float]
    async def teardown(self) -> None
```

### Backend implementations

- `_ShowdownBackend` — wraps `WorkerOpponentFactory`. `run_battle_batch()` calls `factory.prepare_batch_tasks()` + `asyncio.wait()`, drains local trajectory queue.
- `_RustBackend` — wraps `_RustPolicyOpponentPool` + `SyncRustBattleDriver`. `run_battle_batch()` calls `driver.run()` + `driver.consume_completed_trajectories()`.

Both backends propagate `update_weights`, `update_curriculum` (including Option C paths), and `update_sampling` to their underlying components.

### Current state

`VGCEnvironment` is fully implemented but `mp_worker_process` in `worker.py` still owns the full battle loop directly. Migrating `mp_worker_process` to use `VGCEnvironment` is a follow-up step that requires integration testing.

`VGCEnvironment` is exported from `engine/__init__.py`.

## Verification

- `ruff check --select F401,F811,F821,E9 src/elitefurretai/rl/ src/elitefurretai/engine/`: all pass
- `pyright src/elitefurretai/rl/`: 0 errors
- `pytest unit_tests/rl/test_config.py unit_tests/rl/test_learner.py unit_tests/rl/test_multiprocess_actor.py`: all pass
- Import smoke test: VGCEnvironment, worker, train, masking, learners all import cleanly
- Config round-trip: `RNaDConfig.load(yaml)` → `save()` → `load()` matches

## Task 3C Update — VGCEnvironment wired into mp_worker_process (2026-05-01)

`mp_worker_process` in `worker.py` now uses `VGCEnvironment` directly. The `if config.hardware.battle_backend == RUST_ENGINE_BACKEND:` branch has been removed.

### What moved from worker.py → vgc_environment.py

- `_load_agent_team_sources()` — moved to `engine/vgc_environment.py`
- `_build_team_supplier()` — moved to `engine/vgc_environment.py`
- `_RustPolicyOpponentPool` — moved to `engine/vgc_environment.py`
- `_run_rust_worker_loop()` — **deleted** (replaced by `_RustBackend.run_battle_batch()`)

This eliminates the circular import that previously required `vgc_environment.py` to import from `worker.py`.

### New VGCEnvironment API additions

- `BatchResult` dataclass: `trajectories`, `sampled_types`, `completed_counts`, `had_timeout`, `timed_out_types`, `battles_completed`
- `VGCEnvironment.run_battle_batch()` now returns `BatchResult` (was `Tuple[List[Dict], List[str]]`)
- `VGCEnvironment.rebuild()` — rebuild runtime agents after timeout/stale battle
- `VGCEnvironment.reset_battles()` — reset battle state for next batch
- `VGCEnvironment.get_unfinished_summary()` — `{total_unfinished: n, ...}`
- `VGCEnvironment.get_curriculum()` — current curriculum weights (copy)
- `VGCEnvironment.update_sampling(temperature, top_p)` — now accepts `Optional[float]`
- `VGCEnvironment.from_config()` — new `initial_curriculum` parameter for dedicated-worker overrides
- `_ShowdownBackend._batch_timeout_s` — now computed from config (matches original `mp_worker_process` formula)

### worker.py after wiring

`worker.py` now contains only `mp_worker_process`. Its imports dropped from 17 symbols to 9. The Rust-vs-Showdown branching is entirely gone; Phases 2–4 are backend-agnostic.

## Verification

- `ruff check --select F401,F811,F821,E9 src/elitefurretai/rl/worker.py src/elitefurretai/engine/vgc_environment.py`: all pass
- `pyright src/elitefurretai/rl/worker.py src/elitefurretai/engine/vgc_environment.py`: 0 errors
- `pytest unit_tests/rl/test_config.py unit_tests/rl/test_learner.py unit_tests/rl/test_multiprocess_actor.py`: all pass (63 tests)
- Import smoke test: `VGCEnvironment`, `BatchResult`, `mp_worker_process` all import cleanly

## Planned Next Steps

- Update `RL.md` with the new module structure (worker.py, masking.py, vgc_environment.py)
- Run full training smoke test to catch any runtime regressions
