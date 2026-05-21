# Evaluate Process-Pool Refactor — Plan

## Context

Stage II eval pipeline (`src/elitefurretai/rl/analyze/evaluate.py`) is used
for the full Plan B sweep against the balmy-cloud-70 step-10700 checkpoint:
1,764 cells × 100 battles × N opp_types in a single python invocation per
opp_type. Currently producing data in
[data/eval/balmy70_step10700_full_2026-05-19](../../data/eval/balmy70_step10700_full_2026-05-19/).

A parallel side-launch (vgc_bench on ports 8204-7) was started today to
double aggregate wall throughput. This plan covers the structural fix that
makes a *single* evaluate.py invocation much faster, so we don't need to
shard via parallel wrappers for future runs.

## Before State

[evaluate.py:462-493](../../src/elitefurretai/rl/analyze/evaluate.py#L462-L493)
fans cells out across workers using `ThreadPoolExecutor(max_workers=N)`.
Each worker runs an isolated `asyncio.run(_run())` event loop. The
historical assumption (in the docstring on the same lines): "threading is
fine because the heavy work is async network I/O against Showdown."

Empirical reality on 2026-05-20 (PID 27645, `--workers 4 --num-servers 4`,
balmy-cloud-70 step 10700 vs `max_damage`):

| Metric | Value |
|---|---|
| OS processes | 1 (PID 27645) |
| Threads | ~28 (`pstree` shows curly-brace LWPs) |
| CPU util | 100% of **one** core, 0% on the other 7 |
| Throughput | ~3.9 battles/sec across all 4 workers |
| Resident RSS | flat 2.5 GB (with `reset_battles` leak fix in place) |
| GPU usage | ~250 MB, near-idle |

## Problem

The eval pipeline added meaningful CPU work on top of training paths:

- **Embedder** (`etl/embedder.py`) — feature engineering called from
  `RecordingModelPlayer._on_action_selected` per turn.
- **compute_ensemble_advantage** — heuristic + outcome blend, per turn.
- **MaxDamagePlayer** — `calculate_damage` over every legal move × every
  opponent target × team-preview pre-scan, per turn (CPU-bound; no CUDA).
- **Parquet write** — per-shard `pq.write_table` (CPU-bound).

None of those release the GIL. With four threads in one process, the GIL
serializes the CPU work, so the four `asyncio` loops are effectively
co-routinely scheduled on a single core — exactly what we observe.

Net cost: each opp_type takes ~12 h (176,400 battles ÷ 3.9 b/s). With 4
opp_types planned per checkpoint, that is two full days of wall time for a
single sweep.

## Solution

Convert evaluate.py's worker dispatcher from `ThreadPoolExecutor` to
`ProcessPoolExecutor` with the `spawn` start method, so each worker is a
real OS process with its own GIL and its own CUDA context.

This requires three discrete changes; the bulk of the work is making
`PlayerSpec` picklable.

### Change 1 — `PlayerSpec` becomes pure data

Today `PlayerSpec.factory` is a closure built inside `_model_spec` /
`_baseline_spec`
([player_factory.py:193-247](../../src/elitefurretai/rl/analyze/player_factory.py#L193-L247))
that captures `path`, `device`, `battle_format`, `canonical`, etc. as
free vars. Standard `pickle` cannot serialize closures, so submitting a
spec to a process pool fails.

Refactor:

```python
@dataclass(frozen=True)
class PlayerSpec:
    raw: str
    kind: PlayerKind          # "model" | "baseline" | "external"
    name: str
    user_tag: str
    params: Mapping[str, Any] # serializable per-kind config
```

`params` is a frozen mapping the worker uses to reconstruct the player:

- `kind="model"` → `{"path", "device", "battle_format"}`
- `kind="baseline"` → `{"canonical", "battle_format"}`
- `kind="external"` → `{"checkpoint_path", "team_file", "python_executable", "battle_format"}`

`_model_spec` / `_baseline_spec` / `_external_spec` populate `params`
instead of building closures.

Add a module-level **builder**:

```python
def build_player(
    spec: PlayerSpec,
    *,
    team: str,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    accept_open_team_sheet: bool,
) -> Player:
    if spec.kind == "model":
        return SimpleModelPlayer(model_path=spec.params["path"], ...)
    if spec.kind == "baseline":
        return _construct_baseline(spec.params["canonical"], ...)
    raise ValueError(spec.kind)  # "external" handled separately
```

`launch_external` similarly becomes a module-level function
`launch_external(spec, server_url) -> RunningExternal` that switches on
`spec.kind == "external"` and reads `spec.params` — no closure capture.

### Change 2 — `_run_worker` calls the builder

Replace the two existing call sites in
[evaluate.py:294-309](../../src/elitefurretai/rl/analyze/evaluate.py#L294-L309)
that currently invoke `spec.factory(...)` with `build_player(spec, ...)`.
This is a mechanical rename — `_build_player` already centralizes the
team-provider + collector hookup.

### Change 3 — switch the executor

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# At module import time, top-level, before any CUDA touches it:
_SPAWN_CTX = mp.get_context("spawn")

# In the worker fan-out:
with ProcessPoolExecutor(
    max_workers=effective_workers,
    mp_context=_SPAWN_CTX,
) as pool:
    ...
```

Rationale for `spawn` (not `fork`):

- `fork` after CUDA initialization leaves the child with a corrupted CUDA
  context. Even if we don't initialize CUDA in the parent before forking,
  any incidental torch import that touches `torch.cuda` poisons children.
- `spawn` re-imports the module from scratch in each child, so each gets a
  fresh CUDA context. The cost is ~5-10 s of cold-start per worker.

### Change 4 — accept and survive cold-start cost

Each worker spawns from scratch and re-runs the same imports + model
load:

| Per-worker cold-start (estimated) | Time | Memory |
|---|---|---|
| Import `torch`, `poke_env`, project | ~3 s | ~250 MB RAM |
| Load model checkpoint to CUDA | ~5 s | ~250 MB GPU |
| Establish Showdown websocket | ~1 s | — |

Total ~9 s × 4 workers ≈ ~36 s amortized cold-start, paid once per
`evaluate.py` invocation. Cells run for hours afterward, so this is
negligible.

### Change 5 — CLI fallback flag (optional, recommended)

Add `--executor {process,thread}` defaulting to `process`. The thread
path is kept so smoke tests in the unit-test directory keep working
without spawning subprocesses (they're slow under pytest).

## Reasoning

### Why ProcessPool, not subprocess-shard (Option C)

Both achieve the same parallelism. Process pool is cleaner because:

1. Cells are partitioned once inside the parent (same logic as today),
   so the `manifest.json`, `result_*.json`, and any aggregated counts
   stay in one process and remain coherent.
2. The wrapper script stays simple — no `--cell-slice i/N` flag to
   pass around.
3. Tests can still exercise the threaded path for fast feedback.

Subprocess-shard would require:
- A new `--cell-slice` CLI flag.
- A coordinator step (sum cells across N result_*.json shards).
- Per-process port allocation, retry logic that knows about shards.

### Why not centralized inference (Plan C from the 2026-05-15 doc)

Plan C is a multi-week effort meant to unlock training throughput, not
eval. For eval, model inference is a small fraction of per-turn CPU
(embedder + max_damage damage calc dominate), and loading the model
N times costs ~1 GB GPU total — affordable. We do not need the inference
RPC machinery for an N=4 eval workload.

### Why spawn over fork even though spawn is slower

A 9 s cold-start per worker is invisible against multi-hour runs. The
correctness gain from a fresh CUDA context per process is worth it.
Tested in training-side `Plan C` work — `spawn` is the project default
for any multiprocessing that touches CUDA.

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Workers visible in `pstree` | 1 process, 28 threads | 4 processes, ~7 threads each |
| CPU cores active | 1 | 4 |
| Throughput (max_damage) | ~3.9 b/s | ~12 b/s (3× speedup, 25% loss to shared GPU + per-server battle parsing) |
| Wall time per opp_type | ~12 h | ~4 h |
| Full 4-opp_type sweep | ~48 h | ~16 h |

Per-process RSS at steady state will rise to ~2.5 GB × 4 ≈ 10 GB
total (still under our 23 GB WSL2 budget; the [poke-env
battles leak fix](../../src/elitefurretai/rl/analyze/evaluate.py#L319-L331)
keeps each process flat). GPU memory ~1 GB total, well under the 24 GB
RTX 3090 limit.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CUDA spawn quirks (e.g. `RuntimeError: Cannot re-initialize CUDA in forked subprocess`) | Force `mp.get_context("spawn")`; never touch `torch.cuda` in the parent before pool creation. Add an `assert torch.cuda.is_initialized() is False` guard before pool.submit. |
| Pickle errors on non-trivial args (e.g. `cells` containing weird objects) | Cells are `Tuple[str, str]` — strings, already picklable. PlayerSpec becomes pure data per Change 1. |
| In-flight wrappers (v4 + vgcbench-parallel) break if evaluate.py is edited | Land the refactor on a feature branch; merge to main only after the current run completes (or after a smoke test on a non-shared `--collect-trajectories` dir). |
| TrajectoryCollector concurrency assumptions | Each worker still owns one collector; per-worker shard files (`battles_worker_<i>_<call_id>_*.parquet`) are already isolated by `worker_id` — no shared writes. |
| First-cell startup race on Showdown (subprocess all hit the same port at once) | Today's `_run_worker` already handles per-worker `account_configuration` and per-server URL; the only new thing is process boundary, which doesn't change Showdown's view. |
| VGCBenchManager subprocess management | Each worker process launches its own vgc_bench subprocess (same as today, just in a different python process). PID tracking and cleanup in `RunningExternal.shutdown` continues to work. |

## Validation Plan

1. **Unit smoke** — run `pytest unit_tests/rl/analyze/test_eval_collector.py
   -q` after Change 1 to confirm PlayerSpec is still constructible and
   `build_player` returns the right type per `kind`.

2. **Local cell smoke** — 1 cell, 10 battles, `--workers 2 --num-servers 2
   --executor process`, vs `simple_heuristic`. Confirm:
   - 2 OS processes appear in `ps`
   - Each consumes its own CPU core (`top -H -p ...`)
   - Per-worker parquet shards land on disk
   - `result_*.json` aggregates the two workers' wins correctly

3. **Threaded fallback smoke** — same test with `--executor thread` to
   confirm the legacy path still works.

4. **Full-run check** — after the in-flight balmy70 sweep finishes
   (or is killed), run a fresh single-opp_type sweep against
   `simple_heuristic` for ~30 min and confirm:
   - Throughput is ≥10 b/s
   - 4 cores at 90%+ each
   - RSS per worker stays under 4 GB
   - Total GPU memory stays under 2 GB

## Planned Next Steps

1. Branch off `main`. Filename: `evaluate-process-pool`.
2. Implement Change 1 (`PlayerSpec` refactor + `build_player`). Run
   `ruff check`, `pyright`, `pytest unit_tests`.
3. Implement Change 2 (call-site rename in `_run_worker`).
4. Implement Change 3 (`ProcessPoolExecutor` + spawn). Add the
   `--executor` flag.
5. Run validation steps 1-3.
6. Open a PR. Don't merge until in-flight balmy70 sweep is done — the
   running v4/vgcbench-parallel processes are tied to the current
   evaluate.py module signature via the wrapper CLI args.
7. After merge: drop the `vgcbench-parallel` side-launch — single
   evaluate.py invocation per opp_type becomes fast enough on its own.

## Updates

### 2026-05-20 23:15 — Refactor landed on `evaluate-process-pool` branch

Commits on the branch:

* `7730a10` — Change 1: PlayerSpec to pure data, module-level
  `build_player()` + `launch_external_player()`. Closure introspection
  in `_detect_device_from_spec` / `_battle_format_from_spec` replaced
  with direct `spec.params[...]` reads. All 93 `unit_tests/rl/analyze`
  tests pass.
* `f9b4b58` — Change 3: `--executor {process,thread}` flag with
  ProcessPoolExecutor + `mp.get_context("spawn")` as default.

Skipped Change 5 (`--executor` flag *was* added — but kept as the
plan said, defaulting to `process`).

### 2026-05-20 23:16 — Smoke tests passed

Three smoke runs on the branch against `simple_heuristic` with the
balmy70 step-10700 checkpoint and a tiny 2×2 team Cartesian:

| Run | Battles | Workers | Executor | Duration | Notes |
|---|---|---|---|---|---|
| short process | 16 | 2 | process | 15.86 s | rc=0, 4 shards, no pickle/CUDA errors |
| short thread | 16 | 2 | thread | 8.34 s | rc=0, 4 shards, validates legacy path |
| long process | 80 | 2 | process | 24.22 s | rc=0, balanced 40+40 shards across workers |
| sustained process | 400 | 2 | process | ~75 s | rc=0, mid-run `pstree` confirmed 2 OS workers |

Mid-run inspection of the sustained 400-battle run:

```
python(1128) — parent (0% CPU, 0.6 GB)
├── python(1141) — multiprocessing.resource_tracker (housekeeping)
├── python(1142) — worker 0 (80% CPU, 1.4 GB RES, 15+ threads)
└── python(1145) — worker 1 (90% CPU, 1.4 GB RES, ~10 threads)
```

Two cores at ~85% each, validating real CPU parallelism. Compare to
the in-flight v4 process (max_damage, threaded path): single OS
process at 100% of one core.

### Decision: hold merge until in-flight balmy70 sweep completes

The branch is ready, but merging now would change the evaluate.py
module signature *while* the running wrappers ([/tmp/run_full_eval_v4.sh
and /tmp/run_vgcbench_parallel.sh]) depend on the threaded import path.
A wrapper retry could pick up the new code mid-flight and re-invoke
with different process semantics. Easier to wait ~12 h, finish the
sweep, then merge cleanly.

After merge, the parallel-wrapper trick is no longer needed — a single
`evaluate.py --executor process --workers 4` invocation per opp_type
should hit ~12 b/s on its own.
