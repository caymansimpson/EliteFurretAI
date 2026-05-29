# rl/inference_*.py consolidation into inference_service.py

## Context

`src/elitefurretai/rl/` carried four files implementing centralized inference:
`inference_ipc.py` (wire dataclasses), `inference_trainer.py` (trainer-side
service + batch handler), `inference_subprocess.py` (Plan C subprocess host),
and `inference_worker.py` (worker-side client). The four-file layout dates
to the Plan C build-out
([2026-05-15-01-30-plan-c-4-process-implementation.md](2026-05-15-01-30-plan-c-4-process-implementation.md)),
when the IPC layer, service loop, and subprocess host were each landing as
separate units and a clean per-file boundary made review easier.

With Plan C now stable and shipping, those three trainer-side units no longer
need separate files for their original review-time reasons — every consumer
that imports one tends to import another, and the line counts (83 / 434 /
318) are modest enough to live together.

## Before State

Four files under [src/elitefurretai/rl/](../../src/elitefurretai/rl/):

* `inference_ipc.py` (83 lines) — `InferenceRequest`, `InferenceResponse`,
  `EvictRequest`. No torch dependency beyond the type annotation.
* `inference_trainer.py` (434 lines) — `InferenceService`,
  `RealModelBatchHandler`, `echo_batch_handler`, `_COMPILE_LOCK`. Imports
  `inference_ipc`.
* `inference_subprocess.py` (318 lines) — `SyncWeightsMsg`, `ShutdownMsg`,
  `ServiceSpecification`, `SubprocessSpecification`, `run_subprocess`,
  `InferenceSubprocessHandle`. Imports `inference_ipc` and `inference_trainer`.
* `inference_worker.py` (256 lines) — `InferenceClient`,
  `WorkerInferenceClients`. Imports `inference_ipc`.

Consumer map at the start of this refactor:

| Module | Imports from |
|---|---|
| `model_registry.py` | ipc + trainer + subprocess |
| `worker.py` | inference_worker only |
| `opponents.py` | inference_worker only |
| `vgc_environment.py` | inference_worker only |
| `rl_trajectory_player.py` | inference_worker only |
| 5 test files | mixed across all four |

## Problem

The four-file split was not paying for itself anymore. `model_registry.py`
already imported across all three trainer-side files. None of the import
boundaries protected anything from anything — the trainer-side trio always
travels together. But collapsing all four into one is **not** equivalent:
`inference_worker.py` is loaded by code that runs in different processes
than the trainer-side code, and merging the worker stub into the trainer
stack forces every worker, environment, and opponent to pull the full torch
+ subprocess-host surface at import time.

A second question was whether `inference_worker.py` should fold into
`worker.py`. It cannot: `worker.py` already imports `opponents.py`, which
imports `inference_worker`. Folding the client into `worker.py` would create
a `worker → opponents → worker` cycle. The name collision (`worker.py` is
the process-launcher entry, `inference_worker.py` is a shared client
library) is unfortunate but the roles are distinct.

## Solution

Merge the three trainer-side files into a single
[`src/elitefurretai/rl/inference_service.py`](../../src/elitefurretai/rl/inference_service.py)
(~830 lines, organized in four sections: wire format → service → real-model
handler → subprocess host). Keep `inference_worker.py` separate.

Updated imports across the codebase:

* Source: `inference_worker.py`, `model_registry.py` (consolidated from
  three imports to one), plus stale module-name references in `config.py`,
  `rl_utils.py`, `train.py`, and `RL.md` (the three §s for ipc / trainer /
  subprocess collapsed into one § for inference_service).
* Tests: 5 files updated to import from `inference_service` instead of
  `inference_ipc` / `inference_trainer` / `inference_subprocess`. Test
  filenames left alone — `test_inference_subprocess.py` still describes its
  scope (the subprocess host).

## Reasoning

Two anchoring questions drove the cut:

1. *Are imports paying for themselves?* For the trainer-side trio: no.
   `model_registry.py` already pulls all three. The split fragmented the
   import block without preventing the natural co-import.
2. *Does any consumer load only one of the merged units?* For the trainer
   trio: no — they always come as a set. For `inference_worker.py`: yes,
   five different modules import it without touching the trainer-side
   code. That asymmetry is the rule the new boundary preserves.

The `worker.py` name collision is acceptable because the two files have
genuinely different roles (process entry vs shared client). Folding them
together would create a circular import that's only avoided today because
`inference_worker` is its own file.

## Planned Next Steps

None — this was a self-contained cleanup. Future inference-side work
(if/when a new layer joins) should ask the same two questions before
deciding whether to add a separate file or extend `inference_service.py`.

## Updates

* 2026-05-29 21:30 — initial cut. All quality gates pass: ruff check,
  pyright (0 errors), pytest unit_tests/rl (435 pass, 1 pre-existing
  skip). The lone ruff-format warning is on `supervised/fine_tune.py`,
  unrelated to this refactor.
