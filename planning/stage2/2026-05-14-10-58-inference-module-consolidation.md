# Inference Module Consolidation — 4 files → 2

**Date:** 2026-05-14 10:58

## Context

After the centralized-inference work shipped (2026-05-14, commit 3ead76c), the
`src/elitefurretai/rl/` folder had four `inference_*` modules that split along
fine-grained class boundaries rather than along the natural process boundary.
Filing for navigation / IDE jump-to-symbol was getting noisy.

## Before State

```
src/elitefurretai/rl/
├── inference_ipc.py            (95 lines)  — wire dataclasses
├── inference_service.py        (250 lines) — InferenceService (trainer-side)
├── inference_handlers.py       (265 lines) — RealModelBatchHandler (trainer-side)
├── inference_client.py         (177 lines) — InferenceClient (worker-side)
└── worker_inference_clients.py (91 lines)  — WorkerInferenceClients (worker-side)
```

Service and handlers always co-locate (service is constructed by passing a
handler to it). Client and bundle pair tightly (the bundle is a dict of
clients). The four-way split forced consumers to chase imports across four
files when the natural unit was "trainer-side" vs "worker-side".

## Problem

- Trainer code path needed two imports (`from … inference_service import …`,
  `from … inference_handlers import …`) for one logical concept.
- Worker code path needed two imports (`from … inference_client import …`,
  `from … worker_inference_clients import …`) for what is essentially "the
  thing the worker uses to talk to inference."
- Tests importing both halves had four-line import blocks where two would do.

## Solution

Merge along the process boundary:

```
src/elitefurretai/rl/
├── inference_ipc.py     (95 lines)  — wire dataclasses (unchanged)
├── inference_trainer.py (~515 lines) — InferenceService + RealModelBatchHandler + echo_batch_handler
└── inference_worker.py  (~245 lines) — InferenceClient + WorkerInferenceClients
```

Updated imports in 9 callers:
- `src/elitefurretai/rl/worker.py`
- `src/elitefurretai/rl/players.py`
- `src/elitefurretai/rl/opponents.py`
- `src/elitefurretai/rl/model_registry.py`
- `src/elitefurretai/engine/vgc_environment.py`
- `unit_tests/rl/test_inference_handler_real_model.py`
- `unit_tests/rl/test_inference_multistep_and_mp.py`
- `unit_tests/rl/test_inference_ipc_round_trip.py`
- `unit_tests/rl/test_model_registry.py`

Updated `src/elitefurretai/rl/RL.md` "Files in This Module" table to point at the
two new files.

The merged `inference_trainer.py` captures the in-progress LSTM-removal cleanup
that was sitting in the working tree of `inference_handlers.py` (LSTM hidden-state
branching and `_pad_lstm_state` removed; the codebase is transformer-only now).

## Reasoning

- **Process boundary is the natural split.** `InferenceService` only ever runs
  in the trainer; it always wraps a handler. `InferenceClient` only ever runs
  in a worker; the bundle is just a per-model dict of clients. The pairings
  never cross.
- **No coupling lost.** `RealModelBatchHandler` is the only handler shipped
  today (the `echo_batch_handler` stub doesn't need a separate file). If a new
  handler variant ships later — BC handler, exploiter handler, etc. — we can
  split it back out, but until then the merged file is the simpler form.
- **Considered and rejected: fold the client into `worker.py`.** `worker.py`
  is already 724 lines; the client/bundle classes are independently testable
  IPC plumbing and would bloat `worker.py` to ~1000 lines while coupling the
  IPC layer to the actor lifecycle.

## Verification

- `ruff check` + `ruff format --check`: clean on touched files
- `pyright`: 0 errors / 0 warnings on touched files
- `pytest unit_tests/rl/test_inference_*.py unit_tests/rl/test_model_registry.py`:
  27/27 pass
- Pre-existing failures in `unit_tests/rl/test_config.py` (2 tests) are
  unrelated — they were already broken from in-progress config edits before
  this refactor.

## Planned Next Steps

None — this was a localized cleanup. The post-merge layout is the new
steady state; future inference work edits these two files directly.

## Updates

(none yet)
