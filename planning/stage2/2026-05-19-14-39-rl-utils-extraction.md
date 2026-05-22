# rl_utils.py extraction — Tier 1 + Tier 2 cleanup

## Context

`src/elitefurretai/rl/` had grown to ~9k lines across 14 files. A survey of
duplication and off-purpose code identified candidates for extraction into
a new `rl_utils.py` so each file could stay closer to its stated purpose
(orchestration in `train.py`, actor loop in `worker.py`, etc.).

## Before state

Duplicates and off-purpose helpers spread across the package:

- `logging.basicConfig` + `getLogger("elitefurretai").setLevel(INFO)` +
  `getLogger("__main__").setLevel(INFO)` boilerplate at both
  `train.py:2116-2122` and `worker.py:159-166` (worker variant passes
  `force=True` because third-party imports in subprocesses may have
  already attached a root handler).
- `_list_model_checkpoints` listdir+`.pt`-filter loop appeared **twice in
  the same file** (`opponents.py:286-293` and `opponents.py:778-793`).
  The second was on `WorkerOpponentFactory` with an added compatibility
  filter, and a `grep` across `src/` + `unit_tests/` found **zero
  callers** — dead code.
- `_read_pss_bytes`, `_sum_process_tree_rss_bytes`, and
  `start_memory_watchdog` lived in `train.py:118-241` (~125 lines of
  Linux/psutil system instrumentation embedded in the training
  orchestrator).
- `get_memory_usage_mb` closure in `worker.py:136-145` duplicated the
  same KB/MB/GB formatting ladder.
- Three inline `startswith("cuda")` device checks at
  `learners.py:143`, `learners.py:190`, and `inference_subprocess.py:136`.
- Three inline `datetime.now().isoformat()` calls at `train.py:268`,
  `train.py:925`, and `learners.py:906`.

## Problem

Off-purpose plumbing was the largest concentration of non-domain code in
the two highest-signal files (`train.py`, `worker.py`). Reducing it
makes the remaining content (training loop / actor loop) easier to scan.
The literal duplication in `opponents.py` was also a maintenance hazard.

## Solution

Created `src/elitefurretai/rl/rl_utils.py` with:

- `setup_logging(force=False)` — replaces both basicConfig blocks; the
  worker passes `force=True`.
- `list_pt_files(directory)` — replaces both `_list_model_checkpoints`
  helpers in `opponents.py`. The dead WorkerOpponentFactory version was
  deleted outright (also dropped the now-unused
  `is_checkpoint_compatible_with_model_config` import from
  `opponents.py`).
- `read_pss_bytes(pid)`, `format_memory_bytes(n)`,
  `sum_process_tree_pss_bytes()` — moved from `train.py` + `worker.py`.
  The role-classification (showdown/vgcbench/workers/other) stays
  inside `sum_process_tree_pss_bytes` since rl_utils is itself a
  project-specific module.
- `start_memory_watchdog(event, threshold_gb, poll_interval_s)` — moved
  intact from `train.py`. Logs via `rl_utils`'s own logger, still under
  the `elitefurretai` hierarchy so it remains at INFO.
- `is_cuda_device(device)` — replaces three inline `.startswith("cuda")`
  checks (one had `str(...)` wrap, the others did not; helper accepts
  `Any` to normalize).
- `timestamp_iso()` — replaces three inline `datetime.now().isoformat()`
  calls.

The pre-existing `load_checkpoint()` at `learners.py:914` was **not**
re-extracted — it's already in the right place, and the inline
`torch.load` variations in `opponents.py:266-267` and
`load_model_from_checkpoint` have meaningfully different semantics
(`weights_only`, `map_location` target) that aren't worth forcing into a
single helper.

## Reasoning

- Why not include a `checkpoint_path(...)` helper: the actual callsites
  produce wildly different filename shapes
  (`worker_bootstrap_initial.pt`, `main_model_step_{step}.pt`,
  `ghosts/`, `exploiters/`). No consistent extraction.
- Why keep the `get_memory_usage_mb` closure in `worker.py` rather than
  remove it: the closure captures `psutil.Process(os.getpid())` and is
  called at 9 logging sites. Inlining a 1-liner at the closure body
  (`return format_memory_bytes(psutil.Process(os.getpid()).memory_info().rss)`)
  removed the 7-line formatting ladder without touching any callsite.
- Why move `start_memory_watchdog` even though it's project-specific:
  it logs via a module logger and uses `sum_process_tree_pss_bytes`;
  both belong together. Tests had been patching
  `train_mod._sum_process_tree_rss_bytes` directly — updated to patch
  `rl_utils.sum_process_tree_pss_bytes` instead.

## Diff summary

```
 src/elitefurretai/rl/rl_utils.py        | 224 +++++++++++++++++++++++++ (new)
 src/elitefurretai/rl/train.py           | -149 (removed memory helpers, watchdog, basicConfig block)
 src/elitefurretai/rl/worker.py          | -32  (removed memory ladder, basicConfig block + comments)
 src/elitefurretai/rl/opponents.py       | -27  (removed dead WorkerOpponentFactory checkpoint lister)
 src/elitefurretai/rl/learners.py        | -3   (timestamp + is_cuda swaps, dropped datetime import)
 src/elitefurretai/rl/inference_subprocess.py | -1
 unit_tests/rl/test_memory_watchdog.py   | updated patch targets to rl_utils
```

Net: ~210 lines deleted from the five high-signal files, ~224 added to
the single utility file (most of which is doc comments preserved from
the original locations).

## Verification

- `ruff check src unit_tests` — clean.
- `ruff format --check` on the 6 changed files + new file — all
  formatted. (Two pre-existing `Would reformat` hits in
  `analyze/eval_collector.py` and `unit_tests/rl/analyze/test_eval_collector.py`
  were already dirty before this work — unrelated to this refactor.)
- `pyright` on all changed files — 0 errors, 0 warnings.
- `pytest unit_tests` — 596 passed, 1 skipped, 36 deselected. The two
  `test_memory_watchdog.py` tests that patched `train._sum_process_tree_rss_bytes`
  were updated to patch `rl_utils.sum_process_tree_pss_bytes` instead.

## Planned next steps

None — this is a self-contained cleanup. If we later notice further
duplication (e.g. checkpoint-loading variants drifting back together,
or the path-joining patterns stabilizing into a real convention),
revisit then.

## Updates

- 2026-05-19 14:39 — Initial extraction complete. All quality gates pass.
