# Baseline Naming Standardization + Stage II Graduation Revision

**Date**: 2026-05-31
**Status**: Shipped (code + configs + tests + authoritative docs). One pre-existing
unrelated config issue noted under "Updates".

## Context

While diagnosing why rosy-armadillo-80 (may31.yaml) showed ~50% vs vgc_bench but
~20% vs simple_heuristic, two cross-cutting cleanups surfaced:

1. Baseline opponent identifiers were inconsistent — the curriculum side used a
   mix of `_baseline`-suffixed and bare names (`max_damage` bare,
   `simple_heuristic_baseline`/`random_baseline`/`max_base_power_baseline`/
   `vgc_bench_baseline` suffixed), while the eval parser
   (`parse_player_specification`) used bare canonical names. That mismatch was
   silently crashing the eval pass every 1000 updates
   (`simple_heuristic_baseline` is not a resolvable eval spec).
2. The Stage II graduation criterion recorded in the authoritative docs
   (CLAUDE.md, RL.md, memory) was stale — "≥60% vs four baseline" — and no
   longer matches Cayman's current bar.

## Before state

- `OpponentPool` constants: `RANDOM_BASELINE="random_baseline"`,
  `MAX_BASE_POWER_BASELINE="max_base_power_baseline"`,
  `SIMPLE_HEURISTIC_BASELINE="simple_heuristic_baseline"`,
  `VGC_BENCH_BASELINE="vgc_bench_baseline"` (and bare `MAX_DAMAGE="max_damage"`).
- Curriculum-weight keys + win-rate log keys used the suffixed forms; eval used
  bare forms.
- Eval crashed on every attempt (updates 1000, 2000 in rosy-armadillo-80) →
  the only win-rate signal was the noisy 100-battle training window.
- Graduation criterion in CLAUDE.md / RL.md / `project_immediate_goal.md`:
  simultaneous ≥60% vs `vgc_bench`, `max_damage`, `bc_player`, `simple_heuristic`.

## Problem

Inconsistent nomenclature (a) is a footgun — the eval-key crash is one instance —
and (b) makes the codebase harder to reason about. The stale graduation bar
risks future sessions optimizing toward the wrong target.

## Solution

**Naming — standardized on bare names (no `_baseline` suffix):**
`random`, `max_base_power`, `simple_heuristic`, `vgc_bench` (joining the already
bare `max_damage`, `foul_play`). Both the `OpponentPool` constant *names* and
their string *values* were renamed (`SIMPLE_HEURISTIC_BASELINE` →
`SIMPLE_HEURISTIC`, etc.). Applied across `opponents.py`, `config.py`,
`train.py`, `worker.py`, `vgc_environment.py`, `_vgcbench_subprocess.py`,
`vgcbench_manager.py`, all config YAMLs (easy_test, example_eval, may15, may16,
may25, may26, may31), the three affected test files, and RL.md. This also fixes
the eval crash for free (eval keys now resolve).

**Graduation criterion (revised 2026-05-31):** simultaneously
- ≥80% vs `simple_heuristic`, `max_damage`, `bc_player`
- ≥60% vs `vgc_bench`
- ≥45% vs `foul_play`

FoulPlay is non-functional today, so that bucket is blocked until the opponent
is repaired. Updated in CLAUDE.md, RL.md (§1, the curriculum-rationale
paragraph, and the FoulPlay note), the graduation-criteria doc
(`2026-05-16-21-30`, new Updates section marking the 60%×4 bar superseded),
may31.yaml eval targets, and memory (`project_immediate_goal.md`, MEMORY.md
index, new `project_heuristic_exploitation_gap.md`).

## Reasoning

- Bare names match the eval parser's existing canonical set + alias map, are
  shorter, and `max_damage` was already bare — least churn, removes the crash.
- Renaming the constant attribute names (not just values) avoids the confusing
  `SIMPLE_HEURISTIC_BASELINE = "simple_heuristic"` half-measure.
- Backwards compat is explicitly not a concern (CLAUDE.md). One resume caveat:
  a checkpoint whose saved curriculum dict still has old keys, if resumed after
  this change, would carry stale keys that no longer match constants and
  silently never sample. So this is a next-fresh-run change. The currently
  running job already loaded its config and is unaffected.
- The 80% heuristic bar reflects that a strong agent should *dominate*
  deterministic, exploitable opponents — see
  `project_heuristic_exploitation_gap.md` for the diagnosis (structural
  self-play narrowness + `rnad_alpha`/entropy regularization, not OTS, not a
  regression).

## Planned next steps

1. On the next fresh RL run, lower `rnad_alpha` (0.3 → ~0.15, consider
   annealing) as the targeted lever for the heuristic-exploitation gap; keep
   `ent_coef` roughly as-is. Change one knob at a time.
2. Re-enable the `foul_play` eval bucket (target 0.45) once the FoulPlay
   opponent is repaired.
3. Smoke-test the OTS "mixed" training path on the next RL run (still
   un-exercised end-to-end).

## Updates

- `example_eval.yaml` had a **pre-existing, unrelated** load failure: its
  `agent_team_path` and `opponent_team_pool_path` keyed the second format as
  `gen9vgc2024regi` while `battle_formats` declared `gen9vgc2026regi`, and the
  validator requires those per-format dicts to match `battle_formats` exactly.
  Not caused by this work (the rename only touched baseline opponent tokens,
  never format names). **Fixed by Cayman 2026-05-31** (both keys → `gen9vgc2026regi`);
  the config loads cleanly now.
