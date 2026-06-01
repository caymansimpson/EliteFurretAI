# VGCBench Latest-Checkpoint Migration (reg_all/seed1 @ 98.3M steps)

**Date**: 2026-05-29
**Status**: Implemented; offline-verified. Pending one live smoke run.

Adopt the latest `vgc_bench_baseline` checkpoint from Hugging Face
(`cameronangliss/vgc-bench-models .../results/saves_bc_sp/reg_all/seed1/98304000.zip`)
without disturbing the working Feb checkpoint setup.

## Context

EFA's `vgc_bench_baseline` opponent loads an SB3 PPO `.zip` via vgc-bench's
`PolicyPlayer`, run as a subprocess under an isolated venv (poke-env version skew
vs EFA). The checkpoint EFA used was a Feb model. Cayman asked to update to the
latest checkpoint on the BC+self-play, all-regulations run (`reg_all/seed1`).

## Before state

- Checkpoint: `data/models/vgc-bench-sb3-model.zip` (Feb), **754**-wide, flat
  `Box(7030)` obs, dex 316/567/938.
- Code: `../vgc-bench-src` editable @ `468886a` (pre-Dict-obs era).
- Venv: `../venv-vgcbench` with stock PyPI **poke-env 0.11.0**.
- This trio is internally consistent and works.

## Problem

The latest checkpoint is **not a drop-in file swap**. Decoding it:

- New checkpoint: `Dict{observation(6936), action_mask(214)}`, **764**-wide
  `pokemon_proj` input, dex 320/586/955, `policy_kwargs={d_model:256,
  choose_on_teampreview:true}`.
- `vgc-bench-src origin/main` (f9361d0) is the right *architecture* family (Dict
  obs + action masking) but computes **762** with stock poke-env 0.11.0 — matching
  neither the old (754) nor new (764) checkpoint.
- The width is driven by poke-env enum counts. `origin/main` pins
  `poke-env @ git+github.com/cameronangliss/poke-env.git@vgc-bench` (a **fork**,
  not PyPI 0.11.0). The dex JSON files (320/586/955) only set embedding *rows*,
  not width — they live on `origin/main` already.

So the HF model repo is ahead of the public code repo; reproducing **764**
requires the fork at a specific commit.

## Solution

Built an **isolated second stack**, leaving the Feb/754 setup untouched as a
fallback:

1. `../vgc-bench-src-bcsp` — vgc-bench code cloned @ `origin/main` (f9361d0),
   editable-installed into the new venv. Dex confirmed 320/586/955.
2. `../venv-vgcbench-bcsp` — `python3.12 -m venv` with:
   - `poke-env @ git+.../poke-env.git@vgc-bench` (the branch **tip**, currently
     `e9b61cdf`, which yields width **764** live and loads the checkpoint).
     NOTE: an earlier draft of this doc pinned `b3956ae58` based on an
     AST-count of fork enums that was off-by-one — see the 2026-05-30 update
     below; always use the branch tip, which is what vgc-bench `origin/main`
     pins anyway,
   - `stable_baselines3==2.7.1`, `torch==2.9.1`, `supersuit==3.9.3`,
     `gymnasium==1.2.3` (the only deps `PolicyPlayer` transitively needs — the
     heavy training-only deps open-spiel/imitation/transformers/nashpy were
     skipped via `pip install -e ... --no-deps`).
3. Checkpoint staged at `data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip`
   (gitignored; Feb model kept in place).
4. Config wired in `may26.yaml` + `easy_test.yaml`:
   `external_vgcbench_python_executable` → `../venv-vgcbench-bcsp/bin/python`,
   `vgc_bench_checkpoint_path` → the new `.zip`. `may15/16/25` and `multiformat`
   still point at the old venv + Feb checkpoint (working fallback).
5. `agents/vgcbench_manager.py`: replaced the now-meaningless
   `_VGC_BENCH_REQUIRED_POKE_ENV_PREFIX="0.11."` guard (both the fork and EFA's
   poke-env report 0.15.0) with a **width-consistency** check — compares the
   runtime per-Pokemon width (`utils.chunk_obs_len + 6*(embed_len-1)`) to the
   checkpoint's `pokemon_proj.in_features`. `unit_tests/agents/
   test_vgcbench_manager_in_process.py` rewritten to pin the new guard.

## Reasoning

- **Isolated stack over in-place rebuild**: keeps the one currently-working
  baseline intact while the new one is validated; EFA selects the interpreter
  per-config via `external_vgcbench_python_executable`, so the split is free.
- **Separate code checkout**: `vgc_bench` is editable-installed, so a single
  shared `vgc-bench-src` would force both venvs onto one code version. A second
  checkout (`vgc-bench-src-bcsp`) keeps the old venv on `468886a`.
- **Width check over version string**: the original guard assumed vgc-bench ⇒
  poke-env 0.11.x. The fork reports 0.15.0, identical to EFA's poke-env, so the
  version prefix can no longer discriminate. Observation width is the actual
  invariant the guard exists to protect.
- **`--no-deps` for vgc_bench**: `PolicyPlayer` only imports poke-env, SB3,
  torch, numpy, gymnasium, supersuit. The pinned heavy deps are training-only.

## Verification

Offline, under `../venv-vgcbench-bcsp` (cwd = `vgc-bench-src-bcsp`):
- computed width == **764**, dex 320/586/955;
- `PPO.load` OK → `Dict('action_mask': Box(214), 'observation': Box(6936))`;
- `PolicyPlayer` constructed with the exact subprocess kwargs
  (`account_configuration`, `server_configuration`, `team`,
  `accept_open_team_sheet`);
- `policy.predict` on a 764 obs → valid `[93 102]`.

EFA side: both wired configs load via `RNaDConfig`; all wired paths exist; agents
test suite green (26 passed); `ruff`/`pyright` clean on edited files.

## Planned next steps

1. **Live smoke run** (only thing not yet exercised): short `easy_test.yaml`
   training run with `vgc_bench_baseline` weight > 0; confirm the runner logs in
   and battles via `data/logs/vgcbench_runners/`.
2. **Re-baseline Stage II**: this is a far stronger opponent (98.3M-step BCSP).
   The 60%×4 graduation criterion shifts — re-measure `vgc_bench_baseline` win
   rate for the current best agent.
3. Decide whether to migrate the remaining configs (may15/16/25, multiformat) or
   keep them on the Feb fallback.

## Updates

### 2026-05-29 — fallback removed, fully consolidated

Dropped the dual-stack: the new stack is now the *only* stack.

- Deleted `../venv-vgcbench` (old 0.11.0 venv, ~7.6 GB), `../vgc-bench-src`
  (old code @468886a), and `data/models/vgc-bench-sb3-model.zip` + its symlink.
- Repointed all remaining configs (may15, may16, may25, may25_resume,
  multiformat) to `../venv-vgcbench-bcsp` + the new checkpoint.
- Updated code defaults to the new venv/checkpoint: `config.py`
  (`CurriculumConfig.vgc_bench_checkpoint_path`, plus the eval-side
  `vgcbench_checkpoint_path` / `vgcbench_python_executable`),
  `analyze/player_factory.py:parse_player_specification` defaults,
  `analyze/evaluate.py` `--vgc-bench-checkpoint-path` default, and the
  `_create_vgc_bench_player` default arg.
- Fixed all stale "poke_env 0.11.x" / `../venv-vgcbench` comments and
  docstrings (`vgcbench_manager.py`, `_vgcbench_subprocess.py`,
  `player_factory.py`, `AGENTS.md`, `RL.md`). The one remaining `poke_env 0.11`
  mention (`player_factory.py:543`) is FoulPlay's venv, not vgc-bench — left as is.
- Gates green: ruff/pyright clean, all 7 configs load with existing paths,
  agents tests pass (26), new venv still loads the checkpoint at width 764.

Single source of truth now: `../venv-vgcbench-bcsp` (poke-env fork `@vgc-bench` tip e9b61cdf) +
`../vgc-bench-src-bcsp` (vgc-bench @ f9361d0) +
`data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip`.

### 2026-05-30 — version-mismatch resolved + first live eval

**Root cause of the earlier in-battle crash was a poke-env version mismatch**, as
suspected. The fork commit must be the `@vgc-bench` branch **tip** (`e9b61cdf`,
2026-05-09, "add necessary features for vgc-bench operations"), not the older
`b3956ae58` I'd first pinned. My `b3956ae58` choice came from AST-counting enum
members in the fork source, which was off-by-one twice; the **live** width at the
tip is 764 and it loads the checkpoint. With the tip installed, the
`ValueError: Invalid action ... switch <active mon> not in action space` crash is
gone (the missing May-9 commit was exactly the action-handling fix). Always pin the
branch tip — it's what vgc-bench `origin/main` pins anyway.

**First live eval** (vgc_bench 98.3M reg_all vs heuristic baselines, 3 battles ×
41 constrained teams = 123 each, `--cell-iteration`):

| Opponent | vgc_bench W–L | vgc_bench WR |
|---|---|---|
| max_damage | 97–26 | **78.86%** |
| simple_heuristic | 94–29 | **76.42%** |
| max_base_power | — | pending* |

\* max_base_power blocked: the player-vs-player CLI (`main()` + `__main__` guard)
was removed from `analysis_utils.py` at 00:32 during a concurrent refactor of the
eval entrypoints, so `python -m ...analysis_utils` now no-ops. Re-run once the
entrypoint is stable.

**Operational gotcha**: run each baseline as its own fresh process. Running all
three in one shell loop hung the 2nd matchup — tearing down and immediately
relaunching Showdown servers on the same ports desynced the runner/challenger
("room does not exist" PMs, 0 active battles). Separate invocations (ports fully
released between) are reliable.
