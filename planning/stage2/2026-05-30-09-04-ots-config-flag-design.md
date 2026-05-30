# Open Team Sheets (OTS) Config Flag — Design

**Date**: 2026-05-30
**Status**: Design approved; ready for implementation plan.

Add a configurable Open Team Sheets setting to both the training config and the
eval config so every agent in a given run uses a consistent OTS choice.

## Context

Pokémon Showdown VGC formats fall into three OTS categories (see
`pokemon-showdown/data/rulesets.ts`):

- **Soft `Open Team Sheets`** — at team preview each player gets an Accept/Deny
  prompt (`/acceptopenteamsheets`). Opt-in; sheets are revealed only if accepted.
  `gen9vgc2024regg` (our primary format) is in this group, along with VGC 2023
  Reg C/D, VGC 2026 Reg F/I, Champions Reg M-A.
- **Hard `Force Open Team Sheets`** — `onTeamPreview` reveals both sheets
  unconditionally. All the `(Bo3)` formats.
- **No OTS** — sheets never revealed.

In poke-env, whether a player accepts the soft prompt is the Player constructor
arg `accept_open_team_sheet`. In a soft-OTS format the two players must agree:
if one accepts and the other denies, the handshake mismatches and **the battle
is dropped** (documented at
[vgcbench_manager.py:236](../../src/elitefurretai/agents/vgcbench_manager.py)).
Forced formats ignore the arg; non-OTS formats have nothing to accept.

EFA currently runs `gen9vgc2024regg` for both training and eval, so the
`accept_open_team_sheet` arg is live and load-bearing.

## Before State

`accept_open_team_sheet` is set ad hoc and inconsistently across the codebase:

| Site | Value |
|---|---|
| `rl/rl_trajectory_player.py` (main RL agent + opponents) | `False` (default) |
| `rl/opponents.py` baselines (`_make_baseline_pool`) | unset → poke-env default `False` |
| `agents/vgcbench_manager.py` `VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET` | `False` (ClassVar) |
| `rl/analyze/analysis_utils.py` `build_player`, `_launch_vgc_bench_subprocess` | `False` |
| `agents/bc_player.py` | `False` ("avoid OTS deadlock") |
| `agents/verbose_model_player.py`, `rl/analyze/play_human_vs_model.py` | `True` |
| `agents/vgcbench_manager.py` `_create_vgc_bench_player` (in-process eval helper) | `True` default |

The config-driven training and eval paths all land on `False`, so battles
currently run with **closed** sheets — but there is no single knob to change it,
and the scattered defaults are a latent mismatch hazard.

## Problem

There is no way to run training or eval with Open Team Sheets without editing
multiple source files, and doing so risks setting one side but not the other
(dropping every affected battle). Reg G is a soft-OTS format on the real ladder,
so closed-sheet results may not reflect tournament conditions — but switching
must be all-or-nothing within a pool.

## Solution

Two independent config flags, each **global within its context** and defaulting
to `False` (exactly preserving current behavior). Every agent built within a
context reads that one value, guaranteeing both sides of every battle match.

### Config surface

- **Training**: `RNaDConfig.open_team_sheets: bool = False` — a new top-level
  field (it is a run-wide battle setting affecting the main agent *and* all
  opponents, not just the curriculum). Loaded from the top level of the YAML.
- **Eval**: `EvalConfig.open_team_sheets: bool = False` — a new field on
  `EvalConfig` (the `eval:` YAML block), plus an `--open-team-sheets`
  store-true flag on the standalone eval CLI for ad-hoc runs.

### Threading — training

`train.py` reads `config.open_team_sheets` and passes it to:

1. **`OpponentPool`** — new `open_team_sheets: bool = False` constructor
   parameter (added at [opponents.py:653](../../src/elitefurretai/rl/opponents.py)),
   threaded into:
   - the main + opponent `RLTrajectoryPlayer` constructions (opponents.py:925, 944),
   - every baseline Player built in `_make_baseline_pool` (opponents.py:841)
     (`MaxDamagePlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`,
     `RandomPlayer`, and `BCPlayer`).
2. **`VGCBenchManager`** — replace the hardcoded
   `ACCEPT_OPEN_TEAM_SHEET = False` ClassVar with a value read from the
   `RNaDConfig` the manager already holds (`self._config.open_team_sheets`),
   used to decide whether `launch()` appends `--accept-open-team-sheet` to the
   subprocess command.

### Threading — eval

- `EvalConfig.open_team_sheets` flows into the eval driver in
  `rl/analyze/analysis_utils.py`:
  - `build_player(..., accept_open_team_sheet=<flag>)` for in-process players,
  - `_launch_vgc_bench_subprocess` gates `--accept-open-team-sheet` on the flag
    (instead of `VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET`).
- The standalone CLI `--open-team-sheets` flag feeds the same path for ad-hoc
  `--player1/--player2` runs. NOTE: the player-vs-player CLI entrypoint is being
  refactored (its `main()` was moved out of `analysis_utils.py` on 2026-05-30);
  attach the flag to wherever that CLI currently lives, threading it into the
  same `build_player` / `_launch_vgc_bench_subprocess` calls.
- `evaluate_model.py` (the in-training graduation eval driver) reads
  `EvalConfig.open_team_sheets` and passes it through the same way.

### Out of scope

- The scattered per-agent constructor defaults (`verbose_model_player`,
  `play_human_vs_model`, `bc_player`, `_create_vgc_bench_player`) are left as-is.
  They are standalone/interactive conveniences not driven by these configs; the
  config-driven training and eval pools become consistent regardless.
- No change to format selection or the Showdown rulesets. The flag only controls
  the soft-OTS accept handshake; forced/non-OTS formats are unaffected by it.

## Reasoning

- **Two flags, not one** (per design discussion): lets training and eval differ
  (e.g., train closed for stability, eval open to match ladder), while each stays
  internally consistent.
- **Global-within-context, not per-opponent**: the mismatch-drops-battles
  constraint makes per-opponent flags a silent-data-loss trap. One value per
  context removes that class of bug entirely.
- **Default `False`**: zero behavior change for existing configs/runs and keeps
  current win-rate baselines comparable; OTS is strictly opt-in.
- **Top-level for training, `EvalConfig` for eval**: the training flag affects the
  main agent too, so it does not belong under `curriculum:`; the eval flag
  belongs with the rest of the eval knobs.
- **`VGCBenchManager` reads from config rather than a ClassVar**: the external
  runner must match the in-process side, and the manager already receives the
  `RNaDConfig`, so this also removes the stale "must match the main agent
  (currently False)" hand-coordination.

## Risks

1. **Both-sides coverage.** Every construction site within a context must receive
   the flag; a missed site silently reverts to `False` and (if others are `True`)
   drops those battles. Mitigation: enumerate all construction sites (done in
   "Threading" above) and add a test asserting an OTS-enabled `OpponentPool`
   builds players with `accept_open_team_sheet=True`.
2. **vgc_bench external runner.** Its OTS is set at subprocess launch via a CLI
   flag; the in-process side it battles must match. Covered by both reading the
   same context flag.
3. **Forced/Bo3 formats.** The accept arg is ignored there; the flag is a no-op,
   which is correct. No special handling needed.

## Planned Next Steps

1. Write the implementation plan (writing-plans skill).
2. Implement: config fields + YAML loaders, `OpponentPool` param + threading,
   `VGCBenchManager` config read, eval driver + CLI threading.
3. Tests: config round-trip for both flags; `OpponentPool` builds players with
   the expected `accept_open_team_sheet`; eval `build_player` honors the flag.
4. Quality gates (ruff, pyright, pytest).
5. Optional follow-up validation: a short Reg G eval with the eval flag `True`
   to confirm OTS battles complete (both sides accept, no drops) and to compare
   win rates vs the closed-sheet baseline.

## Updates

### 2026-05-30 — implemented (T1–T6) + validated

Branch `feat/open-team-sheets-config`. T1–T6 landed (config fields + `from_dict`,
`WorkerOpponentFactory` threading, `vgc_environment` wiring, `VGCBenchManager`
training gate, eval-driver threading through `run_eval_parallel` → `_run_worker`
→ `build_player`/`launch_external_player` → `_launch_vgc_bench_subprocess`; dead
`VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET` ClassVar removed). All quality gates
green (117 tests across the touched suites, ruff clean).

**T7 deferred**: `evaluate_model.py` (the in-training eval driver that would
forward `EvalConfig.open_team_sheets` into `run_eval_parallel`) had uncommitted
working-tree edits during this work; left untouched to avoid clobbering. One
remaining change: add `open_team_sheets=eval_cfg.open_team_sheets` at its
`run_eval_parallel(...)` call once that refactor is committed. The standalone
player-vs-player CLI was also removed from `analysis_utils.py` during the
refactor; whatever entrypoint replaces it should expose `--open-team-sheets`.

**Finding 1 — vgc_bench is OTS-incompatible.** Running the `reg_all` vgc_bench
checkpoint with OTS=on crashes its policy:
`RuntimeError: shape '[1, 12, -1]' is invalid for input of size 8670`. Revealing
sheets changes vgc-bench's observation width (8670 ≠ the 6936 = 12×578 the policy
expects), so its forward pass fails and battles never complete. The EFA flag is
correct (the runner command gains `--accept-open-team-sheet` and the in-process
side accepts — verified) — it's vgc-bench that can't consume OTS observations.
**Implication:** do not enable `open_team_sheets` while vgc_bench is in the
training curriculum or an eval bucket (with this checkpoint); EFA's own agents +
heuristic baselines are unaffected.

**Finding 2 — keep `run_tag` short.** Driving the eval with a long `run_tag`
pushed Showdown usernames past the 18-char cap; the truncation desynced the
challenge handshake (a flood of "room does not exist" PMs, battles never
completing). The original CLI used a 4-hex tag; the replacement CLI must keep
`run_tag` short. (Not an OTS bug — a latent eval-harness constraint surfaced
while validating.)

**Validation results** — vgc_bench (98.3M reg_all) vs heuristic baselines,
**closed sheets** (OTS=on impossible per Finding 1), constrained Reg G team pool,
3 battles × 41 teams = 123 each:

| Opponent | vgc_bench W–L | vgc_bench WR |
|---|---|---|
| max_damage | 101–22 | **82.11%** |
| max_base_power | 100–23 | **81.30%** |
| simple_heuristic | 90–33 | **73.17%** |
