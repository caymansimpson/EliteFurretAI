# `agents/` Directory Reorganization

**Date**: 2026-05-19
**Status**: Pre-implementation. Design agreed; ready for plan writing.

Consolidate the user-facing battle-participant classes scattered across
`rl/players.py` and `supervised/behavior_clone_player.py` into a new
top-level `src/elitefurretai/agents/` directory. Move the
`VGCBenchManager` (and the upcoming `FoulPlayManager`) into the same
place so the answer to "how do we run X agent in EFA?" is a single `ls`.

---

## Context

`src/elitefurretai/rl/players.py` has grown to ~1948 lines covering five
distinct concerns:

1. `BatchInferencePlayer` — high-throughput async player used during RL
   training, dynamic-batches model decisions across many concurrent
   battles.
2. `RNaDAgent` — `torch.nn.Module` wrapper around the trained model;
   uniform `forward()` API over the underlying transformer.
3. `SimpleModelPlayer` / `VerboseModelPlayer` — eval-time players that
   run inference inline in `choose_move`.
4. `MaxDamagePlayer` — heuristic baseline.
5. `VGCBenchManager` (added 2026-05-18 per
   [vgcbench-consolidation][vgc-consolidation]) — subprocess lifecycle
   for the external vgc-bench bot, plus the three helpers that the
   consolidation pulled in (`_create_vgc_bench_player`, `_temporary_cwd`,
   `_resolve_vgc_bench_root`).

In parallel, `src/elitefurretai/supervised/behavior_clone_player.py`
defines `BCPlayer` — a `Player` subclass loading a supervised checkpoint.
It's structurally the same shape as `SimpleModelPlayer` but lives in a
different subsystem for historical reasons.

[`FoulPlayManager`][foulplay-doc] is queued for landing as a near-exact
mirror of `VGCBenchManager`, currently planned in `rl/players.py`. If we
do nothing, `players.py` grows to ~2200 lines with two semi-related
subprocess managers and four `Player` subclasses sharing a file with a
`torch.nn.Module`.

[vgc-consolidation]: 2026-05-18-14-00-vgcbench-consolidation.md
[foulplay-doc]: 2026-05-18-16-00-foulplay-eval-integration.md

## Before state

```
src/elitefurretai/rl/
  players.py                   # 1948 lines, 5 unrelated concerns
  opponents.py                 # OpponentPool, WorkerOpponentFactory
  _vgcbench_subprocess.py      # subprocess entry, runs under venv-vgcbench
  analyze/
    player_factory.py          # parse_player_spec, _model_spec, _baseline_spec
    team_provider.py
    ...
src/elitefurretai/supervised/
  behavior_clone_player.py     # BCPlayer
```

Import surface from `rl.players` and `supervised.behavior_clone_player`
spans ~30 files across `src/` and `unit_tests/` (full enumeration from
`grep` is in the migration step below).

## Problem

Two problems, both about taxonomy and discoverability:

1. **No single home for "agents you can run."** A new contributor (or
   future-Cayman after a month off) wanting to know "what ways are
   there to play a battle in EFA?" has to read through `rl/players.py`,
   `supervised/behavior_clone_player.py`, and the planned-but-not-yet
   `FoulPlayManager` location. The shared concept — "instantiable
   participant in a battle" — has no module that owns it.
2. **`players.py` mixes unrelated concerns.** The training-time batcher
   (`BatchInferencePlayer`), the model wrapper (`RNaDAgent`), and the
   user-facing players (`SimpleModelPlayer` et al.) all live together.
   Splitting them by concern matches the rest of `rl/` (one purpose per
   file: `learners.py`, `opponents.py`, `model_registry.py`, the
   `inference_*.py` family).

## Solution

### Components

**`src/elitefurretai/agents/`** (new top-level directory). Holds
user-facing, instantiable battle participants — anything someone would
grab to run a battle (eval, analysis, baselines, external bots).
Explicitly **not** for classes that subclass `Player` for training-
plumbing reasons.

```
src/elitefurretai/agents/
  __init__.py                  # public re-exports
  AGENTS.md                    # human-facing guide: what's here, how to use each
  simple_model_player.py       # from rl/players.py (eval-time, inline inference)
  verbose_model_player.py      # from rl/players.py (extends SimpleModelPlayer)
  max_damage_player.py         # from rl/players.py (heuristic baseline)
  bc_player.py                 # from supervised/behavior_clone_player.py
  vgcbench_manager.py          # VGCBenchManager + the three vgc-bench
                               # helpers consolidated 2026-05-18
                               # (_create_vgc_bench_player, _temporary_cwd,
                               # _resolve_vgc_bench_root)
  foulplay_manager.py          # replaces the planned rl/ location for the
                               # FoulPlay integration
  _vgcbench_subprocess.py      # moved from rl/ (entry point, runs under
                               # ../venv-vgcbench/bin/python)
  _foulplay_subprocess.py      # new (replaces the planned rl/ location)
```

The leading-underscore subprocess files signal "internal entry point,
not user-invocable" — matching the established naming convention from
the vgcbench consolidation.

**`AGENTS.md`** in `agents/` is human-facing documentation. Sections:

- **What this directory is** — one-paragraph framing (user-facing
  instantiable agents; not training plumbing).
- **What's in here** — table mapping each file to a one-line role
  description.
- **How to use each agent** — for each Player and Manager:
  - Construction signature (key args, what they default to).
  - Typical call site (which CLI / which training entry point uses it).
  - Any gotchas (e.g. `VerboseModelPlayer` prints to stdout; managers
    need a Showdown server running first).
- **How to add a new agent** — short checklist: subclass `Player`, add
  file, re-export from `__init__.py`, update this `AGENTS.md`.

`AGENTS.md` is documentation, not code — checked into the repo, lives
next to the files it documents (matching the project's existing
`RL.md`, `ENGINE.md`, `SUPERVISED.md` convention).

### What stays in `rl/`

`BatchInferencePlayer` and `RNaDAgent` are training-time internals.
Splitting `rl/players.py` finishes the cleanup:

```
src/elitefurretai/rl/
  batch_inference_player.py    # NEW: BatchInferencePlayer + the three executor
                               # helpers (_request_fingerprint,
                               # get_worker_executor, cleanup_worker_executors).
                               # Only consumer is the RL worker loop, so
                               # co-located.
  rnad_model.py                # NEW: RNaDAgent (renamed for clarity — it's a
                               # torch.nn.Module wrapper, not a Player)
  # players.py deleted
```

Rationale for the `RNaDAgent` → `rnad_model.py` rename: the class is a
`torch.nn.Module` wrapper around `TransformerThreeHeadedModel`, not a
poke-env `Player`. The current name (`*Agent` in `players.py`) is
actively misleading. The rename is a judgment call; if scope-creep is a
concern, keep the class in `rl/players.py` as a solo entry — the
agents-extraction value is independent.

### What does NOT move

| Class / file | Stays | Reason |
|---|---|---|
| `OpponentPool`, `WorkerOpponentFactory` | `rl/opponents.py` | Curriculum sampling / per-worker factory. Orchestrates Players, isn't one. |
| `player_factory.py`, `team_provider.py` | `rl/analyze/` | CLI-eval factories. Consume agents, aren't agents. |
| `rl/__init__.py` re-exports | Updated in place | `rl/__init__.py` publicly names `MaxDamagePlayer`, `BatchInferencePlayer`, `cleanup_worker_executors`. Re-imports from new homes preserve the surface for any downstream callers. |
| `supervised/__init__.py` `BCPlayer` re-export | Removed | `BCPlayer` moves; `supervised/` shouldn't pretend to own it anymore. Per CLAUDE.md, no backwards-compat shim required. |

### Migration mechanics

Five landable commits:

1. **Skeleton.** Create `agents/__init__.py` and stub `AGENTS.md`. No
   moves yet.
2. **Move + split with `git mv` where possible.**
   - `git mv src/elitefurretai/supervised/behavior_clone_player.py
     src/elitefurretai/agents/bc_player.py` (whole-file rename;
     git rename-detection tracks it).
   - `git mv src/elitefurretai/rl/_vgcbench_subprocess.py
     src/elitefurretai/agents/_vgcbench_subprocess.py`.
   - For `rl/players.py` splits: copy-then-delete-sections. Git
     rename-detection can't track partial-file moves, but `git log
     --follow` on the resulting files works against the new content
     and `git blame -C -C` recovers the line-level history.
   - One commit per wave: (a) eval players + heuristic + BC,
     (b) vgcbench manager + helpers + subprocess script,
     (c) batch_inference_player.py + rnad_model.py extraction from
     the remaining `rl/players.py`.
3. **Import sweep.** ~30 files across `src/` and `unit_tests/`. The
   grep enumeration:

   ```
   from elitefurretai.rl.players import ...           # ~20 sites
   from elitefurretai.rl.opponents import ...         # unchanged (stays)
   from elitefurretai.supervised.behavior_clone_player import BCPlayer  # ~5 sites
   ```

   `sed -i` over the import lines, then `ruff format`, then run quality
   gates.
4. **`__init__.py` updates.** `rl/__init__.py` re-imports moved classes
   from `agents/`. `supervised/__init__.py` drops the `BCPlayer`
   re-export entirely.
5. **Smoke test.** `ruff check`, `ruff format --check`, `pyright`,
   `pytest -q`, plus a 5-minute training run with `single_team.yaml`.
   Type-checking does not catch `importlib` / subprocess-script-path
   mistakes in `_vgcbench_subprocess.py`; only running the trainer
   confirms the subprocess launches correctly.

### FoulPlay plan adjustment

The FoulPlay integration design at
[2026-05-18-16-00-foulplay-eval-integration.md][foulplay-doc] places
`FoulPlayManager` and `_foulplay_subprocess.py` in `rl/`. Both move to
`agents/` under this design. Concrete edits to the FoulPlay doc:

1. `FoulPlayManager` class location:
   `src/elitefurretai/rl/players.py` →
   `src/elitefurretai/agents/foulplay_manager.py`.
2. `SUBPROCESS_SCRIPT` ClassVar:
   `"src/elitefurretai/rl/_foulplay_subprocess.py"` →
   `"src/elitefurretai/agents/_foulplay_subprocess.py"`.
3. Subprocess file's actual location matches the ClassVar.
4. References to "mirrors `VGCBenchManager` from `rl/players.py`" →
   "mirrors `VGCBenchManager` from `agents/vgcbench_manager.py`".

**Sequencing**: the `agents/` extraction lands **before** FoulPlay
implementation begins. That way `FoulPlayManager` is written against
the final layout, not the current one. If the FoulPlay timeline is
urgent, FoulPlay can land in `rl/` first and move with the broader
extraction — but that doubles the migration of FoulPlay-related files.
Recommend doing this extraction first.

## Reasoning

**Why a top-level `agents/` and not `rl/agents/`?** `BCPlayer` is one of
the agents we want consolidated, and it lives in `supervised/` for
historical reasons. Putting `agents/` under `rl/` would either leave
`BCPlayer` orphaned in `supervised/` or move it into `rl/agents/` and
break the "rl/ contains only RL training" assumption. Top-level
`agents/` treats "how to run an agent" as its own subsystem, peer to
`engine/`, `etl/`, `inference/`, etc. That matches Cayman's framing
("agents being a place where you can find definitions for how we run
various agents in EFA").

**Why exclude `BatchInferencePlayer` despite it being a `Player`
subclass?** `BatchInferencePlayer`'s real job is dynamic batching of
model decisions for RL training throughput. It's tightly coupled to the
trajectory queue, the inference IPC system, and worker process
orchestration. You'd never instantiate it to run an ad-hoc battle. The
agents/ taxonomy is "would someone outside the RL training loop ever
instantiate this to play a battle?" — answer for `BatchInferencePlayer`
is no, so it stays in `rl/`.

**Why one file per class?** Maximum discoverability. `ls agents/`
shows every agent at a glance; `git blame` is per-file and clean;
adding a new agent is "add one file, add one line to `__init__.py` and
`AGENTS.md`." The alternative (grouping by role) saves a few files but
makes "where is `BCPlayer` defined?" require either grep or knowing the
grouping convention. Files in `agents/` are not big enough for this to
be a maintenance burden.

**Why an `AGENTS.md`?** The directory holds heterogeneous classes that
look superficially similar (all `Player` subclasses or subprocess
managers) but have very different construction patterns and use cases.
A `SimpleModelPlayer` takes a checkpoint path and device; a
`VGCBenchManager` takes a config and server-port list and spawns a
subprocess. Without docs, future-Cayman opening the directory has to
read every file's docstring to figure out which one fits the current
need. `AGENTS.md` is one page that does that lookup. It also matches the
project's existing pattern (`RL.md`, `ENGINE.md`, `SUPERVISED.md`).

**Why rename `RNaDAgent`'s file?** With everything else moved out,
`rl/players.py` would contain only `RNaDAgent` — a torch.nn.Module
wrapper. The filename `players.py` for a one-class-nn.Module file is
actively misleading. Renaming to `rl/rnad_model.py` makes the file's
contents match its name. Flagged as a judgment call because it adds
~15 import-site edits over the strict minimum.

**Why not also reorganize `opponents.py` / `player_factory.py`?**
Different concerns. `OpponentPool` is about curriculum sampling and
opponent diversity — orchestrating Players, not being one.
`player_factory.py` is CLI-string parsing for the eval entry point. The
agents extraction is about the leaf classes; the orchestrators stay
where they belong with their consumers.

## Risks

1. **Blast radius.** ~30 import sites + mock-string updates in tests
   (e.g. `unit_tests/rl/test_players.py:61` does
   `patch("elitefurretai.rl.players.calculate_damage", ...)` — that
   string needs to become
   `elitefurretai.agents.max_damage_player.calculate_damage`). A missed
   import is an `ImportError` at runtime, not at lint time. Mitigation:
   pre-flight grep for `patch(.*rl\.players` and
   `patch(.*supervised\.behavior_clone_player`, plus the smoke training
   run in step 5.
2. **Subprocess `SUBPROCESS_SCRIPT` path drift.**
   `VGCBenchManager.SUBPROCESS_SCRIPT` is a literal string path. After
   the move it needs to be
   `"src/elitefurretai/agents/_vgcbench_subprocess.py"`. The trainer
   launches `../venv-vgcbench/bin/python <script>` — wrong path is a
   silent subprocess-fail-at-startup. Mitigation: the 5-min smoke
   training run will surface this (vgc-bench username never logs in).
3. **`opponents.py` becomes a three-source-import.** Currently
   `from elitefurretai.rl.players import BatchInferencePlayer,
   MaxDamagePlayer, RNaDAgent`. After: `BatchInferencePlayer` from
   `rl.batch_inference_player`, `MaxDamagePlayer` from
   `agents.max_damage_player`, `RNaDAgent` from `rl.rnad_model`.
   Trivial; flagging because it's the one file with imports from three
   new homes.
4. **The `RNaDAgent` rename.** ~15 sites update. If this feels like
   scope creep at implementation time, drop it — keep `rl/players.py`
   as a one-class file. Independent of the rest.

## Out of scope

- Touching `OpponentPool` / `WorkerOpponentFactory`. They consume
  agents but aren't agents. Stay in `rl/opponents.py`.
- Refactoring `player_factory.py` / `team_provider.py` in `rl/analyze/`.
  CLI scaffolding, not agents.
- Reducing `BatchInferencePlayer`'s line count. It's big because
  dynamic batching is intrinsically stateful; orthogonal to this
  reorg.
- Adding a public `elitefurretai.agents` API surface beyond simple
  re-exports — no abstract base, no registry, no factory function.
  Just files and an `__init__.py`.
- Backwards-compat shims from old import paths (`rl.players`,
  `supervised.behavior_clone_player`). Per CLAUDE.md, no.
- Adding `BatchInferencePlayer` to `AGENTS.md`. It's not in `agents/`.

## Planned next steps

1. Confirm this design with Cayman.
2. Write the implementation plan (separate doc, follow the
   subagent-driven-development pattern for the parallelizable waves).
3. Execute waves 1–5 in order. Quality gates + smoke run between
   wave 3 (import sweep) and any merge.
4. Update [foulplay-doc][foulplay-doc] with the path edits listed
   above.
5. Update `RL.md` and `SUPERVISED.md`:
   - `RL.md` — `VGCBenchManager` references point at the new file;
     mentions that `BatchInferencePlayer` and `RNaDAgent` were
     extracted from `players.py` into their own files.
   - `SUPERVISED.md` — `BCPlayer` import-path update.
6. Mark this doc complete in Updates.

## Updates

### 2026-05-19 — implementation complete

Landed in 8 commits per
[2026-05-19-10-00-agents-directory-reorg-implementation-plan.md](2026-05-19-10-00-agents-directory-reorg-implementation-plan.md)
(plus 7 prep commits to clear unrelated working-tree state). All
quality gates green after every phase (`ruff check`, `ruff format --check`,
`pyright`, `pytest unit_tests` → 527 passed, 1 skipped; unchanged from
baseline).

**Commits (chronological):**

1. `449fc1c` — agents: stub AGENTS.md alongside pre-existing HumanPlayer
2. `739b54f` — agents: move BCPlayer from supervised/ to agents/bc_player.py
3. `79cf2d0` — agents: move VGCBenchManager + vgc-bench helpers + subprocess to agents/
4. `90ec07e` — agents: move SimpleModelPlayer, VerboseModelPlayer, MaxDamagePlayer to agents/
5. `0f7cdf2` — rl: split players.py into batch_inference_player.py and rnad_model.py
6. `d9f2bb0` — agents: full __init__.py re-exports and AGENTS.md content
7. `584aaa0` — docs: update FoulPlay plans + RL.md + SUPERVISED.md for agents/ layout
8. (this commit) — planning: mark agents/ reorg spec complete

**Deviations from the design:**

- **Pre-existing `agents/` directory.** The design assumed
  `src/elitefurretai/agents/` did not exist. It actually did — Cayman
  had created it on 2026-05-16 (commit `7e3eb6b`) for `HumanPlayer`.
  Phase 1's implementer subagent overwrote the existing `__init__.py`
  without flagging it; caught via `git show --stat` inspection
  post-commit. Phase 1 was amended to preserve `HumanPlayer`'s
  re-export, and `HumanPlayer` was folded into the public surface in
  Phase 6 (with a row in `AGENTS.md` and a usage section). Memory
  saved: `feedback_check_existing_dir_before_scaffold` — grep for
  prior usage of a "new" package's import path before assuming the
  directory doesn't exist.

- **`RNaDAgent` rename followed through.** The plan's 5.99 escape
  hatch was not taken; `rl/players.py` is now deleted and `RNaDAgent`
  lives at `rl/rnad_model.py`.

- **Phase 3 picked up a stray Phase 2 oversight.** The Phase 2 import
  sweep (`sed` for `behavior_clone_player`) missed
  `src/elitefurretai/supervised/SUPERVISED.md:39`. The Phase 3 implementer
  noticed and fixed it in passing — bundled into commit `79cf2d0`
  rather than carving out a separate fixup commit.

**Smoke verification:**

Type-checking and unit tests are clean. An import-level smoke check
confirmed:

- All six public re-exports import cleanly from `elitefurretai.agents`.
- `train.py`, `worker.py`, `engine.vgc_environment` all import.
- `VGCBenchManager.SUBPROCESS_SCRIPT = "src/elitefurretai/agents/_vgcbench_subprocess.py"`
  resolves to an existing file.

A full 5-minute training smoke run with `single_team.yaml`
(which would also catch any subprocess-spawn regressions in the
vgc-bench runner path) was **not** executed in this session and is
flagged as a follow-up. Recommend running it before the next training
run with `vgc_bench_baseline` weight > 0.

**Follow-ups (out of scope for this work):**

1. Run the 5-min training smoke (above).
2. The FoulPlay integration ([2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md))
   now targets the post-reorg layout; can proceed when ready.
3. `OpponentPool.vgc_bench_baseline_opponents` is still an always-empty
   list (noted in the vgc-bench consolidation Updates section as a
   cosmetic cleanup). Unchanged by this reorg.
