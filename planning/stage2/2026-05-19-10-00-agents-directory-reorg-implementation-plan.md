# `agents/` Directory Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate user-facing battle-participant classes into a new top-level `src/elitefurretai/agents/` directory, finish splitting `rl/players.py` into focused files, and document the new layout in `agents/AGENTS.md`.

**Architecture:** Move-and-split refactor across ~30 import sites. Eight phases, each ending in a working tree (quality gates pass) and a commit. Whole-file moves use `git mv` to preserve history; partial-file splits use `git log --follow` + `git blame -C -C` for blame recovery. No new tests are written — the existing test suite is the safety net.

**Tech Stack:** Python 3.12, ruff, pyright, pytest. Project convention: `source ../venv/bin/activate` before any command.

**Reference spec:** [2026-05-19-09-30-agents-directory-reorg.md](2026-05-19-09-30-agents-directory-reorg.md)

---

## Pre-flight

- [ ] **Step 1: Confirm clean working tree for the relevant files**

Run:
```bash
git status -s src/elitefurretai/rl/players.py src/elitefurretai/supervised/behavior_clone_player.py src/elitefurretai/rl/_vgcbench_subprocess.py
```
Expected: empty output (no staged or unstaged changes in those files). Pre-existing modifications elsewhere (e.g. `data/`, `scripts/`) are fine; do not touch them.

- [ ] **Step 2: Activate venv and confirm baseline quality gates pass**

Run:
```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: all green. If anything is red here, stop — fix or revert before starting the reorg so we don't confuse pre-existing failures with regressions introduced by the move.

- [ ] **Step 3: Snapshot the import surface for the after-comparison**

Run:
```bash
grep -rn "from elitefurretai.rl.players\|from elitefurretai.supervised.behavior_clone_player\|elitefurretai\.rl\.players\." src unit_tests 2>/dev/null | grep -v __pycache__ > /tmp/agents_reorg_before_imports.txt
wc -l /tmp/agents_reorg_before_imports.txt
```
Expected: ~40 lines. Keep this file — phase 6 reruns the grep against the new paths and the count of *remaining* old-path imports must be 0.

---

## Phase 1 — Skeleton

**Files:**
- Create: `src/elitefurretai/agents/__init__.py`
- Create: `src/elitefurretai/agents/AGENTS.md` (stub)

- [ ] **Step 1.1: Create directory and empty __init__.py**

Run:
```bash
mkdir -p src/elitefurretai/agents
```

Create `src/elitefurretai/agents/__init__.py` with:

```python
"""EliteFurretAI agents — instantiable battle participants.

See AGENTS.md in this directory for what's here and how to use each.
"""
```

(No re-exports yet. Phase 6 fills them in.)

- [ ] **Step 1.2: Create stub AGENTS.md**

Create `src/elitefurretai/agents/AGENTS.md` with:

```markdown
# agents/

User-facing, instantiable battle participants — the things you grab to run a battle in EFA.

Contents and usage docs fill in during phase 6 of the reorg
(see planning/stage2/2026-05-19-10-00-agents-directory-reorg-implementation-plan.md).
```

- [ ] **Step 1.3: Run quality gates**

Run:
```bash
source ../venv/bin/activate && ruff check src/elitefurretai/agents && pyright src/elitefurretai/agents
```
Expected: green. `pytest` not needed yet — nothing imports from the new module.

- [ ] **Step 1.4: Commit**

```bash
git add src/elitefurretai/agents/__init__.py src/elitefurretai/agents/AGENTS.md
git commit -m "agents: scaffold src/elitefurretai/agents/ directory"
```

---

## Phase 2 — Move BCPlayer

`BCPlayer` is a single whole file in `supervised/`. Easiest piece to move first because `git mv` preserves history cleanly, and the import surface is small (~5 sites).

**Files:**
- Move: `src/elitefurretai/supervised/behavior_clone_player.py` → `src/elitefurretai/agents/bc_player.py`
- Modify: 5 import sites (enumerated below)
- Modify: `src/elitefurretai/supervised/__init__.py` (drop the re-export)

- [ ] **Step 2.1: Move the file with `git mv`**

Run:
```bash
git mv src/elitefurretai/supervised/behavior_clone_player.py src/elitefurretai/agents/bc_player.py
```

`git mv` records the rename so `git log --follow src/elitefurretai/agents/bc_player.py` shows the full history.

- [ ] **Step 2.2: Update import sites**

Five files need the import path updated. For each, replace:

```
from elitefurretai.supervised.behavior_clone_player import BCPlayer
```

with:

```
from elitefurretai.agents.bc_player import BCPlayer
```

Files to edit:
- `src/elitefurretai/supervised/analyze/behavior_clone_replay.py:37`
- `src/elitefurretai/supervised/analyze/behavior_clone_performance.py:8`
- `src/elitefurretai/supervised/__init__.py:22` — **delete this import line entirely**, not replace. The `__init__.py` also lists `BCPlayer` in `__all__` near line 45; delete that entry too.
- `unit_tests/supervised/test_behavior_clone_player.py:21`

One-shot `sed`:
```bash
grep -rl "from elitefurretai.supervised.behavior_clone_player import BCPlayer" src unit_tests | xargs sed -i 's|from elitefurretai.supervised.behavior_clone_player import BCPlayer|from elitefurretai.agents.bc_player import BCPlayer|g'
```

Then edit `src/elitefurretai/supervised/__init__.py` manually to remove both the now-redundant import (which the sed turned into a from-agents import — but `supervised/__init__.py` shouldn't re-export BCPlayer at all) and the `"BCPlayer"` entry in `__all__`.

- [ ] **Step 2.3: Verify all old paths are gone**

Run:
```bash
grep -rn "behavior_clone_player" src unit_tests 2>/dev/null | grep -v __pycache__
```
Expected: empty (or only `.md` doc references, which we update in phase 8).

- [ ] **Step 2.4: Run quality gates**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: all green.

- [ ] **Step 2.5: Commit**

```bash
git add -A
git commit -m "agents: move BCPlayer from supervised/ to agents/bc_player.py"
```

---

## Phase 3 — Move VGCBenchManager + subprocess + vgc-bench helpers

`VGCBenchManager` (and its three vgc-bench helpers consolidated 2026-05-18) plus the `_vgcbench_subprocess.py` entry script. Move the subprocess script first with `git mv` (whole file), then extract the manager and helpers from `rl/players.py` into a new file.

**Files:**
- Move: `src/elitefurretai/rl/_vgcbench_subprocess.py` → `src/elitefurretai/agents/_vgcbench_subprocess.py`
- Create: `src/elitefurretai/agents/vgcbench_manager.py` (extracted from `rl/players.py:1682-1939`)
- Modify: `src/elitefurretai/rl/players.py` (remove the extracted region)
- Modify: 6 import sites + the `SUBPROCESS_SCRIPT` ClassVar string

- [ ] **Step 3.1: Move the subprocess script**

```bash
git mv src/elitefurretai/rl/_vgcbench_subprocess.py src/elitefurretai/agents/_vgcbench_subprocess.py
```

- [ ] **Step 3.2: Update the docstring reference in the moved subprocess file**

In `src/elitefurretai/agents/_vgcbench_subprocess.py`, the module docstring currently says:

```
spawned automatically by
``elitefurretai.rl.players.VGCBenchManager.launch()``
```

Change to:

```
spawned automatically by
``elitefurretai.agents.vgcbench_manager.VGCBenchManager.launch()``
```

Also update the line referencing `OpponentPool.external_vgcbench_usernames` if it cites a file path; the class itself doesn't move, but the docstring's `rl/opponents.py` reference is still correct so leave that alone.

- [ ] **Step 3.3: Create `agents/vgcbench_manager.py` by copying from `rl/players.py`**

The extraction region in `rl/players.py` is lines **1682–1939** (verify with `grep -n "_VGC_BENCH_POLICY_CACHE\|^_temporary_cwd\|^def _resolve_vgc_bench_root\|^def _create_vgc_bench_player\|^class VGCBenchManager" src/elitefurretai/rl/players.py` — those five symbols are the boundaries). Copy that region into a new file `src/elitefurretai/agents/vgcbench_manager.py`, prepended with a module docstring and the imports it needs.

File header for `src/elitefurretai/agents/vgcbench_manager.py`:

```python
"""VGCBenchManager — subprocess lifecycle for the external vgc-bench bot.

vgc-bench requires poke_env 0.11.x while EFA runs poke_env 0.15.x, so it
runs as a subprocess in its own venv (``../venv-vgcbench/``). This module
owns the EFA-side subprocess lifecycle. The subprocess entry point lives
at ``agents/_vgcbench_subprocess.py``. EFA-side code interacts with
vgc-bench only by Showdown username.

Also exports three in-process helpers used by the eval CLI when running
under a venv whose poke_env vintage matches vgc-bench's
(``_create_vgc_bench_player``, ``_temporary_cwd``, ``_resolve_vgc_bench_root``).
These are NOT safe to call from EFA's training process; see the comment
above ``_create_vgc_bench_player``.

Moved here from ``rl/players.py`` on 2026-05-19 as part of the agents/
directory reorganization (see planning/stage2/2026-05-19-09-30-agents-directory-reorg.md).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TextIO, Tuple

# (other imports — copy exactly what the original lines 1682-1939 reference
# from the players.py import block. The class uses RNaDConfig from
# elitefurretai.rl.config; the helpers use Path, contextmanager, subprocess,
# os, sys. Match what `pyright` reports as undefined and add accordingly.)
```

After copying, the only `players.py` symbols the extracted block references are: `RNaDConfig`. The block does not reference `RLTrajectoryPlayer`, `RNaDModel`, etc.

- [ ] **Step 3.4: Update `SUBPROCESS_SCRIPT` path inside the moved class**

In `src/elitefurretai/agents/vgcbench_manager.py`, change:

```python
SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/rl/_vgcbench_subprocess.py"
```

to:

```python
SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/agents/_vgcbench_subprocess.py"
```

This is **critical** — the trainer launches `python <SUBPROCESS_SCRIPT>` via subprocess. Wrong path = silent fail at startup, only caught by the phase 8 smoke run.

- [ ] **Step 3.5: Delete the extracted region from `rl/players.py`**

Remove lines 1682–1939 inclusive (the `_VGC_BENCH_POLICY_CACHE`, `_temporary_cwd`, `_resolve_vgc_bench_root`, `_create_vgc_bench_player`, and `VGCBenchManager` definitions). Also remove the corresponding entries from the `__all__` list at the bottom — `"VGCBenchManager"` and any vgc-bench-helper entries.

After deletion, run:
```bash
grep -n "VGCBenchManager\|_create_vgc_bench_player\|_resolve_vgc_bench_root\|_temporary_cwd\|_VGC_BENCH_POLICY_CACHE" src/elitefurretai/rl/players.py
```
Expected: empty.

- [ ] **Step 3.6: Update import sites**

Six files import `VGCBenchManager` and/or `_create_vgc_bench_player` from `rl/players.py`:

| File | Symbol(s) | Action |
|---|---|---|
| `src/elitefurretai/rl/worker.py:78` | `VGCBenchManager` (currently with RNaDModel) | Split: keep RNaDModel import from rl.players, add VGCBenchManager import from agents.vgcbench_manager |
| `src/elitefurretai/rl/train.py:97` | `VGCBenchManager` (with RNaDModel, cleanup_worker_executors) | Same split |
| `src/elitefurretai/engine/vgc_environment.py:51` | `VGCBenchManager` (with MaxDamagePlayer, RNaDModel) | Same split |
| `src/elitefurretai/rl/analyze/player_factory.py:25` | `_create_vgc_bench_player` (with MaxDamagePlayer, SimpleModelPlayer) | Same split |
| (Tests referencing VGCBenchManager — grep to find any) | — | Same split |

Diff pattern for `rl/worker.py:78`:

```diff
-from elitefurretai.rl.players import RNaDModel, VGCBenchManager
+from elitefurretai.agents.vgcbench_manager import VGCBenchManager
+from elitefurretai.rl.players import RNaDModel
```

Pattern for the others is analogous. Imports MUST come from
`elitefurretai.agents.vgcbench_manager`, not
`elitefurretai.agents` (the package `__init__.py` re-exports are added in
phase 6; until then, only the per-file path works).

- [ ] **Step 3.7: Verify**

```bash
grep -rn "VGCBenchManager\|_create_vgc_bench_player" src unit_tests | grep -v __pycache__ | grep "elitefurretai.rl.players"
```
Expected: empty.

- [ ] **Step 3.8: Run quality gates**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: green.

- [ ] **Step 3.9: Commit**

```bash
git add -A
git commit -m "agents: move VGCBenchManager + vgc-bench helpers + subprocess script to agents/"
```

---

## Phase 4 — Move SimpleModelPlayer, VerboseModelPlayer, MaxDamagePlayer

Three eval/heuristic players, all currently in `rl/players.py`. Each goes to its own file.

**Files:**
- Create: `src/elitefurretai/agents/simple_model_player.py` (from `rl/players.py:998-1106`)
- Create: `src/elitefurretai/agents/verbose_model_player.py` (from `rl/players.py:1107-1191`)
- Create: `src/elitefurretai/agents/max_damage_player.py` (from `rl/players.py:1192-1681`)
- Modify: `src/elitefurretai/rl/players.py` (remove the extracted regions)
- Modify: ~10 import sites

- [ ] **Step 4.1: Create `simple_model_player.py`**

Copy `rl/players.py:998-1106` (the `SimpleModelPlayer` class). New file header:

```python
"""SimpleModelPlayer — eval-time Player wrapping a trained model.

Loads a checkpoint, runs inference inline in ``choose_move`` (no IPC, no
batching). Trades training-time throughput for setup simplicity — callers
don't need to spawn an InferenceService process or wire up
request/response queues. Use ``rl/rl_trajectory_player.py`` when you
need the trainer-side centralized inference pattern.

See agents/AGENTS.md for usage.
"""

from __future__ import annotations

# Add the imports SimpleModelPlayer actually uses. Reference the original
# import block in rl/players.py for the exhaustive list, then trim to
# what's actually referenced inside the class.
```

The class references (verify by grep within the class body):
- `Player`, `BattleOrder`, `DefaultBattleOrder`, `DoubleBattleOrder` from `poke_env.player`
- `AbstractBattle`, `DoubleBattle` from `poke_env.battle`
- `AccountConfiguration`, `ServerConfiguration` from `poke_env.ps_client`
- `torch`, `numpy as np`
- `Embedder` from `elitefurretai.etl`
- `MDBO` from `elitefurretai.etl.encoder`
- `RNaDModel` — **import from `elitefurretai.rl.players`** for now (phase 5 may rename to `elitefurretai.rl.rnad_model`)
- `fast_get_action_mask`, `get_valid_targets`, `slot_is_commanding` from `elitefurretai.rl.masking`
- `build_model_from_config` from `elitefurretai.rl.learners` (used in `_load_model`)

- [ ] **Step 4.2: Create `verbose_model_player.py`**

Copy `rl/players.py:1107-1191`. Header:

```python
"""VerboseModelPlayer — SimpleModelPlayer + per-turn top-k logging to stdout.

Use for ad-hoc inspection of model behavior during a battle. Subclasses
SimpleModelPlayer; overrides ``_on_action_selected`` to print the top-k
action probabilities at each decision point.

See agents/AGENTS.md for usage.
"""

from __future__ import annotations

from elitefurretai.agents.simple_model_player import SimpleModelPlayer
# (plus whatever else the class references — likely torch, np, MDBO)
```

- [ ] **Step 4.3: Create `max_damage_player.py`**

Copy `rl/players.py:1192-1681`. Header:

```python
"""MaxDamagePlayer — heuristic baseline that picks the highest-damage action.

Non-learning Player used as a curriculum opponent and as a baseline in the
Stage II graduation criterion. Wraps ``poke_env.calc.calculate_damage``.

See agents/AGENTS.md for usage.
"""

from __future__ import annotations
```

Imports the class actually references (verify):
- `Player`, `BattleOrder`, `DoubleBattleOrder`, `DefaultBattleOrder`, `PassBattleOrder`, `SingleBattleOrder` from `poke_env.player` / `poke_env.player.battle_order`
- `AbstractBattle`, `DoubleBattle`, `Pokemon` from `poke_env.battle`
- `calculate_damage` from `poke_env.calc`
- `POKE_LOOP` from `poke_env.concurrency`
- `GenData` from `poke_env.data`
- `compute_raw_stats` from `poke_env.stats`
- `MDBO` from `elitefurretai.etl.encoder`
- `battle_to_str` from `elitefurretai.inference.inference_utils`
- `fast_get_action_mask`, `get_valid_targets`, `slot_is_commanding` from `elitefurretai.rl.masking`
- `np`, `random`, `math`, `re`, `logging`, asyncio bits — verify by grep

- [ ] **Step 4.4: Delete extracted regions from `rl/players.py`**

Remove lines 998–1681 inclusive. Update the `__all__` at the bottom to drop:
- `"SimpleModelPlayer"`
- `"VerboseModelPlayer"`
- `"MaxDamagePlayer"`

After deletion, `rl/players.py` should contain only: imports, `_request_fingerprint`, `_WORKER_EXECUTORS`/`_EXECUTOR_LOCK`/`_FALLBACK_EXECUTOR`, `get_worker_executor`, `cleanup_worker_executors`, `RLTrajectoryPlayer`, `RNaDModel`, and an `__all__` listing just `RNaDModel`, `RLTrajectoryPlayer`, `cleanup_worker_executors`.

Run:
```bash
grep -n "^class \|^def " src/elitefurretai/rl/players.py
```
Expected:
```
115:def _request_fingerprint
217:def get_worker_executor
231:def cleanup_worker_executors
239:class RLTrajectoryPlayer
~966:class RNaDModel
```
(Line numbers will have shifted; the symbols are what matter.)

- [ ] **Step 4.5: Update import sites**

Files importing the three moved classes:

| File:line | Old import | New import |
|---|---|---|
| `src/elitefurretai/rl/opponents.py:78` | `from elitefurretai.rl.players import RLTrajectoryPlayer, MaxDamagePlayer, RNaDModel` | Three lines: RLTrajectoryPlayer + RNaDModel from rl.players; MaxDamagePlayer from agents.max_damage_player |
| `src/elitefurretai/rl/analyze/play_human_vs_model.py:30` | `from elitefurretai.rl.players import VerboseModelPlayer` | `from elitefurretai.agents.verbose_model_player import VerboseModelPlayer` |
| `src/elitefurretai/rl/analyze/player_factory.py:25` | `from elitefurretai.rl.players import (MaxDamagePlayer, SimpleModelPlayer, _create_vgc_bench_player)` | `_create_vgc_bench_player` already moved to vgcbench_manager in phase 3; split into: MaxDamagePlayer from agents.max_damage_player, SimpleModelPlayer from agents.simple_model_player |
| `src/elitefurretai/engine/vgc_environment.py:51` | `from elitefurretai.rl.players import MaxDamagePlayer, RNaDModel, VGCBenchManager` | Already-split-once in phase 3; further split MaxDamagePlayer to agents.max_damage_player |
| `src/elitefurretai/engine/analyze/showdown_benchmark.py:20` | `from elitefurretai.rl.players import SimpleModelPlayer` | `from elitefurretai.agents.simple_model_player import SimpleModelPlayer` |
| `src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py:59` | `from elitefurretai.rl.players import SimpleModelPlayer` | Same |
| `unit_tests/rl/test_players.py:9` | `from elitefurretai.rl.players import RLTrajectoryPlayer, MaxDamagePlayer` | Split: RLTrajectoryPlayer from rl.players; MaxDamagePlayer from agents.max_damage_player |
| `unit_tests/rl/test_players.py:61` | `patch("elitefurretai.rl.players.calculate_damage", ...)` | `patch("elitefurretai.agents.max_damage_player.calculate_damage", ...)` |
| `unit_tests/rl/test_play_human_vs_model.py:76` | `from elitefurretai.rl.players import SimpleModelPlayer, VerboseModelPlayer` | Two imports from the new homes |

- [ ] **Step 4.6: Verify mock-path strings are updated**

```bash
grep -rn 'patch.*"elitefurretai\.rl\.players' unit_tests
grep -rn 'patch.*"elitefurretai\.supervised\.behavior_clone_player' unit_tests
```
Expected: both empty.

- [ ] **Step 4.7: Run quality gates**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: green.

- [ ] **Step 4.8: Commit**

```bash
git add -A
git commit -m "agents: move SimpleModelPlayer, VerboseModelPlayer, MaxDamagePlayer to agents/"
```

---

## Phase 5 — Split `rl/players.py` into `rl_trajectory_player.py` + `rnad_model.py`

After phase 4, `rl/players.py` contains two unrelated concerns: the training-time batcher and the model wrapper. Split them. (The `RNaDModel` rename is the judgment-call piece per the spec — if you want to skip it, see step 5.99 at the end of this phase.)

**Files:**
- Create: `src/elitefurretai/rl/rl_trajectory_player.py` (from remaining `rl/players.py` — `_request_fingerprint`, executor helpers, `RLTrajectoryPlayer`)
- Create: `src/elitefurretai/rl/rnad_model.py` (from remaining `rl/players.py` — `RNaDModel`)
- Delete: `src/elitefurretai/rl/players.py`
- Modify: ~15 RNaDModel import sites + 4 RLTrajectoryPlayer import sites + 1 cleanup_worker_executors import site

- [ ] **Step 5.1: Create `rl_trajectory_player.py`**

Copy these symbols from `rl/players.py`:
- imports it actually uses (verify by pyright after copying)
- `_request_fingerprint`
- `_WORKER_EXECUTORS`, `_EXECUTOR_LOCK`, `_FALLBACK_EXECUTOR` module-level state
- `get_worker_executor`
- `cleanup_worker_executors`
- `RLTrajectoryPlayer` class

Header:

```python
"""RLTrajectoryPlayer — high-throughput async player used during RL training.

Gathers per-turn decisions from many concurrent battles, batches them into
one model forward pass via the inference IPC layer, and pushes finished
trajectories to a queue for the learner. Coupled to the trajectory queue,
the InferenceClient (``elitefurretai.rl.inference_worker``), and the
worker process orchestration — not a user-facing agent.

For user-facing eval/analysis players, see ``elitefurretai.agents``.

Moved here from ``rl/players.py`` on 2026-05-19.
"""
```

The `if TYPE_CHECKING: from elitefurretai.rl.inference_worker import InferenceClient` block from the original players.py imports comes with this class.

- [ ] **Step 5.2: Create `rnad_model.py`**

Copy `RNaDModel` (~lines 966-997 of the post-phase-4 players.py). Header:

```python
"""RNaDModel — torch.nn.Module wrapper around TransformerThreeHeadedModel.

Despite the historical name, this is a *model* wrapper, not a poke-env
Player. Used by the inference subprocess, the learner, the trainer, and
the model registry to present a uniform ``forward(x, hidden_state)`` API
over the underlying transformer architecture.

Moved here from ``rl/players.py`` on 2026-05-19 (file previously named
``players.py`` for historical reasons; renamed for clarity).
"""

from __future__ import annotations

import torch

from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel


class RNaDModel(torch.nn.Module):
    # ... copy class body unchanged from rl/players.py ...
```

- [ ] **Step 5.3: Delete `rl/players.py`**

```bash
git rm src/elitefurretai/rl/players.py
```

- [ ] **Step 5.4: Update `RNaDModel` import sites**

Fifteen files import `RNaDModel` from `elitefurretai.rl.players`. Sweep with sed:

```bash
grep -rl "from elitefurretai.rl.players import.*RNaDModel" src unit_tests | xargs sed -i 's|from elitefurretai\.rl\.players import RNaDModel|from elitefurretai.rl.rnad_model import RNaDModel|g'
```

Some sites import `RNaDModel` together with other symbols (e.g. `from elitefurretai.rl.players import RLTrajectoryPlayer, RNaDModel`). After phase 4 those have already been narrowed. The remaining combined-import sites after phase 4 are:
- `src/elitefurretai/rl/opponents.py:78` — `RLTrajectoryPlayer, RNaDModel`
- `src/elitefurretai/rl/train.py:97` — `RNaDModel, cleanup_worker_executors`
- `unit_tests/rl/test_worker_opponent_factory.py:9` — `RLTrajectoryPlayer, RNaDModel`
- `unit_tests/rl/test_worker.py:25` — `RLTrajectoryPlayer, RNaDModel`

For these four, do not use the sed above; edit manually to split into separate imports from `rl.rl_trajectory_player` and `rl.rnad_model`.

- [ ] **Step 5.5: Update `RLTrajectoryPlayer` import sites**

```bash
grep -rl "from elitefurretai\.rl\.players import.*RLTrajectoryPlayer" src unit_tests
```

Expected after phase 4: four files (the ones listed above). Edit each manually to use `from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer`.

Also: `unit_tests/rl/test_worker_opponent_factory.py:179` patches the mock at `"elitefurretai.rl.opponents.RLTrajectoryPlayer"` — that string does NOT change (it patches the name as bound inside `opponents.py`, not the original definition site).

- [ ] **Step 5.6: Update `cleanup_worker_executors` import site**

One site: `src/elitefurretai/rl/train.py:97`. Change `cleanup_worker_executors` import from `rl.players` to `rl.rl_trajectory_player`.

- [ ] **Step 5.7: Verify all old paths are gone**

```bash
grep -rn "elitefurretai\.rl\.players" src unit_tests 2>/dev/null | grep -v __pycache__
```
Expected: empty (or only doc/comment references in `.md` files — handled in phase 7).

- [ ] **Step 5.8: Run quality gates**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: green.

- [ ] **Step 5.9: Commit**

```bash
git add -A
git commit -m "rl: split players.py into rl_trajectory_player.py and rnad_model.py"
```

**Step 5.99 — escape hatch:** If the `RNaDModel` rename feels like scope creep at this point, do this instead:
- Keep `rl/players.py` (don't delete in step 5.3) — file contains only `RNaDModel` after step 5.1.
- Skip step 5.4's sed (keep `from elitefurretai.rl.players import RNaDModel` everywhere).
- Drop step 5.2 (don't create `rnad_model.py`).
- All other steps in this phase stand. Commit message: `"rl: extract RLTrajectoryPlayer to its own file"`.

---

## Phase 6 — `__init__.py` re-exports and `AGENTS.md` content

Fill in the public surface and the human-readable index.

**Files:**
- Modify: `src/elitefurretai/agents/__init__.py`
- Modify: `src/elitefurretai/rl/__init__.py`
- Replace: `src/elitefurretai/agents/AGENTS.md` (full content)

- [ ] **Step 6.1: Populate `agents/__init__.py`**

Replace the phase-1 stub with:

```python
"""EliteFurretAI agents — user-facing, instantiable battle participants.

See AGENTS.md in this directory for what's here and how to use each.

Re-exports the public classes so ``from elitefurretai.agents import X``
works, but per-file imports (``from elitefurretai.agents.bc_player
import BCPlayer``) are also fine and slightly faster at import time
because they skip loading sibling modules.
"""

from elitefurretai.agents.bc_player import BCPlayer
from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.agents.verbose_model_player import VerboseModelPlayer

__all__ = [
    "BCPlayer",
    "MaxDamagePlayer",
    "SimpleModelPlayer",
    "VerboseModelPlayer",
    "VGCBenchManager",
]
```

Note: `FoulPlayManager` is NOT listed — it lands in a follow-up per the FoulPlay plan, and that plan now points at `agents/foulplay_manager.py`.

- [ ] **Step 6.2: Update `rl/__init__.py`**

Current re-exports from `rl.players`:

```python
from elitefurretai.rl.players import (
    RLTrajectoryPlayer,
    MaxDamagePlayer,
    RNaDModel,
    cleanup_worker_executors,
)
```

Replace with:

```python
from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.rl.rl_trajectory_player import (
    RLTrajectoryPlayer,
    cleanup_worker_executors,
)
from elitefurretai.rl.rnad_model import RNaDModel
```

(If you took the phase 5.99 escape hatch, `RNaDModel` keeps importing from `rl.players`.)

The `__all__` list stays the same.

- [ ] **Step 6.3: Write `AGENTS.md` content**

Replace the phase-1 stub at `src/elitefurretai/agents/AGENTS.md` with:

````markdown
# `agents/` — EliteFurretAI Battle Participants

This directory holds **user-facing, instantiable agents** — the things you grab to run a battle in EFA. Eval/analysis players, heuristic baselines, the behavior-cloned player, and the subprocess managers that wrap external bots (vgc-bench, foul-play).

It is **not** for classes that subclass `poke_env.player.Player` for training-plumbing reasons. `RLTrajectoryPlayer` is a `Player` subclass, but it lives in `rl/rl_trajectory_player.py` because its job is dynamic batching for RL training throughput — coupled to the trajectory queue and the inference IPC layer. You'd never grab it to run an ad-hoc battle.

## What's in here

| File | Class | Role |
|---|---|---|
| `simple_model_player.py` | `SimpleModelPlayer` | Eval-time Player loading a trained checkpoint and running inference inline in `choose_move`. The default choice for evaluation, analysis, and head-to-head matchups. |
| `verbose_model_player.py` | `VerboseModelPlayer` | `SimpleModelPlayer` + top-k action-probability logging to stdout per turn. Use for ad-hoc model-behavior inspection. |
| `max_damage_player.py` | `MaxDamagePlayer` | Heuristic baseline: picks the highest-damage action via `poke_env.calc.calculate_damage`. One of the four Stage II graduation baselines. |
| `bc_player.py` | `BCPlayer` | Player loading a supervised (behavior-cloned) checkpoint. Maintains an across-turn trajectory tensor; used as a curriculum opponent and as one of the four Stage II graduation baselines. |
| `vgcbench_manager.py` | `VGCBenchManager` | Launches and supervises the external vgc-bench subprocess. EFA challenges by Showdown username. Also exports `_create_vgc_bench_player`, `_temporary_cwd`, `_resolve_vgc_bench_root` for in-process construction under a venv whose poke_env vintage matches vgc-bench. |
| `foulplay_manager.py` *(landing later)* | `FoulPlayManager` | Same shape as `VGCBenchManager`, for the foul-play-doubles search bot. See `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md`. |
| `_vgcbench_subprocess.py` | — (script) | Subprocess entry point launched by `VGCBenchManager` under `../venv-vgcbench/bin/python`. Not user-invokable; the leading underscore is the signal. |
| `_foulplay_subprocess.py` *(landing later)* | — (script) | Same shape, for foul-play. |

## How to use each agent

### `SimpleModelPlayer`

```python
from elitefurretai.agents import SimpleModelPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

player = SimpleModelPlayer(
    model_path="data/models/supervised/cool-bee-85-finetune_best.pt",
    device="cuda:0",
    battle_format="gen9vgc2024regg",
    probabilistic=False,                              # argmax; True samples
    account_configuration=AccountConfiguration(...),
    server_configuration=ServerConfiguration(...),
    team=team_string,
    accept_open_team_sheet=False,
)
```

Gotchas:
- Loads the model on construction — instantiation cost is not trivial. Reuse the instance across battles when possible.
- `probabilistic=False` is the default for eval (lowest variance). Pass `True` for stochastic play.

### `VerboseModelPlayer`

Same construction as `SimpleModelPlayer`. Prints top-5 action probabilities to stdout each turn; use only in interactive contexts.

### `MaxDamagePlayer`

```python
from elitefurretai.agents import MaxDamagePlayer

player = MaxDamagePlayer(
    battle_format="gen9vgc2024regg",
    account_configuration=...,
    server_configuration=...,
    team=team_string,
    accept_open_team_sheet=False,
)
```

No model. Cheap to instantiate. Picks the action with highest estimated damage via `poke_env.calc.calculate_damage`; falls through to a default ordering when no damaging move is available (e.g. forced switches).

### `BCPlayer`

```python
from elitefurretai.agents import BCPlayer

player = BCPlayer(
    model_filepath="data/models/supervised/curious-darkness-77_best.pt",
    battle_format="gen9vgc2024regg",
    probabilistic=True,                               # sample from softmax
    device="cuda:0",
    verbose=False,
    accept_open_team_sheet=False,
    account_configuration=...,
    server_configuration=...,
    team=team_string,
)
```

Gotchas:
- Maintains an across-turn trajectory tensor per battle. `reset_battles()` clears it.
- The model checkpoint must be the new format (config + state_dict). Old state-dict-only checkpoints raise `ValueError` — migrate via `scripts/prepare/migrate_model_configs.py`.

### `VGCBenchManager`

Subprocess lifecycle wrapper. EFA-side code never imports vgc-bench; it `/challenge`s the subprocess by Showdown username.

```python
from elitefurretai.agents import VGCBenchManager

manager = VGCBenchManager(config, server_ports=[8000])
usernames = manager.launch()   # subprocess up; returns list of usernames
# ... run training/eval, workers challenge usernames[0] ...
manager.shutdown()             # SIGTERM, close log files
```

Gotchas:
- Requires `../venv-vgcbench/` to exist with vgc-bench installed (`config.curriculum.external_vgcbench_python_executable`).
- Subprocess logs go to `data/logs/vgcbench_runners/runner_*.log` — first place to check if challenges aren't being accepted.
- `WAIT_FOR_SERVER_TIMEOUT_S=180.0` and `STARTUP_WAIT_S=10.0` are the timing parameters; the trainer waits up to `STARTUP_WAIT_S` for the subprocess to log in before issuing the first challenge.

## How to add a new agent

1. Create `src/elitefurretai/agents/<name>.py` with the class.
2. Add a re-export to `src/elitefurretai/agents/__init__.py` and the `__all__` list.
3. Add a row to the "What's in here" table above and a usage section.
4. If the class is a Player subclass usable from the eval CLI, also update `src/elitefurretai/rl/analyze/player_factory.py` to register it.

## What does *not* belong here

- Training-time plumbing that happens to subclass `Player` (`RLTrajectoryPlayer`). Lives in `rl/`.
- Model wrappers (`RNaDModel`). They're `torch.nn.Module`s, not Players. Live in `rl/rnad_model.py`.
- Opponent-sampling / curriculum orchestration (`OpponentPool`, `WorkerOpponentFactory`). They consume agents but aren't ones. Live in `rl/opponents.py`.
- CLI scaffolding for the eval entry point (`player_factory.py`, `team_provider.py`). Live in `rl/analyze/`.
````

- [ ] **Step 6.4: Run quality gates**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: green. The `__init__.py` re-exports may trip ruff's unused-import rule if `__all__` isn't picked up — if so, add `# noqa: F401` to each re-export line.

- [ ] **Step 6.5: Commit**

```bash
git add -A
git commit -m "agents: re-exports in __init__.py and full AGENTS.md content"
```

---

## Phase 7 — Update FoulPlay plan + RL.md + SUPERVISED.md

Bring downstream docs in line with the new layout.

**Files:**
- Modify: `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md`
- Modify: `planning/stage2/2026-05-18-17-00-foulplay-eval-implementation-plan.md` (if it references the old path)
- Modify: `src/elitefurretai/rl/RL.md`
- Modify: `src/elitefurretai/supervised/SUPERVISED.md`

- [ ] **Step 7.1: Update FoulPlay integration doc**

In `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md`:

- Replace `src/elitefurretai/rl/players.py` references (when talking about `FoulPlayManager` location) → `src/elitefurretai/agents/foulplay_manager.py`.
- Replace `SUBPROCESS_SCRIPT` ClassVar value `"src/elitefurretai/rl/_foulplay_subprocess.py"` → `"src/elitefurretai/agents/_foulplay_subprocess.py"`.
- Replace the actual subprocess file path `src/elitefurretai/rl/_foulplay_subprocess.py` → `src/elitefurretai/agents/_foulplay_subprocess.py`.
- Add a note in the Context section: "Targets the post-2026-05-19 agents/ layout (see [2026-05-19-09-30-agents-directory-reorg.md])."

- [ ] **Step 7.2: Update FoulPlay implementation plan doc if it exists**

```bash
grep -n "rl/players\.py\|rl/_foulplay_subprocess\.py\|elitefurretai\.rl\.players" planning/stage2/2026-05-18-17-00-foulplay-eval-implementation-plan.md
```

For each match, apply the same path updates as in step 7.1.

- [ ] **Step 7.3: Update `RL.md`**

```bash
grep -n "rl/players\.py\|rl\.players\|RLTrajectoryPlayer\|MaxDamagePlayer\|SimpleModelPlayer\|VerboseModelPlayer\|VGCBenchManager\|_vgcbench_subprocess\|RNaDModel" src/elitefurretai/rl/RL.md
```

For each match, decide whether the doc text should:
- Refer to the new file path (`agents/...` or `rl/rl_trajectory_player.py`, `rl/rnad_model.py`).
- Remain unchanged because it's describing a concept that doesn't depend on file location.

Apply minimal updates — don't rewrite RL.md.

- [ ] **Step 7.4: Update `SUPERVISED.md`**

```bash
grep -n "behavior_clone_player\|BCPlayer" src/elitefurretai/supervised/SUPERVISED.md
```

The doc's import example at line 39 (`from elitefurretai.supervised.behavior_clone_player import BCPlayer`) needs to become `from elitefurretai.agents.bc_player import BCPlayer`. Also update the module-list near the top of the doc that mentions `behavior_clone_player` if such a list exists.

- [ ] **Step 7.5: Run quality gates one more time**

```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: green. (Docs don't affect gates, but rerun to confirm nothing regressed during the doc edits.)

- [ ] **Step 7.6: Commit**

```bash
git add -A
git commit -m "docs: update FoulPlay plan, RL.md, SUPERVISED.md for agents/ layout"
```

---

## Phase 8 — Smoke training run

Type-checking does not catch wrong `SUBPROCESS_SCRIPT` paths or wrong `importlib`-based dynamic imports. Only an actual run does.

- [ ] **Step 8.1: Re-snapshot import surface and confirm full sweep**

```bash
grep -rn "from elitefurretai\.rl\.players\|from elitefurretai\.supervised\.behavior_clone_player" src unit_tests 2>/dev/null | grep -v __pycache__
```
Expected: empty (unless you took the 5.99 escape hatch, in which case `RNaDModel` imports from `rl.players` are expected and fine).

- [ ] **Step 8.2: Launch a 5-minute training run**

```bash
source ../venv/bin/activate && timeout 300 python src/elitefurretai/rl/train.py --config src/elitefurretai/rl/configs/single_team.yaml
```

The run should:
1. Launch Showdown server(s).
2. Launch the vgc-bench subprocess.
3. Workers start; trajectories flow; learner updates fire.
4. Exit cleanly on `timeout 300` (or after the configured first-checkpoint cadence).

- [ ] **Step 8.3: Verify the vgc-bench subprocess actually started**

```bash
ls -lt data/logs/vgcbench_runners/ | head -3
tail -50 data/logs/vgcbench_runners/runner_VGCBENCH_*.log | head -50
```

Expected: a log file dated within the last 5 minutes showing a successful login to Showdown (look for `'updateuser'` or similar handshake). If the log is empty or the file doesn't exist, the `SUBPROCESS_SCRIPT` path is wrong — fix step 3.4 and rerun.

- [ ] **Step 8.4: Verify there are no `ImportError`s in worker logs**

```bash
ls -lt data/logs/ 2>/dev/null | head
grep -rn "ImportError\|ModuleNotFoundError" data/logs/ 2>/dev/null | head -20
```

Expected: empty for any logs created during this run.

- [ ] **Step 8.5: Mark the spec complete**

Append to `planning/stage2/2026-05-19-09-30-agents-directory-reorg.md` under the `## Updates` section:

```markdown
### 2026-05-19 — implementation complete

Landed across eight commits per
planning/stage2/2026-05-19-10-00-agents-directory-reorg-implementation-plan.md.
All quality gates green; 5-min smoke training run on single_team.yaml
confirmed vgc-bench subprocess launches and accepts challenges. FoulPlay
integration plans updated to target the new layout.

[Whatever else turned out to be relevant — escape hatch taken? RNaDModel
rename skipped? Any unexpected issues? Document them here.]
```

- [ ] **Step 8.6: Final commit**

```bash
git add planning/stage2/2026-05-19-09-30-agents-directory-reorg.md
git commit -m "planning: mark agents/ reorg spec complete"
```

---

## Self-review checklist (writer ran this; engineer doesn't need to)

- [x] Spec coverage: every Components, Migration mechanics, Out-of-scope, Risks bullet in the spec maps to a phase or an explicit non-action.
- [x] Placeholder scan: no TBDs, no "implement later", no "add appropriate error handling", every code block is concrete.
- [x] Type consistency: `SUBPROCESS_SCRIPT` path is "agents/_vgcbench_subprocess.py" in both step 3.4 and AGENTS.md; symbol names match across phases.
- [x] Escape hatch for the `RNaDModel` rename is in phase 5.99 and is referenced in phase 6.2.

## Out of scope (per spec)

- Touching `OpponentPool` / `WorkerOpponentFactory` or `player_factory.py` / `team_provider.py`.
- Reducing `RLTrajectoryPlayer`'s line count.
- Adding an `elitefurretai.agents` abstract base / registry / factory.
- Backwards-compat shims from `rl.players` or `supervised.behavior_clone_player`.
- `RLTrajectoryPlayer` in `AGENTS.md`.
