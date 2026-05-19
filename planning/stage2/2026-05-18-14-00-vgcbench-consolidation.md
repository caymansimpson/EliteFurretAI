# VGCBench Construction Consolidation

**Date**: 2026-05-18
**Status**: Pre-implementation. Design agreed; ready for plan writing.

Collapse the scattered "external vgc-bench runner" plumbing into a single
`VGCBenchManager` in `players.py`, delete the dead in-process construction
path, and rename/relocate the subprocess entry-point script so its role is
unambiguous.

---

## Context

vgc-bench is a third-party SB3-trained baseline EFA challenges as part of
the Stage II graduation criterion (≥60% win rate vs `vgc_bench_baseline`
and three other baselines simultaneously). It is incompatible with EFA's
training venv: vgc-bench was written against `poke_env 0.11.0`, while
EFA's training venv runs `poke_env 0.15.0`. The API gap between those
versions is large enough that vgc-bench's `PolicyPlayer` cannot run
correctly under EFA's poke_env.

The solution shipped historically is **process-level isolation**:
vgc-bench lives in its own venv (`../venv-vgcbench/`) and is launched as
a subprocess by EFA at training startup. The subprocess logs into the
Showdown server under a known username (`VGCBENCH`, optionally
port-suffixed). EFA workers don't import vgc-bench at all; they simply
`/challenge` that username when the curriculum samples `vgc_bench_baseline`.
The architectural reasoning lives at [RL.md:197][rl-md-vgcbench].

[rl-md-vgcbench]: ../../src/elitefurretai/rl/RL.md#L197

## Before state

The current implementation spreads vgc-bench-related code across at least
six files in three subsystems:

- **Subprocess entry point**:
  `src/elitefurretai/rl/analyze/vgcbench_external_runner.py` (~130 lines,
  standalone CLI; runs under `venv-vgcbench`'s interpreter).
- **EFA-side launcher**:
  `src/elitefurretai/engine/showdown_server_manager.py` contains
  `_VGCBENCH_*` constants, `derive_external_vgcbench_username`,
  `launch_external_vgcbench_runners`, `shutdown_external_vgcbench_runners`,
  and `EXTERNAL_VGCBENCH_USERNAMES`.
- **EFA-side proxy plumbing**:
  - `rl/opponents.py:719,795,1086,1250-1252` —
    `OpponentPool.external_vgcbench_usernames` ctor arg, the
    username-routing branch.
  - `rl/analyze/evaluate.py:154-187,345-370,439,488` — same routing
    duplicated for graduation eval.
  - `rl/worker.py:176-177,380-381,491` — startup wait + per-worker
    routing.
  - `engine/vgc_environment.py:636-661,688` — per-port username
    derivation.
- **Dead in-process construction**:
  `rl/opponents.py:122 _create_vgc_bench_player` plus
  `_VGC_BENCH_POLICY_CACHE` at `rl/opponents.py:100`. This is the
  fallback path that runs vgc-bench *inside* EFA's process, gated by
  `not external_vgcbench_usernames` at `rl/opponents.py:1086`. It is
  structurally broken under the poke_env 0.15/0.11 skew (instantiates
  vgc-bench's `PolicyPlayer`, which subclasses `poke_env.player.Player`
  v0.11, against the v0.15 base class actually imported), but the path
  remains in code because no current config exercises it.
- **Re-exports**:
  `engine/__init__.py:14-33` re-exports the three launcher functions.
- **Call sites**:
  `rl/train.py:1139,2100` launches/shuts down at training boundaries.

In addition to the spread, the subprocess entry point lives at
`rl/analyze/vgcbench_external_runner.py` — under a directory whose
sibling files (`evaluate.py`, `play_model.py`, `play_human_vs_model.py`)
are all user-invoked analysis CLIs. The runner isn't a user-facing
analysis tool; it's a training-time component spawned automatically.

## Problem

Two problems, both about structural clarity rather than runtime
correctness:

1. **Discoverability.** A new contributor (or future-Cayman after a
   month off) reading `train.py` follows the call chain
   `launch_external_vgcbench_runners` → `showdown_server_manager.py` →
   constants and helpers scattered across that file →
   `rl/analyze/vgcbench_external_runner.py`. There is no single
   "what is vgc-bench in EFA" entry point.
2. **Misleading file location.** `rl/analyze/vgcbench_external_runner.py`
   reads as a peer of `play_model.py` (an analysis tool) when it is
   actually an internal subprocess entry point invoked automatically by
   the trainer. Following the example of `evaluate.py` etc., one would
   reasonably try to invoke the runner directly — it works in isolation
   if pointed at a Showdown server, but that is not its purpose.

Bonus problem: the dead in-process branch in `opponents.py` is a future
trap. If someone unsets `external_vgcbench_usernames` thinking it's a
toggle, the trainer silently falls into the broken path.

## Solution

### Components

**`VGCBenchManager` in `src/elitefurretai/rl/players.py`** (new class).
Owns the EFA-side lifecycle of the external vgc-bench subprocess(es).

```python
class VGCBenchManager:
    """Launcher and proxy for external vgc-bench bots.

    vgc-bench requires poke_env 0.11.x; EFA runs poke_env 0.15.x. To
    isolate the version gap, vgc-bench is launched as a subprocess in
    its own venv. This manager owns that subprocess lifecycle. EFA-side
    code only interacts with vgc-bench by Showdown username.
    """

    SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/rl/_vgcbench_subprocess.py"
    USERNAMES: ClassVar[List[str]] = ["VGCBENCH"]
    N_CHALLENGES: ClassVar[int] = 1_000_000
    WAIT_FOR_SERVER_TIMEOUT_S: ClassVar[float] = 180.0
    LOG_DIR: ClassVar[str] = "data/logs/vgcbench_runners"
    LOG_TO_FILES: ClassVar[bool] = True
    ACCEPT_OPEN_TEAM_SHEET: ClassVar[bool] = False
    STARTUP_WAIT_S: ClassVar[float] = 10.0

    # Only the first showdown server hosts a vgcbench runner — see
    # planning/stage2/2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md
    # for the memory rationale.
    RUNNER_SERVER_INDEX: ClassVar[int] = 0

    def __init__(self, config: RNaDConfig, server_ports: List[int]) -> None: ...

    @staticmethod
    def derive_username(base: str, server_port: int) -> str: ...

    def launch(self) -> List[str]:
        """Spawn subprocesses; return the list of usernames they will log in as."""

    def shutdown(self) -> None:
        """Terminate subprocesses and close log files."""

    @property
    def usernames(self) -> List[str]:
        """Usernames a worker should /challenge for vgc_bench_baseline battles."""
```

Internal state: `_processes: List[subprocess.Popen]`, `_log_files: List[TextIO]`,
`_usernames: List[str]`.

**`src/elitefurretai/rl/_vgcbench_subprocess.py`** (renamed and moved
from `rl/analyze/vgcbench_external_runner.py`; otherwise unchanged).
Leading underscore signals "internal entry point, not user-invocable."
Sits next to `players.py` so the relationship is obvious.

### Deletions

- `rl/opponents.py:100 _VGC_BENCH_POLICY_CACHE`
- `rl/opponents.py:122 _create_vgc_bench_player`
- `rl/opponents.py:1086` — the `not self.external_vgcbench_usernames`
  branch in `_batch_sample_opponents` that calls the deleted helper.
- The duplicated `_temporary_cwd` / `_resolve_vgc_bench_root` helpers in
  `rl/opponents.py:103-119` (only used by the deleted function).
- `engine/showdown_server_manager.py` loses its eight `_VGCBENCH_*`
  constants, `EXTERNAL_VGCBENCH_USERNAMES`,
  `EXTERNAL_VGCBENCH_STARTUP_WAIT_S`,
  `VGCBENCH_RUNNER_SERVER_INDEX`,
  `derive_external_vgcbench_username`,
  `launch_external_vgcbench_runners`, and
  `shutdown_external_vgcbench_runners`. The module returns to its
  pre-vgcbench scope of "Showdown server lifecycle and allocation."
- `engine/__init__.py:14-33` — drop the three re-exports.

### Call site updates

- `rl/train.py:79,81,1139,2100` — replace
  `launch_external_vgcbench_runners(...)` /
  `shutdown_external_vgcbench_runners(...)` with
  `manager = VGCBenchManager(config, server_ports); manager.launch()`
  and `manager.shutdown()`.
- `rl/opponents.py` — keep `external_vgcbench_usernames` ctor arg and
  the routing branch at `:1250-1252`, but pass it through from
  `train.py` as `manager.usernames` instead of being computed by
  `vgc_environment.py`.
- `engine/vgc_environment.py:636-661,688` — the per-port username
  derivation moves into `VGCBenchManager`. `vgc_environment.py` either
  receives the resolved list from above or constructs its own manager
  for environments that need it (TBD; see Reasoning).
- `rl/analyze/evaluate.py:154-187,345-370,439,488` — graduation eval
  also constructs a `VGCBenchManager`, gets `manager.usernames`, and
  passes them down.
- `rl/worker.py:176-177,380-381,491` —
  `EXTERNAL_VGCBENCH_USERNAMES` import target moves from
  `engine.showdown_server_manager` to
  `elitefurretai.rl.players.VGCBenchManager.USERNAMES` (or whatever
  shape the manager exposes).

### Config compatibility

No yaml changes. The two existing config fields
(`external_vgcbench_python_executable`, `external_vgcbench_team_file`,
under `curriculum:`) remain in `rl/config.py:531-532` and continue to
drive subprocess spawn. `VGCBenchManager` reads them off the passed-in
`RNaDConfig`. Validation in `rl/config.py:759-767` stays put.

## Reasoning

**Why not delete the subprocess script outright?** I checked: `import
vgc_bench` succeeds in EFA's venv, but the version skew check (poke_env
0.15 vs 0.11 — see Before state above) means in-process construction
silently produces a broken Player. Process isolation is genuinely
load-bearing; the standalone script must remain *somewhere*.

**Why a class instead of free functions?** The current three free
functions in `showdown_server_manager.py` are bound by shared state —
processes, log files, usernames, the port list — that callers thread
through arguments and return values. Wrapping that state in an instance
removes the awkward `(processes, log_files) = launch_*` tuple return
and the matching shutdown call that has to receive both.

**Why move the script next to `players.py` instead of into
`scripts/`?** `scripts/` is currently excluded from quality gates
(per CLAUDE.md "Linting exclusion"); we want this file linted and
pyright-checked since it's structurally load-bearing. Sitting next to
`players.py` with a leading-underscore name keeps it in the linted
tree while signaling "internal."

**Why not also remove `external_vgcbench_usernames` from `OpponentPool`?**
It's the only path now (the in-process branch is gone), but the
parameter remains the cleanest way for `OpponentPool` to know which
usernames to route by. Renaming it to `vgcbench_usernames` would be
clearer post-cleanup but is bikeshed-level — defer to the
implementation step.

**Why leave `vgc_environment.py`'s username derivation alone for now?**
That file's relationship to `OpponentPool` is non-trivial and the
multi-port branching at `vgc_environment.py:636-661` is load-bearing.
Cleaner to land the manager first, then look at whether
`vgc_environment.py` should call `VGCBenchManager.derive_username` or
keep its own copy. Flag this as a follow-up.

## Risks

1. **Test coverage.** I'm not aware of any existing test that exercises
   the full launch → challenge → shutdown loop. If one doesn't exist,
   the cleanup is effectively validated only by `train.py` actually
   running. Mitigation: do a smoke training run before merging
   (`single_team.yaml`, ~5 minutes, watch `data/logs/vgcbench_runners/`
   logs for successful login and battle acceptance).
2. **Hidden importers.** `EXTERNAL_VGCBENCH_USERNAMES` and the three
   helper functions are imported in non-obvious places (`worker.py`
   pulls them via the engine package). The cleanup must grep for every
   one. Already enumerated in the "Call site updates" section above
   from a search that hit ~15 import sites — re-run grep before the
   final delete.
3. **Backwards compat with old configs.** Per CLAUDE.md hard
   constraints, backwards compat isn't required, but flag: any saved
   config or external script that imports
   `engine.showdown_server_manager.EXTERNAL_VGCBENCH_USERNAMES` will
   break. Should be no such consumers outside the repo.

## Out of scope

- Replacing the Showdown-WebSocket protocol between EFA and vgc-bench
  with something else (e.g., a shared-memory IPC). The current setup is
  fine; this is a structural cleanup, not a re-architecture.
- Reducing vgc-bench memory footprint (~1.2 GB PSS per runner). Already
  addressed by single-runner-on-server-0 layout — see
  `planning/stage2/2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md`.
- Upgrading vgc-bench to poke_env 0.15 (would eliminate the venv split
  entirely but is a multi-day port of a third-party codebase; not on
  the Stage II critical path).

## Planned next steps

1. Confirm naming: `VGCBenchManager` (per this doc), file name
   `_vgcbench_subprocess.py`. Adjust if either feels wrong.
2. Write the implementation plan (separate doc, follow the
   subagent-driven-development pattern if breaking into steps).
3. Execute: introduce `VGCBenchManager`, update call sites, delete
   dead code, move + rename the subprocess script, run quality gates,
   smoke-test a short training run.
4. Update `RL.md` (the `vgc-bench external runner` section near
   line 197 + the manual-launch example near line 1082) to point at
   the new file path and the new manager class.
5. Mark this doc complete in the Updates section below.

## Updates

### 2026-05-18 — implementation complete

Landed as a single change. All quality gates green
(`ruff check`, `ruff format --check`, `pyright src`, `pyright unit_tests`,
`pytest unit_tests` — 527 passed).

Concrete diff:

- **New**: `VGCBenchManager` class in
  [src/elitefurretai/rl/players.py](../../src/elitefurretai/rl/players.py).
  Owns the subprocess lifecycle. Class constants (`USERNAMES`,
  `N_CHALLENGES`, `WAIT_FOR_SERVER_TIMEOUT_S`, `LOG_DIR`,
  `LOG_TO_FILES`, `ACCEPT_OPEN_TEAM_SHEET`, `STARTUP_WAIT_S`,
  `RUNNER_SERVER_INDEX`, `SUBPROCESS_SCRIPT`) replaced the
  module-level constants from
  [showdown_server_manager.py](../../src/elitefurretai/engine/showdown_server_manager.py).
  Static `derive_username()` plus `should_suffix_port()` give
  [vgc_environment.py](../../src/elitefurretai/engine/vgc_environment.py)
  a single source of truth for per-worker resolution — the
  launcher/worker drift bug from
  [2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md](2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md)
  can no longer recur.
- **Moved**: `_create_vgc_bench_player`,
  `_VGC_BENCH_POLICY_CACHE`, `_temporary_cwd`,
  `_resolve_vgc_bench_root` from `opponents.py` →
  `players.py`. **Plan revision**: I had planned to *delete*
  `_create_vgc_bench_player` as dead code under the poke_env
  version skew, but it turned out to be live —
  [analyze/player_factory.py:25](../../src/elitefurretai/rl/analyze/player_factory.py#L25)
  imports it for the eval pipeline (head-to-head matchups where
  in-process construction is convenient). Move-not-delete keeps eval
  working and still consolidates all vgc-bench construction into
  `players.py` as planned. Updated comment above the helpers
  documents that the in-process path is safe only from interpreters
  whose poke_env matches what vgc-bench expects (so
  `analyze/player_factory.py` is fine when called from a venv
  matching vgc-bench, but never from EFA's training process).
- **Renamed + relocated**:
  `src/elitefurretai/rl/analyze/vgcbench_external_runner.py` →
  `src/elitefurretai/rl/_vgcbench_subprocess.py`
  (`git mv` preserved history). Updated its module docstring to
  reflect the new role.
- **Deleted from opponents.py**: the in-process construction branch
  at the old `:1086` (the `not external_vgcbench_usernames`
  fallback). `vgc_bench_baseline_opponents` list remains as
  always-empty bookkeeping — left in place because the routing code
  guards on `if self.vgc_bench_baseline_opponents` so the dead list
  is harmless.
- **Deleted from showdown_server_manager.py**: all eight
  `_VGCBENCH_*` constants, `EXTERNAL_VGCBENCH_USERNAMES`,
  `EXTERNAL_VGCBENCH_STARTUP_WAIT_S`,
  `VGCBENCH_RUNNER_SERVER_INDEX`,
  `derive_external_vgcbench_username`,
  `launch_external_vgcbench_runners`, and
  `shutdown_external_vgcbench_runners`. The `RNaDConfig` import and
  the `TextIO`/`Union` typing imports went with them. The module is
  back to "Showdown server lifecycle and allocation."
- **engine/__init__.py**: dropped the three external-vgcbench
  re-exports.
- **train.py**: replaced
  `launch_external_vgcbench_runners` /
  `shutdown_external_vgcbench_runners` calls with
  `vgcbench_manager = VGCBenchManager(config, server_ports)` /
  `vgcbench_manager.launch()` /
  `vgcbench_manager.shutdown()`. Dropped the now-unused
  `external_runner_processes` / `external_runner_log_files`
  locals and the `TextIO` import.
- **worker.py**: replaced the two
  `from engine.showdown_server_manager import EXTERNAL_VGCBENCH_*`
  module-constant imports with
  `VGCBenchManager.USERNAMES` /
  `VGCBenchManager.STARTUP_WAIT_S` reads (imported from
  `rl.players`).
- **vgc_environment.py**: dropped the
  `EXTERNAL_VGCBENCH_USERNAMES` /
  `VGCBENCH_RUNNER_SERVER_INDEX` /
  `derive_external_vgcbench_username` imports;
  per-worker resolution now reads
  `VGCBenchManager.USERNAMES`,
  `VGCBenchManager.RUNNER_SERVER_INDEX`,
  `VGCBenchManager.should_suffix_port(...)`, and
  `VGCBenchManager.derive_username(...)`. Comments updated.
- **analyze/player_factory.py**: import path moved from
  `rl.opponents` to `rl.players` for
  `_create_vgc_bench_player`.
- **RL.md**: section near line 197 + manual-launch example near
  line 1082 updated to name the new file
  (`rl/_vgcbench_subprocess.py`), the new class
  (`VGCBenchManager`), and `VGCBenchManager.USERNAMES`. The manual
  launch example was reframed as a debugging/ad-hoc fallback;
  primary path is now automatic via
  `VGCBenchManager(config, server_ports).launch()`.

### Open follow-ups

1. **Smoke test required before next training run.** Quality gates
   exercise type / lint / unit-test correctness but no test
   end-to-end-exercises the subprocess launch + Showdown
   `/challenge` loop. Recommend a ~5-minute training run with
   `single_team.yaml` and `vgc_bench_baseline` weight > 0 to
   confirm: (a) runner subprocess logs in successfully,
   (b) workers can challenge it, (c) shutdown is clean. Check
   `data/logs/vgcbench_runners/runner_VGCBENCH_*.log` for the
   login → battle handshake.
2. **`vgc_bench_baseline_opponents` cleanup** (deferred).
   `OpponentPool.vgc_bench_baseline_opponents` is an always-empty
   list after the in-process branch deletion. The routing code at
   `prepare_batch_tasks` still guards on it. Could be removed
   along with the (already-empty) iteration sites at
   `:1326,:1353,:1449,:1490`. Did not delete since the change is
   purely cosmetic and would expand the diff into the routing
   loop.
3. **`play_model.py` working-tree deletion (out of scope).**
   `src/elitefurretai/rl/analyze/play_model.py` is shown as
   deleted in `git status` (and absent from disk). This deletion
   was already present in the working tree at session start —
   not done by this change. Mentioning for traceability: the
   user-facing CLI it provided (`--mode={challenge,ladder,vs-bot}`)
   is not currently reachable until this is committed or
   reverted. `VerboseModelPlayer` in `players.py` is unaffected.
