# FoulPlay Eval Integration

**Date**: 2026-05-18
**Status**: Pre-implementation. Design agreed; ready for plan writing.

Integrate the external `foul-play-doubles` search bot as a periodic
checkpoint-eval "ground truth" signal for the Stage II RL agent. Mirrors
the just-designed `VGCBenchManager` pattern: subprocess in a dedicated
venv, EFA never imports FoulPlay, all interaction by Showdown username.

---

## Context

EFA's Stage II training currently gauges agent strength against four
baselines (`vgc_bench_baseline`, `max_damage`, `bc_player`,
`simple_heuristic_baseline`) — see
[2026-05-16-21-30-stage2-graduation-criteria.md][grad-criteria]. All four
are NN-based or heuristic; none are search-based. A search bot is a
qualitatively different "ground truth": it exposes the kind of mistakes a
trained policy makes that a forward-search would never make (sacrificing
tempo, missing setup punishes, failing to read forced switches).

[`pmariglia/foul-play-doubles`][fp-repo] is a Python doubles bot powered
by [`pmariglia/poke-engine-doubles`][pe-repo], a Rust search engine. It
connects to Pokemon Showdown via websocket (`PSWebsocketClient`), runs
configurable-budget search (`search_time_ms`, `parallelism`) over the
Rust engine, and accepts challenges with `bot_mode=accept_challenge`.
Architecturally identical to the existing `vgc-bench` integration:
subprocess in its own env, EFA challenges by username.

[grad-criteria]: 2026-05-16-21-30-stage2-graduation-criteria.md
[fp-repo]: https://github.com/pmariglia/foul-play-doubles
[pe-repo]: https://github.com/pmariglia/poke-engine-doubles

## Before state

- No FoulPlay code or env in the repo. `requirements.txt` does not
  reference `poke-engine-doubles`.
- `train.py` does not call `analyze/evaluate.py` (or any eval routine)
  inline during training. Eval against the four baselines is invoked
  manually post-run via `evaluate.py`'s CLI.
- `analyze/evaluate.py` handles `maxdamage`, `maxbasepower`, `shp`, and
  `vgcbench` baselines. Per-baseline routing uses `_make_opponent` at
  `rl/analyze/evaluate.py:178`. `vgcbench` is the only baseline that
  relies on an external subprocess (via the legacy
  `EXTERNAL_VGCBENCH_USERNAMES` path).
- `VGCBenchManager` is being introduced concurrently (see
  [2026-05-18-14-00-vgcbench-consolidation.md][vgc-consolidation]).
  Collapses subprocess-lifecycle plumbing into one class in `players.py`
  and renames `rl/analyze/vgcbench_external_runner.py` to
  `rl/_vgcbench_subprocess.py` (leading underscore = internal entry).

[vgc-consolidation]: 2026-05-18-14-00-vgcbench-consolidation.md

## Problem

Two gaps, both about evaluation coverage:

1. **No search-bot benchmark.** All four current baselines are
   non-search. The Stage II agent could plateau at "beats all four ≥60%"
   while still losing decisively to even a moderately-tuned search bot.
   The graduation criterion is necessary but not sufficient evidence of
   strong play.
2. **No mid-training visibility.** Eval against the four baselines
   happens only at run end (manual `evaluate.py` invocation). The wandb
   curve during training shows training-side win rates against
   curriculum opponents — adequate for monitoring direction, but
   curriculum-side stats are confounded with the curriculum mix itself.
   A fixed, off-curriculum eval pass at checkpoint cadence gives a clean
   "is the model getting better?" signal independent of curriculum
   weights.

## Solution

### Components

Three new units, one config addition, one one-time setup task. All
opponent construction lives in `players.py`, matching the VGCBench
consolidation pattern.

**`FoulPlayManager` in `src/elitefurretai/rl/players.py`** (new class).
Owns the EFA-side lifecycle of the external FoulPlay subprocess. Direct
analog of `VGCBenchManager` from the consolidation doc.

```python
class FoulPlayManager:
    """Launcher and proxy for the external foul-play-doubles bot.

    foul-play-doubles requires `poke-engine-doubles` (a Rust extension)
    and a different poke_env vintage than EFA's training env. To isolate
    the environment, FoulPlay is launched as a subprocess in its own
    venv (`../venv-foulplay/`). EFA-side code only interacts with
    FoulPlay by Showdown username.

    Unlike VGCBenchManager (alive for the lifetime of training), this
    manager is short-lived: launched per checkpoint-eval pass and torn
    down after the eval completes. Re-launchable.
    """

    SUBPROCESS_SCRIPT:        ClassVar[str]  = "src/elitefurretai/rl/_foulplay_subprocess.py"
    USERNAMES:                ClassVar[List[str]] = ["FOULPLAY"]
    WAIT_FOR_SERVER_TIMEOUT_S: ClassVar[float] = 180.0
    LOG_DIR:                  ClassVar[str]  = "data/logs/foulplay_runners"
    LOG_TO_FILES:             ClassVar[bool] = True
    ACCEPT_OPEN_TEAM_SHEET:   ClassVar[bool] = False
    STARTUP_WAIT_S:           ClassVar[float] = 10.0
    RUNNER_SERVER_INDEX:      ClassVar[int]  = 0

    def __init__(self, config: FoulplayEvalConfig, server_ports: List[int]) -> None: ...

    @staticmethod
    def derive_username(base: str, server_port: int) -> str: ...   # same 18-char rule

    def launch(self) -> List[str]:
        """Spawn subprocess; return usernames the model should /challenge."""

    def shutdown(self) -> None:
        """SIGTERM the subprocess, close log files."""

    @property
    def usernames(self) -> List[str]:
        """Usernames the eval driver issues /challenge to."""
```

Internal state: `_processes: List[subprocess.Popen]`,
`_log_files: List[TextIO]`, `_usernames: List[str]`. Single-server
layout (only `server_ports[RUNNER_SERVER_INDEX]` hosts a runner) per
the memory rationale in
[2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md][mem-rationale].

[mem-rationale]: 2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md

**`src/elitefurretai/rl/_foulplay_subprocess.py`** (new file). Standalone
Python entry point, runs under `../venv-foulplay/bin/python`. Imports
FoulPlay's own modules (`config`, `data`, `fp`, `teams`) via
`cwd=../foul-play-doubles`. Configures `FoulPlayConfig` from CLI args,
logs into Showdown, accepts N challenges, plays each as a single battle
(bypassing FoulPlay's `run.py` Bo3 wrapper). CLI surface mirrors the
existing `_vgcbench_subprocess.py` (after the consolidation rename):
`--username`, `--server`, `--battle-format`, `--n-challenges`,
`--team-list-dir`, `--search-time-ms`, `--parallelism`,
`--wait-for-server-timeout`, `--accept-open-team-sheet`.

The leading-underscore name signals "internal entry, not user-invocable"
— consistent with the consolidation doc's naming for
`_vgcbench_subprocess.py`.

**`src/elitefurretai/rl/analyze/foulplay_eval.py`** (new file). The eval
driver. Two surfaces:

- `run(*, checkpoint_path, manager, n_battles, battle_format, device,
       server_url, agent_team_pool, run_tag) -> EvalResult` — callable
  from `train.py`. Builds `SimpleModelPlayer` instances, issues
  `n_battles` challenges to `manager.usernames[0]`, waits for
  completion, returns aggregated wins/losses. Caller owns the
  `FoulPlayManager` lifecycle.
- `main()` argparse entry point — manual CLI use:
  `python -m elitefurretai.rl.analyze.foulplay_eval --checkpoint <path>
   --battles 100`. Constructs and tears down its own
  `FoulPlayManager`.

The driver also exposes a thin helper invoked from `analyze/evaluate.py`
when `--baselines foulplay` is passed, so the existing graduation-eval
CLI gets FoulPlay support uniformly (see "Call site updates" below).

### Config additions

New `FoulplayEvalConfig` dataclass in `src/elitefurretai/rl/config.py`,
sibling to `CurriculumConfig`:

```python
@dataclass
class FoulplayEvalConfig:
    enabled: bool = False
    eval_every_n_updates: int = 50
    n_battles: int = 100
    search_time_ms: int = 750
    parallelism: int = 4
    python_executable: Optional[str] = None         # ../venv-foulplay/bin/python
    team_pool_path: str = "data/teams/gen9vgc2024regg/constrained"
    model_probabilistic: bool = False               # deterministic argmax
```

Hooked into `RNaDConfig`:

```python
foulplay_eval: FoulplayEvalConfig = field(default_factory=FoulplayEvalConfig)
```

Validation in `_validate` (alongside the existing vgcbench validator at
`config.py:759-767`): when `foulplay_eval.enabled`, assert
`python_executable` is set and exists, assert `team_pool_path` exists.

YAML surface (additive — existing configs are unchanged):

```yaml
foulplay_eval:
  enabled: true
  eval_every_n_updates: 50
  n_battles: 100
  search_time_ms: 750
  parallelism: 4
  python_executable: /home/cayman/Repositories/venv-foulplay/bin/python
  team_pool_path: data/teams/gen9vgc2024regg/constrained
```

### Call site updates

- **`rl/train.py`** — at the end of each update loop iteration:
  ```python
  if (config.foulplay_eval.enabled
      and step % config.foulplay_eval.eval_every_n_updates == 0
      and step > 0):
      _run_foulplay_eval_inline(config, step, current_checkpoint_path, server_ports)
  ```
  where `_run_foulplay_eval_inline` is a small helper that constructs
  the manager, calls `foulplay_eval.run(...)`, logs the `EvalResult` to
  wandb under `eval/foulplay/*`, and tears the manager down in a
  `finally`. Training is paused during eval; throughput hit per eval is
  the eval duration (~8–12 min at default settings).
- **`rl/analyze/evaluate.py`** — extend the `_make_opponent` switch at
  `:178` and the `--baselines` arg list at `:409` to accept `foulplay`.
  In the `foulplay` branch, the construction logic calls into
  `foulplay_eval.run` rather than `_make_opponent` (since FoulPlay isn't
  an in-process Player — it's a subprocess routed by username). Same
  username-routing pattern already used for `vgcbench` at
  `evaluate.py:154-187`. Graduation eval reports FoulPlay win rate
  alongside the four existing baselines.

### Setup (one-time)

Documented in `RL.md`, not scripted:

```bash
python3 -m venv ../venv-foulplay
../venv-foulplay/bin/pip install --upgrade pip==24.2
git clone https://github.com/pmariglia/foul-play-doubles ../foul-play-doubles
../venv-foulplay/bin/pip install -v -r ../foul-play-doubles/requirements.txt
```

The `requirements.txt` line
`poke-engine-doubles==0.0.7 --config-settings="build-args=--features poke-engine/terastallization --no-default-features"`
needs `pip>=24.2` to honor `--config-settings`. The build pulls Rust
toolchain (`cargo`) — first install takes 5–10 min on a cold cache.

### Logging & graduation

- Wandb keys per eval: `eval/foulplay/win_rate`,
  `eval/foulplay/n_battles`, `eval/foulplay/wall_time_s`,
  `eval/foulplay/search_time_ms` (logged once for context). Per-step
  logging means the eval shows up as a curve in wandb alongside the
  training metrics.
- **Not** added to Stage II graduation criterion. Tracked as a 5th
  reference signal only. `RL.md` gets a one-paragraph note that
  FoulPlay is a stronger benchmark than the four graduation baselines
  and that we may fold it into the criterion in a future stage. No edit
  to [2026-05-16-21-30-stage2-graduation-criteria.md][grad-criteria].

## Reasoning

**Why eval-only, not curriculum opponent?** FoulPlay is search-based
and slow (~750ms search × ~30 turns = ~22s search compute per battle).
Adding it to the training curriculum at any meaningful weight would
tank actor throughput (currently ~6 traj/s; FoulPlay battles would
process at ~0.05/s). The eval-only role isolates its cost: trainer
pauses, eval runs ~10 min, trainer resumes — visible drop in
updates/hour, but a known bounded cost, paid only at checkpoint
cadence.

**Why launch-per-eval, not run-for-lifetime-of-training?** Two
considerations:

1. Memory budget. VGCBenchManager already costs ~1.2 GB PSS resident
   for an opponent that plays during training. FoulPlay+poke-engine
   resident estimate is similar (Rust engine state + Python
   overhead). Keeping FoulPlay alive between evals burns 1–2 GB for
   no benefit, since the curriculum doesn't sample it.
2. Lifecycle simplicity. Per-eval launch keeps FoulPlay invisible to
   the rest of the training pipeline — no need to plumb its
   username into `OpponentPool`, no curriculum-weight branching, no
   `worker.py` routing changes.

The cost is ~10s subprocess startup per eval pass. At
`eval_every_n_updates=50` and ~10 min wall time per eval, the startup
is ~2% overhead — fine.

**Why mirror `VGCBenchManager` exactly?** Two managers with near-identical
shapes is intentional. The consolidation doc explicitly chose this
class-based pattern to make subprocess-managed opponents discoverable in
`players.py`. Adding a second instance of the pattern is the test of
whether it generalizes — and it does, with only config (search-time,
parallelism, team pool) and lifetime semantics (relaunchable vs.
once-per-training) differing.

**Why put the driver in `analyze/` instead of `players.py`?** Per the
user's framing: `players.py` is for *how opponents are set up* (the
`*Manager` classes and Player subclasses). Eval *orchestration* (which
checkpoint, how many battles, which metrics) is driver work, lives in
`analyze/`. The driver is what calls into `players.py` to build the
participants.

**Why extend `evaluate.py` rather than keep `foulplay_eval.py`
fully separate?** Graduation eval should yield one report covering all
baselines. Forcing the user to run two CLIs (`evaluate.py` for the four
existing baselines, `foulplay_eval.py` for FoulPlay) creates a chore at
the moment the report matters most. `foulplay_eval.py` exists as its
own module because (a) FoulPlay's subprocess-management is heavy enough
to deserve a module, (b) `train.py` needs a clean `run()` entry point
for inline eval. `evaluate.py` then thin-calls into it.

**Why deterministic (argmax) model play during eval?** Per the user's
answer: lowest variance, gives a clean measurement of the policy's
best move. Matches the current `evaluate.py` `probabilistic=False`
default at `:106`. If we later want to compare deterministic vs.
stochastic play, add a config flag — but defer until there's a reason.

**Why sample from the training opponent_team_pool, not a fixed team?**
Per the user's answer: tests generalized play, matches the diverse
opponent distribution the model trains against. The fixed-team option
is lower variance but tests a single matchup — less informative as a
ground-truth signal. Per-battle team rotation is handled by FoulPlay's
own `TeamListIterator`, configured to point at
`team_pool_path`.

**Why a separate venv, not Docker?** Discussed in brainstorming.
Short version: matches the existing `../venv-vgcbench` pattern, simpler
subprocess lifecycle (SIGTERM vs. container lifecycle), localhost
networking is trivial without `--network=host`, no Docker resident
overhead on WSL2. Docker isn't wrong — just heavier than the problem
needs for a solo-dev single-host setup.

## Risks

1. **FoulPlay's `pokemon_battle()` may not run cleanly outside its Bo3
   wrapper.** `fp/run_battle.py:pokemon_battle` is designed to be
   called inside the Bo3 loop in `run.py`. The implementation plan
   should verify a single-battle invocation works without modification,
   or write a minimal wrapper that reuses FoulPlay's `battle_modifier`,
   `search`, and `websocket_client` modules directly. Mitigation: a
   ~10-line smoke test in the implementation plan, before wiring the
   subprocess into the manager.
2. **First-install Rust toolchain.** `poke-engine-doubles` build pulls
   the Rust toolchain via `cargo`. On a system without Rust installed,
   the first `pip install` will fail. Mitigation: document `rustup`
   install in the `RL.md` setup section as a prerequisite.
3. **`requirements-dev.txt` vs. `requirements.txt` divergence.** The
   FoulPlay repo's `requirements.txt` pins `websockets==14.1`. EFA's
   main venv may pin a different version. Mitigation: the dedicated
   venv isolates this; no shared state.
4. **Search-bot strength is config-dependent.** A 100ms search bot is
   not the same opponent as a 1000ms search bot. The win rate trendline
   is only comparable across runs if `search_time_ms` is held constant.
   Mitigation: log `search_time_ms` to wandb with every eval; treat the
   metric as relative-within-run unless config matches.
5. **Pavel's repos warn that the doubles fork is "early stages and
   probably has many broken things on main."** Some battle states may
   crash FoulPlay or produce invalid choices. Mitigation: the
   subprocess logs to `data/logs/foulplay_runners/` and crashes are
   isolated from the trainer; a per-battle exception in
   `pokemon_battle` doesn't tank the eval (loss is recorded, next
   challenge accepted). Worth tracking: if FoulPlay crashes mid-eval,
   the reported win rate is silently biased. Manager should monitor
   the subprocess and abort eval if the process exits unexpectedly.

## Out of scope

- **Adding FoulPlay to the curriculum.** Could happen later if eval
  results suggest it would help; out of scope for this work.
- **Upgrading FoulPlay to a newer poke_env.** Same reasoning as the
  vgc-bench equivalent — multi-day port of third-party code, not on
  the Stage II critical path.
- **Folding FoulPlay into the graduation criterion.** Track as
  reference; revisit after a few runs show how reachable a 60%
  win-rate vs. FoulPlay is.
- **Multiple FoulPlay opponents at different search-time budgets.**
  Could be a useful gradient signal ("model beats 100ms but not
  500ms"); deferred.
- **Replacing the Showdown-WebSocket protocol with shared-memory IPC.**
  Same as vgc-bench's out-of-scope note.

## Planned next steps

1. Confirm dependency ordering: `VGCBenchManager` consolidation should
   land first so `FoulPlayManager` can mirror its final shape without
   re-doing the pattern. If consolidation slips, this work can proceed
   independently — the patterns are isolated.
2. Write the implementation plan (separate doc, following the
   subagent-driven-development split if breaking into steps).
3. Execute, in order:
   a. One-time venv setup (`../venv-foulplay`, clone foul-play-doubles,
      install requirements). Verify `poke-engine-doubles` builds.
   b. Smoke-test `_foulplay_subprocess.py` standalone: launch it, send
      one manual challenge, verify a battle completes.
   c. Introduce `FoulPlayManager` in `players.py`. Wire its lifecycle
      into a manual eval invocation via `foulplay_eval.py` CLI.
   d. Add `FoulplayEvalConfig` to `config.py`, wire YAML loading,
      add validation.
   e. Hook inline eval into `train.py`. Run a 5-update smoke training
      with `eval_every_n_updates=5` to verify lifecycle and wandb
      logging.
   f. Extend `analyze/evaluate.py` with `--baselines foulplay`.
   g. Update `RL.md` setup section.
   h. Run quality gates (`ruff`, `pyright`, `pytest`).
4. Mark this doc complete in Updates.

## Updates

(none yet — pre-implementation)
