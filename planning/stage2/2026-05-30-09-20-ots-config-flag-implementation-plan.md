# Open Team Sheets (OTS) Config Flag — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two config flags — `RNaDConfig.open_team_sheets` (training) and `EvalConfig.open_team_sheets` (eval) — that make every agent in a given run accept (or decline) Open Team Sheets consistently, so soft-OTS battles either all reveal sheets or all stay closed (never a mismatched handshake that drops battles).

**Architecture:** Two independent flags, each global within its context, default `False` (today's behavior). Training: `config.open_team_sheets` → `WorkerOpponentFactory` (applied to the main/opponent `RLTrajectoryPlayer`s and every heuristic baseline) and `VGCBenchManager` (the external vgc_bench runner). Eval: `EvalConfig.open_team_sheets` → the eval driver's `build_player` calls and the vgc_bench subprocess launch, plus a `--open-team-sheets` CLI flag.

**Tech Stack:** Python dataclasses + YAML config, poke-env `Player(accept_open_team_sheet=...)`, pytest.

Design spec: [planning/stage2/2026-05-30-09-04-ots-config-flag-design.md](2026-05-30-09-04-ots-config-flag-design.md).

---

## File Structure

Files modified (no new files):

- `src/elitefurretai/rl/config.py` — add `RNaDConfig.open_team_sheets` (top-level) + `from_dict` handling; add `EvalConfig.open_team_sheets`.
- `src/elitefurretai/rl/opponents.py` — `WorkerOpponentFactory.__init__` param + thread into `create_agents` and `_make_baseline_pool`.
- `src/elitefurretai/engine/vgc_environment.py` — pass `config.open_team_sheets` into `WorkerOpponentFactory(...)`.
- `src/elitefurretai/agents/vgcbench_manager.py` — `launch()` gates `--accept-open-team-sheet` on `self._config.open_team_sheets`; remove the now-unused `ACCEPT_OPEN_TEAM_SHEET` ClassVar (after eval is converted).
- `src/elitefurretai/rl/analyze/analysis_utils.py` — thread an `open_team_sheets` param through `run_eval_parallel` → `_run_worker` → `build_player` calls + `launch_external_player`/`_launch_vgc_bench_subprocess`; add `--open-team-sheets` CLI flag wherever the player-vs-player CLI currently lives.
- `src/elitefurretai/rl/analyze/evaluate_model.py` — pass `eval_cfg.open_team_sheets` into the eval driver.

Tests modified:
- `unit_tests/rl/test_config.py`
- `unit_tests/rl/test_worker_opponent_factory.py`
- `unit_tests/rl/analyze/test_player_factory.py` (eval build_player threading)

> **NOTE — eval files in flux:** `analysis_utils.py` and `evaluate_model.py` were being actively refactored on 2026-05-30 (the player-vs-player `main()`/`__main__` was removed from `analysis_utils.py`). Tasks 6–7 target stable function names (`run_eval_parallel`, `build_player`, `_launch_vgc_bench_subprocess`) and include a verify-current-call-sites step. Confirm the current structure before editing.

---

## Task 1: Training config field (`RNaDConfig.open_team_sheets`)

**Files:**
- Modify: `src/elitefurretai/rl/config.py` (RNaDConfig dataclass ~line 894–905; `from_dict` `cls(...)` ~line 999–1011)
- Test: `unit_tests/rl/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `unit_tests/rl/test_config.py`:

```python
def test_open_team_sheets_defaults_false_and_loads_from_dict():
    from elitefurretai.rl.config import RNaDConfig

    assert RNaDConfig().open_team_sheets is False
    cfg = RNaDConfig.from_dict({"open_team_sheets": True})
    assert cfg.open_team_sheets is True
    # round-trips through to_dict/from_dict
    assert RNaDConfig.from_dict(cfg.to_dict()).open_team_sheets is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py::test_open_team_sheets_defaults_false_and_loads_from_dict -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'open_team_sheets'` or `AttributeError: 'RNaDConfig' object has no attribute 'open_team_sheets'`.

- [ ] **Step 3: Add the dataclass field**

In `config.py`, in the `RNaDConfig` dataclass field block (after `value_head: ValueHeadConfig = ...`, ~line 904), add:

```python
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)

    # Run-wide Open Team Sheets setting. When True, every agent built for
    # this run (main + opponents + baselines + the vgc_bench runner) accepts
    # the soft-OTS prompt in formats that offer it (e.g. gen9vgc2024regg), so
    # both sides reveal sheets at team preview. False = closed (default;
    # preserves prior behavior). Must be uniform across a run — a mismatched
    # accept/deny handshake drops the battle. Forced/non-OTS formats ignore it.
    open_team_sheets: bool = False
```

- [ ] **Step 4: Handle it in `from_dict`**

In `RNaDConfig.from_dict`, the `return cls(...)` block (~line 999), add the field (top-level scalar; `from_dict` builds only sub-configs otherwise):

```python
        return cls(
            algorithm=_make_sub(AlgorithmConfig, data.get("algorithm", {})),
            architecture=_make_sub(ArchitectureConfig, data.get("architecture", {})),
            curriculum=_make_sub(CurriculumConfig, curriculum_data),
            exploiter=_make_sub(ExploiterConfig, data.get("exploiter", {})),
            exploration=_make_sub(ExplorationConfig, data.get("exploration", {})),
            eval=_make_eval_sub(data.get("eval", {})),
            hardware=_make_sub(HardwareConfig, data.get("hardware", {})),
            optimizer=_make_sub(OptimizerConfig, data.get("optimizer", {})),
            portfolio=_make_sub(PortfolioConfig, data.get("portfolio", {})),
            training=_make_sub(TrainingConfig, data.get("training", {})),
            value_head=_make_sub(ValueHeadConfig, data.get("value_head", {})),
            open_team_sheets=bool(data.get("open_team_sheets", False)),
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest unit_tests/rl/test_config.py::test_open_team_sheets_defaults_false_and_loads_from_dict -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "feat(config): add top-level RNaDConfig.open_team_sheets flag"
```

---

## Task 2: Eval config field (`EvalConfig.open_team_sheets`)

**Files:**
- Modify: `src/elitefurretai/rl/config.py` (`EvalConfig` dataclass ~line 752–793)
- Test: `unit_tests/rl/test_config.py`

`_make_eval_sub` already filters YAML by `EvalConfig.__dataclass_fields__`, so adding the field is enough — no loader change.

- [ ] **Step 1: Write the failing test**

Add to `unit_tests/rl/test_config.py`:

```python
def test_eval_open_team_sheets_defaults_false_and_loads_from_dict():
    from elitefurretai.rl.config import RNaDConfig

    assert RNaDConfig().eval.open_team_sheets is False
    cfg = RNaDConfig.from_dict({"eval": {"open_team_sheets": True}})
    assert cfg.eval.open_team_sheets is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest unit_tests/rl/test_config.py::test_eval_open_team_sheets_defaults_false_and_loads_from_dict -v`
Expected: FAIL — `AttributeError: 'EvalConfig' object has no attribute 'open_team_sheets'`.

- [ ] **Step 3: Add the field**

In `EvalConfig`, after `surplus_alpha: float = 1.0` (~line 755), add:

```python
    surplus_alpha: float = 1.0
    # Open Team Sheets for the eval pass. When True, both the model under
    # eval and every opponent (including the vgc_bench runner) accept the
    # soft-OTS prompt. Independent of the training-side flag so you can eval
    # open while training closed. False = closed (default).
    open_team_sheets: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest unit_tests/rl/test_config.py::test_eval_open_team_sheets_defaults_false_and_loads_from_dict -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "feat(config): add EvalConfig.open_team_sheets flag"
```

---

## Task 3: Thread the training flag through `WorkerOpponentFactory`

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` (`WorkerOpponentFactory.__init__` ~653–668; `_make_baseline_pool` ~859–868; `create_agents` `extra_player_kwargs` ~892–896)
- Test: `unit_tests/rl/test_worker_opponent_factory.py`

`RLTrajectoryPlayer` and all baseline classes (`MaxDamagePlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`, `RandomPlayer`) already accept `accept_open_team_sheet` — confirmed because `analyze/analysis_utils.build_player` passes it to all of them.

- [ ] **Step 1: Write the failing test**

Add to `unit_tests/rl/test_worker_opponent_factory.py` (mirrors the existing stub-player pattern around line 360):

```python
def test_open_team_sheets_threads_to_agents_and_baselines(monkeypatch):
    """create_agents must pass accept_open_team_sheet=<factory.open_team_sheets>
    to both the RLTrajectoryPlayer pair and every heuristic baseline."""
    import queue
    from typing import cast
    from elitefurretai.rl.opponents import OpponentPool, WorkerOpponentFactory

    recorded: list[bool] = []

    class _StubPlayer:
        def __init__(self, *, battle_format, accept_open_team_sheet=False, **kwargs):
            self.battle_format = battle_format
            recorded.append(accept_open_team_sheet)

    monkeypatch.setattr("elitefurretai.rl.opponents.RLTrajectoryPlayer", _StubPlayer)
    monkeypatch.setattr("elitefurretai.rl.opponents.MaxDamagePlayer", _StubPlayer)

    factory = _make_factory(
        curriculum={OpponentPool.SELF_PLAY: 0.5, OpponentPool.MAX_DAMAGE: 0.5},
        battle_formats={"gen9vgc2024regg": 1.0},
        worker_inference_clients=_clients_with("main"),
    )
    factory.open_team_sheets = True  # set the field the ctor now stores

    factory.create_agents(num_pairs=2, local_traj_queue=cast(queue.Queue, queue.Queue()))

    assert recorded, "no players were constructed"
    assert all(recorded), f"some players got accept_open_team_sheet=False: {recorded}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest unit_tests/rl/test_worker_opponent_factory.py::test_open_team_sheets_threads_to_agents_and_baselines -v`
Expected: FAIL — `AttributeError` on `factory.open_team_sheets` (field not stored yet), or the assertion fails because the kwarg isn't passed.

- [ ] **Step 3: Add the ctor param**

In `WorkerOpponentFactory.__init__` (~653), add a param at the end of the signature:

```python
        max_concurrent_battles_per_player: Optional[int] = None,
        open_team_sheets: bool = False,
    ):
```

and store it in the body (next to the other `self.` assignments, ~682):

```python
        self.max_battle_steps = max_battle_steps
        self.open_team_sheets = open_team_sheets
```

- [ ] **Step 4: Thread into the RLTrajectoryPlayer pair**

In `create_agents`, the `extra_player_kwargs` block (~892), add the flag unconditionally so both `player` and `opponent` (which both spread `**extra_player_kwargs`) receive it:

```python
        extra_player_kwargs: Dict[str, Any] = {
            "accept_open_team_sheet": self.open_team_sheets,
        }
        if self.max_concurrent_battles_per_player is not None:
            extra_player_kwargs["max_concurrent_battles"] = (
                self.max_concurrent_battles_per_player
            )
```

- [ ] **Step 5: Thread into the baseline pool**

In `_make_baseline_pool` (~859), add the kwarg to the `player_cls(...)` call:

```python
        return [
            player_cls(
                battle_format=fmt,
                account_configuration=AccountConfiguration(
                    self._account_name(role, i), None
                ),
                server_configuration=self.server_config,
                team=self.sample_team(fmt)[0],
                accept_open_team_sheet=self.open_team_sheets,
            )
            for i, fmt in enumerate(pair_formats)
        ]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest unit_tests/rl/test_worker_opponent_factory.py::test_open_team_sheets_threads_to_agents_and_baselines -v`
Expected: PASS

- [ ] **Step 7: Run the full factory + config suites (no regressions)**

Run: `pytest unit_tests/rl/test_worker_opponent_factory.py unit_tests/rl/test_config.py -q`
Expected: PASS (all)

- [ ] **Step 8: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "feat(rl): thread open_team_sheets through WorkerOpponentFactory"
```

---

## Task 4: Wire `config.open_team_sheets` into the factory construction

**Files:**
- Modify: `src/elitefurretai/engine/vgc_environment.py` (`WorkerOpponentFactory(...)` construction ~232; `config` is in scope from `_ShowdownBackend.__init__` at line 172)

- [ ] **Step 1: Add the argument**

In `vgc_environment.py`, at the `WorkerOpponentFactory(...)` call (~232), add the new kwarg (next to `max_concurrent_battles_per_player=...`):

```python
            max_concurrent_battles_per_player=hw.max_concurrent_battles_per_player,
            worker_inference_clients=worker_inference_clients,
            open_team_sheets=config.open_team_sheets,
        )
```

- [ ] **Step 2: Verify it imports and type-checks**

Run: `source ../venv/bin/activate && python -c "import elitefurretai.engine.vgc_environment" && pyright src/elitefurretai/engine/vgc_environment.py 2>&1 | tail -2`
Expected: import OK; `0 errors`.

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/engine/vgc_environment.py
git commit -m "feat(engine): pass config.open_team_sheets into WorkerOpponentFactory"
```

---

## Task 5: Gate the training vgc_bench runner on the config flag

**Files:**
- Modify: `src/elitefurretai/agents/vgcbench_manager.py` (`launch()` — the `if self.ACCEPT_OPEN_TEAM_SHEET:` block, ~line 351)

`VGCBenchManager.__init__(config, server_ports)` already stores `self._config`. Keep the `ACCEPT_OPEN_TEAM_SHEET` ClassVar for now — Task 6 removes the eval-side use, after which it can be deleted.

- [ ] **Step 1: Change the gate**

In `VGCBenchManager.launch()`, replace:

```python
            if self.ACCEPT_OPEN_TEAM_SHEET:
                command.append("--accept-open-team-sheet")
```

with:

```python
            if self._config.open_team_sheets:
                command.append("--accept-open-team-sheet")
```

- [ ] **Step 2: Verify import + type-check + existing agent tests**

Run: `source ../venv/bin/activate && pyright src/elitefurretai/agents/vgcbench_manager.py 2>&1 | tail -2 && pytest unit_tests/agents -q`
Expected: `0 errors`; agent tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/agents/vgcbench_manager.py
git commit -m "feat(agents): gate training vgc_bench OTS on config.open_team_sheets"
```

---

## Task 6: Thread the eval flag through the eval driver + CLI

**Files:**
- Modify: `src/elitefurretai/rl/analyze/analysis_utils.py` (`run_eval_parallel` ~421; `_run_worker` ~164; `launch_external_player` / `_launch_vgc_bench_subprocess`; the player-vs-player CLI `main()` wherever it now lives)
- Modify: `src/elitefurretai/agents/vgcbench_manager.py` (delete the now-unused `ACCEPT_OPEN_TEAM_SHEET` ClassVar)
- Test: `unit_tests/rl/analyze/test_player_factory.py`

> **VERIFY FIRST:** `grep -n "def run_eval_parallel\|def _run_worker\|def launch_external_player\|def _launch_vgc_bench_subprocess\|build_player(\|def main" src/elitefurretai/rl/analyze/analysis_utils.py` and confirm the call chain still matches before editing — these were mid-refactor on 2026-05-30.

- [ ] **Step 1: Write the failing test (in-process build_player already supports the flag; assert the eval worker forwards it)**

Add to `unit_tests/rl/analyze/test_player_factory.py` a test that `_launch_vgc_bench_subprocess` honors an `open_team_sheets` argument by including/excluding the CLI flag. Mirror existing tests in that file for how they patch `subprocess.Popen`:

```python
def test_launch_vgc_bench_subprocess_open_team_sheets_flag(monkeypatch):
    import elitefurretai.rl.analyze.analysis_utils as au

    captured = {}

    class _FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            self.pid = 1

        def poll(self):
            return None

    monkeypatch.setattr(au.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(au.time, "sleep", lambda *_a, **_k: None)

    au._launch_vgc_bench_subprocess(
        server_url="localhost:8000",
        battle_format="gen9vgc2024regg",
        checkpoint_path="data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
        team_file="data/teams/gen9vgc2024regg/vgcbench.txt",
        python_executable="/usr/bin/python",
        open_team_sheets=True,
    )
    assert "--accept-open-team-sheet" in captured["command"]

    au._launch_vgc_bench_subprocess(
        server_url="localhost:8000",
        battle_format="gen9vgc2024regg",
        checkpoint_path="data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
        team_file="data/teams/gen9vgc2024regg/vgcbench.txt",
        python_executable="/usr/bin/python",
        open_team_sheets=False,
    )
    assert "--accept-open-team-sheet" not in captured["command"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest unit_tests/rl/analyze/test_player_factory.py::test_launch_vgc_bench_subprocess_open_team_sheets_flag -v`
Expected: FAIL — `TypeError: _launch_vgc_bench_subprocess() got an unexpected keyword argument 'open_team_sheets'`.

- [ ] **Step 3: Add `open_team_sheets` to `_launch_vgc_bench_subprocess`**

In its signature (keyword-only block), add `open_team_sheets: bool = False,`. In the command-build, replace the `if VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET:` gate with:

```python
    if open_team_sheets:
        command.append("--accept-open-team-sheet")
```

(If the function builds the command without that gate today, add the two lines after the base `command = [...]` list.)

- [ ] **Step 4: Thread through `launch_external_player` → `run_eval_parallel` → `_run_worker`**

- `launch_external_player(specification, server_url, *, open_team_sheets: bool = False)` — pass `open_team_sheets=open_team_sheets` into `_launch_vgc_bench_subprocess`.
- `run_eval_parallel(..., open_team_sheets: bool = False)` (keyword-only) — pass it into each `_run_worker` submission.
- `_run_worker(..., open_team_sheets: bool = False)` — pass `accept_open_team_sheet=open_team_sheets` into every `build_player(...)` / `_build_player(...)` call, and `open_team_sheets=open_team_sheets` into `launch_external_player(...)`.

- [ ] **Step 5: Add the CLI flag**

Wherever the player-vs-player `main()`/argparse currently lives (verify per the note above), add:

```python
    parser.add_argument("--open-team-sheets", action="store_true",
                        help="Both players accept the soft-OTS prompt (Reg G etc.).")
```

and pass `open_team_sheets=args.open_team_sheets` into the `run_eval_parallel(...)` call.

- [ ] **Step 6: Remove the dead ClassVar**

In `vgcbench_manager.py`, delete the `ACCEPT_OPEN_TEAM_SHEET: ClassVar[bool] = False` line and its doc comment (now unused: training uses `self._config.open_team_sheets`, eval uses the param). Confirm no references remain:

Run: `grep -rn "ACCEPT_OPEN_TEAM_SHEET" src/` → expect no matches.

- [ ] **Step 7: Run tests**

Run: `pytest unit_tests/rl/analyze/test_player_factory.py unit_tests/agents -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/elitefurretai/rl/analyze/analysis_utils.py src/elitefurretai/agents/vgcbench_manager.py unit_tests/rl/analyze/test_player_factory.py
git commit -m "feat(eval): thread EvalConfig.open_team_sheets through eval driver + CLI"
```

---

## Task 7: Pass `EvalConfig.open_team_sheets` from the in-training eval driver

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py` (the `run_eval_parallel(...)` call)
- Test: `unit_tests/rl/analyze/test_evaluate_model.py`

> **VERIFY FIRST:** locate the `run_eval_parallel(...)` call (or whatever the multi-bucket driver invokes per opponent) and where `eval_cfg` / `EvalConfig` is in scope.

- [ ] **Step 1: Write the failing test**

In `unit_tests/rl/analyze/test_evaluate_model.py`, add a test that patches `run_eval_parallel` to capture its kwargs and asserts `open_team_sheets` is forwarded from the `EvalConfig`. Mirror the existing tests' setup for invoking the driver with a minimal config:

```python
def test_eval_forwards_open_team_sheets(monkeypatch):
    import elitefurretai.rl.analyze.evaluate_model as em

    captured = {}

    def _fake_run(*args, **kwargs):
        captured.update(kwargs)
        # Return a minimal EvalResult-like object the driver expects.
        from elitefurretai.rl.analyze.analysis_utils import EvalResult
        return EvalResult(label="x", player1_wins=1, player2_wins=0, ties=0, battles_played=1)

    monkeypatch.setattr(em, "run_eval_parallel", _fake_run)
    # ... invoke the driver against one in-process baseline bucket with an
    # EvalConfig whose open_team_sheets=True (mirror existing test setup here) ...
    assert captured.get("open_team_sheets") is True
```

(Fill the `...` using the construction already present in this test file's other cases.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest unit_tests/rl/analyze/test_evaluate_model.py::test_eval_forwards_open_team_sheets -v`
Expected: FAIL — `open_team_sheets` not in captured kwargs.

- [ ] **Step 3: Forward the flag**

At the `run_eval_parallel(...)` call in `evaluate_model.py`, add `open_team_sheets=eval_cfg.open_team_sheets,` (use the actual name the `EvalConfig` is bound to in that scope).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest unit_tests/rl/analyze/test_evaluate_model.py::test_eval_forwards_open_team_sheets -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py unit_tests/rl/analyze/test_evaluate_model.py
git commit -m "feat(eval): forward EvalConfig.open_team_sheets from evaluate_model driver"
```

---

## Task 8: Quality gates + docs

**Files:**
- Modify: `src/elitefurretai/rl/RL.md` (document the two flags under config), one example yaml (e.g. `configs/easy_test.yaml`) showing the keys commented out.

- [ ] **Step 1: Full quality gates**

Run:
```bash
source ../venv/bin/activate
ruff check src unit_tests
ruff format src unit_tests --check
pyright src unit_tests
pytest unit_tests -q
```
Expected: all clean / all pass.

- [ ] **Step 2: Document the flags**

In `src/elitefurretai/rl/RL.md`, add a short note in the config section: top-level `open_team_sheets: bool` (training; all agents accept the soft-OTS prompt) and `eval.open_team_sheets: bool` (eval pass), both default `False`, only meaningful in soft-OTS formats (Reg G etc.); forced/non-OTS formats ignore it; must be uniform within a run.

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/rl/RL.md src/elitefurretai/rl/configs/easy_test.yaml
git commit -m "docs(rl): document open_team_sheets training/eval flags"
```

- [ ] **Step 4 (optional): live validation**

Run a short Reg G eval with the eval flag on (once the CLI entrypoint is stable) and confirm OTS battles complete with no drops, e.g.:
```bash
python -m <eval-cli-module> --player1 vgc_bench --player2 max_damage \
  --team1 /tmp/eval_agent_team --team2 data/teams/gen9vgc2024regg/constrained \
  --cell-iteration --battles 1 --workers 4 --num-servers 4 --launch-servers \
  --battle-format gen9vgc2024regg --device cpu --open-team-sheets
```
Expected: 41 battles complete; compare win rate vs the closed-sheet baseline (max_damage 78.86%).

---

## Notes for the implementer

- Run each baseline eval as its own process (don't loop matchups in one shell — Showdown server teardown→relaunch on the same ports desyncs the runner/challenger).
- `accept_open_team_sheet` only matters in **soft-OTS** formats; in forced/Bo3 formats it's revealed regardless, and in non-OTS formats there's nothing to accept — so these flags are no-ops there (correct, no special handling).
- The whole point of "one value per context" is that both sides always agree — never expose a per-opponent OTS override, which would silently drop mismatched battles.
