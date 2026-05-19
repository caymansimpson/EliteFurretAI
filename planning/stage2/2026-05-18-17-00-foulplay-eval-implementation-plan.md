# FoulPlay Eval Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the search-based `foul-play-doubles` bot as a ground-truth eval opponent at configurable checkpoint intervals during RL training, with results logged to wandb and a uniform CLI entry via `analyze/evaluate.py --player2 foul_play`.

**Architecture:** A `FoulPlayManager` class in `players.py` owns the FoulPlay subprocess lifecycle (mirrors the in-flight `VGCBenchManager` pattern). A new `_foulplay_subprocess.py` entry point runs under a dedicated `../venv-foulplay/` venv. The eval driver in `analyze/foulplay_eval.py` orchestrates per-eval `manager.launch()` → battles → `manager.shutdown()`, and is called both from `train.py` inline at checkpoint cadence and from `analyze/evaluate.py` as a baseline. A new `external_username` field on `PlayerSpec` plus a `send_challenges`-based branch in `_run_worker` adds the routing pattern that subprocess-managed opponents need.

**Tech Stack:** Python 3.10+, poke-env 0.15 (training venv) + a separate venv with poke-env 0.11 + `poke-engine-doubles==0.0.7` (Rust extension w/ Tera feature flag), Pokemon Showdown websocket protocol, wandb, pytest.

**Reference spec:** [planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md)

---

## File Structure

**Create:**
- `src/elitefurretai/rl/_foulplay_subprocess.py` — internal subprocess entry, runs under `../venv-foulplay`. Imports FoulPlay's `fp.run_battle.pokemon_battle` and websocket client; accepts N challenges, plays each as a single (non-Bo3) battle.
- `src/elitefurretai/rl/analyze/foulplay_eval.py` — eval driver exposing `run(...)` for `train.py` and `main()` for CLI.
- `unit_tests/rl/test_foulplay_manager.py` — manager unit tests (mocked `subprocess.Popen`).
- `unit_tests/rl/analyze/__init__.py` — test package marker (if not present).
- `unit_tests/rl/analyze/test_foulplay_eval.py` — eval-driver unit tests (mocked manager + send_challenges).

**Modify:**
- `src/elitefurretai/rl/players.py` — add `FoulPlayManager` class.
- `src/elitefurretai/rl/config.py` — add `FoulplayEvalConfig` dataclass + validator hook.
- `src/elitefurretai/rl/analyze/player_factory.py` — add `external_username` field to `PlayerSpec`; add `foul_play` baseline; tighten factory contract.
- `src/elitefurretai/rl/analyze/evaluate.py` — branch `_run_worker` on external_username (use `send_challenges` instead of `battle_against`).
- `src/elitefurretai/rl/train.py` — inline eval hook at checkpoint boundary; wandb logging under `eval/foulplay/*`.
- `src/elitefurretai/rl/configs/single_team.yaml` — add `foulplay_eval:` block (disabled by default).
- `src/elitefurretai/rl/RL.md` — setup instructions for `../venv-foulplay`.
- `unit_tests/rl/test_config.py` — validation tests for FoulplayEvalConfig.
- `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md` — Updates section.

**Pre-existing files referenced (do not modify):**
- `src/elitefurretai/engine/showdown_server_manager.py` — `launch_showdown_servers` for the CLI test harness.
- `src/elitefurretai/rl/players.py` — `SimpleModelPlayer` for the model side.
- `src/elitefurretai/rl/analyze/team_provider.py` — `parse_team_spec` for team rotation.

**One-time setup (manual, documented in RL.md, not committed as a script):**
- `../venv-foulplay/` venv creation.
- `../foul-play-doubles/` clone + pip install.

---

## Task 1: Add `FoulplayEvalConfig` dataclass and YAML support

**Files:**
- Modify: [src/elitefurretai/rl/config.py](src/elitefurretai/rl/config.py)
- Modify: [unit_tests/rl/test_config.py](unit_tests/rl/test_config.py)

- [ ] **Step 1.1: Write the failing test for default config**

Add to `unit_tests/rl/test_config.py` (append at end of file):

```python
def test_default_foulplay_eval_config_disabled():
    """
    Test that FoulplayEvalConfig defaults to disabled.

    Eval against FoulPlay is opt-in because it requires a separately
    installed venv (`../venv-foulplay`). Default-on would crash any
    smoke training run on a fresh checkout.

    Expected: config.foulplay_eval.enabled is False.
    """
    config = get_default_config()
    assert config.foulplay_eval.enabled is False
    assert config.foulplay_eval.eval_every_n_updates == 50
    assert config.foulplay_eval.n_battles == 100
    assert config.foulplay_eval.search_time_ms == 750
    assert config.foulplay_eval.parallelism == 4
    assert config.foulplay_eval.python_executable is None
    assert (
        config.foulplay_eval.team_pool_path
        == "data/teams/gen9vgc2024regg/constrained"
    )
    assert config.foulplay_eval.model_probabilistic is False
```

- [ ] **Step 1.2: Run the test to confirm it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py::test_default_foulplay_eval_config_disabled -v`

Expected: FAIL with `AttributeError: 'RNaDConfig' object has no attribute 'foulplay_eval'`.

- [ ] **Step 1.3: Add the dataclass and field**

In `src/elitefurretai/rl/config.py`, locate the `CurriculumConfig` dataclass (it ends around line 540 — find the `@dataclass` block containing `external_vgcbench_python_executable`). Just **after** `CurriculumConfig` definition closes (before `RNaDConfig`), add:

```python
@dataclass
class FoulplayEvalConfig:
    """Inline-during-training eval against the external foul-play-doubles bot.

    All fields are config-driven. Disabled by default because it requires
    a separately installed venv at `python_executable`. See
    planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md.
    """

    enabled: bool = False
    eval_every_n_updates: int = 50
    n_battles: int = 100
    search_time_ms: int = 750
    parallelism: int = 4
    python_executable: Optional[str] = None
    team_pool_path: str = "data/teams/gen9vgc2024regg/constrained"
    model_probabilistic: bool = False
```

Then locate the `RNaDConfig` dataclass and add a new field next to `curriculum` (alphabetical or after curriculum is fine):

```python
foulplay_eval: FoulplayEvalConfig = field(default_factory=FoulplayEvalConfig)
```

(`field` and `Optional` should already be imported at the top of `config.py`; if not, add `from dataclasses import field` and `from typing import Optional`.)

- [ ] **Step 1.4: Run the default-config test to confirm it passes**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py::test_default_foulplay_eval_config_disabled -v`

Expected: PASS.

- [ ] **Step 1.5: Write the failing test for validation**

Add to `unit_tests/rl/test_config.py`:

```python
def test_foulplay_eval_validation_requires_python_executable_when_enabled(tmp_path):
    """
    Test that enabling foulplay_eval without python_executable raises.

    The subprocess lives in its own venv; without the interpreter path
    we cannot launch it. The validator should fail loudly at config-load
    time rather than at first checkpoint-eval boundary.

    Expected: AssertionError when enabled=True and python_executable=None.
    """
    config = get_default_config()
    config.foulplay_eval.enabled = True
    config.foulplay_eval.python_executable = None

    with pytest.raises(AssertionError, match="python_executable"):
        config._validate()


def test_foulplay_eval_validation_requires_existing_python_executable(tmp_path):
    """
    Test that a non-existent python_executable path is rejected.

    Expected: AssertionError when the path is set but does not exist.
    """
    config = get_default_config()
    config.foulplay_eval.enabled = True
    config.foulplay_eval.python_executable = str(tmp_path / "does_not_exist")

    with pytest.raises(AssertionError, match="python_executable"):
        config._validate()


def test_foulplay_eval_validation_requires_existing_team_pool(tmp_path):
    """
    Test that a non-existent team_pool_path is rejected when enabled.

    Expected: AssertionError when team_pool_path does not exist.
    """
    config = get_default_config()
    config.foulplay_eval.enabled = True
    # Use a real path for python_executable so we exercise the team_pool check.
    fake_py = tmp_path / "python"
    fake_py.write_text("")
    config.foulplay_eval.python_executable = str(fake_py)
    config.foulplay_eval.team_pool_path = str(tmp_path / "missing_pool")

    with pytest.raises(AssertionError, match="team_pool_path"):
        config._validate()
```

- [ ] **Step 1.6: Run validation tests to confirm they fail**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py -k foulplay -v`

Expected: 3 FAIL (no validator exists yet).

- [ ] **Step 1.7: Add validation logic to `_validate()`**

In `src/elitefurretai/rl/config.py`, find `_validate` on `RNaDConfig` (around line 759). After the existing vgcbench validation block, add:

```python
        # External foul-play-doubles eval validation. Mirror the
        # vgc-bench check style — fail fast if the configured venv
        # interpreter is missing, since the subprocess cannot launch
        # without it.
        if self.foulplay_eval.enabled:
            assert self.foulplay_eval.python_executable, (
                "foulplay_eval.python_executable must be set when "
                "foulplay_eval.enabled is True"
            )
            assert os.path.exists(self.foulplay_eval.python_executable), (
                f"foulplay_eval.python_executable not found: "
                f"{self.foulplay_eval.python_executable}"
            )
            assert os.path.exists(self.foulplay_eval.team_pool_path), (
                f"foulplay_eval.team_pool_path not found: "
                f"{self.foulplay_eval.team_pool_path}"
            )
```

(`os` is already imported at the top of `config.py`.)

- [ ] **Step 1.8: Run validation tests to confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py -k foulplay -v`

Expected: 4 PASS.

- [ ] **Step 1.9: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/config.py unit_tests/rl/test_config.py && pyright src/elitefurretai/rl/config.py unit_tests/rl/test_config.py`

Expected: 0 errors.

- [ ] **Step 1.10: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "$(cat <<'EOF'
feat(rl): add FoulplayEvalConfig with validation

Adds the FoulplayEvalConfig dataclass to RNaDConfig with sensible
defaults (disabled, 100 battles every 50 updates, 750ms search,
constrained team pool, deterministic model play). Validates that
python_executable and team_pool_path exist when enabled.

First step in integrating foul-play-doubles as a periodic
ground-truth eval signal — see
planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `FoulPlayManager` class in `players.py`

**Files:**
- Modify: [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py)
- Create: [unit_tests/rl/test_foulplay_manager.py](unit_tests/rl/test_foulplay_manager.py)

- [ ] **Step 2.1: Write the failing test for username derivation**

Create `unit_tests/rl/test_foulplay_manager.py`:

```python
"""Unit tests for FoulPlayManager (subprocess construction, lifecycle).

These tests mock subprocess.Popen so they can run without ../venv-foulplay
installed. The end-to-end smoke test (Task 5) verifies the real
subprocess flow against a live Showdown server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elitefurretai.rl.config import FoulplayEvalConfig
from elitefurretai.rl.players import FoulPlayManager


def _make_config(tmp_path) -> FoulplayEvalConfig:
    """Build a fully valid FoulplayEvalConfig with disk paths under tmp_path."""
    fake_python = tmp_path / "venv-foulplay" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("")

    team_pool = tmp_path / "teams" / "gen9vgc2024regg" / "constrained"
    team_pool.mkdir(parents=True)
    (team_pool / "dummy.txt").write_text("placeholder team")

    return FoulplayEvalConfig(
        enabled=True,
        python_executable=str(fake_python),
        team_pool_path=str(team_pool),
        n_battles=10,
        search_time_ms=500,
        parallelism=2,
    )


def test_derive_username_single_server_no_suffix(tmp_path):
    """
    Test that with a single server port, no suffix is appended.

    The 18-char username cap means we only append `_<port>` when
    multiple servers compete for the same base name. Single-server
    case should return the base name unchanged.

    Expected: derive_username("FOULPLAY", 8000) → "FOULPLAY" when
    there's only one server. (The append-suffix decision is made by
    the manager based on len(server_ports) > 1, not by derive_username
    itself — derive_username always appends when called.)
    """
    derived = FoulPlayManager.derive_username("FOULPLAY", 8000)
    # derive_username unconditionally appends; the manager decides
    # whether to invoke it. This test pins the formatting contract.
    assert derived == "FOULPLAY_8000"
    assert len(derived) <= 18


def test_derive_username_truncates_long_base(tmp_path):
    """
    Test that a long base name is truncated to fit the 18-char cap.

    Expected: a 20-char base with a 5-char suffix becomes 13 + 5 = 18.
    """
    derived = FoulPlayManager.derive_username("THIS_IS_TOO_LONG_BASE", 8000)
    assert len(derived) == 18
    assert derived.endswith("_8000")
```

- [ ] **Step 2.2: Run the test to confirm it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_foulplay_manager.py -v`

Expected: FAIL with `ImportError: cannot import name 'FoulPlayManager' from 'elitefurretai.rl.players'`.

- [ ] **Step 2.3: Add the `FoulPlayManager` class skeleton**

At the **bottom** of `src/elitefurretai/rl/players.py` (after `MaxDamagePlayer` and before the existing `__all__` list), add:

```python
# ── External subprocess managers ────────────────────────────────────────────
# These classes own the lifecycle of opponent bots that live in their own
# venv (poke_env version skew, native deps, etc.) and communicate with EFA
# only via Showdown websocket. Each manager spawns a subprocess that logs
# into Showdown under a known username; EFA-side code issues `/challenge`
# to that username when the opponent is selected.

import os
import signal
import subprocess
from typing import ClassVar, TextIO


class FoulPlayManager:
    """Launcher and proxy for the external foul-play-doubles bot.

    foul-play-doubles depends on `poke-engine-doubles` (a Rust extension)
    and an older poke_env vintage than EFA's training env. To isolate the
    environment, FoulPlay runs as a subprocess in its own venv. EFA-side
    code only interacts with FoulPlay by Showdown username.

    Unlike VGCBenchManager (alive for the lifetime of training), this
    manager is short-lived: launched per checkpoint-eval pass and torn
    down after the eval completes. Re-launchable.
    """

    SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/rl/_foulplay_subprocess.py"
    USERNAMES: ClassVar[List[str]] = ["FOULPLAY"]
    WAIT_FOR_SERVER_TIMEOUT_S: ClassVar[float] = 180.0
    LOG_DIR: ClassVar[str] = "data/logs/foulplay_runners"
    LOG_TO_FILES: ClassVar[bool] = True
    ACCEPT_OPEN_TEAM_SHEET: ClassVar[bool] = False
    STARTUP_WAIT_S: ClassVar[float] = 10.0
    # Single-runner-on-first-server layout (memory parity with VGCBench);
    # see planning/stage2/2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md.
    RUNNER_SERVER_INDEX: ClassVar[int] = 0

    def __init__(self, config: "FoulplayEvalConfig", server_ports: List[int]) -> None:
        self._config = config
        self._server_ports = list(server_ports)
        self._processes: List[subprocess.Popen] = []
        self._log_files: List[TextIO] = []
        self._usernames: List[str] = []

    @staticmethod
    def derive_username(base: str, server_port: int) -> str:
        """Build a server-scoped Showdown username under the 18-char cap.

        Suffix is `_<port>`; the base name is truncated as needed. Same
        contract as VGCBenchManager.derive_username — kept in lockstep so
        callers can swap managers without rebuilding the routing rules.
        """
        suffix = f"_{server_port}"
        max_base_len = max(1, 18 - len(suffix))
        return f"{base[:max_base_len]}{suffix}"

    @property
    def usernames(self) -> List[str]:
        """Usernames the eval driver issues /challenge to.

        Empty until launch() has been called.
        """
        return list(self._usernames)

    def launch(self) -> List[str]:
        """Spawn the FoulPlay subprocess and return the usernames it will log in as.

        Raises if `launch()` was already called without an intervening
        `shutdown()` — the manager is single-shot per cycle.
        """
        if self._processes:
            raise RuntimeError(
                "FoulPlayManager.launch() called twice without shutdown()"
            )

        if self.LOG_TO_FILES:
            os.makedirs(self.LOG_DIR, exist_ok=True)

        append_port_to_username = len(self._server_ports) > 1
        runner_port = self._server_ports[self.RUNNER_SERVER_INDEX]

        for base_username in self.USERNAMES:
            actual_username = (
                self.derive_username(base_username, runner_port)
                if append_port_to_username
                else base_username
            )
            self._usernames.append(actual_username)

            sanitized = actual_username.replace("/", "_")
            log_handle: Union[TextIO, int]
            log_path: str
            if self.LOG_TO_FILES:
                log_path = os.path.join(
                    self.LOG_DIR, f"runner_{sanitized}_{runner_port}.log"
                )
                handle = open(log_path, "a", encoding="utf-8")
                self._log_files.append(handle)
                log_handle = handle
            else:
                log_path = "<disabled>"
                log_handle = subprocess.DEVNULL

            command = self._build_command(actual_username, runner_port)
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._processes.append(process)
            print(
                "✓ Launched FoulPlay runner "
                f"'{actual_username}' on localhost:{runner_port} "
                f"(PID: {process.pid}) log={log_path}"
            )

        return list(self._usernames)

    def _build_command(self, username: str, server_port: int) -> List[str]:
        """Construct argv for the subprocess. Extracted for testability."""
        assert self._config.python_executable is not None
        command = [
            self._config.python_executable,
            self.SUBPROCESS_SCRIPT,
            "--username",
            username,
            "--server",
            f"localhost:{server_port}",
            "--battle-format",
            # Battle format lives on CurriculumConfig, not FoulplayEvalConfig.
            # The eval driver passes it in via a manager attribute set just
            # before launch() (see foulplay_eval.run). We default to the
            # standard Gen 9 VGC format here; the driver overrides.
            getattr(self, "battle_format", "gen9vgc2024regg"),
            "--n-challenges",
            str(self._config.n_battles),
            "--team-list-dir",
            self._config.team_pool_path,
            "--search-time-ms",
            str(self._config.search_time_ms),
            "--parallelism",
            str(self._config.parallelism),
            "--wait-for-server-timeout",
            str(self.WAIT_FOR_SERVER_TIMEOUT_S),
        ]
        if self.ACCEPT_OPEN_TEAM_SHEET:
            command.append("--accept-open-team-sheet")
        return command

    def shutdown(self) -> None:
        """Terminate the subprocess(es) and close log handles.

        Safe to call from a `finally` even if launch() failed partway.
        """
        for process in self._processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait()
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    print(
                        f"FoulPlayManager.shutdown: error terminating "
                        f"PID {process.pid}: {exc}"
                    )
        self._processes.clear()

        for handle in self._log_files:
            try:
                handle.close()
            except Exception:
                pass
        self._log_files.clear()
        self._usernames.clear()
```

Add at the top of the file (with the other imports) — only if not already present:

```python
from typing import ClassVar, TextIO, Union
```

Add the late import for `FoulplayEvalConfig` near the bottom of the imports (after the `from elitefurretai.rl.masking import ...` block) — guarded by `TYPE_CHECKING` since `config.py` does not import `players.py`:

```python
if TYPE_CHECKING:
    from elitefurretai.rl.config import FoulplayEvalConfig
```

Then update `__all__` at the bottom of the file:

```python
__all__ = [
    "RNaDAgent",
    "SimpleModelPlayer",
    "MaxDamagePlayer",
    "BatchInferencePlayer",
    "FoulPlayManager",
    "cleanup_worker_executors",
]
```

- [ ] **Step 2.4: Run username tests to confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_foulplay_manager.py -v`

Expected: 2 PASS.

- [ ] **Step 2.5: Write the failing test for launch command construction**

Append to `unit_tests/rl/test_foulplay_manager.py`:

```python
def test_launch_builds_expected_argv(tmp_path):
    """
    Test that launch() spawns subprocess.Popen with the configured args.

    All CLI flags should be present, in the order the subprocess script
    expects, with values from the config object.

    Expected: subprocess.Popen called once with a command containing
    --username, --server, --battle-format, --n-challenges,
    --team-list-dir, --search-time-ms, --parallelism,
    --wait-for-server-timeout.
    """
    config = _make_config(tmp_path)
    manager = FoulPlayManager(config, server_ports=[8000])
    manager.battle_format = "gen9vgc2024regg"

    with patch("elitefurretai.rl.players.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        usernames = manager.launch()

    assert usernames == ["FOULPLAY"]  # single server → no suffix
    assert mock_popen.call_count == 1
    argv = mock_popen.call_args.args[0]

    assert argv[0] == config.python_executable
    assert argv[1].endswith("_foulplay_subprocess.py")
    assert "--username" in argv and argv[argv.index("--username") + 1] == "FOULPLAY"
    assert "--server" in argv
    assert argv[argv.index("--server") + 1] == "localhost:8000"
    assert "--n-challenges" in argv
    assert argv[argv.index("--n-challenges") + 1] == "10"
    assert "--search-time-ms" in argv
    assert argv[argv.index("--search-time-ms") + 1] == "500"
    assert "--parallelism" in argv
    assert argv[argv.index("--parallelism") + 1] == "2"
    assert "--team-list-dir" in argv


def test_launch_appends_port_suffix_when_multiple_servers(tmp_path):
    """
    Test that multi-server setups get port-suffixed usernames.

    The suffix avoids username collisions when more than one runner
    might end up sharing the same Showdown server. With multiple
    ports configured, the runner still launches on server[0], but
    its username includes the port for unambiguity.

    Expected: usernames returned have the form "FOULPLAY_8000".
    """
    config = _make_config(tmp_path)
    manager = FoulPlayManager(config, server_ports=[8000, 8001, 8002])

    with patch("elitefurretai.rl.players.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        usernames = manager.launch()

    assert usernames == ["FOULPLAY_8000"]


def test_double_launch_raises(tmp_path):
    """
    Test that launching twice without shutdown raises.

    The manager is single-shot per cycle; double-launch is a caller bug.

    Expected: RuntimeError on second launch().
    """
    config = _make_config(tmp_path)
    manager = FoulPlayManager(config, server_ports=[8000])

    with patch("elitefurretai.rl.players.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        manager.launch()
        with pytest.raises(RuntimeError, match="launch.*twice"):
            manager.launch()


def test_shutdown_after_partial_launch_is_safe(tmp_path):
    """
    Test that shutdown() is safe to call even when launch was not invoked.

    The eval driver calls shutdown() in a `finally`, so the no-launch case
    must not error.

    Expected: shutdown() returns cleanly when _processes is empty.
    """
    config = _make_config(tmp_path)
    manager = FoulPlayManager(config, server_ports=[8000])
    manager.shutdown()  # should not raise
```

- [ ] **Step 2.6: Run all manager tests; confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_foulplay_manager.py -v`

Expected: 6 PASS.

- [ ] **Step 2.7: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/players.py unit_tests/rl/test_foulplay_manager.py && pyright src/elitefurretai/rl/players.py unit_tests/rl/test_foulplay_manager.py`

Expected: 0 errors.

- [ ] **Step 2.8: Commit**

```bash
git add src/elitefurretai/rl/players.py unit_tests/rl/test_foulplay_manager.py
git commit -m "$(cat <<'EOF'
feat(rl): add FoulPlayManager subprocess lifecycle class

Mirrors the in-flight VGCBenchManager pattern. Owns the lifecycle of
a foul-play-doubles subprocess (launched from ../venv-foulplay) that
accepts challenges on a known Showdown username. Single-runner-on-
server-0 layout for memory parity.

Manager is short-lived (one launch per checkpoint-eval pass) versus
VGCBenchManager's training-long lifetime. Adds derive_username,
launch(), shutdown(), usernames property; six unit tests covering
username derivation, argv construction, multi-server suffixing,
double-launch guard, and safe-shutdown-without-launch.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `_foulplay_subprocess.py` entry point

**Files:**
- Create: [src/elitefurretai/rl/_foulplay_subprocess.py](src/elitefurretai/rl/_foulplay_subprocess.py)

**Note on testing:** This script runs under `../venv-foulplay` (NOT the training venv), so unit tests in the main test suite cannot import it. We test it via a smoke test in Task 5 (manual smoke against a live Showdown server). Argv parsing is tested by manually invoking with `--help` in Task 3.4 below.

- [ ] **Step 3.1: Create the subprocess script**

Create `src/elitefurretai/rl/_foulplay_subprocess.py`:

```python
"""Internal entry point for the foul-play-doubles subprocess.

Runs under ../venv-foulplay/bin/python, NOT the EFA training venv.
Imports FoulPlay's own modules (config, data, fp, teams) via
cwd=../foul-play-doubles.

Leading underscore in filename signals "internal subprocess entry —
not user-invocable directly." Spawned by FoulPlayManager.launch().
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import socket
import sys
import time
from pathlib import Path


def _wait_for_server(server: str, timeout_s: float) -> None:
    """Block until the Showdown server's TCP socket accepts connections."""
    if ":" not in server:
        raise ValueError(f"Expected --server as host:port, got: {server}")
    host, port_str = server.rsplit(":", 1)
    port = int(port_str)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for showdown server {server}")


def _resolve_foulplay_root() -> Path:
    """Find the foul-play-doubles checkout directory.

    Defaults to ../foul-play-doubles relative to the EFA repo root.
    Allow override via FOULPLAY_ROOT environment variable for tests.
    """
    override = os.environ.get("FOULPLAY_ROOT")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    # __file__ is .../EliteFurretAI/src/elitefurretai/rl/_foulplay_subprocess.py;
    # repo root is parent[3]; foul-play-doubles is a sibling.
    repo_root = here.parents[3]
    candidate = repo_root.parent / "foul-play-doubles"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"foul-play-doubles checkout not found at {candidate}. "
            f"Clone it: git clone https://github.com/pmariglia/foul-play-doubles "
            f"{candidate}"
        )
    return candidate


def _list_team_files(team_pool_path: str) -> list:
    """Return a sorted list of team-file basenames (without extension) under team_pool_path.

    FoulPlay's `load_team` looks up by basename in its `teams/` directory.
    We symlink (or copy) the requested team pool into FoulPlay's teams/
    dir at startup so load_team can find them. This function returns
    the names to be requested via TeamListIterator.
    """
    pool = Path(team_pool_path)
    if not pool.is_dir():
        raise FileNotFoundError(f"team_pool_path is not a directory: {pool}")
    return sorted(p.stem for p in pool.iterdir() if p.is_file())


async def _accept_one_challenge(ps_websocket_client, pokemon_battle, team_export, team_dict, file_name, battle_format) -> None:
    """Accept exactly one challenge and play it as a single (non-Bo3) battle."""
    await ps_websocket_client.accept_challenge(battle_format, team_export, None)
    # FoulPlay's pokemon_battle handles a single battle; the Bo3 wrapper
    # lives in run.py and we deliberately bypass it. The battle_data list
    # is the per-battle scratchpad FoulPlay uses for telemetry.
    await pokemon_battle(ps_websocket_client, team_export, team_dict, file_name, [])


async def _run(args: argparse.Namespace) -> None:
    _wait_for_server(args.server, args.wait_for_server_timeout)

    # Switch cwd into foul-play-doubles before importing it — FoulPlay
    # uses package-relative imports that assume cwd is the repo root.
    foulplay_root = _resolve_foulplay_root()
    os.chdir(str(foulplay_root))
    sys.path.insert(0, str(foulplay_root))

    # Now safe to import FoulPlay modules.
    from config import FoulPlayConfig, BotModes, init_logging       # type: ignore
    from data.mods.apply_mods import apply_mods                     # type: ignore
    from fp.run_battle import pokemon_battle                        # type: ignore
    from fp.websocket_client import PSWebsocketClient               # type: ignore
    from teams import load_team                                     # type: ignore
    import logging

    # Configure FoulPlay's global config from CLI args.
    FoulPlayConfig.battle_bot_module = "search"
    FoulPlayConfig.websocket_uri = f"ws://{args.server}/showdown/websocket"
    FoulPlayConfig.username = args.username
    FoulPlayConfig.password = ""
    FoulPlayConfig.avatar = None
    FoulPlayConfig.bot_mode = BotModes.accept_challenge
    FoulPlayConfig.pokemon_format = args.battle_format
    FoulPlayConfig.search_time_ms = args.search_time_ms
    FoulPlayConfig.parallelism = args.parallelism
    FoulPlayConfig.user_to_challenge = ""
    FoulPlayConfig.save_replay = False
    FoulPlayConfig.team_list = None
    FoulPlayConfig.team_name = ""

    init_logging("INFO", False)
    apply_mods(FoulPlayConfig.pokemon_format)

    ps = await PSWebsocketClient.create(
        FoulPlayConfig.username,
        FoulPlayConfig.password,
        FoulPlayConfig.websocket_uri,
    )
    await ps.login()

    team_names = _list_team_files(args.team_list_dir)
    if not team_names:
        raise RuntimeError(f"No team files found in {args.team_list_dir}")

    print(
        f"[foulplay-runner] accepting {args.n_challenges} challenges on "
        f"{args.server} as {args.username} ({args.battle_format}), "
        f"search_time_ms={args.search_time_ms}, parallelism={args.parallelism}, "
        f"team_pool_size={len(team_names)}",
        flush=True,
    )

    for i in range(args.n_challenges):
        team_name = random.choice(team_names)
        # FoulPlay's load_team reads from its own teams/<format>/<name>
        # directory. We honor the same lookup by symlinking the pool at
        # setup time (documented in RL.md). load_team returns
        # (team_export, team_dict, file_name).
        try:
            team_export, team_dict, file_name = load_team(team_name)
        except Exception as exc:
            print(f"[foulplay-runner] load_team({team_name!r}) failed: {exc}", flush=True)
            continue

        try:
            await _accept_one_challenge(
                ps, pokemon_battle, team_export, team_dict, file_name, args.battle_format
            )
        except Exception as exc:
            # Per spec risk #5: if pokemon_battle crashes on a battle, log
            # and continue — the eval driver records this as a loss for
            # the model side via Showdown's win-rate accounting, and the
            # next challenge proceeds.
            print(
                f"[foulplay-runner] battle {i + 1}/{args.n_challenges} crashed: {exc}",
                flush=True,
            )
            logging.exception("foulplay-runner battle crash")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="foul-play-doubles bot subprocess entry — internal."
    )
    parser.add_argument("--username", required=True, type=str)
    parser.add_argument("--server", default="localhost:8000", type=str)
    parser.add_argument("--battle-format", default="gen9vgc2024regg", type=str)
    parser.add_argument("--n-challenges", default=1, type=int)
    parser.add_argument("--team-list-dir", required=True, type=str)
    parser.add_argument("--search-time-ms", default=750, type=int)
    parser.add_argument("--parallelism", default=4, type=int)
    parser.add_argument("--wait-for-server-timeout", default=180.0, type=float)
    parser.add_argument("--accept-open-team-sheet", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Verify the file is syntactically valid via `--help`**

Run: `python3 src/elitefurretai/rl/_foulplay_subprocess.py --help`

Expected: argparse help output listing all flags. (This runs under the regular Python, not the FoulPlay venv — the `_run` coroutine is never reached, so the FoulPlay imports never execute.)

- [ ] **Step 3.3: Run quality gates (with FoulPlay-import lines excluded from pyright)**

Add a `# type: ignore[import]` comment to each `from config import …`, `from data.mods.apply_mods import …`, etc. (already present in the snippet above as `# type: ignore`). Then run:

`source ../venv/bin/activate && ruff check src/elitefurretai/rl/_foulplay_subprocess.py && pyright src/elitefurretai/rl/_foulplay_subprocess.py`

Expected: 0 errors. (Ruff and pyright will not chase the runtime FoulPlay imports because they're inside a function and ignored.)

- [ ] **Step 3.4: Commit**

```bash
git add src/elitefurretai/rl/_foulplay_subprocess.py
git commit -m "$(cat <<'EOF'
feat(rl): add _foulplay_subprocess.py internal entry point

Runs under ../venv-foulplay (not the training venv). Configures
FoulPlayConfig from CLI args, logs into the local Showdown server,
and accepts N challenges, playing each as a single (non-Bo3) battle.
Bypasses foul-play's run.py Bo3 wrapper. Per-battle team rotation
picks uniformly from --team-list-dir.

Leading-underscore filename signals "internal subprocess entry,
spawned by FoulPlayManager — not user-invocable directly," matching
the renaming convention planned for vgcbench's runner script.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extend `PlayerSpec` with `external_username` and add `foul_play` baseline

**Files:**
- Modify: [src/elitefurretai/rl/analyze/player_factory.py](src/elitefurretai/rl/analyze/player_factory.py)
- Create: [unit_tests/rl/analyze/__init__.py](unit_tests/rl/analyze/__init__.py) (if not already a package)
- Create: [unit_tests/rl/analyze/test_player_factory.py](unit_tests/rl/analyze/test_player_factory.py)

- [ ] **Step 4.1: Check whether `unit_tests/rl/analyze/` already exists**

Run: `ls unit_tests/rl/analyze/ 2>/dev/null`

If the directory does not exist, create it with an empty `__init__.py`:

```bash
mkdir -p unit_tests/rl/analyze
touch unit_tests/rl/analyze/__init__.py
```

(Otherwise skip this step.)

- [ ] **Step 4.2: Write the failing test for foul_play baseline**

Create `unit_tests/rl/analyze/test_player_factory.py`:

```python
"""Unit tests for PlayerSpec, including the external-username branch."""

from __future__ import annotations

import pytest

from elitefurretai.rl.analyze.player_factory import (
    PlayerSpec,
    canonicalize_baseline,
    parse_player_spec,
)


def test_canonicalize_foul_play_baseline():
    """
    Test that 'foul_play' and 'foulplay' both canonicalize to 'foul_play'.

    Expected: both spellings return "foul_play".
    """
    assert canonicalize_baseline("foul_play") == "foul_play"
    assert canonicalize_baseline("foulplay") == "foul_play"
    assert canonicalize_baseline("FOUL_PLAY") == "foul_play"


def test_parse_player_spec_foul_play_returns_external_spec():
    """
    Test that --player foul_play produces a spec with external_username set.

    foul_play is a subprocess opponent (not an in-process Player), so its
    PlayerSpec must signal that to the eval driver via `external_username`.

    Expected: spec.kind == "baseline", spec.name == "foul_play",
    spec.external_username == "FOULPLAY", spec.factory is None.
    """
    spec = parse_player_spec(
        "foul_play",
        device="cpu",
        battle_format="gen9vgc2024regg",
    )
    assert spec.kind == "baseline"
    assert spec.name == "foul_play"
    assert spec.user_tag == "FP"
    assert spec.external_username == "FOULPLAY"
    assert spec.factory is None
```

- [ ] **Step 4.3: Run the test to confirm it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_player_factory.py -v`

Expected: FAIL (foul_play not in baseline list; `external_username` not a field of `PlayerSpec`).

- [ ] **Step 4.4: Extend `PlayerSpec` and add the `foul_play` baseline**

In `src/elitefurretai/rl/analyze/player_factory.py`:

(a) Update the `PlayerSpec` dataclass and the `PlayerFactory` type alias. Locate `_CANONICAL_BASELINES` (around line 33) and `PlayerSpec` (around line 69).

Replace the `PlayerFactory` alias and the `PlayerSpec` definition with:

```python
PlayerFactory = Callable[
    [Callable[[], str], AccountConfiguration, ServerConfiguration, bool],
    Player,
]


@dataclass(frozen=True)
class PlayerSpec:
    """Parsed player specification.

    Fields:
        raw: original CLI string (for logging / serialization).
        kind: ``"model"`` if ``raw`` resolves to a checkpoint file,
            ``"baseline"`` if it resolves to a known baseline name.
        name: canonical display name. For ``kind="model"`` this is the
            checkpoint filename (without extension); for baselines it
            is the canonical snake_case name (e.g. ``simple_heuristic``).
        user_tag: short uppercase tag used as a username prefix.
        factory: callable that constructs a poke-env ``Player``. None
            for external-subprocess opponents (see ``external_username``).
        external_username: if set, this opponent is NOT an in-process
            ``Player`` — it's a subprocess that has logged into Showdown
            under this username. The eval driver issues
            ``model.send_challenges(external_username, n)`` instead of
            ``model.battle_against(opponent_player_obj, n)``. The
            subprocess lifecycle (launch/shutdown) is the caller's
            responsibility.
    """

    raw: str
    kind: PlayerKind
    name: str
    user_tag: str
    factory: Optional[PlayerFactory]
    external_username: Optional[str] = None
```

(b) Add `"foul_play"` to `_CANONICAL_BASELINES` and `"foulplay"` to `_BASELINE_ALIASES`:

```python
_CANONICAL_BASELINES = (
    "max_damage",
    "max_base_power",
    "simple_heuristic",
    "vgc_bench",
    "random",
    "foul_play",
)

_BASELINE_ALIASES = {
    "maxdamage": "max_damage",
    "maxbasepower": "max_base_power",
    "shp": "simple_heuristic",
    "simpleheuristic": "simple_heuristic",
    "simpleheuristics": "simple_heuristic",
    "vgcbench": "vgc_bench",
    "foulplay": "foul_play",
}

_BASELINE_USER_TAG = {
    "max_damage": "MD",
    "max_base_power": "MBP",
    "simple_heuristic": "SHP",
    "vgc_bench": "VGB",
    "random": "RND",
    "foul_play": "FP",
}
```

(c) In `_baseline_spec`, before the `def factory(...)` definition, short-circuit `foul_play`:

```python
def _baseline_spec(
    raw: str,
    canonical: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str,
) -> PlayerSpec:
    user_tag = _BASELINE_USER_TAG[canonical]

    # foul_play is an external-subprocess opponent — not an in-process
    # Player. The eval driver issues `send_challenges(username, n)` to
    # the FoulPlay subprocess; no factory is needed. The default
    # username "FOULPLAY" matches FoulPlayManager.USERNAMES[0]; multi-
    # server runs derive a port-suffixed username at launch() time and
    # rewrite spec.external_username before _run_worker uses it.
    if canonical == "foul_play":
        return PlayerSpec(
            raw=raw,
            kind="baseline",
            name=canonical,
            user_tag=user_tag,
            factory=None,
            external_username="FOULPLAY",
        )

    def factory(
        team_provider: Callable[[], str],
        account_configuration: AccountConfiguration,
        server_configuration: ServerConfiguration,
        accept_open_team_sheet: bool,
    ) -> Player:
        # ... unchanged ...
```

(`Optional` should already be in the imports list at the top.)

- [ ] **Step 4.5: Run the foul_play tests; confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_player_factory.py -v`

Expected: 2 PASS.

- [ ] **Step 4.6: Write a regression test that existing baselines still get a factory**

Append to `unit_tests/rl/analyze/test_player_factory.py`:

```python
def test_existing_baselines_still_have_factory_and_no_external_username():
    """
    Regression test: adding foul_play with external_username must not
    have broken the existing in-process baselines.

    Expected: max_damage, simple_heuristic, vgc_bench, random all have
    factory != None and external_username is None.
    """
    for name in ("max_damage", "max_base_power", "simple_heuristic", "vgc_bench", "random"):
        spec = parse_player_spec(
            name, device="cpu", battle_format="gen9vgc2024regg"
        )
        assert spec.factory is not None, f"{name} lost its factory"
        assert spec.external_username is None, (
            f"{name} should not have external_username (only foul_play does)"
        )


def test_model_spec_has_factory_and_no_external_username(tmp_path):
    """
    Regression test: model-checkpoint specs still get a factory.

    Expected: factory is not None; external_username is None.
    """
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")
    spec = parse_player_spec(
        str(fake_ckpt), device="cpu", battle_format="gen9vgc2024regg"
    )
    assert spec.kind == "model"
    assert spec.factory is not None
    assert spec.external_username is None
```

- [ ] **Step 4.7: Run the regression tests; confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_player_factory.py -v`

Expected: 4 PASS.

- [ ] **Step 4.8: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/player_factory.py unit_tests/rl/analyze/ && pyright src/elitefurretai/rl/analyze/player_factory.py unit_tests/rl/analyze/`

Expected: 0 errors.

- [ ] **Step 4.9: Commit**

```bash
git add src/elitefurretai/rl/analyze/player_factory.py unit_tests/rl/analyze/
git commit -m "$(cat <<'EOF'
feat(rl/analyze): add external_username field and foul_play baseline

Extends PlayerSpec with an optional external_username for opponents
that live in a separate process and have logged into Showdown — they
have no in-process Player, so .factory is None and the eval driver
routes via send_challenges().

Adds 'foul_play' (alias 'foulplay') as a baseline kind. Existing
baselines (max_damage, vgc_bench, etc.) are unchanged. The same
external_username pattern will be reusable for vgc_bench once the
consolidation work lands.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Route external-username players via `send_challenges` in `_run_worker`

**Files:**
- Modify: [src/elitefurretai/rl/analyze/evaluate.py](src/elitefurretai/rl/analyze/evaluate.py)
- Modify: [unit_tests/rl/analyze/test_player_factory.py](unit_tests/rl/analyze/test_player_factory.py) (cross-check that PlayerSpec carries the expected shape)

- [ ] **Step 5.1: Write the failing test for `_run_worker` routing**

Create `unit_tests/rl/analyze/test_evaluate_routing.py`:

```python
"""Unit tests for evaluate._run_worker's external-username branch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elitefurretai.rl.analyze.evaluate import _run_worker
from elitefurretai.rl.analyze.player_factory import PlayerSpec


def _model_spec_with_mock_factory(mock_player) -> PlayerSpec:
    def factory(team_provider, account_config, server_config, accept_ots):
        return mock_player
    return PlayerSpec(
        raw="model.pt",
        kind="model",
        name="model",
        user_tag="MDL",
        factory=factory,
        external_username=None,
    )


def _external_spec(username: str) -> PlayerSpec:
    return PlayerSpec(
        raw=username.lower(),
        kind="baseline",
        name="foul_play",
        user_tag="FP",
        factory=None,
        external_username=username,
    )


def test_run_worker_uses_send_challenges_when_player2_is_external():
    """
    Test that when p2 has external_username, _run_worker calls
    model_player.send_challenges(username, n) instead of battle_against.

    The external opponent isn't an in-process Player, so there is no
    object to pass to battle_against. The model challenges by username.

    Expected: send_challenges called once with (username, n_battles).
    """
    mock_model = MagicMock()
    mock_model.send_challenges = AsyncMock(return_value=None)
    mock_model.battle_against = AsyncMock(return_value=None)
    mock_model.n_finished_battles = 5
    mock_model.n_won_battles = 3
    mock_model.n_lost_battles = 2

    p1 = _model_spec_with_mock_factory(mock_model)
    p2 = _external_spec("FOULPLAY")

    result = _run_worker(
        worker_id=0,
        p1=p1,
        p2=p2,
        t1=lambda: "team1",
        t2=lambda: "team2",
        battles=5,
        server_url="localhost:8000",
        run_tag="abcd",
    )

    mock_model.send_challenges.assert_awaited_once()
    args, kwargs = mock_model.send_challenges.call_args
    assert args[0] == "FOULPLAY"
    assert args[1] == 5
    mock_model.battle_against.assert_not_awaited()

    assert result.player1_wins == 3
    assert result.player2_wins == 2
    assert result.battles_played == 5


def test_run_worker_uses_battle_against_for_in_process_opponents():
    """
    Test that in-process opponents (both factories present) still go
    through battle_against — the existing path is unchanged.

    Expected: battle_against called once; send_challenges not called.
    """
    mock_model = MagicMock()
    mock_model.battle_against = AsyncMock(return_value=None)
    mock_model.send_challenges = AsyncMock(return_value=None)
    mock_model.n_finished_battles = 4
    mock_model.n_won_battles = 2
    mock_model.n_lost_battles = 2

    mock_opp = MagicMock()
    p1 = _model_spec_with_mock_factory(mock_model)
    p2 = _model_spec_with_mock_factory(mock_opp)

    _run_worker(
        worker_id=0,
        p1=p1,
        p2=p2,
        t1=lambda: "team1",
        t2=lambda: "team2",
        battles=4,
        server_url="localhost:8000",
        run_tag="abcd",
    )

    mock_model.battle_against.assert_awaited_once()
    mock_model.send_challenges.assert_not_awaited()
```

- [ ] **Step 5.2: Run the test to confirm it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_routing.py -v`

Expected: FAIL. The current `_run_worker` always calls `battle_against`; it also tries to call `p2.factory(...)` which is None for external specs.

- [ ] **Step 5.3: Branch `_run_worker` on `external_username`**

In `src/elitefurretai/rl/analyze/evaluate.py`, replace the body of `_run_worker` (lines 97–143) with:

```python
def _run_worker(
    worker_id: int,
    p1: PlayerSpec,
    p2: PlayerSpec,
    t1: TeamProvider,
    t2: TeamProvider,
    battles: int,
    server_url: str,
    run_tag: str,
) -> EvalResult:
    """One worker's slice of an eval matchup.

    Two routing modes:
      • In-process: both p1 and p2 have factories → ``battle_against``.
      • External-username p2: p2.factory is None and p2.external_username
        is set → p1's player issues ``send_challenges`` to that username.
        The external subprocess (e.g. FoulPlay) is launched and torn
        down by the caller; this worker only issues the challenges.

    External-username p1 is intentionally not supported: the model side
    must be the in-process challenger for `n_won_battles`-based win-rate
    accounting. If you need an external-vs-external matchup, set both
    sides up via challenges in a different harness.
    """

    async def _run() -> EvalResult:
        server_config = ServerConfiguration(f"ws://{server_url}/showdown/websocket", "")

        if p1.external_username is not None:
            raise ValueError(
                "External-username player1 is not supported; put the model on "
                "the player1 slot and the subprocess opponent on player2."
            )

        p1_account = AccountConfiguration(
            _username(f"E1{p1.user_tag}", worker_id, run_tag), None
        )
        assert p1.factory is not None  # narrowed by the check above
        player1 = p1.factory(t1, p1_account, server_config, False)

        if p2.external_username is not None:
            try:
                await player1.send_challenges(p2.external_username, battles)
            except Exception as exc:
                print(
                    f"[eval] worker={worker_id} {p1.name} vs {p2.name} "
                    f"(external) failed: {exc}"
                )
        else:
            p2_account = AccountConfiguration(
                _username(f"E2{p2.user_tag}", worker_id, run_tag), None
            )
            assert p2.factory is not None
            player2 = p2.factory(t2, p2_account, server_config, False)
            try:
                await player1.battle_against(player2, n_battles=battles)
            except Exception as exc:
                print(f"[eval] worker={worker_id} {p1.name} vs {p2.name} failed: {exc}")

        played = player1.n_finished_battles
        p1_wins = player1.n_won_battles
        p2_wins = player1.n_lost_battles
        ties = played - p1_wins - p2_wins
        return EvalResult(
            label=f"{p1.name}_vs_{p2.name}",
            player1_wins=p1_wins,
            player2_wins=p2_wins,
            ties=ties,
            battles_played=played,
        )

    return asyncio.run(_run())
```

- [ ] **Step 5.4: Run the routing tests; confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_routing.py -v`

Expected: 2 PASS.

- [ ] **Step 5.5: Add `--player1 foul_play` → friendly error**

The eval driver only supports model-on-player1 + external-on-player2. Verify the new ValueError fires when foul_play is on player1:

Append to `unit_tests/rl/analyze/test_evaluate_routing.py`:

```python
def test_run_worker_rejects_external_player1():
    """
    Test that external-username on player1 raises a clear error.

    External opponents must be on the player2 slot — the model is the
    challenger and owns the win-rate accounting.

    Expected: ValueError mentioning external-username player1.
    """
    p1 = _external_spec("FOULPLAY")
    p2 = _model_spec_with_mock_factory(MagicMock())

    with pytest.raises(ValueError, match="External-username player1"):
        _run_worker(
            worker_id=0,
            p1=p1,
            p2=p2,
            t1=lambda: "t1",
            t2=lambda: "t2",
            battles=1,
            server_url="localhost:8000",
            run_tag="abcd",
        )
```

- [ ] **Step 5.6: Run the new test; confirm it passes**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_routing.py::test_run_worker_rejects_external_player1 -v`

Expected: PASS.

- [ ] **Step 5.7: Run full evaluate.py-touching test suite**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/ -k "evaluate or player_factory or foulplay" -v`

Expected: all PASS.

- [ ] **Step 5.8: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/evaluate.py unit_tests/rl/analyze/ && pyright src/elitefurretai/rl/analyze/evaluate.py unit_tests/rl/analyze/`

Expected: 0 errors.

- [ ] **Step 5.9: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate.py unit_tests/rl/analyze/test_evaluate_routing.py
git commit -m "$(cat <<'EOF'
feat(rl/analyze): route external-username opponents via send_challenges

_run_worker now branches on PlayerSpec.external_username: if set, the
in-process player1 (the model) issues send_challenges(username, n)
instead of battle_against(player2_obj, n). The external subprocess
lifecycle is the caller's responsibility — this commit only adds the
routing.

External-username player1 is explicitly rejected (model must own
win-rate accounting). Three tests cover both branches + the rejection.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `analyze/foulplay_eval.py` driver

**Files:**
- Create: [src/elitefurretai/rl/analyze/foulplay_eval.py](src/elitefurretai/rl/analyze/foulplay_eval.py)
- Create: [unit_tests/rl/analyze/test_foulplay_eval.py](unit_tests/rl/analyze/test_foulplay_eval.py)

- [ ] **Step 6.1: Write the failing test for the driver's `run()`**

Create `unit_tests/rl/analyze/test_foulplay_eval.py`:

```python
"""Unit tests for the foulplay_eval driver.

Mocks the FoulPlayManager (subprocess) and the model player so tests
run without ../venv-foulplay installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elitefurretai.rl.analyze.foulplay_eval import FoulplayEvalResult, run


def test_run_launches_and_shuts_down_manager_around_eval(tmp_path):
    """
    Test that run() invokes manager.launch() before the eval and
    manager.shutdown() after, even when the eval body raises.

    Lifecycle hygiene: the FoulPlay subprocess must never outlive the
    eval pass, or it'll squat on Showdown username + memory.

    Expected: launch called once; shutdown called once; shutdown
    invoked even if run_eval_parallel raises.
    """
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")

    manager = MagicMock()
    manager.launch.return_value = ["FOULPLAY"]
    manager.usernames = ["FOULPLAY"]

    with patch(
        "elitefurretai.rl.analyze.foulplay_eval.run_eval_parallel",
        side_effect=RuntimeError("eval blew up"),
    ) as mock_run_parallel:
        with pytest.raises(RuntimeError, match="eval blew up"):
            run(
                checkpoint_path=str(fake_ckpt),
                manager=manager,
                n_battles=5,
                battle_format="gen9vgc2024regg",
                device="cpu",
                server_url="localhost:8000",
                agent_team_pool=str(tmp_path),
                run_tag="abcd",
            )

    manager.launch.assert_called_once()
    manager.shutdown.assert_called_once()
    mock_run_parallel.assert_called_once()


def test_run_returns_foulplay_eval_result_on_success(tmp_path):
    """
    Test that successful eval returns a FoulplayEvalResult with
    aggregated wins/losses.

    Expected: result.win_rate == player1_wins / battles_played.
    """
    fake_ckpt = tmp_path / "model.pt"
    fake_ckpt.write_bytes(b"placeholder")
    fake_pool = tmp_path / "pool"
    fake_pool.mkdir()
    (fake_pool / "team.txt").write_text("placeholder")

    manager = MagicMock()
    manager.launch.return_value = ["FOULPLAY"]
    manager.usernames = ["FOULPLAY"]

    fake_eval = MagicMock()
    fake_eval.player1_wins = 30
    fake_eval.player2_wins = 20
    fake_eval.ties = 0
    fake_eval.battles_played = 50

    with patch(
        "elitefurretai.rl.analyze.foulplay_eval.run_eval_parallel",
        return_value=fake_eval,
    ):
        result = run(
            checkpoint_path=str(fake_ckpt),
            manager=manager,
            n_battles=50,
            battle_format="gen9vgc2024regg",
            device="cpu",
            server_url="localhost:8000",
            agent_team_pool=str(fake_pool),
            run_tag="abcd",
        )

    assert isinstance(result, FoulplayEvalResult)
    assert result.battles_played == 50
    assert result.model_wins == 30
    assert result.foulplay_wins == 20
    assert result.win_rate == pytest.approx(0.6)
```

- [ ] **Step 6.2: Run the test to confirm it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_foulplay_eval.py -v`

Expected: FAIL with `ImportError: cannot import name 'FoulplayEvalResult'`.

- [ ] **Step 6.3: Create the driver module**

Create `src/elitefurretai/rl/analyze/foulplay_eval.py`:

```python
"""Driver for inline-during-training FoulPlay evaluation.

Two entry points:
  • ``run(...)``: called from ``train.py`` at checkpoint cadence.
    Caller owns the ``FoulPlayManager`` lifecycle; this function calls
    launch() / shutdown() around the eval body.
  • ``main()``: argparse CLI for manual use.

Both build a ``SimpleModelPlayer`` (one per worker), use the manager's
single Showdown username as the external-username opponent, and
aggregate wins/losses via ``run_eval_parallel`` from ``evaluate``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.analyze.evaluate import EvalResult, run_eval_parallel
from elitefurretai.rl.analyze.player_factory import PlayerSpec, parse_player_spec
from elitefurretai.rl.analyze.team_provider import parse_team_spec
from elitefurretai.rl.config import FoulplayEvalConfig
from elitefurretai.rl.players import FoulPlayManager


@dataclass
class FoulplayEvalResult:
    """Outcome of one eval pass, with FoulPlay-specific naming."""

    model_wins: int
    foulplay_wins: int
    ties: int
    battles_played: int
    wall_time_s: float

    @property
    def win_rate(self) -> float:
        if self.battles_played == 0:
            return 0.0
        return self.model_wins / self.battles_played


def run(
    *,
    checkpoint_path: str,
    manager: FoulPlayManager,
    n_battles: int,
    battle_format: str,
    device: str,
    server_url: str,
    agent_team_pool: str,
    run_tag: str,
) -> FoulplayEvalResult:
    """Run one FoulPlay eval pass against the given checkpoint.

    Lifecycle:
      1. manager.launch() — spawns subprocess, waits STARTUP_WAIT_S.
      2. run_eval_parallel — model issues n_battles challenges to
         manager.usernames[0].
      3. manager.shutdown() — SIGTERM the subprocess (in `finally`).

    `agent_team_pool` is the directory the model side samples teams
    from; FoulPlay's team pool comes from the manager's config.
    """
    # Stamp the battle format on the manager so its launch() can build
    # argv. The manager doesn't have a battle_format field of its own —
    # battle_format lives on CurriculumConfig elsewhere.
    manager.battle_format = battle_format  # type: ignore[attr-defined]

    model_spec = parse_player_spec(
        checkpoint_path,
        device=device,
        battle_format=battle_format,
    )
    foulplay_spec = parse_player_spec(
        "foul_play",
        device=device,
        battle_format=battle_format,
    )
    model_team_provider = parse_team_spec(agent_team_pool, battle_format=battle_format)
    # FoulPlay's team comes from its subprocess (--team-list-dir),
    # not from a TeamProvider on the EFA side. Pass a noop provider.
    noop_team_provider = lambda: ""

    started = time.time()
    manager.launch()
    try:
        # Wait for the subprocess to log in to Showdown. The manager's
        # STARTUP_WAIT_S is a coarse "give it time to handshake" delay
        # — without it, the first challenges can hit an unlogged-in user.
        time.sleep(FoulPlayManager.STARTUP_WAIT_S)

        # Override foulplay_spec.external_username with whatever the
        # manager actually logged in as (which may include a port suffix
        # under multi-server configs).
        actual_username = manager.usernames[0]
        spec_with_actual = PlayerSpec(
            raw=foulplay_spec.raw,
            kind=foulplay_spec.kind,
            name=foulplay_spec.name,
            user_tag=foulplay_spec.user_tag,
            factory=foulplay_spec.factory,
            external_username=actual_username,
        )

        eval_result: EvalResult = run_eval_parallel(
            p1=model_spec,
            p2=spec_with_actual,
            t1=model_team_provider,
            t2=noop_team_provider,
            num_battles=n_battles,
            server_urls=[server_url],
            workers=1,  # one worker — the subprocess accepts serially
            run_tag=run_tag,
        )
    finally:
        manager.shutdown()

    return FoulplayEvalResult(
        model_wins=eval_result.player1_wins,
        foulplay_wins=eval_result.player2_wins,
        ties=eval_result.ties,
        battles_played=eval_result.battles_played,
        wall_time_s=round(time.time() - started, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FoulPlay eval against a model checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--search-time-ms", type=int, default=750)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--battle-format", type=str, default="gen9vgc2024regg")
    parser.add_argument(
        "--python-executable",
        required=True,
        type=str,
        help="Path to ../venv-foulplay/bin/python",
    )
    parser.add_argument(
        "--foulplay-team-pool",
        required=True,
        type=str,
        help="Directory FoulPlay samples teams from",
    )
    parser.add_argument(
        "--agent-team-pool",
        required=True,
        type=str,
        help="Directory the model samples teams from",
    )
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument(
        "--launch-servers",
        action="store_true",
        help="Launch local Showdown servers automatically",
    )
    args = parser.parse_args()

    config = FoulplayEvalConfig(
        enabled=True,
        n_battles=args.battles,
        search_time_ms=args.search_time_ms,
        parallelism=args.parallelism,
        python_executable=args.python_executable,
        team_pool_path=args.foulplay_team_pool,
    )

    server_processes = []
    if args.launch_servers:
        server_processes = launch_showdown_servers(args.num_servers, args.start_port)

    try:
        server_ports = [args.start_port + i for i in range(args.num_servers)]
        manager = FoulPlayManager(config, server_ports=server_ports)
        run_tag = format(int(time.time() * 1000) % 65536, "04x")

        result = run(
            checkpoint_path=args.checkpoint,
            manager=manager,
            n_battles=args.battles,
            battle_format=args.battle_format,
            device=args.device,
            server_url=f"localhost:{server_ports[0]}",
            agent_team_pool=args.agent_team_pool,
            run_tag=run_tag,
        )

        print(
            f"FoulPlay eval | battles={result.battles_played} "
            f"model_wins={result.model_wins} foulplay_wins={result.foulplay_wins} "
            f"ties={result.ties} win_rate={result.win_rate * 100:.2f}% "
            f"wall_time={result.wall_time_s:.1f}s"
        )
    finally:
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run the driver tests; confirm they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_foulplay_eval.py -v`

Expected: 2 PASS.

- [ ] **Step 6.5: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/foulplay_eval.py unit_tests/rl/analyze/test_foulplay_eval.py && pyright src/elitefurretai/rl/analyze/foulplay_eval.py unit_tests/rl/analyze/test_foulplay_eval.py`

Expected: 0 errors.

- [ ] **Step 6.6: Commit**

```bash
git add src/elitefurretai/rl/analyze/foulplay_eval.py unit_tests/rl/analyze/test_foulplay_eval.py
git commit -m "$(cat <<'EOF'
feat(rl/analyze): add foulplay_eval driver (run() + CLI)

run(*, checkpoint_path, manager, n_battles, ...) is callable from
train.py at checkpoint cadence; it owns the launch+shutdown lifecycle
around a single eval pass. main() exposes the same flow via argparse
for manual checkpoint eval.

Both routes build SimpleModelPlayer for the model side and treat
FoulPlay as an external-username PlayerSpec, routed through the new
send_challenges branch in _run_worker.

Two tests cover lifecycle hygiene (shutdown always called) and the
success-path aggregation into FoulplayEvalResult.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Hook inline eval into `train.py`

**Files:**
- Modify: [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py)

**Context:** Insert the eval hook at the checkpoint boundary so the model file it evaluates is the just-saved `ghost_checkpoint_path`. The relevant block in `train.py` is the `if updates % config.training.checkpoint_interval == 0:` clause (line ~1843 in current `train.py`).

- [ ] **Step 7.1: Add a helper function for the inline eval**

In `src/elitefurretai/rl/train.py`, add **near the top of the file** (after existing imports — find a spot just below the `from elitefurretai.engine.showdown_server_manager import ...` block, around line 95):

```python
from elitefurretai.rl.analyze import foulplay_eval as _foulplay_eval_mod
from elitefurretai.rl.players import FoulPlayManager
```

Then add a private helper just before the `main()` (or top-level training loop) function definition (search for `def main(` or the outer `if __name__ == "__main__":`):

```python
def _maybe_run_foulplay_eval(
    config,
    updates: int,
    checkpoint_path: str,
    server_ports: list,
) -> None:
    """Inline FoulPlay eval at checkpoint boundary; logs to wandb.

    Caller pre-condition: this is called immediately after
    save_checkpoint, so `checkpoint_path` is the freshly-saved ghost
    checkpoint corresponding to `updates`. Training is paused during
    the eval pass — the duration of the pass is the throughput hit.
    """
    fp = config.foulplay_eval
    if not fp.enabled:
        return
    if updates == 0 or updates % fp.eval_every_n_updates != 0:
        return

    logger.info(
        "[Update %d] Running FoulPlay eval (n_battles=%d, search_time_ms=%d)",
        updates,
        fp.n_battles,
        fp.search_time_ms,
    )

    manager = FoulPlayManager(fp, server_ports=list(server_ports))
    run_tag = format((updates * 1664525 + 1013904223) % 65536, "04x")
    server_url = f"localhost:{server_ports[FoulPlayManager.RUNNER_SERVER_INDEX]}"
    try:
        result = _foulplay_eval_mod.run(
            checkpoint_path=checkpoint_path,
            manager=manager,
            n_battles=fp.n_battles,
            battle_format=config.curriculum.battle_format,
            device=config.hardware.device,
            server_url=server_url,
            agent_team_pool=config.curriculum.opponent_team_pool_path,
            run_tag=run_tag,
        )
        logger.info(
            "[Update %d] FoulPlay eval: win_rate=%.3f wall=%.1fs (model_wins=%d/%d)",
            updates,
            result.win_rate,
            result.wall_time_s,
            result.model_wins,
            result.battles_played,
        )
        wandb.log(
            {
                "eval/foulplay/win_rate": result.win_rate,
                "eval/foulplay/model_wins": result.model_wins,
                "eval/foulplay/foulplay_wins": result.foulplay_wins,
                "eval/foulplay/ties": result.ties,
                "eval/foulplay/n_battles": result.battles_played,
                "eval/foulplay/wall_time_s": result.wall_time_s,
                "eval/foulplay/search_time_ms": fp.search_time_ms,
                "update_step": updates,
            }
        )
    except Exception as exc:
        # Eval failure should NOT kill training. Log and continue.
        logger.exception("[Update %d] FoulPlay eval crashed: %s", updates, exc)
```

(`wandb` and `logger` are already imported at the top of `train.py`.)

- [ ] **Step 7.2: Wire the helper into the checkpoint block**

In `src/elitefurretai/rl/train.py`, find the `save_checkpoint` call inside the `if updates % config.training.checkpoint_interval == 0:` block (line ~1848). Immediately **after** the `opponent_pool.add_ghost(updates, ghost_checkpoint_path)` block and any registry sync (i.e., after the existing `if registry is not None:` block ends around line ~1876), but **before** the broadcast-weights section, add:

```python
                    # Inline FoulPlay eval at the checkpoint boundary.
                    # Training is paused for the duration of the eval.
                    # See planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md.
                    _maybe_run_foulplay_eval(
                        config,
                        updates,
                        ghost_checkpoint_path,
                        server_ports,
                    )
```

(`server_ports` should already be in scope inside this loop — it's set up earlier in `main()` after `launch_showdown_servers`. Verify by grepping `server_ports = ` in train.py.)

- [ ] **Step 7.3: Verify server_ports is in scope**

Run: `grep -n "server_ports" src/elitefurretai/rl/train.py | head -20`

Expected: `server_ports` is defined before the training loop and is in scope at the checkpoint block. If it is not in scope (renamed in train.py, etc.), thread `server_ports` through to the loop via local variable — the grep result will tell you the actual name.

- [ ] **Step 7.4: Run quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/train.py && pyright src/elitefurretai/rl/train.py 2>&1 | tail -40`

Expected: 0 new errors (pyright may already report unrelated errors in train.py; ensure none of the new errors are in your additions).

- [ ] **Step 7.5: Smoke-import test (no actual training)**

Run: `source ../venv/bin/activate && python -c "from elitefurretai.rl.train import _maybe_run_foulplay_eval; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 7.6: Commit**

```bash
git add src/elitefurretai/rl/train.py
git commit -m "$(cat <<'EOF'
feat(rl/train): inline FoulPlay eval at checkpoint boundary

When config.foulplay_eval.enabled and updates % eval_every_n_updates
== 0, the trainer pauses after save_checkpoint, runs a FoulPlay eval
against the freshly-saved ghost checkpoint, and logs win_rate +
n_battles + wall_time_s to wandb under eval/foulplay/*.

Eval failures are caught and logged — they do not kill training.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: YAML wiring + documentation

**Files:**
- Modify: [src/elitefurretai/rl/configs/single_team.yaml](src/elitefurretai/rl/configs/single_team.yaml)
- Modify: [src/elitefurretai/rl/RL.md](src/elitefurretai/rl/RL.md)

- [ ] **Step 8.1: Add `foulplay_eval:` block to single_team.yaml**

In `src/elitefurretai/rl/configs/single_team.yaml`, append at the bottom (the existing blocks are `algorithm`, `portfolio`, `exploration`, `optimizer`, `value_head`, `architecture`, `hardware`, `curriculum`, `training`):

```yaml
foulplay_eval:
  # Disabled by default; enable per-run to add a search-bot ground-truth
  # signal. See planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md.
  # Setup is documented in RL.md ("FoulPlay eval setup").
  enabled: false
  eval_every_n_updates: 50
  n_battles: 100
  search_time_ms: 750
  parallelism: 4
  python_executable: /home/cayman/Repositories/venv-foulplay/bin/python
  team_pool_path: data/teams/gen9vgc2024regg/constrained
  model_probabilistic: false
```

- [ ] **Step 8.2: Add a setup section to RL.md**

In `src/elitefurretai/rl/RL.md`, find a sensible section (e.g. after the existing `vgc-bench external runner` section near line 197 — `grep -n "vgc-bench" src/elitefurretai/rl/RL.md`). Insert a new section:

```markdown
## FoulPlay eval setup

`foul-play-doubles` is the search-based ground-truth eval opponent. It
lives in its own venv (`../venv-foulplay`) because it depends on
`poke-engine-doubles` (a Rust extension) and an older `poke_env` than
EFA's training env. See
`planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md` for the
design rationale.

### One-time setup

```bash
# 1. Ensure the Rust toolchain is installed (poke-engine-doubles builds
#    via cargo on first install).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Create the venv and install dependencies.
python3 -m venv ../venv-foulplay
../venv-foulplay/bin/pip install --upgrade pip==24.2
git clone https://github.com/pmariglia/foul-play-doubles ../foul-play-doubles
../venv-foulplay/bin/pip install -v -r ../foul-play-doubles/requirements.txt

# 3. Make EFA's team pool visible to FoulPlay's load_team(). FoulPlay
#    looks under foul-play-doubles/teams/<format>/. Symlink the EFA pool:
mkdir -p ../foul-play-doubles/teams/gen9vgc2024regg
ln -sft ../foul-play-doubles/teams/gen9vgc2024regg \
    "$(pwd)/data/teams/gen9vgc2024regg/constrained/"*.txt
```

### Manual eval (CLI)

```bash
source ../venv/bin/activate
python -m elitefurretai.rl.analyze.foulplay_eval \
    --checkpoint data/models/rl/<run>/main_model_step_500.pt \
    --battles 50 \
    --python-executable /home/cayman/Repositories/venv-foulplay/bin/python \
    --foulplay-team-pool data/teams/gen9vgc2024regg/constrained \
    --agent-team-pool data/teams/gen9vgc2024regg/constrained \
    --launch-servers
```

### Inline during training

Set in your yaml:

```yaml
foulplay_eval:
  enabled: true
  eval_every_n_updates: 50
  n_battles: 100
  search_time_ms: 750
  python_executable: /home/cayman/Repositories/venv-foulplay/bin/python
  team_pool_path: data/teams/gen9vgc2024regg/constrained
```

The trainer pauses every `eval_every_n_updates` to run one eval pass
against the freshly-saved ghost checkpoint, logs `eval/foulplay/*`
metrics to wandb, then resumes. A 100-battle pass at 750ms search
takes ~8–12 minutes wall time.

**Not** part of the Stage II graduation criterion (currently 60% ×
four other baselines — see
`planning/stage2/2026-05-16-21-30-stage2-graduation-criteria.md`).
Tracked as a reference signal; may be folded into the criterion in a
future stage.
```

- [ ] **Step 8.3: Commit**

```bash
git add src/elitefurretai/rl/configs/single_team.yaml src/elitefurretai/rl/RL.md
git commit -m "$(cat <<'EOF'
docs(rl): add foulplay_eval YAML block and RL.md setup section

Adds the foulplay_eval: block (disabled-by-default) to single_team.yaml
and a complete setup walkthrough to RL.md: Rust toolchain, venv +
foul-play-doubles clone, team-pool symlink, CLI usage, and inline-
training usage.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: End-to-end smoke test

**Files:** none modified (this task verifies the setup against a live Showdown server)

**Pre-requisite:** Tasks 1–8 complete; `../venv-foulplay/` and `../foul-play-doubles/` set up per Task 8.2 instructions; team-pool symlink in place.

- [ ] **Step 9.1: Verify FoulPlay venv is operational**

Run: `../venv-foulplay/bin/python -c "import fp.run_battle; import poke_engine; print('foulplay ok')"`

Expected: prints `foulplay ok`. If it fails, revisit the Task 8.2 setup.

- [ ] **Step 9.2: Launch one Showdown server (if not already running)**

Run in another terminal: `cd ../pokemon-showdown && node pokemon-showdown start --no-security --port 8000`

Expected: server reports `Worker now listening on 0.0.0.0:8000`.

- [ ] **Step 9.3: Run a 2-battle CLI smoke test**

Run:

```bash
source ../venv/bin/activate && \
python -m elitefurretai.rl.analyze.foulplay_eval \
    --checkpoint data/models/supervised/cool-bee-85-finetune_best.pt \
    --battles 2 \
    --search-time-ms 250 \
    --parallelism 2 \
    --device cpu \
    --python-executable /home/cayman/Repositories/venv-foulplay/bin/python \
    --foulplay-team-pool data/teams/gen9vgc2024regg/constrained \
    --agent-team-pool data/teams/gen9vgc2024regg/constrained \
    --num-servers 1 --start-port 8000
```

Expected output:
- FoulPlay runner launches with a PID printed
- 2 battles complete in 1-3 min wall time
- Final line: `FoulPlay eval | battles=2 model_wins=<0-2> foulplay_wins=<0-2> ties=<0> win_rate=<...>% wall_time=<...>s`
- `data/logs/foulplay_runners/runner_FOULPLAY_8000.log` (or similar) contains FoulPlay's per-battle logs

If 2 battles do not complete:
- Inspect `data/logs/foulplay_runners/runner_*.log` for login errors, missing teams, or apply_mods failures.
- Verify the team symlink: `ls ../foul-play-doubles/teams/gen9vgc2024regg/`.
- Check that the model checkpoint loads under `--device cpu` (small/finetuned checkpoint recommended for smoke).

- [ ] **Step 9.4: Run a 5-update inline-training smoke test**

Create a temporary yaml `/tmp/smoke_foulplay.yaml` by copying `single_team.yaml` and overriding (inline-edit):

```yaml
training:
  max_updates: 6
  checkpoint_interval: 5
foulplay_eval:
  enabled: true
  eval_every_n_updates: 5
  n_battles: 2
  search_time_ms: 250
  parallelism: 2
  python_executable: /home/cayman/Repositories/venv-foulplay/bin/python
  team_pool_path: data/teams/gen9vgc2024regg/constrained
```

(Keep the rest of `single_team.yaml` as-is.)

Run: `source ../venv/bin/activate && timeout 1800 python src/elitefurretai/rl/train.py --config /tmp/smoke_foulplay.yaml`

Expected:
- Training runs to update 5
- At update 5, log line `[Update 5] Running FoulPlay eval (n_battles=2, search_time_ms=250)` appears
- 2 battles complete
- Log line `[Update 5] FoulPlay eval: win_rate=X.XXX ...` appears
- wandb console URL is shown; visit it to confirm `eval/foulplay/win_rate` is logged

If eval crashes mid-update, training should continue to update 6 per the `try/except` in `_maybe_run_foulplay_eval`.

- [ ] **Step 9.5: Update the design doc's Updates section**

In `planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md`, replace the "(none yet — pre-implementation)" line in the Updates section with:

```markdown
## Updates

**2026-05-18 (implementation complete)** — All 8 implementation tasks
landed. Smoke test (5-update training run with eval_every_n_updates=5,
n_battles=2, search_time_ms=250) confirmed end-to-end: FoulPlay venv
launches, two battles complete, win rate logged to wandb under
`eval/foulplay/win_rate`. No regressions in the existing four-baseline
graduation eval. Live use at production cadence (`n_battles=100`,
`search_time_ms=750`, `eval_every_n_updates=50`) pending the next
training run.
```

- [ ] **Step 9.6: Commit (the doc update only — no code changes)**

```bash
git add planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md
git commit -m "$(cat <<'EOF'
docs(planning): mark FoulPlay eval implementation complete

Adds the Updates entry to the design doc — smoke test passed (5-update
inline training, 2-battle eval). Production-cadence run pending.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

Run these checks before declaring the plan complete.

**1. Spec coverage** (each spec section → which task implements it):

| Spec section | Task |
|---|---|
| `FoulPlayManager` in players.py | Task 2 |
| `_foulplay_subprocess.py` | Task 3 |
| `foulplay_eval.py` driver | Task 6 |
| `FoulplayEvalConfig` + validation | Task 1 |
| YAML extension | Task 8 |
| RL.md setup section | Task 8 |
| Inline eval in train.py | Task 7 |
| wandb logging `eval/foulplay/*` | Task 7 |
| `evaluate.py --player2 foul_play` | Tasks 4 + 5 |
| Setup symlink for team pool | Task 8 (RL.md) |
| Smoke test | Task 9 |
| Eval NOT in graduation criterion | (no code change — documented in Task 8) |

**2. Placeholder scan:** none — every step has concrete code or commands.

**3. Type consistency:**
- `FoulPlayManager.derive_username` — used at Tasks 2, 6, 7. Same `(base, port) → str` signature throughout.
- `FoulplayEvalConfig` — created in Task 1, consumed by Tasks 2, 6, 7. Same field names.
- `FoulplayEvalResult` (Task 6) vs. `EvalResult` (existing) — deliberately distinct; the eval driver returns its own dataclass with FoulPlay-specific naming (`model_wins`, `foulplay_wins`).
- `PlayerSpec.external_username` — added in Task 4, consumed in Tasks 5, 6.

**4. Dependency check:**
- Task 7 (train.py) depends on Tasks 1, 2, 6 — all complete by that point.
- Task 5 depends on Task 4 (new `external_username` field).
- Task 6 depends on Tasks 1, 2, 4, 5.
- Task 9 depends on every prior task.

**5. Quality-gate cadence:**
- ruff + pyright run after each meaningful code change (Tasks 1, 2, 3, 4, 5, 6, 7).
- pytest runs after each new test addition (Tasks 1, 2, 4, 5, 6).
- End-to-end smoke runs in Task 9.

**6. Frequent commits:** every task ends in a commit. No multi-task batch commits.
