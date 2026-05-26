"""FoulPlayManager — subprocess lifecycle for the external foul-play-doubles bot.

foul-play-doubles depends on ``poke-engine-doubles`` (a Rust extension) and
an older ``poke_env`` than EFA's training env, so it runs as a subprocess in
its own venv (``../venv-foulplay/``). This module owns the EFA-side
subprocess lifecycle. The subprocess entry point lives at
``agents/_foulplay_subprocess.py``. EFA-side code interacts with FoulPlay
only by Showdown username.

Unlike :class:`VGCBenchManager` (training-lifetime), ``FoulPlayManager`` is
short-lived: instantiated per eval pass and per battle format, ``launch()``
spawns the subprocess, eval battles run, ``shutdown()`` tears it down. The
8-core × 750 ms search cost saturates the machine during eval — see
planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import ClassVar, List, TextIO, Union

from elitefurretai.rl.config import FoulplayEvalConfig

logger = logging.getLogger(__name__)


class FoulPlayManager:
    """Launcher for the external foul-play-doubles bot.

    One manager instance covers a single (battle_format, team_pool) eval
    cycle. The multi-format eval driver constructs and tears down one
    manager per active format.

    Usage
    -----
        manager = FoulPlayManager(config, battle_format, team_pool, server_ports)
        manager.launch()       # spawns subprocess, populates .usernames
        # ... eval runs ...
        manager.shutdown()     # SIGTERM the subprocess, close log handles

    Constants mirror :class:`VGCBenchManager` (username derivation,
    runner-on-server-0 layout) so callers can reason about both
    external opponents the same way.
    """

    SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/agents/_foulplay_subprocess.py"
    USERNAMES: ClassVar[List[str]] = ["FOULPLAY"]
    WAIT_FOR_SERVER_TIMEOUT_S: ClassVar[float] = 180.0
    LOG_DIR: ClassVar[str] = "data/logs/foulplay_runners"
    LOG_TO_FILES: ClassVar[bool] = True
    ACCEPT_OPEN_TEAM_SHEET: ClassVar[bool] = False
    STARTUP_WAIT_S: ClassVar[float] = 10.0
    # Single-runner-on-first-server layout, matching VGCBenchManager. The
    # subprocess saturates the machine during search; running more than
    # one would just contend for the same 8 cores.
    RUNNER_SERVER_INDEX: ClassVar[int] = 0

    def __init__(
        self,
        config: FoulplayEvalConfig,
        battle_format: str,
        team_pool_path: str,
        server_ports: List[int],
    ) -> None:
        self._config = config
        self._battle_format = battle_format
        self._team_pool_path = team_pool_path
        self._server_ports = list(server_ports)
        self._processes: List[subprocess.Popen] = []
        self._log_files: List[TextIO] = []
        self._usernames: List[str] = []

    @staticmethod
    def derive_username(base_username: str, server_port: int) -> str:
        """Generate a server-scoped runner username (Showdown max length is 18).

        Same contract as :meth:`VGCBenchManager.derive_username`.
        """
        suffix = f"_{server_port}"
        max_base_len = max(1, 18 - len(suffix))
        return f"{base_username[:max_base_len]}{suffix}"

    @classmethod
    def should_suffix_port(cls, num_servers: int) -> bool:
        """Whether to append ``_<port>`` to the base username."""
        return num_servers > 1

    @property
    def usernames(self) -> List[str]:
        """Usernames the subprocess(es) logged into Showdown as.

        Empty until :meth:`launch` returns.
        """
        return list(self._usernames)

    def launch(self) -> List[str]:
        """Spawn the FoulPlay subprocess and return the usernames it will log in as.

        Raises ``RuntimeError`` if launch() was called twice without an
        intervening shutdown() — the manager is single-shot per cycle.
        """
        if self._processes:
            raise RuntimeError(
                "FoulPlayManager.launch() called twice without shutdown()"
            )

        assert self._config.python_executable is not None, (
            "FoulplayEvalConfig.python_executable must be set when launching "
            "FoulPlayManager"
        )

        if self.LOG_TO_FILES:
            os.makedirs(self.LOG_DIR, exist_ok=True)

        append_port = self.should_suffix_port(len(self._server_ports))
        runner_port = self._server_ports[self.RUNNER_SERVER_INDEX]
        usernames: List[str] = []

        for base_username in self.USERNAMES:
            actual_username = (
                self.derive_username(base_username, runner_port)
                if append_port
                else base_username
            )
            sanitized = actual_username.replace("/", "_")

            log_path = "<disabled>"
            log_handle: Union[TextIO, int]
            if self.LOG_TO_FILES:
                log_path = os.path.join(
                    self.LOG_DIR, f"runner_{sanitized}_{runner_port}.log"
                )
                log_handle = open(log_path, "a", encoding="utf-8")
                self._log_files.append(log_handle)
            else:
                log_handle = subprocess.DEVNULL

            command = [
                self._config.python_executable,
                self.SUBPROCESS_SCRIPT,
                "--username",
                actual_username,
                "--server",
                f"localhost:{runner_port}",
                "--battle-format",
                self._battle_format,
                "--n-challenges",
                str(self._config.n_battles_per_format),
                "--team-list-dir",
                self._team_pool_path,
                "--search-time-ms",
                str(self._config.search_time_ms),
                "--parallelism",
                str(self._config.parallelism),
                "--wait-for-server-timeout",
                str(self.WAIT_FOR_SERVER_TIMEOUT_S),
            ]
            if self.ACCEPT_OPEN_TEAM_SHEET:
                command.append("--accept-open-team-sheet")

            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._processes.append(process)
            usernames.append(actual_username)
            print(
                "✓ Launched FoulPlay runner "
                f"'{actual_username}' on localhost:{runner_port} "
                f"(PID: {process.pid}) log={log_path}"
            )

        self._usernames = usernames
        return list(self._usernames)

    def shutdown(self) -> None:
        """Terminate the subprocess and close log handles.

        Safe to call from a ``finally`` even if :meth:`launch` was never
        invoked or failed partway. Matches :class:`VGCBenchManager.shutdown`
        in shape so eval-driver cleanup paths are uniform.
        """
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass

        for log_handle in self._log_files:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

        self._processes.clear()
        self._log_files.clear()
        self._usernames.clear()
