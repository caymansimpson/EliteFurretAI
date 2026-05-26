# -*- coding: utf-8 -*-
"""Unit tests for FoulPlayManager (subprocess construction, lifecycle).

These mock subprocess.Popen so they run without ../venv-foulplay installed.
The end-to-end smoke test (planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md,
Task 9) verifies the real subprocess flow against a live Showdown server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from elitefurretai.agents.foulplay_manager import FoulPlayManager
from elitefurretai.rl.config import FoulplayEvalConfig


def _make_config(tmp_path) -> FoulplayEvalConfig:
    """Build a fully valid FoulplayEvalConfig with disk paths under tmp_path."""
    fake_python = tmp_path / "venv-foulplay" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("")

    return FoulplayEvalConfig(
        enabled=True,
        python_executable=str(fake_python),
        n_battles_per_format=10,
        search_time_ms=500,
        parallelism=2,
    )


def _make_team_pool(tmp_path) -> str:
    """Create a fake FoulPlay team-pool directory under tmp_path."""
    pool = tmp_path / "teams"
    pool.mkdir()
    (pool / "team_a.txt").write_text("placeholder team a")
    return str(pool)


# ── derive_username / should_suffix_port ──────────────────────────────────────


def test_derive_username_appends_port_suffix():
    """
    derive_username always appends ``_<port>``. The manager decides
    whether to invoke it based on ``should_suffix_port``. This test
    pins the formatting contract.
    """
    derived = FoulPlayManager.derive_username("FOULPLAY", 8000)
    assert derived == "FOULPLAY_8000"
    assert len(derived) <= 18


def test_derive_username_truncates_long_base():
    """A 20-char base + 5-char suffix is truncated to fit Showdown's 18-char cap."""
    derived = FoulPlayManager.derive_username("THIS_IS_TOO_LONG_BASE", 8000)
    assert len(derived) == 18
    assert derived.endswith("_8000")


def test_should_suffix_port_single_server_false():
    """Single-server runs keep the unsuffixed base username."""
    assert FoulPlayManager.should_suffix_port(1) is False


def test_should_suffix_port_multi_server_true():
    """Multi-server runs need port suffixes to avoid username collisions."""
    assert FoulPlayManager.should_suffix_port(3) is True


# ── launch() argv + lifecycle ────────────────────────────────────────────────


def test_launch_builds_expected_argv(tmp_path):
    """
    subprocess.Popen is called once with a command containing every
    FoulPlay-side flag the subprocess script expects. Format and pool
    path are supplied at construction time (one manager per format).
    """
    config = _make_config(tmp_path)
    pool = _make_team_pool(tmp_path)
    manager = FoulPlayManager(
        config=config,
        battle_format="gen9vgc2024regg",
        team_pool_path=pool,
        server_ports=[8000],
    )

    with patch("elitefurretai.agents.foulplay_manager.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        usernames = manager.launch()

    assert usernames == ["FOULPLAY"]  # single server → no suffix
    assert mock_popen.call_count == 1
    argv = mock_popen.call_args.args[0]

    assert argv[0] == config.python_executable
    assert argv[1].endswith("_foulplay_subprocess.py")
    assert "--username" in argv and argv[argv.index("--username") + 1] == "FOULPLAY"
    assert argv[argv.index("--server") + 1] == "localhost:8000"
    assert argv[argv.index("--battle-format") + 1] == "gen9vgc2024regg"
    assert argv[argv.index("--n-challenges") + 1] == "10"
    assert argv[argv.index("--search-time-ms") + 1] == "500"
    assert argv[argv.index("--parallelism") + 1] == "2"
    assert argv[argv.index("--team-list-dir") + 1] == pool


def test_launch_appends_port_suffix_when_multiple_servers(tmp_path):
    """
    Multi-server setups derive a port-suffixed username. The runner
    still lands on the first server (RUNNER_SERVER_INDEX = 0); the
    suffix only avoids username collisions across concurrent runs.
    """
    config = _make_config(tmp_path)
    pool = _make_team_pool(tmp_path)
    manager = FoulPlayManager(
        config=config,
        battle_format="gen9vgc2024regg",
        team_pool_path=pool,
        server_ports=[8000, 8001, 8002],
    )

    with patch("elitefurretai.agents.foulplay_manager.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        usernames = manager.launch()

    assert usernames == ["FOULPLAY_8000"]


def test_double_launch_raises(tmp_path):
    """
    Manager is single-shot per cycle (eval-pass-scoped lifetime).
    Calling launch() twice without an intervening shutdown() is a bug.
    """
    config = _make_config(tmp_path)
    pool = _make_team_pool(tmp_path)
    manager = FoulPlayManager(
        config=config,
        battle_format="gen9vgc2024regg",
        team_pool_path=pool,
        server_ports=[8000],
    )

    with patch("elitefurretai.agents.foulplay_manager.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock(pid=12345, poll=lambda: None)
        manager.launch()
        with pytest.raises(RuntimeError, match="launch.*twice"):
            manager.launch()


def test_shutdown_without_launch_is_safe(tmp_path):
    """
    The eval driver calls shutdown() in a `finally`. Manager must not
    error when launch() was never invoked or partially failed.
    """
    config = _make_config(tmp_path)
    pool = _make_team_pool(tmp_path)
    manager = FoulPlayManager(
        config=config,
        battle_format="gen9vgc2024regg",
        team_pool_path=pool,
        server_ports=[8000],
    )
    manager.shutdown()  # should not raise


def test_shutdown_terminates_running_process(tmp_path):
    """
    A running process is terminated; a finished process is left alone.
    The manager's _processes list is cleared regardless.
    """
    config = _make_config(tmp_path)
    pool = _make_team_pool(tmp_path)
    manager = FoulPlayManager(
        config=config,
        battle_format="gen9vgc2024regg",
        team_pool_path=pool,
        server_ports=[8000],
    )

    mock_process = MagicMock(pid=12345)
    mock_process.poll.return_value = None  # still running

    with patch("elitefurretai.agents.foulplay_manager.subprocess.Popen") as mock_popen:
        mock_popen.return_value = mock_process
        manager.launch()

    manager.shutdown()
    mock_process.terminate.assert_called_once()
    assert manager._processes == []
