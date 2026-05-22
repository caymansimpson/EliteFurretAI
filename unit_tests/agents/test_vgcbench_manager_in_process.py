# -*- coding: utf-8 -*-
"""Tests for ``_create_vgc_bench_player`` guards.

The function is broken from EFA's main venv (poke-env 0.15.x) because
the SB3 checkpoint was trained against poke-env 0.11.x and the
embedder dimensions drifted. These tests pin the two guards that turn
that into a clean actionable error instead of an opaque state_dict
mismatch or a confusing ``vgc-bench-sb3-model.zip.zip`` path failure.

They run from the main venv (where 0.11.x is *not* installed) so they
exercise the failure-mode paths; the success path requires
venv-vgcbench and is covered by the eval acceptance run, not by these
unit tests.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

import pytest

from elitefurretai.agents.vgcbench_manager import _create_vgc_bench_player

if TYPE_CHECKING:
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration


class _DummyKwargs(TypedDict):
    """Typed kwargs for `_create_vgc_bench_player` so `**` unpacking narrows
    each key to its specific parameter type for static analysis."""

    device: str
    player_config: AccountConfiguration
    server_config: ServerConfiguration
    team: str
    battle_format: str
    checkpoint_path: str


def _make_dummy_kwargs(
    checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip",
) -> _DummyKwargs:
    """Minimal kwargs for the function — we exercise pre-load guards only."""
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration

    return _DummyKwargs(
        device="cpu",
        player_config=AccountConfiguration("TEST", None),
        server_config=ServerConfiguration("ws://localhost:8000/showdown/websocket", ""),
        team="",
        battle_format="gen9vgc2024regg",
        checkpoint_path=checkpoint_path,
    )


def test_refuses_to_run_from_wrong_poke_env_version():
    """From EFA's main venv (poke-env 0.15.x) the function must refuse.

    A previous version of the code let the call proceed and crashed
    deep inside SB3.PPO.load with a state_dict size mismatch that's
    nearly impossible to diagnose without knowing the version-drift
    backstory. The guard turns that into a clear error pointing at
    VGCBenchManager.
    """
    runtime = importlib.metadata.version("poke-env")
    if runtime.startswith("0.11."):
        pytest.skip(
            "Running under poke-env 0.11.x — the guard would not fire here; "
            "this test only exercises the mismatched-venv error path."
        )

    with pytest.raises(RuntimeError, match="requires poke-env 0.11.x"):
        _create_vgc_bench_player(**_make_dummy_kwargs())


def test_error_message_points_to_vgcbench_manager():
    """The guard's error must include actionable redirection.

    A bare 'version mismatch' would force the user to spelunk the
    codebase for the right alternative. Mention VGCBenchManager and
    the venv-vgcbench path explicitly.
    """
    runtime = importlib.metadata.version("poke-env")
    if runtime.startswith("0.11."):
        pytest.skip("Guard does not fire under poke-env 0.11.x")

    with pytest.raises(RuntimeError) as exc_info:
        _create_vgc_bench_player(**_make_dummy_kwargs())

    message = str(exc_info.value)
    assert "VGCBenchManager" in message
    assert "venv-vgcbench" in message


def test_missing_checkpoint_raises_clear_error_under_correct_venv(tmp_path):
    """When the version guard passes, a missing file gives FileNotFoundError.

    We can't actually load under venv-vgcbench from a main-venv pytest,
    but we can patch the version check to pass and verify the next
    guard (file existence) fires before SB3 is invoked. This protects
    the user from the legacy ``vgc-bench-sb3-model.zip.zip`` symptom.
    """
    bogus_path = str(tmp_path / "does_not_exist.zip")
    # Patch the version check so we hit the file-existence branch.
    with patch(
        "elitefurretai.agents.vgcbench_manager.importlib.metadata.version",
        return_value="0.11.0",
    ):
        with pytest.raises(FileNotFoundError, match="vgc-bench checkpoint not found"):
            _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=bogus_path))


def test_checkpoint_path_resolved_to_absolute_before_load(tmp_path, monkeypatch):
    """Relative ``checkpoint_path`` must reach SB3.PPO.load as absolute.

    Regression guard for the ``.zip.zip`` failure: under the old code,
    a relative path plus ``_temporary_cwd(vgc_bench_root)`` made SB3
    look in the wrong directory, fall through to its ``path + ".zip"``
    fallback, and surface ``...zip.zip`` not-found errors.

    We patch the version guard and SB3.PPO.load to record the path it
    received, then assert it was absolute.
    """
    # Plant a fake checkpoint so the existence check passes.
    ckpt = tmp_path / "vgc-bench-sb3-model.zip"
    ckpt.write_bytes(b"")
    # Run from a working dir other than tmp_path so a relative path
    # would resolve differently if CWD changed.
    monkeypatch.chdir(tmp_path.parent)
    relative_path = os.path.relpath(str(ckpt))

    received_paths = []

    class _FakePolicy:
        policy = object()

    class _FakePPO:
        @staticmethod
        def load(path, device):
            received_paths.append(path)
            return _FakePolicy()

    class _FakePolicyPlayer:
        def __init__(self, **kwargs):
            pass

    with (
        patch(
            "elitefurretai.agents.vgcbench_manager.importlib.metadata.version",
            return_value="0.11.0",
        ),
        patch(
            "elitefurretai.agents.vgcbench_manager._resolve_vgc_bench_root",
            return_value=tmp_path,
        ),
        patch(
            "elitefurretai.agents.vgcbench_manager.importlib.import_module"
        ) as fake_import,
    ):
        # Two import calls: stable_baselines3, then vgc_bench.src.policy_player.
        def _fake_import(name):
            if name == "stable_baselines3":
                fake_mod = type("_M", (), {"PPO": _FakePPO})
                return fake_mod
            if name == "vgc_bench.src.policy_player":
                fake_mod = type("_M", (), {"PolicyPlayer": _FakePolicyPlayer})
                return fake_mod
            raise ImportError(name)

        fake_import.side_effect = _fake_import

        _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=relative_path))

    assert received_paths, "PPO.load was not called"
    assert os.path.isabs(received_paths[0]), (
        f"PPO.load received non-absolute path {received_paths[0]!r}; "
        f"this is the regression that caused the `.zip.zip` failure"
    )
    assert received_paths[0] == os.path.abspath(relative_path)
