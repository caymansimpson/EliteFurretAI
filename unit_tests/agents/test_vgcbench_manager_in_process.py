# -*- coding: utf-8 -*-
"""Tests for ``_create_vgc_bench_player`` guards.

The function builds the vgc-bench embedder from whatever poke-env is
installed in the *calling* interpreter. If that poke-env produces a
per-Pokemon observation of a different width than the SB3 checkpoint was
trained against, the load (or first forward pass) blows up with an opaque
``pokemon_proj`` state_dict size mismatch. A poke-env *version string* no
longer discriminates — EFA's main venv and the vgc-bench poke-env fork both
report 0.15.0 yet yield different widths — so the guard validates the actual
invariant: runtime observation width must equal the checkpoint's expected
width.

These tests run from the main venv and mock the SB3/vgc_bench imports, so
they exercise the guard logic without needing venv-vgcbench-bcsp. The real
success path (login → battle) is covered by the eval/smoke training run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

import pytest

import elitefurretai.agents.vgcbench_manager as vgcbench_manager
from elitefurretai.agents.vgcbench_manager import _create_vgc_bench_player

if TYPE_CHECKING:
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration

# embed_len is fixed at 32 in vgc-bench's feature extractor; per-Pokemon
# width = chunk_obs_len + 6 * (embed_len - 1) = chunk_obs_len + 186.
_EMBED_LEN = 32
_WIDTH_CONST = 6 * (_EMBED_LEN - 1)


class _DummyKwargs(TypedDict):
    """Typed kwargs for `_create_vgc_bench_player` so `**` unpacking narrows
    each key to its specific parameter type for static analysis."""

    device: str
    player_config: AccountConfiguration
    server_config: ServerConfiguration
    team: str
    battle_format: str
    checkpoint_path: str


def _make_dummy_kwargs(checkpoint_path: str) -> _DummyKwargs:
    from poke_env.ps_client import AccountConfiguration, ServerConfiguration

    return _DummyKwargs(
        device="cpu",
        player_config=AccountConfiguration("TEST", None),
        server_config=ServerConfiguration("ws://localhost:8000/showdown/websocket", ""),
        team="",
        battle_format="gen9vgc2024regg",
        checkpoint_path=checkpoint_path,
    )


def _fake_import_factory(checkpoint_width: int, runtime_chunk_obs_len: int):
    """Build a fake ``importlib.import_module`` for the load path.

    The fake SB3 returns a policy whose ``features_extractor`` reports
    ``checkpoint_width`` (the value baked into the checkpoint), and the fake
    ``vgc_bench.src.utils`` reports ``runtime_chunk_obs_len`` (what the
    installed poke-env yields). The guard compares the two.
    """

    received_paths: list = []

    class _FakeProj:
        in_features = checkpoint_width

    class _FakeFeatureExtractor:
        embed_len = _EMBED_LEN
        pokemon_proj = _FakeProj()

    class _FakePolicy:
        features_extractor = _FakeFeatureExtractor()

    class _FakeLoaded:
        policy = _FakePolicy()

    class _FakePPO:
        @staticmethod
        def load(path, device):
            received_paths.append(path)
            return _FakeLoaded()

    class _FakePolicyPlayer:
        def __init__(self, **kwargs):
            pass

    def _fake_import(name):
        if name == "stable_baselines3":
            return type("_M", (), {"PPO": _FakePPO})
        if name == "vgc_bench.src.policy_player":
            return type("_M", (), {"PolicyPlayer": _FakePolicyPlayer})
        if name == "vgc_bench.src.utils":
            return type("_M", (), {"chunk_obs_len": runtime_chunk_obs_len})
        raise ImportError(name)

    return _fake_import, received_paths


@pytest.fixture(autouse=True)
def _clear_policy_cache():
    """The function caches loaded policies; isolate tests from each other."""
    vgcbench_manager._VGC_BENCH_POLICY_CACHE.clear()
    yield
    vgcbench_manager._VGC_BENCH_POLICY_CACHE.clear()


def _patches(fake_import, root):
    return (
        patch(
            "elitefurretai.agents.vgcbench_manager._resolve_vgc_bench_root",
            return_value=root,
        ),
        patch(
            "elitefurretai.agents.vgcbench_manager.importlib.import_module",
            side_effect=fake_import,
        ),
    )


def test_refuses_on_observation_width_mismatch(tmp_path):
    """A poke-env that yields the wrong width must fail fast and clearly.

    The previous code guarded on a poke-env version prefix, which can no
    longer distinguish the fork (correct) from EFA's poke-env (wrong) — both
    are 0.15.0. The guard now compares the real widths and turns the opaque
    SB3 state_dict mismatch into an actionable error.
    """
    ckpt = tmp_path / "vgc-bench-sb3-model.zip"
    ckpt.write_bytes(b"")
    # checkpoint expects 764; installed poke-env yields 762 (chunk 576).
    fake_import, _ = _fake_import_factory(checkpoint_width=764, runtime_chunk_obs_len=576)
    root_patch, import_patch = _patches(fake_import, tmp_path)

    with root_patch, import_patch:
        with pytest.raises(RuntimeError, match="observation-width mismatch"):
            _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=str(ckpt)))


def test_error_message_points_to_vgcbench_manager(tmp_path):
    """The mismatch error must include actionable redirection."""
    ckpt = tmp_path / "vgc-bench-sb3-model.zip"
    ckpt.write_bytes(b"")
    fake_import, _ = _fake_import_factory(checkpoint_width=764, runtime_chunk_obs_len=576)
    root_patch, import_patch = _patches(fake_import, tmp_path)

    with root_patch, import_patch:
        with pytest.raises(RuntimeError) as exc_info:
            _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=str(ckpt)))

    message = str(exc_info.value)
    assert "VGCBenchManager" in message
    assert "venv-vgcbench" in message


def test_missing_checkpoint_raises_clear_error(tmp_path):
    """A missing file gives FileNotFoundError before SB3 is invoked.

    Protects against the legacy ``vgc-bench-sb3-model.zip.zip`` symptom from
    SB3's relative-path fallback search.
    """
    bogus_path = str(tmp_path / "does_not_exist.zip")
    with pytest.raises(FileNotFoundError, match="vgc-bench checkpoint not found"):
        _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=bogus_path))


def test_checkpoint_path_resolved_to_absolute_before_load(tmp_path, monkeypatch):
    """Relative ``checkpoint_path`` must reach SB3.PPO.load as absolute.

    Regression guard for the ``.zip.zip`` failure: a relative path plus
    ``_temporary_cwd(vgc_bench_root)`` made SB3 look in the wrong directory.
    Widths are matched here so the call proceeds to (fake) construction.
    """
    ckpt = tmp_path / "vgc-bench-sb3-model.zip"
    ckpt.write_bytes(b"")
    monkeypatch.chdir(tmp_path.parent)
    relative_path = os.path.relpath(str(ckpt))

    # checkpoint 764 and chunk 578 -> runtime 764: widths match, guard passes.
    fake_import, received_paths = _fake_import_factory(
        checkpoint_width=764, runtime_chunk_obs_len=764 - _WIDTH_CONST
    )
    root_patch, import_patch = _patches(fake_import, tmp_path)

    with root_patch, import_patch:
        _create_vgc_bench_player(**_make_dummy_kwargs(checkpoint_path=relative_path))

    assert received_paths, "PPO.load was not called"
    assert os.path.isabs(received_paths[0]), (
        f"PPO.load received non-absolute path {received_paths[0]!r}; "
        f"this is the regression that caused the `.zip.zip` failure"
    )
    assert received_paths[0] == os.path.abspath(relative_path)
