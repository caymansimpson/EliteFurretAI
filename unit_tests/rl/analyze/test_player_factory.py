# -*- coding: utf-8 -*-
"""Tests for ``player_factory.parse_player_spec``.

The factories themselves are not invoked in unit tests because
constructing a poke-env ``Player`` requires a running Showdown server.
We verify parsing + canonicalization here; end-to-end construction is
covered by the eval-script integration smoke run.
"""

from __future__ import annotations

import pytest

from elitefurretai.rl.analyze.player_factory import (
    PlayerSpec,
    canonicalize_baseline,
    parse_player_spec,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("max_damage", "max_damage"),
        ("MAX_DAMAGE", "max_damage"),
        ("max-damage", "max_damage"),
        ("maxdamage", "max_damage"),
        ("max_base_power", "max_base_power"),
        ("maxbasepower", "max_base_power"),
        ("simple_heuristic", "simple_heuristic"),
        ("simpleheuristic", "simple_heuristic"),
        ("simpleheuristics", "simple_heuristic"),
        ("shp", "simple_heuristic"),
        ("vgc_bench", "vgc_bench"),
        ("vgcbench", "vgc_bench"),
        ("random", "random"),
        ("RND", None),  # Tag, not a name
        ("totally_made_up", None),
    ],
)
def test_canonicalize_baseline_handles_aliases_and_casing(raw, expected):
    assert canonicalize_baseline(raw) == expected


def test_parse_baseline_returns_spec_with_factory():
    spec = parse_player_spec(
        "simple_heuristic", device="cpu", battle_format="gen9vgc2024regg"
    )
    assert isinstance(spec, PlayerSpec)
    assert spec.kind == "baseline"
    assert spec.name == "simple_heuristic"
    assert spec.user_tag == "SHP"
    assert callable(spec.factory)


def test_parse_alias_returns_canonical_name():
    spec = parse_player_spec("maxdamage", device="cpu", battle_format="gen9vgc2024regg")
    # Raw is preserved as-given (for logging); name canonicalizes.
    assert spec.raw == "maxdamage"
    assert spec.name == "max_damage"
    assert spec.user_tag == "MD"


def test_parse_vgc_bench_is_external():
    """vgc_bench routes to ``kind="external"`` with a ``launch_external`` closure.

    In-process construction is broken (poke-env version drift vs the
    SB3 checkpoint), so the spec instead carries a launcher that will
    spawn ``_vgcbench_subprocess.py`` under the vgc-bench venv. The
    closure isn't invoked here — that requires the venv and a live
    Showdown server.
    """
    spec = parse_player_spec("vgc_bench", device="cpu", battle_format="gen9vgc2024regg")
    assert spec.kind == "external"
    assert spec.name == "vgc_bench"
    assert spec.user_tag == "VGB"
    assert spec.factory is None
    assert spec.launch_external is not None
    assert callable(spec.launch_external)


def test_parse_model_path(tmp_path):
    # File must exist for the path branch to fire.
    ckpt = tmp_path / "fake_model.pt"
    ckpt.write_bytes(b"")  # contents irrelevant — we don't invoke factory
    spec = parse_player_spec(str(ckpt), device="cpu", battle_format="gen9vgc2024regg")
    assert spec.kind == "model"
    assert spec.name == "fake_model"
    assert spec.user_tag == "MDL"
    assert spec.raw == str(ckpt)


def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="Could not resolve player spec"):
        parse_player_spec(
            "definitely_not_a_baseline",
            device="cpu",
            battle_format="gen9vgc2024regg",
        )


def test_path_takes_precedence_over_baseline_name(tmp_path):
    """A checkpoint named ``random.pt`` resolves to a model, not the random baseline.

    The path branch fires first; only files that don't exist fall through
    to baseline-name resolution. This protects against a real model
    checkpoint colliding with a baseline name.
    """
    ckpt = tmp_path / "random.pt"
    ckpt.write_bytes(b"")
    spec = parse_player_spec(str(ckpt), device="cpu", battle_format="gen9vgc2024regg")
    assert spec.kind == "model"
    assert spec.name == "random"
