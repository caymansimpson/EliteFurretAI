# -*- coding: utf-8 -*-
"""Tests for ``player_factory.parse_player_specification``.

The factories themselves are not invoked in unit tests because
constructing a poke-env ``Player`` requires a running Showdown server.
We verify parsing + canonicalization here; end-to-end construction is
covered by the eval-script integration smoke run.
"""

from __future__ import annotations

import pytest

from elitefurretai.rl.analyze.player_factory import (
    PlayerSpecification,
    canonicalize_baseline,
    parse_player_specification,
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


def test_parse_baseline_returns_specification_with_params():
    specification = parse_player_specification(
        "simple_heuristic", device="cpu", battle_format="gen9vgc2024regg"
    )
    assert isinstance(specification, PlayerSpecification)
    assert specification.kind == "baseline"
    assert specification.name == "simple_heuristic"
    assert specification.user_tag == "SHP"
    assert specification.params["canonical"] == "simple_heuristic"
    assert specification.params["battle_format"] == "gen9vgc2024regg"


def test_parse_alias_returns_canonical_name():
    specification = parse_player_specification(
        "maxdamage", device="cpu", battle_format="gen9vgc2024regg"
    )
    # Raw is preserved as-given (for logging); name canonicalizes.
    assert specification.raw == "maxdamage"
    assert specification.name == "max_damage"
    assert specification.user_tag == "MD"


def test_parse_vgc_bench_is_external():
    """vgc_bench routes to ``kind="external"`` carrying subprocess params.

    In-process construction is broken (poke-env version drift vs the
    SB3 checkpoint), so the specification carries the subprocess args needed
    by ``launch_external_player`` (called by the worker once Showdown
    is up). We just verify the params are populated here — actually
    launching requires the vgc-bench venv and a live Showdown server.
    """
    specification = parse_player_specification(
        "vgc_bench", device="cpu", battle_format="gen9vgc2024regg"
    )
    assert specification.kind == "external"
    assert specification.name == "vgc_bench"
    assert specification.user_tag == "VGB"
    assert specification.params["battle_format"] == "gen9vgc2024regg"
    assert "checkpoint_path" in specification.params
    assert "team_file" in specification.params
    assert "python_executable" in specification.params


def test_parse_model_path(tmp_path):
    # File must exist for the path branch to fire.
    ckpt = tmp_path / "fake_model.pt"
    ckpt.write_bytes(b"")  # contents irrelevant — we don't invoke build_player
    specification = parse_player_specification(
        str(ckpt), device="cpu", battle_format="gen9vgc2024regg"
    )
    assert specification.kind == "model"
    assert specification.name == "fake_model"
    assert specification.user_tag == "MDL"
    assert specification.raw == str(ckpt)
    assert specification.params["path"] == str(ckpt)
    assert specification.params["device"] == "cpu"
    assert specification.params["battle_format"] == "gen9vgc2024regg"


def test_parse_unknown_raises():
    with pytest.raises(ValueError, match="Could not resolve player specification"):
        parse_player_specification(
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
    specification = parse_player_specification(
        str(ckpt), device="cpu", battle_format="gen9vgc2024regg"
    )
    assert specification.kind == "model"
    assert specification.name == "random"
