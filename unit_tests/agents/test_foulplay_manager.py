# -*- coding: utf-8 -*-
"""Unit tests for the FoulPlay username helpers.

These pin the username-derivation contract shared with VGCBench. The
subprocess launch itself is exercised by the live eval path
(analyze.analysis_utils._launch_foulplay_subprocess) and the end-to-end
smoke test (planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md,
Task 9) against a real Showdown server.
"""

from __future__ import annotations

from elitefurretai.agents import foulplay_manager


def test_derive_username_appends_port_suffix():
    """
    derive_username always appends ``_<port>``. The caller decides
    whether to invoke it based on ``should_suffix_port``. This test
    pins the formatting contract.
    """
    derived = foulplay_manager.derive_username("FOULPLAY", 8000)
    assert derived == "FOULPLAY_8000"
    assert len(derived) <= 18


def test_derive_username_truncates_long_base():
    """A 20-char base + 5-char suffix is truncated to fit Showdown's 18-char cap."""
    derived = foulplay_manager.derive_username("THIS_IS_TOO_LONG_BASE", 8000)
    assert len(derived) == 18
    assert derived.endswith("_8000")


def test_should_suffix_port_single_server_false():
    """Single-server runs keep the unsuffixed base username."""
    assert foulplay_manager.should_suffix_port(1) is False


def test_should_suffix_port_multi_server_true():
    """Multi-server runs need port suffixes to avoid username collisions."""
    assert foulplay_manager.should_suffix_port(3) is True
