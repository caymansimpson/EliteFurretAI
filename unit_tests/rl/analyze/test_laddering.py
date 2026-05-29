# -*- coding: utf-8 -*-
"""Tests for the laddering script's pure parsers and credentials loader."""

from __future__ import annotations

from elitefurretai.rl.analyze.laddering import (
    _parse_player_line,
)


def test_parse_player_line_rated():
    """A |player| line for a rated battle yields username and integer rating."""
    msg = ["player", "p2", "OpponentUser", "169", "1500"]
    assert _parse_player_line(msg) == ("OpponentUser", 1500)


def test_parse_player_line_unrated():
    """An empty rating field yields None for the rating."""
    msg = ["player", "p2", "OpponentUser", "169", ""]
    assert _parse_player_line(msg) == ("OpponentUser", None)


def test_parse_player_line_missing_rating_field():
    """Older formats omit the rating field entirely."""
    msg = ["player", "p2", "OpponentUser", "169"]
    assert _parse_player_line(msg) == ("OpponentUser", None)


def test_parse_player_line_not_a_player_line():
    """Returns None when the message isn't a |player| message."""
    assert _parse_player_line(["turn", "5"]) is None
