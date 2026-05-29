# -*- coding: utf-8 -*-
"""Tests for the laddering script's pure parsers and credentials loader."""

from __future__ import annotations

from elitefurretai.rl.analyze.laddering import (
    _parse_gxe,
    _parse_player_line,
    _parse_rating_change,
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


def test_parse_rating_change_typical():
    html = (
        "<small>EliteFurret's rating: 1500 &rarr; "
        "<strong>1512</strong></small><br />(+12 for winning)"
    )
    assert _parse_rating_change(html) == (1500, 1512)


def test_parse_rating_change_loss():
    html = (
        "<small>EliteFurret's rating: 1500 &rarr; "
        "<strong>1488</strong></small><br />(-12 for losing)"
    )
    assert _parse_rating_change(html) == (1500, 1488)


def test_parse_rating_change_no_match():
    assert _parse_rating_change("<p>some unrelated raw line</p>") is None


def test_parse_gxe_present():
    html = "<small>Provisional GXE: 54.3%</small>"
    assert _parse_gxe(html) == 54.3


def test_parse_gxe_absent():
    assert _parse_gxe("<p>nothing relevant</p>") is None
