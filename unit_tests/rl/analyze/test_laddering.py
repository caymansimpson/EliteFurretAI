# -*- coding: utf-8 -*-
"""Tests for the laddering script's pure parsers and credentials loader."""

from __future__ import annotations

from typing import Any, cast

from elitefurretai.rl.analyze.laddering import (
    LadderRecord,
    _finalize_record,
    _parse_gxe,
    _parse_player_line,
    _parse_rating_change,
    _parse_replay_url,
    _update_record,
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


def test_parse_replay_url_present():
    html = (
        '<a class="ilink" '
        'href="https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456">'
        "Open replay</a>"
    )
    assert (
        _parse_replay_url(html)
        == "https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456"
    )


def test_parse_replay_url_absent():
    assert _parse_replay_url("<p>nothing</p>") is None


def test_update_record_with_player_line_sets_opponent_and_pre_rating():
    """A |player| line for the opponent's slot populates opponent + pre_rating."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        ["player", "p2", "OpponentUser", "169", "1500"],
        agent_role="p1",
    )
    assert record.opponent == "OpponentUser"
    assert record.pre_rating == 1500


def test_update_record_ignores_own_player_line():
    """The agent's own |player| line shouldn't overwrite opponent/pre_rating."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        ["player", "p1", "EliteFurret", "169", "1500"],
        agent_role="p1",
    )
    assert record.opponent == ""
    assert record.pre_rating is None


def test_update_record_with_raw_rating_change():
    """A |raw| rating-change line populates pre_rating + post_rating."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        [
            "raw",
            "<small>EliteFurret's rating: 1500 &rarr; <strong>1512</strong></small>",
        ],
        agent_role="p1",
    )
    assert record.pre_rating == 1500
    assert record.post_rating == 1512


def test_update_record_with_raw_gxe():
    """A |raw| GXE line populates the gxe field."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(record, ["raw", "<small>GXE: 54.3%</small>"], agent_role="p1")
    assert record.gxe == 54.3


def test_update_record_with_raw_replay_url():
    """A |raw| replay-link line populates replay_url."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        [
            "raw",
            '<a class="ilink" '
            'href="https://replay.pokemonshowdown.com/gen9vgc2024regg-1">x</a>',
        ],
        agent_role="p1",
    )
    assert record.replay_url == "https://replay.pokemonshowdown.com/gen9vgc2024regg-1"


class _FakeBattle:
    def __init__(self, won, lost, turn, battle_tag):
        self.won = won
        self.lost = lost
        self.turn = turn
        self.battle_tag = battle_tag


def test_finalize_record_win():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(record, cast(Any, _FakeBattle(True, False, 18, "battle-x-1")))
    assert record.outcome == "win"
    assert record.final_turn == 18
    assert record.timestamp is not None


def test_finalize_record_loss():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(record, cast(Any, _FakeBattle(False, True, 25, "battle-x-1")))
    assert record.outcome == "loss"
    assert record.final_turn == 25


def test_finalize_record_tie():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(record, cast(Any, _FakeBattle(False, False, 60, "battle-x-1")))
    assert record.outcome == "tie"
