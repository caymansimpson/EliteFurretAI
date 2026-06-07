# -*- coding: utf-8 -*-
import glob
import os
from types import SimpleNamespace

import orjson
import pytest
from poke_env.battle import Observation
from poke_env.player.battle_order import DefaultBattleOrder, ForfeitBattleOrder

from elitefurretai.etl import BattleData, BattleIterator
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.self_play import (
    battle_order_to_input,
    build_self_play_battle_data,
    merge_input_logs,
    teampreview_order_to_input,
)

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "data/fixture/gen9vgc2023regc_logs",
)
FILES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))
IDS = [os.path.basename(f) for f in FILES]


def _load(path):
    with open(path, "rb") as f:
        return BattleData.from_showdown_json(orjson.loads(f.read()))


def _split(input_logs):
    return (
        [s for s in input_logs if s.startswith(">p1")],
        [s for s in input_logs if s.startswith(">p2")],
    )


def _perspective_inputs(bd, perspective):
    it = BattleIterator(bd, perspective=perspective)
    seq = []
    while True:
        try:
            if it.next_input() is None:
                break
        except StopIteration:
            break
        seq.append(it.last_input)
    return seq


def test_teampreview_order_to_input_formats():
    # poke-env may hand either a packed or comma-separated /team order
    assert teampreview_order_to_input("/team 4513", "p1") == ">p1 team 4, 5, 1, 3"
    assert teampreview_order_to_input("/team 4, 5, 1, 3", "p2") == ">p2 team 4, 5, 1, 3"


def test_battle_order_to_input_skips_default_and_forfeit():
    battle = SimpleNamespace(player_role="p1", battle_tag="t")
    assert battle_order_to_input(DefaultBattleOrder(), battle) is None  # type: ignore[arg-type]
    assert battle_order_to_input(ForfeitBattleOrder(), battle) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_last_input_owners_match_input_prefixes(path):
    """The iterator's `last_input_owners` matches the actual ">pX" owners of the
    inputs in its slice, for every real decision (the trailing phantom force switch
    at game end has an out-of-range slice and is skipped)."""
    bd = _load(path)
    it = BattleIterator(bd, perspective="p1", omniscient=True)
    last = list(it.input_nums)
    while not it.battle.finished:
        try:
            it.next()
        except StopIteration:
            break
        if it.input_nums != last and it.input_nums[1] > it.input_nums[0]:
            last = list(it.input_nums)
            start, end = it.input_nums
            if end <= len(bd.input_logs):  # skip the trailing no-input force switch
                expected = [bd.input_logs[i][1:3] for i in range(start, end)]
                assert it.last_input_owners == expected


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_merge_input_logs_raises_on_over_supply(path):
    """Surplus recorded inputs that never get placed must fail loudly rather than
    silently misalign the merged log. (Several extras, since a trailing phantom
    decision can legitimately absorb at most one or two.)"""
    bd = _load(path)
    p1_log, p2_log = _split(bd.input_logs)
    surplus = [">p1 move tackle +1, pass"] * 3
    with pytest.raises(ValueError):
        merge_input_logs(bd, p1_log + surplus, p2_log)


def test_merge_input_logs_raises_on_mid_battle_under_supply():
    """A missing input that is followed by later inputs is caught (here, a dropped p1
    move mid-battle). NOTE: a missing *final* input is NOT detectable -- it is
    indistinguishable from the game's trailing decision; see merge_input_logs."""
    bd = _load(os.path.join(FIXTURE_DIR, "gen9vgc2023regc_anon10.json"))
    p1_log, p2_log = _split(bd.input_logs)
    under = p1_log[:1] + p1_log[2:]  # drop p1's first move, keep teampreview
    with pytest.raises(ValueError):
        merge_input_logs(bd, under, p2_log)


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_merge_input_logs_round_trips(path):
    """Splitting an input log per-player and merging it back reproduces it exactly,
    fully draining both players' streams (the protocol-driven interleaving is correct
    across pivots, one/two-sided force switches and revival blessing)."""
    bd = _load(path)
    original = list(bd.input_logs)
    p1_log, p2_log = _split(original)

    merged = merge_input_logs(bd, p1_log, p2_log)

    assert merged == original
    assert bd.input_logs == original


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_merged_input_logs_replay_with_iterator(path):
    """A BattleData rebuilt via merge_input_logs replays identically for both players."""
    bd = _load(path)
    expected = {p: _perspective_inputs(bd, p) for p in ("p1", "p2")}

    p1_log, p2_log = _split(bd.input_logs)
    merge_input_logs(bd, p1_log, p2_log)

    for perspective in ("p1", "p2"):
        assert _perspective_inputs(bd, perspective) == expected[perspective]


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_battle_order_to_input_round_trips(path):
    """Every real (non-struggle) decision converts back to its original input line:
    last_input -> last_order -> DoubleBattleOrder -> battle_order_to_input."""
    bd = _load(path)
    checked = 0
    for perspective in ("p1", "p2"):
        it = BattleIterator(bd, perspective=perspective)
        while True:
            try:
                if it.next_input() is None:
                    break
            except StopIteration:
                break
            expected = it.last_input
            if expected is None:
                continue
            # struggle/recharge are not in the moveset; they collapse to slot 1 through
            # MDBO and are masked by the training pipeline, so skip them here.
            if "struggle" in expected or "recharge" in expected:
                continue

            if it.last_input_type == MDBO.TEAMPREVIEW:
                raw = "/team " + expected.split(" team ")[1]
                assert teampreview_order_to_input(raw, perspective) == expected
            else:
                dbo = it.last_order().to_double_battle_order(it.battle)  # type: ignore[arg-type]
                assert battle_order_to_input(dbo, it.battle) == expected
            checked += 1
    assert checked > 0


def _faithful_self_play_battle(bd, perspective):
    """A finished DoubleBattle with real teams (reconstructed via the iterator) and
    observations holding every protocol line, as a live poke-env battle keeps them.
    BattleIterator skips "|" separator lines, so we restore the full per-turn events
    to faithfully mirror a real self-play battle's ``observations``."""
    it = BattleIterator(bd, perspective=perspective, omniscient=True)
    while not it.battle.finished:
        try:
            it.next()
        except StopIteration:
            break

    buckets, current, turn = {}, [], 0
    for line in bd.logs:
        current.append(line.split("|"))
        if line.startswith("|turn|"):
            buckets[turn] = Observation(events=current)
            turn += 1
            current = []
    buckets[turn] = Observation(events=current)
    it.battle._observations = buckets
    return it.battle


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_build_self_play_battle_data_end_to_end(path):
    """Full path with no Showdown server: from_self_play reconstructs logs + teams
    from two finished battles, merge_input_logs rebuilds the input log, and the
    result equals the original battle's input log."""
    bd = _load(path)
    original = list(bd.input_logs)
    p1_log, p2_log = _split(original)

    p1_battle = _faithful_self_play_battle(bd, "p1")
    p2_battle = _faithful_self_play_battle(bd, "p2")
    new_bd = build_self_play_battle_data(p1_battle, p2_battle, p1_log, p2_log)

    assert new_bd.input_logs == original
