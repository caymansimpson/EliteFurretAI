# -*- coding: utf-8 -*-
"""Tests for ``TrajectoryCollector`` — buffering, replay sampling, flush.

The ``RecordingModelPlayer`` thin wrapper isn't directly exercised
here (constructing a poke-env Player requires a live Showdown
connection); the smoke test in Phase B7 covers it end-to-end. These
unit tests use fake ``battle`` objects with the minimum interface the
collector touches.
"""

from __future__ import annotations

import gzip
import math
import os

import numpy as np

from elitefurretai.rl.analyze.eval_collector import TrajectoryCollector
from elitefurretai.rl.analyze.eval_schema import (
    read_battles,
    read_turns,
)

# ─── Fake battle objects ────────────────────────────────────────────


class _FakeMon:
    def __init__(self, hp_frac: float, fainted: bool = False):
        self.current_hp_fraction = hp_frac
        self.fainted = fainted


class _FakeBattle:
    """Mimics the poke-env Battle interface that the collector reads."""

    def __init__(
        self,
        *,
        battle_tag: str,
        turn: int = 1,
        won: "bool | None" = None,
        team_alive: int = 6,
        opp_alive: int = 6,
        replay_log: str = ">battle-x\n|move|p1a: Foo|Tackle",
    ):
        self.battle_tag = battle_tag
        self.turn = turn
        self.won = won
        self.team = {f"a{i}": _FakeMon(1.0) for i in range(team_alive)}
        self.team.update(
            {f"a{i}": _FakeMon(0.0, fainted=True) for i in range(team_alive, 6)}
        )
        self.opponent_team = {f"b{i}": _FakeMon(1.0) for i in range(opp_alive)}
        self.opponent_team.update(
            {f"b{i}": _FakeMon(0.0, fainted=True) for i in range(opp_alive, 6)}
        )
        self._replay_log = replay_log

    def _build_replay_log(self) -> str:
        return self._replay_log


_SAMPLE_TEAM_AGENT = """\
Calyrex-Shadow @ Life Orb
Ability: As One (Spectrier)
Tera Type: Normal
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Astral Barrage
- Psychic
- Nasty Plot
- Protect
"""

_SAMPLE_TEAM_OPP = """\
Urshifu @ Focus Sash
Ability: Unseen Fist
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Wicked Blow
- Close Combat
- Detect
- Sucker Punch
"""


def _make_collector(tmp_path, *, replay_rate: float = 0.0, seed: int = 0):
    return TrajectoryCollector(
        eval_run_id="test_run",
        agent_ckpt="data/models/foo.pt",
        agent_team_str=_SAMPLE_TEAM_AGENT,
        opp_team_str=_SAMPLE_TEAM_OPP,
        opp_player_kind="baseline",
        opp_player_name="simple_heuristic",
        battle_format="gen9vgc2024regg",
        run_dir=str(tmp_path),
        worker_id=0,
        replay_sample_rate=replay_rate,
        seed=seed,
    )


# ─── record_turn ────────────────────────────────────────────────────


def test_record_turn_appends_one_row(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="battle-x", turn=1)
    probs = np.zeros(2025, dtype=np.float32)
    probs[42] = 0.7
    probs[43] = 0.3

    collector.record_turn(battle, action=42, probs=probs, value=0.4, is_teampreview=False)

    assert len(collector._turn_rows) == 1
    row = collector._turn_rows[0]
    assert row.battle_id == "battle-x"
    assert row.action_chosen == 42
    assert math.isclose(row.value_predicted, 0.4, abs_tol=1e-6)


def test_record_turn_computes_entropy_over_legal_only(tmp_path):
    """Illegal actions (probs=0) must not contribute to entropy."""
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="battle-x")
    probs = np.zeros(10, dtype=np.float32)
    probs[3] = 0.5
    probs[7] = 0.5  # uniform over 2 legal actions → ln(2) ≈ 0.693

    collector.record_turn(battle, action=3, probs=probs, value=0.0, is_teampreview=False)
    entropy = collector._turn_rows[0].policy_entropy
    assert math.isclose(entropy, math.log(2), abs_tol=1e-4)


def test_record_turn_top_k_drops_zero_prob_actions(tmp_path):
    """Top-K list must contain only nonzero-probability actions."""
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="battle-x")
    probs = np.zeros(20, dtype=np.float32)
    probs[5] = 0.6
    probs[10] = 0.4

    collector.record_turn(battle, action=5, probs=probs, value=0.0, is_teampreview=False)
    import json

    pairs = json.loads(collector._turn_rows[0].top_k_actions_json)
    # Only two nonzero, regardless of TOP_K_ACTIONS=10.
    assert len(pairs) == 2
    # Sorted descending by probability.
    assert pairs[0][0] == 5
    assert pairs[1][0] == 10


def test_record_turn_captures_alive_counts(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b", team_alive=4, opp_alive=2)
    probs = np.zeros(10, dtype=np.float32)
    probs[0] = 1.0

    collector.record_turn(battle, action=0, probs=probs, value=0.0, is_teampreview=False)
    row = collector._turn_rows[0]
    assert row.agent_alive_count == 4
    assert row.opp_alive_count == 2


# ─── record_battle_finished ─────────────────────────────────────────


def test_record_battle_finished_outcome_win(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b1", turn=12, won=True)
    collector.record_battle_finished(battle)
    assert len(collector._battle_rows) == 1
    assert collector._battle_rows[0].outcome == 1.0
    assert collector._battle_rows[0].final_turn == 12


def test_record_battle_finished_outcome_loss(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b2", won=False)
    collector.record_battle_finished(battle)
    assert collector._battle_rows[0].outcome == 0.0


def test_record_battle_finished_outcome_tie_is_nan(tmp_path):
    """Ties don't bucket as wins or losses — outcome is NaN."""
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b3", won=None)
    collector.record_battle_finished(battle)
    assert math.isnan(collector._battle_rows[0].outcome)


def test_record_battle_finished_is_idempotent(tmp_path):
    """Duplicate finished callbacks (poke-env edge cases) must not double-record."""
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b4", won=True)
    collector.record_battle_finished(battle)
    collector.record_battle_finished(battle)
    assert len(collector._battle_rows) == 1


# ─── replay sampling ────────────────────────────────────────────────


def test_replay_rate_zero_never_saves(tmp_path):
    collector = _make_collector(tmp_path, replay_rate=0.0)
    for i in range(20):
        battle = _FakeBattle(battle_tag=f"b{i}", won=True)
        collector.record_battle_finished(battle)
    saved = [r for r in collector._battle_rows if r.replay_saved]
    assert len(saved) == 0
    assert not os.path.exists(os.path.join(tmp_path, "replays"))


def test_replay_rate_one_always_saves(tmp_path):
    collector = _make_collector(tmp_path, replay_rate=1.0)
    for i in range(5):
        battle = _FakeBattle(battle_tag=f"b{i}", won=True)
        collector.record_battle_finished(battle)
    saved = [r for r in collector._battle_rows if r.replay_saved]
    assert len(saved) == 5
    # Replay files exist on disk.
    for i in range(5):
        assert os.path.exists(os.path.join(tmp_path, "replays", f"b{i}.log.gz"))


def test_replay_content_is_gzipped_protocol_log(tmp_path):
    """The saved replay must be the gzipped output of _build_replay_log."""
    collector = _make_collector(tmp_path, replay_rate=1.0)
    expected_log = ">battle-xyz\n|move|p1a: Foo|Tackle|p2a: Bar"
    battle = _FakeBattle(battle_tag="battle-xyz", won=True, replay_log=expected_log)
    collector.record_battle_finished(battle)
    path = os.path.join(tmp_path, "replays", "battle-xyz.log.gz")
    with gzip.open(path, "rb") as f:
        contents = f.read().decode("utf-8")
    assert contents == expected_log


def test_replay_rate_05_samples_approximately_5_percent(tmp_path):
    """Loose statistical check on the sampling rate.

    Uses a fixed seed so the test is deterministic; the assertion
    band is wide enough to catch a bug like "always saves" or "never
    saves" but lenient enough to avoid flakiness.
    """
    collector = _make_collector(tmp_path, replay_rate=0.05, seed=42)
    n = 1000
    for i in range(n):
        battle = _FakeBattle(battle_tag=f"b{i}", won=True)
        collector.record_battle_finished(battle)
    saved = sum(1 for r in collector._battle_rows if r.replay_saved)
    # 5% of 1000 = 50; allow ±25 (5 SD at p=0.05, n=1000 is √(0.05*0.95*1000) ≈ 6.9).
    assert 25 <= saved <= 75, f"saved={saved} far from expected 50"


# ─── flush ──────────────────────────────────────────────────────────


def test_flush_writes_per_worker_parquet_shards(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b1", won=True)
    probs = np.zeros(10, dtype=np.float32)
    probs[0] = 1.0
    collector.record_turn(battle, action=0, probs=probs, value=0.1, is_teampreview=False)
    collector.record_battle_finished(battle)

    collector.flush()

    assert os.path.exists(tmp_path / "battles_worker_0.parquet")
    assert os.path.exists(tmp_path / "turns_worker_0.parquet")

    df_b = read_battles(str(tmp_path))
    assert len(df_b) == 1
    assert df_b["battle_id"].iloc[0] == "b1"
    df_t = read_turns(str(tmp_path))
    assert len(df_t) == 1


def test_flush_is_idempotent_and_clears_buffers(tmp_path):
    collector = _make_collector(tmp_path)
    battle = _FakeBattle(battle_tag="b1", won=True)
    collector.record_battle_finished(battle)

    collector.flush()
    # Buffers are cleared so a re-flush doesn't redundantly serialize.
    assert len(collector._battle_rows) == 0
    collector.flush()
    df_b = read_battles(str(tmp_path))
    assert len(df_b) == 1


def test_team_hashes_set_at_construction(tmp_path):
    """agent_team_hash / opp_team_hash are computed once, used everywhere."""
    collector = _make_collector(tmp_path)
    # Same hash regardless of which battle it's recording for.
    assert len(collector.agent_team_hash) == 12
    assert len(collector.opp_team_hash) == 12
    assert collector.agent_team_hash != collector.opp_team_hash

    battle = _FakeBattle(battle_tag="b", won=True)
    collector.record_battle_finished(battle)
    row = collector._battle_rows[0]
    assert row.agent_team_hash == collector.agent_team_hash
    assert row.opp_team_hash == collector.opp_team_hash
