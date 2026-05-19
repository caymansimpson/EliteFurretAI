# -*- coding: utf-8 -*-
"""Tests for ``eval_schema`` — team-hash canonicalization + parquet I/O.

Most schema fields are dataclass plumbing; the only thing with real
logic worth pinning is ``canonical_team_hash`` (used to identify
unique team builds across the 42×42 matchup matrix). Team-order and
move-order normalization must produce stable hashes or the analysis
cells will fragment.
"""

from __future__ import annotations

import os

import pandas as pd

from elitefurretai.rl.analyze.eval_schema import (
    BattleRecord,
    EvalRunManifest,
    ScheduleEntry,
    TurnRecord,
    canonical_team_hash,
    read_battles,
    read_manifest,
    read_turns,
    write_battles_parquet,
    write_manifest,
    write_turns_parquet,
)

_MON_A = """\
Calyrex-Shadow @ Life Orb
Ability: As One (Spectrier)
Tera Type: Normal
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Astral Barrage
- Psychic
- Nasty Plot
- Protect
"""

_MON_B = """\
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


def _team_of(*mons: str) -> str:
    return "\n\n".join(m.strip() for m in mons)


# ─── canonical_team_hash ────────────────────────────────────────────


def test_team_hash_stable_for_same_team():
    """Same team string → same hash, repeated calls."""
    team = _team_of(_MON_A, _MON_B)
    h1 = canonical_team_hash(team)
    h2 = canonical_team_hash(team)
    assert h1 == h2


def test_team_hash_invariant_to_pokemon_order():
    """Swapping the order of two Pokemon must produce the same hash."""
    team1 = _team_of(_MON_A, _MON_B)
    team2 = _team_of(_MON_B, _MON_A)
    assert canonical_team_hash(team1) == canonical_team_hash(team2)


def test_team_hash_invariant_to_move_order_within_mon():
    """Reordering the move lines within a block must not change the hash."""
    mon_reordered = """\
Calyrex-Shadow @ Life Orb
Ability: As One (Spectrier)
Tera Type: Normal
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Astral Barrage
- Nasty Plot
- Psychic
"""
    assert canonical_team_hash(_MON_A) == canonical_team_hash(mon_reordered)


def test_team_hash_distinguishes_different_items():
    """Same species, different held item → different hash."""
    mon_choice = _MON_A.replace("Life Orb", "Choice Specs")
    assert canonical_team_hash(_MON_A) != canonical_team_hash(mon_choice)


def test_team_hash_distinguishes_different_evs():
    """Same build, different EV spread → different hash."""
    mon_bulky = _MON_A.replace(
        "EVs: 252 SpA / 4 SpD / 252 Spe", "EVs: 252 HP / 252 SpA / 4 SpD"
    )
    assert canonical_team_hash(_MON_A) != canonical_team_hash(mon_bulky)


def test_team_hash_distinguishes_different_moves():
    """One different move → different hash."""
    mon_swapped = _MON_A.replace("- Psychic", "- Substitute")
    assert canonical_team_hash(_MON_A) != canonical_team_hash(mon_swapped)


def test_team_hash_strips_trailing_newlines():
    """Extra blank lines at the end of the team string must not affect the hash."""
    team_clean = _team_of(_MON_A, _MON_B)
    team_with_trailing = team_clean + "\n\n\n\n"
    assert canonical_team_hash(team_with_trailing) == canonical_team_hash(team_clean)


def test_team_hash_strips_trailing_whitespace_within_lines():
    """Trailing spaces on individual lines must not affect the hash."""
    mon_a_spacey = "\n".join(line + "   " for line in _MON_A.split("\n"))
    assert canonical_team_hash(_MON_A) == canonical_team_hash(mon_a_spacey)


def test_team_hash_length_is_12_by_default():
    h = canonical_team_hash(_team_of(_MON_A, _MON_B))
    assert len(h) == 12
    assert all(c in "0123456789abcdef" for c in h)


# ─── Parquet round-trip ─────────────────────────────────────────────


def test_battle_parquet_round_trip(tmp_path):
    records = [
        BattleRecord(
            battle_id=f"b{i}",
            eval_run_id="r0",
            agent_ckpt="data/models/foo.pt",
            agent_team_hash="aaaaaaaaaaaa",
            opp_player_kind="baseline",
            opp_player_name="simple_heuristic",
            opp_team_hash="bbbbbbbbbbbb",
            battle_format="gen9vgc2024regg",
            outcome=float(i % 2),
            final_turn=15 + i,
            agent_final_pokemon_alive=2,
            opp_final_pokemon_alive=0,
            timestamp_started=1700000000.0 + i,
            replay_saved=(i == 0),
        )
        for i in range(3)
    ]
    path = str(tmp_path / "battles_worker_0.parquet")
    write_battles_parquet(records, path)
    df = read_battles(str(tmp_path))
    assert len(df) == 3
    assert set(df.columns) >= {
        "battle_id",
        "agent_team_hash",
        "opp_team_hash",
        "outcome",
        "replay_saved",
    }
    # outcome is float (NaN-able); reads back as float64.
    assert df["outcome"].dtype.kind == "f"


def test_turn_parquet_round_trip(tmp_path):
    records = [
        TurnRecord(
            battle_id="b0",
            turn_number=t,
            is_teampreview=(t == 0),
            action_chosen=42 + t,
            action_chosen_str=f"action{t}",
            top_k_actions_json=f"[[{42 + t}, 0.5]]",
            policy_entropy=2.5 + t * 0.1,
            value_predicted=0.3 + t * 0.05,
            heuristic_adv=-0.1 + t * 0.05,
            agent_hp_frac_sum=5.5 - t * 0.1,
            opp_hp_frac_sum=5.5 - t * 0.1,
            agent_alive_count=6 - t // 5,
            opp_alive_count=6 - t // 4,
            agent_switch_this_turn=(t % 3 == 0),
        )
        for t in range(5)
    ]
    path = str(tmp_path / "turns_worker_0.parquet")
    write_turns_parquet(records, path)
    df = read_turns(str(tmp_path))
    assert len(df) == 5
    assert df["turn_number"].tolist() == [0, 1, 2, 3, 4]


def test_read_empty_run_dir_returns_empty_df(tmp_path):
    df = read_battles(str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_multiple_worker_shards_concatenated(tmp_path):
    """Reading should glob all shards and union them."""
    for worker_id in range(3):
        records = [
            BattleRecord(
                battle_id=f"w{worker_id}_b{i}",
                eval_run_id="r0",
                agent_ckpt="data/models/foo.pt",
                agent_team_hash="aaaaaaaaaaaa",
                opp_player_kind="baseline",
                opp_player_name="simple_heuristic",
                opp_team_hash="bbbbbbbbbbbb",
                battle_format="gen9vgc2024regg",
                outcome=1.0,
                final_turn=10,
                agent_final_pokemon_alive=3,
                opp_final_pokemon_alive=0,
                timestamp_started=1700000000.0,
                replay_saved=False,
            )
            for i in range(2)
        ]
        write_battles_parquet(
            records, str(tmp_path / f"battles_worker_{worker_id}.parquet")
        )

    df = read_battles(str(tmp_path))
    assert len(df) == 6
    assert df["battle_id"].is_unique


# ─── Manifest ────────────────────────────────────────────────────────


def test_manifest_round_trip(tmp_path):
    manifest = EvalRunManifest(
        eval_run_id="r0",
        git_sha="abc123",
        agent_ckpt_path="data/models/foo.pt",
        battle_format="gen9vgc2024regg",
        replay_sample_rate=0.05,
        schedule=[
            ScheduleEntry(opp_player_name="simple_heuristic", battles_total=1000),
            ScheduleEntry(opp_player_name="max_damage", battles_total=1000, completed=500),
        ],
        started_at="2026-05-19T00:00:00",
    )
    write_manifest(manifest, str(tmp_path))
    assert os.path.exists(tmp_path / "manifest.json")
    loaded = read_manifest(str(tmp_path))
    assert loaded.eval_run_id == "r0"
    assert loaded.git_sha == "abc123"
    assert len(loaded.schedule) == 2
    assert loaded.schedule[1].completed == 500
