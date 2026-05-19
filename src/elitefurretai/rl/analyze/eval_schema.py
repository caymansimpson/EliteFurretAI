# -*- coding: utf-8 -*-
"""Data schemas + parquet I/O for the model-analysis pipeline (Plan B).

Two row types are written per eval run:

* ``BattleRecord`` — one row per completed battle. Carries identifiers
  for the agent / opponent / teams, the final outcome, and a few
  cheap aggregate statistics. Used by Q1 (agent-team WR), Q2
  (opp-team WR), Q3 (opp-type WR), Q5 (short-loss patterns), and
  Q8 (saved-game category filters).
* ``TurnRecord`` — one row per agent decision turn. Carries the
  model's policy entropy / value prediction / action choice, plus the
  heuristic position-advantage signal from ``evaluate_state``. Used by
  Q5 (action distribution in short losses), Q6 (confidence vs poor
  situations + swing detection), Q7 (value-head calibration), Q9a
  (persistent value-vs-ensemble disagreement), and Q9b (agree-then-
  diverge).

Per-worker sharding: each worker writes
``<run_dir>/battles_worker_<id>.parquet`` and
``<run_dir>/turns_worker_<id>.parquet`` at shutdown. The analysis CLI
reads all shards via a ``parquet`` glob, so workers don't have to
coordinate writes during the eval. Shard size is small (~1 MB per
worker for a 100,000-battle slice) so writing at the end is fine.

A ``EvalRunManifest`` JSON sits at the top of the run dir as an audit
trail — git SHA, checkpoint path, schedule, replay-sample-rate, etc.

Team hashing: ``canonical_team_hash`` normalizes a Showdown team
string by sorting Pokemon blocks and move lines so the same six mons
in different orders produce the same hash. Item / ability / EVs /
nature / IVs are preserved in the hash because their lines stay
attached to their block. This gives a stable identifier per *unique
team build* without parsing Showdown's text format.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# ─── Static per-battle schema ────────────────────────────────────────


@dataclass
class BattleRecord:
    """One row per completed battle (agent perspective)."""

    battle_id: str
    eval_run_id: str
    agent_ckpt: str
    agent_team_hash: str
    opp_player_kind: str  # "model" | "baseline" | "external"
    opp_player_name: str
    opp_team_hash: str
    battle_format: str
    outcome: float  # 1.0 win / 0.0 loss / NaN tie
    final_turn: int
    agent_final_pokemon_alive: int
    opp_final_pokemon_alive: int
    timestamp_started: float
    replay_saved: bool  # True if .log.gz exists at <run_dir>/replays/<battle_id>.log.gz


# ─── Per-turn schema ─────────────────────────────────────────────────


@dataclass
class TurnRecord:
    """One row per agent decision turn."""

    battle_id: str
    turn_number: int
    is_teampreview: bool
    action_chosen: int  # MDBO action_id
    action_chosen_str: str  # human-readable form for replay sidecars
    top_k_actions_json: str  # JSON: [[action_id, prob], ...] K=10
    policy_entropy: float  # natural-log entropy over legal actions
    value_predicted: float  # scalar from value head
    heuristic_adv: float  # evaluate_position_advantage(battle); [-1, 1]
    agent_hp_frac_sum: float  # sum across all 6 mons; 0..6
    opp_hp_frac_sum: float
    agent_alive_count: int  # 0..6
    opp_alive_count: int
    agent_switch_this_turn: bool  # True iff action_chosen is a switch action


# ─── Run-level manifest ──────────────────────────────────────────────


@dataclass
class ScheduleEntry:
    """One opp_type slice of an eval-run schedule."""

    opp_player_name: str
    battles_total: int
    completed: int = 0


@dataclass
class EvalRunManifest:
    """Top-of-run-dir audit trail. Written as JSON at run start; updated at end."""

    eval_run_id: str
    git_sha: str
    agent_ckpt_path: str
    battle_format: str
    replay_sample_rate: float
    schedule: List[ScheduleEntry] = field(default_factory=list)
    started_at: str = ""
    finished_at: Optional[str] = None
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "EvalRunManifest":
        data = json.loads(raw)
        schedule = [ScheduleEntry(**e) for e in data.pop("schedule", [])]
        return cls(schedule=schedule, **data)


# ─── Team canonicalization ───────────────────────────────────────────


def canonical_team_hash(team_str: str, *, length: int = 12) -> str:
    """Stable hash identifying a unique team build.

    Normalizes the Showdown team string so equivalent teams collapse:

    * Pokemon blocks are sorted alphabetically (team order doesn't matter).
    * Move lines (``- Move``) within each block are sorted (move order
      doesn't matter).
    * Trailing whitespace is stripped from every line.
    * Blank lines collapsed to a single separator.

    What's *preserved*: species, item, ability, tera type, EVs, nature,
    IVs, and the *set* of moves. Two teams that differ on any of these
    will hash to different values.

    Returns a 12-character hex digest by default — enough to make
    collisions vanishingly improbable across the ~42×42 team pool the
    eval will see.
    """
    blocks = _normalize_team_blocks(team_str)
    payload = "\n\n".join(blocks)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _normalize_team_blocks(team_str: str) -> List[str]:
    """Split a team string into normalized, sorted mon blocks."""
    raw_blocks = [b for b in team_str.replace("\r\n", "\n").split("\n\n") if b.strip()]
    normalized: List[str] = []
    for block in raw_blocks:
        lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        # Move lines start with "- "; sort just those to ignore move ordering.
        move_lines = sorted(line for line in lines if line.startswith("- "))
        other_lines = [line for line in lines if not line.startswith("- ")]
        normalized.append("\n".join(other_lines + move_lines))
    normalized.sort()
    return normalized


# ─── Parquet I/O ─────────────────────────────────────────────────────


def write_battles_parquet(records: List[BattleRecord], path: str) -> None:
    """Write a list of ``BattleRecord`` as a parquet shard."""
    _write_dataclass_rows(records, path)


def write_turns_parquet(records: List[TurnRecord], path: str) -> None:
    """Write a list of ``TurnRecord`` as a parquet shard."""
    _write_dataclass_rows(records, path)


def _write_dataclass_rows(records: List[Any], path: str) -> None:
    # Lazy import — pandas is heavy and the schema module is imported
    # at eval-CLI startup before we know whether collection is enabled.
    import pandas as pd

    if not records:
        return
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def read_battles(run_dir: str) -> "Any":
    """Load all battle shards from a run dir into a single DataFrame."""
    return _read_shards(run_dir, glob="battles_worker_*.parquet")


def read_turns(run_dir: str) -> "Any":
    """Load all turn shards from a run dir into a single DataFrame."""
    return _read_shards(run_dir, glob="turns_worker_*.parquet")


def _read_shards(run_dir: str, *, glob: str) -> "Any":
    import pandas as pd

    paths = sorted(Path(run_dir).glob(glob))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ─── Manifest I/O ────────────────────────────────────────────────────


def write_manifest(manifest: EvalRunManifest, run_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        f.write(manifest.to_json())


def read_manifest(run_dir: str) -> EvalRunManifest:
    with open(os.path.join(run_dir, "manifest.json")) as f:
        return EvalRunManifest.from_json(f.read())


# ─── Constants ───────────────────────────────────────────────────────


# Top-K action probabilities preserved per turn. K=10 is enough to
# inspect "what was the model considering" without inflating turn-row
# size; the full 2025-wide distribution would balloon storage 200×.
TOP_K_ACTIONS = 10
