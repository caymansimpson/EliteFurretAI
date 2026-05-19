# -*- coding: utf-8 -*-
"""Trajectory collection for the model-analysis pipeline (Plan B).

Two pieces live here:

* :class:`TrajectoryCollector` — per-worker buffer of ``BattleRecord``
  and ``TurnRecord`` rows, plus replay-sampling state. Owns the
  ``flush()`` that writes the worker's parquet shards at shutdown.
* :class:`RecordingModelPlayer` — a thin ``SimpleModelPlayer`` subclass
  that overrides the two designed extension points
  (``_on_action_selected`` and ``_battle_finished_callback``) to feed
  the collector. No new state on the player; all bookkeeping is in
  the collector.

What gets recorded per turn
---------------------------
* The action chosen and the top-K policy probabilities (K=10), so the
  saved-game sidecars in Q8 can show what the model was "considering"
  alongside what it picked.
* Policy entropy over *legal* actions (illegal actions are masked to
  zero probability before normalization, so they don't contribute).
* The model's value prediction (scalar from the C51 head).
* ``heuristic_adv`` from :func:`evaluate_position_advantage` — an
  outcome-free position score. Used by Q6 to define "poor situation"
  and to detect swings in losses, and by Q9 as the reference signal
  against which the value head's calibration is judged.
* Per-side HP fraction sum and alive count — cheap state features
  needed for the Q6 "poor situation" filters.

Replay sampling
---------------
At battle finish, we draw a uniform random number; if it falls under
``replay_sample_rate`` (default 1.0 — every battle) the full Showdown
protocol log is gzipped to ``<run_dir>/replays/<battle_id>.log.gz``.

Initial designs used a 5% sample to bound storage at ~150 MB total,
but a realistic-scale storage measurement (see commit message of the
B3b smoke run) put gzipped replay size at ~4 KB/battle — 700k full
replays come to ~3 GB, well within disk budget. The user opted to
save all replays so Q8 saved-game selection can use any battle as a
candidate without "no replay available" gaps.

Setting < 1.0 trades replay coverage for inode count if the run is
extremely large. Set to 0.0 to disable replay capture entirely
(parquet rows still record battle outcomes).
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.etl import MDBO, evaluate_position_advantage
from elitefurretai.rl.analyze.eval_schema import (
    TOP_K_ACTIONS,
    BattleRecord,
    TurnRecord,
    canonical_team_hash,
    write_battles_parquet,
    write_turns_parquet,
)


class TrajectoryCollector:
    """Per-worker buffer for ``BattleRecord`` and ``TurnRecord`` rows.

    A worker constructs one collector at startup with the static
    metadata for *its* (agent_team, opp_team, opp_player) slice. The
    collector is then handed to the :class:`RecordingModelPlayer` and
    used during play. At worker shutdown, :meth:`flush` writes parquet
    shards keyed by ``worker_id``.

    Replay sampling uses a per-collector ``random.Random`` so the
    sample fraction is statistically clean per worker (each worker
    seeds its own RNG; aggregate distribution across workers stays
    uniform).
    """

    def __init__(
        self,
        *,
        eval_run_id: str,
        agent_ckpt: str,
        agent_team_str: str,
        opp_team_str: str,
        opp_player_kind: str,
        opp_player_name: str,
        battle_format: str,
        run_dir: str,
        worker_id: int,
        replay_sample_rate: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.eval_run_id = eval_run_id
        self.agent_ckpt = agent_ckpt
        self.agent_team_hash = canonical_team_hash(agent_team_str)
        self.opp_team_hash = canonical_team_hash(opp_team_str)
        self.opp_player_kind = opp_player_kind
        self.opp_player_name = opp_player_name
        self.battle_format = battle_format
        self.run_dir = run_dir
        self.worker_id = worker_id
        self.replay_sample_rate = replay_sample_rate
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._battle_rows: List[BattleRecord] = []
        self._turn_rows: List[TurnRecord] = []
        # battle_tag -> approximate start time. We track first-turn time
        # rather than battle-open time because poke-env's challenge flow
        # opens the battle slightly before the first move is requested,
        # and "started" in BattleRecord is meant as "first decision."
        self._battle_start_times: Dict[str, float] = {}
        self._recorded_battle_tags: set = set()

    # ── cell-iteration support ───────────────────────────────────

    def set_cell(
        self,
        *,
        agent_team_str: str,
        opp_team_str: str,
        opp_player_kind: str,
        opp_player_name: str,
    ) -> None:
        """Update per-cell metadata between matchup cells.

        When the eval CLI iterates the (agent_team × opp_team) matrix
        in a single process, one collector instance services all cells
        — call this between cells so subsequent records carry the right
        team hashes / opp identifiers. Buffers and start-time map are
        preserved across cells so the worker can flush once at the end.

        Recomputing the hashes (rather than passing them in) keeps the
        single source of truth in ``canonical_team_hash``.
        """
        self.agent_team_hash = canonical_team_hash(agent_team_str)
        self.opp_team_hash = canonical_team_hash(opp_team_str)
        self.opp_player_kind = opp_player_kind
        self.opp_player_name = opp_player_name

    # ── per-turn hook ────────────────────────────────────────────

    def record_turn(
        self,
        battle: Any,
        action: int,
        probs: "np.ndarray[Any, Any]",
        value: float,
        is_teampreview: bool,
    ) -> None:
        """Buffer one ``TurnRecord`` for the agent's decision this turn."""
        if battle.battle_tag not in self._battle_start_times:
            self._battle_start_times[battle.battle_tag] = time.time()

        # Entropy over legal actions only. probs has illegal actions
        # masked to zero by SimpleModelPlayer._select_action, so we can
        # treat any nonzero slot as legal.
        legal = probs[probs > 0]
        if legal.size > 0:
            entropy = float(-(legal * np.log(legal + 1e-12)).sum())
        else:
            entropy = 0.0

        # Top-K action indices by probability (descending). Drop zeros
        # so the resulting list contains only legal-and-considered
        # actions; if fewer than K are legal, the list is shorter.
        topk_idx = np.argsort(probs)[-TOP_K_ACTIONS:][::-1]
        topk_pairs = [[int(i), float(probs[i])] for i in topk_idx if probs[i] > 0]
        topk_json = json.dumps(topk_pairs, separators=(",", ":"))

        try:
            heuristic_adv = float(evaluate_position_advantage(battle))
        except Exception:
            # The position evaluator can raise during teampreview when
            # the battle state is incomplete. Use NaN so downstream
            # analysis can drop those rows from heuristic-adv queries
            # without false-zero bias.
            heuristic_adv = float("nan")

        agent_team_iter = list(battle.team.values()) if battle.team else []
        opp_team_iter = list(battle.opponent_team.values()) if battle.opponent_team else []
        agent_hp_sum = sum(
            (m.current_hp_fraction or 0.0) for m in agent_team_iter if not m.fainted
        )
        opp_hp_sum = sum(
            (m.current_hp_fraction or 0.0) for m in opp_team_iter if not m.fainted
        )
        agent_alive = sum(1 for m in agent_team_iter if not m.fainted)
        opp_alive = sum(1 for m in opp_team_iter if not m.fainted)

        action_str = _action_to_str(action, is_teampreview)
        is_switch = _is_switch_action(action, is_teampreview)

        self._turn_rows.append(
            TurnRecord(
                battle_id=battle.battle_tag,
                turn_number=int(battle.turn),
                is_teampreview=bool(is_teampreview),
                action_chosen=int(action),
                action_chosen_str=action_str,
                top_k_actions_json=topk_json,
                policy_entropy=entropy,
                value_predicted=float(value),
                heuristic_adv=heuristic_adv,
                agent_hp_frac_sum=float(agent_hp_sum),
                opp_hp_frac_sum=float(opp_hp_sum),
                agent_alive_count=int(agent_alive),
                opp_alive_count=int(opp_alive),
                agent_switch_this_turn=bool(is_switch),
            )
        )

    # ── battle-finished hook ─────────────────────────────────────

    def record_battle_finished(self, battle: Any) -> None:
        """Buffer one ``BattleRecord`` for a completed battle.

        Idempotent on battle_tag — poke-env can fire the finished
        callback more than once in edge cases (forfeit + timer), so we
        guard against double-recording.
        """
        if battle.battle_tag in self._recorded_battle_tags:
            return
        self._recorded_battle_tags.add(battle.battle_tag)

        outcome = _battle_outcome(battle)
        agent_alive = sum(
            1 for m in (battle.team.values() if battle.team else []) if not m.fainted
        )
        opp_alive = sum(
            1
            for m in (battle.opponent_team.values() if battle.opponent_team else [])
            if not m.fainted
        )

        replay_saved = False
        if self._rng.random() < self.replay_sample_rate:
            replay_saved = self._save_replay(battle)

        self._battle_rows.append(
            BattleRecord(
                battle_id=battle.battle_tag,
                eval_run_id=self.eval_run_id,
                agent_ckpt=self.agent_ckpt,
                agent_team_hash=self.agent_team_hash,
                opp_player_kind=self.opp_player_kind,
                opp_player_name=self.opp_player_name,
                opp_team_hash=self.opp_team_hash,
                battle_format=self.battle_format,
                outcome=outcome,
                final_turn=int(battle.turn),
                agent_final_pokemon_alive=int(agent_alive),
                opp_final_pokemon_alive=int(opp_alive),
                timestamp_started=self._battle_start_times.get(
                    battle.battle_tag, time.time()
                ),
                replay_saved=replay_saved,
            )
        )

    # ── replay capture ──────────────────────────────────────────

    def _save_replay(self, battle: Any) -> bool:
        """Capture the Showdown protocol log to ``replays/<id>.log.gz``."""
        try:
            replay_log = battle._build_replay_log()
        except Exception:
            return False
        replays_dir = os.path.join(self.run_dir, "replays")
        os.makedirs(replays_dir, exist_ok=True)
        path = os.path.join(replays_dir, f"{battle.battle_tag}.log.gz")
        try:
            with gzip.open(path, "wb") as f:
                f.write(replay_log.encode("utf-8"))
            return True
        except Exception:
            return False

    # ── flush ────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write per-worker parquet shards. Idempotent if called twice."""
        battles_path = os.path.join(
            self.run_dir, f"battles_worker_{self.worker_id}.parquet"
        )
        turns_path = os.path.join(self.run_dir, f"turns_worker_{self.worker_id}.parquet")
        write_battles_parquet(self._battle_rows, battles_path)
        write_turns_parquet(self._turn_rows, turns_path)
        # Clear buffers so a redundant flush() doesn't double-write
        # rows on top of the parquet (parquet writers replace files,
        # but clearing keeps in-memory state honest).
        self._battle_rows.clear()
        self._turn_rows.clear()


# ─── Player subclass ────────────────────────────────────────────────


class RecordingModelPlayer(SimpleModelPlayer):
    """``SimpleModelPlayer`` that feeds turn + battle data to a collector.

    All static metadata (team hashes, opp identifiers, run-id) lives
    on the collector; the player just forwards events. This keeps the
    player thin enough that the same instance could in principle be
    swapped between collectors mid-life, though that's not currently
    used.
    """

    def __init__(
        self,
        *args: Any,
        collector: TrajectoryCollector,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._collector = collector

    def _on_action_selected(
        self,
        battle: Any,
        probs: "np.ndarray[Any, Any]",
        value: float,
        selected: int,
        is_teampreview: bool,
    ) -> None:
        self._collector.record_turn(battle, selected, probs, value, is_teampreview)

    def _battle_finished_callback(self, battle: Any) -> None:
        self._collector.record_battle_finished(battle)
        super()._battle_finished_callback(battle)


# ─── action-classification helpers ─────────────────────────────────


def _action_to_str(action: int, is_teampreview: bool) -> str:
    """Best-effort human-readable representation of an MDBO action.

    Falls back to ``"action_<n>"`` if MDBO can't decode the id — this
    happens for invalid/edge-case action ids that the model might
    occasionally produce. The saved-game sidecars don't depend on this
    being parseable, just readable.
    """
    try:
        if is_teampreview:
            return MDBO.from_int(action, type=MDBO.TEAMPREVIEW).message
        return MDBO.from_int(action, type=MDBO.TURN).message
    except Exception:
        return f"action_{action}"


def _is_switch_action(action: int, is_teampreview: bool) -> bool:
    """Whether a turn action is a switch (vs a move / pass / tera-move).

    Teampreview actions never count as switches — that's pre-battle
    ordering, not in-battle position swapping. Returns False on decode
    failure; the cost of a false negative here is one less "this was
    a switch" row in the analysis dataset, not a correctness bug.
    """
    if is_teampreview:
        return False
    try:
        mdbo = MDBO.from_int(action, type=MDBO.TURN)
        return getattr(mdbo, "is_switch", False)
    except Exception:
        return False


# ─── outcome resolution ────────────────────────────────────────────


def _battle_outcome(battle: Any) -> float:
    """Return 1.0 (win) / 0.0 (loss) / NaN (tie or unresolved).

    Mirrors the win/loss semantics ``Player.battle_against`` uses for
    ``n_won_battles`` / ``n_lost_battles``, computed at battle-finish
    time. We use NaN for ties so downstream win-rate aggregations can
    ``.dropna()`` cleanly rather than absorbing them as either bucket.
    """
    if battle.won is True:
        return 1.0
    if battle.won is False:
        return 0.0
    return math.nan
