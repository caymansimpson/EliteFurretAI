"""Generalized model-vs-anything evaluation entry point.

Resolves both players from a single string per slot (checkpoint path
*or* baseline name), with team sources independently specified per
slot (file *or* directory *or* format-default). Replaces the prior
model-vs-model / model-vs-baseline split.

Example
-------
    python -m elitefurretai.rl.analyze.analysis_utils \
        --player1 data/models/rl/may16-run/main_model_step_500.pt \
        --player2 simple_heuristic \
        --team1 data/teams/gen9vgc2024regg/constrained \
        --team2 data/teams/gen9vgc2024regg/vgcbench.txt \
        --battles 200 --workers 4 --launch-servers
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import time
import uuid
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import numpy as np
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.agents.foulplay_manager import FoulPlayManager
from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import MDBO, TeamRepo, evaluate_position_advantage


@dataclass
class EvalResult:
    """Outcome of a single eval matchup, aggregated across workers."""

    label: str
    player1_wins: int
    player2_wins: int
    ties: int
    battles_played: int

    @property
    def player1_win_rate(self) -> float:
        return self.player1_wins / self.battles_played if self.battles_played else 0.0


def _build_server_urls(server_base: str, num_servers: int, start_port: int) -> List[str]:
    if server_base != "localhost":
        return [server_base]
    return [f"localhost:{start_port + i}" for i in range(num_servers)]


def _split_battles(total_battles: int, workers: int) -> List[int]:
    workers = max(1, workers)
    base = total_battles // workers
    rem = total_battles % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _username(prefix: str, worker_id: int, run_tag: str) -> str:
    """Build a Showdown username under the 18-char limit.

    Showdown rejects usernames > 18 chars and silently truncates duplicates,
    so we have to be deliberate. ``run_tag`` is a 4-hex-char per-process
    nonce that disambiguates concurrent eval runs sharing the same server.
    """
    return f"{prefix}{worker_id}{run_tag}"[:18]


def _aggregate_results(label: str, results: List[EvalResult]) -> EvalResult:
    return EvalResult(
        label=label,
        player1_wins=sum(r.player1_wins for r in results),
        player2_wins=sum(r.player2_wins for r in results),
        ties=sum(r.ties for r in results),
        battles_played=sum(r.battles_played for r in results),
    )


def _print_result(result: EvalResult, p1_label: str, p2_label: str) -> None:
    print(
        f"{p1_label:>20} vs {p2_label:<20} | "
        f"Battles={result.battles_played:<5} "
        f"P1Wins={result.player1_wins:<5} "
        f"P2Wins={result.player2_wins:<5} "
        f"Ties={result.ties:<3} "
        f"P1WR={result.player1_win_rate * 100:6.2f}%"
    )


def _build_model_player(
    specification: PlayerSpecification,
    team_str: str,
    account: AccountConfiguration,
    server_config: ServerConfiguration,
    collector: Optional[TrajectoryCollector],
) -> Any:
    """Build a model player, optionally wired with a TrajectoryCollector.

    When ``collector`` is set, returns a ``RecordingModelPlayer`` so
    per-turn data flows into the analysis pipeline (Plan B). When
    None, falls back to the specification's standard factory (plain
    ``SimpleModelPlayer``). All other kinds construct unchanged.

    ``team_str`` is pre-resolved by the worker so the collector and
    the player see the same team string (and so the same team_hash
    appears in BattleRecord and in the player's actual battle).
    """
    if specification.kind == "model" and collector is not None:
        return RecordingModelPlayer(
            model_path=specification.raw,
            device=_detect_device_from_specification(specification),
            battle_format=collector.battle_format,
            probabilistic=False,
            account_configuration=account,
            server_configuration=server_config,
            team=team_str,
            accept_open_team_sheet=False,
            collector=collector,
        )
    return build_player(
        specification,
        team=team_str,
        account_configuration=account,
        server_configuration=server_config,
        accept_open_team_sheet=False,
    )


def _detect_device_from_specification(specification: PlayerSpecification) -> str:
    """Extract device from a model specification.

    For ``kind="model"`` the device was set at parse time and lives in
    ``specification.params["device"]``. For other kinds (baseline / external)
    there is no model, so fall back to ``"cpu"``.
    """
    if specification.kind == "model":
        return str(specification.params.get("device", "cpu"))
    return "cpu"


def _run_worker(
    worker_id: int,
    p1: PlayerSpecification,
    p2: PlayerSpecification,
    cells: List[tuple],
    battles_per_cell: int,
    server_url: str,
    run_tag: str,
    *,
    collect_run_dir: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    replay_sample_rate: float = 1.0,
) -> EvalResult:
    """One worker iterates through its assigned (agent_team, opp_team) cells.

    Each cell runs ``battles_per_cell`` games between the same player
    instances, just with their teams swapped via poke-env's
    ``update_team``. This keeps the model loaded once per worker
    instead of reloading per cell — critical at 1764 cells per opp_type.

    Two flow shapes depending on player kinds:

    * **Both in-process** (``"model"`` / ``"baseline"``): standard
      ``player1.battle_against(player2)``.
    * **One external** (vgc_bench): launch the external subprocess
      once per worker, then call ``send_challenges`` from the in-process
      side each cell.
    * Both external: rejected — there is no in-process side to drive
      challenges from.

    When ``collect_run_dir`` is set, one ``TrajectoryCollector`` is
    built per worker and ``set_cell`` is called before each cell's
    battles so BattleRecord/TurnRecord rows carry the right team
    hashes and opp identifiers. A single parquet shard per worker is
    written at the end.

    ``cells`` is a list of ``(agent_team_str, opp_team_str)`` tuples.
    Single-cell mode is just ``len(cells) == 1``.
    """
    if p1.kind == "external" and p2.kind == "external":
        raise ValueError(
            "Cannot run two external players against each other — at least "
            "one side must be in-process to drive challenges."
        )
    if not cells:
        return EvalResult(
            label=f"{p1.name}_vs_{p2.name}",
            player1_wins=0,
            player2_wins=0,
            ties=0,
            battles_played=0,
        )

    first_agent_team, first_opp_team = cells[0]

    # Per-server, per-session battle_id prefix so the same Showdown
    # ``battle_tag`` produced on two different servers OR across two
    # restarts of the same server doesn't collide in parquet rows or
    # replay filenames. Each Showdown server has its own battle
    # counter that resets on restart, so port alone is not enough to
    # uniquely identify rows that join battles ↔ turns across sessions.
    try:
        server_port = server_url.rsplit(":", 1)[1]
    except IndexError:
        server_port = "0"
    battle_id_prefix = f"p{server_port}_{run_tag}_"

    # One collector per worker, initialized with the first cell's teams.
    # set_cell() updates between cells; all rows go into the same shard.
    collector: Optional[TrajectoryCollector] = None
    if collect_run_dir is not None and eval_run_id is not None:
        if p1.kind == "model":
            collector = TrajectoryCollector(
                eval_run_id=eval_run_id,
                agent_ckpt=p1.raw,
                agent_team_str=first_agent_team,
                opp_team_str=first_opp_team,
                opp_player_kind=p2.kind,
                opp_player_name=p2.name,
                battle_format=_battle_format_from_specification(p1),
                run_dir=collect_run_dir,
                worker_id=worker_id,
                replay_sample_rate=replay_sample_rate,
                seed=worker_id,
                call_id=run_tag,
                battle_id_prefix=battle_id_prefix,
            )
        elif p2.kind == "model":
            # P2 is the model; agent perspective inverts.
            collector = TrajectoryCollector(
                eval_run_id=eval_run_id,
                agent_ckpt=p2.raw,
                agent_team_str=first_opp_team,
                opp_team_str=first_agent_team,
                opp_player_kind=p1.kind,
                opp_player_name=p1.name,
                battle_format=_battle_format_from_specification(p2),
                run_dir=collect_run_dir,
                worker_id=worker_id,
                replay_sample_rate=replay_sample_rate,
                seed=worker_id,
                call_id=run_tag,
                battle_id_prefix=battle_id_prefix,
            )

    async def _run() -> EvalResult:
        server_config = ServerConfiguration(f"ws://{server_url}/showdown/websocket", "")

        external_handle = None
        # Built once on the first cell; reused across cells via update_team.
        player1: Any = None
        player2: Any = None
        p1_account = AccountConfiguration(
            _username(f"E1{p1.user_tag}", worker_id, run_tag), None
        )
        p2_account = AccountConfiguration(
            _username(f"E2{p2.user_tag}", worker_id, run_tag), None
        )

        total_played = 0
        total_p1_wins = 0
        total_p2_wins = 0

        try:
            if p2.kind == "external":
                external_handle = launch_external_player(p2, server_url)
                player1 = _build_player(
                    p1, first_agent_team, p1_account, server_config, collector
                )
            elif p1.kind == "external":
                external_handle = launch_external_player(p1, server_url)
                player2 = _build_player(
                    p2, first_opp_team, p2_account, server_config, collector
                )
            else:
                player1 = _build_player(
                    p1, first_agent_team, p1_account, server_config, collector
                )
                player2 = _build_player(
                    p2, first_opp_team, p2_account, server_config, None
                )

            for cell_idx, (agent_team, opp_team) in enumerate(cells):
                # Swap teams on existing player instances (not the first
                # cell — they were just built with these teams).
                if cell_idx > 0:
                    if player1 is not None:
                        player1.update_team(agent_team)
                    if player2 is not None:
                        player2.update_team(opp_team)
                    # Clear poke-env's per-Player battles dict — it
                    # accumulates finished `Battle` objects across all
                    # cells of this worker (~150 KB each in practice).
                    # At 1764 cells × 100 battles = 176,400 battles this
                    # leaks 25+ GB and crushes the process via swap.
                    # `reset_battles` raises if any battle is still
                    # running, which can't happen here — battle_against
                    # / send_challenges complete before we reach the
                    # next cell. See memory/feedback_poke_env_battles_leak.md.
                    if player1 is not None:
                        player1.reset_battles()
                    if player2 is not None:
                        player2.reset_battles()

                if collector is not None:
                    if p1.kind == "model":
                        collector.set_cell(
                            agent_team_str=agent_team,
                            opp_team_str=opp_team,
                            opp_player_kind=p2.kind,
                            opp_player_name=p2.name,
                        )
                    else:
                        # Model is P2 — agent perspective is from P2.
                        collector.set_cell(
                            agent_team_str=opp_team,
                            opp_team_str=agent_team,
                            opp_player_kind=p1.kind,
                            opp_player_name=p1.name,
                        )

                # Snapshot win/play counts before the cell so we can
                # compute the delta after.
                if p2.kind == "external":
                    snap_played = player1.n_finished_battles
                    snap_p1 = player1.n_won_battles
                    snap_p2 = player1.n_lost_battles
                    try:
                        await player1.send_challenges(
                            external_handle.username, n_challenges=battles_per_cell
                        )
                    except Exception as exc:
                        print(
                            f"[eval] worker={worker_id} cell={cell_idx} "
                            f"{p1.name} vs {p2.name}(ext) failed: {exc}"
                        )
                    total_played += player1.n_finished_battles - snap_played
                    total_p1_wins += player1.n_won_battles - snap_p1
                    total_p2_wins += player1.n_lost_battles - snap_p2

                elif p1.kind == "external":
                    snap_played = player2.n_finished_battles
                    snap_p1 = player2.n_lost_battles  # P2's loss = P1's win
                    snap_p2 = player2.n_won_battles
                    try:
                        await player2.send_challenges(
                            external_handle.username, n_challenges=battles_per_cell
                        )
                    except Exception as exc:
                        print(
                            f"[eval] worker={worker_id} cell={cell_idx} "
                            f"{p1.name}(ext) vs {p2.name} failed: {exc}"
                        )
                    total_played += player2.n_finished_battles - snap_played
                    total_p1_wins += player2.n_lost_battles - snap_p1
                    total_p2_wins += player2.n_won_battles - snap_p2

                else:
                    snap_played = player1.n_finished_battles
                    snap_p1 = player1.n_won_battles
                    snap_p2 = player1.n_lost_battles
                    try:
                        await player1.battle_against(player2, n_battles=battles_per_cell)
                    except Exception as exc:
                        print(
                            f"[eval] worker={worker_id} cell={cell_idx} "
                            f"{p1.name} vs {p2.name} failed: {exc}"
                        )
                    total_played += player1.n_finished_battles - snap_played
                    total_p1_wins += player1.n_won_battles - snap_p1
                    total_p2_wins += player1.n_lost_battles - snap_p2

            ties = total_played - total_p1_wins - total_p2_wins
            return EvalResult(
                label=f"{p1.name}_vs_{p2.name}",
                player1_wins=total_p1_wins,
                player2_wins=total_p2_wins,
                ties=ties,
                battles_played=total_played,
            )
        finally:
            if external_handle is not None:
                external_handle.shutdown()
            if collector is not None:
                collector.flush()

    return asyncio.run(_run())


def _build_player(
    specification: PlayerSpecification,
    team_str: str,
    account: AccountConfiguration,
    server_config: ServerConfiguration,
    collector: Optional[TrajectoryCollector],
) -> Any:
    """Dispatch player construction: ``RecordingModelPlayer`` if recording
    is on for a model specification, otherwise the specification's standard factory."""
    return _build_model_player(specification, team_str, account, server_config, collector)


def _battle_format_from_specification(specification: PlayerSpecification) -> str:
    """Read ``battle_format`` directly from the specification's params.

    All three specification kinds carry ``battle_format`` in ``params``.
    Defaults to ``gen9vgc2024regg`` if unexpectedly missing.
    """
    return str(specification.params.get("battle_format", "gen9vgc2024regg"))


def run_eval_parallel(
    p1: PlayerSpecification,
    p2: PlayerSpecification,
    cells: List[tuple],
    battles_per_cell: int,
    server_urls: List[str],
    workers: int,
    run_tag: str,
    *,
    collect_run_dir: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    replay_sample_rate: float = 1.0,
    executor: str = "process",
) -> EvalResult:
    """Fan out a list of (agent_team, opp_team) cells across workers.

    Cells are distributed round-robin (``cells[i::workers]``) — balanced
    for the uniform workload of "run N battles per cell." Each worker
    runs all of its assigned cells before returning, reusing player
    instances across cells via ``update_team``. With ``collect_run_dir``
    set, one parquet shard per worker captures the union of rows across
    that worker's cells; the analysis CLI globs all shards into one
    DataFrame.

    ``executor`` selects the parallelism model:

    * ``"process"`` (default) — ``ProcessPoolExecutor`` with the
      ``spawn`` start method. Real CPU parallelism: each worker is its
      own OS process with its own GIL. Required at production scale —
      the eval pipeline's embedder + max_damage damage calc are
      GIL-bound and serialize threads onto one core.
    * ``"thread"`` — ``ThreadPoolExecutor``. Kept for tests and
      single-machine debugging where the ~5s spawn cost matters.
      Throughput is capped at one core's worth of CPU work regardless
      of ``workers``.

    Single-cell calls (no matrix iteration) are just
    ``cells = [(agent_team, opp_team)]`` — same code path.
    """
    if not cells:
        return EvalResult(
            label=f"{p1.name}_vs_{p2.name}",
            player1_wins=0,
            player2_wins=0,
            ties=0,
            battles_played=0,
        )

    # Round-robin assign cells to workers. If workers > cells, the
    # tail workers get an empty slice and exit immediately.
    effective_workers = max(1, min(workers, len(cells)))
    cell_slices = [cells[i::effective_workers] for i in range(effective_workers)]

    pool: Executor
    if executor == "process":
        # `spawn` (not `fork`) so each worker gets a fresh CUDA context.
        # Forking after the parent touches torch.cuda corrupts CUDA in
        # the children with "Cannot re-initialize CUDA in forked subprocess".
        pool = ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp.get_context("spawn"),
        )
    elif executor == "thread":
        pool = ThreadPoolExecutor(max_workers=effective_workers)
    else:
        raise ValueError(f"--executor must be 'process' or 'thread', got {executor!r}")

    with pool:
        futures = []
        for worker_id, my_cells in enumerate(cell_slices):
            if not my_cells:
                continue
            server_url = server_urls[worker_id % len(server_urls)]
            futures.append(
                pool.submit(
                    _run_worker,
                    worker_id,
                    p1,
                    p2,
                    my_cells,
                    battles_per_cell,
                    server_url,
                    run_tag,
                    collect_run_dir=collect_run_dir,
                    eval_run_id=eval_run_id,
                    replay_sample_rate=replay_sample_rate,
                )
            )

        results: List[EvalResult] = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:
                print(f"[eval] {p1.name} vs {p2.name} worker failed: {exc}")
                results.append(
                    EvalResult(
                        label=f"{p1.name}_vs_{p2.name}",
                        player1_wins=0,
                        player2_wins=0,
                        ties=0,
                        battles_played=0,
                    )
                )
        return _aggregate_results(f"{p1.name}_vs_{p2.name}", results)


def build_cells(
    t1: TeamProvider,
    t2: TeamProvider,
    *,
    cell_iteration: bool,
    team1_path: Optional[str],
    team2_path: Optional[str],
) -> List[tuple]:
    """Resolve CLI team specifications into a concrete list of
    ``(agent_team_str, opp_team_str)`` cells.

    Single-cell mode (default): call each team provider once. Same as
    Plan A behavior — one team per side for the whole eval.

    Cell-iteration mode: both team specifications must be directories. Lists
    every ``.txt`` file in each, reads them, and yields the Cartesian
    product (N_agent × N_opp cells), then shuffles with a fixed seed.
    The shuffle is reproducible run-to-run but breaks the row-major
    sorted-filename ordering — so partial sweeps (early termination from
    crash or kill) sample uniformly across (agent_team, opp_team) pairs
    instead of missing the same agent_team tail every time.
    """
    if not cell_iteration:
        return [(t1(), t2())]

    if team1_path is None or team2_path is None:
        raise ValueError(
            "--cell-iteration requires both --team1 and --team2 to point at "
            "team directories."
        )
    p1_dir = Path(team1_path)
    p2_dir = Path(team2_path)
    if not p1_dir.is_dir() or not p2_dir.is_dir():
        raise ValueError(
            "--cell-iteration requires both --team1 and --team2 to be "
            "directories, not single files."
        )

    p1_teams = sorted(p1_dir.glob("*.txt"))
    p2_teams = sorted(p2_dir.glob("*.txt"))
    if not p1_teams or not p2_teams:
        raise ValueError(
            f"No .txt team files found under one of --team1={team1_path!r} or "
            f"--team2={team2_path!r}"
        )

    p1_strs = [p.read_text() for p in p1_teams]
    p2_strs = [p.read_text() for p in p2_teams]
    cells = [(a, b) for a in p1_strs for b in p2_strs]
    random.Random(42).shuffle(cells)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generalized model/baseline evaluation runner"
    )
    parser.add_argument(
        "--player1",
        required=True,
        type=str,
        help="Player 1 specification: checkpoint path or baseline name "
        "(max_damage, max_base_power, simple_heuristic, vgc_bench, random)",
    )
    parser.add_argument(
        "--player2",
        required=True,
        type=str,
        help="Player 2 specification (same accepted values as --player1)",
    )
    parser.add_argument(
        "--team1",
        type=str,
        default=None,
        help="Player 1 team source: file, directory, or omit for format default",
    )
    parser.add_argument(
        "--team2",
        type=str,
        default=None,
        help="Player 2 team source: file, directory, or omit for format default",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=100,
        help="Battles per (agent_team, opp_team) cell. In single-cell mode "
        "(default) this is the total. With --cell-iteration this multiplies "
        "by the matrix size.",
    )
    parser.add_argument(
        "--cell-iteration",
        action="store_true",
        help="Iterate the full Cartesian product of --team1 × --team2 "
        "directories, running --battles per cell. Players are reused across "
        "cells via update_team so the model loads once per worker.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--executor",
        choices=("process", "thread"),
        default="process",
        help="Worker dispatch model. 'process' (default) gives real CPU "
        "parallelism via ProcessPoolExecutor+spawn — required at >1 "
        "worker for the GIL-bound embedder + max_damage damage calc. "
        "'thread' uses ThreadPoolExecutor (legacy path; ~3-4x slower at "
        "workers=4 but no spawn cost, useful for tests).",
    )
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument("--server-base", type=str, default="localhost")
    parser.add_argument(
        "--launch-servers",
        action="store_true",
        help="Launch local Showdown servers automatically",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--battle-format", type=str, default="gen9vgc2024regg")
    parser.add_argument(
        "--vgc-bench-checkpoint-path",
        type=str,
        default="data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--collect-trajectories",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Enable Plan B trajectory collection. Writes parquet shards and "
        "(sampled) Showdown replay logs under RUN_DIR. Requires one side to "
        "be a model checkpoint; collection silently no-ops for baseline-vs-"
        "baseline.",
    )
    parser.add_argument(
        "--replay-sample-rate",
        type=float,
        default=1.0,
        help="Fraction of battles whose Showdown protocol log is gzipped to "
        "RUN_DIR/replays/. Battles cannot be re-played deterministically so "
        "logs must be captured live. 0 disables; 1 saves all. Default: 1.0 "
        "(replays are ~4 KB gzipped each — ~3 GB total across the full "
        "4-opp_type 705k-battle schedule, well within disk budget).",
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="Run identifier embedded in every parquet row. Auto-generated "
        "(UUID4 prefix) when omitted.",
    )
    args = parser.parse_args()

    p1 = parse_player_specification(
        args.player1,
        device=args.device,
        battle_format=args.battle_format,
        vgc_bench_checkpoint_path=args.vgc_bench_checkpoint_path,
    )
    p2 = parse_player_specification(
        args.player2,
        device=args.device,
        battle_format=args.battle_format,
        vgc_bench_checkpoint_path=args.vgc_bench_checkpoint_path,
    )
    t1 = parse_team_specification(args.team1, battle_format=args.battle_format)
    t2 = parse_team_specification(args.team2, battle_format=args.battle_format)

    # Resolve cell list before launching servers / collection — a bad
    # CLI combination here should fail fast, not after Showdown is up.
    cells = build_cells(
        t1,
        t2,
        cell_iteration=args.cell_iteration,
        team1_path=args.team1,
        team2_path=args.team2,
    )
    total_battles = args.battles * len(cells)

    run_tag = format(int(time.time() * 1000) % 65536, "04x")

    # Resolve trajectory-collection settings up front so the manifest
    # gets written before any battles fire (so a crash mid-eval still
    # leaves audit metadata on disk).
    collect_run_dir = args.collect_trajectories
    eval_run_id = args.eval_run_id or f"run_{uuid.uuid4().hex[:8]}"
    if collect_run_dir is not None:
        os.makedirs(collect_run_dir, exist_ok=True)
        _write_or_update_manifest(
            run_dir=collect_run_dir,
            eval_run_id=eval_run_id,
            agent_ckpt_path=(p1.raw if p1.kind == "model" else p2.raw),
            battle_format=args.battle_format,
            replay_sample_rate=args.replay_sample_rate,
            opp_player_name=(p2.name if p1.kind == "model" else p1.name),
            battles=total_battles,
        )

    server_processes = []
    if args.launch_servers:
        server_processes = launch_showdown_servers(args.num_servers, args.start_port)

    try:
        server_urls = _build_server_urls(
            args.server_base, args.num_servers, args.start_port
        )

        started = time.time()
        print(f"\n=== Evaluation: {p1.name} vs {p2.name} ===")
        if args.cell_iteration:
            print(
                f"    Cell iteration ON: {len(cells)} cells × "
                f"{args.battles} battles = {total_battles} total"
            )
        if collect_run_dir is not None:
            print(
                f"    Collecting trajectories to {collect_run_dir} (run_id={eval_run_id})"
            )
        result = run_eval_parallel(
            p1=p1,
            p2=p2,
            cells=cells,
            battles_per_cell=args.battles,
            server_urls=server_urls,
            workers=args.workers,
            run_tag=run_tag,
            collect_run_dir=collect_run_dir,
            eval_run_id=eval_run_id,
            replay_sample_rate=args.replay_sample_rate,
            executor=args.executor,
        )
        duration = time.time() - started
        _print_result(result, p1.name, p2.name)

        if collect_run_dir is not None:
            _mark_manifest_finished(collect_run_dir)

        if args.output:
            payload: Dict[str, Any] = {
                "p1": {"raw": p1.raw, "kind": p1.kind, "name": p1.name},
                "p2": {"raw": p2.raw, "kind": p2.kind, "name": p2.name},
                "team1_specification": args.team1,
                "team2_specification": args.team2,
                "battle_format": args.battle_format,
                "duration_sec": round(duration, 2),
                "result": asdict(result),
                "result_p1_win_rate": result.player1_win_rate,
                "eval_run_id": eval_run_id if collect_run_dir else None,
                "collect_run_dir": collect_run_dir,
            }
            with open(args.output, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved evaluation results to {args.output}")
    finally:
        if server_processes:
            shutdown_showdown_servers(server_processes)


def _git_sha_or_empty() -> str:
    """Best-effort current git SHA for audit. Returns empty string outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _write_or_update_manifest(
    *,
    run_dir: str,
    eval_run_id: str,
    agent_ckpt_path: str,
    battle_format: str,
    replay_sample_rate: float,
    opp_player_name: str,
    battles: int,
) -> None:
    """Write a fresh manifest if absent, else append a ScheduleEntry.

    Multiple ``evaluate.py`` invocations may share a run dir (one call
    per opp_type in the user's 4-opp_type schedule). Each call appends
    its slice to the manifest's ``schedule`` list so the audit trail
    captures the full run.
    """
    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = read_manifest(run_dir)
        manifest.schedule.append(
            ScheduleEntry(opp_player_name=opp_player_name, battles_total=battles)
        )
    else:
        manifest = EvalRunManifest(
            eval_run_id=eval_run_id,
            git_sha=_git_sha_or_empty(),
            agent_ckpt_path=agent_ckpt_path,
            battle_format=battle_format,
            replay_sample_rate=replay_sample_rate,
            schedule=[
                ScheduleEntry(opp_player_name=opp_player_name, battles_total=battles)
            ],
            started_at=datetime.datetime.now().isoformat(),
        )
    write_manifest(manifest, run_dir)


def _mark_manifest_finished(run_dir: str) -> None:
    """Stamp ``finished_at`` on the manifest after a successful run."""

    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    manifest = read_manifest(run_dir)
    manifest.finished_at = datetime.datetime.now().isoformat()
    write_manifest(manifest, run_dir)


if __name__ == "__main__":
    main()


# ============================================================================
# Section: eval_schema (was src/elitefurretai/rl/analyze/eval_schema.py)
# ============================================================================


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


# ============================================================================
# Section: team_provider (was src/elitefurretai/rl/analyze/team_provider.py)
# ============================================================================

TeamProvider = Callable[[], str]


def parse_team_specification(raw: Optional[str], *, battle_format: str) -> TeamProvider:
    """Resolve ``raw`` into a ``TeamProvider``.

    Resolution order:
        1. ``raw is None`` or empty → format default
           (``TeamRepo(filepath="data/teams").sample_team(format)``).
        2. ``Path(raw).is_file()`` → read the file once, return a
           closure that hands back the same string every call.
        3. ``Path(raw).is_dir()`` → wrap ``TeamRepo(filepath=raw)`` and
           sample a team per call.
        4. Else raise ``ValueError`` — neither a file nor a directory.
    """
    if not raw:
        return _default_provider(battle_format)

    path = Path(raw)
    if path.is_file():
        return _file_provider(path)
    if path.is_dir():
        return _directory_provider(path, battle_format)

    raise ValueError(
        f"Could not resolve team specification {raw!r}: not a file and not a directory"
    )


def _file_provider(path: Path) -> TeamProvider:
    team_str = path.read_text()
    return lambda: team_str


def _directory_provider(path: Path, battle_format: str) -> TeamProvider:
    repo = TeamRepo(filepath=str(path))
    return lambda: repo.sample_team(battle_format)


def _default_provider(battle_format: str) -> TeamProvider:
    repo = TeamRepo(filepath="data/teams")
    return lambda: repo.sample_team(battle_format)


# ============================================================================
# Section: player_factory (was src/elitefurretai/rl/analyze/player_factory.py)
# ============================================================================

PlayerKind = Literal["model", "baseline", "external"]

# Canonical baseline names (snake_case). Aliases below map legacy
# evaluate.py spellings ("maxdamage", "shp") to the canonical form so
# old CLI invocations keep working.
_CANONICAL_BASELINES = (
    "max_damage",
    "max_base_power",
    "simple_heuristic",
    "random",
)

# vgc_bench and foul_play go down the external path (subprocess in a
# separate venv), not the in-process baseline factory. Kept separate
# so parser logic is clear.
_EXTERNAL_BASELINES = ("vgc_bench", "foul_play")

_BASELINE_ALIASES = {
    "maxdamage": "max_damage",
    "maxbasepower": "max_base_power",
    "shp": "simple_heuristic",
    "simpleheuristic": "simple_heuristic",
    "simpleheuristics": "simple_heuristic",
    "vgcbench": "vgc_bench",
    "foulplay": "foul_play",
}


# 3-letter user tag used as a username prefix. Showdown usernames have
# an 18-character cap (see ``_username`` in evaluate.py), so keep tags
# short and unambiguous across baselines.
_BASELINE_USER_TAG = {
    "max_damage": "MD",
    "max_base_power": "MBP",
    "simple_heuristic": "SHP",
    "vgc_bench": "VGB",
    "foul_play": "FP",
    "random": "RND",
}


@dataclass
class RunningExternal:
    """Handle to a running external opponent subprocess.

    ``username`` is the Showdown username the subprocess logged in as —
    the other player ``send_challenges(username, n)`` against it.
    ``shutdown`` terminates the subprocess and closes any log handles.
    """

    username: str
    shutdown: Callable[[], None]


@dataclass(frozen=True)
class PlayerSpecification:
    """Parsed player specification — pure data, no closures.

    Fields:
        raw: original CLI string (for logging / serialization).
        kind: ``"model"``, ``"baseline"``, or ``"external"``.
        name: canonical display name. For ``kind="model"`` this is the
            checkpoint filename (without extension); for baselines /
            external it is the canonical snake_case name.
        user_tag: short uppercase tag used as a username prefix.
        params: kind-specific config used by ``build_player`` /
            ``launch_external_player`` to construct the Player. Only
            primitive types — must be picklable.

    Per-kind ``params`` schema:

    * ``"model"`` → ``path`` (str), ``device`` (str), ``battle_format`` (str)
    * ``"baseline"`` → ``canonical`` (str), ``battle_format`` (str)
    * ``"external"`` → ``checkpoint_path`` (str), ``team_file`` (str),
      ``python_executable`` (str), ``battle_format`` (str)
    """

    raw: str
    kind: PlayerKind
    name: str
    user_tag: str
    params: Mapping[str, Any]


def canonicalize_baseline(raw: str) -> Optional[str]:
    """Return the canonical baseline name for ``raw``, or ``None`` if not a baseline.

    Includes external baselines (``vgc_bench``) so callers can decide
    routing afterwards.
    """
    key = raw.strip().lower().replace("-", "_")
    if key in _CANONICAL_BASELINES or key in _EXTERNAL_BASELINES:
        return key
    return _BASELINE_ALIASES.get(key)


def parse_player_specification(
    raw: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
    vgc_bench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt",
    vgc_bench_python_executable: str = "/home/cayman/Repositories/venv-vgcbench-bcsp/bin/python",
    foul_play_python_executable: str = "/home/cayman/Repositories/venv-foulplay/bin/python",
    foul_play_team_pool_path: str = "data/teams/gen9vgc2024regg/constrained",
    foul_play_search_time_ms: int = 750,
    foul_play_parallelism: int = 4,
) -> PlayerSpecification:
    """Resolve ``raw`` into a ``PlayerSpecification``.

    Resolution order:
        1. If ``raw`` is a path to an existing file → ``kind="model"``.
        2. Else if ``canonicalize_baseline(raw)`` returns an external
           name → ``kind="external"``.
        3. Else if it returns a regular baseline name → ``kind="baseline"``.
        4. Else raise ``ValueError`` listing accepted names.

    The path check goes first because a model checkpoint named
    ``random.pt`` should resolve to a model, not the random baseline.

    The ``foul_play_*`` kwargs supply the subprocess-config the
    eval-side launcher needs for the FoulPlay path (search-time,
    parallelism, venv interpreter, team pool). They have sensible
    defaults so simple CLI use (``--player2 foul_play``) works without
    extra flags.
    """
    if os.path.isfile(raw):
        return _model_specification(raw, device=device, battle_format=battle_format)

    canonical = canonicalize_baseline(raw)
    if canonical is None:
        accepted = sorted(_CANONICAL_BASELINES + _EXTERNAL_BASELINES)
        raise ValueError(
            f"Could not resolve player specification {raw!r}. Provide a checkpoint "
            f"path that exists, or one of: {accepted} "
            f"(aliases also accepted: {sorted(_BASELINE_ALIASES)})."
        )

    if canonical in _EXTERNAL_BASELINES:
        return _external_specification(
            raw,
            canonical,
            battle_format=battle_format,
            vgc_bench_checkpoint_path=vgc_bench_checkpoint_path,
            vgc_bench_team_file=vgc_bench_team_file,
            vgc_bench_python_executable=vgc_bench_python_executable,
            foul_play_python_executable=foul_play_python_executable,
            foul_play_team_pool_path=foul_play_team_pool_path,
            foul_play_search_time_ms=foul_play_search_time_ms,
            foul_play_parallelism=foul_play_parallelism,
        )

    return _baseline_specification(raw, canonical, battle_format=battle_format)


def _model_specification(
    path: str, *, device: str, battle_format: str
) -> PlayerSpecification:
    name = os.path.splitext(os.path.basename(path))[0]
    return PlayerSpecification(
        raw=path,
        kind="model",
        name=name,
        user_tag="MDL",
        params={"path": path, "device": device, "battle_format": battle_format},
    )


def _baseline_specification(
    raw: str, canonical: str, *, battle_format: str
) -> PlayerSpecification:
    return PlayerSpecification(
        raw=raw,
        kind="baseline",
        name=canonical,
        user_tag=_BASELINE_USER_TAG[canonical],
        params={"canonical": canonical, "battle_format": battle_format},
    )


def _external_specification(
    raw: str,
    canonical: str,
    *,
    battle_format: str,
    vgc_bench_checkpoint_path: str,
    vgc_bench_team_file: str,
    vgc_bench_python_executable: str,
    foul_play_python_executable: str,
    foul_play_team_pool_path: str,
    foul_play_search_time_ms: int,
    foul_play_parallelism: int,
) -> PlayerSpecification:
    """Build a ``kind="external"`` spec for one of the registered external baselines.

    Each external baseline carries its own params shape — there's no
    shared schema because their subprocess interfaces differ
    (vgc_bench loads an SB3 checkpoint + plays one team; foul_play
    runs a search bot at a configured time-budget + samples from a
    team pool).
    """
    if canonical == "vgc_bench":
        return PlayerSpecification(
            raw=raw,
            kind="external",
            name=canonical,
            user_tag=_BASELINE_USER_TAG[canonical],
            params={
                "checkpoint_path": vgc_bench_checkpoint_path,
                "team_file": vgc_bench_team_file,
                "python_executable": vgc_bench_python_executable,
                "battle_format": battle_format,
            },
        )
    if canonical == "foul_play":
        return PlayerSpecification(
            raw=raw,
            kind="external",
            name=canonical,
            user_tag=_BASELINE_USER_TAG[canonical],
            params={
                "python_executable": foul_play_python_executable,
                "team_pool_path": foul_play_team_pool_path,
                "search_time_ms": foul_play_search_time_ms,
                "parallelism": foul_play_parallelism,
                "battle_format": battle_format,
            },
        )
    raise AssertionError(f"unreachable: unknown external baseline {canonical!r}")


def build_player(
    specification: PlayerSpecification,
    *,
    team: str,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    accept_open_team_sheet: bool = False,
) -> Player:
    """Construct a poke-env ``Player`` from a ``PlayerSpecification``.

    Handles ``kind="model"`` and ``kind="baseline"``. For ``kind="external"``
    use ``launch_external_player`` — there is no in-process Player.

    This is a top-level function (not a closure on the specification) so the specification
    itself remains pickleable and process-pool friendly.
    """
    if specification.kind == "model":
        return SimpleModelPlayer(
            model_path=specification.params["path"],
            device=specification.params["device"],
            battle_format=specification.params["battle_format"],
            probabilistic=False,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )

    if specification.kind == "baseline":
        canonical = specification.params["canonical"]
        common = dict(
            battle_format=specification.params["battle_format"],
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )
        if canonical == "max_damage":
            # Deterministic argmax — evaluation needs a fixed-policy baseline,
            # not the curriculum's softmax-sampled default (temperature=0.5).
            return MaxDamagePlayer(temperature=0.0, **common)
        if canonical == "max_base_power":
            return MaxBasePowerPlayer(**common)
        if canonical == "simple_heuristic":
            return SimpleHeuristicsPlayer(**common)
        if canonical == "random":
            return RandomPlayer(**common)
        raise AssertionError(f"unreachable: unknown canonical baseline {canonical!r}")

    raise ValueError(
        f"build_player() does not handle kind={specification.kind!r}; "
        "use launch_external_player() for external opponents."
    )


def launch_external_player(
    specification: PlayerSpecification, server_url: str
) -> RunningExternal:
    """Spawn the external opponent subprocess and return a handle.

    Requires ``specification.kind == "external"``. Dispatches by
    ``specification.name`` to one of the registered external launchers
    (currently ``vgc_bench`` and ``foul_play``).
    """
    if specification.kind != "external":
        raise ValueError(
            f"launch_external_player() requires kind='external', got {specification.kind!r}"
        )
    if specification.name == "vgc_bench":
        return _launch_vgc_bench_subprocess(
            server_url=server_url,
            battle_format=specification.params["battle_format"],
            checkpoint_path=specification.params["checkpoint_path"],
            team_file=specification.params["team_file"],
            python_executable=specification.params["python_executable"],
        )
    if specification.name == "foul_play":
        return _launch_foulplay_subprocess(
            server_url=server_url,
            battle_format=specification.params["battle_format"],
            python_executable=specification.params["python_executable"],
            team_pool_path=specification.params["team_pool_path"],
            search_time_ms=specification.params["search_time_ms"],
            parallelism=specification.params["parallelism"],
        )
    raise ValueError(
        f"launch_external_player() does not handle external opponent {specification.name!r}"
    )


def _launch_vgc_bench_subprocess(
    *,
    server_url: str,
    battle_format: str,
    checkpoint_path: str,
    team_file: str,
    python_executable: str,
) -> RunningExternal:
    """Spawn the vgc-bench subprocess and return a handle.

    Mirrors ``VGCBenchManager.launch`` (under ``src/elitefurretai/agents``)
    but takes raw parameters instead of an ``RNaDConfig`` so the eval
    script doesn't have to construct training config to use vgc_bench.
    The subprocess args are kept identical so any future fix to
    ``_vgcbench_subprocess.py`` applies uniformly.

    Blocks for ``STARTUP_WAIT_S`` seconds before returning so the
    subprocess has time to log into Showdown — without this, a
    challenge sent immediately after launch gets "user not found".
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"vgc-bench checkpoint not found: {checkpoint_path}")
    if not os.path.exists(team_file):
        raise FileNotFoundError(f"vgc-bench team file not found: {team_file}")
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"vgc-bench venv python not found: {python_executable}")

    # Server URL is `host:port`; the subprocess wants the same shape.
    port = int(server_url.rsplit(":", 1)[1])
    base_username = VGCBenchManager.USERNAMES[0]  # "VGCBENCH"
    username = VGCBenchManager.derive_username(base_username, port)

    log_dir = "data/logs/vgcbench_runners_eval"
    os.makedirs(log_dir, exist_ok=True)
    sanitized = username.replace("/", "_")
    log_path = os.path.join(log_dir, f"runner_{sanitized}_{port}.log")
    log_handle = open(log_path, "a", encoding="utf-8")

    command = [
        python_executable,
        VGCBenchManager.SUBPROCESS_SCRIPT,
        "--username",
        username,
        "--server",
        f"localhost:{port}",
        "--battle-format",
        battle_format,
        "--checkpoint-path",
        checkpoint_path,
        "--team-file",
        team_file,
        "--n-challenges",
        str(VGCBenchManager.N_CHALLENGES),
        "--wait-for-server-timeout",
        str(VGCBenchManager.WAIT_FOR_SERVER_TIMEOUT_S),
    ]
    if VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET:
        command.append("--accept-open-team-sheet")

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(
        f"✓ Launched eval vgc-bench runner '{username}' on localhost:{port} "
        f"(PID: {process.pid}) log={log_path}"
    )

    # Wait for Showdown login. VGCBenchManager uses STARTUP_WAIT_S=10s
    # because the subprocess has to import poke-env, load the SB3
    # policy, and complete a websocket handshake. Anything less risks
    # the first /challenge landing before the user exists.
    time.sleep(VGCBenchManager.STARTUP_WAIT_S)

    def shutdown() -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
        try:
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass

    return RunningExternal(username=username, shutdown=shutdown)


def _launch_foulplay_subprocess(
    *,
    server_url: str,
    battle_format: str,
    python_executable: str,
    team_pool_path: str,
    search_time_ms: int,
    parallelism: int,
) -> RunningExternal:
    """Spawn the foul-play-doubles subprocess and return a handle.

    Single-shot CLI variant of ``FoulPlayManager.launch``: takes raw
    parameters instead of a config object so the eval script doesn't
    have to construct training config to use foul_play. The subprocess
    args are kept identical to ``FoulPlayManager`` so any future fix to
    ``_foulplay_subprocess.py`` applies uniformly.

    Blocks for ``STARTUP_WAIT_S`` seconds before returning so the
    subprocess has time to log into Showdown — without this, the first
    challenge sent immediately after launch gets "user not found".

    ``--n-challenges`` is set to a large constant here because the
    eval-side cell loop controls the actual number of challenges issued
    (matching the long-lived FoulPlay-as-baseline pattern). The
    training-loop variant of FoulPlay eval is bounded by
    ``FoulplayEvalConfig.n_battles_per_format`` and goes through
    ``FoulPlayManager`` directly, not this helper.
    """
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"foul-play venv python not found: {python_executable}")
    if not os.path.isdir(team_pool_path):
        raise FileNotFoundError(f"foul-play team pool not found: {team_pool_path}")

    port = int(server_url.rsplit(":", 1)[1])
    base_username = FoulPlayManager.USERNAMES[0]  # "FOULPLAY"
    username = FoulPlayManager.derive_username(base_username, port)

    log_dir = "data/logs/foulplay_runners_eval"
    os.makedirs(log_dir, exist_ok=True)
    sanitized = username.replace("/", "_")
    log_path = os.path.join(log_dir, f"runner_{sanitized}_{port}.log")
    log_handle = open(log_path, "a", encoding="utf-8")

    command = [
        python_executable,
        FoulPlayManager.SUBPROCESS_SCRIPT,
        "--username",
        username,
        "--server",
        f"localhost:{port}",
        "--battle-format",
        battle_format,
        "--n-challenges",
        # Large constant — the cell loop decides how many to issue.
        str(1_000_000),
        "--team-list-dir",
        team_pool_path,
        "--search-time-ms",
        str(search_time_ms),
        "--parallelism",
        str(parallelism),
        "--wait-for-server-timeout",
        str(FoulPlayManager.WAIT_FOR_SERVER_TIMEOUT_S),
    ]
    if FoulPlayManager.ACCEPT_OPEN_TEAM_SHEET:
        command.append("--accept-open-team-sheet")

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(
        f"✓ Launched eval foul-play runner '{username}' on localhost:{port} "
        f"(PID: {process.pid}) log={log_path}"
    )

    # Wait for Showdown login. FoulPlay imports poke-engine-doubles
    # (a Rust extension), poke_env 0.11, and completes a websocket
    # handshake. STARTUP_WAIT_S mirrors VGCBench's 10 s; less and the
    # first /challenge can hit a non-existent user.
    time.sleep(FoulPlayManager.STARTUP_WAIT_S)

    def shutdown() -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
        try:
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass

    return RunningExternal(username=username, shutdown=shutdown)


# ============================================================================
# Section: eval_collector (was src/elitefurretai/rl/analyze/eval_collector.py)
# ============================================================================


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
        call_id: str = "",
        battle_id_prefix: str = "",
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
        # ``call_id`` disambiguates parquet shard filenames when a
        # single run_dir hosts multiple evaluate.py invocations
        # (e.g. one per opp_type in the four-opp_type schedule).
        # Without it, every call writes to battles_worker_<id>.parquet
        # and clobbers the previous opp_type's shards. Empty string is
        # back-compat for single-call usage.
        self.call_id = call_id
        # ``battle_id_prefix`` is prepended to ``battle.battle_tag`` for
        # every record (and replay filename). Necessary because each
        # Showdown server maintains its OWN battle counter — two workers
        # on different servers can produce identical battle_tags, which
        # would collide in BattleRecord rows (the dedup set is per-
        # collector instance) and overwrite each other's replay files.
        # Conventionally set to e.g. ``"p8201_"`` so the canonical id
        # carries server-port context.
        self.battle_id_prefix = battle_id_prefix
        # Periodic flush state. Each flush writes a new suffixed
        # shard (``_b<idx>``) and increments the index, so a crash
        # mid-run only loses the in-memory tail since the last flush
        # rather than the whole worker's history. read_battles /
        # read_turns globs across all shards transparently.
        self._batch_idx = 0
        self._flush_threshold = 50

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
        canonical_id = self.battle_id_prefix + battle.battle_tag
        if canonical_id not in self._battle_start_times:
            self._battle_start_times[canonical_id] = time.time()

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
                battle_id=canonical_id,
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

        Idempotent on canonical battle_id — poke-env can fire the
        finished callback more than once in edge cases (forfeit +
        timer), so we guard against double-recording. The canonical id
        prepends ``self.battle_id_prefix`` (typically per-server) so
        the same ``battle_tag`` produced by two Showdown servers'
        independent counters doesn't collide.
        """
        canonical_id = self.battle_id_prefix + battle.battle_tag
        if canonical_id in self._recorded_battle_tags:
            return
        self._recorded_battle_tags.add(canonical_id)

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
            replay_saved = self._save_replay(battle, canonical_id)

        self._battle_rows.append(
            BattleRecord(
                battle_id=canonical_id,
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
                timestamp_started=self._battle_start_times.get(canonical_id, time.time()),
                replay_saved=replay_saved,
            )
        )

        # Periodic flush: turn-row buffer grows ~15× faster than
        # battle-row buffer, so trigger on battle count which is
        # the easier-to-reason-about cap. At 50 battles/flush, a
        # crash loses at most 50 battles' worth of in-memory state.
        if len(self._battle_rows) >= self._flush_threshold:
            self.flush()

    # ── replay capture ──────────────────────────────────────────

    def _save_replay(self, battle: Any, canonical_id: str) -> bool:
        """Capture the Showdown protocol log to ``replays/<id>.log.gz``.

        ``canonical_id`` is the same prefixed id used in BattleRecord
        so the replay file lines up with the parquet row.
        """
        try:
            replay_log = battle._build_replay_log()
        except Exception:
            return False
        replays_dir = os.path.join(self.run_dir, "replays")
        os.makedirs(replays_dir, exist_ok=True)
        path = os.path.join(replays_dir, f"{canonical_id}.log.gz")
        try:
            with gzip.open(path, "wb") as f:
                f.write(replay_log.encode("utf-8"))
            return True
        except Exception:
            return False

    # ── flush ────────────────────────────────────────────────────

    def flush(self) -> None:
        """Write the current buffer as a fresh batch shard, then clear.

        Each call writes a new ``battles_worker_<i>_<call_id>_b<batch>.parquet``
        file and increments ``_batch_idx``. Splitting into batch
        shards means a mid-run flush is a real safepoint: a later
        crash can't roll back already-written batches. ``read_battles``
        / ``read_turns`` globs ``battles_worker_*.parquet`` so all
        batches are unioned transparently at analysis time.

        Idempotent on empty buffers — the parquet writer skips
        zero-row inputs, so a no-op flush incurs no disk write.
        """
        suffix = f"_{self.call_id}" if self.call_id else ""
        batch_suffix = f"_b{self._batch_idx}"
        battles_path = os.path.join(
            self.run_dir,
            f"battles_worker_{self.worker_id}{suffix}{batch_suffix}.parquet",
        )
        turns_path = os.path.join(
            self.run_dir,
            f"turns_worker_{self.worker_id}{suffix}{batch_suffix}.parquet",
        )
        write_battles_parquet(self._battle_rows, battles_path)
        write_turns_parquet(self._turn_rows, turns_path)
        # Clear buffers — the rows are durably on disk now in a
        # uniquely-named batch shard; the next flush writes the next
        # batch into ``_b<idx+1>``.
        self._battle_rows.clear()
        self._turn_rows.clear()
        self._batch_idx += 1


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
