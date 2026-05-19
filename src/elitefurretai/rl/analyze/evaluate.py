"""Generalized model-vs-anything evaluation entry point.

Resolves both players from a single string per slot (checkpoint path
*or* baseline name), with team sources independently specified per
slot (file *or* directory *or* format-default). Replaces the prior
model-vs-model / model-vs-baseline split.

Example
-------
    python -m elitefurretai.rl.analyze.evaluate \
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
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.analyze.eval_collector import (
    RecordingModelPlayer,
    TrajectoryCollector,
)
from elitefurretai.rl.analyze.eval_schema import (
    EvalRunManifest,
    ScheduleEntry,
    write_manifest,
)
from elitefurretai.rl.analyze.player_factory import PlayerSpec, parse_player_spec
from elitefurretai.rl.analyze.team_provider import TeamProvider, parse_team_spec


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
    spec: PlayerSpec,
    team_str: str,
    account: AccountConfiguration,
    server_config: ServerConfiguration,
    collector: Optional[TrajectoryCollector],
) -> Any:
    """Build a model player, optionally wired with a TrajectoryCollector.

    When ``collector`` is set, returns a ``RecordingModelPlayer`` so
    per-turn data flows into the analysis pipeline (Plan B). When
    None, falls back to the spec's standard factory (plain
    ``SimpleModelPlayer``). All other kinds construct unchanged.

    ``team_str`` is pre-resolved by the worker so the collector and
    the player see the same team string (and so the same team_hash
    appears in BattleRecord and in the player's actual battle).
    """
    if spec.kind == "model" and collector is not None:
        return RecordingModelPlayer(
            model_path=spec.raw,
            device=_detect_device_from_spec(spec),
            battle_format=collector.battle_format,
            probabilistic=False,
            account_configuration=account,
            server_configuration=server_config,
            team=team_str,
            accept_open_team_sheet=False,
            collector=collector,
        )
    assert spec.factory is not None
    return spec.factory(lambda: team_str, account, server_config, False)


def _detect_device_from_spec(spec: PlayerSpec) -> str:
    """Extract device from a model spec's factory closure.

    The factory closes over ``device`` at parse time but doesn't
    expose it. Rather than threading it through PlayerSpec, we
    construct a probe SimpleModelPlayer-free way: ``RecordingModelPlayer``
    needs device, so we inspect the closure. Falls back to "cpu" if
    the closure structure is unexpected (defensive: collection
    shouldn't crash the eval if a future refactor renames the var).
    """
    try:
        # The model factory is a closure with `device` in its co_freevars.
        # __closure__ holds the captured values in the same order.
        names = spec.factory.__code__.co_freevars  # type: ignore[union-attr]
        cells = spec.factory.__closure__  # type: ignore[union-attr]
        if names and cells:
            idx = names.index("device")
            return cells[idx].cell_contents
    except Exception:
        pass
    return "cpu"


def _run_worker(
    worker_id: int,
    p1: PlayerSpec,
    p2: PlayerSpec,
    t1: TeamProvider,
    t2: TeamProvider,
    battles: int,
    server_url: str,
    run_tag: str,
    *,
    collect_run_dir: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    replay_sample_rate: float = 0.05,
) -> EvalResult:
    """One worker's slice of an eval matchup.

    Each worker instantiates fresh players (one per spec slot) and runs
    ``battles`` games between them. Players are constructed *inside* the
    worker so each worker holds its own poke-env client / inference state.

    Two flow shapes depending on player kinds:

    * **Both in-process** (``"model"`` / ``"baseline"``): standard
      ``player1.battle_against(player2)``.
    * **One external** (vgc_bench): launch the external subprocess in
      this worker's server, then have the *in-process* player call
      ``send_challenges(external.username, n_battles)``. The external
      side is identified by Showdown username only; no Python ``Player``
      object on our side. Win/loss accounting is taken from the
      in-process player and inverted if the external is P1.
    * Both external: rejected — there is no in-process side to drive
      challenges from.

    When ``collect_run_dir`` is set, a ``TrajectoryCollector`` is built
    for the model side of the matchup (whichever slot has
    ``kind="model"``). Per-turn and per-battle data is buffered and
    flushed to parquet shards at worker shutdown. If neither side is a
    model, collection is silently skipped — there's nothing to record
    from a baseline-vs-baseline matchup.
    """
    if p1.kind == "external" and p2.kind == "external":
        raise ValueError(
            "Cannot run two external players against each other — at least "
            "one side must be in-process to drive challenges."
        )

    # Resolve team strings once per worker. The collector and players
    # both reference these so the team_hash in BattleRecord matches the
    # team actually played.
    agent_team_str = t1()
    opp_team_str = t2()

    # Build the collector if collection is enabled AND there's a model
    # to record. The collector is attached to whichever side is the
    # model; if both sides are models, P1 wins the recording slot.
    collector: Optional[TrajectoryCollector] = None
    if collect_run_dir is not None and eval_run_id is not None:
        if p1.kind == "model":
            collector = TrajectoryCollector(
                eval_run_id=eval_run_id,
                agent_ckpt=p1.raw,
                agent_team_str=agent_team_str,
                opp_team_str=opp_team_str,
                opp_player_kind=p2.kind,
                opp_player_name=p2.name,
                battle_format=_battle_format_from_spec(p1),
                run_dir=collect_run_dir,
                worker_id=worker_id,
                replay_sample_rate=replay_sample_rate,
                seed=worker_id,
            )
        elif p2.kind == "model":
            # P2 is the model; agent perspective inverts.
            collector = TrajectoryCollector(
                eval_run_id=eval_run_id,
                agent_ckpt=p2.raw,
                agent_team_str=opp_team_str,
                opp_team_str=agent_team_str,
                opp_player_kind=p1.kind,
                opp_player_name=p1.name,
                battle_format=_battle_format_from_spec(p2),
                run_dir=collect_run_dir,
                worker_id=worker_id,
                replay_sample_rate=replay_sample_rate,
                seed=worker_id,
            )

    async def _run() -> EvalResult:
        server_config = ServerConfiguration(f"ws://{server_url}/showdown/websocket", "")

        external_handle = None
        try:
            if p2.kind == "external":
                # P1 is in-process, challenges P2's external username.
                assert p2.launch_external is not None
                external_handle = p2.launch_external(server_url)
                p1_account = AccountConfiguration(
                    _username(f"E1{p1.user_tag}", worker_id, run_tag), None
                )
                player1 = _build_player(
                    p1, agent_team_str, p1_account, server_config, collector
                )
                try:
                    await player1.send_challenges(
                        external_handle.username, n_challenges=battles
                    )
                except Exception as exc:
                    print(
                        f"[eval] worker={worker_id} {p1.name} vs {p2.name} "
                        f"(external) failed: {exc}"
                    )
                played = player1.n_finished_battles
                p1_wins = player1.n_won_battles
                p2_wins = player1.n_lost_battles

            elif p1.kind == "external":
                # P2 is in-process, challenges P1's external username.
                assert p1.launch_external is not None
                external_handle = p1.launch_external(server_url)
                p2_account = AccountConfiguration(
                    _username(f"E2{p2.user_tag}", worker_id, run_tag), None
                )
                player2 = _build_player(
                    p2, opp_team_str, p2_account, server_config, collector
                )
                try:
                    await player2.send_challenges(
                        external_handle.username, n_challenges=battles
                    )
                except Exception as exc:
                    print(
                        f"[eval] worker={worker_id} {p1.name} (external) vs "
                        f"{p2.name} failed: {exc}"
                    )
                played = player2.n_finished_battles
                # Inverted: from EvalResult's "P1 perspective", a P2-side
                # win for the in-process player means a *loss* for the
                # external P1.
                p1_wins = player2.n_lost_battles
                p2_wins = player2.n_won_battles

            else:
                # Both in-process — original path.
                p1_account = AccountConfiguration(
                    _username(f"E1{p1.user_tag}", worker_id, run_tag), None
                )
                p2_account = AccountConfiguration(
                    _username(f"E2{p2.user_tag}", worker_id, run_tag), None
                )
                player1 = _build_player(
                    p1, agent_team_str, p1_account, server_config, collector
                )
                player2 = _build_player(p2, opp_team_str, p2_account, server_config, None)
                try:
                    await player1.battle_against(player2, n_battles=battles)
                except Exception as exc:
                    print(
                        f"[eval] worker={worker_id} {p1.name} vs {p2.name} failed: {exc}"
                    )
                played = player1.n_finished_battles
                p1_wins = player1.n_won_battles
                p2_wins = player1.n_lost_battles

            ties = played - p1_wins - p2_wins
            return EvalResult(
                label=f"{p1.name}_vs_{p2.name}",
                player1_wins=p1_wins,
                player2_wins=p2_wins,
                ties=ties,
                battles_played=played,
            )
        finally:
            if external_handle is not None:
                external_handle.shutdown()
            if collector is not None:
                collector.flush()

    return asyncio.run(_run())


def _build_player(
    spec: PlayerSpec,
    team_str: str,
    account: AccountConfiguration,
    server_config: ServerConfiguration,
    collector: Optional[TrajectoryCollector],
) -> Any:
    """Dispatch player construction: ``RecordingModelPlayer`` if recording
    is on for a model spec, otherwise the spec's standard factory."""
    return _build_model_player(spec, team_str, account, server_config, collector)


def _battle_format_from_spec(spec: PlayerSpec) -> str:
    """Pull ``battle_format`` out of the spec's factory closure.

    Same trick as ``_detect_device_from_spec``; defaults to
    ``gen9vgc2024regg`` if introspection fails.
    """
    try:
        names = spec.factory.__code__.co_freevars  # type: ignore[union-attr]
        cells = spec.factory.__closure__  # type: ignore[union-attr]
        if names and cells:
            idx = names.index("battle_format")
            return cells[idx].cell_contents
    except Exception:
        pass
    return "gen9vgc2024regg"


def run_eval_parallel(
    p1: PlayerSpec,
    p2: PlayerSpec,
    t1: TeamProvider,
    t2: TeamProvider,
    num_battles: int,
    server_urls: List[str],
    workers: int,
    run_tag: str,
    *,
    collect_run_dir: Optional[str] = None,
    eval_run_id: Optional[str] = None,
    replay_sample_rate: float = 0.05,
) -> EvalResult:
    """Fan out ``num_battles`` across ``workers`` and aggregate.

    Workers run in a ``ThreadPoolExecutor``; each calls ``asyncio.run`` on
    its own event loop. Threading (not multiprocessing) is fine here
    because the heavy work is async network I/O against Showdown servers.

    Collection (Plan B): when ``collect_run_dir`` is set, each worker
    builds a TrajectoryCollector and writes parquet shards keyed by
    worker_id under that directory. ``eval_run_id`` distinguishes
    multiple invocations against the same run dir (e.g. one per
    opp_type) and is embedded in every record.
    """
    splits = [s for s in _split_battles(num_battles, workers) if s > 0]
    if not splits:
        return EvalResult(
            label=f"{p1.name}_vs_{p2.name}",
            player1_wins=0,
            player2_wins=0,
            ties=0,
            battles_played=0,
        )

    with ThreadPoolExecutor(max_workers=len(splits)) as pool:
        futures = []
        for worker_id, battles in enumerate(splits):
            server_url = server_urls[worker_id % len(server_urls)]
            futures.append(
                pool.submit(
                    _run_worker,
                    worker_id,
                    p1,
                    p2,
                    t1,
                    t2,
                    battles,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generalized model/baseline evaluation runner"
    )
    parser.add_argument(
        "--player1",
        required=True,
        type=str,
        help="Player 1 spec: checkpoint path or baseline name "
        "(max_damage, max_base_power, simple_heuristic, vgc_bench, random)",
    )
    parser.add_argument(
        "--player2",
        required=True,
        type=str,
        help="Player 2 spec (same accepted values as --player1)",
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
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
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
        default="data/models/vgc-bench-sb3-model.zip",
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
        default=0.05,
        help="Fraction of battles whose Showdown protocol log is gzipped to "
        "RUN_DIR/replays/. Battles cannot be re-played deterministically so "
        "logs must be captured live. 0 disables; 1 saves all. Default: 0.05.",
    )
    parser.add_argument(
        "--eval-run-id",
        type=str,
        default=None,
        help="Run identifier embedded in every parquet row. Auto-generated "
        "(UUID4 prefix) when omitted.",
    )
    args = parser.parse_args()

    p1 = parse_player_spec(
        args.player1,
        device=args.device,
        battle_format=args.battle_format,
        vgc_bench_checkpoint_path=args.vgc_bench_checkpoint_path,
    )
    p2 = parse_player_spec(
        args.player2,
        device=args.device,
        battle_format=args.battle_format,
        vgc_bench_checkpoint_path=args.vgc_bench_checkpoint_path,
    )
    t1 = parse_team_spec(args.team1, battle_format=args.battle_format)
    t2 = parse_team_spec(args.team2, battle_format=args.battle_format)

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
            battles=args.battles,
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
        if collect_run_dir is not None:
            print(
                f"    Collecting trajectories to {collect_run_dir} (run_id={eval_run_id})"
            )
        result = run_eval_parallel(
            p1=p1,
            p2=p2,
            t1=t1,
            t2=t2,
            num_battles=args.battles,
            server_urls=server_urls,
            workers=args.workers,
            run_tag=run_tag,
            collect_run_dir=collect_run_dir,
            eval_run_id=eval_run_id,
            replay_sample_rate=args.replay_sample_rate,
        )
        duration = time.time() - started
        _print_result(result, p1.name, p2.name)

        if collect_run_dir is not None:
            _mark_manifest_finished(collect_run_dir)

        if args.output:
            payload: Dict[str, Any] = {
                "p1": {"raw": p1.raw, "kind": p1.kind, "name": p1.name},
                "p2": {"raw": p2.raw, "kind": p2.kind, "name": p2.name},
                "team1_spec": args.team1,
                "team2_spec": args.team2,
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
        from elitefurretai.rl.analyze.eval_schema import read_manifest

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
    from elitefurretai.rl.analyze.eval_schema import read_manifest

    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    manifest = read_manifest(run_dir)
    manifest.finished_at = datetime.datetime.now().isoformat()
    write_manifest(manifest, run_dir)


if __name__ == "__main__":
    main()
