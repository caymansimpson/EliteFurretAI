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
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
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


def _run_worker(
    worker_id: int,
    p1: PlayerSpec,
    p2: PlayerSpec,
    t1: TeamProvider,
    t2: TeamProvider,
    battles: int,
    server_url: str,
    run_tag: str,
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
    """
    if p1.kind == "external" and p2.kind == "external":
        raise ValueError(
            "Cannot run two external players against each other — at least "
            "one side must be in-process to drive challenges."
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
                assert p1.factory is not None
                player1 = p1.factory(t1, p1_account, server_config, False)
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
                assert p2.factory is not None
                player2 = p2.factory(t2, p2_account, server_config, False)
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
                assert p1.factory is not None and p2.factory is not None
                player1 = p1.factory(t1, p1_account, server_config, False)
                player2 = p2.factory(t2, p2_account, server_config, False)
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

    return asyncio.run(_run())


def run_eval_parallel(
    p1: PlayerSpec,
    p2: PlayerSpec,
    t1: TeamProvider,
    t2: TeamProvider,
    num_battles: int,
    server_urls: List[str],
    workers: int,
    run_tag: str,
) -> EvalResult:
    """Fan out ``num_battles`` across ``workers`` and aggregate.

    Workers run in a ``ThreadPoolExecutor``; each calls ``asyncio.run`` on
    its own event loop. Threading (not multiprocessing) is fine here
    because the heavy work is async network I/O against Showdown servers.
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

    server_processes = []
    if args.launch_servers:
        server_processes = launch_showdown_servers(args.num_servers, args.start_port)

    try:
        server_urls = _build_server_urls(
            args.server_base, args.num_servers, args.start_port
        )

        started = time.time()
        print(f"\n=== Evaluation: {p1.name} vs {p2.name} ===")
        result = run_eval_parallel(
            p1=p1,
            p2=p2,
            t1=t1,
            t2=t2,
            num_battles=args.battles,
            server_urls=server_urls,
            workers=args.workers,
            run_tag=run_tag,
        )
        duration = time.time() - started
        _print_result(result, p1.name, p2.name)

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
            }
            with open(args.output, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved evaluation results to {args.output}")
    finally:
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
