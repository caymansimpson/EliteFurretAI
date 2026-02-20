import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from poke_env.player import MaxBasePowerPlayer, Player
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.etl import TeamRepo
from elitefurretai.rl.model_io import load_agent_from_checkpoint
from elitefurretai.rl.opponents import _create_vgc_bench_player
from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer
from elitefurretai.rl.server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)


@dataclass
class EvalResult:
    label: str
    model_wins: int
    opponent_wins: int
    battles_played: int

    @property
    def win_rate(self) -> float:
        if self.battles_played == 0:
            return 0.0
        return self.model_wins / self.battles_played


def _build_server_urls(server_base: str, num_servers: int, start_port: int) -> List[str]:
    if server_base != "localhost":
        return [f"{server_base}"]
    return [f"localhost:{start_port + i}" for i in range(num_servers)]


def _split_battles(total_battles: int, workers: int) -> List[int]:
    workers = max(1, workers)
    base = total_battles // workers
    rem = total_battles % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _load_sample_team(format_name: str) -> str:
    team_repo = TeamRepo(filepath="data/teams")
    return team_repo.sample_team(format_name)


def _build_baseline_team_provider(
    *,
    battle_format: str,
    baseline_team_file: Optional[str],
    baseline_team_dir: Optional[str],
) -> Callable[[], str]:
    if baseline_team_file and baseline_team_dir:
        raise ValueError("Use only one of --baseline-team-file or --baseline-team-dir")

    if baseline_team_file:
        fixed_team = Path(baseline_team_file).read_text()
        return lambda: fixed_team

    if baseline_team_dir:
        team_repo = TeamRepo(filepath=baseline_team_dir)
        return lambda: team_repo.sample_team(battle_format)

    fallback_team = _load_sample_team(battle_format)
    return lambda: fallback_team


def _baseline_user_tag(baseline_key: str) -> str:
    mapping = {
        "maxdamage": "MD",
        "maxbasepower": "MBP",
        "shp": "SHP",
        "vgcbench": "VGB",
    }
    return mapping.get(baseline_key, baseline_key[:3].upper())


def _username(prefix: str, worker_id: int, run_tag: str) -> str:
    return f"{prefix}{worker_id}{run_tag}"[:18]


def _run_worker_model_vs_model(
    worker_id: int,
    model1_path: str,
    model2_path: str,
    battles: int,
    server_url: str,
    battle_format: str,
    device: str,
    batch_size: int,
    run_tag: str,
) -> EvalResult:
    async def _run() -> EvalResult:
        model1 = load_agent_from_checkpoint(model1_path, device)
        model2 = load_agent_from_checkpoint(model2_path, device)
        team1 = _load_sample_team(battle_format)
        team2 = _load_sample_team(battle_format)

        server_config = ServerConfiguration(f"ws://{server_url}/showdown/websocket", "")
        player1 = BatchInferencePlayer(
            model=model1,
            device=device,
            batch_size=batch_size,
            probabilistic=False,
            worker_id=worker_id * 2,
            account_configuration=AccountConfiguration(_username("EM1", worker_id, run_tag), None),
            server_configuration=server_config,
            battle_format=battle_format,
            team=team1,
            accept_open_team_sheet=True,
        )
        player2 = BatchInferencePlayer(
            model=model2,
            device=device,
            batch_size=batch_size,
            probabilistic=False,
            worker_id=worker_id * 2 + 1,
            account_configuration=AccountConfiguration(_username("EM2", worker_id, run_tag), None),
            server_configuration=server_config,
            battle_format=battle_format,
            team=team2,
            accept_open_team_sheet=True,
        )

        player1.start_inference_loop()
        player2.start_inference_loop()
        await player1.battle_against(player2, n_battles=battles)

        played = player1.n_finished_battles
        wins = player1.n_won_battles
        losses = player1.n_lost_battles
        return EvalResult("model_vs_model", wins, losses, played)

    return asyncio.run(_run())


def _run_worker_model_vs_baseline(
    worker_id: int,
    model_path: str,
    baseline_name: str,
    battles: int,
    server_url: str,
    battle_format: str,
    device: str,
    batch_size: int,
    vgc_bench_checkpoint_path: str,
    baseline_team_file: Optional[str],
    baseline_team_dir: Optional[str],
    external_vgcbench_usernames: Optional[List[str]],
    run_tag: str,
) -> EvalResult:
    async def _run() -> EvalResult:
        model = load_agent_from_checkpoint(model_path, device)
        server_config = ServerConfiguration(f"ws://{server_url}/showdown/websocket", "")
        model_team = _load_sample_team(battle_format)
        baseline_team_provider = _build_baseline_team_provider(
            battle_format=battle_format,
            baseline_team_file=baseline_team_file,
            baseline_team_dir=baseline_team_dir,
        )

        model_player = BatchInferencePlayer(
            model=model,
            device=device,
            batch_size=batch_size,
            probabilistic=False,
            worker_id=worker_id,
            account_configuration=AccountConfiguration(
                _username(f"EM{_baseline_user_tag(baseline_name.lower())}", worker_id, run_tag),
                None,
            ),
            server_configuration=server_config,
            battle_format=battle_format,
            team=model_team,
            accept_open_team_sheet=False,
        )
        model_player.start_inference_loop()

        baseline_key = baseline_name.lower()
        baseline_prefix = f"EB{_baseline_user_tag(baseline_key)}"

        if baseline_key == "vgcbench" and external_vgcbench_usernames:
            opponent_username = external_vgcbench_usernames[
                worker_id % len(external_vgcbench_usernames)
            ]
            await model_player.send_challenges(opponent_username, battles)

            played = model_player.n_finished_battles
            wins = model_player.n_won_battles
            losses = model_player.n_lost_battles
            return EvalResult(baseline_key, wins, losses, played)

        def _make_opponent(baseline_username: str, team: str) -> Player:
            opponent: Player
            if baseline_key == "maxdamage":
                opponent = MaxDamagePlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(baseline_username, None),
                    server_configuration=server_config,
                    team=team,
                    accept_open_team_sheet=False,
                )
            elif baseline_key == "maxbasepower":
                opponent = MaxBasePowerPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(baseline_username, None),
                    server_configuration=server_config,
                    team=team,
                    accept_open_team_sheet=False,
                )
            elif baseline_key == "shp":
                opponent = SimpleHeuristicsPlayer(
                    battle_format=battle_format,
                    account_configuration=AccountConfiguration(baseline_username, None),
                    server_configuration=server_config,
                    team=team,
                    accept_open_team_sheet=False,
                )
            elif baseline_key == "vgcbench":
                opponent = _create_vgc_bench_player(
                    device=device,
                    player_config=AccountConfiguration(baseline_username, None),
                    server_config=server_config,
                    team=team,
                    battle_format=battle_format,
                    checkpoint_path=vgc_bench_checkpoint_path,
                    accept_open_team_sheet=False,
                )
            else:
                raise ValueError(f"Unknown baseline '{baseline_key}'")

            return opponent

        if baseline_team_file:
            shared_opponent = _make_opponent(
                _username(baseline_prefix, worker_id, run_tag), baseline_team_provider()
            )
            try:
                await model_player.battle_against(shared_opponent, n_battles=battles)
            except Exception as exc:
                print(f"[eval] baseline={baseline_key} worker={worker_id} failed: {exc}")
        else:
            for battle_idx in range(battles):
                baseline_username = _username(
                    baseline_prefix,
                    worker_id,
                    f"{run_tag}{battle_idx:03d}",
                )
                team = baseline_team_provider()
                opponent = _make_opponent(baseline_username, team)

                try:
                    await model_player.battle_against(opponent, n_battles=1)
                except Exception as exc:
                    print(f"[eval] baseline={baseline_key} worker={worker_id} failed: {exc}")

        played = model_player.n_finished_battles
        wins = model_player.n_won_battles
        losses = model_player.n_lost_battles
        return EvalResult(baseline_key, wins, losses, played)

    return asyncio.run(_run())


def _aggregate_results(label: str, results: List[EvalResult]) -> EvalResult:
    return EvalResult(
        label=label,
        model_wins=sum(r.model_wins for r in results),
        opponent_wins=sum(r.opponent_wins for r in results),
        battles_played=sum(r.battles_played for r in results),
    )


def _print_result(result: EvalResult, opponent_label: str) -> None:
    print(
        f"{result.label:>16} vs {opponent_label:<16} | "
        f"Battles={result.battles_played:<5} "
        f"Wins={result.model_wins:<5} "
        f"Losses={result.opponent_wins:<5} "
        f"WR={result.win_rate * 100:6.2f}%"
    )


def run_model_vs_model_parallel(
    model1_path: str,
    model2_path: str,
    num_battles: int,
    server_urls: List[str],
    workers: int,
    battle_format: str,
    device: str,
    batch_size: int,
    run_tag: str,
) -> EvalResult:
    splits = [s for s in _split_battles(num_battles, workers) if s > 0]
    if not splits:
        return EvalResult("model", 0, 0, 0)

    with ThreadPoolExecutor(max_workers=len(splits)) as pool:
        futures = []
        for worker_id, battles in enumerate(splits):
            server_url = server_urls[worker_id % len(server_urls)]
            futures.append(
                pool.submit(
                    _run_worker_model_vs_model,
                    worker_id,
                    model1_path,
                    model2_path,
                    battles,
                    server_url,
                    battle_format,
                    device,
                    batch_size,
                    run_tag,
                )
            )

        results: List[EvalResult] = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:
                print(f"[eval] model_vs_model worker failed: {exc}")
                results.append(EvalResult("model_vs_model", 0, 0, 0))
        return _aggregate_results("model", results)


def run_model_vs_baseline_parallel(
    model_path: str,
    baseline_name: str,
    num_battles: int,
    server_urls: List[str],
    workers: int,
    battle_format: str,
    device: str,
    batch_size: int,
    vgc_bench_checkpoint_path: str,
    baseline_team_file: Optional[str],
    baseline_team_dir: Optional[str],
    external_vgcbench_usernames: Optional[List[str]],
    run_tag: str,
) -> EvalResult:
    splits = [s for s in _split_battles(num_battles, workers) if s > 0]
    if not splits:
        return EvalResult("model", 0, 0, 0)

    with ThreadPoolExecutor(max_workers=len(splits)) as pool:
        futures = []
        for worker_id, battles in enumerate(splits):
            server_url = server_urls[worker_id % len(server_urls)]
            futures.append(
                pool.submit(
                    _run_worker_model_vs_baseline,
                    worker_id,
                    model_path,
                    baseline_name,
                    battles,
                    server_url,
                    battle_format,
                    device,
                    batch_size,
                    vgc_bench_checkpoint_path,
                    baseline_team_file,
                    baseline_team_dir,
                    external_vgcbench_usernames,
                    run_tag,
                )
            )
        results: List[EvalResult] = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:
                print(f"[eval] baseline={baseline_name} worker failed: {exc}")
                results.append(EvalResult(baseline_name, 0, 0, 0))
        return _aggregate_results(baseline_name, results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast multi-server/multi-threaded RL model evaluation"
    )
    parser.add_argument("model1", type=str, help="Path to evaluated model checkpoint")
    parser.add_argument(
        "model2",
        nargs="?",
        default=None,
        help="Optional path to opponent model checkpoint",
    )
    parser.add_argument("--num-battles", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument(
        "--launch-servers",
        action="store_true",
        help="Launch local Showdown servers automatically",
    )
    parser.add_argument("--server-base", type=str, default="localhost")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--battle-format", type=str, default="gen9vgc2023regc")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--baselines",
        type=str,
        default="maxdamage,maxbasepower,shp,vgcbench",
        help="Comma-separated baseline suite to run (default: maxdamage,maxbasepower,shp,vgcbench). Use empty string to disable.",
    )
    parser.add_argument(
        "--vgc-bench-checkpoint-path",
        type=str,
        default="data/models/vgc-bench-sb3-model.zip",
        help="Path to the SB3 vgc-bench checkpoint zip used for vgcbench baseline.",
    )
    parser.add_argument(
        "--baseline-team-file",
        type=str,
        default=None,
        help="Optional team file used by all baselines in every battle.",
    )
    parser.add_argument(
        "--baseline-team-dir",
        type=str,
        default=None,
        help="Optional TeamRepo root directory to sample a new baseline team each battle.",
    )
    parser.add_argument(
        "--external-vgcbench-usernames",
        type=str,
        default="",
        help="Comma-separated external Showdown usernames for vgcbench baseline (uses send_challenges instead of local vgc-bench policy player).",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.baseline_team_file and args.baseline_team_dir:
        raise ValueError("Use only one of --baseline-team-file or --baseline-team-dir")

    run_tag = format(int(time.time() * 1000) % 65536, "04x")
    external_vgcbench_usernames = [
        u.strip()
        for u in args.external_vgcbench_usernames.split(",")
        if u.strip()
    ]

    server_processes = []
    if args.launch_servers:
        server_processes = launch_showdown_servers(args.num_servers, args.start_port)

    try:
        server_urls = _build_server_urls(args.server_base, args.num_servers, args.start_port)
        all_results: Dict[str, Any] = {}

        print("\n=== Model Evaluation ===")
        if args.model2:
            mm_result = run_model_vs_model_parallel(
                model1_path=args.model1,
                model2_path=args.model2,
                num_battles=args.num_battles,
                server_urls=server_urls,
                workers=args.workers,
                battle_format=args.battle_format,
                device=args.device,
                batch_size=args.batch_size,
                run_tag=run_tag,
            )
            _print_result(mm_result, "model2")
            all_results["model_vs_model"] = mm_result.__dict__ | {
                "win_rate": mm_result.win_rate
            }

        baseline_names = [b.strip().lower() for b in args.baselines.split(",") if b.strip()]
        if baseline_names:
            print("\n=== Baseline Evaluation ===")
            baseline_results: Dict[str, Any] = {}
            for baseline in baseline_names:
                result = run_model_vs_baseline_parallel(
                    model_path=args.model1,
                    baseline_name=baseline,
                    num_battles=args.num_battles,
                    server_urls=server_urls,
                    workers=args.workers,
                    battle_format=args.battle_format,
                    device=args.device,
                    batch_size=args.batch_size,
                    vgc_bench_checkpoint_path=args.vgc_bench_checkpoint_path,
                    baseline_team_file=args.baseline_team_file,
                    baseline_team_dir=args.baseline_team_dir,
                    external_vgcbench_usernames=external_vgcbench_usernames,
                    run_tag=run_tag,
                )
                _print_result(result, baseline)
                baseline_results[baseline] = result.__dict__ | {"win_rate": result.win_rate}
            all_results["baselines"] = baseline_results

        if args.output:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nSaved evaluation results to {args.output}")
    finally:
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
