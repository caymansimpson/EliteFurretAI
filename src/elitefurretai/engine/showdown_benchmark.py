import argparse
import cProfile
import io
import pstats
import random
import resource
import time
from pathlib import Path
from typing import cast

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player, RandomPlayer

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.model_io import build_model_from_config, load_agent_from_checkpoint
from elitefurretai.rl.players import BatchInferencePlayer, RNaDAgent


def _load_team_text(format_id: str, team_path: str | None, repo: TeamRepo) -> str:
    if team_path is None:
        return repo.sample_team(format_id)
    return Path(team_path).read_text()


def _build_agent(config: RNaDConfig, device: str, checkpoint: str | None) -> RNaDAgent:
    if checkpoint is not None:
        return load_agent_from_checkpoint(checkpoint, device)

    embedder = Embedder(
        format=config.battle_format,
        feature_set=config.embedder_feature_set,
        omniscient=False,
    )
    model = build_model_from_config(config.to_dict(), embedder, device, None)
    return RNaDAgent(model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the local Showdown websocket backend with random or model-backed players."
    )
    parser.add_argument("--format", default="gen9vgc2024regg")
    parser.add_argument("--battles", type=int, default=500)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy", choices=["random", "model"], default="random")
    parser.add_argument("--max-concurrent-battles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=int, default=50)
    parser.add_argument("--team-path")
    parser.add_argument("--opponent-team-path")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-sort", default="cumulative")
    parser.add_argument("--profile-top-n", type=int, default=40)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout", type=float, default=0.01)
    parser.add_argument("--feature-set")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-battle-steps", type=int, default=40)
    return parser


async def _run_benchmark(args: argparse.Namespace) -> None:
    repo = TeamRepo(shuffle=False)
    rng = random.Random(args.seed)
    p1_team = _load_team_text(args.format, args.team_path, repo)
    if args.opponent_team_path is not None:
        p2_team = Path(args.opponent_team_path).read_text()
    elif args.no_mirror:
        p2_team = repo.sample_team(args.format)
    else:
        p2_team = p1_team

    server_launch_start = time.perf_counter()
    server_processes = launch_showdown_servers(1, args.port)
    server_launch_seconds = time.perf_counter() - server_launch_start
    teardown_seconds = 0.0

    profile: cProfile.Profile | None = None
    battle_loop_seconds = 0.0
    player_setup_seconds = 0.0
    player1: Player | None = None
    player2: Player | None = None

    try:
        setup_start = time.perf_counter()
        suffix = rng.randint(100000, 999999)
        server_config = ServerConfiguration(f"ws://localhost:{args.port}/showdown/websocket", "")

        if args.policy == "random":
            player1 = RandomPlayer(
                account_configuration=AccountConfiguration(f"showbenchp1{suffix}", None),
                battle_format=args.format,
                team=p1_team,
                max_concurrent_battles=args.max_concurrent_battles,
                server_configuration=server_config,
                log_level=args.log_level,
            )
            player2 = RandomPlayer(
                account_configuration=AccountConfiguration(f"showbenchp2{suffix}", None),
                battle_format=args.format,
                team=p2_team,
                max_concurrent_battles=args.max_concurrent_battles,
                server_configuration=server_config,
                log_level=args.log_level,
            )
        else:
            if args.config is None:
                raise ValueError("--config is required for --policy model")

            config = RNaDConfig.load(args.config)
            feature_set = args.feature_set or config.embedder_feature_set
            temperature = args.temperature if args.temperature is not None else config.temperature_at_step(0)
            top_p = args.top_p if args.top_p is not None else config.top_p

            p1_agent = _build_agent(config, args.device, args.checkpoint)
            p2_agent = _build_agent(config, args.device, args.opponent_checkpoint or args.checkpoint)
            embedder = Embedder(
                format=config.battle_format,
                feature_set=feature_set,
                omniscient=False,
            )
            player1 = BatchInferencePlayer(
                p1_agent,
                device=args.device,
                batch_size=args.batch_size,
                batch_timeout=args.batch_timeout,
                probabilistic=not args.greedy,
                embedder=embedder,
                max_battle_steps=args.max_battle_steps,
                battle_format=config.battle_format,
                team=p1_team,
                max_concurrent_battles=args.max_concurrent_battles,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"showbenchp1{suffix}", None),
                log_level=args.log_level,
            )
            player2 = BatchInferencePlayer(
                p2_agent,
                device=args.device,
                batch_size=args.batch_size,
                batch_timeout=args.batch_timeout,
                probabilistic=not args.greedy,
                embedder=embedder,
                max_battle_steps=args.max_battle_steps,
                battle_format=config.battle_format,
                team=p2_team,
                max_concurrent_battles=args.max_concurrent_battles,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"showbenchp2{suffix}", None),
                log_level=args.log_level,
            )
            model_player1 = cast(BatchInferencePlayer, player1)
            model_player2 = cast(BatchInferencePlayer, player2)
            model_player1.temperature = temperature
            model_player1.top_p = top_p
            model_player2.temperature = temperature
            model_player2.top_p = top_p
            model_player1.start_inference_loop()
            model_player2.start_inference_loop()

        assert player1 is not None
        assert player2 is not None
        await player1.ps_client.wait_for_login()
        await player2.ps_client.wait_for_login()
        player_setup_seconds = time.perf_counter() - setup_start

        if args.profile_output is not None:
            profile = cProfile.Profile()
            profile.enable()

        battle_loop_start = time.perf_counter()
        await player1.battle_against(player2, n_battles=args.battles)
        battle_loop_seconds = time.perf_counter() - battle_loop_start

        if profile is not None:
            profile.disable()
            profile_buffer = io.StringIO()
            stats = pstats.Stats(profile, stream=profile_buffer)
            stats.sort_stats(args.profile_sort)
            stats.print_stats(args.profile_top_n)
            Path(args.profile_output).write_text(profile_buffer.getvalue())

        total_duration = server_launch_seconds + player_setup_seconds + battle_loop_seconds
        usage = resource.getrusage(resource.RUSAGE_SELF)
        print(f"completed_battles={player1.n_finished_battles}")
        print(f"p1_wins={player1.n_won_battles}")
        print(f"duration_seconds={total_duration:.3f}")
        print(f"battles_per_second={player1.n_finished_battles / max(total_duration, 1e-9):.3f}")
        print(f"server_launch_seconds={server_launch_seconds:.3f}")
        print(f"player_setup_seconds={player_setup_seconds:.3f}")
        print(f"battle_loop_seconds={battle_loop_seconds:.3f}")
        print(f"cpu_user_seconds={usage.ru_utime:.3f}")
        print(f"cpu_system_seconds={usage.ru_stime:.3f}")
        print(f"policy={args.policy}")
        print(f"max_concurrent_battles={args.max_concurrent_battles}")
        if args.policy == "model":
            print(f"batch_size={args.batch_size}")
            print(f"batch_timeout={args.batch_timeout}")
            print(f"device={args.device}")
            print(f"feature_set={args.feature_set or config.embedder_feature_set}")
            print(f"temperature={temperature}")
            print(f"top_p={top_p}")
        print(f"profile_output={args.profile_output or 'disabled'}")
    finally:
        teardown_start = time.perf_counter()
        if isinstance(player1, BatchInferencePlayer):
            player1.teardown_runtime()
        if isinstance(player2, BatchInferencePlayer):
            player2.teardown_runtime()
        shutdown_showdown_servers(server_processes)
        teardown_seconds = time.perf_counter() - teardown_start
        print(f"teardown_seconds={teardown_seconds:.3f}")


def main() -> None:
    import asyncio

    args = build_parser().parse_args()
    random.seed(args.seed)
    asyncio.run(_run_benchmark(args))


if __name__ == "__main__":
    main()
