import argparse
import asyncio
import importlib
import importlib.util
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from poke_env import AccountConfiguration, ServerConfiguration


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_vgc_bench_root() -> Path:
    spec = importlib.util.find_spec("vgc_bench")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("Could not resolve vgc_bench package path")
    package_path = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_path.parent


def _build_vgcbench_player(
    *,
    username: str,
    password: Optional[str],
    server: str,
    battle_format: str,
    checkpoint_path: str,
    team: str,
    accept_open_team_sheet: bool,
):
    vgc_bench_root = _resolve_vgc_bench_root()

    ppo_module = importlib.import_module("stable_baselines3")
    ppo_cls = getattr(ppo_module, "PPO")

    with _temporary_cwd(vgc_bench_root):
        policy_player_module = importlib.import_module("vgc_bench.src.policy_player")
    policy_player_cls = getattr(policy_player_module, "PolicyPlayer")

    policy = ppo_cls.load(checkpoint_path).policy

    return policy_player_cls(
        policy=policy,
        battle_format=battle_format,
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ServerConfiguration(f"ws://{server}/showdown/websocket", ""),
        team=team,
        accept_open_team_sheet=accept_open_team_sheet,
    )


def _wait_for_server(server: str, timeout_s: float) -> None:
    if ":" not in server:
        raise ValueError(f"Expected --server as host:port, got: {server}")

    host, port_str = server.rsplit(":", 1)
    port = int(port_str)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for showdown server {server}")


async def _run(args: argparse.Namespace) -> None:
    _wait_for_server(args.server, args.wait_for_server_timeout)

    team = Path(args.team_file).read_text()
    player = _build_vgcbench_player(
        username=args.username,
        password=args.password if args.password else None,
        server=args.server,
        battle_format=args.battle_format,
        checkpoint_path=args.checkpoint_path,
        team=team,
        accept_open_team_sheet=args.accept_open_team_sheet,
    )

    print(
        f"[vgcbench-runner] accepting {args.n_challenges} challenges on {args.server} "
        f"as {args.username} ({args.battle_format})"
    )
    await player.accept_challenges(
        opponent=args.accept_opponent,
        n_challenges=args.n_challenges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone vgc-bench challenge bot in a dedicated process/environment."
    )
    parser.add_argument("--username", required=True, type=str)
    parser.add_argument("--password", default="", type=str)
    parser.add_argument("--server", default="localhost:8000", type=str)
    parser.add_argument("--battle-format", default="gen9vgc2024regg", type=str)
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--team-file", required=True, type=str)
    parser.add_argument("--n-challenges", default=1, type=int)
    parser.add_argument("--accept-opponent", default=None, type=str)
    parser.add_argument("--accept-open-team-sheet", action="store_true")
    parser.add_argument("--wait-for-server-timeout", default=180.0, type=float)
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
