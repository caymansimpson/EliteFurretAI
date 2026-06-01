"""Standalone subprocess entry point for a vgc-bench bot serving EFA as an opponent.

This file is **not** user-invokable — it is spawned automatically by
``elitefurretai.agents.vgcbench_manager.VGCBenchManager.launch()`` as part of training.
The leading underscore in the filename signals "internal".

Why this exists as a separate script
------------------------------------
vgc-bench depends on the cameronangliss/poke-env fork (pinned to commit
the ``@vgc-bench`` branch tip, currently ``e9b61cdf``, a 0.15.0 build whose
VGC enums yield a 764-wide observation),
which cannot coexist with EFA's own poke-env in one interpreter. To isolate
them, the trainer spawns this script in vgc-bench's own venv
(``../venv-vgcbench-bcsp/`` by convention, configured via
``curriculum.external_vgcbench_python_executable``). The subprocess logs into a
Showdown server, loads a SB3 PPO checkpoint via vgc-bench's ``PolicyPlayer``,
and sits in ``accept_challenges`` accepting battles from EFA workers.

How it plugs into training
--------------------------
``OpponentPool.external_vgcbench_usernames`` (``rl/opponents.py``) holds the
usernames these runners log in as, populated by ``VGCBenchManager.launch()``.
When ``VGC_BENCH`` is sampled from the curriculum, EFA workers route
the battle to one of those usernames via ``/challenge`` instead of
constructing an in-process ``PolicyPlayer``.

Relationship to ``rl/players.py:_create_vgc_bench_player``
----------------------------------------------------------
This script runs in a separate venv that cannot import
``elitefurretai``. Keep the two in sync if vgc-bench's loader contract
changes — they are siblings, not the same code path.

CLI
---
Required: ``--username``, ``--checkpoint-path``, ``--team-file``.
The script waits up to ``--wait-for-server-timeout`` seconds for the Showdown
server's TCP port to come up before connecting (useful because this runner
is typically spawned alongside the server by ``train.py``).
"""

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
