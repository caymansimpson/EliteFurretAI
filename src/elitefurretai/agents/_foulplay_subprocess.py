"""Internal entry point for the foul-play-doubles subprocess.

Runs under ``../venv-foulplay/bin/python``, NOT EFA's training venv.
Imports FoulPlay's own modules (``config``, ``data.mods.apply_mods``,
``fp.run_battle``, ``fp.websocket_client``, ``teams``) by switching
cwd into the ``../foul-play-doubles/`` checkout at startup — FoulPlay
uses package-relative imports that assume the repo root is cwd.

Leading underscore in the filename signals "internal subprocess
entry — not user-invocable directly." Spawned by
:class:`FoulPlayManager.launch`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import List


def _wait_for_server(server: str, timeout_s: float) -> None:
    """Block until the Showdown server's TCP socket accepts connections."""
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


def _resolve_foulplay_root() -> Path:
    """Find the foul-play-doubles checkout directory.

    Defaults to ``../foul-play-doubles`` relative to the EFA repo root.
    Override via the ``FOULPLAY_ROOT`` environment variable for tests
    or non-standard layouts.
    """
    override = os.environ.get("FOULPLAY_ROOT")
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    # here = .../EliteFurretAI/src/elitefurretai/agents/_foulplay_subprocess.py
    # parents[3] is the EFA repo root; foul-play-doubles is a sibling.
    repo_root = here.parents[3]
    candidate = repo_root.parent / "foul-play-doubles"
    if not candidate.is_dir():
        raise FileNotFoundError(
            f"foul-play-doubles checkout not found at {candidate}. "
            f"Clone it: git clone https://github.com/pmariglia/foul-play-doubles "
            f"{candidate}"
        )
    return candidate


def _list_team_names(team_pool_path: str) -> List[str]:
    """Return a sorted list of team-file basenames (without extension).

    FoulPlay's ``load_team`` looks up by basename in its
    ``teams/<format>/`` directory. Caller is responsible for ensuring
    those files are reachable from FoulPlay's lookup path (typically by
    symlinking the EFA team pool — see RL.md setup section).
    """
    pool = Path(team_pool_path)
    if not pool.is_dir():
        raise FileNotFoundError(f"team_pool_path is not a directory: {pool}")
    return sorted(p.stem for p in pool.iterdir() if p.is_file())


async def _accept_one_challenge(
    ps_websocket_client,
    pokemon_battle,
    team_export,
    team_dict,
    file_name,
    battle_format,
) -> None:
    """Accept one challenge and play it as a single (non-Bo3) battle.

    FoulPlay's ``pokemon_battle`` handles one battle; the Bo3 wrapper
    in ``run.py`` is deliberately bypassed because our eval-pass
    aggregation is per-game, not per-best-of-three.
    """
    await ps_websocket_client.accept_challenge(battle_format, team_export, None)
    # ``pokemon_battle`` takes a per-battle scratchpad list for telemetry.
    # We pass an empty list — FoulPlay logs internally; eval-side metrics
    # come from Showdown's win-rate accounting on the model player.
    await pokemon_battle(ps_websocket_client, team_export, team_dict, file_name, [])


async def _run(args: argparse.Namespace) -> None:
    _wait_for_server(args.server, args.wait_for_server_timeout)

    # Switch cwd into foul-play-doubles before importing it — FoulPlay
    # uses package-relative imports that assume cwd is the repo root.
    foulplay_root = _resolve_foulplay_root()
    os.chdir(str(foulplay_root))
    sys.path.insert(0, str(foulplay_root))

    # Now safe to import FoulPlay modules.
    import logging

    from config import BotModes, FoulPlayConfig, init_logging  # type: ignore
    from data.mods.apply_mods import apply_mods  # type: ignore
    from fp.run_battle import pokemon_battle  # type: ignore
    from fp.websocket_client import PSWebsocketClient  # type: ignore
    from teams import load_team  # type: ignore

    # Configure FoulPlay's global config from CLI args.
    FoulPlayConfig.battle_bot_module = "search"
    FoulPlayConfig.websocket_uri = f"ws://{args.server}/showdown/websocket"
    FoulPlayConfig.username = args.username
    FoulPlayConfig.password = ""
    FoulPlayConfig.avatar = None
    FoulPlayConfig.bot_mode = BotModes.accept_challenge
    FoulPlayConfig.pokemon_format = args.battle_format
    FoulPlayConfig.search_time_ms = args.search_time_ms
    FoulPlayConfig.parallelism = args.parallelism
    FoulPlayConfig.user_to_challenge = ""
    FoulPlayConfig.save_replay = False
    FoulPlayConfig.team_list = None
    FoulPlayConfig.team_name = ""

    init_logging("INFO", False)
    apply_mods(FoulPlayConfig.pokemon_format)

    ps = await PSWebsocketClient.create(
        FoulPlayConfig.username,
        FoulPlayConfig.password,
        FoulPlayConfig.websocket_uri,
    )
    await ps.login()

    team_names = _list_team_names(args.team_list_dir)
    if not team_names:
        raise RuntimeError(f"No team files found in {args.team_list_dir}")

    print(
        f"[foulplay-runner] accepting {args.n_challenges} challenges on "
        f"{args.server} as {args.username} ({args.battle_format}), "
        f"search_time_ms={args.search_time_ms}, parallelism={args.parallelism}, "
        f"team_pool_size={len(team_names)}",
        flush=True,
    )

    for i in range(args.n_challenges):
        team_name = random.choice(team_names)
        # load_team returns (team_export, team_dict, file_name).
        try:
            team_export, team_dict, file_name = load_team(team_name)
        except Exception as exc:
            print(
                f"[foulplay-runner] load_team({team_name!r}) failed: {exc}",
                flush=True,
            )
            continue

        try:
            await _accept_one_challenge(
                ps,
                pokemon_battle,
                team_export,
                team_dict,
                file_name,
                args.battle_format,
            )
        except Exception as exc:
            # If a battle crashes, log and continue. The eval driver
            # records this as a loss for the model side via Showdown's
            # win-rate accounting, and the next challenge proceeds.
            print(
                f"[foulplay-runner] battle {i + 1}/{args.n_challenges} crashed: {exc}",
                flush=True,
            )
            logging.exception("foulplay-runner battle crash")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="foul-play-doubles bot subprocess entry — internal."
    )
    parser.add_argument("--username", required=True, type=str)
    parser.add_argument("--server", default="localhost:8000", type=str)
    parser.add_argument("--battle-format", default="gen9vgc2024regg", type=str)
    parser.add_argument("--n-challenges", default=1, type=int)
    parser.add_argument("--team-list-dir", required=True, type=str)
    parser.add_argument("--search-time-ms", default=750, type=int)
    parser.add_argument("--parallelism", default=4, type=int)
    parser.add_argument("--wait-for-server-timeout", default=180.0, type=float)
    parser.add_argument("--accept-open-team-sheet", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
