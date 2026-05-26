"""Internal entry point for the foul-play-doubles subprocess.

Runs under ``../venv-foulplay/bin/python``, NOT EFA's training venv.
Imports FoulPlay's own modules (``config``, ``data.mods.apply_mods``,
``fp.run_battle``, ``fp.websocket_client``, ``teams``) by switching
cwd into the ``../foul-play-doubles/`` checkout at startup — FoulPlay
uses package-relative imports that assume the repo root is cwd.

Leading underscore in the filename signals "internal subprocess
entry — not user-invocable directly." Spawned by
:class:`FoulPlayManager.launch`.

Pinned to foul-play-doubles commit ``8550b93``; later commits use
``poke_engine.TeamPreviewFilters``, which is not exported by
``poke-engine-doubles==0.0.7`` (the only version on PyPI). See
``RL.md`` "FoulPlay eval setup" for the matching setup steps.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time
from pathlib import Path


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


def _ensure_team_pool_visible(foulplay_root: Path, team_list_dir: str) -> str:
    """Make the configured team pool reachable from foul-play's load_team().

    foul-play-doubles' ``load_team(name)`` looks under
    ``<foul-play-doubles>/teams/teams/<name>``. We accept ``--team-list-dir``
    as an absolute path on the EFA side, then symlink (or reuse a matching
    existing symlink) that path into ``<foul-play-doubles>/teams/teams/<basename>``.
    Returns the basename to hand to ``load_team()``.

    Idempotent: if the symlink already points at the right absolute
    target, nothing changes. If a different file exists at that name,
    raises rather than overwriting.
    """
    src = Path(team_list_dir).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"team_list_dir is not a directory: {src}")

    dst_dir = foulplay_root / "teams" / "teams"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dst.is_symlink():
        if dst.resolve() == src:
            return src.name
        raise FileExistsError(
            f"Symlink {dst} already points at {dst.resolve()}, expected {src}. "
            f"Remove the existing symlink to re-link."
        )
    if dst.exists():
        raise FileExistsError(
            f"{dst} already exists and is not a symlink. "
            f"Remove it before launching the FoulPlay subprocess."
        )

    dst.symlink_to(src, target_is_directory=True)
    return src.name


async def _run(args: argparse.Namespace) -> None:
    _wait_for_server(args.server, args.wait_for_server_timeout)

    # Resolve --team-list-dir to absolute BEFORE chdir — once cwd
    # moves to foul-play-doubles, relative paths from the parent
    # process can't be recovered.
    team_list_dir_abs = str(Path(args.team_list_dir).resolve())

    # Switch cwd into foul-play-doubles before importing it — FoulPlay
    # uses package-relative imports that assume cwd is the repo root.
    foulplay_root = _resolve_foulplay_root()
    os.chdir(str(foulplay_root))
    sys.path.insert(0, str(foulplay_root))

    # Now safe to import FoulPlay modules.
    import logging

    from config import BotModes, FoulPlayConfig, SaveReplay, init_logging  # type: ignore
    from data.mods.apply_mods import apply_mods  # type: ignore
    from fp.run_battle import pokemon_battle  # type: ignore
    from fp.websocket_client import PSWebsocketClient  # type: ignore
    from teams import load_team  # type: ignore

    # Make the EFA team pool reachable from load_team() — see helper.
    team_basename = _ensure_team_pool_visible(foulplay_root, team_list_dir_abs)

    # Populate the singleton-style FoulPlayConfig before init_logging
    # and apply_mods read its fields.
    FoulPlayConfig.websocket_uri = f"ws://{args.server}/showdown/websocket"
    FoulPlayConfig.username = args.username
    FoulPlayConfig.password = ""
    FoulPlayConfig.avatar = None
    FoulPlayConfig.bot_mode = BotModes.accept_challenge
    FoulPlayConfig.pokemon_format = args.battle_format
    FoulPlayConfig.smogon_stats = None
    FoulPlayConfig.search_time_ms = args.search_time_ms
    FoulPlayConfig.parallelism = args.parallelism
    FoulPlayConfig.run_count = args.n_challenges
    # team_name is the folder basename under teams/teams/. load_team()
    # samples a random file from inside it for each call.
    FoulPlayConfig.team_name = team_basename
    FoulPlayConfig.team_list = None
    FoulPlayConfig.user_to_challenge = None
    FoulPlayConfig.save_replay = SaveReplay.never
    FoulPlayConfig.room_name = None
    FoulPlayConfig.log_level = "INFO"
    FoulPlayConfig.log_to_file = False

    init_logging("INFO", False)
    apply_mods(FoulPlayConfig.pokemon_format)

    ps = await PSWebsocketClient.create(
        FoulPlayConfig.username,
        FoulPlayConfig.password,
        FoulPlayConfig.websocket_uri,
    )
    await ps.login()

    print(
        f"[foulplay-runner] accepting {args.n_challenges} challenges on "
        f"{args.server} as {args.username} ({args.battle_format}), "
        f"search_time_ms={args.search_time_ms}, parallelism={args.parallelism}, "
        f"team_pool={team_basename}",
        flush=True,
    )

    # One single-battle challenge per iteration. We bypass run.py's
    # Bo3 wrapper by passing best_of_3_room_name=None — pokemon_battle
    # returns once battle_is_finished fires, which is the single-battle
    # exit path. The bo3_is_finished branch never matches None.
    for i in range(args.n_challenges):
        try:
            team_export, team_dict, file_name = load_team(team_basename)
        except Exception as exc:
            print(
                f"[foulplay-runner] load_team({team_basename!r}) failed: {exc}",
                flush=True,
            )
            continue

        try:
            await ps.accept_challenge(args.battle_format, team_export, None)
            winner, _bo3_done = await pokemon_battle(
                ps, args.battle_format, None, True
            )
            print(
                f"[foulplay-runner] battle {i + 1}/{args.n_challenges} "
                f"team={file_name} winner={winner}",
                flush=True,
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

    await ps.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="foul-play-doubles bot subprocess entry — internal."
    )
    parser.add_argument("--username", required=True, type=str)
    parser.add_argument("--server", default="localhost:8000", type=str)
    parser.add_argument("--battle-format", default="gen9vgc2024regg", type=str)
    parser.add_argument("--n-challenges", default=1, type=int)
    parser.add_argument(
        "--team-list-dir",
        required=True,
        type=str,
        help="Absolute path to an EFA team-pool directory. The script "
        "symlinks it under <foul-play-doubles>/teams/teams/<basename> "
        "so load_team() can find it.",
    )
    parser.add_argument("--search-time-ms", default=750, type=int)
    parser.add_argument("--parallelism", default=4, type=int)
    parser.add_argument("--wait-for-server-timeout", default=180.0, type=float)
    parser.add_argument("--accept-open-team-sheet", action="store_true")
    args = parser.parse_args()

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
