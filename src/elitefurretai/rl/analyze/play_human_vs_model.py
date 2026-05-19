"""Play one or more local VGC battles between a human (terminal) and an RL model.

Spawns a local Showdown server (or connects to an existing one), wires up a
HumanPlayer on stdin and a VerboseModelPlayer loaded from a checkpoint, and
runs `human.battle_against(model, n_battles=N)`.

Model debug output (top-k action probabilities, value estimate) is buffered
by default and printed at the start of the NEXT turn alongside the resolved
battle state, so the user does not see the model's pick while choosing
their own action. Pass --reveal to print it inline as the model decides.
"""

import argparse
import asyncio
import contextlib
import io
import sys
from pathlib import Path
from subprocess import Popen
from typing import Any, List, Optional

from poke_env.battle import AbstractBattle
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.agents import HumanPlayer
from elitefurretai.agents.verbose_model_player import VerboseModelPlayer
from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)


class DeferredVerboseModelPlayer(VerboseModelPlayer):
    """VerboseModelPlayer that can buffer its per-turn debug output.

    When `reveal=True`, behaves identically to the parent: prints debug as
    each move is chosen. When `reveal=False`, captures the parent's debug
    output via redirect_stdout and appends it to `pending_debug` for the
    human player to drain on the next turn.
    """

    def __init__(self, *args: Any, reveal: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reveal = reveal
        self.pending_debug: List[str] = []

    def _emit(self, text: str) -> None:
        if self.reveal:
            print(text, end="")
        else:
            self.pending_debug.append(text)

    def flush_debug(self) -> str:
        out = "".join(self.pending_debug)
        self.pending_debug.clear()
        return out

    def _print_debug(self, battle, probs, value, selected, is_teampreview) -> None:  # type: ignore[override]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            super()._print_debug(battle, probs, value, selected, is_teampreview)
        self._emit(buf.getvalue())


class HumanVsModelPlayer(HumanPlayer):
    """HumanPlayer that drains the companion model's pending debug each turn.

    Ensures the user sees what the model decided on the previous turn (and
    why) printed above the freshly resolved state for the new turn.
    """

    def __init__(
        self,
        *args: Any,
        model_player: DeferredVerboseModelPlayer,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_player = model_player

    def choose_move(self, battle: AbstractBattle):  # type: ignore[override]
        pending = self.model_player.flush_debug()
        if pending:
            print("\n--- MODEL (previous turn) ---")
            print(pending, end="")
            print("--- END MODEL ---")
        return super().choose_move(battle)


def _existing_path(value: str) -> Path:
    p = Path(value)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {value}")
    return p


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a local terminal VGC battle against an RL model checkpoint.",
    )
    parser.add_argument("model", type=_existing_path, help="Path to RNaD checkpoint (.pt)")
    parser.add_argument(
        "--human-team",
        type=_existing_path,
        required=True,
        help="Path to team file for the human side",
    )
    parser.add_argument(
        "--model-team",
        type=_existing_path,
        required=True,
        help="Path to team file for the model side",
    )
    parser.add_argument("--battle-format", type=str, default="gen9vgc2023regc")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Sample from the model's action distribution (default: argmax).",
    )
    parser.add_argument(
        "--reveal",
        action="store_true",
        help="Print the model's debug output inline as it decides "
        "(default: buffer and reveal at the start of the next turn).",
    )
    parser.add_argument("--num-battles", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print full battle summary alongside top-k probabilities.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch a local Showdown server (overrides --server to localhost:{start_port}).",
    )
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument("--server", type=str, default="localhost:8000")
    return parser


async def _run(args: argparse.Namespace, server: str) -> None:
    server_config = ServerConfiguration(f"ws://{server}/showdown/websocket", "")

    model_player = DeferredVerboseModelPlayer(
        model_path=str(args.model),
        device=args.device,
        battle_format=args.battle_format,
        probabilistic=args.probabilistic,
        top_k=args.top_k,
        print_summary=args.print_summary,
        account_configuration=AccountConfiguration("Model", None),
        server_configuration=server_config,
        max_concurrent_battles=1,
        start_timer_on_battle_start=False,
        team=str(args.model_team),
        reveal=args.reveal,
    )

    human_player = HumanVsModelPlayer(
        battle_format=args.battle_format,
        account_configuration=AccountConfiguration("Human", None),
        server_configuration=server_config,
        team=str(args.human_team),
        max_concurrent_battles=1,
        accept_open_team_sheet=True,
        model_player=model_player,
    )

    print(f"\nStarting {args.num_battles} battle(s): Human vs {args.model.name}")
    print(f"  Format:    {args.battle_format}")
    print(f"  Human team: {args.human_team}")
    print(f"  Model team: {args.model_team}")
    print(f"  Sampling:  {'probabilistic' if args.probabilistic else 'argmax'}")
    print(f"  Reveal:    {'inline' if args.reveal else 'next-turn'}\n")

    try:
        await human_player.battle_against(model_player, n_battles=args.num_battles)
    finally:
        leftover = model_player.flush_debug()
        if leftover:
            print("\n--- MODEL (final turn) ---")
            print(leftover, end="")
            print("--- END MODEL ---")

    print("\n" + "=" * 60)
    print("FINAL RECORD")
    print("=" * 60)
    print(
        f"Human:  wins={human_player.n_won_battles} "
        f"losses={human_player.n_lost_battles} "
        f"ties={human_player.n_tied_battles}"
    )
    print(f"Total battles finished: {human_player.n_finished_battles}")


def main() -> None:
    args = _build_argparser().parse_args()

    server_processes: Optional[List[Popen[Any]]] = None
    server = args.server
    if args.launch_server:
        server_processes = launch_showdown_servers(1, args.start_port)
        server = f"localhost:{args.start_port}"

    try:
        asyncio.run(_run(args, server))
    finally:
        if server_processes is not None:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    sys.exit(main())
