import argparse
import asyncio
from subprocess import Popen
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.rl.fast_action_mask import fast_get_action_mask
from elitefurretai.rl.model_io import load_agent_from_checkpoint
from elitefurretai.rl.players import RNaDAgent


class VerboseModelPlayer(Player):
    def __init__(
        self,
        model_path: str,
        device: str,
        battle_format: str,
        probabilistic: bool,
        top_k: int,
        print_summary: bool,
        account_configuration: AccountConfiguration,
        server_configuration: ServerConfiguration,
        max_concurrent_battles: int,
        start_timer_on_battle_start: bool,
    ):
        super().__init__(
            battle_format=battle_format,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            accept_open_team_sheet=True,
            max_concurrent_battles=max_concurrent_battles,
            start_timer_on_battle_start=start_timer_on_battle_start,
        )
        self.agent: RNaDAgent = load_agent_from_checkpoint(model_path, device)
        self.device = device
        self.embedder = Embedder(format=battle_format, feature_set=Embedder.FULL, omniscient=False)
        self.probabilistic = probabilistic
        self.top_k = top_k
        self.print_summary = print_summary
        self.hidden_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_hidden(self, battle_tag: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if battle_tag not in self.hidden_states:
            self.hidden_states[battle_tag] = self.agent.get_initial_state(1, self.device)
        return self.hidden_states[battle_tag]

    def _describe_action(self, battle: DoubleBattle, action_idx: int, is_teampreview: bool) -> str:
        try:
            if is_teampreview:
                return MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW).message
            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            mdbo = MDBO.from_int(action_idx, type=action_type)
            order = mdbo.to_double_battle_order(battle)
            if hasattr(order, "message"):
                return str(order.message)
            return str(order)
        except Exception:
            return f"action[{action_idx}]"

    def _print_debug(
        self,
        battle: DoubleBattle,
        probs: np.ndarray,
        value: float,
        selected: int,
        is_teampreview: bool,
    ) -> None:
        print("\n" + "=" * 80)
        print(f"Battle: {battle.battle_tag} | Turn: {battle.turn} | Teampreview: {battle.teampreview}")
        print(f"State value estimate: {value:.4f}")

        topk_indices = np.argsort(probs)[-self.top_k :][::-1]
        print(f"Top-{self.top_k} actions:")
        for rank, idx in enumerate(topk_indices, start=1):
            desc = self._describe_action(battle, int(idx), is_teampreview)
            print(f"  {rank}. p={probs[idx]:.4f} | {desc}")

        selected_desc = self._describe_action(battle, int(selected), is_teampreview)
        print(f"Selected: p={probs[selected]:.4f} | {selected_desc}")

        if self.print_summary:
            print("\nBattle summary:")
            print(battle_to_str(battle))

    def choose_move(self, battle: AbstractBattle) -> Any:
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        if battle.battle_tag in self.hidden_states and battle.finished:
            del self.hidden_states[battle.battle_tag]

        state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        hidden = self._get_hidden(battle.battle_tag)

        with torch.no_grad():
            turn_logits, tp_logits, value, next_hidden = self.agent(state_tensor, hidden)

        self.hidden_states[battle.battle_tag] = (next_hidden[0], next_hidden[1])

        is_teampreview = battle.teampreview
        if is_teampreview:
            probs = torch.softmax(tp_logits[0, 0], dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(turn_logits[0, 0], dim=-1).cpu().numpy()
            mask = fast_get_action_mask(battle)
            probs = probs * mask
            probs = probs / probs.sum() if probs.sum() > 0 else mask / mask.sum()

        selected = int(np.random.choice(np.arange(len(probs)), p=probs)) if self.probabilistic else int(np.argmax(probs))
        state_value = float(value[0, 0].item())
        self._print_debug(battle, probs, state_value, selected, is_teampreview)

        try:
            if is_teampreview:
                return MDBO.from_int(selected, type=MDBO.TEAMPREVIEW).message

            action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
            mdbo = MDBO.from_int(selected, type=action_type)
            return mdbo.to_double_battle_order(battle)
        except Exception:
            return DefaultBattleOrder()


def _build_model_player(args, server_config: ServerConfiguration) -> VerboseModelPlayer:
    password: Optional[str] = args.password if args.password else None
    return VerboseModelPlayer(
        model_path=args.model,
        device=args.device,
        battle_format=args.battle_format,
        probabilistic=args.probabilistic,
        top_k=args.top_k,
        print_summary=args.print_summary,
        account_configuration=AccountConfiguration(args.username, password),
        server_configuration=server_config,
        max_concurrent_battles=args.max_concurrent_battles,
        start_timer_on_battle_start=args.mode == "ladder",
    )


async def run_play(args) -> None:
    server_config = ServerConfiguration(f"ws://{args.server}/showdown/websocket", "")

    model_player = _build_model_player(args, server_config)

    if args.mode == "challenge":
        print(
            f"Awaiting up to {args.num_battles} challenges as '{args.username}' on {args.server} "
            f"({args.battle_format})"
        )
        await model_player.accept_challenges(
            opponent=args.challenge_opponent,
            n_challenges=args.num_battles,
        )
    elif args.mode == "ladder":
        print(
            f"Playing {args.num_battles} ladder battles as '{args.username}' on {args.server} "
            f"({args.battle_format})"
        )
        await model_player.ladder(n_games=args.num_battles)

    print("\n" + "=" * 60)
    print("FINAL RECORD")
    print("=" * 60)
    print(
        f"Wins={model_player.n_won_battles} "
        f"Losses={model_player.n_lost_battles} "
        f"Ties={model_player.n_tied_battles} "
        f"Finished={model_player.n_finished_battles}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a model in challenge, ladder, or local-vs-bot mode with detailed debug output"
    )
    parser.add_argument("model", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="challenge",
        choices=["challenge", "ladder", "vs-bot"],
        help="challenge: accept incoming challenges; ladder: play ladder games; vs-bot: play a local scripted opponent",
    )
    parser.add_argument("--username", type=str, default="VerboseModel", help="Bot account username")
    parser.add_argument("--password", type=str, default="", help="Bot account password")
    parser.add_argument(
        "--challenge-opponent",
        type=str,
        default=None,
        help="Optional opponent username filter when --mode=challenge",
    )
    parser.add_argument(
        "--max-concurrent-battles",
        type=int,
        default=1,
        help="Maximum concurrent battles handled by this bot",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="maxdamage",
        choices=["model", "maxdamage", "maxbasepower", "shp", "random"],
        help="Opponent type when --mode=vs-bot",
    )
    parser.add_argument(
        "--opponent-model",
        type=str,
        default=None,
        help="Path to opponent model checkpoint when --mode=vs-bot and --opponent=model",
    )
    parser.add_argument("--num-battles", type=int, default=1, help="Number of games/challenges")
    parser.add_argument("--battle-format", type=str, default="gen9vgc2023regc")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--server", type=str, default="localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--probabilistic", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--start-port", type=int, default=8000)
    args = parser.parse_args()

    server_processes: Optional[List[Popen[Any]]] = None
    if args.launch_server:
        server_processes = launch_showdown_servers(1, args.start_port)
        args.server = f"localhost:{args.start_port}"

    try:
        asyncio.run(run_play(args))
    finally:
        if server_processes is not None:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
