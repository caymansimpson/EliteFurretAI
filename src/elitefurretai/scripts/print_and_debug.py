# -*- coding: utf-8 -*-
"""This module plays one player against another (need to have a local showdown server running). This
is what I use to examine and print what's happening in a battle to debug; the purpose of this file is
to provide some starter code for others.

It runs a battle, prints the observations, and saves the DoubleBattle object to the Desktop.
"""

import asyncio
import random
from typing import List, Optional, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference


class CustomPlayer(RandomPlayer):

    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        avatar: Optional[str] = None,
        battle_format: str = "gen9randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = False,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = None,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
    ):
        super().__init__(
            account_configuration=account_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            accept_open_team_sheet=accept_open_team_sheet,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team,
        )
        self._inference = None
        self._speed_inference = None
        self._item_inference = None

    def teampreview(self, battle):
        self._inference = BattleInference(battle)

        # Speed and Item Inferences will fill inferences
        # if battle.player_role == "p1":
        #     self._speed_inference = SpeedInference(battle, self._inference, verbose=0)
        #     self._item_inference = ItemInference(battle, self._inference, verbose=0)

        #     self._speed_inference.update(battle)  # pyright: ignore
        #     self._item_inference.update(battle)  # pyright: ignore
        return "/team 1234"

    def choose_random_doubles_move(self, battle: DoubleBattle) -> BattleOrder:
        active_orders: List[List[BattleOrder]] = [[], []]

        if any(battle.force_switch):
            first_order = None
            second_order = None

            if battle.force_switch[0] and battle.available_switches[0]:
                first_switch_in = random.choice(battle.available_switches[0])
                first_order = BattleOrder(first_switch_in)
            else:
                first_switch_in = None

            if battle.force_switch[1] and battle.available_switches[1]:
                available_switches = [
                    s for s in battle.available_switches[1] if s != first_switch_in
                ]

                if available_switches:
                    second_switch_in = random.choice(available_switches)
                    second_order = BattleOrder(second_switch_in)

            if first_order and second_order:
                return DoubleBattleOrder(first_order, second_order)
            return DoubleBattleOrder(first_order or second_order, None)

        for orders, mon, switches, moves, can_tera in zip(
            active_orders,
            battle.active_pokemon,
            battle.available_switches,
            battle.available_moves,
            battle.can_tera,
        ):
            if not mon:
                continue

            targets = {
                move: battle.get_possible_showdown_targets(move, mon) for move in moves
            }
            orders.extend(
                [
                    BattleOrder(move, move_target=target)
                    for move in moves
                    for target in targets[move]
                ]
            )
            orders.extend([BattleOrder(switch) for switch in switches])

            if can_tera:
                orders.extend(
                    [
                        BattleOrder(move, move_target=target, terastallize=True)
                        for move in moves
                        for target in targets[move]
                    ]
                )

        orders = DoubleBattleOrder.join_orders(*active_orders)

        if orders:
            return orders[int(random.random() * len(orders))]
        else:
            return DefaultBattleOrder()

    def choose_move(self, battle):
        # Don't deal with battle.force_switch until
        # if not any(battle.force_switch) and battle.player_role == "p1":
        #     self._speed_inference.update(battle)  # pyright: ignore
        #     self._item_inference.update(battle)  # pyright: ignore
        return self.choose_random_doubles_move(battle)  # pyright: ignore

    # Print the battle upon battle completion, and save the observations in a BattleData object to the Desktop
    def _battle_finished_callback(self, battle: AbstractBattle):
        if self.username == "elitefurretai":
            # print(battle_to_str(battle))
            pass


async def main():
    pokepaste = """
Hatterene
Ability: Magic Bounce  
Tera Type: Psychic  
EVs: 1 HP / 252 SpA  
IVs: 0 Atk  
- Dazzling Gleam  

Furret  
Ability: Run Away  
Tera Type: Normal  
EVs: 1 HP / 252 Atk  
Lonely Nature  
- Baby-Doll Eyes  
- Charm  
- Double-Edge  

Corviknight  
Ability: Mirror Armor  
Level: 50  
Tera Type: Flying  
EVs: 1 HP  
- Fake Tears  
- Metal Sound  
- Screech  
- Brave Bird  

Incineroar  
Ability: Intimidate  
Level: 50  
Tera Type: Fire  
EVs: 1 HP / 252 SpA  
IVs: 0 Atk  
- Fire Blast  
- Will-O-Wisp  
- Scary Face  
        """

    p1 = CustomPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regh",
        team=pokepaste,
    )
    p2 = CustomPlayer(battle_format="gen9vgc2024regh", team=pokepaste)

    # Run the battle
    for i in range(10000):
        random.seed(i)
        print("Starting battle with random seed", i)
        await p1.battle_against(p2)


if __name__ == "__main__":
    asyncio.run(main())
