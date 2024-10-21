# -*- coding: utf-8 -*-
"""This module plays one player against another (need to have a local showdown server running). This
is what I use to examine and print what's happening in a battle to debug; the purpose of this file is
to provide some starter code for others.

It runs a battle, prints the observations, and saves the DoubleBattle object to the Desktop.
"""

import asyncio
import os.path
import pickle
from typing import Dict, List, Optional, Tuple, Union

from poke_env.data import to_id_str
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.model_utils.data_processor import DataProcessor


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
        self._bi = None

    def choose_move(self, battle):

        if self.username == "elitefurretai":
            print(f"in choose_move on turn #{battle.turn}; force_switch: {battle.force_switch}.")
            # print("Current Observation events:", battle.current_observation.events)
            # print()

        return self.choose_random_doubles_move(battle)

    def teampreview(self, battle):

        if self.username == "elitefurretai":
            print("in teampreview")

        return "/team 1234"

    # Print the battle upon battle completion, and save the observations in a BattleData object to the Desktop
    def _battle_finished_callback(self, battle: AbstractBattle):
        address = os.path.expanduser(
            os.path.join("~/Desktop", battle.format + "-" + battle.battle_tag + ".json")
        )

        if self.username == "elitefurretai":
            print(battle_to_str(battle))

        # with open(address, "w") as f:
        #     f.write(DataProcessor.battle_to_json(battle))


async def main():
    pokepaste = """
Golduck @ Aguav Berry  
Ability: Cloud Nine  
Level: 50  
Tera Type: Water  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Flip Turn  
- Aqua Tail  

Pelipper @ Expert Belt  
Ability: Drizzle  
Level: 50  
Tera Type: Water  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
- U-turn  
- Hurricane  

Basculegion (M) @ Choice Band  
Ability: Adaptability  
Level: 50  
Tera Type: Water  
EVs: 252 Atk / 4 SpA / 252 Spe  
Hasty Nature  
- Aqua Jet  
- Wave Crash  

Furret @ Focus Sash  
Ability: Frisk  
Level: 50  
Tera Type: Normal  
EVs: 244 HP / 12 Atk / 252 Spe  
Jolly Nature  
- Double-Edge  
- Super Fang  
- Protect  
- Quick Attack  


        """

    p1 = CustomPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regg",
        team=pokepaste,
    )
    p2 = CustomPlayer(battle_format="gen9vgc2024regg", team="""
Golduck @ Aguav Berry  
Ability: Cloud Nine  
Level: 50  
Tera Type: Water  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Flip Turn  
- Aqua Tail  

Pelipper @ Expert Belt  
Ability: Drizzle  
Level: 50  
Tera Type: Water  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
- U-turn  
- Hurricane  

Basculegion (M) @ Choice Band  
Ability: Adaptability  
Level: 50  
Tera Type: Water  
EVs: 252 Atk / 4 SpA / 252 Spe  
Hasty Nature  
- Aqua Jet  
- Wave Crash  

Furret @ Focus Sash  
Ability: Frisk  
Level: 50  
Tera Type: Normal  
EVs: 244 HP / 12 Atk / 252 Spe  
Jolly Nature  
- Double-Edge  
- Super Fang  
- Protect  
- Quick Attack  
""")

    # Run the battle
    await p1.battle_against(p2)


if __name__ == "__main__":
    asyncio.run(main())
