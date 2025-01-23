# -*- coding: utf-8 -*-
"""This module plays one player against another (need to have a local showdown server running). This
is what I use to examine and print what's happening in a battle to debug; the purpose of this file is
to provide starter code for others.
"""

import asyncio
from typing import Optional, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.utils.inference_utils import battle_to_str


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

    # Place where I can implement basic logic; right now I just print that I'm in choose_move
    def choose_move(self, battle):
        if self.username == "elitefurretai":
            print("in choose_move")

        return self.choose_random_doubles_move(battle)  # pyright: ignore

    # Place where I can implement basic logic; right now I just print that I'm in teampreview
    def teampreview(self, battle):
        if self.username == "elitefurretai":
            print("in teampreview")

        return "/team 1234"

    # Print the battle upon battle completion, and save the observations in a BattleData object to the Desktop
    def _battle_finished_callback(self, battle: AbstractBattle):
        if self.username == "elitefurretai":
            print(battle_to_str(battle))


async def main():
    pokepaste = """
        Delibird @ Safety Goggles
        Ability: Vital Spirit
        Tera Type: Steel
        EVs: 1 HP
        - Fake Out
        - Icy Wind
        - Spikes
        - Substitute

        Raichu @ Light Clay
        Ability: Static
        Tera Type: Ground
        EVs: 1 HP
        - Light Screen
        - Fake Out
        - Reflect
        - Nuzzle

        Tyranitar @ Heavy-Duty Boots
        Ability: Sand Stream
        Tera Type: Psychic
        EVs: 1 HP
        - Bulldoze
        - Dragon Tail
        - Stealth Rock

        Smeargle @ Covert Cloak
        Ability: Moody
        Tera Type: Rock
        EVs: 1 HP
        - U-turn
        - Icy Wind
        - Rage Powder
        - Spikes

        Furret @ Leftovers
        Ability: Frisk
        Tera Type: Rock
        EVs: 1 HP
        - Follow Me
        - Double Edge
        """

    p1 = CustomPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regg",
        team=pokepaste,
    )
    p2 = CustomPlayer(battle_format="gen9vgc2024regg", team=pokepaste)

    # Run the battle
    await p1.battle_against(p2)


if __name__ == "__main__":
    asyncio.run(main())
