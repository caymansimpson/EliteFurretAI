# -*- coding: utf-8 -*-
"""This module plays one player against another (need to have a local showdown server running). This
is what I use to examine and print what's happening in a battle to debug; the purpose of this file is
to provide some starter code for others.

It runs a battle, prints the observations, and saves the DoubleBattle object to the Desktop.
"""

import asyncio
from typing import Optional, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference

from elitefurretai.inference.inference_utils import battle_to_str


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
        if battle.player_role == "p1":
            self._speed_inference = SpeedInference(battle, self._inference, verbose=10)
            self._item_inference = ItemInference(battle, self._inference, verbose=10)

            self._speed_inference.update(battle)  # pyright: ignore
            self._item_inference.update(battle)  # pyright: ignore
        return "/team 1234"

    def choose_move(self, battle):
        # Don't deal with battle.force_switch until
        if not any(battle.force_switch) and battle.player_role == "p1":
            self._speed_inference.update(battle)  # pyright: ignore
            self._item_inference.update(battle)  # pyright: ignore
        return self.choose_random_doubles_move(battle)  # pyright: ignore

    # Print the battle upon battle completion, and save the observations in a BattleData object to the Desktop
    def _battle_finished_callback(self, battle: AbstractBattle):
        if self.username == "elitefurretai":
            print(battle_to_str(battle))


async def main():
    pokepaste = """
Groudon @ Clear Amulet
Ability: Drought
Level: 50
Tera Type: Fire
EVs: 252 HP / 76 Atk / 36 SpD / 140 Spe
Adamant Nature
- Precipice Blades
- Thunder Punch
- Heat Crash
- Protect

Tornadus @ Focus Sash
Ability: Prankster
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Air Slash
- Tailwind
- Heat Wave
- Protect

Flutter Mane @ Choice Specs
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 148 HP / 252 SpA / 108 Spe
Modest Nature
IVs: 18 Atk
- Dazzling Gleam
- Moonblast
- Shadow Ball
- Perish Song

Chi-Yu @ Choice Scarf
Ability: Beads of Ruin
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Heat Wave
- Dark Pulse
- Snarl
- Overheat

Grimmsnarl @ Light Clay
Ability: Prankster
Level: 50
Tera Type: Ghost
EVs: 252 HP / 4 Atk / 180 Def / 36 SpD / 36 Spe
Careful Nature
IVs: 18 SpA
- Foul Play
- Thunder Wave
- Reflect
- Light Screen

Raging Bolt @ Assault Vest
Ability: Protosynthesis
Level: 50
Tera Type: Fire
EVs: 60 HP / 4 Def / 196 SpA / 180 SpD / 68 Spe
Modest Nature
IVs: 20 Atk
- Thunderbolt
- Thunderclap
- Dragon Pulse
- Weather Ball
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
