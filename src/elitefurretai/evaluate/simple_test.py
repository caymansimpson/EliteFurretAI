# -*- coding: utf-8 -*-
"""This module tests one player against another
"""

import asyncio
from typing import Dict, List, Optional, Tuple

from poke_env.data import to_id_str
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration


async def main():
    furret = """
Furret @ Black Sludge
Ability: Frisk
Level: 50
Tera Type: Ghost
EVs: 252 Atk / 1 Def / 252 Spe
Timid Nature
IVs: 0 Atk
- Quick Attack
- Helping Hand

Grimmsnarl @ Leftovers
Ability: Prankster
Level: 50
Tera Type: Fire
EVs: 252 Atk / 1 SpD / 252 Spe
Jolly Nature
- Scary Face
- Thunder Wave
- Draining Kiss

Sandslash @ Choice Scarf
Ability: Sand Rush
Level: 50
Tera Type: Electric
EVs: 252 HP / 1 Def / 252 SpD
Calm Nature
- Bulldoze

Tyranitar @ Flame Orb
Ability: Sand Stream
Level: 50
Tera Type: Electric
EVs: 252 HP / 1 Def / 252 SpD
Calm Nature
- Sandstorm
- Bite

Whimsicott @ Quick Claw
Ability: Prankster
Level: 50
Tera Type: Electric
EVs: 252 HP / 1 Def / 252 SpD
Calm Nature
- Tailwind
- Moonblast
    """
    boring = """
Amoonguss @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 252 Atk / 1 SpD / 252 Spe
Jolly Nature
- Pollen Puff
- Grassy Terrain

Dusclops @ Leftovers
Ability: Frisk
Level: 50
Tera Type: Ghost
EVs: 252 HP / 1 SpD / 252 Spe
Jolly Nature
- Trick Room
- Hex
- Will-O-Wisp

Drifblim @ Iron Ball
Ability: Unburden
Level: 50
Tera Type: Bug
EVs: 244 HP / 252 SpA / 1 SpD
Modest Nature
IVs: 20 Atk
- Tailwind
- Aerial Ace

Bellossom @ Quick Claw
Ability: Chlorophyll
Level: 50
Tera Type: Fire
EVs: 252 Atk / 1 SpD / 252 Spe
Jolly Nature
- Sunny Day
- Trailblaze
- Grassy Terrain
    """
    # gen9doublesou, gen9vgc2024regg
    p1 = RandomPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regg",
        team=furret,
    )
    p2 = RandomPlayer(battle_format="gen9vgc2024regg", team=boring)
    print(p1.username)
    await p1.battle_against(p2)


if __name__ == "__main__":
    asyncio.run(main())
