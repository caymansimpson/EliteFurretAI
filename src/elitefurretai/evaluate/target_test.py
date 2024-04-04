# -*- coding: utf-8 -*-
"""This module tests one player against another
"""
import asyncio
import random
from typing import Dict, List, Optional, Tuple

from poke_env.environment import Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
)
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate


class TeamRepository:
    teams = {
        "common": """
        Tapu Fini @ Wiki Berry
        Level: 50
        Ability: Misty Surge
        EVs: 236 HP / 0 Atk / 4 Def / 204 SpA / 12 SpD / 52 Spe
        Modest Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Muddy Water
        -  Moonblast
        -  Protect
        -  Calm Mind

        Regieleki @ Light Clay
        Level: 50
        EVs: 0 HP / 0 Atk / 0 Def / 252 SpA / 4 SpD / 252 Spe
        Timid Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Electroweb
        -  Thunderbolt
        -  Reflect
        -  Light Screen

        Incineroar @ Figy Berry
        Level: 50
        EVs: 244 HP / 20 Atk / 84 Def / 0 SpA / 148 SpD / 12 Spe
        Careful Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Fake Out
        -  Flare Blitz
        -  Parting Shot
        -  Snarl

        Landorus-Therian @ Assault Vest
        Level: 50
        Ability: Intimidate
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Earthquake
        -  Rock Slide
        -  Fly
        -  Superpower

        Metagross @ Weakness Policy
        Level: 50
        Ability: Clear Body
        EVs: 252 HP / 252 Atk / 0 Def / 0 SpA / 4 SpD / 0 Spe
        Adamant Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Protect
        -  Iron Head
        -  Stomping Tantrum
        -  Ice Punch

        Urshifu @ Focus Sash
        Level: 50
        EVs: 0 HP / 252 Atk / 0 Def / 0 SpA / 4 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Wicked Blow
        -  Close Combat
        -  Sucker Punch
        -  Protect
        """,
        "swampert": """
        Zapdos-Galar @ Choice Scarf
        Level: 50
        Ability: Defiant
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Thunderous Kick
        -  Brave Bird
        -  Coaching
        -  Stomping Tantrum

        Dragapult @ Life Orb
        Level: 50
        Ability: Clear Body
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Phantom Force
        -  Dragon Darts
        -  Fly
        -  Protect

        Swampert @ Assault Vest
        Level: 50
        Ability: Torrent
        EVs: 252 HP / 228 Atk / 4 Def / 0 SpA / 4 SpD / 20 Spe
        Adamant Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Liquidation
        -  Ice Punch
        -  High Horsepower
        -  Hammer Arm

        Clefairy @ Eviolite
        Level: 50
        Ability: Friend Guard
        EVs: 252 HP / 0 Atk / 252 Def / 0 SpA / 4 SpD / 0 Spe
        Bold Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Follow Me
        -  Helping Hand
        -  Sing
        -  Protect

        Celesteela @ Weakness Policy
        Level: 50
        Ability: Beast Boost
        EVs: 92 HP / 0 Atk / 4 Def / 252 SpA / 4 SpD / 156 Spe
        Modest Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Flash Cannon
        -  Air Slash
        -  Meteor Beam
        -  Protect

        Rotom-Heat @ Sitrus Berry
        Level: 50
        Ability: Levitate
        EVs: 252 HP / 0 Atk / 44 Def / 36 SpA / 20 SpD / 156 Spe
        Modest Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Overheat
        -  Thunderbolt
        -  Nasty Plot
        -  Protect
        """,
        "regirock": """
        Raichu @ Focus Sash
        Level: 50
        Ability: Lightning Rod
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Nuzzle
        -  Reflect
        -  Fake Out
        -  Eerie Impulse

        Tapu Fini @ Sitrus Berry
        Level: 50
        Ability: Misty Surge
        EVs: 252 HP / 0 Atk / 68 Def / 116 SpA / 20 SpD / 52 Spe
        Modest Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Moonblast
        -  Muddy Water
        -  Calm Mind
        -  Protect

        Kartana @ Life Orb
        Level: 50
        Ability: Beast Boost
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Leaf Blade
        -  Sacred Sword
        -  Smart Strike
        -  Protect

        Rotom-Heat @ Safety Goggles
        Level: 50
        Ability: Levitate
        EVs: 252 HP / 0 Atk / 44 Def / 36 SpA / 20 SpD / 156 Spe
        Modest Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Overheat
        -  Thunderbolt
        -  Nasty Plot
        -  Protect

        Regirock @ Leftovers
        Level: 50
        Ability: Clear Body
        EVs: 252 HP / 156 Atk / 0 Def / 0 SpA / 100 SpD / 0 Spe
        Adamant Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Rock Slide
        -  Body Press
        -  Earthquake
        -  Curse

        Moltres-Galar @ Weakness Policy
        Level: 50
        Ability: Berserk
        EVs: 204 HP / 0 Atk / 100 Def / 76 SpA / 28 SpD / 100 Spe
        Modest Nature
        IVs: 31 HP / 0 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Fiery Wrath
        -  Air Slash
        -  Nasty Plot
        -  Protect
        """,
        "garchomp": """
        Garchomp @ Weakness Policy
        Ability: Rough Skin
        Level: 50
        EVs: 124 HP / 132 Atk / 4 Def / 4 SpD / 244 Spe
        Jolly Nature
        - Earthquake
        - Rock Slide
        - Swords Dance
        - Protect

        Regieleki @ Light Clay
        Ability: Transistor
        Level: 50
        EVs: 92 HP / 180 Def / 36 SpA / 4 SpD / 196 Spe
        Timid Nature
        IVs: 0 Atk
        - Electroweb
        - Volt Switch
        - Light Screen
        - Reflect

        Moltres @ Life Orb
        Ability: Pressure
        Level: 50
        EVs: 4 HP / 252 SpA / 252 Spe
        Timid Nature
        IVs: 0 Atk
        - Heat Wave
        - Air Slash
        - Scorching Sands
        - Protect

        Ferrothorn @ Leftovers
        Ability: Iron Barbs
        Level: 50
        EVs: 252 HP / 204 Atk / 52 Def
        Brave Nature
        IVs: 0 Spe
        - Power Whip
        - Iron Head
        - Knock Off
        - Protect

        Porygon2 @ Eviolite
        Ability: Download
        Level: 50
        EVs: 244 HP / 76 Def / 4 SpA / 140 SpD / 44 Spe
        Bold Nature
        IVs: 0 Atk
        - Shadow Ball
        - Ice Beam
        - Trick Room
        - Recover

        Tapu Fini @ Sitrus Berry
        Ability: Misty Surge
        Level: 50
        EVs: 252 HP / 68 Def / 116 SpA / 20 SpD / 52 Spe
        Modest Nature
        IVs: 0 Atk
        - Moonblast
        - Muddy Water
        - Calm Mind
        - Protect
        """,
        "mamoswine": """
        Thundurus (M) @ Life Orb
        Ability: Defiant
        Level: 50
        EVs: 4 HP / 252 Atk / 252 Spe
        Jolly Nature
        - Wild Charge
        - Fly
        - Superpower
        - Protect

        Mamoswine @ Focus Sash
        Ability: Oblivious
        Level: 50
        EVs: 4 HP / 252 Atk / 252 Spe
        Jolly Nature
        - Icicle Crash
        - Earthquake
        - Ice Shard
        - Protect

        Nihilego @ Power Herb
        Ability: Beast Boost
        Level: 50
        EVs: 4 HP / 252 SpA / 252 Spe
        Timid Nature
        IVs: 0 Atk
        - Meteor Beam
        - Sludge Bomb
        - Power Gem
        - Protect

        Chandelure @ Sitrus Berry
        Ability: Flash Fire
        Level: 50
        EVs: 252 HP / 4 Def / 100 SpA / 4 SpD / 148 Spe
        Modest Nature
        IVs: 0 Atk
        - Heat Wave
        - Shadow Ball
        - Trick Room
        - Imprison

        Tapu Fini @ Wiki Berry
        Ability: Misty Surge
        Level: 50
        EVs: 252 HP / 116 Def / 12 SpA / 76 SpD / 52 Spe
        Calm Nature
        IVs: 0 Atk
        - Muddy Water
        - Moonblast
        - Calm Mind
        - Protect

        Kartana @ Assault Vest
        Ability: Beast Boost
        Level: 50
        EVs: 84 HP / 52 Atk / 4 Def / 116 SpD / 252 Spe
        Jolly Nature
        - Leaf Blade
        - Smart Strike
        - Sacred Sword
        - Aerial Ace
        """,
        "spectrier": """
        Spectrier @ Grassy Seed
        Ability: Grim Neigh
        Level: 50
        EVs: 140 HP / 92 Def / 36 SpA / 4 SpD / 236 Spe
        Modest Nature
        IVs: 0 Atk
        - Shadow Ball
        - Mud Shot
        - Nasty Plot
        - Protect

        Rillaboom-Gmax @ Rose Incense
        Ability: Grassy Surge
        Level: 50
        EVs: 252 HP / 116 Atk / 4 Def / 132 SpD / 4 Spe
        Adamant Nature
        - Wood Hammer
        - Grassy Glide
        - Fake Out
        - Protect

        Incineroar @ Sitrus Berry
        Ability: Intimidate
        Level: 50
        EVs: 244 HP / 4 Atk / 156 Def / 4 SpA / 100 SpD
        Relaxed Nature
        IVs: 0 Spe
        - Flare Blitz
        - Burning Jealousy
        - Parting Shot
        - Fake Out

        Milotic @ Expert Belt
        Ability: Competitive
        Level: 50
        EVs: 132 HP / 12 Def / 204 SpA / 4 SpD / 156 Spe
        Modest Nature
        IVs: 0 Atk
        - Muddy Water
        - Ice Beam
        - Mud Shot
        - Protect

        Togedemaru @ Focus Sash
        Ability: Lightning Rod
        Level: 50
        EVs: 4 HP / 252 Atk / 252 Spe
        Jolly Nature
        - Fake Out
        - Zing Zap
        - Nuzzle
        - Spiky Shield

        Togekiss (M) @ Scope Lens
        Ability: Super Luck
        Level: 50
        EVs: 236 HP / 100 Def / 12 SpA / 4 SpD / 156 Spe
        Modest Nature
        IVs: 0 Atk
        - Air Slash
        - Dazzling Gleam
        - Follow Me
        - Protect
        """,
        "nochoice": """
        Magikarp
        Ability: Swift Swim
        EVs: 8 HP
        IVs: 0 Atk
        - Splash

        Eevee @ Choice Specs
        Ability: Run Away
        EVs: 20 HP
        IVs: 0 Atk
        - Detect

        Machop @ Choice Band
        Ability: Guts
        EVs: 4 Atk
        IVs: 0 Atk
        - Encore

        Blissey (F) @ Choice Scarf
        Ability: Natural Cure
        EVs: 8 SpA
        IVs: 0 Atk
        - Aromatherapy
        """,
        "doubleturn": """
        Magikarp
        Ability: Swift Swim
        EVs: 8 HP
        IVs: 0 Atk
        - Splash

        Dragapult @ Sitrus Berry
        Ability: Clear Body
        EVs: 20 HP
        IVs: 0 Atk
        -  Phantom Force
        -  Dragon Darts
        -  Fly
        -  Protect


        Landorus-Therian @ Assault Vest
        Level: 50
        Ability: Intimidate
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Fly
        -  Superpower

        Swampert @ Life Orb
        Level: 50
        Ability: Torrent
        EVs: 252 HP / 228 Atk / 4 Def / 0 SpA / 4 SpD / 20 Spe
        Adamant Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        -  Dive
        -  Ice Punch
        -  High Horsepower
        -  Hammer Arm
        """,
        "switch": """
        Landorus-Therian (M)
        Ability: Intimidate
        EVs: 48 Atk
        - U-turn

        Regieleki
        Ability: Transistor
        EVs: 20 Def
        IVs: 0 Atk
        - Volt Switch

        Xatu
        Ability: Synchronize
        EVs: 40 Atk
        - U-turn
        - Teleport

        Vibrava
        Ability: Levitate
        Level: 50
        EVs: 20 Def
        - U-turn
        - Dragon Tail

        Dragapult
        Ability: Clear Body
        Level: 50
        EVs: 12 HP
        - Baton Pass
        - U-turn

        Incineroar
        Ability: Intimidate
        Level: 50
        EVs: 12 HP
        - U-turn
        - Parting Shot
        """,
        "edgecase": """
        Indeedee (M)
        Ability: Psychic Surge
        Level: 50
        EVs: 112 Atk
        IVs: 0 Atk
        - Trick Room
        - Substitute
        - Ally Switch
        - Magic Room

        Pelipper @ Aguav Berry
        Ability: Drizzle
        Level: 50
        EVs: 100 Atk
        - Tailwind
        - U-turn
        - Hail
        - Whirlpool

        Dusclops
        Ability: Pressure
        Level: 50
        EVs: 176 Atk
        IVs: 0 Atk
        - Disable
        - Ally Switch
        - Trick
        - Mean Look

        Tapu Lele
        Ability: Psychic Surge
        Level: 50
        EVs: 152 Atk
        IVs: 0 Atk
        - Ally Switch
        - Aromatic Mist
        - Magic Room
        - Wonder Room
        """,
        "speed": """
        Pelipper
        Ability: Keen Eye
        Level: 50
        EVs: 16 Atk / 252 Spe
        IVs: 0 Atk
        - Tailwind
        - Scald

        Ludicolo
        Ability: Swift Swim
        Level: 50
        EVs: 20 HP
        IVs: 0 Atk
        - Rain Dance

        Rhyperior
        Ability: Reckless
        Level: 50
        EVs: 20 HP
        - Bulldoze
        - Rock Polish

        Raichu
        Ability: Static
        Level: 50
        EVs: 28 HP
        - Nuzzle
        - Thunder Wave
        - Thunder
        """,
        "pledge": """
        Incineroar
        Ability: Blaze
        Level: 50
        EVs: 44 HP
        IVs: 0 Atk
        - Fire Pledge

        Rillaboom
        Ability: Overgrow
        Level: 50
        EVs: 20 HP
        IVs: 0 Atk
        - Grass Pledge

        Primarina
        Ability: Torrent
        Level: 50
        EVs: 20 HP
        IVs: 0 Atk
        - Water Pledge

        Swampert
        Ability: Torrent
        Level: 50
        EVs: 4 HP / 252 Atk / 0 Def / 0 SpA / 0 SpD / 252 Spe
        Jolly Nature
        IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
        - Water Pledge

        """,
        "sample": """
        Incineroar
        Ability: Blaze
        Level: 50
        EVs: 44 HP
        - Acrobatics
        - Blaze Kick
        - Body Slam
        - Brick Break

        Rillaboom
        Ability: Overgrow
        Level: 50
        EVs: 20 HP
        - Acrobatics
        - Body Press
        - Body Slam
        - Brick Break

        Primarina
        Ability: Torrent
        Level: 50
        EVs: 20 HP
        - Acrobatics
        - Aqua Jet
        - Energy Ball
        - Facade

        Swampert
        Ability: Torrent
        Level: 50
        EVs: 36 HP
        - Ancient Power
        - Body Press
        - Body Slam
        - Brick Break
     """,
    }


class TestPlayer(RandomPlayer):
    def choose_move(self, battle):
        active_orders: List[List[BattleOrder]] = [[], []]
        for (
            orders,
            mon,
            switches,
            moves,
            can_mega,
            can_z_move,
            can_dynamax,
            can_tera,
        ) in zip(
            active_orders,
            battle.active_pokemon,
            battle.available_switches,
            battle.available_moves,
            battle.can_mega_evolve,
            battle.can_z_move,
            battle.can_dynamax,
            battle.can_tera,
        ):
            if mon:
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

                if can_mega:
                    orders.extend(
                        [
                            BattleOrder(move, move_target=target, mega=True)
                            for move in moves
                            for target in targets[move]
                        ]
                    )
                if can_z_move:
                    available_z_moves = set(mon.available_z_moves)
                    orders.extend(
                        [
                            BattleOrder(move, move_target=target, z_move=True)
                            for move in moves
                            for target in targets[move]
                            if move in available_z_moves
                        ]
                    )

                if can_dynamax:
                    orders.extend(
                        [
                            BattleOrder(move, move_target=target, dynamax=True)
                            for move in moves
                            for target in targets[move]
                        ]
                    )

                if can_tera:
                    orders.extend(
                        [
                            BattleOrder(move, move_target=target, terastallize=True)
                            for move in moves
                            for target in targets[move]
                        ]
                    )

                if sum(battle.force_switch) == 1:
                    if orders:
                        return orders[int(random.random() * len(orders))]
                    return self.choose_default_move()

        orders = DoubleBattleOrder.join_orders(*active_orders)
        valid_orders = [order for order in orders if order.is_valid(battle)]
        if len(valid_orders) == 0:
            return self.choose_default_move()
        o = random.choice(valid_orders)
        return o


async def main():

    players = [
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["mamoswine"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["mamoswine"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["regirock"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["regirock"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["switch"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["switch"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["edgecase"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["edgecase"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["garchomp"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["garchomp"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["swampert"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["swampert"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["spectrier"],
            ping_timeout=60,
        ),
        TestPlayer(
            max_concurrent_battles=10,
            battle_format="gen8vgc2021",
            team=TeamRepository.teams["spectrier"],
            ping_timeout=60,
        ),
    ]

    # Each player plays n times against eac other
    n = 10

    # Pit players against each other
    print(
        "About to start " + str(n * sum(i for i in range(0, len(players)))) + " battles..."
    )
    cross_evaluation = await cross_evaluate(players, n_challenges=n)

    # gen9vgc2024regg gen9doublesou gen9randomdoublesbattle
    # p1 = TestPlayer(battle_format='gen9randomdoublesbattle')
    # p2 = TestPlayer(battle_format='gen9randomdoublesbattle')

    await p1.battle_against(p2, 100)


if __name__ == "__main__":
    asyncio.run(main())
