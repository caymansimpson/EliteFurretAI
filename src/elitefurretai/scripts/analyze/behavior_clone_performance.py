import argparse
import asyncio

from poke_env.player import MaxBasePowerPlayer, RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from elitefurretai.agents.behavior_clone_player import BehaviorClonePlayer


def main():
    pokepaste = """
Chien-Pao @ Focus Sash
Ability: Sword of Ruin
Level: 60
Tera Type: Fire
EVs: 6 HP / 252 Atk / 252 Spe
Jolly Nature
- Ice Spinner
- Sacred Sword
- Sucker Punch
- Protect

Azumarill (F) @ Assault Vest
Ability: Huge Power
Level: 55
Tera Type: Water
EVs: 236 HP / 236 Atk / 4 Def / 12 SpD / 20 Spe
Adamant Nature
- Aqua Jet
- Liquidation
- Ice Spinner
- Play Rough

Amoonguss (M) @ Wiki Berry
Ability: Regenerator
Level: 55
Tera Type: Water
EVs: 236 HP / 158 Def / 116 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Spore
- Rage Powder
- Pollen Puff
- Clear Smog

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 60
Tera Type: Fairy
EVs: 236 HP / 252 Def / 4 SpA / 12 SpD / 4 Spe
Modest Nature
IVs: 0 Atk
- Shadow Ball
- Dazzling Gleam
- Calm Mind
- Protect

Arcanine (M) @ Sitrus Berry
Ability: Intimidate
Level: 55
Tera Type: Grass
EVs: 244 HP / 52 Atk / 100 Def / 76 SpD / 36 Spe
Careful Nature
- Flare Blitz
- Snarl
- Will-O-Wisp
- Protect

Dragonite (F) @ Choice Band
Ability: Multiscale
Level: 75
Tera Type: Flying
EVs: 236 HP / 252 Atk / 20 Spe
Adamant Nature
- Extreme Speed
- Tera Blast
- Stomping Tantrum
- Dragon Claw
    """

    parser = argparse.ArgumentParser(
        description="Evaluate MrMimePlayer (two_headed_transformer) against poke-env baselines."
    )
    parser.add_argument(
        "teampreview_model_path", type=str, help="Path to the two_headed_transformer model file"
    )
    parser.add_argument(
        "action_model_path", type=str, help="Path to the action model file"
    )
    parser.add_argument(
        "win_model_path", type=str, help="Path to the win model file"
    )
    parser.add_argument(
        "--n_battles", type=int, default=100, help="Number of battles per baseline"
    )
    args = parser.parse_args()

    # Load your model into BehaviorClonePlayer
    mr_mime = BehaviorClonePlayer(
        account_configuration=AccountConfiguration("elitefurretai", password=""),
        teampreview_model_filepath=args.teampreview_model_filepath,
        action_model_filepath=args.action_model_filepath,
        win_model_filepath=args.win_model_filepath,
        teampreview_config={}, # TODO: implement
        action_config={},
        win_config={},
        battle_format="gen9vgc2025regi",
        server_configuration=LocalhostServerConfiguration,
        team=pokepaste,
    )

    # Define baselines
    baselines = {
        "RandomPlayer": RandomPlayer(battle_format="gen9vgc2025regi", team=pokepaste),
        "MaxBasePowerPlayer": MaxBasePowerPlayer(
            battle_format="gen9vgc2025regi", team=pokepaste
        ),
    }

    async def run_battles():
        for name, opponent in baselines.items():
            print(f"Evaluating against {name} for {args.n_battles} battles...")
            await mr_mime.battle_against(opponent, n_battles=args.n_battles)

            wins = mr_mime.n_won_battles
            win_rate = wins / args.n_battles
            print(
                f"Results against {name}: {wins}/{args.n_battles} wins ({win_rate * 100:.2f}%)\n"
            )
            mr_mime.reset_battles()

    asyncio.run(run_battles())


if __name__ == "__main__":
    main()
