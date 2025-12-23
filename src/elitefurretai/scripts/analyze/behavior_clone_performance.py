import argparse
import asyncio

from poke_env.player import MaxBasePowerPlayer, RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from elitefurretai.agents.behavior_clone_player import BCPlayer


def main():
    pokepaste = """
Flutter Mane @ Choice Specs  
Ability: Protosynthesis  
Level: 50  
Tera Type: Fairy  
EVs: 100 HP / 36 Def / 196 SpA / 4 SpD / 172 Spe  
Timid Nature  
IVs: 0 Atk  
- Moonblast  
- Shadow Ball  
- Thunderbolt  
- Dazzling Gleam  

Dragonite @ Lum Berry  
Ability: Multiscale  
Level: 50  
Tera Type: Flying  
EVs: 20 HP / 252 Atk / 4 Def / 4 SpD / 228 Spe  
Jolly Nature  
- Tera Blast  
- Low Kick  
- Extreme Speed  
- Dragon Dance  

Iron Bundle @ Booster Energy  
Ability: Quark Drive  
Level: 50  
Tera Type: Ice  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Freeze-Dry  
- Hydro Pump  
- Icy Wind  
- Protect  

Chi-Yu @ Focus Sash  
Ability: Beads of Ruin  
Level: 50  
Tera Type: Ghost  
EVs: 4 Def / 252 SpA / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Heat Wave  
- Dark Pulse  
- Nasty Plot  
- Protect  

Ting-Lu @ Assault Vest  
Ability: Vessel of Ruin  
Level: 50  
Tera Type: Poison  
EVs: 4 HP / 236 Atk / 4 Def / 84 SpD / 180 Spe  
Adamant Nature  
- Earthquake  
- Heavy Slam  
- Ruination  
- Stomping Tantrum  

Gyarados @ Sitrus Berry  
Ability: Intimidate  
Level: 50  
Tera Type: Steel  
EVs: 244 HP / 36 Atk / 4 Def / 92 SpD / 132 Spe  
Adamant Nature  
- Waterfall  
- Thunder Wave  
- Taunt  
- Protect  
    """

    parser = argparse.ArgumentParser(
        description="Evaluate BCPlayer (three separate models) against poke-env baselines."
    )
    parser.add_argument(
        "teampreview_model_path", type=str, help="Path to the teampreview model file (.pt with embedded config)"
    )
    parser.add_argument(
        "action_model_path", type=str, help="Path to the action model file (.pt with embedded config)"
    )
    parser.add_argument(
        "win_model_path", type=str, help="Path to the win model file (.pt with embedded config)"
    )
    parser.add_argument(
        "--n_battles", type=int, default=100, help="Number of battles per baseline"
    )
    args = parser.parse_args()

    # Load your model into BehaviorClonePlayer
    # Configs are now embedded in the .pt files
    mr_mime = BCPlayer(
        account_configuration=AccountConfiguration("elitefurretai", password=""),
        teampreview_model_filepath=args.teampreview_model_path,
        action_model_filepath=args.action_model_path,
        win_model_filepath=args.win_model_path,
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
