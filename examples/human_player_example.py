"""
Example script demonstrating how to use the HumanPlayer for interactive battles.

This allows you to play battles manually via CLI against random players or other agents.
"""
import asyncio

from poke_env.player import RandomPlayer
from elitefurretai.agents import HumanPlayer


async def main():
    """Run an interactive battle with human vs random player."""

    # Create players
    human = HumanPlayer(battle_format="gen9vgc2023regulationc")
    opponent = RandomPlayer(battle_format="gen9vgc2023regulationc")

    # Run a single battle
    print("Starting battle...")
    print("=" * 80)
    print("Instructions:")
    print("  - For moves: Enter move number (1-4)")
    print("  - For switches: Enter 's' + switch number (e.g., 's1', 's2')")
    print("  - For dynamax: Add 'd' to your move (e.g., '1d')")
    print("  - For terastallize: Add 't' to your move (e.g., '1t')")
    print("  - For doubles: Specify actions for both slots (e.g., '1a 2b')")
    print("  - Type 'quit' to forfeit at any time")
    print("=" * 80)

    await human.battle_against(opponent, n_battles=1)

    # Print results
    print("\n" + "=" * 80)
    print("BATTLE RESULTS")
    print("=" * 80)
    print(f"Human wins: {human.n_won_battles}")
    print(f"Opponent wins: {opponent.n_won_battles}")
    print(f"Total battles: {human.n_finished_battles}")


async def battle_against_custom_team():
    """Example of playing against a specific team."""
    from poke_env.player import MaxBasePowerPlayer

    # You can specify a custom team from a file
    human = HumanPlayer(
        battle_format="gen9vgc2023regulationc",
        team="data/teams/sample_team.txt"  # Optional: specify your team
    )

    # Battle against a stronger opponent
    opponent = MaxBasePowerPlayer(battle_format="gen9vgc2023regulationc")

    print("Battling against MaxBasePowerPlayer (always picks highest base power move)...")
    await human.battle_against(opponent, n_battles=1)


async def play_multiple_battles():
    """Play a series of battles."""
    human = HumanPlayer(battle_format="gen9vgc2023regulationc")
    opponent = RandomPlayer(battle_format="gen9vgc2023regulationc")

    n_battles = 3
    print(f"Starting {n_battles} battles...")

    await human.battle_against(opponent, n_battles=n_battles)

    # Print final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Win rate: {human.n_won_battles / human.n_finished_battles * 100:.1f}%")
    print(f"Wins: {human.n_won_battles}")
    print(f"Losses: {opponent.n_won_battles}")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())

    # Uncomment to try other examples:
    # asyncio.run(battle_against_custom_team())
    # asyncio.run(play_multiple_battles())
