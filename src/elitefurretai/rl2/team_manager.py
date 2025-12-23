"""
Team pool management for RL training.

Loads Pokemon teams from disk and provides random sampling for diverse training.
"""

import random
from pathlib import Path
from typing import List, Optional


class TeamPool:
    """Manages a pool of Pokemon teams for training diversity."""

    def __init__(self, team_dir: str, format_name: str = "gen9vgc2023regulationc"):
        """
        Initialize team pool from directory.

        Args:
            team_dir: Path to directory containing .txt team files (PokePaste format)
            format_name: Pokemon format (e.g., "gen9vgc2023regulationc")
        """
        self.team_dir = Path(team_dir)
        self.format_name = format_name
        self.teams: List[str] = []
        self._load_teams()

    def _load_teams(self) -> None:
        """Load all team files from directory."""
        if not self.team_dir.exists():
            raise ValueError(f"Team directory not found: {self.team_dir}")

        # Find all .txt files
        team_files = list(self.team_dir.glob("*.txt"))

        if len(team_files) == 0:
            raise ValueError(f"No team files found in {self.team_dir}")

        # Load team contents
        for team_file in team_files:
            with open(team_file, "r") as f:
                team_str = f.read().strip()
                if team_str:  # Only add non-empty teams
                    self.teams.append(team_str)

        print(f"Loaded {len(self.teams)} teams from {self.team_dir}")

    def sample_team(self) -> Optional[str]:
        """
        Sample a random team from the pool.

        Returns:
            Team string in PokePaste format, or None if pool is empty
        """
        if not self.teams:
            return None
        return random.choice(self.teams)

    def get_team(self, index: int) -> str:
        """
        Get a specific team by index.

        Args:
            index: Team index (0 to len-1)

        Returns:
            Team string in PokePaste format
        """
        return self.teams[index]

    def __len__(self) -> int:
        """Return number of teams in pool."""
        return len(self.teams)

    def __repr__(self) -> str:
        return f"TeamPool(format={self.format_name}, num_teams={len(self.teams)})"


def validate_team_format(team_str: str) -> bool:
    """
    Basic validation of team format.

    Args:
        team_str: Team in PokePaste format

    Returns:
        True if team appears valid
    """
    lines = [line.strip() for line in team_str.split("\n") if line.strip()]

    # Should have at least 6 Pokemon (each Pokemon = ~6-8 lines)
    if len(lines) < 30:
        return False

    # Should have Pokemon names (first line of each block, no leading whitespace or dashes)
    pokemon_count = sum(
        1 for line in lines if line and not line.startswith("-") and not line.startswith(" ") and "@" in line
    )

    return 4 <= pokemon_count <= 6  # VGC requires 4-6 Pokemon


# Example usage
if __name__ == "__main__":
    # Create a pool from data/teams/regc
    pool = TeamPool("data/teams/gen9vgc2023regulationc")

    print(f"Pool: {pool}")
    print(f"\nSample team:\n{pool.sample_team()}")

    # Validate all teams
    invalid_teams = []
    for i in range(len(pool)):
        team = pool.get_team(i)
        if not validate_team_format(team):
            invalid_teams.append(i)

    if invalid_teams:
        print(f"\nWarning: {len(invalid_teams)} teams failed validation: {invalid_teams}")
    else:
        print("\nAll teams validated successfully!")
