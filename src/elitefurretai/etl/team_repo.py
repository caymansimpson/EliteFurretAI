# -*- coding: utf-8 -*-
"""
Team Repository Management for Pokemon VGC Teams

This module provides TeamRepo, a utility for loading, managing, and sampling Pokemon teams
in PokePaste format. Teams are organized by format (e.g., gen9vgc2023regulationc) and can be
loaded from directories or subdirectories.

Expected File Structure:
    data/teams/
    ├── gen9vgc2023regulationc/        # Format directory
    │   ├── team1.txt                  # Individual team files
    │   ├── team2.txt
    │   ├── rental_teams/              # Subdirectories are supported (recursive)
    │   │   ├── rental1.txt
    │   │   └── rental2.txt
    │   └── tournament_teams/
    │       ├── worlds_2023/           # Arbitrarily deep nesting supported
    │       │   ├── finalist1.txt
    │       │   └── finalist2.txt
    │       └── regionals.txt
    ├── gen9vgc2024regf/               # Another format
    │   ├── team_a.txt
    │   └── team_b.txt
    └── gen8vgc2022/
        └── series12.txt

Usage:
    # Load all teams from repository
    repo = TeamRepo(filepath="data/teams")

    # Sample a random team from a format
    team = repo.sample_team("gen9vgc2023regulationc")

    # Save a new team
    repo.save_team(team_string, "gen9vgc2023regulationc", "my_team")

    # Get specific team
    team = repo.get(format="gen9vgc2023regulationc", name="team1")
"""


import os
import os.path
import random
import subprocess
from contextlib import contextmanager
from typing import Dict, KeysView, Optional, List

_DEFAULT_FILEPATH = "data/teams"


class TeamRepo:
    """
    Repository for managing Pokemon teams organized by format.

    This class loads teams from a directory structure where each format has its own folder,
    and teams are stored as .txt files in PokePaste format. Supports recursive directory
    scanning to allow organizing teams into subdirectories (e.g., by tournament, player, etc.).

    Attributes:
        _teams: Nested dict mapping format -> team_name -> team_string
        _verbose: Whether to print loading progress

    Example:
        >>> repo = TeamRepo("data/teams")
        >>> formats = list(repo.formats)
        >>> team = repo.sample_team("gen9vgc2023regulationc")
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        showdown_path: str = "../pokemon-showdown",
        validate: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize TeamRepo by loading teams from the specified directory.

        Args:
            filepath: Root directory containing format folders (default: data/teams)
            showdown_path: Path to pokemon-showdown for validation (default: ../pokemon-showdown)
            validate: Whether to validate each team against Pokemon Showdown (default: False)
                     WARNING: Validation is slow and requires pokemon-showdown installed
            verbose: Print loading progress and validation results (default: False)

        Directory Structure:
            filepath/
            ├── format1/           # Top-level = format name
            │   ├── team1.txt
            │   ├── team2.txt
            │   └── subdir/        # Subdirectories are scanned recursively
            │       └── team3.txt
            └── format2/
                └── team4.txt
        """
        self._teams: Dict[str, Dict[str, str]] = {}
        self._verbose = verbose

        # If we have the default filepath, use the default
        if filepath is None:
            current_file = os.path.dirname(
                os.path.abspath(__file__)
            )  # Gets current directory
            three_up = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            filepath = os.path.join(three_up, _DEFAULT_FILEPATH)

        self.base_dir = filepath

        # Scan top-level directories (each is a format)
        if not os.path.exists(filepath):
            if verbose:
                print(f"Warning: Team directory not found: {filepath}")
            return

        for format_folder in os.listdir(filepath):
            folder_path = os.path.join(filepath, format_folder)

            if os.path.isdir(folder_path):
                if verbose:
                    print(f"Loading format: {format_folder}")
                if format_folder not in self._teams:
                    self._teams[format_folder] = {}

                # Recursively load all .txt files in this format's directory
                self._load_teams_recursive(
                    format_folder,
                    folder_path,
                    showdown_path if validate else None
                )

    def _load_teams_recursive(
        self,
        format_name: str,
        directory: str,
        showdown_path: Optional[str] = None
    ) -> None:
        """
        Recursively load all .txt files from a directory and its subdirectories.

        This allows organizing teams into subdirectories within a format folder.
        For example: gen9vgc2023regulationc/rental_teams/team1.txt

        Args:
            format_name: The format these teams belong to
            directory: Directory to scan recursively
            showdown_path: If provided, validate teams against this Showdown installation
        """
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)

            if os.path.isdir(item_path):
                # Recursively scan subdirectories
                self._load_teams_recursive(format_name, item_path, showdown_path)
            elif item.endswith('.txt'):
                # Load team file
                if self._verbose:
                    print(f"  Loading: {item_path}")

                try:
                    with open(item_path, "r") as f:
                        team_string = f.read()

                    # Use relative path from format directory as team name for uniqueness
                    # e.g., "rental_teams/team1" instead of just "team1"
                    team_name = os.path.relpath(item_path, os.path.dirname(directory))
                    team_name = team_name.replace(".txt", "").replace(os.sep, "/")

                    self._teams[format_name][team_name] = team_string

                    if showdown_path:
                        self.validate_team(team_string, format_name, showdown_path)
                except Exception as e:
                    if self._verbose:
                        print(f"  Error loading {item_path}: {e}")

    @staticmethod
    @contextmanager
    def change_dir(path):
        origin = os.getcwd()
        try:
            os.chdir(path)
            yield
        finally:
            os.chdir(origin)

    def validate_team(
        self,
        team: str,
        format: str,
        showdown_path: str = "../pokemon-showdown",
    ) -> bool:
        """
        Validate a team against Pokemon Showdown's official validation.

        This uses the pokemon-showdown CLI to pack and validate the team.
        Requires pokemon-showdown to be installed at the specified path.

        Args:
            team: Team string in PokePaste format
            format: Pokemon format to validate against (e.g., "gen9vgc2023regulationc")
            showdown_path: Path to pokemon-showdown directory

        Returns:
            True if team is valid, False otherwise
        """

        SHOWDOWN_PATH = os.path.abspath(os.path.expanduser(showdown_path))
        with self.change_dir(SHOWDOWN_PATH):

            try:
                result = subprocess.run(
                    ["./pokemon-showdown", "pack-team"],
                    input=team,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                packed_team = result.stdout.strip()
            except subprocess.CalledProcessError as e:
                if self._verbose:
                    print(f"Error packing team: {e}")
                    print(f"Error output: {e.stderr}")
                return False

            try:
                result = subprocess.run(
                    ["./pokemon-showdown", "validate-team", format],
                    input=packed_team,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                if self._verbose:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                if self._verbose:
                    print(f"Error validating team: {e}")
                    print(f"Format: {format}")
                    print(f"Team (packed): {packed_team}")
                    print(f"Error output: {e.stderr}")
                return False

        return True

    @property
    def teams(self) -> Dict[str, Dict[str, str]]:
        """Get all teams organized by format."""
        return self._teams

    @property
    def formats(self) -> KeysView[str]:
        """Get all available format names."""
        return self._teams.keys()

    def get(self, format="", name="") -> str:
        """
        Get a specific team by format and name.

        Args:
            format: Pokemon format (e.g., "gen9vgc2023regulationc")
            name: Team name (filename without .txt, may include subdirectory path)

        Returns:
            Team string in PokePaste format

        Example:
            >>> repo.get("gen9vgc2023regulationc", "rental_teams/team1")
        """
        return self._teams[format][name]

    def get_all(self, format: str) -> Dict[str, str]:
        """
        Get all teams for a specific format.

        Args:
            format: Pokemon format (e.g., "gen9vgc2023regulationc")

        Returns:
            Dictionary mapping team names to team strings
        """
        return self._teams[format]

    def sample_team(self, format: str, subdirectory: Optional[str] = None) -> str:
        """
        Sample a random team from the specified format.

        Args:
            format: Pokemon format (e.g., "gen9vgc2023regulationc")
            subdirectory: Optional subdirectory path to sample from (e.g., "rental_teams" or "tournament_teams/worlds_2023")
                         If None, samples from all teams in the format (default: None)

        Returns:
            Team string in PokePaste format

        Raises:
            ValueError: If format not found, no teams available, or subdirectory has no teams

        Examples:
            >>> repo.sample_team("gen9vgc2023regulationc")  # Sample from all teams
            >>> repo.sample_team("gen9vgc2023regulationc", "rental_teams")  # Sample from rental_teams only
        """
        if format not in self._teams:
            raise ValueError(f"Format '{format}' not found. Available formats: {list(self._teams.keys())}")

        format_teams = self._teams[format]
        if not format_teams:
            raise ValueError(f"No teams found for format '{format}'")

        # Filter by subdirectory if provided
        if subdirectory is not None:
            # Normalize subdirectory path separators
            subdirectory = subdirectory.replace(os.sep, "/")
            filtered_teams = {
                name: team for name, team in format_teams.items()
                if name.startswith(subdirectory + "/") or name == subdirectory
            }
            if not filtered_teams:
                raise ValueError(
                    f"No teams found in subdirectory '{subdirectory}' for format '{format}'. "
                    f"Available teams: {list(format_teams.keys())}"
                )
            format_teams = filtered_teams

        # Return a random team from the (filtered) format
        team_name = random.choice(list(format_teams.keys()))
        return format_teams[team_name]

    def sample_n_teams(self, format: str, n: int, with_replacement: bool = False, subdirectory: Optional[str] = None) -> List[str]:
        """
        Sample multiple random teams from the specified format.

        Args:
            format: Pokemon format (e.g., "gen9vgc2023regulationc")
            n: Number of teams to sample
            with_replacement: If True, same team can be sampled multiple times.
                            If False, each team can only be sampled once (default: False)
            subdirectory: Optional subdirectory path to sample from (e.g., "rental_teams" or "tournament_teams/worlds_2023")
                         If None, samples from all teams in the format (default: None)

        Returns:
            List of team strings in PokePaste format

        Raises:
            ValueError: If format not found, no teams available, subdirectory has no teams,
                       or n > available teams when with_replacement=False

        Examples:
            >>> repo.sample_n_teams("gen9vgc2023regulationc", 5)  # 5 unique teams from all
            >>> repo.sample_n_teams("gen9vgc2023regulationc", 10, with_replacement=True)  # 10 teams, may repeat
            >>> repo.sample_n_teams("gen9vgc2023regulationc", 3, subdirectory="rental_teams")  # 3 from rentals only
        """
        if format not in self._teams:
            raise ValueError(f"Format '{format}' not found. Available formats: {list(self._teams.keys())}")

        format_teams = self._teams[format]
        if not format_teams:
            raise ValueError(f"No teams found for format '{format}'")

        # Filter by subdirectory if provided
        if subdirectory is not None:
            # Normalize subdirectory path separators
            subdirectory = subdirectory.replace(os.sep, "/")
            filtered_teams = {
                name: team for name, team in format_teams.items()
                if name.startswith(subdirectory + "/") or name == subdirectory
            }
            if not filtered_teams:
                raise ValueError(
                    f"No teams found in subdirectory '{subdirectory}' for format '{format}'. "
                    f"Available teams: {list(format_teams.keys())}"
                )
            format_teams = filtered_teams

        available_teams = list(format_teams.values())
        num_available = len(available_teams)

        if not with_replacement and n > num_available:
            raise ValueError(
                f"Cannot sample {n} unique teams from format '{format}' "
                f"{'subdirectory ' + repr(subdirectory) if subdirectory else ''} "
                f"which only has {num_available} teams. "
                f"Use with_replacement=True to allow duplicates."
            )

        if with_replacement:
            # Sample with replacement using random.choices
            return random.choices(available_teams, k=n)
        else:
            # Sample without replacement using random.sample
            return random.sample(available_teams, k=n)

    def save_team(
        self,
        team: str,
        format: str,
        relative_path: str,
        validate: bool = True,
        showdown_path: str = "../pokemon-showdown",
    ) -> str:
        """
        Save a team to the appropriate format folder, supporting subdirectories.

        Args:
            team: Team string in PokePaste format
            format: Pokemon format (e.g., "gen9vgc2023regulationc")
            relative_path: Relative path within the format folder (e.g., "my_team" or "rental_teams/team1")
                          Extension (.txt) is optional and will be added if missing
            validate: Whether to validate the team before saving (default: True)
            showdown_path: Path to pokemon-showdown directory for validation

        Returns:
            Path to the saved team file

        Raises:
            ValueError: If validation fails or directory cannot be created

        Examples:
            >>> repo.save_team(team, "gen9vgc2023regulationc", "my_team")  # Saves to format root
            >>> repo.save_team(team, "gen9vgc2023regulationc", "rental_teams/team1")  # Saves to subdirectory
        """
        # Validate team if requested
        if validate and not self.validate_team(team, format, showdown_path):
            raise ValueError(f"Team validation failed for format '{format}'")

        # Normalize path separators and ensure .txt extension
        relative_path = relative_path.replace("/", os.sep).replace("\\", os.sep)
        if not relative_path.endswith(".txt"):
            relative_path = f"{relative_path}.txt"

        # Build full path: base_dir/format/relative_path
        format_dir = os.path.join(self.base_dir, format)
        team_path = os.path.join(format_dir, relative_path)

        # Create subdirectories if needed
        team_dir = os.path.dirname(team_path)
        os.makedirs(team_dir, exist_ok=True)

        # Write team to file
        with open(team_path, "w") as f:
            f.write(team)

        # Add to internal dictionary (use forward slashes for consistency)
        if format not in self._teams:
            self._teams[format] = {}
        team_key = relative_path.replace(".txt", "").replace(os.sep, "/")
        self._teams[format][team_key] = team

        if self._verbose:
            print(f"Saved team to: {team_path}")

        return team_path
