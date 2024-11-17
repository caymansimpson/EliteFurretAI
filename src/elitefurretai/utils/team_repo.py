# -*- coding: utf-8 -*-
"""This module reads several teams in showdown format. There is no team validation as of yet
"""


import os
import os.path
import subprocess
from contextlib import contextmanager
from typing import Dict, KeysView, Optional

_DEFAULT_FILEPATH = "data/teams"


class TeamRepo:

    def __init__(
        self,
        filepath: Optional[str] = None,
        showdown_path: str = "../pokemon-showdown",
        validate: bool = False,
        verbose: bool = False,
    ):
        self._teams: Dict[str, Dict[str, str]] = {}
        self._verbose = verbose

        # If we have the default filepath, use the default
        if filepath is None:
            current_file = os.path.dirname(
                os.path.abspath(__file__)
            )  # Gets current directory
            three_up = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            filepath = os.path.join(three_up, _DEFAULT_FILEPATH)

        directories = set()
        for format_folder in os.listdir(filepath):
            folder_path = os.path.join(filepath, format_folder)

            if os.path.isdir(folder_path):
                if verbose:
                    print(f"format: {format_folder}")
                if format_folder not in self._teams:
                    self._teams[format_folder] = {}
                directories.add(folder_path)

        for directory in directories:
            filenames = os.listdir(directory)
            for filename in filenames:
                team_path = os.path.join(directory, filename)
                frmt = os.path.basename(directory)

                if verbose:
                    print(team_path)

                with open(team_path, "r") as f:
                    team_string = f.read()
                    self._teams[frmt][filename.replace(".txt", "")] = team_string

                if validate:
                    self.validate_team(team_string, frmt, showdown_path)

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
        return self._teams

    @property
    def formats(self) -> KeysView[str]:
        return self._teams.keys()

    def get(self, format="", name="") -> str:
        return self._teams[format][name]

    def get_all(self, format="") -> Dict[str, str]:
        return self._teams[format]
