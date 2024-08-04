# -*- coding: utf-8 -*-
"""This module reads several teams in showdown format. There is no team validation as of yet
"""


import os
import os.path
import subprocess
from contextlib import contextmanager
from typing import Dict, KeysView


# TODO: can convert them into json and add team parsing/format information, with properties?
class TeamRepo:

    def __init__(
        self,
        filepath: str = "data/teams",
        verbose: bool = False,
        showdown_path: str = "../pokemon-showdown",
    ):
        self._teams: Dict[str, Dict[str, str]] = {}

        for format_folder in os.listdir(filepath):
            format_name = format_folder
            folder_path = os.path.join(filepath, format_folder)

            if os.path.isdir(folder_path):
                if verbose:
                    print(f"format: {format_name}")
                if format not in self._teams:
                    self._teams[format_name] = {}

                filenames = os.listdir(folder_path)
                for filename in filenames:
                    team_path = os.path.join(filepath, format_folder, filename)
                    if verbose:
                        print(team_path)
                    with open(team_path, "r") as f:
                        team_string = f.read()
                        self._teams[format_name][
                            os.path.basename(team_path).replace(".txt", "")
                        ] = team_string
                    self.validate_team(team_string, format_name, verbose, showdown_path)
                print(format_name)

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
        verbose: bool = False,
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
                if verbose:
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
                if verbose:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                if verbose:
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
