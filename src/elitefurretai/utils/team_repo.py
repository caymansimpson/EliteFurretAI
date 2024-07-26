# -*- coding: utf-8 -*-
"""This module reads several teams in showdown format. There is no team validation as of yet
"""


import os
import os.path
import subprocess
from typing import Dict


# TODO: can convert them into json and add team parsing/format information, with properties?
class TeamRepo:

    def __init__(self, filepath: str = "data/teams"):
        self._teams: Dict[str, str] = {}

        for format_folder in os.listdir(filepath):
            format_name = format_folder
            folder_path = os.path.join(filepath, format_folder)
            if os.path.isdir(folder_path):
                print(f"format name: {format_name}")
                filenames = os.listdir(folder_path)
                for filename in filenames:
                    team_path = os.path.join(filepath, format_folder, filename)
                    print(team_path)
                    with open(team_path, "r") as f:
                        team_string = f.read()
                    self._teams[
                        format_name + "_" + os.path.basename(team_path).replace(".txt", "")
                    ] = team_string
                    self.validate_team(team_string, format_name)

    # Returns True if valid
    def validate_team(
        self, team: str, format: str, SHOWDOWN_PATH: str = "../pokemon-showdown"
    ) -> bool:

        output = True
        ELITEFURRETAI_PATH = os.getcwd()
        os.chdir(SHOWDOWN_PATH)

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
            print(f"Error packing team: {e}")
            print(f"Error output: {e.stderr}")
            output = False

        try:
            result = subprocess.run(
                ["./pokemon-showdown", "validate-team", format],
                input=packed_team,
                text=True,
                capture_output=True,
                check=True,
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error validating team: {e}")
            print(f"Format: {format}")
            print(f"Team (packed): {packed_team}")
            print(f"Error output: {e.stderr}")
            output = False

        os.chdir(ELITEFURRETAI_PATH)
        return output

    @property
    def teams(self) -> Dict[str, str]:
        return self._teams
