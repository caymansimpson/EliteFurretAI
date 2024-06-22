# -*- coding: utf-8 -*-
"""This module reads several teams in showdown format. There is no team validation as of yet
"""


import os
import os.path
from typing import Dict


# TODO: can convert them into json and add team parsing/format information, with properties?
class TeamRepo:

    def __init__(self, filepath: str = "data/teams"):
        self._teams: Dict[str, str] = {}
        files = [
            os.path.join(filepath, f)
            for f in os.listdir(filepath)
            if os.path.isfile(os.path.join(filepath, f))
        ]
        for file in files:
            with open(file, "r") as f:
                self._teams[os.path.basename(file).replace(".txt", "")] = f.read()

    @property
    def teams(self) -> Dict[str, str]:
        return self._teams
