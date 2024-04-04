# -*- coding: utf-8 -*-

import glob
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from poke_env.data import to_id_str
from poke_env.player import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
)
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer

from elitefurretai.model_utils.data_processor import DataProcessor


def main():
    dp = DataProcessor(omniscient=True, double_data=True)
    files = glob.glob(
        "/Users/cayman/Repositories/EliteFurretAI/data/static/*/*.json", recursive=True
    )
    files += glob.glob(
        "/Users/cayman/Repositories/EliteFurretAI/data/static/*/*/*.json", recursive=True
    )
    data = dp.load_data(files=files)
    print(data.keys())


if __name__ == "__main__":
    main()
