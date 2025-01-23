import os
from typing import Any, Dict, Tuple

import orjson

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.model_battle_order import ModelBattleOrder


# Yield only valid file contents (not directories) from directory_path
def file_generator(directory_path: str, slice: Tuple[float, float] = (0, 1)):
    total_files = int(sum(1 for entry in os.scandir(directory_path) if entry.is_file()))
    file_list = sorted(os.listdir(directory_path))[
        int(total_files * slice[0]) : int(total_files * slice[1])
    ]

    for filename in file_list:
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
            with open(full_path, "r") as f:
                bd = BattleData.from_showdown_json(orjson.loads(f.read()))
                if bd.is_valid_for_supervised_learning:
                    yield (bd, os.path.splitext(os.path.basename(full_path))[0])


# Yields batches of data, from a directory according to cfg
def batch_generator(cfg: Dict[str, Any]):
    assert (
        "format" in cfg and "data" in cfg and "train_slice" in cfg and "batch_size" in cfg
    )

    embedder = Embedder(format=cfg["format"], simple=False)

    X, Y = [], []
    for bd, filename in file_generator(cfg["data"], slice=cfg["train_slice"]):

        # Look at battle from each player's perspective
        for perspective in ["p1", "p2"]:

            # Wrap the battle processing in a try/except block since BattleIterator can throw errors
            try:

                # Create battle from the file, with a battle iterator
                battle = bd.to_battle(perspective)
                iter = BattleIterator(
                    battle,
                    bd,
                    perspective=perspective,
                    custom_parse=BattleData.showdown_translation,
                )

                # Iterate through the battle and get the player's input commands
                while not battle.finished and iter.next_input():

                    # Get the last input command found by the iterator by the player, and stop if there's no more
                    input = iter.last_input
                    if input is None:
                        continue

                    request = iter.simulate_request()
                    battle.parse_request(request)

                    # Convert training data into embedding and state into a label
                    X.append(embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle)))  # type: ignore
                    Y.append(ModelBattleOrder.from_battle_data(input, battle, bd).to_int())

                    if len(X) % cfg["batch_size"] == 0:
                        yield (X, Y)
                        X, Y = [], []

            except AssertionError as e:
                print("Assertion Error reading:", filename)
                print(e)
            except RecursionError as e:
                print("Recursion Error reading:", filename)
                print(e)
            except IndexError as e:
                print("Index Error reading:", filename)
                print(e)

    yield (X, Y)
