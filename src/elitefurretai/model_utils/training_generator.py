import os
from typing import Any, Dict, Tuple, List
import random

import orjson

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.model_battle_order import ModelBattleOrder


# Yield only valid file contents (not directories) from directory_path
def file_generator(directory_path: str, slice: Tuple[float, float] = (0, 1), sort=lambda x: x):
    total_files = int(sum(1 for entry in os.scandir(directory_path) if entry.is_file()))

    file_list = sorted(os.listdir(directory_path), key=sort)[
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
        "format" in cfg and "data" in cfg and "train_slice" in cfg
        and "batch_size" in cfg and "label" in cfg
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
                    if cfg["label"] == "win":
                        Y.append(int(bd.winner == battle.player_username))
                    elif cfg["label"] == "order":
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


def generate_normalizations(cfg: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    return [random.random()], [1]
#     """
#     Calculate mean and std dev in a streaming fashion.
#     data_generator should yield rows of the matrix one at a time.
#     """
#     embedder = Embedder(format=cfg["format"], simple=False)

#     x_mean = [0] * embedder.embedding_size
#     x_std = [1] * embedder.embedding_size
#     n = 0

#     training_data = []
#     for bd, filename in file_generator(cfg["data"], slice=cfg["normalization"], sort=lambda x: random.random()):
#         for perspective in ["p1", "p2"]:
#             # Wrap the battle processing in a try/except block since BattleIterator can throw errors
#             try:

#                 # Create battle from the file, with a battle iterator
#                 battle = bd.to_battle(perspective)
#                 iter = BattleIterator(
#                     battle,
#                     bd,
#                     perspective=perspective,
#                     custom_parse=BattleData.showdown_translation,
#                 )

#                 # Iterate through the battle and get the player's input commands
#                 while not battle.finished and iter.next_input():

#                     # Get the last input command found by the iterator by the player, and stop if there's no more
#                     input = iter.last_input
#                     if input is None:
#                         continue

#                     battle.parse_request(iter.simulate_request())
#                     embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle))  # type: ignore
#                     n += 1
#
#                     # First pass: Calculate mean
#                     # n_samples = 0
#                     # column_sums = None

#                     # for row in data_generator():  # First pass
#                     #     if column_sums is None:
#                     #         column_sums = [0.0] * len(row)

#                     #     for i, value in enumerate(row):
#                     #         column_sums[i] += value
#                     #     n_samples += 1

#                     # means = [s / n_samples for s in column_sums]

#                     # # Second pass: Calculate variance/std dev
#                     # column_var_sums = [0.0] * len(means)

#                     # for row in data_generator():  # Second pass
#                     #     for i, value in enumerate(row):
#                     #         diff = value - means[i]
#                     #         column_var_sums[i] += diff * diff

#                     # stds = [((var_sum / n_samples) ** 0.5) for var_sum in column_var_sums]

#                     # Convert training data into embedding and state into a label
#                     #training_data.append(embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle)))  # type: ignore

#             except AssertionError as e:
#                 print("Assertion Error reading:", filename)
#                 print(e)
#             except RecursionError as e:
#                 print("Recursion Error reading:", filename)
#                 print(e)
#             except IndexError as e:
#                 print("Index Error reading:", filename)
#                 print(e)
#     return means, stds
