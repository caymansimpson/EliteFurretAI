from typing import List, Tuple

import orjson
import torch
from torch.utils.data import Dataset

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.model_double_battle_order import MDBO
from elitefurretai.utils.battle_order_validator import is_valid_order


# An implementation of Dataset for a series of showdown battles stored in a directory as BattleData
class BattleDataset(Dataset):

    @staticmethod
    def _dummy_func(x):
        return True

    # files is a list of filepaths to BattleData files
    def __init__(
        self,
        files: List[str],
        embedder: Embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=True
        ),
        steps_per_battle: int = 17,
    ):
        assert len(files) > 0

        self.files = files
        self.embedder = embedder
        self.steps_per_battle = steps_per_battle

    # Because we can go through each file with a different perspective
    def __len__(self) -> int:
        return len(self.files) * 2

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Because we can go through each file with a different perspective
        # While this is correct behavior, it causes a slowdown with 2x IO
        file_idx = idx // 2
        perspective = "p" + str((idx % 2) + 1)
        file_path = self.files[file_idx]

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Initialize tensors
        states = torch.zeros(self.steps_per_battle, self.embedder.embedding_size)
        actions = torch.zeros(self.steps_per_battle)
        action_masks = torch.zeros(self.steps_per_battle, MDBO.action_space())
        wins = torch.zeros(self.steps_per_battle)
        masks = torch.zeros(self.steps_per_battle)

        # Create a battle iterator
        iter = BattleIterator(
            bd,
            perspective=perspective,
            omniscient=True,
        )

        # Iterate through the battle and get the player's input commands until ended or we exceed the steps ber battle parameter
        i = 0
        while not iter.battle.finished and iter.next_input() and i < self.steps_per_battle:

            # Get the last input command found by the iterator by the player, and stop if there's no more
            input = iter.last_input
            if input is None:
                continue

            iter.simulate_request()

            if iter.last_input_type == MDBO.TEAMPREVIEW:
                for possible_action_int in range(MDBO.action_space()):
                    if possible_action_int < MDBO.teampreview_space():
                        action_masks[i][possible_action_int] = 1
            else:
                assert iter.last_input_type is not None
                for possible_action_int in range(MDBO.action_space()):
                    if is_valid_order(MDBO.from_int(possible_action_int, iter.last_input_type), iter.battle):  # type: ignore
                        action_masks[i][possible_action_int] = 1

            # Convert training data into embedding and state into a label
            states[i] = torch.tensor(self.embedder.feature_dict_to_vector(self.embedder.featurize_double_battle(iter.battle)))  # type: ignore
            actions[i] = iter.last_order().to_int()
            wins[i] = int(bd.winner == iter.battle.player_username)
            masks[i] = 1

            i += 1

        return states, actions, action_masks, wins, masks

    def get_filename(self, idx: int) -> str:
        return self.files[idx // 2]

    def get_perspective(self, idx: int) -> str:
        return "p" + str((idx % 2) + 1)

    def embedding_size(self) -> int:
        return self.embedder.embedding_size
