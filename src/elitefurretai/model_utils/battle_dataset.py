import os
from typing import List, Tuple, Optional, Callable
import orjson
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.model_utils.model_battle_order import ModelBattleOrder


# Pytorch Dataset class
class BattleDataset(Dataset):

    @staticmethod
    def _dummy_func(x):
        return True

    def __init__(
            self,
            files: List[str],
            format: str,
            label_type: Optional[str] = None,
            bd_eligibility_func: Optional[Callable] = None,
            means: Optional[List[float]] = None,
            stds: Optional[List[float]] = None
    ):
        assert len(files) > 0
        assert label_type in ["win", "order", "filename"]

        self.files = files
        self.embedder = Embedder(format=format, simple=False)
        self.label_type = label_type
        self.means = means if means is not None else [0] * self.embedder.embedding_size
        self.stds = stds if stds is not None else [1.0] * self.embedder.embedding_size
        self.bd_eligibility_func = bd_eligibility_func if bd_eligibility_func is not None else self._dummy_func

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]

        # Load and process the showdown files
        X, Y = self.load_battle_data(file_path)

        # Normalize the data
        for i, x in enumerate(X):
            X[i] = [(x - mean) / std for x, mean, std in zip(x, self.means, self.stds)]

        return torch.FloatTensor(X), torch.FloatTensor(Y)

    # Loads and encodes a battle in the official da ta format to an embedding
    def load_battle_data(self, file_path: str) -> Tuple[List[List[float]], List[float]]:

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Check if the battle is valid for whatever task we have
        if not self.bd_eligibility_func(bd):
            return [], []

        X, Y = [], []

        # Look at battle from each player's perspective
        for perspective in ["p1", "p2"]:

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

                battle.parse_request(iter.simulate_request())

                # Convert training data into embedding and state into a label
                X.append(self.embedder.feature_dict_to_vector(self.embedder.featurize_double_battle(battle)))  # type: ignore
                if self.label_type == "win":
                    Y.append(int(bd.winner == battle.player_username))
                elif self.label_type == "order":
                    Y.append(ModelBattleOrder.from_battle_data(input, battle, bd).to_int())
                elif self.label_type == "filename":
                    Y.append(int(os.path.basename(file_path.split(".")[0])))

        return (X, Y)

    @staticmethod
    def generate_normalizations(
        files: List[str],
        format: str,
        bd_eligibility_func: Optional[Callable] = None,
        batch_size: int = 32
    ) -> Tuple[List[float], List[float]]:

        dataset = BattleDataset(
            files=files,
            format=format,
            bd_eligibility_func=bd_eligibility_func,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min(os.cpu_count() or 1, 4),
        )

        # Calculate means and stds across all batches
        total_sum = None
        total_sq_sum = None
        total_count = 0

        for batch_X, _ in data_loader:
            batch_sum = torch.sum(batch_X, dim=0)
            batch_sq_sum = torch.sum(batch_X ** 2, dim=0)
            batch_count = batch_X.shape[0]

            if total_sum is None and total_sq_sum is None:
                total_sum = batch_sum
                total_sq_sum = batch_sq_sum
            elif total_sum is not None and total_sq_sum is not None:
                total_sum += batch_sum
                total_sq_sum += batch_sq_sum

            total_count += batch_count

        assert total_sum is not None and total_sq_sum is not None
        means = (total_sum / total_count).tolist()
        variances = (total_sq_sum / total_count - (total_sum / total_count) ** 2).tolist()
        stds = [max(np.sqrt(var), 1e-6) for var in variances]  # Avoid division by zero

        return means, stds
