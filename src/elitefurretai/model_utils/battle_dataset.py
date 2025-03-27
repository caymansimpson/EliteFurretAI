import os
from typing import List, Tuple, Optional, Callable
import orjson
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import time

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
            label_type: Optional[str] = "win",
            bd_eligibility_func: Optional[Callable] = None,
            max_turns: int = 17,
            means: Optional[torch.Tensor] = None,
            stds: Optional[torch.Tensor] = None
    ):
        assert len(files) > 0
        assert label_type in ["win", "order", "filename"]

        self.files = files
        self.embedder = Embedder(format=format, type="full")
        self.label_type = label_type

        if means is None:
            self.means = torch.zeros(max_turns, self.embedder.embedding_size).to(torch.float32)
        elif len(means.shape) == 1:
            self.means = means.unsqueeze(0).repeat(max_turns, 1).to(torch.float32)
        else:
            self.means = means.to(torch.float32)

        if stds is None:
            self.stds = torch.ones(max_turns, self.embedder.embedding_size).to(torch.float32)
        elif len(stds.shape) == 1:
            self.stds = stds.unsqueeze(0).repeat(max_turns, 1).to(torch.float32)
        else:
            self.stds = stds.to(torch.float32)

        assert self.stds.shape == (max_turns, self.embedder.embedding_size), f"Shape of stds is: {self.stds.shape} and should be {(max_turns, self.embedder.embedding_size)}"
        assert self.means.shape == (max_turns, self.embedder.embedding_size), f"Shape of means is: {self.means.shape} and should be {(max_turns, self.embedder.embedding_size)}"

        self.bd_eligibility_func = bd_eligibility_func if bd_eligibility_func is not None else self._dummy_func
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]

        # Load and process the showdown files
        X, Y = self.load_battle_data(file_path)

        # Create mask for valid steps
        mask = torch.cat((torch.ones(min(len(X), self.max_turns)), torch.zeros(max(0, self.max_turns - len(X)))))

        if len(X) < self.max_turns:
            X = torch.cat((X, torch.zeros(self.max_turns - len(X), self.embedder.embedding_size)))
            Y = torch.cat((Y, torch.zeros(self.max_turns - len(Y))))
        else:
            X = X[:self.max_turns]
            Y = Y[:self.max_turns]

        return (X - self.means) / self.stds, Y.int(), mask.bool()

    # Loads and encodes a battle in the official da ta format to an embedding
    def load_battle_data(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Check if the battle is valid for whatever task we have
        if not self.bd_eligibility_func(bd):
            return torch.tensor([]), torch.tensor([])

        X, Y = [], []

        # Look at battle from each player's perspective
        for perspective in ["p1", "p2"]:

            # Create battle from the file, with a battle iterator
            battle = bd.to_battle(perspective)
            iter = BattleIterator(
                battle,
                bd,
                perspective=perspective,
                # custom_parse=BattleData.showdown_translation, TODO
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

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    @staticmethod
    def generate_normalizations(
        files: List[str],
        format: str,
        bd_eligibility_func: Optional[Callable] = None,
        batch_size: int = 32,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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

        # Caculate progress
        start, last, total_battles = time.time(), 0, len(files)
        if verbose:
            print(f"Starting to calculate means and stds for {total_battles} battles to normalize input data...")

        # Calculate means and stds across all batches
        total_sum = None
        total_sq_sum = None
        total_count = 0

        for batch_X, _, _ in data_loader:
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

            if time.time() - start > last + 10 and verbose:  # Print every 10 seconds
                hours = int((time.time() - start) // 3600)
                minutes = int((time.time() - start) // 60)
                seconds = int((time.time() - start) % 60)
                print(
                    f"Calculated means and stds for {total_count}/{total_battles} battles "
                    f"({round(total_count / total_battles * 100, 2)}% complete) in "
                    f"{hours}h {minutes}m {seconds}s"
                )
                last += 10

        assert total_sum is not None and total_sq_sum is not None
        means = (total_sum / total_count).tolist()
        variances = (total_sq_sum / total_count - (total_sum / total_count) ** 2).tolist()
        stds = [np.maximum(np.sqrt(var), 1e-6) for var in variances]  # Avoid division by zero

        hours = int((time.time() - start) // 3600)
        minutes = int((time.time() - start) // 60)
        seconds = int((time.time() - start) % 60)
        print(f"Done generating normalizations for {total_battles} battles in {hours}h {minutes}m {seconds}s!")

        return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)
