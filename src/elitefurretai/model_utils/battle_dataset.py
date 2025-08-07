import os
from typing import Any, Dict, List, Tuple

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
    """
    Loads raw battle files and produces (states, actions, action_masks, wins, masks) for each perspective.
    Each file is processed twice (once for each player perspective).
    """

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

        self.files = files  # List of filepaths to raw battle JSON files
        self.embedder = embedder  # Feature embedder for converting battle state to vector
        self.steps_per_battle = (
            steps_per_battle  # Max number of steps to process per battle
        )

    # Because we can go through each file with a different perspective
    def __len__(self) -> int:
        # Each battle is processed twice (once for each player perspective)
        return len(self.files) * 2

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # For each index, determine which file and which player perspective
        file_idx = idx // 2
        perspective = "p" + str((idx % 2) + 1)
        file_path = self.files[file_idx]

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Initialize output tensors for this battle/perspective
        states = torch.zeros(self.steps_per_battle, self.embedder.embedding_size)
        actions = torch.zeros(self.steps_per_battle, dtype=torch.long)
        action_masks = torch.zeros(self.steps_per_battle, MDBO.action_space())
        wins = torch.zeros(self.steps_per_battle)
        masks = torch.zeros(self.steps_per_battle)

        # Create a battle iterator for this perspective
        iter = BattleIterator(
            bd,
            perspective=perspective,
            omniscient=True,
        )

        # Iterate through the battle and get the player's input commands until ended or we exceed the steps per battle parameter
        i = 0
        while not iter.battle.finished and iter.next_input() and i < self.steps_per_battle:

            # Get the last input command found by the iterator by the player, and stop if there's no more
            input = iter.last_input
            if input is None:
                continue

            # Build the action mask for this step
            if iter.last_input_type == MDBO.TEAMPREVIEW:
                for possible_action_int in range(MDBO.action_space()):
                    if possible_action_int < MDBO.teampreview_space():
                        action_masks[i, possible_action_int] = 1
            else:
                assert iter.last_input_type is not None
                for possible_action_int in range(MDBO.action_space()):
                    try:
                        dbo = MDBO.from_int(possible_action_int, iter.last_input_type).to_double_battle_order(iter.battle)  # type: ignore
                        if is_valid_order(dbo, iter.battle):  # type: ignore
                            action_masks[i, possible_action_int] = 1
                    except Exception:
                        continue  # Skip invalid orders (if the int creates an invalid order)

            # Get the target action index for this step
            action_idx = iter.last_order().to_int()

            # Convert training data into embedding and state into a label
            states[i] = torch.tensor(self.embedder.feature_dict_to_vector(self.embedder.embed(iter.battle)))  # type: ignore
            actions[i] = action_idx
            wins[i] = int(bd.winner == iter.battle.player_username)
            masks[i] = 1

            # Mask out struggle, revival blessing, and invalid actions
            if (
                iter.last_input
                and ("struggle" in iter.last_input or "revivalblessing" in iter.last_input)
            ) or action_masks[i, action_idx] == 0:
                masks[i] = 0

            i += 1

        return states, actions, action_masks, wins, masks

    def get_filename(self, idx: int) -> str:
        # Returns the filename for a given index (for debugging/analysis)
        return self.files[idx // 2]

    def get_perspective(self, idx: int) -> str:
        # Returns the player perspective ("p1" or "p2") for a given index
        return "p" + str((idx % 2) + 1)

    def embedding_size(self) -> int:
        # Returns the embedding size used by the embedder
        return self.embedder.embedding_size


class BattleIteratorDataset(Dataset):
    """
    Loads raw battle files and produces (states, actions, action_masks, wins, masks) for each perspective.
    Each file is processed twice (once for each player perspective). Returns iterators
    """

    # files is a list of filepaths to BattleData files
    def __init__(
        self,
        files: List[str],
    ):
        assert len(files) > 0
        self.files = files  # List of filepaths to raw battle JSON files

    # Because we can go through each file with a different perspective
    def __len__(self) -> int:
        # Each battle is processed twice (once for each player perspective)
        return len(self.files) * 2

    def __getitem__(self, idx) -> BattleIterator:
        # For each index, determine which file and which player perspective
        file_idx = idx // 2
        perspective = "p" + str((idx % 2) + 1)
        file_path = self.files[file_idx]

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Return a battle iterator for this perspective
        return BattleIterator(
            bd,
            perspective=perspective,
            omniscient=True,
        )


class PreprocessedBattleDataset(Dataset):
    """
    Loads preprocessed trajectory files (each file contains a list of trajectories, each trajectory is a tuple of tensors).
    Returns (states, actions, action_masks, wins, masks) for a single step, just like BattleDataset.
    Only one file is loaded into memory at a time.
    """

    def __init__(
        self,
        folder_path,
        embedder: Embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=True
        ),
        steps_per_battle=17,
    ):
        self.embedder = embedder
        self.steps_per_battle = steps_per_battle
        self.folder_path = folder_path

        # List all .pt files in the folder, sorted for reproducibility
        self.trajectory_files = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".pt")
            ]
        )

        # Build a mapping from global step index to (file_idx, traj_idx, step_idx)
        self._idx_map = []
        self._file_traj_lengths = []  # For debugging/analysis
        for file_idx, file_path in enumerate(self.trajectory_files):
            trajectories = torch.load(file_path, map_location="cpu")
            self._file_traj_lengths.append(len(trajectories))
            for traj_idx, traj in enumerate(trajectories):
                states, actions, action_masks, wins, masks = traj
                num_steps = states.shape[0]
                for step_idx in range(num_steps):
                    self._idx_map.append((file_idx, traj_idx, step_idx))
        self._length = len(self._idx_map)

        # Cache for last loaded file
        self._cache: Dict[str, Any] = {"file_idx": None, "trajectories": None}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        file_idx, traj_idx, step_idx = self._idx_map[idx]
        if self._cache["file_idx"] != file_idx:
            self._cache["trajectories"] = torch.load(
                self.trajectory_files[file_idx], map_location="cpu"
            )
            self._cache["file_idx"] = file_idx
        traj = self._cache["trajectories"][traj_idx]
        states, actions, action_masks, wins, masks = traj
        return (
            states[step_idx],
            actions[step_idx],
            action_masks[step_idx],
            wins[step_idx],
            masks[step_idx],
        )

    def embedding_size(self) -> int:
        # Returns the embedding size used by the embedder
        return self.embedder.embedding_size


class PreprocessedTrajectoryDataset(torch.utils.data.Dataset):
    """
    Loads full trajectories from multiple .pt files, only one file in memory at a time.
    Each __getitem__ returns (states, actions, action_masks, wins, masks) for one trajectory.
    """

    def __init__(
        self,
        folder_path,
        embedder: Embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=True
        ),
    ):
        self.embedder = embedder

        self.trajectory_files = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".pt")
            ]
        )

        # Build index mapping: for each file, how many trajectories does it contain?
        self.file_lengths = []
        self._length = 0
        for file in self.trajectory_files:
            n = len(torch.load(file, map_location="cpu"))
            self.file_lengths.append(n)
            self._length += n

        # Build global idx -> (file_idx, local_idx)
        self._idx_map = []
        for file_idx, file_len in enumerate(self.file_lengths):
            for local_idx in range(file_len):
                self._idx_map.append((file_idx, local_idx))

        # Cache for last loaded file
        self._cache = {"file_idx": None, "trajectories": None}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        file_idx, local_idx = self._idx_map[idx]
        if self._cache["file_idx"] != file_idx:
            self._cache["trajectories"] = torch.load(
                self.trajectory_files[file_idx], map_location="cpu"
            )
            self._cache["file_idx"] = file_idx
        assert self._cache["trajectories"]
        return self._cache["trajectories"][local_idx]

    def embedding_size(self) -> int:
        # Returns the embedding size used by the embedder
        return self.embedder.embedding_size
