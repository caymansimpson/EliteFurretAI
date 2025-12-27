import os
from typing import Any, Dict, List, Optional, Union
import math
import random

import orjson
import torch
from torch.utils.data import Dataset

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.battle_iterator import BattleIterator
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO, MoveOrderEncoder
from elitefurretai.etl.battle_order_validator import is_valid_order
from elitefurretai.etl.evaluate_state import evaluate_position_advantage
from elitefurretai.etl.compress_utils import load_compressed

import platform
import warnings


# An implementation of Dataset for a series of showdown battles stored in a directory as BattleData
class BattleDataset(Dataset):
    """
    Loads raw battlefiles and generates training data for them.
    Returns (variable length ≤ steps_per_battle):

    states         : (T, embed_dim)
    actions        : (T,)
    action_masks   : (T, action_space)
    wins           : (T,) final outcome repeated each step (0/1)
    move_orders    : (T,) move order on next step
    kos            : (T, 4) ko order on next step
    switches       : (T, 4) switch order on next step
    masks          : (T,) 1 valid, 0 invalid
    """

    # files is a list of filepaths to BattleData files
    def __init__(
        self,
        files: List[str],
        embedder: Embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=True
        ),
        steps_per_battle: int = 17,
        augment_teampreview: bool = True,
    ):
        assert len(files) > 0

        self.files = files  # List of filepaths to raw battle JSON files
        self.embedder = embedder  # Feature embedder for converting battle state to vector
        self.steps_per_battle = (
            steps_per_battle  # Max number of steps to process per battle
        )
        self.augment_teampreview = augment_teampreview  # Whether to apply team order randomization

    # Because we can go through each file with a different perspective
    def __len__(self) -> int:
        # Each battle is processed twice (once for each player perspective)
        return len(self.files) * 2

    def __getitem__(self, idx: int):
        # For each index, determine which file and which player perspective
        file_idx = idx // 2
        perspective = "p" + str((idx % 2) + 1)
        file_path = self.files[file_idx].replace('\\', '/')

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # =============================================================================
        # TEAMPREVIEW DATA AUGMENTATION
        # =============================================================================
        # PROBLEM: Analysis showed 88.6% of unique team compositions (specific 4 Pokemon
        # in positions MON:0-3) always map to the same teampreview action. This means
        # the model memorizes "Team X always chooses action Y" rather than learning
        # strategic reasoning about matchups.
        #
        # EVIDENCE:
        # - 98.7% determinism on 200 battles, 88.6% on 20K battles
        # - Model achieves 99.9% validation accuracy by lookup table memorization
        # - Only 1.3-11.4% of team patterns show strategic variation
        #
        # ROOT CAUSE: Competitive players using team archetypes consistently bring
        # the same 4 Pokemon and lead with the same pair (e.g., Rain teams always
        # lead Pelipper+Barraskewda). Dataset lacks counterfactual examples where
        # the same team makes different choices against different opponents.
        #
        # SOLUTION: Randomize team order before creating BattleIterator, then adjust
        # teampreview action labels to match the new ordering. This forces the model
        # to learn position-invariant team representations and reason about matchups
        # rather than memorize "Pokemon at positions [0,1,2,3] → action N".
        #
        # This creates up to 720 different views per battle (6! permutations) and
        # breaks the deterministic team→action mapping that causes overfitting. Well,
        # it doesn't because the model can still memorize the teams, but it at least
        # makes the memorization harder.
        # =============================================================================

        if self.augment_teampreview:
            # Store original teams and teampreview inputs for augmentation
            p1_team_original = bd.p1_team.copy()
            p2_team_original = bd.p2_team.copy()

            # Generate random permutations for both teams
            # VGC teams can have 4-6 Pokemon, so use actual team size
            p1_permutation = list(range(len(p1_team_original)))
            p2_permutation = list(range(len(p2_team_original)))
            random.shuffle(p1_permutation)
            random.shuffle(p2_permutation)

            # Apply permutations to team lists
            bd.p1_team = [p1_team_original[i] for i in p1_permutation]
            bd.p2_team = [p2_team_original[i] for i in p2_permutation]

            # We also need to update the input logs to reflect the new team ordering
            # The input logs contain teampreview choices like "/team 1234" which refer
            # to Pokemon by their original positions. We need to remap these to the
            # new shuffled positions.
            updated_input_logs = []
            for input_log in bd.input_logs:
                if input_log.startswith(">p1 team") or input_log.startswith(">p2 team"):
                    # Parse the teampreview choice (e.g., ">p1 team 1234")
                    parts = input_log.split(" team ")
                    if len(parts) == 2:
                        player_prefix = parts[0]  # e.g., ">p1"
                        team_choice = parts[1]    # e.g., "1234"

                        # Determine which permutation to use
                        permutation = p1_permutation if player_prefix == ">p1" else p2_permutation

                        # Convert team choice from string to 0-indexed positions
                        # e.g., "4, 5, 1, 3" → [3, 4, 0, 2]
                        original_positions = [int(x.strip()) - 1 for x in team_choice.split(",")]

                        # Map original positions through the inverse permutation
                        # If permutation is [2,0,5,1,3,4] (new[i] = old[perm[i]]),
                        # then inverse is: old_pos → new_pos where new[new_pos] = old[old_pos]
                        inverse_perm = [0] * len(permutation)
                        for new_idx, old_idx in enumerate(permutation):
                            inverse_perm[old_idx] = new_idx

                        # Apply inverse permutation: original_pos → new_pos
                        new_positions = [inverse_perm[pos] for pos in original_positions]

                        # Convert back to 1-indexed string with comma separation
                        # e.g., [2,0,4,1] → "3, 1, 5, 2" (matching the format "4, 5, 1, 3")
                        new_team_choice = ", ".join(str(pos + 1) for pos in new_positions)

                        # Reconstruct the input log
                        updated_input_logs.append(f"{player_prefix} team {new_team_choice}")
                    else:
                        # Shouldn't happen, but keep original if parsing fails
                        updated_input_logs.append(input_log)
                else:
                    # Non-teampreview inputs remain unchanged
                    updated_input_logs.append(input_log)

            bd.input_logs = updated_input_logs

        # Initialize output tensors for this battle/perspective
        states = torch.zeros(self.steps_per_battle, self.embedder.embedding_size)
        actions = torch.zeros(self.steps_per_battle, dtype=torch.long)
        action_masks = torch.zeros(self.steps_per_battle, MDBO.action_space())
        wins = torch.full((self.steps_per_battle,), -1.0)
        masks = torch.zeros(self.steps_per_battle)

        # Create a battle iterator for this perspective
        # The iterator will now see Pokemon in shuffled order, forcing position-invariant learning
        iter = BattleIterator(
            bd,
            perspective=perspective,
            omniscient=self.embedder.omniscient,
        )

        # Store advantage by turn, and log indices by input number
        advantages: List[float] = []
        input_indices: List[int] = []

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
            masks[i] = 1

            # Store advantages and input_indices
            advantages.append(evaluate_position_advantage(iter.battle))  # type: ignore
            input_indices.append(iter.index)

            # Mask out struggle, revival blessing, and invalid actions
            if (
                iter.last_input
                and ("struggle" in iter.last_input or "revivalblessing" in iter.last_input)
            ) or action_masks[i, action_idx] == 0:
                masks[i] = 0

            i += 1

        # Compute advantages; need to do this after because advantages this turn
        # I compute from advantages in future turns
        wins = self._compute_ensemble_advantage(iter, advantages)

        # Compute move order and KO status -- these can be auxiliary tasks to help with learning
        move_orders = self._compute_move_order(iter, input_indices)
        kos = self._compute_ko_status(iter, input_indices)
        switches = self._compute_switch(iter, input_indices)

        return {
            "states": states,
            "actions": actions,
            "action_masks": action_masks,
            "wins": wins,
            "move_orders": move_orders,
            "kos": kos,
            "switches": switches,
            "masks": masks,
        }

    def _compute_move_order(
        self, iter: BattleIterator, input_indices: List[int]
    ) -> torch.Tensor:
        speed_orders = torch.zeros(self.steps_per_battle, dtype=torch.long)
        prev_idx = 0
        for i, end_idx in enumerate(input_indices):

            # Look for move order in the log range
            move_order = []
            for log_idx in range(prev_idx, min(end_idx, len(iter.bd.logs))):
                log = iter.bd.logs[log_idx]
                if log.startswith("|move|"):
                    split_message = log.split("|")
                    if len(split_message) >= 3:
                        pokemon_id = split_message[2][:3]
                        move_order.append(pokemon_id)

            # Encode the order (simplified to first 4 moves); things like Dancer/Instruct can mess with this
            encoded_order = MoveOrderEncoder.encode(move_order[:4])
            speed_orders[i] = encoded_order
            prev_idx = end_idx

        return speed_orders

    def _compute_ko_status(
        self, iter: BattleIterator, input_indices: List[int]
    ) -> torch.Tensor:
        prev_idx = 0
        ko_targets = torch.zeros((self.steps_per_battle, 4), dtype=torch.float32)
        for i, end_idx in enumerate(input_indices):

            # Track which pokemon faint: [p1a, p1b, p2a, p2b]
            positions = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
            fainted = [0, 0, 0, 0]

            # Look ahead in logs for faint messages
            for log_idx in range(prev_idx, min(end_idx, len(iter.bd.logs))):
                log = iter.bd.logs[log_idx]
                if log.startswith("|faint|"):
                    split_message = log.split("|")
                    if len(split_message) >= 3:
                        position = split_message[2][:3]
                        fainted[positions[position]] = 1

            ko_targets[i] = torch.Tensor(fainted)
            prev_idx = end_idx

        return ko_targets

    def _compute_switch(
        self, iter: BattleIterator, input_indices: List[int]
    ) -> torch.Tensor:
        switch_targets = torch.zeros((self.steps_per_battle, 4), dtype=torch.float32)
        prev_idx = 0
        for i, end_idx in enumerate(input_indices):

            # Track which pokemon switch: [p1a, p1b, p2a, p2b]
            positions = {"p1a": 0, "p1b": 1, "p2a": 2, "p2b": 3}
            switches = [0, 0, 0, 0]

            # Look for switch messages in the log range
            for log_idx in range(prev_idx, min(end_idx, len(iter.bd.logs))):
                log = iter.bd.logs[log_idx]
                if log.startswith("|switch|"):
                    split_message = log.split("|")
                    if len(split_message) >= 3:
                        position = split_message[2][:3]
                        switches[positions[position]] = 1
                elif log.startswith("|upkeep|"):
                    break

            switch_targets[i] = torch.Tensor(switches)
            prev_idx = end_idx

        return switch_targets

    def _compute_ensemble_advantage(
        self, iter: BattleIterator, advantages: List[float]
    ) -> torch.Tensor:
        """
        Compute ensemble advantage combining multiple heuristics:
        1. Estimated Position advantage of next few turns
        2. Final outcome
        3. Step-dependent weighting converging to true outcome
        """

        win_labels = torch.zeros(self.steps_per_battle)
        n = len(advantages)
        final_outcome = 1 if iter.bd.winner == iter.battle.player_username else -1

        for i in range(n):
            # How close are we to the end?
            progress = i * 1.0 / max(n - 1, 1)

            # Final outcome weight ramps up near the end (e.g., quadratic ramp)
            outcome_weight = min(max(progress**2, 0.05), 0.95)
            position_weight = 1.0 - outcome_weight

            # Current advantage
            current_adv = advantages[i]

            # Average advantage over next 3 turns (including current)
            next_window = advantages[i + 1: min(i + 4, n)]
            if len(next_window) == 0:
                avg_next_adv = current_adv
            else:
                avg_next_adv = sum(next_window) / len(next_window)

            # Blend: two stages
            # 1. position advantage: 50% of it goes to current turn, 50% goes to average
            #    over next 3 turns
            # 2. outcome advantage: final outcome of battle. Starts at 5%, ramps to 95%
            if n > 1:
                blended_adv = (
                    position_weight * (0.5 * current_adv + 0.5 * avg_next_adv)
                    + outcome_weight * final_outcome
                )
            else:
                blended_adv = final_outcome

            win_labels[i] = blended_adv

        return win_labels


class BattleIteratorDataset(Dataset):
    """
    Loads raw battle files and produces (states, actions, action_masks, wins, masks) for each perspective.
    Each file is processed twice (once for each player perspective). Returns iterators
    """

    # files is a list of filepaths to BattleData files
    def __init__(
        self,
        files: List[str],
        augment_teampreview: bool = True,
    ):
        assert len(files) > 0
        self.files = files  # List of filepaths to raw battle JSON files
        self.augment_teampreview = augment_teampreview

    # Because we can go through each file with a different perspective
    def __len__(self) -> int:
        # Each battle is processed twice (once for each player perspective)
        return len(self.files) * 2

    def __getitem__(self, idx) -> BattleIterator:
        # For each index, determine which file and which player perspective
        file_idx = idx // 2
        perspective = "p" + str((idx % 2) + 1)
        file_path = self.files[file_idx].replace('\\', '/')

        # Read the file into BattleData
        bd = None
        with open(file_path, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        # Apply same team order augmentation as BattleDataset for consistency
        # (See BattleDataset.__getitem__ for full explanation of the augmentation strategy)
        if self.augment_teampreview:
            p1_team_original = bd.p1_team.copy()
            p2_team_original = bd.p2_team.copy()

            p1_permutation = list(range(len(p1_team_original)))
            p2_permutation = list(range(len(p2_team_original)))
            random.shuffle(p1_permutation)
            random.shuffle(p2_permutation)

            bd.p1_team = [p1_team_original[i] for i in p1_permutation]
            bd.p2_team = [p2_team_original[i] for i in p2_permutation]

            # Update input logs to reflect new team ordering
            updated_input_logs = []
            for input_log in bd.input_logs:
                if input_log.startswith(">p1 team") or input_log.startswith(">p2 team"):
                    parts = input_log.split(" team ")
                    if len(parts) == 2:
                        player_prefix = parts[0]
                        team_choice = parts[1]
                        permutation = p1_permutation if player_prefix == ">p1" else p2_permutation
                        original_positions = [int(x.strip()) - 1 for x in team_choice.split(",")]
                        inverse_perm = [0] * len(permutation)
                        for new_idx, old_idx in enumerate(permutation):
                            inverse_perm[old_idx] = new_idx
                        new_positions = [inverse_perm[pos] for pos in original_positions]
                        new_team_choice = ", ".join(str(pos + 1) for pos in new_positions)
                        updated_input_logs.append(f"{player_prefix} team {new_team_choice}")
                    else:
                        updated_input_logs.append(input_log)
                else:
                    updated_input_logs.append(input_log)

            bd.input_logs = updated_input_logs

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
        metadata_filename: str = "_metadata.json",
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
                if f.endswith(".pt.zst")
            ]
        )

        # Build a mapping from global step index to (file_idx, traj_idx, step_idx)
        self._idx_map = []

        # If metadata file exists, use it to avoid reading all files
        metadata_path = os.path.join(folder_path, metadata_filename)
        if os.path.exists(metadata_path):
            # Load trajectory counts from metadata file
            with open(metadata_path, 'r') as f:
                metadata = orjson.loads(f.read())

            file_trajectory_counts = metadata.get('file_trajectory_counts', [])
            if len(file_trajectory_counts) != len(self.trajectory_files):
                raise ValueError("Metadata file trajectory counts do not match number of trajectory files.")

            # Build index map using metadata
            for file_idx, num_trajectories in enumerate(file_trajectory_counts):
                # Need to load one file to get steps per trajectory
                # Assume all trajectories in a file have same length (based on steps_per_battle)
                for traj_idx in range(num_trajectories):
                    for step_idx in range(steps_per_battle):
                        self._idx_map.append((file_idx, traj_idx, step_idx))
        else:
            # No metadata file; load each file to count trajectories and steps
            for file_idx, file_path in enumerate(self.trajectory_files):
                trajectories = torch.load(file_path, map_location="cpu")
                for traj_idx, traj_data in enumerate(trajectories):
                    states = traj_data["states"]
                    num_steps = states.shape[0]
                    for step_idx in range(num_steps):
                        self._idx_map.append((file_idx, traj_idx, step_idx))
                del trajectories  # Free memory

        self._length = len(self._idx_map)

        # Cache for last loaded file
        self._cache: Dict[str, Any] = {"file_idx": None, "trajectories": None}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        file_idx, traj_idx, step_idx = self._idx_map[idx]
        if self._cache["file_idx"] != file_idx:
            self._cache["trajectories"] = load_compressed(
                self.trajectory_files[file_idx], map_location="cpu"
            )
            self._cache["file_idx"] = file_idx
        data = self._cache["trajectories"][traj_idx]

        # Build return dictionary dynamically based on what's in the data
        result = {}
        for key in data.keys():
            value = data[key]
            if isinstance(value, torch.Tensor):
                # For teampreview data (steps_per_battle=1), tensors are already single-step (no step dimension)
                # For trajectory data, tensors have a step dimension that needs indexing
                if self.steps_per_battle == 1 or value.dim() <= 1:
                    # Teampreview: return as-is (already single step)
                    # or 0-d/1-d tensors that shouldn't be indexed
                    result[key] = value
                else:
                    # Trajectory: index into the step dimension
                    result[key] = value[step_idx]
            else:
                # For non-tensor data, return as-is
                result[key] = value

        return result

    def embedding_size(self) -> int:
        # Returns the embedding size used by the embedder
        return self.embedder.embedding_size


class OptimizedPreprocessedTrajectoryDataset(Dataset):
    """
    On Windows, PyTorch's DataLoader with num_workers > 0 uses spawn multiprocessing (not fork like on Linux), which means:
    - Each worker starts a fresh Python process
    - Your dataset object gets pickled and copied to each worker
    - Each worker has its own isolated cache
    - No sharing possible between workers

    This is why you'll see 0% cache hit rates with BattleDataset and PreprocessedBattleDataset in scripts/analyze/dataset_profiler.py on Windows!
    The cache exists, but only within each worker's isolated memory. A normally implemented cache will not work with multiprocessing on Windows.

    Instead, we need to implement smarter file-level batching with worker-aware caching. The key insight to this dataset: Have each worker
    "own" specific files; each worker loads and caches only its assigned files. Note that this also requires a custom Sampler, and also
    for us to set some system level parameters to avoid Windows shared memory limits.
    """
    def __init__(
        self,
        folder_path: str,
        metadata_filename: str = "_metadata.json",
        embedder: Embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=False
        ),
        files_per_worker: Union[int, str] = "default",
    ):
        """
        Args:
            folder_path: Path to folder containing .pt files
            metadata_filename: filename to metadata file containing trajectory counts; prevents read through dataset on init if exists
            embedder: Your embedder instance
            files_per_worker: How many files each worker should cache at once
            verbose: Print debug info
        """

        # On Windows, PyTorch's default shared memory strategy can hit limits with large datasets
        # on specifially Windows, that happens with multiprocessing DataLoader
        # See https://pytorch.org/docs/stable/multiprocessing.html#sharing-strategies
        if platform.system() == 'Windows' or "microsoft" in platform.uname()[2].lower():
            # Use file_system sharing to avoid Windows shared memory limits
            torch.multiprocessing.set_sharing_strategy('file_system')
            warnings.filterwarnings('ignore', message='.*socket.send.*')

        self.folder_path = folder_path
        self.embedder = embedder

        self._worker_id: Optional[int] = None
        self._worker_files: Optional[set] = None

        # Get all trajectory files
        self.trajectory_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.pt.zst')
        ])

        # If no files found, raise error
        if not self.trajectory_files:
            raise ValueError(f"No .pt.zst files found in {folder_path}")

        # Set how many files each worker should cache
        if isinstance(files_per_worker, int):
            self.files_per_worker: int = files_per_worker
        elif files_per_worker == "default":
            self.files_per_worker = max(2, len(self.trajectory_files) // ((os.cpu_count() or 2) * 2))

        # Build index: global_idx -> (file_idx, local_idx)
        self._idx_map = []
        self.file_trajectory_counts = []

        # If metadata file provided, use it to avoid reading all files now
        if os.path.exists(os.path.join(self.folder_path, metadata_filename)):
            # Load trajectory counts from metadata file; it is just a JSON with a list of counts
            # of number of trajectories per file
            with open(os.path.join(self.folder_path, metadata_filename), 'r') as f:
                metadata = orjson.loads(f.read())

            self.file_trajectory_counts = metadata.get('file_trajectory_counts', [])
            if len(self.file_trajectory_counts) != len(self.trajectory_files):
                raise ValueError("Metadata file trajectory counts do not match number of trajectory files.")

            for file_idx, num_trajectories in enumerate(self.file_trajectory_counts):
                for local_idx in range(num_trajectories):
                    self._idx_map.append((file_idx, local_idx))

        else:
            # No metadata file; load each file to count trajectories
            for file_idx, file_path in enumerate(self.trajectory_files):
                trajectories = torch.load(file_path, map_location='cpu', weights_only=False)
                num_trajectories = len(trajectories)
                self.file_trajectory_counts.append(num_trajectories)

                for local_idx in range(num_trajectories):
                    self._idx_map.append((file_idx, local_idx))

                del trajectories  # Free memory

        self._length = len(self._idx_map)

        # Worker-specific state (initialized in worker_init_fn)
        self._worker_id = None
        self._worker_files = None  # Files this worker is responsible for
        self._worker_cache: dict = {}  # This worker's file cache

    # Defines how we initialize and run each worker
    def _init_worker(self):
        """Initialize worker-specific state. Called automatically by DataLoader."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process loading
            self._worker_id = 0
            self.num_workers = 1
            self._worker_files = set(range(len(self.trajectory_files)))
        else:
            # Multi-process loading
            self._worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

            # Divide files among workers
            files_per_worker_calc = math.ceil(len(self.trajectory_files) / self.num_workers)
            start_file = self._worker_id * files_per_worker_calc
            end_file = min(start_file + files_per_worker_calc, len(self.trajectory_files))

            self._worker_files = set(range(start_file, end_file))

    def _load_file(self, file_idx: int):
        """Load file with worker-local caching"""

        # Lazy worker initialization
        if self._worker_id is None:
            self._init_worker()

        # Check worker-local cache
        if file_idx in self._worker_cache:
            return self._worker_cache[file_idx]

        # Load file
        file_path = self.trajectory_files[file_idx]
        trajectories = load_compressed(file_path, map_location='cpu')

        # Cache management: keep only files_per_worker most recent
        if len(self._worker_cache) >= self.files_per_worker:
            # Evict oldest (first key)
            oldest_file_idx = next(iter(self._worker_cache))
            del self._worker_cache[oldest_file_idx]

        self._worker_cache[file_idx] = trajectories

        return trajectories

    def __len__(self):
        return len(self._idx_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self._idx_map[idx]

        # Lazy worker initialization
        if self._worker_id is None:
            self._init_worker()

        # Skip if this file isn't assigned to this worker
        # (DataLoader will handle distributing indices appropriately)
        if self._worker_files and file_idx not in self._worker_files:
            # This shouldn't happen with proper sampler, but handle gracefully
            # Return from any cached file as fallback
            if self._worker_cache:
                fallback_file_idx = next(iter(self._worker_cache))
                trajectories = self._worker_cache[fallback_file_idx]
                return trajectories[0]  # Return first trajectory

            # If no cache, load any file we're responsible for
            fallback_file_idx = next(iter(self._worker_files))
            trajectories = self._load_file(fallback_file_idx)
            return trajectories[0]

        trajectories = self._load_file(file_idx)
        return trajectories[local_idx]
