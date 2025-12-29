import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.evaluate import load_model
from elitefurretai.rl.worker import BatchInferencePlayer
from elitefurretai.supervised.behavior_clone_player import BCPlayer


class ExploiterRegistry:
    """Manages the registry of trained exploiter models."""

    def __init__(self, registry_path: str = "data/models/exploiter_registry.json"):
        self.registry_path = registry_path

        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.exploiters: List[Dict[str, Any]] = data.get("exploiters", [])
        else:
            self.exploiters = []

    def save(self):
        """Save exploiters to registry file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump({"exploiters": self.exploiters}, f, indent=2)

    def add_exploiter(self, exploiter_info: Dict[str, Any]):
        """
        Add a new exploiter to the registry.

        Args:
            exploiter_info: Dictionary with keys:
                - id: Unique identifier
                - filepath: Path to model checkpoint
                - team: Team string (PokePaste format) used for training
                - victim_step: Training step of victim model
                - win_rate: Evaluation win rate
                - trained_date: Training date (YYYY-MM-DD)
                - notes: Optional notes
        """
        if "id" not in exploiter_info:
            raise ValueError("Exploiter info must contain 'id' field.")
        elif "filepath" not in exploiter_info:
            raise ValueError("Exploiter info must contain 'filepath' field.")
        elif "team" not in exploiter_info:
            raise ValueError("Exploiter info must contain 'team' field.")
        elif any(exp["id"] == exploiter_info["id"] for exp in self.exploiters):
            raise ValueError(f"Exploiter with id {exploiter_info['id']} already exists.")
        elif any(exp["filepath"] == exploiter_info["filepath"] for exp in self.exploiters):
            raise ValueError(
                f"Exploiter with filepath {exploiter_info['filepath']} already exists."
            )

        self.exploiters.append(exploiter_info)
        self.save()

    def get_active_exploiters(self, min_win_rate: float = 0.55) -> List[Dict[str, Any]]:
        """Get list of exploiters above minimum win rate threshold."""
        return [exp for exp in self.exploiters if exp.get("win_rate", 0) >= min_win_rate]

    def get_exploiter_by_id(self, exploiter_id: str) -> Optional[Dict[str, Any]]:
        """Get specific exploiter by ID."""
        for exp in self.exploiters:
            if exp["id"] == exploiter_id:
                return exp
        return None


class OpponentPool:
    """
    Manages a pool of opponents for training including:
    - Self-play (current main model)
    - Behavior Clone player (human baseline)
    - Exploiters (adversarial models)
    - Past versions (historical checkpoints)
    """

    def __init__(
        self,
        main_model: RNaDAgent,
        device: str,
        battle_format: str = "gen9vgc2023regc",
        bc_teampreview_path: Optional[str] = None,
        bc_action_path: Optional[str] = None,
        bc_win_path: Optional[str] = None,
        exploiter_registry_path: str = "data/models/exploiter_registry.json",
        past_models_dir: str = "data/models/ghosts",
        max_past_models: int = 10,
        curriculum: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize opponent pool.

        Args:
            main_model: Current main model being trained
            device: Device to load models on
            battle_format: Pokemon Showdown format string for battles
            bc_teampreview_path: Path to BC teampreview model
            bc_action_path: Path to BC action model
            bc_win_path: Path to BC win prediction model
            exploiter_registry_path: Path to exploiter registry JSON
            past_models_dir: Directory containing past model checkpoints
            max_past_models: Maximum number of past versions to keep
            curriculum: Dict of opponent type -> sampling probability
                       If None, uses default curriculum
        """
        self.main_model = main_model
        self.device = device
        self.battle_format = battle_format
        self.max_past_models = max_past_models

        # Default curriculum: weights for sampling different opponent types
        self.curriculum = curriculum or {
            "self_play": 0.40,  # 40% self-play
            "bc_player": 0.20,  # 20% human baseline
            "exploiters": 0.20,  # 20% exploiters
            "ghosts": 0.20,  # 20% past versions
        }

        # Validate curriculum sums to 1.0
        total = sum(self.curriculum.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Curriculum weights must sum to 1.0, got {total}")

        # Load BC player if paths provided
        self.bc_player_config = None
        if bc_teampreview_path and bc_action_path and bc_win_path:
            self.bc_player_config = {
                "teampreview": bc_teampreview_path,
                "action": bc_action_path,
                "win": bc_win_path,
            }

        # Load exploiter registry
        self.exploiter_registry = ExploiterRegistry(exploiter_registry_path)
        self.loaded_exploiters: Dict[
            str, RNaDAgent
        ] = {}  # Cache: exploiter_id -> RNaDAgent

        # Track past model checkpoints
        self.past_models_dir = past_models_dir
        os.makedirs(past_models_dir, exist_ok=True)
        self.past_models: List[Tuple[int, str]] = []  # List of (step, filepath)
        self._load_past_models()

        # Win rate tracking
        self.win_rates: Dict[str, List[float]] = {
            "self_play": [],
            "bc_player": [],
            "exploiters": [],
            "ghosts": [],  # fka past_versions
        }

    def _load_past_models(self):
        """Scan past_models_dir and load available checkpoints."""
        if not os.path.exists(self.past_models_dir):
            return

        files = os.listdir(self.past_models_dir)
        model_files = [f for f in files if f.endswith(".pt")]

        # Extract step numbers and sort
        models = []
        for f in model_files:
            # Expected format: main_model_step_XXXXX.pt
            try:
                step = int(f.split("_step_")[1].split(".pt")[0])
                models.append((step, os.path.join(self.past_models_dir, f)))
            except ValueError:
                continue

        # Keep only the most recent max_past_models
        models.sort(key=lambda x: x[0], reverse=True)
        self.past_models = models[: self.max_past_models]

    def _load_exploiter_model(self, exploiter_info: Dict[str, Any]) -> RNaDAgent:
        """Load an exploiter model from disk."""
        exploiter_id = exploiter_info["id"]

        # Check cache
        if exploiter_id in self.loaded_exploiters:
            return self.loaded_exploiters[exploiter_id]

        # Load from disk
        model = load_model(exploiter_info["filepath"], self.device)

        # Cache it
        self.loaded_exploiters[exploiter_id] = model
        return model

    def sample_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        """
        Sample an opponent based on curriculum.

        Args:
            player_config: AccountConfiguration for the opponent
            server_config: ServerConfiguration for battles

        Returns:
            Player instance
        """
        # Sample opponent type
        roll = np.random.rand()
        cumsum = 0.0
        opponent_type = None

        for opp_type, weight in self.curriculum.items():
            cumsum += weight
            if roll < cumsum:
                opponent_type = opp_type
                break

        # Fallback
        if opponent_type is None:
            opponent_type = "self_play"

        # Create opponent based on type
        if opponent_type == "self_play":
            return self._create_self_play_opponent(player_config, server_config, team)
        elif opponent_type == "bc_player":
            return self._create_bc_opponent(player_config, server_config, team)
        elif opponent_type == "exploiters":
            return self._create_exploiter_opponent(player_config, server_config, team)
        elif opponent_type == "ghosts":
            return self._create_past_opponent(player_config, server_config, team)
        else:
            # Fallback to self-play
            return self._create_self_play_opponent(player_config, server_config, team)

    def _create_self_play_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        """Create a self-play opponent (copy of main model)."""
        return BatchInferencePlayer(
            model=self.main_model,
            device=self.device,
            batch_size=16,
            account_configuration=player_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=team,
        )

    def _create_bc_opponent(self, player_config, server_config, team: str) -> Player:
        """Create a BC player opponent."""
        if self.bc_player_config is None:
            # Fallback to self-play if BC not configured
            return self._create_self_play_opponent(player_config, server_config, team)

        return BCPlayer(
            teampreview_model_filepath=self.bc_player_config["teampreview"],
            action_model_filepath=self.bc_player_config["action"],
            win_model_filepath=self.bc_player_config["win"],
            battle_format=self.battle_format,
            probabilistic=True,
            device=self.device,
            account_configuration=player_config,
            server_configuration=server_config,
            team=team,
        )

    def _create_exploiter_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        """Create an exploiter opponent using its trained team."""
        active_exploiters = self.exploiter_registry.get_active_exploiters()

        if not active_exploiters:
            # Fallback to BC or self-play
            if self.bc_player_config:
                return self._create_bc_opponent(player_config, server_config, team)
            else:
                return self._create_self_play_opponent(player_config, server_config, team)

        # Sample random exploiter
        exploiter_info = random.choice(active_exploiters)
        exploiter_model = self._load_exploiter_model(exploiter_info)

        # Use the exploiter's trained team
        exploiter_team = exploiter_info.get(
            "team", team
        )  # Fallback to provided team if not found

        return BatchInferencePlayer(
            model=exploiter_model,
            device=self.device,
            batch_size=16,
            account_configuration=player_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=exploiter_team,
        )

    def _create_past_opponent(self, player_config, server_config, team: str) -> Player:
        """Create an opponent from past model versions."""
        if not self.past_models:
            # Fallback to BC or self-play
            if self.bc_player_config:
                return self._create_bc_opponent(player_config, server_config, team)
            else:
                return self._create_self_play_opponent(player_config, server_config, team)

        # Sample random past model
        step, filepath = self.past_models[np.random.randint(len(self.past_models))]

        past_model = load_model(filepath, self.device)

        return BatchInferencePlayer(
            model=past_model,
            device=self.device,
            batch_size=16,
            account_configuration=player_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=team,
        )

    def add_past_model(self, step: int, filepath: str):
        """Add a new past model checkpoint."""
        self.past_models.append((step, filepath))
        # Sort by step descending
        self.past_models.sort(key=lambda x: x[0], reverse=True)
        # Keep only max_past_models
        self.past_models = self.past_models[: self.max_past_models]

    def update_win_rate(self, opponent_type: str, won: bool):
        """Track win rate against different opponent types."""
        if opponent_type in self.win_rates:
            self.win_rates[opponent_type].append(1.0 if won else 0.0)

    def get_win_rate_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get recent win rate statistics for each opponent type.

        Args:
            window: Number of recent games to average over

        Returns:
            Dict mapping opponent_type -> win_rate
        """
        stats: Dict[str, float] = {}
        for opp_type, results in self.win_rates.items():
            if results:
                recent = results[-window:]
                stats[opp_type] = float(np.mean(recent))
            else:
                stats[opp_type] = 0.0
        return stats

    def update_curriculum(self, adaptive: bool = True):
        """
        Update curriculum based on performance (adaptive learning).

        If the agent is dominating exploiters (>70% WR), reduce their weight.
        If struggling against BC (<40% WR), increase BC weight.
        """
        if not adaptive:
            return

        stats = self.get_win_rate_stats()

        # Adaptive adjustments
        new_curriculum = self.curriculum.copy()

        # If dominating exploiters, reduce their weight
        if (
            stats.get("exploiters", 0) > 0.70
            and self.exploiter_registry.get_active_exploiters()
        ):
            new_curriculum["exploiters"] = max(0.10, new_curriculum["exploiters"] - 0.05)
            new_curriculum["self_play"] += 0.05

        # If struggling against BC, increase BC weight
        if stats.get("bc_player", 0) < 0.40 and self.bc_player_config:
            new_curriculum["bc_player"] = min(0.40, new_curriculum["bc_player"] + 0.05)
            new_curriculum["self_play"] = max(0.20, new_curriculum["self_play"] - 0.05)

        # Renormalize
        total = sum(new_curriculum.values())
        for key in new_curriculum:
            new_curriculum[key] /= total

        self.curriculum = new_curriculum
