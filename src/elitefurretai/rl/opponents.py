"""Opponent management for RL training.

This module contains three related but intentionally different classes:

1) `ExploiterRegistry`
    - Lightweight persistence layer for exploiter metadata
    - Stores/loads exploiter records from JSON
    - Does not create battle players or sample curriculum

2) `OpponentPool`
    - Learner/main-process opponent manager
    - Samples opponent *types* using curriculum probabilities
    - Lazily loads/caches models (BC, exploiters, ghost checkpoints)
    - Creates concrete `Player` instances for evaluation/training contexts
    - Tracks win rates and can adapt curriculum over time

3) `WorkerOpponentFactory`
    - Worker-process battle wiring helper
    - Creates paired `BatchInferencePlayer` instances used by mp workers
    - Reconfigures opponents each batch (self-play/BC/max-damage)
    - Handles per-batch team randomization and battle-state cleanup

In short: `OpponentPool` is global/curriculum-focused, while
`WorkerOpponentFactory` is local/worker-loop-focused.
"""

import json
import os
import queue
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import Player
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.evaluate import load_model
from elitefurretai.rl.model_io import build_model_from_config, load_model_from_checkpoint
from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer, RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel
from elitefurretai.supervised.behavior_clone_player import BCPlayer


class ExploiterRegistry:
    """Persistent index of exploiter checkpoints and metadata.

    This class only manages bookkeeping (IDs, filepaths, team strings,
    performance metadata). It intentionally does not build models or players.
    """

    def __init__(self, registry_path: str = "data/models/exploiter_registry.json"):
        self.registry_path = registry_path

        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.exploiters: List[Dict[str, Any]] = data.get("exploiters", [])
        else:
            self.exploiters = []

    def save(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump({"exploiters": self.exploiters}, f, indent=2)

    def add_exploiter(self, exploiter_info: Dict[str, Any]):
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
        return [exp for exp in self.exploiters if exp.get("win_rate", 0) >= min_win_rate]

    def get_exploiter_by_id(self, exploiter_id: str) -> Optional[Dict[str, Any]]:
        for exp in self.exploiters:
            if exp["id"] == exploiter_id:
                return exp
        return None


class OpponentPool:
    """Main-process opponent sampler and model cache.

    Responsibilities:
    - Own the curriculum distribution over opponent types
    - Resolve sampled opponent types into concrete `Player` instances
    - Cache expensive model artifacts (BC/exploiter/ghost models)
    - Track win rates per opponent type and optionally adapt curriculum

    Notably different from `WorkerOpponentFactory`:
    - `OpponentPool` is global and policy-level (sampling + adaptation)
    - `WorkerOpponentFactory` is per-worker and execution-level
    """

    SELF_PLAY = "self_play"
    BC_PLAYER = "bc_player"
    EXPLOITERS = "exploiters"
    GHOSTS = "ghosts"
    MAX_DAMAGE = "max_damage"

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
        self.main_model = main_model
        self.device = device
        self.battle_format = battle_format
        self.max_past_models = max_past_models

        self.curriculum = curriculum or {
            self.SELF_PLAY: 0.40,
            self.BC_PLAYER: 0.20,
            self.EXPLOITERS: 0.20,
            self.GHOSTS: 0.20,
            self.MAX_DAMAGE: 0.0,
        }

        total = sum(self.curriculum.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Curriculum weights must sum to 1.0, got {total}")

        self.bc_player_config = None
        self._cached_bc_model: Optional[Any] = None
        self._cached_bc_embedder: Optional[Any] = None
        self._cached_bc_config: Optional[Dict] = None

        if bc_teampreview_path and bc_action_path and bc_win_path:
            self.bc_player_config = {
                "teampreview": bc_teampreview_path,
                "action": bc_action_path,
                "win": bc_win_path,
            }
            self._preload_bc_model()

        self.exploiter_registry = ExploiterRegistry(exploiter_registry_path)
        self.loaded_exploiters: Dict[str, RNaDAgent] = {}

        self.past_models_dir = past_models_dir
        os.makedirs(past_models_dir, exist_ok=True)
        self.past_models: List[Tuple[int, str]] = []
        self._load_past_models()

        self.win_rates: Dict[str, List[float]] = {
            self.SELF_PLAY: [],
            self.BC_PLAYER: [],
            self.EXPLOITERS: [],
            self.GHOSTS: [],
            self.MAX_DAMAGE: [],
        }

    def _preload_bc_model(self):
        if not self.bc_player_config:
            return

        from elitefurretai.etl.embedder import Embedder
        from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

        filepath = self.bc_player_config["teampreview"]
        print(f"[OpponentPool] Pre-loading BC model from {filepath}...")

        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint["config"]

        embedder = Embedder(
            format="gen9vgc2023regc",
            feature_set=config.get("embedder_feature_set", "raw"),
            omniscient=False,
        )

        model = FlexibleThreeHeadedModel(
            input_size=embedder.embedding_size,
            early_layers=config["early_layers"],
            late_layers=config["late_layers"],
            lstm_layers=config.get("lstm_layers", 2),
            lstm_hidden_size=config.get("lstm_hidden_size", 512),
            dropout=config.get("dropout", 0.1),
            gated_residuals=config.get("gated_residuals", False),
            early_attention_heads=config.get("early_attention_heads", 8),
            late_attention_heads=config.get("late_attention_heads", 8),
            use_grouped_encoder=config.get("use_grouped_encoder", False),
            group_sizes=(
                embedder.group_embedding_sizes
                if config.get("use_grouped_encoder", False)
                else None
            ),
            grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
            grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
            pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
            teampreview_head_layers=config.get("teampreview_head_layers", []),
            teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
            teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
            turn_head_layers=config.get("turn_head_layers", []),
            num_actions=2025,
            num_teampreview_actions=90,
            max_seq_len=config.get("max_seq_len", 17),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._cached_bc_model = model
        self._cached_bc_embedder = embedder
        self._cached_bc_config = config

        print("[OpponentPool] BC model cached (saves ~500MB per BCPlayer instance)")

    def _load_past_models(self):
        if not os.path.exists(self.past_models_dir):
            return

        files = os.listdir(self.past_models_dir)
        model_files = [f for f in files if f.endswith(".pt")]

        models = []
        for f in model_files:
            try:
                step = int(f.split("_step_")[1].split(".pt")[0])
                models.append((step, os.path.join(self.past_models_dir, f)))
            except ValueError:
                continue

        models.sort(key=lambda x: x[0], reverse=True)
        self.past_models = models[: self.max_past_models]

    def _load_exploiter_model(self, exploiter_info: Dict[str, Any]) -> RNaDAgent:
        exploiter_id = exploiter_info["id"]
        if exploiter_id in self.loaded_exploiters:
            return self.loaded_exploiters[exploiter_id]

        model = load_model(exploiter_info["filepath"], self.device)
        self.loaded_exploiters[exploiter_id] = model
        return model

    def sample_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
        worker_id: Optional[int] = None,
    ) -> Player:
        """Sample and instantiate one opponent `Player` from curriculum.

        Flow:
        1) Sample opponent type from `self.curriculum`
        2) Dispatch to the corresponding builder method
        3) Return a concrete `Player` implementation
        """
        roll = np.random.rand()
        cumsum = 0.0
        opponent_type = None

        for opp_type, weight in self.curriculum.items():
            cumsum += weight
            if roll < cumsum:
                opponent_type = opp_type
                break

        if opponent_type is None:
            opponent_type = self.SELF_PLAY

        if opponent_type == self.SELF_PLAY:
            return self._create_self_play_opponent(player_config, server_config, team, worker_id)
        elif opponent_type == self.BC_PLAYER:
            return self._create_bc_opponent(player_config, server_config, team)
        elif opponent_type == self.EXPLOITERS:
            return self._create_exploiter_opponent(player_config, server_config, team, worker_id)
        elif opponent_type == self.GHOSTS:
            return self._create_past_opponent(player_config, server_config, team, worker_id)
        elif opponent_type == self.MAX_DAMAGE:
            return self._create_max_damage_opponent(player_config, server_config, team)
        else:
            return self._create_self_play_opponent(player_config, server_config, team, worker_id)

    def _create_max_damage_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        new_config = AccountConfiguration(
            f"MaxDmg_{player_config.username}", player_config.password
        )
        return MaxDamagePlayer(
            battle_format=self.battle_format,
            account_configuration=new_config,
            server_configuration=server_config,
            team=team,
        )

    def _create_self_play_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
        worker_id: Optional[int] = None,
    ) -> Player:
        new_config = AccountConfiguration(
            f"Self_{player_config.username}", player_config.password
        )
        return BatchInferencePlayer(
            model=self.main_model,
            device=self.device,
            batch_size=16,
            account_configuration=new_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=team,
            worker_id=worker_id,
        )

    def _create_bc_opponent(self, player_config, server_config, team: str) -> Player:
        if self.bc_player_config is None or self._cached_bc_model is None:
            return self._create_self_play_opponent(player_config, server_config, team)

        new_config = AccountConfiguration(
            f"BC_{player_config.username}", player_config.password
        )

        bc_player = BCPlayer.__new__(BCPlayer)
        Player.__init__(
            bc_player,
            account_configuration=new_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            team=team,
            accept_open_team_sheet=True,
        )

        bc_player._battle_format = self.battle_format
        bc_player._probabilistic = True
        bc_player._trajectories = {}
        bc_player._device = self.device
        bc_player._verbose = False

        bc_player.teampreview_model = self._cached_bc_model
        bc_player.action_model = self._cached_bc_model
        bc_player.win_model = self._cached_bc_model
        bc_player.teampreview_embedder = self._cached_bc_embedder  # type: ignore[assignment]
        bc_player.action_embedder = self._cached_bc_embedder  # type: ignore[assignment]
        bc_player.win_embedder = self._cached_bc_embedder  # type: ignore[assignment]
        bc_player.teampreview_config = self._cached_bc_config  # type: ignore[assignment]
        bc_player.action_config = self._cached_bc_config  # type: ignore[assignment]
        bc_player.win_config = self._cached_bc_config  # type: ignore[assignment]

        bc_player._last_message_error = {}
        bc_player._last_message = {}
        bc_player._last_win_advantage = {}
        bc_player._win_advantage_threshold = 0.5

        return bc_player

    def _create_exploiter_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
        worker_id: Optional[int] = None,
    ) -> Player:
        active_exploiters = self.exploiter_registry.get_active_exploiters()

        if not active_exploiters:
            return self._create_self_play_opponent(player_config, server_config, team, worker_id)

        exploiter_info = random.choice(active_exploiters)
        exploiter_model = self._load_exploiter_model(exploiter_info)

        exploiter_team = exploiter_info.get("team", team)
        exploiter_id = exploiter_info.get("id", "Unknown")
        new_config = AccountConfiguration(
            f"Exp_{exploiter_id}_{player_config.username}", player_config.password
        )

        return BatchInferencePlayer(
            model=exploiter_model,
            device=self.device,
            batch_size=16,
            account_configuration=new_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=exploiter_team,
            worker_id=worker_id,
        )

    def _create_past_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
        worker_id: Optional[int] = None,
    ) -> Player:
        if not self.past_models:
            return self._create_self_play_opponent(player_config, server_config, team, worker_id)

        step, filepath = self.past_models[np.random.randint(len(self.past_models))]
        past_model = load_model(filepath, self.device)

        new_config = AccountConfiguration(
            f"Ghost_{step}_{player_config.username}", player_config.password
        )

        return BatchInferencePlayer(
            model=past_model,
            device=self.device,
            batch_size=16,
            account_configuration=new_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=team,
            worker_id=worker_id,
        )

    def add_past_model(self, step: int, filepath: str):
        self.past_models.append((step, filepath))
        self.past_models.sort(key=lambda x: x[0], reverse=True)
        self.past_models = self.past_models[: self.max_past_models]

    def update_win_rate(self, opponent_type: str, won: bool):
        if opponent_type in self.win_rates:
            self.win_rates[opponent_type].append(1.0 if won else 0.0)

    def get_win_rate_stats(self, window: int = 100) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for opp_type, results in self.win_rates.items():
            if results:
                recent = results[-window:]
                stats[opp_type] = float(np.mean(recent))
            else:
                stats[opp_type] = 0.0
        return stats

    def update_curriculum(self, adaptive: bool = True):
        """Adapt curriculum weights from recent matchup performance.

        Current heuristic:
        - If exploiters are too easy, reduce exploiter weight
        - If BC baseline is too hard, increase BC weight
        - Renormalize to a valid probability distribution
        """
        if not adaptive:
            return

        stats = self.get_win_rate_stats()
        new_curriculum = self.curriculum.copy()

        if (
            stats.get(self.EXPLOITERS, 0) > 0.70
            and self.exploiter_registry.get_active_exploiters()
        ):
            new_curriculum[self.EXPLOITERS] = max(0.10, new_curriculum[self.EXPLOITERS] - 0.05)
            new_curriculum[self.SELF_PLAY] += 0.05

        if stats.get(self.BC_PLAYER, 0) < 0.40 and self.bc_player_config:
            new_curriculum[self.BC_PLAYER] = min(0.40, new_curriculum[self.BC_PLAYER] + 0.05)
            new_curriculum[self.SELF_PLAY] = max(0.20, new_curriculum[self.SELF_PLAY] - 0.05)

        total = sum(new_curriculum.values())
        for key in new_curriculum:
            new_curriculum[key] /= total

        self.curriculum = new_curriculum


class WorkerOpponentFactory:
    """Worker-local helper for battle pair construction and batch reconfiguration.

    This class is intentionally simpler than `OpponentPool`: it does not track
    global win-rate stats or persist metadata. Instead, it focuses on the hot
    path inside worker loops:
    - create player/opponent pairs once
    - swap opponent policy per batch
    - randomize teams per batch
    - reset battle state to avoid memory growth
    """

    SELF_PLAY = "self_play"
    BC_PLAYER = "bc_player"
    EXPLOITERS = "exploiters"
    GHOSTS = "ghosts"
    MAX_DAMAGE = "max_damage"

    def __init__(
        self,
        team_repo: TeamRepo,
        battle_format: str,
        team_subdirectory: Optional[str],
        server_config: ServerConfiguration,
        main_agent: RNaDAgent,
        bc_agent: Optional[RNaDAgent],
        curriculum: Optional[Dict[str, float]],
        embedder: Embedder,
        worker_id: int,
        run_id: str,
        device: str,
        batch_size: int = 16,
        batch_timeout: float = 0.01,
        max_battle_steps: int = 40,
    ):
        self.team_repo = team_repo
        self.battle_format = battle_format
        self.team_subdirectory = team_subdirectory
        self.server_config = server_config
        self.main_agent = main_agent
        self.bc_agent = bc_agent
        self.curriculum = curriculum or {self.SELF_PLAY: 1.0}
        self.embedder = embedder
        self.worker_id = worker_id
        self.run_id = run_id
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_battle_steps = max_battle_steps

        self.players: List[BatchInferencePlayer] = []
        self.opponents: List[BatchInferencePlayer] = []
        self.max_damage_opponents: List[MaxDamagePlayer] = []

    def sample_team(self) -> str:
        return self.team_repo.sample_team(
            self.battle_format,
            subdirectory=self.team_subdirectory,
        )

    def create_player_pairs(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[List[BatchInferencePlayer], List[BatchInferencePlayer]]:
        """Create mirrored player/opponent `BatchInferencePlayer` pairs.

        Players collect trajectories (`trajectory_queue` attached); opponents do
        not collect trajectories. Both are reused across batches.
        """
        self.players = []
        self.opponents = []

        for i in range(num_pairs):
            player = BatchInferencePlayer(
                model=self.main_agent,
                device=self.device,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                account_configuration=AccountConfiguration(
                    f"MP{self.worker_id}SELF{i}R{self.run_id}", None
                ),
                server_configuration=self.server_config,
                trajectory_queue=local_traj_queue,
                battle_format=self.battle_format,
                team=self.sample_team(),
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
                opponent_type=self.SELF_PLAY,
            )
            self.players.append(player)

            opponent = BatchInferencePlayer(
                model=self.main_agent,
                device=self.device,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                account_configuration=AccountConfiguration(
                    f"MP{self.worker_id}OPP{i}R{self.run_id}", None
                ),
                server_configuration=self.server_config,
                trajectory_queue=None,
                battle_format=self.battle_format,
                team=self.sample_team(),
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
            )
            self.opponents.append(opponent)

        return self.players, self.opponents

    def create_max_damage_opponents(self, num_opponents: int) -> List[MaxDamagePlayer]:
        self.max_damage_opponents = []

        if self.curriculum.get(self.MAX_DAMAGE, 0) > 0:
            for i in range(num_opponents):
                md_opponent = MaxDamagePlayer(
                    battle_format=self.battle_format,
                    account_configuration=AccountConfiguration(
                        f"MP{self.worker_id}MD{i}R{self.run_id}", None
                    ),
                    server_configuration=self.server_config,
                    team=self.sample_team(),
                )
                self.max_damage_opponents.append(md_opponent)

        return self.max_damage_opponents

    def sample_opponent_type(self) -> str:
        rand = random.random()
        cumulative = 0.0

        for opp_type, prob in self.curriculum.items():
            cumulative += prob
            if rand < cumulative:
                if opp_type in (self.GHOSTS, self.EXPLOITERS):
                    return self.SELF_PLAY
                return opp_type

        return self.SELF_PLAY

    def configure_opponent_for_batch(
        self,
        player: BatchInferencePlayer,
        opponent: BatchInferencePlayer,
    ) -> str:
        """Configure one pair for the next batch and return chosen type.

        - `max_damage` is handled by separate MaxDamagePlayer instances
        - `bc_player` swaps opponent.model to the frozen BC agent
        - fallback is self-play against `main_agent`
        """
        selected_type = self.sample_opponent_type()

        if selected_type == self.MAX_DAMAGE:
            pass
        elif selected_type == self.BC_PLAYER and self.bc_agent is not None:
            opponent.model = self.bc_agent
        else:
            if selected_type not in (self.SELF_PLAY, self.MAX_DAMAGE):
                selected_type = self.SELF_PLAY
            opponent.model = self.main_agent

        player.opponent_type = selected_type
        return selected_type

    def randomize_all_teams(self) -> None:
        """Resample teams for all participants before the next batch."""
        for player in self.players:
            player._team = ConstantTeambuilder(self.sample_team())

        for opponent in self.opponents:
            opponent._team = ConstantTeambuilder(self.sample_team())

        for md_opp in self.max_damage_opponents:
            md_opp._team = ConstantTeambuilder(self.sample_team())

    def start_inference_loops(self) -> None:
        for player in self.players + self.opponents:
            player.start_inference_loop()

    def reset_all_battles(self) -> None:
        """Clear per-battle runtime state after a batch completes."""
        for player in self.players:
            player.reset_battles()
            player.clear_completed_trajectories()
            player.hidden_states.clear()
            player.current_trajectories.clear()

        for opponent in self.opponents:
            opponent.reset_battles()
            opponent.clear_completed_trajectories()
            opponent.hidden_states.clear()

        for md_opp in self.max_damage_opponents:
            md_opp.reset_battles()


__all__ = [
    "OpponentPool",
    "ExploiterRegistry",
    "WorkerOpponentFactory",
    "build_model_from_config",
    "load_model_from_checkpoint",
    "FlexibleThreeHeadedModel",
]
