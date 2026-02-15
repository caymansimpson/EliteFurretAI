"""
WorkerOpponentFactory: Encapsulates opponent creation and sampling logic for worker processes.

This module extracts the opponent logic from train.py's mp_worker_process to:
1. Consolidate opponent creation in one place
2. Enable easy team randomization for all opponent types
3. Simplify the worker process code
"""

import queue
import random
from typing import Any, Dict, List, Optional, Tuple

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.model_builder import (
    build_model_from_config as _build_model_from_config,
)
from elitefurretai.rl.model_builder import (
    load_model_from_checkpoint as _load_model_from_checkpoint,
)
from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer, RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel


def build_model_from_config(
    model_config: Dict[str, Any],
    embedder: Embedder,
    device: str,
    state_dict: Optional[Dict[str, Any]] = None,
) -> FlexibleThreeHeadedModel:
    return _build_model_from_config(model_config, embedder, device, state_dict)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> Tuple[FlexibleThreeHeadedModel, Embedder, Dict[str, Any]]:
    return _load_model_from_checkpoint(checkpoint_path, device, embedder)


class WorkerOpponentFactory:
    """
    Manages opponent creation and sampling within a worker process.

    Responsibilities:
    - Create player/opponent pairs with proper account configurations
    - Sample opponent types based on curriculum probabilities
    - Randomize teams for all players before each battle batch
    - Manage MaxDamagePlayer instances

    This class encapsulates logic previously scattered in mp_worker_process(),
    making the worker code cleaner and opponent management more consistent.
    """

    # Opponent type constants (matching OpponentPool)
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
        """
        Initialize the factory.

        Args:
            team_repo: TeamRepo for sampling teams
            battle_format: Pokemon Showdown format string
            team_subdirectory: Subdirectory within format for teams
            server_config: Server configuration for battles
            main_agent: Main RNaDAgent being trained
            bc_agent: Optional frozen BC agent for curriculum
            curriculum: Dict of opponent type -> sampling probability
            embedder: Shared embedder instance
            worker_id: Unique worker identifier
            run_id: Unique run identifier
            device: Device for inference
            batch_size: Batch size for BatchInferencePlayer
            batch_timeout: Batch timeout for BatchInferencePlayer
            max_battle_steps: Max steps before forfeiting
        """
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

        # Storage for created players
        self.players: List[BatchInferencePlayer] = []
        self.opponents: List[BatchInferencePlayer] = []
        self.max_damage_opponents: List[MaxDamagePlayer] = []

    def sample_team(self) -> str:
        """Sample a random team from the team repository."""
        return self.team_repo.sample_team(
            self.battle_format,
            subdirectory=self.team_subdirectory
        )

    def create_player_pairs(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[List[BatchInferencePlayer], List[BatchInferencePlayer]]:
        """
        Create player/opponent pairs for battles.

        Args:
            num_pairs: Number of player/opponent pairs to create
            local_traj_queue: Queue for collecting trajectories (players only)

        Returns:
            Tuple of (players list, opponents list)
        """
        self.players = []
        self.opponents = []

        for i in range(num_pairs):
            # Create player (collects trajectories)
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

            # Create opponent (no trajectory collection)
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
        """
        Create MaxDamagePlayer opponents if curriculum includes them.

        Args:
            num_opponents: Number of MaxDamagePlayer instances to create

        Returns:
            List of MaxDamagePlayer instances
        """
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
        """
        Sample an opponent type based on curriculum probabilities.

        Returns:
            String identifying the opponent type
        """
        rand = random.random()
        cumulative = 0.0

        for opp_type, prob in self.curriculum.items():
            cumulative += prob
            if rand < cumulative:
                # Handle unsupported types by falling back to self_play
                if opp_type in (self.GHOSTS, self.EXPLOITERS):
                    # These require broadcasting weights from main process
                    # Fall back to self_play until implemented
                    return self.SELF_PLAY
                return opp_type

        return self.SELF_PLAY

    def configure_opponent_for_batch(
        self,
        player: BatchInferencePlayer,
        opponent: BatchInferencePlayer,
    ) -> str:
        """
        Configure an opponent for a battle batch based on curriculum sampling.

        Updates the opponent's model reference and sets the player's opponent_type
        for trajectory metadata.

        Args:
            player: The player (whose opponent_type will be set)
            opponent: The opponent (whose model may be swapped)

        Returns:
            The selected opponent type string
        """
        selected_type = self.sample_opponent_type()

        if selected_type == self.MAX_DAMAGE:
            # MaxDamagePlayer is handled separately in battle loop
            pass
        elif selected_type == self.BC_PLAYER and self.bc_agent is not None:
            opponent.model = self.bc_agent
        else:
            # self_play or fallback
            if selected_type not in (self.SELF_PLAY, self.MAX_DAMAGE):
                selected_type = self.SELF_PLAY
            opponent.model = self.main_agent

        player.opponent_type = selected_type
        return selected_type

    def randomize_all_teams(self) -> None:
        """
        Randomize teams for ALL players before a battle batch.

        This improves generalization by exposing the model to diverse team
        matchups. Previously only MaxDamagePlayer got fresh teams each batch.
        """
        # Randomize player teams
        for player in self.players:
            player._team = ConstantTeambuilder(self.sample_team())

        # Randomize BatchInferencePlayer opponent teams
        for opponent in self.opponents:
            opponent._team = ConstantTeambuilder(self.sample_team())

        # Randomize MaxDamagePlayer teams
        for md_opp in self.max_damage_opponents:
            md_opp._team = ConstantTeambuilder(self.sample_team())

    def start_inference_loops(self) -> None:
        """Start inference loops for all BatchInferencePlayer instances."""
        for player in self.players + self.opponents:
            player.start_inference_loop()

    def reset_all_battles(self) -> None:
        """Reset battle state and clean up memory for all players."""
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
