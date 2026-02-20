"""Opponent management for RL training.

This module contains two related but intentionally different classes:

1) `OpponentPool`
    - Learner/main-process opponent manager
    - Samples opponent *types* using curriculum probabilities
    - Lazily loads/caches models (BC, exploiters, ghost checkpoints)
    - Creates concrete `Player` instances for evaluation/training contexts
    - Tracks win rates and can adapt curriculum over time

2) `WorkerOpponentFactory`
    - Worker-process battle wiring helper
    - Creates paired `BatchInferencePlayer` instances used by mp workers
    - Reconfigures opponents each batch (self-play/BC/exploiter/ghost/max-damage)
    - Handles per-batch team randomization and battle-state cleanup

In short: `OpponentPool` is global/curriculum-focused, while
`WorkerOpponentFactory` is local/worker-loop-focused.
"""

import importlib
import importlib.util
import logging
import os
import queue
import random
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.model_io import (
    build_model_from_config,
    is_checkpoint_compatible_with_model_config,
    load_agent_from_checkpoint,
    load_model_from_checkpoint,
)
from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer, RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel
from elitefurretai.supervised.behavior_clone_player import BCPlayer

logger = logging.getLogger(__name__)

SimpleHeuristicBaselineCls: Optional[type]
try:
    _baselines_module = importlib.import_module("poke_env.player.baselines")
    SimpleHeuristicBaselineCls = cast(
        Optional[type],
        getattr(_baselines_module, "SimpleHeuristicsPlayer", None),
    )
except Exception:
    SimpleHeuristicBaselineCls = None


_VGC_BENCH_POLICY_CACHE: Dict[Tuple[str, str], Any] = {}


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_vgc_bench_root() -> Optional[Path]:
    spec = importlib.util.find_spec("vgc_bench")
    if spec is None or not spec.submodule_search_locations:
        return None

    package_path = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_path.parent


def _create_vgc_bench_player(
    device: str,
    player_config: AccountConfiguration,
    server_config: ServerConfiguration,
    team: str,
    battle_format: str='gen9vgc2024regg',
    checkpoint_path: str = 'data/models/vgc-bench-sb3-model.zip',
    accept_open_team_sheet: bool = True,
) -> Player:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"vgc-bench checkpoint not found: {checkpoint_path}")

    cache_key = (checkpoint_path, device)

    ppo_module = importlib.import_module("stable_baselines3")
    ppo_cls = getattr(ppo_module, "PPO")

    vgc_bench_root = _resolve_vgc_bench_root()
    if vgc_bench_root is None:
        raise ModuleNotFoundError("Could not resolve vgc_bench package path")

    with _temporary_cwd(vgc_bench_root):
        policy_player_module = importlib.import_module("vgc_bench.src.policy_player")
        policy_player_cls = getattr(policy_player_module, "PolicyPlayer")

        policy = _VGC_BENCH_POLICY_CACHE.get(cache_key)
        if policy is None:
            policy = ppo_cls.load(cache_key[0], device=device).policy
            _VGC_BENCH_POLICY_CACHE[cache_key] = policy

    player = policy_player_cls(
        policy=policy,
        battle_format=battle_format,
        account_configuration=player_config,
        server_configuration=server_config,
        accept_open_team_sheet=accept_open_team_sheet,
        team=team,
    )
    return cast(Player, player)


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
    RANDOM_BASELINE = "random_baseline"
    MAX_BASE_POWER_BASELINE = "max_base_power_baseline"
    SIMPLE_HEURISTIC_BASELINE = "simple_heuristic_baseline"
    VGC_BENCH_BASELINE = "vgc_bench_baseline"

    def __init__(
        self,
        main_model: RNaDAgent,
        device: str,
        battle_format: str = "gen9vgc2023regc",
        bc_teampreview_path: Optional[str] = None,
        bc_action_path: Optional[str] = None,
        bc_win_path: Optional[str] = None,
        exploiter_models_dir: str = "data/models/exploiters",
        past_models_dir: str = "data/models/ghosts",
        vgc_bench_checkpoint_path: Optional[str] = None,
        max_past_models: int = 10,
        tracking_window: int = 100,
        curriculum: Optional[Dict[str, float]] = None,
    ):
        self.main_model = main_model
        self.device = device
        self.battle_format = battle_format
        self.max_past_models = max_past_models
        self.vgc_bench_checkpoint_path = vgc_bench_checkpoint_path
        self.tracking_window = tracking_window

        self.curriculum = curriculum or {
            self.SELF_PLAY: 0.40,
            self.BC_PLAYER: 0.20,
            self.EXPLOITERS: 0.20,
            self.GHOSTS: 0.20,
            self.MAX_DAMAGE: 0.0,
            self.RANDOM_BASELINE: 0.0,
            self.MAX_BASE_POWER_BASELINE: 0.0,
            self.SIMPLE_HEURISTIC_BASELINE: 0.0,
            self.VGC_BENCH_BASELINE: 0.0,
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

        self.exploiter_models_dir = exploiter_models_dir
        os.makedirs(self.exploiter_models_dir, exist_ok=True)
        self.exploiter_models: List[Tuple[float, str]] = []
        self.loaded_exploiters: Dict[str, RNaDAgent] = {}
        self._load_exploiter_models()

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
            self.RANDOM_BASELINE: [],
            self.MAX_BASE_POWER_BASELINE: [],
            self.SIMPLE_HEURISTIC_BASELINE: [],
            self.VGC_BENCH_BASELINE: [],
        }
        self.win_rate_tracking: Dict[str, deque[float]] = {
            opp_type: deque(maxlen=self.tracking_window)
            for opp_type in self.win_rates
        }
        self.battle_length_tracking: Dict[str, deque[int]] = {
            opp_type: deque(maxlen=self.tracking_window)
            for opp_type in self.win_rates
        }
        self.total_battles_tracked = 0
        self.total_forfeits_tracked = 0

    def _ensure_tracking_key(self, opponent_type: str) -> None:
        if opponent_type not in self.win_rate_tracking:
            self.win_rate_tracking[opponent_type] = deque(maxlen=self.tracking_window)
        if opponent_type not in self.battle_length_tracking:
            self.battle_length_tracking[opponent_type] = deque(maxlen=self.tracking_window)
        if opponent_type not in self.win_rates:
            self.win_rates[opponent_type] = []

    def _preload_bc_model(self):
        if not self.bc_player_config:
            return

        # TODO fix lazy loading
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

    def _list_model_checkpoints(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        return [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.endswith(".pt")
        ]

    def _load_exploiter_models(self) -> None:
        files = self._list_model_checkpoints(self.exploiter_models_dir)
        models = [
            (os.path.getmtime(filepath), filepath)
            for filepath in files
            if os.path.isfile(filepath)
        ]
        models.sort(key=lambda item: item[0], reverse=True)
        self.exploiter_models = models[: self.max_past_models]

    def _load_past_models(self) -> None:
        files = self._list_model_checkpoints(self.past_models_dir)
        models: List[Tuple[int, str]] = []
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (ValueError, IndexError):
                step = int(os.path.getmtime(filepath))
            models.append((step, filepath))

        models.sort(key=lambda item: item[0], reverse=True)
        self.past_models = models[: self.max_past_models]

    def _load_exploiter_model(self, filepath: str) -> RNaDAgent:
        if filepath in self.loaded_exploiters:
            return self.loaded_exploiters[filepath]

        model = load_agent_from_checkpoint(filepath, self.device)
        self.loaded_exploiters[filepath] = model
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
        elif opponent_type == self.RANDOM_BASELINE:
            return self._create_random_baseline_opponent(player_config, server_config, team)
        elif opponent_type == self.MAX_BASE_POWER_BASELINE:
            return self._create_max_base_power_baseline_opponent(player_config, server_config, team)
        elif opponent_type == self.SIMPLE_HEURISTIC_BASELINE:
            return self._create_simple_heuristic_baseline_opponent(player_config, server_config, team)
        elif opponent_type == self.VGC_BENCH_BASELINE:
            return self._create_vgc_bench_baseline_opponent(player_config, server_config, team)
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

    def _create_random_baseline_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        return RandomPlayer(
            battle_format=self.battle_format,
            account_configuration=AccountConfiguration(
                f"Rnd_{player_config.username}", player_config.password
            ),
            server_configuration=server_config,
            team=team,
        )

    def _create_max_base_power_baseline_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        return MaxBasePowerPlayer(
            battle_format=self.battle_format,
            account_configuration=AccountConfiguration(
                f"MBP_{player_config.username}", player_config.password
            ),
            server_configuration=server_config,
            team=team,
        )

    def _create_simple_heuristic_baseline_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        if SimpleHeuristicBaselineCls is None:
            return self._create_max_base_power_baseline_opponent(
                player_config, server_config, team
            )
        return SimpleHeuristicBaselineCls(
            battle_format=self.battle_format,
            account_configuration=AccountConfiguration(
                f"SHP_{player_config.username}", player_config.password
            ),
            server_configuration=server_config,
            team=team,
        )

    def _create_vgc_bench_baseline_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
    ) -> Player:
        return _create_vgc_bench_player(
            device=self.device,
            battle_format=self.battle_format,
            player_config=AccountConfiguration(
                f"VGCBench_{player_config.username}", player_config.password
            ),
            server_config=server_config,
            team=team,
            checkpoint_path=self.vgc_bench_checkpoint_path or "data/models/vgc-bench-sb3-model.zip",
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
        self._load_exploiter_models()
        if not self.exploiter_models:
            return self._create_self_play_opponent(player_config, server_config, team, worker_id)

        _, exploiter_path = random.choice(self.exploiter_models)
        exploiter_model = self._load_exploiter_model(exploiter_path)

        exploiter_team = team
        exploiter_id = os.path.splitext(os.path.basename(exploiter_path))[0]
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
        past_model = load_agent_from_checkpoint(filepath, self.device)

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

    def record_battle_result(
        self,
        opponent_type: str,
        won: bool,
        battle_length: int = 0,
        forfeited: bool = False,
    ) -> None:
        self._ensure_tracking_key(opponent_type)

        win_value = 1.0 if won else 0.0
        self.win_rates[opponent_type].append(win_value)
        self.win_rate_tracking[opponent_type].append(win_value)

        if battle_length > 0:
            self.battle_length_tracking[opponent_type].append(battle_length)

        self.total_battles_tracked += 1
        if forfeited:
            self.total_forfeits_tracked += 1

    def get_win_rate_stats(self, window: int = 100) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for opp_type, results in self.win_rate_tracking.items():
            if results:
                recent = list(results)[-window:]
                stats[opp_type] = float(np.mean(recent))
            else:
                stats[opp_type] = 0.0
        return stats

    def get_battle_length_stats(self, window: int = 100) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for opp_type, lengths in self.battle_length_tracking.items():
            if lengths:
                recent = list(lengths)[-window:]
                stats[opp_type] = float(np.mean(recent))
            else:
                stats[opp_type] = 0.0
        return stats

    def get_training_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        used_window = window or self.tracking_window

        for opp_type, win_rate in self.get_win_rate_stats(used_window).items():
            metrics[f"win_rate_{opp_type}"] = win_rate

        length_stats = self.get_battle_length_stats(used_window)
        for opp_type, avg_len in length_stats.items():
            if len(self.battle_length_tracking.get(opp_type, [])) > 0:
                metrics[f"avg_battle_length_{opp_type}"] = avg_len

        all_lengths = [
            length
            for lengths in self.battle_length_tracking.values()
            for length in lengths
        ]
        if all_lengths:
            metrics["avg_battle_length_overall"] = float(np.mean(all_lengths))

        if self.total_battles_tracked > 0:
            metrics["forfeit_rate"] = self.total_forfeits_tracked / self.total_battles_tracked

        return metrics

    def update_curriculum(self, adaptive: bool = True):
        """Adapt curriculum weights from recent matchup performance.

        Current heuristic:
        - If exploiters are too easy, reduce exploiter weight
        - If BC baseline is too hard, increase BC weight
        - Renormalize to a valid probability distribution
        """
        if not adaptive:
            return

        self._load_exploiter_models()
        stats = self.get_win_rate_stats()
        new_curriculum = self.curriculum.copy()

        if (
            stats.get(self.EXPLOITERS, 0) > 0.70
            and len(self.exploiter_models) > 0
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
    RANDOM_BASELINE = "random_baseline"
    MAX_BASE_POWER_BASELINE = "max_base_power_baseline"
    SIMPLE_HEURISTIC_BASELINE = "simple_heuristic_baseline"
    VGC_BENCH_BASELINE = "vgc_bench_baseline"

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
        exploiter_models_dir: Optional[str] = None,
        max_exploiter_models: int = 10,
        max_loaded_exploiter_models: int = 2,
        ghost_models_dir: Optional[str] = None,
        max_ghost_models: int = 10,
        max_loaded_ghost_models: int = 2,
        vgc_bench_checkpoint_path: Optional[str] = None,
        external_vgcbench_usernames: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
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
        self.exploiter_models_dir = exploiter_models_dir
        self.max_exploiter_models = max_exploiter_models
        self.max_loaded_exploiter_models = max_loaded_exploiter_models
        self.ghost_models_dir = ghost_models_dir
        self.max_ghost_models = max_ghost_models
        self.max_loaded_ghost_models = max_loaded_ghost_models
        self.vgc_bench_checkpoint_path = vgc_bench_checkpoint_path
        self.model_config = model_config
        self.external_vgcbench_usernames = [
            username.strip()
            for username in (external_vgcbench_usernames or [])
            if username and username.strip()
        ]

        self.players: List[BatchInferencePlayer] = []
        self.opponents: List[BatchInferencePlayer] = []
        self.max_damage_opponents: List[MaxDamagePlayer] = []
        self.random_baseline_opponents: List[RandomPlayer] = []
        self.max_base_power_baseline_opponents: List[MaxBasePowerPlayer] = []
        self.simple_heuristic_baseline_opponents: List[Player] = []
        self.vgc_bench_baseline_opponents: List[Player] = []
        self.active_exploiters: List[Tuple[float, str]] = []
        self.loaded_exploiters: Dict[str, RNaDAgent] = {}
        self._exploiter_cache_order: List[str] = []
        self.past_models: List[Tuple[int, str]] = []
        self.loaded_ghosts: Dict[str, RNaDAgent] = {}
        self._ghost_cache_order: List[str] = []
        self._batch_count = 0
        self._load_exploiter_models()
        self._load_past_models()

    def _list_model_checkpoints(self, directory: Optional[str]) -> List[str]:
        if not directory or not os.path.exists(directory):
            return []
        model_paths = [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if filename.endswith(".pt")
        ]
        if not self.model_config:
            return model_paths

        compatible_paths: List[str] = []
        for model_path in model_paths:
            if is_checkpoint_compatible_with_model_config(model_path, self.model_config):
                compatible_paths.append(model_path)

        return compatible_paths

    def _load_exploiter_models(self) -> None:
        files = self._list_model_checkpoints(self.exploiter_models_dir)
        if not files:
            self.active_exploiters = []
            return

        models = [
            (os.path.getmtime(filepath), filepath)
            for filepath in files
            if os.path.isfile(filepath)
        ]
        models.sort(key=lambda item: item[0], reverse=True)
        self.active_exploiters = models[: self.max_exploiter_models]

    def _refresh_exploiters_if_needed(self) -> None:
        if self._batch_count % 10 == 0:
            self._load_exploiter_models()

    def _get_cached_model(
        self,
        filepath: str,
        loaded: Dict[str, RNaDAgent],
        cache_order: List[str],
        max_loaded: int,
    ) -> RNaDAgent:
        if filepath in loaded:
            if filepath in cache_order:
                cache_order.remove(filepath)
            cache_order.append(filepath)
            return loaded[filepath]

        loaded_model = load_agent_from_checkpoint(filepath, self.device)
        loaded[filepath] = loaded_model
        cache_order.append(filepath)

        while len(cache_order) > max_loaded:
            evict_path = cache_order.pop(0)
            loaded.pop(evict_path, None)

        return loaded_model

    def _get_exploiter_agent(self) -> Optional[RNaDAgent]:
        if not self.active_exploiters:
            return None

        _, filepath = random.choice(self.active_exploiters)
        return self._get_cached_model(
            filepath,
            self.loaded_exploiters,
            self._exploiter_cache_order,
            self.max_loaded_exploiter_models,
        )

    def _load_past_models(self) -> None:
        if not self.ghost_models_dir or not os.path.exists(self.ghost_models_dir):
            self.past_models = []
            return

        model_files = self._list_model_checkpoints(self.ghost_models_dir)

        models: List[Tuple[int, str]] = []
        for filepath in model_files:
            filename = os.path.basename(filepath)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (ValueError, IndexError):
                step = int(os.path.getmtime(filepath))
            models.append((step, filepath))

        models.sort(key=lambda x: x[0], reverse=True)
        self.past_models = models[: self.max_ghost_models]

    def _refresh_past_models_if_needed(self) -> None:
        # Periodically refresh to pick up newly saved checkpoints from learner.
        if self._batch_count % 10 == 0:
            self._load_past_models()

    def _get_ghost_agent(self) -> Optional[RNaDAgent]:
        if not self.past_models:
            return None

        _, filepath = self.past_models[np.random.randint(len(self.past_models))]
        return self._get_cached_model(
            filepath,
            self.loaded_ghosts,
            self._ghost_cache_order,
            self.max_loaded_ghost_models,
        )

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

    def create_agents(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[List[BatchInferencePlayer], List[BatchInferencePlayer], List[MaxDamagePlayer]]:
        """Create and cache all worker-local battle participants.

        This is the high-level setup entrypoint used by worker startup.
        It creates training players/opponents and conditionally initializes
        max-damage opponents based on curriculum.
        """
        self.create_player_pairs(num_pairs, local_traj_queue)
        self.create_max_damage_opponents(num_pairs)
        self.create_baseline_opponents(num_pairs)
        return self.players, self.opponents, self.max_damage_opponents

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

    def create_baseline_opponents(self, num_opponents: int) -> None:
        self.random_baseline_opponents = []
        self.max_base_power_baseline_opponents = []
        self.simple_heuristic_baseline_opponents = []
        self.vgc_bench_baseline_opponents = []

        if self.curriculum.get(self.RANDOM_BASELINE, 0) > 0:
            for i in range(num_opponents):
                self.random_baseline_opponents.append(
                    RandomPlayer(
                        battle_format=self.battle_format,
                        account_configuration=AccountConfiguration(
                            f"MP{self.worker_id}RND{i}R{self.run_id}", None
                        ),
                        server_configuration=self.server_config,
                        team=self.sample_team(),
                    )
                )

        if self.curriculum.get(self.MAX_BASE_POWER_BASELINE, 0) > 0:
            for i in range(num_opponents):
                self.max_base_power_baseline_opponents.append(
                    MaxBasePowerPlayer(
                        battle_format=self.battle_format,
                        account_configuration=AccountConfiguration(
                            f"MP{self.worker_id}MBP{i}R{self.run_id}", None
                        ),
                        server_configuration=self.server_config,
                        team=self.sample_team(),
                    )
                )

        if self.curriculum.get(self.SIMPLE_HEURISTIC_BASELINE, 0) > 0:
            baseline_cls = SimpleHeuristicBaselineCls or MaxBasePowerPlayer
            for i in range(num_opponents):
                self.simple_heuristic_baseline_opponents.append(
                    baseline_cls(
                        battle_format=self.battle_format,
                        account_configuration=AccountConfiguration(
                            f"MP{self.worker_id}SHP{i}R{self.run_id}", None
                        ),
                        server_configuration=self.server_config,
                        team=self.sample_team(),
                    )
                )

        if (
            self.curriculum.get(self.VGC_BENCH_BASELINE, 0) > 0
            and not self.external_vgcbench_usernames
        ):
            for i in range(num_opponents):
                vgc_player = _create_vgc_bench_player(
                    device=self.device,
                    battle_format=self.battle_format,
                    player_config=AccountConfiguration(
                        f"MP{self.worker_id}VGB{i}R{self.run_id}", None
                    ),
                    server_config=self.server_config,
                    team=self.sample_team(),
                    checkpoint_path=self.vgc_bench_checkpoint_path or "data/models/vgc-bench-sb3-model.zip",
                )
                self.vgc_bench_baseline_opponents.append(vgc_player)

    def sample_opponent_type(self) -> str:
        rand = random.random()
        cumulative = 0.0

        for opp_type, prob in self.curriculum.items():
            cumulative += prob
            if rand < cumulative:
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
        elif selected_type == self.EXPLOITERS:
            exploiter_agent = self._get_exploiter_agent()
            if exploiter_agent is not None:
                opponent.model = exploiter_agent
            else:
                selected_type = self.SELF_PLAY
                opponent.model = self.main_agent
        elif selected_type == self.GHOSTS:
            ghost_agent = self._get_ghost_agent()
            if ghost_agent is not None:
                opponent.model = ghost_agent
            else:
                selected_type = self.SELF_PLAY
                opponent.model = self.main_agent
        else:
            if selected_type not in (
                self.SELF_PLAY,
                self.MAX_DAMAGE,
                self.RANDOM_BASELINE,
                self.MAX_BASE_POWER_BASELINE,
                self.SIMPLE_HEURISTIC_BASELINE,
                self.VGC_BENCH_BASELINE,
            ):
                selected_type = self.SELF_PLAY
            opponent.model = self.main_agent

        player.opponent_type = selected_type
        return selected_type

    # TODO: might want to bring this back into train.py -- seems like train.py should
    # decide who to battle when
    def prepare_batch_tasks(
        self,
        num_battles_per_pair: int,
    ) -> Tuple[List[Any], List[str]]:
        """Prepare all battles for the next batch according to curriculum.

        Returns:
            tasks: list of awaitables ready for asyncio.gather
            batch_opponent_types: sampled opponent type per player
        """
        self._batch_count += 1
        self._refresh_exploiters_if_needed()
        self._refresh_past_models_if_needed()
        self.randomize_all_teams()

        tasks: List[Any] = []
        batch_opponent_types: List[str] = []

        for i, player in enumerate(self.players):
            opponent = self.opponents[i]
            opp_type = self.configure_opponent_for_batch(player, opponent)
            batch_opponent_types.append(opp_type)

            if opp_type == self.VGC_BENCH_BASELINE and self.external_vgcbench_usernames:
                username = self.external_vgcbench_usernames[
                    (self._batch_count + i) % len(self.external_vgcbench_usernames)
                ]
                tasks.append(
                    player.send_challenges(
                        username,
                        num_battles_per_pair,
                    )
                )
                continue

            target_opponent: Player

            if opp_type == self.MAX_DAMAGE and self.max_damage_opponents:
                target_opponent = self.max_damage_opponents[i % len(self.max_damage_opponents)]
            elif opp_type == self.RANDOM_BASELINE and self.random_baseline_opponents:
                target_opponent = self.random_baseline_opponents[
                    i % len(self.random_baseline_opponents)
                ]
            elif (
                opp_type == self.MAX_BASE_POWER_BASELINE
                and self.max_base_power_baseline_opponents
            ):
                target_opponent = self.max_base_power_baseline_opponents[
                    i % len(self.max_base_power_baseline_opponents)
                ]
            elif (
                opp_type == self.SIMPLE_HEURISTIC_BASELINE
                and self.simple_heuristic_baseline_opponents
            ):
                target_opponent = self.simple_heuristic_baseline_opponents[
                    i % len(self.simple_heuristic_baseline_opponents)
                ]
            elif opp_type == self.VGC_BENCH_BASELINE and self.vgc_bench_baseline_opponents:
                target_opponent = self.vgc_bench_baseline_opponents[
                    i % len(self.vgc_bench_baseline_opponents)
                ]
            else:
                target_opponent = opponent

            tasks.append(
                player.battle_against(
                    target_opponent,
                    n_battles=num_battles_per_pair,
                )
            )

        return tasks, batch_opponent_types

    def randomize_all_teams(self) -> None:
        """Resample teams for all participants before the next batch."""
        for player in self.players:
            player._team = ConstantTeambuilder(self.sample_team())

        for opponent in self.opponents:
            opponent._team = ConstantTeambuilder(self.sample_team())

        for md_opp in self.max_damage_opponents:
            md_opp._team = ConstantTeambuilder(self.sample_team())

        for random_opp in self.random_baseline_opponents:
            random_opp._team = ConstantTeambuilder(self.sample_team())

        for maxbp_opp in self.max_base_power_baseline_opponents:
            maxbp_opp._team = ConstantTeambuilder(self.sample_team())

        for heuristic_opp in self.simple_heuristic_baseline_opponents:
            heuristic_opp._team = ConstantTeambuilder(self.sample_team())

        for vgc_opp in self.vgc_bench_baseline_opponents:
            vgc_opp._team = ConstantTeambuilder(self.sample_team())

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

        for random_opp in self.random_baseline_opponents:
            random_opp.reset_battles()

        for maxbp_opp in self.max_base_power_baseline_opponents:
            maxbp_opp.reset_battles()

        for heuristic_opp in self.simple_heuristic_baseline_opponents:
            heuristic_opp.reset_battles()

        for vgc_opp in self.vgc_bench_baseline_opponents:
            vgc_opp.reset_battles()


__all__ = [
    "OpponentPool",
    "WorkerOpponentFactory",
    "build_model_from_config",
    "load_model_from_checkpoint",
    "FlexibleThreeHeadedModel",
]
