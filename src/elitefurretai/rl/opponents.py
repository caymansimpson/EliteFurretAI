"""opponents.py — Opponent management for RL training.

What this file is
-----------------
The system that decides "who does the RL agent play against this game?" and
constructs the actual `Player` objects that show up on the other side of the
battle. Two cooperating classes:

1) `OpponentPool` — main-process curriculum manager
    - Owns the curriculum: a probability distribution over opponent *types*
      (self-play, BC clone, ghosts, exploiters, MaxDamage, …).
    - Caches model artifacts so we don't reload from disk every battle.
    - Tracks per-opponent win rates and can adapt curriculum over time
      (PFSP-style: focus on opponents we're currently losing to).
    - Lives in the trainer process; speaks to workers via the broadcast queue.

2) `WorkerOpponentFactory` — per-worker actor factory
    - Lives inside each worker process.
    - When the curriculum says "play self vs BC", this is what actually
      *constructs* the BatchInferencePlayer / BCPlayer / MaxDamagePlayer
      instances and connects them to Showdown.
    - Reuses player objects across battles where possible (each player
      construction triggers a websocket login, which is expensive).
    - Handles team randomization per battle.

Why two classes
---------------
OpponentPool answers "what kind of opponent should this battle have?". It's
about the overall *training plan*. WorkerOpponentFactory answers "given that
choice, instantiate and wire up the actual websocket-connected battle
participants". It's about *execution mechanics*. Keeping them separate means
the curriculum logic doesn't have to know about Showdown account
configuration, and the websocket wiring code doesn't have to know about
adaptive PFSP weighting.

Glossary of opponent types
--------------------------
- self_play          — A second copy of the current learning policy.
- bc_player          — Frozen behavior-cloned policy from human data.
- ghosts             — Past checkpoints of the main agent (from earlier in
                       this same training run). Prevents catastrophic forgetting.
- exploiters         — Adversarially-trained policies whose only goal was to
                       beat a previous version of the agent. Patches blind spots.
- max_damage         — Fixed heuristic (highest damage move). Sanity baseline.
- random_baseline    — Random legal action.
- max_base_power     — Heuristic: highest base-power move regardless of target.
- simple_heuristic   — poke-env's built-in SimpleHeuristicsPlayer.
- vgc_bench_baseline — External SB3-trained agent we compare against.
"""

import asyncio
import gc
import importlib
import importlib.util
import logging
import os
import queue
import random
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast

if TYPE_CHECKING:
    from elitefurretai.rl.inference_worker import (
        InferenceClient,
        WorkerInferenceClients,
    )

import numpy as np
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.learners import (
    build_model_from_config,
    is_checkpoint_compatible_with_model_config,
    load_agent_from_checkpoint,
    load_model_from_checkpoint,
)
from elitefurretai.rl.players import BatchInferencePlayer, MaxDamagePlayer, RNaDAgent
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


# Cached vgc-bench policies keyed by (checkpoint_path, device).
# Loading a stable_baselines3 PPO checkpoint is slow (hundreds of ms); cache
# them so swapping vgc-bench opponents in/out of the curriculum is cheap.
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
    battle_format: str = "gen9vgc2024regg",
    checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip",
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
    - Own the curriculum distribution over opponent types.
    - Resolve sampled opponent types into concrete `Player` instances.
    - Cache expensive model artifacts (BC/exploiter/ghost models).
    - Track win rates per opponent type and (optionally) adapt curriculum.

    Curriculum adaptation (PFSP-style)
    ----------------------------------
    "Prioritized Fictitious Self Play" — the basic idea: focus more battles on
    opponents the agent is currently losing to. After each batch of wins/losses
    is reported to `record_battle_result()`, we periodically `update_curriculum`
    to nudge the distribution toward the harder opponents (clamped, smoothed,
    and gated by sample count to avoid overreacting to noise).

    Win-rate tracking is windowed (default last 100 battles per opponent type)
    so the curriculum responds to *recent* performance, not lifetime stats.

    Why this is in the main process, not workers
    --------------------------------------------
    - Curriculum changes need to be applied uniformly across all workers; the
      main process is the natural coordinator.
    - Aggregating per-worker win rates into a single signal is simpler when
      the aggregator lives in one place.
    - Loading large model checkpoints (BC, exploiters) repeatedly per worker
      would be wasteful; the main process loads once and distributes paths
      to workers, which load themselves on demand.
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
    # Exploiter-vs-victim battles whose trajectories feed the in-process
    # exploiter learner (separate from EXPLOITERS, which is main fighting
    # frozen exploiter snapshots from disk).
    TRAIN_EXPLOITER = "train_exploiter"

    def __init__(
        self,
        main_model: RNaDAgent,
        device: str,
        battle_format: str = "gen9vgc2023regc",
        bc_model_path: Optional[str] = None,
        exploiter_models_dir: str = "data/models/exploiters",
        ghosts_dir: str = "data/models/ghosts",
        vgc_bench_checkpoint_path: Optional[str] = None,
        max_ghosts: int = 10,
        max_exploiter_models: int = 10,
        tracking_window: int = 100,
        curriculum: Optional[Dict[str, float]] = None,
    ):
        self.main_model = main_model
        self.device = device
        self.battle_format = battle_format
        self.max_ghosts = max_ghosts
        # Cap on the number of graduated exploiter snapshots kept in the
        # curriculum pool. Older ones get evicted when new ones graduate.
        # Mirrors max_ghosts semantics; was previously implicit (the list
        # got trimmed by max_ghosts due to a typo, fixed in this change).
        self.max_exploiter_models = max_exploiter_models
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
            self.TRAIN_EXPLOITER: 0.0,
        }

        total = sum(self.curriculum.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Curriculum weights must sum to 1.0, got {total}")

        self.bc_model_path: Optional[str] = bc_model_path
        self._cached_bc_model: Optional[Any] = None
        self._cached_bc_embedder: Optional[Any] = None
        self._cached_bc_config: Optional[Dict] = None

        if bc_model_path:
            self._preload_bc_model()

        self.exploiter_models_dir = exploiter_models_dir
        os.makedirs(self.exploiter_models_dir, exist_ok=True)
        self.exploiter_models: List[Tuple[float, str]] = []
        self.loaded_exploiters: Dict[str, RNaDAgent] = {}
        self._load_exploiter_models()

        self.ghosts_dir = ghosts_dir
        os.makedirs(ghosts_dir, exist_ok=True)
        self.ghosts: List[Tuple[int, str]] = []
        # Ghost slot lifecycle: each path is assigned a slot 0..max_ghosts-1.
        # `slot_for_ghost_path` maps file path -> slot index. `_slot_lru` is
        # an insertion-ordered list of currently-occupied slots; head is
        # oldest. Loaded ghosts from disk on startup get slots 0..K-1.
        self.slot_for_ghost_path: Dict[str, int] = {}
        self._slot_lru: List[int] = []
        self._load_ghosts()

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
            self.TRAIN_EXPLOITER: [],
        }
        self.win_rate_tracking: Dict[str, deque[float]] = {
            opp_type: deque(maxlen=self.tracking_window) for opp_type in self.win_rates
        }
        self.battle_length_tracking: Dict[str, deque[int]] = {
            opp_type: deque(maxlen=self.tracking_window) for opp_type in self.win_rates
        }
        self.total_battles_tracked = 0
        self.total_forfeits_tracked = 0

    def _normalize_curriculum(self, curriculum: Dict[str, float]) -> Dict[str, float]:
        total = float(sum(curriculum.values()))
        if total <= 0:
            return {self.SELF_PLAY: 1.0}
        return {key: value / total for key, value in curriculum.items()}

    def _opponent_available(self, opponent_type: str) -> bool:
        if opponent_type == self.BC_PLAYER:
            return self.bc_model_path is not None
        if opponent_type == self.EXPLOITERS:
            return len(self.exploiter_models) > 0
        if opponent_type == self.GHOSTS:
            return len(self.ghosts) > 0
        return True

    def _ensure_tracking_key(self, opponent_type: str) -> None:
        if opponent_type not in self.win_rate_tracking:
            self.win_rate_tracking[opponent_type] = deque(maxlen=self.tracking_window)
        if opponent_type not in self.battle_length_tracking:
            self.battle_length_tracking[opponent_type] = deque(maxlen=self.tracking_window)
        if opponent_type not in self.win_rates:
            self.win_rates[opponent_type] = []

    def _preload_bc_model(self):
        if not self.bc_model_path:
            return

        from elitefurretai.etl.embedder import Embedder
        from elitefurretai.rl.learners import build_model_from_config

        filepath = self.bc_model_path
        logger.info("Pre-loading BC model from %s...", filepath)

        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint["config"]

        embedder = Embedder(
            format=self.battle_format,
            feature_set=config.get("embedder_feature_set", "raw"),
            omniscient=False,
        )

        model = build_model_from_config(
            config, embedder, self.device, checkpoint["model_state_dict"]
        )
        model.eval()

        self._cached_bc_model = model
        self._cached_bc_embedder = embedder
        self._cached_bc_config = config

        logger.info("BC model cached (saves ~500MB per BCPlayer instance)")

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
        self.exploiter_models = models[: self.max_exploiter_models]

    def _load_ghosts(self) -> None:
        files = self._list_model_checkpoints(self.ghosts_dir)
        models: List[Tuple[int, str]] = []
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (ValueError, IndexError):
                step = int(os.path.getmtime(filepath))
            models.append((step, filepath))

        models.sort(key=lambda item: item[0], reverse=True)
        self.ghosts = models[: self.max_ghosts]
        # Assign slots in oldest-first order (slot 0 = oldest = first to
        # be evicted by LRU). `self.ghosts` is sorted newest-first, so
        # iterate in reverse for slot assignment.
        self.slot_for_ghost_path = {}
        self._slot_lru = []
        for slot, (_, path) in enumerate(reversed(self.ghosts)):
            self.slot_for_ghost_path[path] = slot
            self._slot_lru.append(slot)

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
            return self._create_self_play_opponent(
                player_config, server_config, team, worker_id
            )
        elif opponent_type == self.BC_PLAYER:
            return self._create_bc_opponent(player_config, server_config, team)
        elif opponent_type == self.EXPLOITERS:
            return self._create_exploiter_opponent(
                player_config, server_config, team, worker_id
            )
        elif opponent_type == self.GHOSTS:
            return self._create_ghost_opponent(
                player_config, server_config, team, worker_id
            )
        elif opponent_type == self.MAX_DAMAGE:
            return self._create_max_damage_opponent(player_config, server_config, team)
        elif opponent_type == self.RANDOM_BASELINE:
            return self._create_random_baseline_opponent(
                player_config, server_config, team
            )
        elif opponent_type == self.MAX_BASE_POWER_BASELINE:
            return self._create_max_base_power_baseline_opponent(
                player_config, server_config, team
            )
        elif opponent_type == self.SIMPLE_HEURISTIC_BASELINE:
            return self._create_simple_heuristic_baseline_opponent(
                player_config, server_config, team
            )
        elif opponent_type == self.VGC_BENCH_BASELINE:
            return self._create_vgc_bench_baseline_opponent(
                player_config, server_config, team
            )
        else:
            return self._create_self_play_opponent(
                player_config, server_config, team, worker_id
            )

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
            checkpoint_path=self.vgc_bench_checkpoint_path
            or "data/models/vgc-bench-sb3-model.zip",
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
        if self.bc_model_path is None or self._cached_bc_model is None:
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

        bc_player.model = self._cached_bc_model
        bc_player.embedder = self._cached_bc_embedder  # type: ignore[assignment]
        bc_player.config = self._cached_bc_config  # type: ignore[assignment]

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
            return self._create_self_play_opponent(
                player_config, server_config, team, worker_id
            )

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

    def _create_ghost_opponent(
        self,
        player_config: AccountConfiguration,
        server_config: ServerConfiguration,
        team: str,
        worker_id: Optional[int] = None,
    ) -> Player:
        if not self.ghosts:
            return self._create_self_play_opponent(
                player_config, server_config, team, worker_id
            )

        step, filepath = self.ghosts[np.random.randint(len(self.ghosts))]
        ghost_model = load_agent_from_checkpoint(filepath, self.device)

        new_config = AccountConfiguration(
            f"Ghost_{step}_{player_config.username}", player_config.password
        )

        return BatchInferencePlayer(
            model=ghost_model,
            device=self.device,
            batch_size=16,
            account_configuration=new_config,
            server_configuration=server_config,
            battle_format=self.battle_format,
            probabilistic=True,
            team=team,
            worker_id=worker_id,
        )

    def add_ghost(self, step: int, filepath: str):
        # Determine slot: reuse if path already known, else allocate
        # next free or evict LRU.
        if filepath in self.slot_for_ghost_path:
            slot = self.slot_for_ghost_path[filepath]
        elif len(self._slot_lru) < self.max_ghosts:
            slot = len(self._slot_lru)
            self._slot_lru.append(slot)
        else:
            # Evict the LRU slot. Find which path currently holds it,
            # drop the mapping, and reuse the slot for the new path.
            slot = self._slot_lru.pop(0)
            evicted_path = next(
                p for p, s in self.slot_for_ghost_path.items() if s == slot
            )
            del self.slot_for_ghost_path[evicted_path]
            self._slot_lru.append(slot)
        self.slot_for_ghost_path[filepath] = slot

        self.ghosts.append((step, filepath))
        self.ghosts.sort(key=lambda x: x[0], reverse=True)
        self.ghosts = self.ghosts[: self.max_ghosts]

    def active_ghost_slots(self) -> Set[int]:
        """Slots currently populated with real ghost weights."""
        return set(self._slot_lru)

    def add_exploiter(self, filepath: str):
        """Register a newly-graduated exploiter snapshot.

        Used by the in-process exploiter pipeline (train.py): after an
        exploiter generation graduates and its weights are saved to disk,
        this records the file in the pool so the next broadcast's
        `exploiter_paths` key includes it. Sorted by mtime descending so
        the newest snapshots stay if we exceed `max_exploiter_models`.

        Mirrors `add_ghost` for symmetry. The startup-only directory scan
        in `_load_exploiter_models` still handles initial population
        (e.g., on resume); this method handles in-flight additions.
        """
        if not os.path.isfile(filepath):
            return
        self.exploiter_models.append((os.path.getmtime(filepath), filepath))
        self.exploiter_models.sort(key=lambda x: x[0], reverse=True)
        self.exploiter_models = self.exploiter_models[: self.max_exploiter_models]

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

    def update_win_rate(self, opponent_type: str, won: bool) -> None:
        """Backward-compatible wrapper used by older tests and callers."""
        self.record_battle_result(opponent_type=opponent_type, won=won)

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
            metrics["forfeit_rate"] = (
                self.total_forfeits_tracked / self.total_battles_tracked
            )

        for opp_type, weight in self.curriculum.items():
            metrics[f"curriculum_weight_{opp_type}"] = float(weight)

        return metrics

    # TODO: revisit
    def update_curriculum(self):
        """Adapt curriculum weights from recent matchup performance.

        Hybrid strategy:
        - PFSP score favors opponents near 50/50 win rate (high learning signal)
        - Weakness score upweights hard opponents where main model underperforms
        - Availability gating prevents sampling unsupported opponent types
        - Anchor floors preserve self-play/BC/ghost exposure to reduce forgetting
        - Bayesian smoothing + sample gating reduce noisy swings under high-variance battles
        """

        # Refresh checkpoint-backed pools before scoring availability.
        self._load_exploiter_models()
        self._load_ghosts()
        base = self.curriculum.copy()

        # Blend factor between PFSP (learn near decision boundary) and weakness targeting.
        pfsp_mix = 0.70
        weakness_mix = 0.30

        # Weakness objective: push more games toward opponents where we underperform.
        target_win_rate = 0.55

        # Numerical floor to avoid exact zeros in intermediate score math.
        epsilon = 1e-2

        # Ignore tiny sample windows to avoid adapting to variance noise.
        min_samples = 40

        # Beta prior parameters for smoothed win-rate estimate.
        # Equivalent pseudo-counts make small-n win rates less extreme.
        prior_alpha = 8.0
        prior_beta = 8.0

        # Step 1: build per-opponent candidate learning scores.
        candidate_scores: Dict[str, float] = {}
        for opp_type, base_weight in base.items():
            # Skip unsupported opponents (e.g., no ghost/exploiter checkpoints loaded).
            if not self._opponent_available(opp_type):
                candidate_scores[opp_type] = 0.0
                continue

            samples = len(self.win_rate_tracking.get(opp_type, []))
            # With insufficient data, retain prior curriculum preference.
            if samples < min_samples:
                candidate_scores[opp_type] = max(base_weight, epsilon)
                continue

            # Step 1a: Bayesian-smoothed win rate (wins+alpha)/(n+alpha+beta).
            recent_results = self.win_rate_tracking.get(opp_type, deque())
            wins = float(sum(recent_results))
            losses = float(samples) - wins
            win_rate = (wins + prior_alpha) / (wins + losses + prior_alpha + prior_beta)

            # Step 1b: PFSP favors ~50/50 opponents where policy gradients are most useful.
            pfsp_score = max(0.0, 1.0 - (2.0 * abs(win_rate - 0.5)))

            # Step 1c: Weakness score increases as win rate drops below target.
            weakness_score = max(0.0, (target_win_rate - win_rate) / target_win_rate)
            learning_value = (pfsp_mix * pfsp_score) + (weakness_mix * weakness_score)

            # Step 1d: Mix adaptive signal with base curriculum for stability.
            candidate_scores[opp_type] = max(
                epsilon,
                (0.50 * base_weight) + (0.50 * learning_value),
            )

        # Keep explicit anchor exposure to avoid forgetting and mode collapse.
        floors: Dict[str, float] = {}
        if self._opponent_available(self.SELF_PLAY):
            floors[self.SELF_PLAY] = 0.20
        if self._opponent_available(self.BC_PLAYER):
            floors[self.BC_PLAYER] = 0.10
        if self._opponent_available(self.GHOSTS):
            floors[self.GHOSTS] = 0.10

        floor_sum = sum(floors.values())
        if floor_sum > 0.95:
            # Keep at least 5% free mass for adaptive allocation to non-anchor types.
            scale = 0.95 / floor_sum
            floors = {key: value * scale for key, value in floors.items()}
            floor_sum = sum(floors.values())

        # Step 2: distribute remaining mass by residual score above anchor floors.
        remaining_mass = max(0.0, 1.0 - floor_sum)
        residual_scores: Dict[str, float] = {}
        for opp_type, score in candidate_scores.items():
            if not self._opponent_available(opp_type):
                continue
            residual_scores[opp_type] = max(epsilon, score - floors.get(opp_type, 0.0))

        residual_total = sum(residual_scores.values())
        if residual_total <= 0:
            # Degenerate case fallback: anchor-only curriculum (or pure self-play).
            new_curriculum = {
                opp_type: floors.get(opp_type, 0.0)
                for opp_type in base
                if self._opponent_available(opp_type)
            }
            if self.SELF_PLAY not in new_curriculum:
                new_curriculum[self.SELF_PLAY] = 1.0
        else:
            new_curriculum = {}
            for opp_type in base:
                if not self._opponent_available(opp_type):
                    continue
                floor = floors.get(opp_type, 0.0)
                residual = residual_scores.get(opp_type, 0.0) / residual_total
                new_curriculum[opp_type] = floor + (remaining_mass * residual)

            # Final normalization protects against drift from rounding and availability gating.
        self.curriculum = self._normalize_curriculum(new_curriculum)


class WorkerOpponentFactory:
    """Worker-local helper for battle pair construction and batch reconfiguration.

    This class is intentionally simpler than `OpponentPool`: it does not track
    global win-rate stats or persist metadata. Instead, it focuses on the hot
    path inside worker loops:
    - create player/opponent pairs once
    - swap opponent policy per batch
    - randomize teams per batch
    - reset battle state to avoid memory growth

    Why "factory" not "pool"
    ------------------------
    OpponentPool decides *which* type of opponent to play; this class actually
    *constructs* the player objects. They have related responsibilities but
    different concerns (planning vs. execution), and live in different
    processes (main vs. worker).

    Account-name mechanics
    ----------------------
    Every player needs a unique Showdown account name. We can't reuse names
    across worker rebuilds because Showdown rejects duplicates with `|nametaken|`
    until the previous socket fully disconnects. The `_account_name()` helper
    composes a unique-ish name from worker_id + role + index + run_id +
    factory_tag + rebuild_generation. See its docstring for the gory details.
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
    # Exploiter-vs-victim battles whose trajectories train the in-process
    # exploiter learner. Distinct from EXPLOITERS (main fights frozen
    # exploiter snapshots from disk).
    TRAIN_EXPLOITER = "train_exploiter"

    def __init__(
        self,
        team_repo: TeamRepo,
        battle_format: str,
        team_subdirectory: Optional[str],
        server_config: ServerConfiguration,
        main_agent: Optional[RNaDAgent],
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
        vgc_bench_checkpoint_path: Optional[str] = None,
        external_vgcbench_usernames: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        agent_team_path: Optional[str] = None,
        exploiter_agent: Optional[RNaDAgent] = None,
        victim_agent: Optional[RNaDAgent] = None,
        max_concurrent_battles_per_player: Optional[int] = None,
        # Centralized inference: when set, the factory wires
        # BatchInferencePlayers to submit through the trainer-side
        # InferenceService instead of holding a model copy. `main_agent`
        # may be None when this is set.
        main_inference_client: Optional["InferenceClient"] = None,
        # Full bundle of inference clients (one per registered model
        # name in the trainer's ModelRegistry). When set, takes
        # precedence over the singular `main_inference_client` —
        # `configure_opponent_for_batch` uses bundle.get("main") /
        # bundle.get("bc") / etc. to hot-swap opponent.inference_client
        # per curriculum pick. `main_inference_client` is retained as
        # a fallback for callers that haven't been updated to the
        # bundle interface.
        worker_inference_clients: Optional["WorkerInferenceClients"] = None,
    ):
        if main_agent is None and main_inference_client is None:
            raise ValueError(
                "WorkerOpponentFactory requires either main_agent (legacy "
                "per-player inference) or main_inference_client (centralized)"
            )
        self.team_repo = team_repo
        self.battle_format = battle_format
        self.team_subdirectory = team_subdirectory
        self.server_config = server_config
        self.main_agent = main_agent
        # The bundle is the source of truth in centralized mode (step 3+).
        # `main_inference_client` is kept as a back-compat fallback for
        # callers that haven't switched to the bundle interface.
        self.worker_inference_clients = worker_inference_clients
        self.main_inference_client = main_inference_client
        self.bc_agent = bc_agent
        # poke-env Player's `max_concurrent_battles` kwarg. None preserves
        # the library default of 1. See Track C in
        # planning/stage2/2026-05-12-09-30-second-wsl-crash-and-watchdog.md.
        self.max_concurrent_battles_per_player = max_concurrent_battles_per_player
        # Co-training agents (None until train_exploiter > 0). exploiter_agent
        # is the live policy being optimized to beat main; victim_agent is a
        # frozen, periodically-refreshed copy of main that the exploiter
        # trains against. Both are CPU model copies in the worker — main
        # process owns the GPU originals and broadcasts state_dicts.
        self.exploiter_agent = exploiter_agent
        self.victim_agent = victim_agent
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
        self.vgc_bench_checkpoint_path = vgc_bench_checkpoint_path
        self.model_config = model_config
        self.agent_team_path = agent_team_path
        self._agent_team_string: Optional[str] = None
        self._agent_team_strings: Optional[List[str]] = None
        if agent_team_path:
            if os.path.isdir(agent_team_path):
                self._agent_team_strings = []
                for fname in sorted(os.listdir(agent_team_path)):
                    if fname.endswith(".txt"):
                        with open(os.path.join(agent_team_path, fname)) as f:
                            self._agent_team_strings.append(f.read())
                logger.info(
                    "Loaded %d agent teams from directory %s",
                    len(self._agent_team_strings),
                    agent_team_path,
                )
            else:
                with open(agent_team_path) as f:
                    self._agent_team_string = f.read()
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
        self._active_ghost_slots: Set[int] = set()
        self._batch_count = 0
        # Rebuild generation increments every time we recreate runtime agents.
        # Why: Showdown usernames must be unique among currently connected clients,
        # and stale sockets can briefly outlive a rebuild.
        self._rebuild_generation = 0
        # Compact entropy tags for username uniqueness across fast restarts.
        # Why: run_id alone can still collide in close launches if stale sockets
        # have not fully disconnected yet.
        cleaned_run_id = "".join(ch for ch in str(self.run_id) if ch.isalnum()).upper()
        self._run_tag = cleaned_run_id[-4:] if cleaned_run_id else "0000"
        self._factory_tag = f"{random.getrandbits(8):02X}"
        self._load_exploiter_models()

    def _account_name(self, role: str, idx: int) -> str:
        """Create compact, rebuild-unique account names.

        Why: the previous naming scheme reused identical usernames during rebuilds,
        which produced `|nametaken|` errors and prevented worker recovery.
        """
        worker_token = f"{self.worker_id % 256:02X}"
        idx_token = f"{idx % 256:02X}"
        gen_token = f"{self._rebuild_generation % 256:02X}"
        return f"M{worker_token}{role}{idx_token}{self._run_tag}{self._factory_tag}{gen_token}"

    def update_curriculum(self, curriculum: Dict[str, float]) -> None:
        """Update worker-local curriculum and refresh dependent opponent pools."""
        total = float(sum(curriculum.values()))
        if total <= 0:
            self.curriculum = {self.SELF_PLAY: 1.0}
        else:
            self.curriculum = {
                key: float(value) / total for key, value in curriculum.items() if value > 0
            }
            if not self.curriculum:
                self.curriculum = {self.SELF_PLAY: 1.0}

        # Important: do NOT recreate baseline/max-damage opponent clients here.
        # Why: curriculum updates can happen repeatedly at runtime (including local
        # vgc-bench disable fallbacks). Recreating players with the same generation
        # can trigger duplicate login attempts and `|nametaken|` collisions.
        # Existing baseline pools are kept alive; sampling already follows the new
        # probabilities in `self.curriculum`.

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

    def _get_cached_model(
        self,
        filepath: str,
        loaded: Dict[str, RNaDAgent],
    ) -> RNaDAgent:
        if filepath in loaded:
            return loaded[filepath]

        loaded_model = load_agent_from_checkpoint(filepath, self.device)
        loaded[filepath] = loaded_model
        return loaded_model

    def _get_exploiter_agent(self) -> Optional[RNaDAgent]:
        if not self.active_exploiters:
            return None

        _, filepath = random.choice(self.active_exploiters)
        return self._get_cached_model(filepath, self.loaded_exploiters)

    def set_exploiter_paths(self, paths: List[str]) -> None:
        """Apply explicit exploiter file list from learner broadcast (Option C)."""
        models = [(os.path.getmtime(p), p) for p in paths if os.path.isfile(p)]
        models.sort(key=lambda item: item[0], reverse=True)
        self.active_exploiters = models

    def set_active_ghost_slots(self, slots: List[int]) -> None:
        """Update the set of populated ghost slots from a trainer broadcast.

        Workers use this to know which `ghost_<slot>` clients in the
        inference bundle correspond to real ghost weights vs placeholders.
        Only slots in this set are valid targets for GHOSTS opponent
        routing in `configure_opponent_for_batch`.
        """
        self._active_ghost_slots = set(slots)

    def update_exploiter_weights(self, state_dict: Dict) -> None:
        """Apply broadcasted exploiter weights to the worker-local exploiter agent.

        No-op if `exploiter_agent` is None (co-training disabled). Loading is
        in-place so any BatchInferencePlayer holding a reference to the same
        nn.Module picks up the new weights without rewiring.
        """
        if self.exploiter_agent is not None:
            self.exploiter_agent.model.load_state_dict(state_dict)

    def update_victim_weights(self, state_dict: Dict) -> None:
        """Apply broadcasted victim weights to the worker-local victim agent.

        Refreshed less frequently than exploiter weights (typically every
        `victim_refresh_interval` main updates) — the victim is the
        stationary target the exploiter optimizes against.
        """
        if self.victim_agent is not None:
            self.victim_agent.model.load_state_dict(state_dict)

    def sample_team(self) -> str:
        return self.team_repo.sample_team(
            self.battle_format,
            subdirectory=self.team_subdirectory,
        )

    def get_agent_team(self) -> str:
        """Return the agent's team, shuffled. Uses fixed team if agent_team_path is set.

        If agent_team_path was a directory, samples randomly from all loaded teams.
        """
        if self._agent_team_strings is not None:
            team = random.choice(self._agent_team_strings)
            return self.team_repo._shuffle_team_order(team)
        if self._agent_team_string is not None:
            return self.team_repo._shuffle_team_order(self._agent_team_string)
        return self.sample_team()

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

        # Conditionally pass `max_concurrent_battles` so when the config
        # leaves it None we don't override poke-env's default of 1.
        extra_player_kwargs: Dict[str, Any] = {}
        if self.max_concurrent_battles_per_player is not None:
            extra_player_kwargs["max_concurrent_battles"] = (
                self.max_concurrent_battles_per_player
            )

        # Centralized vs legacy: in centralized mode the player owns
        # an `inference_client` and submits requests to the trainer-side
        # InferenceService; in legacy mode it owns a model copy and runs
        # its own inference loop. Same constructor accepts either.
        # Prefer the bundle (worker_inference_clients.get("main")) so
        # that the same player can be re-pointed at "bc" / ghost slots
        # later by configure_opponent_for_batch.
        main_client = self._resolve_centralized_client("main")
        main_kwargs: Dict[str, Any]
        if main_client is not None:
            main_kwargs = {"inference_client": main_client}
        else:
            main_kwargs = {"model": self.main_agent}

        for i in range(num_pairs):
            player = BatchInferencePlayer(
                device=self.device,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                account_configuration=AccountConfiguration(
                    self._account_name("SF", i), None
                ),
                server_configuration=self.server_config,
                trajectory_queue=local_traj_queue,
                battle_format=self.battle_format,
                team=self.get_agent_team(),
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
                opponent_type=self.SELF_PLAY,
                **main_kwargs,
                **extra_player_kwargs,
            )
            self.players.append(player)

            opponent = BatchInferencePlayer(
                device=self.device,
                batch_size=self.batch_size,
                batch_timeout=self.batch_timeout,
                account_configuration=AccountConfiguration(
                    self._account_name("OP", i), None
                ),
                server_configuration=self.server_config,
                trajectory_queue=None,
                battle_format=self.battle_format,
                team=self.sample_team(),
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
                **main_kwargs,
                **extra_player_kwargs,
            )
            self.opponents.append(opponent)

        return self.players, self.opponents

    def create_agents(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[
        List[BatchInferencePlayer], List[BatchInferencePlayer], List[MaxDamagePlayer]
    ]:
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
                        self._account_name("MD", i), None
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
                            self._account_name("RD", i), None
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
                            self._account_name("MB", i), None
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
                            self._account_name("SH", i), None
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
                    player_config=AccountConfiguration(self._account_name("VG", i), None),
                    server_config=self.server_config,
                    team=self.sample_team(),
                    checkpoint_path=self.vgc_bench_checkpoint_path
                    or "data/models/vgc-bench-sb3-model.zip",
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

    def _resolve_centralized_client(self, model_name: str) -> Optional["InferenceClient"]:
        """Return the centralized InferenceClient for `model_name` if
        available. Order of precedence:
          1. `worker_inference_clients` bundle (step 3+ canonical path)
          2. legacy single `main_inference_client` parameter (for "main"
             only — kept so step 2 callers continue to work)
          3. None — caller falls back to legacy `model = X` swap
        """
        if self.worker_inference_clients is not None and self.worker_inference_clients.has(
            model_name
        ):
            return self.worker_inference_clients.get(model_name)
        if model_name == "main" and self.main_inference_client is not None:
            return self.main_inference_client
        return None

    def _swap_to(
        self,
        slot: BatchInferencePlayer,
        centralized_name: Optional[str],
        legacy_agent: Optional[RNaDAgent],
    ) -> bool:
        """Re-point one player or opponent slot at a different model.

        Centralized mode: assign `slot.inference_client = clients.get(name)`.
        Legacy mode: assign `slot.model = legacy_agent`.

        Returns True if the swap succeeded (the requested model is
        available in the active mode), False if neither was available
        (caller should fall back to self-play / main).
        """
        if centralized_name is not None:
            client = self._resolve_centralized_client(centralized_name)
            if client is not None:
                slot.inference_client = client
                return True
        if legacy_agent is not None:
            slot.model = legacy_agent
            return True
        return False

    def configure_opponent_for_batch(
        self,
        player: BatchInferencePlayer,
        opponent: BatchInferencePlayer,
    ) -> str:
        """Configure one pair for the next batch and return chosen type.

        - `max_damage` is handled by separate MaxDamagePlayer instances
        - `bc_player` swaps opponent to the BC client (centralized) or
          model (legacy)
        - `train_exploiter` swaps player→exploiter, opponent→victim
        - fallback is self-play against the main agent

        In centralized mode (worker_inference_clients set), the swap
        targets `slot.inference_client`; in legacy mode it targets
        `slot.model`. `_swap_to` resolves which to use.
        """
        selected_type = self.sample_opponent_type()

        # Default: self-play. Each branch may override.
        opponent_swapped = False

        if selected_type == self.MAX_DAMAGE:
            pass
        elif selected_type == self.BC_PLAYER:
            opponent_swapped = self._swap_to(opponent, "bc", self.bc_agent)
            if not opponent_swapped:
                selected_type = self.SELF_PLAY
        elif selected_type == self.EXPLOITERS:
            exploiter_agent = self._get_exploiter_agent()
            opponent_swapped = self._swap_to(opponent, None, exploiter_agent)
            if not opponent_swapped:
                selected_type = self.SELF_PLAY
        elif selected_type == self.TRAIN_EXPLOITER:
            # Exploiter (player) vs. frozen victim (opponent). Trajectory
            # is tagged "train_exploiter" so the main process routes it
            # to the exploiter learner instead of the main learner.
            player_swapped = self._swap_to(player, "exploiter", self.exploiter_agent)
            opponent_swapped = self._swap_to(opponent, "victim", self.victim_agent)
            if not (player_swapped and opponent_swapped):
                # Co-training agents not provisioned (e.g., curriculum
                # slot ramped up before main process built them). Fall
                # back to self-play; trajectory will be routed as main.
                selected_type = self.SELF_PLAY
                self._swap_to(player, "main", self.main_agent)
        elif selected_type == self.GHOSTS:
            if not self._active_ghost_slots:
                # Curriculum sampled GHOSTS but no slots are populated —
                # treat as a desync bug and fall back to self-play loudly
                # via the standard path. (The curriculum's
                # _opponent_available guard should prevent this.)
                selected_type = self.SELF_PLAY
            else:
                slot = random.choice(tuple(self._active_ghost_slots))
                opponent_swapped = self._swap_to(opponent, f"ghost_{slot}", None)
                if not opponent_swapped:
                    selected_type = self.SELF_PLAY
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

        # If we fell back to self-play (or were self-play to begin
        # with), or the chosen branch didn't swap the opponent, point
        # opponent at main.
        if not opponent_swapped:
            self._swap_to(opponent, "main", self.main_agent)

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
                target_opponent = self.max_damage_opponents[
                    i % len(self.max_damage_opponents)
                ]
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
        """Resample teams for all participants before the next batch.

        Players use the fixed agent team if agent_team_path is set,
        otherwise sample randomly. Opponents always sample randomly.
        """
        for player in self.players:
            player._team = ConstantTeambuilder(self.get_agent_team())

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

    def teardown_runtime_agents(self) -> None:
        """Best-effort teardown of all worker-local players/opponents.

        Why: rebuilds need to disconnect previous websocket clients before creating
        replacements, otherwise old sessions can still hold usernames.
        """
        # Stop inference/listening for batch-inference players first.
        for player in self.players + self.opponents:
            try:
                player.teardown_runtime()
            except Exception:
                pass

        # Non-batched baseline players (max damage / heuristics / etc.) may not expose
        # teardown_runtime; attempt a generic listener stop when available.
        for participant in (
            self.max_damage_opponents
            + self.random_baseline_opponents
            + self.max_base_power_baseline_opponents
            + self.simple_heuristic_baseline_opponents
            + self.vgc_bench_baseline_opponents
        ):
            stop_fn = getattr(participant, "stop_listening", None)
            if stop_fn is None:
                continue
            try:
                maybe_coro = stop_fn()
                if asyncio.iscoroutine(maybe_coro):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If a loop is already running in this context, schedule and continue.
                        asyncio.create_task(maybe_coro)
                    else:
                        loop.run_until_complete(maybe_coro)
            except Exception:
                pass

    @staticmethod
    def _count_unfinished_for_player(player: Player) -> int:
        battles = getattr(player, "_battles", {})
        if not isinstance(battles, dict):
            return 0
        return sum(1 for battle in battles.values() if not battle.finished)

    def get_unfinished_battle_summary(self) -> Dict[str, Any]:
        # This summary powers two things:
        # 1) first-trigger diagnostics, and
        # 2) deciding whether we can safely call reset_battles().
        by_player: Dict[str, int] = {}
        total = 0

        for player in self.players + self.opponents:
            count = self._count_unfinished_for_player(player)
            if count > 0:
                username = player.username if hasattr(player, "username") else str(player)
                by_player[username] = count
                total += count

        return {
            "total_unfinished": total,
            "by_player": by_player,
        }

    def get_runtime_diagnostics(self) -> Dict[str, Dict[str, float]]:
        def aggregate(participants: List[Any]) -> Dict[str, float]:
            aggregated: Dict[str, float] = {}
            for participant in participants:
                snapshot_fn = getattr(participant, "get_diagnostics_snapshot", None)
                if snapshot_fn is None:
                    continue
                snapshot = snapshot_fn()
                if not isinstance(snapshot, dict):
                    continue
                for key, value in snapshot.items():
                    aggregated[key] = aggregated.get(key, 0.0) + float(value)
            return aggregated

        players_diag = aggregate(self.players)
        opponents_diag = aggregate(self.opponents)
        combined = dict(players_diag)
        for key, value in opponents_diag.items():
            combined[key] = combined.get(key, 0.0) + value

        return {
            "players": players_diag,
            "opponents": opponents_diag,
            "combined": combined,
        }

    def rebuild_runtime_agents(self, local_traj_queue: queue.Queue) -> None:
        """Rebuild player/opponent runtime state after websocket/battle desync.

        This resets all battle participants to fresh connections instead of trying to
        forcibly clear active battles.
        """
        num_pairs = len(self.players)
        if num_pairs <= 0:
            return

        # Explicitly tear down old runtime state before re-creating players.
        self.teardown_runtime_agents()

        # Drop references to old player/opponent objects so they can be GC'd.
        # Without this, asyncio coroutine frames may prevent collection.
        for p in self.players:
            p.current_trajectories.clear()
            p.completed_trajectories.clear()
            p.hidden_states.clear()
        for o in self.opponents:
            o.hidden_states.clear()
        self.players.clear()
        self.opponents.clear()
        self.max_damage_opponents.clear()
        self.random_baseline_opponents.clear()
        self.max_base_power_baseline_opponents.clear()
        self.simple_heuristic_baseline_opponents.clear()
        self.vgc_bench_baseline_opponents.clear()

        # Force a GC cycle to reclaim memory from the old objects.
        gc.collect()

        # Bump generation so newly-created players use unique usernames.
        self._rebuild_generation += 1

        self.create_agents(num_pairs, local_traj_queue)
        self.start_inference_loops()

    def reset_all_battles(self) -> None:
        """Clear per-battle runtime state after a batch completes."""
        for player in self.players:
            player.reset_battles()
            player.clear_completed_trajectories()
            player.hidden_states.clear()
            player.current_trajectories.clear()
            player._discarded_battles.clear()
            player._request_generation.clear()

        for opponent in self.opponents:
            opponent.reset_battles()
            opponent.clear_completed_trajectories()
            opponent.hidden_states.clear()
            opponent.current_trajectories.clear()
            opponent._discarded_battles.clear()
            opponent._request_generation.clear()

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
]
