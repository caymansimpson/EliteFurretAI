"""opponents.py — Opponent management for RL training.

What this file is
-----------------
The system that decides "who does the RL agent play against this game?" and
constructs the actual `Player` objects that show up on the other side of the
battle. Two cooperating classes:

1) `OpponentPool` — main-process curriculum manager
    - Owns the curriculum: a probability distribution over opponent *types*
      (self-play, BC clone, ghosts, exploiters, MaxDamage, …).
    - Tracks per-opponent win rates and can adapt curriculum over time
      (PFSP-style: focus on opponents we're currently losing to).
    - Lives in the trainer process; speaks to workers via the broadcast queue.

2) `WorkerOpponentFactory` — per-worker actor factory
    - Lives inside each worker process.
    - When the curriculum says "play self vs BC", this is what actually
      *constructs* the RLTrajectoryPlayer / BCPlayer / MaxDamagePlayer
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
import logging
import os
import queue
import random
from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

import numpy as np
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.inference_worker import WorkerInferenceClients
from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer
from elitefurretai.rl.rl_utils import list_pt_files, normalize_curriculum

logger = logging.getLogger(__name__)


# TypeVar for `_make_baseline_pool`: lets the helper return the concrete
# player class (MaxDamagePlayer/RandomPlayer/...) without a cast.
_P = TypeVar("_P", bound=Player)


def largest_remainder_apportionment(
    num_items: int, weights: Dict[str, float]
) -> List[str]:
    """Distribute ``num_items`` slots across ``weights.keys()`` proportionally.

    Uses the largest-remainder (Hamilton) method: floor every quota first,
    then assign leftover slots one-at-a-time to the keys with the largest
    fractional remainders. Deterministic — ties are broken by sorted key
    order. Returns a list of length ``num_items``.

    Example:

        >>> largest_remainder_apportionment(5, {"a": 0.7, "b": 0.3})
        ['a', 'a', 'a', 'a', 'b']
    """
    if num_items <= 0 or not weights:
        return []
    quotas = {fmt: weights[fmt] * num_items for fmt in weights}
    floors = {fmt: int(quotas[fmt]) for fmt in weights}
    remainders = sorted(
        ((quotas[fmt] - floors[fmt], fmt) for fmt in weights),
        key=lambda pair: (-pair[0], pair[1]),
    )
    leftover = num_items - sum(floors.values())
    for _, fmt in remainders[:leftover]:
        floors[fmt] += 1
    result: List[str] = []
    for fmt in sorted(weights):
        result.extend([fmt] * floors[fmt])
    return result


class OpponentPool:
    """Main-process opponent sampler.

    Responsibilities:
    - Own the curriculum distribution over opponent types.
    - Resolve sampled opponent types into concrete `Player` instances.
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

    # Canonical set of opp-type tags. Used by the worker factory's
    # sample step to drop unrecognized samples.
    KNOWN_OPP_TYPES: Tuple[str, ...] = (
        SELF_PLAY,
        BC_PLAYER,
        EXPLOITERS,
        GHOSTS,
        TRAIN_EXPLOITER,
        MAX_DAMAGE,
        RANDOM_BASELINE,
        MAX_BASE_POWER_BASELINE,
        SIMPLE_HEURISTIC_BASELINE,
        VGC_BENCH_BASELINE,
    )

    def __init__(
        self,
        curriculum: Dict[str, float],
        bc_model_path: Optional[str] = None,
        exploiter_models_dir: str = "data/models/exploiters",
        ghosts_dir: str = "data/models/ghosts",
        max_ghosts: int = 10,
        max_exploiter_models: int = 10,
        tracking_window: int = 100,
    ):
        self.max_ghosts = max_ghosts
        self.max_exploiter_models = max_exploiter_models
        self.tracking_window = tracking_window
        self.curriculum = curriculum

        total = sum(self.curriculum.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Curriculum weights must sum to 1.0, got {total}")

        self.bc_model_path: Optional[str] = bc_model_path

        self.exploiter_models_dir = exploiter_models_dir
        os.makedirs(self.exploiter_models_dir, exist_ok=True)
        self.slot_for_exploiter_path: "OrderedDict[str, int]" = OrderedDict()
        self._load_exploiter_models()

        self.ghosts_dir = ghosts_dir
        os.makedirs(ghosts_dir, exist_ok=True)
        self.slot_for_ghost_path: "OrderedDict[str, int]" = OrderedDict()
        self._load_ghosts()

        self.win_rates: Dict[str, List[float]] = defaultdict(
            list,
            {
                OpponentPool.SELF_PLAY: [],
                OpponentPool.BC_PLAYER: [],
                OpponentPool.EXPLOITERS: [],
                OpponentPool.GHOSTS: [],
                OpponentPool.MAX_DAMAGE: [],
                OpponentPool.RANDOM_BASELINE: [],
                OpponentPool.MAX_BASE_POWER_BASELINE: [],
                OpponentPool.SIMPLE_HEURISTIC_BASELINE: [],
                OpponentPool.VGC_BENCH_BASELINE: [],
                OpponentPool.TRAIN_EXPLOITER: [],
            },
        )
        self.win_rate_tracking: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.tracking_window),
            {opp_type: deque(maxlen=self.tracking_window) for opp_type in self.win_rates},
        )
        self.battle_length_tracking: Dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self.tracking_window),
            {opp_type: deque(maxlen=self.tracking_window) for opp_type in self.win_rates},
        )
        self.total_battles_tracked = 0
        self.total_forfeits_tracked = 0

    def _opponent_available(self, opponent_type: str) -> bool:
        if opponent_type == OpponentPool.BC_PLAYER:
            return self.bc_model_path is not None
        if opponent_type == OpponentPool.EXPLOITERS:
            return len(self.slot_for_exploiter_path) > 0
        if opponent_type == OpponentPool.GHOSTS:
            return len(self.slot_for_ghost_path) > 0
        return True

    @staticmethod
    def _add_to_slot_pool(
        pool: "OrderedDict[str, int]", filepath: str, max_size: int
    ) -> None:
        """Register `filepath` in a slot pool with stable slot IDs.

        Insertion order is the eviction queue: re-adding a known path is a
        no-op (keeps its slot and FIFO position); a new path takes the next
        free slot, or recycles the slot of the longest-resident entry when
        the pool is full. Workers reference slots by integer in their
        inference bundle (`ghost_<slot>` / `exploiter_snap_<slot>`), so
        stability minimizes weight reloads on eviction.
        """
        if filepath in pool:
            return
        if len(pool) < max_size:
            slot = len(pool)
        else:
            _, slot = pool.popitem(last=False)
        pool[filepath] = slot

    def _load_exploiter_models(self) -> None:
        files = list_pt_files(self.exploiter_models_dir)
        models = sorted(
            ((os.path.getmtime(fp), fp) for fp in files if os.path.isfile(fp)),
            reverse=True,
        )[: self.max_exploiter_models]
        # Insert oldest-first so slot 0 = oldest = first to evict.
        self.slot_for_exploiter_path.clear()
        for _, path in reversed(models):
            self._add_to_slot_pool(
                self.slot_for_exploiter_path, path, self.max_exploiter_models
            )

    def _load_ghosts(self) -> None:
        files = list_pt_files(self.ghosts_dir)
        models: List[Tuple[int, str]] = []
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (ValueError, IndexError):
                step = int(os.path.getmtime(filepath))
            models.append((step, filepath))

        models.sort(reverse=True)
        models = models[: self.max_ghosts]
        # Insert oldest-first so slot 0 = oldest = first to evict.
        self.slot_for_ghost_path.clear()
        for _, path in reversed(models):
            self._add_to_slot_pool(self.slot_for_ghost_path, path, self.max_ghosts)

    def add_ghost(self, filepath: str) -> None:
        self._add_to_slot_pool(self.slot_for_ghost_path, filepath, self.max_ghosts)

    def active_ghost_slots(self) -> Set[int]:
        """Slots currently populated with real ghost weights."""
        return set(self.slot_for_ghost_path.values())

    def add_exploiter(self, filepath: str) -> None:
        """Register a newly-graduated exploiter snapshot.

        Used by the in-process exploiter pipeline (train.py): after an
        exploiter generation graduates and its weights are saved to disk,
        this records the file in the pool so the next broadcast's
        `exploiter_paths` key includes it. Eviction is FIFO by registration
        order — the longest-resident snapshot loses its slot when full.

        Mirrors `add_ghost` for symmetry. The startup-only directory scan
        in `_load_exploiter_models` still handles initial population
        (e.g., on resume); this method handles in-flight additions.
        """
        if not os.path.isfile(filepath):
            return
        self._add_to_slot_pool(
            self.slot_for_exploiter_path, filepath, self.max_exploiter_models
        )

    def active_exploiter_slots(self) -> Set[int]:
        """Slots currently populated with real exploiter snapshot weights."""
        return set(self.slot_for_exploiter_path.values())

    def record_battle_result(
        self,
        opponent_type: str,
        won: bool,
        battle_length: int = 0,
        forfeited: bool = False,
    ) -> None:
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
            metrics[f"win_rate/{opp_type}"] = win_rate

        length_stats = self.get_battle_length_stats(used_window)
        for opp_type, avg_len in length_stats.items():
            if len(self.battle_length_tracking.get(opp_type, [])) > 0:
                metrics[f"battle_length/{opp_type}"] = avg_len

        all_lengths = [
            length
            for lengths in self.battle_length_tracking.values()
            for length in lengths
        ]
        if all_lengths:
            metrics["battle_length/overall"] = float(np.mean(all_lengths))

        for opp_type, weight in self.curriculum.items():
            metrics[f"curriculum/{opp_type}"] = float(weight)

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
        if self._opponent_available(OpponentPool.SELF_PLAY):
            floors[OpponentPool.SELF_PLAY] = 0.20
        if self._opponent_available(OpponentPool.BC_PLAYER):
            floors[OpponentPool.BC_PLAYER] = 0.10
        if self._opponent_available(OpponentPool.GHOSTS):
            floors[OpponentPool.GHOSTS] = 0.10

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
            if OpponentPool.SELF_PLAY not in new_curriculum:
                new_curriculum[OpponentPool.SELF_PLAY] = 1.0
        else:
            new_curriculum = {}
            for opp_type in base:
                if not self._opponent_available(opp_type):
                    continue
                floor = floors.get(opp_type, 0.0)
                residual = residual_scores.get(opp_type, 0.0) / residual_total
                new_curriculum[opp_type] = floor + (remaining_mass * residual)

            # Final normalization protects against drift from rounding and availability gating.
        self.curriculum = normalize_curriculum(new_curriculum)


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

    def __init__(
        self,
        team_repo: TeamRepo,
        battle_formats: Dict[str, float],
        opponent_team_subdirectories: Dict[str, Optional[str]],
        server_config: ServerConfiguration,
        curriculum: Optional[Dict[str, float]],
        embedder: Embedder,
        worker_id: int,
        run_id: str,
        worker_inference_clients: WorkerInferenceClients,
        max_battle_steps: int = 40,
        external_vgcbench_usernames: Optional[List[str]] = None,
        agent_team_paths: Optional[Dict[str, str]] = None,
        max_concurrent_battles_per_player: Optional[int] = None,
    ):
        self.team_repo = team_repo
        self.battle_formats = dict(battle_formats)
        self.opponent_team_subdirectories = dict(opponent_team_subdirectories)
        self.server_config = server_config
        self.worker_inference_clients = worker_inference_clients
        self.max_concurrent_battles_per_player = max_concurrent_battles_per_player
        self.curriculum = curriculum or {OpponentPool.SELF_PLAY: 1.0}
        self.embedder = embedder
        self.worker_id = worker_id
        self.run_id = run_id
        self.max_battle_steps = max_battle_steps

        # Per-format agent team strings. Keys match self.battle_formats; each
        # value is the list of team strings loaded from disk for that format
        # (or [] if no agent_team_paths entry was given for that format,
        # which means "sample from opponent_team_subdirectories instead").
        self._agent_teams_by_format: Dict[str, List[str]] = {
            fmt: [] for fmt in self.battle_formats
        }
        if agent_team_paths:
            for fmt, path in agent_team_paths.items():
                teams: List[str] = []
                if os.path.isdir(path):
                    for fname in sorted(os.listdir(path)):
                        if fname.endswith(".txt"):
                            with open(os.path.join(path, fname)) as f:
                                teams.append(f.read())
                    logger.info(
                        "Loaded %d agent teams for %s from directory %s",
                        len(teams),
                        fmt,
                        path,
                    )
                else:
                    with open(path) as f:
                        teams.append(f.read())
                self._agent_teams_by_format[fmt] = teams

        self.external_vgcbench_usernames = [
            username.strip()
            for username in (external_vgcbench_usernames or [])
            if username and username.strip()
        ]

        self.players: List[RLTrajectoryPlayer] = []
        self.opponents: List[RLTrajectoryPlayer] = []
        self.max_damage_opponents: List[MaxDamagePlayer] = []
        self.random_baseline_opponents: List[RandomPlayer] = []
        self.max_base_power_baseline_opponents: List[MaxBasePowerPlayer] = []
        self.simple_heuristic_baseline_opponents: List[Player] = []
        self._active_ghost_slots: Set[int] = set()
        self._active_exploiter_slots: Set[int] = set()
        self._batch_count = 0
        # Rebuild generation increments every time we recreate runtime agents.
        # Why: Showdown usernames must be unique among currently connected clients,
        # and stale sockets can briefly outlive a rebuild.
        self._rebuild_generation = 0
        # Per-pair format assignment. Populated in create_agents() (Task 4).
        self.pair_formats: List[str] = []

        # Compact entropy tags for username uniqueness across fast restarts.
        # Why: run_id alone can still collide in close launches if stale sockets
        # have not fully disconnected yet.
        cleaned_run_id = "".join(ch for ch in str(self.run_id) if ch.isalnum()).upper()
        self._run_tag = cleaned_run_id[-4:] if cleaned_run_id else "0000"
        self._factory_tag = f"{random.getrandbits(8):02X}"

    def _account_name(self, role: str, idx: int) -> str:
        """Create compact, rebuild-unique account names.

        Format: `{role}{worker:02X}{idx:02X}{run_tag}{factory_tag}{gen:02X}`,
        e.g. `MaxD0100000A3B43AF`. Role token leads so the player type is
        visible at a glance in the Showdown UI; the trailing tokens
        guarantee uniqueness across workers, indices, runs, and rebuilds
        (`|nametaken|` errors from reused names prevented worker recovery
        before this scheme).
        """
        worker_token = f"{self.worker_id % 256:02X}"
        idx_token = f"{idx % 256:02X}"
        gen_token = f"{self._rebuild_generation % 256:02X}"
        return (
            f"{role}{worker_token}{idx_token}{self._run_tag}{self._factory_tag}{gen_token}"
        )

    def update_curriculum(self, curriculum: Dict[str, float]) -> None:
        """Update worker-local curriculum and refresh dependent opponent pools."""
        self.curriculum = normalize_curriculum(curriculum)

    def set_active_ghost_slots(self, slots: List[int]) -> None:
        """Update the set of populated ghost slots from a trainer broadcast.

        Workers use this to know which `ghost_<slot>` clients in the
        inference bundle correspond to real ghost weights vs placeholders.
        Only slots in this set are valid targets for GHOSTS opponent
        routing in `apply_opp_type_to_pair`.
        """
        self._active_ghost_slots = set(slots)

    def set_active_exploiter_slots(self, slots: List[int]) -> None:
        """Update the set of populated exploiter snapshot slots from a
        trainer broadcast. Workers use this to know which
        `exploiter_snap_<slot>` clients in the inference bundle
        correspond to real weights vs placeholders.
        """
        self._active_exploiter_slots = set(slots)

    def sample_team(self, battle_format: str) -> str:
        return self.team_repo.sample_team(
            battle_format,
            subdirectory=self.opponent_team_subdirectories.get(battle_format),
        )

    def get_agent_team(self, battle_format: str) -> str:
        """Return the agent's team for the given format, shuffled.

        Uses fixed team(s) from ``self._agent_teams_by_format[battle_format]``
        when any are loaded for this format; otherwise falls back to
        sampling from the opponent team pool for that format.
        """
        teams = self._agent_teams_by_format.get(battle_format, [])
        if teams:
            return self.team_repo._shuffle_team_order(random.choice(teams))
        return self.sample_team(battle_format)

    def _make_baseline_pool(
        self,
        opp_type: str,
        player_cls: Type[_P],
        role: str,
        pair_formats: List[str],
    ) -> List[_P]:
        """Create one ``player_cls`` per entry in ``pair_formats``, each pinned
        to that pair's format and team. Returns [] when curriculum weight is 0.

        Pools are not backfilled when `update_curriculum` later raises a
        weight from 0 — see the note in `update_curriculum` about
        `|nametaken|` collisions on rebuild.
        """
        if self.curriculum.get(opp_type, 0) <= 0:
            return []
        return [
            player_cls(
                battle_format=fmt,
                account_configuration=AccountConfiguration(
                    self._account_name(role, i), None
                ),
                server_configuration=self.server_config,
                team=self.sample_team(fmt),
            )
            for i, fmt in enumerate(pair_formats)
        ]

    def create_agents(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[List[RLTrajectoryPlayer], List[RLTrajectoryPlayer], List[MaxDamagePlayer]]:
        """Create and cache all worker-local battle participants.

        Always builds `num_pairs` mirrored `RLTrajectoryPlayer` pairs:
        players ship trajectories (`trajectory_queue` attached), opponents
        do not. The opponent's `inference_client` is hot-swapped per
        batch in `apply_opp_type_to_pair` to cover every neural opponent
        type (self_play / bc / ghosts / exploiters / train_exploiter).

        Heuristic baseline pools (`MaxDamage`, `Random`, `MaxBasePower`,
        `SimpleHeuristic`) are each `num_pairs` long, created only when
        the curriculum sampled them at startup. VGCBench is external —
        workers `/challenge` the usernames in `external_vgcbench_usernames`
        instead of holding local Player objects.
        """
        # Conditionally pass `max_concurrent_battles` so when the config
        # leaves it None we don't override poke-env's default of 1.
        extra_player_kwargs: Dict[str, Any] = {}
        if self.max_concurrent_battles_per_player is not None:
            extra_player_kwargs["max_concurrent_battles"] = (
                self.max_concurrent_battles_per_player
            )

        # Centralized inference: the player owns an `inference_client` and
        # submits requests to the trainer-side InferenceService. "main" is
        # required; `apply_opp_type_to_pair` can later re-point the
        # opponent at optional slots ("bc", ghost/exploiter) via .get().
        main_kwargs: Dict[str, Any] = {
            "inference_client": self.worker_inference_clients["main"]
        }

        # Apportion formats across pairs deterministically (see Task 3 plan).
        self.pair_formats = largest_remainder_apportionment(
            num_items=num_pairs, weights=self.battle_formats
        )
        missing_formats = set(self.battle_formats) - set(self.pair_formats)
        if missing_formats:
            logger.warning(
                "Worker %s has only %d pairs but %d configured formats; %s got zero pairs. "
                "Recommend num_pairs >= len(battle_formats) for stable distributions.",
                self.worker_id,
                num_pairs,
                len(self.battle_formats),
                sorted(missing_formats),
            )

        self.players = []
        self.opponents = []
        for i, fmt in enumerate(self.pair_formats):
            self.players.append(
                RLTrajectoryPlayer(
                    account_configuration=AccountConfiguration(
                        self._account_name("Self", i), None
                    ),
                    server_configuration=self.server_config,
                    trajectory_queue=local_traj_queue,
                    battle_format=fmt,
                    team=self.get_agent_team(fmt),
                    worker_id=self.worker_id,
                    embedder=self.embedder,
                    max_battle_steps=self.max_battle_steps,
                    opponent_type=OpponentPool.SELF_PLAY,
                    **main_kwargs,
                    **extra_player_kwargs,
                )
            )
            self.opponents.append(
                RLTrajectoryPlayer(
                    account_configuration=AccountConfiguration(
                        self._account_name("Opp", i), None
                    ),
                    server_configuration=self.server_config,
                    trajectory_queue=None,
                    battle_format=fmt,
                    team=self.sample_team(fmt),
                    worker_id=self.worker_id,
                    embedder=self.embedder,
                    max_battle_steps=self.max_battle_steps,
                    **main_kwargs,
                    **extra_player_kwargs,
                )
            )

        self.max_damage_opponents = self._make_baseline_pool(
            OpponentPool.MAX_DAMAGE, MaxDamagePlayer, "MaxD", self.pair_formats
        )
        self.random_baseline_opponents = self._make_baseline_pool(
            OpponentPool.RANDOM_BASELINE, RandomPlayer, "Rand", self.pair_formats
        )
        self.max_base_power_baseline_opponents = self._make_baseline_pool(
            OpponentPool.MAX_BASE_POWER_BASELINE,
            MaxBasePowerPlayer,
            "MaxB",
            self.pair_formats,
        )
        self.simple_heuristic_baseline_opponents = self._make_baseline_pool(
            OpponentPool.SIMPLE_HEURISTIC_BASELINE,
            SimpleHeuristicsPlayer,
            "Heur",
            self.pair_formats,
        )

        return self.players, self.opponents, self.max_damage_opponents

    def sample_opponent_type(self) -> str:
        rand = random.random()
        cumulative = 0.0

        for opp_type, prob in self.curriculum.items():
            cumulative += prob
            if rand < cumulative:
                return opp_type

        return OpponentPool.SELF_PLAY

    def _swap_to(
        self,
        slot: RLTrajectoryPlayer,
        centralized_name: str,
    ) -> bool:
        """Re-point one player or opponent slot at a different model.

        Returns True if the named client is registered in the bundle,
        False otherwise (caller should fall back to self-play / main).
        """
        client = self.worker_inference_clients.get(centralized_name)
        if client is None:
            return False
        slot.inference_client = client
        return True

    def sample_opp_type_for(self, player: RLTrajectoryPlayer) -> str:
        """Sample an opponent type from the curriculum, resolve it against
        currently-available pool state, and align `player.inference_client`.

        Falls back to `SELF_PLAY` when:
        - the sampled type is unrecognized (defensive guard),
        - `EXPLOITERS` was sampled but no slots are populated,
        - `GHOSTS` was sampled but no slots are populated, or
        - `TRAIN_EXPLOITER` was sampled but no "exploiter" client is
          registered in the worker's bundle.

        Aligns `player.inference_client` to the chosen type — "exploiter"
        for `TRAIN_EXPLOITER` so the player's actions train the exploiter
        learner; "main" for everything else. The reset runs every batch
        (including heuristic and VGCBench) so a prior `TRAIN_EXPLOITER`
        batch never leaks its "exploiter" client into a subsequent main
        trajectory.

        The opponent's client (BIP only) is handled separately by
        `apply_opp_type_to_pair`.
        """
        selected_type = self.sample_opponent_type()

        if selected_type not in OpponentPool.KNOWN_OPP_TYPES:
            selected_type = OpponentPool.SELF_PLAY
        elif selected_type == OpponentPool.EXPLOITERS and not self._active_exploiter_slots:
            # Curriculum normally guards this via _opponent_available; this
            # is the defensive fallback.
            selected_type = OpponentPool.SELF_PLAY
        elif selected_type == OpponentPool.GHOSTS and not self._active_ghost_slots:
            selected_type = OpponentPool.SELF_PLAY

        if selected_type == OpponentPool.TRAIN_EXPLOITER:
            if not self._swap_to(player, "exploiter"):
                # Co-training not provisioned (race during ramp-up). Fall
                # back to self-play and reset player to main.
                selected_type = OpponentPool.SELF_PLAY
                self._swap_to(player, "main")
        else:
            self._swap_to(player, "main")

        player.opponent_type = selected_type
        return selected_type

    def apply_opp_type_to_pair(
        self,
        player: RLTrajectoryPlayer,
        opponent: RLTrajectoryPlayer,
        opp_type: str,
    ) -> str:
        """Re-point the BIP opponent's inference client to match
        `opp_type`. Called only for opp types that battle the BIP
        opponent pool (`SELF_PLAY` / `BC_PLAYER` / `EXPLOITERS` /
        `GHOSTS` / `TRAIN_EXPLOITER`). Heuristic and VGCBench types
        don't use the BIP so they skip this step.

        The player's client is already aligned by `sample_opp_type_for`;
        this method only touches the player on a TRAIN_EXPLOITER opponent
        fall-back, where the victim client is missing and we revert the
        player from "exploiter" back to "main".

        May fall back to `SELF_PLAY` if a swap target client is missing
        from the worker's bundle. Updates `player.opponent_type` on fall
        back; returns the final opp type.
        """
        final_type = opp_type
        opponent_swapped = False

        if opp_type == OpponentPool.BC_PLAYER:
            opponent_swapped = self._swap_to(opponent, "bc")
            if not opponent_swapped:
                final_type = OpponentPool.SELF_PLAY
        elif opp_type == OpponentPool.EXPLOITERS:
            slot = random.choice(tuple(self._active_exploiter_slots))
            opponent_swapped = self._swap_to(opponent, f"exploiter_snap_{slot}")
            if not opponent_swapped:
                final_type = OpponentPool.SELF_PLAY
        elif opp_type == OpponentPool.GHOSTS:
            slot = random.choice(tuple(self._active_ghost_slots))
            opponent_swapped = self._swap_to(opponent, f"ghost_{slot}")
            if not opponent_swapped:
                final_type = OpponentPool.SELF_PLAY
        elif opp_type == OpponentPool.TRAIN_EXPLOITER:
            opponent_swapped = self._swap_to(opponent, "victim")
            if not opponent_swapped:
                # No "victim" client. Player was set to "exploiter" by
                # sample_opp_type_for; revert to "main" for the SELF_PLAY
                # fall-back so the trajectory routes as main correctly.
                final_type = OpponentPool.SELF_PLAY
                self._swap_to(player, "main")

        if not opponent_swapped:
            self._swap_to(opponent, "main")

        if final_type != opp_type:
            player.opponent_type = final_type
        return final_type

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
            opp_type = self.sample_opp_type_for(player)

            if (
                opp_type == OpponentPool.VGC_BENCH_BASELINE
                and self.external_vgcbench_usernames
            ):
                username = self.external_vgcbench_usernames[
                    (self._batch_count + i) % len(self.external_vgcbench_usernames)
                ]
                task = player.send_challenges(username, num_battles_per_pair)
            elif opp_type == OpponentPool.MAX_DAMAGE and self.max_damage_opponents:
                target_opponent = self.max_damage_opponents[
                    i % len(self.max_damage_opponents)
                ]
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif (
                opp_type == OpponentPool.RANDOM_BASELINE and self.random_baseline_opponents
            ):
                target_opponent = self.random_baseline_opponents[
                    i % len(self.random_baseline_opponents)
                ]
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif (
                opp_type == OpponentPool.MAX_BASE_POWER_BASELINE
                and self.max_base_power_baseline_opponents
            ):
                target_opponent = self.max_base_power_baseline_opponents[
                    i % len(self.max_base_power_baseline_opponents)
                ]
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif (
                opp_type == OpponentPool.SIMPLE_HEURISTIC_BASELINE
                and self.simple_heuristic_baseline_opponents
            ):
                target_opponent = self.simple_heuristic_baseline_opponents[
                    i % len(self.simple_heuristic_baseline_opponents)
                ]
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            else:
                # Neural opp type (SELF_PLAY / BC / EXPLOITERS / GHOSTS /
                # TRAIN_EXPLOITER). Configure the BIP opponent's inference
                # client and battle it; `apply_opp_type_to_pair` may
                # further fall back to SELF_PLAY if a client is missing.
                opponent = self.opponents[i]
                opp_type = self.apply_opp_type_to_pair(player, opponent, opp_type)
                task = player.battle_against(opponent, n_battles=num_battles_per_pair)

            batch_opponent_types.append(opp_type)
            tasks.append(task)

        return tasks, batch_opponent_types

    def randomize_all_teams(self) -> None:
        """Resample teams for all participants before the next batch.

        Each slot is pinned to its pair's format (see `self.pair_formats`).
        Players use the fixed agent team(s) for that format if configured,
        otherwise sample from the format's opponent team pool. Baseline
        pools mirror the same per-slot format pinning.
        """
        for i, player in enumerate(self.players):
            fmt = self.pair_formats[i]
            player._team = ConstantTeambuilder(self.get_agent_team(fmt))

        for i, opponent in enumerate(self.opponents):
            fmt = self.pair_formats[i]
            opponent._team = ConstantTeambuilder(self.sample_team(fmt))

        for i, md_opp in enumerate(self.max_damage_opponents):
            fmt = self.pair_formats[i]
            md_opp._team = ConstantTeambuilder(self.sample_team(fmt))

        for i, random_opp in enumerate(self.random_baseline_opponents):
            fmt = self.pair_formats[i]
            random_opp._team = ConstantTeambuilder(self.sample_team(fmt))

        for i, maxbp_opp in enumerate(self.max_base_power_baseline_opponents):
            fmt = self.pair_formats[i]
            maxbp_opp._team = ConstantTeambuilder(self.sample_team(fmt))

        for i, heuristic_opp in enumerate(self.simple_heuristic_baseline_opponents):
            fmt = self.pair_formats[i]
            heuristic_opp._team = ConstantTeambuilder(self.sample_team(fmt))

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

        # Force a GC cycle to reclaim memory from the old objects.
        gc.collect()

        # Bump generation so newly-created players use unique usernames.
        self._rebuild_generation += 1

        self.create_agents(num_pairs, local_traj_queue)

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


__all__ = [
    "OpponentPool",
    "WorkerOpponentFactory",
]
