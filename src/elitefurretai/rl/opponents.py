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
- random    — Random legal action.
- max_base_power     — Heuristic: highest base-power move regardless of target.
- simple_heuristic   — poke-env's built-in SimpleHeuristicsPlayer.
- vgc_bench — External SB3-trained agent we compare against.
"""

import asyncio
import gc
import logging
import os
import queue
import random
from collections import OrderedDict, defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

import numpy as np
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder

from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.config import open_team_sheets_for_battle
from elitefurretai.rl.inference_worker import WorkerInferenceClients
from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer
from elitefurretai.rl.rl_utils import list_pt_files, normalize_curriculum

if TYPE_CHECKING:
    from elitefurretai.rl.config import AdaptiveAxisConfig

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
    RANDOM = "random"
    MAX_BASE_POWER = "max_base_power"
    SIMPLE_HEURISTIC = "simple_heuristic"
    VGC_BENCH = "vgc_bench"
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
        RANDOM,
        MAX_BASE_POWER,
        SIMPLE_HEURISTIC,
        VGC_BENCH,
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
        team_repo: Optional["TeamRepo"] = None,
        battle_formats: Optional[Dict[str, float]] = None,
        opponent_team_subdirectories: Optional[Dict[str, Optional[str]]] = None,
        adaptive_team_axis: Optional["AdaptiveAxisConfig"] = None,
        adaptive_agent_axis: Optional["AdaptiveAxisConfig"] = None,
    ):
        # Resolve adaptive-axis configs to their defaults. Local import
        # keeps the dependency from `opponents` -> `config` one-directional
        # at call time (rather than at module import time), which is a
        # convention this file already follows for other RL-side imports.
        from elitefurretai.rl.config import AdaptiveAxisConfig

        if adaptive_team_axis is None:
            adaptive_team_axis = AdaptiveAxisConfig.team_axis_defaults()
        if adaptive_agent_axis is None:
            adaptive_agent_axis = AdaptiveAxisConfig.agent_axis_defaults()
        self.adaptive_team_axis = adaptive_team_axis
        self.adaptive_agent_axis = adaptive_agent_axis

        self.max_ghosts = max_ghosts
        self.max_exploiter_models = max_exploiter_models
        self.tracking_window = tracking_window
        self.curriculum = curriculum

        # Master switch for the in-process exploiter pipeline. train.py only
        # provisions the exploiter learner and its "exploiter"/"victim"
        # inference clients when the configured train_exploiter weight is
        # positive. Captured here from the original config because the
        # adaptive update rewrites self.curriculum, which would otherwise
        # erase the base-zero signal and let the agent axis leak weight onto
        # a slot that has no learner behind it.
        self._train_exploiter_enabled = (
            curriculum.get(OpponentPool.TRAIN_EXPLOITER, 0.0) > 0
        )

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
                OpponentPool.RANDOM: [],
                OpponentPool.MAX_BASE_POWER: [],
                OpponentPool.SIMPLE_HEURISTIC: [],
                OpponentPool.VGC_BENCH: [],
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
        # Agent-axis EWMA store: maps opp_type → (decayed_wins, decayed_n).
        # `win_rate_tracking` (deque) stays for the metrics emitter so the
        # win_rate/<opp> Wandb panels keep showing a sliding-window rate;
        # the adaptive update (Phase 5.3) reads from this EWMA pair instead.
        self.agent_win_rates: Dict[str, Tuple[float, float]] = {
            opp_type: (0.0, 0.0) for opp_type in self.win_rates
        }
        self.total_battles_tracked = 0
        self.total_forfeits_tracked = 0

        # ── team-axis adaptive curriculum state ──
        # Keep scalar copies for the fields that are read repeatedly on hot
        # paths inside this class (warm-up latch check, per-team floor pass,
        # record_battle_result gate). Other params (half_life, pfsp/weakness
        # mixing) are pulled directly from `self.adaptive_team_axis` at
        # call time inside update_team_distribution / record_battle_result.
        self.team_axis_enabled = adaptive_team_axis.enabled
        self.team_warmup_threshold = adaptive_team_axis.min_samples
        self.team_per_team_floor = adaptive_team_axis.per_key_floor

        self.known_teams: Dict[str, List[str]] = {}
        self.team_win_rates: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.team_sample_counts: Dict[str, Dict[str, int]] = {}
        self._team_axis_warm: Dict[str, bool] = {}
        # Format keys are tracked regardless of team_axis_enabled so the
        # disabled-path of update_team_distribution can still enumerate
        # the configured formats (returning {fmt: None} for each).
        self._team_axis_format_keys: List[str] = (
            list(battle_formats.keys()) if battle_formats else []
        )

        if self.team_axis_enabled and team_repo is not None and battle_formats:
            subs = opponent_team_subdirectories or {}
            for fmt in battle_formats:
                names = sorted(team_repo.get_all(fmt).keys())
                # Apply subdirectory filter symmetric with sample_team.
                sub = subs.get(fmt)
                if sub is not None:
                    sub = sub.replace(os.sep, "/")
                    names = [n for n in names if n.startswith(sub + "/") or n == sub]
                self.known_teams[fmt] = names
                self.team_win_rates[fmt] = {n: (0.0, 0.0) for n in names}
                self.team_sample_counts[fmt] = {n: 0 for n in names}
                self._team_axis_warm[fmt] = False

        if self.team_axis_enabled:
            for fmt, teams in self.known_teams.items():
                if teams and len(teams) * self.team_per_team_floor >= 1.0:
                    logger.warning(
                        "OpponentPool team-axis: format %r has %d teams * "
                        "team_per_team_floor=%s >= 1.0; per-team floor will "
                        "effectively pin all teams uniform. Consider lowering "
                        "team_per_team_floor (e.g., to %.4f) or using a "
                        "subdirectory filter to reduce team count.",
                        fmt,
                        len(teams),
                        self.team_per_team_floor,
                        1.0 / (3 * len(teams)),
                    )

    def _opponent_available(self, opponent_type: str) -> bool:
        if opponent_type == OpponentPool.BC_PLAYER:
            return self.bc_model_path is not None
        if opponent_type == OpponentPool.EXPLOITERS:
            return len(self.slot_for_exploiter_path) > 0
        if opponent_type == OpponentPool.GHOSTS:
            return len(self.slot_for_ghost_path) > 0
        if opponent_type == OpponentPool.TRAIN_EXPLOITER:
            return self._train_exploiter_enabled
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
        battle_format: Optional[str] = None,
        team_name: Optional[str] = None,
    ) -> None:
        win_value = 1.0 if won else 0.0
        self.win_rates[opponent_type].append(win_value)
        self.win_rate_tracking[opponent_type].append(win_value)

        if battle_length > 0:
            self.battle_length_tracking[opponent_type].append(battle_length)

        self.total_battles_tracked += 1
        if forfeited:
            self.total_forfeits_tracked += 1

        # Agent-axis EWMA update. Same shape as the team-axis update
        # below: decay = 0.5 ** (1 / half_life), then add this battle's
        # contribution. Phase 5.3 reads agent_win_rates via the
        # adaptive_score primitive in update_curriculum. Forfeits are
        # included here to match the sliding-window deque update above,
        # which also does not gate on `forfeited`; both signals stay
        # consistent for the metrics emitter.
        agent_half_life = self.adaptive_agent_axis.half_life
        decay = 0.5 ** (1.0 / max(agent_half_life, 1e-9))
        prev_wins, prev_n = self.agent_win_rates.get(opponent_type, (0.0, 0.0))
        new_wins = prev_wins * decay + win_value
        new_n = prev_n * decay + 1.0
        self.agent_win_rates[opponent_type] = (new_wins, new_n)

        # ── per-(format, team) EWMA update ──
        if (
            self.adaptive_team_axis.enabled
            and battle_format is not None
            and team_name is not None
            and not forfeited
            and battle_format in self.team_win_rates
            and team_name in self.team_win_rates[battle_format]
        ):
            decay = 0.5 ** (1.0 / max(self.adaptive_team_axis.half_life, 1e-9))
            prev_wins, prev_n = self.team_win_rates[battle_format][team_name]
            new_wins = prev_wins * decay + (1.0 if won else 0.0)
            new_n = prev_n * decay + 1.0
            self.team_win_rates[battle_format][team_name] = (new_wins, new_n)
            self.team_sample_counts[battle_format][team_name] += 1

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

    def update_curriculum(self):
        """Adapt agent-axis curriculum weights using the shared
        adaptive primitive (`rl_utils.adaptive_distribution`).

        See `AdaptiveAxisConfig.agent_axis_defaults` for the parameter
        choices that reproduce the pre-unification behavior. The only
        substantive change from the legacy code path is that smoothing
        now uses EWMA (configured by `half_life`) instead of a fixed
        sliding window; for a stationary signal the long-run estimate
        is unchanged.
        """
        from elitefurretai.rl.rl_utils import adaptive_distribution, adaptive_score

        self._load_exploiter_models()
        self._load_ghosts()

        cfg = self.adaptive_agent_axis
        if not cfg.enabled:
            return

        base = self.curriculum.copy()

        scores: Dict[str, float] = {}
        for opp_type, base_weight in base.items():
            if not self._opponent_available(opp_type):
                scores[opp_type] = 0.0
                continue
            wins, n = self.agent_win_rates.get(opp_type, (0.0, 0.0))
            if n < cfg.min_samples:
                scores[opp_type] = max(base_weight, 1e-2)
                continue
            scores[opp_type] = max(
                1e-2,
                adaptive_score(
                    wins=wins,
                    n=n,
                    prior_alpha=cfg.prior_alpha,
                    prior_beta=cfg.prior_beta,
                    pfsp_mix=cfg.pfsp_mix,
                    weakness_mix=cfg.weakness_mix,
                    weakness_exponent=cfg.weakness_exponent,
                    target_win_rate=cfg.target_win_rate,
                ),
            )

        floors: Dict[str, float] = {}
        if self._opponent_available(OpponentPool.SELF_PLAY):
            floors[OpponentPool.SELF_PLAY] = 0.20
        if self._opponent_available(OpponentPool.BC_PLAYER):
            floors[OpponentPool.BC_PLAYER] = 0.10
        if self._opponent_available(OpponentPool.GHOSTS):
            floors[OpponentPool.GHOSTS] = 0.10

        scores = {k: v for k, v in scores.items() if self._opponent_available(k)}
        if not scores:
            self.curriculum = {OpponentPool.SELF_PLAY: 1.0}
            return

        new_curriculum = adaptive_distribution(
            scores,
            base=base,
            base_blend=cfg.base_blend,
            floors=floors,
        )
        self.curriculum = normalize_curriculum(new_curriculum)

    def update_team_distribution(
        self,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """Recompute the per-format team sampling distribution using the
        shared adaptive_distribution primitive.

        See AdaptiveAxisConfig.team_axis_defaults for the parameter
        choices that drive the team-axis behavior.
        """
        from elitefurretai.rl.rl_utils import adaptive_distribution, adaptive_score

        result: Dict[str, Optional[Dict[str, float]]] = {
            fmt: None for fmt in self._team_axis_format_keys
        }
        if not self.adaptive_team_axis.enabled:
            return result

        cfg = self.adaptive_team_axis
        for fmt, teams in self.known_teams.items():
            if not self._team_axis_warm.get(fmt, False):
                warm = all(
                    self.team_sample_counts[fmt][t] >= cfg.min_samples for t in teams
                )
                if warm:
                    self._team_axis_warm[fmt] = True
                else:
                    result[fmt] = None
                    continue

            scores: Dict[str, float] = {}
            for t in teams:
                wins, n = self.team_win_rates[fmt][t]
                scores[t] = adaptive_score(
                    wins=wins,
                    n=n,
                    prior_alpha=cfg.prior_alpha,
                    prior_beta=cfg.prior_beta,
                    pfsp_mix=cfg.pfsp_mix,
                    weakness_mix=cfg.weakness_mix,
                    weakness_exponent=cfg.weakness_exponent,
                    target_win_rate=cfg.target_win_rate,
                )

            floors = (
                {t: cfg.per_key_floor for t in teams} if cfg.per_key_floor > 0 else None
            )
            result[fmt] = adaptive_distribution(
                scores,
                base=None,
                base_blend=0.0,
                floors=floors,
            )

        return result


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
        open_team_sheets: str = "off",
    ):
        self.team_repo = team_repo
        self.battle_formats = dict(battle_formats)
        self.opponent_team_subdirectories = dict(opponent_team_subdirectories)
        # ── per-format biased team distribution from broadcast ──
        # None or absent value for a format → fall back to uniform.
        self.team_distribution_by_format: Dict[str, Optional[Dict[str, float]]] = {}
        self.server_config = server_config
        self.worker_inference_clients = worker_inference_clients
        self.max_concurrent_battles_per_player = max_concurrent_battles_per_player
        self.curriculum = curriculum or {OpponentPool.SELF_PLAY: 1.0}
        self.embedder = embedder
        self.worker_id = worker_id
        self.run_id = run_id
        self.max_battle_steps = max_battle_steps
        # OTS mode ("off"|"on"|"mixed"); resolved per battle in prepare_batch_tasks.
        # _ots_rng makes the "mixed" per-batch coin flip reproducible per worker.
        self.open_team_sheets = open_team_sheets
        self._ots_rng = random.Random(f"{run_id}:{worker_id}:ots")

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
        self.random_opponents: List[RandomPlayer] = []
        self.max_base_power_opponents: List[MaxBasePowerPlayer] = []
        self.simple_heuristic_opponents: List[Player] = []
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

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
        team_distribution_by_format: Optional[
            Dict[str, Optional[Dict[str, float]]]
        ] = None,
    ) -> None:
        """Update worker-local curriculum and refresh dependent opponent pools.

        team_distribution_by_format: per-format biased team sampling
            distributions broadcast from the trainer. May
            contain None values for formats still in warm-up;
            sample_team falls back to uniform for those.
        """
        self.curriculum = normalize_curriculum(curriculum)
        if team_distribution_by_format is not None:
            self.team_distribution_by_format = dict(team_distribution_by_format)

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

    def sample_team(
        self,
        battle_format: str,
        biased: bool = True,
    ) -> Tuple[str, str]:
        """Return (team_string, team_name) for the given battle format.

        biased=True (default): if the trainer has broadcast a non-None
            distribution for ``battle_format``, sample a name from it
            via numpy.random.choice and look up the corresponding
            team string. Otherwise (no distribution set, or value is
            None during warm-up) fall back to the uniform path.

        biased=False: always uniform via team_repo.sample_team_name.
            Reserved for future eval-at-checkpoint code paths that
            want the natural team distribution.
        """
        dist = self.team_distribution_by_format.get(battle_format) if biased else None
        if dist:
            names = list(dist.keys())
            weights = list(dist.values())
            name = str(np.random.choice(names, p=weights))
        else:
            name = self.team_repo.sample_team_name(
                battle_format,
                subdirectory=self.opponent_team_subdirectories.get(battle_format),
            )
        team_string = self.team_repo.get(battle_format, name)
        return self.team_repo._shuffle_team_order(team_string), name

    def get_agent_team(self, battle_format: str) -> Tuple[str, str]:
        """Return ``(team_string, team_name)`` for the agent in this format.

        Uses fixed team(s) from ``self._agent_teams_by_format[battle_format]``
        when any are loaded for this format; otherwise falls back to
        sampling from the opponent team pool for that format.

        For the fixed-team path the team name is ``"agent_fixed"`` —
        the file-level identity is irrelevant because the trajectory's
        per-team adaptive curriculum only tracks teams the pool samples.
        """
        teams = self._agent_teams_by_format.get(battle_format, [])
        if teams:
            return (
                self.team_repo._shuffle_team_order(random.choice(teams)),
                "agent_fixed",
            )
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
        # Heuristic baselines don't produce trajectories, so the team
        # name is discarded.
        return [
            player_cls(
                battle_format=fmt,
                account_configuration=AccountConfiguration(
                    self._account_name(role, i), None
                ),
                server_configuration=self.server_config,
                team=self.sample_team(fmt)[0],
                # Initial value; prepare_batch_tasks sets the real per-batch
                # accept on whichever baseline a player actually battles.
                accept_open_team_sheet=(self.open_team_sheets == "on"),
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
        extra_player_kwargs: Dict[str, Any] = {
            # Initial value; prepare_batch_tasks sets the real per-batch accept
            # on both the player and its opponent before each batch's battles.
            "accept_open_team_sheet": self.open_team_sheets == "on",
        }
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
            player_team_string, player_team_name = self.get_agent_team(fmt)
            player = RLTrajectoryPlayer(
                account_configuration=AccountConfiguration(
                    self._account_name("Self", i), None
                ),
                server_configuration=self.server_config,
                trajectory_queue=local_traj_queue,
                battle_format=fmt,
                team=player_team_string,
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
                opponent_type=OpponentPool.SELF_PLAY,
                **main_kwargs,
                **extra_player_kwargs,
            )
            player._pending_team_name = player_team_name
            self.players.append(player)

            opp_team_string, opp_team_name = self.sample_team(fmt)
            opponent = RLTrajectoryPlayer(
                account_configuration=AccountConfiguration(
                    self._account_name("Opp", i), None
                ),
                server_configuration=self.server_config,
                trajectory_queue=None,
                battle_format=fmt,
                team=opp_team_string,
                worker_id=self.worker_id,
                embedder=self.embedder,
                max_battle_steps=self.max_battle_steps,
                **main_kwargs,
                **extra_player_kwargs,
            )
            opponent._pending_team_name = opp_team_name
            self.opponents.append(opponent)

        self.max_damage_opponents = self._make_baseline_pool(
            OpponentPool.MAX_DAMAGE, MaxDamagePlayer, "MaxD", self.pair_formats
        )
        self.random_opponents = self._make_baseline_pool(
            OpponentPool.RANDOM, RandomPlayer, "Rand", self.pair_formats
        )
        self.max_base_power_opponents = self._make_baseline_pool(
            OpponentPool.MAX_BASE_POWER,
            MaxBasePowerPlayer,
            "MaxB",
            self.pair_formats,
        )
        self.simple_heuristic_opponents = self._make_baseline_pool(
            OpponentPool.SIMPLE_HEURISTIC,
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

        # Open Team Sheets: resolve the per-battle accept. "mixed" rolls once
        # per batch so every (non-vgc_bench) battle this batch shares the same
        # regime — both sides and any SHARED heuristic opponent agree, so a
        # mismatch never drops a battle. vgc_bench is always ON, so the agent
        # facing it is forced ON for that battle.
        batch_ots_roll = self.open_team_sheets == "mixed" and self._ots_rng.random() < 0.5

        for i, player in enumerate(self.players):
            opp_type = self.sample_opp_type_for(player)
            accept_ots = open_team_sheets_for_battle(
                self.open_team_sheets,
                is_vgc_bench=(opp_type == OpponentPool.VGC_BENCH),
                mixed_roll=batch_ots_roll,
            )
            player._accept_open_team_sheet = accept_ots

            if opp_type == OpponentPool.VGC_BENCH and self.external_vgcbench_usernames:
                username = self.external_vgcbench_usernames[
                    (self._batch_count + i) % len(self.external_vgcbench_usernames)
                ]
                task = player.send_challenges(username, num_battles_per_pair)
            elif opp_type == OpponentPool.MAX_DAMAGE and self.max_damage_opponents:
                target_opponent = self.max_damage_opponents[
                    i % len(self.max_damage_opponents)
                ]
                target_opponent._accept_open_team_sheet = accept_ots
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif opp_type == OpponentPool.RANDOM and self.random_opponents:
                target_opponent = self.random_opponents[i % len(self.random_opponents)]
                target_opponent._accept_open_team_sheet = accept_ots
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif opp_type == OpponentPool.MAX_BASE_POWER and self.max_base_power_opponents:
                target_opponent = self.max_base_power_opponents[
                    i % len(self.max_base_power_opponents)
                ]
                target_opponent._accept_open_team_sheet = accept_ots
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            elif (
                opp_type == OpponentPool.SIMPLE_HEURISTIC
                and self.simple_heuristic_opponents
            ):
                target_opponent = self.simple_heuristic_opponents[
                    i % len(self.simple_heuristic_opponents)
                ]
                target_opponent._accept_open_team_sheet = accept_ots
                task = player.battle_against(
                    target_opponent, n_battles=num_battles_per_pair
                )
            else:
                # Neural opp type (SELF_PLAY / BC / EXPLOITERS / GHOSTS /
                # TRAIN_EXPLOITER). Configure the BIP opponent's inference
                # client and battle it; `apply_opp_type_to_pair` may
                # further fall back to SELF_PLAY if a client is missing.
                opponent = self.opponents[i]
                opponent._accept_open_team_sheet = accept_ots
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
            team_string, team_name = self.get_agent_team(fmt)
            player._team = ConstantTeambuilder(team_string)
            player._pending_team_name = team_name

        for i, opponent in enumerate(self.opponents):
            fmt = self.pair_formats[i]
            team_string, team_name = self.sample_team(fmt)
            opponent._team = ConstantTeambuilder(team_string)
            opponent._pending_team_name = team_name

        # Heuristic baselines below don't produce trajectories, so the
        # team name is discarded.
        for i, md_opp in enumerate(self.max_damage_opponents):
            fmt = self.pair_formats[i]
            team_string, _ = self.sample_team(fmt)
            md_opp._team = ConstantTeambuilder(team_string)

        for i, random_opp in enumerate(self.random_opponents):
            fmt = self.pair_formats[i]
            team_string, _ = self.sample_team(fmt)
            random_opp._team = ConstantTeambuilder(team_string)

        for i, maxbp_opp in enumerate(self.max_base_power_opponents):
            fmt = self.pair_formats[i]
            team_string, _ = self.sample_team(fmt)
            maxbp_opp._team = ConstantTeambuilder(team_string)

        for i, heuristic_opp in enumerate(self.simple_heuristic_opponents):
            fmt = self.pair_formats[i]
            team_string, _ = self.sample_team(fmt)
            heuristic_opp._team = ConstantTeambuilder(team_string)

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
            + self.random_opponents
            + self.max_base_power_opponents
            + self.simple_heuristic_opponents
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
        self.random_opponents.clear()
        self.max_base_power_opponents.clear()
        self.simple_heuristic_opponents.clear()

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

        for random_opp in self.random_opponents:
            random_opp.reset_battles()

        for maxbp_opp in self.max_base_power_opponents:
            maxbp_opp.reset_battles()

        for heuristic_opp in self.simple_heuristic_opponents:
            heuristic_opp.reset_battles()


__all__ = [
    "OpponentPool",
    "WorkerOpponentFactory",
]
