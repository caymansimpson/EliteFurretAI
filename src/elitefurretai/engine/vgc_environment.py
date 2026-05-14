"""
vgc_environment.py — Unified environment abstraction over both battle backends.

The worker process creates one VGCEnvironment, calls run_battle_batch() repeatedly,
and receives completed trajectories. The backend (Showdown websocket or Rust
in-process) is selected at construction and transparent to the caller.

Lifecycle:
  env = VGCEnvironment.from_config(config, worker_id, agent, embedder, ...)
  await env.setup()
  while training:
      result = await env.run_battle_batch(n_battles)
      env.update_weights(state_dict)
      env.update_curriculum(curriculum)
  await env.teardown()
"""

import asyncio
import logging
import os
import queue
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from elitefurretai.rl.inference_worker import (
        InferenceClient,
        WorkerInferenceClients,
    )

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import MaxBasePowerPlayer, RandomPlayer

from elitefurretai.engine.showdown_server_manager import (
    derive_external_vgcbench_username,
)
from elitefurretai.engine.sync_battle_driver import (
    SyncBaselineController,
    SyncPolicyPlayer,
    SyncRustBattleDriver,
)
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.config import RUST_ENGINE_BACKEND, RNaDConfig
from elitefurretai.rl.learners import (
    is_checkpoint_compatible_with_model_config,
    load_agent_from_checkpoint,
)
from elitefurretai.rl.opponents import (
    OpponentPool,
    SimpleHeuristicBaselineCls,
    WorkerOpponentFactory,
)
from elitefurretai.rl.players import MaxDamagePlayer, RNaDAgent

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from one call to VGCEnvironment.run_battle_batch()."""

    trajectories: List[Dict]
    sampled_types: List[str]  # opponent types sampled for this batch
    completed_counts: Dict[str, int]  # {opp_type: n} for completed trajectories
    had_timeout: bool
    timed_out_types: List[str]
    battles_completed: int  # total n_finished_battles this batch


# ── Shared helpers used by _RustBackend ──────────────────────────────────────


def _load_agent_team_sources(
    agent_team_path: Optional[str],
) -> Tuple[Optional[str], Optional[List[str]]]:
    if not agent_team_path:
        return None, None
    if os.path.isdir(agent_team_path):
        team_strings: List[str] = []
        for fname in sorted(os.listdir(agent_team_path)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(agent_team_path, fname)) as f:
                team_strings.append(f.read())
        return None, team_strings
    with open(agent_team_path) as f:
        return f.read(), None


def _build_team_supplier(
    *,
    team_repo: TeamRepo,
    battle_format: str,
    team_subdirectory: Optional[str],
    fixed_team: Optional[str] = None,
    team_options: Optional[List[str]] = None,
) -> Any:
    def supplier() -> str:
        if team_options:
            return team_repo._shuffle_team_order(random.choice(team_options))
        if fixed_team is not None:
            return team_repo._shuffle_team_order(fixed_team)
        return team_repo.sample_team(battle_format, subdirectory=team_subdirectory)

    return supplier


# ── Rust opponent pool (Rust backend only) ────────────────────────────────────


class _RustPolicyOpponentPool:
    def __init__(
        self,
        *,
        main_agent: RNaDAgent,
        bc_agent: Optional[RNaDAgent],
        config: RNaDConfig,
        team_repo: TeamRepo,
        model_config: Dict[str, Any],
        feature_set: str,
        device: str,
    ) -> None:
        self.main_agent = main_agent
        self.bc_agent = bc_agent
        self.config = config
        self.team_repo = team_repo
        self.model_config = model_config
        self.feature_set = feature_set
        self.device = device
        self.curriculum = dict(config.curriculum.curriculum_weights)
        self.temperature = config.temperature_at_step(0)
        self.top_p = config.exploration.top_p
        self._batch_count = 0
        self._baseline_controllers: Dict[str, SyncBaselineController] = {}

        self._static_policies: Dict[str, SyncPolicyPlayer] = {
            OpponentPool.SELF_PLAY: self._build_policy(main_agent, OpponentPool.SELF_PLAY),
        }
        if bc_agent is not None:
            self._static_policies[OpponentPool.BC_PLAYER] = self._build_policy(
                bc_agent, OpponentPool.BC_PLAYER
            )

        self._loaded_exploiters: Dict[str, RNaDAgent] = {}
        self._exploiter_policies: Dict[str, SyncPolicyPlayer] = {}
        self._active_exploiters: List[Tuple[float, str]] = []
        self._loaded_ghosts: Dict[str, RNaDAgent] = {}
        self._ghost_policies: Dict[str, SyncPolicyPlayer] = {}
        self._ghosts: List[Tuple[int, str]] = []
        self._refresh_exploiters()
        self._refresh_ghosts()
        self._init_baselines()

    def _init_baselines(self) -> None:
        server_config = ServerConfiguration("", "")
        account = AccountConfiguration(f"rust-baseline-{random.getrandbits(16):04x}", None)
        sample_team = self.team_repo.sample_team(
            self.config.curriculum.battle_format,
            subdirectory=self.config.curriculum.team_pool_path,
        )
        self._baseline_controllers[OpponentPool.MAX_DAMAGE] = SyncBaselineController(
            MaxDamagePlayer(
                battle_format=self.config.curriculum.battle_format,
                account_configuration=account,
                server_configuration=server_config,
                start_listening=False,
                team=sample_team,
            ),
            OpponentPool.MAX_DAMAGE,
        )
        self._baseline_controllers[OpponentPool.RANDOM_BASELINE] = SyncBaselineController(
            RandomPlayer(
                battle_format=self.config.curriculum.battle_format,
                account_configuration=account,
                server_configuration=server_config,
                start_listening=False,
                team=sample_team,
            ),
            OpponentPool.RANDOM_BASELINE,
        )
        self._baseline_controllers[OpponentPool.MAX_BASE_POWER_BASELINE] = (
            SyncBaselineController(
                MaxBasePowerPlayer(
                    battle_format=self.config.curriculum.battle_format,
                    account_configuration=account,
                    server_configuration=server_config,
                    start_listening=False,
                    team=sample_team,
                ),
                OpponentPool.MAX_BASE_POWER_BASELINE,
            )
        )
        heuristic_cls = SimpleHeuristicBaselineCls or MaxBasePowerPlayer
        self._baseline_controllers[OpponentPool.SIMPLE_HEURISTIC_BASELINE] = (
            SyncBaselineController(
                heuristic_cls(
                    battle_format=self.config.curriculum.battle_format,
                    account_configuration=account,
                    server_configuration=server_config,
                    start_listening=False,
                    team=sample_team,
                ),
                OpponentPool.SIMPLE_HEURISTIC_BASELINE,
            )
        )

    def _build_policy(self, agent: RNaDAgent, opponent_type: str) -> SyncPolicyPlayer:
        return SyncPolicyPlayer(
            agent,
            self.config.curriculum.battle_format,
            device=self.device,
            feature_set=self.feature_set,
            collect_trajectories=False,
            probabilistic=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_battle_steps=self.config.hardware.max_battle_steps,
            opponent_type=opponent_type,
        )

    def update_curriculum(self, curriculum: Dict[str, float]) -> None:
        total = float(sum(curriculum.values()))
        if total <= 0:
            self.curriculum = {OpponentPool.SELF_PLAY: 1.0}
            return
        self.curriculum = {
            key: float(value) / total for key, value in curriculum.items() if value > 0
        }
        if not self.curriculum:
            self.curriculum = {OpponentPool.SELF_PLAY: 1.0}

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        for policy in self._all_policies():
            policy.update_sampling(temperature=temperature, top_p=top_p)

    def sample_opponent(self) -> Tuple[str, str, Any]:
        self._batch_count += 1
        roll = random.random()
        cumulative = 0.0
        selected_type = OpponentPool.SELF_PLAY
        for opp_type, weight in self.curriculum.items():
            cumulative += weight
            if roll < cumulative:
                selected_type = opp_type
                break

        team_text = self.team_repo.sample_team(
            self.config.curriculum.battle_format,
            subdirectory=self.config.curriculum.team_pool_path,
        )

        if (
            selected_type == OpponentPool.BC_PLAYER
            and OpponentPool.BC_PLAYER in self._static_policies
        ):
            return selected_type, team_text, self._static_policies[OpponentPool.BC_PLAYER]
        if selected_type == OpponentPool.EXPLOITERS:
            exploiter_policy = self._sample_exploiter_policy()
            if exploiter_policy is not None:
                return selected_type, team_text, exploiter_policy
        if selected_type == OpponentPool.GHOSTS:
            ghost_policy = self._sample_ghost_policy()
            if ghost_policy is not None:
                return selected_type, team_text, ghost_policy
        if selected_type in self._baseline_controllers:
            return selected_type, team_text, self._baseline_controllers[selected_type]
        return (
            OpponentPool.SELF_PLAY,
            team_text,
            self._static_policies[OpponentPool.SELF_PLAY],
        )

    def _all_policies(self) -> List[SyncPolicyPlayer]:
        return [
            *self._static_policies.values(),
            *self._exploiter_policies.values(),
            *self._ghost_policies.values(),
        ]

    def _list_compatible_checkpoints(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        compatible_paths: List[str] = []
        for filename in os.listdir(directory):
            if not filename.endswith(".pt"):
                continue
            filepath = os.path.join(directory, filename)
            if is_checkpoint_compatible_with_model_config(filepath, self.model_config):
                compatible_paths.append(filepath)
        return compatible_paths

    def _refresh_exploiters(self) -> None:
        files = self._list_compatible_checkpoints(
            os.path.join(str(self.config.training.run_dir), "exploiters")
        )
        models = [
            (os.path.getmtime(filepath), filepath)
            for filepath in files
            if os.path.isfile(filepath)
        ]
        models.sort(key=lambda item: item[0], reverse=True)
        self._active_exploiters = models[: self.config.curriculum.max_exploiter_models]

    def _refresh_ghosts(self) -> None:
        files = self._list_compatible_checkpoints(
            os.path.join(str(self.config.training.run_dir), "ghosts")
        )
        models: List[Tuple[int, str]] = []
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (IndexError, ValueError):
                step = int(os.path.getmtime(filepath))
            models.append((step, filepath))
        models.sort(key=lambda item: item[0], reverse=True)
        self._ghosts = models[: self.config.curriculum.max_ghosts]

    def _get_cached_agent(
        self,
        filepath: str,
        *,
        loaded: Dict[str, RNaDAgent],
    ) -> RNaDAgent:
        if filepath in loaded:
            return loaded[filepath]

        loaded_model = load_agent_from_checkpoint(filepath, self.device)
        loaded[filepath] = loaded_model
        return loaded_model

    def _sample_exploiter_policy(self) -> Optional[SyncPolicyPlayer]:
        if not self._active_exploiters:
            return None
        _, filepath = random.choice(self._active_exploiters)
        agent = self._get_cached_agent(filepath, loaded=self._loaded_exploiters)
        if filepath not in self._exploiter_policies:
            self._exploiter_policies[filepath] = self._build_policy(
                agent, OpponentPool.EXPLOITERS
            )
        return self._exploiter_policies[filepath]

    def _sample_ghost_policy(self) -> Optional[SyncPolicyPlayer]:
        if not self._ghosts:
            return None
        _, filepath = random.choice(self._ghosts)
        agent = self._get_cached_agent(filepath, loaded=self._loaded_ghosts)
        if filepath not in self._ghost_policies:
            self._ghost_policies[filepath] = self._build_policy(agent, OpponentPool.GHOSTS)
        return self._ghost_policies[filepath]

    def set_ghost_paths(self, paths: List[str]) -> None:
        """Apply explicit ghost file list from learner broadcast (Option C)."""
        models: List[Tuple[int, str]] = []
        for p in paths:
            if not os.path.isfile(p):
                continue
            filename = os.path.basename(p)
            try:
                step = int(filename.split("_step_")[1].split(".pt")[0])
            except (IndexError, ValueError):
                step = int(os.path.getmtime(p))
            models.append((step, p))
        models.sort(key=lambda item: item[0], reverse=True)
        self._ghosts = models


# ── Public facade ─────────────────────────────────────────────────────────────


class VGCEnvironment:
    """
    Unified environment abstraction over both battle backends.

    The backend (Showdown websocket or Rust in-process) is selected at construction
    via config.hardware.battle_backend and is fully transparent to the caller.
    """

    def __init__(self, backend: "_BackendBase") -> None:
        self._backend = backend

    @classmethod
    def from_config(
        cls,
        config: RNaDConfig,
        worker_id: int,
        agent: Optional[RNaDAgent],
        model: Any,
        model_config: Dict[str, Any],
        embedder: Embedder,
        team_repo: TeamRepo,
        bc_agent: Optional[RNaDAgent] = None,
        server_port: Optional[int] = None,
        run_id: str = "",
        initial_curriculum: Optional[Dict[str, float]] = None,
        exploiter_agent: Optional[RNaDAgent] = None,
        victim_agent: Optional[RNaDAgent] = None,
        # M4 centralized inference: when set, the Showdown backend wires
        # BatchInferencePlayers to the trainer-side InferenceService
        # instead of the in-worker model. The Rust backend ignores it
        # (centralization not yet implemented there).
        main_inference_client: Optional["InferenceClient"] = None,
        # Step 3+: full bundle of per-model inference clients. Threaded
        # through to WorkerOpponentFactory so it can hot-swap
        # opponent.inference_client between main / bc / ghost slots.
        worker_inference_clients: Optional["WorkerInferenceClients"] = None,
    ) -> "VGCEnvironment":
        """Create a VGCEnvironment for the given config and worker."""
        if config.hardware.battle_backend == RUST_ENGINE_BACKEND:
            assert agent is not None, "Rust backend requires an in-worker agent"
            backend: _BackendBase = _RustBackend(
                config=config,
                worker_id=worker_id,
                agent=agent,
                model=model,
                model_config=model_config,
                embedder=embedder,
                team_repo=team_repo,
                bc_agent=bc_agent,
                initial_curriculum=initial_curriculum,
            )
        else:
            assert server_port is not None, "server_port required for Showdown backend"
            backend = _ShowdownBackend(
                config=config,
                worker_id=worker_id,
                agent=agent,
                model=model,
                model_config=model_config,
                embedder=embedder,
                team_repo=team_repo,
                bc_agent=bc_agent,
                server_port=server_port,
                run_id=run_id,
                initial_curriculum=initial_curriculum,
                exploiter_agent=exploiter_agent,
                victim_agent=victim_agent,
                main_inference_client=main_inference_client,
                worker_inference_clients=worker_inference_clients,
            )
        return cls(backend)

    async def setup(self) -> None:
        """Start websocket connections (Showdown) or initialize Rust driver."""
        await self._backend.setup()

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
        """Run n_battles; return a BatchResult with trajectories and timeout info."""
        return await self._backend.run_battle_batch(n_battles)

    def update_weights(self, state_dict: Dict) -> None:
        """Swap policy weights in-place; propagates to all active players."""
        self._backend.update_weights(state_dict)

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
    ) -> None:
        """Update opponent sampling distribution."""
        self._backend.update_curriculum(curriculum)

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        """Update exploration temperature and nucleus-sampling threshold."""
        self._backend.update_sampling(temperature, top_p)

    def update_exploiter_weights(self, state_dict: Dict) -> None:
        """Apply new exploiter weights to the worker-local exploiter agent."""
        self._backend.update_exploiter_weights(state_dict)

    def update_victim_weights(self, state_dict: Dict) -> None:
        """Apply new victim weights to the worker-local victim agent."""
        self._backend.update_victim_weights(state_dict)

    def get_curriculum(self) -> Dict[str, float]:
        """Return current curriculum weights (copy)."""
        return self._backend.get_curriculum()

    def get_diagnostics(self) -> Dict[str, float]:
        """Return backend-specific diagnostic metrics for logging."""
        return self._backend.get_diagnostics()

    async def rebuild(self) -> None:
        """Rebuild runtime agents after a timeout or stale battle state."""
        await self._backend.rebuild()

    def reset_battles(self) -> None:
        """Reset all battle state for the next batch."""
        self._backend.reset_battles()

    def get_unfinished_summary(self) -> Dict[str, int]:
        """Return {total_unfinished: n, ...} for any battles still in progress."""
        return self._backend.get_unfinished_summary()

    async def teardown(self) -> None:
        """Disconnect websockets / clean up Rust resources."""
        await self._backend.teardown()


# ── Backend protocol ──────────────────────────────────────────────────────────


class _BackendBase:
    """Abstract base defining the interface each concrete backend must implement."""

    async def setup(self) -> None:
        raise NotImplementedError

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
        raise NotImplementedError

    def update_weights(self, state_dict: Dict) -> None:
        raise NotImplementedError

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
    ) -> None:
        raise NotImplementedError

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        raise NotImplementedError

    def update_exploiter_weights(self, state_dict: Dict) -> None:
        """Apply broadcasted exploiter weights. No-op by default; backends
        that support in-process exploiter co-training override this."""
        pass

    def update_victim_weights(self, state_dict: Dict) -> None:
        """Apply broadcasted victim weights. No-op by default; backends
        that support in-process exploiter co-training override this."""
        pass

    def get_curriculum(self) -> Dict[str, float]:
        return {}

    def get_diagnostics(self) -> Dict[str, float]:
        return {}

    async def rebuild(self) -> None:
        pass

    def reset_battles(self) -> None:
        pass

    def get_unfinished_summary(self) -> Dict[str, int]:
        return {"total_unfinished": 0}

    async def teardown(self) -> None:
        pass


# ── Showdown websocket backend ────────────────────────────────────────────────


class _ShowdownBackend(_BackendBase):
    """
    Showdown websocket backend — async, high-throughput.

    Wraps WorkerOpponentFactory + BatchInferencePlayers. run_battle_batch()
    calls factory.prepare_batch_tasks() + asyncio.wait(), returns collected
    trajectories from the local queue.
    """

    def __init__(
        self,
        *,
        config: RNaDConfig,
        worker_id: int,
        agent: Optional[RNaDAgent],
        model: Any,
        model_config: Dict[str, Any],
        embedder: Embedder,
        team_repo: TeamRepo,
        bc_agent: Optional[RNaDAgent],
        server_port: int,
        run_id: str,
        initial_curriculum: Optional[Dict[str, float]] = None,
        exploiter_agent: Optional[RNaDAgent] = None,
        victim_agent: Optional[RNaDAgent] = None,
        main_inference_client: Optional["InferenceClient"] = None,
        worker_inference_clients: Optional["WorkerInferenceClients"] = None,
    ) -> None:
        self._config = config
        self._worker_id = worker_id
        self._agent = agent
        self._model = model
        self._embedder = embedder
        self._team_repo = team_repo
        self._bc_agent = bc_agent
        self._exploiter_agent = exploiter_agent
        self._victim_agent = victim_agent
        self._server_port = server_port
        self._run_id = run_id
        self._model_config = model_config
        self._local_traj_queue: queue.Queue = queue.Queue()
        self._factory: Optional[WorkerOpponentFactory] = None
        self._main_inference_client = main_inference_client
        self._worker_inference_clients = worker_inference_clients

        cur = config.curriculum
        hw = config.hardware
        self._curriculum = (
            dict(initial_curriculum)
            if initial_curriculum is not None
            else dict(cur.curriculum_weights)
        )
        self._num_pairs = hw.players_per_worker // 2

        # Matches the timeout calculation in the original mp_worker_process:
        # battles_per_task = max(1, min(4, num_battles_per_pair))
        # timeout = min(600, max(120, battles_per_task * max(6, max_battle_steps // 2)))
        battles_per_task = max(1, min(4, hw.num_battles_per_pair))
        self._batch_timeout_s = min(
            600.0, max(120.0, float(battles_per_task * max(6, hw.max_battle_steps // 2)))
        )

    async def setup(self) -> None:
        cur = self._config.curriculum
        hw = self._config.hardware
        server_config = ServerConfiguration(
            f"ws://localhost:{self._server_port}/showdown/websocket", ""
        )
        external_vgcbench_usernames = cur.external_vgcbench_usernames
        if (
            external_vgcbench_usernames
            and cur.auto_launch_external_vgcbench
            and hw.num_servers > 1
        ):
            external_vgcbench_usernames = [
                derive_external_vgcbench_username(username, self._server_port)
                for username in external_vgcbench_usernames
            ]
        self._factory = WorkerOpponentFactory(
            team_repo=self._team_repo,
            battle_format=cur.battle_format,
            team_subdirectory=cur.team_pool_path,
            server_config=server_config,
            main_agent=self._agent,
            bc_agent=self._bc_agent,
            curriculum=self._curriculum,
            embedder=self._embedder,
            worker_id=self._worker_id,
            run_id=self._run_id,
            device="cpu",
            batch_size=hw.batch_size,
            batch_timeout=hw.batch_timeout,
            max_battle_steps=hw.max_battle_steps,
            vgc_bench_checkpoint_path=cur.vgc_bench_checkpoint_path,
            external_vgcbench_usernames=external_vgcbench_usernames,
            model_config=self._model_config,
            agent_team_path=cur.agent_team_path,
            exploiter_agent=self._exploiter_agent,
            victim_agent=self._victim_agent,
            max_concurrent_battles_per_player=hw.max_concurrent_battles_per_player,
            main_inference_client=self._main_inference_client,
            worker_inference_clients=self._worker_inference_clients,
        )
        self._factory.create_agents(self._num_pairs, self._local_traj_queue)
        self._factory.start_inference_loops()
        await asyncio.sleep(0.2)

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
        assert self._factory is not None, (
            "setup() must be called before run_battle_batch()"
        )
        tasks, sampled_types = self._factory.prepare_batch_tasks(n_battles)

        had_timeout = False
        timed_out_types: List[str] = []

        if tasks:
            running_tasks = [asyncio.create_task(t) for t in tasks]
            pending_map: Dict[asyncio.Task, str] = {
                running_tasks[i]: (
                    sampled_types[i] if i < len(sampled_types) else "unknown"
                )
                for i in range(len(running_tasks))
            }
            done_tasks, unfinished_tasks = await asyncio.wait(
                running_tasks,
                timeout=self._batch_timeout_s,
                return_when=asyncio.ALL_COMPLETED,
            )
            for done_task in done_tasks:
                try:
                    _ = done_task.exception()
                except asyncio.CancelledError:
                    pass
            if unfinished_tasks:
                had_timeout = True
                for t in unfinished_tasks:
                    timed_out_types.append(pending_map.get(t, "unknown"))
                    t.cancel()
                await asyncio.gather(*unfinished_tasks, return_exceptions=True)

        trajectories: List[Dict] = []
        completed_counts: Dict[str, int] = {}
        while not self._local_traj_queue.empty():
            try:
                traj = self._local_traj_queue.get_nowait()
                trajectories.append(traj)
                if isinstance(traj, dict):
                    opp_type = traj.get("opponent_type", "unknown")
                    completed_counts[opp_type] = completed_counts.get(opp_type, 0) + 1
                else:
                    completed_counts["legacy"] = completed_counts.get("legacy", 0) + 1
            except Exception:
                break

        battles_completed = sum(p.n_finished_battles for p in self._factory.players)
        return BatchResult(
            trajectories=trajectories,
            sampled_types=sampled_types,
            completed_counts=completed_counts,
            had_timeout=had_timeout,
            timed_out_types=timed_out_types,
            battles_completed=battles_completed,
        )

    def update_weights(self, state_dict: Dict) -> None:
        # Centralized inference: trainer-side InferenceService owns the
        # model; this backend has none in that mode. Drop silently — the
        # trainer state-syncs its own service copy at the same cadence.
        if self._model is None:
            return
        self._model.load_state_dict(state_dict)

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
    ) -> None:
        self._curriculum = curriculum
        if self._factory is not None:
            self._factory.update_curriculum(curriculum)

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        if self._factory is not None:
            for p in self._factory.players:
                if temperature is not None:
                    p.temperature = temperature
                if top_p is not None:
                    p.top_p = top_p

    def update_exploiter_weights(self, state_dict: Dict) -> None:
        if self._factory is not None:
            self._factory.update_exploiter_weights(state_dict)

    def update_victim_weights(self, state_dict: Dict) -> None:
        if self._factory is not None:
            self._factory.update_victim_weights(state_dict)

    def get_curriculum(self) -> Dict[str, float]:
        if self._factory is not None:
            return dict(self._factory.curriculum)
        return dict(self._curriculum)

    def get_diagnostics(self) -> Dict[str, float]:
        if self._factory is None:
            return {}
        return self._factory.get_runtime_diagnostics().get("summary", {})

    async def rebuild(self) -> None:
        if self._factory is not None:
            self._factory.rebuild_runtime_agents(self._local_traj_queue)
            await asyncio.sleep(0.2)

    def reset_battles(self) -> None:
        if self._factory is not None:
            self._factory.reset_all_battles()

    def get_unfinished_summary(self) -> Dict[str, int]:
        if self._factory is None:
            return {"total_unfinished": 0}
        return self._factory.get_unfinished_battle_summary()

    async def teardown(self) -> None:
        if self._factory is not None:
            self._factory.teardown_runtime_agents()


# ── Rust in-process backend ───────────────────────────────────────────────────


class _RustBackend(_BackendBase):
    """
    Rust in-process backend — synchronous, deterministic.

    Wraps _RustPolicyOpponentPool + SyncRustBattleDriver. run_battle_batch()
    calls driver.run(n_battles) and drains the completed trajectory list.
    """

    def __init__(
        self,
        *,
        config: RNaDConfig,
        worker_id: int,
        agent: RNaDAgent,
        model: Any,
        model_config: Dict[str, Any],
        embedder: Embedder,
        team_repo: TeamRepo,
        bc_agent: Optional[RNaDAgent],
        initial_curriculum: Optional[Dict[str, float]] = None,
    ) -> None:
        self._config = config
        self._worker_id = worker_id
        self._agent = agent
        self._model = model
        self._embedder = embedder
        self._team_repo = team_repo
        self._bc_agent = bc_agent
        self._model_config = model_config
        self._initial_curriculum = initial_curriculum
        self._driver: Optional[SyncRustBattleDriver] = None
        self._pool: Optional[_RustPolicyOpponentPool] = None
        self._p1_policy: Optional[SyncPolicyPlayer] = None
        self._feature_set = config.training.embedder_feature_set
        self._max_concurrent = max(1, config.hardware.players_per_worker // 2)

    async def setup(self) -> None:
        cur = self._config.curriculum
        hw = self._config.hardware
        fixed_team, team_options = _load_agent_team_sources(cur.agent_team_path)
        agent_team_supplier = _build_team_supplier(
            team_repo=self._team_repo,
            battle_format=cur.battle_format,
            team_subdirectory=cur.team_pool_path,
            fixed_team=fixed_team,
            team_options=team_options,
        )
        opponent_team_supplier = _build_team_supplier(
            team_repo=self._team_repo,
            battle_format=cur.battle_format,
            team_subdirectory=cur.team_pool_path,
        )
        initial_temp = self._config.temperature_at_step(0)
        self._p1_policy = SyncPolicyPlayer(
            self._agent,
            cur.battle_format,
            device="cpu",
            feature_set=self._feature_set,
            collect_trajectories=True,
            probabilistic=True,
            temperature=initial_temp,
            top_p=self._config.exploration.top_p,
            max_battle_steps=hw.max_battle_steps,
            opponent_type=OpponentPool.SELF_PLAY,
        )
        self._pool = _RustPolicyOpponentPool(
            main_agent=self._agent,
            bc_agent=self._bc_agent,
            config=self._config,
            team_repo=self._team_repo,
            model_config=self._model_config,
            feature_set=self._feature_set,
            device="cpu",
        )
        if self._initial_curriculum is not None:
            self._pool.update_curriculum(self._initial_curriculum)

        def battle_setup(_battle_index: int) -> Dict[str, Any]:
            opp_type, p2_team, p2_policy = self._pool.sample_opponent()  # type: ignore[union-attr]
            return {
                "p1_policy": self._p1_policy,
                "p2_policy": p2_policy,
                "p1_team_text": agent_team_supplier(),
                "p2_team_text": p2_team,
                "opponent_type": opp_type,
            }

        self._driver = SyncRustBattleDriver(
            format_id=cur.battle_format,
            p1_team=agent_team_supplier(),
            p2_team=opponent_team_supplier(),
            battle_tag_prefix=f"rust-env-{self._worker_id}",
            collect_rollouts=False,
            feature_set=self._feature_set,
            max_turns_per_battle=max(100, hw.max_battle_steps * 2),
            max_stalled_steps_per_battle=max(25, hw.max_battle_steps // 2),
            p1_policy=self._p1_policy,
            p1_team_supplier=agent_team_supplier,
            p2_team_supplier=opponent_team_supplier,
            battle_setup_callback=battle_setup,
        )

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
        assert self._driver is not None, "setup() must be called before run_battle_batch()"
        stats = self._driver.run(
            total_battles=n_battles, max_concurrent=self._max_concurrent
        )
        trajs = self._driver.consume_completed_trajectories()

        completed_counts: Dict[str, int] = {}
        opp_types: List[str] = []
        for t in trajs:
            opp_type = (
                t.get("opponent_type", "self_play") if isinstance(t, dict) else "unknown"
            )
            completed_counts[opp_type] = completed_counts.get(opp_type, 0) + 1
            opp_types.append(opp_type)

        battles_completed = stats.completed_battles if stats is not None else len(trajs)
        return BatchResult(
            trajectories=trajs,
            sampled_types=opp_types,
            completed_counts=completed_counts,
            had_timeout=False,
            timed_out_types=[],
            battles_completed=battles_completed,
        )

    def update_weights(self, state_dict: Dict) -> None:
        # Centralized inference: trainer-side InferenceService owns the
        # model; this backend has none in that mode. Drop silently — the
        # trainer state-syncs its own service copy at the same cadence.
        if self._model is None:
            return
        self._model.load_state_dict(state_dict)

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
    ) -> None:
        if self._pool is not None:
            self._pool.update_curriculum(curriculum)

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        if self._pool is not None:
            self._pool.update_sampling(temperature, top_p)
        if self._p1_policy is not None:
            self._p1_policy.update_sampling(temperature=temperature, top_p=top_p)

    def get_curriculum(self) -> Dict[str, float]:
        if self._pool is not None:
            return dict(self._pool.curriculum)
        return {}

    def get_diagnostics(self) -> Dict[str, float]:
        if self._driver is None:
            return {}
        stats = self._driver.get_last_batch_stats()  # type: ignore[attr-defined]
        if stats is None:
            return {}
        return {
            "battles_completed": float(stats.completed_battles),
            "battles_truncated": float(stats.truncated_battles),
            "battles_per_second": float(stats.battles_per_second),
        }

    async def teardown(self) -> None:
        pass
