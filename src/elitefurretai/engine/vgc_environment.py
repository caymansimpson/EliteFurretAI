"""
vgc_environment.py — Worker-side environment over the Showdown websocket backend.

The worker process creates one VGCEnvironment, calls run_battle_batch()
repeatedly, and receives completed trajectories. All battle execution flows
through Pokemon Showdown via poke-env.

Lifecycle:
  env = VGCEnvironment.from_config(config, worker_id, embedder, ...)
  await env.setup()
  while training:
      result = await env.run_battle_batch(n_battles)
      env.update_curriculum(curriculum)
  await env.teardown()
"""

import asyncio
import logging
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional

from poke_env import ServerConfiguration

from elitefurretai.agents.vgcbench_manager import VGCBenchManager
from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.inference_worker import WorkerInferenceClients
from elitefurretai.rl.opponents import (
    OpponentPool,
    WorkerOpponentFactory,
)
from elitefurretai.rl.rl_utils import normalize_curriculum

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


class VGCEnvironment:
    """Worker-side environment over the Showdown websocket backend.

    Thin facade over _ShowdownBackend; the indirection is a holdover from
    when this file housed both a Showdown and a Rust backend. Kept because
    callers already use VGCEnvironment as the public type, with the possibility
    of moving to Rust again.
    """

    def __init__(self, backend: "_ShowdownBackend") -> None:
        self._backend = backend

    @classmethod
    def from_config(
        cls,
        config: RNaDConfig,
        worker_id: int,
        embedder: Embedder,
        team_repo: TeamRepo,
        server_port: int,
        # Bundle of per-model inference clients. Threaded through to
        # WorkerOpponentFactory so it can hot-swap opponent.inference_client
        # between main / bc / ghost slots / etc.
        worker_inference_clients: WorkerInferenceClients,
        run_id: str = "",
    ) -> "VGCEnvironment":
        """Create a VGCEnvironment for the given config and worker."""
        backend = _ShowdownBackend(
            config=config,
            worker_id=worker_id,
            embedder=embedder,
            team_repo=team_repo,
            server_port=server_port,
            run_id=run_id,
            worker_inference_clients=worker_inference_clients,
        )
        return cls(backend)

    async def setup(self) -> None:
        """Start websocket connections."""
        await self._backend.setup()

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
        """Run n_battles; return a BatchResult with trajectories and timeout info."""
        return await self._backend.run_battle_batch(n_battles)

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

    def set_active_ghost_slots(self, slots: List[int]) -> None:
        """Set which ghost slots are currently valid routing targets.

        Used both at worker startup (seed from spawn args) and on each
        control_queue broadcast (track curriculum changes).
        """
        self._backend.set_active_ghost_slots(slots)

    def set_active_exploiter_slots(self, slots: List[int]) -> None:
        """Set which exploiter-snapshot slots are currently valid routing targets.

        Used both at worker startup (seed from spawn args) and on each
        control_queue broadcast (track curriculum changes).
        """
        self._backend.set_active_exploiter_slots(slots)

    def get_curriculum(self) -> Dict[str, float]:
        """Return current curriculum weights (copy)."""
        return self._backend.get_curriculum()

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
        """Disconnect websockets / clean up resources."""
        await self._backend.teardown()


class _ShowdownBackend:
    """
    Showdown websocket backend — async, high-throughput.

    Wraps WorkerOpponentFactory + RLTrajectoryPlayers. run_battle_batch()
    calls factory.prepare_batch_tasks() + asyncio.wait(), returns collected
    trajectories from the local queue.
    """

    def __init__(
        self,
        *,
        config: RNaDConfig,
        worker_id: int,
        embedder: Embedder,
        team_repo: TeamRepo,
        server_port: int,
        run_id: str,
        worker_inference_clients: WorkerInferenceClients,
    ) -> None:
        self._local_traj_queue: queue.Queue = queue.Queue()

        cur = config.curriculum
        hw = config.hardware
        self._curriculum = dict(cur.curriculum_weights)
        self._num_pairs = hw.players_per_worker // 2
        self._batch_timeout_s = 120.0

        # Vgcbench username derivation (server-affinity routing). Only
        # one server hosts an external vgcbench runner (memory
        # mitigation — see VGCBenchManager.RUNNER_SERVER_INDEX). Workers
        # on other servers can't challenge a user logged into a
        # different Showdown server, so:
        #   - the worker on the runner's server: derive the suffixed
        #     username and keep the vgc_bench_baseline weight as-is.
        #   - workers on every other server: clear the username list AND
        #     zero out the local vgc_bench_baseline weight so they don't
        #     sample an opponent they can't play. The freed mass falls
        #     through to self_play via OpponentPool.sample_opponent_type's
        #     un-normalized random.random() (anything past the cumulative
        #     defaults to SELF_PLAY).
        # The launcher's port-suffix derivation condition lives on the
        # manager itself; we mirror it via `should_suffix_port` so the
        # two branches can't drift.
        external_vgcbench_usernames: List[str] = list(VGCBenchManager.USERNAMES)
        vgc_bench_weight = cur.curriculum_weights.get(OpponentPool.VGC_BENCH_BASELINE, 0.0)
        runner_port = hw.showdown_start_port + VGCBenchManager.RUNNER_SERVER_INDEX
        this_worker_has_runner = server_port == runner_port
        if (
            external_vgcbench_usernames
            and vgc_bench_weight > 0
            and VGCBenchManager.should_suffix_port(hw.num_servers)
        ):
            if this_worker_has_runner:
                external_vgcbench_usernames = [
                    VGCBenchManager.derive_username(username, runner_port)
                    for username in external_vgcbench_usernames
                ]
            else:
                external_vgcbench_usernames = []
                if self._curriculum.get(OpponentPool.VGC_BENCH_BASELINE, 0.0) > 0:
                    logger.info(
                        "worker %s on port %d has no vgcbench runner "
                        "(runner_port=%d); zeroing local vgc_bench_baseline "
                        "weight and renormalizing across remaining slots",
                        worker_id,
                        server_port,
                        runner_port,
                    )
                    self._curriculum[OpponentPool.VGC_BENCH_BASELINE] = 0.0
                    self._curriculum = normalize_curriculum(self._curriculum)

        self._factory = WorkerOpponentFactory(
            team_repo=team_repo,
            battle_formats=cur.battle_formats,
            opponent_team_subdirectories=cur.resolved_opponent_team_pool_paths(),
            server_config=ServerConfiguration(
                f"ws://localhost:{server_port}/showdown/websocket", ""
            ),
            curriculum=self._curriculum,
            embedder=embedder,
            worker_id=worker_id,
            run_id=run_id,
            max_battle_steps=hw.max_battle_steps,
            external_vgcbench_usernames=external_vgcbench_usernames,
            agent_team_paths=cur.resolved_agent_team_paths() or None,
            max_concurrent_battles_per_player=hw.max_concurrent_battles_per_player,
            worker_inference_clients=worker_inference_clients,
        )

    async def setup(self) -> None:
        # Side-effectful work that must run on the event loop:
        # create_agents opens RLTrajectoryPlayer websocket sessions; the
        # asyncio.sleep gives them a beat to connect before the first batch.
        self._factory.create_agents(self._num_pairs, self._local_traj_queue)
        await asyncio.sleep(0.2)

    async def run_battle_batch(self, n_battles: int) -> BatchResult:
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
                opp_type = traj.get("opponent_type", "unknown")
                completed_counts[opp_type] = completed_counts.get(opp_type, 0) + 1
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

    def update_curriculum(
        self,
        curriculum: Dict[str, float],
    ) -> None:
        self._factory.update_curriculum(curriculum)

    def update_sampling(
        self, temperature: Optional[float], top_p: Optional[float]
    ) -> None:
        for p in self._factory.players:
            if temperature is not None:
                p.temperature = temperature
            if top_p is not None:
                p.top_p = top_p

    def set_active_ghost_slots(self, slots: List[int]) -> None:
        self._factory.set_active_ghost_slots(slots)

    def set_active_exploiter_slots(self, slots: List[int]) -> None:
        self._factory.set_active_exploiter_slots(slots)

    def get_curriculum(self) -> Dict[str, float]:
        return dict(self._factory.curriculum)

    async def rebuild(self) -> None:
        self._factory.rebuild_runtime_agents(self._local_traj_queue)
        await asyncio.sleep(0.2)

    def reset_battles(self) -> None:
        self._factory.reset_all_battles()

    def get_unfinished_summary(self) -> Dict[str, int]:
        return self._factory.get_unfinished_battle_summary()

    async def teardown(self) -> None:
        self._factory.teardown_runtime_agents()
