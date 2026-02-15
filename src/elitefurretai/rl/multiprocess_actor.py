# -*- coding: utf-8 -*-
"""
TODO: consider including in agent.py
RL training actors and players for Pokemon VGC.

This module provides two types of players for RL training:

1. BatchInferencePlayer: High-throughput player for threaded training
   - Batches observations from multiple concurrent battles
   - Uses async queue for GPU inference batching
   - Runs in threads (shares model with learner)

2. ActorPlayer: IMPALA-style multiprocessing player
   - Runs in separate process (CPU inference)
   - Simple single-battle inference
   - Sends trajectories via multiprocessing Queue

Architecture (IMPALA-style):
    Main Process (Learner)
    ├── Model on GPU
    ├── Receives trajectories from Queue
    ├── Computes gradients, updates weights
    └── Broadcasts weights to actors via shared memory

    Actor Process 0          Actor Process 1          ...
    ├── Model copy (CPU)     ├── Model copy (CPU)
    ├── Showdown server      ├── Showdown server
    ├── Runs battles         ├── Runs battles
    └── Sends trajectories   └── Sends trajectories
"""

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
)

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.fast_action_mask import fast_get_action_mask

if TYPE_CHECKING:
    from elitefurretai.rl.players import RNaDAgent

# Registry of worker-specific executors: worker_id -> ThreadPoolExecutor
_WORKER_EXECUTORS: Dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = Lock()

# Global fallback executor for non-worker contexts (e.g., testing)
_FALLBACK_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fallback_inference")

def get_worker_executor(worker_id: Optional[int] = None) -> ThreadPoolExecutor:
    """Get or create a ThreadPoolExecutor for a specific worker.

    Each worker gets its own executor to reduce thread contention.
    """
    if worker_id is None:
        return _FALLBACK_EXECUTOR

    with _EXECUTOR_LOCK:
        if worker_id not in _WORKER_EXECUTORS:
            _WORKER_EXECUTORS[worker_id] = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"worker{worker_id}_inference"
            )
        return _WORKER_EXECUTORS[worker_id]


def cleanup_worker_executors():
    """Shutdown all worker executors (call at training end)."""
    with _EXECUTOR_LOCK:
        for executor in _WORKER_EXECUTORS.values():
            executor.shutdown(wait=False)
        _WORKER_EXECUTORS.clear()


# =============================================================================
# BatchInferencePlayer - High-throughput threaded player with batched inference
# =============================================================================


class BatchInferencePlayer(Player):
    """
    High-performance player that batches observations from multiple concurrent battles.

    This player is designed for threaded training where multiple battles run
    concurrently and share a single GPU model. It batches inference requests
    to maximize GPU utilization.

    Key features:
    - Async queue-based batching of inference requests
    - Thread pool for GPU inference to avoid blocking poke-env's event loop
    - Trajectory collection for RL training
    - Supports both teampreview and turn actions through unified pipeline
    """

    def __init__(
        self,
        model: "RNaDAgent",
        device="cpu",
        batch_size=16,
        batch_timeout=0.01,
        probabilistic=True,
        trajectory_queue=None,
        accept_open_team_sheet=False,  # Default False to avoid OTS deadlock
        worker_id: Optional[int] = None,
        embedder: Optional[Embedder] = None,  # Share embedder to save memory
        max_battle_steps: int = 40,  # Maximum steps before forfeiting battle
        opponent_type: str = "self_play",  # Type of opponent for trajectory metadata
        **kwargs,
    ):
        # IMPORTANT: Set up instance attributes BEFORE calling super().__init__()
        # because poke-env Player starts listening for websocket messages immediately
        # (start_listening=True by default), and message handlers may access these
        # attributes before __init__ completes.

        # Extract battle_format from kwargs since we need it before super().__init__
        battle_format = kwargs.get("battle_format", "gen9vgc2023regc")

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.probabilistic = probabilistic
        self.trajectory_queue = trajectory_queue
        self.worker_id = worker_id  # For worker-specific executor
        self.opponent_type = opponent_type  # Track opponent type for trajectory metadata
        # Use shared embedder if provided, otherwise create one
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = Embedder(
                format=battle_format, feature_set=Embedder.FULL, omniscient=False
            )
        # CRITICAL: Create the queue in POKE_LOOP since that's where battle handling runs
        # poke-env runs all websocket/battle handling in a separate event loop (POKE_LOOP)
        # If we create the queue in a different loop, cross-loop access will deadlock
        self.queue: asyncio.Queue = create_in_poke_loop(asyncio.Queue)
        self.hidden_states: Dict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = {}  # battle_tag -> (h, c)
        self._inference_task: Optional[asyncio.Task] = None

        # Trajectory storage
        # battle_tag -> list of dicts
        self.current_trajectories: Dict[str, List[Dict[str, Any]]] = {}
        self.completed_trajectories: List[
            Dict[str, Any]
        ] = []  # For debugging/inspection when no training queue

        # Battles to discard (forfeited due to exceeding step limit)
        # These won't contribute to training - no win/loss reward
        self._discarded_battles: set = set()

        # Maximum steps before forfeiting a battle to prevent memory explosion
        # VGC battles typically end in 10-20 turns; 40 is very generous
        self.max_battle_steps = max_battle_steps

        # Now call parent init - this starts the websocket listener
        super().__init__(accept_open_team_sheet=accept_open_team_sheet, **kwargs)

    def clear_completed_trajectories(self):
        """Clear accumulated completed trajectories to free memory.

        For opponent players (trajectory_queue=None), trajectories accumulate
        in completed_trajectories. Call this periodically to prevent memory leaks.
        """
        self.completed_trajectories.clear()

    async def stop_listening(self):
        """Stop listening to the server and close the websocket connection."""
        await self.ps_client.stop_listening()

    def start_inference_loop(self):
        """
        Start the inference loop in POKE_LOOP (poke-env's event loop).

        CRITICAL: This is NOT async because we need to schedule the task in POKE_LOOP,
        not in whatever event loop is currently running. All battle handling happens
        in POKE_LOOP, so the inference loop must also run there to avoid cross-loop
        deadlocks when accessing the shared asyncio.Queue.
        """
        future = asyncio.run_coroutine_threadsafe(self._inference_loop(), POKE_LOOP)
        self._inference_future = future

    async def _inference_loop(self):
        while True:
            batch: List[Any] = []
            futures: List[Any] = []
            battle_tags: List[str] = []
            is_tps: List[bool] = []
            masks: List[Any] = []
            try:
                item = await self.queue.get()
                self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)

                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        self._add_to_batch(
                            batch, futures, battle_tags, is_tps, masks, item
                        )
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break

            if batch:
                await self._run_batch(batch, futures, battle_tags, is_tps, masks)

    def _add_to_batch(self, batch, futures, battle_tags, is_tps, masks, item):
        batch.append(item[0])
        futures.append(item[1])
        battle_tags.append(item[2])
        is_tps.append(item[3])
        masks.append(item[4])

    def _gpu_inference_sync(self, states_np, h_cpu, c_cpu):
        """Synchronous GPU inference - runs in ThreadPoolExecutor."""
        states_tensor = torch.tensor(states_np, dtype=torch.float32).to(self.device).unsqueeze(1)
        hidden = (h_cpu.to(self.device), c_cpu.to(self.device))

        with torch.no_grad():
            turn_logits, tp_logits, values, next_hidden = self.model(states_tensor, hidden)

        turn_probs = torch.softmax(turn_logits, dim=-1).cpu().numpy()
        tp_probs = torch.softmax(tp_logits, dim=-1).cpu().numpy()
        values_np = values.cpu().numpy()
        h_next_cpu = next_hidden[0].cpu()
        c_next_cpu = next_hidden[1].cpu()

        return turn_probs, tp_probs, values_np, h_next_cpu, c_next_cpu

    async def _run_batch(self, states, futures, battle_tags, is_tps, masks):
        states_np = np.array(states)

        h_list = []
        c_list = []
        for tag in battle_tags:
            if tag not in self.hidden_states:
                h, c = self.model.get_initial_state(1, "cpu")
                self.hidden_states[tag] = (h, c)
            h_list.append(self.hidden_states[tag][0])
            c_list.append(self.hidden_states[tag][1])

        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)

        loop = asyncio.get_running_loop()
        executor = get_worker_executor(self.worker_id)
        turn_probs, tp_probs, values, h_next_cpu, c_next_cpu = await loop.run_in_executor(
            executor,
            self._gpu_inference_sync,
            states_np,
            h_batch,
            c_batch
        )

        for i, tag in enumerate(battle_tags):
            self.hidden_states[tag] = (h_next_cpu[:, i : i + 1, :], c_next_cpu[:, i : i + 1, :])

        for i, future in enumerate(futures):
            is_tp = is_tps[i]
            mask = masks[i]

            if is_tp:
                probs = tp_probs[i, 0]
                valid_actions = list(range(len(probs)))
            else:
                probs = turn_probs[i, 0]
                if mask is not None:
                    probs = probs * mask
                    if probs.sum() == 0:
                        probs = mask / mask.sum()
                    else:
                        probs = probs / probs.sum()
                valid_actions = list(range(len(probs)))

            if self.probabilistic:
                action = np.random.choice(valid_actions, p=probs)
            else:
                action = np.argmax(probs)

            log_prob = np.log(probs[action] + 1e-10)

            result = {
                "action": action,
                "log_prob": log_prob,
                "value": values[i, 0],
                "probs": probs,
            }
            future.set_result(result)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        """Override poke-env's request handler to route ALL decisions through async pipeline."""
        if maybe_default_order:
            if random.random() < self.DEFAULT_CHOICE_CHANCE:
                message = self.choose_default_move().message
                await self.ps_client.send_message(message, battle.battle_tag)
                return

        choice = self.choose_move(battle)
        if asyncio.iscoroutine(choice):
            choice = await choice

        if isinstance(choice, str):
            message = choice
        elif hasattr(choice, "message"):
            message = choice.message
        else:
            message = str(choice)

        if message:
            await self.ps_client.send_message(message, battle.battle_tag)

    def choose_move(self, battle: AbstractBattle) -> Any:
        """Main decision method - handles BOTH teampreview and turn actions."""
        return self._choose_move_async(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        """NOT USED - we override _handle_battle_request to bypass this."""
        raise RuntimeError(
            "teampreview() should not be called - we override _handle_battle_request "
            "to route teampreview through choose_move() for async batched inference."
        )

    async def _choose_move_async(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()

        # Check if battle has exceeded step limit - forfeit to prevent memory explosion
        # Only the training player (with trajectory_queue) forfeits to avoid both sides
        # sending forfeit simultaneously and causing "not in that room" errors
        current_steps = len(self.current_trajectories.get(battle.battle_tag, []))
        if current_steps >= self.max_battle_steps:
            # Mark for discard
            self._discarded_battles.add(battle.battle_tag)

            # Clear trajectory immediately to free memory
            if battle.battle_tag in self.current_trajectories:
                del self.current_trajectories[battle.battle_tag]
            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]

            # Only the training player (trajectory_queue is not None) sends forfeit
            # Opponent just returns default action and waits for the other side
            if self.trajectory_queue is not None:
                # Send forfeit command - returns a string message directly
                return "/forfeit"  # type: ignore
            else:
                # Opponent: return default action - training player will forfeit
                return DefaultBattleOrder()

        state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))

        if battle.teampreview:
            mask = None
        else:
            mask = fast_get_action_mask(battle)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self.queue.put((state, future, battle.battle_tag, battle.teampreview, mask))
        result = await future

        action_idx = result["action"]

        # Track this step in trajectory (only if we're collecting for training)
        # Opponents (trajectory_queue=None) only need step count for forfeit logic
        if self.trajectory_queue is not None:
            if battle.battle_tag not in self.current_trajectories:
                self.current_trajectories[battle.battle_tag] = []

            step = {
                "state": state,
                "action": action_idx,
                "log_prob": result["log_prob"],
                "value": result["value"],
                "reward": 0,
                "is_teampreview": battle.teampreview,
                "mask": mask,
            }
            self.current_trajectories[battle.battle_tag].append(step)
        else:
            # Opponents: just track step count for forfeit logic (no actual data)
            if battle.battle_tag not in self.current_trajectories:
                self.current_trajectories[battle.battle_tag] = []
            self.current_trajectories[battle.battle_tag].append(None)  # type: ignore

        if battle.teampreview:
            try:
                mdbo = MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW)
                order = mdbo.message
            except (ValueError, AssertionError):
                order = "/team 123456"
        else:
            try:
                action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
                mdbo = MDBO.from_int(action_idx, type=action_type)
                order = mdbo.to_double_battle_order(battle)
            except (ValueError, KeyError, AttributeError, IndexError, AssertionError) as e:
                # Track error count for diagnostics
                if not hasattr(self, '_action_error_count'):
                    self._action_error_count = 0
                self._action_error_count += 1
                if self._action_error_count <= 5 or self._action_error_count % 100 == 0:
                    print(
                        f"WARNING: Action {action_idx} conversion failed (error #{self._action_error_count}): {type(e).__name__}: {e}"
                    )
                order = DefaultBattleOrder()

        return order  # type: ignore

    def _battle_finished_callback(self, battle: AbstractBattle):
        """Called automatically when a battle ends. Finalizes trajectory."""
        # Track finished battles for diagnostics
        if not hasattr(self, '_finished_battle_count'):
            self._finished_battle_count = 0
        self._finished_battle_count += 1

        # Check if this battle was forfeited due to step limit
        # If so, discard trajectory entirely - no win/loss reward for training
        if battle.battle_tag in self._discarded_battles:
            self._discarded_battles.discard(battle.battle_tag)
            # Clean up any remaining state (should already be cleared, but belt-and-suspenders)
            if battle.battle_tag in self.current_trajectories:
                del self.current_trajectories[battle.battle_tag]
            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]
            return  # Don't process trajectory - battle was forfeited

        # Only process trajectories if we have a queue to send them to
        # Opponents (trajectory_queue=None) don't need trajectories - skip to save memory
        if self.trajectory_queue is None:
            # Just clean up state without accumulating trajectory
            if battle.battle_tag in self.current_trajectories:
                del self.current_trajectories[battle.battle_tag]
            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]
            return

        if battle.battle_tag in self.current_trajectories:
            traj = self.current_trajectories.pop(battle.battle_tag)
            for t, step in enumerate(traj):
                # Base: small negative per turn (encourages faster wins)
                step_reward = -0.005

                # For now, just terminal reward at end
                if t == len(traj) - 1:
                    step_reward += 1.0 if battle.won else -1.0
                step["reward"] = step_reward

            # Create trajectory wrapper with metadata
            traj_with_meta = {
                "steps": traj,
                "opponent_type": self.opponent_type,
                "won": battle.won,
                "battle_length": len(traj),
                "forfeited": False,
            }
            self.trajectory_queue.put(traj_with_meta)

            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]
