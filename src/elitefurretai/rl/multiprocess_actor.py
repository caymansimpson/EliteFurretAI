# -*- coding: utf-8 -*-
"""
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
import multiprocessing as mp
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.fast_action_mask import fast_get_action_mask
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

# =============================================================================
# Option 2a: Worker-specific model copies and executors
# Each worker gets its own model copy and executor to reduce contention
# =============================================================================

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
        model: RNaDAgent,
        device="cpu",
        batch_size=16,
        batch_timeout=0.01,
        probabilistic=True,
        trajectory_queue=None,
        accept_open_team_sheet=False,  # Default False to avoid OTS deadlock
        worker_id: Optional[int] = None,  # Option 2a: worker-specific executor
        embedder: Optional[Embedder] = None,  # Share embedder to save memory
        max_battle_steps: int = 40,  # Maximum steps before forfeiting battle
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
        # Option 2a: Use worker-specific executor to reduce contention
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
            self.current_trajectories[battle.battle_tag].append(None)  # Placeholder for count

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
            reward = 1.0 if battle.won else -1.0
            for step in traj:
                step["reward"] = reward

            self.trajectory_queue.put(traj)

            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]


# =============================================================================
# ActorPlayer - IMPALA-style multiprocessing player (CPU inference)
# =============================================================================


@dataclass
class ActorConfig:
    """Configuration for an actor process."""

    actor_id: int
    server_port: int
    model_path: str
    model_config: Dict[str, Any]
    battle_format: str = "gen9vgc2023regc"
    num_battles: int = 100
    device: str = "cpu"  # Actors use CPU
    probabilistic: bool = True
    team_path: Optional[str] = None  # Path to team file or directory


@dataclass
class Trajectory:
    """A single trajectory from one player's perspective."""

    states: np.ndarray  # (T, state_dim)
    actions: np.ndarray  # (T,)
    rewards: np.ndarray  # (T,)
    action_masks: np.ndarray  # (T, action_dim)
    is_teampreview: np.ndarray  # (T,)
    values: np.ndarray  # (T,) - for GAE
    win: float  # 1.0 for win, 0.0 for loss, 0.5 for draw


class ActorPlayer(Player):
    """
    A player that runs in an actor process for IMPALA-style training.

    Key differences from BatchInferencePlayer:
    - Runs in separate process (no GIL sharing)
    - Uses CPU inference (GPU for learner only)
    - Simpler batching (one battle at a time per actor)
    - Sends completed trajectories to queue
    """

    def __init__(
        self,
        model: FlexibleThreeHeadedModel,
        embedder: Embedder,
        trajectory_queue: mp.Queue,
        actor_id: int,
        device: str = "cpu",
        probabilistic: bool = True,
        accept_open_team_sheet: bool = False,
        **kwargs,
    ):
        self.model = model
        self.embedder = embedder
        self.trajectory_queue = trajectory_queue
        self.actor_id = actor_id
        self.device = device
        self.probabilistic = probabilistic

        # Trajectory buffers for current battle
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.action_masks: List[np.ndarray] = []
        self.is_teampreview: List[bool] = []
        self.values: List[float] = []

        # LSTM hidden state
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        super().__init__(accept_open_team_sheet=accept_open_team_sheet, **kwargs)

    def _reset_trajectory(self):
        """Reset trajectory buffers for new battle."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.action_masks = []
        self.is_teampreview = []
        self.values = []
        self.hidden = None

    def choose_move(self, battle: AbstractBattle) -> DoubleBattleOrder:
        """Choose a move using the model."""
        assert isinstance(battle, DoubleBattle)
        # Get state embedding
        state = self.embedder.embed_to_vector(battle)
        state_np = np.array(state, dtype=np.float32)

        # Get action mask
        is_tp = battle.teampreview
        if is_tp:
            mask = np.ones(MDBO.teampreview_space(), dtype=np.float32)
        else:
            mask = fast_get_action_mask(battle)

        # Forward pass - use forward_with_hidden for incremental RL inference
        # Pass None for hidden to let the model initialize it with correct dimensions
        state_tensor = torch.tensor(state_np, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            turn_logits, tp_logits, values, next_hidden = self.model.forward_with_hidden(
                state_tensor, self.hidden  # self.hidden is None on first call
            )
            self.hidden = next_hidden

        # Get action probabilities
        if is_tp:
            logits = tp_logits.squeeze()
            action_dim = MDBO.teampreview_space()
        else:
            logits = turn_logits.squeeze()
            action_dim = MDBO.action_space()

        # Apply mask
        mask_tensor = torch.tensor(mask, device=self.device)
        logits = logits[:action_dim]
        masked_logits = logits.masked_fill(mask_tensor == 0, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)

        # Sample action
        if self.probabilistic:
            action = int(torch.multinomial(probs, 1).item())
        else:
            action = int(probs.argmax().item())

        # Store trajectory step
        self.states.append(state_np)
        self.actions.append(action)
        self.action_masks.append(mask)
        self.is_teampreview.append(is_tp)
        self.values.append(values.item())
        self.rewards.append(0.0)  # Will be filled in at end

        # Convert action to order using MDBO
        if is_tp:
            mdbo = MDBO.from_int(action, type=MDBO.TEAMPREVIEW)
            # Teampreview returns the raw message string
            return mdbo.message  # type: ignore
        else:
            mdbo = MDBO.from_int(action, type=MDBO.TURN)
            try:
                order = mdbo.to_double_battle_order(battle)
                return order if order else self.choose_default_move()
            except Exception:
                return self.choose_default_move()

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Called when battle finishes. Send trajectory to learner."""
        if not self.states:
            return

        # Determine win/loss
        if battle.won:
            win = 1.0
            final_reward = 1.0
        elif battle.lost:
            win = 0.0
            final_reward = -1.0
        else:
            win = 0.5
            final_reward = 0.0

        # Set final reward
        if self.rewards:
            self.rewards[-1] = final_reward

        # Create trajectory
        trajectory = Trajectory(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            action_masks=np.array(self.action_masks),
            is_teampreview=np.array(self.is_teampreview),
            values=np.array(self.values),
            win=win,
        )

        # Send to learner
        try:
            self.trajectory_queue.put_nowait(trajectory)
        except Exception as e:
            print(f"Actor {self.actor_id}: Failed to send trajectory: {e}")

        # Reset for next battle
        self._reset_trajectory()


def actor_process(
    config: ActorConfig,
    trajectory_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
):
    """
    Main function for actor process.

    Args:
        config: Actor configuration
        trajectory_queue: Queue to send trajectories to learner
        weight_queue: Queue to receive updated weights from learner
        stop_event: Event to signal shutdown
    """
    try:
        print(f"Actor {config.actor_id}: Starting on port {config.server_port}")

        # Set up for multiprocessing
        torch.set_num_threads(1)  # Avoid thread contention

        # Load model
        checkpoint = torch.load(config.model_path, map_location="cpu", weights_only=False)
        model = FlexibleThreeHeadedModel(**config.model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(config.device)

        # Create embedder
        embedder = Embedder(
            format=config.battle_format, feature_set=Embedder.FULL, omniscient=False
        )

        # Load team if provided
        team = None
        if config.team_path:
            import os
            if os.path.isfile(config.team_path):
                with open(config.team_path, "r") as f:
                    team = f.read()
            elif os.path.isdir(config.team_path):
                # Sample random team from directory
                import glob
                team_files = glob.glob(os.path.join(config.team_path, "*.txt"))
                if team_files:
                    team_file = random.choice(team_files)
                    with open(team_file, "r") as f:
                        team = f.read()

        print(f"Actor {config.actor_id}: Model loaded, embedding size {embedder.embedding_size}")

        # Run battles
        battles_completed = 0
        last_weight_check = time.time()

        async def run_battles():
            nonlocal battles_completed, last_weight_check

            from poke_env import AccountConfiguration, ServerConfiguration

            server_config = ServerConfiguration(
                f"ws://localhost:{config.server_port}/showdown/websocket",
                None,  # type: ignore[arg-type]
            )

            # Create players with teams
            actor_player = ActorPlayer(
                model=model,
                embedder=embedder,
                trajectory_queue=trajectory_queue,
                actor_id=config.actor_id,
                device=config.device,
                probabilistic=config.probabilistic,
                account_configuration=AccountConfiguration(
                    f"Actor{config.actor_id}", None
                ),
                server_configuration=server_config,
                battle_format=config.battle_format,
                team=team,
            )

            opponent = RandomPlayer(
                account_configuration=AccountConfiguration(
                    f"Opponent{config.actor_id}", None
                ),
                server_configuration=server_config,
                battle_format=config.battle_format,
                team=team,  # RandomPlayer also needs a team for VGC
            )

            while not stop_event.is_set() and battles_completed < config.num_battles:
                # Check for weight updates
                if time.time() - last_weight_check > 1.0:
                    try:
                        while not weight_queue.empty():
                            new_weights = weight_queue.get_nowait()
                            model.load_state_dict(new_weights)
                            print(f"Actor {config.actor_id}: Updated weights")
                    except Exception:
                        pass
                    last_weight_check = time.time()

                # Run a battle
                try:
                    await actor_player.battle_against(opponent, n_battles=1)
                    battles_completed += 1
                    if battles_completed % 10 == 0:
                        print(
                            f"Actor {config.actor_id}: Completed {battles_completed} battles"
                        )
                except Exception as e:
                    print(f"Actor {config.actor_id}: Battle error: {e}")
                    await asyncio.sleep(0.5)

        # Run the async battle loop
        asyncio.get_event_loop().run_until_complete(run_battles())
        print(f"Actor {config.actor_id}: Finished {battles_completed} battles")

    except Exception as e:
        print(f"Actor {config.actor_id}: Fatal error: {e}")
        traceback.print_exc()


class MultiprocessingTrainer:
    """
    Coordinator for IMPALA-style multiprocessing training.

    Manages:
    - Spawning actor processes
    - Collecting trajectories
    - Updating model weights
    - Broadcasting weights to actors
    """

    def __init__(
        self,
        model_path: str,
        model_config: Dict[str, Any],
        num_actors: int = 4,
        server_ports: Optional[List[int]] = None,
        battle_format: str = "gen9vgc2023regc",
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.num_actors = num_actors
        self.server_ports = server_ports or [8000 + i for i in range(num_actors)]
        self.battle_format = battle_format
        self.device = device

        # Queues for communication
        self.trajectory_queue: mp.Queue = mp.Queue(maxsize=1000)
        self.weight_queues: List[mp.Queue] = [mp.Queue(maxsize=10) for _ in range(num_actors)]
        self.stop_event = mp.Event()

        # Actor processes
        self.actors: List[mp.Process] = []

        # Load model for learner
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = FlexibleThreeHeadedModel(**model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.train()

    def start_actors(self, num_battles_per_actor: int = 100):
        """Start actor processes."""
        for i in range(self.num_actors):
            config = ActorConfig(
                actor_id=i,
                server_port=self.server_ports[i],
                model_path=self.model_path,
                model_config=self.model_config,
                battle_format=self.battle_format,
                num_battles=num_battles_per_actor,
            )
            p = mp.Process(
                target=actor_process,
                args=(config, self.trajectory_queue, self.weight_queues[i], self.stop_event),
            )
            p.start()
            self.actors.append(p)
            print(f"Started actor {i} (PID {p.pid})")

    def broadcast_weights(self):
        """Send current model weights to all actors."""
        weights = self.model.state_dict()
        # Convert to CPU for transfer
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        for q in self.weight_queues:
            try:
                # Clear old weights
                while not q.empty():
                    try:
                        q.get_nowait()
                    except Exception:
                        break
                q.put_nowait(cpu_weights)
            except Exception as e:
                print(f"Failed to broadcast weights: {e}")

    def collect_trajectories(self, timeout: float = 1.0) -> List[Trajectory]:
        """Collect available trajectories from actors."""
        trajectories = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                traj = self.trajectory_queue.get(timeout=0.1)
                trajectories.append(traj)
            except Exception:
                break
        return trajectories

    def stop(self):
        """Stop all actors."""
        self.stop_event.set()
        for p in self.actors:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

    def run_training(
        self,
        num_batches: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        broadcast_interval: int = 10,
    ):
        """
        Run training loop.

        Args:
            num_batches: Number of training batches
            batch_size: Trajectories per batch
            learning_rate: Learning rate
            broadcast_interval: How often to broadcast weights
        """
        torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        trajectories_collected = 0
        batches_trained = 0

        while batches_trained < num_batches:
            # Collect trajectories
            trajs = self.collect_trajectories(timeout=2.0)
            trajectories_collected += len(trajs)

            if len(trajs) >= batch_size:
                # TODO: Implement actual training step here
                # This is a placeholder - you'd compute loss and backprop
                batches_trained += 1
                print(
                    f"Batch {batches_trained}: {len(trajs)} trajectories "
                    f"(total collected: {trajectories_collected})"
                )

                # Broadcast updated weights periodically
                if batches_trained % broadcast_interval == 0:
                    self.broadcast_weights()
                    print("Broadcast weights to actors")

        self.stop()
        return trajectories_collected


def main():
    """Test multiprocessing trainer."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/supervised/dauntless-hill-95.pt")
    parser.add_argument("--num-actors", type=int, default=2)
    parser.add_argument("--battles-per-actor", type=int, default=20)
    args = parser.parse_args()

    # Get model config from checkpoint
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    full_config = checkpoint.get("model_config", checkpoint.get("config", {}))

    # Extract model kwargs
    from elitefurretai.etl.embedder import Embedder

    embedder = Embedder(format="gen9vgc2023regulationc", feature_set="full", omniscient=True)
    model_config = {
        "input_size": embedder.embedding_size,
        "group_sizes": embedder.group_embedding_sizes,
        "dropout": full_config.get("dropout", 0.1),
        "gated_residuals": full_config.get("gated_residuals", False),
        "use_grouped_encoder": full_config.get("use_grouped_encoder", True),
        "grouped_encoder_hidden_dim": full_config.get("grouped_encoder_hidden_dim", 512),
        "grouped_encoder_aggregated_dim": full_config.get("grouped_encoder_aggregated_dim", 4096),
        "pokemon_attention_heads": full_config.get("pokemon_attention_heads", 16),
        "early_layers": full_config.get("early_layers", [4096, 2048, 2048, 1024]),
        "early_attention_heads": full_config.get("early_attention_heads", 16),
        "lstm_layers": full_config.get("lstm_layers", 4),
        "lstm_hidden_size": full_config.get("lstm_hidden_size", 512),
        "late_layers": full_config.get("late_layers", [2048, 2048, 1024, 1024]),
        "late_attention_heads": full_config.get("late_attention_heads", 32),
        "teampreview_head_layers": full_config.get("teampreview_head_layers", [512, 256]),
        "teampreview_head_dropout": full_config.get("teampreview_head_dropout", 0.3),
        "teampreview_attention_heads": full_config.get("teampreview_attention_heads", 8),
        "turn_head_layers": full_config.get("turn_head_layers", [2048, 1024, 1024, 1024]),
        "max_seq_len": full_config.get("max_seq_len", 40),
    }

    trainer = MultiprocessingTrainer(
        model_path=args.model,
        model_config=model_config,
        num_actors=args.num_actors,
        device="cpu",  # Use CPU for testing
    )

    print(f"Starting {args.num_actors} actors with {args.battles_per_actor} battles each")
    trainer.start_actors(num_battles_per_actor=args.battles_per_actor)

    # Let actors run for a bit
    import time

    start = time.time()
    total_trajs = 0
    while time.time() - start < 60:  # Run for 60 seconds max
        trajs = trainer.collect_trajectories(timeout=1.0)
        total_trajs += len(trajs)
        if trajs:
            print(f"Collected {len(trajs)} trajectories (total: {total_trajs})")

        # Check if all actors finished
        if all(not p.is_alive() for p in trainer.actors):
            break

    print(f"Total trajectories collected: {total_trajs}")
    print(f"Time elapsed: {time.time() - start:.1f}s")
    trainer.stop()


if __name__ == "__main__":
    main()
