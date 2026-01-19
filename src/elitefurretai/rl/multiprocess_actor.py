# -*- coding: utf-8 -*-
"""
IMPALA-style multiprocessing actors for RL training.

This module provides multiprocessing-based actors that run in separate Python
processes, bypassing the GIL limitation. Each actor:
- Owns its own model copy (CPU inference)
- Connects to its own Showdown server
- Sends trajectories to the learner via multiprocessing Queue
- Periodically receives updated weights from the learner

Architecture:
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
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.concurrency import POKE_LOOP
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.fast_action_mask import fast_get_action_mask
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


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

    def choose_move(self, battle: DoubleBattle) -> DoubleBattleOrder:
        """Choose a move using the model."""
        # Get state embedding
        state = self.embedder.embed_to_vector(battle)
        state_np = np.array(state, dtype=np.float32)

        # Get action mask
        is_tp = battle.teampreview
        if is_tp:
            mask = np.ones(MDBO.teampreview_space(), dtype=np.float32)
        else:
            mask = fast_get_action_mask(battle)

        # Initialize hidden state if needed
        if self.hidden is None:
            h = torch.zeros(4, 1, 512, device=self.device)  # lstm_layers, batch, hidden
            c = torch.zeros(4, 1, 512, device=self.device)
            self.hidden = (h, c)

        # Forward pass
        state_tensor = torch.tensor(state_np, device=self.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            turn_logits, tp_logits, values, next_hidden = self.model(
                state_tensor, self.hidden
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
            action = torch.multinomial(probs, 1).item()
        else:
            action = probs.argmax().item()

        # Store trajectory step
        self.states.append(state_np)
        self.actions.append(action)
        self.action_masks.append(mask)
        self.is_teampreview.append(is_tp)
        self.values.append(values.item())
        self.rewards.append(0.0)  # Will be filled in at end

        # Convert action to order
        if is_tp:
            return MDBO.decode_teampreview_order(action, battle)
        else:
            order = MDBO.decode_doubles_order(action, battle)
            return order if order else self.choose_default_move()

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
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

        print(f"Actor {config.actor_id}: Model loaded, embedding size {embedder.embedding_size}")

        # Run battles
        battles_completed = 0
        last_weight_check = time.time()

        async def run_battles():
            nonlocal battles_completed, last_weight_check

            # Create players
            actor_player = ActorPlayer(
                model=model,
                embedder=embedder,
                trajectory_queue=trajectory_queue,
                actor_id=config.actor_id,
                device=config.device,
                probabilistic=config.probabilistic,
                username=f"Actor{config.actor_id}",
                server_configuration={
                    "server_url": f"localhost:{config.server_port}",
                    "authentication_url": None,
                },
                battle_format=config.battle_format,
            )

            opponent = RandomPlayer(
                username=f"Opponent{config.actor_id}",
                server_configuration={
                    "server_url": f"localhost:{config.server_port}",
                    "authentication_url": None,
                },
                battle_format=config.battle_format,
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
