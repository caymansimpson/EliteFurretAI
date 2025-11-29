# -*- coding: utf-8 -*-
"""
Worker process for collecting battle experience in parallel.
Each worker runs its own poke-env player and battles, collecting rollouts for the learner.
"""

import torch
import asyncio
import queue
from typing import Dict, Any
from poke_env.player import Player, RandomPlayer
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.rl.environment import VGCDoublesEnv
from elitefurretai.rl.agent import ActorCritic
from elitefurretai.rl.memory import ExperienceBuffer


async def run_battle_episode(env: VGCDoublesEnv, model: ActorCritic, memory: ExperienceBuffer,
                             opponent: Player, max_steps: int = 100):
    """
    Run a single battle episode, collecting experience.

    Args:
        env: VGC environment with integrated model
        model: Local actor-critic model
        memory: Experience buffer to store rollout
        opponent: Opponent player to battle against
        max_steps: Maximum steps per episode

    Returns:
        dict: Episode statistics
    """
    # Set the model and memory on the environment
    env.model = model
    env.memory = memory
    env.steps = 0
    env.max_steps = max_steps

    # Run battle using poke_env's coordination
    await env.battle_against(opponent, n_battles=1)

    # Get battle statistics
    battle = list(env._battles.values())[0] if env._battles else None
    if battle:
        episode_reward = env.calc_reward(battle)  # type: ignore
        episode_length = env.steps

        # Assign final reward to last step (sparse reward)
        if memory.rewards:
            memory.rewards[-1] = episode_reward

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'won': battle.won if battle.finished else None
        }
    else:
        return {
            'episode_reward': 0.0,
            'episode_length': 0,
            'won': None
        }


def worker_fn(worker_id: int, weights_queue, experience_queue, config: Dict):
    """
    The main function for an Actor-Worker process.

    This function runs battle episodes, collects experience, and communicates
    with the central learner.

    Args:
        worker_id (int): A unique ID for this worker.
        weights_queue (multiprocessing.Queue): Queue to receive new model weights.
        experience_queue (multiprocessing.Queue): Queue to send collected experience.
        config (dict): Configuration parameters.
    """
    print(f"Worker {worker_id} started.")

    # Set torch file system sharing (required for WSL)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Get configuration
    battle_format = config['battle_format']
    buffer_size = config['buffer_size']
    state_dim = config['state_dim']
    action_dim = config['action_dim']

    # Create team and opponent_team
    team = config.get('team', None)
    if team and isinstance(team, str):
        team = ConstantTeambuilder(team)
    opponent_team = config.get('opponent_team', None)
    if opponent_team and isinstance(opponent_team, str):
        opponent_team = ConstantTeambuilder(opponent_team)

    # Initialize opponent
    opponent = RandomPlayer(
        battle_format=battle_format,
        team=opponent_team
    )

    # Initialize environment
    env = VGCDoublesEnv(
        battle_format=battle_format,
        opponent=opponent,
        team=team,
        max_concurrent_battles=1
    )

    # Initialize local model
    local_model = ActorCritic(state_dim, action_dim, hidden_sizes=config.get('hidden_sizes', [512, 256]))
    local_model.to('cpu')
    local_model.eval()

    # Initialize memory buffer
    memory = ExperienceBuffer(
        buffer_size=buffer_size,
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        device='cpu',
        checkpoint_dir=config.get('checkpoint_dir', None)
    )

    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Main worker loop
    episodes_collected = 0
    try:
        while True:
            # Sync weights with the learner (non-blocking)
            try:
                new_weights = weights_queue.get_nowait()
                local_model.load_state_dict(new_weights)
                print(f"Worker {worker_id} updated weights.")
            except queue.Empty:
                pass  # No new weights, continue with current ones

            # Collect rollout
            steps_collected = 0
            episode_stats = []

            while steps_collected < buffer_size:
                # Run one episode (max 20 steps to speed up training iterations)
                stats = loop.run_until_complete(
                    run_battle_episode(env, local_model, memory, opponent, max_steps=20)
                )
                episode_stats.append(stats)
                steps_collected += stats['episode_length']
                episodes_collected += 1

                print(f"Worker {worker_id}: Episode {episodes_collected}, "
                      f"Reward: {stats['episode_reward']:.2f}, "
                      f"Length: {stats['episode_length']}, "
                      f"Won: {stats['won']}")

            # Calculate advantages for the collected rollout
            # Get last value for bootstrapping
            if memory.states:
                last_state = torch.tensor(memory.states[-1], dtype=torch.float32).unsqueeze(0)
                last_action_mask = None
                if memory.action_masks[-1] is not None:
                    last_action_mask = torch.tensor(memory.action_masks[-1], dtype=torch.int8).unsqueeze(0)

                with torch.no_grad():
                    _, _, last_value = local_model.get_action_and_value(last_state, last_action_mask)

                memory.calculate_advantages(last_value.item(), memory.dones[-1])

            # Send experience to learner
            experience_batch: Dict[str, Any] = memory.get()
            experience_batch['worker_id'] = worker_id
            experience_batch['episode_stats'] = episode_stats

            experience_queue.put(experience_batch)
            print(f"Worker {worker_id} sent {steps_collected} steps to learner.")

            # Clear memory for next rollout
            memory.clear()

    except KeyboardInterrupt:
        print(f"Worker {worker_id} shutting down...")
    finally:
        loop.close()
