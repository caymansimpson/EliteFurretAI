# -*- coding: utf-8 -*-
"""
Simple RL Training Script with Hardcoded Teams

This script provides a simple way to start RL training with obvious Pokemon teams
for testing and validation. Teams are designed with clear type advantages to make
learning observable.

Usage:
    python -m elitefurretai.rl.simple_train --num-workers 4 --bc-model path/to/model.pt
"""

import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path

from elitefurretai.rl.learner import PPOLearner, MMDLearner, BaseLearner
from elitefurretai.rl.worker import worker_fn
from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.model_utils.embedder import Embedder


# Hardcoded teams with obvious type matchups for testing
FIRE_TEAM = """
Charizard @ Choice Scarf
Ability: Blaze
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Flamethrower
- Air Slash
- Dragon Pulse
- Focus Blast

Arcanine @ Sitrus Berry
Ability: Intimidate
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Flare Blitz
- Close Combat
- Extreme Speed
- Protect

Talonflame @ Life Orb
Ability: Gale Wings
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Brave Bird
- Flare Blitz
- U-turn
- Protect

Torkoal @ Choice Specs
Ability: Drought
Level: 50
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 0 Atk
- Heat Wave
- Earth Power
- Eruption
- Flamethrower
"""

WATER_TEAM = """
Greninja @ Life Orb
Ability: Protean
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Hydro Pump
- Ice Beam
- Dark Pulse
- Protect

Gyarados @ Lum Berry
Ability: Intimidate
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Waterfall
- Ice Fang
- Stone Edge
- Protect

Politoed @ Sitrus Berry
Ability: Drizzle
Level: 50
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Scald
- Ice Beam
- Protect
- Helping Hand

Kingdra @ Choice Specs
Ability: Swift Swim
Level: 50
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Hydro Pump
- Draco Meteor
- Ice Beam
- Surf
"""

BALANCED_TEAM = """
Garchomp @ Focus Sash
Ability: Rough Skin
Level: 50
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Earthquake
- Dragon Claw
- Rock Slide
- Protect

Togekiss @ Sitrus Berry
Ability: Serene Grace
Level: 50
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 0 Atk
- Air Slash
- Dazzling Gleam
- Follow Me
- Protect

Metagross @ Choice Band
Ability: Clear Body
Level: 50
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Meteor Mash
- Zen Headbutt
- Bullet Punch
- Earthquake

Rotom-Wash @ Leftovers
Ability: Levitate
Level: 50
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 0 Atk
- Hydro Pump
- Thunderbolt
- Volt Switch
- Protect
"""


def get_config(args):
    """
    Create configuration dictionary from arguments.
    """
    # Get embedder to determine state dimensions
    embedder = Embedder(
        format=args.battle_format,
        feature_set=Embedder.FULL,
        omniscient=False
    )

    config = {
        # Battle settings
        'battle_format': args.battle_format,
        'team': BALANCED_TEAM,  # Use balanced team for both sides initially
        'opponent_team': WATER_TEAM,  # Opponent uses water team

        # Dimensions
        'state_dim': embedder.embedding_size,
        'action_dim': MDBO.action_space(),
        'hidden_sizes': [512, 256],

        # Training hyperparameters
        'learner_type': args.learner,
        'num_workers': args.num_workers,
        'learning_rate': args.lr,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'buffer_size': args.buffer_size,

        # PPO specific
        'ppo_epochs': 10,
        'num_minibatches': 32,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,

        # MMD specific
        'mmd_epochs': 10,
        'mmd_beta': 1.0,

        # BC initialization
        'bc_model_path': args.bc_model if args.bc_model else None,

        # File system sharing
        'checkpoint_dir': args.checkpoint_dir,
    }

    return config


def main():
    parser = argparse.ArgumentParser(description='Simple RL Training for VGC Doubles')

    # Core arguments
    parser.add_argument('--learner', type=str, default='PPO', choices=['PPO', 'MMD'],
                        help='Learning algorithm to use (default: PPO)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--battle-format', type=str, default='gen9vgc2023regulationc',
                        help='Pokemon Showdown battle format')
    parser.add_argument('--buffer-size', type=int, default=2048,
                        help='Experience buffer size per worker (default: 2048)')

    # BC initialization
    parser.add_argument('--bc-model', type=str, default=None,
                        help='Path to pretrained behavioral cloning model for warm-start')

    # Training parameters
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')

    # File system
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')

    args = parser.parse_args()

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Get configuration
    config = get_config(args)

    # Print configuration
    print("=" * 60)
    print("Starting VGC Doubles RL Training")
    print("=" * 60)
    print(f"Learner: {config['learner_type']}")
    print(f"Battle Format: {config['battle_format']}")
    print(f"Number of Workers: {config['num_workers']}")
    print(f"Buffer Size: {config['buffer_size']}")
    print(f"State Dimension: {config['state_dim']}")
    print(f"Action Dimension: {config['action_dim']}")
    print(f"Learning Rate: {config['learning_rate']}")
    if config['bc_model_path']:
        print(f"BC Model: {config['bc_model_path']}")
    print("=" * 60)

    # Set multiprocessing start method to 'spawn' for WSL compatibility
    try:
        mp.set_start_method('spawn')
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Multiprocessing start method already set")

    # Enable file system sharing for WSL
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("Set torch sharing strategy to 'file_system'")

    # Create queues for communication
    weights_queue = mp.Queue(maxsize=config['num_workers'])
    experience_queue = mp.Queue(maxsize=config['num_workers'] * 2)

    # Initialize the learner
    print(f"\nInitializing {config['learner_type']} Learner...")
    if config['learner_type'] == 'PPO':
        learner: BaseLearner = PPOLearner(
            config['state_dim'],
            config['action_dim'],
            weights_queue,
            experience_queue,
            config
        )
    elif config['learner_type'] == 'MMD':
        learner = MMDLearner(
            config['state_dim'],
            config['action_dim'],
            weights_queue,
            experience_queue,
            config
        )
    else:
        raise ValueError(f"Unknown learner type: {config['learner_type']}")

    # Start worker processes
    print(f"\nStarting {config['num_workers']} worker processes...")
    processes = []
    for worker_id in range(config['num_workers']):
        p = mp.Process(
            target=worker_fn,
            args=(worker_id, weights_queue, experience_queue, config),
            daemon=True
        )
        p.start()
        processes.append(p)
        print(f"  Worker {worker_id} started (PID: {p.pid})")

    # Run the learner's main loop
    print("\nStarting learner training loop...")
    print("Press Ctrl+C to stop training\n")

    try:
        learner.learn()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        # Cleanup
        print("\nShutting down workers...")
        for i, p in enumerate(processes):
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                print(f"  Worker {i} did not terminate cleanly, forcing...")
                p.kill()
        print("All workers shut down")

        # Save final model
        model_path = Path(args.checkpoint_dir) / 'final_model.pt'
        torch.save(learner.model.state_dict(), model_path)
        print(f"\nFinal model saved to: {model_path}")


if __name__ == '__main__':
    main()
