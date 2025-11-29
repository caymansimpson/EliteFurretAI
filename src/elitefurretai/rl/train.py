import torch
from typing import Dict, Union
from elitefurretai.rl.learner import PPOLearner, MMDLearner, BaseLearner
from elitefurretai.rl.worker import worker_fn


def main():
    """
    The main entrypoint for starting the training process.

    This script initializes the shared queues, the central learner,
    and spawns the actor-worker processes.
    """
    # --- Configuration Hyperparameters ---
    config: Dict[str, Union[str, int, float]] = {
        'learner_type': 'PPO',      # Choose 'PPO' or 'MMD'
        'num_workers': 8,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'buffer_size': 2048,

        # PPO specific
        'ppo_epochs': 10,
        'num_minibatches': 32,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,

        # MMD specific
        'mmd_epochs': 10,
        'mmd_beta': 1.0,            # Weight for the KL divergence term

        'battle_format': 'gen9vgc2023regulationc',
    }

    # You need to define these based on your environment
    state_dim, action_dim = 100, 10  # Placeholder values

    # --- Initialization ---
    try:
        torch.multiprocessing.set_start_method('spawn')
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        pass

    weights_queue = torch.multiprocessing.Queue()
    experience_queue = torch.multiprocessing.Queue()

    # --- Start the Learner ---
    if config['learner_type'] == 'PPO':
        learner: BaseLearner = PPOLearner(state_dim, action_dim, weights_queue, experience_queue, config)
    elif config['learner_type'] == 'MMD':
        learner = MMDLearner(state_dim, action_dim, weights_queue, experience_queue, config)
    else:
        raise ValueError(f"Unknown learner_type: {config['learner_type']}")

    # --- Start the Workers ---
    processes = []
    for worker_id in range(int(config['num_workers'])):
        p = torch.multiprocessing.Process(
            target=worker_fn,
            args=(worker_id, weights_queue, experience_queue, config)
        )
        p.start()
        processes.append(p)

    # --- Run the Learner's main loop ---
    print(f"Starting {config['learner_type']} learner and {config['num_workers']} workers...")
    learner.learn()

    # --- Cleanup ---
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
