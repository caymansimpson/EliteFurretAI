"""
EliteFurretAI rl: Reinforcement Learning Training System

This module implements RNaD (Regularized Nash Dynamics) for training Pokemon VGC agents.
See RL.md for detailed documentation.

Core Components:
    - RNaDAgent: RL-compatible agent wrapper
    - PortfolioRNaDLearner: RNaD learner with portfolio of reference models
      (set max_portfolio_size=1, portfolio_update_strategy="recent" for standard RNaD)
    - BatchInferencePlayer: High-performance battle worker with batched inference
    - OpponentPool: Manages diverse opponent sampling for training
    - RNaDConfig: Configuration system for all hyperparameters
"""

from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import (
    PortfolioRNaDLearner,
    build_model_from_config,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from elitefurretai.rl.opponents import (
    OpponentPool,
    WorkerOpponentFactory,
)
from elitefurretai.rl.players import (
    BatchInferencePlayer,
    RNaDAgent,
    cleanup_worker_executors,
)

__all__ = [
    # Core training components
    "RNaDAgent",
    "PortfolioRNaDLearner",
    # Workers and infrastructure
    "BatchInferencePlayer",
    "cleanup_worker_executors",
    "OpponentPool",
    "WorkerOpponentFactory",
    "build_model_from_config",
    "load_model_from_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "launch_showdown_servers",
    "shutdown_showdown_servers",
    "allocate_server_ports",
    # Players
    "MaxDamagePlayer",
    # Configuration
    "RNaDConfig",
]

__version__ = "2.0.0"
