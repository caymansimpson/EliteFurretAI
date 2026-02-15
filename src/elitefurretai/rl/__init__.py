"""
EliteFurretAI rl: Reinforcement Learning Training System

This module implements RNaD (Regularized Nash Dynamics) for training Pokemon VGC agents.
See COMPREHENSIVE_GUIDE.md for detailed documentation.

Core Components:
    - RNaDAgent: RL-compatible agent wrapper
    - RNaDLearner: Standard RNaD learner with single reference model
    - PortfolioRNaDLearner: Advanced learner with portfolio of reference models
    - BatchInferencePlayer: High-performance battle worker with batched inference
    - OpponentPool: Manages diverse opponent sampling for training
    - RNaDConfig: Configuration system for all hyperparameters

Usage:
    from elitefurretai.rl import RNaDConfig, RNaDAgent, RNaDLearner

    config = RNaDConfig.load("config.yaml")
    agent = RNaDAgent(model)
    learner = RNaDLearner(agent, ref_agent, lr=config.lr, device="cuda")
"""

from elitefurretai.rl.checkpoint import load_checkpoint, save_checkpoint
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import PortfolioRNaDLearner, RNaDLearner
from elitefurretai.rl.model_builder import build_model_from_config
from elitefurretai.rl.opponents import ExploiterRegistry, OpponentPool, WorkerOpponentFactory
from elitefurretai.rl.players import (
    BatchInferencePlayer,
    MaxDamagePlayer,
    RNaDAgent,
    cleanup_worker_executors,
)
from elitefurretai.rl.server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)

__all__ = [
    # Core training components
    "RNaDAgent",
    "RNaDLearner",
    "PortfolioRNaDLearner",
    # Workers and infrastructure
    "BatchInferencePlayer",
    "cleanup_worker_executors",
    "OpponentPool",
    "ExploiterRegistry",
    "WorkerOpponentFactory",
    "build_model_from_config",
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
