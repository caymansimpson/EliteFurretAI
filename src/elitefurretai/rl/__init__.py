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

from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learner import RNaDLearner
from elitefurretai.rl.opponent_pool import ExploiterRegistry, OpponentPool
from elitefurretai.rl.portfolio_learner import PortfolioRNaDLearner
from elitefurretai.rl.worker import BatchInferencePlayer

__all__ = [
    # Core training components
    "RNaDAgent",
    "RNaDLearner",
    "PortfolioRNaDLearner",
    # Workers and infrastructure
    "BatchInferencePlayer",
    "OpponentPool",
    "ExploiterRegistry",
    # Configuration
    "RNaDConfig",
]

__version__ = "2.0.0"
