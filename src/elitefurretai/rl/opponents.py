"""Unified opponent management exports for RL.

Option B structure entrypoint for opponent-related classes and helpers.
"""

from elitefurretai.rl.opponent_pool import ExploiterRegistry, OpponentPool
from elitefurretai.rl.worker_factory import WorkerOpponentFactory

__all__ = [
    "OpponentPool",
    "ExploiterRegistry",
    "WorkerOpponentFactory",
]
