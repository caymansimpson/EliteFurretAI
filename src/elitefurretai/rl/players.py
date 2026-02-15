"""Unified player and agent exports for RL.

Option B structure entrypoint for player-related classes.
"""

from elitefurretai.rl.agent import MaxDamagePlayer, RNaDAgent
from elitefurretai.rl.multiprocess_actor import (
    BatchInferencePlayer,
    cleanup_worker_executors,
)

__all__ = [
    "RNaDAgent",
    "MaxDamagePlayer",
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
