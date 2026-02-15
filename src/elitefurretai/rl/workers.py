"""Unified worker exports for RL.

Option B structure entrypoint for worker-related player/process helpers.
"""

from elitefurretai.rl.multiprocess_actor import (
    BatchInferencePlayer,
    cleanup_worker_executors,
)

__all__ = [
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
