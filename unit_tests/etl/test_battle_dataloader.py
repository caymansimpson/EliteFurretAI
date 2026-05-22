"""Tests for OptimizedBattleDataLoader and _trajectory_collate_fn."""

import torch

from elitefurretai.etl.battle_dataloader import _trajectory_collate_fn


def test_collate_casts_states_to_bf16():
    """The states tensor should be downcast to bf16 to halve H2D payload.
    All other tensors should retain their original dtypes."""
    batch = [
        {
            "states": torch.randn(5, 10, dtype=torch.float32),  # (seq, features)
            "actions": torch.zeros(5, dtype=torch.long),
            "masks": torch.ones(5, dtype=torch.bool),
        }
        for _ in range(3)
    ]
    out = _trajectory_collate_fn(batch)
    assert out["states"].dtype == torch.bfloat16, (
        f"states should be bf16, got {out['states'].dtype}"
    )
    assert out["actions"].dtype == torch.long, (
        f"actions should stay long, got {out['actions'].dtype}"
    )
    assert out["masks"].dtype == torch.bool, (
        f"masks should stay bool, got {out['masks'].dtype}"
    )
    # Shape check
    assert out["states"].shape == (3, 5, 10)


def test_collate_handles_empty_batch():
    """Empty batch should return an empty dict (pre-existing behavior)."""
    assert _trajectory_collate_fn([]) == {}
