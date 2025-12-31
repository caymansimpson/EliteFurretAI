# -*- coding: utf-8 -*-
"""
Unit tests for RL training utilities.

These tests verify:
1. Trajectory collation for batched training
2. GAE (Generalized Advantage Estimation) computation
3. Padding and masking for variable-length trajectories
4. Checkpoint saving and loading
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

# =============================================================================
# MOCK IMPORTS (avoid loading full training infrastructure)
# =============================================================================

# We test the collate_trajectories function logic without importing train.py
# to avoid loading heavy dependencies. The function is copied here for testing.


def collate_trajectories_for_test(trajectories: List[List[Dict[str, Any]]], device: str):
    """
    Collate list of trajectories into batched tensors with padding.

    This is a copy of the function from train.py for isolated testing.

    Args:
        trajectories: List of trajectories, where each trajectory is a list of step dicts
        device: Device to place tensors on

    Returns:
        Dict of batched tensors with padding
    """
    batch_size = len(trajectories)
    max_len = max(len(t) for t in trajectories)

    # Get dim from first state
    dim = len(trajectories[0][0]["state"])

    states = torch.zeros(batch_size, max_len, dim)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    rewards = torch.zeros(batch_size, max_len)
    log_probs = torch.zeros(batch_size, max_len)
    values = torch.zeros(batch_size, max_len)
    is_tp = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    processed_advantages = []
    processed_returns = []

    for i, traj in enumerate(trajectories):
        seq_len = len(traj)

        # Extract fields
        traj_states = torch.tensor(np.array([step["state"] for step in traj]))
        traj_actions = torch.tensor([step["action"] for step in traj])
        traj_rewards = torch.tensor([step["reward"] for step in traj])
        traj_log_probs = torch.tensor([step["log_prob"] for step in traj])
        traj_values = torch.tensor([step["value"] for step in traj])
        traj_is_tp = torch.tensor([step["is_teampreview"] for step in traj])

        states[i, :seq_len] = traj_states
        actions[i, :seq_len] = traj_actions
        rewards[i, :seq_len] = traj_rewards
        log_probs[i, :seq_len] = traj_log_probs
        values[i, :seq_len] = traj_values
        is_tp[i, :seq_len] = traj_is_tp
        padding_mask[i, :seq_len] = 1

        # Compute GAE
        advs = torch.zeros(seq_len)
        rets = torch.zeros(seq_len)
        last_gae_lam = torch.tensor(0.0)
        gamma = 0.99
        lam = 0.95

        next_val = torch.tensor(0.0)

        for t in reversed(range(seq_len)):
            delta = traj_rewards[t] + gamma * next_val - traj_values[t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advs[t] = last_gae_lam
            rets[t] = traj_values[t] + last_gae_lam
            next_val = traj_values[t]

        processed_advantages.append(advs)
        processed_returns.append(rets)

    # Pad advantages and returns
    advantages = torch.zeros(batch_size, max_len)
    returns = torch.zeros(batch_size, max_len)

    for i in range(batch_size):
        seq_len = len(processed_advantages[i])
        advantages[i, :seq_len] = processed_advantages[i]
        returns[i, :seq_len] = processed_returns[i]

    return {
        "states": states.to(device),
        "actions": actions.to(device),
        "rewards": rewards.to(device),
        "log_probs": log_probs.to(device),
        "values": values.to(device),
        "is_teampreview": is_tp.to(device),
        "advantages": advantages.to(device),
        "returns": returns.to(device),
        "padding_mask": padding_mask.to(device),
    }


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_step():
    """
    Create a sample step dictionary.

    This represents one decision point in a battle trajectory.
    """
    return {
        "state": np.random.randn(100).astype(np.float32),
        "action": 42,
        "log_prob": -0.5,
        "value": 0.1,
        "reward": 0.0,
        "is_teampreview": False,
        "mask": None,
    }


@pytest.fixture
def sample_trajectory(sample_step):
    """
    Create a sample trajectory of 5 steps.

    A trajectory is a complete battle from teampreview to game end.
    """
    steps = []
    for i in range(5):
        step = sample_step.copy()
        step["state"] = np.random.randn(100).astype(np.float32)
        step["action"] = i
        step["is_teampreview"] = i == 0  # First step is teampreview
        steps.append(step)

    # Assign final reward
    for step in steps:
        step["reward"] = 1.0  # Win

    return steps


@pytest.fixture
def multiple_trajectories(sample_trajectory):
    """
    Create multiple trajectories of varying lengths.

    Tests padding logic when trajectories have different lengths.
    """
    trajectories = []

    # Short trajectory (3 steps)
    short_traj = sample_trajectory[:3]
    trajectories.append(short_traj)

    # Medium trajectory (5 steps)
    trajectories.append(sample_trajectory.copy())

    # Long trajectory (8 steps)
    long_traj = sample_trajectory.copy()
    for i in range(3):
        step = sample_trajectory[0].copy()
        step["state"] = np.random.randn(100).astype(np.float32)
        step["action"] = 10 + i
        long_traj.append(step)
    trajectories.append(long_traj)

    return trajectories


# =============================================================================
# COLLATE TRAJECTORIES TESTS
# =============================================================================


def test_collate_single_trajectory(sample_trajectory):
    """
    Test collating a single trajectory.

    With one trajectory, batch dimension should be 1.
    Sequence length should match trajectory length.

    Expected: Batch of shape (1, seq_len, ...).
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    assert batch["states"].shape[0] == 1  # Batch size
    assert batch["states"].shape[1] == 5  # Sequence length
    assert batch["states"].shape[2] == 100  # Feature dim


def test_collate_multiple_trajectories(multiple_trajectories):
    """
    Test collating multiple trajectories with padding.

    Trajectories of different lengths should be padded to max length.
    Padding positions should be marked in padding_mask.

    Expected: All trajectories padded to longest length.
    """
    batch = collate_trajectories_for_test(multiple_trajectories, "cpu")

    # Should have 3 trajectories
    assert batch["states"].shape[0] == 3

    # Max length is 8 (longest trajectory)
    assert batch["states"].shape[1] == 8

    # Check padding mask
    # Trajectory 0: length 3, so mask[:3] = True, mask[3:] = False
    assert batch["padding_mask"][0, :3].all()
    assert not batch["padding_mask"][0, 3:].any()

    # Trajectory 1: length 5
    assert batch["padding_mask"][1, :5].all()
    assert not batch["padding_mask"][1, 5:].any()

    # Trajectory 2: length 8 (no padding)
    assert batch["padding_mask"][2, :].all()


def test_collate_actions_preserved(sample_trajectory):
    """
    Test that action values are correctly preserved in collation.

    Actions should match input trajectory actions.

    Expected: batch['actions'] contains original action indices.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    for i, step in enumerate(sample_trajectory):
        assert batch["actions"][0, i].item() == step["action"]


def test_collate_rewards_preserved(sample_trajectory):
    """
    Test that rewards are correctly preserved in collation.

    Expected: batch['rewards'] contains original reward values.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    for i, step in enumerate(sample_trajectory):
        assert batch["rewards"][0, i].item() == step["reward"]


def test_collate_log_probs_preserved(sample_trajectory):
    """
    Test that log probabilities are correctly preserved.

    Expected: batch['log_probs'] contains original log prob values.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    for i, step in enumerate(sample_trajectory):
        assert np.isclose(batch["log_probs"][0, i].item(), step["log_prob"])


def test_collate_is_teampreview_preserved(sample_trajectory):
    """
    Test that is_teampreview flags are correctly preserved.

    Expected: batch['is_teampreview'] matches original flags.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    for i, step in enumerate(sample_trajectory):
        assert batch["is_teampreview"][0, i].item() == step["is_teampreview"]


# =============================================================================
# GAE COMPUTATION TESTS
# =============================================================================


def test_advantages_computed(sample_trajectory):
    """
    Test that advantages are computed during collation.

    GAE (Generalized Advantage Estimation) computes advantages
    for policy gradient updates.

    Expected: Advantages tensor exists and has correct shape.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    assert "advantages" in batch
    assert batch["advantages"].shape == (1, 5)


def test_returns_computed(sample_trajectory):
    """
    Test that returns are computed during collation.

    Returns = values + advantages, used for value function training.

    Expected: Returns tensor exists and has correct shape.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    assert "returns" in batch
    assert batch["returns"].shape == (1, 5)


def test_gae_with_positive_reward():
    """
    Test GAE computation with positive (win) reward.

    For a win (reward=1.0), advantages should generally be positive
    in the later steps of the trajectory.

    Expected: Final step advantage is positive.
    """
    # Create simple trajectory with win
    steps = [
        {
            "state": np.zeros(10),
            "action": 0,
            "log_prob": -0.5,
            "value": 0.0,
            "reward": 1.0,
            "is_teampreview": False,
        },
    ]

    batch = collate_trajectories_for_test([steps], "cpu")

    # With value=0 and reward=1, advantage should be positive
    assert batch["advantages"][0, 0].item() > 0


def test_gae_with_negative_reward():
    """
    Test GAE computation with negative (loss) reward.

    For a loss (reward=-1.0), advantages should generally be negative.

    Expected: Final step advantage is negative.
    """
    # Create simple trajectory with loss
    steps = [
        {
            "state": np.zeros(10),
            "action": 0,
            "log_prob": -0.5,
            "value": 0.0,
            "reward": -1.0,
            "is_teampreview": False,
        },
    ]

    batch = collate_trajectories_for_test([steps], "cpu")

    # With value=0 and reward=-1, advantage should be negative
    assert batch["advantages"][0, 0].item() < 0


def test_gae_discounting():
    """
    Test that GAE properly discounts future rewards.

    Earlier steps should have smaller absolute advantages than later
    steps when all rewards are at the end (sparse reward setting).

    Expected: Advantages decay towards earlier steps.
    """
    # Multi-step trajectory with reward only at end
    steps = [
        {
            "state": np.zeros(10),
            "action": 0,
            "log_prob": -0.5,
            "value": 0.0,
            "reward": 0.0,
            "is_teampreview": True,
        },
        {
            "state": np.zeros(10),
            "action": 1,
            "log_prob": -0.5,
            "value": 0.0,
            "reward": 0.0,
            "is_teampreview": False,
        },
        {
            "state": np.zeros(10),
            "action": 2,
            "log_prob": -0.5,
            "value": 0.0,
            "reward": 1.0,
            "is_teampreview": False,
        },
    ]

    batch = collate_trajectories_for_test([steps], "cpu")

    # Advantage at step 2 (reward step) should be largest
    # Advantages at earlier steps should be discounted
    advs = batch["advantages"][0, :3]

    # All advantages should be positive (we won)
    assert (advs > 0).all()

    # Last step should have largest advantage
    assert advs[2] > advs[1] > advs[0]


# =============================================================================
# DEVICE PLACEMENT TESTS
# =============================================================================


def test_collate_places_tensors_on_cpu(sample_trajectory):
    """
    Test that tensors are placed on CPU when specified.

    Expected: All tensors on CPU device.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cpu")

    for key, tensor in batch.items():
        assert tensor.device.type == "cpu", f"{key} not on CPU"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_collate_places_tensors_on_cuda(sample_trajectory):
    """
    Test that tensors are placed on CUDA when specified.

    Expected: All tensors on CUDA device.
    """
    batch = collate_trajectories_for_test([sample_trajectory], "cuda")

    for key, tensor in batch.items():
        assert tensor.device.type == "cuda", f"{key} not on CUDA"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_collate_single_step_trajectory():
    """
    Test collating a trajectory with only one step.

    This can happen for very short battles (forfeit, error, etc.).

    Expected: Works without error, correct shapes.
    """
    steps = [
        {
            "state": np.zeros(50),
            "action": 0,
            "log_prob": -1.0,
            "value": 0.5,
            "reward": 1.0,
            "is_teampreview": True,
        },
    ]

    batch = collate_trajectories_for_test([steps], "cpu")

    assert batch["states"].shape == (1, 1, 50)
    assert batch["actions"].shape == (1, 1)


def test_collate_large_batch():
    """
    Test collating a large batch of trajectories.

    Training uses batches of 32-128 trajectories.

    Expected: Works efficiently for realistic batch sizes.
    """
    # Create 64 trajectories of length 10
    trajectories = []
    for _ in range(64):
        steps = []
        for i in range(10):
            steps.append(
                {
                    "state": np.random.randn(100).astype(np.float32),
                    "action": i,
                    "log_prob": -0.5,
                    "value": 0.1,
                    "reward": 1.0 if i == 9 else 0.0,
                    "is_teampreview": (i == 0),
                }
            )
        trajectories.append(steps)

    batch = collate_trajectories_for_test(trajectories, "cpu")

    assert batch["states"].shape == (64, 10, 100)


def test_collate_preserves_dtypes():
    """
    Test that collation uses correct data types.

    - States: float32
    - Actions: long (int64)
    - Rewards: float32
    - Masks: bool

    Expected: Each tensor has appropriate dtype.
    """
    steps = [
        {
            "state": np.zeros(10, dtype=np.float32),
            "action": 0,
            "log_prob": -0.5,
            "value": 0.1,
            "reward": 1.0,
            "is_teampreview": False,
        },
    ]

    batch = collate_trajectories_for_test([steps], "cpu")

    assert batch["states"].dtype == torch.float32
    assert batch["actions"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float32
    assert batch["padding_mask"].dtype == torch.bool
    assert batch["is_teampreview"].dtype == torch.bool


# =============================================================================
# TRAJECTORY TRUNCATION TESTS
# These test the fix for "Sequence length X exceeds max_seq_len" error.
# See src/elitefurretai/rl/DEBUG_LEARNINGS.md for details.
# =============================================================================


def collate_trajectories_with_truncation(
    trajectories: List[List[Dict[str, Any]]], device: str, max_seq_len: int = 17
):
    """
    Copy of collate_trajectories with truncation logic from train.py.

    Trajectories longer than max_seq_len are truncated to last N steps,
    since later game decisions are typically more impactful.
    """
    # Truncate long trajectories
    truncated_trajectories = []
    for traj in trajectories:
        if len(traj) > max_seq_len:
            truncated_trajectories.append(traj[-max_seq_len:])
        else:
            truncated_trajectories.append(traj)

    # Use the standard collation on truncated data
    return collate_trajectories_for_test(truncated_trajectories, device)


def test_trajectory_truncation_enforces_max_length():
    """
    Test that trajectories exceeding max_seq_len are truncated.

    Scenario: Battle with 39 decision points (many force_switches).
    Model has max_seq_len=17 for positional embeddings.

    Expected: Trajectory truncated to last 17 steps.
    """
    # Create a long trajectory (39 steps like we saw in debugging)
    long_trajectory = []
    for i in range(39):
        long_trajectory.append(
            {
                "state": np.random.randn(100).astype(np.float32),
                "action": i % 2025,
                "log_prob": -0.5,
                "value": 0.1,
                "reward": 1.0 if i == 38 else 0.0,
                "is_teampreview": (i == 0),
            }
        )

    batch = collate_trajectories_with_truncation([long_trajectory], "cpu", max_seq_len=17)

    # Sequence length should be capped at 17
    assert batch["states"].shape[1] == 17
    assert batch["actions"].shape[1] == 17


def test_trajectory_truncation_keeps_last_steps():
    """
    Test that truncation keeps the LAST N steps, not first N.

    Rationale: Later game decisions (endgame) are more impactful
    and shouldn't be discarded.
    """
    trajectory = []
    for i in range(25):
        trajectory.append(
            {
                "state": np.ones(10, dtype=np.float32) * i,  # State encodes step index
                "action": i,
                "log_prob": -0.5,
                "value": 0.1,
                "reward": 1.0 if i == 24 else 0.0,
                "is_teampreview": (i == 0),
            }
        )

    batch = collate_trajectories_with_truncation([trajectory], "cpu", max_seq_len=10)

    # Should keep steps 15-24 (last 10)
    actions = batch["actions"][0].tolist()
    assert actions == list(range(15, 25)), f"Expected last 10 steps, got {actions}"


def test_trajectory_truncation_preserves_short_trajectories():
    """
    Test that trajectories shorter than max_seq_len are not modified.
    """
    short_trajectory = []
    for i in range(5):
        short_trajectory.append(
            {
                "state": np.random.randn(10).astype(np.float32),
                "action": i,
                "log_prob": -0.5,
                "value": 0.1,
                "reward": 1.0 if i == 4 else 0.0,
                "is_teampreview": (i == 0),
            }
        )

    batch = collate_trajectories_with_truncation([short_trajectory], "cpu", max_seq_len=17)

    # Should preserve original length
    assert batch["states"].shape[1] == 5
    actions = batch["actions"][0].tolist()
    assert actions == [0, 1, 2, 3, 4]


def test_trajectory_truncation_mixed_lengths():
    """
    Test batch with mix of short and long trajectories.

    Expected: Long ones truncated, short ones preserved, proper padding.
    """
    trajectories = []

    # Short trajectory (5 steps)
    short = [
        {
            "state": np.ones(10, dtype=np.float32) * i,
            "action": i,
            "log_prob": -0.5,
            "value": 0.1,
            "reward": 0.0,
            "is_teampreview": False,
        }
        for i in range(5)
    ]
    trajectories.append(short)

    # Long trajectory (25 steps)
    long = [
        {
            "state": np.ones(10, dtype=np.float32) * (100 + i),
            "action": 100 + i,
            "log_prob": -0.5,
            "value": 0.1,
            "reward": 0.0,
            "is_teampreview": False,
        }
        for i in range(25)
    ]
    trajectories.append(long)

    batch = collate_trajectories_with_truncation(trajectories, "cpu", max_seq_len=10)

    # Max length should be 10 (from truncated long trajectory)
    assert batch["states"].shape == (2, 10, 10)

    # First trajectory should be padded (original 5 steps + 5 padding)
    assert batch["padding_mask"][0].sum() == 5

    # Second trajectory should be full (10 steps after truncation)
    assert batch["padding_mask"][1].sum() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
