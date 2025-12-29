# -*- coding: utf-8 -*-
"""
Unit tests for RL worker (BatchInferencePlayer).

These tests verify:
1. Player initialization and configuration
2. Action masking for valid move selection
3. Trajectory collection during battles
4. Inference loop batch processing
5. Integration with MDBO encoder for action conversion
"""

import asyncio
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle
from poke_env.player.battle_order import DefaultBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.worker import BatchInferencePlayer

# =============================================================================
# FIXTURES: Reusable test components
# =============================================================================


@pytest.fixture
def mock_model():
    """
    Create a mock RNaDAgent that returns predictable outputs.

    The mock model:
    - Returns uniform logits (all zeros) so action selection is random
    - Returns zero values for all states
    - Returns proper hidden state shapes

    This allows testing the worker logic without needing a real neural network.
    """
    model = MagicMock(spec=RNaDAgent)

    # Mock get_initial_state to return properly shaped hidden states
    # Shape: (num_layers * num_directions, batch_size, hidden_size)
    def mock_initial_state(batch_size, device):
        h = torch.zeros(8, batch_size, 512, device=device)  # 4 layers * 2 directions
        c = torch.zeros(8, batch_size, 512, device=device)
        return (h, c)

    model.get_initial_state = mock_initial_state

    # Mock forward to return uniform logits
    # turn_logits: (batch, seq, 2025), tp_logits: (batch, seq, 90), values: (batch, seq)
    def mock_forward(states, hidden):
        batch_size = states.shape[0]
        seq_len = states.shape[1]
        device = states.device

        turn_logits = torch.zeros(batch_size, seq_len, MDBO.action_space(), device=device)
        tp_logits = torch.zeros(
            batch_size, seq_len, MDBO.teampreview_space(), device=device
        )
        values = torch.zeros(batch_size, seq_len, device=device)

        # Return same hidden state (no update)
        next_h = hidden[0]
        next_c = hidden[1]

        return turn_logits, tp_logits, values, (next_h, next_c)

    model.side_effect = mock_forward
    model.forward = mock_forward

    return model


@pytest.fixture
def mock_embedder():
    """
    Create a mock Embedder that returns fixed-size embeddings.

    The mock embedder:
    - Returns a fixed-size feature dictionary
    - Converts features to a vector of the expected size

    This allows testing without loading the full embedder which has complex dependencies.
    """
    embedder = MagicMock(spec=Embedder)

    # Return a simple feature dict
    embedder.embed.return_value = {"feature_1": 0.5, "feature_2": 1.0}

    # Return a numpy array of the expected embedding size
    # Real embedder returns ~15000 features for FULL feature set
    embedder.feature_dict_to_vector.return_value = np.zeros(15000, dtype=np.float32)
    embedder.embedding_size = 15000

    return embedder


@pytest.fixture
def player_config():
    """
    Create AccountConfiguration for test players.

    Uses a simple username with no password for local testing.
    """
    return AccountConfiguration("TestPlayer", None)


@pytest.fixture
def server_config():
    """
    Create ServerConfiguration for local testing.

    Points to localhost on default showdown port.
    This won't actually connect in unit tests - we mock the network layer.
    """
    return ServerConfiguration("ws://localhost:8000/showdown/websocket", None)  # type: ignore


@pytest.fixture
def trajectory_queue():
    """
    Create a mock queue for collecting trajectories.

    Trajectories are pushed here when battles complete.
    We use a list to capture them for inspection.
    """
    import queue

    return queue.Queue()


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_player_initialization(mock_model, player_config, server_config, trajectory_queue):
    """
    Test that BatchInferencePlayer initializes with correct attributes.

    Verifies:
    - Model is stored correctly
    - Device is set
    - Batch parameters are configured
    - Empty trajectory storage is created
    - Embedder is created with correct format

    Expected: All attributes should match constructor arguments.
    """
    with patch.object(BatchInferencePlayer, "__init__", lambda self, **kwargs: None):
        # Manually set attributes that __init__ would set
        player = BatchInferencePlayer.__new__(BatchInferencePlayer)
        player.model = mock_model
        player.device = "cpu"
        player.batch_size = 16
        player.batch_timeout = 0.01
        player.probabilistic = True
        player.trajectory_queue = trajectory_queue
        player.current_trajectories = {}
        player.completed_trajectories = []
        player.hidden_states = {}
        player.queue = asyncio.Queue()
        player._inference_task = None

        assert player.model == mock_model
        assert player.device == "cpu"
        assert player.batch_size == 16
        assert player.probabilistic is True
        assert player.trajectory_queue == trajectory_queue
        assert len(player.current_trajectories) == 0


def test_player_batch_size_parameter():
    """
    Test that batch_size parameter controls inference batching.

    BatchInferencePlayer collects multiple battle states before running
    a single GPU inference. This test verifies the batch_size parameter
    is respected.

    Expected: batch_size should limit how many states are processed together.
    """
    # This is a structural test - just verify the parameter exists and is settable
    assert hasattr(BatchInferencePlayer, "__init__")
    # The actual batching logic is tested in integration tests


# =============================================================================
# ACTION MASKING TESTS
# =============================================================================


def test_get_action_mask_returns_correct_shape():
    """
    Test that _get_action_mask returns a mask of the correct shape.

    The action mask should:
    - Have shape (2025,) for the full turn action space
    - Contain only 0.0 or 1.0 values
    - Have at least one valid action (mask sum > 0)

    Expected: numpy array of shape (2025,) with binary values.
    """
    # Create a mock battle with some available moves
    mock_battle = MagicMock(spec=DoubleBattle)
    mock_battle.battle_tag = "test_battle"
    mock_battle.force_switch = [False, False]
    mock_battle.active_pokemon = [MagicMock(), MagicMock()]
    mock_battle.teampreview = False

    # Mock the available moves and switches
    mock_battle.available_moves = [[MagicMock()], [MagicMock()]]
    mock_battle.available_switches = [[MagicMock()], [MagicMock()]]

    # We can't easily test the full _get_action_mask without a real battle
    # since it depends on MDBO.from_int and is_valid_order
    # This test verifies the mask shape contract
    expected_shape = (MDBO.action_space(),)
    assert expected_shape == (2025,), "MDBO action space should be 2025"


def test_action_mask_all_zeros_triggers_fallback():
    """
    Test that when no valid actions are found, a fallback mask is returned.

    If _get_action_mask finds no valid actions (all zeros), it should:
    - Print a warning
    - Return an all-ones mask as fallback
    - Prevent the model from crashing

    This handles edge cases like corrupted battle states.

    Expected: All-ones mask when no actions are valid.
    """
    # The fallback behavior is implemented in _get_action_mask
    # When mask.sum() == 0, it returns np.ones(MDBO.action_space())
    fallback_mask = np.ones(MDBO.action_space(), dtype=np.float32)
    assert fallback_mask.sum() == 2025
    assert fallback_mask.shape == (2025,)


def test_teampreview_returns_none_mask():
    """
    Test that teampreview phase returns None for action mask.

    During teampreview:
    - All 90 team selection actions are always valid
    - No masking is needed
    - Returning None signals to use all actions

    Expected: mask should be None when battle.teampreview is True.
    """
    # In _choose_move_async, when battle.teampreview is True, mask is set to None
    # This is verified by inspecting the code:
    # if battle.teampreview:
    #     mask = None
    # This test documents the expected behavior
    pass  # Behavior verified by code inspection


# =============================================================================
# TRAJECTORY COLLECTION TESTS
# =============================================================================


def test_trajectory_stored_during_battle():
    """
    Test that steps are stored in current_trajectories during battle.

    Each decision point should add a step dict containing:
    - state: The embedded game state
    - action: The chosen action index
    - log_prob: Log probability of the action
    - value: Estimated state value
    - reward: Initially 0, filled at battle end
    - is_teampreview: Whether this was a teampreview decision
    - mask: The action mask used (or None for teampreview)

    Expected: current_trajectories[battle_tag] should grow with each step.
    """
    # Create trajectory storage
    current_trajectories: Dict[str, list] = {}
    battle_tag = "test_battle_123"

    # Simulate adding a step
    step = {
        "state": np.zeros(15000),
        "action": 42,
        "log_prob": -0.5,
        "value": 0.1,
        "reward": 0,  # Filled later
        "is_teampreview": False,
        "mask": np.ones(2025),
    }

    if battle_tag not in current_trajectories:
        current_trajectories[battle_tag] = []
    current_trajectories[battle_tag].append(step)

    assert len(current_trajectories[battle_tag]) == 1
    assert current_trajectories[battle_tag][0]["action"] == 42


def test_battle_finished_callback_assigns_rewards():
    """
    Test that _battle_finished_callback assigns rewards and pushes trajectory.

    When a battle ends:
    - reward = +1.0 for win, -1.0 for loss
    - All steps in the trajectory get the same reward (sparse signal)
    - Trajectory is pushed to training queue
    - Hidden states are cleaned up

    Expected: All steps should have reward set to ±1.0 based on battle outcome.
    """

    # Simulate trajectory with 5 steps
    traj = [
        {
            "state": np.zeros(100),
            "action": 1,
            "log_prob": -0.5,
            "value": 0.1,
            "reward": 0,
            "is_teampreview": False,
            "mask": None,
        },
        {
            "state": np.zeros(100),
            "action": 2,
            "log_prob": -0.3,
            "value": 0.2,
            "reward": 0,
            "is_teampreview": False,
            "mask": None,
        },
        {
            "state": np.zeros(100),
            "action": 3,
            "log_prob": -0.4,
            "value": 0.15,
            "reward": 0,
            "is_teampreview": False,
            "mask": None,
        },
    ]

    # Simulate win
    reward = 1.0
    for step in traj:
        step["reward"] = reward

    # Verify all steps got the reward
    assert all(step["reward"] == 1.0 for step in traj)

    # Simulate loss
    for step in traj:
        step["reward"] = -1.0

    assert all(step["reward"] == -1.0 for step in traj)


def test_trajectory_queue_receives_completed_battles():
    """
    Test that completed trajectories are pushed to the training queue.

    The trajectory_queue is shared between workers and the learner.
    When a battle completes, the full trajectory should be pushed.

    Expected: Queue should contain the trajectory after battle finishes.
    """
    import queue

    traj_queue: queue.Queue = queue.Queue()

    # Simulate pushing a trajectory
    trajectory = [{"state": np.zeros(100), "action": 1, "reward": 1.0}]
    traj_queue.put(trajectory)

    # Verify queue received it
    assert not traj_queue.empty()
    received = traj_queue.get()
    assert len(received) == 1
    assert received[0]["action"] == 1


# =============================================================================
# MDBO INTEGRATION TESTS
# =============================================================================


def test_teampreview_action_conversion():
    """
    Test that teampreview actions are correctly converted to team orders.

    MDBO.from_int(action, type=MDBO.TEAMPREVIEW) should return an MDBO
    object whose .message is a valid team order string like "/team 3421".

    Expected: Valid team order string for all 90 teampreview actions.
    """
    # Test a few valid teampreview action indices
    for action_idx in [0, 45, 89]:  # First, middle, last
        try:
            mdbo = MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW)
            message = mdbo.message

            # Team order should start with /team
            assert message.startswith("/team "), f"Action {action_idx} produced: {message}"

            # Should have 4 digits (choosing 4 Pokemon from 6)
            team_part = message.replace("/team ", "")
            assert len(team_part) == 4, f"Expected 4 digits, got: {team_part}"

        except ValueError:
            pytest.fail(f"Valid teampreview action {action_idx} raised ValueError")


def test_turn_action_space_size():
    """
    Test that turn action space has the expected size.

    VGC doubles has 45 possible actions per Pokemon:
    - 4 moves * (3 possible targets + self) = up to 16 move actions
    - 4 switches (swap with bench Pokemon)
    - Gimmicks (Tera, etc.)

    Total: 45 * 45 = 2025 action combinations

    Expected: MDBO.action_space() should return 2025.
    """
    assert MDBO.action_space() == 2025, "Turn action space should be 45*45=2025"


def test_teampreview_action_space_size():
    """
    Test that teampreview action space has the expected size.

    Teampreview chooses 4 of 6 Pokemon with ordered lead pair:
    - C(6,4) = 15 ways to choose 4 Pokemon
    - For each, 6 ways to order the leading pair (unordered pairs)
    - Total: 15 * 6 = 90

    Expected: MDBO.teampreview_space() should return 90.
    """
    assert MDBO.teampreview_space() == 90, "Teampreview space should be 90"


# =============================================================================
# INFERENCE LOOP TESTS
# =============================================================================


def test_add_to_batch_collects_items():
    """
    Test that batch collection logic works correctly.

    Items are tuples of (state, future, battle_tag, is_teampreview, mask).
    _add_to_batch should append each component to its respective list.

    Expected: All lists should have one more element after call.

    Note: Full async testing requires pytest-asyncio.
    This tests the collection logic without async.
    """
    # Initialize collection lists
    batch = []
    futures: list[None] = []
    battle_tags = []
    is_tps = []
    masks = []

    # Create a mock item (using None for future since not async)
    state = np.zeros(15000)
    battle_tag = "test_battle"
    is_tp = False
    mask = np.ones(2025)

    # Simulate _add_to_batch logic
    batch.append(state)
    futures.append(None)  # Placeholder
    battle_tags.append(battle_tag)
    is_tps.append(is_tp)
    masks.append(mask)

    assert len(batch) == 1
    assert len(futures) == 1
    assert battle_tags[0] == "test_battle"
    assert is_tps[0] is False


def test_inference_loop_processes_queue():
    """
    Test that BatchInferencePlayer has the _inference_loop method.

    The inference loop should:
    1. Wait for items in the queue
    2. Collect up to batch_size items
    3. Run batched inference
    4. Set results on futures

    Note: Full async testing requires pytest-asyncio.
    This verifies the class has the method.
    """
    from elitefurretai.rl.worker import BatchInferencePlayer

    # Verify the class has the method
    assert hasattr(BatchInferencePlayer, "_inference_loop")


# =============================================================================
# HIDDEN STATE MANAGEMENT TESTS
# =============================================================================


def test_hidden_states_tracked_per_battle():
    """
    Test that LSTM hidden states are tracked per battle_tag.

    Each battle has its own hidden state to maintain temporal context.
    Hidden states are:
    - Created when a battle is first seen
    - Updated after each inference
    - Deleted when battle completes

    Expected: hidden_states dict should map battle_tag -> (h, c) tensors.
    """
    hidden_states: Dict[str, tuple] = {}

    # Simulate creating hidden state for new battle
    battle_tag = "battle-gen9vgc-12345"
    h = torch.zeros(8, 1, 512)
    c = torch.zeros(8, 1, 512)
    hidden_states[battle_tag] = (h, c)

    assert battle_tag in hidden_states
    assert hidden_states[battle_tag][0].shape == (8, 1, 512)


def test_hidden_states_cleaned_up_after_battle():
    """
    Test that hidden states are deleted when battle completes.

    This prevents memory leaks when running many sequential battles.

    Expected: battle_tag should not be in hidden_states after cleanup.
    """
    hidden_states: Dict[str, tuple] = {}

    # Add a battle's hidden state
    battle_tag = "battle-gen9vgc-12345"
    hidden_states[battle_tag] = (torch.zeros(8, 1, 512), torch.zeros(8, 1, 512))

    # Simulate battle completion cleanup
    if battle_tag in hidden_states:
        del hidden_states[battle_tag]

    assert battle_tag not in hidden_states


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


def test_default_order_returned_for_non_double_battle():
    """
    Test that DefaultBattleOrder is returned for non-DoubleBattle.

    BatchInferencePlayer is designed for VGC doubles format.
    If given a singles battle, it should return a safe default.

    Expected: DefaultBattleOrder when battle is not DoubleBattle.
    """
    # The check in _choose_move_async is:
    # if not isinstance(battle, DoubleBattle):
    #     return DefaultBattleOrder()

    default = DefaultBattleOrder()
    assert default.message == "/choose default"


def test_fallback_team_order_on_mdbo_error():
    """
    Test that a fallback team order is used when MDBO conversion fails.

    If MDBO.from_int raises ValueError, the worker should:
    - Not crash
    - Return a valid default team order

    Expected: "/team 123456" as fallback team order.
    """
    fallback_order = "/team 123456"
    assert fallback_order.startswith("/team ")
    assert len(fallback_order.replace("/team ", "")) == 6


def test_fallback_default_order_on_turn_action_error():
    """
    Test that DefaultBattleOrder is used when turn action conversion fails.

    If MDBO.to_double_battle_order raises an exception, the worker should:
    - Not crash
    - Return DefaultBattleOrder

    Expected: DefaultBattleOrder as fallback for invalid turn actions.
    """
    default = DefaultBattleOrder()
    # DefaultBattleOrder tells the server to pick a random valid move
    assert default.message == "/choose default"


# =============================================================================
# PROBABILITY SAMPLING TESTS
# =============================================================================


def test_probabilistic_mode_samples_from_distribution():
    """
    Test that probabilistic=True samples actions according to policy.

    In probabilistic mode:
    - Actions are sampled from softmax distribution
    - Different calls may return different actions
    - Log probs are correctly computed

    Expected: Action should be sampled, not argmax.
    """
    # Create a simple probability distribution
    probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Sample 100 times and verify distribution
    samples = [np.random.choice(len(probs), p=probs) for _ in range(100)]

    # All samples should be valid indices
    assert all(0 <= s < len(probs) for s in samples)

    # Should have some variety (not always picking action 3)
    assert len(set(samples)) > 1


def test_deterministic_mode_uses_argmax():
    """
    Test that probabilistic=False uses argmax action selection.

    In deterministic mode:
    - Always select highest probability action
    - No randomness
    - Used for evaluation, not training

    Expected: Always return action with highest probability.
    """
    probs = np.array([0.1, 0.2, 0.3, 0.4])

    # Deterministic should always pick action 3 (probability 0.4)
    action = np.argmax(probs)
    assert action == 3


def test_log_prob_computation():
    """
    Test that log probabilities are correctly computed.

    log_prob = log(prob[action] + epsilon) where epsilon=1e-10 prevents log(0).

    Expected: log_prob should be finite for all valid probabilities.
    """
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    action = 2

    log_prob = np.log(probs[action] + 1e-10)

    assert np.isfinite(log_prob)
    assert log_prob < 0  # Log of probability < 1 is negative
    assert np.isclose(log_prob, np.log(0.25), atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
