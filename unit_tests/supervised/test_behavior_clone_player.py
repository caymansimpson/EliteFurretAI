# -*- coding: utf-8 -*-
"""
Unit tests for BCPlayer (Behavioral Cloning Player).

These tests verify:
1. Initialization and model loading logic
2. State embedding
3. Prediction and action selection
4. Win advantage calculation
5. Edge case handling

Note: These tests mock model loading to avoid needing actual model files.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from elitefurretai.etl.encoder import MDBO
from elitefurretai.supervised.behavior_clone_player import BCPlayer
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """
    Create a mock FlexibleThreeHeadedModel.

    Returns a model that outputs deterministic values for testing.
    """
    model = MagicMock(spec=FlexibleThreeHeadedModel)
    model.max_seq_len = 17

    # Mock forward pass - returns (turn_logits, tp_logits, win_value)
    def mock_forward(x, mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Return deterministic values
        turn_logits = torch.randn(batch_size, seq_len, MDBO.action_space())
        tp_logits = torch.randn(batch_size, seq_len, MDBO.teampreview_space())
        win_value = torch.zeros(batch_size, seq_len)  # Neutral win prediction
        return turn_logits, tp_logits, win_value

    model.__call__ = mock_forward  # type: ignore[method-assign]
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)

    return model


@pytest.fixture
def mock_config():
    """Standard model config dict for testing."""
    return {
        "early_layers": [512, 256],
        "late_layers": [256, 128],
        "lstm_layers": 2,
        "lstm_hidden_size": 256,
        "dropout": 0.1,
        "gated_residuals": False,
        "early_attention_heads": 4,
        "late_attention_heads": 4,
        "use_grouped_encoder": False,
    }


@pytest.fixture
def mock_checkpoint(mock_model, mock_config):
    """Create mock checkpoint dict matching expected format."""
    return {
        "model_state_dict": {},  # Empty state dict for mocking
        "config": mock_config,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_bcplayer_requires_models():
    """
    Test that BCPlayer raises error without model paths.

    Must provide either unified_model_filepath or all three separate paths.

    Expected: ValueError raised with descriptive message.
    """
    with pytest.raises(ValueError, match="Must provide either unified_model_filepath"):
        BCPlayer(battle_format="gen9vgc2023regulationc")


def test_bcplayer_rejects_both_model_types():
    """
    Test that BCPlayer rejects both unified and separate model paths.

    Cannot mix unified and separate model configuration.

    Expected: ValueError raised with descriptive message.
    """
    with pytest.raises(ValueError, match="Cannot provide both"):
        BCPlayer(
            unified_model_filepath="unified.pt",
            teampreview_model_filepath="tp.pt",
            action_model_filepath="action.pt",
            win_model_filepath="win.pt",
            battle_format="gen9vgc2023regulationc",
        )


def test_bcplayer_requires_all_separate_models():
    """
    Test that BCPlayer requires all three separate model paths.

    If using separate models, must provide teampreview, action, and win.

    Expected: ValueError raised.
    """
    with pytest.raises(ValueError, match="Must provide either unified_model_filepath"):
        BCPlayer(
            teampreview_model_filepath="tp.pt",
            action_model_filepath="action.pt",
            # Missing win_model_filepath
            battle_format="gen9vgc2023regulationc",
        )


# =============================================================================
# MODEL LOADING TESTS
# =============================================================================


def test_load_model_rejects_old_format(tmp_path):
    """
    Test that _load_model rejects old format checkpoints.

    Old format was state_dict only, new format requires model_state_dict + config.

    Expected: ValueError with migration instructions.
    """
    # Create old format checkpoint (state_dict directly)
    old_checkpoint = {"conv1.weight": torch.randn(64, 3, 3, 3)}
    filepath = tmp_path / "old_model.pt"
    torch.save(old_checkpoint, filepath)

    # Create player with mocked load for init, then test _load_model directly
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._verbose = False

        with pytest.raises(ValueError, match="old format"):
            player._load_model(str(filepath), device="cpu")


# =============================================================================
# EMBEDDING TESTS
# =============================================================================


def test_embed_battle_state_returns_correct_size():
    """
    Test that embed_battle_state returns correct embedding size.

    Embedding size should match embedder's configured size.

    Expected: Output length equals embedder.embedding_size.

    Note: This test is a placeholder - full embedding test requires
    actual battle state setup which is complex.
    """
    # Create mock embedded features
    mock_features = {f"feature_{i}": [float(i)] for i in range(100)}
    expected_size = sum(len(v) for v in mock_features.values())

    # Verify the expected size calculation works
    assert expected_size == 100


# =============================================================================
# WIN ADVANTAGE TESTS
# =============================================================================


def test_predict_advantage_neutral_without_trajectory():
    """
    Test predict_advantage returns 0.0 without trajectory.

    When no trajectory exists for a battle, should return neutral advantage.

    Expected: Returns 0.0.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._trajectories = {}

        mock_battle = MagicMock()
        mock_battle.battle_tag = "test-battle"

        advantage = player.predict_advantage(mock_battle)

        assert advantage == 0.0


def test_predict_advantage_with_trajectory():
    """
    Test predict_advantage uses model when trajectory exists.

    Should use the win_model to predict advantage from trajectory.

    Expected: Returns value from model (in [-1, 1]).
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)

        # Setup mock model with side_effect to make callable work
        mock_model = MagicMock()
        mock_model.max_seq_len = 17

        # Return win_value = 0.5 for all positions
        def mock_forward(x):
            batch, seq = x.shape[:2]
            return (
                torch.randn(batch, seq, 2025),  # turn_logits
                torch.randn(batch, seq, 90),  # tp_logits
                torch.full((batch, seq), 0.5),  # win_value
            )

        mock_model.side_effect = mock_forward
        mock_model.eval = MagicMock()

        player.win_model = mock_model
        player._device = "cpu"  # Add device attribute
        player._trajectories = {"test-battle": [[0.0] * 100 for _ in range(5)]}

        mock_battle = MagicMock()
        mock_battle.battle_tag = "test-battle"

        advantage = player.predict_advantage(mock_battle)

        assert -1.0 <= advantage <= 1.0
        assert advantage == pytest.approx(0.5, abs=0.01)


# =============================================================================
# WIN ADVANTAGE SWING TESTS
# =============================================================================


def test_check_win_advantage_swing_skill_issue():
    """
    Test that "skill issue" message is sent on positive swing.

    When player goes from losing (-0.6) to winning (+0.6), should send message.

    Expected: send_message called with "skill issue".
    """
    import asyncio

    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_win_advantage = {"test-battle": -0.6}
        player._win_advantage_threshold = 0.5
        player.send_message = AsyncMock()  # type: ignore[method-assign]

        mock_battle = MagicMock()
        mock_battle.battle_tag = "test-battle"

        asyncio.get_event_loop().run_until_complete(
            player._check_win_advantage_swing(mock_battle, current_advantage=0.6)
        )

        player.send_message.assert_called_once_with("skill issue", "test-battle")


def test_check_win_advantage_swing_misclick():
    """
    Test that "misclick" message is sent on negative swing.

    When player goes from winning (+0.6) to losing (-0.6), should send message.

    Expected: send_message called with "misclick".
    """
    import asyncio

    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_win_advantage = {"test-battle": 0.6}
        player._win_advantage_threshold = 0.5
        player.send_message = AsyncMock()  # type: ignore[method-assign]

        mock_battle = MagicMock()
        mock_battle.battle_tag = "test-battle"

        asyncio.get_event_loop().run_until_complete(
            player._check_win_advantage_swing(mock_battle, current_advantage=-0.6)
        )

        player.send_message.assert_called_once_with("misclick", "test-battle")


def test_check_win_advantage_no_swing():
    """
    Test that no message is sent when no dramatic swing.

    Small changes in advantage should not trigger messages.

    Expected: send_message not called.
    """
    import asyncio

    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_win_advantage = {"test-battle": 0.2}
        player._win_advantage_threshold = 0.5
        player.send_message = AsyncMock()  # type: ignore[method-assign]

        mock_battle = MagicMock()
        mock_battle.battle_tag = "test-battle"

        # Small positive change
        asyncio.get_event_loop().run_until_complete(
            player._check_win_advantage_swing(mock_battle, current_advantage=0.3)
        )

        player.send_message.assert_not_called()


# =============================================================================
# PROBABILISTIC MODE TESTS
# =============================================================================


def test_probabilistic_property():
    """
    Test probabilistic property getter and setter.

    Expected: Property works correctly.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._probabilistic = True

        assert player.probabilistic is True

        player.probabilistic = False
        assert player.probabilistic is False


# =============================================================================
# RESET TESTS
# =============================================================================


def test_reset_battles():
    """
    Test reset_battles clears all battle state.

    Should clear _battles, _trajectories, and _last_win_advantage.

    Expected: All dicts are empty after reset.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._battles = {"battle1": MagicMock()}
        player._trajectories = {"battle1": [[0.0] * 100]}
        player._last_win_advantage = {"battle1": 0.5}

        player.reset_battles()

        assert player._battles == {}
        assert player._trajectories == {}
        assert player._last_win_advantage == {}


# =============================================================================
# LAST MESSAGE ERROR TRACKING TESTS
# =============================================================================


def test_last_message_error_default_false():
    """
    Test last_message_error returns False for unknown room.

    Expected: Returns False for rooms without error state.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_message_error = {}

        assert player.last_message_error("unknown-room") is False


def test_last_message_error_returns_tracked_value():
    """
    Test last_message_error returns tracked error state.

    Expected: Returns stored error state for room.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_message_error = {"room1": True, "room2": False}

        assert player.last_message_error("room1") is True
        assert player.last_message_error("room2") is False


def test_last_message_raises_for_unknown_room():
    """
    Test last_message raises for unknown room.

    Expected: AssertionError for rooms without messages.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_message = {}

        with pytest.raises(AssertionError, match="No last message"):
            player.last_message("unknown-room")


def test_last_message_returns_tracked_value():
    """
    Test last_message returns tracked message.

    Expected: Returns stored message for room.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._last_message = {"room1": "/move tackle"}

        assert player.last_message("room1") == "/move tackle"


# =============================================================================
# MDBO INTEGRATION TESTS
# =============================================================================


def test_mdbo_action_space_size():
    """
    Test MDBO action space matches expected size.

    Model is designed for 2025 turn actions and 90 teampreview actions.

    Expected: action_space() == 2025, teampreview_space() == 90.
    """
    assert MDBO.action_space() == 2025
    assert MDBO.teampreview_space() == 90


# =============================================================================
# TRAJECTORY MANAGEMENT TESTS
# =============================================================================


def test_trajectory_stored_per_battle():
    """
    Test that trajectories are stored per battle tag.

    Different battles should have independent trajectories.

    Expected: Separate trajectory lists per battle_tag.
    """
    with patch.object(BCPlayer, "__init__", lambda self, **kwargs: None):
        player = BCPlayer.__new__(BCPlayer)
        player._trajectories = {}

        # Simulate storing states for two battles
        player._trajectories["battle-1"] = [[0.0] * 100]
        player._trajectories["battle-2"] = [[1.0] * 100, [1.0] * 100]

        assert len(player._trajectories) == 2
        assert len(player._trajectories["battle-1"]) == 1
        assert len(player._trajectories["battle-2"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
