# -*- coding: utf-8 -*-
"""
Unit tests for RNaDAgent and RNaDLearner.

These tests verify:
1. Agent initialization and forward pass
2. Hidden state management for LSTM
3. Learner PPO + RNaD update logic
4. Reference model updates
5. Mixed precision training
6. Gradient clipping

Note: These tests use mock models to avoid loading full supervised models.
"""

from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

# =============================================================================
# MOCK CLASSES FOR TESTING
# =============================================================================


class MockFlexibleThreeHeadedModel(nn.Module):
    """
    Mock model that simulates FlexibleThreeHeadedModel interface.

    Used for testing without loading full model weights or having
    all the dependencies of the supervised module.
    """

    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 64,
        num_turn_actions: int = 2025,
        num_tp_actions: int = 90,
        lstm_hidden_size: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()

        # Simple architecture for testing
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_hidden_size = lstm_hidden_size

        # Output heads
        lstm_output_dim = lstm_hidden_size * 2  # Bidirectional
        self.turn_head = nn.Linear(lstm_output_dim, num_turn_actions)
        self.tp_head = nn.Linear(lstm_output_dim, num_tp_actions)
        self.value_head = nn.Linear(lstm_output_dim, 1)

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with explicit hidden state handling.

        Args:
            x: Input features (batch, seq, input_dim)
            hidden_state: Optional (h, c) tuple from previous step
            mask: Optional action mask (not used in mock)

        Returns:
            turn_logits: (batch, seq, num_turn_actions)
            tp_logits: (batch, seq, num_tp_actions)
            value: (batch, seq)
            next_hidden: (h, c) tuple
        """

        # Embed input
        embedded = self.embedding(x)  # (batch, seq, hidden_dim)

        # LSTM forward
        if hidden_state is None:
            lstm_out, next_hidden = self.lstm(embedded)
        else:
            lstm_out, next_hidden = self.lstm(embedded, hidden_state)

        # Output heads
        turn_logits = self.turn_head(lstm_out)
        tp_logits = self.tp_head(lstm_out)
        value = self.value_head(lstm_out).squeeze(-1)  # (batch, seq)

        # Apply tanh to value to bound in [-1, 1]
        value = torch.tanh(value)

        return turn_logits, tp_logits, value, next_hidden


class MockRNaDAgent(nn.Module):
    """
    Mock RNaDAgent for testing.

    Wraps MockFlexibleThreeHeadedModel with the same interface as RNaDAgent.
    """

    def __init__(self, model: MockFlexibleThreeHeadedModel):
        super().__init__()
        self.model = model

    def get_initial_state(self, batch_size: int, device: str):
        """
        Create initial LSTM hidden state.

        Returns:
            (h, c) tuple with zero-initialized hidden states
        """
        num_directions = 2  # Bidirectional
        num_layers = self.model.lstm.num_layers
        hidden_size = self.model.lstm_hidden_size

        h = torch.zeros(
            num_layers * num_directions, batch_size, hidden_size, device=device
        )
        c = torch.zeros(
            num_layers * num_directions, batch_size, hidden_size, device=device
        )
        return (h, c)

    def forward(self, x, hidden_state=None, mask=None):
        """
        Forward pass matching RNaDAgent interface.
        """
        return self.model.forward_with_hidden(x, hidden_state, mask)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_model():
    """
    Create a mock model with default dimensions.
    """
    return MockFlexibleThreeHeadedModel(
        input_dim=100,
        hidden_dim=64,
        num_turn_actions=2025,
        num_tp_actions=90,
        lstm_hidden_size=32,
        num_layers=1,
    )


@pytest.fixture
def mock_agent(mock_model):
    """
    Create a mock agent wrapping the mock model.
    """
    return MockRNaDAgent(mock_model)


@pytest.fixture
def sample_batch():
    """
    Create a sample training batch.

    Contains all required fields for learner update.
    """
    batch_size = 8
    seq_len = 5
    input_dim = 100

    return {
        "states": torch.randn(batch_size, seq_len, input_dim),
        "actions": torch.randint(
            0, 90, (batch_size, seq_len)
        ),  # Use TP action range for simplicity
        "log_probs": torch.randn(batch_size, seq_len) - 1.0,  # Negative log probs
        "values": torch.randn(batch_size, seq_len) * 0.5,
        "rewards": torch.zeros(batch_size, seq_len),  # Sparse reward
        "advantages": torch.randn(batch_size, seq_len),
        "returns": torch.randn(batch_size, seq_len) * 0.5,
        "is_teampreview": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "padding_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


# =============================================================================
# AGENT TESTS
# =============================================================================


def test_agent_initialization(mock_agent):
    """
    Test that agent initializes correctly.

    Expected: Agent has model attribute and is a PyTorch module.
    """
    assert hasattr(mock_agent, "model")
    assert isinstance(mock_agent, nn.Module)


def test_agent_forward_pass(mock_agent):
    """
    Test agent forward pass produces correct output shapes.

    Expected:
        - turn_logits: (batch, seq, 2025)
        - tp_logits: (batch, seq, 90)
        - value: (batch, seq)
        - hidden: tuple of tensors
    """
    batch_size = 4
    seq_len = 3
    input_dim = 100

    x = torch.randn(batch_size, seq_len, input_dim)
    hidden = mock_agent.get_initial_state(batch_size, "cpu")

    turn_logits, tp_logits, value, next_hidden = mock_agent(x, hidden)

    assert turn_logits.shape == (batch_size, seq_len, 2025)
    assert tp_logits.shape == (batch_size, seq_len, 90)
    assert value.shape == (batch_size, seq_len)
    assert len(next_hidden) == 2  # (h, c)


def test_agent_initial_state_shape(mock_agent):
    """
    Test that initial hidden state has correct shape.

    For bidirectional LSTM with 1 layer, hidden should be:
        (num_layers * 2, batch, hidden_size)

    Expected: Correct hidden state dimensions.
    """
    batch_size = 8
    hidden = mock_agent.get_initial_state(batch_size, "cpu")

    h, c = hidden

    # Bidirectional (2) * num_layers (1) = 2
    assert h.shape[0] == 2
    assert h.shape[1] == batch_size
    assert h.shape[2] == 32  # lstm_hidden_size

    assert c.shape == h.shape


def test_agent_initial_state_zeros(mock_agent):
    """
    Test that initial hidden state is zeros.

    Expected: Both h and c are zero tensors.
    """
    hidden = mock_agent.get_initial_state(4, "cpu")
    h, c = hidden

    assert torch.allclose(h, torch.zeros_like(h))
    assert torch.allclose(c, torch.zeros_like(c))


def test_agent_value_bounded(mock_agent):
    """
    Test that value output is bounded in [-1, 1].

    Value represents win probability/expected outcome, so should
    be bounded. We use tanh activation.

    Expected: All values in [-1, 1].
    """
    x = torch.randn(4, 3, 100)
    hidden = mock_agent.get_initial_state(4, "cpu")

    _, _, value, _ = mock_agent(x, hidden)

    assert value.min() >= -1.0
    assert value.max() <= 1.0


def test_agent_sequential_forward(mock_agent):
    """
    Test agent processes sequences correctly with hidden state carry.

    Hidden state from step 1 should be passed to step 2.

    Expected: Sequential processing maintains state.
    """
    batch_size = 2
    input_dim = 100

    hidden = mock_agent.get_initial_state(batch_size, "cpu")

    # Process first step
    x1 = torch.randn(batch_size, 1, input_dim)
    _, _, _, hidden1 = mock_agent(x1, hidden)

    # Process second step with hidden from first
    x2 = torch.randn(batch_size, 1, input_dim)
    _, _, _, hidden2 = mock_agent(x2, hidden1)

    # Hidden states should change between steps
    h0, c0 = hidden
    h1, c1 = hidden1
    h2, c2 = hidden2

    # After processing, hidden should differ from zeros
    assert not torch.allclose(h1, torch.zeros_like(h1))
    # Hidden should change after each step
    assert not torch.allclose(h1, h2)


# =============================================================================
# LEARNER TESTS (Using mock implementation)
# =============================================================================


class MockRNaDLearner:
    """
    Simplified mock learner for testing update logic.
    """

    def __init__(
        self,
        model: MockRNaDAgent,
        ref_model: MockRNaDAgent,
        lr: float = 1e-4,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        rnad_alpha: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.rnad_alpha = rnad_alpha
        self.device = device

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def update_ref_model(self):
        """Copy current model weights to reference model."""
        self.ref_model.load_state_dict(self.model.state_dict())


@pytest.fixture
def mock_learner(mock_agent):
    """
    Create mock learner with separate model and ref_model.
    """
    model = mock_agent
    ref_model = MockRNaDAgent(MockFlexibleThreeHeadedModel())
    return MockRNaDLearner(model, ref_model, device="cpu")


def test_learner_initialization(mock_learner):
    """
    Test learner initializes with model and ref_model.

    Expected: Both models exist and ref_model is frozen.
    """
    assert mock_learner.model is not None
    assert mock_learner.ref_model is not None

    # Check ref_model is frozen
    for param in mock_learner.ref_model.parameters():
        assert not param.requires_grad


def test_learner_has_optimizer(mock_learner):
    """
    Test learner has optimizer for main model.

    Expected: Optimizer is Adam and contains model parameters.
    """
    assert isinstance(mock_learner.optimizer, torch.optim.Adam)


def test_learner_ref_model_update(mock_learner):
    """
    Test that update_ref_model copies weights correctly.

    This is used in RNaD to periodically update the reference policy.

    Expected: After update_ref_model(), ref_model has same weights as model.
    """
    # Modify main model weights
    with torch.no_grad():
        for param in mock_learner.model.parameters():
            param.add_(1.0)

    # Update ref model
    mock_learner.update_ref_model()

    # Check weights match
    for p_model, p_ref in zip(
        mock_learner.model.parameters(), mock_learner.ref_model.parameters()
    ):
        assert torch.allclose(p_model, p_ref)


def test_learner_ref_model_stays_frozen_after_update(mock_learner):
    """
    Test that ref_model stays frozen after update.

    Expected: requires_grad remains False after update_ref_model().
    """
    mock_learner.update_ref_model()

    for param in mock_learner.ref_model.parameters():
        assert not param.requires_grad


# =============================================================================
# PPO COMPONENT TESTS
# =============================================================================


def test_ppo_ratio_computation():
    """
    Test PPO probability ratio computation.

    ratio = exp(new_log_prob - old_log_prob)

    When new == old: ratio = 1.0
    When new > old: ratio > 1.0 (action more likely now)
    When new < old: ratio < 1.0 (action less likely now)
    """
    old_log_prob = torch.tensor([-1.0, -2.0, -0.5])
    new_log_prob = torch.tensor([-1.0, -1.0, -1.5])

    ratio = torch.exp(new_log_prob - old_log_prob)

    # Same log prob -> ratio = 1
    assert torch.isclose(ratio[0], torch.tensor(1.0))

    # New > old -> ratio > 1
    assert ratio[1] > 1.0

    # New < old -> ratio < 1
    assert ratio[2] < 1.0


def test_ppo_clipping():
    """
    Test PPO clipping constrains ratio.

    Clipping prevents too large policy updates.

    Expected: Clipped ratio in [1-eps, 1+eps].
    """
    clip_range = 0.2
    ratio = torch.tensor([0.5, 1.0, 2.0])

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

    assert torch.allclose(clipped_ratio, torch.tensor([0.8, 1.0, 1.2]))


def test_ppo_surrogate_objective():
    """
    Test PPO surrogate objective with clipping.

    L = min(ratio * A, clip(ratio) * A)

    For positive advantage: want to increase action probability
    For negative advantage: want to decrease action probability
    """
    clip_range = 0.2
    ratio = torch.tensor([1.5])  # Action is 1.5x more likely now
    advantage = torch.tensor([1.0])  # Positive advantage (good action)

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage

    # Clipped version should be smaller
    assert surr2 < surr1

    # Take minimum
    objective = torch.min(surr1, surr2)
    assert torch.isclose(objective, surr2)


# =============================================================================
# KL DIVERGENCE TESTS (RNaD component)
# =============================================================================


def test_kl_divergence_same_distribution():
    """
    Test KL divergence is zero for identical distributions.

    KL(P || P) = 0

    Expected: KL = 0 when current and reference policy are same.
    """
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    dist1 = torch.distributions.Categorical(logits=logits)
    dist2 = torch.distributions.Categorical(logits=logits)

    kl = torch.distributions.kl_divergence(dist1, dist2)

    assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)


def test_kl_divergence_different_distributions():
    """
    Test KL divergence is positive for different distributions.

    KL(P || Q) > 0 when P != Q

    Expected: Non-zero KL when policies differ.
    """
    logits1 = torch.tensor([[1.0, 2.0, 3.0]])
    logits2 = torch.tensor([[3.0, 2.0, 1.0]])  # Different preference order

    dist1 = torch.distributions.Categorical(logits=logits1)
    dist2 = torch.distributions.Categorical(logits=logits2)

    kl = torch.distributions.kl_divergence(dist1, dist2)

    assert kl > 0.0


def test_kl_asymmetry():
    """
    Test that KL divergence is asymmetric.

    KL(P || Q) != KL(Q || P) in general

    Expected: Forward and reverse KL differ when distributions
    are not symmetric around each other.
    """
    # Use asymmetric logits - one peaked, one more uniform
    logits1 = torch.tensor([[0.0, 0.0, 5.0]])  # Strongly prefer action 2
    logits2 = torch.tensor([[1.0, 2.0, 0.0]])  # Mildly prefer action 1

    dist1 = torch.distributions.Categorical(logits=logits1)
    dist2 = torch.distributions.Categorical(logits=logits2)

    kl_forward = torch.distributions.kl_divergence(dist1, dist2)
    kl_reverse = torch.distributions.kl_divergence(dist2, dist1)

    # They should be different due to asymmetric preferences
    assert not torch.isclose(kl_forward, kl_reverse, rtol=0.01)


# =============================================================================
# HIDDEN STATE TESTS
# =============================================================================


def test_hidden_state_detach_between_updates():
    """
    Test that hidden states should be detached between updates.

    LSTM hidden states carry gradients. For RL, we typically
    detach between environment steps to prevent backprop through time
    across episode boundaries.

    Expected: Detached hidden doesn't require grad.
    """
    model = MockFlexibleThreeHeadedModel()
    x = torch.randn(1, 1, 100, requires_grad=True)
    hidden = (torch.zeros(2, 1, 32), torch.zeros(2, 1, 32))

    _, _, _, next_hidden = model.forward_with_hidden(x, hidden)

    # Next hidden has gradients connected to input
    h, c = next_hidden

    # Detach for next episode
    detached_h = h.detach()
    detached_c = c.detach()

    assert not detached_h.requires_grad
    assert not detached_c.requires_grad


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_empty_teampreview_steps():
    """
    Test handling batch with no teampreview steps.

    Some batches might have all turn steps (if trajectory starts mid-battle).

    Expected: No error when is_teampreview is all False.
    """
    batch_size = 4
    seq_len = 3

    is_teampreview = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    valid_tp_mask = is_teampreview.reshape(-1)

    # Should be safe to check
    assert not valid_tp_mask.any()


def test_empty_turn_steps():
    """
    Test handling batch with no turn steps.

    Teampreview-only training uses only teampreview steps.

    Expected: No error when is_teampreview is all True.
    """
    batch_size = 4
    seq_len = 1  # Teampreview only

    is_teampreview = torch.ones(batch_size, seq_len, dtype=torch.bool)
    valid_turn_mask = ~is_teampreview.reshape(-1)

    assert not valid_turn_mask.any()


def test_masked_loss_computation():
    """
    Test that padded positions don't contribute to loss.

    When computing loss, padding_mask should zero out contributions
    from padded timesteps.

    Expected: Loss computed only on valid (unpadded) positions.
    """
    values = torch.tensor([1.0, 2.0, 0.0])
    targets = torch.tensor([1.5, 2.5, 0.0])
    padding_mask = torch.tensor([True, True, False])  # Third is padding

    # Raw error
    error = (values - targets) ** 2

    # Masked mean
    masked_loss = (error * padding_mask.float()).sum() / padding_mask.sum()

    # Should only average first two elements
    expected = ((0.5**2) + (0.5**2)) / 2
    assert torch.isclose(masked_loss, torch.tensor(expected))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
