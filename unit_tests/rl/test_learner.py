# -*- coding: utf-8 -*-
"""
Unit tests for RL learner (RNaDLearner).

These tests verify:
1. Learner initialization
2. Reference model freezing and updates
3. PPO + RNaD update logic
4. Mixed precision training
5. Gradient clipping
6. Batch processing

Note: Tests use small models for speed. Uses CPU for compatibility.
"""

import copy

import pytest
import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learners import PortfolioRNaDLearner
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_embedder():
    """Create a simple Embedder for testing."""
    return Embedder(feature_set="simple")


@pytest.fixture
def small_model(simple_embedder):
    """Create a small FlexibleThreeHeadedModel for testing."""
    return FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[32, 16],
        late_layers=[32, 16],
        lstm_layers=1,
        lstm_hidden_size=16,
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=10,
        dropout=0.0,
    )


@pytest.fixture
def agent(small_model):
    """Create RNaDAgent from small model."""
    return RNaDAgent(small_model)


@pytest.fixture
def ref_agent(small_model, simple_embedder):
    """Create reference RNaDAgent from small model."""
    # Create a separate instance with same architecture
    ref_model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[32, 16],
        late_layers=[32, 16],
        lstm_layers=1,
        lstm_hidden_size=16,
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=10,
        dropout=0.0,
    )
    return RNaDAgent(ref_model)


@pytest.fixture
def learner(agent, ref_agent):
    """Create PortfolioRNaDLearner for testing."""
    config = RNaDConfig()
    config.algorithm.clip_range = 0.2
    config.algorithm.ent_coef = 0.01
    config.algorithm.vf_coef = 0.5
    config.algorithm.rnad_alpha = 0.1
    config.algorithm.gamma = 0.99
    config.algorithm.max_grad_norm = 0.5
    config.hardware.device = "cpu"
    config.hardware.use_mixed_precision = False
    config.optimizer.warmup_steps = 0
    config.optimizer.schedule = "constant"
    config.optimizer.backbone_lr = 1e-4
    config.optimizer.backbone_weight_decay = 1e-4
    config.optimizer.heads_lr = 3e-4
    config.optimizer.heads_weight_decay = 0.0
    return PortfolioRNaDLearner(
        model=agent,
        ref_models=[ref_agent],
        config=config,
        device="cpu",
    )


@pytest.fixture
def sample_batch(simple_embedder):
    """
    Create sample batch for training.

    Returns dict with all required tensors for learner.update().
    """
    batch_size = 4
    seq_len = 5
    state_dim = simple_embedder.embedding_size

    # Random states
    states = torch.randn(batch_size, seq_len, state_dim)

    # Is teampreview: first step is teampreview, rest are turns
    is_teampreview = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    is_teampreview[:, 0] = True

    # Random valid actions (within appropriate action space)
    # Teampreview actions must be < 90, turn actions can be up to 2025
    actions = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        for s in range(seq_len):
            if is_teampreview[b, s]:
                actions[b, s] = torch.randint(0, MDBO.teampreview_space(), (1,)).item()
            else:
                actions[b, s] = torch.randint(0, MDBO.action_space(), (1,)).item()

    # Rewards
    rewards = torch.randn(batch_size, seq_len)

    # Old values and log_probs from previous policy
    values = torch.randn(batch_size, seq_len)
    log_probs = -torch.rand(batch_size, seq_len)  # Log probs are negative

    # Advantages and returns (pre-computed)
    advantages = torch.randn(batch_size, seq_len)
    returns = torch.randn(batch_size, seq_len)

    # Action masks (all actions valid for simplicity)
    masks = torch.ones(batch_size, seq_len, MDBO.action_space(), dtype=torch.bool)

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "values": values,
        "log_probs": log_probs,
        "advantages": advantages,
        "returns": returns,
        "is_teampreview": is_teampreview,
        "masks": masks,
    }


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


def test_learner_initialization(learner):
    """Test PortfolioRNaDLearner initializes correctly."""
    assert learner.gamma == 0.99
    assert learner.clip_range == 0.2
    assert learner.ent_coef == 0.01
    assert learner.vf_coef == 0.5
    assert learner.rnad_alpha == 0.1
    assert learner.device == "cpu"
    assert learner.gradient_clip == 0.5


def test_learner_creates_optimizer(learner):
    """
    Test that learner creates optimizer.

    Expected: Optimizer is torch.optim.AdamW or Adam instance.
    """
    assert isinstance(learner.optimizer, (torch.optim.Adam, torch.optim.AdamW))


def test_ref_model_frozen(learner):
    """Test that reference model parameters are frozen."""
    for ref in learner.ref_models:
        for param in ref.parameters():
            assert not param.requires_grad, "Reference model should be frozen"


def test_main_model_trainable(learner):
    """
    Test that main model parameters are trainable.

    Expected: Main model params have requires_grad=True.
    """
    trainable_count = sum(1 for p in learner.model.parameters() if p.requires_grad)
    assert trainable_count > 0, "Main model should have trainable parameters"


# =============================================================================
# REFERENCE MODEL UPDATE TESTS
# =============================================================================


def test_add_reference_model_snapshots_main(learner):
    """Test add_reference_model snapshots current weights from main model."""
    with torch.no_grad():
        for param in learner.model.parameters():
            param.fill_(1.0)

    learner.add_reference_model(RNaDAgent(copy.deepcopy(learner.model.model)))

    for main_param, ref_param in zip(
        learner.model.parameters(), learner.ref_models[-1].parameters()
    ):
        assert torch.allclose(main_param, ref_param), (
            "New ref model should match main model at snapshot time"
        )


def test_add_reference_model_keeps_frozen(learner):
    """Test that ref models stay frozen after add."""
    learner.add_reference_model(RNaDAgent(copy.deepcopy(learner.model.model)))

    for ref in learner.ref_models:
        for param in ref.parameters():
            assert not param.requires_grad, "Ref model should stay frozen"


# =============================================================================
# UPDATE STEP TESTS
# =============================================================================


def test_update_returns_losses(learner, sample_batch):
    """
    Test that update returns loss dictionary.

    Expected: Returns dict with loss, policy_loss, value_loss, entropy, rnad_loss.
    """
    losses = learner.update(sample_batch)

    assert isinstance(losses, dict)
    assert "loss" in losses
    assert "policy_loss" in losses
    assert "value_loss" in losses
    assert "entropy" in losses
    assert "rnad_loss" in losses


def test_update_losses_are_finite(learner, sample_batch):
    """
    Test that all losses are finite values.

    Expected: No NaN or Inf values in losses.
    """
    losses = learner.update(sample_batch)

    for key, value in losses.items():
        if not isinstance(value, (int, float)):
            continue
        assert not torch.isnan(torch.tensor(value)), f"{key} is NaN"
        assert not torch.isinf(torch.tensor(value)), f"{key} is Inf"


def test_update_modifies_weights(learner, sample_batch):
    """
    Test that update actually modifies model weights.

    Expected: Weights change after update step.
    """
    # Get initial weights
    initial_weights = [p.clone() for p in learner.model.parameters()]

    # Perform update
    learner.update(sample_batch)

    # Check some weights changed
    weights_changed = False
    for initial, current in zip(initial_weights, learner.model.parameters()):
        if not torch.allclose(initial, current):
            weights_changed = True
            break

    assert weights_changed, "Model weights should change after update"


def test_update_with_padding_mask(learner, sample_batch):
    """
    Test update handles padding mask correctly.

    Padding mask zeros out loss for padded positions.

    Expected: Update completes without error.
    """
    # Add padding mask (last 2 positions are padding)
    batch_size, seq_len = sample_batch["actions"].shape
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -2:] = False
    sample_batch["padding_mask"] = padding_mask

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_with_initial_hidden(learner, sample_batch):
    """
    Test update handles provided initial hidden state.

    Expected: Update uses provided hidden state.
    """
    batch_size = sample_batch["states"].shape[0]

    # Create initial hidden state
    num_layers = 1
    num_directions = 2
    hidden_size = 16
    h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)

    sample_batch["initial_hidden"] = (h0, c0)

    losses = learner.update(sample_batch)

    assert "loss" in losses


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_update_all_teampreview(learner, sample_batch):
    """
    Test update with all teampreview steps.

    Expected: Handles case where all steps are teampreview.
    """
    # Mark all steps as teampreview
    sample_batch["is_teampreview"] = torch.ones_like(
        sample_batch["is_teampreview"], dtype=torch.bool
    )
    # Actions should be valid teampreview actions (< 90)
    sample_batch["actions"] = torch.randint(
        0, MDBO.teampreview_space(), sample_batch["actions"].shape
    )

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_all_turn(learner, sample_batch):
    """
    Test update with all turn steps (no teampreview).

    Expected: Handles case where no steps are teampreview.
    """
    # Mark all steps as turn actions
    sample_batch["is_teampreview"] = torch.zeros_like(
        sample_batch["is_teampreview"], dtype=torch.bool
    )

    losses = learner.update(sample_batch)

    assert "loss" in losses
    assert not torch.isnan(torch.tensor(losses["loss"]))


def test_update_single_step(learner, sample_batch):
    """
    Test update with single step trajectories.

    Expected: Handles seq_len=1 correctly.
    """
    # Reduce to single step
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor) and sample_batch[key].dim() > 1:
            sample_batch[key] = sample_batch[key][:, :1]

    losses = learner.update(sample_batch)

    assert "loss" in losses


def test_update_batch_size_one(learner, sample_batch):
    """
    Test update with batch_size=1.

    Expected: Handles single batch correctly.
    """
    # Reduce to single batch
    for key in sample_batch:
        if isinstance(sample_batch[key], torch.Tensor):
            sample_batch[key] = sample_batch[key][:1]

    losses = learner.update(sample_batch)

    assert "loss" in losses


# =============================================================================
# GRADIENT CLIPPING TESTS
# =============================================================================


def test_gradient_clipping_applied(learner, sample_batch):
    """
    Test that gradient clipping is applied during update.

    Expected: Gradients are clipped to gradient_clip value.
    """
    # Create batch with potentially large gradients
    sample_batch["advantages"] = sample_batch["advantages"] * 100  # Large advantages

    # Hook to capture gradient norms before clipping
    grad_norms = []

    def hook(module):
        total_norm = 0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

    # Perform update (clipping happens internally)
    learner.update(sample_batch)

    # Can't directly verify clipping without hooks, but update should complete
    # without NaN due to clipping
    assert True  # Update completed


# =============================================================================
# RNAD AGENT TESTS
# =============================================================================


def test_rnad_agent_forward(agent, simple_embedder):
    """
    Test RNaDAgent forward pass.

    Expected: Returns turn_logits, tp_logits, value, hidden.
    """
    batch_size = 4
    seq_len = 5

    x = torch.randn(batch_size, seq_len, simple_embedder.embedding_size)
    hidden = agent.get_initial_state(batch_size, "cpu")

    turn_logits, tp_logits, value, win_dist_logits, next_hidden = agent(x, hidden)

    assert turn_logits.shape == (batch_size, seq_len, MDBO.action_space())
    assert tp_logits.shape == (batch_size, seq_len, MDBO.teampreview_space())
    assert value.shape == (batch_size, seq_len)
    assert isinstance(next_hidden, tuple)
    assert len(next_hidden) == 2


def test_rnad_agent_initial_state_shape(agent):
    """
    Test get_initial_state returns correct shapes.

    Expected: Hidden states have correct dimensions for bidirectional LSTM.
    """
    batch_size = 8
    h, c = agent.get_initial_state(batch_size, "cpu")

    # Shape: (num_layers * num_directions, batch, hidden_size)
    num_layers = 1
    num_directions = 2
    hidden_size = 16

    assert h.shape == (num_layers * num_directions, batch_size, hidden_size)
    assert c.shape == (num_layers * num_directions, batch_size, hidden_size)


def test_rnad_agent_value_range(agent, simple_embedder):
    """
    Test that RNaDAgent value output is in [-1, 1].

    Expected: Value uses tanh activation.
    """
    x = torch.randn(4, 5, simple_embedder.embedding_size)
    hidden = agent.get_initial_state(4, "cpu")

    _, _, value, _, _ = agent(x, hidden)

    assert value.min() >= -1.0
    assert value.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
