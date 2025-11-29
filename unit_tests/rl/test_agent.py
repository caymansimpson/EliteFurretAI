# -*- coding: utf-8 -*-
"""
Unit tests for RL agent (ActorCritic model).
"""

import pytest
import torch
import tempfile
import os

from elitefurretai.rl.agent import ActorCritic
from elitefurretai.model_utils.encoder import MDBO


@pytest.fixture
def state_dim():
    """Sample state dimension."""
    return 128


@pytest.fixture
def action_dim():
    """Action dimension matching MDBO space."""
    return MDBO.action_space()


@pytest.fixture
def model(state_dim, action_dim):
    """Create a test model."""
    return ActorCritic(state_dim, action_dim, hidden_sizes=[64, 32])


def test_model_initialization(model, state_dim, action_dim):
    """Test model initializes with correct architecture."""
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Test that output heads have correct dimensions
    sample_state = torch.randn(1, state_dim)
    action_logits, state_value = model.forward(sample_state)
    
    assert action_logits.shape == (1, action_dim)
    assert state_value.shape == (1, 1)


def test_forward_pass(model, state_dim):
    """Test forward pass through model."""
    batch_size = 4
    states = torch.randn(batch_size, state_dim)
    
    action_logits, state_values = model.forward(states)
    
    assert action_logits.shape[0] == batch_size
    assert state_values.shape[0] == batch_size


def test_forward_with_action_mask(model, state_dim, action_dim):
    """Test forward pass with action masking."""
    states = torch.randn(1, state_dim)
    action_mask = torch.zeros(1, action_dim, dtype=torch.int8)
    action_mask[0, :10] = 1  # Only first 10 actions are valid
    
    action_logits, _ = model.forward(states, action_mask)
    
    # Masked actions should have -inf logits
    assert torch.all(torch.isinf(action_logits[0, 10:]))
    assert torch.all(torch.isfinite(action_logits[0, :10]))


def test_get_action_and_value(model, state_dim):
    """Test action sampling and value estimation."""
    state = torch.randn(1, state_dim)
    
    action, log_prob, value = model.get_action_and_value(state)
    
    assert action.shape == (1,)
    assert log_prob.shape == (1,)
    assert value.shape == (1, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_get_action_and_value_with_mask(model, state_dim, action_dim):
    """Test action sampling with action mask."""
    state = torch.randn(1, state_dim)
    action_mask = torch.zeros(1, action_dim, dtype=torch.int8)
    valid_actions = [0, 5, 10, 15, 20]
    action_mask[0, valid_actions] = 1
    
    action, log_prob, value = model.get_action_and_value(state, action_mask)
    
    # Action should be one of the valid actions
    assert action.item() in valid_actions
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_get_value_and_log_prob(model, state_dim, action_dim):
    """Test value and log probability computation."""
    batch_size = 8
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    
    values, log_probs, entropy = model.get_value_and_log_prob(states, actions)
    
    assert values.shape == (batch_size, 1)
    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert torch.isfinite(values).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()


def test_model_save_and_load(model, state_dim):
    """Test saving and loading model weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'test_model.pt')
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        # Create new model and load weights
        new_model = ActorCritic(state_dim, MDBO.action_space(), hidden_sizes=[64, 32])
        new_model.load_state_dict(torch.load(model_path))
        
        # Test that outputs match
        test_state = torch.randn(1, state_dim)
        with torch.no_grad():
            logits1, value1 = model.forward(test_state)
            logits2, value2 = new_model.forward(test_state)
        
        assert torch.allclose(logits1, logits2)
        assert torch.allclose(value1, value2)


def test_custom_hidden_sizes(state_dim, action_dim):
    """Test model with custom hidden layer sizes."""
    hidden_sizes = [256, 128, 64]
    model = ActorCritic(state_dim, action_dim, hidden_sizes=hidden_sizes)
    
    # Verify model works
    state = torch.randn(1, state_dim)
    logits, value = model.forward(state)
    
    assert logits.shape == (1, action_dim)
    assert value.shape == (1, 1)


def test_gradient_flow(model, state_dim):
    """Test that gradients flow through the model."""
    state = torch.randn(1, state_dim, requires_grad=True)
    
    logits, value = model.forward(state)
    loss = logits.sum() + value.sum()
    loss.backward()
    
    # Check that gradients exist
    assert state.grad is not None
    for param in model.parameters():
        assert param.grad is not None or not param.requires_grad


def test_action_distribution_sums_to_one(model, state_dim):
    """Test that action probabilities sum to 1."""
    state = torch.randn(1, state_dim)
    
    logits, _ = model.forward(state)
    probs = torch.softmax(logits, dim=-1)
    
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
