# -*- coding: utf-8 -*-
"""
Unit tests for experience buffer/memory.
"""

import pytest
import torch
import numpy as np
import tempfile
import os

from elitefurretai.rl.memory import ExperienceBuffer


@pytest.fixture
def buffer():
    """Create a test experience buffer."""
    return ExperienceBuffer(
        buffer_size=100,
        gamma=0.99,
        gae_lambda=0.95,
        device='cpu',
        checkpoint_dir=None
    )


def test_buffer_initialization(buffer):
    """Test buffer initializes correctly."""
    assert buffer.buffer_size == 100
    assert buffer.gamma == 0.99
    assert buffer.gae_lambda == 0.95
    assert buffer.ptr == 0
    assert len(buffer.states) == 0


def test_store_experience(buffer):
    """Test storing a single experience."""
    state = np.random.randn(128)
    action = 5
    reward = 1.0
    value = 0.5
    log_prob = -2.0
    done = False
    action_mask = np.ones(2025, dtype=np.int8)
    
    buffer.store(state, action, reward, value, log_prob, done, action_mask)
    
    assert len(buffer.states) == 1
    assert buffer.ptr == 1
    assert np.array_equal(buffer.states[0], state)
    assert buffer.actions[0] == action


def test_store_multiple_experiences(buffer):
    """Test storing multiple experiences."""
    for i in range(50):
        state = np.random.randn(128)
        buffer.store(state, i, 1.0, 0.5, -1.0, False)
    
    assert len(buffer.states) == 50
    assert buffer.ptr == 50


def test_buffer_wraparound(buffer):
    """Test buffer wraps around correctly when full."""
    # Fill buffer beyond capacity
    for i in range(150):
        state = np.random.randn(128)
        buffer.store(state, i, 1.0, 0.5, -1.0, False)
    
    # Buffer should be at capacity
    assert len(buffer.states) == 100
    # Pointer should have wrapped
    assert buffer.ptr == 50


def test_calculate_advantages():
    """Test GAE advantage calculation."""
    buffer = ExperienceBuffer(10, gamma=0.99, gae_lambda=0.95, device='cpu')
    
    # Store simple trajectory
    for i in range(10):
        buffer.store(
            state=np.zeros(128),
            action=0,
            reward=1.0,  # Constant reward
            value=0.0,   # Zero value estimates
            log_prob=-1.0,
            done=False
        )
    
    buffer.calculate_advantages(last_value=0.0, done=True)
    
    # Check that advantages were calculated
    assert hasattr(buffer, 'advantages')
    assert hasattr(buffer, 'returns')
    assert len(buffer.advantages) == 10
    assert len(buffer.returns) == 10


def test_advantages_with_terminal_state():
    """Test advantages when episode terminates."""
    buffer = ExperienceBuffer(5, gamma=0.99, gae_lambda=0.95, device='cpu')
    
    for i in range(5):
        done = (i == 4)  # Last state is terminal
        buffer.store(np.zeros(128), 0, 1.0, 0.5, -1.0, done)
    
    buffer.calculate_advantages(last_value=1.0, done=True)
    
    # With terminal state, last value shouldn't be bootstrapped
    assert buffer.returns[-1] == buffer.advantages[-1] + buffer.values[-1]


def test_get_batch():
    """Test retrieving batch of experiences."""
    buffer = ExperienceBuffer(10, gamma=0.99, gae_lambda=0.95, device='cpu')
    
    # Fill buffer
    for i in range(10):
        buffer.store(np.random.randn(128), i, 1.0, 0.5, -1.0, False)
    
    buffer.calculate_advantages(0.0, True)
    batch = buffer.get()
    
    assert 'states' in batch
    assert 'actions' in batch
    assert 'log_probs' in batch
    assert 'returns' in batch
    assert 'advantages' in batch
    
    assert batch['states'].shape == (10, 128)
    assert batch['actions'].shape == (10,)
    assert batch['returns'].shape == (10,)
    assert batch['advantages'].shape == (10,)


def test_advantages_normalized():
    """Test that advantages are normalized."""
    buffer = ExperienceBuffer(10, gamma=0.99, gae_lambda=0.95, device='cpu')
    
    for i in range(10):
        buffer.store(np.random.randn(128), i, np.random.randn(), 0.5, -1.0, False)
    
    buffer.calculate_advantages(0.0, True)
    batch = buffer.get()
    
    advantages = batch['advantages']
    # Normalized advantages should have approximately mean=0, std=1
    assert torch.abs(advantages.mean()) < 0.1
    assert torch.abs(advantages.std() - 1.0) < 0.1


def test_action_masks_in_batch():
    """Test that action masks are included in batch."""
    buffer = ExperienceBuffer(10, gamma=0.99, gae_lambda=0.95, device='cpu')
    
    for i in range(10):
        mask = np.random.randint(0, 2, 2025, dtype=np.int8)
        buffer.store(np.zeros(128), i, 1.0, 0.5, -1.0, False, action_mask=mask)
    
    buffer.calculate_advantages(0.0, True)
    batch = buffer.get()
    
    assert 'action_masks' in batch
    assert batch['action_masks'].shape == (10, 2025)


def test_clear_buffer(buffer):
    """Test buffer clearing."""
    # Add some data
    for i in range(10):
        buffer.store(np.zeros(128), i, 1.0, 0.5, -1.0, False)
    
    buffer.clear()
    
    assert len(buffer.states) == 0
    assert buffer.ptr == 0


def test_save_and_load_buffer():
    """Test saving and loading buffer to/from disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        buffer = ExperienceBuffer(
            buffer_size=10,
            gamma=0.99,
            gae_lambda=0.95,
            device='cpu',
            checkpoint_dir=tmpdir
        )
        
        # Add data
        for i in range(10):
            buffer.store(np.random.randn(128), i, 1.0, 0.5, -1.0, False)
        
        buffer.calculate_advantages(0.0, True)
        
        # Save buffer
        worker_id = 0
        buffer.save_to_disk(worker_id)
        
        # Check file exists
        checkpoint_path = os.path.join(tmpdir, f'worker_{worker_id}_buffer.pt')
        assert os.path.exists(checkpoint_path)
        
        # Load buffer
        loaded_data = ExperienceBuffer.load_from_disk(tmpdir, worker_id, 'cpu')
        
        assert loaded_data is not None
        assert len(loaded_data['states']) == 10
        assert loaded_data['advantages'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
