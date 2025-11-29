# -*- coding: utf-8 -*-
"""
Unit tests for RL learners (PPO and MMD).
"""

import pytest
import torch
import torch.multiprocessing as mp

from elitefurretai.rl.learner import PPOLearner, MMDLearner
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
def ppo_config():
    """PPO configuration."""
    return {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'buffer_size': 100,
        'ppo_epochs': 2,
        'num_minibatches': 4,
        'clip_coef': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'hidden_sizes': [64, 32]
    }


@pytest.fixture
def mmd_config():
    """MMD configuration."""
    return {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'buffer_size': 100,
        'mmd_epochs': 2,
        'num_minibatches': 4,
        'mmd_beta': 1.0,
        'state_dim': 128,
        'action_dim': MDBO.action_space(),
        'hidden_sizes': [64, 32]
    }


@pytest.fixture
def sample_experience(state_dim, action_dim):
    """Create sample experience batch."""
    batch_size = 100
    return {
        'states': torch.randn(batch_size, state_dim),
        'actions': torch.randint(0, action_dim, (batch_size,)),
        'log_probs': torch.randn(batch_size),
        'returns': torch.randn(batch_size),
        'advantages': torch.randn(batch_size),
        'action_masks': torch.randint(0, 2, (batch_size, action_dim), dtype=torch.int8),
        'worker_id': 0,
        'episode_stats': []
    }


def test_ppo_learner_initialization(state_dim, action_dim, ppo_config):
    """Test PPO learner initializes correctly."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = PPOLearner(state_dim, action_dim, weights_queue, experience_queue, ppo_config)
    
    assert learner is not None
    assert learner.model is not None
    assert learner.optimizer is not None
    assert learner.device is not None


def test_mmd_learner_initialization(state_dim, action_dim, mmd_config):
    """Test MMD learner initializes correctly."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = MMDLearner(state_dim, action_dim, weights_queue, experience_queue, mmd_config)
    
    assert learner is not None
    assert learner.model is not None
    assert learner.actor_optimizer is not None
    assert learner.critic_optimizer is not None
    assert learner.reference_model is not None


def test_ppo_learner_update(state_dim, action_dim, ppo_config, sample_experience):
    """Test PPO learner update step."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = PPOLearner(state_dim, action_dim, weights_queue, experience_queue, ppo_config)
    
    # Perform update
    update_info = learner._update(sample_experience)
    
    assert update_info is not None
    assert 'policy_loss' in update_info
    assert 'value_loss' in update_info
    assert 'entropy' in update_info
    assert isinstance(update_info['policy_loss'], float)


def test_mmd_learner_update(state_dim, action_dim, mmd_config, sample_experience):
    """Test MMD learner update step."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = MMDLearner(state_dim, action_dim, weights_queue, experience_queue, mmd_config)
    
    # Perform update
    update_info = learner._update(sample_experience)
    
    assert update_info is not None
    assert 'policy_loss' in update_info
    assert 'value_loss' in update_info
    assert 'kl_divergence' in update_info
    assert isinstance(update_info['kl_divergence'], float)


def test_ppo_distribute_weights(state_dim, action_dim, ppo_config):
    """Test PPO learner distributes weights correctly."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = PPOLearner(state_dim, action_dim, weights_queue, experience_queue, ppo_config)
    learner.distribute_weights()
    
    # Check that weights were put in queue
    assert not weights_queue.empty()
    weights = weights_queue.get()
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_mmd_distribute_weights(state_dim, action_dim, mmd_config):
    """Test MMD learner distributes weights correctly."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = MMDLearner(state_dim, action_dim, weights_queue, experience_queue, mmd_config)
    learner.distribute_weights()
    
    # Check that weights were put in queue
    assert not weights_queue.empty()
    weights = weights_queue.get()
    assert isinstance(weights, dict)
    assert len(weights) > 0


def test_mmd_reference_model_updates(state_dim, action_dim, mmd_config, sample_experience):
    """Test that MMD reference model gets updated after training."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = MMDLearner(state_dim, action_dim, weights_queue, experience_queue, mmd_config)
    
    # Get initial reference model weights
    initial_ref_weights = learner.reference_model.state_dict()['actor_head.weight'].clone()
    
    # Perform update (which should update reference model)
    learner._update(sample_experience)
    
    # Get updated reference model weights
    updated_ref_weights = learner.reference_model.state_dict()['actor_head.weight']
    
    # Reference model should match current model after update
    current_weights = learner.model.state_dict()['actor_head.weight']
    assert torch.allclose(updated_ref_weights, current_weights)


def test_learner_statistics(state_dim, action_dim, ppo_config):
    """Test that learner tracks statistics correctly."""
    weights_queue = mp.Queue()
    experience_queue = mp.Queue()
    
    learner = PPOLearner(state_dim, action_dim, weights_queue, experience_queue, ppo_config)
    
    assert learner.total_steps == 0
    assert learner.total_episodes == 0
    assert learner.update_count == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
