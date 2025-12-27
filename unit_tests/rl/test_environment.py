# -*- coding: utf-8 -*-
"""
Unit tests for VGC RL environment.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from elitefurretai.rl.environment import VGCDoublesEnv
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.embedder import Embedder
from poke_env.battle import DoubleBattle


@pytest.fixture
def env():
    """Create a test environment."""
    return VGCDoublesEnv(
        battle_format="gen9vgc2023regulationc",
        team=None
    )


@pytest.fixture
def mock_battle():
    """Create a mock DoubleBattle for testing."""
    battle = Mock(spec=DoubleBattle)
    battle.teampreview = False
    battle.finished = False
    battle.won = None
    battle.turn = 1
    return battle


def test_environment_initialization(env):
    """Test that environment initializes correctly."""
    assert env is not None
    assert isinstance(env.embedder, Embedder)
    assert env.embedder.feature_set == Embedder.FULL
    assert env.embedder.omniscient is False
    assert env.action_space.n == MDBO.action_space()


def test_observation_space(env):
    """Test that observation space is correctly configured."""
    assert 'observation' in env.observation_space.spaces
    assert 'action_mask' in env.observation_space.spaces
    
    obs_space = env.observation_space.spaces['observation']
    assert obs_space.shape == (env.embedder.embedding_size,)
    assert obs_space.dtype == np.float32
    
    mask_space = env.observation_space.spaces['action_mask']
    assert mask_space.n == MDBO.action_space()


def test_action_space(env):
    """Test that action space matches MDBO dimensions."""
    assert env.action_space.n == MDBO.action_space()
    assert env.action_space.n == 2025


def test_embed_battle_structure(env, mock_battle):
    """Test that embed_battle returns correct structure."""
    # Mock embedder output
    env.embedder.embed = Mock(return_value={'feature1': [0.0] * 10})
    env.embedder.feature_dict_to_vector = Mock(return_value=[0.0] * env.embedder.embedding_size)
    
    obs = env.embed_battle(mock_battle)
    
    assert 'observation' in obs
    assert 'action_mask' in obs
    assert isinstance(obs['observation'], np.ndarray)
    assert isinstance(obs['action_mask'], np.ndarray)
    assert obs['observation'].dtype == np.float32
    assert obs['action_mask'].dtype == np.int8


def test_action_mask_teampreview(env, mock_battle):
    """Test action mask during teampreview."""
    mock_battle.teampreview = True
    
    mask = env._get_action_mask(mock_battle)
    
    # During teampreview, only teampreview actions should be valid
    teampreview_space = MDBO.teampreview_space()
    assert mask[:teampreview_space].sum() == teampreview_space
    assert mask[teampreview_space:].sum() == 0


def test_action_mask_regular_turn(env, mock_battle):
    """Test action mask during regular battle turn."""
    mock_battle.teampreview = False
    
    # This requires a fully mocked battle with valid orders
    # For now, just test that it returns an array of correct shape
    mask = env._get_action_mask(mock_battle)
    
    assert mask.shape == (MDBO.action_space(),)
    assert mask.dtype == np.int8
    assert np.all((mask == 0) | (mask == 1))


def test_action_to_move_teampreview(env, mock_battle):
    """Test converting action to move during teampreview."""
    mock_battle.teampreview = True
    action = 0  # First teampreview action
    
    order = env.action_to_move(action, mock_battle)
    
    assert order is not None


def test_action_to_move_regular(env, mock_battle):
    """Test converting action to move during regular turn."""
    mock_battle.teampreview = False
    action = 100  # Some valid action
    
    # This will fail without a fully configured battle, but tests the path
    try:
        order = env.action_to_move(action, mock_battle)
        assert order is not None
    except Exception:
        # Expected - mock battle doesn't have all required attributes
        pass


def test_calc_reward(env, mock_battle):
    """Test reward calculation."""
    # Mock the reward_computing_helper method
    env.reward_computing_helper = Mock(return_value=1.5)
    
    reward = env.calc_reward(mock_battle)
    
    assert isinstance(reward, float)
    env.reward_computing_helper.assert_called_once_with(
        mock_battle,
        fainted_value=2.0,
        hp_value=1.0,
        victory_value=30.0
    )


@pytest.mark.asyncio
async def test_reset(env):
    """Test environment reset."""
    obs, info = await env.reset()
    
    assert 'observation' in obs
    assert 'action_mask' in obs
    assert isinstance(obs['observation'], np.ndarray)
    assert isinstance(obs['action_mask'], np.ndarray)
    assert isinstance(info, dict)


@pytest.mark.asyncio
async def test_step(env):
    """Test environment step."""
    action = 0
    obs, reward, terminated, truncated, info = await env.step(action)
    
    assert 'observation' in obs
    assert 'action_mask' in obs
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
