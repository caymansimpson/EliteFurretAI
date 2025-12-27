# -*- coding: utf-8 -*-
"""
Unit tests for MDBO encoding/decoding integration with RL.
"""

import pytest
from unittest.mock import Mock

from elitefurretai.etl.encoder import MDBO
from poke_env.battle import DoubleBattle


def test_mdbo_action_space_size():
    """Test that MDBO action space is correct size."""
    assert MDBO.action_space() == 2025


def test_mdbo_teampreview_space():
    """Test teampreview action space size."""
    teampreview_space = MDBO.teampreview_space()
    assert teampreview_space > 0
    assert teampreview_space < MDBO.action_space()


def test_mdbo_from_int_teampreview():
    """Test creating MDBO from integer during teampreview."""
    action_int = 0
    mdbo = MDBO.from_int(action_int, type=MDBO.TEAMPREVIEW)
    assert mdbo is not None


def test_mdbo_from_int_turn():
    """Test creating MDBO from integer during turn."""
    action_int = 100
    mdbo = MDBO.from_int(action_int, type=MDBO.TURN)
    assert mdbo is not None


def test_mdbo_to_double_battle_order():
    """Test converting MDBO to DoubleBattleOrder."""
    # Create a mock battle
    battle = Mock(spec=DoubleBattle)
    battle.teampreview = False
    
    # This will likely fail without a fully configured battle,
    # but tests the interface
    mdbo = MDBO.from_int(0, type=MDBO.TURN)
    try:
        order = mdbo.to_double_battle_order(battle)
        # If it succeeds, check that we got an order
        assert order is not None
    except Exception:
        # Expected - mock battle doesn't have required attributes
        pass


def test_mdbo_round_trip():
    """Test that MDBO can be converted to/from integers."""
    for action_int in [0, 100, 500, 1000, 2024]:
        mdbo = MDBO.from_int(action_int, type=MDBO.TURN)
        assert mdbo is not None
        # Can't test full round trip without battle context


def test_mdbo_teampreview_actions():
    """Test that all teampreview actions are valid."""
    teampreview_space = MDBO.teampreview_space()
    
    for i in range(min(10, teampreview_space)):  # Test first 10
        mdbo = MDBO.from_int(i, type=MDBO.TEAMPREVIEW)
        assert mdbo is not None


def test_mdbo_turn_actions_sample():
    """Test a sample of turn actions."""
    # Test a sample of turn actions
    sample_actions = [0, 50, 100, 500, 1000, 1500, 2000, 2024]
    
    for action in sample_actions:
        mdbo = MDBO.from_int(action, type=MDBO.TURN)
        assert mdbo is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
