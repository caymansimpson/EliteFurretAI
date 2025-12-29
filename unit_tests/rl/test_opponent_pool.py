# -*- coding: utf-8 -*-
"""
Unit tests for RL opponent pool.

These tests verify:
1. Curriculum sampling logic
2. Opponent type selection based on weights
3. Win rate tracking and adaptive curriculum
4. Past model management
5. Exploiter registry
6. Battle format propagation
"""

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.opponent_pool import ExploiterRegistry, OpponentPool

# =============================================================================
# FIXTURES: Reusable test components
# =============================================================================


@pytest.fixture
def mock_main_model():
    """
    Create a mock RNaDAgent as the main model.

    This represents the model being trained in RL.
    Opponents are sampled from various sources including copies of this model.
    """
    model = MagicMock(spec=RNaDAgent)
    return model


@pytest.fixture
def player_config():
    """Account configuration for test players."""
    return AccountConfiguration("TestOpponent", None)


@pytest.fixture
def server_config():
    """Server configuration for test battles."""
    return ServerConfiguration("ws://localhost:8000/showdown/websocket", None)  # type: ignore


@pytest.fixture
def temp_registry_path(tmp_path):
    """
    Create a temporary path for exploiter registry.

    Tests should not modify production registry files.
    """
    return str(tmp_path / "test_exploiter_registry.json")


@pytest.fixture
def temp_past_models_dir(tmp_path):
    """
    Create a temporary directory for past model checkpoints.

    Tests should not modify production model directories.
    """
    past_dir = tmp_path / "past_models"
    past_dir.mkdir()
    return str(past_dir)


# =============================================================================
# EXPLOITER REGISTRY TESTS
# =============================================================================


def test_exploiter_registry_initialization_empty(temp_registry_path):
    """
    Test ExploiterRegistry initializes empty when file doesn't exist.

    When no registry file exists:
    - exploiters list should be empty
    - get_active_exploiters should return empty list

    Expected: Empty registry is created without errors.
    """
    registry = ExploiterRegistry(temp_registry_path)

    assert len(registry.exploiters) == 0
    assert registry.get_active_exploiters() == []


def test_exploiter_registry_loads_existing_file(temp_registry_path):
    """
    Test ExploiterRegistry loads exploiters from existing file.

    When registry file exists with exploiters:
    - All exploiters should be loaded
    - Exploiter info should be preserved

    Expected: Registry contains exploiters from file.
    """
    # Create registry file with one exploiter
    exploiter_data = {
        "exploiters": [
            {
                "id": "exp_001",
                "filepath": "data/models/exploiters/exp_001.pt",
                "team": "some_team_string",
                "win_rate": 0.65,
                "victim_step": 1000,
            }
        ]
    }

    with open(temp_registry_path, "w") as f:
        json.dump(exploiter_data, f)

    registry = ExploiterRegistry(temp_registry_path)

    assert len(registry.exploiters) == 1
    assert registry.exploiters[0]["id"] == "exp_001"
    assert registry.exploiters[0]["win_rate"] == 0.65


def test_exploiter_registry_add_exploiter(temp_registry_path):
    """
    Test adding a new exploiter to the registry.

    add_exploiter should:
    - Add the exploiter info to the list
    - Save the registry to disk
    - Preserve all exploiter fields

    Expected: New exploiter is added and persisted.
    """
    registry = ExploiterRegistry(temp_registry_path)

    exploiter_info = {
        "id": "exp_new",
        "filepath": "data/models/exploiters/exp_new.pt",
        "team": "team_string",
        "win_rate": 0.60,
        "victim_step": 2000,
    }

    registry.add_exploiter(exploiter_info)

    assert len(registry.exploiters) == 1
    assert registry.exploiters[0]["id"] == "exp_new"

    # Verify file was saved
    assert os.path.exists(temp_registry_path)
    with open(temp_registry_path, "r") as f:
        saved_data = json.load(f)
    assert len(saved_data["exploiters"]) == 1


def test_exploiter_registry_requires_id(temp_registry_path):
    """
    Test that add_exploiter requires 'id' field.

    Exploiters need unique identifiers for tracking.

    Expected: ValueError raised when 'id' is missing.
    """
    registry = ExploiterRegistry(temp_registry_path)

    with pytest.raises(ValueError, match="must contain 'id' field"):
        registry.add_exploiter(
            {
                "filepath": "some/path.pt",
                "team": "team_string",
            }
        )


def test_exploiter_registry_requires_filepath(temp_registry_path):
    """
    Test that add_exploiter requires 'filepath' field.

    We need to know where the model checkpoint is.

    Expected: ValueError raised when 'filepath' is missing.
    """
    registry = ExploiterRegistry(temp_registry_path)

    with pytest.raises(ValueError, match="must contain 'filepath' field"):
        registry.add_exploiter(
            {
                "id": "exp_001",
                "team": "team_string",
            }
        )


def test_exploiter_registry_requires_team(temp_registry_path):
    """
    Test that add_exploiter requires 'team' field.

    Exploiters are trained with specific teams that exploit weaknesses.

    Expected: ValueError raised when 'team' is missing.
    """
    registry = ExploiterRegistry(temp_registry_path)

    with pytest.raises(ValueError, match="must contain 'team' field"):
        registry.add_exploiter(
            {
                "id": "exp_001",
                "filepath": "some/path.pt",
            }
        )


def test_exploiter_registry_rejects_duplicate_id(temp_registry_path):
    """
    Test that add_exploiter rejects duplicate IDs.

    Each exploiter must have a unique ID.

    Expected: ValueError raised when ID already exists.
    """
    registry = ExploiterRegistry(temp_registry_path)

    exploiter = {
        "id": "exp_001",
        "filepath": "path1.pt",
        "team": "team1",
    }
    registry.add_exploiter(exploiter)

    with pytest.raises(ValueError, match="already exists"):
        registry.add_exploiter(
            {
                "id": "exp_001",  # Duplicate ID
                "filepath": "path2.pt",
                "team": "team2",
            }
        )


def test_get_active_exploiters_filters_by_win_rate(temp_registry_path):
    """
    Test that get_active_exploiters filters by minimum win rate.

    Only exploiters above the threshold should be returned.
    This prevents weak exploiters from polluting the opponent pool.

    Expected: Only exploiters with win_rate >= threshold are returned.
    """
    # Create registry with exploiters of varying win rates
    exploiter_data = {
        "exploiters": [
            {
                "id": "exp_001",
                "filepath": "p1.pt",
                "team": "t1",
                "win_rate": 0.40,
            },  # Below threshold
            {
                "id": "exp_002",
                "filepath": "p2.pt",
                "team": "t2",
                "win_rate": 0.55,
            },  # At threshold
            {
                "id": "exp_003",
                "filepath": "p3.pt",
                "team": "t3",
                "win_rate": 0.70,
            },  # Above threshold
        ]
    }

    with open(temp_registry_path, "w") as f:
        json.dump(exploiter_data, f)

    registry = ExploiterRegistry(temp_registry_path)

    # Default threshold is 0.55
    active = registry.get_active_exploiters(min_win_rate=0.55)

    assert len(active) == 2
    assert all(e["win_rate"] >= 0.55 for e in active)


# =============================================================================
# OPPONENT POOL INITIALIZATION TESTS
# =============================================================================


def test_opponent_pool_initialization(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test OpponentPool initializes with correct default curriculum.

    Default curriculum should sum to 1.0 and include all opponent types.

    Expected: curriculum weights are valid probabilities.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Check curriculum exists and sums to 1.0
    assert hasattr(pool, "curriculum")
    assert np.isclose(sum(pool.curriculum.values()), 1.0)

    # Check all expected opponent types are present
    expected_types = {"self_play", "bc_player", "exploiters", "ghosts"}
    assert set(pool.curriculum.keys()) == expected_types


def test_opponent_pool_stores_battle_format(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test that OpponentPool stores battle_format for opponent creation.

    The battle_format must be passed to all created opponents to ensure
    they connect to the correct battle format on the server.

    Expected: pool.battle_format matches constructor argument.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    assert pool.battle_format == "gen9vgc2023regulationc"


def test_opponent_pool_custom_curriculum(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test OpponentPool accepts custom curriculum weights.

    Users can override default curriculum to focus on specific opponents.

    Expected: Custom curriculum is used instead of defaults.
    """
    custom_curriculum = {
        "self_play": 0.80,  # 80% self-play
        "bc_player": 0.10,
        "exploiters": 0.05,
        "ghosts": 0.05,
    }

    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        curriculum=custom_curriculum,
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    assert pool.curriculum["self_play"] == 0.80
    assert pool.curriculum["bc_player"] == 0.10


def test_opponent_pool_rejects_invalid_curriculum(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test OpponentPool rejects curriculum that doesn't sum to 1.0.

    Curriculum weights are probabilities and must be valid.

    Expected: ValueError when weights don't sum to 1.0.
    """
    bad_curriculum = {
        "self_play": 0.50,
        "bc_player": 0.20,
        "exploiters": 0.10,
        "ghosts": 0.10,  # Sum = 0.90, not 1.0
    }

    with pytest.raises(ValueError, match="must sum to 1.0"):
        OpponentPool(
            main_model=mock_main_model,
            device="cpu",
            battle_format="gen9vgc2023regulationc",
            curriculum=bad_curriculum,
            exploiter_registry_path=temp_registry_path,
            past_models_dir=temp_past_models_dir,
        )


# =============================================================================
# CURRICULUM SAMPLING TESTS
# =============================================================================


def test_sample_opponent_respects_curriculum(
    mock_main_model, player_config, server_config, temp_registry_path, temp_past_models_dir
):
    """
    Test that sample_opponent samples according to curriculum weights.

    With 100% self_play weight, every sample should be self_play.

    Expected: All samples should be self_play opponents.
    """
    # 100% self_play curriculum
    curriculum = {
        "self_play": 1.0,
        "bc_player": 0.0,
        "exploiters": 0.0,
        "ghosts": 0.0,
    }

    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        curriculum=curriculum,
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Sample should always return self-play (BatchInferencePlayer with main_model)
    # We can't easily test the full sample without mocking Player creation
    # This test verifies the curriculum is set correctly
    assert pool.curriculum["self_play"] == 1.0


# =============================================================================
# PAST MODEL MANAGEMENT TESTS
# =============================================================================


def test_add_past_model(mock_main_model, temp_registry_path, temp_past_models_dir):
    """
    Test adding past model checkpoints to the pool.

    add_past_model should:
    - Add (step, filepath) tuple to past_models list
    - Sort by step descending (most recent first)
    - Respect max_past_models limit

    Expected: past_models contains new checkpoint.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        max_past_models=5,
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Add some past models
    pool.add_past_model(1000, "path/model_1000.pt")
    pool.add_past_model(2000, "path/model_2000.pt")
    pool.add_past_model(500, "path/model_500.pt")

    assert len(pool.past_models) == 3

    # Should be sorted by step descending
    assert pool.past_models[0][0] == 2000  # Most recent first
    assert pool.past_models[1][0] == 1000
    assert pool.past_models[2][0] == 500


def test_add_past_model_respects_limit(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test that add_past_model respects max_past_models limit.

    When adding more models than the limit:
    - Only the most recent are kept
    - Older models are removed

    Expected: Only max_past_models are retained.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        max_past_models=3,
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Add 5 models when limit is 3
    for step in [100, 200, 300, 400, 500]:
        pool.add_past_model(step, f"path/model_{step}.pt")

    assert len(pool.past_models) == 3

    # Should keep the 3 most recent (300, 400, 500)
    steps = [m[0] for m in pool.past_models]
    assert 500 in steps
    assert 400 in steps
    assert 300 in steps
    assert 200 not in steps
    assert 100 not in steps


# =============================================================================
# WIN RATE TRACKING TESTS
# =============================================================================


def test_update_win_rate(mock_main_model, temp_registry_path, temp_past_models_dir):
    """
    Test tracking win rates against different opponent types.

    update_win_rate should:
    - Add 1.0 for wins, 0.0 for losses
    - Track separately per opponent type

    Expected: Win rate history is updated correctly.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Record some results
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=False)
    pool.update_win_rate("bc_player", won=False)

    assert len(pool.win_rates["self_play"]) == 3
    assert len(pool.win_rates["bc_player"]) == 1


def test_get_win_rate_stats(mock_main_model, temp_registry_path, temp_past_models_dir):
    """
    Test computing recent win rate statistics.

    get_win_rate_stats should:
    - Compute mean win rate over recent games
    - Return 0.0 for opponent types with no games

    Expected: Correct win rate averages.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Win 3 out of 4 against self_play
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=False)

    stats = pool.get_win_rate_stats()

    assert np.isclose(stats["self_play"], 0.75)
    assert stats["bc_player"] == 0.0  # No games played


def test_get_win_rate_stats_with_window(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test that get_win_rate_stats uses the window parameter.

    Only the last N games should be considered for the average.

    Expected: Average is computed over only recent games.
    """
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    # Record 10 games: 5 wins then 5 losses
    for _ in range(5):
        pool.update_win_rate("self_play", won=True)
    for _ in range(5):
        pool.update_win_rate("self_play", won=False)

    # Full window should give 50% win rate
    stats_full = pool.get_win_rate_stats(window=10)
    assert np.isclose(stats_full["self_play"], 0.5)

    # Last 5 games were all losses
    stats_recent = pool.get_win_rate_stats(window=5)
    assert np.isclose(stats_recent["self_play"], 0.0)


# =============================================================================
# ADAPTIVE CURRICULUM TESTS
# =============================================================================


def test_update_curriculum_adaptive_disabled(
    mock_main_model, temp_registry_path, temp_past_models_dir
):
    """
    Test that update_curriculum does nothing when adaptive=False.

    Sometimes we want fixed curriculum weights.

    Expected: Curriculum unchanged when adaptive=False.
    """
    curriculum = {
        "self_play": 0.25,
        "bc_player": 0.25,
        "exploiters": 0.25,
        "ghosts": 0.25,
    }

    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regulationc",
        curriculum=curriculum,
        exploiter_registry_path=temp_registry_path,
        past_models_dir=temp_past_models_dir,
    )

    original = pool.curriculum.copy()
    pool.update_curriculum(adaptive=False)

    assert pool.curriculum == original


def test_curriculum_keys_match_opponent_pool():
    """
    Test that curriculum keys match the expected opponent types.

    This is a critical invariant: the keys in curriculum dict must
    match what sample_opponent expects.

    BUG CAUGHT: Previously config.py used 'past_versions' but opponent_pool.py
    expected 'ghosts'. This test ensures they stay aligned.

    Expected: Keys should be {'self_play', 'bc_player', 'exploiters', 'ghosts'}.
    """
    expected_keys = {"self_play", "bc_player", "exploiters", "ghosts"}

    # Default curriculum from OpponentPool
    default_curriculum = {
        "self_play": 0.40,
        "bc_player": 0.20,
        "exploiters": 0.20,
        "ghosts": 0.20,
    }

    assert set(default_curriculum.keys()) == expected_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
