# -*- coding: utf-8 -*-
"""
Unit tests for RL configuration.

These tests verify:
1. Config loading and saving
2. Default values
3. YAML serialization
4. Curriculum key consistency with opponent_pool
5. Parameter validation
"""

import os
import tempfile

import pytest
import yaml

from elitefurretai.rl.config import HardwareConfig, RNaDConfig, get_default_config

# =============================================================================
# DEFAULT CONFIG TESTS
# =============================================================================


def test_get_default_config():
    """
    Test that get_default_config returns a valid RNaDConfig.

    The default config should be usable without modification for testing.

    Expected: Returns RNaDConfig instance with sensible defaults.
    """
    config = get_default_config()

    assert isinstance(config, RNaDConfig)
    assert config.curriculum.battle_format == "gen9vgc2023regc"
    assert config.hardware.battle_backend == "showdown_websocket"
    assert config.hardware.device in ["cuda", "cpu"]


def test_invalid_battle_backend_raises_value_error():
    with pytest.raises(ValueError, match="battle_backend"):
        HardwareConfig(battle_backend="not_a_backend")


def test_default_curriculum_sums_to_one():
    """
    Test that default curriculum weights sum to 1.0.

    Curriculum is a probability distribution over opponent types.

    Expected: sum(curriculum_weights.values()) == 1.0
    """
    config = get_default_config()

    total = sum(config.curriculum.curriculum_weights.values())
    assert abs(total - 1.0) < 1e-6, f"Curriculum sums to {total}, expected 1.0"


def test_default_curriculum_uses_correct_keys():
    """
    Test that default curriculum uses keys matching opponent_pool.py.

    This is CRITICAL: The curriculum keys must match what OpponentPool expects:
    - 'self_play': Play against current model
    - 'bc_player': Play against behavioral cloning baseline
    - 'exploiters': Play against trained exploiter agents
    - 'ghosts': Play against past checkpoint snapshots

    BUG PREVENTION: Previously there was a mismatch where config used
    'past_versions' but opponent_pool expected 'ghosts'. This test
    ensures they stay aligned.

    Expected: Keys are exactly {'self_play', 'bc_player', 'exploiters', 'ghosts'}
    """
    config = get_default_config()

    expected_keys = {"self_play", "bc_player", "exploiters", "ghosts"}
    actual_keys = set(config.curriculum.curriculum_weights.keys())

    assert actual_keys == expected_keys, (
        f"Curriculum keys mismatch!\n"
        f"  Expected: {expected_keys}\n"
        f"  Got: {actual_keys}\n"
        f"  Missing: {expected_keys - actual_keys}\n"
        f"  Extra: {actual_keys - expected_keys}"
    )


# =============================================================================
# SAVE AND LOAD TESTS
# =============================================================================


def test_config_save_and_load():
    """
    Test that config can be saved to YAML and loaded back.

    All config values should survive the round-trip.

    Expected: Loaded config matches saved config.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.yaml")

        # Create config with custom values
        config = get_default_config()
        config.curriculum.battle_format = "gen9vgc2024regg"
        config.optimizer.lr = 0.0005
        config.hardware.num_players = 4
        config.training.train_batch_size = 64

        # Save
        config.save(config_path)

        # Verify file exists
        assert os.path.exists(config_path)

        # Load
        loaded = RNaDConfig.load(config_path)

        # Verify values match
        assert loaded.curriculum.battle_format == "gen9vgc2024regg"
        assert loaded.optimizer.lr == 0.0005
        assert loaded.hardware.num_players == 4
        assert loaded.training.train_batch_size == 64


def test_config_save_creates_directory():
    """
    Test that config.save creates parent directories if needed.

    Expected: Directories are created, no error raised.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Path with non-existent subdirectories
        config_path = os.path.join(tmpdir, "subdir1", "subdir2", "config.yaml")

        config = get_default_config()
        config.save(config_path)

        assert os.path.exists(config_path)


def test_config_yaml_format():
    """
    Test that saved config is valid YAML with nested structure.

    Expected: File can be parsed as YAML with nested sub-config keys.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "test_config.yaml")

        config = get_default_config()
        config.save(config_path)

        # Parse with yaml directly
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        # New nested structure
        assert "curriculum" in data
        assert "hardware" in data
        assert "algorithm" in data
        assert isinstance(data["curriculum"], dict)
        assert "battle_format" in data["curriculum"]
        assert "curriculum_weights" in data["curriculum"]


# =============================================================================
# TO_DICT TESTS
# =============================================================================


def test_config_to_dict():
    """
    Test that config.to_dict returns all configuration values in nested form.

    This is used for wandb logging and checkpointing.

    Expected: Dictionary contains nested sub-config keys.
    """
    config = get_default_config()
    d = config.to_dict()

    assert isinstance(d, dict)

    # Check nested sub-config keys are present
    assert "algorithm" in d
    assert "hardware" in d
    assert "curriculum" in d
    assert "training" in d
    assert "optimizer" in d

    # Check nested access
    assert "battle_format" in d["curriculum"]
    assert "lr" in d["optimizer"]
    assert "num_players" in d["hardware"]
    assert "use_wandb" in d["training"]


def test_config_str():
    """
    Test that config.__str__ returns readable output.

    This is used for logging configuration at training start.

    Expected: String representation includes key values.
    """
    config = get_default_config()
    s = str(config)

    assert isinstance(s, str)
    assert "RNaD Training Configuration" in s


# =============================================================================
# HYPERPARAMETER VALIDATION TESTS
# =============================================================================


def test_default_learning_rate():
    """
    Test default learning rate is reasonable.

    Expected: lr should be in typical range [1e-5, 1e-3].
    """
    config = get_default_config()

    assert 1e-5 <= config.optimizer.lr <= 1e-3, (
        f"Learning rate {config.optimizer.lr} seems unusual"
    )


def test_default_clip_range():
    """
    Test default PPO clip range is reasonable.

    Expected: clip_range should be in [0.1, 0.3] typically.
    """
    config = get_default_config()

    assert 0.1 <= config.algorithm.clip_range <= 0.5, (
        f"Clip range {config.algorithm.clip_range} seems unusual"
    )


def test_default_gamma():
    """
    Test default discount factor is reasonable.

    Expected: gamma should be close to 1.0 for episodic tasks.
    """
    config = get_default_config()

    assert 0.9 <= config.algorithm.gamma <= 1.0, (
        f"Gamma {config.algorithm.gamma} seems unusual"
    )


def test_default_gae_lambda():
    """
    Test default GAE lambda is reasonable.

    Expected: gae_lambda should be in [0.9, 1.0].
    """
    config = get_default_config()

    assert 0.9 <= config.algorithm.gae_lambda <= 1.0, (
        f"GAE lambda {config.algorithm.gae_lambda} seems unusual"
    )


# =============================================================================
# PATH CONFIGURATION TESTS
# =============================================================================


def test_default_paths_are_strings():
    """
    Test that all path configurations are strings.

    Expected: All *_path and *_dir fields are strings.
    """
    config = get_default_config()

    assert isinstance(config.curriculum.base_team_path, str)
    assert isinstance(config.training.save_dir, str)


def test_team_pool_path_can_be_none():
    """
    Test that team_pool_path can be None.

    When None, all teams in the format directory are sampled.
    When set, only teams in that subdirectory are used.

    Expected: Default is None, can be set to string.
    """
    config = get_default_config()

    # Default should be None or a string
    assert config.curriculum.team_pool_path is None or isinstance(
        config.curriculum.team_pool_path, str
    )


# =============================================================================
# BOOLEAN FLAG TESTS
# =============================================================================


def test_use_wandb_is_boolean():
    """
    Test that use_wandb is a boolean.

    Expected: use_wandb is True or False.
    """
    config = get_default_config()

    assert isinstance(config.training.use_wandb, bool)


def test_use_mixed_precision_is_boolean():
    """
    Test that use_mixed_precision is a boolean.

    Expected: use_mixed_precision is True or False.
    """
    config = get_default_config()

    assert isinstance(config.hardware.use_mixed_precision, bool)


def test_train_exploiters_is_boolean():
    """
    Test that train_exploiters is a boolean.

    Expected: train_exploiters is True or False.
    """
    config = get_default_config()

    assert isinstance(config.training.train_exploiters, bool)


# =============================================================================
# PORTFOLIO CONFIG TESTS
# =============================================================================


def test_portfolio_config_options():
    """
    Test portfolio regularization configuration options.

    Portfolio regularization maintains multiple reference models
    instead of just one for RNaD KL penalty.

    Expected: Portfolio config fields have valid defaults.
    """
    config = get_default_config()

    assert isinstance(config.portfolio.max_portfolio_size, int)
    assert config.portfolio.max_portfolio_size > 0
    assert config.portfolio.portfolio_update_strategy in ["diverse", "best", "recent"]


# =============================================================================
# WORKER CONFIGURATION TESTS
# =============================================================================


def test_worker_config():
    """
    Test worker-related configuration.

    Workers generate battle data by playing games.

    Expected: Worker config fields have valid defaults.
    """
    config = get_default_config()

    assert config.hardware.num_workers >= 1
    assert config.hardware.players_per_worker >= 1
    assert config.hardware.batch_size >= 1
    assert config.training.train_batch_size >= 1


def test_server_config():
    """
    Test showdown server configuration.

    Expected: Server config fields have valid defaults.
    """
    config = get_default_config()

    assert config.hardware.num_servers >= 1
    assert config.hardware.showdown_start_port > 0


# =============================================================================
# CONFIG LOADING FROM YAML EDGE CASES
# =============================================================================


def test_load_partial_yaml():
    """
    Test that loading a partial nested YAML uses defaults for missing fields.

    This allows config files to only specify changed values.

    Expected: Missing fields get default values.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "partial_config.yaml")

        # Write partial nested config
        partial = {
            "optimizer": {"lr": 0.001},
            "hardware": {"num_players": 8},
        }
        with open(config_path, "w") as f:
            yaml.dump(partial, f)

        # Load - should use defaults for missing fields
        loaded = RNaDConfig.load(config_path)

        # Specified values
        assert loaded.optimizer.lr == 0.001
        assert loaded.hardware.num_players == 8

        # Default values for unspecified fields
        assert loaded.curriculum.battle_format == "gen9vgc2023regc"
        assert loaded.algorithm.gamma == 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
