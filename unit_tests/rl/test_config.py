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

from elitefurretai.rl.config import CurriculumConfig, RNaDConfig, get_default_config
from elitefurretai.rl.opponents import OpponentPool

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
    assert config.curriculum.primary_format == "gen9vgc2023regc"
    assert config.curriculum.battle_formats == {"gen9vgc2023regc": 1.0}
    assert config.hardware.device in ["cuda", "cpu"]


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
    Test that default curriculum keys are aligned with OpponentPool slots.

    Two checks:
      1. Every key in `curriculum_weights` must correspond to a real
         OpponentPool slot — extra/typo keys would silently never sample.
      2. The default config exposes the *full* set of valid slots so
         operators can dial any of them up via YAML without first having
         to add the key. Today this is a perfect match; if a new slot is
         added to OpponentPool, the default config must register it.

    BUG PREVENTION: Previously there was a mismatch where config used
    'past_versions' but opponent_pool expected 'ghosts'. This test
    ensures they stay aligned.
    """
    config = get_default_config()
    actual_keys = set(config.curriculum.curriculum_weights.keys())
    valid_keys = {
        OpponentPool.SELF_PLAY,
        OpponentPool.BC_PLAYER,
        OpponentPool.EXPLOITERS,
        OpponentPool.GHOSTS,
        OpponentPool.TRAIN_EXPLOITER,
        OpponentPool.MAX_DAMAGE,
        OpponentPool.RANDOM,
        OpponentPool.MAX_BASE_POWER,
        OpponentPool.SIMPLE_HEURISTIC,
        OpponentPool.VGC_BENCH,
    }

    extra = actual_keys - valid_keys
    assert not extra, (
        f"curriculum_weights contains keys that don't match any OpponentPool "
        f"slot (will silently never sample): {extra}"
    )

    missing = valid_keys - actual_keys
    assert not missing, (
        f"Default curriculum should register every OpponentPool slot so "
        f"operators can tune it via YAML without adding keys. Missing: "
        f"{missing}"
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
        config.curriculum.battle_formats = {"gen9vgc2024regg": 1.0}
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
        assert loaded.curriculum.primary_format == "gen9vgc2024regg"
        assert loaded.curriculum.battle_formats == {"gen9vgc2024regg": 1.0}
        assert loaded.optimizer.lr == 0.0005
        assert loaded.hardware.num_players == 4
        assert loaded.training.train_batch_size == 64


def test_max_concurrent_battles_per_player_default_and_roundtrip():
    """The hardware knob defaults to 20 and survives a YAML round-trip
    both as an int and as None."""
    config = get_default_config()
    assert config.hardware.max_concurrent_battles_per_player == 20

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "concurrency.yaml")
        config.hardware.max_concurrent_battles_per_player = 16
        config.save(config_path)
        assert (
            RNaDConfig.load(config_path).hardware.max_concurrent_battles_per_player == 16
        )

        config.hardware.max_concurrent_battles_per_player = None
        config.save(config_path)
        assert (
            RNaDConfig.load(config_path).hardware.max_concurrent_battles_per_player is None
        )


def test_compile_inference_model_default_and_roundtrip():
    """The torch.compile knob defaults to None (eager) and round-trips
    through YAML both as a mode string and as None.
    """
    config = get_default_config()
    assert config.hardware.compile_inference_model is None

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "compile.yaml")
        config.hardware.compile_inference_model = "default"
        config.save(config_path)
        assert RNaDConfig.load(config_path).hardware.compile_inference_model == "default"

        config.hardware.compile_inference_model = "reduce-overhead"
        config.save(config_path)
        assert (
            RNaDConfig.load(config_path).hardware.compile_inference_model
            == "reduce-overhead"
        )

        config.hardware.compile_inference_model = None
        config.save(config_path)
        assert RNaDConfig.load(config_path).hardware.compile_inference_model is None


def test_memory_watchdog_threshold_default_and_roundtrip():
    """Memory watchdog threshold defaults to 20.0 GB and round-trips through YAML.

    Disabling (None) must also survive the round-trip — the trainer treats
    None or 0 as "disabled" and that pathway has to be configurable.
    """
    config = get_default_config()
    assert config.training.memory_watchdog_threshold_gb == 20.0

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "watchdog.yaml")
        config.training.memory_watchdog_threshold_gb = 14.5
        config.save(config_path)
        assert RNaDConfig.load(config_path).training.memory_watchdog_threshold_gb == 14.5

        config.training.memory_watchdog_threshold_gb = None
        config.save(config_path)
        assert RNaDConfig.load(config_path).training.memory_watchdog_threshold_gb is None


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
        assert "battle_formats" in data["curriculum"]
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
    assert "battle_formats" in d["curriculum"]
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


def test_opponent_team_pool_path_can_be_none():
    """
    Test that opponent_team_pool_path can be None.

    When None, all teams in the format directory are sampled.
    When set, only teams in that subdirectory are used.

    Expected: Default is None, can be set to string.
    """
    config = get_default_config()

    # Default should be None or a string
    assert config.curriculum.opponent_team_pool_path is None or isinstance(
        config.curriculum.opponent_team_pool_path, str
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
    assert config.portfolio.portfolio_update_strategy in [
        "diverse",
        "best",
        "recent",
        "random",
    ]


# =============================================================================
# EXPLOITER CONFIG TESTS
# =============================================================================


def test_exploiter_config_defaults():
    """ExploiterConfig holds all in-process exploiter co-training knobs.

    Defaults must produce a sensible (if conservative) configuration
    even when train_exploiter == 0 — the dataclass is always constructed
    regardless of whether the pipeline is active.
    """
    config = get_default_config()

    # Schedule knobs — must be positive ints with sane ranges
    assert isinstance(config.exploiter.batch_size, int)
    assert config.exploiter.batch_size > 0
    assert isinstance(config.exploiter.graduation_window, int)
    assert config.exploiter.graduation_window > 0
    assert isinstance(config.exploiter.max_updates_per_generation, int)
    assert config.exploiter.max_updates_per_generation > 0
    assert isinstance(config.exploiter.victim_refresh_interval, int)
    assert config.exploiter.victim_refresh_interval > 0
    assert isinstance(config.exploiter.warmup_updates, int)
    assert config.exploiter.warmup_updates >= 0  # 0 disables warmup

    # Threshold is a probability
    assert 0.0 <= config.exploiter.graduation_threshold <= 1.0

    # Optimizer knobs
    assert config.exploiter.lr > 0
    assert config.exploiter.ent_coef >= 0


def test_exploiter_config_round_trip():
    """ExploiterConfig survives to_dict/from_dict round-trip."""
    config = get_default_config()
    config.exploiter.graduation_threshold = 0.70
    config.exploiter.batch_size = 128

    d = config.to_dict()
    restored = type(config).from_dict(d)

    assert restored.exploiter.graduation_threshold == 0.70
    assert restored.exploiter.batch_size == 128


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
        assert loaded.curriculum.primary_format == "gen9vgc2023regc"
        assert loaded.curriculum.battle_formats == {"gen9vgc2023regc": 1.0}
        assert loaded.algorithm.gamma == 0.99


# =============================================================================
# BATTLE_FORMATS DISTRIBUTION TESTS
# =============================================================================


def test_battle_formats_default_is_single_format_distribution():
    """Default CurriculumConfig has battle_formats == {default: 1.0}."""
    config = get_default_config()
    assert config.curriculum.battle_formats == {"gen9vgc2023regc": 1.0}
    assert config.curriculum.primary_format == "gen9vgc2023regc"


def test_battle_formats_validation_rejects_non_unit_sum():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        CurriculumConfig(battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.4})


def test_battle_formats_validation_rejects_negative_weight():
    with pytest.raises(ValueError, match="positive"):
        CurriculumConfig(battle_formats={"gen9vgc2024regg": 1.2, "gen9vgc2024regh": -0.2})


def test_primary_format_returns_highest_weight():
    cur = CurriculumConfig(battle_formats={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3})
    assert cur.primary_format == "gen9vgc2024regg"


def test_battle_formats_validation_rejects_mixed_gens():
    """battle_formats with entries from different gens (format[3] differs) is rejected."""
    with pytest.raises(ValueError, match="same gen"):
        CurriculumConfig(battle_formats={"gen8vgc2022": 0.5, "gen9vgc2024regg": 0.5})


def test_battle_formats_validation_rejects_short_format_string():
    """Format strings shorter than 4 chars cannot identify a gen."""
    with pytest.raises(ValueError, match="too short"):
        CurriculumConfig(battle_formats={"foo": 1.0})


# =============================================================================
# PER-FORMAT RESOLVED PATH TESTS (Task 2)
# =============================================================================


def test_resolved_agent_team_paths_string_form_broadcasts_to_all_formats():
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
        base_team_path="data/teams",
        agent_team_path="constrained",
    )
    paths = cur.resolved_agent_team_paths()
    assert paths == {
        "gen9vgc2024regg": "data/teams/gen9vgc2024regg/constrained",
        "gen9vgc2024regh": "data/teams/gen9vgc2024regh/constrained",
    }


def test_resolved_agent_team_paths_dict_form_used_verbatim_per_format():
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3},
        base_team_path="data/teams",
        agent_team_path={"gen9vgc2024regg": "constrained", "gen9vgc2024regh": "rentals"},
    )
    paths = cur.resolved_agent_team_paths()
    assert paths == {
        "gen9vgc2024regg": "data/teams/gen9vgc2024regg/constrained",
        "gen9vgc2024regh": "data/teams/gen9vgc2024regh/rentals",
    }


def test_resolved_agent_team_paths_dict_form_rejects_missing_format():
    with pytest.raises(ValueError, match="missing="):
        CurriculumConfig(
            battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
            agent_team_path={"gen9vgc2024regg": "constrained"},  # missing regh
        )


def test_resolved_agent_team_paths_dict_form_rejects_extra_format():
    """Dict-form path config with extra keys not in battle_formats is rejected."""
    with pytest.raises(ValueError, match="extra="):
        CurriculumConfig(
            battle_formats={"gen9vgc2024regg": 1.0},
            agent_team_path={
                "gen9vgc2024regg": "constrained",
                "gen9vgc2024regh": "rentals",
            },
        )


def test_resolved_agent_team_paths_none_returns_empty_dict():
    cur = CurriculumConfig(battle_formats={"gen9vgc2024regg": 1.0}, agent_team_path=None)
    assert cur.resolved_agent_team_paths() == {}


def test_resolved_opponent_team_pool_paths_matches_format_keys():
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.6, "gen9vgc2024regh": 0.4},
        opponent_team_pool_path={"gen9vgc2024regg": "ranked", "gen9vgc2024regh": None},
    )
    assert cur.resolved_opponent_team_pool_paths() == {
        "gen9vgc2024regg": "ranked",
        "gen9vgc2024regh": None,
    }


def test_resolved_opponent_team_pool_paths_string_form_broadcasts():
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
        opponent_team_pool_path="ranked",
    )
    assert cur.resolved_opponent_team_pool_paths() == {
        "gen9vgc2024regg": "ranked",
        "gen9vgc2024regh": "ranked",
    }


def test_resolved_opponent_team_pool_paths_none_maps_every_format_to_none():
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
        opponent_team_pool_path=None,
    )
    assert cur.resolved_opponent_team_pool_paths() == {
        "gen9vgc2024regg": None,
        "gen9vgc2024regh": None,
    }


# =============================================================================
# FOULPLAY EVAL CONFIG TESTS
# =============================================================================


def test_default_eval_config_disabled():
    """
    EvalConfig defaults to disabled.

    Inline eval is opt-in; default-on would either crash a fresh
    checkout (missing venvs / executables) or slow training unexpectedly.
    """
    config = get_default_config()
    assert config.eval.enabled is False
    assert config.eval.eval_every_n_updates == 500
    assert config.eval.pause_training is True
    assert config.eval.surplus_alpha == 1.0
    assert config.eval.opponents["foul_play"].weight == 0.0
    assert config.eval.foulplay_team_pool_paths is None
    assert config.eval.foulplay_model_probabilistic is False


def _eval_test_config(tmp_path) -> RNaDConfig:
    """Build a config with all prerequisites for verify() satisfied except
    eval-specific fields. Tests then mutate only the eval slice."""
    base_team_path = tmp_path / "teams"
    base_team_path.mkdir()
    fmt_dir = base_team_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    agent_team_dir = fmt_dir / "agent"
    agent_team_dir.mkdir()
    (agent_team_dir / "team.txt").write_text("placeholder")

    config = get_default_config()
    config.curriculum.battle_formats = {"gen9vgc2024regg": 1.0}
    config.curriculum.base_team_path = str(base_team_path)
    config.curriculum.agent_team_path = "agent"
    config.curriculum.opponent_team_pool_path = None
    config.curriculum.bc_model_path = None
    # Existing vgcbench validator fires when the baseline has positive weight.
    config.curriculum.curriculum_weights = {"vgc_bench": 0.0}
    # Zero out all opponent weights so only the one under test triggers.
    for spec in config.eval.opponents.values():
        spec.weight = 0.0
    return config


def test_foulplay_eval_verify_requires_existing_foulplay_executable(tmp_path):
    """Verify() rejects an enabled FoulPlay opponent with a non-existent executable."""
    config = _eval_test_config(tmp_path)
    config.eval.enabled = True
    config.eval.opponents["foul_play"].weight = 1.0
    config.eval.foulplay_python_executable = str(tmp_path / "does_not_exist")

    with pytest.raises(ValueError, match="foulplay_python_executable"):
        config.verify()


def test_foulplay_eval_verify_foulplay_disabled_when_weight_zero(tmp_path):
    """When foul_play weight is 0, foulplay_python_executable is not checked."""
    config = _eval_test_config(tmp_path)
    config.eval.enabled = True
    config.eval.opponents["foul_play"].weight = 0.0
    config.eval.foulplay_python_executable = str(tmp_path / "does_not_exist")

    # Should not raise — weight=0 means FoulPlay is inactive.
    config.verify()


def test_vgcbench_eval_verify_requires_existing_executable(tmp_path):
    """Verify() rejects an enabled vgc_bench opponent with a non-existent executable."""
    config = _eval_test_config(tmp_path)
    config.eval.enabled = True
    config.eval.opponents["vgc_bench"].weight = 1.0
    config.eval.vgcbench_python_executable = str(tmp_path / "does_not_exist")

    with pytest.raises(ValueError, match="vgcbench_python_executable"):
        config.verify()


def test_eval_yaml_round_trip(tmp_path):
    """
    EvalConfig survives a save/load cycle through YAML.

    Guards against from_dict silently dropping the eval sub-config
    when partial YAML only sets some keys.
    """
    config = get_default_config()
    config.eval.enabled = True
    config.eval.eval_every_n_updates = 250
    config.eval.foulplay_python_executable = "/tmp/python-foulplay"
    config.eval.foulplay_team_pool_paths = {"gen9vgc2023regc": "/tmp/teams"}

    path = tmp_path / "config.yaml"
    config.save(str(path))
    loaded = RNaDConfig.load(str(path))

    assert loaded.eval.enabled is True
    assert loaded.eval.eval_every_n_updates == 250
    assert loaded.eval.foulplay_python_executable == "/tmp/python-foulplay"
    assert loaded.eval.foulplay_team_pool_paths == {"gen9vgc2023regc": "/tmp/teams"}


# =============================================================================
# ADAPTIVE AXIS CONFIG TESTS (Phase 4)
# =============================================================================


def test_adaptive_axis_config_team_defaults():
    """Team-axis default values reproduce the Change 7 settings: EWMA
    half_life=50, pure asymmetric weakness with linear exponent."""
    from elitefurretai.rl.config import AdaptiveAxisConfig

    cfg = AdaptiveAxisConfig.team_axis_defaults()
    assert cfg.enabled is True
    assert cfg.half_life == 50.0
    assert cfg.pfsp_mix == 0.0
    assert cfg.weakness_mix == 1.0
    assert cfg.weakness_exponent == 1.0
    assert cfg.base_blend == 0.0
    assert cfg.per_key_floor == 0.005
    assert cfg.min_samples == 20  # team_warmup_threshold


def test_adaptive_axis_config_agent_defaults():
    """Agent-axis default values reproduce the existing update_curriculum
    behavior: PFSP-weighted with weakness side, base-curriculum blended."""
    from elitefurretai.rl.config import AdaptiveAxisConfig

    cfg = AdaptiveAxisConfig.agent_axis_defaults()
    assert cfg.enabled is True
    assert cfg.half_life == 100.0
    assert cfg.pfsp_mix == 0.70
    assert cfg.weakness_mix == 0.30
    assert cfg.weakness_exponent == 1.0
    assert cfg.target_win_rate == 0.55
    assert cfg.base_blend == 0.50
    assert cfg.min_samples == 40
    assert cfg.prior_alpha == 8.0
    assert cfg.prior_beta == 8.0


def test_curriculum_config_nests_both_axes():
    from elitefurretai.rl.config import AdaptiveAxisConfig, CurriculumConfig

    cfg = CurriculumConfig()
    assert isinstance(cfg.adaptive_team_axis, AdaptiveAxisConfig)
    assert isinstance(cfg.adaptive_agent_axis, AdaptiveAxisConfig)
    # Distinct defaults for each axis
    assert cfg.adaptive_team_axis.pfsp_mix == 0.0
    assert cfg.adaptive_agent_axis.pfsp_mix == 0.70


def test_curriculum_config_yaml_round_trip_with_nested_axes(tmp_path):
    """Loading a YAML with nested adaptive_*_axis blocks reconstructs
    the AdaptiveAxisConfig sub-dataclasses correctly."""
    payload = {
        "curriculum": {
            "battle_formats": {"gen9vgc2024regg": 1.0},
            "adaptive_team_axis": {
                "enabled": True,
                "half_life": 75.0,
                "weakness_exponent": 2.0,
                "per_key_floor": 0.01,
            },
            "adaptive_agent_axis": {
                "enabled": False,
                "min_samples": 60,
                "base_blend": 0.25,
            },
        },
    }
    path = tmp_path / "test_cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = RNaDConfig.load(str(path))
    assert cfg.curriculum.adaptive_team_axis.half_life == 75.0
    assert cfg.curriculum.adaptive_team_axis.weakness_exponent == 2.0
    assert cfg.curriculum.adaptive_team_axis.per_key_floor == 0.01
    assert cfg.curriculum.adaptive_agent_axis.enabled is False
    assert cfg.curriculum.adaptive_agent_axis.min_samples == 60
    assert cfg.curriculum.adaptive_agent_axis.base_blend == 0.25


def test_curriculum_open_team_sheets_mode_default_and_loads():
    from elitefurretai.rl.config import OPEN_TEAM_SHEETS_MODES, RNaDConfig

    # Lives on CurriculumConfig now (not top-level); default is "mixed".
    assert RNaDConfig().curriculum.open_team_sheets == "mixed"
    for mode in OPEN_TEAM_SHEETS_MODES:
        cfg = RNaDConfig.from_dict({"curriculum": {"open_team_sheets": mode}})
        assert cfg.curriculum.open_team_sheets == mode
        # round-trips through to_dict/from_dict
        assert RNaDConfig.from_dict(cfg.to_dict()).curriculum.open_team_sheets == mode
    # YAML parses unquoted on/off as booleans — coerced back to strings.
    assert (
        RNaDConfig.from_dict(
            {"curriculum": {"open_team_sheets": True}}
        ).curriculum.open_team_sheets
        == "on"
    )
    assert (
        RNaDConfig.from_dict(
            {"curriculum": {"open_team_sheets": False}}
        ).curriculum.open_team_sheets
        == "off"
    )
    with pytest.raises(ValueError):
        RNaDConfig.from_dict({"curriculum": {"open_team_sheets": "bogus"}})


def test_eval_open_team_sheets_default_on_and_rejects_mixed():
    from elitefurretai.rl.config import RNaDConfig

    assert RNaDConfig().eval.open_team_sheets == "on"
    assert (
        RNaDConfig.from_dict({"eval": {"open_team_sheets": "off"}}).eval.open_team_sheets
        == "off"
    )
    # unquoted on/off booleans coerced
    assert (
        RNaDConfig.from_dict({"eval": {"open_team_sheets": False}}).eval.open_team_sheets
        == "off"
    )
    with pytest.raises(ValueError):
        RNaDConfig.from_dict({"eval": {"open_team_sheets": "mixed"}})


def test_open_team_sheets_for_battle_resolution():
    from elitefurretai.rl.config import open_team_sheets_for_battle

    assert (
        open_team_sheets_for_battle("off", is_vgc_bench=False, mixed_roll=False) is False
    )
    assert open_team_sheets_for_battle("on", is_vgc_bench=False, mixed_roll=False) is True
    assert (
        open_team_sheets_for_battle("mixed", is_vgc_bench=False, mixed_roll=True) is True
    )
    assert (
        open_team_sheets_for_battle("mixed", is_vgc_bench=False, mixed_roll=False) is False
    )
    # vgc_bench is always ON regardless of mode
    for mode in ("off", "on", "mixed"):
        assert (
            open_team_sheets_for_battle(mode, is_vgc_bench=True, mixed_roll=False) is True
        )
    with pytest.raises(ValueError):
        open_team_sheets_for_battle("bogus", is_vgc_bench=False, mixed_roll=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
