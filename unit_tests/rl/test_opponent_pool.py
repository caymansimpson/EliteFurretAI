# -*- coding: utf-8 -*-
"""Unit tests for RL OpponentPool."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.opponents import OpponentPool
from elitefurretai.rl.players import RNaDAgent


@pytest.fixture
def mock_main_model():
    return MagicMock(spec=RNaDAgent)


@pytest.fixture
def player_config():
    return AccountConfiguration("TestOpponent", None)


@pytest.fixture
def server_config():
    return ServerConfiguration("ws://localhost:8000/showdown/websocket", None)  # type: ignore


@pytest.fixture
def temp_exploiters_dir(tmp_path):
    d = tmp_path / "exploiters"
    d.mkdir()
    return str(d)


@pytest.fixture
def temp_ghosts_dir(tmp_path):
    d = tmp_path / "ghosts"
    d.mkdir()
    return str(d)


def _make_pool(mock_main_model, temp_exploiters_dir, temp_ghosts_dir, curriculum=None):
    return OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regc",
        curriculum=curriculum,
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=temp_ghosts_dir,
    )


def test_opponent_pool_initialization(
    mock_main_model, temp_exploiters_dir, temp_ghosts_dir
):
    pool = _make_pool(mock_main_model, temp_exploiters_dir, temp_ghosts_dir)

    assert hasattr(pool, "curriculum")
    assert np.isclose(sum(pool.curriculum.values()), 1.0)


def test_opponent_pool_custom_curriculum(
    mock_main_model, temp_exploiters_dir, temp_ghosts_dir
):
    curriculum = {
        "self_play": 0.8,
        "bc_player": 0.1,
        "exploiters": 0.05,
        "ghosts": 0.05,
        "max_damage": 0.0,
        "random_baseline": 0.0,
        "max_base_power_baseline": 0.0,
        "simple_heuristic_baseline": 0.0,
    }
    pool = _make_pool(
        mock_main_model,
        temp_exploiters_dir,
        temp_ghosts_dir,
        curriculum=curriculum,
    )

    assert pool.curriculum["self_play"] == 0.8


def test_opponent_pool_rejects_invalid_curriculum(
    mock_main_model, temp_exploiters_dir, temp_ghosts_dir
):
    curriculum = {
        "self_play": 0.5,
        "bc_player": 0.2,
        "exploiters": 0.1,
        "ghosts": 0.1,
    }

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _make_pool(
            mock_main_model,
            temp_exploiters_dir,
            temp_ghosts_dir,
            curriculum=curriculum,
        )


def test_add_ghost_respects_limit(mock_main_model, temp_exploiters_dir, temp_ghosts_dir):
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regc",
        max_ghosts=3,
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=temp_ghosts_dir,
    )

    for step in [100, 200, 300, 400, 500]:
        pool.add_ghost(step, f"path/model_{step}.pt")

    assert len(pool.ghosts) == 3
    steps = [m[0] for m in pool.ghosts]
    assert 500 in steps and 400 in steps and 300 in steps


def test_update_and_get_win_rate_stats(
    mock_main_model, temp_exploiters_dir, temp_ghosts_dir
):
    pool = _make_pool(mock_main_model, temp_exploiters_dir, temp_ghosts_dir)

    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=True)
    pool.update_win_rate("self_play", won=False)

    stats = pool.get_win_rate_stats()
    assert np.isclose(stats["self_play"], 2.0 / 3.0)


def test_load_exploiter_models_from_directory(
    mock_main_model, temp_exploiters_dir, temp_ghosts_dir
):
    for idx in range(3):
        path = os.path.join(temp_exploiters_dir, f"exploiter_{idx}.pt")
        with open(path, "wb"):
            pass

    pool = _make_pool(mock_main_model, temp_exploiters_dir, temp_ghosts_dir)
    pool._load_exploiter_models()

    assert len(pool.exploiter_models) == 3


def test_sample_opponent_self_play_only(
    mock_main_model, player_config, server_config, temp_exploiters_dir, temp_ghosts_dir
):
    curriculum = {
        "self_play": 1.0,
        "bc_player": 0.0,
        "exploiters": 0.0,
        "ghosts": 0.0,
        "max_damage": 0.0,
        "random_baseline": 0.0,
        "max_base_power_baseline": 0.0,
        "simple_heuristic_baseline": 0.0,
    }
    pool = _make_pool(
        mock_main_model,
        temp_exploiters_dir,
        temp_ghosts_dir,
        curriculum=curriculum,
    )

    opponent = pool.sample_opponent(
        player_config, server_config, team="Pikachu @ Light Ball"
    )
    assert opponent is not None


def test_opponent_pool_tracks_active_ghost_slots(tmp_path, mock_main_model, temp_exploiters_dir):
    """OpponentPool exposes active_ghost_slots reflecting which slots
    hold real ghost weights, and rotates LRU when full."""
    ghosts_dir = str(tmp_path / "ghosts")
    os.makedirs(ghosts_dir)
    pool = OpponentPool(
        main_model=mock_main_model,
        device="cpu",
        battle_format="gen9vgc2023regc",
        max_ghosts=3,
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=ghosts_dir,
    )
    assert pool.active_ghost_slots() == set()
    assert pool.slot_for_ghost_path == {}

    # First three add_ghost calls fill slots 0, 1, 2
    for step, name in [(10, "g10.pt"), (20, "g20.pt"), (30, "g30.pt")]:
        p = tmp_path / name
        p.write_bytes(b"x")  # add_ghost only stores the path
        pool.add_ghost(step, str(p))
    assert pool.active_ghost_slots() == {0, 1, 2}
    assert set(pool.slot_for_ghost_path.values()) == {0, 1, 2}

    # Fourth add evicts LRU (slot 0 — oldest by insertion)
    p4 = tmp_path / "g40.pt"
    p4.write_bytes(b"x")
    pool.add_ghost(40, str(p4))
    assert pool.active_ghost_slots() == {0, 1, 2}  # still full
    assert pool.slot_for_ghost_path[str(p4)] == 0
