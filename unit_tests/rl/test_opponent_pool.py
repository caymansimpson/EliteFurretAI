# -*- coding: utf-8 -*-
"""Unit tests for RL OpponentPool."""

import os

import numpy as np
import pytest
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.opponents import OpponentPool


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


_DEFAULT_TEST_CURRICULUM = {
    OpponentPool.SELF_PLAY: 0.40,
    OpponentPool.BC_PLAYER: 0.20,
    OpponentPool.EXPLOITERS: 0.20,
    OpponentPool.GHOSTS: 0.20,
}


def _make_pool(temp_exploiters_dir, temp_ghosts_dir, curriculum=None):
    return OpponentPool(
        curriculum=curriculum
        if curriculum is not None
        else dict(_DEFAULT_TEST_CURRICULUM),
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=temp_ghosts_dir,
    )


def test_opponent_pool_initialization(temp_exploiters_dir, temp_ghosts_dir):
    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir)

    assert hasattr(pool, "curriculum")
    assert np.isclose(sum(pool.curriculum.values()), 1.0)


def test_opponent_pool_custom_curriculum(temp_exploiters_dir, temp_ghosts_dir):
    curriculum = {
        "self_play": 0.8,
        "bc_player": 0.1,
        "exploiters": 0.05,
        "ghosts": 0.05,
        "max_damage": 0.0,
        "random": 0.0,
        "max_base_power": 0.0,
        "simple_heuristic": 0.0,
    }
    pool = _make_pool(
        temp_exploiters_dir,
        temp_ghosts_dir,
        curriculum=curriculum,
    )

    assert pool.curriculum["self_play"] == 0.8


def test_opponent_pool_rejects_invalid_curriculum(temp_exploiters_dir, temp_ghosts_dir):
    curriculum = {
        "self_play": 0.5,
        "bc_player": 0.2,
        "exploiters": 0.1,
        "ghosts": 0.1,
    }

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _make_pool(
            temp_exploiters_dir,
            temp_ghosts_dir,
            curriculum=curriculum,
        )


def test_add_ghost_respects_limit(temp_exploiters_dir, temp_ghosts_dir):
    pool = OpponentPool(
        curriculum=dict(_DEFAULT_TEST_CURRICULUM),
        max_ghosts=3,
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=temp_ghosts_dir,
    )

    for step in [100, 200, 300, 400, 500]:
        pool.add_ghost(f"path/model_{step}.pt")

    assert len(pool.slot_for_ghost_path) == 3
    retained = set(pool.slot_for_ghost_path.keys())
    assert retained == {f"path/model_{s}.pt" for s in (300, 400, 500)}


def test_update_and_get_win_rate_stats(temp_exploiters_dir, temp_ghosts_dir):
    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir)

    pool.record_battle_result(opponent_type="self_play", won=True)
    pool.record_battle_result(opponent_type="self_play", won=True)
    pool.record_battle_result(opponent_type="self_play", won=False)

    stats = pool.get_win_rate_stats()
    assert np.isclose(stats["self_play"], 2.0 / 3.0)


def test_load_exploiter_models_from_directory(temp_exploiters_dir, temp_ghosts_dir):
    for idx in range(3):
        path = os.path.join(temp_exploiters_dir, f"exploiter_{idx}.pt")
        with open(path, "wb"):
            pass

    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir)
    pool._load_exploiter_models()

    assert len(pool.slot_for_exploiter_path) == 3


def test_opponent_pool_tracks_active_ghost_slots(tmp_path, temp_exploiters_dir):
    """OpponentPool exposes active_ghost_slots reflecting which slots
    hold real ghost weights, and rotates LRU when full."""
    ghosts_dir = str(tmp_path / "ghosts")
    os.makedirs(ghosts_dir)
    pool = OpponentPool(
        curriculum=dict(_DEFAULT_TEST_CURRICULUM),
        max_ghosts=3,
        exploiter_models_dir=temp_exploiters_dir,
        ghosts_dir=ghosts_dir,
    )
    assert pool.active_ghost_slots() == set()
    assert pool.slot_for_ghost_path == {}

    # First three add_ghost calls fill slots 0, 1, 2
    for name in ["g10.pt", "g20.pt", "g30.pt"]:
        p = tmp_path / name
        p.write_bytes(b"x")  # add_ghost only stores the path
        pool.add_ghost(str(p))
    assert pool.active_ghost_slots() == {0, 1, 2}
    assert set(pool.slot_for_ghost_path.values()) == {0, 1, 2}

    # Fourth add evicts LRU (slot 0 — oldest by insertion)
    p4 = tmp_path / "g40.pt"
    p4.write_bytes(b"x")
    pool.add_ghost(str(p4))
    assert pool.active_ghost_slots() == {0, 1, 2}  # still full
    assert pool.slot_for_ghost_path[str(p4)] == 0


def test_opponent_pool_tracks_active_exploiter_slots(tmp_path, temp_exploiters_dir):
    """OpponentPool exposes active_exploiter_slots reflecting which slots
    hold real exploiter snapshot weights, and rotates LRU when full."""
    pool = OpponentPool(
        curriculum=dict(_DEFAULT_TEST_CURRICULUM),
        max_exploiter_models=3,
        exploiter_models_dir=str(tmp_path),
        ghosts_dir=temp_exploiters_dir,
    )
    assert pool.active_exploiter_slots() == set()
    assert pool.slot_for_exploiter_path == {}

    # First three add_exploiter calls fill slots 0, 1, 2
    for i, name in enumerate(["e10.pt", "e20.pt", "e30.pt"]):
        p = tmp_path / name
        p.write_bytes(b"x")
        import time

        os.utime(p, (time.time() + i, time.time() + i))
        pool.add_exploiter(str(p))
    assert pool.active_exploiter_slots() == {0, 1, 2}
    assert set(pool.slot_for_exploiter_path.values()) == {0, 1, 2}

    # Fourth add evicts LRU (slot 0 — oldest by insertion)
    p4 = tmp_path / "e40.pt"
    p4.write_bytes(b"x")
    import time

    os.utime(p4, (time.time() + 4, time.time() + 4))
    pool.add_exploiter(str(p4))
    assert pool.active_exploiter_slots() == {0, 1, 2}  # still full
    assert pool.slot_for_exploiter_path[str(p4)] == 0


def test_train_exploiter_unavailable_when_disabled(temp_exploiters_dir, temp_ghosts_dir):
    """train_exploiter is gated on its configured base weight: with the
    pipeline off (weight 0 / absent), the slot is unavailable so the
    adaptive agent axis never leaks weight onto a learner that doesn't
    exist."""
    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir)
    assert pool._opponent_available(OpponentPool.TRAIN_EXPLOITER) is False


def test_train_exploiter_available_when_enabled(temp_exploiters_dir, temp_ghosts_dir):
    """A positive configured train_exploiter weight marks the slot
    available, matching the pipeline train.py provisions in that case."""
    curriculum = {
        OpponentPool.SELF_PLAY: 0.6,
        OpponentPool.GHOSTS: 0.2,
        OpponentPool.TRAIN_EXPLOITER: 0.2,
    }
    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir, curriculum=curriculum)
    assert pool._opponent_available(OpponentPool.TRAIN_EXPLOITER) is True


def test_disabled_train_exploiter_stays_zero_after_adaptive_update(
    temp_exploiters_dir, temp_ghosts_dir
):
    """The adaptive agent axis must not raise train_exploiter off zero when
    the pipeline is disabled, even though the slot is a curriculum key."""
    curriculum = {
        OpponentPool.SELF_PLAY: 0.8,
        OpponentPool.GHOSTS: 0.2,
        OpponentPool.TRAIN_EXPLOITER: 0.0,
    }
    pool = _make_pool(temp_exploiters_dir, temp_ghosts_dir, curriculum=curriculum)

    # Some real self_play history so the adaptive update has a non-trivial
    # distribution to redistribute across the available slots.
    for _ in range(50):
        pool.record_battle_result(opponent_type=OpponentPool.SELF_PLAY, won=True)

    pool.update_curriculum()

    # train_exploiter is filtered out as unavailable rather than floored to
    # 1e-2, so it carries no weight in the adapted curriculum.
    assert pool.curriculum.get(OpponentPool.TRAIN_EXPLOITER, 0.0) == 0.0


def test_apply_mega_megas_first_eligible_move_slot():
    import types

    from poke_env.battle import Move
    from poke_env.player.battle_order import SingleBattleOrder

    from elitefurretai.agents.max_damage_player import MaxDamagePlayer

    o0 = SingleBattleOrder(order=Move("flamethrower", gen=9))
    o1 = SingleBattleOrder(order=Move("surf", gen=9))
    battle = types.SimpleNamespace(can_mega_evolve=[True, False])
    MaxDamagePlayer._apply_mega([o0, o1], battle)
    assert o0.mega is True
    assert o1.mega is False  # slot 1 cannot mega


def test_apply_mega_megas_at_most_one_slot():
    import types

    from poke_env.battle import Move
    from poke_env.player.battle_order import SingleBattleOrder

    from elitefurretai.agents.max_damage_player import MaxDamagePlayer

    o0 = SingleBattleOrder(order=Move("flamethrower", gen=9))
    o1 = SingleBattleOrder(order=Move("surf", gen=9))
    battle = types.SimpleNamespace(can_mega_evolve=[True, True])
    MaxDamagePlayer._apply_mega([o0, o1], battle)
    assert o0.mega is True
    assert o1.mega is False  # only the first eligible slot megas (one per turn)


def test_apply_mega_skips_switch_orders():
    import types

    from poke_env.battle import Move, Pokemon
    from poke_env.player.battle_order import SingleBattleOrder

    from elitefurretai.agents.max_damage_player import MaxDamagePlayer

    sw = SingleBattleOrder(order=Pokemon(gen=9, species="venusaur"))
    mv = SingleBattleOrder(order=Move("flamethrower", gen=9))
    battle = types.SimpleNamespace(can_mega_evolve=[True, True])
    MaxDamagePlayer._apply_mega([sw, mv], battle)
    assert sw.mega is False  # switch orders are not mega'd
    assert mv.mega is True


def test_apply_mega_noop_when_unavailable():
    import types

    from poke_env.battle import Move
    from poke_env.player.battle_order import SingleBattleOrder

    from elitefurretai.agents.max_damage_player import MaxDamagePlayer

    o0 = SingleBattleOrder(order=Move("flamethrower", gen=9))
    battle = types.SimpleNamespace(can_mega_evolve=[False, False])
    MaxDamagePlayer._apply_mega([o0], battle)
    assert o0.mega is False
