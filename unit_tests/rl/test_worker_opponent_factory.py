import queue
from types import SimpleNamespace
from typing import cast

from poke_env import ServerConfiguration

from elitefurretai.etl import Embedder, TeamRepo
from elitefurretai.rl.opponents import WorkerOpponentFactory
from elitefurretai.rl.players import BatchInferencePlayer, RNaDAgent


class _DummyTeamRepo:
    def sample_team(self, battle_format, subdirectory=None):
        return "Pikachu @ Light Ball"


class _DummyPlayer:
    def __init__(self, model=None):
        self.opponent_type = "self_play"
        self.model = model


class _DummyOpponent:
    def __init__(self, model):
        self.model = model
        self.inference_client: object = None


def _make_factory(
    curriculum, exploiter_agent=None, victim_agent=None, worker_inference_clients=None
):
    return WorkerOpponentFactory(
        team_repo=cast(TeamRepo, _DummyTeamRepo()),
        battle_format="gen9vgc2023regc",
        team_subdirectory=None,
        server_config=cast(ServerConfiguration, SimpleNamespace()),
        main_agent=cast(RNaDAgent, SimpleNamespace(name="main")),
        bc_agent=cast(RNaDAgent, SimpleNamespace(name="bc")),
        curriculum=curriculum,
        embedder=cast(Embedder, SimpleNamespace()),
        worker_id=0,
        run_id="0000",
        device="cpu",
        exploiter_agent=cast(RNaDAgent, exploiter_agent) if exploiter_agent else None,
        victim_agent=cast(RNaDAgent, victim_agent) if victim_agent else None,
        worker_inference_clients=worker_inference_clients,
    )


def test_configure_opponent_for_batch_supports_train_exploiter():
    """When TRAIN_EXPLOITER is sampled and both co-training agents are
    provisioned, player swaps to the live exploiter and opponent to the
    frozen victim. The trajectory tag (`opponent_type`) is what main-process
    routing uses to direct the trajectory to the exploiter learner."""
    exploiter = SimpleNamespace(name="exploiter")
    victim = SimpleNamespace(name="victim")
    factory = _make_factory(
        {"train_exploiter": 1.0},
        exploiter_agent=exploiter,
        victim_agent=victim,
    )

    player = _DummyPlayer(model=factory.main_agent)
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.TRAIN_EXPLOITER
    assert player.opponent_type == factory.TRAIN_EXPLOITER
    assert player.model is exploiter
    assert opponent.model is victim


def test_update_exploiter_weights_loads_into_agent_model():
    """update_exploiter_weights should call load_state_dict on the exploiter
    agent's model. The state_dict isn't validated here — that's the model's
    job — we just verify the call is dispatched correctly."""
    captured = {}

    class _StubModel:
        def load_state_dict(self, sd):
            captured["sd"] = sd

    exploiter = SimpleNamespace(model=_StubModel())
    factory = _make_factory({"train_exploiter": 1.0}, exploiter_agent=exploiter)

    factory.update_exploiter_weights({"layer.weight": "tensor"})

    assert captured["sd"] == {"layer.weight": "tensor"}


def test_update_exploiter_weights_no_op_when_disabled():
    """When co-training is off (no exploiter_agent), update is a silent no-op."""
    factory = _make_factory({"self_play": 1.0})  # no exploiter_agent
    factory.update_exploiter_weights({"layer.weight": "tensor"})  # must not raise


def test_update_victim_weights_loads_into_agent_model():
    captured = {}

    class _StubModel:
        def load_state_dict(self, sd):
            captured["sd"] = sd

    victim = SimpleNamespace(model=_StubModel())
    factory = _make_factory({"train_exploiter": 1.0}, victim_agent=victim)

    factory.update_victim_weights({"layer.weight": "tensor"})

    assert captured["sd"] == {"layer.weight": "tensor"}


def test_update_victim_weights_no_op_when_disabled():
    factory = _make_factory({"self_play": 1.0})
    factory.update_victim_weights({"layer.weight": "tensor"})  # must not raise


def test_configure_opponent_for_batch_train_exploiter_falls_back_when_unprovisioned():
    """If the curriculum activates train_exploiter before the main process
    has provisioned exploiter/victim agents, fall back to self-play. The
    trajectory routes as main; main process is responsible for ensuring
    this race doesn't matter (warmup gate keeps weight at 0 until ready)."""
    factory = _make_factory({"train_exploiter": 1.0})  # no agents passed

    player = _DummyPlayer(model=factory.main_agent)
    opponent = _DummyOpponent(model=SimpleNamespace(name="other"))

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.SELF_PLAY
    assert player.opponent_type == factory.SELF_PLAY
    assert opponent.model is factory.main_agent


def _patch_batch_inference_player(monkeypatch):
    """Replace BatchInferencePlayer in opponents.py with a recording stub.

    Returns the list that captures kwargs of each construction so a test
    can assert what got passed.
    """
    captured_kwargs = []

    class _RecordingPlayer:
        def __init__(self, **kwargs):
            captured_kwargs.append(kwargs)

    monkeypatch.setattr(
        "elitefurretai.rl.opponents.BatchInferencePlayer", _RecordingPlayer
    )
    return captured_kwargs


def test_create_player_pairs_omits_max_concurrent_when_none(monkeypatch):
    """Default config leaves max_concurrent_battles_per_player=None — the
    kwarg must not be passed to BatchInferencePlayer so poke-env's library
    default (1) stays in force. Forwarding a None would surface as an
    unexpected override.
    """
    factory = _make_factory({"self_play": 1.0})
    assert factory.max_concurrent_battles_per_player is None

    captured_kwargs = _patch_batch_inference_player(monkeypatch)
    factory.create_player_pairs(num_pairs=2, local_traj_queue=cast(queue.Queue, None))

    assert len(captured_kwargs) == 4  # 2 pairs × (player + opponent)
    for kw in captured_kwargs:
        assert "max_concurrent_battles" not in kw


def test_create_player_pairs_passes_max_concurrent_when_set(monkeypatch):
    """When the config sets a concurrent-battle cap, every player and
    opponent constructed by create_player_pairs must receive it.
    """
    factory = _make_factory({"self_play": 1.0})
    factory.max_concurrent_battles_per_player = 16

    captured_kwargs = _patch_batch_inference_player(monkeypatch)
    factory.create_player_pairs(num_pairs=3, local_traj_queue=cast(queue.Queue, None))

    assert len(captured_kwargs) == 6  # 3 pairs × (player + opponent)
    for kw in captured_kwargs:
        assert kw.get("max_concurrent_battles") == 16


def test_set_active_ghost_slots_updates_state():
    factory = _make_factory({"self_play": 1.0})
    assert factory._active_ghost_slots == set()
    factory.set_active_ghost_slots([0, 2, 4])
    assert factory._active_ghost_slots == {0, 2, 4}
    factory.set_active_ghost_slots([])
    assert factory._active_ghost_slots == set()


def test_configure_opponent_for_batch_ghosts_routes_via_active_slot():
    """When GHOSTS is sampled and slots are active, opponent gets a
    ghost_<slot> InferenceClient from the bundle."""
    from unittest.mock import MagicMock

    ghost_clients = {f"ghost_{i}": MagicMock(name=f"ghost_client_{i}") for i in range(3)}
    main_client = MagicMock(name="main_client")

    clients = MagicMock()
    clients.has = MagicMock(
        side_effect=lambda n: n in {"main", "ghost_0", "ghost_1", "ghost_2"}
    )
    clients.get = MagicMock(side_effect=lambda n: ghost_clients.get(n, main_client))

    factory = _make_factory({"ghosts": 1.0})
    factory.worker_inference_clients = clients
    factory.set_active_ghost_slots([0, 1, 2])

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.GHOSTS
    assert player.opponent_type == factory.GHOSTS
    # opponent.inference_client should now point to one of the ghost clients
    assigned = opponent.inference_client
    assert assigned in ghost_clients.values()


def test_configure_opponent_for_batch_ghosts_falls_back_when_no_active_slots():
    """When GHOSTS is sampled but no slots are active, fall back to self-play.
    (Curriculum normally guards this via _opponent_available; this test pins
    the defensive fallback inside _swap_to.)"""
    from unittest.mock import MagicMock

    clients = MagicMock()
    clients.has = MagicMock(side_effect=lambda n: n == "main")
    main_client = MagicMock(name="main_client")
    clients.get = MagicMock(return_value=main_client)

    factory = _make_factory({"ghosts": 1.0}, worker_inference_clients=clients)
    # Note: deliberately NOT calling set_active_ghost_slots — empty by default.

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.SELF_PLAY


def test_set_active_exploiter_slots_updates_state():
    factory = _make_factory({"self_play": 1.0})
    assert factory._active_exploiter_slots == set()
    factory.set_active_exploiter_slots([0, 1, 2])
    assert factory._active_exploiter_slots == {0, 1, 2}
    factory.set_active_exploiter_slots([])
    assert factory._active_exploiter_slots == set()


def test_configure_opponent_for_batch_exploiters_routes_via_active_slot():
    """When EXPLOITERS is sampled and slots are active, opponent gets an
    exploiter_snap_<slot> InferenceClient from the bundle."""
    from unittest.mock import MagicMock

    snap_clients = {
        f"exploiter_snap_{i}": MagicMock(name=f"snap_client_{i}") for i in range(3)
    }
    main_client = MagicMock(name="main_client")

    clients = MagicMock()
    clients.has = MagicMock(
        side_effect=lambda n: n in {"main", "exploiter_snap_0", "exploiter_snap_1", "exploiter_snap_2"}
    )
    clients.get = MagicMock(side_effect=lambda n: snap_clients.get(n, main_client))

    factory = _make_factory({"exploiters": 1.0})
    factory.worker_inference_clients = clients
    factory.set_active_exploiter_slots([0, 1, 2])

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.EXPLOITERS
    assert player.opponent_type == factory.EXPLOITERS
    # opponent.inference_client should now point to one of the snap clients
    assigned = opponent.inference_client
    assert assigned in snap_clients.values()


def test_configure_opponent_for_batch_exploiters_falls_back_when_no_active_slots():
    """When EXPLOITERS is sampled but no slots are active, fall back to self-play.
    (Curriculum normally guards this via _opponent_available; this test pins
    the defensive fallback inside configure_opponent_for_batch.)"""
    from unittest.mock import MagicMock

    clients = MagicMock()
    clients.has = MagicMock(side_effect=lambda n: n == "main")
    main_client = MagicMock(name="main_client")
    clients.get = MagicMock(return_value=main_client)

    factory = _make_factory({"exploiters": 1.0}, worker_inference_clients=clients)
    # Note: deliberately NOT calling set_active_exploiter_slots — empty by default.

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.SELF_PLAY
