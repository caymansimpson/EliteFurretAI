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
    def __init__(self):
        self.opponent_type = "self_play"


class _DummyOpponent:
    def __init__(self, model):
        self.model = model


def _make_factory(curriculum):
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
        ghost_models_dir=None,
    )


def test_configure_opponent_for_batch_supports_ghosts(monkeypatch):
    factory = _make_factory({"ghosts": 1.0})
    ghost_agent = SimpleNamespace(name="ghost")
    monkeypatch.setattr(factory, "_get_ghost_agent", lambda: ghost_agent)

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.GHOSTS
    assert player.opponent_type == factory.GHOSTS
    assert opponent.model is ghost_agent


def test_configure_opponent_for_batch_ghosts_fallback_to_self_play(monkeypatch):
    factory = _make_factory({"ghosts": 1.0})
    monkeypatch.setattr(factory, "_get_ghost_agent", lambda: None)

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=SimpleNamespace(name="other"))

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.SELF_PLAY
    assert player.opponent_type == factory.SELF_PLAY
    assert opponent.model is factory.main_agent


def test_configure_opponent_for_batch_supports_exploiters(monkeypatch):
    factory = _make_factory({"exploiters": 1.0})
    exploiter_agent = SimpleNamespace(name="exploiter")
    monkeypatch.setattr(factory, "_get_exploiter_agent", lambda: exploiter_agent)

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=factory.main_agent)

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.EXPLOITERS
    assert player.opponent_type == factory.EXPLOITERS
    assert opponent.model is exploiter_agent


def test_configure_opponent_for_batch_exploiters_fallback_to_self_play(monkeypatch):
    factory = _make_factory({"exploiters": 1.0})
    monkeypatch.setattr(factory, "_get_exploiter_agent", lambda: None)

    player = _DummyPlayer()
    opponent = _DummyOpponent(model=SimpleNamespace(name="other"))

    selected = factory.configure_opponent_for_batch(
        cast(BatchInferencePlayer, player),
        cast(BatchInferencePlayer, opponent),
    )

    assert selected == factory.SELF_PLAY
    assert player.opponent_type == factory.SELF_PLAY
    assert opponent.model is factory.main_agent
