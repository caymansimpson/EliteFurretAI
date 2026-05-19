"""Unit tests for the human-vs-model terminal CLI.

The interactive parts (stdin loop, websocket battle) can't be unit tested;
this file covers the deferred-debug routing logic and the import smoke.
"""

import io
from contextlib import redirect_stdout

from elitefurretai.rl.analyze.play_human_vs_model import DeferredVerboseModelPlayer


def _make_bare_player(reveal: bool) -> DeferredVerboseModelPlayer:
    """Build a DeferredVerboseModelPlayer without running __init__.

    The base VerboseModelPlayer.__init__ loads a real checkpoint and opens
    a websocket; both are out of scope for these tests. We only exercise
    the buffer-routing helpers, so we instantiate the minimal attrs.
    """
    player = DeferredVerboseModelPlayer.__new__(DeferredVerboseModelPlayer)
    player.reveal = reveal
    player.pending_debug = []
    return player


def test_emit_buffers_when_reveal_false():
    player = _make_bare_player(reveal=False)
    captured = io.StringIO()
    with redirect_stdout(captured):
        player._emit("hello world\n")

    assert player.pending_debug == ["hello world\n"]
    assert captured.getvalue() == ""


def test_emit_prints_when_reveal_true():
    player = _make_bare_player(reveal=True)
    captured = io.StringIO()
    with redirect_stdout(captured):
        player._emit("hello world\n")

    assert player.pending_debug == []
    assert captured.getvalue() == "hello world\n"


def test_flush_debug_concatenates_and_clears():
    player = _make_bare_player(reveal=False)
    player.pending_debug.extend(["alpha\n", "beta\n", "gamma\n"])

    flushed = player.flush_debug()

    assert flushed == "alpha\nbeta\ngamma\n"
    assert player.pending_debug == []


def test_flush_debug_on_empty_buffer_returns_empty_string():
    player = _make_bare_player(reveal=False)
    assert player.flush_debug() == ""


def test_module_smoke_import():
    """Ensure the module imports without instantiating Players."""
    import elitefurretai.rl.analyze.play_human_vs_model as mod

    assert hasattr(mod, "DeferredVerboseModelPlayer")
    assert hasattr(mod, "HumanVsModelPlayer")
    assert hasattr(mod, "main")


def test_simple_model_player_is_verbose_parent():
    """VerboseModelPlayer should inherit from SimpleModelPlayer.

    This factoring lets analyze/benchmark scripts use SimpleModelPlayer
    (no per-turn print noise) without copying the inference loop.
    """
    from elitefurretai.rl.players import SimpleModelPlayer, VerboseModelPlayer

    assert issubclass(VerboseModelPlayer, SimpleModelPlayer)
    assert issubclass(DeferredVerboseModelPlayer, SimpleModelPlayer)
