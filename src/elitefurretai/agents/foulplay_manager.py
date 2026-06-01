"""FoulPlay subprocess constants and username helpers.

foul-play-doubles depends on ``poke-engine-doubles`` (a Rust extension) and
an older ``poke_env`` than EFA's training env, so it runs as a subprocess in
its own venv (``../venv-foulplay/``). The subprocess entry point lives at
``agents/_foulplay_subprocess.py``; EFA-side code interacts with FoulPlay only
by Showdown username.

The live eval path spawns the subprocess via
``analyze.analysis_utils._launch_foulplay_subprocess``, which borrows the
constants and username helpers below. This module previously also hosted a
``FoulPlayManager`` class for an inline-during-training eval; that path was
superseded by the generic multi-bucket eval in ``analyze.evaluate_model`` and
removed.
"""

from __future__ import annotations

SUBPROCESS_SCRIPT: str = "src/elitefurretai/agents/_foulplay_subprocess.py"
BASE_USERNAME: str = "FOULPLAY"
WAIT_FOR_SERVER_TIMEOUT_S: float = 180.0
ACCEPT_OPEN_TEAM_SHEET: bool = False
STARTUP_WAIT_S: float = 10.0


def derive_username(base_username: str, server_port: int) -> str:
    """Generate a server-scoped runner username (Showdown max length is 18).

    Same contract as :meth:`VGCBenchManager.derive_username`.
    """
    suffix = f"_{server_port}"
    max_base_len = max(1, 18 - len(suffix))
    return f"{base_username[:max_base_len]}{suffix}"


def should_suffix_port(num_servers: int) -> bool:
    """Whether to append ``_<port>`` to the base username."""
    return num_servers > 1
