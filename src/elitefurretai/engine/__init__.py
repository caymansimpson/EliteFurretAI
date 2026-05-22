"""Battle engine runtime components for EliteFurretAI.

This package contains the Showdown-websocket battle execution path and the
helpers that manage the Showdown server processes. Engine code lives here
rather than under `rl/` because it's an execution concern, not an RL
algorithm concern.

See `ENGINE.md` in this package for current architecture notes.
"""

from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.engine.vgc_environment import VGCEnvironment

__all__ = [
    "VGCEnvironment",
    "allocate_server_ports",
    "launch_showdown_servers",
    "shutdown_showdown_servers",
]
