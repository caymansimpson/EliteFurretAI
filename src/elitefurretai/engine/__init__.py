"""Battle engine runtime components for EliteFurretAI.

This package contains the Showdown-websocket battle execution path and the
helpers that manage the Showdown server processes. Engine code lives here
rather than under `rl/` because it's an execution concern, not an RL
algorithm concern.

See `ENGINE.md` in this package for current architecture notes.

`VGCEnvironment` is exposed via lazy ``__getattr__`` so that lightweight
sibling modules (e.g. `battle_renderer`) can be imported without paying
the cost of loading the full RL training stack — and so that those
modules can be safely imported from `inference` without creating a
circular dependency through `etl -> inference`.
"""

from typing import TYPE_CHECKING

from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    launch_showdown_servers,
    shutdown_showdown_servers,
)

if TYPE_CHECKING:
    from elitefurretai.engine.vgc_environment import VGCEnvironment

__all__ = [
    "VGCEnvironment",
    "allocate_server_ports",
    "launch_showdown_servers",
    "shutdown_showdown_servers",
]


def __getattr__(name: str):
    if name == "VGCEnvironment":
        from elitefurretai.engine.vgc_environment import VGCEnvironment

        return VGCEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
