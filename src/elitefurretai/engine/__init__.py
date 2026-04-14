"""Battle engine runtime components for EliteFurretAI.

This package contains the concrete engine backends and shared runtime helpers used
by RL training and evaluation. Rust-backed self-play runtime pieces and
Showdown-server management live here because they are execution concerns rather
than RL algorithm concerns.

See `ENGINE.md` in this package for the current backend recommendation and the
consolidated Stage 2 engine learnings.
"""

from elitefurretai.engine.showdown_server_manager import (
    allocate_server_ports,
    derive_external_vgcbench_username,
    launch_external_vgcbench_runners,
    launch_showdown_servers,
    shutdown_external_vgcbench_runners,
    shutdown_showdown_servers,
)
from elitefurretai.engine.sync_battle_driver import (
    SyncBaselineController,
    SyncPolicyPlayer,
    SyncRustBattleDriver,
)

__all__ = [
    "allocate_server_ports",
    "derive_external_vgcbench_username",
    "launch_external_vgcbench_runners",
    "launch_showdown_servers",
    "shutdown_external_vgcbench_runners",
    "shutdown_showdown_servers",
    "SyncBaselineController",
    "SyncPolicyPlayer",
    "SyncRustBattleDriver",
]
