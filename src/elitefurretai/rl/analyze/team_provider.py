# -*- coding: utf-8 -*-
"""Team-source parsing for the unified evaluation entry point.

A ``TeamProvider`` is a no-arg callable that returns a Showdown team
string. Three sources are supported:

* file (.txt / .team) — read once, fixed team for every battle.
* directory — wraps ``TeamRepo.sample_team`` for per-battle random
  sampling.
* default — sample from the format's root team directory under
  ``data/teams/<format>/`` (mirrors evaluate.py's prior fallback).

``parse_team_specification`` does file-or-directory dispatch via filesystem
checks; callers don't need to switch on type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from elitefurretai.etl import TeamRepo

TeamProvider = Callable[[], str]


def parse_team_specification(raw: Optional[str], *, battle_format: str) -> TeamProvider:
    """Resolve ``raw`` into a ``TeamProvider``.

    Resolution order:
        1. ``raw is None`` or empty → format default
           (``TeamRepo(filepath="data/teams").sample_team(format)``).
        2. ``Path(raw).is_file()`` → read the file once, return a
           closure that hands back the same string every call.
        3. ``Path(raw).is_dir()`` → wrap ``TeamRepo(filepath=raw)`` and
           sample a team per call.
        4. Else raise ``ValueError`` — neither a file nor a directory.
    """
    if not raw:
        return _default_provider(battle_format)

    path = Path(raw)
    if path.is_file():
        return _file_provider(path)
    if path.is_dir():
        return _directory_provider(path, battle_format)

    raise ValueError(
        f"Could not resolve team specification {raw!r}: not a file and not a directory"
    )


def _file_provider(path: Path) -> TeamProvider:
    team_str = path.read_text()
    return lambda: team_str


def _directory_provider(path: Path, battle_format: str) -> TeamProvider:
    repo = TeamRepo(filepath=str(path))
    return lambda: repo.sample_team(battle_format)


def _default_provider(battle_format: str) -> TeamProvider:
    repo = TeamRepo(filepath="data/teams")
    return lambda: repo.sample_team(battle_format)
