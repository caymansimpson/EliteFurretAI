# -*- coding: utf-8 -*-
"""Player specification parsing for the unified evaluation entry point.

A ``PlayerSpec`` is a small typed record produced from a single CLI
string. The string is either a path to a model checkpoint or one of a
fixed set of baseline names. ``parse_player_spec`` does the routing;
each resulting spec carries a ``factory`` closure that builds the
poke-env ``Player`` once per worker, given the per-worker dynamic
inputs (team provider, account configuration, server configuration).

The factory signature is uniform across baselines and model checkpoints
so callers don't switch on ``kind`` to construct players.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.players import (
    MaxDamagePlayer,
    SimpleModelPlayer,
    _create_vgc_bench_player,
)

PlayerKind = Literal["model", "baseline"]

# Canonical baseline names (snake_case). Aliases below map legacy
# evaluate.py spellings ("maxdamage", "shp") to the canonical form so
# old CLI invocations keep working.
_CANONICAL_BASELINES = (
    "max_damage",
    "max_base_power",
    "simple_heuristic",
    "vgc_bench",
    "random",
)

_BASELINE_ALIASES = {
    "maxdamage": "max_damage",
    "maxbasepower": "max_base_power",
    "shp": "simple_heuristic",
    "simpleheuristic": "simple_heuristic",
    "simpleheuristics": "simple_heuristic",
    "vgcbench": "vgc_bench",
}


# 3-letter user tag used as a username prefix. Showdown usernames have
# an 18-character cap (see ``_username`` in evaluate.py), so keep tags
# short and unambiguous across baselines.
_BASELINE_USER_TAG = {
    "max_damage": "MD",
    "max_base_power": "MBP",
    "simple_heuristic": "SHP",
    "vgc_bench": "VGB",
    "random": "RND",
}


PlayerFactory = Callable[
    [Callable[[], str], AccountConfiguration, ServerConfiguration, bool],
    Player,
]


@dataclass(frozen=True)
class PlayerSpec:
    """Parsed player specification.

    Fields:
        raw: original CLI string (for logging / serialization).
        kind: ``"model"`` if ``raw`` resolves to a checkpoint file,
            ``"baseline"`` if it resolves to a known baseline name.
        name: canonical display name. For ``kind="model"`` this is the
            checkpoint filename (without extension); for baselines it
            is the canonical snake_case name (e.g. ``simple_heuristic``).
        user_tag: short uppercase tag used as a username prefix.
        factory: callable that constructs a poke-env ``Player`` from
            (team_provider, account_configuration, server_configuration,
            accept_open_team_sheet). The factory captures any static
            configuration (model path, device, format, vgc-bench
            checkpoint, etc.) at parse time.
    """

    raw: str
    kind: PlayerKind
    name: str
    user_tag: str
    factory: PlayerFactory


def canonicalize_baseline(raw: str) -> Optional[str]:
    """Return the canonical baseline name for ``raw``, or ``None`` if not a baseline.

    Matches case-insensitively against canonical names and the alias
    table. Returning ``None`` is the signal to fall through to the
    path-based ``"model"`` branch in ``parse_player_spec``.
    """
    key = raw.strip().lower().replace("-", "_")
    if key in _CANONICAL_BASELINES:
        return key
    return _BASELINE_ALIASES.get(key)


def parse_player_spec(
    raw: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip",
) -> PlayerSpec:
    """Resolve ``raw`` into a ``PlayerSpec``.

    Resolution order:
        1. If ``raw`` is a path to an existing file → ``kind="model"``.
        2. Else if ``canonicalize_baseline(raw)`` returns a known name
           → ``kind="baseline"``.
        3. Else raise ``ValueError`` listing the accepted baseline names.

    The path check goes first because a model checkpoint named
    ``random.pt`` should resolve to a model, not the random baseline.
    """
    if os.path.isfile(raw):
        return _model_spec(raw, device=device, battle_format=battle_format)

    baseline = canonicalize_baseline(raw)
    if baseline is None:
        raise ValueError(
            f"Could not resolve player spec {raw!r}. Provide a checkpoint "
            f"path that exists, or one of: {sorted(_CANONICAL_BASELINES)} "
            f"(aliases also accepted: {sorted(_BASELINE_ALIASES)})."
        )

    return _baseline_spec(
        raw,
        baseline,
        device=device,
        battle_format=battle_format,
        vgc_bench_checkpoint_path=vgc_bench_checkpoint_path,
    )


def _model_spec(path: str, *, device: str, battle_format: str) -> PlayerSpec:
    name = os.path.splitext(os.path.basename(path))[0]

    def factory(
        team_provider: Callable[[], str],
        account_configuration: AccountConfiguration,
        server_configuration: ServerConfiguration,
        accept_open_team_sheet: bool,
    ) -> Player:
        return SimpleModelPlayer(
            model_path=path,
            device=device,
            battle_format=battle_format,
            probabilistic=False,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team_provider(),
            accept_open_team_sheet=accept_open_team_sheet,
        )

    return PlayerSpec(raw=path, kind="model", name=name, user_tag="MDL", factory=factory)


def _baseline_spec(
    raw: str,
    canonical: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str,
) -> PlayerSpec:
    user_tag = _BASELINE_USER_TAG[canonical]

    def factory(
        team_provider: Callable[[], str],
        account_configuration: AccountConfiguration,
        server_configuration: ServerConfiguration,
        accept_open_team_sheet: bool,
    ) -> Player:
        team = team_provider()
        common = dict(
            battle_format=battle_format,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )
        if canonical == "max_damage":
            return MaxDamagePlayer(**common)
        if canonical == "max_base_power":
            return MaxBasePowerPlayer(**common)
        if canonical == "simple_heuristic":
            return SimpleHeuristicsPlayer(**common)
        if canonical == "random":
            return RandomPlayer(**common)
        if canonical == "vgc_bench":
            return _create_vgc_bench_player(
                device=device,
                player_config=account_configuration,
                server_config=server_configuration,
                team=team,
                battle_format=battle_format,
                checkpoint_path=vgc_bench_checkpoint_path,
                accept_open_team_sheet=accept_open_team_sheet,
            )
        raise AssertionError(f"unreachable: unknown canonical baseline {canonical!r}")

    return PlayerSpec(
        raw=raw, kind="baseline", name=canonical, user_tag=user_tag, factory=factory
    )
