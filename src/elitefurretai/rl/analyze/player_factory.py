# -*- coding: utf-8 -*-
"""Player specification parsing for the unified evaluation entry point.

A ``PlayerSpecification`` is a small typed record produced from a single CLI
string. The string is either a path to a model checkpoint or one of a
fixed set of baseline names. ``parse_player_specification`` does the routing.

Three player kinds are supported:

* ``"model"`` — a trained RL checkpoint. ``build_player`` returns a
  ``SimpleModelPlayer`` that runs inference inline.
* ``"baseline"`` — a heuristic poke-env Player (``max_damage``,
  ``max_base_power``, ``simple_heuristic``, ``random``). ``build_player``
  returns a fresh instance per worker.
* ``"external"`` — vgc-bench, which can't run inline because its SB3
  policy needs the cameronangliss/poke-env fork (a 0.15.0 build whose VGC
  enums yield a 764-wide observation) that can't share an interpreter with
  EFA's own poke-env. ``launch_external_player`` spawns a subprocess in the
  ``../venv-vgcbench-bcsp`` venv and exposes a username for the in-process
  side to challenge. The worker handles this asymmetry — see
  ``_run_worker`` in ``evaluate.py``.

PlayerSpecification carries only pickleable data (no closures), so it can be
shipped across process boundaries — required by the ProcessPoolExecutor
fan-out in ``run_eval_parallel``.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional

from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.baselines import SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.agents.foulplay_manager import FoulPlayManager
from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.agents.vgcbench_manager import VGCBenchManager

PlayerKind = Literal["model", "baseline", "external"]

# Canonical baseline names (snake_case). Aliases below map legacy
# evaluate.py spellings ("maxdamage", "shp") to the canonical form so
# old CLI invocations keep working.
_CANONICAL_BASELINES = (
    "max_damage",
    "max_base_power",
    "simple_heuristic",
    "random",
)

# vgc_bench and foul_play go down the external path (subprocess in a
# separate venv), not the in-process baseline factory. Kept separate
# so parser logic is clear.
_EXTERNAL_BASELINES = ("vgc_bench", "foul_play")

_BASELINE_ALIASES = {
    "maxdamage": "max_damage",
    "maxbasepower": "max_base_power",
    "shp": "simple_heuristic",
    "simpleheuristic": "simple_heuristic",
    "simpleheuristics": "simple_heuristic",
    "vgcbench": "vgc_bench",
    "foulplay": "foul_play",
}


# 3-letter user tag used as a username prefix. Showdown usernames have
# an 18-character cap (see ``_username`` in evaluate.py), so keep tags
# short and unambiguous across baselines.
_BASELINE_USER_TAG = {
    "max_damage": "MD",
    "max_base_power": "MBP",
    "simple_heuristic": "SHP",
    "vgc_bench": "VGB",
    "foul_play": "FP",
    "random": "RND",
}


@dataclass
class RunningExternal:
    """Handle to a running external opponent subprocess.

    ``username`` is the Showdown username the subprocess logged in as —
    the other player ``send_challenges(username, n)`` against it.
    ``shutdown`` terminates the subprocess and closes any log handles.
    """

    username: str
    shutdown: Callable[[], None]


@dataclass(frozen=True)
class PlayerSpecification:
    """Parsed player specification — pure data, no closures.

    Fields:
        raw: original CLI string (for logging / serialization).
        kind: ``"model"``, ``"baseline"``, or ``"external"``.
        name: canonical display name. For ``kind="model"`` this is the
            checkpoint filename (without extension); for baselines /
            external it is the canonical snake_case name.
        user_tag: short uppercase tag used as a username prefix.
        params: kind-specific config used by ``build_player`` /
            ``launch_external_player`` to construct the Player. Only
            primitive types — must be picklable.

    Per-kind ``params`` schema:

    * ``"model"`` → ``path`` (str), ``device`` (str), ``battle_format`` (str)
    * ``"baseline"`` → ``canonical`` (str), ``battle_format`` (str)
    * ``"external"`` → ``checkpoint_path`` (str), ``team_file`` (str),
      ``python_executable`` (str), ``battle_format`` (str)
    """

    raw: str
    kind: PlayerKind
    name: str
    user_tag: str
    params: Mapping[str, Any]


def canonicalize_baseline(raw: str) -> Optional[str]:
    """Return the canonical baseline name for ``raw``, or ``None`` if not a baseline.

    Includes external baselines (``vgc_bench``) so callers can decide
    routing afterwards.
    """
    key = raw.strip().lower().replace("-", "_")
    if key in _CANONICAL_BASELINES or key in _EXTERNAL_BASELINES:
        return key
    return _BASELINE_ALIASES.get(key)


def parse_player_specification(
    raw: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
    vgc_bench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt",
    vgc_bench_python_executable: str = "/home/cayman/Repositories/venv-vgcbench-bcsp/bin/python",
    foul_play_python_executable: str = "/home/cayman/Repositories/venv-foulplay/bin/python",
    foul_play_team_pool_path: str = "data/teams/gen9vgc2024regg/constrained",
    foul_play_search_time_ms: int = 750,
    foul_play_parallelism: int = 4,
) -> PlayerSpecification:
    """Resolve ``raw`` into a ``PlayerSpecification``.

    Resolution order:
        1. If ``raw`` is a path to an existing file → ``kind="model"``.
        2. Else if ``canonicalize_baseline(raw)`` returns an external
           name → ``kind="external"``.
        3. Else if it returns a regular baseline name → ``kind="baseline"``.
        4. Else raise ``ValueError`` listing accepted names.

    The path check goes first because a model checkpoint named
    ``random.pt`` should resolve to a model, not the random baseline.

    The ``foul_play_*`` kwargs supply the subprocess-config the
    eval-side launcher needs for the FoulPlay path (search-time,
    parallelism, venv interpreter, team pool). They have sensible
    defaults so simple CLI use (``--player2 foul_play``) works without
    extra flags.
    """
    if os.path.isfile(raw):
        return _model_specification(raw, device=device, battle_format=battle_format)

    canonical = canonicalize_baseline(raw)
    if canonical is None:
        accepted = sorted(_CANONICAL_BASELINES + _EXTERNAL_BASELINES)
        raise ValueError(
            f"Could not resolve player specification {raw!r}. Provide a checkpoint "
            f"path that exists, or one of: {accepted} "
            f"(aliases also accepted: {sorted(_BASELINE_ALIASES)})."
        )

    if canonical in _EXTERNAL_BASELINES:
        return _external_specification(
            raw,
            canonical,
            battle_format=battle_format,
            vgc_bench_checkpoint_path=vgc_bench_checkpoint_path,
            vgc_bench_team_file=vgc_bench_team_file,
            vgc_bench_python_executable=vgc_bench_python_executable,
            foul_play_python_executable=foul_play_python_executable,
            foul_play_team_pool_path=foul_play_team_pool_path,
            foul_play_search_time_ms=foul_play_search_time_ms,
            foul_play_parallelism=foul_play_parallelism,
        )

    return _baseline_specification(raw, canonical, battle_format=battle_format)


def _model_specification(
    path: str, *, device: str, battle_format: str
) -> PlayerSpecification:
    name = os.path.splitext(os.path.basename(path))[0]
    return PlayerSpecification(
        raw=path,
        kind="model",
        name=name,
        user_tag="MDL",
        params={"path": path, "device": device, "battle_format": battle_format},
    )


def _baseline_specification(
    raw: str, canonical: str, *, battle_format: str
) -> PlayerSpecification:
    return PlayerSpecification(
        raw=raw,
        kind="baseline",
        name=canonical,
        user_tag=_BASELINE_USER_TAG[canonical],
        params={"canonical": canonical, "battle_format": battle_format},
    )


def _external_specification(
    raw: str,
    canonical: str,
    *,
    battle_format: str,
    vgc_bench_checkpoint_path: str,
    vgc_bench_team_file: str,
    vgc_bench_python_executable: str,
    foul_play_python_executable: str,
    foul_play_team_pool_path: str,
    foul_play_search_time_ms: int,
    foul_play_parallelism: int,
) -> PlayerSpecification:
    """Build a ``kind="external"`` spec for one of the registered external baselines.

    Each external baseline carries its own params shape — there's no
    shared schema because their subprocess interfaces differ
    (vgc_bench loads an SB3 checkpoint + plays one team; foul_play
    runs a search bot at a configured time-budget + samples from a
    team pool).
    """
    if canonical == "vgc_bench":
        return PlayerSpecification(
            raw=raw,
            kind="external",
            name=canonical,
            user_tag=_BASELINE_USER_TAG[canonical],
            params={
                "checkpoint_path": vgc_bench_checkpoint_path,
                "team_file": vgc_bench_team_file,
                "python_executable": vgc_bench_python_executable,
                "battle_format": battle_format,
            },
        )
    if canonical == "foul_play":
        return PlayerSpecification(
            raw=raw,
            kind="external",
            name=canonical,
            user_tag=_BASELINE_USER_TAG[canonical],
            params={
                "python_executable": foul_play_python_executable,
                "team_pool_path": foul_play_team_pool_path,
                "search_time_ms": foul_play_search_time_ms,
                "parallelism": foul_play_parallelism,
                "battle_format": battle_format,
            },
        )
    raise AssertionError(f"unreachable: unknown external baseline {canonical!r}")


def build_player(
    specification: PlayerSpecification,
    *,
    team: str,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    accept_open_team_sheet: bool = False,
) -> Player:
    """Construct a poke-env ``Player`` from a ``PlayerSpecification``.

    Handles ``kind="model"`` and ``kind="baseline"``. For ``kind="external"``
    use ``launch_external_player`` — there is no in-process Player.

    This is a top-level function (not a closure on the specification) so the specification
    itself remains pickleable and process-pool friendly.
    """
    if specification.kind == "model":
        return SimpleModelPlayer(
            model_path=specification.params["path"],
            device=specification.params["device"],
            battle_format=specification.params["battle_format"],
            probabilistic=False,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )

    if specification.kind == "baseline":
        canonical = specification.params["canonical"]
        common = dict(
            battle_format=specification.params["battle_format"],
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )
        if canonical == "max_damage":
            # Deterministic argmax — evaluation needs a fixed-policy baseline,
            # not the curriculum's softmax-sampled default (temperature=0.5).
            return MaxDamagePlayer(temperature=0.0, **common)
        if canonical == "max_base_power":
            return MaxBasePowerPlayer(**common)
        if canonical == "simple_heuristic":
            return SimpleHeuristicsPlayer(**common)
        if canonical == "random":
            return RandomPlayer(**common)
        raise AssertionError(f"unreachable: unknown canonical baseline {canonical!r}")

    raise ValueError(
        f"build_player() does not handle kind={specification.kind!r}; "
        "use launch_external_player() for external opponents."
    )


def launch_external_player(
    specification: PlayerSpecification, server_url: str
) -> RunningExternal:
    """Spawn the external opponent subprocess and return a handle.

    Requires ``specification.kind == "external"``. Dispatches by
    ``specification.name`` to one of the registered external launchers
    (currently ``vgc_bench`` and ``foul_play``).
    """
    if specification.kind != "external":
        raise ValueError(
            f"launch_external_player() requires kind='external', got {specification.kind!r}"
        )
    if specification.name == "vgc_bench":
        return _launch_vgc_bench_subprocess(
            server_url=server_url,
            battle_format=specification.params["battle_format"],
            checkpoint_path=specification.params["checkpoint_path"],
            team_file=specification.params["team_file"],
            python_executable=specification.params["python_executable"],
        )
    if specification.name == "foul_play":
        return _launch_foulplay_subprocess(
            server_url=server_url,
            battle_format=specification.params["battle_format"],
            python_executable=specification.params["python_executable"],
            team_pool_path=specification.params["team_pool_path"],
            search_time_ms=specification.params["search_time_ms"],
            parallelism=specification.params["parallelism"],
        )
    raise ValueError(
        f"launch_external_player() does not handle external opponent {specification.name!r}"
    )


def _launch_vgc_bench_subprocess(
    *,
    server_url: str,
    battle_format: str,
    checkpoint_path: str,
    team_file: str,
    python_executable: str,
) -> RunningExternal:
    """Spawn the vgc-bench subprocess and return a handle.

    Mirrors ``VGCBenchManager.launch`` (under ``src/elitefurretai/agents``)
    but takes raw parameters instead of an ``RNaDConfig`` so the eval
    script doesn't have to construct training config to use vgc_bench.
    The subprocess args are kept identical so any future fix to
    ``_vgcbench_subprocess.py`` applies uniformly.

    Blocks for ``STARTUP_WAIT_S`` seconds before returning so the
    subprocess has time to log into Showdown — without this, a
    challenge sent immediately after launch gets "user not found".
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"vgc-bench checkpoint not found: {checkpoint_path}")
    if not os.path.exists(team_file):
        raise FileNotFoundError(f"vgc-bench team file not found: {team_file}")
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"vgc-bench venv python not found: {python_executable}")

    # Server URL is `host:port`; the subprocess wants the same shape.
    port = int(server_url.rsplit(":", 1)[1])
    base_username = VGCBenchManager.USERNAMES[0]  # "VGCBENCH"
    username = VGCBenchManager.derive_username(base_username, port)

    log_dir = "data/logs/vgcbench_runners_eval"
    os.makedirs(log_dir, exist_ok=True)
    sanitized = username.replace("/", "_")
    log_path = os.path.join(log_dir, f"runner_{sanitized}_{port}.log")
    log_handle = open(log_path, "a", encoding="utf-8")

    command = [
        python_executable,
        VGCBenchManager.SUBPROCESS_SCRIPT,
        "--username",
        username,
        "--server",
        f"localhost:{port}",
        "--battle-format",
        battle_format,
        "--checkpoint-path",
        checkpoint_path,
        "--team-file",
        team_file,
        "--n-challenges",
        str(VGCBenchManager.N_CHALLENGES),
        "--wait-for-server-timeout",
        str(VGCBenchManager.WAIT_FOR_SERVER_TIMEOUT_S),
    ]
    if VGCBenchManager.ACCEPT_OPEN_TEAM_SHEET:
        command.append("--accept-open-team-sheet")

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(
        f"✓ Launched eval vgc-bench runner '{username}' on localhost:{port} "
        f"(PID: {process.pid}) log={log_path}"
    )

    # Wait for Showdown login. VGCBenchManager uses STARTUP_WAIT_S=10s
    # because the subprocess has to import poke-env, load the SB3
    # policy, and complete a websocket handshake. Anything less risks
    # the first /challenge landing before the user exists.
    time.sleep(VGCBenchManager.STARTUP_WAIT_S)

    def shutdown() -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
        try:
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass

    return RunningExternal(username=username, shutdown=shutdown)


def _launch_foulplay_subprocess(
    *,
    server_url: str,
    battle_format: str,
    python_executable: str,
    team_pool_path: str,
    search_time_ms: int,
    parallelism: int,
) -> RunningExternal:
    """Spawn the foul-play-doubles subprocess and return a handle.

    Single-shot CLI variant of ``FoulPlayManager.launch``: takes raw
    parameters instead of a config object so the eval script doesn't
    have to construct training config to use foul_play. The subprocess
    args are kept identical to ``FoulPlayManager`` so any future fix to
    ``_foulplay_subprocess.py`` applies uniformly.

    Blocks for ``STARTUP_WAIT_S`` seconds before returning so the
    subprocess has time to log into Showdown — without this, the first
    challenge sent immediately after launch gets "user not found".

    ``--n-challenges`` is set to a large constant here because the
    eval-side cell loop controls the actual number of challenges issued
    (matching the long-lived FoulPlay-as-baseline pattern). The
    training-loop variant of FoulPlay eval is bounded by
    ``FoulplayEvalConfig.n_battles_per_format`` and goes through
    ``FoulPlayManager`` directly, not this helper.
    """
    if not os.path.exists(python_executable):
        raise FileNotFoundError(f"foul-play venv python not found: {python_executable}")
    if not os.path.isdir(team_pool_path):
        raise FileNotFoundError(f"foul-play team pool not found: {team_pool_path}")

    port = int(server_url.rsplit(":", 1)[1])
    base_username = FoulPlayManager.USERNAMES[0]  # "FOULPLAY"
    username = FoulPlayManager.derive_username(base_username, port)

    log_dir = "data/logs/foulplay_runners_eval"
    os.makedirs(log_dir, exist_ok=True)
    sanitized = username.replace("/", "_")
    log_path = os.path.join(log_dir, f"runner_{sanitized}_{port}.log")
    log_handle = open(log_path, "a", encoding="utf-8")

    command = [
        python_executable,
        FoulPlayManager.SUBPROCESS_SCRIPT,
        "--username",
        username,
        "--server",
        f"localhost:{port}",
        "--battle-format",
        battle_format,
        "--n-challenges",
        # Large constant — the cell loop decides how many to issue.
        str(1_000_000),
        "--team-list-dir",
        team_pool_path,
        "--search-time-ms",
        str(search_time_ms),
        "--parallelism",
        str(parallelism),
        "--wait-for-server-timeout",
        str(FoulPlayManager.WAIT_FOR_SERVER_TIMEOUT_S),
    ]
    if FoulPlayManager.ACCEPT_OPEN_TEAM_SHEET:
        command.append("--accept-open-team-sheet")

    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(
        f"✓ Launched eval foul-play runner '{username}' on localhost:{port} "
        f"(PID: {process.pid}) log={log_path}"
    )

    # Wait for Showdown login. FoulPlay imports poke-engine-doubles
    # (a Rust extension), poke_env 0.11, and completes a websocket
    # handshake. STARTUP_WAIT_S mirrors VGCBench's 10 s; less and the
    # first /challenge can hit a non-existent user.
    time.sleep(FoulPlayManager.STARTUP_WAIT_S)

    def shutdown() -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass
        try:
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass

    return RunningExternal(username=username, shutdown=shutdown)
