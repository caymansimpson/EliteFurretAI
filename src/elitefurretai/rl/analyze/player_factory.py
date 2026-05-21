# -*- coding: utf-8 -*-
"""Player specification parsing for the unified evaluation entry point.

A ``PlayerSpec`` is a small typed record produced from a single CLI
string. The string is either a path to a model checkpoint or one of a
fixed set of baseline names. ``parse_player_spec`` does the routing.

Three player kinds are supported:

* ``"model"`` — a trained RL checkpoint. ``build_player`` returns a
  ``SimpleModelPlayer`` that runs inference inline.
* ``"baseline"`` — a heuristic poke-env Player (``max_damage``,
  ``max_base_power``, ``simple_heuristic``, ``random``). ``build_player``
  returns a fresh instance per worker.
* ``"external"`` — vgc-bench, which can't run inline because its SB3
  policy requires a different poke-env vintage.
  ``launch_external_player`` spawns a subprocess in the
  ``../venv-vgcbench`` venv and exposes a username for the in-process
  side to challenge. The worker handles this asymmetry — see
  ``_run_worker`` in ``evaluate.py``.

PlayerSpec carries only pickleable data (no closures), so it can be
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

# vgc_bench is canonical but goes down the external path, not the
# baseline factory. Kept separate so parser logic is clear.
_EXTERNAL_BASELINES = ("vgc_bench",)

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
class PlayerSpec:
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


def parse_player_spec(
    raw: str,
    *,
    device: str,
    battle_format: str,
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip",
    vgc_bench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt",
    vgc_bench_python_executable: str = "/home/cayman/Repositories/venv-vgcbench/bin/python",
) -> PlayerSpec:
    """Resolve ``raw`` into a ``PlayerSpec``.

    Resolution order:
        1. If ``raw`` is a path to an existing file → ``kind="model"``.
        2. Else if ``canonicalize_baseline(raw)`` returns an external
           name → ``kind="external"``.
        3. Else if it returns a regular baseline name → ``kind="baseline"``.
        4. Else raise ``ValueError`` listing accepted names.

    The path check goes first because a model checkpoint named
    ``random.pt`` should resolve to a model, not the random baseline.
    """
    if os.path.isfile(raw):
        return _model_spec(raw, device=device, battle_format=battle_format)

    canonical = canonicalize_baseline(raw)
    if canonical is None:
        accepted = sorted(_CANONICAL_BASELINES + _EXTERNAL_BASELINES)
        raise ValueError(
            f"Could not resolve player spec {raw!r}. Provide a checkpoint "
            f"path that exists, or one of: {accepted} "
            f"(aliases also accepted: {sorted(_BASELINE_ALIASES)})."
        )

    if canonical in _EXTERNAL_BASELINES:
        return _external_spec(
            raw,
            canonical,
            battle_format=battle_format,
            checkpoint_path=vgc_bench_checkpoint_path,
            team_file=vgc_bench_team_file,
            python_executable=vgc_bench_python_executable,
        )

    return _baseline_spec(raw, canonical, battle_format=battle_format)


def _model_spec(path: str, *, device: str, battle_format: str) -> PlayerSpec:
    name = os.path.splitext(os.path.basename(path))[0]
    return PlayerSpec(
        raw=path,
        kind="model",
        name=name,
        user_tag="MDL",
        params={"path": path, "device": device, "battle_format": battle_format},
    )


def _baseline_spec(raw: str, canonical: str, *, battle_format: str) -> PlayerSpec:
    return PlayerSpec(
        raw=raw,
        kind="baseline",
        name=canonical,
        user_tag=_BASELINE_USER_TAG[canonical],
        params={"canonical": canonical, "battle_format": battle_format},
    )


def _external_spec(
    raw: str,
    canonical: str,
    *,
    battle_format: str,
    checkpoint_path: str,
    team_file: str,
    python_executable: str,
) -> PlayerSpec:
    assert canonical == "vgc_bench", (
        f"unreachable: unknown external baseline {canonical!r}"
    )
    return PlayerSpec(
        raw=raw,
        kind="external",
        name=canonical,
        user_tag=_BASELINE_USER_TAG[canonical],
        params={
            "checkpoint_path": checkpoint_path,
            "team_file": team_file,
            "python_executable": python_executable,
            "battle_format": battle_format,
        },
    )


def build_player(
    spec: PlayerSpec,
    *,
    team: str,
    account_configuration: AccountConfiguration,
    server_configuration: ServerConfiguration,
    accept_open_team_sheet: bool = False,
) -> Player:
    """Construct a poke-env ``Player`` from a ``PlayerSpec``.

    Handles ``kind="model"`` and ``kind="baseline"``. For ``kind="external"``
    use ``launch_external_player`` — there is no in-process Player.

    This is a top-level function (not a closure on the spec) so the spec
    itself remains pickleable and process-pool friendly.
    """
    if spec.kind == "model":
        return SimpleModelPlayer(
            model_path=spec.params["path"],
            device=spec.params["device"],
            battle_format=spec.params["battle_format"],
            probabilistic=False,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            team=team,
            accept_open_team_sheet=accept_open_team_sheet,
        )

    if spec.kind == "baseline":
        canonical = spec.params["canonical"]
        common = dict(
            battle_format=spec.params["battle_format"],
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
        f"build_player() does not handle kind={spec.kind!r}; "
        "use launch_external_player() for external opponents."
    )


def launch_external_player(spec: PlayerSpec, server_url: str) -> RunningExternal:
    """Spawn the external opponent subprocess and return a handle.

    Requires ``spec.kind == "external"``. Currently the only external
    opponent is ``vgc_bench``; this dispatch grows when more arrive.
    """
    if spec.kind != "external":
        raise ValueError(
            f"launch_external_player() requires kind='external', got {spec.kind!r}"
        )
    if spec.name != "vgc_bench":
        raise ValueError(
            f"launch_external_player() does not handle external opponent {spec.name!r}"
        )
    return _launch_vgc_bench_subprocess(
        server_url=server_url,
        battle_format=spec.params["battle_format"],
        checkpoint_path=spec.params["checkpoint_path"],
        team_file=spec.params["team_file"],
        python_executable=spec.params["python_executable"],
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
