"""VGCBenchManager — subprocess lifecycle for the external vgc-bench bot.

vgc-bench depends on the cameronangliss/poke-env fork (pinned to commit
``b3956ae58``, reporting version 0.15.0) whose VGC enums produce a 764-wide
observation, while EFA runs its own poke-env 0.15.0. The two cannot coexist in
one interpreter, so vgc-bench runs as a subprocess in its own venv
(``../venv-vgcbench-bcsp/``). This module owns the EFA-side subprocess
lifecycle. The subprocess entry point lives at ``agents/_vgcbench_subprocess.py``.
EFA-side code interacts with vgc-bench only by Showdown username.

Also exports three in-process helpers used by the eval CLI when running under a
venv whose poke-env produces the same observation width the checkpoint expects
(``_create_vgc_bench_player``, ``_temporary_cwd``, ``_resolve_vgc_bench_root``).
These are NOT safe to call from EFA's training process; see the comment
above ``_create_vgc_bench_player``.

Moved here from ``rl/players.py`` on 2026-05-19 as part of the agents/
directory reorganization (see planning/stage2/2026-05-19-09-30-agents-directory-reorg.md).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
    cast,
)

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.config import RNaDConfig

logger = logging.getLogger(__name__)

# `_create_vgc_bench_player` builds the embedder from whatever poke-env is
# installed in the *calling* interpreter. If that poke-env's enum/dex widths
# don't match the checkpoint the embedder feeds a wrong-width observation and
# SB3.PPO.load (or the first forward pass) fails on a state_dict size mismatch
# — e.g. "copying a param with shape [256, 764] ... current model [256, 762]".
# A poke-env *version string* no longer discriminates: EFA's main venv and the
# vgc-bench poke-env fork both report 0.15.0 yet produce different widths. So
# we validate the actual invariant at call time — the runtime per-Pokemon
# observation width must equal the checkpoint's `pokemon_proj` input — instead
# of guessing from a version number.

# `_create_vgc_bench_player` is the legacy in-process fallback for evaluation
# matchups where head-to-head play inside a single process is convenient. It is
# *only* safe to call from a Python interpreter whose poke-env produces the same
# observation width the checkpoint was trained with (currently the
# cameronangliss/poke-env fork installed in `../venv-vgcbench-bcsp`); calling it
# from EFA's training venv produces a broken PolicyPlayer. The training path
# (`VGCBenchManager` → subprocess) sidesteps this by running under that venv.

# Cached vgc-bench policies keyed by (checkpoint_path, device).
# Loading a stable_baselines3 PPO checkpoint is slow (hundreds of ms); cache
# them so swapping vgc-bench opponents in/out of the curriculum is cheap.
_VGC_BENCH_POLICY_CACHE: Dict[Tuple[str, str], Any] = {}


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_vgc_bench_root() -> Optional[Path]:
    spec = importlib.util.find_spec("vgc_bench")
    if spec is None or not spec.submodule_search_locations:
        return None

    package_path = Path(next(iter(spec.submodule_search_locations))).resolve()
    return package_path.parent


def _create_vgc_bench_player(
    device: str,
    player_config: AccountConfiguration,
    server_config: ServerConfiguration,
    team: str,
    battle_format: str = "gen9vgc2024regg",
    checkpoint_path: str = "data/models/vgc-bench-bcsp-reg_all-seed1-98304000.zip",
    accept_open_team_sheet: bool = True,
) -> Player:
    """Construct an in-process vgc-bench PolicyPlayer.

    **Only callable from a venv whose poke-env produces the same
    observation width the checkpoint was trained with** (currently the
    cameronangliss/poke-env fork in ``../venv-vgcbench-bcsp``). From EFA's
    main venv use ``VGCBenchManager`` to launch the subprocess flow
    instead — this function will refuse to run on a width mismatch with
    an actionable error.

    Fails fast in two cases the in-process flow could otherwise hit:

    1. Observation-width mismatch (the common failure when called from a
       venv whose poke-env differs from the one the checkpoint was trained
       against) — surfaced before the first forward pass, since the
       resulting state_dict size error is opaque ("size mismatch for
       features_extractor.pokemon_proj.weight: copying a param with shape
       torch.Size([256, 764]) ... current model torch.Size([256, 762])").
       Checked after load by comparing the runtime per-Pokemon width
       (``utils.chunk_obs_len`` + embeddings) to the checkpoint's
       ``pokemon_proj`` input.
    2. Missing checkpoint file — checked before ``_temporary_cwd``
       because SB3's PPO.load doubles the ``.zip`` suffix when its
       fallback search fires, producing a confusing
       ``...98304000.zip.zip`` error.

    The checkpoint path is also resolved to absolute *before* entering
    ``_temporary_cwd(vgc_bench_root)`` — SB3 treats the path as
    relative to the current working directory, so a relative path
    plus a chdir into ``vgc_bench_root`` gives the ``.zip.zip``
    failure even when the file exists.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"vgc-bench checkpoint not found: {checkpoint_path}")

    # Absolute path BEFORE _temporary_cwd. See docstring for why.
    checkpoint_abspath = os.path.abspath(checkpoint_path)
    cache_key = (checkpoint_abspath, device)

    ppo_module = importlib.import_module("stable_baselines3")
    ppo_cls = getattr(ppo_module, "PPO")

    vgc_bench_root = _resolve_vgc_bench_root()
    if vgc_bench_root is None:
        raise ModuleNotFoundError("Could not resolve vgc_bench package path")

    with _temporary_cwd(vgc_bench_root):
        policy_player_module = importlib.import_module("vgc_bench.src.policy_player")
        policy_player_cls = getattr(policy_player_module, "PolicyPlayer")

        policy = _VGC_BENCH_POLICY_CACHE.get(cache_key)
        if policy is None:
            policy = ppo_cls.load(checkpoint_abspath, device=device).policy
            _VGC_BENCH_POLICY_CACHE[cache_key] = policy

        # Validate the real invariant: the per-Pokemon observation width this
        # interpreter's poke-env produces must equal what the checkpoint expects.
        # A version string can't catch this — EFA's main venv and the vgc-bench
        # poke-env fork both report 0.15.0 but yield different widths.
        utils_module = importlib.import_module("vgc_bench.src.utils")
        feature_extractor = policy.features_extractor
        runtime_width = utils_module.chunk_obs_len + 6 * (feature_extractor.embed_len - 1)
        checkpoint_width = feature_extractor.pokemon_proj.in_features
        if runtime_width != checkpoint_width:
            raise RuntimeError(
                f"vgc-bench observation-width mismatch: this interpreter's "
                f"poke-env produces a {runtime_width}-wide per-Pokemon "
                f"observation but checkpoint {checkpoint_path!r} expects "
                f"{checkpoint_width}. The installed poke-env does not match the "
                f"one this checkpoint was trained against. Run from the venv "
                f"whose cameronangliss/poke-env fork yields {checkpoint_width} "
                f"(currently ../venv-vgcbench-bcsp), or from EFA's main venv use "
                f"`VGCBenchManager` to launch the subprocess flow instead "
                f"(see analyze/player_factory.py for the wiring)."
            )

    player = policy_player_cls(
        policy=policy,
        battle_format=battle_format,
        account_configuration=player_config,
        server_configuration=server_config,
        accept_open_team_sheet=accept_open_team_sheet,
        team=team,
    )
    return cast(Player, player)


class VGCBenchManager:
    """Launcher and proxy for external vgc-bench bots.

    vgc-bench needs the cameronangliss/poke-env fork (a 0.15.0 build whose
    VGC enums yield a 764-wide observation), which can't share an interpreter
    with EFA's own poke-env. To isolate them, vgc-bench is launched as a
    subprocess in its own venv (``../venv-vgcbench-bcsp``). This manager owns
    that subprocess lifecycle. EFA-side code only interacts with vgc-bench by
    Showdown username — workers `/challenge` `manager.usernames[i]` like any
    other opponent.

    Usage
    -----
        manager = VGCBenchManager(config, server_ports)
        manager.launch()         # spawns subprocesses, populates .usernames
        ...                       # training loop runs
        manager.shutdown()        # terminates and closes log handles

    Per-worker resolution (which subset of usernames *this* worker can
    reach) currently lives in `engine/vgc_environment.py:setup()`. The
    static helper `derive_username()` and class constant
    `RUNNER_SERVER_INDEX` are exposed here so that file can resolve the
    same values without re-deriving them.
    """

    # Path to the standalone subprocess entry point, relative to the
    # repository root (subprocess inherits the trainer's CWD).
    SUBPROCESS_SCRIPT: ClassVar[str] = "src/elitefurretai/agents/_vgcbench_subprocess.py"

    # Base usernames each runner logs into Showdown as. One subprocess
    # per name. The port is appended (`{name}_{port}`) when num_servers
    # > 1 so concurrent runs on different ports don't collide.
    USERNAMES: ClassVar[List[str]] = ["VGCBENCH"]

    # How many challenges the runner accepts before exiting. Set to a
    # number larger than any plausible training run.
    N_CHALLENGES: ClassVar[int] = 1_000_000

    # Seconds the runner waits for the Showdown server's TCP port to
    # come up before giving up. Larger than launch_showdown_servers'
    # `time.sleep(2)` warmup to allow for slow boots.
    WAIT_FOR_SERVER_TIMEOUT_S: ClassVar[float] = 180.0

    # Directory for runner stdout/stderr. Created on launch.
    LOG_DIR: ClassVar[str] = "data/logs/vgcbench_runners"
    LOG_TO_FILES: ClassVar[bool] = True

    # accept_open_team_sheet must match the main agent's
    # RLTrajectoryPlayer (currently False); a mismatched handshake
    # drops battles.
    ACCEPT_OPEN_TEAM_SHEET: ClassVar[bool] = False

    # Seconds a worker waits after env.setup() before sending its first
    # challenge, giving the subprocess time to log in.
    STARTUP_WAIT_S: ClassVar[float] = 10.0

    # Only the first Showdown server hosts a runner. Each SB3
    # PolicyPlayer runner costs ~1.2 GB resident PSS; launching one per
    # server (the pre-2026-05-16 layout) cost ~4.8 GB just for an
    # opponent that plays ~20% of battles. Workers on other servers
    # detect this and zero out their local vgc_bench_baseline weight.
    # See planning/stage2/2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md.
    RUNNER_SERVER_INDEX: ClassVar[int] = 0

    def __init__(self, config: RNaDConfig, server_ports: List[int]) -> None:
        self._config = config
        self._server_ports = server_ports
        self._processes: List[subprocess.Popen] = []
        self._log_files: List[TextIO] = []
        self._usernames: List[str] = []

    @staticmethod
    def derive_username(base_username: str, server_port: int) -> str:
        """Generate a server-scoped runner username (Showdown max length is 18)."""
        suffix = f"_{server_port}"
        max_base_len = max(1, 18 - len(suffix))
        return f"{base_username[:max_base_len]}{suffix}"

    @classmethod
    def should_suffix_port(cls, num_servers: int) -> bool:
        """Whether the launcher appends `_{port}` to usernames.

        Single source of truth for the suffix condition. Both `launch`
        and worker-side username resolution (vgc_environment.py) must
        agree exactly — drift between them produces "user not found"
        stalls (see planning/stage2/2026-05-16-08-13-...).
        """
        return num_servers > 1

    @property
    def usernames(self) -> List[str]:
        """Resolved usernames the subprocess(es) logged into Showdown as.

        Empty until `launch()` has been called.
        """
        return list(self._usernames)

    def launch(self) -> List[str]:
        """Spawn subprocess(es) and return the list of usernames they logged in as."""
        if not self.USERNAMES:
            return []

        cur = self._config.curriculum
        assert cur.external_vgcbench_python_executable is not None, (
            "external_vgcbench_python_executable must be set when launching VGCBenchManager"
        )

        if len(cur.battle_formats) > 1:
            logger.warning(
                "VGCBench v1 is single-format (bound to primary_format=%s). "
                "Off-format pairs (%s) cannot challenge VGCBench and will hit "
                "Showdown |formaterror|. Multi-format VGCBench v2 will replace this.",
                cur.primary_format,
                sorted(set(cur.battle_formats) - {cur.primary_format}),
            )

        if self.LOG_TO_FILES:
            os.makedirs(self.LOG_DIR, exist_ok=True)

        append_port = self.should_suffix_port(len(self._server_ports))
        runner_port = self._server_ports[self.RUNNER_SERVER_INDEX]
        usernames: List[str] = []

        for username in self.USERNAMES:
            actual_username = (
                self.derive_username(username, runner_port) if append_port else username
            )
            sanitized = actual_username.replace("/", "_")

            log_path = "<disabled>"
            log_handle: Union[TextIO, int]
            if self.LOG_TO_FILES:
                log_path = os.path.join(
                    self.LOG_DIR,
                    f"runner_{sanitized}_{runner_port}.log",
                )
                log_handle = open(log_path, "a", encoding="utf-8")
                self._log_files.append(log_handle)
            else:
                log_handle = subprocess.DEVNULL

            command = [
                cur.external_vgcbench_python_executable,
                self.SUBPROCESS_SCRIPT,
                "--username",
                actual_username,
                "--server",
                f"localhost:{runner_port}",
                "--battle-format",
                # VGCBench v1 is single-format; bind to primary_format. Off-format
                # challenges hit Showdown |formaterror| and time out. Multi-format
                # VGCBench v2 will replace this binding.
                cur.primary_format,
                "--checkpoint-path",
                cur.vgc_bench_checkpoint_path,
                "--team-file",
                cur.external_vgcbench_team_file,
                "--n-challenges",
                str(self.N_CHALLENGES),
                "--wait-for-server-timeout",
                str(self.WAIT_FOR_SERVER_TIMEOUT_S),
            ]
            if self.ACCEPT_OPEN_TEAM_SHEET:
                command.append("--accept-open-team-sheet")

            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._processes.append(process)
            usernames.append(actual_username)
            print(
                "✓ Launched external vgc-bench runner "
                f"'{actual_username}' on localhost:{runner_port} (PID: {process.pid}) "
                f"log={log_path}"
            )

        self._usernames = usernames
        return usernames

    def shutdown(self) -> None:
        """Terminate subprocess(es) and close log files."""
        for process in self._processes:
            if process.poll() is not None:
                continue
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass

        for log_handle in self._log_files:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass

        self._processes.clear()
        self._log_files.clear()
