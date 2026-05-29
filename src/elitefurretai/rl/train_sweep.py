# -*- coding: utf-8 -*-
"""W&B Bayesian sweep harness for RL training.

Each sweep agent run spawns ``python -m elitefurretai.rl.train`` as a
subprocess with ``WANDB_RUN_ID`` inherited via env so the subprocess
reattaches to the sweep's wandb run. Subprocess isolation guarantees
that DataLoader workers, Showdown servers, and external eval
subprocesses from one run cannot leak into the next.

Three-layer cleanup contract: layer 1 (per-eval, in baseline_eval),
layer 2 (per-train-process, in train.main), layer 3 (per-sweep-agent,
here). See planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md.
"""

from __future__ import annotations

import copy
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

import yaml

logger = logging.getLogger(__name__)


# Filled in by main() before wandb.agent dispatches to sweep_train().
BASE_CONFIG_PATH: str = ""
EVAL_OVERRIDES: Dict[str, Any] = {}
PATCHED_CONFIG_DIR: str = ""


def load_sweep_config(path: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Parse the sweep YAML into (wandb_sweep_dict, base_config_path,
    eval_overrides).

    Required top-level keys: ``base_config``, ``sweep``. ``eval_overrides``
    is optional and defaults to an empty dict.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"sweep YAML {path!r} must be a mapping at the top level")
    if "base_config" not in data:
        raise KeyError("sweep YAML missing required key 'base_config'")
    if "sweep" not in data:
        raise KeyError("sweep YAML missing required key 'sweep'")
    return data["sweep"], data["base_config"], data.get("eval_overrides", {})


def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Patch ``dotted_key`` into the nested ``target`` dict, creating
    intermediate dicts as needed. Existing values at intermediate paths
    that are not dicts are replaced with dicts.
    """
    parts = dotted_key.split(".")
    cur = target
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _write_patched_config(
    base_config_path: str,
    eval_overrides: Dict[str, Any],
    sweep_params: Dict[str, Any],
    run_name: str,
    out_dir: str,
) -> str:
    """Write a patched config YAML for one sweep run.

    Loads ``base_config_path``, applies ``eval_overrides`` first, then
    ``sweep_params`` on top (sweep wins on key collision). Writes to
    ``<out_dir>/<run_name>.yaml`` and returns the path.
    """
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)
    for k, v in eval_overrides.items():
        _set_dotted(cfg, k, v)
    for k, v in sweep_params.items():
        _set_dotted(cfg, k, v)
    out_path = os.path.join(out_dir, f"{run_name}.yaml")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_path


def _terminate_with_grace(proc: subprocess.Popen, timeout: int) -> None:
    """SIGTERM → wait up to timeout → SIGKILL → wait. Safe to call when
    proc already exited.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Process %d ignored SIGKILL; giving up", proc.pid)


def _sweep_orphan_cleanup(run_name: str) -> None:
    """pgrep fallback at the sweep-agent layer: catch showdown / vgcbench
    / foulplay subprocesses whose command line still mentions this
    sweep run's tag.
    """
    import shutil

    if shutil.which("pgrep") is None:
        return
    for pat in (
        f"showdown.*{run_name}",
        f"vgcbench.*{run_name}",
        f"foulplay.*{run_name}",
    ):
        try:
            out = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True)
            for line in out.stdout.strip().splitlines():
                pid_str = line.split(None, 1)[0]
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.warning(
                        "Sweep orphan cleanup: SIGTERM pid=%d (matched %r)",
                        pid,
                        pat,
                    )
                except ProcessLookupError:
                    pass
        except Exception:
            logger.exception("Sweep orphan cleanup failed for pattern %r", pat)


CLEANUP_TIMEOUT_S = 30


def sweep_train() -> None:
    """Called by wandb.agent per sweep run. wandb.init has already
    happened (the agent does it before calling this function), so the
    run name and config are available on wandb.run.

    Spawns ``python -m elitefurretai.rl.train`` as a subprocess with the
    sweep run's wandb run id in the env. The subprocess's own
    ``wandb.init()`` reattaches to that run, so metrics it logs land
    under the correct sweep run.
    """
    import wandb

    sampled = dict(wandb.config)
    run_name = (
        wandb.run.name if wandb.run and wandb.run.name else f"sweep-{int(time.time())}"
    )
    cfg_path = _write_patched_config(
        base_config_path=BASE_CONFIG_PATH,
        eval_overrides=EVAL_OVERRIDES,
        sweep_params=sampled,
        run_name=run_name,
        out_dir=PATCHED_CONFIG_DIR,
    )

    env = {**os.environ}
    if wandb.run is not None:
        env["WANDB_RUN_ID"] = wandb.run.id
        env["WANDB_RESUME"] = "must"
    env["EFAI_SWEEP_RUN_TAG"] = run_name

    logger.info("Spawning sweep run %s with config %s", run_name, cfg_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "elitefurretai.rl.train", "--config", cfg_path],
        env=env,
        preexec_fn=os.setsid,
    )
    try:
        proc.wait()
        logger.info("Sweep run %s finished with returncode %d", run_name, proc.returncode)
    finally:
        _terminate_with_grace(proc, timeout=CLEANUP_TIMEOUT_S)
        _sweep_orphan_cleanup(run_name=run_name)
        # wandb.agent owns the run lifecycle; do not call wandb.finish here


def main() -> None:
    import argparse

    import wandb

    global BASE_CONFIG_PATH, EVAL_OVERRIDES, PATCHED_CONFIG_DIR
    parser = argparse.ArgumentParser(
        description="W&B Bayesian sweep over RL hyperparameters."
    )
    parser.add_argument("--config", required=True, help="Path to sweep YAML")
    parser.add_argument("--count", type=int, default=40, help="Number of runs")
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Existing sweep ID to add runs to (skips wandb.sweep create)",
    )
    parser.add_argument(
        "--project",
        default="elitefurretai-rnad-sweep",
        help="Wandb project for sweep + runs",
    )
    parser.add_argument(
        "--patched-config-dir",
        default="/tmp/efai-sweep-configs",
        help="Where to write per-run patched config YAMLs",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    sweep_dict, base_cfg_path, eval_overrides = load_sweep_config(args.config)
    BASE_CONFIG_PATH = base_cfg_path
    EVAL_OVERRIDES = eval_overrides
    PATCHED_CONFIG_DIR = args.patched_config_dir
    os.makedirs(PATCHED_CONFIG_DIR, exist_ok=True)

    sweep_id = args.sweep_id or wandb.sweep(sweep_dict, project=args.project)
    logger.info("Sweep ID: %s", sweep_id)
    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)


if __name__ == "__main__":
    main()
