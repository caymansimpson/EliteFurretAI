# -*- coding: utf-8 -*-
"""Unit tests for train_sweep (YAML loader, dotted-key patching, termination)."""

import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

_SWEEP_YAML = """
base_config: "src/elitefurretai/rl/configs/may26.yaml"

eval_overrides:
  training.wandb_project: "test-project"
  eval.enabled: true
  eval.eval_every_n_updates: 500
  eval.opponents.foul_play.weight: 0.0

sweep:
  method: bayes
  metric:
    name: "eval/score"
    goal: maximize
  parameters:
    learner.learning_rate:
      min: 1.0e-5
      max: 1.0e-3
      distribution: log_uniform_values
"""


class TestLoadSweepConfig:
    def test_returns_three_components(self, tmp_path):
        from elitefurretai.rl.train_sweep import load_sweep_config

        p = tmp_path / "sweep.yaml"
        p.write_text(_SWEEP_YAML)
        sweep_dict, base_cfg_path, eval_overrides = load_sweep_config(str(p))
        assert sweep_dict["method"] == "bayes"
        assert sweep_dict["metric"]["name"] == "eval/score"
        assert base_cfg_path == "src/elitefurretai/rl/configs/may26.yaml"
        assert eval_overrides["eval.enabled"] is True
        assert eval_overrides["training.wandb_project"] == "test-project"

    def test_raises_on_missing_base_config(self, tmp_path):
        from elitefurretai.rl.train_sweep import load_sweep_config

        p = tmp_path / "bad.yaml"
        p.write_text("sweep: {method: bayes}\n")
        with pytest.raises((KeyError, ValueError)):
            load_sweep_config(str(p))

    def test_raises_on_missing_sweep(self, tmp_path):
        from elitefurretai.rl.train_sweep import load_sweep_config

        p = tmp_path / "bad.yaml"
        p.write_text("base_config: x.yaml\n")
        with pytest.raises((KeyError, ValueError)):
            load_sweep_config(str(p))


class TestDottedKeyPatching:
    def _base_yaml(self, tmp_path):
        data = {
            "training": {"wandb_project": "default"},
            "eval": {
                "enabled": False,
                "eval_every_n_updates": 1000,
                "opponents": {
                    "foul_play": {"weight": 1.0, "target": 0.5, "n_battles": 40},
                },
            },
            "learner": {"learning_rate": 1e-4, "eta": 0.0},
        }
        p = tmp_path / "base.yaml"
        p.write_text(yaml.safe_dump(data))
        return str(p)

    def test_top_level_field_patched(self, tmp_path):
        from elitefurretai.rl.train_sweep import _write_patched_config

        base = self._base_yaml(tmp_path)
        out = _write_patched_config(
            base_config_path=base,
            eval_overrides={"training.wandb_project": "patched-project"},
            sweep_params={},
            run_name="r1",
            out_dir=str(tmp_path),
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["training"]["wandb_project"] == "patched-project"

    def test_nested_dataclass_field_patched(self, tmp_path):
        from elitefurretai.rl.train_sweep import _write_patched_config

        base = self._base_yaml(tmp_path)
        out = _write_patched_config(
            base_config_path=base,
            eval_overrides={},
            sweep_params={"learner.learning_rate": 5e-5, "learner.eta": 0.01},
            run_name="r2",
            out_dir=str(tmp_path),
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["learner"]["learning_rate"] == 5e-5
        assert data["learner"]["eta"] == 0.01

    def test_dict_of_dataclass_field_patched(self, tmp_path):
        from elitefurretai.rl.train_sweep import _write_patched_config

        base = self._base_yaml(tmp_path)
        out = _write_patched_config(
            base_config_path=base,
            eval_overrides={"eval.opponents.foul_play.weight": 0.0},
            sweep_params={},
            run_name="r3",
            out_dir=str(tmp_path),
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["eval"]["opponents"]["foul_play"]["weight"] == 0.0

    def test_sweep_params_override_eval_overrides(self, tmp_path):
        from elitefurretai.rl.train_sweep import _write_patched_config

        base = self._base_yaml(tmp_path)
        out = _write_patched_config(
            base_config_path=base,
            eval_overrides={"learner.learning_rate": 1e-3},
            sweep_params={"learner.learning_rate": 7e-5},
            run_name="r4",
            out_dir=str(tmp_path),
        )
        with open(out) as f:
            data = yaml.safe_load(f)
        # Sweep wins
        assert data["learner"]["learning_rate"] == 7e-5


class TestTerminateWithGrace:
    def test_clean_exit_on_sigterm(self):
        from elitefurretai.rl.train_sweep import _terminate_with_grace

        code = (
            "import signal,sys,time; "
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); "
            "time.sleep(10)"
        )
        proc = subprocess.Popen([sys.executable, "-c", code], preexec_fn=os.setsid)
        time.sleep(0.5)
        _terminate_with_grace(proc, timeout=3)
        assert proc.poll() is not None
        assert proc.returncode == 0

    def test_sigkill_escalation_for_unresponsive_child(self):
        from elitefurretai.rl.train_sweep import _terminate_with_grace

        code = (
            "import signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
        proc = subprocess.Popen([sys.executable, "-c", code], preexec_fn=os.setsid)
        time.sleep(0.5)
        t0 = time.time()
        _terminate_with_grace(proc, timeout=2)
        elapsed = time.time() - t0
        assert proc.poll() is not None
        # Escalation should fire within timeout + SIGKILL grace, not wait
        # the full 60 the child wanted
        assert elapsed < 10.0
        # Killed by SIGKILL → negative return code on POSIX
        assert proc.returncode == -signal.SIGKILL or proc.returncode < 0

    def test_already_exited_child_is_noop(self):
        from elitefurretai.rl.train_sweep import _terminate_with_grace

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        _terminate_with_grace(proc, timeout=1)  # should not raise
        assert proc.returncode == 0
