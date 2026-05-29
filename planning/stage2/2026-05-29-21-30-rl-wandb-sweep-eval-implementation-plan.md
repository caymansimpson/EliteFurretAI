# RL W&B Sweep with Multi-Baseline Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Bayesian W&B sweep harness that drives RL training and optimizes a single scalar score derived from win rates against four baselines (SHP, MaxDamage, VGCBench, BCPlayer), implemented behind a generic multi-bucket eval driver that subsumes the existing FoulPlay eval.

**Architecture:** Replace `FoulplayEvalConfig` with a unified `EvalConfig` whose `opponents` dict drives a per-checkpoint eval pass through a new `analyze/baseline_eval.py` driver. The driver dispatches each active opponent (in-process or external-subprocess) through the existing `parse_player_specification` path. Scoring uses hinge L2 deficit + linear surplus on percentage-point-scaled win rates so floors dominate. A new `train_sweep.py` runs `wandb.agent` with subprocess-per-run isolation (each sweep run spawns `python -m elitefurretai.rl.train` with `WANDB_RUN_ID` env), with three-layer cleanup to keep Showdown servers and external subprocesses from leaking across sweep runs.

**Tech Stack:** Python 3, PyTorch, poke-env, Weights & Biases (`wandb`), `pyyaml`, `subprocess` + `os.killpg`, pytest, ruff, pyright.

**Design reference:** [planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md](2026-05-29-21-00-rl-wandb-sweep-eval-design.md)

---

## File Structure

**Created:**
- [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) — driver, scoring, log payload, standalone CLI
- [src/elitefurretai/rl/train_sweep.py](../../src/elitefurretai/rl/train_sweep.py) — sweep harness (YAML loader, config patcher, sweep_train, main)
- [src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml](../../src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml) — first sweep config
- [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py) — tests for compute_score, dispatch, payload, layer-1 cleanup
- [unit_tests/rl/test_train_sweep.py](../../unit_tests/rl/test_train_sweep.py) — tests for YAML loader, dotted-key patching, layer-3 termination
- [unit_tests/rl/analyze/__init__.py](../../unit_tests/rl/analyze/__init__.py) — empty package marker if needed

**Modified:**
- [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py) — replace `FoulplayEvalConfig` with `EvalConfig` + `OpponentEvalSpec`; rename `RNaDConfig.foulplay_eval` → `eval`; update `__post_init__` validation; update `from_yaml`
- [src/elitefurretai/rl/train.py](../../src/elitefurretai/rl/train.py) — replace `_maybe_run_foulplay_eval` with `_maybe_run_eval`; reinforce layer-2 cleanup; remove `import elitefurretai.rl.analyze.foulplay_eval`
- [src/elitefurretai/rl/configs/may25.yaml](../../src/elitefurretai/rl/configs/may25.yaml), [may26.yaml](../../src/elitefurretai/rl/configs/may26.yaml), [multiformat.yaml](../../src/elitefurretai/rl/configs/multiformat.yaml), [may25_resume.yaml](../../src/elitefurretai/rl/configs/may25_resume.yaml) — rename top-level `foulplay_eval:` section to `eval:` with the new shape

**Deleted (last task only, after validation):**
- [src/elitefurretai/rl/analyze/foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py) — subsumed by `baseline_eval.py`

---

## Task 1: `compute_score` pure function

**Files:**
- Create: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (scaffold + function)
- Create: [unit_tests/rl/analyze/__init__.py](../../unit_tests/rl/analyze/__init__.py)
- Create: [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py)

- [ ] **Step 1.1: Create empty test package marker**

```bash
mkdir -p unit_tests/rl/analyze
: > unit_tests/rl/analyze/__init__.py
```

- [ ] **Step 1.2: Write failing tests for `compute_score`**

Create `unit_tests/rl/analyze/test_baseline_eval.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for baseline_eval (compute_score, dispatch, payload, cleanup)."""

import math

import pytest

from elitefurretai.rl.analyze.baseline_eval import compute_score


def _targets_all(t: float):
    return {"a": t, "b": t, "c": t, "d": t}


def _weights_unit():
    return {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}


class TestComputeScore:
    def test_all_floors_met_exactly_returns_zero(self):
        rates = {"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.8}
        score, br = compute_score(rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0)
        assert score == 0.0
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == 0.0

    def test_uniform_overshoot_pure_surplus(self):
        # 10pp surplus on each of 4 buckets at alpha=1 → 10*4 = 40
        rates = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}
        score, br = compute_score(rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0)
        assert score == pytest.approx(40.0)
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == pytest.approx(40.0)

    def test_uniform_deficit_pure_l2_penalty(self):
        # 10pp deficit on each of 4 buckets → 10^2 * 4 = 400, score is negative
        rates = {"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.7}
        score, br = compute_score(rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0)
        assert score == pytest.approx(-400.0)
        assert br["deficit_l2_pp"] == pytest.approx(400.0)
        assert br["surplus_sum_pp"] == 0.0

    def test_mixed_deficit_and_surplus(self):
        # a: 20pp deficit → 400 ; b: 10pp surplus → 10 ; c,d on target → 0
        rates = {"a": 0.6, "b": 0.9, "c": 0.8, "d": 0.8}
        score, br = compute_score(rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0)
        assert br["deficit_l2_pp"] == pytest.approx(400.0)
        assert br["surplus_sum_pp"] == pytest.approx(10.0)
        assert score == pytest.approx(10.0 - 400.0)

    def test_weights_scale_both_terms(self):
        rates = {"a": 0.7, "b": 0.9}   # 10pp deficit / 10pp surplus
        targets = {"a": 0.8, "b": 0.8}
        weights = {"a": 2.0, "b": 3.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        # deficit: 2.0 * 10^2 = 200 ; surplus: 3.0 * 10 = 30
        assert br["deficit_l2_pp"] == pytest.approx(200.0)
        assert br["surplus_sum_pp"] == pytest.approx(30.0)
        assert score == pytest.approx(30.0 - 200.0)

    def test_surplus_alpha_scales_surplus_term_only(self):
        rates = {"a": 0.9}   # 10pp surplus
        targets = {"a": 0.8}
        weights = {"a": 1.0}
        s0, _ = compute_score(rates, targets, weights, surplus_alpha=0.0)
        s5, _ = compute_score(rates, targets, weights, surplus_alpha=0.5)
        s1, _ = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert s0 == 0.0
        assert s5 == pytest.approx(5.0)
        assert s1 == pytest.approx(10.0)

    def test_edge_rate_zero_max_deficit(self):
        rates = {"a": 0.0}
        targets = {"a": 1.0}
        weights = {"a": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert br["deficit_l2_pp"] == pytest.approx(10000.0)
        assert br["surplus_sum_pp"] == 0.0
        assert score == pytest.approx(-10000.0)

    def test_edge_rate_one_max_surplus(self):
        rates = {"a": 1.0}
        targets = {"a": 0.0}
        weights = {"a": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == pytest.approx(100.0)
        assert score == pytest.approx(100.0)

    def test_empty_inputs_return_zero(self):
        score, br = compute_score({}, {}, {}, surplus_alpha=1.0)
        assert score == 0.0
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == 0.0

    def test_finite_score(self):
        # Sanity: no NaNs/Infs for plausible inputs
        rates = {"a": 0.55, "b": 0.62, "c": 0.81, "d": 0.99}
        targets = {"a": 0.8, "b": 0.6, "c": 0.8, "d": 0.8}
        weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert math.isfinite(score)
        assert math.isfinite(br["deficit_l2_pp"])
        assert math.isfinite(br["surplus_sum_pp"])
```

- [ ] **Step 1.3: Run tests, verify they fail**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py -v`
Expected: `ImportError: cannot import name 'compute_score'` (module does not exist yet).

- [ ] **Step 1.4: Implement `compute_score` + module scaffold**

Create `src/elitefurretai/rl/analyze/baseline_eval.py`:

```python
# -*- coding: utf-8 -*-
"""Multi-bucket eval driver, scoring, log payload, and standalone CLI.

Replaces the FoulPlay-specific driver in foulplay_eval.py with a generic
driver that handles every opponent in EvalConfig.opponents identically.
FoulPlay becomes one bucket; its weight defaults to 0.0 until the
FoulPlay subprocess is stable.
"""

from __future__ import annotations

from typing import Dict, Tuple


def compute_score(
    win_rates: Dict[str, float],
    targets: Dict[str, float],
    weights: Dict[str, float],
    surplus_alpha: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute the W&B sweep metric from per-opponent win rates.

    Hinge L2 on deficit + linear surplus, both scaled to percentage
    points (×100) so floors dominate at any non-trivial deficit. See
    planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md
    Section 1 for the rationale.

    Maximize. score=0 means every floor met exactly.
    """
    deficit_pp = {k: max(0.0, 100.0 * (targets[k] - win_rates[k])) for k in weights}
    surplus_pp = {k: max(0.0, 100.0 * (win_rates[k] - targets[k])) for k in weights}
    deficit_term = sum(weights[k] * deficit_pp[k] ** 2 for k in weights)
    surplus_term = sum(weights[k] * surplus_pp[k] for k in weights)
    score = surplus_alpha * surplus_term - deficit_term
    return score, {"deficit_l2_pp": deficit_term, "surplus_sum_pp": surplus_term}
```

- [ ] **Step 1.5: Run tests, verify they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py -v`
Expected: 10 passing.

- [ ] **Step 1.6: Run quality gates**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/
```

Expected: all pass.

- [ ] **Step 1.7: Commit**

```bash
git add src/elitefurretai/rl/analyze/baseline_eval.py \
        unit_tests/rl/analyze/__init__.py \
        unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add compute_score for multi-baseline sweep metric

Hinge L2 deficit + linear surplus on pp-scaled win rates so floors
dominate. Pure function; exhaustively unit-tested before any I/O is
introduced. Scaffolds analyze/baseline_eval.py for subsequent tasks."
```

---

## Task 2: `OpponentEvalSpec` + `EvalConfig` (replaces FoulplayEvalConfig)

**Files:**
- Modify: [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py) — replace `FoulplayEvalConfig` (line 704), rename `RNaDConfig.foulplay_eval` field (line 814), update `from_yaml` (line 916), update `__post_init__` validation (lines 984-991)
- Modify: [src/elitefurretai/rl/configs/may25.yaml](../../src/elitefurretai/rl/configs/may25.yaml), [may26.yaml](../../src/elitefurretai/rl/configs/may26.yaml), [multiformat.yaml](../../src/elitefurretai/rl/configs/multiformat.yaml), [may25_resume.yaml](../../src/elitefurretai/rl/configs/may25_resume.yaml) — rename `foulplay_eval:` section to `eval:` with new shape
- Test: extend [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py) (config defaults + YAML roundtrip)

- [ ] **Step 2.1: Read existing FoulplayEvalConfig + RNaDConfig.from_yaml for context**

```bash
sed -n '700,740p' src/elitefurretai/rl/config.py    # FoulplayEvalConfig
sed -n '800,925p' src/elitefurretai/rl/config.py    # RNaDConfig + from_yaml
sed -n '980,995p' src/elitefurretai/rl/config.py    # validation
```

Note: `_make_sub` helper at ~line 890 is the existing pattern for sub-dataclass construction. Reuse it.

- [ ] **Step 2.2: Write failing tests for `EvalConfig` defaults + YAML roundtrip**

Append to `unit_tests/rl/analyze/test_baseline_eval.py`:

```python
from elitefurretai.rl.config import EvalConfig, OpponentEvalSpec


class TestEvalConfig:
    def test_defaults_include_four_baselines_plus_foulplay_off(self):
        cfg = EvalConfig()
        assert cfg.enabled is False
        assert cfg.eval_every_n_updates == 500
        assert cfg.pause_training is True
        assert cfg.surplus_alpha == 1.0
        # All five expected canonical names present
        assert set(cfg.opponents.keys()) == {
            "simple_heuristic_baseline",
            "max_damage",
            "vgc_bench",
            "bc_player",
            "foul_play",
        }
        # FoulPlay shipped off
        assert cfg.opponents["foul_play"].weight == 0.0
        # Other four shipped on with weight 1.0
        for k in ["simple_heuristic_baseline", "max_damage", "vgc_bench", "bc_player"]:
            assert cfg.opponents[k].weight == 1.0
        # Targets match the design spec
        assert cfg.opponents["simple_heuristic_baseline"].target == 0.80
        assert cfg.opponents["max_damage"].target == 0.80
        assert cfg.opponents["vgc_bench"].target == 0.60
        assert cfg.opponents["bc_player"].target == 0.80

    def test_opponent_spec_defaults(self):
        spec = OpponentEvalSpec(target=0.5, weight=1.0, n_battles=100)
        assert spec.target == 0.5
        assert spec.weight == 1.0
        assert spec.n_battles == 100

    def test_eval_section_loads_from_yaml(self, tmp_path):
        import yaml
        from elitefurretai.rl.config import RNaDConfig

        # Minimal RNaDConfig YAML override of the eval section
        data = {
            "eval": {
                "enabled": True,
                "eval_every_n_updates": 1234,
                "surplus_alpha": 0.5,
                "opponents": {
                    "vgc_bench": {"target": 0.65, "weight": 2.0, "n_battles": 80},
                },
            },
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data))
        cfg = RNaDConfig.from_yaml(str(p))
        assert cfg.eval.enabled is True
        assert cfg.eval.eval_every_n_updates == 1234
        assert cfg.eval.surplus_alpha == 0.5
        # YAML-supplied vgc_bench overrides defaults
        assert cfg.eval.opponents["vgc_bench"].target == 0.65
        assert cfg.eval.opponents["vgc_bench"].weight == 2.0
        assert cfg.eval.opponents["vgc_bench"].n_battles == 80
        # Other opponents fall back to defaults
        assert cfg.eval.opponents["foul_play"].weight == 0.0
        assert cfg.eval.opponents["simple_heuristic_baseline"].weight == 1.0
```

- [ ] **Step 2.3: Run tests, verify they fail**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestEvalConfig -v`
Expected: `ImportError: cannot import name 'EvalConfig'`.

- [ ] **Step 2.4: Replace `FoulplayEvalConfig` with `EvalConfig` + `OpponentEvalSpec`**

Edit `src/elitefurretai/rl/config.py`. Replace the `FoulplayEvalConfig` block (line 704 onward) with:

```python
@dataclass
class OpponentEvalSpec:
    """Per-opponent eval config. Lives in EvalConfig.opponents keyed by
    canonical opponent name. Fields apply across all curriculum formats:
    n_battles is split per-format using curriculum.battle_formats weights.
    """

    target: float
    weight: float
    n_battles: int


@dataclass
class EvalConfig:
    """Inline-during-training multi-bucket eval (replaces FoulplayEvalConfig).

    Drives a generic eval pass against every opponent in `opponents` whose
    weight > 0. The pass runs at checkpoint cadence (every
    `eval_every_n_updates`), pauses training, and emits a scalar
    `eval/score` metric for W&B sweeps. See
    planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md.

    FoulPlay is one opponent in this dict. Its default weight is 0.0
    until the subprocess is stable; flipping the weight in YAML is the
    only thing needed to enable it.
    """

    enabled: bool = False
    eval_every_n_updates: int = 500
    pause_training: bool = True
    surplus_alpha: float = 1.0

    opponents: Dict[str, OpponentEvalSpec] = field(
        default_factory=lambda: {
            "simple_heuristic_baseline": OpponentEvalSpec(
                target=0.80, weight=1.0, n_battles=150
            ),
            "max_damage": OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
            "vgc_bench": OpponentEvalSpec(target=0.60, weight=1.0, n_battles=100),
            "bc_player": OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
            "foul_play": OpponentEvalSpec(target=0.50, weight=0.0, n_battles=40),
        }
    )

    # Opponent-specific runtime knobs (read only when the corresponding
    # opponent's weight > 0). Naming follows the `<opp>_*` convention
    # used by player_factory for cross-venv subprocess kwargs.
    vgcbench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"
    vgcbench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt"
    vgcbench_python_executable: str = (
        "/home/cayman/Repositories/venv-vgcbench/bin/python"
    )

    foulplay_search_time_ms: int = 750
    foulplay_python_executable: str = (
        "/home/cayman/Repositories/venv-foulplay/bin/python"
    )
    foulplay_team_pool_paths: Optional[Dict[str, str]] = None
    foulplay_parallelism: int = 8
    foulplay_model_probabilistic: bool = False
```

Imports at the top of `config.py` should already include `Dict`, `Optional`, `field`, `dataclass`. Verify and add any missing.

- [ ] **Step 2.5: Update `RNaDConfig` to use `EvalConfig`**

At line 814, change:

```python
    foulplay_eval: FoulplayEvalConfig = field(default_factory=FoulplayEvalConfig)
```

to:

```python
    eval: EvalConfig = field(default_factory=EvalConfig)
```

At line 916 (inside `from_yaml`), change:

```python
            foulplay_eval=_make_sub(FoulplayEvalConfig, data.get("foulplay_eval", {})),
```

to:

```python
            eval=_make_eval_sub(data.get("eval", {})),
```

The `eval` field's nested `opponents` dict can't use the plain `_make_sub` helper because its values are dataclasses. Add a small helper above `RNaDConfig.from_yaml`:

```python
def _make_eval_sub(data: dict) -> "EvalConfig":
    """Construct EvalConfig from YAML dict, merging the opponents map
    with EvalConfig defaults so partial YAML overrides work.
    """
    defaults = EvalConfig()
    opponents_override = data.pop("opponents", {})
    merged_opponents = dict(defaults.opponents)
    for name, opp_data in opponents_override.items():
        existing = merged_opponents.get(
            name, OpponentEvalSpec(target=0.5, weight=0.0, n_battles=0)
        )
        merged_opponents[name] = OpponentEvalSpec(
            target=opp_data.get("target", existing.target),
            weight=opp_data.get("weight", existing.weight),
            n_battles=opp_data.get("n_battles", existing.n_battles),
        )
    return EvalConfig(opponents=merged_opponents, **data)
```

- [ ] **Step 2.6: Update `RNaDConfig.__post_init__` validation**

Replace the FoulPlay validation block (lines ~984-991) with:

```python
        ev = self.eval
        if ev.enabled:
            fp_spec = ev.opponents.get("foul_play")
            if fp_spec is not None and fp_spec.weight > 0:
                if not ev.foulplay_python_executable:
                    raise ValueError(
                        "eval.foulplay_python_executable must be set when "
                        "eval.opponents['foul_play'].weight > 0"
                    )
                if not os.path.exists(ev.foulplay_python_executable):
                    raise ValueError(
                        f"eval.foulplay_python_executable not found: "
                        f"{ev.foulplay_python_executable}"
                    )
            vgc_spec = ev.opponents.get("vgc_bench")
            if vgc_spec is not None and vgc_spec.weight > 0:
                if not os.path.exists(ev.vgcbench_python_executable):
                    raise ValueError(
                        f"eval.vgcbench_python_executable not found: "
                        f"{ev.vgcbench_python_executable}"
                    )
```

- [ ] **Step 2.7: Update YAML configs**

For each of [may25.yaml](../../src/elitefurretai/rl/configs/may25.yaml), [may26.yaml](../../src/elitefurretai/rl/configs/may26.yaml), [multiformat.yaml](../../src/elitefurretai/rl/configs/multiformat.yaml), [may25_resume.yaml](../../src/elitefurretai/rl/configs/may25_resume.yaml):

Read the existing `foulplay_eval:` section. Replace it with:

```yaml
eval:
  enabled: <preserved>
  eval_every_n_updates: <preserved>
  surplus_alpha: 1.0
  opponents:
    foul_play:
      target: 0.50
      weight: <1.0 if old enabled was true else 0.0>
      n_battles: <preserved n_battles_per_format value or 40>
  foulplay_search_time_ms: <preserved old search_time_ms>
  foulplay_parallelism: <preserved old parallelism>
  foulplay_python_executable: <preserved old python_executable>
  foulplay_team_pool_paths: <preserved old foulplay_team_pool_paths>
  foulplay_model_probabilistic: <preserved old model_probabilistic>
```

If the old section had FoulPlay disabled and used only the default placeholder, the new section can be the minimal `eval:` block with no `opponents` override (defaults handle everything).

- [ ] **Step 2.8: Run tests, verify they pass**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestEvalConfig -v
```

Expected: 3 passing.

Also run the existing test suite to catch any RNaDConfig regressions:

```bash
source ../venv/bin/activate && pytest unit_tests/rl/ -q
```

Expected: all existing tests pass (or surface clean failures that need rename fix-ups).

- [ ] **Step 2.9: Run quality gates**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/config.py unit_tests/rl/ && \
  ruff format src/elitefurretai/rl/config.py unit_tests/rl/ --check && \
  pyright src/elitefurretai/rl/config.py
```

Expected: all pass.

- [ ] **Step 2.10: Commit**

```bash
git add src/elitefurretai/rl/config.py \
        src/elitefurretai/rl/configs/*.yaml \
        unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "refactor(rl-config): replace FoulplayEvalConfig with EvalConfig

EvalConfig unifies per-checkpoint eval against all opponents, with
FoulPlay folded in as one bucket (weight=0.0 by default until
FoulPlay subprocess is stable). RNaDConfig.foulplay_eval renamed to
.eval; from_yaml learns to merge opponents-dict partial overrides.
Existing configs migrated to the new shape with preserved semantics."
```

---

## Task 3: `BucketRunResult` + `MultiBucketEvalResult` dataclasses

**Files:**
- Modify: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (append dataclasses)
- Test: extend [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py) (instantiate-and-introspect smoke)

- [ ] **Step 3.1: Write failing instantiation test**

Append to test file:

```python
from elitefurretai.rl.analyze.baseline_eval import (
    BucketRunResult,
    MultiBucketEvalResult,
)
from elitefurretai.rl.analyze.evaluate import EvalResult


class TestResultDataclasses:
    def test_bucket_run_result_instantiation(self):
        ev = EvalResult(
            player1_wins=80, player2_wins=20, ties=0, battles_played=100
        )
        b = BucketRunResult(
            win_rate=0.80,
            n_battles=100,
            per_format={"gen9vgc2023regc": ev},
            wall_time_s=12.5,
        )
        assert b.win_rate == 0.80
        assert b.n_battles == 100
        assert "gen9vgc2023regc" in b.per_format
        assert b.wall_time_s == 12.5

    def test_multi_bucket_eval_result_instantiation(self):
        ev = EvalResult(player1_wins=80, player2_wins=20, ties=0, battles_played=100)
        bucket = BucketRunResult(
            win_rate=0.80, n_battles=100,
            per_format={"gen9vgc2023regc": ev}, wall_time_s=1.0,
        )
        r = MultiBucketEvalResult(
            per_bucket={"max_damage": bucket},
            score=-5.0,
            breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 0.0},
            wall_time_s=2.0,
        )
        assert r.score == -5.0
        assert r.per_bucket["max_damage"].win_rate == 0.80
```

- [ ] **Step 3.2: Run, verify ImportError**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestResultDataclasses -v
```

Expected: ImportError for `BucketRunResult`.

- [ ] **Step 3.3: Add dataclasses to baseline_eval.py**

Append to `src/elitefurretai/rl/analyze/baseline_eval.py`:

```python
from dataclasses import dataclass, field
from typing import Optional

from elitefurretai.rl.analyze.evaluate import EvalResult


@dataclass
class BucketRunResult:
    """Aggregated outcome for ONE opponent (one entry in
    EvalConfig.opponents) after running its full eval cycle across all
    curriculum.battle_formats.

    Granularity: one BucketRunResult per active opponent per eval pass.

    win_rate is the format-weighted mean of per-format win rates and is
    the single number that feeds compute_score for this opponent.
    per_format preserves the per-format breakdown so the wandb logger
    can compute eval/<format>/win_rate cross-opponent aggregates.
    """

    win_rate: float
    n_battles: int
    per_format: Dict[str, EvalResult] = field(default_factory=dict)
    wall_time_s: float = 0.0


@dataclass
class MultiBucketEvalResult:
    """Outcome of ONE full multi-bucket eval pass.

    Granularity: one MultiBucketEvalResult per call to baseline_eval.run,
    i.e. one per checkpoint boundary during training or one per
    standalone CLI invocation.

    per_bucket has one entry per active (weight > 0) opponent.
    Disabled opponents (weight == 0) do not appear here.
    score is the scalar W&B sweep metric.
    """

    per_bucket: Dict[str, BucketRunResult] = field(default_factory=dict)
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0
```

- [ ] **Step 3.4: Run, verify pass**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py -v
```

- [ ] **Step 3.5: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py
git add src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add BucketRunResult and MultiBucketEvalResult

Result dataclasses for the multi-bucket eval driver. Docstrings
explicitly name the granularity (per-opponent-per-eval-pass vs
per-eval-pass) since the names alone don't make it self-evident."
```

---

## Task 4: `_run_opponent_bucket` (per-opponent dispatch)

**Files:**
- Modify: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (add `_run_opponent_bucket`)
- Test: extend [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py)

The bucket runner iterates `curriculum.battle_formats`, calls `parse_player_specification` with opponent-name-prefixed kwargs from EvalConfig, runs `run_eval_parallel` per format, and aggregates per-format results using format weights.

- [ ] **Step 4.1: Read parse_player_specification signature to confirm kwargs**

```bash
sed -n '140,200p' src/elitefurretai/rl/analyze/player_factory.py
```

Note the `vgc_bench_*` and `foul_play_*` kwargs the function accepts. The bucket runner translates EvalConfig's `vgcbench_*` / `foulplay_*` field names (no underscore between `vgc` and `bench`) to the kwargs `parse_player_specification` expects.

- [ ] **Step 4.2: Write failing tests with mocks**

Append to test file:

```python
from unittest.mock import MagicMock, patch

from elitefurretai.rl.analyze.baseline_eval import _run_opponent_bucket
from elitefurretai.rl.config import CurriculumConfig, EvalConfig, OpponentEvalSpec


def _mock_curriculum_single_format():
    cur = MagicMock(spec=CurriculumConfig)
    cur.battle_formats = ["gen9vgc2023regc"]
    cur.format_weights = {"gen9vgc2023regc": 1.0}
    return cur


def _mock_curriculum_two_formats():
    cur = MagicMock(spec=CurriculumConfig)
    cur.battle_formats = ["gen9vgc2023regc", "gen9vgc2024regg"]
    cur.format_weights = {"gen9vgc2023regc": 0.6, "gen9vgc2024regg": 0.4}
    return cur


class TestRunOpponentBucket:
    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_vgc_bench_receives_vgcbench_kwargs(self, mock_parse, mock_run):
        from elitefurretai.rl.analyze.evaluate import EvalResult

        mock_parse.return_value = MagicMock()
        mock_run.return_value = EvalResult(
            player1_wins=60, player2_wins=40, ties=0, battles_played=100
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["vgc_bench"]
        result, externals = _run_opponent_bucket(
            opp_name="vgc_bench", spec=spec, eval_cfg=eval_cfg,
            curriculum=_mock_curriculum_single_format(),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        # parse_player_specification got vgc_bench-prefixed kwargs
        kwargs = mock_parse.call_args.kwargs
        assert kwargs.get("vgc_bench_checkpoint_path") == eval_cfg.vgcbench_checkpoint_path
        assert kwargs.get("vgc_bench_team_file") == eval_cfg.vgcbench_team_file
        assert kwargs.get("vgc_bench_python_executable") == eval_cfg.vgcbench_python_executable
        # No foul_play kwargs leaked in
        assert "foul_play_python_executable" not in kwargs
        assert result.win_rate == pytest.approx(0.60)

    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_foul_play_receives_foulplay_kwargs(self, mock_parse, mock_run):
        from elitefurretai.rl.analyze.evaluate import EvalResult

        mock_parse.return_value = MagicMock()
        mock_run.return_value = EvalResult(
            player1_wins=20, player2_wins=20, ties=0, battles_played=40
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["foul_play"]
        _run_opponent_bucket(
            opp_name="foul_play", spec=spec, eval_cfg=eval_cfg,
            curriculum=_mock_curriculum_single_format(),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        kwargs = mock_parse.call_args.kwargs
        assert kwargs.get("foul_play_python_executable") == eval_cfg.foulplay_python_executable
        assert kwargs.get("foul_play_search_time_ms") == eval_cfg.foulplay_search_time_ms
        assert kwargs.get("foul_play_parallelism") == eval_cfg.foulplay_parallelism
        # No vgc_bench kwargs leaked in
        assert "vgc_bench_checkpoint_path" not in kwargs

    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_inprocess_baseline_receives_no_external_kwargs(self, mock_parse, mock_run):
        from elitefurretai.rl.analyze.evaluate import EvalResult

        mock_parse.return_value = MagicMock()
        mock_run.return_value = EvalResult(
            player1_wins=120, player2_wins=30, ties=0, battles_played=150
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["max_damage"]
        _run_opponent_bucket(
            opp_name="max_damage", spec=spec, eval_cfg=eval_cfg,
            curriculum=_mock_curriculum_single_format(),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        kwargs = mock_parse.call_args.kwargs
        assert "vgc_bench_checkpoint_path" not in kwargs
        assert "foul_play_python_executable" not in kwargs

    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_two_formats_aggregates_with_format_weights(self, mock_parse, mock_run):
        from elitefurretai.rl.analyze.evaluate import EvalResult

        mock_parse.return_value = MagicMock()
        # Format A: 90% win rate ; Format B: 50% win rate ;
        # weights 0.6 / 0.4 → aggregate = 0.6*0.9 + 0.4*0.5 = 0.74
        mock_run.side_effect = [
            EvalResult(player1_wins=54, player2_wins=6, ties=0, battles_played=60),
            EvalResult(player1_wins=20, player2_wins=20, ties=0, battles_played=40),
        ]
        eval_cfg = EvalConfig()
        spec = OpponentEvalSpec(target=0.5, weight=1.0, n_battles=100)
        result, _ = _run_opponent_bucket(
            opp_name="max_damage", spec=spec, eval_cfg=eval_cfg,
            curriculum=_mock_curriculum_two_formats(),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        assert result.win_rate == pytest.approx(0.74)
        assert result.n_battles == 100
        assert set(result.per_format.keys()) == {"gen9vgc2023regc", "gen9vgc2024regg"}
```

- [ ] **Step 4.3: Run, verify failures**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestRunOpponentBucket -v
```

Expected: ImportError for `_run_opponent_bucket`.

- [ ] **Step 4.4: Implement `_run_opponent_bucket`**

Append to `src/elitefurretai/rl/analyze/baseline_eval.py`:

```python
import time
from typing import Any, List, Tuple

from elitefurretai.rl.analyze.evaluate import EvalResult, run_eval_parallel
from elitefurretai.rl.analyze.player_factory import (
    RunningExternal,
    launch_external_player,
    parse_player_specification,
)
from elitefurretai.rl.config import CurriculumConfig, EvalConfig, OpponentEvalSpec


# Opponent canonical names that require external-subprocess construction
# via launch_external_player rather than a plain in-process Player. Mirrors
# player_factory._EXTERNAL_BASELINES.
_EXTERNAL_OPPONENTS = ("vgc_bench", "foul_play")


def _opponent_kwargs(opp_name: str, eval_cfg: EvalConfig) -> Dict[str, Any]:
    """Translate EvalConfig's `vgcbench_*` / `foulplay_*` fields into the
    kwarg names parse_player_specification expects (`vgc_bench_*`,
    `foul_play_*`). In-process baselines get an empty dict.
    """
    if opp_name == "vgc_bench":
        return {
            "vgc_bench_checkpoint_path": eval_cfg.vgcbench_checkpoint_path,
            "vgc_bench_team_file": eval_cfg.vgcbench_team_file,
            "vgc_bench_python_executable": eval_cfg.vgcbench_python_executable,
        }
    if opp_name == "foul_play":
        return {
            "foul_play_python_executable": eval_cfg.foulplay_python_executable,
            "foul_play_search_time_ms": eval_cfg.foulplay_search_time_ms,
            "foul_play_parallelism": eval_cfg.foulplay_parallelism,
            "foul_play_team_pool_paths": eval_cfg.foulplay_team_pool_paths,
            "foul_play_model_probabilistic": eval_cfg.foulplay_model_probabilistic,
        }
    return {}


def _split_battles_by_format(
    n_total: int, format_weights: Dict[str, float]
) -> Dict[str, int]:
    """Allocate per-format battle counts proportional to curriculum format
    weights. Uses largest-remainder so the sum matches n_total exactly.
    """
    if not format_weights:
        return {}
    total_weight = sum(format_weights.values())
    raw = {fmt: n_total * w / total_weight for fmt, w in format_weights.items()}
    floors = {fmt: int(v) for fmt, v in raw.items()}
    remainder = n_total - sum(floors.values())
    # Distribute the remainder to formats with the largest fractional parts
    fracs = sorted(
        format_weights.keys(), key=lambda f: raw[f] - floors[f], reverse=True
    )
    for fmt in fracs[:remainder]:
        floors[fmt] += 1
    return floors


def _run_opponent_bucket(
    opp_name: str,
    spec: OpponentEvalSpec,
    eval_cfg: EvalConfig,
    curriculum: CurriculumConfig,
    checkpoint_path: str,
    server_urls: List[str],
    device: str,
    run_tag: str,
) -> Tuple[BucketRunResult, List[RunningExternal]]:
    """Run one opponent across all curriculum formats. Returns
    (BucketRunResult, list of RunningExternal handles spawned during this
    run) so the caller's layer-1 cleanup can shut them down.
    """
    t0 = time.time()
    per_format: Dict[str, EvalResult] = {}
    externals: List[RunningExternal] = []

    battles_per_format = _split_battles_by_format(spec.n_battles, curriculum.format_weights)

    kwargs = _opponent_kwargs(opp_name, eval_cfg)

    for fmt in curriculum.battle_formats:
        n_for_fmt = battles_per_format.get(fmt, 0)
        if n_for_fmt <= 0:
            continue
        opponent_spec = parse_player_specification(opp_name, battle_format=fmt, **kwargs)
        running_ext: Optional[RunningExternal] = None
        if opp_name in _EXTERNAL_OPPONENTS:
            running_ext = launch_external_player(
                opponent_spec, server_url=server_urls[0], run_tag=run_tag
            )
            externals.append(running_ext)
        fmt_result = run_eval_parallel(
            p1_specification=parse_player_specification(
                "model", checkpoint_path=checkpoint_path, battle_format=fmt, device=device
            ),
            p2_specification=opponent_spec,
            n_battles=n_for_fmt,
            server_urls=server_urls,
            run_tag=run_tag,
            external_player=running_ext,
        )
        per_format[fmt] = fmt_result

    # Aggregate win rate using format weights restricted to formats we ran
    total_w = sum(curriculum.format_weights[f] for f in per_format)
    if total_w > 0:
        win_rate = sum(
            curriculum.format_weights[f] * (per_format[f].player1_wins / max(per_format[f].battles_played, 1))
            for f in per_format
        ) / total_w
    else:
        win_rate = 0.0

    n_total = sum(r.battles_played for r in per_format.values())
    return BucketRunResult(
        win_rate=win_rate,
        n_battles=n_total,
        per_format=per_format,
        wall_time_s=time.time() - t0,
    ), externals
```

Note: the exact signature of `parse_player_specification` and `run_eval_parallel` may differ slightly. Verify by reading [player_factory.py:140](../../src/elitefurretai/rl/analyze/player_factory.py#L140) and [evaluate.py:439](../../src/elitefurretai/rl/analyze/evaluate.py#L439), and adjust kwargs accordingly. The test mocks both functions so it does not care about real signatures.

- [ ] **Step 4.5: Run tests, verify pass**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestRunOpponentBucket -v
```

If the real `run_eval_parallel` signature differs, the mocks still match; only update the production call site if pyright fails.

- [ ] **Step 4.6: Quality gates**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py
```

- [ ] **Step 4.7: Commit**

```bash
git add src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add _run_opponent_bucket for per-opponent dispatch

Iterates curriculum formats, calls parse_player_specification with
opponent-name-prefixed kwargs, runs run_eval_parallel per format,
aggregates with format weights. Returns RunningExternal handles to
the caller so layer-1 cleanup can shut them down."
```

---

## Task 5: `baseline_eval.run` (driver + layer-1 cleanup)

**Files:**
- Modify: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (add top-level `run`)
- Test: extend [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py)

- [ ] **Step 5.1: Write failing tests — weight==0 skip + layer-1 cleanup**

Append:

```python
class TestRunDriver:
    @patch("elitefurretai.rl.analyze.baseline_eval._run_opponent_bucket")
    def test_weight_zero_bucket_is_skipped(self, mock_bucket):
        eval_cfg = EvalConfig(enabled=True)
        # Only foul_play has weight 0.0 in defaults
        from elitefurretai.rl.analyze.baseline_eval import run

        mock_bucket.return_value = (
            BucketRunResult(win_rate=0.8, n_battles=100, per_format={}, wall_time_s=1.0),
            [],
        )
        result = run(
            eval_cfg=eval_cfg, curriculum=_mock_curriculum_single_format(),
            checkpoint_path="/tmp/m.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        # _run_opponent_bucket called exactly 4 times (foul_play skipped)
        assert mock_bucket.call_count == 4
        called_opponents = {c.kwargs.get("opp_name") or c.args[0]
                            for c in mock_bucket.call_args_list}
        assert "foul_play" not in called_opponents
        assert "foul_play" not in result.per_bucket
        # Score is computed (4 buckets at 0.8 win_rate vs varying targets)
        assert isinstance(result.score, float)

    @patch("elitefurretai.rl.analyze.baseline_eval._run_opponent_bucket")
    def test_layer1_cleanup_on_exception(self, mock_bucket):
        from elitefurretai.rl.analyze.baseline_eval import run

        ext_a = MagicMock(spec=RunningExternal); ext_a.username = "X-A"
        ext_b = MagicMock(spec=RunningExternal); ext_b.username = "X-B"
        mock_bucket.side_effect = [
            (BucketRunResult(win_rate=0.8, n_battles=150, per_format={}, wall_time_s=1.0), [ext_a]),
            (BucketRunResult(win_rate=0.7, n_battles=150, per_format={}, wall_time_s=1.0), [ext_b]),
            RuntimeError("simulated mid-loop crash"),
        ]
        eval_cfg = EvalConfig(enabled=True)
        with pytest.raises(RuntimeError):
            run(
                eval_cfg=eval_cfg, curriculum=_mock_curriculum_single_format(),
                checkpoint_path="/tmp/m.pt",
                server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
            )
        # Both externals from completed buckets must have been shut down
        ext_a.shutdown.assert_called_once()
        ext_b.shutdown.assert_called_once()

    @patch("elitefurretai.rl.analyze.baseline_eval._run_opponent_bucket")
    def test_skipped_bucket_has_no_log_key(self, mock_bucket):
        from elitefurretai.rl.analyze.baseline_eval import build_eval_log_payload, run

        mock_bucket.return_value = (
            BucketRunResult(win_rate=0.8, n_battles=150, per_format={}, wall_time_s=1.0),
            [],
        )
        eval_cfg = EvalConfig(enabled=True)
        result = run(
            eval_cfg=eval_cfg, curriculum=_mock_curriculum_single_format(),
            checkpoint_path="/tmp/m.pt",
            server_urls=["localhost:8000"], device="cpu", run_tag="abcd",
        )
        payload = build_eval_log_payload(result, update_step=100, eval_cfg=eval_cfg)
        assert "eval/foul_play/win_rate" not in payload
```

The `build_eval_log_payload` test is forward-looking; it will fail at import until Task 6 lands. Mark that test with `@pytest.mark.skip(reason='build_eval_log_payload arrives in Task 6')` for now, and remove the marker in Task 6.

- [ ] **Step 5.2: Run failing tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestRunDriver -v
```

Expected: ImportError for `run`.

- [ ] **Step 5.3: Implement `run`**

Append to `baseline_eval.py`:

```python
import logging

logger = logging.getLogger(__name__)


def run(
    eval_cfg: EvalConfig,
    curriculum: CurriculumConfig,
    checkpoint_path: str,
    server_urls: List[str],
    device: str,
    run_tag: str,
) -> MultiBucketEvalResult:
    """Run one full multi-bucket eval pass.

    Iterates eval_cfg.opponents, skipping any with weight == 0.0 (no
    player construction at all for those). For each active opponent,
    dispatches to _run_opponent_bucket and aggregates the results into
    a MultiBucketEvalResult. compute_score is called over the active
    opponents to produce the scalar W&B sweep metric.

    Layer-1 cleanup: every RunningExternal spawned during the pass is
    shut down before this function returns, on every exit path.
    """
    t0 = time.time()
    per_bucket: Dict[str, BucketRunResult] = {}
    externals: List[RunningExternal] = []
    try:
        for opp_name, spec in eval_cfg.opponents.items():
            if spec.weight == 0.0:
                continue
            bucket_result, bucket_externals = _run_opponent_bucket(
                opp_name=opp_name,
                spec=spec,
                eval_cfg=eval_cfg,
                curriculum=curriculum,
                checkpoint_path=checkpoint_path,
                server_urls=server_urls,
                device=device,
                run_tag=run_tag,
            )
            per_bucket[opp_name] = bucket_result
            externals.extend(bucket_externals)
    finally:
        for ext in externals:
            try:
                ext.shutdown()
            except Exception:
                logger.exception("Failed to shutdown external player %s", ext.username)

    win_rates = {k: r.win_rate for k, r in per_bucket.items()}
    targets = {k: eval_cfg.opponents[k].target for k in per_bucket}
    weights = {k: eval_cfg.opponents[k].weight for k in per_bucket}
    score, breakdown = compute_score(win_rates, targets, weights, eval_cfg.surplus_alpha)
    return MultiBucketEvalResult(
        per_bucket=per_bucket,
        score=score,
        breakdown=breakdown,
        wall_time_s=time.time() - t0,
    )
```

- [ ] **Step 5.4: Run, verify pass**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestRunDriver -v
```

Expected: 3 passing (one skipped pending Task 6).

- [ ] **Step 5.5: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py
git add src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add baseline_eval.run with layer-1 cleanup contract

Iterates EvalConfig.opponents and skips weight=0 buckets (no
construction). On every exit path, every RunningExternal spawned
during the pass gets shutdown() called, with shutdown failures
logged and swallowed so cleanup drains as much as possible."
```

---

## Task 6: `build_eval_log_payload` (wandb payload + per-format aggregation)

**Files:**
- Modify: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (add `build_eval_log_payload`)
- Test: extend [unit_tests/rl/analyze/test_baseline_eval.py](../../unit_tests/rl/analyze/test_baseline_eval.py)

- [ ] **Step 6.1: Remove the `@pytest.mark.skip` from `test_skipped_bucket_has_no_log_key`** added in Task 5.

- [ ] **Step 6.2: Write failing tests for payload shape + per-format aggregation**

Append:

```python
class TestBuildEvalLogPayload:
    def _mk(self, win_rates_per_fmt: Dict[str, Dict[str, float]]):
        """Helper: per_bucket dict from a {opp: {fmt: win_rate}} map."""
        per_bucket = {}
        for opp, fmts in win_rates_per_fmt.items():
            per_format = {}
            n_total = 0
            for fmt, wr in fmts.items():
                n = 100
                per_format[fmt] = EvalResult(
                    player1_wins=int(wr * n), player2_wins=n - int(wr * n),
                    ties=0, battles_played=n,
                )
                n_total += n
            per_bucket[opp] = BucketRunResult(
                win_rate=sum(fmts.values()) / len(fmts),
                n_battles=n_total, per_format=per_format, wall_time_s=1.0,
            )
        return MultiBucketEvalResult(
            per_bucket=per_bucket, score=-3.0,
            breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 2.0},
            wall_time_s=10.0,
        )

    def test_payload_has_top_level_keys(self):
        from elitefurretai.rl.analyze.baseline_eval import build_eval_log_payload
        cfg = EvalConfig(enabled=True)
        r = self._mk({"max_damage": {"gen9vgc2023regc": 0.8}})
        p = build_eval_log_payload(r, update_step=200, eval_cfg=cfg)
        assert p["eval/score"] == -3.0
        assert p["eval/deficit_l2_pp"] == 5.0
        assert p["eval/surplus_sum_pp"] == 2.0
        assert p["eval/wall_time_s"] == 10.0
        assert p["eval/update_step"] == 200

    def test_payload_has_per_opponent_win_rate(self):
        from elitefurretai.rl.analyze.baseline_eval import build_eval_log_payload
        cfg = EvalConfig(enabled=True)
        r = self._mk({"max_damage": {"gen9vgc2023regc": 0.8}})
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        assert p["eval/max_damage/win_rate"] == pytest.approx(0.8)

    def test_per_format_uses_opponent_weighted_mean(self):
        from elitefurretai.rl.analyze.baseline_eval import build_eval_log_payload
        # Two opponents both ran on one format; cross-opponent weighted mean
        cfg = EvalConfig(enabled=True)
        # Set weights deliberately so the math is non-trivial
        cfg.opponents["max_damage"].weight = 2.0
        cfg.opponents["bc_player"].weight = 1.0
        r = self._mk({
            "max_damage": {"gen9vgc2023regc": 0.9},
            "bc_player":  {"gen9vgc2023regc": 0.6},
        })
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        # Cross-opponent weighted mean = (2*0.9 + 1*0.6) / (2+1) = 0.8
        assert p["eval/gen9vgc2023regc/win_rate"] == pytest.approx(0.8)

    def test_no_per_opponent_per_format_keys(self):
        from elitefurretai.rl.analyze.baseline_eval import build_eval_log_payload
        cfg = EvalConfig(enabled=True)
        r = self._mk({"max_damage": {"gen9vgc2023regc": 0.8, "gen9vgc2024regg": 0.7}})
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        # No nested per-opp-per-fmt keys (intentional design — design Section 4)
        assert "eval/max_damage/gen9vgc2023regc/win_rate" not in p
        assert "eval/max_damage/gen9vgc2024regg/win_rate" not in p
```

- [ ] **Step 6.3: Run failing tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestBuildEvalLogPayload -v
```

Expected: ImportError.

- [ ] **Step 6.4: Implement `build_eval_log_payload`**

Append:

```python
def build_eval_log_payload(
    result: MultiBucketEvalResult,
    update_step: int,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    """Shape the eval result for `wandb.log`. See design Section 4 for
    the key contract.

    Per-format aggregate: opponent-weight-weighted mean of per-(opp, fmt)
    win rates across active opponents, restricted to opponents that
    actually ran that format.
    """
    payload: Dict[str, Any] = {
        "eval/score": result.score,
        "eval/deficit_l2_pp": result.breakdown.get("deficit_l2_pp", 0.0),
        "eval/surplus_sum_pp": result.breakdown.get("surplus_sum_pp", 0.0),
        "eval/wall_time_s": result.wall_time_s,
        "eval/update_step": update_step,
    }
    # Per-opponent aggregate win rate
    for opp, bucket in result.per_bucket.items():
        payload[f"eval/{opp}/win_rate"] = bucket.win_rate

    # Per-format cross-opponent weighted aggregate
    all_formats = set()
    for bucket in result.per_bucket.values():
        all_formats.update(bucket.per_format.keys())
    for fmt in all_formats:
        weighted_sum = 0.0
        weight_total = 0.0
        for opp, bucket in result.per_bucket.items():
            if fmt not in bucket.per_format:
                continue
            w = eval_cfg.opponents[opp].weight
            fmt_result = bucket.per_format[fmt]
            n = max(fmt_result.battles_played, 1)
            wr = fmt_result.player1_wins / n
            weighted_sum += w * wr
            weight_total += w
        if weight_total > 0:
            payload[f"eval/{fmt}/win_rate"] = weighted_sum / weight_total
    return payload
```

- [ ] **Step 6.5: Run all tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py -v
```

Expected: all passing.

- [ ] **Step 6.6: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py
git add src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add build_eval_log_payload for wandb logging

Shapes MultiBucketEvalResult into wandb.log() keys. Per-format value
is the opponent-weight-weighted mean of per-(opp,fmt) win rates, so
the eval/<format>/win_rate metric uses the same weighting as the
score function."
```

---

## Task 7: Wire `_maybe_run_eval` into train.py (replaces `_maybe_run_foulplay_eval`)

**Files:**
- Modify: [src/elitefurretai/rl/train.py](../../src/elitefurretai/rl/train.py) — replace `_maybe_run_foulplay_eval` (line 665), update call site, remove `foulplay_eval` import
- Manual smoke test (no unit test — covered by integration)

- [ ] **Step 7.1: Identify call site for `_maybe_run_foulplay_eval`**

```bash
grep -n "_maybe_run_foulplay_eval\|foulplay_eval" src/elitefurretai/rl/train.py
```

Note the location of the call site and the import.

- [ ] **Step 7.2: Replace `_maybe_run_foulplay_eval` with `_maybe_run_eval`**

In `src/elitefurretai/rl/train.py`, delete the entire `_maybe_run_foulplay_eval` function (lines ~665-740) and replace with:

```python
def _maybe_run_eval(
    config: RNaDConfig,
    updates: int,
    checkpoint_path: str,
    server_ports: List[int],
    run_id: str,
) -> None:
    """Inline multi-bucket eval at checkpoint boundary. Pauses training.

    Replaces _maybe_run_foulplay_eval. Routes every active opponent
    (including foul_play when its weight > 0) through
    baseline_eval.run. Exceptions during eval are caught and logged so
    eval failures cannot kill a training run.
    """
    eval_cfg = config.eval
    if not eval_cfg.enabled:
        return
    if updates == 0 or updates % eval_cfg.eval_every_n_updates != 0:
        return

    server_urls = [f"localhost:{p}" for p in server_ports]
    run_tag = format((updates * 1664525 + 1013904223) % 65536, "04x")

    logger.info(
        "[Update %d] Running eval (active opponents: %s)",
        updates,
        sorted(k for k, v in eval_cfg.opponents.items() if v.weight > 0),
    )
    try:
        result = baseline_eval.run(
            eval_cfg=eval_cfg,
            curriculum=config.curriculum,
            checkpoint_path=checkpoint_path,
            server_urls=server_urls,
            device=config.hardware.device,
            run_tag=run_tag,
        )
    except Exception:
        logger.exception("[Update %d] Eval crashed; training continues", updates)
        return

    logger.info(
        "[Update %d] Eval: score=%.2f deficit_l2_pp=%.2f surplus_sum_pp=%.2f wall=%.1fs",
        updates,
        result.score,
        result.breakdown.get("deficit_l2_pp", 0.0),
        result.breakdown.get("surplus_sum_pp", 0.0),
        result.wall_time_s,
    )
    if config.training.use_wandb:
        payload = baseline_eval.build_eval_log_payload(
            result, update_step=updates, eval_cfg=eval_cfg
        )
        wandb.log(payload)
```

- [ ] **Step 7.3: Update import + call site**

Replace this import (top of train.py):

```python
import elitefurretai.rl.analyze.foulplay_eval as _foulplay_eval_mod
```

with:

```python
import elitefurretai.rl.analyze.baseline_eval as baseline_eval
```

Find the call site (was `_maybe_run_foulplay_eval(...)` somewhere in the training loop) and rename to `_maybe_run_eval(...)`. The signature is the same so the call doesn't need restructuring.

- [ ] **Step 7.4: Run the full test suite to catch regressions**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/ -q
```

Expected: all pass.

- [ ] **Step 7.5: Quality gates**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train.py && \
  ruff format src/elitefurretai/rl/train.py --check && \
  pyright src/elitefurretai/rl/train.py
```

- [ ] **Step 7.6: Manual smoke check — config loads without raising**

```bash
source ../venv/bin/activate && python -c "
from elitefurretai.rl.config import RNaDConfig
c = RNaDConfig.from_yaml('src/elitefurretai/rl/configs/may26.yaml')
print('eval.enabled =', c.eval.enabled)
print('eval.opponents.keys() =', sorted(c.eval.opponents.keys()))
"
```

Expected: prints `eval.enabled = <whatever the YAML says>` and the five opponent canonical names.

- [ ] **Step 7.7: Commit**

```bash
git add src/elitefurretai/rl/train.py
git commit -m "feat(rl-train): route eval through baseline_eval.run

Replaces _maybe_run_foulplay_eval with _maybe_run_eval, routing every
active opponent (including foul_play when its weight > 0) through
the unified baseline_eval driver. Eval failures continue to be
swallowed so they cannot kill a training run."
```

---

## Task 8: Reinforce train.main layer-2 cleanup (atexit + orphan sweep)

**Files:**
- Modify: [src/elitefurretai/rl/train.py:742](../../src/elitefurretai/rl/train.py#L742) — `main()` body + finally block (~1305)

Existing `train.main` already has a `try/finally` that calls `shutdown_showdown_servers` (line 1350) and `wandb.finish` (line 1338). Layer 2 needs an additional `atexit.register` registration and an orphan-process cleanup sweep that runs at the end of the `finally` block, looking for processes whose Showdown username carries this run's `run_id`.

- [ ] **Step 8.1: Read existing main() try/finally block**

```bash
sed -n '742,780p' src/elitefurretai/rl/train.py
sed -n '1295,1355p' src/elitefurretai/rl/train.py
```

Confirm where `server_processes` is set and where `shutdown_showdown_servers` is called.

- [ ] **Step 8.2: Add `atexit.register` right after server launch**

After the line `server_processes: List[subprocess.Popen] = launch_showdown_servers(...)` (~line 774), insert:

```python
import atexit
atexit.register(shutdown_showdown_servers, server_processes)
```

`atexit` handlers run on interpreter exit even when the `try/finally` is bypassed (e.g. by an OS-level SIGKILL on a different thread). `shutdown_showdown_servers` is idempotent (it checks `process.poll()` before signaling), so calling it twice is safe.

- [ ] **Step 8.3: Add orphan-cleanup sweep inside the final `finally`**

In the existing `finally:` block (~line 1305), after `shutdown_showdown_servers(server_processes)`, add:

```python
        # Layer-2 cleanup sweep: catch any showdown / vgcbench / foulplay
        # subprocess that survived the per-process shutdown above, by
        # matching this run's run_id embedded in usernames or process args.
        _orphan_cleanup_sweep(run_id)
```

Add the helper near the top of train.py (after imports):

```python
def _orphan_cleanup_sweep(run_id: str) -> None:
    """pgrep -based fallback: kill any showdown / vgcbench / foulplay
    process whose command line still mentions this run's run_id. Idempotent.
    """
    import shutil
    if shutil.which("pgrep") is None:
        logger.warning("pgrep not available; skipping orphan cleanup sweep")
        return
    patterns = [
        f"showdown.*{run_id}",
        f"vgcbench.*{run_id}",
        f"foulplay.*{run_id}",
    ]
    for pat in patterns:
        try:
            out = subprocess.run(
                ["pgrep", "-af", pat], capture_output=True, text=True
            )
            for line in out.stdout.strip().splitlines():
                pid_str = line.split(None, 1)[0]
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                logger.warning("Orphan cleanup: SIGTERM pid=%d (matched %r)", pid, pat)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        except Exception:
            logger.exception("Orphan cleanup sweep failed for pattern %r", pat)
```

If `signal` or `subprocess` is not already imported at top of train.py, add the imports.

- [ ] **Step 8.4: Type check + lint**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train.py && \
  ruff format src/elitefurretai/rl/train.py --check && \
  pyright src/elitefurretai/rl/train.py
```

- [ ] **Step 8.5: Manual smoke check — train.main parses and loads with eval enabled**

```bash
source ../venv/bin/activate && python -c "
import elitefurretai.rl.train as t
import elitefurretai.rl.analyze.baseline_eval as be
print('_maybe_run_eval exists:', hasattr(t, '_maybe_run_eval'))
print('_orphan_cleanup_sweep exists:', hasattr(t, '_orphan_cleanup_sweep'))
print('baseline_eval.run exists:', hasattr(be, 'run'))
"
```

Expected: all three `True`.

- [ ] **Step 8.6: Commit**

```bash
git add src/elitefurretai/rl/train.py
git commit -m "feat(rl-train): reinforce layer-2 cleanup with atexit + orphan sweep

atexit registration handles hard-crash exit paths the try/finally
misses. _orphan_cleanup_sweep uses pgrep to catch showdown/vgcbench/
foulplay subprocesses that survived per-process shutdown by matching
this run's run_id in their command lines."
```

---

## Task 9: Standalone CLI — `baseline_eval.main`

**Files:**
- Modify: [src/elitefurretai/rl/analyze/baseline_eval.py](../../src/elitefurretai/rl/analyze/baseline_eval.py) (add `main()`)

The CLI uses one `--config` flag pointing to a full RNaDConfig YAML (reads `config.curriculum`, `config.eval`, `config.hardware.device`), launches its own Showdown servers, runs `baseline_eval.run`, writes a JSON result file specified by `--output`, tears down servers in `finally`, exits. No wandb.

- [ ] **Step 9.1: Write minimal failing test for JSON output shape**

Append to `unit_tests/rl/analyze/test_baseline_eval.py`:

```python
import json


class TestStandaloneCLIOutput:
    def test_serialize_result_to_json(self, tmp_path):
        from elitefurretai.rl.analyze.baseline_eval import (
            _serialize_result_to_dict,
        )
        ev = EvalResult(player1_wins=80, player2_wins=20, ties=0, battles_played=100)
        bucket = BucketRunResult(
            win_rate=0.80, n_battles=100,
            per_format={"gen9vgc2023regc": ev}, wall_time_s=1.0,
        )
        result = MultiBucketEvalResult(
            per_bucket={"max_damage": bucket},
            score=-5.0,
            breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 0.0},
            wall_time_s=2.0,
        )
        d = _serialize_result_to_dict(
            result=result,
            checkpoint_path="/tmp/m.pt",
            config_path="/tmp/cfg.yaml",
            run_tag="abcd",
        )
        # Required fields present
        assert d["checkpoint_path"] == "/tmp/m.pt"
        assert d["config_path"] == "/tmp/cfg.yaml"
        assert d["run_tag"] == "abcd"
        assert "timestamp_utc" in d
        assert d["score"] == -5.0
        assert d["breakdown"]["deficit_l2_pp"] == 5.0
        assert d["per_opponent"]["max_damage"]["win_rate"] == 0.80
        assert d["per_opponent"]["max_damage"]["n_battles"] == 100
        assert d["per_opponent"]["max_damage"]["per_format"]["gen9vgc2023regc"]["win_rate"] == 0.80
        # Must be JSON-serializable
        json.dumps(d)
```

- [ ] **Step 9.2: Run, verify failure**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py::TestStandaloneCLIOutput -v
```

Expected: ImportError for `_serialize_result_to_dict`.

- [ ] **Step 9.3: Add `_serialize_result_to_dict` + `main()`**

Append to `baseline_eval.py`:

```python
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.config import RNaDConfig


def _serialize_result_to_dict(
    result: MultiBucketEvalResult,
    checkpoint_path: str,
    config_path: str,
    run_tag: str,
) -> Dict[str, Any]:
    """Convert MultiBucketEvalResult to a JSON-safe dict matching the
    schema documented in design Section 8.
    """
    per_opponent: Dict[str, Any] = {}
    all_formats = set()
    weight_by_opp = {}  # used to recompute per-format cross-opp aggregate offline
    for opp, bucket in result.per_bucket.items():
        per_format_out = {}
        for fmt, ev in bucket.per_format.items():
            n = max(ev.battles_played, 1)
            per_format_out[fmt] = {
                "win_rate": ev.player1_wins / n,
                "n_battles": ev.battles_played,
            }
            all_formats.add(fmt)
        per_opponent[opp] = {
            "win_rate": bucket.win_rate,
            "n_battles": bucket.n_battles,
            "wall_time_s": bucket.wall_time_s,
            "per_format": per_format_out,
        }

    return {
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "score": result.score,
        "breakdown": dict(result.breakdown),
        "wall_time_s": result.wall_time_s,
        "per_opponent": per_opponent,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone multi-bucket eval against configured baselines."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument(
        "--config", required=True, help="Path to full RNaDConfig YAML"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write JSON result file"
    )
    parser.add_argument("--num-servers", type=int, default=4)
    parser.add_argument("--server-port-start", type=int, default=8000)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    config = RNaDConfig.from_yaml(args.config)
    run_tag = format(int(time.time() * 1000) & 0xFFFF, "04x")

    server_processes = launch_showdown_servers(
        num_servers=args.num_servers, start_port=args.server_port_start
    )
    server_urls = [f"localhost:{args.server_port_start + i}" for i in range(args.num_servers)]
    try:
        result = run(
            eval_cfg=config.eval,
            curriculum=config.curriculum,
            checkpoint_path=args.checkpoint,
            server_urls=server_urls,
            device=config.hardware.device,
            run_tag=run_tag,
        )
        out_dict = _serialize_result_to_dict(
            result=result,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            run_tag=run_tag,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Wrote {args.output}")
        print(f"score={result.score:.2f}  wall_time_s={result.wall_time_s:.1f}")
        for opp, bucket in result.per_bucket.items():
            print(f"  {opp}: win_rate={bucket.win_rate:.3f}  n={bucket.n_battles}")
    finally:
        shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
```

If the real `launch_showdown_servers` signature uses different kwargs, adjust them to match. Look at how train.py calls it (line ~774) and mirror.

- [ ] **Step 9.4: Run tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_baseline_eval.py -v
```

- [ ] **Step 9.5: Quality gates**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ && \
  ruff format src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/ --check && \
  pyright src/elitefurretai/rl/analyze/baseline_eval.py
```

- [ ] **Step 9.6: Commit**

```bash
git add src/elitefurretai/rl/analyze/baseline_eval.py unit_tests/rl/analyze/test_baseline_eval.py
git commit -m "feat(rl): add standalone baseline_eval CLI with JSON output

Single --config flag takes a full RNaDConfig YAML (same format
training uses). Writes results to --output as JSON; no wandb. Server
launch + shutdown wrapped in try/finally so showdown processes
exit cleanly on any error path."
```

---

## Task 10: Sweep YAML loader + dotted-key config patcher

**Files:**
- Create: [src/elitefurretai/rl/train_sweep.py](../../src/elitefurretai/rl/train_sweep.py) (load_sweep_config + _write_patched_config)
- Create: [unit_tests/rl/test_train_sweep.py](../../unit_tests/rl/test_train_sweep.py)

- [ ] **Step 10.1: Write failing tests for YAML loader**

Create `unit_tests/rl/test_train_sweep.py`:

```python
# -*- coding: utf-8 -*-
"""Unit tests for train_sweep (YAML loader, dotted-key patching, termination)."""

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
        # Minimal valid RNaDConfig-shape YAML
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
```

- [ ] **Step 10.2: Run failing tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/test_train_sweep.py -v
```

Expected: ImportError for `train_sweep`.

- [ ] **Step 10.3: Create train_sweep.py with loader + patcher**

Create `src/elitefurretai/rl/train_sweep.py`:

```python
# -*- coding: utf-8 -*-
"""W&B Bayesian sweep harness for RL training.

Each sweep agent run spawns `python -m elitefurretai.rl.train` as a
subprocess with WANDB_RUN_ID inherited via env so the subprocess
reattaches to the sweep's wandb run. Subprocess isolation guarantees
that DataLoader workers, Showdown servers, and external eval
subprocesses from one run cannot leak into the next.

Three-layer cleanup contract: layer 1 (per-eval, in baseline_eval),
layer 2 (per-train-process, in train.main), layer 3 (per-sweep-agent,
here). See planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

import yaml

import wandb

logger = logging.getLogger(__name__)


# Filled in by main() before wandb.agent dispatches to sweep_train().
BASE_CONFIG_PATH: str = ""
EVAL_OVERRIDES: Dict[str, Any] = {}
PATCHED_CONFIG_DIR: str = ""


def load_sweep_config(path: str) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Parse the sweep YAML into (wandb_sweep_dict, base_config_path,
    eval_overrides).

    Required top-level keys: `base_config`, `sweep`. `eval_overrides` is
    optional and defaults to an empty dict.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if "base_config" not in data:
        raise KeyError("sweep YAML missing required key 'base_config'")
    if "sweep" not in data:
        raise KeyError("sweep YAML missing required key 'sweep'")
    return data["sweep"], data["base_config"], data.get("eval_overrides", {})


def _set_dotted(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Patch `dotted_key` into the nested `target` dict, creating
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

    Loads base_config_path, applies eval_overrides first, then
    sweep_params on top (sweep wins on key collision). Writes to
    `<out_dir>/<run_name>.yaml` and returns the path.
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
```

- [ ] **Step 10.4: Run, verify pass**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/test_train_sweep.py -v
```

- [ ] **Step 10.5: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py && \
  ruff format src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py --check && \
  pyright src/elitefurretai/rl/train_sweep.py
git add src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py
git commit -m "feat(rl): sweep YAML loader + dotted-key config patcher

load_sweep_config returns (sweep_dict, base_config_path,
eval_overrides). _write_patched_config applies eval_overrides then
sweep_params (sweep wins on collision) into the nested config dict,
writing a per-run patched YAML for the train subprocess to consume."
```

---

## Task 11: Layer-3 termination + orphan cleanup in train_sweep

**Files:**
- Modify: [src/elitefurretai/rl/train_sweep.py](../../src/elitefurretai/rl/train_sweep.py)
- Test: extend [unit_tests/rl/test_train_sweep.py](../../unit_tests/rl/test_train_sweep.py)

- [ ] **Step 11.1: Write failing tests for `_terminate_with_grace`**

Append to `unit_tests/rl/test_train_sweep.py`:

```python
import os
import signal
import subprocess
import sys
import time


class TestTerminateWithGrace:
    def test_clean_exit_on_sigterm(self):
        """Child that respects SIGTERM exits during the grace period."""
        from elitefurretai.rl.train_sweep import _terminate_with_grace
        # Spawns a Python that sleeps 10s but exits on SIGTERM
        code = "import signal,sys,time; signal.signal(signal.SIGTERM, lambda *a: sys.exit(0)); time.sleep(10)"
        proc = subprocess.Popen([sys.executable, "-c", code], preexec_fn=os.setsid)
        time.sleep(0.5)  # let it install the handler
        _terminate_with_grace(proc, timeout=3)
        assert proc.poll() is not None  # exited
        assert proc.returncode == 0

    def test_sigkill_escalation_for_unresponsive_child(self):
        """Child that ignores SIGTERM gets escalated to SIGKILL."""
        from elitefurretai.rl.train_sweep import _terminate_with_grace
        code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-c", code], preexec_fn=os.setsid)
        time.sleep(0.5)
        t0 = time.time()
        _terminate_with_grace(proc, timeout=2)
        elapsed = time.time() - t0
        assert proc.poll() is not None
        # Should have escalated within ~timeout seconds, not waited the full 60
        assert elapsed < 8.0
        assert proc.returncode in (-signal.SIGKILL, signal.SIGKILL + 128)  # killed by SIGKILL

    def test_already_exited_child_is_noop(self):
        from elitefurretai.rl.train_sweep import _terminate_with_grace
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        _terminate_with_grace(proc, timeout=1)  # should not raise
        assert proc.returncode == 0
```

- [ ] **Step 11.2: Run, verify failures**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/test_train_sweep.py::TestTerminateWithGrace -v
```

- [ ] **Step 11.3: Implement `_terminate_with_grace` + `_sweep_orphan_cleanup`**

Append to `train_sweep.py`:

```python
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
    for pat in (f"showdown.*{run_name}", f"vgcbench.*{run_name}", f"foulplay.*{run_name}"):
        try:
            out = subprocess.run(
                ["pgrep", "-af", pat], capture_output=True, text=True
            )
            for line in out.stdout.strip().splitlines():
                pid_str = line.split(None, 1)[0]
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.warning("Sweep orphan cleanup: SIGTERM pid=%d (matched %r)", pid, pat)
                except ProcessLookupError:
                    pass
        except Exception:
            logger.exception("Sweep orphan cleanup failed for pattern %r", pat)
```

- [ ] **Step 11.4: Run tests**

```bash
source ../venv/bin/activate && pytest unit_tests/rl/test_train_sweep.py -v
```

Some subprocess tests can be flaky on overloaded CI runners. If `test_sigkill_escalation_for_unresponsive_child` flakes, bump the assertion timeout to 12s and retry.

- [ ] **Step 11.5: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py && \
  ruff format src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py --check && \
  pyright src/elitefurretai/rl/train_sweep.py
git add src/elitefurretai/rl/train_sweep.py unit_tests/rl/test_train_sweep.py
git commit -m "feat(rl-sweep): add layer-3 _terminate_with_grace + orphan cleanup

SIGTERM → wait → SIGKILL escalation on the train subprocess, plus a
pgrep fallback to catch showdown/vgcbench/foulplay processes that
survived. Keeps Showdown port conflicts from leaking into the next
sweep run."
```

---

## Task 12: `sweep_train` agent function

**Files:**
- Modify: [src/elitefurretai/rl/train_sweep.py](../../src/elitefurretai/rl/train_sweep.py)

`sweep_train` is called in-process by `wandb.agent` per sweep run. It spawns `python -m elitefurretai.rl.train` as a subprocess with `WANDB_RUN_ID` env so the subprocess reattaches to the sweep's run, waits, then runs layer-3 cleanup.

- [ ] **Step 12.1: Implement `sweep_train`**

Append to `train_sweep.py`:

```python
CLEANUP_TIMEOUT_S = 30


def sweep_train() -> None:
    """Called by wandb.agent per sweep run. wandb.init has already
    happened (the agent does it before calling this function), so the
    run name and config are available on wandb.run.

    Spawns `python -m elitefurretai.rl.train` as a subprocess with the
    sweep run's wandb run id in the env. The subprocess's own
    `wandb.init()` reattaches to that run, so metrics it logs land
    under the correct sweep run.
    """
    sampled = dict(wandb.config)
    run_name = wandb.run.name if wandb.run and wandb.run.name else f"sweep-{int(time.time())}"
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
        # do NOT call wandb.finish — wandb.agent owns the run lifecycle
```

- [ ] **Step 12.2: Run linter and type checker (no new unit tests; covered by smoke test in Task 13)**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train_sweep.py && \
  ruff format src/elitefurretai/rl/train_sweep.py --check && \
  pyright src/elitefurretai/rl/train_sweep.py
```

- [ ] **Step 12.3: Commit**

```bash
git add src/elitefurretai/rl/train_sweep.py
git commit -m "feat(rl-sweep): add sweep_train agent function

Spawns train.py as a subprocess per sweep run with WANDB_RUN_ID in
env so the subprocess reattaches to the sweep's wandb run. Subprocess
isolation guarantees Showdown servers, DataLoader workers, and
external eval subprocesses can't leak across runs. Layer-3 cleanup
runs unconditionally in the finally."
```

---

## Task 13: `train_sweep.main` (CLI entry point)

**Files:**
- Modify: [src/elitefurretai/rl/train_sweep.py](../../src/elitefurretai/rl/train_sweep.py)

- [ ] **Step 13.1: Implement `main()`**

Append to `train_sweep.py`:

```python
def main() -> None:
    global BASE_CONFIG_PATH, EVAL_OVERRIDES, PATCHED_CONFIG_DIR
    parser = argparse.ArgumentParser(
        description="W&B Bayesian sweep over RL hyperparameters."
    )
    parser.add_argument("--config", required=True, help="Path to sweep YAML")
    parser.add_argument("--count", type=int, default=40, help="Number of runs")
    parser.add_argument(
        "--sweep-id", default=None,
        help="Existing sweep ID to add runs to (skips wandb.sweep create)",
    )
    parser.add_argument(
        "--project", default="elitefurretai-rnad-sweep",
        help="Wandb project for sweep + runs",
    )
    parser.add_argument(
        "--patched-config-dir", default="/tmp/efai-sweep-configs",
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
```

- [ ] **Step 13.2: Smoke test — CLI parses --help cleanly**

```bash
source ../venv/bin/activate && python -m elitefurretai.rl.train_sweep --help
```

Expected: prints argparse help with the listed flags, exit 0.

- [ ] **Step 13.3: Quality gates + commit**

```bash
source ../venv/bin/activate && \
  ruff check src/elitefurretai/rl/train_sweep.py && \
  ruff format src/elitefurretai/rl/train_sweep.py --check && \
  pyright src/elitefurretai/rl/train_sweep.py
git add src/elitefurretai/rl/train_sweep.py
git commit -m "feat(rl-sweep): add train_sweep.main CLI entry point

Argparse + wandb.sweep + wandb.agent. Supports --sweep-id for
resuming existing sweeps. Sets module-level BASE_CONFIG_PATH etc.
before dispatching to sweep_train via wandb.agent."
```

---

## Task 14: Author `initial_rl_baselines.yaml`

**Files:**
- Create: [src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml](../../src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml)

- [ ] **Step 14.1: Create sweep_configs directory and write YAML**

```bash
mkdir -p src/elitefurretai/rl/sweep_configs
```

Create `src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml`:

```yaml
# ===========================================================================
# initial_rl_baselines.yaml — first RL hyperparameter sweep targeting
# Stage II graduation thresholds (80% vs SHP/MaxDamage/BC, 60% vs VGCBench).
#
# See planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md and
# planning/stage2/2026-05-29-21-30-rl-wandb-sweep-eval-implementation-plan.md.
#
# The sweep parameter list below is intentionally minimal; the implementation
# plan task 14 includes a follow-up to expand it (gradient clip, batch sizes,
# target update rate, etc.) with Cayman before launching at scale.
# ===========================================================================

base_config: "src/elitefurretai/rl/configs/may26.yaml"

# Applied to every sweep run before sweep_params. These turn eval on, point
# wandb at the sweep project, and lock the surplus_alpha that the score
# function uses.
eval_overrides:
  training.wandb_project: "elitefurretai-rnad-sweep"
  eval.enabled: true
  eval.eval_every_n_updates: 500
  eval.surplus_alpha: 1.0
  # FoulPlay shipped off until its subprocess is stable; flip to 1.0 here
  # when ready.
  eval.opponents.foul_play.weight: 0.0

sweep:
  method: bayes
  metric:
    name: "eval/score"
    goal: maximize
  early_terminate:
    type: hyperband
    min_iter: 3
    eta: 2
  parameters:
    learner.learning_rate:
      min: 1.0e-5
      max: 1.0e-3
      distribution: log_uniform_values
    learner.eta:
      values: [0.0, 0.001, 0.01, 0.1]
    learner.entropy_weight:
      values: [0.0, 0.001, 0.01, 0.05]
```

- [ ] **Step 14.2: Smoke-test the loader can parse it**

```bash
source ../venv/bin/activate && python -c "
from elitefurretai.rl.train_sweep import load_sweep_config
s, b, e = load_sweep_config('src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml')
print('sweep keys:', sorted(s.keys()))
print('base_config:', b)
print('eval_overrides:', e)
"
```

Expected: prints `sweep keys: ['early_terminate', 'method', 'metric', 'parameters']`, the base config path, and the eval_overrides dict.

- [ ] **Step 14.3: Commit**

```bash
git add src/elitefurretai/rl/sweep_configs/initial_rl_baselines.yaml
git commit -m "feat(rl-sweep): add initial sweep config for Stage II baselines

Minimal sweep parameter list (lr, eta, entropy_weight); will be
expanded before the first at-scale launch. eval_overrides turn eval
on, lock surplus_alpha=1.0, and keep foul_play disabled."
```

---

## Task 15: Validate, then delete `foulplay_eval.py`

This task is intentionally last: the prior tasks must produce a driver that reproduces existing FoulPlay eval behavior on a real checkpoint before the parallel driver gets deleted.

**Files:**
- Manual validation run + comparison
- Delete: [src/elitefurretai/rl/analyze/foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py)
- Update: any imports referencing `analyze.foulplay_eval`

- [ ] **Step 15.1: Find remaining `foulplay_eval` references**

```bash
grep -rn "foulplay_eval\|analyze.foulplay_eval" src/elitefurretai/ unit_tests/ | grep -v "planning/" | grep -v ".pyc"
```

Expected: should already be limited to the file itself after Tasks 2 and 7. Note any stragglers (e.g. docs, scripts).

- [ ] **Step 15.2: Run an at-scale standalone eval with FoulPlay enabled, against an existing checkpoint**

Author a one-off test config:

```yaml
# /tmp/foulplay_validation.yaml — overrides from may26.yaml turning ONLY
# foul_play on (other baselines off) so the eval result is directly
# comparable to the legacy FoulPlay eval driver's output for the same
# checkpoint + n_battles.
eval:
  enabled: true
  eval_every_n_updates: 1
  surplus_alpha: 1.0
  opponents:
    simple_heuristic_baseline: {weight: 0.0, target: 0.8, n_battles: 1}
    max_damage:                {weight: 0.0, target: 0.8, n_battles: 1}
    vgc_bench:                 {weight: 0.0, target: 0.6, n_battles: 1}
    bc_player:                 {weight: 0.0, target: 0.8, n_battles: 1}
    foul_play:                 {weight: 1.0, target: 0.5, n_battles: 40}
```

Use the standalone CLI:

```bash
source ../venv/bin/activate && python -m elitefurretai.rl.analyze.baseline_eval \
    --checkpoint data/models/supervised/cool-bee-85-finetune_best.pt \
    --config /tmp/foulplay_validation.yaml \
    --output /tmp/foulplay_validation.json \
    --num-servers 4
```

- [ ] **Step 15.3: Compare against legacy driver**

In a separate shell, run the legacy FoulPlay CLI on the same checkpoint with the same n_battles (if the legacy driver supports it):

```bash
source ../venv/bin/activate && python -m elitefurretai.rl.analyze.foulplay_eval \
    --checkpoint data/models/supervised/cool-bee-85-finetune_best.pt \
    --n-battles 40 --search-time-ms 750 --parallelism 8
```

Win rates should be within statistical noise (40 battles ⇒ ±~16% at 1σ). If win rates differ by more than ~25 percentage points, investigate before deleting the legacy driver.

- [ ] **Step 15.4: Delete `foulplay_eval.py`**

```bash
git rm src/elitefurretai/rl/analyze/foulplay_eval.py
```

- [ ] **Step 15.5: Update any stragglers found in Step 15.1**

For each remaining reference, update to use `baseline_eval.run` with an `EvalConfig` that has FoulPlay enabled. If the reference is in a script under [src/elitefurretai/scripts/](../../src/elitefurretai/scripts/), it is intentionally excluded from quality gates so just update the import.

- [ ] **Step 15.6: Run full quality gates one more time**

```bash
source ../venv/bin/activate && \
  ruff check src unit_tests && \
  ruff format src unit_tests --check && \
  pyright src unit_tests && \
  pytest unit_tests -q
```

- [ ] **Step 15.7: Commit**

```bash
git add -u
git commit -m "refactor(rl): remove foulplay_eval.py after unified driver validation

The unified baseline_eval driver reproduces FoulPlay eval behavior
(validated against a sample checkpoint with weight=1.0 on foul_play
and other opponents disabled). The parallel driver is now dead
code; removing it.

Closes the migration started in:
- planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md
- planning/stage2/2026-05-29-21-30-rl-wandb-sweep-eval-implementation-plan.md"
```

---

## Self-Review Notes

**Spec coverage:** Every section of the design doc maps to one or more tasks. Section 1 (score) → Task 1; Section 2 (EvalConfig) → Task 2; Section 3 (bucket dispatch) → Tasks 3, 4, 5; Section 4 (train.py integration + payload) → Tasks 6, 7; Section 5 (cleanup contract) → Tasks 5 (layer 1), 8 (layer 2), 11 (layer 3); Section 6 (sweep harness) → Tasks 10, 11, 12, 13; Section 7 (YAML schema) → Task 14; Section 8 (standalone CLI) → Task 9; Section 9 (files removed) → Task 15; Section 10 (testing) → distributed through Tasks 1-11. The empirical `eval_every_n_updates` measurement noted in the design's Open Question is left as a follow-up in Task 14's docstring rather than a task — measuring requires a working sweep, which only exists after Task 14.

**Placeholder scan:** All code blocks contain executable Python (no `# TODO`, no `pass # implement me`). All test code shows actual assertions, not "write tests for the above." Task 14's docstring explicitly notes the sweep parameter list is minimal pending follow-up with Cayman — this is a flagged scope boundary, not a placeholder.

**Type consistency:** `compute_score` signature is `(win_rates, targets, weights, surplus_alpha) -> (float, Dict[str, float])` in Task 1, reused identically in Task 5. `EvalConfig`, `OpponentEvalSpec`, `BucketRunResult`, `MultiBucketEvalResult` types defined in Tasks 2-3 are referenced consistently throughout Tasks 4-9. `_run_opponent_bucket` returns `Tuple[BucketRunResult, List[RunningExternal]]` in Task 4 and is unpacked the same way in Task 5. `build_eval_log_payload(result, update_step, eval_cfg)` signature in Task 6 matches the call site in Task 7.
