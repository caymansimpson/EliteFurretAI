# RL W&B Sweep with Multi-Baseline Eval Metric — Design

## Context

Stage II graduation requires simultaneously beating four reference opponents
(`simple_heuristic_baseline`, `max_damage`, `vgc_bench_baseline`, `bc_player`)
at thresholds tracked in [planning/stage2/2026-05-16-21-30-stage2-graduation-criteria.md](2026-05-16-21-30-stage2-graduation-criteria.md).
Today we tune RL hyperparameters by hand and read multi-axis baseline
performance off ad-hoc evals. Cayman wants a Weights & Biases sweep that
optimizes a single scalar derived from these baseline win rates, so the
hyperparameter search can be Bayesian rather than manual.

The targets for this sweep are stricter than the graduation floor: 80% vs
SHP, 80% vs MaxDamage, 60% vs VGCBench, 80% vs BCPlayer (graduation is 60%
across all four).

This design also collapses the existing per-checkpoint FoulPlay eval driver
([src/elitefurretai/rl/analyze/foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py))
into the same unified driver. FoulPlay becomes one more opponent bucket
rather than its own subsystem. While FoulPlay is broken, its bucket weight
defaults to 0.0 and its subprocess machinery is skipped.

## Before State

- W&B is already wired into RL training via [src/elitefurretai/rl/train.py:73](../../src/elitefurretai/rl/train.py#L73)
  and [config.py:767-771](../../src/elitefurretai/rl/config.py#L767).
- A per-checkpoint inline eval exists for FoulPlay only:
  [`_maybe_run_foulplay_eval`](../../src/elitefurretai/rl/train.py#L666) pauses
  training, runs one FoulPlay eval cycle per curriculum format, aggregates
  with format weights, and logs to W&B.
- The four baselines exist in [opponents.py](../../src/elitefurretai/rl/opponents.py)
  with canonical names `simple_heuristic_baseline`, `max_damage`, `bc_player`,
  `vgc_bench_baseline`.
- VGCBench and FoulPlay both already run as external subprocesses via
  [`launch_external_player`](../../src/elitefurretai/rl/analyze/player_factory.py#L331)
  and share the `_EXTERNAL_BASELINES` abstraction at
  [player_factory.py:59](../../src/elitefurretai/rl/analyze/player_factory.py#L59).
- The supervised side already has a mature W&B sweep harness
  ([train_sweep.py](../../src/elitefurretai/supervised/train_sweep.py),
  [sweep_configs/](../../src/elitefurretai/supervised/sweep_configs/)) with YAML config,
  Hyperband early termination, and dotted-key parameter patching. The
  RL side has no equivalent yet.

## Problem

Three concrete gaps:

1. There is no scalar sweep metric that captures multi-baseline
   performance. Sweep agents need a single number to optimize.
2. There is no per-checkpoint eval pass that runs all four baselines.
   Today only FoulPlay is wired in.
3. There is no sweep harness for RL: no agent script, no YAML schema, no
   subprocess isolation between sweep runs, no orphan-process cleanup.

A fourth implicit gap is that FoulPlay eval lives in its own parallel
driver. Once a unified multi-bucket eval driver exists, the FoulPlay
driver is redundant and should be removed.

## Solution

### 1. Score function (scalar sweep metric)

A pure function in `baseline_eval.py`:

```python
def compute_score(
    win_rates: Dict[str, float],     # [0,1]
    targets:   Dict[str, float],     # [0,1]
    weights:   Dict[str, float],
    surplus_alpha: float,
) -> Tuple[float, Dict[str, float]]:
    deficit_pp = {k: max(0.0, 100*(targets[k] - win_rates[k])) for k in weights}
    surplus_pp = {k: max(0.0, 100*(win_rates[k] - targets[k])) for k in weights}
    deficit_term = sum(weights[k] * deficit_pp[k] ** 2 for k in weights)
    surplus_term = sum(weights[k] * surplus_pp[k]      for k in weights)
    score = surplus_alpha * surplus_term - deficit_term
    return score, {"deficit_l2_pp": deficit_term, "surplus_sum_pp": surplus_term}
```

Hinge L2 on the deficit side preserves the "missing the floor by a lot is
super bad" property. Linear L1 surplus contributes additive bonus once
floors are met. Percentage-point scaling resolves the scale mismatch
between L2 on [0,1] (which produces tiny numbers) and L1 on [0,1]: at pp
scale, a 10pp deficit contributes 100 to the negative side while a 10pp
surplus contributes 10 to the positive side at `α=1.0`, so floors
dominate the metric.

W&B sweep direction: maximize. All-floors-met-exactly produces score=0.
All-floors-exceeded produces a positive number that grows with weighted
surplus.

### 2. Unified `EvalConfig`

Replaces `FoulplayEvalConfig` in [config.py](../../src/elitefurretai/rl/config.py).

```python
@dataclass
class OpponentEvalSpec:
    target: float        # in [0,1]
    weight: float        # 0.0 disables the bucket entirely
    n_battles: int       # total across all curriculum formats

@dataclass
class EvalConfig:
    enabled: bool = False
    eval_every_n_updates: int = 500
    pause_training: bool = True
    surplus_alpha: float = 1.0

    opponents: Dict[str, OpponentEvalSpec] = field(default_factory=lambda: {
        "simple_heuristic_baseline": OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
        "max_damage":                OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
        "vgc_bench":                 OpponentEvalSpec(target=0.60, weight=1.0, n_battles=100),
        "bc_player":                 OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
        "foul_play":                 OpponentEvalSpec(target=0.50, weight=0.0, n_battles=40),
    })

    # Opponent-specific runtime knobs, read only when that opponent's weight > 0
    vgcbench_checkpoint_path:    str = "data/models/vgc-bench-sb3-model.zip"
    vgcbench_team_file:          str = "data/teams/gen9vgc2024regg/vgcbench.txt"
    vgcbench_python_executable:  str = "/home/cayman/Repositories/venv-vgcbench/bin/python"

    foulplay_search_time_ms:    int = 750
    foulplay_python_executable: str = "/home/cayman/Repositories/venv-foulplay/bin/python"
    foulplay_team_pool:         Optional[str] = None
    foulplay_parallelism:       int = 8
```

The `<opp>_*` prefix convention mirrors how
[player_factory.py](../../src/elitefurretai/rl/analyze/player_factory.py)
already names cross-venv subprocess kwargs. Setting `opponents["foul_play"].weight = 0.0`
short-circuits all FoulPlay construction; the `foulplay_*` knobs become
dead config until weight is raised.

### 3. Generic bucket dispatch — `analyze/baseline_eval.py`

New module. Single dispatch loop handles in-process baselines and external
subprocesses identically because
[parse_player_specification](../../src/elitefurretai/rl/analyze/player_factory.py#L140)
already abstracts that distinction.

```python
@dataclass
class EvalBucket:
    name: str
    target: float
    weight: float
    n_battles: int

@dataclass
class BucketRunResult:
    win_rate: float
    n_battles: int
    per_format: Dict[str, EvalResult]
    wall_time_s: float

@dataclass
class MultiBucketEvalResult:
    per_bucket: Dict[str, BucketRunResult]
    score: float
    breakdown: Dict[str, float]
    wall_time_s: float

def run(eval_cfg, curriculum, checkpoint_path, server_urls, device, run_tag) -> MultiBucketEvalResult:
    results = {}
    externals = []
    try:
        for opp_name, spec in eval_cfg.opponents.items():
            if spec.weight == 0.0:
                continue
            result, ext_list = _run_opponent_bucket(
                opp_name, spec, eval_cfg, curriculum,
                checkpoint_path, server_urls, device, run_tag,
            )
            results[opp_name] = result
            externals.extend(ext_list)
    finally:
        for ext in externals:
            try:
                ext.shutdown()
            except Exception:
                logger.exception("Failed to shutdown %s", ext.username)

    win_rates = {k: r.win_rate for k, r in results.items()}
    targets   = {k: eval_cfg.opponents[k].target for k in results}
    weights   = {k: eval_cfg.opponents[k].weight for k in results}
    score, breakdown = compute_score(win_rates, targets, weights, eval_cfg.surplus_alpha)
    return MultiBucketEvalResult(per_bucket=results, score=score, breakdown=breakdown, ...)
```

`_run_opponent_bucket` iterates `curriculum.battle_formats`, calls
`parse_player_specification(opp_name, **eval_cfg_kwargs_for_opp)`, runs
[run_eval_parallel](../../src/elitefurretai/rl/analyze/evaluate.py#L439) per
format, then aggregates the per-format win rates using curriculum format
weights. The same code path handles every opponent.

### 4. `train.py` integration

[`_maybe_run_foulplay_eval`](../../src/elitefurretai/rl/train.py#L666) is
replaced by `_maybe_run_eval`, which routes everything through
`baseline_eval.run`. Cadence gating, exception swallowing, and W&B
logging follow the existing pattern.

W&B log payload per eval:

```
eval/score                              (sweep metric, maximize)
eval/deficit_l2_pp
eval/surplus_sum_pp
eval/wall_time_s
eval/update_step
eval/<opp>/win_rate                     one per active bucket
eval/<format>/win_rate                  weighted-by-opp-weight mean over active opps
```

The per-format value uses the same opponent weights as the score, so the
two metrics tell a consistent story. Per-opponent deficit, surplus, and
n_battles are deliberately omitted from the log payload; they are
derivable from the win rate and config if needed in a notebook.

### 5. Cleanup contract (three layers)

The user's hard requirement: before any process is killed, ensure all
Showdown servers and external subprocesses exit cleanly.

| Layer | Owner | Scope | Mechanism |
|---|---|---|---|
| 1 | `baseline_eval.run` | One eval pass | `try/finally` around the bucket loop, every spawned `RunningExternal` gets `.shutdown()` called before return |
| 2 | `train.main` | One sweep run | `try/finally` around the training loop + `atexit.register(shutdown_showdown_servers, ...)`. A final cleanup sweep uses `pgrep` to catch orphaned showdown/vgcbench/foulplay processes by matching the `run_tag` embedded in their Showdown usernames (the same `run_tag` already used by `_run_worker` and external player launchers) |
| 3 | `train_sweep.sweep_train` | One sweep agent run | `_terminate_with_grace` on the train subprocess: SIGTERM → wait `cleanup_timeout_s` → SIGKILL → wait. Followed by orphan-cleanup sweep before returning to `wandb.agent` |

The three layers exist because each handles a different failure mode.
Layer 1 covers eval-loop exceptions. Layer 2 covers training-loop
exceptions and normal process exit. Layer 3 covers train-process hangs
that bypass layer 2 entirely. If layer 3 inherits an orphaned showdown
server from a prior run, the next run silently fails on port conflict,
so the orphan sweep is load-bearing.

### 6. Sweep harness — `src/elitefurretai/rl/train_sweep.py`

Ports the supervised
[train_sweep.py](../../src/elitefurretai/supervised/train_sweep.py)
structure but spawns each sweep run as a subprocess of
`python -m elitefurretai.rl.train` rather than calling `train.main()`
in-process. The subprocess inherits `WANDB_RUN_ID`, so its own
`wandb.init()` reattaches to the run the sweep agent created and the
`eval/score` metric posts under the correct run.

Subprocess isolation guarantees that multiprocessing state, DataLoader
workers, Showdown server processes, and GPU memory from one sweep run
cannot affect the next. The cost is a few seconds of subprocess startup
per run, which is negligible against the multi-hour run length.

### 7. YAML schema — `sweep_configs/initial_rl_baselines.yaml`

```yaml
base_config: "src/elitefurretai/rl/configs/may26.yaml"

eval_overrides:
  training.wandb_project: "elitefurretai-rnad-sweep"
  eval.enabled: true
  eval.eval_every_n_updates: 500
  eval.surplus_alpha: 1.0
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
    # The full sweep parameter set (gradient clip, batch sizes, target
    # update rate, etc.) is intentionally not fixed here; the
    # implementation plan should pick the parameters and ranges with
    # Cayman before the first sweep launches. The three above are
    # placeholders to make the schema concrete.
```

Dotted keys (`learner.learning_rate`, `eval.opponents.foul_play.weight`) are
parsed by `_write_patched_config` into the nested `RNaDConfig` structure.
Sweep-sampled values override `eval_overrides` on key collision.

### 8. Standalone CLI

`baseline_eval.py` exposes `main()` for manual checkpoint evaluation,
mirroring the existing
[foulplay_eval.main](../../src/elitefurretai/rl/analyze/foulplay_eval.py#L599)
scaffolding it replaces:

```bash
python -m elitefurretai.rl.analyze.baseline_eval \
    --checkpoint data/models/.../checkpoint.pt \
    --eval-config path/to/eval_config.yaml \
    --curriculum-config path/to/curriculum.yaml \
    --num-servers 4 \
    [--wandb-project foo --wandb-run-name bar]
```

The CLI launches its own Showdown servers, calls `baseline_eval.run`,
shuts down servers in `finally`, and prints results. Optional `--wandb-*`
flags log to W&B for record-keeping. The CLI and the training-loop
integration share the same `run(...)` core, so all real work happens in
one place.

### 9. Files removed

- [src/elitefurretai/rl/analyze/foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py)
  — driver and `FoulplayEvalResult` are subsumed by `baseline_eval.py`.
- `FoulplayEvalConfig` in [config.py](../../src/elitefurretai/rl/config.py)
  — folded into `EvalConfig`.

The subprocess plumbing
([`_launch_foulplay_subprocess`](../../src/elitefurretai/rl/analyze/player_factory.py#L463),
[`agents/_foulplay_subprocess.py`](../../src/elitefurretai/agents/_foulplay_subprocess.py),
[`agents/foulplay_manager.py`](../../src/elitefurretai/agents/foulplay_manager.py))
is unchanged.

### 10. Testing strategy

Layout:

```
unit_tests/elitefurretai/rl/analyze/test_baseline_eval.py
unit_tests/elitefurretai/rl/test_train_sweep.py
```

Pytest-only, no Showdown servers, no GPU.

| Component | Tests |
|---|---|
| `compute_score` | Exhaustive: all floors met (score=0), uniform overshoot, uniform deficit, mixed, weight scaling, edge cases (rate=0, rate=1, target=0, target=1), pp-scale arithmetic |
| `weight==0` skip semantics | Mock `parse_player_specification`; assert zero calls for a `weight=0.0` bucket; assert no `eval/<opp>/*` keys in payload |
| Bucket dispatch | Mock `parse_player_specification` + `run_eval_parallel`; verify a `vgc_bench` bucket receives `vgcbench_*` kwargs and a `foul_play` bucket receives `foulplay_*` kwargs; verify in-process baselines receive neither |
| Log payload builder | Given synthetic `MultiBucketEvalResult`, assert top-level keys, per-opp `eval/<opp>/win_rate`, per-format `eval/<fmt>/win_rate` cross-opponent aggregation arithmetic |
| Layer-1 cleanup | Inject exception mid-bucket-loop; assert every prior-bucket `RunningExternal.shutdown` was called |
| YAML loader | Roundtrip on a fixture YAML; assert `(sweep_dict, base_config_path, eval_overrides)` extracted correctly; assert clear error on missing required keys |
| Dotted-key patching | `learner.learning_rate=1e-4` patches the right nested field; `eval.opponents.foul_play.weight=0.0` patches the dict-of-dataclass field; sweep params override eval_overrides on conflict |
| Layer-3 termination | Spawn a fixture sleep subprocess; assert clean wait on SIGTERM-respecting child; assert SIGKILL escalation on SIGTERM-ignoring child |

Out of scope: W&B internals, Showdown server lifecycle (covered elsewhere),
per-baseline win-rate accuracy (covered by `evaluate.py` tests), layer-2
cleanup (relies on structural simplicity plus manual smoke test before
declaring done), end-to-end sweep-with-real-servers run (belongs in the
implementation plan's manual verification step).

## Reasoning

**Why a generic bucket abstraction from day one rather than baselines-only.**
Cayman explicitly wants FoulPlay folded in once it stabilizes. The
generic abstraction costs one dataclass and one dispatch loop, and lets
the future FoulPlay integration land as a config change rather than a
refactor of tested code. The unification also removes a duplicate driver
([foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py))
that exists only because the two evals were built at different times.

**Why hinge L2 deficit + linear surplus rather than symmetric L2.**
The Stage II framing prioritizes meeting floors over maximizing
surpluses, so the metric should reward closing deficits much harder than
extending surpluses. Hinge L2 gives quadratic penalty for any
under-target rate; linear surplus contributes additive tiebreaker once
floors are met. Symmetric L2 would treat surplus and deficit as the same
shape, which would lead the sweep to chase imbalanced overshoots.

**Why percentage-point scaling.** L2 over [0,1] produces values in [0,1]
while L1 over [0,1] also produces values in [0,1] — but linear L1
dominates quadratic L2 in that range. Scaling deficits and surpluses to
percentage points expands the L2 term's dynamic range to [0,10000] while
L1 stays bounded by 100 per opponent, so floors dominate at any
non-trivial deficit. The resulting score range is large but
interpretable: a score of -100 means "one bucket is 10pp below target."

**Why subprocess-per-sweep-run rather than in-process.** RL training
spawns many child processes (Showdown servers, DataLoader workers, VGCBench
and FoulPlay subprocesses) and uses torch multiprocessing. Cleaning up all
of that between sweep runs in-process is error-prone; one missed shutdown
leaks resources into the next run and the sweep silently produces
nonsense. A subprocess boundary makes cleanup the OS's problem after we
issue SIGTERM/SIGKILL, at the cost of a few seconds of startup per run.

**Why three layers of cleanup rather than one.** Each layer covers a
different failure mode. Layer 1 (driver) handles eval-loop exceptions.
Layer 2 (training process) handles training-loop exceptions and normal
exit. Layer 3 (sweep harness) handles training-process hangs that bypass
layer 2. If layer 3 misses an orphan, the next sweep run inherits a port
conflict on Showdown servers and silently fails. The orphan cleanup
sweep at layer 3 is the only thing standing between us and silent sweep
corruption when a training run hangs.

**Why omit per-opponent deficit/surplus/n_battles from W&B logs.** Those
are derivable from `eval/<opp>/win_rate` plus the config. Logging them
inflates the metrics namespace without adding information and clutters
the W&B run UI.

**Open empirical question, deferred to implementation:** the right value
of `eval_every_n_updates`. At ~8.5 traj/s and 128 traj/update, 500 updates
is ~2 hours of training between evals, and Hyperband's min_iter=3 means a
run needs ~6 hours before it can be pruned. 1500 updates would be ~6
hours between evals and ~19 hours before pruning — too coarse for a
useful sweep. The right number depends on actual per-eval pause cost
(550 battles across 4 baselines at training throughput), which should be
measured in the implementation plan rather than locked here.

## Planned Next Steps

1. Hand off to writing-plans for the implementation plan.
2. Plan should call out the empirical measurement of per-eval pause cost
   before locking `eval_every_n_updates`.
3. Plan should land [foulplay_eval.py](../../src/elitefurretai/rl/analyze/foulplay_eval.py)
   removal as the last step, after the unified driver is validated to
   reproduce existing FoulPlay eval behavior on a sample checkpoint.

## Updates

(none yet)
