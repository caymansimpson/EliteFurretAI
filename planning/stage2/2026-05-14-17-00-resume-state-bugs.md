# Resume-state bugs: random RNaD anchor + scheduler/_step rewind + curriculum override

## Context

While auditing earthy-wind-35 (currently running; cbdxgedw, resumed from
`pretty-jazz-4/main_model_step_365.pt`), the user observed loss improving while
all baseline win rates regressed and entropy climbed. Diagnostic dive uncovered
three independent bugs in the resume path, all exposed by the same wandb history.

The mapping from chart labels to wandb runs (via `wandb.Api`, since display
names aren't in local files):

| Display name | wandb id | Created (UTC) | Resumed from |
|---|---|---|---|
| earthy-wind-35 | `cbdxgedw` | 2026-05-14T23:22Z (running) | `pretty-jazz-4/main_model_step_365.pt` |
| balmy-firefly-11 | `7uiqaizt` | 2026-05-14T00:11Z | `pretty-jazz-4/main_model_step_151.pt` |
| pretty-jazz-4 | `ovcxrhy9` | 2026-05-13T17:37Z | None (BC init) |
| morning-energy-1 | `c4wcjdlt` | 2026-05-12T18:09Z | `bright-pine-189/ghosts/main_model_step_875.pt` |

## Before State

In [src/elitefurretai/rl/train.py:254-272](../../src/elitefurretai/rl/train.py)
the resume branch executed in this order:

```python
base_model = build_model_from_config(cfg, embedder, device, None)  # random init
agent = RNaDModel(base_model)
learner = initialize_learner(config, agent, base_model)            # deepcopies base_model as ref
start_step, old_config = load_checkpoint(                          # mutates agent.model AFTER deepcopy
    config.training.resume_from, agent, learner.optimizer, device,
)
```

`initialize_learner` ([train.py:198-210](../../src/elitefurretai/rl/train.py))
calls `ref_model = copy.deepcopy(base_model)` — captured BEFORE the checkpoint
load. `load_checkpoint` in
[learners.py:819-835](../../src/elitefurretai/rl/learners.py) restored only
`model_state_dict` and `optimizer_state_dict` — never scheduler or
`learner._step`. The opponent pool wiring at
[train.py:1119](../../src/elitefurretai/rl/train.py) was
`curriculum=resume_curriculum or config.curriculum.curriculum_weights`, which
silently preferred the checkpoint's saved curriculum over the yaml.

## Problem

Three bugs, all surfaced in earthy-wind-35's wandb history:

### Bug 1: RNaD anchor is a freshly random model after every resume

The deepcopy at `initialize_learner` runs BEFORE `load_checkpoint`, so
`learner.ref_models[0]` holds random weights. With `rnad_alpha=0.05`, the
`α * KL(curr || ref)` loss pulls the trained policy toward a uniform random
policy on every resume.

Empirical confirmation from an offline reproduction:

```
After resume — agent.model (loaded) vs ref_models[0] (deepcopied BEFORE load):
Total L1 diff: 1,406,949 / weight magnitude 1,100,842 → 127.8%
actor_token:  agent.sum=25.51, ref.sum=8.44   (~Xavier-init magnitude)
critic_token: agent.sum=36.95, ref.sum=8.02
```

earthy-wind-35's first 35 updates showed the textbook signature:

| update | rnad_loss | entropy | policy_loss | win_self_play | win_vgc | win_max_dmg |
|---|---|---|---|---|---|---|
| 366 | **1.81** | 5.49 | +0.073 | 0.21 | 0.12 | 0.14 |
| 399 | **0.34** | 6.93 | **−0.035** | 0.14 | 0.01 | 0.09 |
| 401 | 0.09 | 7.13 | +0.361 | **0.54** | 0.10 | 0.14 ← portfolio refresh |

The "improvement at update 401" the user observed is the portfolio's automatic
self-rescue: `portfolio_add_interval=100` fires, a snapshot of the (still
mostly-intact) current policy gets added as `ref_models[1]`. From update 402
onward, `_compute_portfolio_kl`'s `min_kl` selection picks the sane ref every
single step — `portfolio_selections.0` (the random ref) froze at 70 forever,
`portfolio_selections.1` accumulated 1 per update. Win rates recovered to
roughly pretty-jazz-4's level (self_play 0.69, bc 0.52, vgc 0.14, max_dmg 0.29
at update 464; vgc_bench is still slightly below the 0.18 target — the
residual damage from 35 updates of training-toward-random).

### Bug 2: LR scheduler `last_epoch` and learner `_step` reset to zero

`load_checkpoint` only round-tripped `optimizer.state_dict()` — never the
scheduler. `PortfolioRNaDLearner.__init__` set `self._step = 0`
([learners.py:174](../../src/elitefurretai/rl/learners.py)) and built the
`LambdaLR` with `last_epoch=-1`. Both are absent from the resume restoration.

Confirmed in wandb (queried directly because the on-disk LR series rounds to
0.0 in display):

| update | lr_backbone | warmup step (intended peak = 1e-5) |
|---|---|---|
| 366 | 1.00e-7 | 1/100 ← `last_epoch` reset |
| 393 | 2.80e-6 | 28/100 |
| 464 | 9.90e-6 | ~99/100 — warmup just completed |

`learner._step` drives `ent_coef_at_step` at
[learners.py:297](../../src/elitefurretai/rl/learners.py). With
`temperature_anneal_steps=10000` and `ent_coef_end ≈ ent_coef_start * 0.2`,
the drift inside 50 updates is small in absolute terms — but accumulated
across many resumes the entropy bonus never actually anneals.

### Bug 3: curriculum override silently used the checkpoint's curriculum

Confirmed in earthy-wind-35 history: `curriculum_weight_train_exploiter=0` and
`curriculum_weight_exploiters=0` throughout the run, even though sep_arch.yaml
sets both to 0.1. The model registry at
[train.py:1218-1264](../../src/elitefurretai/rl/train.py) is gated on the NEW
yaml (so it registered exploiter+victim services), but the OpponentPool
sampling used the OLD saved curriculum — services were registered but never
sampled. The yaml-vs-checkpoint mismatch was invisible to the user.

## Solution

Three changes, all in `learners.py` + `train.py`:

### 1. `save_checkpoint` now takes the full `learner`

`save_checkpoint(model, learner, step, config, curriculum, save_dir)` writes
`optimizer_state_dict`, `scheduler_state_dict`, and `learner_step` in
addition to `model_state_dict`. Both train.py callsites updated to pass
`learner` instead of `learner.optimizer`.

### 2. `load_checkpoint` returns the raw checkpoint dict; caller wires state

`load_checkpoint(filepath, device) -> Dict[str, Any]` is now side-effect-free.
A new method `PortfolioRNaDLearner.load_resume_state(checkpoint)` restores
optimizer + scheduler + `_step` — but pointedly NOT model weights, since
those must be loaded into `base_model` BEFORE the learner is constructed
(otherwise Bug 1 returns).

The resume branch in train.py now reads:

```python
checkpoint = load_checkpoint(resume_from, device)
base_model = build_model_from_config(cfg, embedder, device, None)
base_model.load_state_dict(checkpoint["model_state_dict"])
agent = RNaDModel(base_model)
learner = initialize_learner(config, agent, base_model)  # ref now captures TRAINED weights
learner.load_resume_state(checkpoint)
start_step = int(checkpoint["step"])
old_config = RNaDConfig.from_dict(checkpoint["config"])
```

`load_resume_state` falls back gracefully for pre-fix checkpoints (no
`scheduler_state_dict` / `learner_step` keys) by anchoring `scheduler.last_epoch
= step - 1` and `_step = step`, so older `.pt` files don't replay the warmup
from zero either.

### 3. yaml curriculum wins; resume curriculum diff is logged loudly

```python
active_curriculum = config.curriculum.curriculum_weights
if resume_curriculum and resume_curriculum != active_curriculum:
    logger.warning("Curriculum override on resume: yaml weights differ from "
                   "checkpoint's saved weights. Using yaml.\n"
                   "  yaml:       %s\n"
                   "  checkpoint: %s", active_curriculum, resume_curriculum)
opponent_pool = OpponentPool(..., curriculum=active_curriculum)
```

This makes the yaml the source of truth — the user's primary mental model when
editing `sep_arch.yaml` — and makes the previous silent override an explicit
WARNING line in the boot log.

## Reasoning

- Why move `load_checkpoint` before learner construction instead of adding a
  ref-sync method on the learner: the temporal coupling between "model loaded"
  and "ref deepcopied" is fundamental to how `PortfolioRNaDLearner.__init__`
  works. A `learner.sync_ref()` helper would have papered over the order
  dependency. Restructuring the resume branch makes the order explicit and
  matches the natural order in `initialize_learner` itself (the deepcopy
  documents that the ref IS the agent at construction time).

- Why split `load_checkpoint` into "read dict" vs `load_resume_state` rather
  than pass the learner: the old API made it look like `load_checkpoint`
  modified the model+optimizer together. With Bug 1 the model load needs to
  happen at one phase and optimizer/scheduler at another, so splitting the
  call site clarifies which state lands where and when.

- Why fall back to `step - 1` for legacy checkpoints rather than treating
  them as fresh: existing `.pt` files under
  [data/models/rl/pretty-jazz-4/](../../data/models/rl/pretty-jazz-4/) are all
  pre-fix. Anchoring scheduler `last_epoch` to the saved global step gets us
  approximately the right LR; anchoring `_step` similarly keeps the entropy
  bonus close. Both are best-effort — the new save path will persist exact
  scheduler state going forward.

- Why yaml wins (per Cayman's preference): the user edits the yaml between
  resumes to change curriculum mixes (e.g. enable exploiters mid-run). The
  silent checkpoint override violated that mental model. Logging when the two
  differ keeps the change visible without requiring an explicit
  `--reset-curriculum` flag.

## Planned Next Steps

- **Now**: Kill PID 8566 (`pkill -f "train.py.*sep_arch"`, then verify with
  `pgrep -f train.py`) and re-resume from `pretty-jazz-4/main_model_step_365.pt`
  with the patched code. Expected first-update signature:
  `rnad_loss ≈ 0.0–0.1`, `policy_loss` near 0 (not negative), `lr_backbone`
  near peak (~1e-5), win rates close to pretty-jazz-4's final state.
- **Future-proofing**: Consider also resuming from a cleaner upstream
  (`bright-pine-189/ghosts/main_model_step_875.pt`, morning-energy-1's source)
  if the residual policy damage in step_365.pt proves too sticky to recover
  cleanly.
- **Documentation**: Note in `RL.md` that resume preserves scheduler + `_step`
  state going forward; legacy checkpoints get best-effort restoration.

## Updates

- 2026-05-14 17:00 — patch landed. Files touched:
  `src/elitefurretai/rl/learners.py` (save/load API, `load_resume_state`
  helper), `src/elitefurretai/rl/train.py` (resume branch reorder, curriculum
  precedence + warning, two save callsites), `unit_tests/rl/test_learner.py`
  (5 new tests covering scheduler/_step round-trip, legacy-checkpoint
  fallback, and the invariant that `load_resume_state` doesn't touch model
  weights). `ruff check` / `ruff format --check` / `pyright` clean.
  `pytest unit_tests/rl/test_learner.py` 31/31 pass. Two pre-existing failures
  in `test_compile_race_reproducer.py` and `test_worker_opponent_factory.py`
  are unrelated to this change (verified by `git stash` + retest).
