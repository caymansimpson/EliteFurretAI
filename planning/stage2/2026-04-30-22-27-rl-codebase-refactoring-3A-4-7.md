# RL Codebase Refactoring: Tasks 7, 4, 3A

## Context

Structural cleanup of `src/elitefurretai/rl/` to improve readability for researchers, remove duplication, consolidate action-space logic, and clarify the IMPALA actor-learner boundary. Covers tasks 7 (masking merge), 4 (remove RNaDLearner, absorb model_io), and 3A (hierarchical config). Tasks 3B (worker.py) and 3C (VGCEnvironment) are pending.

## Before State

- `rl/fast_action_mask.py` + `rl/request_targeting.py` — separate files for the same concern
- `rl/learners.py` — both `RNaDLearner` (base) and `PortfolioRNaDLearner` (portfolio variant) existed
- `rl/model_io.py` — separate file for checkpoint I/O utilities
- `rl/central_inference.py` — deprecated cross-worker GPU batching server (dead code)
- `rl/config.py` — flat `RNaDConfig` with ~530 fields accessed as `config.clip_range`, `config.battle_format`, etc.
- YAML configs used flat structure

## Problem

- Duplication: RNaDLearner and PortfolioRNaDLearner had 80% identical code; callers had `if portfolio` branches
- Scattering: model I/O functions in a separate file from the learner that owns the checkpoint format
- Dead code: `central_inference.py` was flagged deprecated, unused in current training path
- Flat config: ~530 fields on one object made it impossible to know which subsystem owned which field; typos silently added new attrs instead of raising AttributeError

## Solution

### Task 7 — masking.py
- Created `src/elitefurretai/rl/masking.py` merging `fast_action_mask.py` + `request_targeting.py`
- Module docstring explains the 3-step pipeline: target resolution → slot-level legality → full pair mask
- Deleted both source files; updated `players.py` imports

### Task 4a — Remove RNaDLearner
- Deleted `RNaDLearner` class from `learners.py` entirely
- `PortfolioRNaDLearner` subsumes all use cases (set `max_portfolio_size=1` + `portfolio_update_strategy="recent"` for standard single-reference RNaD)
- Removed `use_portfolio_regularization` field from config (documented in `PortfolioConfig` docstring instead)
- Updated `train.py` `initialize_learner()` to always use `PortfolioRNaDLearner`

### Task 4b — Absorb model_io.py
- Moved all functions (`build_model_from_config`, `load_model_from_checkpoint`, `load_agent_from_checkpoint`, `save_checkpoint`, `load_checkpoint`, `is_checkpoint_compatible_with_model_config`) into `learners.py` under a `## Model I/O` section header
- Deleted `model_io.py` and `central_inference.py`
- Updated all importers (`train.py`, `opponents.py`, `exploiter_train.py`, engine analyze scripts)

### Task 3A — Hierarchical Config
Replaced flat `RNaDConfig` with 9 typed sub-config dataclasses:

| Sub-config | Key fields |
|---|---|
| `AlgorithmConfig` | clip_range, ent_coef, gamma, rnad_alpha, ref_update_interval |
| `PortfolioConfig` | max_portfolio_size, portfolio_add_interval, portfolio_update_strategy |
| `ExplorationConfig` | temperature_start/end, temperature_anneal_steps, top_p |
| `OptimizerConfig` | backbone_lr, heads_lr, schedule, warmup_steps |
| `ValueHeadConfig` | num_value_bins, value_min/max |
| `ArchitectureConfig` | early/late_layers, lstm_*, transformer_*, number_bank_* |
| `HardwareConfig` | num_workers, num_players, device, battle_backend, use_mixed_precision |
| `CurriculumConfig` | battle_format, curriculum_weights, bc_*_path, vgcbench_*, team paths |
| `TrainingConfig` | max_updates, save_dir, train_batch_size, use_wandb, exploiter_* |

Access pattern: `config.algorithm.clip_range`, `config.hardware.num_workers`, etc.

`config.curriculum` (dict of weights) renamed to `config.curriculum.curriculum_weights` — the sub-object is `config.curriculum`.

**Checkpoint backwards compatibility**: Old `.pt` files store flat config dicts. `_migrate_flat_config()` + `_is_nested_format()` in `config.py` auto-migrate on load. `_config_to_flat_arch()` in `learners.py` normalizes for `build_model_from_config`.

**YAML structure**: Both `easy_test.yaml` and `single_team.yaml` restructured to nested format matching sub-config names.

**Files updated**: `train.py`, `learners.py`, `opponents.py`, `players.py`, `exploiter_train.py`, `showdown_server_manager.py`, all `engine/analyze/*.py` files.

**Tests updated**: `unit_tests/rl/test_config.py`, `unit_tests/rl/test_learner.py`.

## Reasoning

- Hierarchical config makes ownership explicit — `HardwareConfig` owns hardware fields, `CurriculumConfig` owns opponent-sampling fields; typos now raise `AttributeError` immediately
- Single `PortfolioRNaDLearner` with `max_portfolio_size=1` is strictly equivalent to the old `RNaDLearner`; no behavioral change
- Co-locating model I/O with the learner makes the coupling explicit — the checkpoint format is defined by the learner's optimizer/config schema
- Flat migration in `from_dict` means existing checkpoints load without manual intervention

## Verification

- Config round-trip: `RNaDConfig.load(yaml)` → `save()` → `load()` matches ✓
- Import smoke tests: masking, learners, config all import cleanly ✓
- `ruff check --select F401,F811,F821,E9 src/elitefurretai/rl/ src/elitefurretai/engine/`: all pass ✓
- `pyright src/elitefurretai/rl/ src/elitefurretai/engine/`: 0 RNaDConfig errors (14 pre-existing non-config errors remain) ✓
- `pytest unit_tests/rl/test_config.py unit_tests/rl/test_learner.py unit_tests/rl/test_multiprocess_actor.py`: all pass ✓

## Planned Next Steps

- **Task 3B**: Create `src/elitefurretai/rl/worker.py` — extract `_RustPolicyOpponentPool`, `_run_rust_worker_loop`, `mp_worker_process` from `train.py`; add IMPALA architecture comments; implement Option C broadcast extension (add `exploiter_paths`/`ghost_paths` to weight_queue messages)
- **Task 3C**: Create `src/elitefurretai/engine/vgc_environment.py` — `VGCEnvironment` class wrapping both Showdown and Rust backends behind a single interface
