# 2026-05-14 — Delete `FlexibleThreeHeadedModel` (LSTM backbone)

## Context

All active checkpoints (`cool-bee-85-finetune_best.pt`, `cool-bee-85_best.pt`,
`curious-darkness-77_best.pt`) are `TransformerThreeHeadedModel`. Every YAML
config sets `use_transformer: true`. The LSTM branch in
`build_model_from_config`, the `_is_transformer` flag cascading through
players/engine/workers, and the LSTM-specific config fields were all dead in
practice. Decision: delete the legacy class entirely and simplify the
plumbing that branched on architecture.

## Before State

- `FlexibleThreeHeadedModel` class: 588 lines in `supervised/model_archs.py`
- LSTM-vs-Transformer branching in:
  - `rl/learners.py` — `MODEL_ARCH_CONFIG_KEYS`, `build_model_from_config`,
    `update()` PPO inner loop, `load_model_from_checkpoint` return type
  - `rl/players.py` — `RLTrajectoryPlayer._gpu_inference_sync`,
    `RLTrajectoryPlayer._run_batch`, `RNaDModel` wrapper, `StateHandle`
  - `rl/inference_handlers.py` — `_is_transformer` flag,
    `_pad_lstm_state`, branch in `_slice_next_hidden`
  - `rl/opponents.py` — `main_is_transformer` param, `WorkerOpponentFactory`
    assertion, `BCPlayer._load_model` (was hardcoded LSTM despite all
    checkpoints being transformer)
  - `engine/vgc_environment.py` — `main_is_transformer` param threaded
    through `VGCEnvironment.create` and `_ShowdownBackend`
  - `engine/sync_battle_driver.py` — `_is_transformer` flag,
    `choose_actions_from_snapshots` LSTM batching path
  - `rl/worker.py`, `rl/train.py` — `main_is_transformer` plumbing
- Dead config: `lstm_layers`, `lstm_hidden_size`, `early_attention_heads`,
  `late_attention_heads`, `use_transformer` in `ArchitectureConfig` and
  every yaml config (`single_team`, `easy_test`, `sep_arch`, all
  `supervised/configs/*.yaml`)

## Problem

Two model paths existed in the type system but only one was reachable. The
dual-path code carried real cost:

1. **Cognitive overhead** — readers had to constantly resolve "which branch
   matters here." Every hidden-state handling site had two implementations.
2. **Latent bugs** — `BCPlayer._load_model` was hardcoded to LSTM, so the
   analyze scripts (`behavior_clone_performance.py`, `behavior_clone_replay.py`)
   could not actually load any current checkpoint. The error wouldn't surface
   until someone tried to run those scripts.
3. **Drift surface** — every refactor needed to touch both branches and
   risked breaking the unused one silently.

## Solution

Full deletion: removed `FlexibleThreeHeadedModel` class, every isinstance
check, every `_is_transformer` flag, every `main_is_transformer` param, every
LSTM-only config field. Collapsed dual-branch code into the transformer path
that was the only one running anyway.

### Files touched

- **Deleted class**: `src/elitefurretai/supervised/model_archs.py` (-588 lines)
- **Simplified architecture branches**:
  `rl/learners.py`, `rl/players.py`, `rl/inference_handlers.py`,
  `rl/opponents.py`, `rl/worker.py`, `rl/train.py`,
  `engine/vgc_environment.py`, `engine/sync_battle_driver.py`
- **Cleaned config**: `rl/config.py` — dropped `lstm_layers`,
  `lstm_hidden_size`, `early_attention_heads`, `late_attention_heads`,
  `use_transformer`
- **Stripped yamls**: 3 RL configs + 7 supervised configs — removed dead
  `use_transformer: true` / `lstm_*` entries
- **Rewrote `BCPlayer._load_model`** to go through `build_model_from_config`
  (lazy-imported to avoid a `supervised → rl` cycle), so it now actually
  loads transformer checkpoints
- **Supervised cleanup**: `supervised/train.py`, `supervised/fine_tune.py`,
  `supervised/analyze/{action_model,win_model}_diagnostics.py`,
  `supervised/analyze/training_profiler.py`
- **Tests**: rewrote `test_agent_learner.py` mocks to mirror the transformer
  interface (context tensor instead of `(h, c)`); deleted the 13
  `test_flexible_model_*` cases in `test_model_archs.py`; updated remaining
  fixtures/assertions in `test_learner.py`, `test_smoke.py`,
  `test_behavior_clone_player.py`, `test_sync_battle_driver.py`,
  `test_players.py`
- **Docs**: `RL.md`, `SUPERVISED.md` — removed legacy-model sections

## Reasoning

- All current checkpoints are transformers; no backwards-compat constraint
  in CLAUDE.md (`backwards-compat is not a concern`).
- Deleting eliminates a latent bug (broken `BCPlayer._load_model` for
  transformer checkpoints, which all real checkpoints now are).
- Tests that exercised LSTM internals had no value once LSTM was unreachable.
  The transformer test coverage in `test_transformer_model_*` already exists
  and is sufficient.
- `SyncPolicyPlayer.choose_actions_from_snapshots` always fanned out to
  per-snapshot inference in the transformer path anyway, so collapsing the
  LSTM batching branch is purely a code reduction, no perf impact.

## Verification

- `ruff check` on changed files: clean.
- `ruff format` on changed files: clean (auto-applied).
- `pyright` on changed src: 0 errors. On changed tests: 5 errors, all
  pre-existing (verified by stash + re-run).
- `pytest unit_tests` (excluding `test_smoke.py` which needs servers and
  `test_config.py` which has unrelated drift from concurrent curriculum
  expansion): **499 passed, 1 skipped**.

## Updates

None yet.

## Planned Next Steps

- Remove the lingering `early_attention_heads`/`late_attention_heads` yaml
  entries from the active configs (already dropped from the dataclass; yaml
  values are now silently unused). Low-priority — they don't break anything.
- Investigate whether the `_load_model` lazy import in `behavior_clone_player.py`
  should be replaced by moving `build_model_from_config` to a shared module
  to eliminate the supervised→rl layering inversion. Cosmetic.
