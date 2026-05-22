# Multi-Format Doubles RL — Shipped

## Context

Reference plan: [2026-05-22-12-16-multi-format-doubles-rl-implementation-plan.md](./2026-05-22-12-16-multi-format-doubles-rl-implementation-plan.md)

Implementation executed via subagent-driven development on branch `feature/multi-format-doubles` (worktree at `.worktrees/multi-format-doubles/`). Seven planned tasks plus post-flight, all merged to the feature branch.

## Before State

- `CurriculumConfig.battle_format: str` was a single-format string.
- `WorkerOpponentFactory` hardcoded a single battle format per worker.
- Stage II training pipeline could only run one format at a time.

## Problem

Goal was to train a single RL model across multiple gen9 doubles formats without retraining from scratch per format. Vocab is gen-keyed (shared across doubles regs at the same gen), so the same embedder works for every gen9 doubles format — the engineering challenge was wiring per-pair format assignment through the existing single-format machinery.

## Solution

- `battle_formats: Dict[str, float]` distribution on `CurriculumConfig` (Task 1).
- Per-format path resolution helpers `resolved_agent_team_paths()` / `resolved_opponent_team_pool_paths()` (Task 2).
- `WorkerOpponentFactory` accepts the distribution and per-format paths (Task 3).
- `create_agents` apportions formats across pairs via largest-remainder (Hamilton); `randomize_all_teams` indexes per-slot format (Task 4).
- `VGCEnvironment.from_config`, `worker.py`, `train.py`, `learners.py`, `exploiters.py`, `vgcbench_manager.py` all migrated to use `primary_format` for embedder/vocab and the dict-form distribution where appropriate (Task 5). Also: replaced `"battle_format"` in `MODEL_ARCH_CONFIG_KEYS` with synthetic `"_gen"` derived from `battle_formats`, so checkpoint compatibility is gen-keyed instead of format-keyed.
- 5 yamls migrated; new `multi_format_smoke.yaml` for integration testing (Task 6).
- Per-format graduation matrix: `q_format_opp_type_win_rate`, `graduation_summary`, and `multi_format_graduation_eval.py` driver (Task 7).

## Reasoning

Apportionment chosen over per-battle resampling because Showdown usernames are persistent — destroying/recreating players per battle would be expensive and racy. Per-pair format pinning means each (player, opponent) pair runs one format for the entire batch, which is the cleanest contract given the existing player lifecycle.

Embedder uses `primary_format` for vocab construction; vocab is keyed on `format_str[3]` (the gen number), so any gen9 doubles format produces the same vocab. The same-gen constraint in `__post_init__` enforces this invariant explicitly.

VGCBench v1 is single-format; off-`primary_format` pairs cannot challenge it (Showdown rejects mismatched formats). VGCBenchManager logs a warning at launch when this configuration is detected. The user's spec accepts this trade-off; VGCBench v2 (multi-format trained) will replace it later.

## Planned Next Steps

1. **Showdown server requires `gen9vgc2024regh` format definition.** The smoke test fired regg battles successfully but Showdown rejected regh with `|popup|Unrecognized format`. The local `pokemon-showdown/config/formats.ts` needs a `[Gen 9] VGC 2024 Reg H` entry before any real multi-format run can use that format.
2. **`easy_test.yaml` has a stale `resume_from` pointing at a nonexistent checkpoint.** Out of scope for this plan, but a small follow-up.
3. **Per-format adaptive curriculum.** Format weights are currently static. If a multi-format run shows uneven performance, dynamic re-weighting (à la the existing `adaptive_curriculum`) could help — but only after VGCBench v2 ships.

## Updates

(Add dated entries here as the work continues.)
