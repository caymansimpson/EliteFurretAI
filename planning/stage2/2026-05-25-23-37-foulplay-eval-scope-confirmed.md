# FoulPlay Eval — Scope Confirmed (Eval-Only) and Plan Refresh Required

## Context

The original FoulPlay integration was specified on 2026-05-18 across two docs:

- Design: [planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md)
- Implementation plan: [planning/stage2/2026-05-18-17-00-foulplay-eval-implementation-plan.md](2026-05-18-17-00-foulplay-eval-implementation-plan.md)

Both pre-date the May 18–25 work (multi-format doubles, agents directory reorg, agent-team-axis adaptive curriculum, BC action quality fixes). Before resuming implementation, we revisited scope — specifically, whether to expand from eval-only to use FoulPlay battles as a learning signal too.

## Before State

- Original plan: 8 implementation tasks + smoke test, eval-only. Adds `FoulPlayManager` (subprocess lifecycle), `_foulplay_subprocess.py` (separate venv), `foulplay_eval.py` driver, `external_username` field on `PlayerSpecification`, train.py checkpoint hook, YAML + RL.md docs.
- Recent code drift: `PlayerSpecification` now has `kind ∈ {model, baseline, external}` + `params: Mapping` (no `factory`, no `external_username`); vgc_bench already wired as `kind="external"`; `_run_worker` is cell-iterated with an existing external branch; `battle_format → battle_formats` (multi-format distribution); `opponent_team_pool_path → opponent_team_pool_paths`; team-axis adaptive sampling (Change 7).

## Problem

Two real questions had to be answered before writing the refreshed implementation plan:

1. **Is the original architecture (subprocess-managed websocket bot, eval-only) still the right shape?** Yes — the May 18–25 refactors didn't change the eval surface, only its specifics.
2. **Should we expand scope to capture FoulPlay battles as training data (BC corpus) or as a teacher signal (online distillation)?** Explored both; both ruled out for this round.

## Solution

**Scope confirmed as eval-only.** FoulPlay runs as a periodic ground-truth eval opponent at checkpoint cadence, logs win-rate to wandb, does not participate in the RL curriculum and does not produce a BC corpus.

The original architecture stands:

- FoulPlay subprocess in `../venv-foulplay/` (poke-engine-doubles + old poke_env).
- `FoulPlayManager` owns subprocess lifecycle (mirrors `VGCBenchManager`).
- `analyze/foulplay_eval.py` driver orchestrates per-eval launch/battles/shutdown.
- Called from `train.py` at checkpoint cadence and from `analyze/evaluate.py` as a baseline.

What changes vs. the original 84KB plan is the **realization against current code**, not the architecture. Six concrete diffs no longer apply and need re-anchoring:

| # | Original plan assumes | Current code | Refreshed plan does |
|---|---|---|---|
| 1 | Add `external_username` field to `PlayerSpecification` | `kind="external"` + `params` pattern already exists; vgc_bench uses it | Add `foul_play` to `_EXTERNAL_BASELINES`; extend `launch_external_player` dispatch by `name`. Tasks 4+5 collapse to ~30 lines. |
| 2 | Modify `agents/foulplay_manager.py` | File does not exist | Create new file mirroring `agents/vgcbench_manager.py` shape |
| 3 | `_run_worker` patch at lines 97–143 | File reorganized; external branch already present at line ~293, cell-iterated | Verify foul_play flows through existing external branch; no `_run_worker` patch needed |
| 4 | `FoulplayEvalConfig.battle_format: str`, validator checks `battle_format` | `battle_formats: Dict[str, float]`, `primary_format` property | Multi-format eval from v1. `FoulplayEvalConfig.n_battles_per_format: int = 100`; the eval driver iterates over `config.curriculum.battle_formats.keys()`, runs one self-contained FoulPlay cycle per format (launch → battles → teardown), aggregates results. WandB logs both per-format (`eval/foulplay/<format>/win_rate`) and a weight-averaged overall (`eval/foulplay/win_rate`). |
| 5 | Flat YAML keys (`battle_format: ...`, `opponent_team_pool_path: ...`) | Multi-format schema | YAML uses per-format `foulplay_team_pool_paths: Dict[str, str]` (one team-list directory per format FoulPlay needs to play). Default: fall back to `opponent_team_pool_paths[fmt]` if `foulplay_team_pool_paths` is omitted, so users only override when they want a separate FoulPlay-side team set. Validator checks all referenced paths exist when `enabled=True`. |
| 6 | No decision on agent-side team sampling | Change 7 team-axis sampler is adaptive during training | Eval uses a **fixed** agent team pool (clean signal across checkpoints), not the adaptive distribution |

## Reasoning

### Why not BC corpus

Two routes considered:

- **Offline sidecar + `from_external_play`**: capture the model's `p1_battle` state at battle finish, reconstruct opponent perspective offline via a new `BattleData` constructor. Cost: ~3 days, most risk in synthesizing `input_logs` from protocol events. Recoverable.
- **In-process proxy player + stdin/stdout-adapted FoulPlay**: in-process `FoulPlayProxyPlayer` holds `p2_battle` directly; `from_self_play` works as-is. Cost: ~4–5 days; risk concentrated in a foul-play-doubles refactor (strip its websocket layer) — unbounded if their main loop is awkward to drive from stdin.

Both deferred. Reasons:

- **Premature.** We don't yet know whether FoulPlay's win-rate signal is even useful at training cadence. Investing 3–5 days in capture infrastructure before having seen the signal is overbuild.
- **The infrastructure is reusable.** Capture can be added as a phase 2 on either pathway without retraining; the trajectory parquet + gzipped protocol logs from the standard eval pipeline already record the data shape we'd need.
- **Stage II graduation pressure.** The user is pushing for 60% × 4 baselines; eval-only is the minimum viable artifact to learn whether FoulPlay belongs in the graduation criterion at all.

### Why not online distillation

Online distillation against FoulPlay (FoulPlay as a curriculum opponent + `cross_entropy(model_policy, foulplay_action)` mixed into the RL loss) was the most ambitious option and has the cleanest "always on-policy" properties. Ruled out by a hard throughput constraint:

- FoulPlay's search uses **8 cores at 750 ms/move**.
- One FoulPlay battle saturates the machine; no other RL battles can run concurrently with FoulPlay's search.
- Curriculum integration would require serializing the entire training loop around FoulPlay's search. Even at 5% curriculum mix and 200 ms search, projected throughput drop is large (and 200 ms search degrades the teacher quality the distillation depends on).

The 8-core × 750 ms × no-parallelism constraint is a hard machine-level ceiling. Eval-only is the only shape compatible with FoulPlay's resource profile on a single machine.

### Why eval-only is the right shape anyway

- **Wall-clock cost is bounded**: one eval pass = 100 battles at full search ≈ 30–60 min training pause every N checkpoints. Acceptable as periodic interruption.
- **The signal is the deliverable**: a periodic ground-truth win-rate against a top-100 search bot is information training does not otherwise have access to.
- **The data comes free**: the standard eval pipeline (`TrajectoryCollector`) writes per-battle parquet + gzipped protocol logs. FoulPlay-vs-model battles land in the same place as every other eval. If BC or distillation becomes worth doing later, the data is on disk.

## Planned Next Steps

1. **Refresh the implementation plan** to align with current code (mechanical task — same 8 tasks + smoke test, updated diffs throughout). To be produced via the writing-plans skill in the next session.
2. **Execute the refreshed plan** task-by-task with TDD discipline (per the original plan's structure: write failing test → implement → run quality gates → commit).
3. **Smoke test** at low cadence (5 updates, 2 battles, 250 ms search) before turning on production cadence.
4. **First production run**: enable in next training run at `n_battles=100`, `search_time_ms=750`, `eval_every_n_updates=50`. Observe whether win-rate signal is informative across checkpoints.
5. **FoulPlay is informational signal; promotion to graduation criterion deferred to post-data decision.** Stage II's current bar is 60% × 4 baselines (`vgc_bench_baseline`, `max_damage`, `bc_player`, `simple_heuristic_baseline`) per [2026-05-16-21-30-stage2-graduation-criteria.md](2026-05-16-21-30-stage2-graduation-criteria.md). FoulPlay would be a candidate 5th baseline, but its win-rate distribution against your models is unknown until we have data — promoting it to a hard requirement now would raise the bar before we've measured it. After 200–500 FoulPlay battles land in wandb, revisit: is FoulPlay-60% a meaningful gate, or too easy / too hard to be useful?
6. **Revisit BC/distillation** only if (a) FoulPlay win-rate signal is informative and (b) Stage II graduation is in reach. Both are open questions until phase 1 ships.

## Updates

(none yet — pre-implementation; this doc is the scope decision)
