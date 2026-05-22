# Adaptive Curriculum Overhaul

**Date**: 2026-05-20 15:00 (refresh of 2026-05-03 23:37 plan); 2026-05-22 11:55 reprioritization in light of MODEL_EVALUATION.md
**Status**: Planned, not implemented — refreshed for current code state
**Owner**: Cayman

> **Why this refresh**: the original plan dated 2026-05-03 was still entirely on paper as of 2026-05-20. Between those dates the codebase shifted in ways that change scope and priority for several of the proposed changes (centralized inference, ghost centralization + model registry, slot-addressed ghosts/exploiters, sampling relocated into the worker, most invalid-choice forfeit bugs fixed, Stage II graduation criterion broadened to 4 baselines, foulplay held-out eval shipped). The "Before State" math is unchanged; the *plan around it* needed re-anchoring.

> **Why this reprioritization (2026-05-22)**: the balmy70 step-10700 full evaluation in [src/elitefurretai/rl/analyze/MODEL_EVALUATION.md](../../src/elitefurretai/rl/analyze/MODEL_EVALUATION.md) (n = 476,650 battles) surfaced two findings that constrain this plan. First, per-agent-team WR spans 35 points against heuristics and 28 points against vgc_bench (Q1), while per-opp-team WR std against vgc_bench is 0.006 (Q2) — the curriculum's opponent-only axis is missing the dimension that actually varies. Second, the model exhibits a single-strategy pathology (direct damage commitment): strong against setup opponents, weak against direct-offensive box mons. Foul-play is search-based and a single-strategy policy cannot beat it. Concrete adjustments: added Change 7 (agent-team-axis adaptive sampling) as a new top priority, split Change 3 into exploiter sub-PFSP (kept) and ghost sub-PFSP (deferred pending a diagnostic), restored a modest priority to Change 1 because per-cell sample efficiency matters more once we track team × opp_type cells.

---

## Context

The adaptive curriculum lives in `OpponentPool.update_curriculum()` at
[src/elitefurretai/rl/opponents.py:367](../../src/elitefurretai/rl/opponents.py#L367)
(was lines 719-830 in the original plan; the file was reorganized since).
It is called from `train.py` at the trajectory-ingest loop, currently once per
`checkpoint_interval` immediately before broadcasting weights + curriculum to
workers ([train.py:1110](../../src/elitefurretai/rl/train.py#L1110)).

**Architectural change since the original plan**: opponent *sampling* now lives
in `WorkerOpponentFactory.sample_opponent_type()` at
[opponents.py:819](../../src/elitefurretai/rl/opponents.py#L819), not in
`OpponentPool`. The main process computes the curriculum and broadcasts it;
each worker samples from its local copy. This matters for Change 3 — any
within-bucket distribution (per-ghost or per-exploiter sampling weights) must
now be broadcast as new fields, not just maintained in main-process dicts.

The curriculum is a probability distribution over opponent *types*
(`self_play`, `bc_player`, `exploiters`, `ghosts`, plus heuristic baselines and
`vgc_bench_baseline`).

Per-opponent state used by the algorithm (in `OpponentPool`):
- `self.win_rate_tracking[opp_type]: deque(maxlen=tracking_window=100)` of 0/1 outcomes.
- `self.battle_length_tracking[opp_type]: deque(maxlen=100)` of ints — **collected but unused in update**.
- `self.total_forfeits_tracked` — forfeits aggregated globally; not separated per-type and not excluded from win rate.
- `record_battle_result(opponent_type, won, battle_length=0, forfeited=False)` at [opponents.py:298](../../src/elitefurretai/rl/opponents.py#L298), called once per finished trajectory from [train.py:936-941](../../src/elitefurretai/rl/train.py#L936-L941). The trajectory protocol already carries `forfeited` and `battle_length` end-to-end — so the data plumbing for Change 1 is already done.

Ghosts and exploiters are now **slot-addressed** by path via
`slot_for_ghost_path` and `slot_for_exploiter_path` (model registry, shipped
2026-05-14). The natural id Change 3 needs already exists.

Project goals (per `CLAUDE.md`, `RL.md`, and the project-vision memory):
- Stage II: Single-Team League Mastery via r-NaD initialized from BC.
- **Stage II graduation criterion**: simultaneously ≥60% win rate vs
  `vgc_bench_baseline`, `max_damage`, `bc_player`, and
  `simple_heuristic_baseline` (see [2026-05-16-21-30-stage2-graduation-criteria.md](2026-05-16-21-30-stage2-graduation-criteria.md)).
- Self-play + ghosts + exploiters ensemble; portfolio of reference snapshots
  (loss-side anchor) is distinct from the curriculum (sampling-side anchor).
- We want Nash-style convergence with low cyclic drift; BC anchor preserves
  human-likeness.

---

## Before State (current algorithm, summary)

Unchanged from the original plan. Per call to `update_curriculum`:

1. Refresh `exploiter_models` and `ghosts` lists from disk.
2. For each opponent type with a base weight:
   - If unavailable → score 0.
   - If `samples < 40` → keep base weight (no adaptation).
   - Else compute Bayesian-smoothed win rate `wr = (wins+8)/(n+16)` over last 100 battles.
   - PFSP score: `1 - 2|wr - 0.5|` (symmetric tent peaking at 0.5).
   - Weakness score: `max(0, (0.55 - wr) / 0.55)`.
   - `learning_value = 0.7 * pfsp + 0.3 * weakness`.
   - `candidate = max(eps, 0.5 * base_weight + 0.5 * learning_value)`.
3. Apply hard floors (self_play=0.20, bc=0.10, ghosts=0.10 when available).
4. Distribute remaining mass by `residual = candidate - floor` proportionally.
5. Renormalize.

Default starting curriculum: self=0.40, bc=0.20, exploiters=0.20, ghosts=0.20,
all heuristic baselines=0.0.

A `# TODO: revisit` comment sits just above `update_curriculum` in
`opponents.py` — this plan is what to revisit it with.

---

## Problem

Ten identified weaknesses; severity has shifted since 2026-05-03 and again on 2026-05-22 in light of MODEL_EVALUATION.

1. **Flat ghost/exploiter buckets** *(split: exploiter side still high, ghost side deferred — see Group B)*: all ghost checkpoints (and all exploiters) sampled uniformly within their bucket. AlphaStar credits within-bucket PFSP as a major contributor to skill gains; we currently get none of that. With slot-addressed ghosts/exploiters now in place, the *implementation* of within-bucket PFSP got cheaper — the per-checkpoint identity already exists in `slot_for_ghost_path` / `slot_for_exploiter_path`. The *broadcast* side got slightly more involved because sampling now lives in workers. MODEL_EVALUATION Q2 caveat: WR std across opp_teams against vgc_bench is 0.006, so if ghosts inherit the same lack of per-opponent-team variation they may also have low cross-ghost variation, and within-bucket PFSP over ghosts could just rotate among same-flavor checkpoints. Exploiters are trained to differ from each other by construction, so the lift on the exploiter side is more credible.
2. **Symmetric PFSP score** *(still high)*: `1 - 2|wr - 0.5|` doesn't differentiate "we crush them" from "we're crushed."
3. **Double-stabilization** *(promoted to highest priority within Group A; corroborated by MODEL_EVALUATION)*: `0.5 * base_weight + 0.5 * learning_value` ties the curriculum to its initial values forever. The "update-100 cliff was VGCBench, not ghosts" diagnostic ([2026-05-16-08-13](2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md)) confirmed VGCBench dominates the in-curriculum learning signal — and yet adaptation is being squelched by this base-mix. MODEL_EVALUATION Q2's WR std = 0.006 against vgc_bench is an independent symptom that the in-curriculum signal is not moving the policy.
4. **`battle_length_tracking` collected but unused** *(unchanged)*.
5. **Forfeits counted as losses** *(modestly re-elevated relative to 2026-05-20)*: most invalid-choice forfeit sources have been fixed (Commander, Uproar, force-switch families per `CLAUDE.md`); two residual bugs remain ([2026-04-26-22-00](2026-04-26-22-00-two-residual-bugs.md)). Once Change 7 lands and tracking expands to 42 × ≥4 = ≥168 (agent_team, opp_type) cells, the per-cell sample budget shrinks proportionally and residual forfeit noise per cell matters more than at the type-only granularity. Still not load-bearing on its own; worth re-bundling with Change 7 when that ships.
6. **Stale-window risk** *(unchanged, still high after Change 2)*: the deque doesn't drain when samples don't arrive; min-sample gate of 40 compounds the lag.
7. **Update cadence = checkpoint cadence** *(downgraded)*: at current throughput (~5-6 traj/s, per the throughput memory), each checkpoint interval already accumulates plenty of battles for adaptation to be statistically meaningful. The decoupling is no longer urgent.
8. **Zero-base opponents stay zero** *(unchanged)*: `RANDOM_BASELINE=0.0` etc. → 0 forever; no exploration mechanism.
9. **Agent-team axis ignored** *(new, 2026-05-22; designated top priority alongside Group A)*: `team_provider` samples uniformly across the 42 teams in `data/teams/gen9vgc2024regg/constrained/`. MODEL_EVALUATION Q1 shows a 35-point per-agent-team WR spread against heuristics, 28 points against vgc_bench, and only one cell (`38dessert` × simple_heuristic) where the CI crosses 50%. Q2 shows zero per-opponent-team adaptation against vgc_bench. The cross-pattern summary (top-5 vs bottom-5 piloted teams) shows the policy can pilot direct-offensive cores and cannot pilot setup-reliant teams (Calm Mind, Trick Room, Friend Guard). With uniform team sampling, training pressure on the under-performed teams is diluted by the teams the policy already wins or loses on consistently. Change 7 addresses this; design pending brainstorm.
10. **Single-strategy pathology limits foul-play viability** *(new, 2026-05-22; out of curriculum scope, flagged for cross-plan coordination)*: MODEL_EVALUATION's headline is that the model has one strategy (direct damage commitment), exploits opponents that need setup, and loses to opponents that don't. Foul-play is search-based and multi-strategy by construction; beating it requires the policy to develop a second strategy, most plausibly patient setup execution. Curriculum can over-sample setup-team cells (Change 7) to *force* the policy to live in that regime, but cannot supply the inductive bias to learn setup execution from scratch. Pairing with BC data on setup play or with exploiter training that explicitly targets setup-team mastery is required, and lives outside this plan's scope. Listed here so future readers know the curriculum is necessary-but-insufficient for the foul-play target.

**Updated measurement methodology** (replaces "primary metric = `win_rate_vgc_bench_baseline`"):

- The Stage II graduation criterion is now a 4-tuple: ≥60% simultaneously vs `vgc_bench_baseline`, `max_damage`, `bc_player`, `simple_heuristic_baseline`. Optimizing a single in-curriculum win rate can pull others down — measurement must track all four.
- **Foulplay held-out eval** ([2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md)) shipped. Use it as the held-out slice the original plan called for: in-curriculum win rates are contaminated by curriculum shift; foulplay eval is sampled from a fixed distribution and is comparable across runs.
- When comparing two adaptive-curriculum variants, hold the VGCBench *curriculum weight* fixed across the runs so per-GPU-hour matchup counts are comparable.
- **Per-agent-team WR distributions** are now a first-class signal (2026-05-22). Use the eval pipeline that produced MODEL_EVALUATION Q1/Q2 to check whether Change 7 actually moves the bottom of the per-team distribution. The headline metric is the WR of the *worst-piloted* team against each baseline; secondary metric is the variance across teams (lower = more uniform competence).

---

## Solution: Seven changes, re-grouped and re-sequenced (2026-05-22)

Re-grouping after MODEL_EVALUATION:
- **Group A** (Changes 2 + 4 + 5) is unchanged; ships as one combined PR; designed and implementable now.
- **Group A+** (new Change 7, agent-team-axis sampling) is co-priority with Group A but design is pending a brainstorm session; ships independently once designed. The two groups touch orthogonal axes (opponent vs agent_team) and do not block each other.
- **Group B** (Change 3) narrowed: exploiter sub-PFSP kept at moderate priority; ghost sub-PFSP deferred until a diagnostic confirms ghost-vs-ghost variation is non-trivial.
- **Group C** (Changes 1, 6) remain optional polish, with Change 1 re-bundling with Change 7 once the per-cell tracking expands.

The primary measurement is the **4-tuple of baseline win rates** plus the per-agent-team WR distribution (with worst-piloted-team WR as headline), with the foulplay held-out eval as the cross-run-comparable signal.

### Group A (one PR): Changes 2 + 4 + 5 — the algorithm cleanup

#### Change 4 — Asymmetric PFSP, drop weakness term

Replace the hybrid PFSP+weakness with a single asymmetric PFSP score:
```
pfsp_score = (1.0 - wr) ** p   # default p = 1.0, configurable
learning_value = pfsp_score
```
Drop the weakness term entirely. Make `p` a config field
(`config.curriculum.pfsp_exponent: float = 1.0`).

**Why first in this group**: the asymmetric form is cleaner to reason about when removing other stabilization terms below.

**Acceptance**: unit test that an opponent with `wr=0.05` scores higher than one at `wr=0.50`.

#### Change 2 — Drop the `0.5 * base_weight` mix

Change `candidate = max(eps, 0.5 * base + 0.5 * learning_value)` to
`candidate = max(eps, learning_value)`. Anchor floors remain the only
stabilizer.

**Acceptance**: integration test — simulate an opponent that the policy crushes (wr=0.95) for many windows; weight falls to its floor (or eps for non-anchor types). Currently it stabilizes around `0.5 * base + small`.

#### Change 5 — Half-life decay window for win-rate tracking

Replace `deque(maxlen=100)` with an exponentially-weighted estimator:
- Maintain `ewma_wins[opp_type]: float` and `ewma_n[opp_type]: float`.
- On each `record_battle_result`: multiply both by decay `λ = 0.5 ** (1/half_life)`, then add the new sample. Default `half_life = 50` battles.
- `wr = (ewma_wins + alpha) / (ewma_n + alpha + beta)`.

**Why now**: Change 2 removes the base-mix stabilizer; without EWMA, an underweighted opponent's stale deque produces oscillations.

**Acceptance**: unit test that an opponent sampled rarely doesn't keep an indefinitely stale wr estimate.

**Files touched (whole group)**: `opponents.py`, `config.py`. Optionally migrate `get_win_rate_stats` reporting to EWMA throughout (cleaner) rather than keeping the deque for backward-compatible reporting — `CLAUDE.md` notes backward compat is not a concern.

### Group A+ (co-priority with Group A; design pending brainstorm): Change 7 — Agent-team-axis adaptive sampling

**Status (2026-05-22)**: stub. Design to be filled in by a brainstorm session immediately following this re-scope. This subsection captures motivation, constraints, and open design questions; implementation details are TBD.

**Motivation**: MODEL_EVALUATION Q1 shows a 35-point WR spread across 42 agent teams against heuristics and 28 points against vgc_bench. Q2 shows zero per-opponent-team adaptation against vgc_bench (WR std = 0.006). Only one cell (`38dessert` × simple_heuristic) has a CI upper bound > 50%. The cross-pattern summary shows the policy can pilot direct-offensive cores and cannot pilot setup-reliant teams. Without team-axis adaptation, training pressure on the under-performed teams is diluted across the 35+ teams the policy already handles consistently.

**Why co-priority with Group A**: this is likely the dominant Elo lever for Stage II graduation given the eval data, and the only realistic curriculum response to the single-strategy pathology that gates foul-play viability (problem item #10). It is orthogonal to the opponent axis Group A operates on, so the two groups can proceed independently.

**Open design questions for the brainstorm**:
- Granularity: per-`agent_team` only (42 cells), or per-`(agent_team, opp_type)` cell (42 × K cells). Latter is more informative but sample-hungry.
- Where the sampling lives: worker-side via a broadcast team distribution, or main-process via a curriculum-aware `team_provider`. Worker-side mirrors how opponent sampling works post-2026-05-14; main-process is a smaller change.
- Relationship to the opponent-axis curriculum: independent samplers, or a joint distribution over (team, opp_type) cells. Independent is simpler; joint may be required to handle interaction effects (e.g. setup teams might need over-sampling specifically against direct-offensive opponents).
- Score function: reuse Change 4's asymmetric PFSP `(1 - wr)**p`, or use a different scoring shape (e.g. UCB-style with cell-sample counts).
- Treatment of zero-sample cells at start: uniform initialization, BC-derived prior, or eval-derived prior from MODEL_EVALUATION's per-team distribution.

**Acceptance (placeholder)**: integration test demonstrating that biased team sampling shifts a synthetic policy's per-team WR distribution toward uniformity within K updates. Real-run validation: bottom-quartile per-team WR climbs faster under Change 7 than under uniform team sampling, holding Group A's algorithm constant.

**Files touched (placeholder)**: `opponents.py` or a new sibling module for team-axis tracking, `team_provider.py`, `worker.py` (if broadcast-driven), `train.py` (record per-team WR; broadcast team distribution if applicable).

### Group B (separate PR, narrowed): Change 3 — Sub-PFSP within exploiters; ghost sub-PFSP deferred

Within-bucket PFSP for **exploiters** stays at moderate priority. Exploiters are trained to differ from each other by construction, so cross-exploiter WR variation is plausible and within-bucket PFSP should produce a real lift.

Within-bucket PFSP for **ghosts** is **deferred (2026-05-22)** pending a diagnostic. MODEL_EVALUATION Q2 found WR std across 42 opp_teams against vgc_bench is 0.006 — the model has essentially no per-opponent-team adaptation. If ghosts are snapshots of a policy with that pathology, they likely also have low cross-ghost WR variance, and within-bucket PFSP just rotates among same-flavor checkpoints. Before implementing ghost sub-PFSP:

- **Diagnostic**: pit each ghost slot against the current main model for ~500 battles per slot and compute WR mean + std across slots. If std < 0.02, ghost sub-PFSP gives no signal and stays deferred; if std ≥ 0.05, lift the deferral and implement as originally planned.

**Exploiter-side implementation** (the kept half of original Change 3):
- Add `self.exploiter_win_rates: Dict[str, (ewma_wins, ewma_n)]` keyed by exploiter path (or slot index). Use the EWMA estimator from Change 5.
- Trajectory protocol: workers emit an `opponent_id` (path or slot index) on the trajectory dict when `opponent_type == "exploiters"`. Currently `traj["opponent_type"]` is the only identifier ([train.py:937](../../src/elitefurretai/rl/train.py#L937)).
- `record_battle_result` extended to accept `opponent_id: Optional[str] = None`; routes to the per-id EWMA when type is `exploiters`.
- **Broadcast change**: sampling lives in `WorkerOpponentFactory.sample_opponent_type()` ([opponents.py:819](../../src/elitefurretai/rl/opponents.py#L819)). Main process broadcasts `exploiter_subdist: Dict[slot_index, weight]` in addition to type-level curriculum.
- Worker-side: when type is `exploiters`, sample slot from the broadcast subdistribution. The factory already chooses the per-slot model in its battle setup — the change is which slot it picks.
- Main-process score: per-id `sample_weight_i ∝ (1 - wr_i) ** p` using the same `p` from Change 4.

**Files touched**: `opponents.py` (per-id exploiter tracking, subdistribution build, broadcast extension), `worker.py` (trajectory dict — emit `opponent_id` for exploiters), `train.py` (propagate `opponent_id` into `record_battle_result`, broadcast `exploiter_subdist`).

**Acceptance**: integration test with a synthetic 3-exploiter setup where one is "hard" (50% wr) and two are "easy" (90% wr); within-bucket sampling converges to ≥60% on the hard exploiter. Type-level exploiter weight unchanged.

### Group C (optional polish, do only if motivated by data)

#### Change 1 — Drop forfeits from win-rate signal + length-weight

**Status downgraded.** Motivation has weakened: most invalid-choice forfeit sources are fixed. Forfeit drop is still principled (don't credit invalid-action losses to the policy), but it's no longer load-bearing for adaptation quality.

If implemented:
- In `record_battle_result`, when `forfeited=True`, **do not update the EWMA** for that opponent type. Still increment a new per-type `forfeits_per_type` counter for observability.
- Length-weighted EWMA update: weight `= min(length / median_length, 2.0)` applied to both `ewma_wins` and `ewma_n` increments.

**Acceptance**: unit test that synthetic forfeit-heavy data doesn't change curriculum weights when forfeits dominate. Unit test that long battles weight more than short ones.

#### Change 6 — Decouple curriculum recompute from checkpoint cadence

**Status downgraded.** At current ~5-6 traj/s throughput, each checkpoint interval already accumulates enough battles for adaptation to be meaningful. Defer unless `curriculum_weight_*` traces in wandb show clear stair-step artifacts attributable to recompute cadence (currently they don't).

If implemented: call `update_curriculum()` every N updates (config: `config.curriculum.update_interval_steps: int = 10`); broadcast still on checkpoint cadence using latest curriculum.

### Exploration handle (Change 8 from problem list — now addressed)

Zero-base opponents will not become non-zero through this algorithm. If we want to surface unused baselines (e.g., `random_baseline`), assign them a small non-zero base weight (e.g., 0.01) so the learning signal can grow them, or add a small uniform floor across all available types. Recommend the former — explicit, no new mechanism.

### **NOTE on exploiter training**:
Adaptive curriculum shouldn't touch *whether/how we train exploiters*, only *how the main model samples against the frozen exploiter snapshots*. The `train_exploiter` opponent type (exploiter learner's own data plane) is out of scope.

---

## Reasoning

**Why group A as one PR?** Each of Changes 2, 4, 5 in isolation creates instability that the next one fixes:
- Change 4 alone keeps the base-mix that prevents real adaptation.
- Change 2 alone exposes deque staleness for underweighted opponents.
- Change 5 alone leaves the symmetric PFSP + base-mix combo that already squelches adaptation.

Together they are coherent; separately each is hard to evaluate. Trying to A/B them one-at-a-time would also burn GPU-hours on three short runs that mostly tell us "the algorithm is still over-stabilized."

**Why Change 7 as co-priority with Group A (2026-05-22)?** MODEL_EVALUATION showed the agent-team axis carries 35 points of WR variation while the opponent-team axis carries 0.6 (Q1 vs Q2 against vgc_bench). The curriculum currently has zero visibility into agent_team. Among the curriculum-shaped responses available, this one targets the variance that actually exists in the data. The brainstorm precedes implementation because the design has open questions (granularity, joint vs independent with opponent axis, sample initialization) that materially change the scope.

**Why split Change 3 by bucket (2026-05-22)?** Group B's expected lift was originally framed as "AlphaStar's reported gain from within-bucket PFSP." That estimate assumed within-bucket diversity. MODEL_EVALUATION Q2's WR std = 0.006 against vgc_bench is evidence the policy lacks per-opponent variation; ghosts (snapshots of that policy) likely inherit it. Exploiters do not — they are trained to differ from each other. Implementing exploiter sub-PFSP first preserves the cheap lift; running a 500-battle ghost-vs-main diagnostic before implementing ghost sub-PFSP avoids burning protocol-change scope on a no-signal axis.

**Why drop the weakness term in Change 4 rather than tuning it?** Same reasoning as the original plan: `weakness_score = max(0, (target - wr) / target)` is a clipped, rescaled version of `(1 - wr)` near the threshold. Asymmetric PFSP gives the same behavior with one fewer hyperparameter and a smoother curve.

**Why half-life rather than larger `tracking_window`?** A larger window doesn't solve the underweighted-opponent staleness — the deque still doesn't drain when samples don't arrive. EWMA shrinks naturally toward the prior over wall-clock time, which is the property we want.

**Why per-checkpoint tracking in Change 3 keyed by path/slot?** Both are unique. Slot index is more compact for broadcast; path is more readable for logging. Keyed by path internally, broadcast as slot-indexed dict.

**Why foulplay eval instead of held-out VGCBench slice?** Foulplay eval is a separately-trained agent (held-out by construction) and was integrated specifically as a cross-run-comparable signal ([2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md)). The original plan's "eval slice" was conceptual; foulplay is concrete.

**Why downgrade Change 1?** Forfeit volume dropped substantially after the invalid-choice bug fixes. The fix is still correct, but it no longer gates measurement quality. Length-weighting is still valuable but is now a polish item, not foundational.

---

## Hard constraints to respect during implementation

- `pin_memory=False` always (WSL2).
- `torch.multiprocessing.set_sharing_strategy('file_system')` already handled via `configure_torch_multiprocessing`.
- Rust backend must keep working — opponent-path changes go through the worker protocol, which both backends share.
- No try/except hiding errors — fix root causes (`CLAUDE.md`).
- Lint exclusion for `src/elitefurretai/scripts/` only.
- Backward compat is not a concern — when migrating reporting from deque to EWMA in Change 5, just change the code rather than keeping both.

---

## Planned Next Steps

When implementation starts (a future session may pick this up cold):

1. **Read first**:
   - This doc (you're reading it).
   - [src/elitefurretai/rl/opponents.py:104-479](../../src/elitefurretai/rl/opponents.py#L104-L479) — `OpponentPool`. Especially `record_battle_result` (line 298), `update_curriculum` (line 367).
   - [src/elitefurretai/rl/opponents.py:481-900](../../src/elitefurretai/rl/opponents.py#L481-L900) — `WorkerOpponentFactory`, especially `sample_opponent_type` (line 819) — this is where Change 3's within-bucket sampling lives.
   - [src/elitefurretai/rl/train.py:920-1115](../../src/elitefurretai/rl/train.py#L920-L1115) — main loop where `record_battle_result` is called (line 936) and `update_curriculum` fires (line 1110).
   - [src/elitefurretai/rl/worker.py](../../src/elitefurretai/rl/worker.py) — confirm trajectory dict shape; Change 3 adds an `opponent_id` field here.
   - [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py) — curriculum config for new fields (`pfsp_exponent`, `half_life`, `update_interval_steps`).
   - [src/elitefurretai/rl/model_registry.py](../../src/elitefurretai/rl/model_registry.py) — slot-addressed ghost/exploiter identity that Change 3 leverages.
   - [src/elitefurretai/rl/RL.md](../../src/elitefurretai/rl/RL.md) §7 ("The Opponent Pool & Adaptive Curriculum") — update after each PR.
   - [planning/stage2/2026-05-16-21-30-stage2-graduation-criteria.md](2026-05-16-21-30-stage2-graduation-criteria.md) — the 4-baseline measurement target.
   - [planning/stage2/2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md) — the held-out eval slice.
   - [planning/stage2/2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md](2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md) — the diagnostic that promoted Change 2 to top priority.

2. **Implementation order (2026-05-22)** — separate PRs so each is measurable against the 4-baseline tuple + foulplay eval + per-team WR distribution:
   1. **Group A**: Changes 4 + 2 + 5 in one PR. Unit tests for each, then one short run to confirm adaptation actually moves under the new algorithm. Implementable now; designed.
   2. **Brainstorm Change 7** design before any code lands. Output is a separate planning doc with granularity, sampler location, joint-vs-independent decision, initialization strategy, and acceptance criteria.
   3. **Group A+ (Change 7)** ships independently once designed. Can interleave with Group A or follow it depending on which finishes its run first.
   4. **Ghost-variance diagnostic** for Group B: 500 battles per ghost slot against current main model; compute WR std.
   5. **Group B (Change 3, exploiter-only)**: trajectory protocol update + worker sampling + per-id tracking for exploiters. Unit + integration tests. Medium-length run to measure the within-exploiter lift. If the diagnostic in step 4 returns std ≥ 0.05, extend Group B to ghosts in a follow-up PR.
   6. **Group C** (optional): Change 1 re-bundled with Change 7 if per-cell forfeit volume becomes a measurement-noise concern; Change 6 only if recompute-cadence stair-stepping shows up in wandb.

3. **Per PR**: update `RL.md` §7 and create a short completion doc in `planning/stage2/` per project convention.

4. **Final validation**: one full Stage II run (whatever is the standard length at that time) measuring the 4-baseline tuple per GPU-hour vs. a baseline run with the original algorithm. Target: at least one of the four baselines climbs ≥10% faster; ideally adaptation pushes the *limiting* baseline (whichever is furthest from 60%) hardest, since the graduation criterion is the min over all four.

---

## Updates

- **2026-05-22 11:55** — Reprioritization in light of MODEL_EVALUATION.md (balmy70 step-10700 full eval, n=476,650). Added problem item #9 (agent-team axis ignored) and #10 (single-strategy pathology limits foul-play viability). Added Change 7 (agent-team-axis adaptive sampling) as a new co-priority with Group A; design pending a brainstorm session. Split original Change 3 into exploiter sub-PFSP (kept, moderate priority) and ghost sub-PFSP (deferred pending a 500-battle ghost-vs-main variance diagnostic), motivated by Q2's WR std = 0.006 against vgc_bench suggesting ghosts inherit a no-per-opponent-variation pathology. Modestly re-elevated Change 1 because Change 7 expands per-cell tracking and per-cell forfeit noise matters more at that granularity. Added per-agent-team WR distribution as a first-class measurement signal alongside the 4-baseline tuple and foulplay eval. Renamed "Six changes" → "Seven changes" in the Solution section heading. Reordered Planned Next Steps to surface the Change 7 brainstorm + ghost-variance diagnostic as new gating steps.
- **2026-05-20 15:00** — Refreshed for current code state. Promoted Change 2 to top priority based on the 2026-05-16 update-100 cliff diagnostic. Re-grouped Changes 2/4/5 as one PR. Downgraded Changes 1 and 6 to optional polish (forfeit volume dropped after invalid-choice fixes; throughput regime makes decoupled recompute unnecessary). Updated measurement methodology from single-metric (`win_rate_vgc_bench_baseline`) to the 4-baseline graduation tuple + foulplay held-out eval. Added broadcast-side scope to Change 3 reflecting that sampling now lives in `WorkerOpponentFactory`. File renamed from `2026-05-03-23-37-adaptive-curriculum-overhaul.md` via `git mv` so it sorts as current.
- **2026-05-03 23:37** — Original plan written.
