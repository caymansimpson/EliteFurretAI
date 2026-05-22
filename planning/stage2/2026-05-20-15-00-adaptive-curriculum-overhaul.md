# Adaptive Curriculum Overhaul

**Date**: 2026-05-20 15:00 (refresh of 2026-05-03 23:37 plan)
**Status**: Planned, not implemented — refreshed for current code state
**Owner**: Cayman

> **Why this refresh**: the original plan dated 2026-05-03 was still entirely on paper as of 2026-05-20. Between those dates the codebase shifted in ways that change scope and priority for several of the proposed changes (centralized inference, ghost centralization + model registry, slot-addressed ghosts/exploiters, sampling relocated into the worker, most invalid-choice forfeit bugs fixed, Stage II graduation criterion broadened to 4 baselines, foulplay held-out eval shipped). The "Before State" math is unchanged; the *plan around it* needed re-anchoring.

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

Eight identified weaknesses; severity has shifted since 2026-05-03 in light of recent diagnostics.

1. **Flat ghost/exploiter buckets** *(still high)*: all ghost checkpoints (and all exploiters) sampled uniformly within their bucket. AlphaStar credits within-bucket PFSP as a major contributor to skill gains; we currently get none of that. With slot-addressed ghosts/exploiters now in place, the *implementation* of within-bucket PFSP got cheaper — the per-checkpoint identity already exists in `slot_for_ghost_path` / `slot_for_exploiter_path`. The *broadcast* side got slightly more involved because sampling now lives in workers.
2. **Symmetric PFSP score** *(still high)*: `1 - 2|wr - 0.5|` doesn't differentiate "we crush them" from "we're crushed."
3. **Double-stabilization** *(promoted to highest priority)*: `0.5 * base_weight + 0.5 * learning_value` ties the curriculum to its initial values forever. The "update-100 cliff was VGCBench, not ghosts" diagnostic ([2026-05-16-08-13](2026-05-16-08-13-update100-cliff-was-vgcbench-not-ghosts.md)) confirmed VGCBench dominates the in-curriculum learning signal — and yet adaptation is being squelched by this base-mix.
4. **`battle_length_tracking` collected but unused** *(unchanged)*.
5. **Forfeits counted as losses** *(downgraded)*: most invalid-choice forfeit sources have been fixed (Commander, Uproar, force-switch families per `CLAUDE.md`). Two residual bugs remain ([2026-04-26-22-00](2026-04-26-22-00-two-residual-bugs.md)) but baseline forfeit volume is much lower than when the original plan was drafted. Still worth dropping forfeits from the signal for cleanliness, but it is no longer load-bearing.
6. **Stale-window risk** *(unchanged, still high after Change 2)*: the deque doesn't drain when samples don't arrive; min-sample gate of 40 compounds the lag.
7. **Update cadence = checkpoint cadence** *(downgraded)*: at current throughput (~5-6 traj/s, per the throughput memory), each checkpoint interval already accumulates plenty of battles for adaptation to be statistically meaningful. The decoupling is no longer urgent.
8. **Zero-base opponents stay zero** *(unchanged)*: `RANDOM_BASELINE=0.0` etc. → 0 forever; no exploration mechanism.

**Updated measurement methodology** (replaces "primary metric = `win_rate_vgc_bench_baseline`"):

- The Stage II graduation criterion is now a 4-tuple: ≥60% simultaneously vs `vgc_bench_baseline`, `max_damage`, `bc_player`, `simple_heuristic_baseline`. Optimizing a single in-curriculum win rate can pull others down — measurement must track all four.
- **Foulplay held-out eval** ([2026-05-18-16-00-foulplay-eval-integration.md](2026-05-18-16-00-foulplay-eval-integration.md)) shipped. Use it as the held-out slice the original plan called for: in-curriculum win rates are contaminated by curriculum shift; foulplay eval is sampled from a fixed distribution and is comparable across runs.
- When comparing two adaptive-curriculum variants, hold the VGCBench *curriculum weight* fixed across the runs so per-GPU-hour matchup counts are comparable.

---

## Solution: Six changes, re-grouped and re-sequenced

Re-grouping vs. the original plan: Changes 2, 4, 5 are tightly coupled (each addresses a different failure mode that the others expose), are all `opponents.py`-local, and should ship as **one combined PR**. Change 3 follows as its own PR because it now requires a small protocol addition. Changes 1 and 6 are optional polish given the downgrades above.

The primary measurement is the **4-tuple of baseline win rates** (above), with the foulplay held-out eval as the cross-run-comparable signal.

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

### Group B (separate PR): Change 3 — Sub-PFSP within ghosts and exploiters

Highest expected Elo lift. Stop treating "ghosts" and "exploiters" as flat buckets.

**What** (revised given current architecture):
- Add `self.ghost_win_rates: Dict[str, (ewma_wins, ewma_n)]` keyed by ghost path (or slot index), same for `exploiter_win_rates`. Use the EWMA estimator from Change 5.
- Trajectory protocol: workers must emit an `opponent_id` (path or slot index) on the trajectory dict for `ghosts` and `exploiters`. Currently `traj["opponent_type"]` is the only opponent identifier ([train.py:937](../../src/elitefurretai/rl/train.py#L937)) — `opponent_id` is the new field.
- `record_battle_result` extended to accept `opponent_id: Optional[str] = None`; routes to the per-id EWMA when type is `ghosts`/`exploiters`.
- **Broadcast change** (this is the part the original plan understated): because sampling now lives in `WorkerOpponentFactory.sample_opponent_type()` ([opponents.py:819](../../src/elitefurretai/rl/opponents.py#L819)), main process must broadcast per-bucket distributions, not just type-level curriculum. Add to the curriculum broadcast: `ghost_subdist: Dict[slot_index, weight]` and `exploiter_subdist: Dict[slot_index, weight]`.
- Worker-side: extend `WorkerOpponentFactory` to (a) sample type from the type-level curriculum, then (b) if type ∈ {ghosts, exploiters}, sample slot from the broadcast subdistribution. The factory already chooses the per-slot model in its battle setup — the change is *which* slot it picks.
- Main-process score: per-id `sample_weight_i ∝ (1 - wr_i) ** p` using the same `p` from Change 4.

**Files touched**: `opponents.py` (per-id tracking, subdistribution build, broadcast extension), `worker.py` (trajectory dict — emit `opponent_id`), `train.py` (propagate `opponent_id` into `record_battle_result`, broadcast subdistributions).

**Acceptance**: integration test with a synthetic 3-ghost setup where one ghost is "hard" (50% wr) and two are "easy" (90% wr); within-bucket sampling converges to ≥60% on the hard ghost. Type-level ghost weight unchanged.

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

**Why Change 3 as its own PR?** It touches the worker → trainer protocol (adds `opponent_id`), the broadcast format (adds subdistributions), and adds new tracking dicts. It's also the biggest expected lift on its own (AlphaStar's reported gain from within-bucket PFSP), so isolating it makes attribution clean.

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

2. **Implementation order** (separate PRs so each is measurable against the 4-baseline tuple + foulplay eval):
   1. **Group A**: Changes 4 + 2 + 5 in one PR. Unit tests for each, then one short run to confirm adaptation actually moves under the new algorithm.
   2. **Group B**: Change 3 alone. Trajectory protocol update + worker sampling + per-id tracking. Unit + integration tests. Medium-length run to measure the within-bucket lift.
   3. **Group C** (optional): Change 1 first if forfeit telemetry shows non-trivial residual volume after `2026-04-26-22-00-two-residual-bugs.md` is closed; Change 6 only if recompute-cadence stair-stepping shows up in wandb.

3. **Per PR**: update `RL.md` §7 and create a short completion doc in `planning/stage2/` per project convention.

4. **Final validation**: one full Stage II run (whatever is the standard length at that time) measuring the 4-baseline tuple per GPU-hour vs. a baseline run with the original algorithm. Target: at least one of the four baselines climbs ≥10% faster; ideally adaptation pushes the *limiting* baseline (whichever is furthest from 60%) hardest, since the graduation criterion is the min over all four.

---

## Updates

- **2026-05-20 15:00** — Refreshed for current code state (this version). Promoted Change 2 to top priority based on the 2026-05-16 update-100 cliff diagnostic. Re-grouped Changes 2/4/5 as one PR. Downgraded Changes 1 and 6 to optional polish (forfeit volume dropped after invalid-choice fixes; throughput regime makes decoupled recompute unnecessary). Updated measurement methodology from single-metric (`win_rate_vgc_bench_baseline`) to the 4-baseline graduation tuple + foulplay held-out eval. Added broadcast-side scope to Change 3 reflecting that sampling now lives in `WorkerOpponentFactory`. File renamed from `2026-05-03-23-37-adaptive-curriculum-overhaul.md` via `git mv` so it sorts as current.
- **2026-05-03 23:37** — Original plan written.
