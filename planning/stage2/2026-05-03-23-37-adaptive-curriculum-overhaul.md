# Adaptive Curriculum Overhaul

**Date**: 2026-05-03 23:37
**Branch**: `rl-throughput-opts` (or successor)
**Status**: Planned, not implemented
**Owner**: Cayman

---

## Context

The adaptive curriculum lives in `OpponentPool.update_curriculum()` at
[src/elitefurretai/rl/opponents.py:719-830](../../src/elitefurretai/rl/opponents.py#L719-L830).
It is called from [src/elitefurretai/rl/train.py:798-799](../../src/elitefurretai/rl/train.py#L798-L799)
once per `checkpoint_interval`, immediately before broadcasting weights + curriculum
to workers. The curriculum is a probability distribution over opponent *types*
(`self_play`, `bc_player`, `exploiters`, `ghosts`, plus heuristic baselines and
`vgc_bench_baseline`); workers sample from it via `sample_opponent()`
([opponents.py:370-432](../../src/elitefurretai/rl/opponents.py#L370-L432)).

Per-opponent state used by the algorithm:
- `self.win_rate_tracking[opp_type]: deque(maxlen=tracking_window=100)` of 0/1 outcomes.
- `self.battle_length_tracking[opp_type]: deque(maxlen=100)` of ints — **collected but unused in update**.
- `self.total_forfeits_tracked` — forfeits *aggregated globally* but not separated per-type and not excluded from win rate.
- `record_battle_result(opponent_type, won, battle_length, forfeited)` is called once per finished trajectory from [train.py:654-659](../../src/elitefurretai/rl/train.py#L654-L659).

Project goals (per `CLAUDE.md`, `RL.md`, and the project-vision memory):
- Stage II: Single-Team League Mastery via r-NaD initialized from BC.
- Self-play + ghosts + exploiters ensemble; portfolio of reference snapshots
  (loss-side anchor) is distinct from the curriculum (sampling-side anchor).
- We want Nash-style convergence with low cyclic drift; BC anchor preserves
  human-likeness.

---

## Before State (current algorithm, summary)

Per call to `update_curriculum`:

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

---

## Problem

Eight identified weaknesses, ordered by severity:

1. **Flat ghost/exploiter buckets**: all ghost checkpoints (and all exploiters) sampled
   uniformly within their bucket. AlphaStar credits within-bucket PFSP as a major
   contributor to skill gains; we currently get none of that.
2. **Symmetric PFSP score**: `1 - 2|wr - 0.5|` doesn't differentiate "we crush them"
   from "we're crushed." The weakness term partially compensates but is mixed
   asymmetrically (0.3 weight) and saturates at `wr ≥ 0.55`.
3. **Double-stabilization**: `0.5 * base_weight + 0.5 * learning_value` ties the
   curriculum to its initial values forever. Combined with 40% locked anchor
   floors, real adaptation has only ~30% of the mass to play with.
4. **`battle_length_tracking` collected but unused**: long battles carry more
   decision density (more learning signal per battle); ignoring this miscredits
   short forfeit-heavy matchups.
5. **Forfeits counted as losses**: forfeits caused by invalid-choice bugs (still
   present per `2026-04-26-22-00-two-residual-bugs.md`) corrupt win-rate signal.
6. **Stale-window risk**: when an opponent's curriculum weight drops near floor,
   its deque drains slowly and the win-rate it reports lags many policy updates
   behind reality. Min-sample gate (40) compounds this — the type can get stuck.
7. **Update cadence = checkpoint cadence**: curriculum recomputation only happens
   when weights are broadcast. Math is cheap; we could update internally more
   often even if broadcast stays on checkpoint cadence.
8. **Zero-base opponents stay zero**: `RANDOM_BASELINE=0.0` (etc.) initially → 0
   forever. No exploration mechanism to surface useful baselines.

Progress will be measured by performance against `vgc_bench_baseline` (in-curriculum,
already tracked) rather than by a separate held-out eval slice — VGC-Bench
matchups are the documented progress signal for Stage II.

---

## Solution: Six changes, sequenced by dependency

The primary metric for evaluating each change is **win rate vs.
`vgc_bench_baseline` per GPU-hour**, as logged today via
`win_rate_vgc_bench_baseline` in `get_training_metrics`. Secondary metrics:
time-to-plateau on that win rate, and exploitability gap (vs. an exploiter
trained against the final policy).

### Change 1 — Length-weight + drop forfeits

**Goal**: clean the win-rate signal feeding every other adaptation step.

**What**:
- In `record_battle_result`, when `forfeited=True`, **do not append to
  `win_rate_tracking`** for that opponent type. Still increment
  `total_forfeits_tracked` and a new per-type `forfeits_per_type` counter for
  observability.
- Replace the simple win-rate computation in `update_curriculum` with a
  length-weighted version:
  - For each battle in the window, weight = `min(length / median_length, 2.0)`
    where `median_length` is the global median across all tracked battles.
  - `wins_w = sum(weight_i * outcome_i)`; `n_w = sum(weight_i)`.
  - `wr = (wins_w + alpha) / (n_w + alpha + beta)` (still Beta(8,8) prior).
- Keep the Beta prior. Keep `min_samples=40` but apply it on `n_w` (weighted).

**Files touched**: `opponents.py` only.

**Acceptance**: unit test that synthetic forfeit-heavy data doesn't change
curriculum weights when forfeits dominate. Unit test that long battles weight
more than short ones.

### Change 4 — Asymmetric PFSP, drop weakness term

**Goal**: cleaner semantics, fewer hyperparameters.

**What**: replace the hybrid PFSP+weakness with a single asymmetric PFSP score:
```
pfsp_score = (1.0 - wr) ** p   # default p = 1.0, configurable
```
Drop the weakness term entirely. `learning_value = pfsp_score`. Make `p` a
config field (`config.curriculum.pfsp_exponent: float = 1.0`).

**Why this is safe after Change 1**: the symmetric tent's main motivation was
"learn most where gradients are densest"; with a clean win-rate signal, the
asymmetric form aligns directly with "play your hardest matchups," which is
what we actually want for Nash convergence in a single-team setting.

**Files touched**: `opponents.py`, `config.py`.

**Acceptance**: unit test that an opponent with wr=0.05 gets a higher score
than one at wr=0.50 (currently the reverse for the PFSP component).

### Change 2 — Drop the `0.5 * base_weight` mix

**Goal**: let adaptation actually adapt.

**What**: change `candidate = max(eps, 0.5 * base + 0.5 * learning_value)` to
`candidate = max(eps, learning_value)`. Anchor floors remain the only
stabilizer.

**Sequenced after Change 4** because asymmetric PFSP is cleaner to reason about
when removing a stabilization term.

**Files touched**: `opponents.py`.

**Acceptance**: integration test: simulate an opponent that the policy crushes
(wr=0.95) for many windows; weight should fall to its floor (or eps for
non-anchor types). Currently it would stabilize around `0.5 * base + small`.

### Change 5 — Half-life decay window for win-rate tracking

**Goal**: kill the stale-deque problem that becomes more visible after Change 2.

**What**: replace the `deque(maxlen=100)` with an exponentially-weighted estimator:
- Maintain `ewma_wins[opp_type]: float` and `ewma_n[opp_type]: float`.
- On each `record_battle_result`: multiply both by decay `λ = 0.5 ** (1/half_life)`,
  then add the new sample. Default `half_life = 50` battles.
- `wr = (ewma_wins + alpha) / (ewma_n + alpha + beta)`.
- For length-weighting (Change 1), apply weight to both the wins and n updates.

**Why now**: Change 2 removed the base-mix stabilizer; the deque's stale-when-
underweighted behavior would otherwise produce oscillations.

**Files touched**: `opponents.py`. Keep the existing `win_rate_tracking` deque
for backward-compatible `get_win_rate_stats` reporting, *or* migrate stats to
EWMA throughout (cleaner).

**Acceptance**: unit test that an opponent sampled rarely doesn't keep an
indefinitely stale wr estimate.

### Change 3 — Sub-PFSP within ghosts and exploiters

**Goal**: highest expected Elo lift. Stop treating "ghosts" and "exploiters" as
flat buckets.

**What**:
- Track per-checkpoint win rate: `self.ghost_win_rates: Dict[str, EWMA]` and
  `self.exploiter_win_rates: Dict[str, EWMA]`, keyed by checkpoint path.
- `record_battle_result` extended (or wrapped) to accept an `opponent_id` (path
  or step) when the type is `ghosts`/`exploiters`. Workers must report this in
  the trajectory dict.
- `sample_opponent` for ghost/exploiter types: instead of uniform, sample
  proportionally to `(1 - wr_i) ** p` with the same `p` as Change 4.
- Bucket-level mass in the curriculum stays governed by the type-level adaptive
  algorithm; the within-bucket distribution is sub-PFSP.

**Files touched**: `worker.py` (emit opponent id in trajectory), `train.py`
(propagate id when calling `record_battle_result`), `opponents.py` (track,
sample).

**Acceptance**: integration test with a synthetic 3-ghost setup where one
ghost is "hard" (50% wr) and two are "easy" (90% wr); within-bucket sampling
should converge to ≥60% on the hard ghost.

### Change 6 — Decouple curriculum recompute from checkpoint cadence

**Goal**: smoother metrics, faster reaction.

**What**: call `update_curriculum()` every N updates (config:
`config.curriculum.update_interval_steps: int = 10`), independent of checkpoint
broadcast. The broadcast still happens at `checkpoint_interval` and uses the
*latest* curriculum at that moment.

**Files touched**: `train.py` (add the inner cadence check), `opponents.py`
(idempotent, already safe to call repeatedly).

**Acceptance**: no behavioral test needed beyond verifying no regression; just
confirm wandb traces of `curriculum_weight_*` are smoother.

---

## Reasoning

**Why this ordering?** Changes 1, 4 simplify and clean signals; Change 2
removes a stabilizer that fights real adaptation; Change 5 patches the noise
that 2 exposes; Change 3 is the biggest expected lift but also the most
invasive (touches worker → trainer protocol). Change 6 is polish.

**Measurement note**: VGC-Bench is already in the curriculum and its win rate
is already logged. Each change is evaluated by comparing
`win_rate_vgc_bench_baseline` trajectories across runs (current algorithm vs.
modified). This is in-curriculum so sampling shifts can confound — when in
doubt, hold the VGC-Bench curriculum weight fixed across the runs being
compared so the matchup count per GPU-hour is comparable.

**Why drop the weakness term in Change 4 rather than tuning it?** The original
`weakness_score = max(0, (target_win_rate - wr) / target_win_rate)` is just a
clipped, rescaled version of `(1 - wr)` near the threshold. An asymmetric PFSP
exponent gives the same behavior with one fewer hyperparameter and a smoother
curve.

**Why half-life rather than larger `tracking_window`?** A larger window doesn't
solve the underweighted-opponent staleness — the deque still doesn't drain when
samples don't arrive. EWMA shrinks naturally toward the prior over wall-clock
time, which is the property we want.

**Why per-checkpoint tracking in Change 3 instead of per-version?** Ghosts are
identified by step number in the filename; exploiters by mtime. Both have
unique paths. The path *is* the natural id and is already plumbed through
`add_ghost` / `_load_exploiter_models`.

**Why an eval slice and not just lifetime metrics?** Lifetime win rate by
opponent type is contaminated by curriculum shift. Eval slice is sampled from
a fixed distribution and is therefore comparable across runs and across time.

---

## Hard constraints to respect during implementation

- `pin_memory=False` always (WSL2).
- `torch.multiprocessing.set_sharing_strategy('file_system')` already handled
  via `configure_torch_multiprocessing`.
- Rust backend must keep working — opponent path changes go through worker
  protocol, which both backends share. Verify with both.
- No try/except hiding errors — fix root causes.
- Lint exclusion for `src/elitefurretai/scripts/` only.

---

## Planned Next Steps

When implementation starts (a future session may pick this up cold):

1. **Read first**:
   - This doc (you're reading it).
   - [src/elitefurretai/rl/opponents.py:160-830](../../src/elitefurretai/rl/opponents.py#L160-L830) — `OpponentPool`, especially `record_battle_result`, `update_curriculum`, `sample_opponent`.
   - [src/elitefurretai/rl/train.py:644-836](../../src/elitefurretai/rl/train.py#L644-L836) — main loop where `record_battle_result` is called and broadcast happens.
   - [src/elitefurretai/rl/worker.py](../../src/elitefurretai/rl/worker.py) — confirm trajectory dict shape and where `opponent_type` is set; this is where Change 3 needs an `opponent_id` field added.
   - [src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py) — `RNaDConfig.curriculum` for new fields.
   - [src/elitefurretai/rl/RL.md](../../src/elitefurretai/rl/RL.md) §7 ("The Opponent Pool & Adaptive Curriculum") — update after each change.

2. **Implementation order** (do them as separate commits/PRs so each is
   measurable against `win_rate_vgc_bench_baseline`):
   1. Change 1 (length-weight + forfeit drop) — unit tests + one short run.
   2. Change 4 (asymmetric PFSP, drop weakness) — unit tests + one short run.
   3. Change 2 (drop base-mix) — short run; should be a noticeable adaptation
      speed-up.
   4. Change 5 (EWMA windows) — short run; check `curriculum_weight_*` traces
      are stable.
   5. Change 3 (sub-PFSP within buckets) — biggest change; needs:
      - Trajectory protocol update (worker → trainer).
      - New tracking dicts.
      - Unit + integration tests.
      - Full medium-length run for VGC-Bench win-rate measurement.
   6. Change 6 (decoupled cadence) — last, polish.

3. **Per change**: update `RL.md` (architecture-level) and create a short
   completion doc in `planning/stage2/` per project convention.

4. **Final validation**: one full Stage II run (whatever is the standard
   length at that time) measuring `win_rate_vgc_bench_baseline` per GPU-hour
   vs. a baseline run with the original algorithm. Target: ≥10% improvement;
   null result is still a publishable conclusion.

---

## Updates

(none yet — this doc is a plan)
