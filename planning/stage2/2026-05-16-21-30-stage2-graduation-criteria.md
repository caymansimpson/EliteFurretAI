# Stage II Graduation Criteria

**Date**: 2026-05-16
**Status**: Operational target adopted. Open — no run has met it yet.

The concrete bar that defines "the Stage II training regime is finished and we
can advance." Codifies an intuition Cayman has held for a while into a single
falsifiable success condition.

---

## Context

Stage II is Single-Team League Mastery — train an RL agent that plays one
fixed team well against a broad opponent distribution. Up to this point, runs
have been evaluated on a mix of qualitative ("does this look better than the
last run?") and per-baseline numeric signals, but without a single agreed
graduation threshold. Without a threshold:

- Runs that improve on one axis while regressing on others (firm-field-41,
  hopeful-wood-69 as of update ~750) get judged inconsistently.
- It's unclear when to declare the BC-init + RNaD-anchor recipe "done" and
  move to Stage III work (multi-team, league play with exploiters, etc).
- Curriculum and architecture decisions can't be evaluated against a fixed
  bar — they get justified relative to whatever the prior run did.

## Before state

- `may15.yaml` defines "success looks like" only in opposition to
  firm-field-41 (vgc_bench ≥ 0.10, bc_player ≥ 0.50 not trending down). That's
  a regression-avoidance criterion, not a Stage-II-done criterion.
- `RL.md` and `project_approach_and_vision.md` (memory) describe the
  five-stage plan but don't pin a numeric exit bar for Stage II.
- Win rates against the four fixed baselines are already logged every update,
  so the criterion is observable from existing infrastructure with no new
  instrumentation.

## Problem

Need a single, observable, falsifiable threshold that gates Stage II → Stage
III. Without it, runs drift on multi-axis tradeoffs and "are we improving?"
becomes a debate every session.

## Solution

**Graduation criterion**: a single training run produces a checkpoint that,
when evaluated for ≥200 battles against each of the following four baselines
*independently*, achieves **≥60% win rate against all four simultaneously**:

1. `vgc_bench_baseline` — external Stable-Baselines3 VGCBench checkpoint.
   Diverse RL-style play; the closest thing to "another serious bot."
2. `max_damage` — picks the move that maximizes immediate damage. Tests that
   the model isn't getting baited into bad trades by greedy play.
3. `bc_player` — behavior-cloned model (`cool-bee-85-finetune_best.pt`).
   Tests that RL learning hasn't regressed below the human-imitation prior
   it was initialized from.
4. `simple_heuristic_baseline` — hand-coded VGC rules engine. Tests basic
   tempo/positioning understanding — the canary for whether the model has
   internalized fundamental VGC, not just a narrow self-play equilibrium.

Out of scope as graduation criteria (still logged, but not gating):

- `self_play` — expected to hover near 50% by construction (mirror match).
  A diagnostic for policy stability, not capability.
- `ghosts` — rotating prior selves. A learning-progress indicator, not a
  capability bar.
- `random_baseline`, `max_base_power_baseline` — too easy to be informative.
- `exploiters`, `train_exploiter` — adversarial training partners, not
  evaluation opponents.

### Why ≥60% on all four simultaneously

The conjunction matters more than the threshold value. Hitting 60% against
any single baseline is achievable by overfitting to that opponent's style.
Hitting 60% against all four at once requires generalizing across:

- **vgc_bench**: a serious RL policy, so the model can hold its own against
  another optimizer.
- **max_damage**: pure greed, so the model has learned to *not* trade naively
  when it shouldn't.
- **bc_player**: human-like play, so the model still understands the
  distribution it was initialized from.
- **simple_heuristic**: rule-based play, so the model has learned basic
  tempo/protect/switch patterns, not just RL-vs-RL exotica.

These four span the opponent space — RL, greedy, human-imitation, rule-based.
A policy that beats all four is robust across the strategy distribution it
will face from any future opponent.

The threshold of 60% (not 50% or 70%) reflects that:

- 50% means "not consistently losing" — too weak a bar for a model that
  initialized from BC and burned tens of millions of trajectories.
- 70% means "dominant" — likely demands either much longer training or
  baseline-specific exploitation that defeats the generalization-test purpose.
- 60% means "consistently winning the majority" — a stable advantage that's
  hard to achieve by accident.

## Reasoning

- **Multi-baseline conjunction over single-baseline target.** Single-baseline
  targets (e.g. "beat vgc_bench at 70%") are gameable. Conjunction tests
  whether the policy generalizes, not whether it exploits.
- **Fixed baselines over Elo/ladder.** Elo requires a stable population and
  a long ladder; we don't have one. Fixed baselines are reproducible,
  comparable across runs, and already instrumented.
- **60% over 55% or 65%.** Threshold round-numbering matters less than the
  conjunction; 60% chosen for stability margin over the noise floor (100-
  battle sliding window has ~±5% sampling variance at p=0.5).
- **Evaluation independent of training curriculum.** The graduation check is
  ≥200 battles vs each baseline, *outside the training loop*, with the
  checkpoint frozen. Per-update curriculum win rates are training signal,
  not evaluation evidence.

## Current run status (hopeful-wood-69, update 753)

| Baseline | Current | Trend | Gap |
|---|---|---|---|
| vgc_bench_baseline | 33–50% | ↑ improving from 0% | -10 to -27 pts |
| bc_player | 36–51% | peaked 60% at update 400, drifting | -9 to -24 pts |
| max_damage | 27–45% | ↓ declining from 65% early | -15 to -33 pts |
| simple_heuristic_baseline | 11–26% | flat the entire run | -34 to -49 pts |

All four below target. vgc_bench is the only one trending up; the other three
are flat or declining. simple_heuristic is the worst dimension — the strongest
evidence that the policy is not generalizing.

## Diagnostic playbook (when off-target)

Ordered by ROI. Run highest-ROI probes first; their results determine which
intervention to try.

1. **BC baseline sanity check.** Evaluate `cool-bee-85-finetune_best.pt`
   directly against all four baselines for 200 battles each. If BC already
   hits >50% on any of them, then RL is *regressing* from BC, not failing
   to learn. Cheapest probe; rules out the largest hypothesis class first.
2. **Per-opponent loss/advantage breakdown.** Tag trajectories by opponent
   type in the rollout buffer; log mean advantage, value loss, and policy
   loss per opponent. Identifies which slice the optimizer is ignoring.
3. **Behavioral mining.** Play 10–20 battles vs the worst baseline (today:
   simple_heuristic) using `play_human_vs_model.py --reveal` (or the existing
   `play_model.py --mode=vs-bot --opponent=simple_heuristic`). Look for
   recurring failure patterns: bad target selection, premature tera, switch
   into bad matchup, etc. Converts "unknown failure" into "categorized
   failure" — see Q2 in [[project-immediate-goal]] context for the action
   table.
4. **Gradient-dominance ratio.** `vf_coef × value_loss / |policy_loss|` from
   wandb. If still ≥5× after many updates, value head is hijacking the
   shared trunk — fix is gradient routing or per-head architecture (see
   `2026-05-15-10-49-per-run-dir-and-firm-field-41-diagnosis.md`).
5. **Curriculum reweighting (last resort).** If the above identify a slice
   the model is undersampling, bump that opponent's weight. Hold self_play
   high (≥0.30) so generalization pressure stays on — see "Why not just
   train on the baselines" below.
6. **Action entropy.** If <0.5 nats, policy has collapsed to a narrow
   strategy; raise `ent_coef_end` or extend `temperature_anneal_steps`.

## Why not just train directly against the baselines

Tempting shortcut: bump `simple_heuristic_baseline` and `max_damage` to 0.4+
and let SGD do the rest. Reject this in the general case because:

- **Goodhart's law.** The baselines are a measure of generalization, not the
  thing we want. Training directly against them produces a model that beats
  *those four specific opponents* — not a strong VGC player.
- **Self-play is the generalization engine.** A model that reaches 60%×4
  *through* self-play has learned VGC; one that gets there by direct
  training has learned four specific exploits. The former is what we want
  for Stage III and beyond.
- **The baselines are diagnostics, not targets.** simple_heuristic being
  stuck is *information about what's missing from training*, not a signal
  to train on simple_heuristic harder. The action is usually "fix what's
  missing" (representation, value calibration, opponent diversity in
  self-play, ghost rotation rate), not "boost the weight."

A small allocation to each baseline in the curriculum (current 0.10 to
simple_heuristic, 0.10 to max_damage, 0.25 to vgc_bench) is fine as anchor
against pathological self-play drift. The line we don't want to cross is
making the baselines a majority of training mass.

## Planned next steps

1. Hold hopeful-wood-69 through update ~1500 to see whether bc_player
   stabilizes or keeps drifting from its update-400 peak.
2. Once it's clear hopeful-wood-69 won't meet criterion, run the BC sanity
   check (diagnostic probe 1) — that's the highest-ROI single experiment.
3. Behavioral mining vs simple_heuristic — convert the flat-line into a
   named failure pattern.
4. Based on probes 1 + 3, decide whether the next run is:
   - A curriculum tweak (small, safe), or
   - An architecture change (gradient routing or per-head trunks per
     `2026-05-15-10-49-per-run-dir-and-firm-field-41-diagnosis.md`), or
   - A representation fix (if behavioral mining shows the model is missing
     observable state).

## Updates

### 2026-05-31 — criterion revised (supersedes the 60%×4 bar above)

The graduation bar is no longer "≥60% against all four simultaneously." The
current, authoritative criterion is:

- **≥80%** vs `simple_heuristic`, `max_damage`, and `bc_player`
- **≥60%** vs `vgc_bench`
- **≥45%** vs `foul_play`

…all simultaneously. Everything above this section describing a uniform 60%
threshold is retained for historical context but is **superseded** by these
numbers.

Two notable shifts in intent:

- **The heuristic bar moved up to 80%.** A genuinely strong agent should
  *dominate* deterministic, exploitable opponents (max_damage, SHP), not merely
  edge them. As of rosy-armadillo-80 (may31.yaml) the agent sits at ~20–27% on
  exactly these two — the same pattern in every recent run (may22/25/26) — so
  this bucket is the binding constraint and the central open problem (narrow
  self-play population + reference/entropy regularization suppressing
  exploitation).
- **FoulPlay is now a requirement (≥45%), not just an informational signal.**
  It is non-functional today, so that bucket is blocked until the opponent is
  repaired.

Baseline opponent identifiers were also standardized to **bare names** (no
`_baseline` suffix): `random`, `max_base_power`, `simple_heuristic`,
`vgc_bench`, alongside `max_damage` and `foul_play`. This matches the eval
parser's canonical names and fixed a silent eval crash (`simple_heuristic_baseline`
was not a resolvable eval spec).

_(none yet)_
