# Value-trunk gradient scaling + mean-KL portfolio anchor

**Date**: 2026-05-16 22:30
**Status**: Implemented; new run pending kickoff
**Triggers**: hopeful-wood-69 win-rate decline (see
`2026-05-16-21-30-stage2-graduation-criteria.md`), gradient-dominance
diagnostic confirming the firm-field-41 failure mode recurring.

## Context

hopeful-wood-69 reproduced the firm-field-41 pattern: value_loss and
loss steadily improve while win rates against all four graduation
baselines decline together. The 2026-05-15 firm-field-41 diagnosis
attributed this to two stacked pathologies:

- Value-gradient dominance through the shared trunk
  (`vf_coef × value_loss / |policy_loss|` ratio ≥ 4×).
- R-NaD regularization functionally off (α × rnad_loss ≪ 1% of total).

The recommended fix bundled (a) a 6× `rnad_alpha` bump (0.05 → 0.3) with
(b) a head-deepening (`turn_head_layers` and `value_head_layers` extended
to `[768, ...]`). `may15.yaml` shipped (a) but reverted (b) for BC partial-
load compatibility, leaving the structural fix incomplete. hopeful-wood-69
ran on this partial fix.

## Before State

### Diagnostic measurements (hopeful-wood-69, 959 updates)

Gradient-dominance ratio computed from
`wandb/run-20260516_182240-ocusmofu/files/output.log`:

| Window | \|policy_loss\| | value_loss | vf · val / \|pol\| | value share | rnad share |
|---|---|---|---|---|---|
| 1–137 | 0.350 | 3.671 | **5.25×** | 76% | 8.6% |
| 138–274 | 0.075 | 3.154 | **20.89×** | 95% | 2.5% |
| 275–411 | 0.144 | 3.026 | **10.47×** | 111% | 1.6% |
| 412–548 | 0.194 | 2.828 | **7.29×** | 115% | 1.6% |
| 549–685 | 0.212 | 2.594 | **6.13×** | 120% | 1.6% |
| 686–822 | 0.082 | 2.194 | **13.33×** | 108% | 1.6% |
| 823–959 | 0.062 | 1.776 | **14.42×** | 97% | 2.3% |

Ratio ≥ 5× throughout, ≥ 10× for five of seven windows. The May-15
diagnosis playbook flagged ≥ 4× as "value head hijacking the shared
trunk." This is well above that threshold from update 1, not after slow
drift.

### Failure signature (from W&B panels, run name `hopeful-wood-69`)

- `value_loss` step-decreases at ~50-update intervals (matches
  `portfolio_add_interval=50`).
- `policy_loss` regressing from ~-0.2 back toward 0 (advantage signal
  flattening).
- `entropy` monotonically rising from 3.2 → 5.0+ (policy uniformizing).
- Win rates against all fixed opponents declining in lockstep:
  `bc_player` peaked 0.60 → 0.25; `max_damage` 0.65 → 0.27;
  `vgc_bench` 0.40 → 0.32; `simple_heuristic` 0.20 → 0.10.
- Self-play pinned at 0.50 (mirror-match by construction; meaningless).
- `learner_steps_per_second_recent` declining from 150 → 100 (longer
  battles as policy randomizes).

### Why `rnad_loss` being low isn't a fix

The portfolio uses `min KL` across `max_portfolio_size=10` references
spaced 50 updates apart, with `portfolio_update_strategy=recent`. After
~update 500 the portfolio fills with recent-self-snapshots and BC gets
evicted. `min KL to recent self` is small by construction regardless of
absolute drift direction — the regularizer becomes structurally unable
to catch slow degradation. Until update 500 BC was anchoring; after
500, the loss reading became approximately tautological.

## Problem

The trunk's parameter updates from the value-head gradient path are
4–20× larger than from the policy-head path. Over hundreds of updates
this shapes the trunk's representations to minimize observed-return CE
on a self-play distribution that is itself degrading against fixed
opponents. The policy head can't recover useful advantage signal from
the resulting features, entropy rises, and win rates fall.

The R-NaD term — which is supposed to be the brake on this drift — can't
fire because the min-KL aggregation only requires staying close to *any*
recent self.

## Solution

Two surgical changes, no architecture rewrite, no BC re-train.

### Change 1: Gradient scaling on the value→trunk path

- Add `_ValueTrunkGradScale` (forward-identity, backward × `scale`) in
  `src/elitefurretai/supervised/model_archs.py`.
- Insert one call between `late_ff_stack(critic_out)` and
  `value_ff_stack(out_critic)` in both `forward` and `forward_with_hidden`.
- Add `value_to_trunk_grad_scale: float` kwarg on
  `TransformerThreeHeadedModel.__init__`, default `1.0`.
- Add `architecture.value_to_trunk_grad_scale: float = 1.0` to
  `ArchitectureConfig`. Lives in architecture (not algorithm) because it
  is a model-construction attribute set on `self` (parallels
  `use_decision_tokens`), not a loss coefficient — and because
  `build_model_from_config` already reads from the flat-arch dict, so
  this matches the existing plumbing pattern. No new MODEL_ARCH_CONFIG_KEYS
  entry — different scale values are checkpoint-compatible because
  weight shapes are unchanged.
- Added `value_to_trunk_grad_scale` (default 1.0) to all existing YAMLs
  (`easy_test.yaml`, `may15.yaml`, `sep_arch.yaml`, `single_team.yaml`)
  for visibility; `may16.yaml` overrides to 0.15.

Effect on backward (scale `s`):
- `win_head`, `value_ff_stack` parameters: full gradient (downstream of the
  scale op).
- `late_ff_stack` parameters via the value-side forward call: scaled by `s`.
- Trunk parameters (everything upstream of `critic_out`): scaled by `s` on
  the value path; policy path is untouched.

The policy path is structurally unaffected. The value head still learns
at full speed within its own subtree; only its share of the *shared*
representation budget shrinks.

### Change 2: Mean-KL portfolio anchor

- `PortfolioRNaDLearner._compute_portfolio_kl` now returns
  `torch.stack(kls).mean()` instead of `min(kls)`.
- The diagnostic `portfolio_selection_counts` still tracks the closest
  reference for visibility, but does not gate the loss.
- Old-portfolio anchors (e.g. BC while it remains in the portfolio) now
  contribute their drift cost on every update, not just when they happen
  to be the closest reference.

### Initial values

| Knob | Value | Reasoning |
|---|---|---|
| `value_to_trunk_grad_scale` | 0.15 | Middle of the 0.1–0.2 range. Targets ratio ~1.5× given the run-1 starting condition. Tune downward to 0.1 if ratio rebounds > 2×; upward to 0.2 if value learning stalls. |
| `rnad_alpha` | 0.3 (unchanged from may15) | Mean-KL across diverse refs raises the *value* of rnad_loss, so α × rnad_loss climbs without re-bumping α. |
| Portfolio shape | unchanged | `max_portfolio_size=10`, `add_interval=50`, `strategy=recent`. |

## Reasoning

### Why grad-scale over the alternatives

- **Cutting `vf_coef`** would slow value learning system-wide. Value head
  still has headroom (CE at ~1.8 over 51 bins is well above floor), so we
  don't want global value-learning damping — we want the value path to
  stop dictating shared-representation shape.
- **Per-head transformers** (the May-15 alternative) would decouple the
  paths fully but invalidates BC partial-load and roughly doubles
  transformer parameter count. Reserved as escalation if 0.15 is too high
  and 0.1 is too low.
- **Head-deepening** (the original May-15 fix) was reverted in may15.yaml
  for BC compat reasons. Re-attempting it would require a fresh trunk
  with no BC initialization — much larger blast radius than a 10-LOC
  backward-only hook.

### Why mean-KL over BC-pinning

- The user picked mean-KL specifically over BC-pinning as the
  conservative first move. Mean-KL alone has limited effect once the
  portfolio fills with recent-selves (~update 500), but it raises the
  cost of *all* simultaneous drift in early training while BC is still
  in the portfolio. If that's not enough, BC pinning is the natural
  follow-up — a permanent reference slot — and doesn't conflict with the
  current change.
- Selection-counter semantics preserved (closest ref still tracked) so
  W&B dashboards continue to render meaningfully.

### Why both, not one at a time

Either change alone is plausibly insufficient. The grad-scale addresses
the dominant mechanism (representational hijacking); mean-KL addresses
the second-order failure of the regularizer to catch the resulting drift.
Running them together costs one experiment instead of two; if win rates
recover we won't know which mattered more, but that's acceptable — the
goal is graduation, not ablation.

## Implementation summary

Files changed:
- `src/elitefurretai/supervised/model_archs.py`: add
  `_ValueTrunkGradScale` autograd Function + `scale_value_trunk_gradient`
  wrapper; insert in both `forward` paths; add `value_to_trunk_grad_scale`
  kwarg to `TransformerThreeHeadedModel`.
- `src/elitefurretai/rl/config.py`: add
  `AlgorithmConfig.value_to_trunk_grad_scale: float = 1.0`.
- `src/elitefurretai/rl/learners.py`: pass new kwarg through
  `build_model_from_config`; switch `_compute_portfolio_kl` from
  min-KL to mean-KL.
- `src/elitefurretai/rl/configs/may16.yaml`: new config — fork of
  may15.yaml with `value_to_trunk_grad_scale: 0.15`.
- `unit_tests/rl/test_learner.py`: 7 new tests covering forward-identity,
  trunk-grad scaling linearity, value-head invariance, mean-KL semantics,
  selection-counter preservation, empty-portfolio handling.

Quality gates: `ruff check`, `ruff format`, `pyright` clean on changed
files; 38 learner tests pass (+7 new); 85 tests pass across
`test_learner`, `test_model_archs`, `test_agent_learner`, `test_smoke`,
`test_config`, `test_players` with no regressions.

## Planned Next Steps

1. **Kick off may16.yaml run** with BC initialization from
  `cool-bee-85-finetune_best.pt`. Expect a fresh wandb run name (no
  resume). Per-run-dir behavior writes checkpoints to
  `data/models/rl/<new-run-name>/`.
2. **Within the first ~50 updates**, sanity-check on W&B:
  - Gradient-dominance ratio should drop from the hopeful-wood-69 5–20×
    range into ~1–2×.
  - `value_loss` may rise initially (smaller share of trunk capacity), then
    settle.
  - `rnad_loss` should be visibly larger than the 0.05–0.15 hopeful-wood-69
    range because mean-KL aggregates across diverse refs.
3. **At update ~200**, decide:
   - If ratio ~1–2× and win rates flat/up vs fixed opponents → hold the
     run, let it ride to update ~1500.
   - If ratio still ≥ 3× → lower `value_to_trunk_grad_scale` to 0.10 and
     restart.
   - If win rates *crash* immediately → value head may be starved of trunk
     gradient; raise scale to 0.25 and restart.
4. **At update ~500**, evaluate: does the entropy curve flatten or keep
   climbing? Climbing entropy past update 500 means mean-KL alone wasn't
   enough — next escalation is pinning BC as a permanent reference (or
   switching to a fixed-BC `rnad_alpha` boost just for that reference).

## Risks

- **Value head undertraining.** If grad-scale starves the trunk of
  value signal, value_loss plateaus high and advantage estimates stay
  noisy. Mitigation: monitor value_loss trajectory — should still
  decrease, just slower than hopeful-wood-69's aggressive staircase.
- **Mean-KL pulling policy toward stale references.** While BC is in
  the portfolio (first ~500 updates), mean-KL anchors against BC's
  policy distribution. If BC is meaningfully worse than the current
  policy in some regime, mean-KL will pull the policy back toward BC's
  style. Acceptable risk — that's the point of an anchor — but watch
  bc_player win rate to make sure we're not regressing.
- **Numerical interaction with PPO clip.** PPO importance-weight clip
  ε=0.2 caps per-step policy update size. With the trunk now receiving
  less value-gradient pressure, the policy gradient may push harder
  through PPO clipping. Should be self-regulating but worth watching
  `clip_fraction` metric if logged.

## Updates

_None yet — run not launched._
