# Deep value head ablation — sep_arch.yaml

## Context

The 2026-05-11 single_team run at update 657 showed value loss dominating
total loss 3:1 through the shared transformer backbone:
`policy=0.91, vf_coef × value = 1.0 × 2.87 = 2.87, rnad ≈ 0.02`.
single_team.yaml subsequently dropped `vf_coef` from 1.0 to 0.5 as the
cheapest rebalance, but the value loss signal still flows through the
*same* shared trunk and continues to shape representation alongside
policy gradients. The question this experiment addresses: does giving
the value head its own integrative depth (deep value head pattern, à la
AlphaStar / MuZero-style prediction nets) free policy learning by
removing value's pressure on shared representation?

## Before state

`FlexibleThreeHeadedModel` and `TransformerThreeHeadedModel` both
hardcoded the value head as a 2-layer MLP:
```
win_head = Sequential(Linear(output_size, 128), LayerNorm, ReLU, Dropout, Linear(128, num_value_bins))
```
There was no `value_head_layers` config knob. `late_ff_stack` output
was shared between policy and value, and the only value-specific
computation was this thin 2-layer MLP.

`build_model_from_config` always called `load_state_dict` with strict=True,
and `train.py`'s `initialize_path` branch built the model from the
*checkpoint's* config rather than the user's runtime config — meaning
architecture knobs in the user's YAML were ignored when initializing
weights from a BC checkpoint.

## Problem

Three coupled issues blocked a clean ablation of "deep value head,
shallow trunk":
1. No way to deepen the value head from config — required model code change.
2. No way to BC-initialize a *modified* architecture — `initialize_path`
   used the checkpoint's architecture verbatim and load_state_dict was
   always strict.
3. New value-head parameters would silently land in the backbone
   parameter group rather than the heads group.

## Solution

Five-part change, kept narrow:

1. **`model_archs.py`** — Added `value_head_layers: Optional[list]` to both
   `TransformerThreeHeadedModel` and `FlexibleThreeHeadedModel`. When non-empty,
   builds a `value_ff_stack` of `ResidualBlock`s between `late_ff_stack`
   output and the final value linear. When empty (legacy), `value_ff_stack`
   is `Identity` and `win_head` keeps the old 2-layer MLP shape — existing
   checkpoints continue to strict-load.

2. **`config.py`** — Added `value_head_layers: List[int]` (default `[]`)
   to `ArchitectureConfig`.

3. **`learners.py`** — Threaded `value_head_layers` through
   `build_model_from_config` common_kwargs and `MODEL_ARCH_CONFIG_KEYS`.
   Added `strict: bool = True` to the loader and a partial-load logger
   that prints missing/unexpected keys when `strict=False`. Added
   `"value_ff_stack"` to `head_keywords` in `_build_optimizer` so new
   modules join the heads LR group.

4. **`train.py`** — Changed the `initialize_path` branch to pass the
   user's `cfg` (not `checkpoint_cfg`) and `strict=False`. This is a
   semantic change to `initialize_path`: it now means "use my config,
   load BC weights where shapes match." Per CLAUDE.md's
   no-backcompat rule, flagged but not gated.

5. **`sep_arch.yaml`** — New config modeled on single_team.yaml. Only
   delta: `value_head_layers: [512, 512, 256]` and
   `initialize_path: data/models/supervised/cool-bee-85-finetune_best.pt`
   (BC partial-load) with `resume_from: null`. Trunk depth, vf_coef,
   topology, curriculum mix all held constant against single_team.yaml.

## Reasoning

**Why not also shallow the trunk.** The user briefly considered combining
"shallow trunk + deep value head" but settled on holding trunk depth
constant. This lets any observed effect be attributed to head separation
alone, not a confounded trunk-depth change. It also preserves clean BC
init for the trunk (all transformer layer weights load strictly).

**Why no extra self-attention layers in the value head.** Initially
considered but ruled out: the value head's direct input is `critic_out`,
the broadcast `[CRITIC]` decision token output. Every position is
identical, so self-attention has nothing to attend across. Adding
attention would have required either (a) switching the value head's
input from `critic_out` to `turn_out` (per-position features) — a
semantic change to the `[CRITIC]` token mechanism — or (b) a complex
cross-attention design. Neither was worth the scope for a first
ablation. The ResidualBlock stack alone tests whether *capacity* is
the bottleneck.

**Why partial BC load instead of fresh init.** Throwing away the BC
trunk and policy head wastes weeks of supervised pretraining. The
only piece whose shape changed is the value head, and the existing
value head was barely-better-than-uniform anyway (cross-entropy 2.87
vs ceiling log(51) ≈ 3.93), so fresh-initializing it costs little.

**Why head_keywords contains `value_ff_stack`.** Without this, the
new value-specific Residual blocks would land in the backbone LR
group (`backbone_lr=1e-5`) and learn 3× slower than `heads_lr=3e-5`.
The whole point of a deep value head is to give value its own faster
learning dynamics — putting its params in the heads group enforces this.

## Outcome

Smoke test confirms:
- Legacy `single_team.yaml` strict-loads from `cool-bee-85-finetune` (unchanged behavior).
- `sep_arch.yaml` partial-loads: 18 missing keys (the fresh
  `value_ff_stack.*` + new 1-layer `win_head`), 6 unexpected keys
  dropped (old `win_head.0.*`, `win_head.1.*`, `win_head.4.*`).
- Forward pass shapes correct: `turn=(B,S,2025) tp=(B,S,90) win_vals=(B,S) win_dist=(B,S,51)`.
- Total params: ~26.28M (vs 25.55M baseline) — `value_ff_stack` adds ~791K params.
- Optimizer routing: all 16 `value_ff_stack` tensors placed in the heads group.
- ruff check + pyright + full RL test suite (211 tests) + supervised tests (53 tests) all pass.

## What to watch when running

Side-by-side comparison against single_team.yaml at matching `vf_coef=0.5`,
same topology and curriculum. Expected signal:

- **If sep_arch wins:** policy loss starts moving sooner and reaches
  a lower equilibrium than single_team baseline; value loss converges
  at similar rate to baseline. Interpretation: the deep value head
  did useful decoupling — value capacity was a real bottleneck.
- **If sep_arch ties or loses:** the gradient dominance problem isn't
  about value *capacity*, it's about gradient *magnitude*. Next
  experiment should be stop-gradient from the value head into the
  shared trunk (zero LOC change in config, near-zero in code).
- **Specific things to log:** per-update policy_loss, value_loss
  (raw, pre-vf_coef), gradient norm into the trunk (`transformer.*`
  vs `value_ff_stack.*` separately if possible).

## Planned next steps

1. Launch sep_arch.yaml side-by-side with current single_team.yaml run
   (or A/B by alternating runs). Use a meaningful enough sample window —
   the policy/value loss split takes ~500-1000 updates to stabilize after
   the BC init.
2. If sep_arch shows policy unlock, consider whether to keep deep value
   head as the default for future architectures or graduate the pattern
   into a re-pretrained supervised checkpoint.
3. If sep_arch shows no signal, write up stop-gradient experiment as
   the next branch.

## Files touched

- `src/elitefurretai/supervised/model_archs.py` — new `value_head_layers`
  parameter + `value_ff_stack` module in both Transformer and LSTM models.
- `src/elitefurretai/rl/config.py` — `value_head_layers` field added to
  `ArchitectureConfig`.
- `src/elitefurretai/rl/learners.py` — common_kwargs threading, strict
  flag on `build_model_from_config`, `value_ff_stack` in head_keywords,
  `value_head_layers` in `MODEL_ARCH_CONFIG_KEYS`.
- `src/elitefurretai/rl/train.py` — `initialize_path` branch now uses
  runtime cfg + strict=False (semantic change flagged here).
- `src/elitefurretai/rl/configs/sep_arch.yaml` — new config.

## Updates

(none yet — will be appended after first launch and any post-run findings.)
