# Second Sweep: YAML Config Refactor + Action-Type Weighting

## Context
After analyzing the first 40-run sweep (ID: 8ltntdzf), we identified key findings
and implemented infrastructure improvements + a second sweep to test action-type
loss weighting.

## Before State
- `train_sweep.py` had all sweep configuration hardcoded as Python dicts (~300 lines)
- No mechanism to weight move vs switch actions differently in the loss
- First sweep showed MOVE:SWITCH Top3 ratio of 0.39 — switches 2.5x easier to predict

## Problem
1. Rigid sweep config: adding new parameters (like move/switch weights) required
   editing 300+ lines of Python code per sweep
2. Switch-bias: the model defaults to easy switch predictions, starving move+target
   gradient signal
3. Two pre-existing bugs discovered during deployment:
   - Teampreview cross_entropy crash: data has TP steps with turn-space encoded
     actions (0-2024) but TP head only outputs 90 logits
   - No-grad backward: rare batches with no valid samples produce a loss tensor
     without gradient, crashing backward()

## Solution

### 1. YAML Config Refactor (`train_sweep.py`)
- Extracted all config into YAML files under `supervised/sweep_configs/`
- Added `--config` CLI argument (required)
- `load_sweep_config(yaml_path)` loads sweep/variants/fixed configs from YAML
- Created `first_config.yaml` (original sweep) and `second_config.yaml` (new)

### 2. Move/Switch Action-Type Weighting (`train.py`)
- Added `move_loss_weight` and `switch_loss_weight` config params (default: 1.0)
- Per-sample weights computed from action encoding:
  - slot1 = action // 45, slot2 = action % 45
  - Slots 40-43 are switches; if either slot switches, apply switch_loss_weight
  - Otherwise apply move_loss_weight
- Weights passed to `topk_cross_entropy_loss` and `focal_topk_cross_entropy_loss`

### 3. Bug Fixes
- **TP range guard**: Filter out teampreview actions >= 90 before calling cross_entropy
- **No-grad skip**: Skip backward() when `loss.requires_grad` is False (empty batch)

### 4. Second Sweep Config (second_config.yaml)
Fixed from first sweep: topk+k=2025, batch_size=128, entropy_weight=0,
teampreview_loss_weight=0. New parameters: move_loss_weight [1.0-2.0],
switch_loss_weight [0.5-1.0]. Extended to 15 epochs. Narrow LR/dropout around
dark-sweep-15's sweet spot.

## Reasoning
- YAML configs enable rapid iteration on sweep parameters without code changes
- Move/switch weighting directly addresses the core quality gap (MOVE Top3 = 38% vs
  SWITCH Top3 = 97%) — the biggest bottleneck for VGC bot performance
- Bug fixes ensure reliable overnight sweeps without manual intervention
- 15 epochs allow convergence (dark-sweep-15 was still improving at epoch 8)

## Sweep Details
- Sweep ID: xiup5jp7
- W&B Project: elitefurretai-hydreigon
- 15 runs, 15 epochs each (~19 hours total)
- Epoch 1 metrics (first run): Turn Top3=0.402, MOVE Top3=0.300, SWITCH Top3=0.997

## Planned Next Steps
1. Analyze second sweep results — compare MOVE Top3 across move_loss_weight values
2. If weighting helps, consider finer-grained weighting (move-only vs move+switch)
3. Extended single-run training on best config (30+ epochs) to find convergence
4. Investigate data quality: fix teampreview action encoding in preprocessing
5. Begin RL finetuning with best supervised checkpoint
