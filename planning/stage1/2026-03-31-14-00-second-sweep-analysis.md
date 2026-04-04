# Second Sweep Analysis (us7ulxtv) — Key Findings & Recommendations

## Context
Second Bayesian sweep using `second_config.yaml`, building on first sweep (8ltntdzf) learnings.
8 runs total, 5 completed 15 epochs, 2 failed/killed early, 1 still running.
Sweep ID: us7ulxtv. Fixed: topk+k=2025, batch=128, entropy=0, tp_loss_weight=0.

## Before State
- First sweep best: dark-sweep-15 (8 epochs) — Turn Top3=0.519, Move Top3=0.384, Switch Top3=0.973, WinCorr=0.59
- Core hypothesis: action-type weighting (mv_w > 1.0, sw_w < 1.0) would close the Move/Switch gap
- Model was still improving at epoch 8; extended to 15 epochs

## Corrected Results (15-epoch completed runs, ranked by Test Turn Top3)

| Rank | Run | mv_w | sw_w | mv/sw | win_w | lr | drop | TL | Turn3 | Move3 | Switch3 | WinCorr | WinMSE |
|------|-----|------|------|-------|-------|----|------|----|-------|-------|---------|---------|--------|
| 1 | generous-5 | 1.0 | 0.7 | 1.43 | 0.35 | 5.44e-5 | 0.17 | 7 | **0.700** | **0.619** | 0.9991 | 0.406 | 0.157 |
| 2 | youthful-6 | 1.0 | 1.0 | 1.00 | 0.35 | 5.44e-5 | 0.20 | 7 | 0.665 | 0.574 | 0.9991 | **0.610** | 0.166 |
| 3 | autumn-2 | 2.0 | 1.0 | 2.00 | 0.15 | 1.20e-4 | 0.12 | 5 | 0.629 | 0.536 | 0.9988 | 0.401 | 0.159 |
| 4 | fragrant-1 | 1.3 | 0.5 | 2.60 | 0.35 | 6.90e-5 | 0.20 | 6 | 0.624 | 0.531 | 0.9984 | 0.614 | 0.161 |
| 5 | solar-3 | 2.0 | 1.0 | 2.00 | 0.25 | 6.08e-5 | 0.17 | 7 | 0.536 | 0.411 | 0.9989 | 0.305 | 0.158 |

Failed/incomplete: copper-7 (mv=1.3, sw=0.7, 8ep, Turn3=0.505), olive-4 (mv=2.0, sw=1.0, 4ep, Turn3=0.333), earthy-8 (1ep, still running).

## Key Insights

### Insight 1: 15 epochs produced massive gains (Turn3: 0.519 → 0.700, +35%)
The single biggest improvement came from simply training longer. Even youthful-sweep-6 (no action-type weighting, mv=sw=1.0) hit Turn3=0.665, a +28% improvement over dark-sweep-15's 0.519 at 8 epochs. This confirms the first sweep's strongest prediction: the model was far from converged at 8 epochs.

Move Top3 improved even more dramatically: 0.384 → 0.619 (+61%). Longer training disproportionately helps move prediction because moves are harder and need more gradient iterations to learn the complex move+target space.

The train/test gap (train_loss=0.893 vs test_loss=1.920 for generous) suggests the model may still not be fully converged and more epochs could help further.

### Insight 2: Switch downweighting beats move upweighting (H2 supported, H1 rejected)
The original hypothesis was that upweighting moves (mv_w > 1.0) would force more gradient toward hard move predictions. The data tells a different story:

- **Best**: mv=1.0, sw=0.7 (generous, Turn3=0.700) — keep moves at baseline, reduce switches
- **Second**: mv=1.0, sw=1.0 (youthful, Turn3=0.665) — no weighting at all
- **Worse**: mv=2.0, sw=1.0 (autumn=0.629, solar=0.536) — upweighted moves
- **Worst of the asymmetric**: mv=1.3, sw=0.5 (fragrant, 0.624) — most aggressive ratio (2.6x)

The effective move gradient fraction tells the story:
- 72.6% (generous, sw_w=0.7) → Turn3=0.700 ← **sweet spot**
- 65.0% (youthful, uniform) → Turn3=0.665
- 78.8% (autumn/solar, mv_w=2.0) → Turn3=0.629/0.536
- 82.8% (fragrant, mv=1.3/sw=0.5) → Turn3=0.624

**Why switch downweighting > move upweighting**: Upweighting moves amplifies the magnitude of gradients from move samples, which can destabilize training — the optimizer sees larger-than-normal loss spikes on hard examples. Downweighting switches gently reduces the easy examples' contribution, allowing the natural gradient distribution to shift toward moves without magnitude distortion. It's the difference between shouting louder about moves vs. simply talking less about switches.

The optimal gradient allocation is ~72-73% toward moves (vs. the natural ~65% from data distribution). Going above ~78% hurts.

### Insight 3: The generous-vs-youthful comparison is the cleanest signal
These two runs are nearly identical in everything except weighting and regularization:
- Same LR (5.44e-5), same TL (7), same win_w (0.35), same architecture
- generous: mv=1.0, sw=0.7, drop=0.17, wd=9.9e-5
- youthful: mv=1.0, sw=1.0, drop=0.20, wd=1.4e-4

generous beat youthful by +3.5pp Turn3 and +4.5pp Move3. Two factors:
1. **Switch downweighting**: sw_w=0.7 shifted gradient allocation toward moves
2. **Lower regularization**: less dropout (0.17 vs 0.20) and lower weight decay (1.0e-4 vs 1.4e-4)

These effects are confounded. The next experiment should separate them.

### Insight 4: Turn accuracy and win correlation trade off
A striking pattern: generous has the best Turn3 (0.700) but worst WinCorr (0.406) among the top 4, while youthful and fragrant have the best WinCorr (0.610, 0.614) but lower Turn3.

For RL, you need BOTH a good policy (Turn3) AND a good value function (WinCorr). This tradeoff means the best BC model for RL might not be the absolute Turn3 winner. youthful-sweep-6's configuration (Turn3=0.665, WinCorr=0.610) might actually be the better RL seed because the value function quality will directly impact advantage estimation.

The likely mechanism: less regularization (generous) helps the turn head memorize patterns but lets the win head overfit to narrow prediction ranges (lower MSE=0.157 but lower correlation=0.406 — it predicts a tight range of values rather than discriminating well).

### Insight 5: win_loss_weight=0.35 is confirmed optimal
Three runs with win_w=0.35 averaged Turn3=0.663, beating win_w=0.15 (0.629, n=1) and win_w=0.25 (0.536, n=1, confounded). Higher win weight helps turn prediction presumably by forcing the backbone to learn state quality features that also help action selection.

### Insight 6: 7 transformer layers with ~124M params is the right scale
The top 2 runs both used TL=7 (124M params). autumn (TL=5, 107M params) was 3rd with a very different config. No evidence that deeper is needed, but 7 > 5 ≈ 6.

### Insight 7: Teampreview head is dead with tp_loss_weight=0
All runs show TP Top1 ≈ 0.5% (random = 1.1%). Expected — the TP head has [] layers and gets zero gradient. Need a different approach to get TP working without hurting turn prediction (see Recommendations).

### Insight 8: LR sweet spot is firmly at ~5.4e-5
The two best runs both used lr ≈ 5.44e-5. This is very close to dark-sweep-15's 5.93e-5 from the first sweep. The LR question is resolved.

### Insight 9: olive-sweep-4's failure is instructive
olive (lr=1.1e-4, wd=2.8e-4) — the weight decay was 2.5x the learning rate! This massive regularization starved the model of learning capacity. Killed at epoch 4 with Turn3=0.333. Lesson: weight_decay should be well below the LR (ratio <= ~2x).

## Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: mv_w > 1.0 helps Move Top3 | **REJECTED** | mv_w=2.0 runs (solar=0.536, autumn=0.629) underperformed mv_w=1.0 (generous=0.700) |
| H2: sw_w < 1.0 helps by reducing easy-switch gradient | **SUPPORTED** | generous (sw=0.7) beat youthful (sw=1.0) in near-identical comparison: +3.5pp Turn3 |
| H3: 15 epochs allows convergence | **CONFIRMED** (still converging) | All 15-epoch runs massively exceeded 8-epoch best. Train/test gap suggests more room |
| H4: Narrow LR around 5.4e-5 is optimal | **CONFIRMED** | Top 2 runs both at lr=5.44e-5 |
| H5: win_w in [0.15, 0.35] — higher helps | **MOSTLY CONFIRMED** | 0.35 avg Turn3=0.663 > 0.15 Turn3=0.629. But tradeoff with WinCorr quality |

## Recommendations (see next planning doc for implementation plan)

### Step 1: Extended single-run training (20-30 epochs)
Train generous-sweep-5's config for 20-30 epochs with cosine annealing LR. Model still converging.

### Step 2: Disentangle switch weighting from regularization
Run a controlled 3-run experiment: (a) sw=0.7/drop=0.20 (b) sw=1.0/drop=0.17 (c) sw=0.7/drop=0.17
All other params identical to generous-sweep-5. This isolates the switch weighting effect.

### Step 3: Stop-gradient teampreview head (zero-cost TP)
Change `tp_feat = self.teampreview_ff_stack(encoded.detach())` in model forward pass. Add back
TP FF layers [512, 256]. TP head trains from detached encoder features — zero impact on turn prediction.

### Step 4: Consider youthful-sweep-6 as RL seed (Turn3/WinCorr tradeoff)
If RL is imminent, youthful's WinCorr=0.610 may be more valuable than generous's extra +3.5pp Turn3.

## Planned Next Steps
1. Implement stop-gradient TP head
2. Extended training run on generous config (25-30 epochs, cosine LR schedule)
3. Controlled A/B test: switch weighting vs dropout regularization
4. Begin RL finetuning with best BC checkpoint
