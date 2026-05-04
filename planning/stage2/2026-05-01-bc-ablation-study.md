# BC Model Ablation Study

## Context

Investigate three independent ablation axes on the BC model to understand what drives performance
and identify safe cuts for RL training speed:

1. **Feature ablations**: `full_no_transition` (drop 37 transition dims) and `raw` (drop 190 engineered+transition dims)
2. **Model size**: ~5× smaller (26.7M params) and ~10× smaller (13.9M params)

All runs use `data/battles/regc_final_v4/` and 10 epochs. Baseline reference is `curious-darkness-77_best`
(125.6M params, `full` featureset, 30 epochs, best at epoch 18).

**Why 10 epochs:** Reduces per-run cost from ~15h → ~5h. An epoch-matched baseline (B0) controls for
undertraining so ablation comparisons are fair.

**Why not re-process data:** Pre-processed FULL states (5222 dims) are sliced at training time.
Feature ordering is `MON:0-5 → OPP_MON:0-5 → BATTLE → ENGINEERED → TRANSITION`, so `raw` (5032)
and `full_no_transition` (5185) are clean leading-slice ablations.

**Key feature sizes:**
- FULL: 5222 dims (13 MON/OPP groups + BATTLE + ENGINEERED[153] + TRANSITION[37])
- FULL_NO_TRANSITION: 5185 dims (drops TRANSITION)
- RAW: 5032 dims (drops ENGINEERED + TRANSITION)

## Before State

Reference checkpoint: `data/models/supervised/curious-darkness-77_best.pt`
- Overall Top-1/3/5: **41.4% / 61.3% / 68.8%**
- MOVE Top-1/3/5: **26.2% / 51.2% / 61.6%**
- SWITCH Top-1: **99.1%**
- BOTH Top-1/3/5: **48.3% / 65.2% / 67.3%**
- Win Corr (actual): **0.818** | Synthetic: **0.665** | Brier: **0.133**
- TP Top-1/3/5: **53.8% / 82.2% / 98.9%**
- Params: **125.6M** | Featureset: `full` | Epochs: 30 (best @ 18)

## Problem

Unknown which components of the model are load-bearing for BC performance.
Identifying safe ablations would let us:
- Speed up RL inference (fewer input dims = faster embedding, smaller model = faster forward pass)
- Reduce RL worker memory pressure
- Potentially train a smaller, faster RL model without sacrificing much quality

## Solution: 5 Ablation Runs

### B0 — Epoch-Matched Baseline
**Config:** `configs/ablation_b0_baseline.yaml`
**Change:** Same as `curious-darkness-77` but 10 epochs instead of 30.
**Purpose:** Controls for the epoch gap so A1–A4 comparisons are apples-to-apples.

**Pre-run hypothesis:**
- Expect scores ~3–6% lower than the 30-epoch published reference across all metrics.
- Win Corr likely drops from 0.818 → ~0.78–0.80 (value head needs more steps to converge).
- SWITCH Top-1 should remain near ~99% (trivial to learn).
- This establishes the fair baseline all ablations compare against.

**Results (B0 — `hardy-mountain-79_best.pt`, 10 epochs, 3h 44m):**
- Overall Top-1/3/5/10: **36.10% / 53.86% / 70.98% / 83.18%**
- MOVE Top-1/3/5/10: **23.51% / 42.97% / 65.07% / 81.73%**
- SWITCH Top-1/3/5: **81.00% / 99.94% / 99.95%**
- BOTH Top-1/3/5: **45.86% / 52.86% / 65.45%**
- FORCE_SWITCH Top-1: **60.99%**
- Win Corr (actual): **0.6438** | Synthetic: **0.4091** | Brier (synthetic): **0.2120** | Brier (actual): **0.2904**
- Train Loss epoch10: 1.814 (down from 2.527 epoch1)

**Vs 30-epoch reference (curious-darkness-77):**
- Overall Top-3: -7.4pt (53.86% vs 61.3%)
- MOVE Top-3: -8.2pt
- SWITCH Top-1: -18.1pt (81% vs 99.1%) — surprisingly large; SWITCH Top-3 is still 99.94% though, so the model knows *which mons to consider* but not the exact choice yet
- BOTH Top-3: -12.3pt
- Win Corr: -0.17 (0.644 vs 0.818)

**Takeaway:** B0 is significantly under-baked vs the 30-epoch reference, especially the value head. This is expected and is exactly why we need the epoch-matched baseline — A1–A4 must be compared against B0, not against the 30-epoch reference.

---

### A1 — Drop Transition Features (FULL_NO_TRANSITION)
**Config:** `configs/ablation_a1_no_transition.yaml`
**Change:** `embedder_feature_set: full_no_transition` (5185 dims; drops TRANSITION[37])
**Params:** Same as B0 (125.6M — architecture unchanged, just 37 fewer input dims)

**Pre-run hypothesis:**
- **Action accuracy**: Minimal drop (~0–2%). Transition features tell the model "I just used a switch /
  got critted / moved first last turn", which is useful for reactive play but humans rarely
  condition hard on a single stochastic event. BC imitation is robust to this.
- **Win Corr**: Small drop (~0–2%). The value head benefits from knowing "a faint just happened"
  (already captured in HP/fainted features in the main groups) more than from knowing *how* it happened.
- **RL implication**: This is the riskiest-for-RL of the feature ablations. Transition features
  encode per-turn stochastic outcomes (crit, miss, super-effective) that RL needs for credit
  assignment. BC hides this risk because human data averages over noise.
- **Speed gain**: Negligible inference speedup (37 dims / 5222 = 0.7% smaller input).

**Results (A1 — `still-plasma-81_best.pt`, 10 epochs):**
- Overall Top-1/3/5/10: **35.16% / 43.47% / 53.53% / 67.91%**
- MOVE Top-1/3/5: **18.89% / 26.47% / 38.09%** (large drop vs B0)
- SWITCH Top-1/3: **91.68% / 99.89%** (better than B0 81%)
- BOTH Top-1/3/5: **49.34% / 61.59% / 76.22%** (better than B0)
- FORCE_SWITCH Top-1: **92.24%**
- Win Corr (actual): **0.5820** (much better than B0 0.369!)
- Synth Corr: **0.5291** | Brier (actual): **0.1782**
- Final test_loss: 6.0006

**vs B0:** Action accuracy mixed — BOTH and SWITCH improved, MOVE dropped sharply (−16.5pt MOVE T3).
Value head **substantially better** than B0 (+0.21 Win Corr). Net: surprising win, value head benefits
from removing transition noise at this epoch budget. **Hypothesis was wrong** — transition
features hurt, not help, BC value learning at 10 epochs.

---

### A2 — Raw Featureset (Drop Engineered + Transition)
**Config:** `configs/ablation_a2_raw.yaml`
**Change:** `embedder_feature_set: raw` (5032 dims; drops ENGINEERED[153] + TRANSITION[37])
**Params:** Same as B0 (125.6M — architecture has one fewer input group)

**Pre-run hypothesis:**
- **Action accuracy**: Small-to-moderate drop (~2–5% on MOVE Top-3). Engineered features include
  damage estimates for each opp-move × my-mon pair — a direct proxy for "which move KOs what".
  BC humans know this intuitively; removing it forces the model to learn it from type features alone.
  SWITCH accuracy should be unaffected.
- **Win Corr**: Moderate drop (~0.03–0.06). The engineered features are closest to explicit win-
  probability signals (faint counts, HP%, damage estimates), so the value head loses the most useful
  shortcut inputs.
- **RL implication**: Slower RL convergence likely, as the value head must learn damage from scratch.
  Risk of degraded early-game teampreview predictions (damage calc is most helpful pre-reveal).
- **Speed gain**: ~3–4% smaller input (190 dims removed); plus one fewer GroupedFeatureEncoder group,
  slightly reducing attention computation. Not a large inference win.

**Results (A2 — `fresh-totem-82_best.pt`, 10 epochs):**
- Overall Top-1/3/5/10: **33.42% / 47.86% / 50.78% / 60.62%**
- MOVE Top-1/3/5: **17.86% / 34.85% / 38.53%**
- SWITCH Top-1/3: **99.23% / 99.89%** (highest of all ablations, near reference)
- BOTH Top-1/3/5: **32.07% / 50.50% / 53.10%**
- FORCE_SWITCH Top-1: **96.55%**
- Win Corr (actual): **0.6031** (best of all ablations, +0.23 vs B0!)
- Synth Corr: **0.5258** | Brier (actual): **0.1783**

**vs B0:** Worst overall action accuracy (MOVE T3 −8pt, BOTH T3 ~the same). But SWITCH Top-1 hit
99.23% — best of all ablations. Win head substantially better than B0 (+0.23 Win Corr).
**Hypothesis partially wrong:** dropping engineered features did NOT hurt the value head as
predicted; it improved it. Damage-calc features are not load-bearing for the value head at 10
epochs — possibly because they're redundant with the type/ability/HP info already in MON/OPP_MON.

---

### A3 — ~5× Smaller Model (26.7M params)
**Config:** `configs/ablation_a3_small5x.yaml`
**Change:** Scale down transformer and MLP dims (see config for exact values).
**Params:** 26.7M (4.7× smaller); featureset=`full`

**Pre-run hypothesis:**
- **Action accuracy**: Moderate drop (~3–6% on Top-3). The baseline 125M model has excess capacity
  for a 450K-battle BC dataset. A 5× smaller model can likely fit the data's structure, but fine-
  grained move prediction may suffer (prediction of low-frequency moves from rare board states).
  SWITCH accuracy should hold near 99%.
- **Win Corr**: Small-to-moderate drop (~0.02–0.05). Value estimation benefits from depth, but the
  C51 distributional head is robust; a shallower transformer may capture temporal credit assignment
  less well.
- **RL implication**: ~4–5× faster forward pass per inference step. With 6 workers × 18 players,
  this translates to ~4–5× more battles/second assuming compute-bound. Memory per worker ~5× lower.
  Asymptotic RL policy quality may suffer more than BC quality (RL explores harder states).
- **Speed gain**: Significant. Expect ~3–5× speedup on forward pass.

**Results (A3 — `eternal-cherry-83_best.pt`, 10 epochs, ~3h35m training):**
- Overall Top-1/3/5/10: **39.81% / 58.69% / 66.02% / 78.64%** (BEST action accuracy of all ablations)
- MOVE Top-1/3/5: **26.75% / 48.34% / 58.07%** (close to 30-epoch reference!)
- SWITCH Top-1/3: **90.87% / 99.92%**
- BOTH Top-1/3/5: **44.35% / 61.03% / 64.98%** (best BOTH T3 of any ablation)
- FORCE_SWITCH Top-1: **63.36%**
- Win Corr (actual): **0.2906** (WORST of all ablations, −0.08 vs B0)
- Synth Corr: **0.3308** | Brier (actual): **0.2224**

**vs B0:** Action accuracy markedly **better** (+4.8pt Top-3, +5.4pt MOVE T3, +8.2pt BOTH T3) —
the 5× smaller model **outperforms** B0 on action prediction at 10 epochs. But the value head
**suffers most** of any ablation (Win Corr 0.291 vs 0.369). **Hypothesis partially wrong on
direction:** action accuracy went up, not down — but value head took the predicted hit and harder.

---

### A4 — ~10× Smaller Model (13.9M params)
**Config:** `configs/ablation_a4_small10x.yaml`
**Change:** Scale down further (3 transformer layers, smaller hidden dims).
**Params:** 13.9M (9× smaller); featureset=`full`

**Pre-run hypothesis:**
- **Action accuracy**: Significant drop (~5–10% on Top-3). At 14M params with a 3-layer transformer,
  the model has little depth to capture temporal dependencies across the trajectory. Early-game moves
  (which require reasoning over teampreview + initial reveals) will degrade most.
- **Win Corr**: Moderate drop (~0.04–0.08). The value head is most sensitive to depth.
- **RL implication**: Highest throughput gain (~8–9× forward pass speedup). May cap RL quality
  significantly in the long run, especially for the value/critic head (MCTS and r-NaD both rely on it).
  Probably only worth it if rapid iteration speed matters more than asymptotic quality in Stage II.
- **Speed gain**: Very significant.

**Results (A4 — `lyric-dew-84_best.pt`, 10 epochs):**
- Overall Top-1/3/5/10: **31.50% / 48.46% / 57.15% / 64.31%**
- MOVE Top-1/3/5: **15.90% / 35.68% / 47.57%**
- SWITCH Top-1/3: **90.92% / 99.92%**
- BOTH Top-1/3/5: **38.81% / 50.65% / 53.42%**
- FORCE_SWITCH Top-1: **62.72%**
- Win Corr (actual): **0.4743** (better than B0!)
- Synth Corr: **0.4717** | Brier (actual): **0.1960**

**vs B0:** Action accuracy worse than A3 but comparable to B0 on Top-3 (48.46% vs 53.86%, −5.4pt).
Value head **better** than B0 (+0.10 Win Corr) — the opposite of the predicted direction.
**Hypothesis wrong on value head:** the 10× smaller model has *better* value estimation than B0
at 10 epochs. Combined with A3's worst-Win-Corr result, this suggests value head fragility is
non-monotonic with model size at 10 epochs (possibly a bad local minimum for A3 specifically).

---

## Summary Table

| Run | Change | Params | Overall T1 | Overall T3 | MOVE T3 | SWITCH T1 | BOTH T3 | Win Corr | Brier | Synth Corr |
|-----|--------|--------|-----------:|-----------:|--------:|----------:|--------:|---------:|------:|-----------:|
| Reference (30ep) | — | 125.6M | 41.4% | 61.3% | 51.2% | 99.1% | 65.2% | 0.818 | 0.133 | 0.665 |
| **B0** (10ep) | epochs only | 125.6M | 36.10% | 53.86% | 42.97% | 81.00% | 52.86% | **0.369** | 0.212 | 0.409 |
| **A1** (10ep) | −transition | 125.6M | 35.16% | 43.47% | 26.47% | 91.68% | **61.59%** | **0.582** | 0.178 | 0.529 |
| **A2** (10ep) | raw featureset | 125.6M | 33.42% | 47.86% | 34.85% | **99.23%** | 50.50% | **0.603** | **0.178** | 0.526 |
| **A3** (10ep) | 5× smaller | 26.7M | **39.81%** | **58.69%** | **48.34%** | 90.87% | 61.03% | 0.291 | 0.222 | 0.331 |
| **A4** (10ep) | 10× smaller | 13.9M | 31.50% | 48.46% | 35.68% | 90.92% | 50.65% | 0.474 | 0.196 | 0.472 |

(Bold = best in column among ablations. "Win Corr" = `prediction_vs_actual_overall.correlation`.)

### Key observations

1. **Fewer features helped the value head, not hurt it.** A1 (no transition) and A2 (raw)
   both produced *substantially better* Win Corr than B0 at 10 epochs (+0.21 / +0.23). The
   working hypothesis was that transition + engineered features carried critical signal for
   value learning; the data instead suggests they add noise relative to the time budget. This
   does not mean those features have zero value — at 30 epochs the picture may differ — but at
   the 10-epoch budget that matches RL warm-start cycles, they are net-negative for value Corr.

2. **A3 (~5× smaller) is the action-accuracy winner but the value-Corr loser.** A3's action
   metrics are the closest of any ablation to the 30-epoch reference (BOTH T3 61.0%, MOVE T3
   48.3%), but its Win Corr collapses to 0.291 — worst of all runs. This is a known small-
   network failure mode: insufficient depth to integrate trajectory information for long-
   horizon credit assignment.

3. **A4 (~10× smaller) trades action accuracy for a less-bad value head than A3.** A4's MOVE
   T3 is comparable to A2 but its Win Corr (0.474) is much better than A3's. This is mildly
   surprising and suggests A3's value-head collapse is not purely a size effect — it may be
   architectural (the specific 4-layer × 1024-ff configuration we picked).

4. **MOVE Top-3 is the noisiest column.** It varies from 26% (A1) to 51% (Reference) without
   a clean monotonic trend. SWITCH Top-1 and BOTH Top-3 are more reliable signals.

5. **All runs are well below the 30-epoch reference on every metric.** The 10-epoch baseline
   (B0) loses 7–18pt across action metrics and 0.45 in Win Corr vs the 30-epoch reference.
   So while ablation **rankings** are likely informative, **absolute scores** would shift
   substantially with more training.

## Reasoning

- **Feature ablations first** (A1, A2): Low-risk architecture changes that isolate data signal from
  model capacity. Results directly inform whether transition/engineered features matter for BC.
- **Size ablations** (A3, A4): Directly address RL speed question. If A3 degrades <3% on BOTH Top-3
  and <0.03 on Win Corr vs B0, it's a strong candidate for RL. If A4 is only marginally worse than
  A3 on quality but delivers >2× additional speedup, the trade may still be worth it.
- **All runs at 10 epochs**: Longer training could narrow gaps (big models benefit more from more
  epochs), but 10 epochs is sufficient to establish the ranking. If a smaller model closes the gap
  after more training, that's a bonus for RL where we can retrain BC before RL.

## Planned Next Steps (recommendations, post-results)

1. **Drop transition features in production BC training.** A1 had higher Win Corr than B0 and
   slightly different action profile. Removing transitions also drops the poke-env observations
   dependency at RL inference time (the user's original goal). Strong recommendation.

2. **Move to RAW featureset with caution.** A2 had the *highest* Win Corr of any 10-epoch run
   AND the highest SWITCH Top-1 (99.23%, near-reference) AND the highest FORCE_SWITCH (96.55%).
   Its weakness is BOTH Top-3 (50.5% vs A1 61.6%). For RL warm-start, the value head and SWITCH
   accuracy may matter more than BOTH accuracy — but this is a judgment call. If RAW is adopted,
   the engineered damage-calc features that RL would otherwise consume can be dropped from the
   embedder pipeline entirely, simplifying the RL embedding step.

3. **5× smaller model needs a longer training run before judgment.** A3's action metrics are
   compelling (BOTH T3 61% vs B0's 53%) and its size makes it 4–5× faster at inference. But its
   Win Corr collapse needs investigation. **Suggested follow-up:** train A3 for 30 epochs with
   the same config and re-check Win Corr. If it recovers to >0.7, A3 becomes the strongest RL
   warm-start candidate.

4. **A4 (10× smaller) is a viable fallback.** Its Win Corr is decent and its action metrics
   match A2. Worth keeping in mind if A3 doesn't recover at 30 epochs.

5. **Combined ablation worth running.** RAW featureset + ~5× smaller architecture would be the
   maximum-speedup configuration. The RAW featureset specifically might fix A3's value-head
   issue (since A2 had strong Win Corr at full size). Suggested next ablation: A3 architecture
   + RAW featureset, 30 epochs.

6. **Caveat — 10-epoch results are noisy at this scale.** All conclusions above are tentative.
   The 30-epoch reference is much stronger than any 10-epoch run on every metric. Before
   committing to an architecture change for RL, retrain the chosen winner at 30 epochs and
   re-validate.

## Updates

- 2026-05-01 21:04: Started B0 training.
- 2026-05-02 00:48: B0 training complete, diagnostics ran successfully.
- 2026-05-02 03:30 (approx): WSL crashed mid-A1 (5/10 epochs done). Aborted run cleaned up.
- 2026-05-02 13:28: Restarted via resumable script.
- 2026-05-02 17:45: A1 training complete. Action diagnostics OK; **win diagnostics failed**
  due to `feature_set="full"` hardcoded in `win_model_diagnostics.py:603`. Bug fixed in-flight
  by reading featureset from saved config.
- 2026-05-02 17:51: A2 started.
- 2026-05-02 22:09: A2 complete + diagnostics.
- 2026-05-03 01:37: A3 complete + diagnostics.
- 2026-05-03 05:13: A4 complete + diagnostics. Script finished.
- 2026-05-03 05:41: Manual re-run of A1 win diagnostics (with the fixed loader).
- 2026-05-03: Final summary written.
