# Supervised Learning

## Executive summary

- This folder holds code to train a behavioral clone on millions of human VGC battles to (1) warm-start Stage II r-NaD RL and (2) provide a policy + value backbone for the Stage V search agent.
- **Current best model**: `data/models/supervised/rose-sun-108-mega_best.pt` (ask me for weights) — [~30.9M param transformer model with three heads (turn action 2025-way, teampreview 90-way, distributional C51 value), grouped features, action/critic/field tokens](model_archs.py), trained on the **mega-aware featureset** (Mega Stones in the item vocab and a prospective mega-form feature block per mon) on `data/battles/mega_final_v1/` — 460K 1500+ VGC Regulation C non-omniscient battles ([config here](configs/may24.yaml)).
- **Model Performance**: Overall Top-1 / 3 = **55.0% / 83.5%**, Teampreview Top-1 = 99.9% (overfit; players are consistent with their teampreview selections on Showdown), SWITCH Top-1 = 99.6%, FORCE_SWITCH Top-1 = 99.7%. Advantage prediction correlates with probability of winning at **0.60**, and the model emits **zero invalid actions** — confirming the mega vocab + feature block are wired end-to-end correctly. Full diagnostics table in [Current state](#current-state-and-findings).
- **Main entry points**: [`train.py`](train.py) (BC from scratch), [`fine_tune.py`](fine_tune.py) (continue from a checkpoint), [`agents/bc_player.py`](../agents/bc_player.py) (play / RL-evaluate via poke-env).


## What's in this folder

| Path | Purpose |
|---|---|
| [`model_archs.py`](model_archs.py) | `TransformerThreeHeadedModel`, `GroupedFeatureEncoder`, `NumberBankEncoder`, `SinusoidalPositionalEncoding`. |
| [`train.py`](train.py) | Production trainer for `TransformerThreeHeadedModel`. Config-driven (YAML), mixed precision, gradient accumulation, WandB logging, best/last checkpoint saving. |
| [`fine_tune.py`](fine_tune.py) | Loads a `.pt` checkpoint, optionally overrides non-architecture hyperparameters from YAML, and continues training using the same `train_epoch` / `evaluate` infra. |
| [`train_sweep.py`](train_sweep.py) | Sweep entry point — reuses `train_epoch` from `train.py`, consumes wandb sweep configs. |
| [`train_non_traj.py`](train_non_traj.py) | Non-trajectory ablation trainer (state → next action, no RNN / temporal component). Benchmarking only. |
| [`utils.py`](utils.py) | Shared training/eval utilities: `topk_cross_entropy_loss`, `focal_topk_cross_entropy_loss`, `evaluate` (top-1/3/5 action and TP accuracy), CUDA prefetcher integration. |
| [`configs/`](configs/) | YAML training configs. Current production: [`may24.yaml`](configs/may24.yaml). |
| [`sweep_configs/`](sweep_configs/) | YAML sweep configs. |
| [`analyze/`](analyze/) | Diagnostics: `action_model_diagnostics.py` (the metrics table below), `win_model_diagnostics.py`, `behavior_clone_performance.py`, `behavior_clone_replay.py`, `state_eval_baseline.py`, `training_profiler.py`. |

> *The agent wrapper for poke-env lives outside this folder at [`src/elitefurretai/agents/bc_player.py`](../agents/bc_player.py) (was `behavior_clone_player.py`).*

### `TransformerThreeHeadedModel` detail

The model used for all supervised + RL training.

- **Decision tokens**: three learned vectors `[ACTOR]`, `[CRITIC]`, `[FIELD]` prepended to the sequence. ACTOR output feeds the turn head; CRITIC feeds the value head.
- **Positional encoding**: sinusoidal, so variable-length sequences work at inference (up to `max_len`).
- **Causal mask**: past turns attend to themselves and earlier turns. Decision tokens attend to everything.
- **Hidden state**: growing context tensor of past encoded features; each turn appends.
- **Detached TP head**: teampreview uses `encoded.detach()` so TP gradients do not flow back into the shared encoder.
- **Heads**:
  1. **Turn head** (2025 classes) — Cartesian product of legal move/target/switch/tera combinations for the two active Pokémon, flattened by `MDBO`.
  2. **Teampreview head** (90 classes) — $\binom{6}{2}\binom{4}{2}$ unordered lead/back picks.
  3. **Win head** (distributional, C51) — 51 bins over [-1, 1]; expected value = `(softmax(logits) * support).sum(-1)`. Targets are two-hot encoded via `twohot_encode()`. See [Win head target](#win-head-target--ensemble-advantage) for what the label actually is.
- `forward()` returns `(turn_logits, tp_logits, win_values, win_dist_logits)`; `forward_with_hidden()` also returns the next context tensor.

`NumberBankEncoder` swaps raw floats for learned embedding lookups on selected numeric features (HP% → 100 bins, stats → 600 bins, base power → 250 bins). Pattern-matched on `Embedder.feature_names` and applied inside `GroupedFeatureEncoder`; gated by `use_number_banks` (off by default). The Embedder output format is unchanged.

### Win head target — ensemble advantage

The win head does not regress to the binary battle outcome. Its per-turn label is a blended **ensemble advantage** in [-1, 1] computed by [`BattleDataset._compute_ensemble_advantage`](../etl/battle_dataset.py):

1. **Position component** — [`evaluate_position_advantage(battle)`](../etl/evaluate_state.py) sums per-Pokémon HP / status / hazards / boost contributions on each side, adds speed-comparison bonuses, and normalizes the raw score by 500 into [-1, 1] (positive = favorable for the training perspective). The turn label uses `0.5 * pos[i] + 0.5 * mean(pos[i+1 : i+4])`, so it folds in a small amount of near-future position signal.
2. **Outcome component** — `+1` if the player won the battle, `-1` otherwise.
3. **Weighting** — for a battle of `n` turns, `outcome_weight = clip((i / (n-1))², 0.05, 0.95)` and `position_weight = 1 - outcome_weight`. Turn 0 is ~95% position / 5% outcome; the final turn is ~5% position / 95% outcome. The value head therefore sees a smooth, dense signal early in the battle and converges to the true outcome by the end. Single-turn battles fall back to pure outcome.

Final per-turn label: `position_weight * (0.5 * pos[i] + 0.5 * mean(pos[i+1 : i+4])) + outcome_weight * final_outcome`. The C51 head's 51 logits over [-1, 1] are trained against this scalar via `twohot_encode()`. The "Win Correlation" row in the diagnostics table is Pearson correlation between the head's expected-value scalar `(softmax(logits) * support).sum(-1)` and the true battle outcome (per turn); "Brier Score" is against this ensemble advantage label.

## How to use it

### 1. Data prep

Raw Showdown logs → `src/elitefurretai/etl/process_training_data.py` → `.pt.zst` files. Training scripts load these via `BattleDataset`. It takes in a json file that links your replays. I filter my replays first via [`../etl/filter_battle_data.py`](../etl/filter_battle_data.py)

### 2. Train

```bash
python src/elitefurretai/supervised/train.py \
    data/battles/regc_final_v5/ \
    --config src/elitefurretai/supervised/configs/may24.yaml \
    --save-best
```

`may24.yaml` is the current production config. This command saves the best model encountered,

### 3. Fine-tune

```bash
python src/elitefurretai/supervised/fine_tune.py \
    data/battles/specific_team_data \
    data/models/pretrained_checkpoint.pt \
    "finetune_experiment"
```

`fine_tune.py` reconstructs the exact `TransformerThreeHeadedModel` from the embedded config, optionally overrides non-architecture hyperparameters from YAML (LR, num_epochs, weight_decay, dropout, lr_schedule), recomputes embedder-derived indices (`teampreview_idx`, `force_switch_indices`, `state_input_dim`) from a fresh embedder so they stay in sync if the feature schema evolved, and reuses the same `train_epoch` / `evaluate` / `analyze` infra. Architecture keys must not change — they would mismatch the loaded weight shapes.

### 4. Play / RL-evaluate

```python
from elitefurretai.agents.bc_player import BCPlayer

player = BCPlayer(
    model_filepath="data/models/my_model.pt",
    battle_format="gen9vgc2024regg",
    device="cuda",
)
await player.battle_against(opponent, n_battles=1)
```

`BCPlayer` runs the forward pass, masks invalid actions before selection, and supports greedy or probabilistic action choice. You can play with it using `exmaples/human_player_example.py`

### 5. Diagnose a checkpoint

You can run [`./analyze/action_model_diagnostics.py`](analyze/action_model_diagnostics.py) on a data split to generate metrics and deeper analysis on your models' behaviors to better understand its strengths and weaknesses for iteration.

## Current state and findings

### Production checkpoint: `rose-sun-108-mega_best.pt`
> *This model is not pushed. You can ask me for the weights if you'd like.*

- **Config**: [`configs/may24.yaml`](configs/may24.yaml).
- **Architecture (~30.9M params, 124MB)**: transformer 4 layers × 8 heads, ff_dim=2048, agg=2048, hidden=256, early=[1024, 512, 512], late=[512, 512], turn_head=[512, 256, 256], teampreview_head=[256, 128].
- **Featureset**: `raw` featureset in `Embedder` class, **5,374 input dims**. Includes the prospective mega-form feature block (`MEGA_STAT:*`, multi-hot `MEGA_TYPE:*`, `mega_ability_id`) per mon plus scalar `is_mega_evolved` / `can_mega` / `can_tera` / `gimmick_spent` flags. On tera-format data these features are constant (-1 / unknown) and behave inertly; they only carry signal in the mega format.
- **Vocab**: `ITEM_TO_ID` includes 91 Mega Stones (derived from `requiredItem` on mega formes in gen9 GenData); the item embedding is sized **132**.
- **Data**: `data/battles/mega_final_v1/` — 460K-battle regC corpus with the mega-aware embedder; 90/5/5 train/val/test chunks at 512 trajectories/chunk.

Diagnostic metrics (full test split, `data/battles/mega_final_v1/test`, 676 batches × 64):

| Metric | Value |
|---|---:|
| **Overall Action Top-1 / 3** (incl. teampreview) | **55.0% / 83.5%** |
| Overall Action Top-1 / 3 / 5 / 10 (turn actions only, excl. TP) | 49.14% / 81.35% / 90.79% / 95.86% |
| MOVE Top-1 / 3 / 5 / 10 | 34.55% / 76.19% / 86.96% / 94.18% |
| BOTH Top-1 / 3 / 5 (both slots act) | 45.83% / 78.47% / 98.46% |
| SWITCH Top-1 / 3 | 99.56% / 99.97% |
| FORCE_SWITCH Top-1 | 99.71% |
| Teampreview Top-1 | 99.88% |
| Move-id correct (move-vs-move pairs) | 81.87% |
| **Win Correlation** (advantage → win) | **0.601** |
| **Invalid predictions** | **0** (0.00%) |

Teampreview / SWITCH / FORCE_SWITCH saturate near 100% (deterministic decisions). MOVE is the genuine hard problem with the right move in top-5 ~87% of the time. The model emits **zero invalid actions** — confirming the mega vocab and feature block are wired end-to-end correctly through embedder → model → mask.


## Operating notes

### `eval_every` config knob

`eval_every: N` evaluates on the test set every N epochs. The final epoch always evaluates. Useful when fine-tunes are eval-dominated (`may22_finetune.yaml` uses `eval_every: 5`; other configs default to 1).

When `eval_every > 1`:
- `ReduceLROnPlateau` only steps on eval epochs (it needs `test_loss`).
- The cosine scheduler still steps every epoch.
- `save_best` only triggers on eval epochs.

### WSL2 / 8-core dataloader notes

The current paramaters are highly optimized for my work setup (RTX 3090, i7-7700K 8-core, 24 GB RAM, WSL2, NVMe) and so you need to readdress these parameters to your setup to get similar speed and throughput that I get. Right now:

- **CPU contention is the bottleneck**, not I/O. On 8 logical cores, `num_workers=3` outperformed `num_workers=7` by ~4.4×.
- **`pin_memory=True` causes OOM on WSL2.** Always `pin_memory=False`.
- **Worker batch sizes above the WSL2 shared-memory / file-descriptor cap crash with bus errors.** 
- Decompression of `.pt.zst` files (~201× ratio) is CPU-intensive, so fewer workers means each worker gets more CPU and finishes decompression sooner. Taking a step back, I precompute and store them compressed so that I only need I/O and decompression load to pass to GPU -- the size they're compressed with (`chunk-size` in `etl/process_training_data.py`) is aligned with my batch_sizes to maximize throughput.

If hardware changes (more cores, more RAM, native Linux), re-measure rather than transplant these numbers.

### Diagnostic commands

To help you understand what your bottlenecks are, to optimize training on your hardware:

```bash
# GPU SM utilization (30s, 1Hz)
nvidia-smi dmon -s u -d 1 -c 30 | awk 'NR>2 {sum+=$2; n++; if($2>max)max=$2} END {print "sm util avg=" sum/n " max=" max}'

# py-spy flamegraph against a live training PID
sudo /home/cayman/Repositories/venv/bin/py-spy record \
    -o ~/diag/train.svg --pid <PID> --idle --subprocesses --native --rate 10 -d 180

# Look for sync-stall frames in the flamegraph (should be near zero after the .item() deferral)
grep -oE '<title>[^<]+\([0-9]+ samples[^<]*</title>' ~/diag/train.svg | \
    grep -iE "_local_scalar_dense|cudaStreamSynchronize|aten::item" | head -5
```
