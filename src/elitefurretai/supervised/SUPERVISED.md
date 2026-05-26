# Supervised Learning

## Executive summary

- This folder holds code to train a behavioral clone on millions of human VGC battles to (1) warm-start Stage II r-NaD RL and (2) provide a policy + value backbone for the Stage V search agent.
- **Current best model**: `data/models/supervised/rose-sun-108_best.pt` (ask me for weights) — [30.5M param transformer model with three heads (turn action 2025-way, teampreview 90-way, distributional C51 value), grouped features, action/critic/field tokens](model_archs.py), trained for 30 epochs of 460K 1500+ VGC Regulation C non-omniscient battles ([config here](configs/may24.yaml)) at ~5m per epoch due to [optimized training and pre-computed features](#operating-notes).
- **Model Performance**: It can predict player actions Top-1/3 = 55.1% / 84.1%, Teampreview Top-1 = 99.9% (overfit; players are consistent with their teampreview selections on showdown), advantage prediction correlates with probability of winning w/ correlation coefficient of 0.55. Full table in [Current state](#current-state-and-findings).
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
  3. **Win head** (distributional, C51) — 51 bins over [-1, 1]; expected value = `(softmax(logits) * support).sum(-1)`. Targets are two-hot encoded via `twohot_encode()`.
- `forward()` returns `(turn_logits, tp_logits, win_values, win_dist_logits)`; `forward_with_hidden()` also returns the next context tensor.

`NumberBankEncoder` swaps raw floats for learned embedding lookups on selected numeric features (HP% → 100 bins, stats → 600 bins, base power → 250 bins). Pattern-matched on `Embedder.feature_names` and applied inside `GroupedFeatureEncoder`; gated by `use_number_banks` (off by default). The Embedder output format is unchanged.

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

### Production checkpoint: `rose-sun-108_best.pt`
> *This model is not pushed. You can ask me for the weights if you'd like.*

- **Config**: [`configs/may24.yaml`](configs/may24.yaml).
- **Architecture (~30.5M params, 117MB)**: transformer 4 layers × 8 heads, ff_dim=2048, agg=2048, hidden=256, early=[1024, 512, 512], late=[512, 512], turn_head=[512, 256, 256], teampreview_head=[256, 128].
- **Featureset**: `raw` featureset in `Embedding` class (5056 input dims). 
- **Data**: `data/battles/regc_final_v5/` — 90/5/5 train/val/test chunks at 512 trajectories/chunk.

Diagnostic metrics (200-batch run, ~55K predictions on the test split):

| Metric | Value |
|---|---:|
| Overall Action Top-1 / 3 / 5 / 10 | **55.09% / 84.11% / 87.67% / 96.54%** |
| MOVE Top-1 / 3 / 5 (where player chose attacks) | **44.93% / 80.45% / 84.72%** |
| BOTH Top-1 / 3 (where player switched + attacks) | **77.18% / 91.35%** |
| SWITCH Top-1 / 3 (where player switched) | 97.29% / 99.81% |
| FORCE_SWITCH Top-1 (where player was forced to switch) | 94.12% |
| Teampreview Top-1 | 99.9% |
| Win Correlation (advantage prediction correlated with winning battles) | 0.548 |
| Brier Score (against true advantage score) | 0.185 |


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
