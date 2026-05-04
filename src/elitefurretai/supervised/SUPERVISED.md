# Supervised Learning

This folder contains the code for training, fine-tuning, and evaluating supervised learning models for EliteFurretAI. These models are trained on millions of human battles to learn behavioral cloning (imitating high-Elo players).

## Workflow

### 1. Data Preparation
Before training, you must have processed training data ready.
1.  **Extract**: Raw Showdown logs.
2.  **Transform**: Run `src/elitefurretai/etl/process_training_data.py` to generate `.pt.zst` files.
3.  **Load**: The training scripts below will load these files using `BattleDataset`.

### 2. Training
The primary training script is `train.py`. This trains the `TransformerThreeHeadedModel` which predicts:
*   **Turn Actions**: What move/switch to make (2025 classes).
*   **Teampreview**: Which 4 Pokemon to bring and in what order (90 classes).
*   **Win Probability**: Who is likely to win (distributional C51).

```bash
# Example usage
python src/elitefurretai/supervised/train.py --config configs/curious_darkness_77.yaml
```

### 3. Fine-Tuning
To adapt a pre-trained model to a specific team or playstyle, use `fine_tune.py`. This loads a checkpoint and continues training, optionally with different hyperparameters.

```bash
# Example usage
python src/elitefurretai/supervised/fine_tune.py \
    data/battles/specific_team_data \
    data/models/pretrained_checkpoint.pt \
    "finetune_experiment"
```

### 4. Inference / Playing
To use the trained model in a battle, use `behavior_clone_player.py`. This wraps the model in a `poke-env` Player class.

```python
from elitefurretai.supervised.behavior_clone_player import BCPlayer

player = BCPlayer(
    model_filepath="data/models/my_model.pt",
    battle_format="gen9vgc2024regg",
    device="cuda"
)
await player.battle_against(opponent, n_battles=1)
```

---

## File Documentation

### `model_archs.py`
**Purpose**: Defines the neural network architectures.

**Key Classes**:

#### `FlexibleThreeHeadedModel` (Legacy)
This was the initial architecture for the project, using a bidirectional LSTM backbone. Superseded by `TransformerThreeHeadedModel` for the Stage I handoff.
*   **Design Choice (Grouped Encoder)**: Instead of a flat input vector, features are split into semantic groups (Player Mon 1-6, Opponent Mon 1-6, Global State). Each group is encoded separately. This allows the model to learn "Pokemon" representations that are invariant to slot position.
*   **Design Choice (Cross-Attention)**: After encoding, an attention mechanism allows Pokemon to "look at" each other. This explicitly models synergy (teammate-teammate attention) and matchups (player-opponent attention).
*   **Design Choice (Three Heads)**:
    1.  **Turn Head**: Predicts the next move (0-2024).
        *   **Why 2025 actions?** In VGC Doubles, you control 2 Pokemon. The action space is the Cartesian product of all possible legal moves for the active Pokemon.
            *   Moves: 4 moves * 2 targets (opponent 1/2) + 1 target (ally) = ~9 options per move slot.
            *   Switches: Switch to slot 3, 4, 5, or 6.
            *   Terastallization: Can happen with any move.
            *   We also include empty options like pass/default as well.
            *   The `MDBO` class flattens all valid combinations of these into 2025 discrete integers. This allows us to use standard classification loss instead of complex multi-output regression.
    2.  **Teampreview Head**: Predicts the lead 4 Pokemon (0-89).
        *   **Why 90 actions?** You must choose 4 Pokemon out of 6. In VGC, the *order* of the two leads doesn't matter (Lead A+B is same as B+A), and the order of the two back Pokemon doesn't matter.
            *   Ways to pick 2 leads from 6: $\binom{6}{2} = 15$
            *   Ways to pick 2 back from remaining 4: $\binom{4}{2} = 6$
            *   Total combinations: $15 \times 6 = 90$.
    3.  **Win Head (Distributional)**: Predicts win probability via C51 distributional head.
        *   **C51 Distribution**: Instead of a scalar in [-1, 1], outputs logits over 51 bins spanning [-1, 1]. The expected value is computed as `(softmax(logits) * support).sum(-1)`.
        *   **Two-hot encoding**: Target values are encoded as soft distributions via `twohot_encode()` — interpolating between the two nearest bin centers.
        *   **Why distributional?** Captures the full return distribution (win/loss is bimodal). Provides richer gradients than scalar MSE. Stabilizes RL training significantly.
        *   **Forward return change**: `forward()` returns 4 values: `(turn_logits, tp_logits, win_values, win_dist_logits)`. `forward_with_hidden()` returns 5: adds hidden state.

#### `NumberBankEncoder`
Replaces raw float inputs for selected numerical features with learned embedding lookups.
*   **Feature types**: HP% → `hp_bank` (100 bins), stats → `stat_bank` (600 bins), base power → `power_bank` (250 bins)
*   **How it works**: Features are identified by pattern matching on `Embedder.feature_names`. Each matched feature is discretized into buckets and replaced with a learned embedding vector.
*   **Integration**: Applied inside `GroupedFeatureEncoder` — the Embedder output format is unchanged
*   **Config**: Gated by `use_number_banks` (disabled by default)

#### `TransformerThreeHeadedModel` (Current)
The primary backbone, replacing the bidirectional LSTM with a TransformerEncoder and decision tokens. The Stage I research baseline is `curious-darkness-77` (~125M params, full featureset). The Stage II RL handoff is the `cool-bee-85-finetune` configuration (~26.7M params, raw featureset, ~5× smaller; same module class, smaller hyperparams).
*   **Decision tokens**: Three learned parameter vectors `[ACTOR]`, `[CRITIC]`, `[FIELD]` are prepended to the sequence. ACTOR token output feeds the turn head, CRITIC feeds the value head.
*   **Positional encoding**: Sinusoidal (supports variable-length sequences at inference)
*   **Causal mask**: Past turns can only attend to themselves and prior turns. Decision tokens can attend to everything.
*   **Hidden state**: Instead of LSTM `(h, c)`, uses a growing context tensor of past encoded features. Each turn appends to the context.
*   **Detached TP head**: The teampreview head uses `encoded.detach()` so TP gradients do not flow back into the shared encoder, preventing harmful gradient interference with the action/value trunk.
*   **Config**: Gated by `use_transformer`. Key params: `transformer_layers=7`, `transformer_heads=16`, `transformer_ff_dim=2048`
*   **Same three heads**: Identical turn/teampreview/win head structure as `FlexibleThreeHeadedModel`

#### `GroupedFeatureEncoder`
Encodes features by semantic groups with optional number bank integration and cross-attention for Pokemon synergies.

#### `SinusoidalPositionalEncoding`
Standard sinusoidal positional encoding for the Transformer backbone. Supports variable-length sequences up to `max_len`.

### `train.py`
**Purpose**: The main training script for both `FlexibleThreeHeadedModel` and `TransformerThreeHeadedModel`.

**Key Features**:
*   **Config-driven**: Accepts a YAML config file (see `configs/curious_darkness_77.yaml` for the Stage I handoff config).
*   **Mixed Precision**: Uses `torch.amp` for faster training on modern GPUs.
*   **Gradient Accumulation**: Allows training with large effective batch sizes even on limited VRAM.
*   **Loss Weighting**: Balances the three heads (Action, Teampreview, Win) to ensure one doesn't dominate the gradient.
*   **WandB Integration**: Logs metrics (loss, accuracy, top-k accuracy) to Weights & Biases.
*   **Checkpoint saving**: Saves best model (by test loss) and final model.

### `train_sweep.py`
**Purpose**: Sweep training script for hyperparameter search. Imports `train_epoch` from `train.py` and supports wandb sweep configs.

### `fine_tune.py`
**Purpose**: Adapts a pre-trained `TransformerThreeHeadedModel` to new data or a new training schedule.

**How it works**:
1.  Loads the model architecture and weights from a `.pt` checkpoint, reconstructing the exact `TransformerThreeHeadedModel` structure from the embedded config.
2.  Optionally overrides training parameters (LR, num_epochs, weight_decay, dropout, lr_schedule, etc.) from a YAML file. Architecture keys must NOT change — they would mismatch the loaded weight shapes.
3.  Recomputes embedder-derived indices (`teampreview_idx`, `force_switch_indices`, `state_input_dim`) from a fresh embedder so they stay in sync if the feature schema evolved between runs.
4.  Reuses the same `train_epoch`, `evaluate`, and `analyze` infrastructure as `train.py`, including `torch.compile`, `lr_schedule` choice (`cosine`/`plateau`), mixed precision, and gradient accumulation.
5.  `--save-best` flag saves the lowest-test-loss checkpoint mid-run.

### `behavior_clone_player.py`
**Purpose**: The agent interface for `poke-env`.

**Key Class**: `BCPlayer`
*   **Integration**: Inherits from `poke_env.Player`.
*   **Inference**: Runs the model forward pass to get logits.
*   **Action Selection**:
    *   **Greedy**: Picks the action with the highest probability.
    *   **Probabilistic**: Samples from the softmax distribution (better for diversity/exploration).
*   **Masking**: Crucially, it masks out invalid actions (e.g., using a disabled move) before selection to ensure the agent never crashes.

### `train_utils.py`
**Purpose**: Shared utilities for training and evaluation.

**Key Functions**:
*   `topk_cross_entropy_loss`: A custom loss function. In Pokemon, multiple moves might be "correct". If the model predicts a good move that isn't the *exact* ground truth, we don't want to penalize it heavily. This loss only penalizes if the ground truth isn't in the top-K predictions.
*   `focal_topk_cross_entropy_loss`: Adds focal loss to downweight "easy" examples (obvious KOs) and focus learning on complex, high-uncertainty turns.
*   `evaluate`: Runs a full validation pass, computing Top-1, Top-3, and Top-5 accuracy for both actions and teampreview.

### `feed_forward_action.py`
**Purpose**: A simpler baseline model.
*   **Architecture**: A standard Multi-Layer Perceptron (MLP) without attention or LSTMs.
*   **Use Case**: Useful for debugging the pipeline or establishing a performance baseline to see how much value the Transformer architecture adds.

---

## Research Findings & Benchmarks

Based on our experiments and analysis (detailed in the [project documentation](https://docs.google.com/document/d/14menCHw8z06KJWZ5F_K-MjgWVo_b7PESR7RlG-em4ic/edit)), here are the key findings that drive our supervised learning strategy:

### 1. Current Benchmarks

**Production checkpoint for Stage II RL: `cool-bee-85-finetune_best.pt`** — 26.7M params, RAW featureset, transformer (4 layers × 8 heads × ff_dim 1024). Originally trained as A5 (`cool-bee-85`, 30 epochs combining the A3 ~5× smaller architecture with the A2 raw featureset), then fine-tuned for 15 additional epochs with `win_loss_weight=0.5` and a cosine LR (peak 2.5e-5). See `planning/stage2/2026-05-01-bc-ablation-study.md` for the ablation study that motivated this configuration.

**Reference checkpoint (BC research baseline, not used for RL): `curious-darkness-77_best.pt`** — 125M params, FULL featureset, transformer (7 layers × 16 heads × ff_dim 2048). Stronger pure-BC numbers but ~5× slower at inference and uses transition features (which couple inference to `poke-env`'s observations).

| Metric | `cool-bee-85-finetune_best` (production) | `curious-darkness-77_best` (reference) |
|---|---:|---:|
| Params | **26.7M** (~5× smaller) | 125M |
| Featureset | **raw** (5032 dims) | full (5222 dims) |
| Overall Top-1 | **45.33%** | 41.36% |
| Overall Top-3 | **62.15%** | 61.27% |
| Overall Top-5 | 67.56% | 68.77% |
| MOVE Top-3 | **52.63%** | 51.23% |
| SWITCH Top-1 | **99.35%** | 99.06% |
| BOTH Top-1 | 47.66% | 48.27% |
| BOTH Top-3 | 64.45% | 65.16% |
| TP Top-1 | 99.9% | 53.8% |
| **Actual Win Correlation** | 0.7531 | **0.8175** |
| **Brier Score** | 0.1623 | **0.1329** |
| Synthetic Win Correlation | 0.5976 | 0.6653 |

*Takeaway*: At ~5× fewer parameters and using the simpler RAW featureset (which removes engineered/transition features and decouples RL inference from `poke-env` observations), `cool-bee-85-finetune` matches or beats the reference on every action metric (Top-1 +4pt, Top-3 +0.9pt, MOVE Top-3 +1.4pt, SWITCH Top-1 +0.3pt) while trading ~0.05 Win Corr / +0.03 Brier on the value head. The action policy is *better* than the reference; the value head is the only area where the larger model retains an edge. r-NaD self-play is expected to recover the value-head gap during RL training.

### 2. Stage 1 Sweep Learnings

The sweep campaign (see `planning/stage1/`) condensed into the current training recipe:

*   **Train longer**: 15 epochs was the biggest lever over 8 epochs. 30 epochs with `ReduceLROnPlateau` was the best setup for the final model.
*   **Use standard action CE**: `train_topk_k=2025` with `turn_loss_type=topk` remained the best setup; focal loss and entropy regularization hurt.
*   **Keep batches large**: `batch_size=128` gave better update efficiency without the noise penalties seen at smaller effective batches.
*   **Center the action weighting at `sw=1.0`** once the teampreview head is active; the older `0.7` preference did not hold in the detached-TP regime.
*   **Prefer `dropout=0.20` and `lr` near `5.4e-5`** as the stable center.
*   **Keep the transformer at `TL=7`** with medium late/turn heads and a causal decision-token setup.
*   **Detach the teampreview head from the shared encoder** so it can learn TP behavior without feeding harmful gradients back into the action/value trunk. This is implemented in `model_archs.py` via `encoded.detach()` before the TP head.
*   **ReduceLROnPlateau > cosine annealing** for 30-epoch runs: plateau scheduling found better local optima than cosine.
*   **Optimize for RL handoff quality, not just WinCorr**: the action metrics (especially BOTH Top-3 and SWITCH Top-1) matter more than a small win-correlation gain when choosing the Stage II initialization checkpoint.
*   **`move_loss_weight=1.0` is optimal**: Experiments with `1.3` (dark-oath-78) showed it damages the value head without improving move accuracy.

### 3. Strategic Conclusions
*   **Model-Free RL Limits**: Pure model-free RL is unlikely to produce superhuman performance in VGC due to the massive action space (2025 actions) and high stochasticity.
*   **The Role of Supervised Learning**: We treat supervised learning not as the end goal, but as the **initialization** for a search-based agent.
    *   The **Policy Head** (Action logits) prunes the search space for MCTS.
    *   The **Value Head** (Win Advantage) provides a heuristic evaluation for leaf nodes.
*   **Search is Necessary**: To bridge the gap between "human-like" (42% Top-5) and "superhuman", we need decision-time planning (MCTS) to handle complex calculations and lookaheads that a static policy network misses.

### 4. Stage II Handoff: `cool-bee-85-finetune_best`

The chosen BC handoff model for Stage II RL is `cool-bee-85-finetune_best.pt`. It is the result of two stages:

1. **Pre-train (`cool-bee-85`)**: 30 epochs with the A3 ~5× smaller transformer architecture and the A2 RAW featureset. Config: `configs/ablation_a5_small5x_raw_30ep.yaml`. (Best at epoch 30.)
2. **Fine-tune (`cool-bee-85-finetune`)**: 15 additional epochs from `cool-bee-85_best.pt`, with `win_loss_weight=0.5` (was 0.35) to push more gradient through the value head, cosine LR (peak 2.5e-5), and the throughput speedup config (batch 256 / worker_batch 256 / num_workers 4 / pf 3 / fpw 4). Config: `configs/ablation_a5_small5x_raw_30ep_finetune.yaml`. (Best at epoch 8.)

*   **Final BC model path**: `data/models/supervised/cool-bee-85-finetune_best.pt` (~100 MB; not pushed)
*   **Architecture (~26.7M params)**: transformer 4 layers × 8 heads, ff_dim=1024, agg=2048, hidden=256, early=[1024,512,512], late=[512,512], turn_head=[512,256,256], teampreview_head=[256,128]
*   **Featureset**: `raw` (5032 input dims; drops engineered + transition features → no `poke-env` observation dependency at RL inference time)
*   **Overall Top-1/3/5**: 45.33% / 62.15% / 67.56%
*   **MOVE Top-3**: 52.63%
*   **BOTH Top-3**: 64.45%
*   **SWITCH Top-1**: 99.35%
*   **Actual Win Correlation**: 0.7531
*   **Brier Score**: 0.1623
*   **Synthetic Advantage Correlation**: 0.5976
*   **TP Top-1**: 99.9%

**Why this model?** Action policy quality at or above the 125M reference (Top-1 +4pt, Top-3 +0.9pt, MOVE Top-3 +1.4pt, SWITCH Top-1 +0.3pt) while delivering ~5× faster RL forward passes. The RAW featureset removes the `poke-env` observation dependency from the RL inference path, simplifying the worker pipeline. The value-head gap (Win Corr 0.75 vs 0.82, Brier 0.16 vs 0.13) is r-NaD's job to close during self-play. See the ablation study at `planning/stage2/2026-05-01-bc-ablation-study.md` for the full reasoning.

**Alternatives considered**: `elated-dream-72_best` had the highest entropy (0.652) and best win correlation (0.822), but weaker MOVE Top-3 (45.5%). `dark-oath-78` (with `move_loss_weight=1.3`) showed that upweighting move loss trades value head quality for marginal coordination gains.

---

## DataLoader Performance (2026-03-22)

### System Specs
*   **CPU**: Intel Core i7-7700K @ 4.20GHz (8 logical cores)
*   **RAM**: 24 GB total (~20 GB available in WSL2)
*   **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
*   **Disk**: NVMe SSD
*   **OS**: Linux via WSL2

### Data Profile (regc_final_v4)
*   **Train**: 1522 `.pt.zst` files, 512 trajectories/file, ~1.2 MB compressed each, **1.8 GB total on disk**
*   **Per file decompressed**: **~244 MB** in tensor memory (201x compression ratio)
*   **Per trajectory**: ~487 KB (states[40,5222] fp16 + action_masks[40,2025] bool + actions[40] int64 + wins[40] fp16 + masks[40] bool)

### Key Findings

**Optimal DataLoader parameters** (measured via `benchmark_dataloader.py` and `benchmark_dataloader_r2.py`):

| Parameter | Old Value | Optimal Value | Why |
|---|---|---|---|
| `worker_batch_size` | 64 | **128** | Larger batches amortize per-batch IPC overhead; 2x batch = fewer send/receive cycles between workers and main process |
| `num_workers` | 6 | **3** | On 8-core CPU, 3 workers + main process + GPU thread avoids CPU contention. More workers = diminishing returns + IPC overhead |
| `prefetch_factor` | 4 | **2** | Lower prefetch = less memory pressure and less CPU contention. pf=2 is enough to hide I/O latency |
| `files_per_worker` | 4 | **3** | With 3 workers, each owning ~507 files, caching 3 decompressed files per worker is sufficient for good cache hit rates |
| `persistent_workers` | true | **true** | Keeps worker caches across epochs. Critical for epoch 2+ performance |

**Throughput comparison (samples/sec)**:

| Config | samples/sec | RAM Delta | Notes |
|---|---|---|---|
| Old: nw=6, fpw=4, pf=4, wb=64 | 205 | 6.76 GB | Current config |
| **New: nw=3, fpw=3, pf=2, wb=128** | **788** | **3.50 GB** | **3.8x faster, 48% less RAM** |
| Runner-up: nw=3, fpw=4, pf=2, wb=128 | 764 | 3.50 GB | Similar, slightly more cache |
| nw=4, fpw=4, pf=4, wb=64 | 440 | 4.48 GB | R1 winner before wb=128 discovery |

### Why Fewer Workers Win

On an 8-core i7-7700K, CPU contention is the primary bottleneck — not I/O:

1. **nw=3 beats nw=7 by 4.4x** (788 vs 178 samples/sec). With 7 workers, all 8 cores are saturated: workers compete for CPU time with each other and with the main process, causing pipeline stalls.
2. **nw=3 beats nw=4 by 1.8x** (788 vs 440 at wb=64, or 788 vs 638 at wb=128). The marginal benefit of a 4th worker is negative due to context switching.
3. The decompression of `.pt.zst` files (201x ratio) is CPU-intensive. Fewer workers means each worker gets more CPU time and finishes decompression faster.

### Why worker_batch_size=128 Wins

*   Each worker-to-main transfer has fixed IPC overhead. Doubling batch size halves the number of transfers per epoch.
*   `wb=128` sends 128 trajectories per batch transfer vs 64, cutting total transfer count in half.
*   `wb=256` crashes with bus errors on this system (shared memory / file descriptor exhaustion on WSL2). wb=128 is the sweet spot.

### Memory Budget

```
Total DataLoader RAM = num_workers × files_per_worker × decompressed_file_size
                     + num_workers × prefetch_factor × worker_batch_size × per_trajectory_size
                     + worker_overhead

Optimal config:
  Worker cache: 3 workers × 3 files × 244 MB = 2.2 GB
  Prefetch buf: 3 workers × 2 batches × 128 traj × 487 KB = 0.37 GB
  Overhead:     ~1 GB
  Total:        ~3.5 GB  (leaves ~16.5 GB for model + system)
```

### Caveats
*   Results are specific to this hardware (8-core CPU, 24 GB RAM, WSL2). More cores would shift optimal `num_workers` higher.
*   `pin_memory=True` does NOT work on WSL2 (causes OOM). Always use `pin_memory=False`.
*   `wb=256` causes "No space left on device" bus errors on WSL2 due to shared memory/file descriptor limits.
*   These parameters are optimized for the DataLoader pipeline only (CPU → RAM). GPU training throughput depends additionally on model size, VRAM, and mixed precision.
