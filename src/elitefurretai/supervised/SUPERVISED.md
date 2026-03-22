# Supervised Learning

This folder contains the code for training, fine-tuning, and evaluating supervised learning models for EliteFurretAI. These models are trained on millions of human battles to learn behavioral cloning (imitating high-Elo players).

## Workflow

### 1. Data Preparation
Before training, you must have processed training data ready.
1.  **Extract**: Raw Showdown logs.
2.  **Transform**: Run `src/elitefurretai/etl/process_training_data.py` to generate `.pt.zst` files.
3.  **Load**: The training scripts below will load these files using `BattleDataset`.

### 2. Training
The primary training script is `three_headed_transformer.py`. This trains the `FlexibleThreeHeadedModel` which predicts:
*   **Turn Actions**: What move/switch to make (2025 classes).
*   **Teampreview**: Which 4 Pokemon to bring and in what order (90 classes).
*   **Win Probability**: Who is likely to win (scalar).

```bash
# Example usage
python src/elitefurretai/supervised/three_headed_transformer.py \
    data/battles/processed_vgc_data \
    --name "my_experiment_v1" \
    --batch_size 512 \
    --epochs 10
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

#### `FlexibleThreeHeadedModel`
This is the primary architecture for the project, using a bidirectional LSTM backbone.
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

#### `TransformerThreeHeadedModel`
Alternative backbone that replaces the bidirectional LSTM with a TransformerEncoder and decision tokens.
*   **Decision tokens**: Three learned parameter vectors `[ACTOR]`, `[CRITIC]`, `[FIELD]` are prepended to the sequence. ACTOR token output feeds the turn head, CRITIC feeds the value head.
*   **Positional encoding**: Sinusoidal (supports variable-length sequences at inference)
*   **Causal mask**: Past turns can only attend to themselves and prior turns. Decision tokens can attend to everything.
*   **Hidden state**: Instead of LSTM `(h, c)`, uses a growing context tensor of past encoded features. Each turn appends to the context.
*   **Config**: Gated by `use_transformer` (disabled by default). Key params: `transformer_layers`, `transformer_heads`, `transformer_ff_dim`
*   **Same three heads**: Identical turn/teampreview/win head structure as `FlexibleThreeHeadedModel`

#### `GroupedFeatureEncoder`
Encodes features by semantic groups with optional number bank integration and cross-attention for Pokemon synergies.

#### `SinusoidalPositionalEncoding`
Standard sinusoidal positional encoding for the Transformer backbone. Supports variable-length sequences up to `max_len`.

### `three_headed_transformer.py`
**Purpose**: The main training loop for the transformer model.

**Key Features**:
*   **Mixed Precision**: Uses `torch.amp` for faster training on modern GPUs.
*   **Gradient Accumulation**: Allows training with large effective batch sizes even on limited VRAM.
*   **Loss Weighting**: Balances the three heads (Action, Teampreview, Win) to ensure one doesn't dominate the gradient.
*   **WandB Integration**: Logs metrics (loss, accuracy, top-k accuracy) to Weights & Biases.

### `fine_tune.py`
**Purpose**: Adapts a pre-trained model to new data.

**How it works**:
1.  Loads the model architecture and weights from a `.pt` checkpoint.
2.  Reads the embedded config to reconstruct the exact model structure.
3.  Optionally overrides training parameters (LR, batch size) from a config file.
4.  Freezes early layers (optional) to preserve feature extractors while adapting the decision heads.

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
The `FlexibleThreeHeadedModel` achieves the following accuracy on human ladder data (w/ ~130M parameters):

*   **Teampreview**:
    *   **Top-1**: 99.9%
    *   **Top-3**: 99.9%
    *   **Top-5**: 99.9%
    *   *Takeaway*: The dataset has little strategic variation; analysis shows that for a given team composition, 88.6% of the time, a player will choose the same teampreview choice. This means our model is just creating a look-up table. This model is overfit, but is the nature of this dataset.

*   **Turn Actions (Move Choice)**:
    *   **Top-1**: 28.9%
    *   **Top-3**: 43.5%
    *   **Top-5**: 53.8%
    *   *Takeaway*: Predicting the *exact* move a human makes is difficult due to playstyle variety and "rock-paper-scissors" scenarios. However, the Top-5 accuracy suggests the model consistently identifies the pool of reasonable moves.

*   **Win Advantage**:
    *   **Correlation**: 0.856
    *   **Accuracy w/ Win Prediction**: 68.7%
    *   **Brier Score**:  0.2026
    *   *Takeaway*: The win model is doing a decent job at predicting state advantage, better than the average state evaluation used by Foul Play.

*   **Unified Model**:
    *   **Win Correlation**: 0.726
    *   **Top 1/3/5/10 Move Accuracy**: 26%/40%/48%
    *   **Top 1/3/5 Teampreview Accuracy**:  79%/95%/99%
    *   *Takeaway*: This model is a single well-rounded model that balances all tasks. It doesn't overfit on teampreview and slightly underperforms against models that are trained on a single objective.

### 2. Strategic Conclusions
*   **Model-Free RL Limits**: Pure model-free RL is unlikely to produce superhuman performance in VGC due to the massive action space (2025 actions) and high stochasticity.
*   **The Role of Supervised Learning**: We treat supervised learning not as the end goal, but as the **initialization** for a search-based agent.
    *   The **Policy Head** (Action logits) prunes the search space for MCTS.
    *   The **Value Head** (Win Advantage) provides a heuristic evaluation for leaf nodes.
*   **Search is Necessary**: To bridge the gap between "human-like" (42% Top-5) and "superhuman", we need decision-time planning (MCTS) to handle complex calculations and lookaheads that a static policy network misses.
