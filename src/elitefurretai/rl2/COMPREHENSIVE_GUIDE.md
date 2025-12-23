# EliteFurretAI RL2: The Complete Guide

This document provides a comprehensive, in-depth explanation of the EliteFurretAI RL2 training system. It covers everything from the high-level architecture and core components to advanced features, optimization techniques, and the rationale behind the design choices.

## Table of Contents
1.  [**Architecture Overview**](#1-architecture-overview)
    -   The Actor-Learner Model
    -   Key Architectural Principles
2.  [**The RNaD Algorithm Explained**](#2-the-rnad-algorithm-explained)
    -   Why Regularized Nash Dynamics?
    -   The RNaD Loss Function
    -   Inspiration from Ataraxos (State-of-the-Art Stratego AI)
3.  [**Core Components**](#3-core-components)
    -   `agent.py`: The RL-Compatible Agent Wrapper
    -   `learner.py`: The Standard RNaD Learner
    -   `worker.py`: The High-Performance Battle Worker
    -   `config.py`: The Configuration System
    -   `team_manager.py`: The Team Pool System
4.  [**Training Workflow & Features**](#4-training-workflow--features)
    -   The Multi-Stage Training Process
    -   Configuration-Driven Training
    -   Resume Training from Checkpoints
    -   Automatic Exploiter Training
    -   Comprehensive Monitoring with Wandb
5.  [**Exploiter Training: The Adversarial Heartbeat**](#5-exploiter-training-the-adversarial-heartbeat)
    -   What is an Exploiter?
    -   The Exploiter Training Workflow
    -   Automating the Exploiter Pipeline
    -   The Opponent Pool & Adaptive Curriculum
6.  [**Advanced Features & Optimization**](#6-advanced-features--optimization)
    -   Portfolio Regularization: Preventing Strategy Collapse
    -   Mixed Precision Training: 2x Speed on Modern GPUs
    -   The Training Profiler: Maximizing Hardware Utilization
7.  [**Performance & Hardware Optimization**](#7-performance--hardware-optimization)
    -   Why `BatchInferencePlayer` is Fast: Async + Batching
    -   Expected Performance Benchmarks
    -   Troubleshooting Common Issues
8.  [**Quick Start Guide**](#8-quick-start-guide)
    -   Basic Usage & Commands
    -   Example Configurations

---

## 1. Architecture Overview

The system follows a **single-machine, multi-threaded Actor-Learner architecture**, specifically optimized for a setup with a high-end GPU (like an RTX 3090) and a multi-core CPU.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING COORDINATOR                     │
│                    (train_v2.py)                            │
└────────────┬────────────────────────────────────┬───────────┘
             │                                    │
             │                                    ▼
             │                       ┌────────────────────────┐
             │                       │    TRAJECTORY QUEUE    │
             │                       │    (asyncio.Queue)     │
             │                       └────────────────────────┘
             │                                    ▲
             ▼                                    │
┌─────────────────────────┐          ┌────────────────────────────┐
│     WORKER THREADS      │          │      LEARNER THREAD        │
│      (worker.py)        │          │ (learner.py/portfolio.py)  │
│                         │          │                            │
│ ┌─────────────────────┐ │          │ ┌────────────────────────┐ │
│ │ BatchInferencePlayer├─┼───────────>│       MAIN MODEL       │ │
│ └─────────────────────┘ │          │ │ (On GPU, Being Trained)│ │
│          │ ▲            │          │ └────────────┬───────────┘ │
│          ▼ |            │          │              │             │
│ ┌─────────────────────┐ │          │              ▼             │
│ │  Showdown Servers   │ │          │ ┌────────────────────────┐ │
│ │ (Local Subprocesses)│ │          │ |   REFERENCE MODEL(s)   │ │
│ └─────────────────────┘ │          │ |    (On GPU, Frozen)    │ │
└─────────────────────────┘          | └──────────────────---───┘ | 
                                     └────────────────────────────┘
```

### Key Architectural Principles

This design may seem complex, but it's built on a few core, corrected principles:

1.  **A Single, Shared Model on GPU**: The "GPU Learner" and "GPU Inference" are not separate entities. Both the Learner thread and all Worker threads access the **exact same model object** in GPU memory. The Learner computes gradients and updates its weights, while the Workers use it for inference (`torch.no_grad()`) to choose actions. Updates are visible to all threads instantly due to Python's shared memory model for threads.
2.  **Trajectories are the Currency**: Workers do **not** pass gradients to the Learner. They collect battle experience—`(state, action, reward, log_prob, value)` tuples—and put these "trajectories" onto a queue. The Learner is the only component that computes gradients, using these trajectories as training data.
3.  **Bidirectional Communication with the Simulator**: The `BatchInferencePlayer` sends actions to the Pokémon Showdown server, which then sends back battle state updates via a WebSocket. This loop continues until the battle is over.
4.  **Efficiency Through Batching**: Each `BatchInferencePlayer` manages multiple concurrent battles (e.g., 4-8). It collects observations from all of them and batches them into a single tensor for one efficient GPU call, dramatically increasing throughput. This is explained in detail in the [Performance Section](#7-performance--hardware-optimization).

---

## 2. The RNaD Algorithm Explained

### Why Regularized Nash Dynamics?

Standard Reinforcement Learning can be unstable in complex games like Pokémon. An agent might discover a simple, exploitative strategy (e.g., "always use Protect on turn 1"), over-optimize for it, and completely forget the more general, robust strategies it learned from previous phases (e.g. Behavior Cloning (BC) phase). This is known as **catastrophic forgetting**.

**Regularized Nash Dynamics (RNaD)** is an algorithm designed to combat this. It forces the learning agent (the "Main Model") to stay "close" to a stable, trusted policy (the "Reference Model").

The Reference Model is a frozen snapshot of the Main Model from a few thousand steps ago. It acts as an anchor, preventing the Main Model from drifting too far, too fast. This ensures:
-   **Stability**: Training is smoother and less prone to sudden collapses in performance.
-   **Retention of Priors**: The valuable, human-like strategies learned during the initial BC phase are not easily discarded.
-   **Robustness**: The agent is encouraged to find improvements that are good in a general sense, not just against its current self.

### The RNaD Loss Function

The total loss that the `RNaDLearner` minimizes is a combination of four components:

$$L_{total} = L_{policy} + \beta \cdot L_{value} - \gamma \cdot H + \alpha \cdot L_{RNaD}$$

-   **$L_{policy}$ (PPO Policy Loss)**: The standard RL loss. It increases the probability of actions that led to better-than-expected outcomes (high advantage).
-   **$L_{value}$ (Value Loss)**: Trains the value head of the model to accurately predict the probability of winning from a given state.
-   **$H$ (Entropy Bonus)**: Encourages exploration by rewarding the agent for having a less "peaked" (more uncertain) action distribution. This helps it discover new strategies.
-   **$L_{RNaD}$ (KL Divergence Loss)**: This is the core of RNaD. It measures the "difference" between the action probabilities of the Main Model and the Reference Model. If the Main Model tries to change its strategy too drastically, this term becomes large, penalizing the update.

**How KL Divergence Works in Practice:**
```python
# For each state in the batch:
ref_probs = softmax(ref_model(state))  # What would the old model do?
curr_probs = softmax(main_model(state))  # What does the current model want to do?

# KL divergence measures "how different" these two distributions are
kl_divergence = sum(curr_probs * (log(curr_probs) - log(ref_probs)))

# If curr_probs is similar to ref_probs, kl_divergence is close to 0 (no penalty)
# If they're very different, kl_divergence is high (large penalty)
```

This regularization term effectively says: *"You are free to improve, but do not stray too far from the sensible strategies you already know."*

### Inspiration from Ataraxos (State-of-the-Art Stratego AI)

Many design choices in this project are inspired by DeepMind's Ataraxos, the AI that achieved superhuman performance in Stratego, a game with significant hidden information like Pokémon.

-   **Separate Heads for Setup**: Ataraxos uses a separate model head for the initial piece setup phase. We mirror this by having a dedicated **teampreview head** (90 actions) and a **turn action head** (2025 actions).
-   **Belief States**: Ataraxos models a probability distribution over hidden opponent pieces. Our `Embedder` with `omniscient=False` and `BattleInference` module are steps in this direction, attempting to infer hidden information like opponent items and speed tiers.
-   **Regularized Policy Improvement**: Ataraxos uses R-NaD, validating our choice of algorithm. A key insight we've adopted is using a **portfolio** of reference models, which is detailed in the [Advanced Features](#6-advanced-features--optimization) section.

---

## 3. Core Components

### `agent.py`: The RL-Compatible Agent Wrapper

-   **What it is**: `RNaDAgent` is a thin wrapper around the `FlexibleThreeHeadedModel` that was trained via Behavior Cloning.
-   **How it fits in**: The BC model is designed for supervised learning on entire battle trajectories. However, in RL, the agent must make decisions one step at a time.
-   **Why it's implemented this way**: This wrapper solves the "impedance mismatch" between the BC model and the RL loop. It manages the LSTM's hidden state from one turn to the next, providing a clean `forward(state, hidden_state)` interface that the RL worker can call on each turn of the battle. This avoids modifying the original, validated BC model code.

**Key Methods and Logic**:
```python
# In RNaDAgent:

def get_initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a zero-initialized hidden state for the LSTM."""
    return self.model.get_initial_state(batch_size, device)

def forward(self, x: Dict[str, torch.Tensor], hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Runs a single forward pass for the current turn.
    `x` contains the features for the current state (batch=1, seq_len=1).
    `hidden` is the LSTM state from the previous turn.
    """
    # The BC model expects a sequence, so we unsqueeze the time dimension
    turn_features = x["turn_features"].unsqueeze(1)
    
    # Forward pass through the underlying model
    turn_logits, tp_logits, value, next_hidden = self.model.forward_for_rl(
        turn_features,
        x["teampreview_features"],
        x["is_teampreview"],
        hidden,
        x["action_mask"],
    )
    
    return turn_logits, tp_logits, value, next_hidden
```
This shows how the agent takes a single step's features and the previous hidden state, formats it for the sequence-based BC model, and returns the action logits, value, and the *next* hidden state, ready for the next turn.

### `learner.py`: The Standard RNaD Learner

-   **What it is**: `RNaDLearner` is the heart of the training process. It holds both the `main_model` (which is learning) and the `ref_model` (which is frozen).
-   **How it fits in**: It continuously pulls completed battle trajectories from the `Trajectory Queue`, computes the RNaD loss function across these batches of data, and performs backpropagation to update the `main_model`'s weights.
-   **Why it's implemented this way**: It isolates all the learning logic—loss calculation, optimization, and model updates—into a single class. Every `checkpoint_interval`, it updates the `ref_model` by copying the `main_model`'s weights, advancing the regularization anchor.

### `worker.py`: The High-Performance Battle Worker

-   **What it is**: The `BatchInferencePlayer` is a highly optimized `poke-env` player responsible for battling, collecting data, and putting it on the queue for the learner.
-   **How it fits in**: Multiple workers run in parallel threads. Each worker instantiates a `BatchInferencePlayer` which, in turn, manages multiple concurrent battles against opponents drawn from the `OpponentPool`.
-   **Why it's implemented this way**: Performance. Instead of making one GPU call per action for each battle, it batches observations from all its concurrent battles into a single tensor. This maximizes GPU utilization and is the single most important optimization for data collection throughput. The performance gains are detailed in [Section 7](#7-performance--hardware-optimization).

### `config.py`: The Configuration System

-   **What it is**: A `RNaDConfig` dataclass that holds all hyperparameters for training, hardware, paths, exploiters, and logging.
-   **How it fits in**: The main training script, `train_v2.py`, is driven by a single config file. This makes experiments reproducible and easy to manage.
-   **Why it's implemented this way**: It centralizes all tunable parameters, moving them out of hardcoded script values. This allows for easy saving, loading, and modification of entire training setups without changing the code.

### `team_manager.py`: The Team Pool System

-   **What it is**: The `TeamPool` class loads a directory of Pokémon teams (in PokePaste `.txt` format) and provides them to workers.
-   **How it fits in**: If a `team_pool_path` is specified in the config, each worker will sample a random team for every battle.
-   **Why it's implemented this way**: It prevents the agent from overfitting to a single team composition. By training on a wide variety of teams, the agent learns more general strategies that are not dependent on a specific set of Pokémon.

---

## 4. Training Workflow & Features

### The Multi-Stage Training Process

1.  **Behavioral Cloning (BC)**: The model is first pre-trained on a large dataset of human battles (e.g., 1M+ games). This provides a strong, sensible starting point. This is done via the `/supervised` folder.
2.  **RL Finetuning**: The BC model is then loaded into the RL system and finetuned using the RNaD algorithm described above.

### Configuration-Driven Training

The entire system is managed through YAML config files.

```bash
# 1. Create a default config file
python -c "from elitefurretai.rl2.config import RNaDConfig; RNaDConfig().save('my_config.yaml')"

# 2. Edit my_config.yaml to set parameters

# 3. Launch training
python src/elitefurretai/rl2/train_v2.py --config my_config.yaml
```

### Resume Training from Checkpoints

The training script saves comprehensive checkpoints that include the model weights, optimizer state, update count, and the full configuration. Training can be seamlessly resumed.

```bash
# Resume from the most recent checkpoint in the save directory
python src/elitefurretai/rl2/train_v2.py --resume

# Resume from a specific checkpoint
python src/elitefurretai/rl2/train_v2.py --checkpoint path/to/checkpoint.pt
```

### Automatic Exploiter Training

As detailed in the next section, the training loop can automatically train and integrate exploiters. This is enabled via the config:

```yaml
# In config.yaml
train_exploiters: true
exploiter_check_interval: 5000    # Train a new exploiter every 5000 updates
exploiter_train_steps: 50000      # Train each exploiter for 50k steps
exploiter_win_threshold: 0.6      # Win rate needed to be added to the pool
```

### Comprehensive Monitoring with Wandb

When enabled (`use_wandb: true`), the system logs a rich set of metrics to Weights & Biases, crucial for understanding agent performance:
-   **Loss Components**: `policy_loss`, `value_loss`, `entropy`, `rnad_loss`.
-   **Win Rates**: `win_rate/self`, `win_rate/bc`, `win_rate/exploiter`, `win_rate/past`.
-   **Curriculum Weights**: The sampling probability for each opponent type.
-   **Advanced Metrics**: Portfolio selection counts and KL stats (if using portfolio).

---

## 5. Exploiter Training

Self-play alone is not enough. An agent that only ever plays against itself can develop blind spots and brittle strategies. Exploiters are the solution.

### What is an Exploiter?

An **exploiter** is a specialized agent trained with a single, ruthless goal: **to beat a specific, frozen version of the main model**. It is trained with:
-   **No RNaD Regularization**: It is free to find any winning strategy, no matter how weird or "un-human".
-   **A Fixed Opponent**: It only ever plays against one "victim" model.

This process is designed to uncover weaknesses. For example, if the main model has a tendency to always switch out a burned Pokémon, an exploiter will learn to predict this switch and punish it, a pattern the main model would never discover by playing against itself.

### The Exploiter Training Workflow

The process is a continuous cycle that runs in the background of main training:

1.  **Freeze Victim**: At a regular interval (e.g., every 5,000 updates), the current main model is saved as a "victim".
2.  **Train Exploiter**: A new agent is trained from scratch (or from the BC model) against this frozen victim. PPO is used, but with `rnad_alpha = 0`.
3.  **Evaluate**: The trained exploiter is benchmarked against its victim.
4.  **Register**: If the exploiter achieves a minimum win rate (e.g., >60%), its model path is added to the `exploiter_registry.json`.
5.  **Integrate**: The `OpponentPool` reads this registry, and the new exploiter is immediately added to the pool of possible opponents for the main agent to face.

The main agent is then forced to learn a defense against this new, highly specific threat, patching its weakness and becoming more robust.

### The Opponent Pool & Adaptive Curriculum

Workers don't just play against the current main model. They sample opponents from an `OpponentPool` according to a curriculum:
-   **Self**: The most recent main model.
-   **BC**: The original, human-like behavior-cloned model.
-   **Past**: Past versions (checkpoints) of the main model.
-   **Exploiters**: The adversarial agents from the registry.

The system tracks win rates against each category and can adapt the sampling probabilities, forcing the agent to spend more time training against opponents it is struggling with.

---

## 6. Advanced Features & Optimization

To push performance further, several advanced features have been implemented, inspired by state-of-the-art research.

### Portfolio Regularization: Preventing Strategy Collapse

-   **What it is**: An evolution of RNaD. Instead of one reference model, we maintain a **portfolio** of 3-5 diverse, high-performing past models.
-   **How it works**: In each training step, the learner calculates the KL divergence against *all* models in the portfolio and **regularizes against the closest one** (`min(KL_losses)`).

    ```python
    # Traditional RNaD (single reference)
    kl_loss = KL(current_policy || reference_policy)

    # Portfolio RNaD (multiple references)
    kl_losses = [KL(current_policy || ref_i) for ref_i in portfolio]
    kl_loss = min(kl_losses)  # Regularize to the CLOSEST reference
    ```

-   **Why it helps**: A single reference model can still "forget" older, but still valid, strategies (e.g., forgetting how to play against a specific archetype if it hasn't seen it recently). A portfolio forces the agent to remain competent against a wider range of playstyles, preventing strategy collapse and increasing robustness.
-   **Implementation**: A separate `PortfolioRNaDLearner` is used when `use_portfolio_regularization: true`. It manages adding, removing, and selecting from the portfolio based on strategies like "diverse" or "best".

    **Configuration**:
    ```yaml
    # In config.yaml
    use_portfolio_regularization: true
    max_portfolio_size: 5              # Keep 5 reference models
    portfolio_update_strategy: "diverse"  # "diverse", "best", or "recent"
    portfolio_add_interval: 5000       # Add a new candidate every 5k updates
    ```

### Mixed Precision Training: 2x Speed on Modern GPUs

-   **What it is**: Using FP16 (half-precision) floating-point numbers for model computations instead of FP32 (full-precision).
-   **How it works**: PyTorch's `torch.cuda.amp` (Automatic Mixed Precision) automatically casts model weights and operations to FP16 where safe, using special hardware units called Tensor Cores on modern NVIDIA GPUs (RTX 20xx and newer).

    ```python
    # Traditional FP32
    output = model(input)
    loss.backward()

    # Mixed Precision FP16 (in the learner's update step)
    with torch.cuda.amp.autocast():
        output = model(input)  # Model forward pass runs in FP16
        loss = criterion(output, target)

    # Scales loss to prevent gradients from vanishing
    scaler.scale(loss).backward()
    # Unscales gradients and updates FP32 weights
    scaler.step(optimizer)
    scaler.update()
    ```

-   **Why it helps**: This provides a nearly **2x speedup** in training and reduces VRAM usage by ~30%, allowing for larger batch sizes. It is a "free" performance boost on compatible hardware.
-   **Implementation**: The `RNaDLearner` and `PortfolioRNaDLearner` wrap the forward and backward passes in `autocast` and use a `GradScaler` to prevent numerical underflow, ensuring training stability. This is enabled by default.

    **Configuration**:
    ```yaml
    # In config.yaml
    use_mixed_precision: true  # Enable FP16 (default: true)
    gradient_clip: 0.5         # Recommended for stability with FP16
    ```

### The Training Profiler: Maximizing Hardware Utilization

-   **What it is**: A powerful diagnostic tool (`profiler.py`) that analyzes the entire training pipeline to find the optimal hyperparameters for your specific hardware.
-   **How it works**: It runs short tests to measure data collection throughput, inference speed, and training update speed while monitoring CPU, GPU, and RAM usage. It can run a full sweep over different numbers of workers and batch sizes.
-   **Why it helps**: It answers critical questions like: "What is my bottleneck: CPU, GPU, or network?" and "What are the best batch sizes and worker counts for my machine?". Running the profiler before a long training run ensures you are getting the most out of your hardware.
-   **Usage**:
    ```bash
    # Run a full sweep to find the best possible configuration
    python src/elitefurretai/rl2/profiler.py --sweep --output results.json
    ```
    The output provides recommended settings for your `config.yaml`.

---

## 7. Performance & Hardware Optimization

### Why `BatchInferencePlayer` is Fast: Async + Batching

A common bottleneck in RL is the time spent waiting for the GPU. A naive implementation would process one action at a time, leaving the GPU idle for most of the duration.

-   **Synchronous (Slow)**:
    -   Battle 1 needs action -> Call GPU (2ms) -> Wait for network (50ms) -> GPU is idle.
    -   Total time for 16 battles = 16 * (2ms + 50ms) = **832ms**. GPU utilization is ~2.4%.

-   **Asynchronous + Batched (Fast)**:
    -   All 16 battles request an action at roughly the same time.
    -   The `BatchInferencePlayer` collects these 16 observations into a single batch.
    -   One large, efficient GPU call is made on the batch (e.g., 15ms).
    -   While the GPU is working, the CPU and network are processing other things.
    -   Total time for 16 battles = **~25ms**. GPU utilization is ~60%.

This represents an **8-10x speedup** in data collection and is the most critical optimization in the `worker.py` implementation.

### Expected Performance Benchmarks

On an RTX 3090, 8-core CPU, and 32GB RAM:

| Configuration              | Updates/Hr | Time to 100k Updates | GPU Util % | Notes                   |
| -------------------------- | ---------- | -------------------- | ---------- | ----------------------- |
| Baseline (FP32, single ref) | ~2000      | ~50 hours            | ~60%       | Original implementation   |
| + Mixed Precision          | ~3800      | ~26 hours            | ~75%       | **Nearly 2x faster!**     |
| + Portfolio (no MP)        | ~2000      | ~50 hours            | ~60%       | More robust, same speed   |
| + Both Features            | ~3800      | ~26 hours            | ~75%       | **Fast AND robust!**      |
| + Profiler-Optimized       | **~4200**  | **~24 hours**        | **~85%**   | **Optimal utilization**   |

### Troubleshooting Common Issues

-   **Out of Memory (OOM)**: Reduce `train_batch_size` and `players_per_worker`.
-   **Slow Training**: Your bottleneck is likely CPU or network. Run the profiler. You may need to reduce `num_workers` if CPU-bound.
-   **Loss -> NaN**: Usually happens with mixed precision. Try increasing `gradient_clip` (e.g., to `1.0`) or, as a last resort, disable `use_mixed_precision`.
-   **Model Overfits**: The agent is winning too easily. Enable `train_exploiters` and ensure your `team_pool_path` is populated with diverse teams.

---

## 8. Quick Start Guide

### Basic Usage & Commands

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Launch Showdown Servers**:
    ```bash
    # Launches 4 Showdown servers on ports 8000-8003
    python src/elitefurretai/rl2/launch_servers.py --num-servers 6
    ```
3.  **Create a Config**:
    ```bash
    python -c "from elitefurretai.rl2.config import RNaDConfig; RNaDConfig().save('my_config.yaml')"
    ```
4.  **Edit `my_config.yaml`**: Set `checkpoint_path` to your BC model and configure other settings.
5.  **Train**:
    ```bash
    python src/elitefurretai/rl2/train_v2.py --config my_config.yaml
    ```

### Example Configurations

#### Minimal Config (Fast Testing)
```yaml
# fast_test.yaml
checkpoint_path: "data/models/bc_action_model.pt"
num_workers: 2
players_per_worker: 2
train_batch_size: 16
max_updates: 10000
use_wandb: false
```

#### Production Config (Full Training with Advanced Features)
```yaml
# production.yaml
# Model
checkpoint_path: "data/models/bc_action_model.pt"
team_pool_path: "data/teams/gen9vgc2023regulationc"

# Training
learning_rate: 0.0001
rnad_alpha: 0.01
num_workers: 4            # Use profiler to tune
players_per_worker: 6     # Use profiler to tune
train_batch_size: 48      # Use profiler to tune

# Advanced Features
use_mixed_precision: true
use_portfolio_regularization: true
max_portfolio_size: 5
portfolio_add_interval: 5000

# Checkpointing
checkpoint_interval: 1000
ref_update_interval: 1000
save_dir: "data/models/production"

# Exploiters
train_exploiters: true
exploiter_check_interval: 5000
exploiter_train_steps: 50000

# Wandb
use_wandb: true
wandb_project: "elitefurretai-production"
wandb_run_name: "rnad_advanced_v1"
```
