# RL Training System

This document is the **comprehensive, one-stop guide** to EliteFurretAI's reinforcement learning training system. It covers architecture, algorithms, optimization history, benchmarks, and practical guidance for running and extending the system.

## Table of Contents

1.  [**Hardware & Environment**](#1-hardware--environment)
2.  [**Architecture Overview**](#2-architecture-overview)
    -   IMPALA-Style Multiprocessing
    -   Architectural Principles
3.  [**RNaD Algorithm Overview**](#3-rnad-algorithm-overview)
    -   Why Regularized Nash Dynamics?
    -   The RNaD Loss Function
    -   Inspiration from Ataraxos
4.  [**Core Components and Files**](#4-core-components)
    -   `agent.py`: The RL-Compatible Agent Wrapper
    -   `learner.py`: The Standard RNaD Learner
    -   `multiprocess_actor.py`: Actor Processes and Trainer
    -   `config.py`: The Configuration System
    -   `fast_action_mask.py`: Optimized Action Masking
    -   `utils/team_repo.py`: The Team Repository System
5.  [**Training Workflow & Features**](#5-training-workflow--features)
    -   The Multi-Stage Training Process
    -   Configuration-Driven Training
    -   Resume Training from Checkpoints
    -   Automatic Exploiter Training
    -   Comprehensive Monitoring with WandB
6.  [**Exploiter Training Details**](#6-exploiter-training-details)
    -   What is an Exploiter?
    -   Design Decision: Single-Team Exploiters
    -   The Exploiter Training Workflow
    -   The Opponent Pool & Adaptive Curriculum
7.  [**Performance & Optimization**](#7-performance--optimization)
    -   Understanding the Bottlenecks
    -   Fast Action Masking (52,000x speedup)
    -   Embedder Move Caching (2.75x speedup)
    -   Mixed Precision Training (2x speedup)
    -   Why Multiprocessing? GIL Limitations
    -   Multi-Server Showdown Architecture
8.  [**Scaling Experiments & Benchmarks**](#8-scaling-experiments--benchmarks)
    -   Baseline Measurements
    -   Multi-Server Scaling Results
    -   Hardware Stress Testing
    -   Memory Requirements
    -   Optimal Configurations
9.  [**Advanced Features**](#9-advanced-features)
    -   Portfolio Regularization
    -   The Training Profiler
10. [**Quick Start Guide**](#10-quick-start-guide)
    -   Basic Usage & Commands
    -   Example Configurations
11. [**Implementation Notes & Bug Fixes**](#11-implementation-notes--bug-fixes)
    -   Critical Bug Fixes
    -   OTS Deadlock Fix
    -   Known Issues & Workarounds
12. [**Design Philosophy & Key Takeaways**](#12-design-philosophy--key-takeaways)
    -   Core Principles
    -   Lessons Learned
    -   Future Directions

---

## 1. Hardware & Environment

### Reference Hardware
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **CPU**: 8 cores
- **RAM**: 32GB (24GB available in WSL2)
- **Storage**: 2TB NVMe SSD
- **OS**: Linux via WSL2

### Model Specifications
- **Model**: `FlexibleThreeHeadedModel` (LSTM backbone) or `TransformerThreeHeadedModel` (Transformer backbone) with FULL embedder
- **Parameters**: ~138.8M (LSTM variant)
- **Weights Size**: ~529 MB
- **Embedding Dimensions**: 9,223
- **Value Head**: C51 distributional (51 bins over [-1, 1]) — richer gradients than scalar MSE
- **Action Space**: 2,025 turn actions + 90 teampreview actions

### Critical WSL2 Notes
```python
# REQUIRED at start of any training script
torch.multiprocessing.set_sharing_strategy('file_system')  # Required for WSL

# DataLoader settings
pin_memory=False  # MUST be False on WSL2 - causes OOM otherwise
```

---

## 2. Architecture Overview

The system uses an **IMPALA-style multiprocessing architecture** with separate Python processes for actors and learner. This design bypasses Python's GIL limitation to achieve maximum throughput.

### IMPALA-Style Multiprocessing

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LEARNER PROCESS (GPU)                        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │  Main Model     │    │ Reference Model │    │ Optimizer       │  │
│  │  (GPU, FP32)    │    │ (GPU, FP32)     │    │ (Adam states)   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Trajectory Queue                          │   │
│  │              (receives from all actors)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           ▲                     ▲                     ▲           │
└───────────┼─────────────────────┼─────────────────────┼───────────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │  ACTOR 0      │     │  ACTOR 1      │     │  ACTOR 2      │
    │  (CPU only)   │     │  (CPU only)   │     │  (CPU only)   │
    │               │     │               │     │               │
    │ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
    │ │Model Copy │ │     │ │Model Copy │ │     │ │Model Copy │ │
    │ └───────────┘ │     │ └───────────┘ │     │ └───────────┘ │
    │       ↓       │     │       ↓       │     │       ↓       │
    │ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
    │ │ Embedder  │ │     │ │ Embedder  │ │     │ │ Embedder  │ │
    │ └───────────┘ │     │ └───────────┘ │     │ └───────────┘ │
    │       ↓       │     │       ↓       │     │       ↓       │
    │ ┌───────────┐ │     │ ┌───────────┐ │     │ ┌───────────┐ │
    │ │ Showdown  │ │     │ │ Showdown  │ │     │ │ Showdown  │ │
    │ │ :8000     │ │     │ │ :8001     │ │     │ │ :8002     │ │
    │ └───────────┘ │     │ └───────────┘ │     │ └───────────┘ │
    └───────────────┘     └───────────────┘     └───────────────┘
```

**Key Characteristics:**
- Each actor is a **separate Python process** with its own model copy
- Actors use **CPU inference** (GPU reserved for learner)
- **Bypasses GIL** for true parallelism and 2-3x throughput improvement
- Learner periodically **broadcasts updated weights** to all actors
- Actors send completed trajectories via `multiprocessing.Queue`

### Why Multiprocessing Over Threading?

Python's Global Interpreter Lock (GIL) prevents true parallel execution across threads:

```
Thread 1: [===RUN===][--wait--][===RUN===][--wait--]
Thread 2: [--wait--][===RUN===][--wait--][===RUN===]
          ↑ Only one thread runs Python bytecode at any moment
```

With multiprocessing, each actor has its own Python interpreter, enabling true parallelism:

```
Actor 0: [===RUN===][===RUN===][===RUN===][===RUN===]
Actor 1: [===RUN===][===RUN===][===RUN===][===RUN===]
Actor 2: [===RUN===][===RUN===][===RUN===][===RUN===]
          ↑ All processes run simultaneously
```

### Architectural Principles

1.  **Trajectories are the Currency**: Actors collect `(state, action, reward, log_prob, value)` tuples and send them to the learner via queue. The Learner computes gradients from these trajectories.
2.  **Bidirectional Communication**: Actors send actions to Pokémon Showdown and receive state updates via WebSocket.
3.  **CPU Actors, GPU Learner**: Actors run inference on CPU (fast enough for individual battles), freeing the GPU entirely for gradient computation.
4.  **Periodic Weight Sync**: Learner broadcasts updated weights every N trajectories, keeping actors reasonably up-to-date without constant synchronization overhead.

---

## 3. RNaD Algorithm Overview

### Why Regularized Nash Dynamics?

Standard RL can be unstable in games like Pokémon. An agent might discover a simple exploitative strategy, over-optimize for it, and forget robust strategies learned from Behavioral Cloning.

**Regularized Nash Dynamics (RNaD)** forces the learning agent to stay "close" to a stable reference policy, preventing catastrophic forgetting and ensuring:
- **Stability**: Smoother training, less prone to sudden collapses
- **Retention of Priors**: Human-like strategies from BC are preserved
- **Robustness**: Agent finds improvements that generalize well

### The RNaD Loss Function

$$L_{total} = L_{policy} + \beta \cdot L_{value} - \gamma \cdot H + \alpha \cdot L_{RNaD}$$

| Component | Description |
|-----------|-------------|
| $L_{policy}$ | PPO policy loss - increases probability of high-advantage actions |
| $L_{value}$ | Value loss - trains value head to predict win probability |
| $H$ | Entropy bonus - encourages exploration |
| $L_{RNaD}$ | **KL divergence** from reference model - the core of RNaD |

**KL Divergence in Practice:**
```python
ref_probs = softmax(ref_model(state))   # What would the old model do?
curr_probs = softmax(main_model(state)) # What does current model want?

# KL measures "how different" these distributions are
kl_divergence = sum(curr_probs * (log(curr_probs) - log(ref_probs)))

# If similar: kl ≈ 0 (no penalty)
# If very different: kl is large (big penalty)
```

This says: *"You can improve, but don't stray too far from what you already know."*

### Inspiration from Ataraxos

Our design is inspired by DeepMind's Ataraxos (superhuman Stratego AI):
- **Separate heads** for setup (teampreview: 90 actions) and gameplay (turns: 2,025 actions)
- **Belief states** via `BattleInference` module for hidden information
- **Portfolio regularization** - multiple reference models instead of one

---

## 4. Core Components

### `agent.py`: The RL-Compatible Agent Wrapper

**Purpose**: Wraps the BC-trained `FlexibleThreeHeadedModel` or `TransformerThreeHeadedModel` for step-by-step RL inference.

The BC model expects full trajectories, but RL requires one decision at a time. `RNaDAgent` manages hidden state between turns:
- **LSTM variant**: Manages `(h, c)` hidden state tuple
- **Transformer variant**: Manages a growing context tensor (previous turns' encoded features); returns `None` for initial state

```python
class RNaDAgent(torch.nn.Module):
    def __init__(self, model: Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel]):
        self._is_transformer = isinstance(model, TransformerThreeHeadedModel)
    
    def get_initial_state(self, batch_size, device):
        if self._is_transformer:
            return None  # Context starts empty
        return (h_zeros, c_zeros)  # LSTM hidden state
```

### `learner.py`: The Standard RNaD Learner

**Purpose**: Heart of training. Holds `main_model` (learning) and `ref_model` (frozen anchor).

- Pulls trajectories from queue
- Computes RNaD loss (PPO policy + distributional value + entropy + KL)
- Uses **C51 distributional value loss** (cross-entropy against two-hot encoded targets) instead of scalar MSE
- Updates `main_model` via backpropagation with **topology-aware optimizer** (AdamW with separate param groups for backbone vs heads)
- Applies **LR scheduler** (linear warmup + cosine/linear decay)
- Periodically copies weights to `ref_model`
- Broadcasts updated weights to actors

### `multiprocess_actor.py`: Actor Processes and Trainer

**Purpose**: IMPALA-style multiprocessing architecture for high-throughput training.

**Key Classes:**

#### `ActorConfig`
Configuration dataclass for actor processes:
```python
@dataclass
class ActorConfig:
    actor_id: int           # Unique identifier
    server_port: int        # Showdown server port
    model_path: str         # Path to model checkpoint
    model_config: Dict      # Model architecture config
    battle_format: str      # e.g., "gen9vgc2023regc"
    num_battles: int        # Battles before sending trajectory
    device: str = "cpu"     # Always CPU for actors
    probabilistic: bool = True  # Sample from policy vs argmax
```

#### `Trajectory`
Data container sent from actor to learner:
```python
@dataclass
class Trajectory:
    states: np.ndarray        # (T, state_dim) - embedded observations
    actions: np.ndarray       # (T,) - action indices taken
    rewards: np.ndarray       # (T,) - per-step rewards
    action_masks: np.ndarray  # (T, action_dim) - valid action masks
    is_teampreview: np.ndarray  # (T,) - teampreview vs turn steps
    values: np.ndarray        # (T,) - value predictions for GAE
    win: float                # 1.0=win, 0.0=loss, 0.5=draw
```

#### `ActorPlayer`
A `Player` subclass that runs in an actor process:
- Performs CPU inference using its local model copy
- Accumulates trajectory data during battles
- Sends completed trajectories to the learner's queue
- Supports both deterministic (argmax) and probabilistic (sampling) action selection

#### `MultiprocessingTrainer`
The coordinator class that:
- Spawns actor processes using `mp.Process`
- Manages the trajectory queue
- Broadcasts updated weights to actors at configurable intervals
- Collects and batches trajectories for the learner

### `config.py`: The Configuration System

**Purpose**: `RNaDConfig` dataclass holding all hyperparameters.

Centralizes all tunable parameters in YAML files for reproducibility. Includes:

- **Exploration**: `temperature_start/end`, `temperature_anneal_steps`, `top_p`, `ent_coef_end`
- **Optimizer**: `optimizer` dict with `type` (adam/adamw), `weight_decay`, `lr_backbone`, `lr_heads`, `lr_warmup_steps`, `lr_schedule`
- **Distributional Value**: `num_value_bins`, `value_min`, `value_max`
- **Number Banks**: `use_number_banks`, `number_bank_embedding_dim`, `number_bank_hp/stat/power_bins`
- **Transformer**: `use_transformer`, `transformer_layers/heads/ff_dim/dropout`, `use_decision_tokens`, `use_causal_mask`

### `fast_action_mask.py`: Optimized Action Masking

**Purpose**: Fast generation of valid action masks (52,000x faster than naive approach).

Instead of iterating over all 2,025 actions, directly enumerates valid actions from `battle.last_request`:
```python
def fast_get_action_mask(battle: DoubleBattle) -> np.ndarray:
    """Generate action mask in O(moves × targets) instead of O(2025)."""
    # Parse request JSON directly
    # Build valid action sets
    # Return boolean mask
```

### `utils/team_repo.py`: The Team Repository System

**Purpose**: Manages Pokémon teams in PokePaste format.

```python
team_repo.sample_team("gen9vgc2023regc")  # Random team
team_repo.sample_n_teams(format, "trickroom")  # Teams from subfolder
team_repo.save_team(team, format, path, name)  # Save new team
```

---

## 5. Training Workflow & Features

### The Multi-Stage Training Process

1. **Behavioral Cloning (BC)**: Pre-train on 1M+ human battles (see `/supervised`)
2. **RL Finetuning**: Load BC model, finetune with RNaD using multiprocessing actors

### Configuration-Driven Training

```bash
# 1. Create default config
python src/elitefurretai/rl/config.py my_config.yaml

# 2. Edit my_config.yaml

# 3. Train
python src/elitefurretai/rl/train.py --config my_config.yaml
```

### Resume Training from Checkpoints

```bash
# Resume from most recent
python src/elitefurretai/rl/train.py --resume

# Resume from specific checkpoint
python src/elitefurretai/rl/train.py --checkpoint path/to/checkpoint.pt
```

### Automatic Exploiter Training

```yaml
train_exploiters: true
exploiter_check_interval: 5000    # Every 5k updates
exploiter_train_steps: 50000      # 50k steps per exploiter
exploiter_win_threshold: 0.6      # 60% win rate to join pool
```

### Comprehensive Monitoring with WandB

Logged metrics include:
- **Loss Components**: `policy_loss`, `value_loss`, `entropy`, `rnad_loss`
- **Win Rates**: `win_rate/self`, `win_rate/bc`, `win_rate/exploiter`, `win_rate/past`
- **Curriculum Weights**: Sampling probabilities for each opponent type

---

## 6. Exploiter Training Details

### What is an Exploiter?

An **exploiter** is trained with one ruthless goal: **beat a frozen version of the main model**.

- **No RNaD Regularization**: Free to find any winning strategy
- **Fixed Opponent**: Only plays against one "victim" model
- **Single, Fixed Team**: Rapidly specializes in one playstyle

### Design Decision: Single-Team Exploiters

**Rationale**:
1. **Faster Specialization**: Learns one playstyle deeply
2. **Clearer Patterns**: Discovers specific exploitation strategies
3. **Reproducibility**: Consistent behavior when loaded later
4. **Complementary**: Main agent generalizes, exploiters specialize

### The Exploiter Training Workflow

```
MAIN TRAINING LOOP
        │
        ├─ Every exploiter_check_interval updates
        ▼
  Save Current Model as "Victim"
        │
        ▼
┌────────────────────────────────────────┐
│     EXPLOITER TRAINING SUBPROCESS      │
│                                        │
│  1. Sample ONE team for exploiter      │
│  2. Train against victim (PPO only)    │
│  3. Save model + team together         │
└────────────────────────────────────────┘
        │
        ▼
  If win_rate > threshold:
    Register exploiter with team in registry
        │
        ▼
  OpponentPool now includes this exploiter
  (always uses its training team)
```

### The Opponent Pool & Adaptive Curriculum

Actors sample opponents according to curriculum:
- **Self**: Most recent main model
- **BC**: Original behavior-cloned model
- **Past**: Checkpoint history
- **Exploiters**: Adversarial agents with their specific teams

Win rates tracked per category; sampling adapts to weaknesses.

---

## 7. Performance & Optimization

This section documents the optimization journey from **540 battles/hr to 2,750 battles/hr** (5x improvement).

### Understanding the Bottlenecks

**Initial Profiling (Before Optimizations):**

| Component | Time | % of Total | Notes |
|-----------|------|------------|-------|
| Action Masking | 3-4 sec | **~99%** | Iterating 2,025 actions |
| Embedding | 11.5ms | - | Pure Python |
| Inference | 8.1ms | - | GPU forward pass |
| Network/Async | - | 69% of wall time | WebSocket I/O |

**Key Insight**: Only 31% of wall-clock time is computation. The majority is async/network overhead.

### Fast Action Masking (52,000x Speedup)

**Problem**: `_get_action_mask()` iterated over all 2,025 actions, calling `is_valid_order()` for each.

**Solution**: `fast_action_mask.py` directly enumerates valid actions from `battle.last_request`:

| Metric | Old Method | Fast Method | Improvement |
|--------|-----------|-------------|-------------|
| Avg mask time | 3-4 sec | 0.057 ms | **52,000x** |
| Mask overhead | ~99% | 2.4% | Negligible |

### Embedder Move Caching (2.75x Speedup)

**Problem**: `generate_move_features()` called 48 times per embed (6 mons × 4 moves × 2 sides), computing static features repeatedly.

**Solution**: Cache static move features by `move.id`:

```python
def _generate_static_move_features(self, move):
    if move.id in self._move_cache:
        return self._move_cache[move.id]
    # Compute and cache...
    
def generate_move_features(self, move, mon, battle):
    static = self._generate_static_move_features(move)
    dynamic = [move.current_pp / move.max_pp, ...]
    return np.concatenate([static, dynamic])
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Time per embed | 11.3 ms | 4.1 ms | **2.75x** |
| Embedding % of inference | 55% | 31% | Significant |

### Mixed Precision Training (2x Speedup)

Using FP16 instead of FP32 with `torch.cuda.amp`:

```python
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Result**: Nearly **2x speedup**, ~30% VRAM reduction.

### Why Multiprocessing? GIL Limitations

**Python's GIL** prevents true parallel execution across threads. Even with 16 "concurrent" battles in threads, they share one interpreter and compete for execution time.

**Solution**: IMPALA-style multiprocessing:
- Each actor is a separate Python process with its own interpreter
- Actors use CPU inference (GPU for learner only)
- **Bypasses GIL for 2-3x throughput improvement**

### Multi-Server Showdown Architecture

**Problem**: Pokemon Showdown is CPU-bound (single-threaded Node.js).

**Solution**: Run multiple servers on different ports:

```bash
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    node pokemon-showdown start --no-security --port $port &
    sleep 1
done
```

Each actor connects to a different server, distributing load across CPU cores.

---

## 8. Scaling Experiments & Benchmarks

### Baseline Measurements

| Configuration | Battles/hr | Notes |
|--------------|-----------|-------|
| Before action mask fix | ~540 | 3-4 sec mask time |
| After action mask fix (1 server, 1 actor) | 528-935 | ~1x |

### Multi-Server Scaling Results

| Servers | Actors | Total Pairs | Rate/hr | Status |
|---------|--------|-------------|---------|--------|
| 1 | 1 | 1 | 140 | Timeouts |
| 1 | 2 | 2 | 955 | ✅ Stable |
| 1 | 4 | 4 | 660 | Timeouts |
| 2 | 2 | 4 | 920 | Timeouts |
| **4** | **4** | **4** | **2,586** | ✅ Best |

**Key Finding**: More servers > more actors per server. Single-server contention causes timeouts.

### Hardware Stress Testing (Maximum Throughput)

| Config | Servers | Actors/Srv | Concurrent | Rate/hr | CPU avg | RAM |
|--------|---------|-----------|------------|---------|---------|-----|
| 1 | 4 | 1 | 4 | 1,579 | 18% | 3.4 GB |
| 2 | 4 | 2 | 8 | 2,513 | 15% | 5.9 GB |
| **8** | **8** | **2** | **16** | **2,756** | 16% | 9.0 GB |
| 10 | 8 | 4 | 32 | 2,269 | 16% | 7.6 GB |

**Maximum Achieved: ~2,750 battles/hour** (8 servers × 2 actors)

**Scaling Efficiency:**

| Concurrent | Expected (linear) | Actual | Efficiency |
|------------|-------------------|--------|------------|
| 4 | 1,579/hr | 1,579/hr | 100% |
| 8 | 3,158/hr | 2,513/hr | 80% |
| 16 | 6,315/hr | 2,756/hr | 44% |
| 32 | 12,630/hr | 2,269/hr | 18% |

**Conclusion**: Scaling is sub-linear due to I/O bottleneck.

### Memory Requirements

**Per Actor Process (~1.1 GB):**
| Component | Memory |
|-----------|--------|
| Python baseline | 495 MB |
| Embedder (with caches) | 94 MB |
| Model weights | 532 MB |

**Learner Process (~2.9 GB VRAM + 0.6 GB RAM):**
| Component | Memory |
|-----------|--------|
| Main model (GPU) | 558 MB |
| Reference model (GPU) | 558 MB |
| Optimizer states | 1.1 GB |
| Gradient buffers | 558 MB |

**RAM Budget for 23 GB:**
| Configuration | RAM Used | Verdict |
|--------------|----------|---------|
| 1 actor + learner | 4.0 GB | ✓ |
| 4 actors + learner | 7.4 GB | ✓ RECOMMENDED |
| 8 actors + learner | 11.9 GB | ✓ FITS |

### Optimal Configurations

**For Maximum Throughput (~2,750/hr):**
```yaml
num_actors: 8
num_showdown_servers: 8
```

**For Stability & Efficiency (~2,500/hr):**
```yaml
num_actors: 4
num_showdown_servers: 4
```

**For Quick Testing:**
```yaml
num_actors: 2
num_showdown_servers: 2
```

---

## 9. Advanced Features

### Portfolio Regularization: Preventing Strategy Collapse

Instead of one reference model, maintain **3-5 diverse past models**:

```python
# Traditional RNaD (single reference)
kl_loss = KL(current_policy || reference_policy)

# Portfolio RNaD (multiple references)
kl_losses = [KL(current_policy || ref_i) for ref_i in portfolio]
kl_loss = min(kl_losses)  # Regularize to CLOSEST reference
```

**Why it helps**: Single reference can "forget" older strategies. Portfolio maintains competence across playstyles.

```yaml
use_portfolio_regularization: true
max_portfolio_size: 5
portfolio_update_strategy: "diverse"  # or "best", "recent"
portfolio_add_interval: 5000
```

### The Training Profiler

Diagnostic tool to find optimal hyperparameters:

```bash
python src/elitefurretai/rl/profiler.py --sweep --output results.json
```

Measures:
- Data collection throughput
- Inference speed
- Training update speed
- CPU/GPU/RAM utilization

---

## 10. Quick Start Guide

### Basic Usage & Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Showdown servers (4-8 recommended)
python src/elitefurretai/rl/launch_servers.py --num-servers 4

# 3. Create config
python -c "from elitefurretai.rl.config import RNaDConfig; RNaDConfig().save('config.yaml')"

# 4. Edit config.yaml - set checkpoint_path to your BC model

# 5. Train
python src/elitefurretai/rl/train.py --config config.yaml
```

### Example Configurations

**Minimal Config (Testing):**
```yaml
checkpoint_path: "data/models/bc_action_model.pt"
num_actors: 2
num_showdown_servers: 2
train_batch_size: 16
max_updates: 10000
use_wandb: false
train_exploiters: false
```

**Production Config:**
```yaml
# Model
checkpoint_path: "data/models/bc_action_model.pt"
team_pool_path: "data/teams/gen9vgc2023regc"

# Training
learning_rate: 0.0001
rnad_alpha: 0.01
num_actors: 4
num_showdown_servers: 4
train_batch_size: 48

# Advanced
use_mixed_precision: true
use_portfolio_regularization: true
max_portfolio_size: 5

# Exploiters
train_exploiters: true
exploiter_check_interval: 5000
exploiter_train_steps: 50000
exploiter_win_threshold: 0.6

# Wandb
use_wandb: true
wandb_project: "elitefurretai-production"
```

---

## 11. Implementation Notes & Bug Fixes

### Critical Bug Fixes

**Loss Accumulation Type Mixing (`learner.py`):**
```python
# WRONG
total_loss = 0
total_loss += loss_tensor  # Can't add tensor to float!

# CORRECT
total_loss = 0.0
total_loss += loss_tensor.item()
```

**Invalid Turn Mask Logic:**
```python
# Properly check for valid turns
is_teampreview = batch["is_teampreview"]
valid_turn_mask = batch.get("valid_turn_mask", ~is_teampreview)
valid_turn_steps = ~is_teampreview & valid_turn_mask
```

### OTS Deadlock Fix

**Problem**: `accept_open_team_sheet=True` (default) caused poke-env to wait indefinitely when players disagreed on OTS.

**Fix**: Changed BCPlayer default to `accept_open_team_sheet=False`.

### Known Issues & Workarounds

| Issue | Workaround |
|-------|------------|
| OOM on WSL2 | Set `pin_memory=False` in DataLoader |
| Loss → NaN | Increase `gradient_clip` or disable mixed precision |
| Showdown timeouts | Reduce actors per server, add more servers |

### External `vgc-bench` Opponents (Fork-Safe)

`vgc-bench` may require a different `poke-env` fork than EliteFurretAI. To avoid import/API conflicts, run `vgc-bench` in a separate environment and challenge it externally.

1. Start external runner (in dedicated env):
```bash
source ../venv-vgcbench/bin/activate
python src/elitefurretai/rl/analyze/vgcbench_external_runner.py \
    --username VGCBENCHX \
    --server localhost:8000 \
    --battle-format gen9vgc2024regg \
    --checkpoint-path data/models/vgc-bench-sb3-model.zip \
    --team-file data/teams/gen9vgc2024regg/vgcbench.txt \
    --n-challenges 100
```

2. In RL config, set external usernames to bypass in-process vgc-bench player creation:
```yaml
external_vgcbench_usernames:
    - VGCBENCHX
```

When `external_vgcbench_usernames` is set, worker curriculum entries for `vgc_bench_baseline` use `send_challenges(...)` to those usernames instead of constructing local vgc-bench policy players.

### torch.compile() Results

Tested on forward pass (5.85ms baseline):

| Mode | Time | Change |
|------|------|--------|
| Original | 5.85ms | - |
| torch.compile (default) | 6.96ms | **-19%** (slower!) |
| torch.compile (reduce-overhead) | 5.48ms | +6% |

**Conclusion**: `torch.compile` is **not recommended** - overhead exceeds benefits for small batches.

---

## 12. Design Philosophy & Key Takeaways

### Core Principles

1. **Specialization Through Diversity**: Main agent trains on 100+ teams for generalization; exploiters specialize with 1 team each
2. **Stability Through Regularization**: RNaD prevents catastrophic forgetting and strategy collapse
3. **Adversarial Robustness**: Continuous exploiter training exposes weaknesses
4. **Hardware Optimization**: Multi-server + multiprocessing maximizes throughput
5. **Reproducibility**: Configuration-driven design

### Lessons Learned

**On Performance:**
- Pokemon Showdown is the bottleneck, not GPU
- 4-8 parallel servers fully utilizes 8-core CPU
- Mixed precision nearly doubles training speed
- Embedder was 55% of time → caching reduced to 31%
- **Multiprocessing bypasses GIL for 2-3x throughput**

**On Exploiter Training:**
- Single-team exploiters learn 2-3x faster
- Team persistence is critical for reproducibility
- Win threshold: 40% too low (noise), 70% too high (misses)
- Optimal interval: 3k-5k updates

**On Training Dynamics:**
- RNaD essential - pure PPO collapses ~20-30k updates
- Portfolio (3-5 models) better than single reference
- Scaling is sub-linear due to I/O bottleneck

### Future Directions

1. **Adaptive Exploiter Allocation**: Dynamic adjustment based on win rate stability
2. **Multi-Format Training**: Single agent across Reg C, D, E, F
3. **Team Generation**: Generate novel teams instead of sampling
4. **Native Battle Engine**: Port Showdown to Python/Rust to eliminate WebSocket overhead
5. **Transformer Evaluation**: Benchmark `TransformerThreeHeadedModel` vs LSTM backbone on full training runs
6. **Number Bank Tuning**: Optimize bin counts and embedding dimensions for production training
7. **Batched Transformer Inference**: Pad variable-length contexts for true batched inference in actors (currently sequential per-battle)

### Files in This Module

| File | Purpose |
|------|---------|
| `agent.py` | RNaDAgent wrapper for step-by-step RL (supports LSTM + Transformer) |
| `learner.py` | RNaDLearner with PPO + KL regularization + distributional value |
| `multiprocess_actor.py` | IMPALA-style multiprocessing actors and trainer |
| `config.py` | RNaDConfig dataclass (exploration, optimizer, architecture flags) |
| `fast_action_mask.py` | Optimized action mask generation |
| `model_io.py` | Model construction, checkpoint save/load, config key management |
| `opponent_pool.py` | Opponent sampling (self, BC, exploiters) |
| `portfolio_learner.py` | Portfolio regularization extension |
| `players.py` | BatchInferencePlayer with LSTM/Transformer hidden state management |
| `train.py` | Main training coordinator |
| `exploiter_train.py` | Exploiter training subprocess |
| `launch_servers.py` | Multi-server Showdown launcher |
| `evaluate.py` | Model evaluation utilities |

---

## ps-ppo-Inspired Improvements (February 2026)

Five architectural and training improvements inspired by the ps-ppo project have been implemented. All are gated by config flags. See [IMPLEMENTATION_PLAN.md](../../../docs/IMPLEMENTATION_PLAN.md) for the full design rationale.

### 1. Temperature Annealing & Top-p Sampling
- **What**: Temperature-scaled softmax for exploration control + nucleus sampling to filter low-probability actions
- **Config**: `temperature_start=1.5`, `temperature_end=0.5`, `temperature_anneal_steps=50000`, `top_p=0.95`
- **Key detail**: Log-probs for PPO ratios are computed at T=1 (unscaled) to avoid biasing importance weights
- **Entropy coefficient** also anneals via `ent_coef_end`

### 2. Topology-Aware Optimizer
- **What**: AdamW with separate learning rates and weight decay for backbone vs heads; LR scheduler with warmup + cosine/linear decay
- **Config**: `optimizer` dict with `type`, `weight_decay`, `lr_backbone`, `lr_heads`, `lr_warmup_steps`, `lr_schedule`
- **Purpose**: Prevents value head under-training (common PPO failure mode) and regularizes large backbone layers

### 3. Distributional Value Head (C51)
- **What**: 51-bin categorical distribution over [-1, 1] replaces scalar Tanh value head
- **Config**: `num_value_bins=51`, `value_min=-1.0`, `value_max=1.0`
- **Loss**: Cross-entropy against two-hot encoded targets (via `twohot_encode()`)
- **Model output change**: `forward()` returns 4 values, `forward_with_hidden()` returns 5 (extra: `win_dist_logits`)
- **Actor impact**: None — actors use the scalar expected value (3rd return element), computed as `(softmax(logits) * support).sum(-1)`

### 4. Number Bank Embeddings
- **What**: Learned embedding lookup for numerical features (HP%, stats, base power) instead of raw floats
- **Config**: `use_number_banks=false` (disabled by default), `number_bank_embedding_dim=16`, `number_bank_hp/stat/power_bins`
- **Design**: Discretization + embedding happens inside `GroupedFeatureEncoder` via `NumberBankEncoder` — the Embedder output format is unchanged
- **Feature identification**: Pattern matching on `Embedder.feature_names` (HP_PATTERNS, STAT_PATTERNS, POWER_PATTERNS)
- **Requires**: Fresh training when enabled (changes model input dimensions)

### 5. Transformer Architecture
- **What**: `TransformerThreeHeadedModel` replaces LSTM backbone with TransformerEncoder + decision tokens
- **Config**: `use_transformer=false` (disabled by default), `transformer_layers=6`, `transformer_heads=16`, `transformer_ff_dim=2048`, `use_decision_tokens=true`, `use_causal_mask=true`
- **Decision tokens**: Learned [ACTOR], [CRITIC], [FIELD] vectors prepended to the sequence; ACTOR → turn head, CRITIC → value head
- **Hidden state**: Context tensor (growing sequence of past encoded features) instead of LSTM `(h, c)`. Each turn appends to context.
- **Inference**: Per-battle sequential inference in actors (contexts differ in length across battles). LSTM batched path preserved.
- **Requires**: Fresh training when enabled

---

## IMPALA Benchmark Results - 2026-01-18

After fixing the poke-env import issue and enabling proper trajectory collection, 
the IMPALA multiprocessing architecture achieves the following throughput:

### Configuration Comparison

| Config | Actors | Servers | Act/Srv | Rate/hr | Notes |
|--------|--------|---------|---------|---------|-------|
| 1×1 | 1 | 1 | 1.0 | 2,569 | Baseline |
| 2×2 | 2 | 2 | 1.0 | 2,667 | 1.04x baseline |
| **4×4** | 4 | 4 | 1.0 | **3,106** | **1.21x baseline** |
| 4×2 | 4 | 2 | 2.0 | 2,736 | 2 actors/server |
| **6×3** | 6 | 3 | 2.0 | **2,912** | Good efficiency |
| 6×6 | 6 | 6 | 1.0 | 2,763 | 1.08x baseline |
| 8×4 | 8 | 4 | 2.0 | 2,190 | CPU saturated |
| 8×8 | 8 | 8 | 1.0 | 1,793 | Server startup failures |

### Key Findings

1. **Optimal configuration: 4 actors × 4 servers = ~3,100 battles/hr**
2. **CPU is the bottleneck** - with 8 cores, more than 6 actors causes contention
3. **1:1 actor-to-server ratio** works best for lower actor counts
4. **2:1 actor-to-server ratio** can work with 4-6 actors and 2-3 servers
5. **Server startup time matters** - 8 servers need longer warmup (4+ seconds)

### Recommended Configuration

For this hardware (8-core CPU, 138.8M param model):

```yaml
# Optimal throughput
num_actors: 4
num_showdown_servers: 4

# Alternative (similar performance, less servers)
num_actors: 6
num_showdown_servers: 3
```

### Per-Actor Efficiency

| Config | Total Rate | Per-Actor Rate | Efficiency |
|--------|-----------|----------------|------------|
| 1×1 | 2,569/hr | 2,569/hr | 100% |
| 4×4 | 3,106/hr | 777/hr | 30% |
| 6×3 | 2,912/hr | 485/hr | 19% |

Efficiency drops with more actors due to CPU contention for model inference.
However, total throughput increases up to ~4 actors.

