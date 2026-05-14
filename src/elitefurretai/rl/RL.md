# RL Training System

This document is the **comprehensive, one-stop guide** to EliteFurretAI's reinforcement learning training system. It covers architecture, algorithms, optimization history, benchmarks, and practical guidance for running and extending the system.

## Table of Contents

1.  [**Hardware & Environment**](#1-hardware--environment)
2.  [**Architecture Overview**](#2-architecture-overview)
    -   IMPALA-Style Multiprocessing
    -   Architectural Principles
3.  [**Battle Backends**](#3-battle-backends)
    -   Supported Backends
    -   Rust Backend Status
    -   Core Rust Runtime Files
4.  [**RNaD Algorithm Overview**](#4-rnad-algorithm-overview)
    -   Why Regularized Nash Dynamics?
    -   The RNaD Loss Function
    -   Inspiration from Ataraxos
5.  [**Core Components and Files**](#5-core-components)
    -   `players.py`: Actor-side player + agent wrapper
    -   `learners.py`: RNaD learner + model construction
    -   `worker.py`: Actor process body
    -   `train.py`: Trainer entrypoint and coordinator
    -   `config.py`: Configuration system
    -   `masking.py`: Optimized action masking
    -   Team management lives in `etl/`, not here
6.  [**Training Workflow & Features**](#6-training-workflow--features)
    -   The Multi-Stage Training Process
    -   Configuration-Driven Training
    -   Resume Training from Checkpoints
    -   Automatic Exploiter Training
    -   Comprehensive Monitoring with WandB
7.  [**Exploiter Training Details**](#7-exploiter-training-details)
    -   What is an Exploiter?
    -   Design Decision: Single-Team Exploiters
    -   The Exploiter Training Workflow
    -   The Opponent Pool & Adaptive Curriculum
8.  [**Performance & Optimization**](#8-performance--optimization)
    -   Understanding the Bottlenecks
    -   Fast Action Masking (52,000x speedup)
    -   Embedder Move Caching (2.75x speedup)
    -   Mixed Precision Training (2x speedup)
    -   Why Multiprocessing? GIL Limitations
    -   Multi-Server Showdown Architecture
8b. [**Centralized Inference (May 2026)**](#8b-centralized-inference-may-2026)
    -   ModelRegistry + WorkerInferenceClients architecture
    -   Hot-swap opponent.inference_client between named models
    -   +34% throughput on full curriculum
    -   torch.compile multi-thread caveat
9.  [**Scaling Experiments & Benchmarks**](#9-scaling-experiments--benchmarks)
    -   Baseline Measurements
    -   Multi-Server Scaling Results
    -   Hardware Stress Testing
    -   Memory Requirements
    -   Optimal Configurations
10. [**Advanced Features**](#10-advanced-features)
    -   Portfolio Regularization
    -   The Training Profiler
11. [**Quick Start Guide**](#11-quick-start-guide)
    -   Basic Usage & Commands
    -   Example Configurations
12. [**Implementation Notes & Bug Fixes**](#12-implementation-notes--bug-fixes)
    -   Critical Bug Fixes
    -   OTS Deadlock Fix
    -   Known Issues & Workarounds
13. [**Design Philosophy & Key Takeaways**](#13-design-philosophy--key-takeaways)
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
- **Model**: `TransformerThreeHeadedModel` (Transformer backbone) with FULL embedder
- **Parameters**: ~26.7M (cool-bee-85-finetune; raw featureset)
- **Weights Size**: ~100 MB
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
- Each actor is a **separate Python process** (still true — bypasses GIL).
- **Legacy mode** (the dataclass default, `enable_centralized_inference: false`):
  each actor holds a CPU model copy and runs its own batched inference
  loop. The diagram above shows this mode.
- **Centralized inference** (opt-in via `enable_centralized_inference: true`,
  enabled in the production configs like `sep_arch.yaml`): actors no
  longer hold their own model copies. A trainer-side `ModelRegistry`
  owns one `InferenceService` per model name; actors construct a
  `WorkerInferenceClients` bundle and submit via mp.Queue. The inference
  forward runs on `config.hardware.device` (typically the trainer's GPU)
  in the trainer process. See "Centralized Inference" section below for
  details.
- **Bypasses GIL** for true parallelism — featurization + battle
  stepping in actor processes, inference either CPU-per-actor (legacy)
  or batched on the trainer's GPU (centralized).
- In legacy mode the learner **broadcasts updated weights** to actors
  via mp.Queue. In centralized mode actors don't hold the model;
  instead the trainer calls `registry.sync_weights(...)` to update the
  inference service's copy at the same cadence.
- Actors send completed trajectories via `multiprocessing.Queue`.

---

## 3. Battle Backends

The RL system now supports two execution backends:

- `showdown_websocket`: the traditional local Pokemon Showdown server plus poke-env `Player` path
- `rust_engine`: the in-process Rust simulator plus standalone `DoubleBattle` synchronization path

### Supported Backends

The project should keep both backends working unless there is an explicit decision to retire one. The websocket path is now the primary optimization target for RL training, and the Rust path remains useful for fallback, parity checks, and debugging.

### Rust Backend Status

The current Rust path remains supported, but it is no longer the primary optimization path.

- It is still integrated into the real [train.py](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/rl/train.py) entrypoint.
- It still has dedicated runtime and benchmark coverage under [unit_tests/engine](/home/cayman/Repositories/EliteFurretAI/unit_tests/engine).
- The latest paired learner-facing comparison favored Showdown on wall-clock time to the same update count, while Rust remained useful as a fallback and comparison backend.

The current recommendation is:

- use `showdown_websocket` as the primary optimization and training focus
- keep `rust_engine` available for fallback, parity checks, and debugging

### Core Rust Runtime Files

The Rust path is centered on these files:

- [rust_battle_engine.py](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine/rust_battle_engine.py): binding adapter, request sanitization, protocol replay, standalone `DoubleBattle` synchronization
- [sync_battle_driver.py](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine/sync_battle_driver.py): synchronous battle loop, policy batching, trajectory collection, diagnostics, synthetic teampreview compatibility
- [battle_snapshot.py](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine/battle_snapshot.py): policy-facing observation object for the Rust self-play path
- [rust_model_benchmark.py](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine/rust_model_benchmark.py): model-backed Rust benchmark entrypoint
- [ENGINE.md](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine/ENGINE.md): current engine-package guide, backend decision summary, and Stage 2 runtime learnings

### Engine Package Layout

The execution-side modules now live in [src/elitefurretai/engine](/home/cayman/Repositories/EliteFurretAI/src/elitefurretai/engine) rather than directly under the RL package.

That split is intentional:

- `elitefurretai.engine` owns battle-execution concerns
    - Rust request/state synchronization
    - synchronous battle driving
    - Showdown server process management
    - engine comparison benchmarks
- `elitefurretai.rl` owns algorithm and training concerns
    - config
    - learners
    - players
    - opponent sampling
    - train entrypoint

This keeps runtime/backend code from getting mixed together with RNaD-specific training code.

The Rust binding team-conversion helpers now live directly in `rust_battle_engine.py` so the Rust engine boundary stays consolidated in one module. If the project later introduces a backend-agnostic team representation shared by multiple runtimes, that would be the point to split them back out.

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
3.  **Inference placement is mode-dependent**: Legacy mode runs inference on per-actor CPU model copies, freeing the GPU entirely for gradient computation. Centralized mode runs batched inference on the trainer's GPU (`config.hardware.device`), trading some GPU contention with the learner for much higher inference batch sizes and one model copy total.
4.  **Periodic Weight Sync**: Every N trajectories, fresh weights propagate from the learner — to the actors' model copies in legacy mode, or to the trainer-side `InferenceService`'s model copy in centralized mode (`registry.sync_weights`).

---

## 4. RNaD Algorithm Overview

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

## 5. Core Components

The full file inventory lives in the "Files in This Module" table near
the bottom of this doc. This section walks the main concepts in the
order data flows through them.

### `players.py`: Actor-side player + agent wrapper

Two classes co-locate here because they pair with each other inside the
actor:

`RNaDAgent` wraps the BC-trained `TransformerThreeHeadedModel` for
step-by-step RL inference. The BC model expects full trajectories, but
RL requires one decision at a time; `RNaDAgent` carries the growing
transformer context between turns. `get_initial_state` returns `None`
so the first turn starts with an empty context.

`BatchInferencePlayer` is the poke-env `Player` subclass each actor
runs. It is **dual-mode**:

- **Legacy mode**: owns a CPU `RNaDAgent`, runs its own per-player
  batched inference loop (`_inference_loop` thread + per-request
  asyncio futures) over actions taken across concurrent battles.
- **Centralized mode**: receives an `InferenceClient` instead of a
  model and `await client.submit(...)`s every action. Hidden state
  lives on the trainer side and is not held here.

It also accumulates per-step `(state, action, log_prob, value, mask, …)`
tuples during each battle and pushes a completed-battle dict onto
`trajectory_queue` (keys: `steps`, `opponent_type`, `won`,
`battle_length`, `forfeited`) when a battle ends.

### `learners.py`: RNaD learner + model construction

`PortfolioRNaDLearner` is the heart of training: holds `main_model`
(learning) and one or more frozen reference models (portfolio
regularization), pulls trajectories from the queue, and:

- Computes the RNaD loss (PPO policy + distributional value + entropy + KL)
- Uses **C51 distributional value loss** (cross-entropy against two-hot
  encoded targets) instead of scalar MSE
- Updates `main_model` via backprop with a **topology-aware optimizer**
  (AdamW with separate param groups for backbone vs heads)
- Applies an **LR scheduler** (linear warmup + cosine/linear decay)
- Periodically copies weights to the reference model(s)

The same module also owns model-construction helpers used by both the
trainer and the workers:
`build_model_from_config`, `load_model_from_checkpoint`,
`load_agent_from_checkpoint`, `save_checkpoint`, `load_checkpoint`,
`is_checkpoint_compatible_with_model_config`.

Weight propagation is mode-dependent:
- **Legacy**: the trainer broadcasts state dicts to each worker via
  per-worker `weight_queue`s; each worker calls `agent.model.load_state_dict`.
- **Centralized**: the trainer calls `registry.sync_weights(name, sd)`
  to update the inference service's model copy in-place. Workers never
  touch model weights.

### `worker.py`: Actor process body

`mp_worker_process` is the function each actor subprocess runs (spawned
from `train.py` via `mp.Process`). Lifecycle:

1. **Model loading** (legacy only): build/load `RNaDAgent` from
   checkpoint on CPU. In centralized mode this is skipped — workers
   build a `WorkerInferenceClients` bundle from spawn-time mp.Queue
   handles instead.
2. **Environment setup**: build a `VGCEnvironment` over the configured
   backend (Showdown websocket or Rust in-process).
3. **Battle loop**: repeatedly poll for new weights (legacy) → run a
   batch of battles → push completed trajectories to the learner via
   `mp_traj_queue`.

There is no `MultiprocessingTrainer` class — the trainer-side
orchestration (spawning workers, draining trajectories, weight
broadcast, checkpointing) lives directly in `train.py`'s `main()`.

### `train.py`: Trainer entrypoint and coordinator

`main()` owns the full trainer side: builds the learner, optionally
the `ModelRegistry` (centralized inference), spawns
`config.hardware.num_workers` worker processes, runs the trajectory
collection loop, calls `learner.update(...)`, periodically broadcasts
weights / `registry.sync_weights(...)`, and writes checkpoints.

Notable helpers in this module: `initialize_learner`,
`initialize_training_state`, `collate_trajectories`, `start_memory_watchdog`,
`_maybe_run_exploiter_update`, `get_dead_workers`.

### `config.py`: Configuration system

`RNaDConfig` dataclass holds all hyperparameters. Tunable knob groups
(non-exhaustive):

- **Exploration**: `temperature_start/end`, `temperature_anneal_steps`,
  `top_p`, `ent_coef_end`
- **Optimizer**: `optimizer` dict with `type` (adam/adamw),
  `weight_decay`, `lr_backbone`, `lr_heads`, `lr_warmup_steps`,
  `lr_schedule`
- **Distributional Value**: `num_value_bins`, `value_min`, `value_max`
- **Number Banks**: `use_number_banks`, `number_bank_embedding_dim`,
  `number_bank_hp/stat/power_bins`
- **Transformer**: `transformer_layers/heads/ff_dim/dropout`,
  `use_decision_tokens`, `use_causal_mask`
- **Hardware**: `num_workers`, `batch_size`, `batch_timeout`,
  `battle_backend`, `device`, `max_concurrent_battles_per_player`,
  `enable_centralized_inference`, `compile_inference_model`

### `masking.py`: Optimized action masking

`fast_get_action_mask(battle: DoubleBattle) -> np.ndarray` is the fast
path for valid-action masking (≈52,000× faster than the naive
2025-action enumeration). It reads `battle.last_request` and directly
enumerates valid (move, target) and switch pairs.

### Team management lives in `etl/`, not here

Pokémon team sampling and management uses `etl.TeamRepo`
(`elitefurretai.etl.team_repo`), not anything under `rl/`. The RL
modules import it from `elitefurretai.etl import TeamRepo`.

---

## 6. Training Workflow & Features

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

## 7. Exploiter Training Details

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

## 8. Performance & Optimization

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

**Solution**: `masking.py` directly enumerates valid actions from `battle.last_request`:

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

## 8b. Centralized Inference (May 2026)

Shipped on `main` 2026-05-14 (commit `3ead76c`). The full design + measurement
history is in `planning/stage2/2026-05-13-18-00-centralized-inference-implementation-plan.md`
and `planning/stage2/2026-05-14-00-15-model-registry-plan.md`.

### What changed

Pre-merge architecture: each actor process held its own CPU copy of the
main agent, ran its own per-player `BatchInferencePlayer` queue +
inference loop, and pulled weight updates from the learner via mp.Queue
broadcasts. With `num_workers=4` and a 27M-param transformer, that's
4 model copies in worker memory and 4 separate, CPU-bound inference
loops.

Post-merge architecture (opt-in via `enable_centralized_inference: true`):

- Trainer owns a `ModelRegistry` with one `InferenceService` per model
  name. Currently registered: `main` (torch.compile'd), optionally
  `bc` / `exploiter` / `victim` (gated on curriculum weight).
- Each service runs as a daemon thread in the trainer process. The
  service drains an `mp.Queue` of `InferenceRequest`s, batches up to
  `batch_size` (or until `batch_timeout` elapses), runs ONE batched
  forward, and dispatches `InferenceResponse`s to per-worker response
  queues.
- Workers no longer hold model copies. Each constructs a
  `WorkerInferenceClients` bundle from per-worker mp.Queue handles.
  Players in the worker call `client.submit(...)` and await a future.
- Hidden state lives on the trainer side, in
  `RealModelBatchHandler.hidden_states` keyed by
  `(worker_id, player_id, battle_tag)`. Wire payloads carry only the
  lightweight battle_tag, not the (1, T, hidden_size) tensor.
- Eviction: workers send an `EvictRequest` on battle completion so the
  trainer can free its hidden-state slot. Cleanup also fires on every
  stale-request / timeout / send-failure path
  (`BatchInferencePlayer._reset_battle_hidden_state`).

### Hot-swap (multi-model curriculum)

`WorkerOpponentFactory.configure_opponent_for_batch` chooses an
opponent type per battle pair. In centralized mode, it re-points
`opponent.inference_client` to the right client by name:
`clients.get("main")`, `clients.get("bc")`, etc.
`WorkerOpponentFactory._swap_to(slot, name, legacy_agent)` resolves
"centralized client by name, falling back to legacy `slot.model = X`."

### What's NOT centralized

- **Ghost models**: each worker still loads ghost checkpoints from
  disk lazily (`_get_ghost_agent`). Ghost rotation through the registry
  is a deferred follow-up — see the registry plan's "Future work".
- **OpponentPool's main-process eval battles**: still use legacy
  per-call inference. Cheap enough that centralizing them isn't
  motivated.

### Throughput

Measured on sep_arch.yaml, full original curriculum (self_play 0.3 /
bc 0.1 / ghosts 0.1 / max_damage 0.1 / simple_heuristic 0.1 /
vgc_bench 0.3), post-warmup updates:

| Variant | Throughput | Learner steps/s |
|---|---|---|
| Per-player baseline (pre-merge) | 3.7 traj/s | ~60 |
| Centralized (post-merge) | **4.98 traj/s** | **~87** |
| Δ | **+34%** | **+45%** |

Plus two collateral improvements landed in the same commit:
- Vectorized `GroupedFeatureEncoder._dual_expand` (was 32% of worker
  py-spy OwnTime — per-position Python loop calling `nn.Embedding`
  N times; vectorized version batches per-bank lookups). ~1.9× CUDA
  microbenchmark speedup.
- F9 legacy bug fix: `RealModelBatchHandler._slice_next_hidden`
  correctly handles mixed-length batches. Pre-fix legacy code sliced a
  padding-derived position instead of the real new-state position,
  corrupting next-turn hidden state for any battle that started in a
  heterogeneous batch.

### Caveats / known issues

- **torch.compile + multi-threaded service calls**: `mode='default',
  dynamic=True` is not thread-safe across multiple compiled services
  in concurrent threads. Symptom: `RuntimeError: Detected that you are
  using FX to symbolically trace a dynamo-optimized function`.
  Workaround: `registry.register(name, agent, compile=False)` for any
  non-main model. Documented in `ModelRegistry.register`'s docstring.
- **Memory watchdog**: bumped from 20 GB → 22 GB in sep_arch.yaml.
  VGCBench external runners (~5.5 GB) + 4 Showdown servers + workers
  + trainer combined RSS edges over 20 GB on the 24 GB WSL2.

### How to revert (escape hatch)

`enable_centralized_inference: false` in your config. The dual-mode
`BatchInferencePlayer` keeps the legacy per-player path fully intact;
this is one config knob away.

---

## 9. Scaling Experiments & Benchmarks

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

## 10. Advanced Features

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

## 11. Quick Start Guide

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

## 12. Implementation Notes & Bug Fixes

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

## 13. Design Philosophy & Key Takeaways

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
5. **Number Bank Tuning**: Optimize bin counts and embedding dimensions for production training
7. **Batched Transformer Inference**: Pad variable-length contexts for true batched inference in actors (currently sequential per-battle)

### Files in This Module

| File | Purpose |
|------|---------|
| `players.py` | `RNaDAgent` wrapper (model adapter) + `BatchInferencePlayer` (poke-env Player that bridges battles → inference). Dual-mode: legacy per-player batcher OR centralized via `inference_client`. |
| `learners.py` | `PortfolioRNaDLearner` with PPO + KL regularization + distributional value (C51). Model construction lives here too (`build_model_from_config`, `load_agent_from_checkpoint`). |
| `worker.py` | `mp_worker_process` — the actor subprocess body. Spawns once per `num_workers`; sets up VGCEnvironment, runs battles, ships trajectories. In centralized mode skips loading the main model and constructs `WorkerInferenceClients` from spawn args. |
| `train.py` | Main training coordinator. Owns the learner, the `ModelRegistry` (centralized inference), worker spawn, weight broadcast, checkpointing. |
| `config.py` | `RNaDConfig` dataclass. Knobs: hardware (num_workers, batch_size, enable_centralized_inference, compile_inference_model, max_concurrent_battles_per_player, ...), algorithm (PPO/RNaD), curriculum, etc. |
| `opponents.py` | `OpponentPool` (trainer-side curriculum manager) + `WorkerOpponentFactory` (worker-side player builder, opponent hot-swap via `_swap_to`). |
| `masking.py` | `fast_get_action_mask` and helpers (the optimized action-mask path). |
| `model_registry.py` | Trainer-side `ModelRegistry`: one `InferenceService` per registered model name. Used in centralized inference mode (post-2026-05-14). |
| `inference_trainer.py` | Trainer-side: `InferenceService` (batched inference daemon thread, one per registered model) + `RealModelBatchHandler` (the model-forward path; owns `hidden_states` keyed by `(worker_id, player_id, battle_tag)`) + `echo_batch_handler` for plumbing tests. |
| `inference_worker.py` | Worker-side: `InferenceClient` (submits `InferenceRequest`, awaits response via per-request asyncio future) + `WorkerInferenceClients` (per-worker bundle of clients keyed by model name; counterpart to `ModelRegistry`). |
| `inference_ipc.py` | `InferenceRequest` / `InferenceResponse` / `EvictRequest` dataclasses (the wire protocol). |
| `launch_servers.py` | Multi-server Showdown launcher. |
| `analyze/` | Evaluation utilities, plotters, VGCBench external runner. |

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
- **What**: `TransformerThreeHeadedModel` — TransformerEncoder + decision tokens (the only supported backbone)
- **Config**: `transformer_layers=6`, `transformer_heads=16`, `transformer_ff_dim=2048`, `use_decision_tokens=true`, `use_causal_mask=true`
- **Decision tokens**: Learned [ACTOR], [CRITIC], [FIELD] vectors prepended to the sequence; ACTOR → turn head, CRITIC → value head
- **Hidden state**: Context tensor (growing sequence of past encoded features). Each turn appends to context.
- **Inference**: Per-battle sequential inference in actors (contexts differ in length across battles).

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

