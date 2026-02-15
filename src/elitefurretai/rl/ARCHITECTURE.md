# RL Module Architecture

This document explains how all the RL module files work together and suggests simplifications.

## File Overview

```
rl/
├── train.py              # Main training orchestrator (entry point)
├── config.py             # RNaDConfig dataclass
├── agent.py              # RNaDAgent wrapper + MaxDamagePlayer
├── learner.py            # RNaDLearner (PPO + KL loss)
├── portfolio_learner.py  # PortfolioRNaDLearner (multi-reference models)
├── multiprocess_actor.py # BatchInferencePlayer (battle worker)
├── opponent_pool.py      # OpponentPool + ExploiterRegistry
├── worker_factory.py     # WorkerOpponentFactory + build_model_from_config
├── evaluate.py           # Evaluation utilities
├── exploiter_train.py    # Exploiter training subprocess
├── fast_action_mask.py   # Optimized action masking (~52,000x speedup)
├── configs/              # YAML config files
└── analyze/              # Profiling and analysis scripts
```

## How Files Work Together

### 1. Training Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              train.py (main)                                 │
│  • Parses args, loads config                                                 │
│  • Launches Showdown servers                                                 │
│  • Creates learner + queues                                                  │
│  • Spawns mp_worker_process() as separate Python processes                   │
│  • Main loop: consumes trajectories, calls learner.update(), broadcasts wts  │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────────┐
          │                          │                              │
          ▼                          ▼                              ▼
┌──────────────────────┐  ┌──────────────────────┐      ┌──────────────────────┐
│  mp_worker_process   │  │  mp_worker_process   │      │  mp_worker_process   │
│  (Actor Process 0)   │  │  (Actor Process 1)   │ ...  │  (Actor Process N)   │
│                      │  │                      │      │                      │
│  WorkerOpponentFact. │  │  WorkerOpponentFact. │      │  WorkerOpponentFact. │
│        │             │  │        │             │      │        │             │
│        ▼             │  │        ▼             │      │        ▼             │
│  BatchInference-     │  │  BatchInference-     │      │  BatchInference-     │
│  Player (players)    │  │  Player (players)    │      │  Player (players)    │
│        │             │  │        │             │      │        │             │
│        ▼             │  │        ▼             │      │        ▼             │
│  Showdown :8000      │  │  Showdown :8001      │      │  Showdown :800N      │
└──────────────────────┘  └──────────────────────┘      └──────────────────────┘
          │                          │                              │
          └──────────────────────────┼──────────────────────────────┘
                                     │ (trajectories via MPQueue)
                                     ▼
                        ┌────────────────────────┐
                        │     RNaDLearner        │
                        │  • Computes PPO loss   │
                        │  • Computes KL loss    │
                        │  • Updates model       │
                        │  • Updates ref model   │
                        └────────────────────────┘
```

### 2. Class Responsibilities

| Class | File | Responsibility |
|-------|------|----------------|
| `RNaDConfig` | config.py | Dataclass holding all hyperparameters. Loads/saves YAML. |
| `RNaDAgent` | agent.py | Wraps `FlexibleThreeHeadedModel` for RL. Manages LSTM hidden state. |
| `MaxDamagePlayer` | agent.py | Simple baseline opponent that picks highest damage move. |
| `RNaDLearner` | learner.py | PPO + RNaD updates. Single reference model. |
| `PortfolioRNaDLearner` | portfolio_learner.py | RNaD with multiple reference models for diversity. |
| `BatchInferencePlayer` | multiprocess_actor.py | Player that runs battles, batches inference, collects trajectories. |
| `OpponentPool` | opponent_pool.py | **Learner-side**: Tracks win rates by opponent type, caches BC model. |
| `ExploiterRegistry` | opponent_pool.py | JSON registry of trained exploiter models. |
| `WorkerOpponentFactory` | worker_factory.py | **Worker-side**: Creates players/opponents, samples based on curriculum. |
| `build_model_from_config` | worker_factory.py | Utility to construct `FlexibleThreeHeadedModel` from config dict. |

### 3. Data Flow

```
1. BATTLE STATE
   DoubleBattle (from poke-env)
        │
        ▼
2. EMBEDDING
   Embedder.embed_battle() → tensor [9223 dims]
        │
        ▼
3. INFERENCE
   RNaDAgent.forward() → (turn_logits, tp_logits, value, hidden)
        │
        ▼
4. ACTION SELECTION
   Sample from masked logits → MDBO action index
        │
        ▼
5. ACTION EXECUTION
   MDBO.decode() → BattleOrder → send to Showdown
        │
        ▼
6. TRAJECTORY COLLECTION
   (state, action, reward, log_prob, value) → trajectory queue
        │
        ▼
7. LEARNING
   RNaDLearner.update(batch) → gradient descent
```

### 4. Process vs Main Thread Responsibilities

| Location | Creates | Manages |
|----------|---------|---------|
| **Main Process** | `RNaDLearner`, `OpponentPool`, `RNaDAgent` (main + ref) | Gradient updates, weight broadcasting, checkpointing, wandb logging |
| **Worker Process** | `WorkerOpponentFactory`, `BatchInferencePlayer`, local `RNaDAgent` | Battles, trajectory collection, weight receiving |

### 5. Configuration Flow

```
config.yaml
    │
    ▼
RNaDConfig.load("config.yaml")
    │
    ├─► train.py: Uses for training hyperparameters
    │
    ├─► mp_worker_process: Receives as serialized dict
    │       │
    │       └─► WorkerOpponentFactory: Uses curriculum dict
    │
    └─► RNaDLearner: Uses lr, clip_range, ent_coef, etc.
```

---

## Current Pain Points

### 1. **Overlapping Responsibilities: OpponentPool vs WorkerOpponentFactory**

Both classes deal with opponent management but in different contexts:

| OpponentPool | WorkerOpponentFactory |
|--------------|----------------------|
| Lives in main process | Lives in worker process |
| Tracks win rates | Creates BatchInferencePlayers |
| Caches BC model | Samples opponent types |
| Has `sample_opponent()` method | Has `sample_opponent_type()` method |

**Problem**: The dual-class design exists because:
- `OpponentPool` can't be pickled across process boundaries (holds torch models)
- Workers need local opponent management

### 2. **File Size: train.py is 1100+ lines**

`train.py` handles too many concerns:
- Argument parsing
- Server management
- Worker process definition
- Trajectory collation
- Checkpointing
- Main training loop
- Wandb logging

### 3. **Agent Types Scattered**

- `RNaDAgent` in agent.py
- `MaxDamagePlayer` in agent.py
- `BCPlayer` imported from supervised/ in opponent_pool.py
- `BatchInferencePlayer` in multiprocess_actor.py

### 4. **Duplicate Constants**

Opponent type strings defined in multiple places:
- `OpponentPool.SELF_PLAY`, `OpponentPool.BC_PLAYER`, etc.
- `WorkerOpponentFactory.SELF_PLAY`, `WorkerOpponentFactory.BC_PLAYER`, etc.

---

## Suggested Simplifications

### Option A: Minimal Refactor (Low Risk)

Keep current structure but:

1. **Extract constants to shared module**
   ```python
   # rl/constants.py
   class OpponentType:
       SELF_PLAY = "self_play"
       BC_PLAYER = "bc_player"
       MAX_DAMAGE = "max_damage"
       GHOSTS = "ghosts"
       EXPLOITERS = "exploiters"
   ```

2. **Move server management to separate file**
   ```python
   # rl/server_manager.py
   def launch_showdown_servers(...)
   def shutdown_showdown_servers(...)
   def allocate_server_ports(...)
   ```

3. **Move checkpoint functions to separate file**
   ```python
   # rl/checkpoint.py
   def save_checkpoint(...)
   def load_checkpoint(...)
   ```

**Result**: train.py drops from ~1100 to ~700 lines.

### Option B: Moderate Refactor

In addition to Option A:

4. **Merge agent types into one file**
   ```
   # rl/players.py (rename from agent.py)
   - RNaDAgent
   - MaxDamagePlayer
   - Move BatchInferencePlayer here (from multiprocess_actor.py)
   ```

5. **Rename multiprocess_actor.py → worker.py**
   - Contains only `mp_worker_process` and helper functions
   - Clearer name reflecting its role

6. **Consolidate opponent management**
   ```
   # rl/opponents.py (merge opponent_pool.py + parts of worker_factory.py)
   - OpponentType constants
   - ExploiterRegistry
   - OpponentPool
   - WorkerOpponentFactory
   ```

**New structure**:
```
rl/
├── train.py              # Main loop only (~500 lines)
├── config.py             # Unchanged
├── players.py            # All player/agent classes
├── learner.py            # RNaDLearner
├── portfolio_learner.py  # PortfolioRNaDLearner
├── worker.py             # mp_worker_process + helpers
├── opponents.py          # All opponent management
├── checkpoint.py         # Save/load checkpoints
├── server_manager.py     # Showdown server management
├── model_builder.py      # build_model_from_config
├── fast_action_mask.py   # Unchanged
├── evaluate.py           # Unchanged
└── exploiter_train.py    # Unchanged
```

### Option C: Aggressive Refactor (Higher Risk)

In addition to Option B:

7. **Unify OpponentPool and WorkerOpponentFactory**
   
   Create a serializable `OpponentConfig` that can cross process boundaries:
   ```python
   @dataclass
   class OpponentConfig:
       curriculum: Dict[str, float]
       bc_model_path: Optional[str]
       exploiter_registry_path: Optional[str]
       battle_format: str
       team_pool_path: str
   
   class OpponentManager:
       """Single class for both main and worker process."""
       
       @classmethod
       def for_learner(cls, config: OpponentConfig) -> "OpponentManager":
           """Create instance for main process (loads models, tracks wins)."""
           
       @classmethod  
       def for_worker(cls, config: OpponentConfig, ...) -> "OpponentManager":
           """Create instance for worker process (creates players)."""
   ```

8. **Extract trajectory handling**
   ```python
   # rl/trajectory.py
   @dataclass
   class Trajectory:
       states: torch.Tensor
       actions: torch.Tensor
       rewards: torch.Tensor
       ...
   
   def collate_trajectories(trajectories, device, gamma, ...) -> Dict[str, torch.Tensor]
   ```

---

## Recommended Approach

**Start with Option A** (extract constants, server management, checkpointing).

This provides immediate benefits (smaller train.py, clearer separation) with minimal risk. The existing code works, so major restructuring should be done incrementally.

**Then evaluate Option B** after Option A is stable. The key question: does having `BatchInferencePlayer` in a separate file from `RNaDAgent` cause confusion? If engineers frequently need to understand both together, merge them.

**Avoid Option C** unless the current dual-class opponent management becomes a significant source of bugs. The process-boundary constraint is fundamental and won't go away with renaming.

---

## Key Insight: Why Two Opponent Classes?

The fundamental constraint is **Python multiprocessing**:

```
Main Process                          Worker Process
─────────────────                     ─────────────────
OpponentPool                          WorkerOpponentFactory
├─ self.bc_model (torch.nn.Module)    ├─ self.main_agent (RNaDAgent)
├─ self.exploiter_models              ├─ self.bc_agent (RNaDAgent)
└─ win_rate tracking                  └─ self.players (BatchInferencePlayer)
     ↑                                     ↑
     │                                     │
     └─── Cannot be pickled ───────────────┘
```

`torch.nn.Module` objects cannot be efficiently serialized across process boundaries. So:
- **Main process** holds the "source of truth" models and tracks global statistics
- **Worker processes** create their own model copies and manage local battles

This is why we have two classes - it's not redundancy, it's a necessary split due to multiprocessing constraints.
