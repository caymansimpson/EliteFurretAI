# RL Training System for VGC Doubles

This directory contains a complete reinforcement learning training system for Pokemon VGC doubles battles, with support for behavioral cloning initialization and memory-efficient parallel training.

## Architecture Overview

The system uses a distributed actor-learner architecture:

- **Learner**: Central process that updates the model using PPO on a GPU (if available)
- **Workers**: Multiple parallel processes that collect battle experience using CPUs
- **Communication**: Queues for sharing model weights and experience data
- **Memory**: File-system based tensor sharing for WSL/memory-constrained environments

## Key Features

### 1. **MDBO Action Space (2025 discrete actions)**
- Comprehensive encoding of all move/switch/target/gimmick combinations
- Integrated action masking for valid moves only
- Compatible with poke-env's DoubleBattle format

### 2. **Behavioral Cloning Warm-Start**
- Initialize RL training from pretrained BC models
- Leverages human gameplay data for faster convergence
- Optional freezing of shared layers

### 3. **Memory-Efficient Design**
- Configurable buffer sizes per worker
- File-system tensor sharing for WSL compatibility
- Periodic checkpoint saving
- Limited disk space usage (~200GB safe)

### 4. **Flexible Configuration**
- Adjustable number of workers (recommended: 4-6 for 8-core systems)
- Configurable learning rates and PPO hyperparameters
- Support for different battle formats

## Components

### Files

- **`environment.py`**: VGC doubles gym environment with MDBO and Embedder integration
- **`agent.py`**: ActorCritic model with action masking and BC initialization
- **`learner.py`**: PPO learner with experience aggregation and logging
- **`worker.py`**: Async worker for collecting battle episodes
- **`memory.py`**: Experience buffer with GAE and disk checkpointing
- **`simple_train.py`**: Training script with hardcoded teams for testing
- **`train.py`**: Main training entry point (legacy, use simple_train.py)

### Unit Tests

- **`unit_tests/rl/test_environment.py`**: Environment tests
- **`unit_tests/rl/test_agent.py`**: Model tests
- **`unit_tests/rl/test_memory.py`**: Buffer tests
- **`unit_tests/rl/test_mdbo_integration.py`**: MDBO encoding tests

## Usage

### Basic Training

Start training with default settings (4 workers, no BC initialization):

```bash
python -m elitefurretai.rl.simple_train --num-workers 4
```

### Training with Behavioral Cloning

Initialize from a pretrained BC model:

```bash
python -m elitefurretai.rl.simple_train \
    --num-workers 4 \
    --bc-model data/models/your_bc_model.pt \
    --lr 1e-4
```

### Custom Configuration

```bash
python -m elitefurretai.rl.simple_train \
    --num-workers 6 \
    --buffer-size 4096 \
    --battle-format gen9vgc2023regulationc \
    --checkpoint-dir ./my_checkpoints \
    --lr 3e-4
```

### Resource-Constrained Setup (WSL / 8 cores)

```bash
python -m elitefurretai.rl.simple_train \
    --num-workers 4 \
    --buffer-size 1024 \
    --checkpoint-dir /mnt/c/checkpoints
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workers` | 4 | Number of parallel worker processes |
| `--battle-format` | gen9vgc2023regulationc | Pokemon Showdown format |
| `--buffer-size` | 2048 | Experience buffer size per worker |
| `--bc-model` | None | Path to pretrained BC model |
| `--lr` | 3e-4 | Learning rate |
| `--checkpoint-dir` | ./checkpoints | Directory for checkpoints |

## PPO Hyperparameters

Configured in `simple_train.py`:

```python
'gamma': 0.99,            # Discount factor
'gae_lambda': 0.95,       # GAE lambda
'ppo_epochs': 10,         # Epochs per update
'num_minibatches': 32,    # Minibatches per epoch
'clip_coef': 0.2,         # PPO clipping coefficient
'ent_coef': 0.01,         # Entropy regularization
'vf_coef': 0.5,           # Value function coefficient
```

## Running Unit Tests

Test the entire RL system:

```bash
# Run all RL tests
pytest unit_tests/rl/ -v

# Run specific test file
pytest unit_tests/rl/test_environment.py -v

# Run with coverage
pytest unit_tests/rl/ --cov=elitefurretai.rl --cov-report=html
```

## Workflow

### 1. **Training Loop**

```
Learner initializes model (optionally from BC)
    ↓
Learner distributes initial weights to workers
    ↓
Workers collect battle episodes in parallel
    ↓
Workers send experience batches to learner
    ↓
Learner performs PPO updates
    ↓
Learner distributes new weights
    ↓
(Repeat)
```

### 2. **Worker Process**

```
Initialize environment and opponent
    ↓
Sync weights from learner
    ↓
Run battle episodes until buffer full
    ↓
Calculate GAE advantages
    ↓
Send experience to learner
    ↓
Clear buffer and repeat
```

### 3. **Experience Flow**

```
Battle State → Embedder → Features (observation)
    ↓
Features → Actor → Action (with masking)
    ↓
Action → Environment → Reward
    ↓
Store (state, action, reward, value, log_prob)
    ↓
After rollout: Calculate GAE advantages
    ↓
Send to learner for PPO update
```

## Memory Management

### RAM Usage Estimate

- **Learner**: ~2GB (model + optimizer + batch)
- **Worker**: ~500MB each (environment + local model + buffer)
- **Total for 4 workers**: ~4GB RAM

### Disk Usage

- Checkpoints: ~100MB per saved model
- Final model: ~100MB
- Total with periodic saves: ~500MB - 1GB

### Tips for Memory-Constrained Systems

1. **Reduce buffer size**: Use `--buffer-size 1024` or `--buffer-size 512`
2. **Fewer workers**: Use `--num-workers 2` or `--num-workers 3`
3. **Disable action mask storage**: Modify memory.py if needed
4. **Use CPU-only**: PPO learner will automatically use CPU if no GPU

## Troubleshooting

### Common Issues

**Issue**: Workers not starting
- **Solution**: Check that Pokemon Showdown server is running locally
- **Solution**: Verify battle format is valid

**Issue**: Out of memory
- **Solution**: Reduce `--num-workers` or `--buffer-size`
- **Solution**: Enable file-system sharing (automatically done)

**Issue**: Slow training
- **Solution**: Increase `--num-workers` if you have CPU headroom
- **Solution**: Use GPU for learner
- **Solution**: Reduce `ppo_epochs` or `num_minibatches`

**Issue**: BC model fails to load
- **Solution**: Check model architecture compatibility
- **Solution**: Ensure model file exists and is readable

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor worker output:

```bash
python -m elitefurretai.rl.simple_train --num-workers 1  # Single worker for debugging
```

## Integration with Behavioral Cloning

### How BC Initialization Works

1. BC model (`TwoHeadedHybridModel`) is loaded from disk
2. Compatible weights (action head) are extracted
3. RL model's actor head is initialized with BC weights
4. Value head (critic) is randomly initialized
5. Training continues with RL updates

### Benefits

- **Faster convergence**: Start with reasonable policy
- **Better exploration**: BC provides good action distribution
- **Reduced variance**: Less random initial behavior
- **Data efficiency**: Leverage existing human gameplay data

### Limitations

- Architecture must be compatible (same action space)
- BC model's feed-forward layers don't directly transfer (different architecture)
- Value function still needs to be learned from scratch

## Performance Expectations

### On 8-core CPU with 4 workers

- **Steps/sec**: ~50-100 (depending on battle complexity)
- **Episodes/hour**: ~500-1000
- **GPU speedup**: 2-3x faster learner updates

### Convergence

- **Random initialization**: 10K-50K episodes to see improvement
- **BC initialization**: 1K-10K episodes to see improvement
- **Human-level play**: 100K+ episodes (estimated)

## Next Steps

1. **Monitor training**: Check win rates, rewards, and loss metrics
2. **Tune hyperparameters**: Adjust learning rate, clip coefficient
3. **Add logging**: Integrate wandb or tensorboard
4. **Evaluate**: Test against different opponents
5. **Scale up**: Increase workers if training is stable

## References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **poke-env**: [Documentation](https://poke-env.readthedocs.io/)

## Support

For issues or questions:
1. Check unit tests for usage examples
2. Review code comments in each module
3. Open an issue with detailed error messages
