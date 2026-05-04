# Showdown Single-Team Sample Profile

## Context

The goal of this profiling pass was to produce a fresh, lightweight Showdown-backed Stage 2 RL measurement based on the single-team training shape, then use that sample to reason about the most likely training-speed bottlenecks and the best next optimization targets.

This pass used:

- the current single-team comparison Showdown config at `src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml`
- a short sample config at `src/elitefurretai/rl/configs/single_team_showdown_profile_sample.yaml`
- the maintained websocket benchmark at `src/elitefurretai/engine/analyze/showdown_benchmark.py`

## Before State

Before this pass, the repo already had strong Showdown-focused notes from 2026-04-13 showing that:

- the websocket path was slower than the raw environment ceiling by a large factor once the real policy stack was enabled
- invalid-choice churn and actor-side inference were both material contributors
- learner-facing throughput ramped over time as the actor pipeline filled

However, a fresh sample run was still useful to confirm what remains true in the current code after the recent request snapshot and order-decode fixes.

## Problem

The user wants to reason about realistic paths to a `2x-5x` Showdown training speedup.

That requires separating three different questions:

1. what the raw Showdown environment can do
2. what the model-backed actor loop can do
3. what the full learner loop spends its time on during a representative single-team run

## Solution

### Sample measurements

#### 1. Random-policy Showdown ceiling

Command:

`/home/cayman/Repositories/venv/bin/python -m elitefurretai.engine.analyze.showdown_benchmark --policy random --battles 12 --max-concurrent-battles 4 --port 8110`

Observed:

- `12` battles completed
- `duration_seconds=5.951`
- `battles_per_second=2.016`
- `server_launch_seconds=2.512`
- `player_setup_seconds=1.933`
- `battle_loop_seconds=1.506`

Interpretation:

- even this small run shows the raw websocket path can move far faster than the real training stack
- setup cost is a large fraction of tiny benchmark runs, so longer runs still matter for stable ceilings

#### 2. Model-backed Showdown benchmark

Command:

`/home/cayman/Repositories/venv/bin/python -m elitefurretai.engine.analyze.showdown_benchmark --policy model --config src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml --checkpoint data/models/supervised/curious-darkness-77_best.pt --battles 6 --max-concurrent-battles 4 --batch-size 8 --batch-timeout 0.01 --device cpu --port 8111 --profile-output /tmp/showdown_profile_sample_cpu.txt --profile-top-n 50`

Observed:

- `6` battles completed
- `duration_seconds=16.206`
- `battles_per_second=0.370`
- `cpu_user_seconds=25.261`

Key cumulative hotspots from cProfile:

- `_gpu_inference_sync` and model forward path: about `8.03s`
- `_encode_features`: about `4.79s`
- poke-env message handling (`_handle_message` / `_handle_battle_message`): about `6.64s` combined
- `_embed_battle_state`: about `2.91s`
- `embed_to_array`: about `2.79s`
- transformer encoder layers: about `2.38s`
- `torch._C._nn.linear`: about `2.14s`
- `generate_feature_engineered_features`: about `1.93s`
- `calculate_damage`: about `1.88s`

Interpretation:

- actor-side model execution is still the largest measured hotspot
- websocket / poke-env message handling is still large enough that pure model optimization alone will not deliver a clean `2x-5x`
- embedding and damage calculation remain meaningful secondary costs

#### 3. One-update learner sample run

Command:

`/home/cayman/Repositories/venv/bin/python -m elitefurretai.rl.train --config src/elitefurretai/rl/configs/single_team_showdown_profile_sample.yaml`

Observed from `Update 1`:

- `64` battles in about `62s`
- `1.03 battles/s`
- `636 learner steps`
- `10.26 learner steps/s`
- `1.03 learner trajectories/s`
- win rates during this first sample were reasonable, so the run was not obviously degenerate

Important code-path observation:

- `src/elitefurretai/rl/train.py` computes `time_collecting_battles_pct`, `time_training_pct`, and `time_broadcasting_pct` but only stores them in metrics; they are not printed in the console logger path when wandb is off

Interpretation:

- the first learner update is not screaming “GPU learner bottleneck”; the actor pipeline can feed enough data to keep a single update healthy
- the lack of console visibility into collection-vs-training time is currently slowing down future profiling work more than necessary

### Training-flow observations from code inspection

#### Actor path

- `src/elitefurretai/rl/train.py` hard-pins Showdown workers to `device = "cpu"`, which is the correct default for avoiding worker-side CUDA duplication and startup OOMs
- `src/elitefurretai/rl/players.py` still runs transformer inference per battle in the async path because contexts are handled one battle at a time instead of padded for a real batched transformer forward
- the same file snapshots requests and drops stale results, which improves correctness but can increase wasted actor work when request churn is high

#### Learner path

- `collate_trajectories()` in `src/elitefurretai/rl/train.py` constructs tensors on CPU trajectory-by-trajectory, computes GAE in Python loops, then moves all tensors to the learner device
- `RNaDLearner.update()` in `src/elitefurretai/rl/learners.py` immediately calls `.to(self.device)` on batch tensors again even though collation already placed them on the configured device
- this likely does not dominate wall-clock today, but it is unnecessary data motion and should be cleaned up before larger-scale sweeps

## Reasoning

The strongest current bottleneck picture for Showdown single-team training is:

1. actor-side policy inference and transformer forward cost
2. websocket / poke-env message handling and request churn
3. embedding plus feature engineering plus damage calculation
4. learner-side batching / copying overhead

The fresh sample run supports the same broad conclusion as the earlier April 13 notes: the path to a real `2x-5x` gain is cumulative, and the biggest wins are likely to come from actor-side throughput improvements rather than learner-side math alone.

The most important code-level guess is that the current Showdown transformer path is under-batched in practice. `BatchInferencePlayer._run_batch()` still processes transformer requests one battle at a time because variable-length contexts are not padded into a true batch. That means the configured `batch_size` helps much less than it appears on paper.

## Planned Next Steps/Implementation Plan

1. Expose collection/training/broadcasting percentages in the console log path.
   - This is the fastest way to make future short runs self-diagnosing even with wandb disabled.

2. Measure actual stale-drop and invalid-choice volume after the recent request fixes.
   - If stale drops or invalid retries are still high, reducing that churn should be prioritized ahead of pure model tuning.

3. Investigate true batching for transformer actors in `BatchInferencePlayer`.
   - Padding per-battle contexts into a real batch is a plausible high-leverage improvement because it directly attacks the largest measured hotspot.

4. Profile `Embedder` and damage-calculation cost separately.
   - This remains a realistic `10-25%` class optimization bucket, but probably not the whole `2x-5x` target by itself.

5. Remove redundant learner-side device transfers and CPU-loop overhead in `collate_trajectories()` and `RNaDLearner.update()`.
   - This is probably a second-order win, but it is low-risk cleanup that also improves measurement clarity.

## Updates

### Logging and longer-run follow-up

Completed after the initial sample pass:

- added raw timing totals and per-update timing deltas to the trainer log path
- added worker batch summary and worker phase logs
- added aggregated actor diagnostics for request counts, embedding time, inference wait time, executor time, stale/drop counters, and completed trajectory counts
- created `src/elitefurretai/rl/configs/single_team_showdown_profile_longer.yaml` for a slightly longer concurrent Showdown profile

### Longer run outcome

The longer concurrent trainer run materially sharpened the bottleneck picture.

Update 1 showed:

- total update wall time `289.45s`
- collection `287.87s` (`99.5%`)
- training `1.40s` (`0.5%`)
- `64` trajectories / `584` steps
- `queue_success=64`
- `queue_empty=278`

Update 2 showed:

- total update wall time `325.66s`
- cumulative collection `323.63s`
- training `1.77s`
- another `64` trajectories / `494` steps

### Revised interpretation

The most important new finding is that the integrated longer run is dominated by collection-side starvation rather than learner compute.

Specifically, worker logs repeatedly showed `vgc_bench_baseline` batches with:

- `120s+` await-heavy durations
- `0` completed trajectories
- `0` transferred steps
- `req=0`

That means these batches are not simply slow policy battles. They appear to be failing to reach the point where battle requests are generated at all. In this measured run, this path is a higher-priority fix than pure transformer or embedding optimization, because it consumes concurrency while yielding no training signal.

### New priority order

1. debug and eliminate `vgc_bench_baseline` zero-trajectory timeout batches
2. then optimize actor-side transformer batching / inference throughput
3. then tune embedding and message-handling overhead
4. finally clean up learner-side copy and collation overhead as a second-order improvement