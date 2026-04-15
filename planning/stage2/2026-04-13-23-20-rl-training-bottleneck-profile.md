# RL Training Bottleneck Profile On Current Hardware

## Context

The goal for this pass was to profile current Stage 2 RL training speed on the existing workstation and identify the highest-leverage path to roughly 2x faster training.

This profile used three evidence sources:

- the paired 10-update learner-facing comparison runs already completed for the current Transformer + `full` feature single-team setup
- the maintained Rust model benchmark under the same config family
- direct hardware inspection on the current WSL machine

## Before State

Before this pass, earlier benchmark notes already suggested that actor-side CPU inference was dominant, but we still needed a sharper answer to two practical questions:

1. which metric actually moves on the real learner loop: battles per second, updates per hour, or learner steps per second?
2. what is the realistic path to 2x speed on the current machine without guessing?

## Problem

The current hardware is heavily asymmetric:

- CPU: Intel i7-7700K under WSL, 4 physical cores / 8 logical CPUs
- RAM: about 23 GiB visible to Linux, ~18 GiB available during the measurement
- GPU: training configs target CUDA, but shell-side GPU tooling was not available in this session (`nvidia-smi` missing)

That means the RL system is vulnerable to a bad split where thousands of small Python-side operations and large CPU forward passes dominate trajectory generation while the learner GPU spends much of wall time waiting for data.

## Solution

### 1. Learner-facing 10-update comparison

Using the paired logs:

- Rust backend log: `data/models/rl/single_team_compare_rust_10_updates/nohup_train_2026-04-13-remote.log`
- Showdown backend log: `data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log`

Observed end-of-run metrics:

#### Rust, 10 updates

- `640` battles in `35m 23s`
- `0.30 battles/s`
- `19011 learner steps`
- `8.95 learner steps/s`
- about `29.7 learner steps / battle`

#### Showdown, 10 updates

- `640` battles in `13m 27s`
- `0.79 battles/s`
- `6447 learner steps`
- `7.98 learner steps/s`
- about `10.1 learner steps / battle`

Interpretation:

- Showdown reaches the same 10 updates about `2.63x` faster in wall clock.
- Rust still produces about `1.12x` more learner steps per second.
- Rust also produces about `2.95x` more learner steps per completed battle.

So there is no single scalar notion of "training speed" here. On this setup:

- if the goal is **updates per hour**, Showdown is already the faster backend
- if the goal is **learner steps per second**, Rust is still slightly ahead
- if the goal is **battle throughput**, Showdown is far ahead

### 2. Current Rust hot-path benchmark sweep

Using the maintained benchmark:

- `python -m elitefurretai.engine.rust_model_benchmark --config src/elitefurretai/rl/configs/single_team_compare_rust_10_updates.yaml --checkpoint data/models/supervised/curious-darkness-77_best.pt --battles 6`

Measured results:

#### max_concurrent=1

- `duration_seconds=26.676`
- `battles_per_second=0.225`
- `decisions_per_second=8.285`
- `policy_embed_seconds=1.037`
- `policy_inference_seconds=24.720`
- `engine_step_seconds=0.594`

#### max_concurrent=3

- `duration_seconds=30.931`
- `battles_per_second=0.194`
- `decisions_per_second=7.759`
- `policy_embed_seconds=1.168`
- `policy_inference_seconds=28.655`
- `engine_step_seconds=0.691`
- `p1_rejected_choices=40`

#### max_concurrent=4

- `duration_seconds=83.509`
- `battles_per_second=0.072`
- `truncated_battles=6`
- `decisions_per_second=7.197`
- `policy_embed_seconds=2.890`
- `policy_inference_seconds=78.037`
- `engine_step_seconds=1.647`
- `p1_rejected_choices=25`
- `p2_rejected_choices=26`

Key takeaway from the current run shape:

- actor-side inference is the bottleneck by a wide margin
- at `max_concurrent=1`, inference alone consumed about `92.7%` of wall time
- embedding was only about `3.9%` of wall time in the same run
- engine stepping was only about `2.2%` of wall time
- pushing concurrency higher without fixing inference quality / legality interactions made the run worse, not better

### 3. GPU experiment status

A direct local GPU probe through the Rust sync benchmark failed with a device mismatch in the Transformer hidden-state path:

- `RuntimeError: Expected all tensors to be on the same device ... cuda:0 ... cpu`

This means the local single-process CUDA path is not currently a trustworthy measurement harness for actor inference upside.

Earlier notes already showed that the current centralized GPU transport was slower than the local CPU path because transport overhead dominated. That result still stands as the best available evidence for the current implementation.

## Reasoning

The most important conclusion is that the current machine is not bottlenecked by the Rust simulator itself and not primarily by embedding.

It is bottlenecked by **running a large Transformer actor policy on CPU in the actor loop**.

On a 4-core CPU box, that has several consequences:

1. naive worker or per-worker concurrency increases do not reliably help
2. embedding optimizations are still useful, but they are second-order improvements unless actor inference cost drops first
3. a centralized GPU design only helps if its transport overhead is kept far below the CPU forward cost
4. backend choice changes which metric looks fastest, so recommendations must be tied to the metric we care about

## Planned Next Steps/Implementation Plan

### If the target is 2x faster updates per hour right now

1. Prefer the Showdown backend for current training sweeps on this hardware.
2. Keep the validated batching settings from the paired run (`batch_size=8`, `batch_timeout=0.01`).
3. Measure quality-per-update carefully, because the faster backend is also feeding shorter trajectories.

### If the target is 2x faster learner steps per second on the Rust path

1. Do not expect worker scaling alone to get there on this CPU.
2. Prioritize reducing actor inference cost first:
   - smaller distilled actor model for workers, or
   - a redesigned GPU inference transport with lower overhead than the current pipe-based path
3. Keep embedding work as a secondary pass after inference cost is reduced.

### Concrete engineering follow-ups

1. Add durable learner logs for `time_collecting_battles_pct`, `time_training_pct`, and `time_broadcasting_pct` even when wandb is off, so future runs expose idle-vs-train split directly in the console log.
2. Fix the Rust sync local-CUDA hidden-state device mismatch so we can benchmark the true single-process GPU ceiling cleanly.
3. Treat `max_concurrent=4` in the current Rust benchmark shape as a regression point, not an optimization target.

## Updates

- 2026-04-13 23:20: Confirmed from the paired 10-update comparison that Showdown is already `2.63x` faster in wall-clock time to the same update count, while Rust remains slightly better on learner steps per second.
- 2026-04-13 23:20: Re-ran a short Rust model benchmark sweep on the current config family and confirmed that actor inference is still the dominant hot path by a very large margin.
- 2026-04-13 23:20: Attempted a direct local GPU Rust benchmark probe and hit a real hidden-state device mismatch bug, so local CUDA benchmark numbers remain blocked until that path is fixed.