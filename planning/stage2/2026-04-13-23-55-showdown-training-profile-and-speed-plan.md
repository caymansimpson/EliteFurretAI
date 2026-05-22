# Showdown Training Profile And Speed Plan

## Context

Stage 2 runtime work still needs a clear answer for the websocket-backed RL path: where does wall-clock time actually go on the current hardware, and which changes have a realistic chance of doubling learner-facing training speed.

This profiling pass focused specifically on the Showdown server path used by `battle_backend: showdown_websocket` in the current single-team Transformer configuration.

Hardware verified during this pass:

- CPU: Intel i7-7700K, `4` physical cores / `8` hardware threads
- GPU: NVIDIA GeForce RTX 3090, `24 GB`
- Current Showdown comparison config: `4` workers, `12` players, `4` servers, `batch_size: 8`, `batch_timeout: 0.01`, `max_battle_steps: 40`

## Before State

Before this pass, the repo already had:

- a validated Showdown benchmark harness in [src/elitefurretai/engine/showdown_benchmark.py](src/elitefurretai/engine/showdown_benchmark.py)
- a completed paired 10-update Showdown learner run in [data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log](data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log)
- older notes saying the Showdown path was primarily actor-side CPU inference and asyncio overhead

However, the current runtime had drifted enough that the practical bottleneck picture needed to be rechecked against the live code:

- [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py) currently passes the YAML `device` through worker model construction and into `WorkerOpponentFactory`
- [src/elitefurretai/rl/opponents.py](src/elitefurretai/rl/opponents.py) passes that same device into every `RLTrajectoryPlayer`
- [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py) still runs the battle loop through websocket request handling, embedding, action masking, and batched model inference

So the key question became: is the Showdown bottleneck still mostly model compute, or is the system now dominated by websocket / legality churn and actor orchestration.

## Problem

The user wants to aim for roughly `2x` training speed on the Showdown-backed RL path.

The problem is that multiple costs stack together in this backend:

1. websocket battle/message handling in poke-env
2. Python-side embedding and damage-calculation work
3. actor-side model inference on a `125,612,746` parameter policy (`~0.468 GB` of FP32 weights before activations)
4. repeated invalid-choice retries that burn battle-loop time without producing useful learner signal

Without separating those costs, it is too easy to over-invest in the wrong optimization.

## Solution

### Measured datapoints

#### End-to-end learner-facing Showdown baseline

From [data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log](data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log):

- `Update 10`: `640` battles in `13m 27s`
- `0.79` battles / trajectories per second
- `6447` learner steps total
- `7.98` learner steps / second at the end of the 10-update run

Warmup shape across the same run:

- `Update 1`: `3.33` learner steps / second
- `Update 5`: `5.61` learner steps / second
- `Update 10`: `7.98` learner steps / second

Interpretation:

- the learner is not compute-bound at startup; throughput ramps as the actor pipeline fills
- wall-clock is dominated by how fast the Showdown workers can generate valid trajectories

#### Invalid-choice churn in the real Showdown training run

Total logged legality errors in the completed 10-update Showdown run:

- `24976` `PS_ERROR` lines

Most common invalid-choice families:

- `8134`: Dondozo asked to use `dracometeor`
- `5426`: Dondozo asked to use `muddywater`
- `4900`: `Uproar needs a target`
- `2872`: `You sent more choices than unfainted Pokémon`
- `2666`: Dondozo asked to use `icywind`

Interpretation:

- the Showdown actor path is spending a large amount of wall-clock retrying or recovering from obviously invalid actions
- this is not just cosmetic log noise; it is a training-speed bottleneck because invalid requests lengthen episodes and occupy websocket / inference / embedding time without generating equivalent learner value

#### Websocket environment ceiling versus model-backed throughput

Fresh benchmark on this machine:

- Random policy, `200` battles, `max_concurrent_battles=4`
  - `12.402` battles / second
- Model-backed policy, `12` battles, `max_concurrent_battles=4`, `batch_size=8`, `batch_timeout=0.01`, `device=cpu`
  - `0.424` battles / second

Interpretation:

- the raw Showdown websocket environment can drive roughly `12.4` battles / second on this box
- the real policy stack drops that by about `29x`
- therefore the performance budget is overwhelmingly in policy execution, legality recovery, and request handling, not in server launch or raw simulator stepping

#### CPU versus CUDA actor inference in the Showdown benchmark

Fresh model-backed benchmark, same shape except device:

- CPU actors: `0.424` battles / second, `battle_loop_seconds=23.809`, `cpu_user_seconds=72.371`
- CUDA actors: `0.388` battles / second, `battle_loop_seconds=24.081`, `cpu_user_seconds=25.983`

Interpretation:

- moving the benchmarked Showdown actors from CPU to CUDA did **not** improve throughput on this machine
- CUDA reduced CPU time, but the saved CPU work did not translate into faster battles
- that means the websocket path is bottlenecked by orchestration, synchronization, and/or request churn before raw matrix multiplication speed becomes the dominant limiter

This is important because the current worker bootstrap in [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py) and [src/elitefurretai/rl/opponents.py](src/elitefurretai/rl/opponents.py) allows actor models to follow the same `device` setting as the learner. On a single 3090, that invites contention without proven speed benefit.

#### cProfile hotspots for the current Showdown model path

CPU benchmark profile (`/tmp/showdown_profile_cpu.txt`), `12` battles, `max_concurrent_battles=4`, `batch_size=8`:

- `selectors.select`: `23.053s`
- `RLTrajectoryPlayer._gpu_inference_sync`: `21.925s`
- model forward path: `21.8s+`
- `torch._C._nn.linear`: `7.654s`
- `poke_env.player.Player._handle_battle_message`: `5.731s`
- `RLTrajectoryPlayer._embed_battle_state`: `4.736s`
- `Embedder.embed_to_array`: `4.567s`
- `generate_feature_engineered_features`: `3.025s`
- `calculate_damage`: `3.312s`

CUDA benchmark profile (`/tmp/showdown_profile_cuda.txt`), same shape:

- `selectors.select`: `23.026s`
- `RLTrajectoryPlayer._gpu_inference_sync`: `20.258s`
- `poke_env.player.Player._handle_battle_message`: `5.685s`
- `RLTrajectoryPlayer._embed_battle_state`: `4.905s`
- `Embedder.embed_to_array`: `4.717s`
- `generate_feature_engineered_features`: `3.122s`
- `calculate_damage`: `3.148s`

Interpretation:

- the model path is still expensive, but the event-loop / websocket-request side remains large enough that CUDA actor inference alone does not move battle throughput
- embedding plus engineered feature generation plus damage calculation is still a material secondary hotspot
- the biggest waste unique to the Showdown path is the invalid-choice churn seen in the real training log

#### Small control sweep on batch shape

Fresh short CPU control runs:

- `max_concurrent_battles=1`, `batch_size=4`, `batch_timeout=0.005`: `0.482` battles / second over `8` battles
- `max_concurrent_battles=4`, `batch_size=16`, `batch_timeout=0.01`: `0.497` battles / second over `8` battles

Interpretation:

- once setup noise is included, these short sweeps are close enough that batch-shape tuning is not the path to a clean `2x`
- batching still matters, but the next big win will not come from re-sweeping `8` versus `16` alone

## Reasoning

This helps build the best VGC bot because the training system only gets stronger when wall-clock is turned into valid learner signal rather than invalid websocket retries.

The measured bottleneck order for the current Showdown backend is:

1. invalid-choice churn and the extra battle-loop work it causes
2. websocket / asyncio request handling overhead
3. actor-side policy inference and synchronization
4. embedding, engineered features, and damage calculation
5. minor batch-shape differences within the already-validated range

The CPU-versus-CUDA benchmark result is the key correction from older assumptions. On this machine, simply putting Showdown actors on the 3090 does not increase battles per second. That means a naive “move more actor inference to GPU” plan is not enough for the websocket backend.

If the goal is roughly `2x` faster Showdown training, the path with the best expected return is:

1. remove large classes of invalid Showdown actions first
2. decouple learner device from actor device so the learner can keep the GPU while Showdown actors stay on CPU unless a centralized inference design proves faster
3. reduce embedding / damage-calc cost after the validity bug is fixed

## Planned Next Steps/Implementation Plan

1. Fix invalid action generation before doing more throughput tuning.
   - Prioritize the error families already proven dominant in the real log:
   - stale / impossible move-name selection on Dondozo
   - target selection failures such as `Uproar needs a target`
   - malformed joint orders such as sending too many choices or passing on forced switch
   - likely starting inspection points are [src/elitefurretai/rl/players.py](src/elitefurretai/rl/players.py), [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py), and [src/elitefurretai/etl/encoder.py](src/elitefurretai/etl/encoder.py)

2. Split actor and learner device configuration.
   - Add separate config fields such as `learner_device` and `actor_device`
   - keep learner updates on the 3090
   - default Showdown actors to CPU until a centralized GPU inference path shows real end-to-end wins on the websocket backend

3. Re-profile after the legality fix with the same benchmark and 10-update training shape.
   - success criterion: large drop in `PS_ERROR` volume and higher learner steps / second at the same battle count

4. Optimize the secondary hotspot only after the invalid-choice churn is reduced.
   - target [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py) feature-engineering and damage-calc-heavy paths
   - the current profile suggests this is worth roughly a `10-20%` class improvement, but not the full `2x` by itself

5. Only then revisit websocket batch / concurrency tuning.
   - the existing `batch_size: 8`, `batch_timeout: 0.01`, `4` servers / `4` workers / `12` players setup already appears close enough that it is not the main blocker

## Expected Outcome

The most realistic route to roughly doubling Showdown learner-facing speed on this hardware is cumulative rather than single-knob:

- legality-fix win: likely the biggest gain, potentially `1.3x-1.6x` if it removes substantial retry churn
- actor/learner device split and reduced GPU contention: likely another `1.05x-1.2x` if the current YAML is letting actors compete with the learner on the same GPU
- embedding / feature-engineering optimization: likely `1.1x-1.2x`

Those gains multiply if they stack cleanly. By contrast, another round of batch-size-only tuning is unlikely to produce a reliable `2x` on its own.