# Showdown Optimization Sweep Notes

## Context

This document records an April 2026 parameter sweep for the Showdown websocket benchmark after implementing true batched transformer actor inference.

The goal is to maximize actor-side throughput without choosing settings that are likely to destabilize long-running WSL2 training runs.

## Safety Constraints

Hardware and environment constraints that shape the sweep:

- WSL2 with about `24 GB` usable RAM
- `8` CPU threads / `4` physical cores
- RTX `3090`, but learner GPU contention makes actor-on-GPU settings risky for week-long training unless proven clearly better
- prior RL failures included OOM/SIGKILL events under memory pressure, so this sweep should avoid aggressive concurrency settings that could scale badly in full training

Because of those constraints, the initial search is limited to parameters that directly affect the local Showdown actor batching path while staying conservative on memory:

- `batch_size`
- `batch_timeout`
- `max_concurrent_battles`

Parameters intentionally **not** treated as primary sweep dimensions for the first pass:

- `device`
  - CPU is the safer default for long-running training because Showdown actors otherwise compete with the learner GPU
- `temperature`, `top_p`, `greedy`
  - these mostly affect action choice behavior rather than the hot inference/embedding path
- `max_battle_steps`
  - more of a truncation/training-quality parameter than a direct benchmark throughput knob for this search

## Parameter Categories

### Parameters That Could Matter For Speed

1. `batch_size`
   - controls how many pending requests the actor tries to process together
2. `batch_timeout`
   - controls how long the inference loop waits for more requests before flushing a batch
3. `max_concurrent_battles`
   - controls how many battles each benchmarked player runs simultaneously and therefore how many requests are available to fill batches
4. `device`
   - `cpu` versus `cuda`; relevant, but treated as a secondary dimension because of learner contention and long-run stability concerns
5. `feature_set`
   - relevant to speed, but not part of this sweep because it changes semantics and would confound the batching-specific comparison
6. `max_battle_steps`
   - can change battle duration and memory footprint, but not the main target of this actor batching sweep

### Parameters Chosen For This Sweep

The first constrained sweep will test:

- `batch_size ∈ {4, 8, 16}`
- `batch_timeout ∈ {0.003, 0.005, 0.01, 0.02}`
- `max_concurrent_battles ∈ {2, 4}`
- `device = cpu`

Rationale:

- keeps the search space manageable
- avoids obviously risky long-run settings like actor CUDA or very high concurrency on a single Showdown server
- focuses on the parameters that should most directly expose whether the new batching implementation is being used effectively

## Benchmark Methodology

1. Run repeated `20` battle baseline tests with the same parameters to estimate timing variance.
2. Run a constrained `20` battle sweep over the selected parameter grid.
3. Select the strongest candidates from the `20` battle sweep.
4. Re-run the finalists over `100` battles to reduce noise and pick the best stable variant.
5. Run a `cProfile` and a manual timing/profile pass on the winner.

## Baseline For Variance

Baseline settings chosen for repeatability checks:

- `batch_size=8`
- `batch_timeout=0.01`
- `max_concurrent_battles=4`
- `device=cpu`

## Results

### 2026-04-19: `20`-Battle Baseline Variance Check

Repeated baseline settings:

- `batch_size=8`
- `batch_timeout=0.01`
- `max_concurrent_battles=4`
- `device=cpu`
- `battles=20`

Observed across `5` runs:

- `duration_seconds`: mean `32.198`, min `26.470`, max `37.411`
- `battles_per_second`: mean `0.633`, min `0.535`, max `0.756`
- `battle_loop_seconds`: mean `25.764`, min `19.957`, max `30.978`
- `player_setup_seconds`: mean `3.922`

Variance interpretation:

- `battle_loop_seconds` had about `42.8%` relative spread from min to max on this `20` battle shape
- that is too noisy to trust small improvements
- `20` battle runs are still useful as a coarse screening tool, but changes in the `5-15%` range should not be treated as real improvements without a longer rerun

Updated decision rule for the sweep:

1. use `20` battle runs only to eliminate obviously worse settings
2. promote only clearly better candidates to `100` battle reruns
3. choose the winner based primarily on `100` battle `battle_loop_seconds` and `battles_per_second`

### 2026-04-20: Full `20`-Battle CPU Sweep

Completed constrained sweep over:

- `batch_size ∈ {4, 8, 16}`
- `batch_timeout ∈ {0.003, 0.005, 0.01, 0.02}`
- `max_concurrent_battles ∈ {2, 4}`
- `device = cpu`

Best coarse results by `battle_loop_seconds`:

| BS | BT | MCB | Duration | Battles/s | Loop Sec | Setup Sec |
|---|---|---|---|---|---|---|
| 8 | 0.01 | 2 | 23.100 | 0.866 | 16.764 | 3.822 |
| 8 | 0.005 | 4 | 28.319 | 0.706 | 21.863 | 3.940 |
| 4 | 0.01 | 2 | 28.871 | 0.693 | 22.594 | 3.763 |
| 16 | 0.02 | 4 | 29.705 | 0.673 | 22.959 | 4.231 |
| 8 | 0.02 | 4 | 29.723 | 0.673 | 23.242 | 3.964 |

Important interpretation:

- the coarse winner was `batch_size=8`, `batch_timeout=0.01`, `max_concurrent_battles=2`
- `batch_size=16` produced one strong-looking coarse result, but most other `16` runs were substantially worse
- because the `16` region looked less stable and is less attractive for long-running WSL2 safety, it was not promoted ahead of similarly strong `batch_size=8` settings

Promoted to `100`-battle confirmation:

- `batch_size=8`, `batch_timeout=0.01`, `max_concurrent_battles=2`
- `batch_size=8`, `batch_timeout=0.005`, `max_concurrent_battles=4`
- `batch_size=4`, `batch_timeout=0.01`, `max_concurrent_battles=2`
- `batch_size=8`, `batch_timeout=0.02`, `max_concurrent_battles=4`

### 2026-04-20: `100`-Battle Finalists

Confirmed finalists over `100` battles:

| BS | BT | MCB | Duration | Battles/s | Loop Sec | Setup Sec |
|---|---|---|---|---|---|---|
| 8 | 0.02 | 4 | 122.300 | 0.818 | 116.040 | 3.740 |
| 4 | 0.01 | 2 | 130.410 | 0.767 | 124.090 | 3.810 |
| 8 | 0.005 | 4 | 132.020 | 0.757 | 125.760 | 3.750 |
| 8 | 0.01 | 2 | 140.980 | 0.709 | 134.400 | 4.070 |

Winner:

- `batch_size=8`
- `batch_timeout=0.02`
- `max_concurrent_battles=4`

Why this won:

- the longer run reversed the short-run ranking, which confirms the earlier point that `20` battles are only good for coarse filtering
- the winning configuration appears to wait long enough to form materially better batches while still keeping four concurrent battles busy
- `batch_size=8` remained in the best region, while `batch_size=16` was consistently worse and `batch_size=4` left batching performance on the table

Practical takeaway:

- use `batch_size=8`, `batch_timeout=0.02`, `max_concurrent_battles=4` as the current safest high-throughput CPU Showdown actor setting on this WSL2 machine

### 2026-04-20: Winner Profile

Profiled configuration:

- `batch_size=8`
- `batch_timeout=0.02`
- `max_concurrent_battles=4`
- `battles=100`

Non-profiled benchmark result:

- `duration_seconds=122.300`
- `battles_per_second=0.818`
- `player_setup_seconds=3.740`
- `battle_loop_seconds=116.040`

Manual timing interpretation:

- `battle_loop_seconds` accounts for about `94.9%` of total end-to-end runtime
- server launch and player setup are now relatively small fixed costs, so further speedups must mostly come from the battle loop
- the remaining optimization targets are therefore the async websocket wait path, model inference, and embedding/feature engineering

Profiled run note:

- the `cProfile` run was slower (`duration_seconds=147.410`, `battle_loop_seconds=141.080`) because profiling overhead is material at this scale
- use the profiled run to identify hotspots, not to compare throughput directly against non-profiled runs

Top `cProfile` hotspots for the winner:

- `asyncio.base_events._run_once` / `selectors.select`: large cumulative time in the async event loop and socket waiting path
- `rl/players.py:_gpu_inference_sync`: main synchronous inference wrapper used by the actor runtime
- `supervised/model_archs.py:forward_with_hidden`: core batched transformer forward path
- `torch._C._nn.linear`: dominant low-level PyTorch kernel on CPU
- `poke_env.ps_client._handle_message`: websocket message handling cost remains meaningful
- `etl/embedder.py:embed_to_array` and `generate_feature_engineered_features`: embedding and feature engineering remain major overhead
- `poke_env.calc.damage_calc_gen9.calculate_damage`: damage calculation is still one of the most expensive feature-building subroutines
- `supervised/model_archs.py:_dual_expand`: notable architecture-specific model cost

Quantified profile evidence from `/tmp/showdown_profile_winner_8_0.02_4.txt`:

- profiled battle-loop runtime was `141.053s` over `100` battles
- `selectors.select` had `143.321s` inclusive cumulative time and `_run_once` had `207.515s`
  - these are inclusive times in the async scheduler, so they overlap heavily, but they confirm that websocket wait / event-loop coordination is still a first-order part of runtime
- `rl/players.py:_gpu_inference_sync` had `137.219s` inclusive cumulative time (`97.3%` of profiled battle-loop time)
- `supervised/model_archs.py:forward_with_hidden` had `134.816s` inclusive cumulative time (`95.6%`)
- low-level `torch._C._nn.linear` alone consumed `42.791s` cumulative (`30.3%`)
- `poke_env.ps_client._handle_message` consumed `42.519s` cumulative (`30.1%`)
- `rl/players.py:_embed_battle_state` consumed `33.230s` cumulative (`23.6%`)
- `etl/embedder.py:embed_to_array` consumed `31.894s` cumulative (`22.6%`)
- `etl/embedder.py:embed` consumed `27.913s` cumulative (`19.8%`)
- `poke_env.calc.damage_calc_gen9.calculate_damage` consumed `23.953s` cumulative (`17.0%`)
- `etl/embedder.py:generate_feature_engineered_features` consumed `21.598s` cumulative (`15.3%`)
- `supervised/model_archs.py:_dual_expand` consumed `20.313s` cumulative (`14.4%`)

Interpretation of those numbers:

- the cumulative times overlap, so they should not be added together
- they do clearly show that the battle loop is still dominated by three interacting regions:
  - async websocket/event-loop wait and message handling
  - CPU transformer inference
  - embedding plus damage-calculation feature work
- that is why the next optimization pass should stay focused on those three areas rather than spending time on setup, launch, or larger batch-size experiments

Priority optimization directions after batching:

1. reduce embedder and damage-calc work per request
2. cut CPU transformer inference cost in the online actor path
3. reduce websocket/event-loop idle time or improve request accumulation efficiency without regressing latency

## Implementation Ideas

### 1. Reduce Embedder And Damage-Calc Work Per Request

Why this is prioritized:

- `_embed_battle_state` alone accounts for `33.230s` inclusive cumulative time
- `embed_to_array` accounts for `31.894s`
- `generate_feature_engineered_features` accounts for `21.598s`
- `calculate_damage` accounts for `23.953s`

Implementation ideas:

1. Cache per-turn feature subcomputations that do not change across repeated request handling for the same battle state.
   - likely targets: move feature blocks, stat computations, repeated active-pokemon traversals, and deterministic board-level derived values
2. Split the embedder into cheap always-on features and expensive conditional features, then gate the expensive path behind a config flag for runtime benchmarking.
   - this would let training keep `full` features while making it easier to test a cheaper actor-only online feature path
3. Memoize or reuse damage-calculation inputs within a single request.
   - many damage calls differ only by attacker / target / move combinations over a small local state; avoid rebuilding all intermediate values each time
4. Precompute team- and pokemon-static features once per battle and only update the dynamic fields each request.
   - species/item/move-slot encodings should not be rebuilt from scratch on every decision point
5. Add targeted instrumentation around embedder phases.
   - break out timers for `embed_to_array`, `generate_feature_engineered_features`, `calculate_damage`, and move-feature generation so future regressions are attributable

Concrete implementation plan:

1. Add timing counters inside `players.py:_embed_battle_state` and the main embedder phases.
2. Identify repeated subcomputations inside `embed_to_array` and `generate_feature_engineered_features`.
3. Introduce a per-battle request-local cache object owned by the player runtime.
4. Re-run the winner benchmark to verify that reduced embedder cost converts into lower `battle_loop_seconds` rather than just shifting time elsewhere.

### 2. Cut CPU Transformer Inference Cost In The Online Actor Path

Why this is prioritized:

- `_gpu_inference_sync` accounts for `137.219s` inclusive cumulative time
- `forward_with_hidden` accounts for `134.816s`
- `torch._C._nn.linear` alone accounts for `42.791s`
- `_dual_expand` accounts for `20.313s`

Implementation ideas:

1. Reduce avoidable tensor reshaping and expansion work in the online path.
   - `_dual_expand` is large enough to justify inspecting whether tensor duplication or format conversion can be reduced or fused
2. Revisit whether the actor model needs the exact same transformer depth / width as the learner checkpoint for online inference.
   - if a smaller distilled actor or reduced-width inference head can preserve action quality, CPU throughput could improve materially
3. Reduce per-batch padding waste by improving batch formation heuristics.
   - current batching is correct, but smarter grouping by similar context length may reduce wasted transformer work on padded tokens
4. Explore actor-side inference compilation only if it is stable on WSL2.
   - `torch.compile` or similar should be treated as an experiment, not a default, because the current goal is safe long-run training
5. Benchmark lower-precision or quantized CPU inference only as a controlled follow-up.
   - again, only if action quality and runtime stability are acceptable

Concrete implementation plan:

1. Instrument per-batch context statistics further: average real tokens, average padded tokens, and padding ratio.
2. Inspect `_dual_expand` and the immediate pre-forward tensor preparation path for avoidable copies.
3. Add an experiment path that groups pending requests by context-length bucket before forming a batch.
4. Re-run the benchmark on the current winner configuration and compare `battle_loop_seconds`, `forward_with_hidden`, and `torch._C._nn.linear` cumulative time.

### 3. Reduce Websocket/Event-Loop Idle Time Or Improve Request Accumulation Efficiency

Why this is prioritized:

- `selectors.select` shows `143.321s` inclusive cumulative time
- `asyncio.base_events._run_once` shows `207.515s` inclusive cumulative time
- `ps_client._handle_message` still shows `42.519s`

Those numbers overlap, but together they show that even after real model batching, the runtime still spends a large amount of time waiting on or coordinating asynchronous Showdown traffic.

Implementation ideas:

1. Measure batch fill efficiency directly.
   - record actual batch-size histograms, flush reasons, and time spent waiting before flush
2. Tune request accumulation policy using real observed fill rates rather than only benchmark grid search.
   - for example: flush earlier when the queue is cold, but wait longer when the queue is already filling quickly
3. Reduce per-message overhead in request handling.
   - inspect whether `_handle_message` and `_handle_battle_request` are doing avoidable parsing or repeated work before a request reaches batching
4. Consider separating message parsing from expensive state construction more aggressively.
   - the goal would be to let the websocket loop stay responsive while deferring heavier work to the batched decision path
5. Benchmark whether a small increase in concurrent battles beyond `4` is worthwhile only after adding better instrumentation.
   - this should not be the first next step, but may become safe to test once fill-rate and memory behavior are visible

Concrete implementation plan:

1. Add counters for actual formed batch sizes, timeout-triggered flushes, and queue-depth-at-flush.
2. Add timing around `_handle_message`, `_handle_battle_request`, and `_choose_move_async`.
3. Compare the winner against one lower-timeout variant using those counters to see whether the gain comes from better batch fill or less idle waiting.
4. Only after that, decide whether to change the batching policy itself or the number of concurrent battles.
