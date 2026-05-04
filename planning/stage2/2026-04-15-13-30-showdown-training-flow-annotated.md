# Showdown Training Flow Annotated

## Context

This note records the architecture flow of the Stage 2 single-team Showdown training path after adding broader runtime diagnostics to the trainer, worker loop, and websocket actor path.

The goal was not just to identify percentages, but to pin concrete counts and raw times onto each stage of a representative longer run so future speed work can target the actual limiting stages.

This run used:

- `src/elitefurretai/rl/configs/single_team_showdown_profile_longer.yaml`
- `battle_backend: showdown_websocket`
- `4` Showdown servers
- `4` multiprocessing workers
- `12` total players (`3` per server)
- a two-update learner run with output captured to `/tmp/showdown_profile_longer_train.log`

## Before State

Before this pass:

- the trainer logged only coarse timing buckets internally
- raw timing totals and per-update deltas were not printed in a way that made short or medium runs easy to interpret offline
- worker activity was largely opaque once execution entered the websocket battle loop
- actor-side request, embedding, inference, and stale/drop counters were not aggregated per worker batch

That made it too easy to see that training was slow without being able to state exactly where wall-clock was going.

## Problem

The user asked for three things:

1. add broader logging, including raw times rather than percentages alone
2. reprofile with a slightly longer concurrent Showdown run
3. create a markdown description of the architectural training flow annotated with counts and timings

The central profiling question was: which stage is actually dominant in the current Showdown path, and which subpath most likely blocks a `2x-5x` speedup?

## Solution

### Instrumentation added

The following runtime surfaces were added:

- learner timing totals and per-update timing deltas in `src/elitefurretai/rl/train.py`
- learner dataflow logging for trajectories, steps, average battle length, queue successes, and queue timeouts in `src/elitefurretai/rl/train.py`
- worker phase timing logs in `src/elitefurretai/rl/train.py`
- aggregated actor diagnostics in `src/elitefurretai/rl/players.py`
- worker-level aggregation of player and opponent diagnostics in `src/elitefurretai/rl/opponents.py`

### Annotated architecture flow

#### 1. Config and process startup

Flow:

- load RL config
- load main model and BC reference assets
- launch local Showdown servers
- start multiprocessing workers
- connect websocket players and opponents

Measured run facts:

- `4` Showdown servers launched on ports `8000-8003`
- `4` workers started
- server allocation was balanced at `3/3` players per server

Interpretation:

- startup cost exists, but the longer run confirms it is not the main wall-clock driver once updates begin

#### 2. Worker batch preparation

Flow:

- each worker samples opponents/tasks
- worker prepares battle tasks and runtime objects
- runtime diagnostics begin accumulating per player and opponent

Measured worker phase bucket:

- `prepare_tasks_seconds` is logged, but in representative successful batches this phase is tiny relative to battle waiting time

Interpretation:

- task preparation is not currently the main bottleneck

#### 3. Battle launch and websocket wait loop

Flow:

- workers schedule battle tasks
- players wait for Showdown battle requests
- actor requests are embedded, forwarded through model inference, and sent back as choices
- completed trajectories are buffered until worker transfer

This stage is where the run spent almost all of its wall-clock.

Representative successful worker batches:

- self-play example:
  - sampled `{'self_play': 1}`
  - completed `{'self_play': 4}`
  - transferred `4` trajectories / `27` steps
  - batch duration `5.96s`
  - await time `5.96s`
  - request count `97`
  - inference executor time `5.78s`

- max-damage example:
  - sampled `{'max_damage': 1}`
  - completed `{'max_damage': 4}`
  - transferred `4` trajectories / `60` steps
  - batch duration `6.35s`
  - await time `6.34s`
  - request count `60`
  - inference executor time `4.12s`

Later successful self-play batches also showed:

- request counts like `81` and `128`
- inference executor time around `3.51s` to `7.24s`
- transferred `4` trajectories with step counts ranging from `14` to `64`

Interpretation:

- for batches that actually run, websocket wait time and actor-side inference time are tightly coupled
- actor inference remains a real cost center, but these successful batches are still short enough that they are not the catastrophic throughput killer in this run

#### 4. VGC benchmark stall path

Flow:

- some worker batches sample `vgc_bench_baseline`
- those batches enter the wait loop and fail to generate usable trajectories before timing out

Representative failed batches:

- Worker 3 batch 1:
  - sampled `{'vgc_bench_baseline': 1}`
  - completed `{}`
  - transferred `0` trajectories / `0` steps
  - duration `120.11s`
  - timeout rate `100.0%`
  - zero-output rate `100.0%`
  - await time `120.10s`
  - request count `0`

- Worker 0 batch 1:
  - sampled `{'vgc_bench_baseline': 1}`
  - completed `{}`
  - transferred `0` trajectories / `0` steps
  - duration `120.10s`
  - timeout rate `100.0%`
  - zero-output rate `100.0%`
  - await time `120.10s`
  - request count `0`

Repeated pattern later in the run:

- more `vgc_bench_baseline` batches timed out in the `120-150s` range
- these continued to show zero requests and zero transferred trajectories

Interpretation:

- this is the dominant pathological subpath in the measured run
- these batches are not merely slow; they are starving the learner by occupying workers while producing no requests, no actions, and no trajectories
- because `req=0`, the evidence points to battle initiation / integration failure rather than a slow actor-policy path inside the battle

#### 5. Trajectory transfer back to learner

Flow:

- completed worker trajectories are pushed to the multiprocessing queue
- learner drains queue until it has enough trajectories for the update

Measured learner dataflow for update 1:

- `traj/update=64`
- `steps/update=584`
- `avg_steps/traj=9.12`
- `avg_battle_len=9.12`
- `forfeits=0`
- `queue_success=64`
- `queue_empty=278`

Measured learner dataflow for update 2:

- `traj/update=64`
- `steps/update=494`
- `avg_steps/traj=7.72`
- `forfeits=0`

Interpretation:

- the learner is frequently polling an empty queue while workers are blocked elsewhere
- the queue is not the root bottleneck; the queue is exposing upstream starvation

#### 6. Learner collation and update

Flow:

- collected trajectories are collated into padded tensors
- learner computes PPO/RNaD update
- metrics are logged

Measured update 1 timing:

- total update wall time `289.45s`
- collection `287.87s` (`99.5%`)
- training `1.40s` (`0.5%`)
- collation about `0.09s`

Measured update 2 timing:

- total update wall time `325.66s`
- cumulative collection `323.63s`
- this-update collection `35.77s`
- training `1.77s` (`0.5%`)

Measured learner metrics:

- update 1 total battles `64`
- update 2 total battles `128`
- update 1 loss `2.9800`
- update 2 loss `3.1021`

Interpretation:

- the learner is decisively not the wall-clock bottleneck in this sample
- moving learner math alone will not produce a meaningful end-to-end speedup until the collection side is fixed

#### 7. Shutdown behavior

Flow:

- once the configured updates complete, the main process attempts worker shutdown
- workers still in blocked waits are terminated
- final checkpoint is saved

Measured shutdown facts:

- all `4` workers were still running and had to be terminated
- final checkpoint saved to `data/models/rl/single_team_showdown_profile_longer/main_model_step_2.pt`

Interpretation:

- shutdown friction is another symptom of workers sitting in long-lived blocked battle waits

### Supporting benchmark evidence

The separate model-backed Showdown benchmark from the same profiling thread still points to the expected secondary hotspots once battles are actually active:

- `selectors.select` about `20.27s`
- `_gpu_inference_sync` about `18.75s`
- transformer forward path about `18.7s`
- `_encode_features` about `11.70s`
- `torch.linear` about `6.65s`
- `SetTransformer.forward` about `6.59s`
- `TransformerEncoder.forward` about `6.05s`
- `_choose_move_async` about `4.10s`
- `embed_to_array` about `3.58s`
- `calculate_damage` about `2.68s`

Interpretation:

- after the zero-trajectory stall path is fixed, actor inference, feature encoding, and websocket/message overhead are still the most likely next optimization buckets

## Reasoning

This annotated flow changes the prioritization in an important way.

If we only looked at model-backed microbenchmarks, we would prioritize transformer batching, embedding reductions, and message overhead. Those are still valid and likely necessary for large gains.

But the longer trainer run shows that, in the current integrated single-team Showdown path, the first-order bottleneck is even earlier: some opponent batches never reach the request/inference stage at all. A worker stuck for `120s` with `req=0` is worse than a worker spending `6s` on heavy inference, because the former produces no learning signal while still consuming concurrency.

For building the best VGC bot, this means the correct order of attack is:

1. eliminate zero-trajectory timeout paths, especially `vgc_bench_baseline`
2. then improve actor throughput on the batches that do run
3. only after that revisit learner-side micro-optimizations as cleanup rather than mainline acceleration

## Planned Next Steps/Implementation Plan

1. Debug the `vgc_bench_baseline` integration path specifically.
   - The strongest clue is `req=0`, which suggests battle startup or handshake failure rather than slow policy execution.

2. Add explicit per-opponent startup and first-request timestamps.
   - This will separate “battle never started” from “battle started but never produced a decision request.”

3. Keep the new worker and learner logging enabled for future sweeps.
   - These logs are now sufficient to compare concurrency changes without relying on ad hoc transcript scraping.

4. Once timeout batches are removed, re-run the same two-update profile.
   - That will give a cleaner post-fix breakdown of actor inference, websocket overhead, and embedding cost.

5. After collection stalls are fixed, investigate true batching for transformer actor inference.
   - The microbenchmark evidence still suggests this remains a high-leverage speedup candidate.

## Updates

- Added raw timing, per-update timing, and worker-phase logs to the trainer.
- Added aggregated actor diagnostics to the websocket player path.
- Reprofiled the Showdown single-team path with a two-update concurrent run.
- Confirmed that the dominant bottleneck in this run is collection-side worker starvation caused by repeated `vgc_bench_baseline` timeout batches.
- Traced the immediate root cause of the `vgc_bench_baseline` stalls: the profiling configs still set `external_vgcbench_usernames: [VGCBENCH]` while `auto_launch_external_vgcbench` was false, so workers sent `/challenge VGCBENCH` to a username that was never logged in. The trainer log showed `|popup|The user 'VGCBENCH' was not found.`
- Fixed the profiling configs by setting `external_vgcbench_usernames: null` and added a trainer warning when external usernames are configured without auto-launch, because that mode requires a manually started external runner.
- Ran a direct A/B comparison of the two-update Showdown profile with and without the in-process `vgc_bench_baseline`. The VGC-bench-enabled run completed only after using an absolute temporary checkpoint path workaround for the current SB3 loader/path bug. Even with the checkpoint loading, the `vgc_bench_baseline` path remained pathological: all sampled VGC-bench batches timed out at roughly `120s` with `0` completed games and `0` transferred trajectories.
- Quantitatively, the VGC-bench-enabled run took about `333s` for two updates versus about `93s` without VGC bench, or roughly `3.6x` slower end-to-end. Collection stayed dominant in both cases, but throughput dropped from about `13.11` steps/s without VGC bench to about `3.32` steps/s with VGC bench.