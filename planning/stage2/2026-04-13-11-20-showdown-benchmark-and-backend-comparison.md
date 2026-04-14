# Showdown Benchmark And Backend Comparison

## Context

Stage 2 backend work needed a checked-in Showdown benchmark path so Rust and websocket execution could be compared without going through the full learner loop. The immediate goal was to finish a clean apples-to-apples benchmark harness for both:

- pure random-policy engine throughput
- model-backed self-play throughput through the real action-selection stack

The user also asked for Showdown profiling and for two benchmark comparison modes:

1. a simple straightforward run
2. a maxed-out run using the best-performing parameters we could verify quickly

## Before State

Before this pass:

- Rust benchmark entrypoints already existed in [src/elitefurretai/engine/rust_engine_benchmark.py](src/elitefurretai/engine/rust_engine_benchmark.py) and [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py)
- the new checked-in Showdown benchmark existed in [src/elitefurretai/engine/showdown_benchmark.py](src/elitefurretai/engine/showdown_benchmark.py), but it had not been validated end to end
- the first Showdown smoke runs appeared to hang after server launch, so the harness itself was still suspect

## Problem

We needed to answer four concrete questions:

1. Is the checked-in Showdown benchmark runner correct and stable?
2. What do Rust and Showdown look like in a simple one-concurrency comparison?
3. Which parameters actually improve throughput enough to justify a maxed comparison?
4. Where is Showdown spending its time during model-backed play?

## Solution

### Harness fix kept

The Showdown benchmark hang was caused by awaiting `player.ps_client.logged_in.wait()` directly from the local `asyncio.run(...)` loop even though the event is created in poke-env's separate `POKE_LOOP`.

Kept fix in [src/elitefurretai/engine/showdown_benchmark.py](src/elitefurretai/engine/showdown_benchmark.py):

- replace direct `logged_in.wait()` awaits with `await player.ps_client.wait_for_login()`

That preserves the intended semantics while avoiding cross-loop event waits.

### Benchmark result summary

Simple runs:

#### Random, simple
- Rust: `500 battles`, `max_concurrent=1`
  - `duration_seconds=41.483`
  - `battles_per_second=12.053`
- Showdown: `500 battles`, `max_concurrent_battles=1`
  - `duration_seconds=44.974`
  - `battles_per_second=11.117`

#### Model-backed, simple
- Rust: `10 battles`, `max_concurrent=1`
  - `duration_seconds=25.460`
  - `battles_per_second=0.393`
  - profiled Rust path was dominated by `policy_inference_seconds=22.867`
- Showdown: `10 battles`, `max_concurrent_battles=1`, `batch_size=4`, `batch_timeout=0.005`
  - `duration_seconds=24.242`
  - `battles_per_second=0.413`
  - cProfile artifact: `/tmp/showdown_model_simple_profile.txt`

### Parameter sweep results

#### Random sweep

Rust random sweep, `200 battles`:

| Backend | Concurrency | Battles/s |
| --- | ---: | ---: |
| Rust | 1 | `12.053` (from 500-battle simple run) |
| Rust | 2 | `8.847` |
| Rust | 4 | `9.745` |
| Rust | 8 | `9.544` |

Showdown random sweep, `200 battles`:

| Backend | Concurrency | Battles/s |
| --- | ---: | ---: |
| Showdown | 1 | `11.117` (from 500-battle simple run) |
| Showdown | 2 | `11.630` |
| Showdown | 4 | `12.117` |
| Showdown | 8 | `12.012` |

Conclusion:

- Rust random benchmark did **not** benefit from additional concurrent battles in this harness; `1` remained the best tested point.
- Showdown random benchmark did benefit from overlap, peaking at `4` among tested values.

#### Model-backed sweep

Rust model sweep, `6 battles`:

| Backend | Concurrency | Battles/s |
| --- | ---: | ---: |
| Rust | 1 | `0.393` (from 10-battle simple run) |
| Rust | 2 | `0.753` |
| Rust | 3 | `0.851` |
| Rust | 4 | `0.623` |

Showdown model sweep, `6 battles`:

| Backend | Concurrency | Batch Size | Batch Timeout | Battles/s |
| --- | ---: | ---: | ---: | ---: |
| Showdown | 1 | 4 | 0.005 | `0.413` (from 10-battle simple run) |
| Showdown | 2 | 4 | 0.005 | `0.297` |
| Showdown | 3 | 4 | 0.005 | `0.327` |
| Showdown | 4 | 4 | 0.005 | `0.367` |
| Showdown | 4 | 8 | 0.01 | `0.414` |
| Showdown | 4 | 16 | 0.01 | `0.363` |

Conclusion:

- Rust model-backed throughput peaked at `max_concurrent=3` among tested values.
- Showdown model-backed throughput improved again when batching was relaxed to `batch_size=8`, `batch_timeout=0.01` at concurrency `4`.

### Maxed comparison runs

#### Random, maxed
- Rust best tested setting remained the same as the simple run:
  - `max_concurrent=1`
  - `battles_per_second=12.053`
- Showdown maxed run:
  - `500 battles`
  - `max_concurrent_battles=4`
  - `duration_seconds=33.122`
  - `battles_per_second=15.096`

Interpretation:

- For pure random-policy execution, Showdown's websocket overlap can outperform the current synchronous Rust random harness once several battles are in flight.
- This reinforces the earlier conclusion that pure random throughput is not the right proxy for learner-facing training throughput.

#### Model-backed, maxed
- Rust maxed run:
  - `12 battles`
  - `max_concurrent=3`
  - `duration_seconds=15.876`
  - `battles_per_second=0.756`
- Showdown maxed run:
  - `12 battles`
  - `max_concurrent_battles=4`
  - `batch_size=8`
  - `batch_timeout=0.01`
  - `duration_seconds=23.601`
  - `battles_per_second=0.508`
  - cProfile artifact: `/tmp/showdown_model_maxed_profile.txt`

Interpretation:

- Once the real model path is exercised, Rust regained a clear throughput lead.
- The maxed model-backed comparison is the important one for Stage 2 runtime decisions because it includes embedding, inference, and action selection rather than just environment stepping.

## Reasoning

The benchmark results support a stronger version of the earlier architectural argument:

- Showdown can look strong in random-policy runs because websocket overlap hides a lot of the environment cost.
- Rust looks stronger when the benchmark path includes real policy work because the Python-side synchronous self-play path can batch model calls effectively without websocket transport overhead and without the websocket-side action-selection path dominating wall time.

The profiling also shows that Showdown model-backed time is still dominated by model execution and event-loop waiting rather than by one exotic hotspot. That means the biggest practical knobs are still batching and concurrency, not a one-line websocket fix.

## Showdown profiling notes

Simple Showdown model profile (`/tmp/showdown_model_simple_profile.txt`):

- top cumulative frames were dominated by:
  - `asyncio.base_events._run_once`
  - `selectors.select`
  - `elitefurretai.rl.players._gpu_inference_sync`
  - `torch` forward path in [src/elitefurretai/supervised/model_archs.py](src/elitefurretai/supervised/model_archs.py)

Maxed Showdown model profile (`/tmp/showdown_model_maxed_profile.txt`):

- same dominant shape, but with fewer inference calls and better aggregate throughput:
  - `208` profiled inference calls in the maxed profile versus `338` in the simple profile
  - more work per inference batch is what improved wall-clock throughput

That matches the benchmark sweep result: the Showdown model path benefits when concurrency and batch sizing allow fewer, larger inference batches.

## Planned Next Steps

1. Keep [src/elitefurretai/engine/showdown_benchmark.py](src/elitefurretai/engine/showdown_benchmark.py) as the checked-in websocket comparison harness.
2. Prefer model-backed benchmark results over random-policy benchmark results when deciding backend priorities.
3. If we want more Showdown speed, the next knobs worth sweeping are player count and batch-shape interactions, not `asyncio` rewrites by themselves.
4. If we want a more apples-to-apples actor-runtime comparison, add learner-step-style metrics to both benchmark harnesses rather than only battles/sec.

## Updates

- 2026-04-13 11:20: Fixed the Showdown benchmark login stall by switching the harness from direct cross-loop `logged_in.wait()` awaits to `PSClient.wait_for_login()`.
- 2026-04-13 11:20: Validated [src/elitefurretai/engine/showdown_benchmark.py](src/elitefurretai/engine/showdown_benchmark.py) in both random and model-backed smoke modes.
- 2026-04-13 11:20: Recorded simple and maxed benchmark comparisons for both random and model-backed paths, plus Showdown cProfile outputs at:
  - `/tmp/showdown_model_simple_profile.txt`
  - `/tmp/showdown_model_maxed_profile.txt`
- 2026-04-13 20:10: Rebased the Rust model benchmark on the current Transformer + `full` feature path and fixed the missing Rust-side Transformer batching.
  - Root cause before the fix:
    - [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) forced all Transformer snapshots through per-snapshot inference instead of the batched path.
    - [src/elitefurretai/rl/central_inference.py](src/elitefurretai/rl/central_inference.py) also processed Transformer requests one-by-one in the central server.
  - Fix landed:
    - Rust sync inference now batches Transformer requests when their context lengths match.
    - The central inference server now batches Transformer requests by shared context length instead of handling them individually.
    - Added regression coverage in [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) for equal-length and mixed-length Transformer batches.
  - Reprofiled Rust Transformer benchmark, `12 battles`, `max_concurrent=3`, `device=cpu`, same shape as the earlier regression run:
    - before fix: `duration_seconds=53.148`, `battles_per_second=0.226`, `decisions_per_second=7.639`, `policy_inference_seconds=49.326`
    - after fix: `duration_seconds=51.011`, `battles_per_second=0.235`, `decisions_per_second=9.331`, `policy_inference_seconds=46.622`
  - Interpretation:
    - the batching fix is real but the gain is modest at battle-level throughput (`0.226 -> 0.235`, about `+4%`)
    - the better signal is decisions/s (`7.639 -> 9.331`, about `+22%`) and reduced pure inference time (`49.326s -> 46.622s`)
    - battle throughput under this benchmark shape is still distorted by battle-quality effects such as rejected choices and longer trajectories
  - Additional probe, `max_concurrent=4`, `device=cpu`:
    - `duration_seconds=75.992`, `battles_per_second=0.158`, `truncated_battles=5`
    - `p1_rejected_choices=64`, `p2_rejected_choices=38`
    - this is worse than the 3-concurrent point and shows the next limit is not just raw forward speed; action quality / legality drift starts to dominate when more battles are in flight
  - Central GPU inference probe after restoring Transformer batching, `12 battles`, `max_concurrent=3`, `--use-central-gpu-inference --inference-device cuda`:
    - `duration_seconds=58.017`, `battles_per_second=0.207`, `policy_inference_seconds=53.004`
    - this remains slower than the local CPU Transformer path, which confirms the transport layer is still the dominant blocker for the centralized GPU design even after batching is fixed
  - Updated backend takeaway:
    - the earlier Rust-vs-Showdown Transformer gap was partly unfair because Rust had a missing batching path
    - after fixing batching, Rust still does not become dramatically faster overall because model inference remains dominant and same-length Transformer batching opportunities are limited once concurrent battles diverge in turn count
 - 2026-04-13 15:56: Created paired `train.py` configs for a learner-facing 10-update comparison on the current Transformer + `full` single-team setup.
   - Rust config: [src/elitefurretai/rl/configs/single_team_compare_rust_10_updates.yaml](src/elitefurretai/rl/configs/single_team_compare_rust_10_updates.yaml)
   - Showdown config: [src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml](src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml)
   - Shared comparison shape:
     - `max_updates: 10`
     - `train_batch_size: 64`
     - current Transformer architecture + `full` feature set + `curious-darkness-77_best.pt` initialization
     - `num_workers: 4`, `num_players: 12`, `max_battle_steps: 40`
   - Backend-specific choice kept from the latest evidence:
     - Rust stays on local actor inference (`use_central_gpu_inference: false`)
     - Showdown uses the best validated websocket batching settings so far: `batch_size: 8`, `batch_timeout: 0.01`
 - 2026-04-13 15:56: User needed the work to continue remotely after ending the local session, so the active Rust 10-update comparison was relaunched under `nohup`.
   - active PID at handoff time: `19843`
   - command:
     - `/home/cayman/Repositories/venv/bin/python -m elitefurretai.rl.train --config src/elitefurretai/rl/configs/single_team_compare_rust_10_updates.yaml`
   - durable log:
     - `data/models/rl/single_team_compare_rust_10_updates/nohup_train_2026-04-13-remote.log`
   - startup confirmation in the log showed all four Rust workers launching successfully.
   - pending next step once the Rust run finishes:
     - launch the paired Showdown run with [src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml](src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml) and compare learner-facing throughput / stability across the two 10-update runs.
  - 2026-04-13 21:48: Resumed the paired 10-update learner comparison from the nohup handoff.
    - Rust run completed cleanly before resume; the original nohup PID `19843` had exited.
    - Rust final checkpoint:
      - `data/models/rl/single_team_compare_rust_10_updates/main_model_step_10.pt`
    - Rust end-of-run learner-facing metrics from `data/models/rl/single_team_compare_rust_10_updates/nohup_train_2026-04-13-remote.log`:
      - `Update 10`
      - `Total Battles=640 in 0h 35m 23s (0.30 b/s)`
      - `Learner Steps=19011 (8.95 steps/s)`
      - `Learner Trajectories=640 (0.30 traj/s)`
      - `Win rates: self_play: 61.0% | bc_player: 46.6%`
    - Rust shutdown behavior at completion:
      - trainer saved the final model and then force-terminated still-running workers during shutdown after the 5-second join window
      - this did not prevent checkpoint save or normal run completion, but it is worth keeping in mind when comparing clean-stop behavior across backends
    - Launched the paired Showdown run under `nohup` with:
      - `/home/cayman/Repositories/venv/bin/python -m elitefurretai.rl.train --config src/elitefurretai/rl/configs/single_team_compare_showdown_10_updates.yaml`
    - Active Showdown trainer PID at resume time:
      - `24526`
    - Durable Showdown log:
      - `data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log`
    - Startup confirmation captured at resume time:
      - all four Showdown servers launched on ports `8000-8003`
      - all four multiprocessing workers started successfully on those ports
      - websocket clients were already logging `updatesearch` / `updateuser` traffic, indicating the paired run had progressed past bare server boot
    - Pending next step from here:
      - once the Showdown 10-update run finishes, extract the same learner-facing metrics and compare Rust vs Showdown on throughput, truncation/stability signals, and shutdown behavior
  - 2026-04-13 22:02: The paired Showdown 10-update learner comparison completed.
    - Showdown final checkpoint:
      - `data/models/rl/single_team_compare_showdown_10_updates/main_model_step_10.pt`
    - Showdown end-of-run learner-facing metrics from `data/models/rl/single_team_compare_showdown_10_updates/nohup_train_2026-04-13-remote.log`:
      - `Update 10`
      - `Total Battles=640 in 0h 13m 27s (0.79 b/s)`
      - `Learner Steps=6447 (7.98 steps/s)`
      - `Learner Trajectories=640 (0.79 traj/s)`
      - `Win rates: self_play: 62.0% | bc_player: 60.8% | max_damage: 27.0% | simple_heuristic_baseline: 19.0%`
    - Showdown shutdown behavior:
      - trainer also hit the same post-update worker join timeout and force-terminated still-running workers before saving the final model
      - unlike Rust, Showdown then shut down all four local servers gracefully
    - Direct paired comparison against the completed Rust run:
      - Rust: `640 battles` in `35m 23s`, `0.30 b/s`, `19011 learner steps`, `8.95 steps/s`
      - Showdown: `640 battles` in `13m 27s`, `0.79 b/s`, `6447 learner steps`, `7.98 steps/s`
      - Wall-clock interpretation:
        - Showdown completed the same 10 updates about `2.6x` faster by battle / trajectory throughput
        - Rust still produced about `12%` more learner steps per second
        - Rust generated far more learner steps per trajectory (`19011 / 640 ~= 29.7`) than Showdown (`6447 / 640 ~= 10.1`), so battle count alone is not an apples-to-apples efficiency metric
    - Stability / validity signals observed during the paired comparison:
      - Rust worker summaries continued to show substantial truncation pressure late in the run, with cumulative truncation rates on reporting workers still roughly in the `0.167-0.266` range
      - Showdown did not emit comparable truncation summaries, but it logged a large number of explicit websocket legality errors during play
      - counted Showdown legality errors: `24976` `PS_ERROR` lines in the 10-update run
      - most common Showdown invalid-choice families included:
        - move-name mismatches such as Dondozo being asked to use moves it does not have
        - missing or illegal targeting (for example `Uproar needs a target`, `Behemoth Blade needs a target`, `You can't choose a target for Protect`)
        - malformed joint choices such as sending more choices than unfainted Pokemon or passing when a forced switch was required
      - Rust log did not show equivalent `PS_ERROR` messages, but its truncation signal strongly suggests its own action-quality / battle-resolution issue remains unsolved
    - Backend decision implication from the current evidence:
      - the earlier model-backed benchmark still says Rust can beat Showdown on isolated backend throughput when the action-selection path is controlled
      - the learner-facing 10-update run says the current end-to-end system reaches updates much faster through Showdown, but with heavy invalid-choice churn
      - therefore the current bottleneck is not just simulator speed; it is action validity, episode length / truncation behavior, and whether learner steps are semantically comparable across backends
    - Recommended next decision criterion:
      - do not choose solely on battles/s or solely on learner steps/s from this pair
      - first measure which backend delivers more valid training signal per wall-clock hour after fixing the dominant legality/truncation issue on each side