# Rust Model-Backed Throughput Sweep

## Context
We have already established that the random legal-action Rust benchmark is a poor proxy for actual RL training throughput. The relevant benchmark is now [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py), because it exercises the real policy path: embedding, action masking, model forward passes, and synchronous battle stepping.

The prior Stage 2 throughput document captured the architectural direction and short ablation smokes. This document is the dedicated record for the first longer 1000-battle sweep using the model-backed benchmark path.

## Before State
We had:
- an exact benchmark-facing Rust adapter contract in [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py)
- model-backed benchmark toggles in [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py)
- short 20-battle smokes suggesting batched inference matters a lot, while request-cache effects were less obvious

We did not yet have longer runs that were stable enough to support decisions about what to keep, what to simplify away, and what bottlenecks remain after these optimizations.

## Problem
We need to decide, based on longer model-backed runs:
1. Which implementation optimizations materially improve throughput and should remain.
2. Which implementation ideas are low-value enough that we should remove or ignore them for simplicity.
3. Which runtime parameters are worth tuning further.
4. Whether the longer sweeps reveal new bottlenecks or creative paths to increase RL battle throughput.

## Solution
Run a focused 1000-battle sweep using the model-backed benchmark path only.

The sweep matrix is intentionally limited to the highest-signal dimensions:

### Implementation ablations
- baseline exact-shape path
- disable batched inference
- disable fast embed path
- disable cached request wrapper
- disable binding snapshots

### Runtime parameter sweep
- baseline concurrency
- lower concurrency (`max_concurrent=2`)
- higher concurrency (`max_concurrent=6`)

This separates code-path optimizations from workload-shape tuning without exploding total benchmark time.

## Reasoning
The short smokes already suggested that not all optimizations are equally valuable. Batched inference had a large observed effect, while some transport-side changes appeared second-order. A 1000-battle run is long enough to smooth out noise from truncation variance and unusual battle paths. By restricting the matrix to the most consequential axes, we can make decisions that improve the real training path without overfitting to microbenchmark noise.

## Planned Next Steps
1. Run the 1000-battle model-backed baseline.
2. Run the four implementation ablations.
3. Run the concurrency sweep.
4. Record all results in this document.
5. Synthesize recommendations on what to keep, remove, and investigate next.

## Updates

- 2026-04-09 09:35: Created this dedicated sweep document and ported the benchmark context from the broader throughput plan. The active sweep matrix is:
    - `baseline`
    - `disable_batched_inference`
    - `disable_fast_embed`
    - `disable_request_cache_wrapper`
    - `disable_binding_snapshots`
    - `max_concurrent_2`
    - `max_concurrent_6`

- 2026-04-09 09:35: All runs in this document will use the model-backed benchmark entrypoint:
    - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu`

- 2026-04-09 10:45: Baseline 1000-battle result recorded.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=482`
        - `duration_seconds=3691.351`
        - `battles_per_second=0.271`
        - `p1_decisions=85107`
        - `decisions_per_second=23.056`
        - `max_concurrent=4`
        - `feature_set=raw`
        - `batched_inference=True`
        - `fast_embed_to_vector=True`
        - `binding_snapshots=True`
        - `cached_request_wrapper=True`
        - `wall_seconds=3697.80`
    - Immediate implication:
        - The long-horizon benchmark is substantially worse than the short 20-battle smoke. The main new fact is not just lower throughput, but the very high truncation rate (`48.2%`). That means long-run stability is now a first-class throughput bottleneck, not just a quality issue.

- 2026-04-09 10:45: The sweep is being narrowed adaptively based on the baseline runtime.
    - A full 7-scenario matrix at ~1 hour per run would not be a good use of time because some scenarios are already clearly dominated by the 20-battle smoke.
    - The remaining 1000-battle runs are being prioritized as:
        1. `max_concurrent=2` to test whether lower concurrency reduces truncation enough to improve net throughput.
        2. `disable_request_cache_wrapper` because its short-run effect was ambiguous.
        3. `disable_fast_embed` if time remains, because it is a true simplicity/performance trade-off.
    - Batched inference is already strongly favored from the short model-backed smoke and is unlikely to justify another hour-long dominated run just to reconfirm an obvious loss.

- 2026-04-09 11:55: Lower-concurrency 1000-battle result recorded.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 2`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=172`
        - `duration_seconds=3953.771`
        - `battles_per_second=0.253`
        - `p1_decisions=54702`
        - `decisions_per_second=13.835`
        - `max_concurrent=2`
        - `wall_seconds=3957.89`
    - Comparison vs baseline:
        - Truncations improved dramatically (`482 -> 172`).
        - Raw throughput still got worse (`0.271 -> 0.253` battles/s).
        - Decision throughput dropped heavily (`23.056 -> 13.835` decisions/s), which implies that reduced concurrency lowers policy batching efficiency enough to offset the stability gain.
    - Immediate implication:
        - `max_concurrent=2` is a stability knob, not a throughput knob. It may be attractive if trajectory quality or completion rate matters more than raw battles/sec, but it is not the fastest setting among the tested values so far.

- 2026-04-09 11:55: Higher-concurrency run started.
    - Active command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6`

- 2026-04-09 12:15: Higher-concurrency 1000-battle result recorded.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=292`
        - `duration_seconds=660.486`
        - `battles_per_second=1.514`
        - `p1_decisions=18233`
        - `decisions_per_second=27.605`
        - `max_concurrent=6`
        - `wall_seconds=664.62`
    - Comparison vs baseline (`max_concurrent=4`):
        - Throughput improved enormously (`0.271 -> 1.514` battles/s).
        - Decision throughput improved (`23.056 -> 27.605` decisions/s).
        - Truncation rate also improved in absolute count (`482 -> 292`) despite the higher concurrency.
    - Immediate implication:
        - The current benchmark config was not operating near its throughput optimum.
        - `max_concurrent=6` is now the best tested operating point by a wide margin, so implementation ablations should be judged there rather than at `4`.

- 2026-04-09 12:15: The implementation sweep has been re-anchored onto `max_concurrent=6`.
    - Active command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-request-cache-wrapper`
    - Rationale:
        - If an optimization does not help under the best tested runtime regime, it should not be treated as throughput-critical.

- 2026-04-09 12:35: Request-cache wrapper ablation recorded at `max_concurrent=6`.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-request-cache-wrapper`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=203`
        - `duration_seconds=999.551`
        - `battles_per_second=1.000`
        - `p1_decisions=27662`
        - `decisions_per_second=27.674`
        - `wall_seconds=1003.68`
    - Comparison vs `max_concurrent=6` baseline:
        - Throughput dropped substantially (`1.514 -> 1.000` battles/s).
        - Decision throughput stayed almost identical (`27.605 -> 27.674` decisions/s).
        - Truncations decreased (`292 -> 203`), but not enough to offset the much slower wall time.
    - Immediate implication:
        - The cached request wrapper is a genuine throughput optimization, not just interface cleanup.
        - Without centralized request reuse, the runtime appears to spend more wall time per decision even when overall decision throughput remains similar.

- 2026-04-09 12:35: Fast-embed ablation started at `max_concurrent=6`.
    - Active command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-fast-embed`

- 2026-04-09 12:55: Fast-embed ablation recorded at `max_concurrent=6`.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-fast-embed`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=195`
        - `duration_seconds=744.422`
        - `battles_per_second=1.343`
        - `p1_decisions=20481`
        - `decisions_per_second=27.513`
        - `wall_seconds=748.52`
    - Comparison vs `max_concurrent=6` baseline:
        - Throughput dropped moderately (`1.514 -> 1.343` battles/s).
        - Decision throughput remained almost unchanged (`27.605 -> 27.513` decisions/s).
        - Truncations decreased (`292 -> 195`).
    - Immediate implication:
        - The fast embed path is worth keeping as a throughput optimization.
        - Like the request-cache result, this looks less like "more decisions per second" and more like "less wall time per completed battle" under the same broad policy workload.

- 2026-04-09 12:55: Binding-snapshot ablation started at `max_concurrent=6`.
    - Active command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-binding-snapshots`

- 2026-04-09 13:15: Binding-snapshot ablation recorded at `max_concurrent=6`.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-binding-snapshots`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=197`
        - `duration_seconds=725.296`
        - `battles_per_second=1.379`
        - `p1_decisions=20127`
        - `decisions_per_second=27.750`
        - `wall_seconds=729.39`
    - Comparison vs `max_concurrent=6` baseline:
        - Throughput dropped moderately (`1.514 -> 1.379` battles/s).
        - Decision throughput was effectively flat to slightly higher (`27.605 -> 27.750` decisions/s).
        - Truncations decreased (`292 -> 197`).
    - Immediate implication:
        - Binding snapshots are not the primary throughput win in the current Python implementation.
        - They still appear mildly beneficial overall, but their strongest justification remains architectural consistency and a stable future binding API rather than a massive standalone speedup.

- 2026-04-09 13:15: Final long-run batched-inference ablation started at `max_concurrent=6`.
    - Active command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-batched-inference`

- 2026-04-09 14:40: Final long-run batched-inference ablation recorded at `max_concurrent=6`.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6 --disable-batched-inference`
    - Result:
        - `completed_battles=1000`
        - `truncated_battles=121`
        - `duration_seconds=3408.864`
        - `battles_per_second=0.293`
        - `p1_decisions=35997`
        - `decisions_per_second=10.560`
        - `wall_seconds=3412.97`
    - Comparison vs `max_concurrent=6` baseline:
        - Throughput collapsed (`1.514 -> 0.293` battles/s).
        - Decision throughput collapsed (`27.605 -> 10.560` decisions/s).
        - Truncations improved (`292 -> 121`), but nowhere near enough to compensate for the throughput loss.
    - Immediate implication:
        - Batched inference is non-negotiable for the model-backed Rust self-play path.

- 2026-04-09 14:40: Consolidated sweep table.
    - Raw benchmark throughput (`completed_battles / duration_seconds`):

| Scenario | Battles/s | Decisions/s | Truncations | Truncation Rate |
| --- | ---: | ---: | ---: | ---: |
| baseline (`max_concurrent=4`) | 0.271 | 23.056 | 482 | 48.2% |
| `max_concurrent=2` | 0.253 | 13.835 | 172 | 17.2% |
| `max_concurrent=6` | 1.514 | 27.605 | 292 | 29.2% |
| `max_concurrent=6`, no request cache | 1.000 | 27.674 | 203 | 20.3% |
| `max_concurrent=6`, no fast embed | 1.343 | 27.513 | 195 | 19.5% |
| `max_concurrent=6`, no binding snapshots | 1.379 | 27.750 | 197 | 19.7% |
| `max_concurrent=6`, no batched inference | 0.293 | 10.560 | 121 | 12.1% |

    - Training-oriented derived metric (`non_truncated_battles / duration_seconds`):
        - baseline (`max_concurrent=4`): `0.140`
        - `max_concurrent=2`: `0.209`
        - `max_concurrent=6`: `1.072`
        - `max_concurrent=6`, no request cache: `0.797`
        - `max_concurrent=6`, no fast embed: `1.081`
        - `max_concurrent=6`, no binding snapshots: `1.107`
        - `max_concurrent=6`, no batched inference: `0.258`

- 2026-04-09 14:40: Synthesis and recommendations.
    - (a) Parameters and optimizations to keep:
        - Keep batched inference. It is the single largest confirmed throughput win in the model-backed path. Disabling it reduced battles/sec by about `80.6%` relative to the `max_concurrent=6` baseline.
        - Keep a higher concurrent battle count than the current default benchmark path. Among tested values, `max_concurrent=6` is by far the best raw-throughput point and also the best non-truncated throughput point among the clearly throughput-oriented configurations.
        - Keep the cached request wrapper. Removing it dropped battles/sec from `1.514` to `1.000` at `max_concurrent=6`.
        - Keep the fast `embed_to_vector()` path. Removing it dropped battles/sec from `1.514` to `1.343` at `max_concurrent=6`.
        - Keep the exact binding adapter contract in [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py). It gives us a stable surface to compare implementations and will still be the right target shape for the future native PyO3 binding.
        - Practical config recommendation from this sweep: move the benchmark and Rust training configs toward `players_per_worker ~= 12`, because current training derives `max_concurrent_battles = players_per_worker // 2`, and the best tested point was `max_concurrent=6`.
    - (b) Optimizations to get rid of or ignore for simplicity:

- 2026-04-10 12:50: Added paired `train.py` backend-comparison configs and reran the comparison after the Showdown commander fix.
    - New reproducible configs:
        - [src/elitefurretai/rl/configs/backend_compare_rust.yaml](src/elitefurretai/rl/configs/backend_compare_rust.yaml)
        - [src/elitefurretai/rl/configs/backend_compare_showdown.yaml](src/elitefurretai/rl/configs/backend_compare_showdown.yaml)
        - [src/elitefurretai/rl/configs/backend_compare_rust_long.yaml](src/elitefurretai/rl/configs/backend_compare_rust_long.yaml)
        - [src/elitefurretai/rl/configs/backend_compare_showdown_long.yaml](src/elitefurretai/rl/configs/backend_compare_showdown_long.yaml)
    - Short one-update comparison artifacts:
        - Rust: `data/benchmarks/training_throughput_2026_04_10/backend_compare_rust.txt`
        - Showdown pre-fix: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown.txt`
        - Showdown post-fix smoke: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_post_commander_fix_smoke.txt`
    - Longer three-update comparison artifacts:
        - Rust: `data/benchmarks/training_throughput_2026_04_10/backend_compare_rust_long.txt`
        - Showdown: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_long.txt`
    - Post-fix short smoke result:
        - Showdown no longer emits the old commander-specific error `Can't move: Your Dondozo doesn't have a move matching dracometeor`.
        - Update 1: `0.17 battles/s`, `2.03 learner steps/s`, `47 learner steps over 4 battles`.
    - Longer three-update result:
        - Rust update 3 cumulative totals: `12 battles in 44s` (`0.27 battles/s`), `259 learner steps` (`5.79 steps/s`).
        - Showdown update 3 cumulative totals: `12 battles in 40s` (`0.30 battles/s`), `156 learner steps` (`3.86 steps/s`).
        - Derived average learner steps per battle:
            - Rust: `259 / 12 = 21.6`
            - Showdown: `156 / 12 = 13.0`
    - Interpretation:
        - Rust remains materially better on learner-facing throughput even when raw battle throughput looks similar.
        - The longer run explains why: the Showdown backend still accumulates many invalid-choice events (`276` matching log lines), and its most common surviving error is unrelated to commander (`180` instances of `Terapagos ... terastarstorm`).
        - Those websocket-side errors shorten or distort battles, which can make `battles/s` look competitive or even slightly better while simultaneously producing fewer learner-ingested steps per battle.
        - In other words, the current Rust advantage is mostly battle quality and step retention, not just a lower wall-clock cost per finished battle.

- 2026-04-10 13:50: Reran the paired three-update comparison after the transformed-target and request-filter fixes.
    - New artifacts:
        - Rust: `data/benchmarks/training_throughput_2026_04_10/backend_compare_rust_long_post_target_fix.txt`
        - Showdown: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_long_post_target_fix.txt`
        - Showdown smoke: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown_post_target_fix_smoke.txt`
    - Specific targeted improvement:
        - pre-fix Showdown longer run had `180` occurrences of `Can't move: Your Terapagos doesn't have a move matching terastarstorm`.
        - post-fix Showdown longer run had `0` occurrences of that signature.
        - the post-fix smoke also had `0` combined matches for the targeted transformed-move signatures:
            - `Terapagos ... terastarstorm`
            - `... expandingforce`
    - Updated three-update results:
        - Rust update 3 cumulative totals: `12 battles in 46s` (`0.26 battles/s`), `290 learner steps` (`6.24 steps/s`).
        - Showdown update 3 cumulative totals: `12 battles in 35s` (`0.34 battles/s`), `142 learner steps` (`3.97 steps/s`).
        - Derived learner steps per battle:
            - Rust: `290 / 12 = 24.2`
            - Showdown: `142 / 12 = 11.8`
    - Comparison versus the immediately prior paired run:
        - Rust learner steps improved from `259 -> 290`.
        - Showdown targeted `terastarstorm` failures improved from `180 -> 0`.
        - Showdown total invalid-choice log lines did not improve overall (`276 -> 308`) because different websocket mismatches took over the rejection budget immediately.
    - Interpretation:
        - the transformed-target fixes removed a real class of websocket legality bugs, but they did not close the backend throughput gap because the Showdown path is still losing step yield to other request/move mismatches.
        - Rust remains materially better on learner throughput even when Showdown finishes battles faster in wall-clock terms, because Rust is retaining far more learner-ingested decisions per battle.
        - updated takeaway:
            - battle completion rate alone is still not the right proxy for training throughput.
            - until the websocket path stops leaking rewritten or stale move ids, Rust's main advantage will continue to be trajectory quality and step retention.

- 2026-04-09 16:40: Added structured rejection and truncation diagnostics to [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) and threaded them through [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py).
    - New benchmark option:
        - `--diagnostic-log-path <jsonl>`
    - Logged event types:
        - `rejected_choice`
        - `battle_truncated`
    - Each rejection event now records:
        - side, turn, request type, submitted choice, fallback choice, whether the choice was in the Python legal set, force-switch state, trapped flags, tera availability, a short preview of the legal choices, and the raw request when available.

- 2026-04-09 16:40: Confirmed and fixed one concrete legality bug.
    - Problem:
        - During force-switch requests, the sync driver was still enumerating normal move actions for the non-forced slot.
        - This allowed obviously invalid joint actions such as `switch X, move Y` in request phases where the engine only accepts switch/pass semantics.
    - Fix kept:
        - In [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py), when any slot is in `forceSwitch`, the non-forced slot is now constrained to `pass` instead of normal move/switch enumeration.
        - Healthy forced-switch slots no longer unconditionally receive `pass` as an extra action.
    - Focused regression coverage added in [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py).

- 2026-04-09 16:40: Short post-fix diagnostic run recorded for the kept fix set.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 200 --device cpu --max-concurrent 6 --diagnostic-log-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_final_kept_fix_200.jsonl`
    - Result artifact:
        - `data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_final_kept_fix_200.txt`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=46`
        - `non_truncated_battles=154`
        - `battles_per_second=1.329`
        - `non_truncated_battles_per_second=1.023`
        - `p1_rejected_choices=1949`
        - `p1_fallback_recoveries=1025`
        - `p1_unrecovered_rejections=924`
    - Dominant remaining rejection clusters in the JSONL:
        - `move, pass` during move requests
        - `switch, pass` and `pass, switch` during switch requests
        - stall truncations remain strongly coupled to those repeated rejected-choice loops

- 2026-04-09 16:40: Additional hypotheses were tested and then explicitly not kept because they did not improve the measured system.
    - Tried:
        - trusting request `active` entries over poke-env active-slot cache for normal move enumeration
        - tightening dual force-switch pass handling further
        - inferring blank request targets from move IDs
        - retrying both fallback choices together when both sides were rejected
    - Outcome:
        - these experiments helped narrow the remaining failure surface, but the measured 200-battle throughput and non-truncated throughput did not improve in the tested runs
        - they were reverted so the repository lands on the best proven state from this loop rather than on speculative changes that made end-to-end behavior worse

- 2026-04-09 16:40: Current best understanding after the iterative loop.
    - Confirmed relationship:
        - stall-limit is still acting mainly as a circuit breaker for repeated rejected-choice / no-progress loops, not as the root cause itself
    - Proven bug fixed:
        - incorrect force-switch legality enumeration in the sync driver
    - Remaining unresolved bottleneck:
        - move-phase legality is still diverging from the Rust engine in some states, especially in requests that repeatedly collapse into `move/pass` and `switch/pass` style rejection loops
    - Why the loop is stopping here:
        - further changes in this pass stopped producing measured improvement, and the remaining bug likely sits deeper in the Rust request/battle-state synchronization boundary rather than in one more obvious local legality patch
    - Most informed next step:
        - inspect the remaining `move/pass` and `switch/pass` rejection clusters against the raw request plus the synchronized poke-env battle state, especially around situations where `request["active"]` and `side.pokemon[*].active` disagree or where the request leaves target semantics underspecified
        - Do not treat Python-side binding snapshots as a primary throughput optimization. Their standalone speed effect is modest compared with batching, concurrency, request caching, and fast embedding.
        - Do not spend more time pursuing `max_concurrent=2` as a speed optimization. It is a stability knob only.
        - Do not consider disabling request caching or the fast embed path in the name of simplicity. Their code cost is low relative to their measured throughput benefit.
        - Do not spend more benchmark time on non-batched inference variants. The long-run run confirmed it is decisively dominated.

- 2026-04-09 18:25: Additional root-cause pass completed and explicitly reverted.
    - What was tested in code:
        - synchronizing poke-env active-slot cache directly from request parsing
        - filtering switch actions against `battle.available_switches` instead of every healthy `side.pokemon` entry
        - inferring move target types from move metadata when the Rust request left `target` blank
    - Why this was attempted:
        - live reproductions confirmed that the remaining rejection clusters were not random noise; they came from concrete state mismatches between the Rust request payload and the synchronized Python battle view
    - What the live reproductions showed:
        - some move requests expose two acting entries in `request["active"]` while `side.pokemon` still marks one of those mons as inactive or even `0 fnt`
        - some force-switch requests expose only a partial or stale view of `side.pokemon[*].active`, so enumerating switch targets from raw request health alone over-approximates legal choices
        - after repairing the obvious `move/pass` and `switch/pass` bugs locally, the dominant remaining failures became `move, move` rejections driven by deeper disagreement between `request["active"]`, `side.pokemon`, and protocol-tracked battle state
    - Measured result:
        - the experimental patch set reduced the old `move/pass` and `switch/pass` clusters, but 100-battle model-backed diagnostics still regressed against the kept baseline, including worse `non_truncated_battles_per_second`
        - the patch set was therefore reverted so the repository remains on the best proven measured state rather than on a net-worse intermediate
    - Updated best understanding:
        - the next real bug is not just local legality enumeration; it is a deeper Rust request contract issue where `active[]` can describe an acting mon while the paired `side.pokemon` entry still carries stale active/faint state
        - a durable fix likely needs request sanitization that rebuilds active-slot identity and condition from protocol-tracked slot occupancy plus the `active[]` payload, rather than trusting raw `side.pokemon` as authoritative during those turns
    - (c) Additional bottlenecks and ideas revealed by the sweep:
        - The current benchmark headline metric is misleading for training. `completed_battles` includes truncated battles, but truncated trajectories are often not the data we want. We should add a first-class benchmark metric for non-truncated battle throughput and probably trajectory-seconds or retained-steps/sec.
        - We need explicit truncation-cause instrumentation. Right now we know truncation volume, but not how much comes from `max_turns_per_battle` versus `max_stalled_steps_per_battle` versus rejected-choice recovery patterns.
        - Exploration settings are now an obvious candidate sweep axis. All runs here used `temperature=1.5` and `top_p=0.95`. That likely increases battle degeneracy and invalid/low-value action chains. A lower-temperature or narrower-top-p actor policy may improve usable trajectory throughput substantially even if it slightly reduces raw decision diversity.
        - The fact that `max_concurrent=6` dramatically improved throughput indicates we have been under-batching the policy relative to the CPU model cost. This strengthens the case for even more centralized or SEED-RL-like batched inference, not less.
        - The difference between raw battles/sec and non-truncated battles/sec means the next optimization target is not only faster stepping. It is also reducing pathological long or stalled games so that more wall time converts into retained training data.
        - A future native Rust-side snapshot API and headless mode still matter, but after this sweep they no longer look like the first-order bottleneck. The first-order bottlenecks are now: batching regime, truncation behavior, and policy/exploration settings under long-horizon self-play.

- 2026-04-09 20:15: Protocol-faithful target-sign fix recorded and kept.
    - Code change kept:
        - [src/elitefurretai/rl/fast_action_mask.py](src/elitefurretai/rl/fast_action_mask.py) now matches poke-env / Showdown target signs in the fast legality path.
        - [unit_tests/rl/test_fast_action_mask.py](unit_tests/rl/test_fast_action_mask.py) and [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) gained regression coverage for that behavior.
    - Command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 200 --device cpu --max-concurrent 6 --diagnostic-log-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_fix_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=30`
        - `non_truncated_battles=170`
        - `duration_seconds=178.047`
        - `battles_per_second=1.123`
        - `non_truncated_battles_per_second=0.955`
        - `p1_rejected_choices=2405`
        - `p1_fallback_recoveries=1602`
        - `p1_unrecovered_rejections=803`
    - Comparison vs prior kept baseline (`rejection_diagnostics_final_kept_fix_200`):
        - better stall behavior and fewer unrecovered rejections
        - worse raw throughput and worse retained-throughput
    - Interpretation:
        - the protocol correction is still worth keeping because the previous legality path was definitely wrong
        - it is not the main throughput fix; the dominant `move, pass` rejection loop remains after the correction

- 2026-04-09 20:15: Narrow active-entry reorder experiment recorded and reverted.
    - Experimental command:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 200 --device cpu --max-concurrent 6 --diagnostic-log-path data/benchmarks/training_throughput_2026_04_09/rejection_diagnostics_target_sign_reorder_fix_200.jsonl`
    - Result:
        - `completed_battles=200`
        - `truncated_battles=44`
        - `non_truncated_battles=156`
        - `duration_seconds=182.913`
        - `battles_per_second=1.093`
        - `non_truncated_battles_per_second=0.853`
        - `p1_rejected_choices=2860`
        - `p1_fallback_recoveries=1850`
        - `p1_unrecovered_rejections=1010`
    - Outcome:
        - the simple repair of promoting unmatched `active: true` roster entries before `parse_request()` made the measured system worse and was reverted
    - Updated bottleneck picture:
        - the best current kept code state from this pass is now: force-switch legality fix plus target-sign correctness fix
        - the remaining root issue still looks like malformed move requests at the adapter boundary, especially requests with blank move-target metadata and non-contiguous active roster entries

- 2026-04-09 14:40: Recommended immediate follow-up sweep order after this document.
    1. Sweep `max_concurrent` above `6` through the actual train-derived config path (`num_players -> players_per_worker -> max_concurrent_battles`) rather than only the benchmark CLI.
    2. Sweep `temperature` and `top_p` using the model-backed benchmark, but score scenarios by both raw battles/sec and non-truncated battles/sec.
    3. Add truncation-cause counters to [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) and report them from [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py).
    4. Re-run a short `train.py` smoke with the best candidate config, which currently looks like "batched inference + cached request wrapper + fast embed path + higher concurrency".

- 2026-04-09 15:05: Recommended metric for total training throughput.
    - Primary metric: `learner_steps_per_second`.
        - Definition: the number of trajectory steps actually ingested by the learner per wall second, after battle completion filtering and after sequence-length truncation at collate time.
        - Why this is the right primary metric:
            - It measures the data the optimizer really consumes, not just actor-side activity.
            - It is backend-agnostic: both Rust and websocket workers eventually send trajectories into the same learner queue.
            - It avoids the main failure mode of `battles_per_second`, which counts truncated or otherwise low-value battles as if they were equally useful training work.
    - Secondary diagnostic metrics:
        - `learner_trajectories_per_second` for update cadence / batch assembly behavior.
        - `non_truncated_battles_per_second` for actor-side environment quality.
        - truncation-cause counters and rejection/fallback counters for bottleneck diagnosis.

- 2026-04-09 15:05: Implemented the shared throughput metric in [src/elitefurretai/rl/train.py](src/elitefurretai/rl/train.py).
    - The main training loop now tracks and logs:
        - `learner_steps_per_second`
        - `learner_trajectories_per_second`
        - `received_steps_per_second`
        - `received_trajectories_per_second`
    - The console update line now prints those metrics even when `use_wandb: false`, which makes smoke benchmarks reproducible without external logging.

- 2026-04-09 15:05: Completed next step #1 through the actual train-derived config path.
    - Setup:
        - one-update `train.py` smokes
        - one worker
        - benchmark config derived from [src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml](src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml)
        - `num_players` swept over `8, 12, 16, 20`
        - derived `max_concurrent_battles = players_per_worker // 2` therefore swept `4, 6, 8, 10`
    - Result table (shared primary metric = `learner_steps_per_second`):

| Scenario | Derived Max Concurrent Battles | Learner Steps/s | Learner Traj/s | Battles/s | Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust train smoke, `num_players=8` | 4 | 2.18 | 0.14 | 0.14 | 44.26 |
| Rust train smoke, `num_players=12` | 6 | 4.19 | 0.22 | 0.22 | 33.28 |
| Rust train smoke, `num_players=16` | 8 | 3.60 | 0.17 | 0.17 | 39.19 |
| Rust train smoke, `num_players=20` | 10 | 3.24 | 0.19 | 0.19 | 36.79 |

    - Interpretation:
        - The train-derived optimum among tested values is still the same shape the benchmark suggested: `num_players=12` / derived `max_concurrent_battles=6`.
        - Pushing above `6` did not improve the training metric. The extra concurrency likely creates more actor churn and lower-value completions than the learner-step gains can justify.

- 2026-04-09 15:05: Completed next step #4 using the shared metric and compared it against the classic showdown backend.
    - Rust comparison point:
        - `num_players=12`, one worker, one update
        - `learner_steps_per_second=4.19`
        - `learner_trajectories_per_second=0.22`
        - `battles_per_second=0.22`
        - `wall_seconds=33.28`
    - Classic showdown comparison point:
        - same benchmark-derived config shape, but `battle_backend: showdown_websocket`
        - `num_players=12`, `num_servers=1`, one worker, one update
        - `learner_steps_per_second=2.01`
        - `learner_trajectories_per_second=0.15`
        - `battles_per_second=0.15`
        - `wall_seconds=43.52`
    - Comparison:
        - Rust delivered about `2.08x` the learner-ingested step throughput of websocket (`4.19 / 2.01`).
        - Rust also improved learner trajectory rate by about `1.47x` and reduced time-to-first-update materially.
    - Interpretation:
        - On the metric that matters for real training, Rust is now clearly ahead of the classic showdown path in this benchmark-shaped smoke.

- 2026-04-09 15:05: Completed next step #3 by adding and reporting truncation-cause counters in the Rust model benchmark.
    - Best-path benchmark rerun:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 1000 --device cpu --max-concurrent 6`
    - Result highlights:
        - `battles_per_second=1.385`
        - `non_truncated_battles_per_second=0.960`
        - `truncated_battles=307`
        - `turn_limit_truncations=0`
        - `stalled_limit_truncations=307`
        - `dual_limit_truncations=0`
        - `p1_rejected_choices=9943`
        - `p2_rejected_choices=10253`
        - `p1_fallback_recoveries=4950`

- 2026-04-10 12:10: Re-ran the Rust-vs-Showdown comparison through paired one-update `train.py` smokes using checked-in config files.
    - New reproducible configs:
        - `src/elitefurretai/rl/configs/backend_compare_rust.yaml`
        - `src/elitefurretai/rl/configs/backend_compare_showdown.yaml`
    - Shared benchmark shape:
        - one worker
        - `num_players=12`
        - `max_updates=1`
        - `battle_format=gen9vgc2024regg`
        - `embedder_feature_set=raw`
        - `battle_backend` is the only intentional backend difference
    - Artifacts:
        - Rust log: `data/benchmarks/training_throughput_2026_04_10/backend_compare_rust.txt`
        - Showdown log: `data/benchmarks/training_throughput_2026_04_10/backend_compare_showdown.txt`
    - Rust result:
        - `Update 1: ... Total Battles=4 in 0h 0m 17s (0.23 b/s) | Learner Steps=102 (5.77 steps/s) | Learner Trajectories=4 (0.23 traj/s)`
    - Showdown result:
        - `Update 1: ... Total Battles=4 in 0h 0m 17s (0.22 b/s) | Learner Steps=34 (1.90 steps/s) | Learner Trajectories=4 (0.22 traj/s)`
    - Comparison:
        - learner-ingested step throughput improved by about `3.04x` (`5.77 / 1.90`)
        - learner trajectory rate is effectively flat in this one-update smoke (`0.23 / 0.22 ~= 1.05x`)
        - raw completed battle rate is also nearly flat in this short horizon (`0.23 / 0.22 ~= 1.05x`)
    - Interpretation:
        - the backend win is currently showing up primarily as more learner-usable sequence steps per unit time, not as dramatically more completed battles before the first optimizer step.
        - this is still the metric that matters more for training because the learner consumes padded sequence steps, not just battle count.
    - Notable websocket-side log evidence:
        - the Showdown smoke emitted repeated invalid-choice errors of the form `Can't move: Your Dondozo doesn't have a move matching dracometeor`, which is additional evidence that commander-related legality mismatches are still visible in the legacy websocket path as well.
        - `p2_fallback_recoveries=4924`
        - `p1_unrecovered_rejections=4993`
        - `p2_unrecovered_rejections=5329`
    - Interpretation:
        - The current long-horizon truncation problem is not a turn-limit problem at all.
        - It is entirely a stall-limit problem under this configuration.
        - Rejected choices remain very common, and roughly half of them are not recovered by the fallback path. That is now one of the clearest remaining throughput bottlenecks.

- 2026-04-09 15:05: Updated recommendations after using the shared training-throughput metric.
    - Keep `learner_steps_per_second` as the primary benchmark metric for total training throughput.
    - Keep `num_players=12` / derived `max_concurrent_battles=6` as the best tested training config point for the current one-worker benchmark shape.
    - Keep batched inference, cached request reuse, and the fast embed path.
    - Treat websocket as the fallback/comparison backend, but not the throughput leader for this workload.
    - Prioritize rejected-choice reduction and stall-limit diagnosis before pursuing more generic transport cleanups.

- 2026-04-09 15:35: Baseline rejection and stall diagnostics.
    - Best-path 1000-battle benchmark baseline (`max_concurrent=6`):
        - `truncated_battles=307`
        - `stalled_limit_truncations=307`
        - `turn_limit_truncations=0`
        - `p1_decisions=19978`
        - `p1_rejected_choices=9943`
        - `p1_fallback_recoveries=4950`
        - `p1_unrecovered_rejections=4993`
    - Derived baseline rates:
        - `p1_rejection_rate=49.8%`
        - `p1_recovery_rate_given_rejection=49.8%`
        - `p1_unrecovered_rate_given_rejection=50.2%`
        - `stall_truncation_share=100%`
        - `truncation_rate=30.7%`
    - Interpretation:
        - In the current best-path long-horizon benchmark, every recorded truncation came from the stall limit rather than the turn limit.
        - About half of sampled `p1` choices were rejected by the Rust engine, and only about half of those rejections were recovered by the fallback path.

- 2026-04-09 15:35: Targeted stall-limit sweep to quantify why stall handling affects throughput.
    - Setup:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 200 --device cpu --max-concurrent 6 --max-stalled-steps-per-battle {5,10,25,50}`
    - Result table:

| Stall Limit | Battles/s | Non-Truncated Battles/s | Truncated Battles | p1 Reject Rate | p1 Recovery Given Reject |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5` | 1.974 | 1.283 | 70 | 42.6% | 70.7% |
| `10` | 1.415 | 1.047 | 52 | 43.0% | 68.4% |
| `25` | 1.028 | 0.858 | 33 | 50.3% | 68.2% |
| `50` | 1.093 | 0.781 | 57 | 57.2% | 35.0% |

    - Interpretation:
        - A looser stall limit does not help throughput here. It keeps pathological battles alive longer, which burns wall time and worsens usable throughput.
        - In this targeted 200-battle test, a tighter stall limit (`5` or `10`) materially improved non-truncated throughput relative to the current benchmark default (`25`).
        - The signal is not "stall limit causes more truncations so training is worse". The signal is "allowing a battle to remain stalled for too long is itself expensive", because those battles keep consuming actor/inference time while contributing low-value or discarded data.