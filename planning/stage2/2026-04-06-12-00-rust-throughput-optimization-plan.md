# Rust Backend Throughput Optimization Plan

## Context
During Stage 2, we introduced a local in-process Rust battle engine (`pokemon-showdown-rs` with PyO3 bindings) to replace the external Node.js Showdown websocket servers. The theoretical expectation was a 10-100x speedup in environment stepping. However, initial benchmarks revealed that the multi-update training run handles only ~0.33 battles/sec per worker, which is significantly slower than the older websocket baseline. 

## Before State
The current integration uses a `SyncRustBattleDriver` that runs completely serially. For each active battle, it steps the engine, extracts the raw Showdown text protocol strings, normalizes them, and feeds them into two separate `poke-env` `DoubleBattle` instances (one for each player perspective). It then calls the policy `Embedder`, converts the state to a tensor, and runs a PyTorch `forward()` pass with a batch size of exactly 1. Additionally, battles often stall with rejected actions, leading to a spin-loop that heavily inflates the truncation rate (~71%) and wastes CPU cycles.

## Problem
1. **The PyTorch Batching Bottleneck**: Running a large Transformer or LSTM model on the CPU with a batch size of 1 completely eliminates the high-dimensional vectorization benefits of Modern CPUs, severely bottlenecking throughput.
2. **The Serialization Bottleneck**: `poke-env` operates by parsing raw Showdown `.txt` logs. The Python string manipulation and regex-matching essentially replicates the work the Rust engine already did natively.
3. **The Spin-Loop Trap**: Actions rejected by the engine cause the driver to immediately re-prompt the policy without advancing the battle state, trapping the worker in an infinite loop until `max_stalled_steps` causes a truncation.

## Constraint Verification
Any long-term architectural shift must respect two critical constraints:
1. **Supervised BC Training**: We rely on historical Showdown logs stored as text. `poke-env` is very helpful for replaying these logs and reconstructing battle state for BC data generation.
2. **Live Ladder Play**: Our final RL policy must interact with human players on the live Showdown ladder, which streams Showdown protocol text over websockets and is best handled through `poke-env`.

This changes the integration boundary. We should not treat `poke-env` as something to eliminate globally; we should treat it as the canonical adapter for text-driven environments (historical logs and live ladder). The Rust engine should instead accelerate the self-play simulator path underneath a shared EliteFurretAI-owned observation interface.

If we build an embedder natively in Rust that circumvents the shared Python pipeline entirely, we risk severe train-test skew: the RL agent would train on Rust-generated feature tensors, but evaluate and play on Python-generated feature tensors derived from `poke-env`. Maintaining perfect parity between a Rust embedder and a Python embedder is a large ongoing maintenance burden and the wrong abstraction boundary.

## Exploration of Possibilities

### Goal 1: Maximize RL Self-Play Throughput Without Fragmenting The Policy Stack

If we are willing to decouple the RL self-play worker from `poke-env`'s `DoubleBattle` object while still preserving `poke-env` for text-driven BC and ladder play, we have several architectural paths to achieve peak performance:

*   **Option A: Full Rust Rollout Worker (C++ / Rust RL Integration)**
    We write the rollout collection loop entirely in Rust. Rust collects states, calls an exported TorchScript/ONNX C++ model, acts, and returns uncompressed trajectory bytes back to Python `train.py`. 
    *Pros:* Absolute maximum speed; zero Python GIL interference. *Cons:* Exceptionally high engineering complexity; requires exporting dynamic RL models to ONNX/LibTorch and reproducing the RNaD action sampling in Rust/C++. It also makes it harder to keep the live ladder policy path aligned with the training path.
*   **Option B: Canonical `BattleSnapshot` Transport**
    Instead of Rust returning Showdown protocol strings, the Rust engine exports a strictly typed intermediate state object owned by EliteFurretAI, such as `BattleSnapshot`, containing the battle fields the shared Python embedding pipeline needs. The Python side keeps one canonical `Embedder`, but it consumes this snapshot instead of reparsing text.
    *Pros:* Removes the string parsing bottleneck in RL while preserving one shared embedding stack for Rust self-play, BC replay, and live ladder play. This is the best long-term boundary.
    *Cons:* Requires designing and maintaining a stable intermediate schema plus adapters from both Rust engine state and `poke-env` battle state.
*   **Option C: FlatBuffer / Cap'n Proto Transport Encoding**
    The `BattleSnapshot` transport is serialized through FlatBuffers, Cap'n Proto, or a compact PyO3-backed struct for low-overhead Python access.
    *Pros:* Useful implementation detail for Option B if Python object creation becomes the next bottleneck.
    *Cons:* This is an encoding choice, not the core architecture by itself.

### Goal 2: Maximize Throughput while keeping the `poke-env` framework

If we want to maintain the `poke-env` ecosystem (ensuring model inference and heuristics maintain a shared, cohesive API across all environments), we need creative Python optimization paths:

*   **Option A: Parallelized Multi-Threaded Logging (`ThreadPoolExecutor`)**
    Because text parsing and string manipulation dominate the `poke-env` overhead, we could fan out the `_apply_requests` and `_drain_messages` to a Python ThreadPool. 
    *Pros:* Easy to write. *Cons:* The Python GIL will heavily throttle string parsing anyway. It likely won't yield more than a 10-20% speedup.
*   **Option B: The `poke-env` Vectorized Batching Driver**
    We modify `SyncRustBattleDriver` so that it doesn't step one battle at a time. Instead, it advances *all* active Rust battles that need an action. It then grabs the `poke-env` embedded states for *every* active battle, stacks them into a single `[N, feature_dim]` PyTorch tensor, and runs the `SyncPolicyPlayer` inference once for the entire batch.
    *Pros:* Massively improves CPU cache locality and PyTorch vectorization (batch_size=1 -> batch_size=32). Gives the largest immediate performance leap without changing `poke-env`'s parsing role.
*   **Option C: JIT-Compiled or Cythonized `poke-env` Text Parsers**
    Use `numba` or write a custom C extension to rewrite the specific string-split and regex logic inside `_apply_protocol_line()`.

The recommendation is to treat Goal 2 as the immediate path and Goal 1 Option B as the durable destination. In other words: first batch aggressively while still using `poke-env` battle objects where needed, then replace the RL-only text parsing boundary with an EliteFurretAI-owned `BattleSnapshot` interface rather than a second embedder.

## Solution

We will execute a phased approach that prioritizes immediate, high-impact fixes while paving the way for maximum theoretical throughput.

The architectural rule is:

- `poke-env` remains the adapter for text-based environments: historical Showdown logs and live ladder/websocket play.
- Rust accelerates only the self-play simulator backend.
- EliteFurretAI owns the shared policy-facing observation boundary so both backends feed the same Python embedding and action-selection code.

1.  **Phase 1 (Immediate & `poke-env` compliant): Vectorized Inference Batching & Spin-Loop Mitigation**
    We will modify `SyncRustBattleDriver` to decouple state collection from action execution (Option 2B). We will gather all battles requiring an action, embed them sequentially via `poke-env`, stack the `numpy` arrays into a single batched CPU tensor `(batch_size=N)`, run a single forward pass, and map the resultant actions back to the individual instances. 
    *We will simultaneously fix the spin-loop behavior* by ensuring that if an action is rejected by the Rust engine, the driver forces a fallback valid action or penalizes the sequence to prevent endless re-requests.

2.  **Phase 2 (Long-Term & Maximum Throughput): Shared `BattleSnapshot` Observation Layer**
    To achieve the final 10-100x theoretical multiplier without fragmenting the policy stack, we will introduce an EliteFurretAI-owned intermediate observation type such as `BattleSnapshot`. The Rust engine will produce this snapshot directly for RL self-play. The BC and ladder paths will continue to use `poke-env`, but they will convert `DoubleBattle` state into the same `BattleSnapshot` before invoking the shared Python `Embedder`. This preserves one policy-facing feature pipeline while eliminating RL-side text parsing.

3.  **Phase 3 (Optional Encoding Optimization): Compact Snapshot Transport**
    If Python object materialization becomes a measurable bottleneck after Phase 2, we will serialize `BattleSnapshot` through FlatBuffers, Cap'n Proto, or a compact PyO3 struct. This is an optimization layer on top of the shared observation boundary, not a substitute for it.

## Reasoning
The biggest bottleneck is currently batch size 1 inference on the CPU. The easiest way to get an order of magnitude improvement immediately is to batch requests across multiple active simulated games before invoking the network. Fixing the truncation spin-loop will directly recover wasted CPU steps. In the long run, Python string manipulation is inherently bounded. But because BC training and live ladder play are fundamentally text-oriented Showdown workflows, the right answer is not a Rust-only embedder. The right answer is a shared policy-facing observation boundary inside EliteFurretAI, with `poke-env` serving text-based inputs and Rust serving simulator-native inputs. That preserves parity where it matters and still removes the expensive RL-only serialization path.

## Planned Next Steps / Implementation Plan
1. **Address Truncation**: Update the `SyncRustBattleDriver.step()` loop to reliably catch choices that are legal according to the Python-side request enumeration but still rejected by the Rust engine, then fall back to safe recovery behavior instead of thrashing `max_stalled_steps`.
2. **Vectorize Python Inferencing**: Rewrite the active battle loop. Collect a list of `NeedAction` states across all 32+ concurrent active battles in a worker, call `p1_policy.choose_actions(batch)`, and distribute the results.
3. **Define `BattleSnapshot`**: Create the minimal EliteFurretAI-owned observation schema required by the shared `Embedder`, action masker, and policy code. Keep it constrained to fields we already trust across both backends.
4. **Add Dual Adapters**: Implement `DoubleBattle -> BattleSnapshot` for BC/ladder paths and `RustBattleState -> BattleSnapshot` for RL self-play.
5. **Benchmark**: Rerun `rust_multiupdate_benchmark.yaml` to measure the speedup of vectorized inference first, then compare Phase 2 snapshot transport against the current protocol-text path.
6. **Extract `BattleRuntimeFactory`**: Formalize the backend abstraction so standard training scripts are agnostic to this engine structure.

## Updates

- 2026-04-08 23:35: Implemented the Python-side subset of this plan that can land inside EliteFurretAI without changing the external `pokemon_showdown_py` binding.
    - Added `BattleSnapshot` in [src/elitefurretai/engine/battle_snapshot.py](src/elitefurretai/engine/battle_snapshot.py) as the policy-facing observation object for the Rust self-play path.
    - Updated [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py) to cache per-side request objects once per drain cycle and to accept future native dict-returning bindings through `get_request_dict()` when that becomes available upstream.
    - Updated [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py) to build snapshots, use `embed_to_vector()` on the fast path, retry rejected choices with a fallback action, and batch `SyncPolicyPlayer` inference across concurrent active battles that share a policy.
    - Added focused regression coverage in [unit_tests/engine/test_rust_battle_engine.py](unit_tests/engine/test_rust_battle_engine.py) and [unit_tests/engine/test_sync_battle_driver.py](unit_tests/engine/test_sync_battle_driver.py) for request caching, snapshot usage, batched inference, and rejected-choice fallback.

- 2026-04-08 23:35: Benchmark checkpoints recorded during implementation.
    - Baseline microbenchmark: `python src/elitefurretai/engine/rust_engine_benchmark.py --max-concurrent 7 --battles 200 --collect-rollouts`
        - `battles_per_second=2.940`
        - `decisions_per_second=140.597`
        - `truncated_battles=53`
        - `wall_seconds=71.98`
    - Baseline end-to-end smoke: `python src/elitefurretai/rl/train.py --config src/elitefurretai/rl/configs/rust_smoke.yaml`
        - `wall_seconds=31.30`
    - Slice 1, after request caching + snapshot scaffolding + fallback retry + `embed_to_vector()`:
        - Microbenchmark: `battles_per_second=2.396`, `decisions_per_second=91.991`, `truncated_battles=23`, `wall_seconds=85.22`
        - Train smoke: `wall_seconds=25.24`
    - Slice 2, after batched `SyncPolicyPlayer` inference:
        - Microbenchmark: `battles_per_second=2.793`, `decisions_per_second=91.253`, `truncated_battles=36`, `wall_seconds=73.35`
        - Train smoke: `wall_seconds=29.19`
    - Post-change concurrency probe on the rollout benchmark:
        - `max_concurrent=4` -> `battles_per_second=2.493`, `decisions_per_second=92.728`
        - `max_concurrent=7` -> `battles_per_second=2.566`, `decisions_per_second=91.670`
        - `max_concurrent=10` -> `battles_per_second=2.857`, `decisions_per_second=92.178`
    - Higher-concurrency train smoke probe with a temporary config override (`num_players: 8`, which yields `max_concurrent_battles=4` in the Rust worker loop):
        - `wall_seconds=35.87`
        - first worker batch completed `8` battles and emitted `5` trajectories

- 2026-04-08 23:35: What the benchmarks imply.
    - The request-cache and fallback-retry work was worth doing for the real `train.py` path. It improved the tiny Rust train smoke from `31.30s` to `25.24s` even though it did not improve the no-model microbenchmark.
    - The rollout microbenchmark is a poor proxy for policy throughput when no model-backed policy is attached. That benchmark exercises action enumeration and battle stepping heavily, but it does not exercise the model-forward bottleneck that motivated batching in the first place.
    - `BattleSnapshot` is valuable as an interface simplification and as the right long-term boundary, but the current implementation is still backed by `poke-env` `DoubleBattle` state and Python-generated vectors. On its own, it is not yet the big speed win.
    - The batched policy path is only valuable when the worker actually has multiple simultaneous active battles for the same policy. The checked-in Rust smoke and benchmark configs both use `num_players: 2`, which means `players_per_worker // 2 == 1`, so the new batching path is mostly starved in those runs.
    - The small parameter sweep suggests current throughput is slightly better with more active concurrent battles (`max_concurrent=10` beat `4` and `7` on completed battles/sec), but the gain is modest until the binding-level serialization/logging overhead is removed.

- 2026-04-08 23:35: Deliberate non-implementation / external blockers.
    - Native PyO3 request dict returns were not fully implemented because the installed `pokemon_showdown_py.RustBattle` binding in this environment still only exposes `get_request_json`, `get_messages`, `choose`, `turn`, `winner`, and `ended`. The Python side is now ready to consume `get_request_dict()` if the binding adds it.
    - A true headless or silent Rust-engine mode was not implemented here because that has to happen in the external Rust binding/engine package. The current Python repo cannot suppress protocol-string allocation if the binding insists on emitting those logs.
    - A full Phase 2 snapshot transport that bypasses `poke-env` text parsing entirely also depends on new Rust binding surface area. The new `BattleSnapshot` is therefore the Python-side contract, not yet the final zero-text transport.

- 2026-04-08 23:35: Recommended next optimization order after this pass.
    1. Add native `get_request_dict()` and a native observation snapshot to the Rust binding, then feed that directly into `BattleSnapshot`.
    2. Add a headless Rust battle mode that suppresses protocol-string generation during RL self-play.
    3. Raise Rust-worker concurrency in the benchmark configs (`num_players > 2`) so the batched policy path is exercised enough to matter.
    4. Re-benchmark with a model-backed benchmark path instead of relying primarily on the random-rollout microbenchmark.

- 2026-04-09 08:00: Implemented items 1-3 as far as this workspace allows.
    - Item 1, practical binding-side step inside this repo: added a repo-owned `CachedRustBattleBinding` plus `RustBattleSideSnapshot` in [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py).
        - This does not modify the external compiled PyO3 module directly, because the installed `pokemon_showdown_py.RustBattle` object is still a builtin extension type with no editable Rust source in this workspace.
        - It does provide the interfaces the Python RL path wanted from that binding layer today: `get_request_dict()` and `get_side_snapshot()` without repeated JSON parsing by every consumer.
        - `SyncRustBattleDriver` now wraps the builtin binding in `CachedRustBattleBinding` and threads the resulting side snapshot into `BattleSnapshot.binding_snapshot`.
    - Item 2: raised the benchmark training config concurrency in [src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml](src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml) from `num_players: 2` to `num_players: 8`, which yields `max_concurrent_battles=4` in the current Rust worker loop.
    - Item 3: added a new model-backed benchmark entrypoint in [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py) that builds or loads a real `RNaDAgent`, attaches `SyncPolicyPlayer` policies, and measures Rust self-play throughput under actual CPU inference rather than random legal-action sampling.

- 2026-04-09 08:00: Validation and benchmark results for items 1-3.
    - Focused validation:
        - `pytest unit_tests/engine/test_rust_battle_engine.py unit_tests/engine/test_sync_battle_driver.py -q` -> passed (`11` tests)
        - `ruff check` on the changed Rust-driver and benchmark files -> passed
    - Random/legal-action benchmark after the wrapper path, using `--max-concurrent 10 --battles 100 --collect-rollouts`:
        - `completed_battles=100`
        - `truncated_battles=7`
        - `battles_per_second=2.247`
        - `decisions_per_second=92.206`
        - `wall_seconds=46.42`
    - New model-backed benchmark, using [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py) with [src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml](src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml):
        - `completed_battles=100`
        - `truncated_battles=17`
        - `battles_per_second=0.780`
        - `decisions_per_second=20.892`
        - `max_concurrent=4`
        - `wall_seconds=132.34`
    - Higher-concurrency train smoke using a one-update temporary derivative of [src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml](src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml):
        - `wall_seconds=39.75`
        - first two worker batches each completed `8` battles and each truncated `5`

- 2026-04-09 08:00: What these new numbers mean.
    - The repo-owned wrapper for request dicts/snapshots is worth keeping because it simplifies the Python integration boundary and removes redundant JSON parsing points, even though it is not yet the true native PyO3 implementation.
    - Raising concurrency does make the batched policy path active, but once real CPU model inference is in the loop, inference cost dominates. The model-backed benchmark is much slower than the random rollout benchmark, which confirms that the next throughput wall is real model execution, not just transport overhead.
    - The new model-backed benchmark is more decision-useful than the random rollout benchmark. It tells us the current Rust path can step battles quickly enough, but the end-to-end training throughput is still bounded heavily by CPU inference and truncation rate under higher concurrency.
    - The remaining high-value work is therefore still outside this repo boundary: a real native binding-side snapshot API and a headless Rust engine mode that suppresses protocol-string generation entirely.

- 2026-04-09 09:10: Turned the Python-side Rust binding contract into the exact benchmark-facing API shape and exposed implementation ablations in the model-backed benchmark.
    - [src/elitefurretai/engine/rust_battle_engine.py](src/elitefurretai/engine/rust_battle_engine.py) now defines one exact adapter contract for the Python RL path:
        - `get_request_json(side)`
        - `get_request_dict(side)`
        - `get_side_snapshot(side)`
        - `get_messages(side)`
        - `choose(side, choice)`
        - `turn`, `ended`, `winner`
    - That exact contract now has two adapter implementations around the compiled builtin binding:
        - `DirectRustBattleBinding`: exact-shape API, but intentionally no request caching so we can measure repeated JSON parsing cost.
        - `CachedRustBattleBinding`: exact-shape API with centralized request caching.
    - `SyncRustBattleDriver` now accepts explicit implementation toggles instead of baking in one path:
        - `use_cached_binding_wrapper`
        - `include_binding_snapshots`
    - `SyncPolicyPlayer` now accepts explicit policy-side ablation toggles:
        - `enable_batch_inference`
        - `use_fast_embed_to_vector`
    - [src/elitefurretai/engine/rust_model_benchmark.py](src/elitefurretai/engine/rust_model_benchmark.py) now exposes all of those as CLI flags so we can benchmark the real model-backed path with or without each optimization.

- 2026-04-09 09:10: Parameter inventory for the final model-backed throughput sweep.
    - We should use the model-backed benchmark as the primary decision tool because it reflects the actual training bottlenecks much better than the random legal-action benchmark.
    - Implementation-ablation parameters now exposed directly in `rust_model_benchmark.py`:
        - `cached_request_wrapper`: on/off via `--disable-request-cache-wrapper`
        - `binding_snapshots`: on/off via `--disable-binding-snapshots`
        - `batched_inference`: on/off via `--disable-batched-inference`
        - `fast_embed_to_vector`: on/off via `--disable-fast-embed`
    - Battle-runtime parameters already available and worth sweeping:
        - `max_concurrent`
        - `max_turns_per_battle`
        - `max_stalled_steps_per_battle`
        - `collect_rollouts`
        - mirror vs non-mirror opponents (`--no-mirror`, `--opponent-team-path`)
        - checkpoint asymmetry (`--checkpoint`, `--opponent-checkpoint`)
    - Policy-sampling parameters already available and worth sweeping because they can affect branching cost and truncation patterns:
        - `temperature`
        - `top_p`
        - `greedy`
    - Config-backed model/runtime parameters worth sweeping through the benchmark config or temporary config variants:
        - `embedder_feature_set`
        - `num_players` and therefore `players_per_worker`
        - `num_workers` for train-smoke and full training runs
        - `num_battles_per_pair`
        - `max_battle_steps`
        - `battle_backend` should stay `rust_engine` for this sweep, but websocket remains supported outside this experiment
        - model architecture knobs that materially change inference throughput: `use_transformer`, hidden size / layer counts in the model config, `use_number_banks`
    - Training-only parameters to record alongside the sweep, even if the first pass keeps them fixed:
        - `batch_size`
        - `use_mixed_precision`
        - optimizer schedule and LR groups
        - `num_workers` x `num_players` interaction
        - team sampling mode (`use_random_teams`, fixed team vs directory sampling)

- 2026-04-09 09:10: Recommended sweep order so we separate implementation wins from workload-shape wins.
    1. Implementation ablation sweep on the model-backed benchmark with one fixed config.
        - Goal: quantify the marginal value of request caching, binding snapshots, batched inference, and the fast embed path.
    2. Concurrency and truncation sweep.
        - Goal: jointly tune `max_concurrent`, `num_players`, `max_turns_per_battle`, and `max_stalled_steps_per_battle`.
    3. Feature and model-cost sweep.
        - Goal: compare `embedder_feature_set` values and lightweight-vs-heavier model settings under the now-tuned runtime.
    4. Short `train.py` smoke confirmation on the best benchmark candidates.
        - Goal: confirm the benchmark winner actually improves end-to-end learner throughput rather than just raw battle stepping.

- 2026-04-09 09:10: Immediate benchmark command surface we can now use for those sweeps.
    - Baseline exact-shape model benchmark:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 100 --device cpu`
    - Disable batched inference:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 100 --device cpu --disable-batched-inference`
    - Disable fast embed path:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 100 --device cpu --disable-fast-embed`
    - Disable request cache wrapper:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 100 --device cpu --disable-request-cache-wrapper`
    - Disable binding snapshots:
        - `python src/elitefurretai/engine/rust_model_benchmark.py --config src/elitefurretai/rl/configs/rust_multiupdate_benchmark.yaml --battles 100 --device cpu --disable-binding-snapshots`

- 2026-04-09 09:20: Short model-backed ablation smoke after exposing the new flags.
    - Baseline exact-shape path, `20` battles, CPU, current benchmark config:
        - `truncated_battles=4`
        - `battles_per_second=1.092`
        - `decisions_per_second=19.337`
        - `max_concurrent=4`
        - `batched_inference=True`
        - `fast_embed_to_vector=True`
        - `binding_snapshots=True`
        - `cached_request_wrapper=True`
    - Batched inference disabled under the same setup:
        - `truncated_battles=5`
        - `battles_per_second=0.609`
        - `decisions_per_second=12.179`
        - This is a large enough drop that batched inference should stay enabled unless a later train-smoke contradicts it.
    - Request cache wrapper disabled under the same setup:
        - `truncated_battles=3`
        - `battles_per_second=0.914`
        - `decisions_per_second=19.642`
        - This result is close enough to baseline that it needs a larger sweep before we treat it as meaningful; the main value right now is that the exact-shape adapter toggle is working.