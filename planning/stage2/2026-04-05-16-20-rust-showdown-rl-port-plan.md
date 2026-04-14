# Rust Showdown RL Port Plan

## Context
We want to replace the current reinforcement learning battle execution path, which uses local TypeScript Pokemon Showdown servers plus poke-env websocket clients, with a Rust implementation of the battle engine from `vjeux/pokemon-showdown-rs`. The goal is not to change the learner or the policy architecture; the goal is to change how worker processes execute battles so that RL data collection is faster, simpler, and less memory-fragile.

This plan is Stage II work because it affects the RL self-play training substrate rather than the supervised baseline. The current codebase is still finishing Stage I model tuning, but this document records the detailed Stage II migration design so it can be reviewed and implemented incrementally.

## Before State
The current RL system uses an IMPALA-style architecture:

1. `train.py` launches one or more local TypeScript Showdown servers.
2. Each worker process creates `BatchInferencePlayer` instances, which inherit from poke-env's `Player` class.
3. poke-env manages websocket login, challenge negotiation, battle room lifecycle, request handling, and message sending.
4. `BatchInferencePlayer` embeds a `DoubleBattle`, batches model inference asynchronously, and returns `BattleOrder` objects back through poke-env.
5. Completed trajectories are pushed back to the learner process via multiprocessing queues.

This architecture works, but it has several costs:

- Network-style overhead even though everything is local.
- Tight coupling between battle execution and poke-env's `Player` websocket model.
- Async battle orchestration complexity inside workers.
- Worker rebuild logic and stale coroutine/websocket state that have already shown up as OOM or timeout failure modes.
- A battle runtime that is harder to reason about because battle progress is callback-driven instead of a worker-owned loop.

## Problem
The project currently depends on poke-env's `Player` class for responsibilities that are orthogonal to the actual RL problem:

1. Transport to Showdown.
2. Challenge and battle room lifecycle.
3. Callback-based decision requests.
4. Websocket listener management.
5. Battle end notification.

If the Rust Showdown project is used as an in-process engine instead of as a websocket server, these responsibilities are no longer naturally provided by poke-env. That means we need a concrete plan for:

- removing the `Player` dependency from the Rust path,
- preserving the existing `DoubleBattle`-based embedding pipeline,
- keeping IMPALA training intact under our memory and compute limits,
- and breaking the migration into reviewable, separable gates.

We also need to avoid a vague migration plan. For this change to be safely reviewed and implemented, every gate needs to specify what files change, what abstractions are added, what risks are introduced, and what can be validated independently.

## Solution
The migration is a staged replacement of the worker-side battle runtime, not of the learner.

### Overall Approach
The recommended architecture is:

1. Fork `pokemon-showdown-rs` and add Python bindings directly in that fork using PyO3.
2. Treat the Rust engine as an in-process simulator, not as a websocket server.
3. Keep poke-env's `DoubleBattle` and all existing embedding/action-mask code, but stop using poke-env's `Player` class for the Rust path.
4. Create a Python adapter that drives the Rust battle engine and feeds emitted protocol messages into standalone `DoubleBattle.parse_message()` calls.
5. Replace the current async callback-driven worker loop with a synchronous worker-owned battle stepping loop.
6. Preserve the existing IMPALA multiprocessing boundaries: workers still emit trajectories, and the learner still trains from queue-fed batches.

### What Is Not Changing
The following parts of the system remain in place unless a later gate proves otherwise:

- `Embedder`
- `MDBO`
- `fast_action_mask`
- `RNaDLearner` and `PortfolioRNaDLearner`
- model architectures and optimizer logic
- curriculum weights and opponent type sampling logic at the policy level
- multiprocessing queue-based worker/learner topology
- the ability for end users to choose either backend for RL training via configuration rather than forcing an irreversible simulator migration

### Backend Support Requirement
This migration is not a one-way replacement plan anymore. The project should continue to support both RL execution backends:

- `showdown_websocket` for the traditional Pokemon Showdown server plus poke-env path
- `rust_engine` for the in-process Rust simulator path

This means the Rust work should be treated as a first-class alternative backend, not as a temporary branch that immediately deletes the existing simulator path. Unless Cayman explicitly changes that direction later, future implementation work should preserve:

- runtime backend selection in config,
- a working websocket training path for fallback and comparison,
- and side-by-side validation so throughput and training quality can be compared instead of inferred.

### What Replaces poke-env `Player`
For the Rust path, `Player` is replaced by two layers:

1. `RustBattleEngine`
   - Owns a Rust battle instance and two standalone `DoubleBattle` objects.
   - Converts Rust protocol output into poke-env battle state via `parse_message()`.
   - Owns battle lifecycle directly: reset, submit choice, inspect requests, detect completion.

2. `SyncBattleDriver`
   - Owns many `RustBattleEngine` instances per worker.
   - Detects which battles need actions.
   - Batches inference synchronously across those battles.
   - Records trajectories and finalizes rewards.

This means the Rust path will no longer rely on:

- websocket transport,
- poke-env `POKE_LOOP`,
- `BatchInferencePlayer`,
- `battle_against()` challenge negotiation,
- `stop_listening()` teardown,
- or async request callback ordering.

### Gate 0: Internal Abstraction Pass in EliteFurretAI
This gate happens before touching the Rust fork in a serious way.

#### Problem
The current RL code assumes that battle execution means `Player` objects talking to a Showdown server. That makes later integration brittle because there is no abstraction boundary around battle execution.

#### Changes
- Add an RL-side backend selection/config surface so the training system can distinguish between `showdown_websocket` and `rust_engine` battle execution.
- Introduce internal modules for:
  - team conversion from PokePaste to structured JSON,
  - a standalone Rust battle adapter,
  - and a synchronous stepping driver.
- Refactor code only enough to make backend-specific execution a first-class concept.

#### Files Expected To Change
- `src/elitefurretai/rl/config.py`
- new `src/elitefurretai/engine/team_converter.py`
- new `src/elitefurretai/engine/rust_battle_engine.py`
- new `src/elitefurretai/engine/sync_battle_driver.py`
- targeted test files under `unit_tests/rl/`

#### Why
This isolates the migration surface early and lets us write tests for the standalone `DoubleBattle` path without waiting for the full worker rewrite.

#### Validation
- New unit tests can validate protocol replay into standalone `DoubleBattle`.
- Team conversion can be tested independently from live battles.

### Gate 1: Validate Functional VGC Doubles Support in Rust

#### Problem
Before any integration work, we need to prove the Rust engine can run VGC doubles battles to completion in the formats we care about. We do not need exact JS parity at this gate; we need a functional green light.

#### Changes
Inside the forked `pokemon-showdown-rs` repo:

- Add a small executable/test harness that:
  - creates a VGC doubles battle with custom teams,
  - runs team preview,
  - submits legal choices for both sides,
  - and advances battles to completion.
- Run many seeded battles to look for panics or obviously broken state transitions.
- Spot-check that the engine handles:
  - doubles targeting,
  - team preview 4-of-6 selection,
  - Terastallization,
  - switch requests after KO,
  - and spread move execution.

#### Files Expected To Change
In the Rust fork only:

- `Cargo.toml` if helper binaries/examples are added
- a new example or test binary for VGC battle execution
- any missing doubles-specific fixes uncovered during validation

#### Why
There is no value in refactoring Python around a Rust engine that still cannot run our battles. This gate de-risks the entire effort with a functional criterion instead of a perfect parity criterion.

#### Validation
- Approximately 100 VGC battles complete without panics.
- Basic battle progression is sane enough to continue.

### Gate 2: Add Python Bindings Directly in the Rust Fork

#### Problem
The Rust engine is a library, not a Python package. We need a stable Python-facing API before the EliteFurretAI repo can integrate it.

#### Changes
Inside the Rust fork:

- Add PyO3 and maturin support.
- Expose a minimal `RustBattle` API to Python:
  - create battle from format id, seed, and structured teams,
  - submit per-side choices,
  - fetch per-side protocol messages,
  - fetch per-side request JSON,
  - inspect completion state and winner.
- Reuse the Rust engine's existing player-stream separation instead of inventing a second visibility model.

#### Files Expected To Change
In the Rust fork:

- `Cargo.toml`
- `pyproject.toml`
- binding module source files
- optionally a small Python smoke-test example

#### Why
Adding bindings directly in the fork keeps the build story simple. EliteFurretAI will consume a single package artifact instead of juggling an external wrapper crate plus a simulator crate.

#### Validation
- A Python smoke test can instantiate a battle, read messages, submit choices, and finish a battle.

### Gate 3: Standalone Rust Battle Adapter in EliteFurretAI

#### Problem
The RL code still expects poke-env battle state. We do not want to rewrite `Embedder`, action masking, or battle-state-dependent helpers immediately.

#### Changes
Create a Python adapter in EliteFurretAI that bridges the Rust binding to poke-env `DoubleBattle`.

The adapter should:

- create two standalone `DoubleBattle` objects, one for each perspective,
- feed Rust-emitted protocol lines into the correct perspective battle object,
- expose a worker-friendly API such as:
  - `reset()`
  - `needs_action(side)`
  - `request_type(side)`
  - `step(p1_choice, p2_choice)`
  - `battle_for(side)`
  - `ended`, `winner`, `turn`

This gate also includes the team conversion utility, because the Rust binding will take structured teams rather than raw PokePaste strings.

#### Files Expected To Change
- new `src/elitefurretai/engine/team_converter.py`
- new `src/elitefurretai/engine/rust_battle_engine.py`
- tests for:
  - protocol replay into `DoubleBattle`,
  - request handling,
  - team conversion,
  - end-of-battle propagation

#### Why
This gate removes the strongest conceptual dependency on poke-env `Player` while preserving the rest of the RL data pipeline.

#### Validation
- Fake-binding tests can verify that emitted protocol lines reconstruct usable `DoubleBattle` state.
- Real-binding smoke tests can confirm standalone battle progression.

### Gate 4: Synchronous Worker-Side Battle Driver

#### Problem
The current worker loop is centered around async callbacks and futures. That is unnecessary if battle stepping is local and synchronous.

#### Changes
Introduce a `SyncBattleDriver` that:

- manages a batch of live battles in one worker,
- gathers every pending decision from those battles,
- groups decisions by model identity,
- batches inference per model,
- applies actions back into the battles,
- stores per-battle hidden state,
- and finalizes trajectories when a battle ends.

The trajectory format should remain compatible with the current learner path.

#### Files Expected To Change
- new `src/elitefurretai/engine/sync_battle_driver.py`
- tests for trajectory collection, hidden-state management, reward finalization, and mixed request types

#### Why
This converts worker execution from callback-driven orchestration to explicit stepping under worker control. That is simpler to debug, easier to profile, and easier to make memory-safe.

#### IMPALA / Compute / Memory Notes
This gate still fits the IMPALA model:

- workers remain independent CPU-heavy actor processes,
- learner remains a separate GPU-heavy process,
- trajectories still cross a multiprocessing queue boundary,
- and weight updates still fan back out from learner to workers.

Expected memory effects:

- remove websocket client/listener state,
- remove stale `Player` coroutine lifetime issues,
- remove rebuild loops that retain old battle/player state,
- add only modest Rust-side battle state and Dex cache overhead.

#### Validation
- Run many battles inside a worker without RSS growth.
- Compare throughput to the websocket path.

### Gate 5: Worker Integration and Backend Switch

#### Problem
Even with a working adapter and sync driver, training still routes through the old showdown-launching path.

#### Changes
Modify worker startup and training orchestration so that `train.py` can choose backend execution at runtime.

For the Rust path:

- do not launch Showdown servers,
- do not construct `BatchInferencePlayer` instances,
- do not allocate websocket server ports,
- and do not use `server_manager` for actor execution.

For the existing websocket path:

- keep the current implementation available as a supported training option even after the Rust path is usable.

#### Files Expected To Change
- `src/elitefurretai/rl/train.py`
- `src/elitefurretai/rl/opponents.py`
- optionally new backend-specific worker modules if the split becomes cleaner than conditional branching

#### Why
This gate is the point where the new runtime becomes usable in training runs while keeping rollback cheap and preserving an apples-to-apples comparison path for future tuning.

#### Validation
- End-to-end short RL run on the Rust backend.
- Existing websocket backend still works as a supported alternative backend.

### Gate 6: Cleanup, Profiling, and Full Adoption

#### Problem
After integration, there will likely be duplicate code paths and unproven performance claims.

#### Changes
- Profile battles/sec, decisions/sec, and worker RSS.
- Compare training stability and rollout quality.
- Remove dead or now-obsolete worker rebuild logic if the Rust path proves stable.
- Keep the websocket backend unless Cayman explicitly decides to retire it later; the default expectation is continued dual-backend support.

#### Why
The point of this migration is operational leverage. We should not keep the extra complexity indefinitely unless it continues to buy us something concrete.

#### Validation
- sustained training without actor OOM or hung battle loops,
- clear throughput improvement or operational reliability win,
- and no meaningful regression in data quality.

## Reasoning
This approach is better for building the best VGC bot for four reasons.

1. It attacks the data collection bottleneck directly.
   The learner is only as good as the battle data it receives. Faster, more reliable rollouts improve the throughput of self-play and league training.

2. It removes infrastructure that does not help policy quality.
   Websocket orchestration, server lifecycle, and async callback wiring do not make the bot stronger. They only increase system complexity.

3. It preserves the most battle-tested parts of the current stack.
   The embedding, masking, and action encoding layers already understand poke-env `DoubleBattle`. Keeping those stable reduces migration risk.

4. It matches our hardware constraints better.
   We are operating under WSL memory limits, and the existing worker leaks/timeouts are already evidence that simpler worker lifetimes matter.

## Planned Next Steps
1. Land Gate 0 in EliteFurretAI:
   - backend selection plumbing,
   - team conversion utility,
   - standalone Rust battle adapter scaffold,
   - and tests that validate standalone `DoubleBattle` updates.
2. Fork `pokemon-showdown-rs` and run Gate 1 validation.
3. Add PyO3 bindings in the fork once Gate 1 is green.
4. Wire the real binding into the EliteFurretAI adapter.
5. Replace worker execution with the sync driver under a config flag.
6. Run side-by-side backend comparisons before making Rust the default.

## Updates
- 2026-04-05 16:20: Initial detailed migration plan written.
- The first implementation target in this repo is Gate 0, because it is the smallest slice that removes conceptual dependence on poke-env `Player` without requiring the Rust fork to be finished first.
- 2026-04-05 17:05: Gate 0 partially implemented in EliteFurretAI.
   - Added `battle_backend` config support with explicit backend validation in `src/elitefurretai/rl/config.py`.
   - Added `src/elitefurretai/engine/team_converter.py` to convert PokePaste exports into structured team JSON compatible with the planned Rust binding API.
   - Added `src/elitefurretai/engine/rust_battle_engine.py`, which owns standalone `DoubleBattle` objects and synchronizes them from a Rust-style binding interface without using poke-env `Player`.
   - Added focused unit tests covering config validation, team conversion, and standalone battle adapter stepping.
   - Verified the new code with focused pytest runs.
- 2026-04-05 17:05: Deliberate non-implementation at this stage.
   - The live Rust backend is not wired into worker execution yet because the forked `pokemon-showdown-rs` bindings do not exist inside this workspace.
   - `BatchInferencePlayer`, websocket battle execution, and server launching remain the active training path until Gate 1 and Gate 2 are completed in the Rust fork.
- 2026-04-06 00:10: Gate 1 through Gate 4 were advanced into a live benchmarkable path across both repos.
   - Added a real PyO3 binding in the forked `pokemon-showdown-rs` repo with a `RustBattle` class that exposes battle construction, per-side request JSON, log draining, winner state, and combined doubles choice submission.
   - Confirmed the Rust fork is an in-process simulator, not a websocket server, so the Python integration path remains binding-driven rather than transport-driven.
   - Added `src/elitefurretai/engine/sync_battle_driver.py` as the first usable synchronous actor loop and `src/elitefurretai/engine/rust_engine_benchmark.py` as a constrained RL-style benchmark entrypoint.
   - Verified the benchmark path can collect lightweight rollout steps when requested, but the stable 1000-battle benchmark was run in the minimal no-rollout mode.
- 2026-04-06 00:10: Concrete protocol/request compatibility learnings from the real Rust integration.
   - The Rust fork emits valid battle information, but its protocol and request shapes are not a drop-in match for poke-env doubles parsing.
   - The binding originally forwarded a full combined doubles choice string into `Battle.choose`, which only accepts one per-Pokemon choice at a time; the binding had to be updated to split combined side choices before committing the turn.
   - Rust logs include unslotted identifiers such as `p1: Iron Hands`, bare slot tokens such as `p1a`, and perspective-relative slot tokens such as `1a` and `2a`; Python-side normalization is required before these lines can be consumed by standalone `DoubleBattle` objects.
   - Rust request JSON condition strings include trailing percentage tokens such as `261/261 100`; these must be sanitized before reuse on the poke-env side.
   - Rust side request ordering can diverge from the active slot ordering that poke-env expects, so request sanitization must reorder active entries defensively.
   - Some Rust protocol events still do not map cleanly into poke-env's parser, and some late-battle request shapes can still desynchronize move ownership from poke-env state. For the throughput benchmark, the pragmatic fix was to drive legality directly from Rust request JSON and treat poke-env state as best-effort auxiliary state rather than the source of truth for action legality.
- 2026-04-06 00:10: Benchmark result for the first constrained Rust actor path.
   - Command run: `PYTHONPATH=/home/cayman/Repositories/EliteFurretAI/src /home/cayman/Repositories/venv/bin/python -m elitefurretai.engine.rust_engine_benchmark --battles 1000`
   - Result: `completed_battles=1000`, `truncated_battles=254`, `duration_seconds=118.030`, `battles_per_second=8.472`, `p1_decisions=34779`, `decisions_per_second=294.663`.
   - Compared to the old websocket baseline of `1.4 battles/sec`, this benchmarked path is approximately `6.05x` faster on raw completed-battle throughput.
   - Important caveat: this result comes from a bounded benchmark path with turn-cap and stall-cap protections, and it does not yet represent a full training-quality replacement for the websocket worker path. It is a performance green light for continuing the migration, not final parity proof.
- 2026-04-06 00:10: Immediate next implementation gap after the benchmark.
   - The next quality gate is to replace the benchmark-specific best-effort state handling with a training-safe state synchronization path so that rollouts can rely on fully coherent poke-env battle state without truncation-heavy safeguards.
   - `train.py` still does not route actor execution through the Rust backend; the live 1000-battle benchmark proves the core in-process stepping path, but not end-to-end learner integration.
   - We want the `train.py` integration step because a standalone benchmark only proves local stepping throughput. The actual training path also has to preserve queue semantics, hidden-state lifetimes, opponent sampling, worker restart behavior, and learner-facing trajectory formatting. Until the Rust path goes through the real actor entrypoint, we have not proven that it can replace production self-play.
   - We want the training-safe state synchronization step because the model, embedder, reward shaping, and debugging tools all still assume that poke-env battle state is trustworthy. In the benchmark path, legality comes from Rust request JSON and battle state is only maintained on a best-effort basis. That is acceptable for throughput measurement, but it is not acceptable for learning because corrupted or partially synchronized state would poison observations, action masks, auxiliary features, and any future debugging of bad policies.
   - We want focused tests around truncation, stall handling, and combined-choice parsing because the current benchmark relies on pragmatic safeguards rather than clean semantic parity. If those safeguards regress silently, we could end up measuring the fallback behavior rather than the real simulator path, or worse, training on silently malformed trajectories.
- 2026-04-06 01:05: Gate 5 now has an initial Rust-backed training branch in the real entrypoint.
   - `src/elitefurretai/rl/train.py` now branches on `battle_backend == "rust_engine"` inside `mp_worker_process` and routes that worker through `SyncRustBattleDriver` instead of `WorkerOpponentFactory` and websocket battles.
   - The Rust worker path currently runs bounded self-play only: the same main `RNaDAgent` is used on both sides, the learner-facing side collects trajectories, and the opponent side exists only to supply legal self-play actions.
   - Weight broadcasts from the learner now update the live Rust actor model in place and also refresh sampling controls (`temperature` and `top_p`) on the synchronous policy players, so the Rust path matches the existing learner-to-worker control surface.
   - `train.py` also now skips Showdown server launch, websocket port allocation, and external vgc-bench runner launch when the Rust backend is selected. This keeps the old websocket backend intact while removing unnecessary process infrastructure from the Rust path.
- 2026-04-06 01:05: Gate 4 trajectory handling was tightened enough for learner integration.
   - `src/elitefurretai/engine/sync_battle_driver.py` now includes a model-backed `SyncPolicyPlayer` that emits learner-compatible trajectory steps with `state`, `action`, `log_prob`, `value`, reward finalization, and optional masks.
   - This is intentionally aligned with the semantics already used by `BatchInferencePlayer`, so the learner continues to consume the same rollout contract instead of needing a separate Rust-specific batch format.
   - Added a focused unit test covering the new synchronous policy path and verified it together with the existing config and Rust adapter tests.
- 2026-04-06 01:20: Backend-support direction clarified.
   - The project should continue to support both RL training backends: the traditional `showdown_websocket` simulator path and the `rust_engine` path.
   - Future migration work should preserve configuration-based backend selection and treat the websocket path as a supported comparison and fallback path rather than planning to remove it by default.
- 2026-04-06 07:30: Rust backend now passes a real `train.py` smoke run and has moved beyond pure self-play.
   - The Rust worker loop now samples opponent policies from curriculum instead of hard-wiring mirror self-play. The implemented Rust-side opponent set currently covers the policy-backed opponent types that matter for the default curriculum: `self_play`, `ghosts`, and `exploiters`, plus `bc_player` when a compatible BC checkpoint is provided.
   - `SyncRustBattleDriver` now carries per-battle policy selection and per-battle opponent labels, so learner trajectories keep the correct `opponent_type` metadata even when one worker batch mixes opponent sources.
   - `RustBattleEngine` now prefers `DoubleBattle.parse_request()` over the old flag-only update path, which materially improves synchronization of available moves, switches, force-switch state, and active-slot bookkeeping. The older private-field path remains only as a fallback for request shapes poke-env still rejects.
   - A one-update Rust smoke config was added at `src/elitefurretai/rl/configs/rust_smoke.yaml` and executed through the real `train.py` entrypoint. The run completed with a final checkpoint at `data/models/rl/rust_smoke/main_model_step_1.pt`, which confirms that Rust worker trajectories were collected and consumed into at least one learner update.
   - The smoke run also exposed and fixed a real team-conversion bug: PokePaste headers with trailing gender markers such as `(F)` were being misparsed as species names. `src/elitefurretai/engine/team_converter.py` now strips header-level gender markers before nickname/species parsing, and a regression test was added.
- 2026-04-06 07:30: Remaining limitations after the first end-to-end smoke validation.
   - The smoke run had to generate a local current-architecture bootstrap checkpoint for BC/ghost validation because none of the checked-in historical RL or supervised checkpoints are compatible with the current model architecture. This is now a concrete repo-state issue rather than an abstract risk.
   - The Rust path no longer depends primarily on benchmark-era legality hacks, but it still retains battle turn/stall guardrails as a safety net. Those guards are now defensive rather than foundational, and future work should shrink or remove them only after longer multi-update runs show stable request parsing over time.
- 2026-04-06 07:58: Mixed-opponent Rust training now completes a short multi-update run and exposes real truncation telemetry.
   - Expanded the Rust-side opponent layer beyond policy-only sampling by adding local baseline controllers for `max_damage`, `simple_heuristic_baseline`, `max_base_power_baseline`, and `random_baseline`. These controllers reuse battle-state-driven heuristic players without starting poke-env websocket listeners, which keeps the Rust path transport-free while preserving curriculum semantics.
   - Fixed a sync-driver regression where `SyncPolicyPlayer` stopped issuing legal choices after `max_battle_steps` and returned `default` forever. In practice this could stall mixed-opponent battles until the driver's turn/stall guardrails fired, producing zero learner trajectories even when the underlying battle engine was still healthy. The sync path now continues making legal choices after the trajectory cap while only recording the first `max_battle_steps` rollout steps.
   - Added focused regression coverage for the above behavior in `unit_tests/engine/test_sync_battle_driver.py`.
   - Added worker-process logging initialization inside `mp_worker_process` so Rust actor batch summaries are visible from subprocesses during real `train.py` runs. This matters because the parent process was otherwise hiding the very truncation telemetry needed to judge whether request parsing is stable enough for longer curriculum runs.
   - Re-ran `src/elitefurretai/rl/configs/rust_multiupdate.yaml` through the real `train.py` entrypoint. The run completed `3` learner updates and saved `data/models/rl/rust_multiupdate/main_model_step_3.pt`.
   - Measured Rust worker truncation during that mixed curriculum run from live worker summaries: after `18` batches / `36` completed battles, cumulative truncation settled at `23 / 36 = 0.639`. Batch-level truncation varied materially by matchup, ranging from `0.000` to `1.000`, which indicates request parsing is no longer failing catastrophically but mixed-opponent battle completion is still guardrail-heavy.
   - Immediate interpretation: the Rust backend is now good enough for short mixed-opponent learner updates, but it is not yet training-clean. The dominant remaining quality issue is not worker startup anymore; it is the high proportion of battles still ending via turn/stall truncation rather than natural resolution under the current guardrail settings and state-sync quality.