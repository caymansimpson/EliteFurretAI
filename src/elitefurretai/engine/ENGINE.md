# Engine Runtime Guide

This document is the durable reference for EliteFurretAI's battle-execution layer. It consolidates the current backend decision, the key Stage 2 learnings, and the ownership boundaries for the engine package.

## Current Decision

There are still two supported RL execution backends:

- `showdown_websocket`: local Pokemon Showdown servers plus poke-env websocket players
- `rust_engine`: the in-process Rust simulator plus synchronized standalone `DoubleBattle` state

The current project decision is:

- optimize and operate around `showdown_websocket` for near-term RL work
- keep `rust_engine` as a simpler fallback, comparison, and debugging backend

This is not a theoretical preference. It is based on the latest paired learner-facing comparison and the broader Stage 2 benchmark history.

## What Stage 2 Taught Us

### 1. Random battle throughput is a weak decision metric

Random-policy engine stepping overstated Rust's advantage and sometimes overstated Showdown's concurrency benefit. The meaningful comparison had to include:

- embedding
- action masking
- policy inference
- legality handling
- actual training-loop update cadence

The most decision-relevant numbers came from model-backed benchmarks and paired `train.py` runs, not from pure environment stepping.

### 2. Showdown won the latest learner-facing wall-clock comparison

In the paired 10-update comparison on the current Transformer + `full` feature path:

- Rust completed `640` battles in `35m 23s`
  - `0.30 battles/s`
  - `19011 learner steps`
  - `8.95 learner steps/s`
- Showdown completed `640` battles in `13m 27s`
  - `0.79 battles/s`
  - `6447 learner steps`
  - `7.98 learner steps/s`

Interpretation:

- Showdown reached the same `10` learner updates about `2.6x` faster in wall-clock time.
- Rust still produced slightly more learner steps per second.
- The runs were not semantically identical because Rust produced many more learner steps per trajectory than Showdown.

That means the backend decision cannot be made from one scalar metric, but it is enough to justify moving forward with Showdown as the actively optimized path.

### 3. Rust's hardest problems were not raw simulator speed

The important Rust learnings from the Stage 2 investigation were:

- request sanitization was mandatory
- cached request reuse was useful and worth keeping
- batched local policy inference was important and worth keeping
- the faster embedder array path was useful and worth keeping as an automatic default
- synthetic teampreview compatibility was necessary because the binding did not expose the full preview stage directly
- battle validity and truncation behavior mattered more than pure step speed

In practice, the main Rust losses came from:

- long / truncated trajectories
- legality drift and retry behavior
- Python-side synchronization and observation reconstruction costs

### 4. Several Rust optimizations were not worth their maintenance cost

The following ideas were explored and are now intentionally not part of the maintained Rust fallback path:

- centralized GPU inference transport for Rust actors
- benchmark-only dual adapter modes for cached vs uncached request access in normal runtime code
- extra public toggles for turning batching and fast embedding on and off in the maintained runtime
- Transformer context-length batch grouping complexity in the sync policy path

Why they were removed or de-emphasized:

- centralized inference added transport complexity and did not win in the learner-facing comparisons
- the extra adapter and optimization toggles were mostly useful for historical sweeps, not ongoing maintenance
- the Rust backend is no longer the primary optimization target, so the code should bias toward clarity over maximum throughput tuning

### 5. Showdown is not “solved” either

The latest Showdown run was faster to updates, but it still logged a large number of invalid websocket choices. The recommendation to move forward with Showdown means:

- Showdown is the better place to spend optimization effort now
- not that Showdown is already clean
- not that Rust learned nothing useful

The best Rust lessons transferred directly into the broader backend decision:

- validity and usable training signal matter more than raw battle stepping speed
- battle throughput without legality quality can be misleading
- end-to-end learner behavior must dominate backend choices

## Simplified Rust Runtime Policy

The maintained Rust fallback runtime is intentionally simpler now.

### Kept

- cached request access around the compiled binding
- request sanitization and active-slot reconciliation
- standalone `DoubleBattle` synchronization
- local policy inference inside the worker
- batched local LSTM inference when natural batch opportunities exist
- automatic use of `embed_to_array()` when the embedder exposes it
- diagnostics and trace capture for rejected choices and problematic battles

### Removed Or De-Prioritized

- centralized GPU inference for Rust workers
- dual binding adapter surfaces as a maintained runtime feature
- benchmark-only optimization switches in the normal Rust runtime path
- extra Transformer batching complexity in the sync policy layer

### Design Rule

The Rust backend should remain:

- runnable
- documented
- debuggable
- easy to compare against Showdown

It does not need to remain the most aggressively tuned path unless the project explicitly re-prioritizes it later.

## Engine Package Layout

The engine package owns battle-execution concerns rather than RL algorithm concerns.

### Rust runtime modules

#### `battle_snapshot.py`

Purpose:

- defines the policy-facing observation object for the Rust self-play path
- packages synchronized battle state, legal choices, request state, and optional cached vectors

How it helped speed up Rust:

- created a stable Python-owned observation boundary
- let the sync driver pass prebuilt legality and optional precomputed vectors without rebuilding ad hoc structures at each consumer
- made it possible to separate runtime stepping from policy evaluation more cleanly

#### `rust_battle_engine.py`

Purpose:

- wraps the compiled Rust battle binding
- converts PokePaste text into the Rust binding team schema
- caches request JSON parsing
- sanitizes request payloads for poke-env compatibility
- replays Rust protocol into standalone `DoubleBattle` objects
- exposes protocol history and snapshots for diagnostics

How it helped speed up Rust:

- eliminated repeated request parsing at every Python consumer
- moved team conversion into the same module that owns the Rust binding boundary
- reduced synchronization bugs that previously created wasted retries and invalid state handling
- kept the binding boundary centralized so fixes landed once instead of in many driver call sites

What we keep from Stage 2:

- cached request access
- request sanitization
- protocol-history diagnostics

What we intentionally do not keep as a maintained runtime surface:

- multiple adapter modes for benchmark ablations

#### `sync_battle_driver.py`

Purpose:

- runs synchronous Rust battles inside a worker-owned loop
- gathers pending decisions
- asks policies for choices
- retries rejected choices with fallbacks
- records trajectories and diagnostics
- finalizes completed battles for the learner

How it helped speed up Rust:

- replaced callback-heavy websocket orchestration with a worker-owned step loop
- enabled batched local policy inference across simultaneous battles
- reduced wasted time from infinite retry loops by adding fallback and truncation handling
- automatically uses the faster embedder array path when available

What we keep from Stage 2:

- local batched LSTM inference
- fallback handling and diagnostics
- automatic fast embedding path

What we intentionally simplified:

- no central GPU inference path
- no public runtime optimization toggles
- no maintained Transformer context-length batching machinery

#### `rust_engine_benchmark.py`

Purpose:

- lightweight engine-level benchmark for the Rust fallback runtime
- useful when checking that battle stepping still works after engine changes

How it helped speed up Rust:

- gave a quick way to separate pure stepping issues from model-backed training issues
- helped show that raw engine stepping was not the dominant bottleneck in the full RL path

#### `rust_model_benchmark.py`

Purpose:

- model-backed benchmark for the Rust fallback runtime
- exercises embedding, masking, policy inference, and synchronous battle stepping together

How it helped speed up Rust:

- identified that batched policy inference and request caching were real wins
- exposed that centralized GPU transport was not worth its complexity
- made it obvious that learner-facing validity and truncation behavior mattered more than raw simulator speed

This benchmark now tracks the simplified maintained runtime rather than the old optimization-ablation surface.

### Showdown runtime modules

#### `showdown_server_manager.py`

Purpose:

- owns local Showdown server lifecycle and external vgc-bench runner lifecycle

#### `showdown_benchmark.py`

Purpose:

- provides the checked-in websocket comparison harness used to compare Showdown and Rust under both random and model-backed conditions

Important Stage 2 learning:

- the Showdown benchmark had to be fixed to wait for login through `PSClient.wait_for_login()` rather than cross-loop event waiting

## Ownership Guide

Use this when deciding where work belongs.

### Put work in `rust_battle_engine.py` when

- requests are malformed
- active-slot ordering is wrong
- protocol normalization is wrong
- poke-env synchronization from Rust output is wrong

### Put work in `sync_battle_driver.py` when

- batching or stepping behavior is wrong
- fallback behavior is wrong
- truncation behavior is wrong
- trajectory formatting is wrong
- diagnostics need more runtime context

### Put work in `showdown_server_manager.py` when

- server launch, shutdown, or port allocation is wrong
- external vgc-bench process management is wrong

### Put work in RL modules when

- legality should be shared across both backends
- masking logic belongs to both backends
- policy or learner logic is changing regardless of backend

## Practical Recommendation Going Forward

If the question is “where should we spend engineering time next for RL performance?”, the default answer should be Showdown.

Choose Showdown-first work when the goal is:

- faster end-to-end learner updates
- improving websocket legality handling
- improving training signal per wall-clock hour
- stabilizing the primary RL training path

Choose Rust work only when the goal is:

- keeping the fallback backend healthy
- debugging backend parity questions
- testing whether a new validity fix changes the Rust-vs-Showdown tradeoff
- revisiting Rust as the primary path after Showdown improvements materially plateau

## Related Docs

- `src/elitefurretai/rl/RL.md`
- `planning/stage2/2026-04-13-11-20-showdown-benchmark-and-backend-comparison.md`
- `planning/stage2/2026-04-13-14-00-training-speed-recommendations.md`
- `planning/stage2/2026-04-09-09-35-rust-model-backed-throughput-sweep.md`
- `planning/stage2/2026-04-06-12-00-rust-throughput-optimization-plan.md`
