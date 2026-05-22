# Showdown Training Summary And Transformer Actor Batching Plan

## Context

This note records a simplified Stage 2 description of the current Showdown-backed single-team RL training flow after the recent profiling passes, explicitly ignoring the pathological `vgc_bench_baseline` path.

The immediate goal is not to document every micro-optimization, but to answer three practical questions:

1. what the healthy Showdown training flow looks like end to end
2. where wall-clock time is currently spent in that flow
3. whether true batched actor inference for the current transformer policy is a realistic path toward roughly `2x` faster learner-facing training

This note is based on:

- `planning/stage2/2026-04-15-13-30-showdown-training-flow-annotated.md`
- `planning/stage2/2026-04-15-13-00-showdown-single-team-sample-profile.md`
- `planning/stage2/2026-04-13-23-55-showdown-training-profile-and-speed-plan.md`
- the current Showdown actor implementation in `src/elitefurretai/rl/players.py`
- the current transformer model implementation in `src/elitefurretai/supervised/model_archs.py`

Representative current config shape:

- `battle_backend: showdown_websocket`
- `use_transformer: true`
- `transformer_layers: 7`
- `batch_size: 8`
- `batch_timeout: 0.01`
- `num_workers: 4`
- `num_players: 12`
- `num_servers: 4`
- `use_central_gpu_inference: false`

## Before State

Before this note:

- the project already had strong evidence that learner-side math was not the main Showdown bottleneck
- the longer annotated run established that collection time dominates update wall-clock
- the sample benchmark established that actor inference, websocket/message handling, and embedding are the main healthy-path costs
- there was still no simple planning note focused specifically on how the non-VGCBench Showdown flow should be described or why transformer actor batching is valuable enough to prioritize

There was also an important code-level gap in the current implementation:

- `RLTrajectoryPlayer` already batches the LSTM path
- but the transformer path does **not** run a true batched forward pass across multiple battles
- instead, it loops over battles one by one inside `_run_batch()` with the in-code comment: `TODO: pad contexts for true batched Transformer inference`

That means the current async queue batches request scheduling, but not the expensive transformer forward itself.

## Problem

The user wants a simple Stage 2 summary of the Showdown training flow and a clearer implementation plan for batched actor inference, including:

1. why batching the actor is expected to help
2. how it would actually work in code
3. how it interacts with future choices such as actor CPU vs GPU placement, batch size, timeout, and concurrency

Without writing that down clearly, it is easy to overestimate or underestimate the likely value of batching.

## Solution

### 1. Simple description of the healthy Showdown training flow

Ignoring VGCBench stalls, the healthy Showdown flow is:

1. the trainer launches local Showdown servers and multiprocessing workers
2. each worker creates paired players and enters the websocket battle loop
3. when Showdown sends a decision request, the actor:
   - reads the request and battle state
   - embeds the current battle into a feature vector
   - computes the action mask
   - runs the policy forward pass
   - samples or selects an action
   - sends the chosen command back to Showdown
4. completed battles become trajectories and are transferred back to the learner queue
5. the learner waits until enough trajectories arrive, collates them, performs the PPO/RNaD update, and broadcasts weights

The critical operational point is that the learner mostly waits for trajectories. The expensive part is generating battle decisions fast enough, not applying gradients.

### 2. Time spent at each stage

#### End-to-end update level

For the healthy no-VGCBench path, the learner update is cheap relative to collection.

Observed examples from the profiling notes:

- one-update sample run: about `62s` for `64` battles and `636` learner steps
- longer two-update no-VGCBench comparison run: about `93s` total for `2` updates
- learner-side training time inside longer runs: about `1-2s` per update once data is available

Interpretation:

- end-to-end wall-clock is overwhelmingly collection-side
- learner math is not the current first-order bottleneck

#### Worker batch level

Representative successful worker batches in the annotated flow note were about `6s` to produce `4` trajectories.

Examples:

- self-play batch: about `5.96s`, `4` trajectories, `27` steps, `97` requests, `5.78s` inference executor time
- max-damage batch: about `6.35s`, `4` trajectories, `60` steps, `60` requests, `4.12s` inference executor time

Interpretation:

- healthy batches spend most of their time inside the battle request loop
- request handling and actor inference are tightly coupled in wall-clock

#### Actor-side profile level

The model-backed Showdown benchmark and profile show the main healthy-path hot buckets.

Representative cumulative hotspots:

- `selectors.select`: about `20-23s`
- actor inference path (`_gpu_inference_sync` plus model forward): about `20-22s`
- `Player._handle_battle_message`: about `5.7s`
- `_embed_battle_state`: about `4.7-4.9s`
- `Embedder.embed_to_array`: about `4.6-4.7s`
- `generate_feature_engineered_features`: about `3.0s`
- `calculate_damage`: about `3.1-3.3s`

Interpretation:

- the actor loop is a combined systems bottleneck, not a single isolated line of code
- the biggest buckets are websocket/event-loop waiting, policy inference, and embedding/feature generation

### 3. Why transformer actor batching is valuable

The evidence is stronger than “batching is usually good.” It is specific to this code path.

#### Evidence A: the current training configs use the transformer path

The active Showdown Stage 2 configs use:

- `use_transformer: true`
- `transformer_layers: 7`
- `transformer_heads: 16`

So the relevant actor path is the transformer branch, not the LSTM branch.

#### Evidence B: the current transformer branch is not truly batched

In `src/elitefurretai/rl/players.py`, `_run_batch()` explicitly handles transformer inference one battle at a time because contexts have variable length. The code comment says:

- `TODO: pad contexts for true batched Transformer inference`

So today:

- multiple requests are gathered into a batch
- but the expensive transformer forward is still called once per battle
- the current queue provides scheduling aggregation, not compute aggregation

This means the current design still pays Python overhead, model-call overhead, and device-transfer overhead per battle request instead of per batch.

#### Evidence C: the model already has the pieces needed for padded batch execution

`TransformerThreeHeadedModel.forward()` already supports padded sequences with a mask for supervised training.

The missing piece is the online RL method:

- `forward_with_hidden()` accumulates a per-battle context tensor
- but it does not currently accept a per-sample padding mask for variable-length contexts

That is an implementation gap, not a conceptual blocker.

#### Evidence D: inference is already one of the biggest healthy-path buckets

The current actor-side profiles show about `20-22s` in inference/model-forward buckets within the benchmarked run shape. Even a partial reduction there matters because collection dominates end-to-end update time.

The realistic argument is not that batching alone will deliver `2x`. The argument is that batching attacks one of the very largest healthy-path costs and is likely one of the few changes with enough leverage to matter materially.

### 4. What true transformer actor batching would do

The target behavior is simple:

- instead of running the transformer once per ready battle request
- gather all ready battle states in the current micro-batch
- pad their historical transformer contexts to a common length
- run **one** transformer forward over the whole batch
- split the outputs back into one result per battle

Conceptually:

Current path:

- battle A forward
- battle B forward
- battle C forward
- battle D forward

Target path:

- stack A/B/C/D into one padded batch
- run one forward on batch size `4`
- slice outputs back to A/B/C/D

Benefits:

- fewer Python-to-PyTorch transitions
- fewer device transfers and dispatches
- better BLAS / kernel utilization on both CPU and GPU
- lower per-request overhead relative to useful compute

### 5. Implementation plan for true transformer batching

#### Step 1: represent batched transformer context explicitly

Today each battle stores:

- `hidden_states[tag] = context_tensor` with shape `(1, T, H)` or `None`

The batched path should build:

- `context_batch`: `(B, T_max, H)`
- `context_mask`: `(B, T_max)` where `True` means real token and `False` means padding
- `context_lengths`: optional helper list for slicing outputs back

This is directly analogous to standard sequence padding.

#### Step 2: extend `forward_with_hidden()` to accept padded context masks

The transformer online forward should accept something like:

- `hidden_state`: `(B, T_max, H)`
- `hidden_mask`: `(B, T_max)`

Then it should:

- append the current encoded step to each sequence
- extend the mask by one valid position per sample
- prepend decision tokens as usual
- build a `src_key_padding_mask` that marks padded context positions as masked out
- run the transformer once over the padded batch

This mirrors what `forward()` already does for supervised sequence padding.

#### Step 3: batch all ready transformer requests inside `_run_batch()`

Instead of the current per-item loop, `_run_batch()` should:

1. collect all `states_np`
2. collect each battle’s stored context tensor
3. pad contexts to `T_max`
4. run one executor call to `_gpu_inference_sync`
5. receive batched logits, values, and next contexts
6. unpad or slice the returned contexts back into per-battle `hidden_states[tag]`

The key point is that the expensive call becomes one batched call.

#### Step 4: keep masks and stale-request safeguards unchanged

The existing request snapshot, stale-generation checks, and action-mask logic should remain in place.

The batching change should affect only how inference is performed, not the legality/correctness guards around request handling.

#### Step 5: add batching diagnostics before and after

The change should log at least:

- transformer batch size distribution
- transformer padded context length distribution
- average real context length versus padded length
- inference executor seconds before and after
- learner steps/s before and after on the same config

This is necessary because padding can introduce wasted compute if most batches contain one long sequence and many short ones.

### 6. Expected value and realistic speedup range

The correct expectation is “meaningful but not magical.”

Best-case reasoning:

- today the transformer path pays forward-call overhead once per battle request
- batching amortizes Python overhead and improves dense-math utilization
- inference is one of the largest healthy-path buckets

Conservative expectation:

- true transformer batching alone is more likely a `1.2x-1.6x` class end-to-end win than a guaranteed `2x`

Stronger upside exists if:

- batches are consistently well-filled
- actor inference remains local and avoids expensive inter-process transport
- embedding is also optimized afterward

This is why batching looks like one of the highest-leverage single improvements, but not a complete performance plan by itself.

### 7. Interaction with future parameter choices

#### Actor CPU vs GPU

Batching helps on both CPU and GPU, but for different reasons.

On CPU:

- fewer model invocations
- better use of large dense linear algebra kernels
- less Python overhead per decision

On GPU:

- batching is even more important because small single-item forwards underutilize the device
- but naive per-worker GPU actors can contend with the learner GPU and may still lose end-to-end

The current Showdown evidence says that simply moving actors to CUDA is not enough. Batched CPU actor inference is therefore a sensible first target, and batched GPU inference only becomes attractive if the transport and contention story is also good.

#### `batch_size`

Larger `batch_size` increases the upside of batching because more requests can be processed together.

But larger `batch_size` only helps if the queue fills quickly enough. If not, it mostly increases waiting or leaves batch slots unused.

#### `batch_timeout`

This setting becomes more important after real batching exists.

- too low: batches stay tiny and batching gains are muted
- too high: actors wait too long before responding, increasing stale-request risk and battle latency

The right value depends on request arrival rate from all concurrent battles.

#### `num_workers`, `num_players`, `num_servers`

More concurrent battles generally improve batch fill and make batching more valuable.

But more concurrency also increases:

- websocket scheduling pressure
- stale-result risk
- memory usage
- legal-action churn if correctness bugs still exist

So batching and concurrency tuning should be measured together, not independently.

#### Model family

This specific optimization is highest-value for the current transformer configs because that path is currently under-batched.

For LSTM models, the player already performs batched hidden-state concatenation, so the marginal value of this exact project is much smaller.

## Reasoning

This plan helps build the best VGC bot because it focuses on turning wall-clock into learner signal instead of optimizing already-cheap learner math.

The strongest current non-VGCBench bottleneck picture is:

1. collection dominates update wall-clock
2. inside collection, actor inference is one of the largest healthy-path costs
3. the current transformer actor path is provably under-batched in code
4. the model already has enough padding infrastructure that true online batching is a practical implementation target

So transformer actor batching is worth prioritizing not because batching is fashionable, but because the current code is leaving a large obvious optimization on the table.

## Planned Next Steps/Implementation Plan

1. Add a padding-mask-aware transformer online forward path in `TransformerThreeHeadedModel.forward_with_hidden()`.
2. Replace the transformer per-battle loop in `RLTrajectoryPlayer._run_batch()` with padded batched execution.
3. Preserve existing stale-request and action-mask correctness guardrails unchanged.
4. Add transformer-specific batch diagnostics so padding efficiency can be measured directly.
5. Re-run the current no-VGCBench two-update Showdown profile on the same config.
6. If batching improves inference materially, tune `batch_size` and `batch_timeout` afterward rather than before.
7. Only revisit actor-on-GPU experiments after the batched local path is measured, because naive CUDA actors were not enough by themselves.

## Updates

- Created this planning note to capture the simplified non-VGCBench Showdown flow and the rationale for true transformer actor batching.
- Confirmed from `src/elitefurretai/rl/players.py` that the transformer path is still not truly batched and currently falls back to one-battle-at-a-time execution.
- Confirmed from `src/elitefurretai/supervised/model_archs.py` that the supervised transformer path already supports padded masking, making online batched transformer inference a practical extension rather than a ground-up redesign.
- Implemented a first-pass true batched transformer actor path in `src/elitefurretai/rl/players.py` by padding per-battle transformer contexts to a batch-local maximum length, constructing a boolean context mask, running a single forward pass for the batch, and slicing the returned contexts back per battle tag.
- Extended `TransformerThreeHeadedModel.forward_with_hidden()` in `src/elitefurretai/supervised/model_archs.py` to accept an optional `hidden_mask` so padded online contexts can participate in the transformer without attending to padding tokens.
- Added transformer batching diagnostics for batched-call count, number of context items, real context tokens, padded context tokens, and max context length so the padding efficiency of future profiles can be measured directly.
- Restored missing runtime helpers in `src/elitefurretai/rl/fast_action_mask.py` (`get_valid_targets_for_request_move`, `slot_is_commanding`, and the optional `request_override` parameter) after a pre-existing import break blocked direct Python validation and benchmark startup.
- Verified the new masked transformer online path with a focused Python smoke: a padded hidden-state batch with shape `(3, 5, 1024)` and mask shape `(3, 5)` produced valid outputs and an updated context with shape `(3, 6, 1024)`.
- Re-ran the model-backed Showdown benchmark on the no-VGCBench shape. The first profiled run showed a cold-start-heavy total duration (`62.701s`) but still improved the actual `battle_loop_seconds` to `12.794s`. A second non-profile rerun on the same shape completed in `14.163s` at `0.424` battles/s with `battle_loop_seconds=7.612`, confirming that the healthy execution loop is materially faster than the previously documented `16.206s` sample.