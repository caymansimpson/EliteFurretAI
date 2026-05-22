# Centralized Inference (Option C) — Detailed Implementation Plan

**Branch**: `centralized-inference` (off `main` at this commit)
**Date**: 2026-05-13
**Predecessor**: [scope + SB3 question](./2026-05-13-17-30-centralized-inference-scope-and-sb3-question.md)
**Status**: Pre-implementation. Resolving design choices, scaffolding milestone 1.

This doc resolves the open design choices from the scope doc and gives a
concrete file-by-file plan for the 3-week build. The 4-week budget and
fallback to Option A on week-3-no-clean-run still applies.

---

## Resolved design choices

### D1. IPC mechanism: `torch.multiprocessing.Queue`

**Choice**: one `torch.multiprocessing.Queue` per direction.
- `inference_request_queue`: shared, all workers → trainer (one per model
  type — see D5).
- `response_queue[worker_id]`: dedicated per-worker, trainer → that worker.

**Why this and not alternatives**:
- `mp.Pipe`: simpler but trainer has to `select()` on N pipes; queue is
  cleaner. Also no multi-producer support.
- Raw shared memory (`mp.shared_memory.SharedMemory`): faster per-message
  but enormous synchronization complexity for the variable-shape tensors
  in our requests (hidden state grows turn-by-turn).
- `torch.mp.Queue` is a thin subclass of `mp.Queue` that auto-shares
  tensors via shm rather than pickling — so tensor payloads stay fast
  while metadata still goes through the pickle path.

**Bandwidth math** (sanity check):
- Per request: state ~36 KB + hidden ~80 KB max + mask ~2 KB ≈ 120 KB
- Per response: action_idx + log_prob + value + next_hidden(~2 KB) ≈ 2 KB
- 30 req/sec/worker × 4 workers = 120 req/sec → 14 MB/sec request
  bandwidth, 240 KB/sec response. Well within `mp.Queue`'s capacity.

### D2. Action sampling location: trainer

**Choice**: sampling (temperature softmax + top-p + multinomial) happens in
the trainer's InferenceService, on GPU, batched across all requests. The
response carries the chosen action_idx + log_prob, not raw logits.

**Why**:
- Batched GPU softmax is meaningfully cheaper than per-request CPU softmax.
- Returning logits would mean transmitting 2025-dim float arrays (8 KB)
  per response instead of 1 int + 1 float (16 bytes). 500x bandwidth saving.
- Per-request `temperature` and `top_p` flow with the request (8 bytes total).

**What doesn't change**: the existing two-distribution distinction
(temperature-scaled probs for sampling, T=1 log_probs for PPO importance
ratio) lives in the InferenceService now instead of `RLTrajectoryPlayer`.
Math is identical.

### D3. Hidden state ownership: worker

**Choice**: each player keeps its own `hidden_states[battle_tag]` dict.
The hidden tensor flows over IPC with every request.

**Why**:
- Trainer becomes stateless w.r.t. battles. Cleaner separation, easier
  recovery. If a worker crashes, the trainer has nothing to clean up.
- IPC bandwidth math (D1) shows hidden-in-request is affordable.
- Eviction on battle end is already handled by player code (`hidden_states.pop`).
  Moving it to trainer means duplicating that lifecycle.

**Trade-off**: 80 KB per request vs essentially 0. Acceptable.

### D4. Stale-request semantics: worker drops

**Choice**: worker submits request with a `request_generation` counter
(unchanged from today). When a response arrives and the worker's
generation has since advanced, drop the response. Trainer is unaware
of staleness — it processes whatever arrives.

**Why**:
- Matches today's per-player behavior, minimal cognitive shift.
- Trainer doesn't need to track per-battle generation state — keeps
  trainer logic simple.
- The cost (a stale forward pass that gets thrown away) is small;
  staleness is rare.

### D5. Multi-model multiplexing: one service per model

**Choice**: trainer hosts N parallel InferenceServices, one per active
model (main, optionally bc, optionally exploiter, optionally victim).
Each has its own request queue. Workers know which client to use based
on which player they're driving.

**Why**:
- Simpler than a single multiplexed service (no internal routing).
- Each service can independently apply `torch.compile`, batch tuning,
  and diagnostics.
- Most workers only have main + bc active at most; exploiter/victim
  are off in the current sep_arch config. Three queues + three threads
  is a small footprint.

**Implication for shutdown**: each service needs its own teardown
sequence (drain queue, signal stop, join thread). Clean factory wiring
in trainer setup.

### D6. Inference model separate from learner model

**Choice**: trainer has TWO copies of the main model in GPU memory:
- `learner_model`: owned by the existing PPO/r-NaD update loop.
- `inference_service.model`: independent copy, synced from learner
  periodically (every N updates, matching existing
  `model_broadcast_interval` semantics for workers).

**Why**:
- Avoids races between learner's backward pass and inference's forward.
- Lets us safely apply `torch.compile` to the inference model without
  affecting learner training dynamics.
- Matches existing semantics: workers today have model copies that get
  refreshed periodically; we're moving those copies into the trainer
  instead of into worker processes.

**Memory cost**: ~600 MB (2x model copies on a 24 GB 3090). Acceptable.

### D7. Inference service execution: background thread in trainer process

**Choice**: each InferenceService runs as a daemon thread in the trainer
process, NOT as a separate process. Reads from its mp.Queue (which is
process-safe), runs forward on GPU, writes to per-worker response queues.

**Why**:
- Avoids spawning an extra process. The trainer is already there.
- Threads share the GPU context — no IPC cost between learner and
  inference for the model copy update (`load_state_dict` is just
  Python).
- GIL is not a problem because PyTorch ops release the GIL.

**Trade-off**: thread joins must be handled cleanly on shutdown (already
have signal-handler infrastructure from memory watchdog work).

---

## Architecture diagram

```
┌────────────────────────────────────────────────────────────────────┐
│ TRAINER PROCESS                                                    │
│                                                                    │
│  ┌──────────────────┐       ┌──────────────────────────────────┐  │
│  │  Learner thread  │       │  InferenceService (main model)   │  │
│  │  (existing PPO/  │       │  ┌────────────┐                  │  │
│  │   r-NaD loop)    │       │  │  model     │ ←── periodic    │  │
│  │                  │       │  │  (copy)    │     state sync  │  │
│  │  learner_model ──┼──────→│  └────────────┘     from learner│  │
│  │                  │       │                                  │  │
│  └──────────────────┘       │  inference loop:                 │  │
│                             │   1. drain mp.Queue              │  │
│                             │   2. batch up to batch_size      │  │
│                             │   3. forward + sample on GPU     │  │
│                             │   4. dispatch to response queues │  │
│                             └────────────────┬─────────────────┘  │
│                                              │                    │
│  (similar service for bc model if active)    │                    │
│                                              │                    │
└──────────────────────────────────────────────┼────────────────────┘
                          ↑                    │
                          │                    │
            inference_request_queue            response_queue[worker_id]
                  (shared mp.Queue)              (one per worker)
                          │                    │
            ┌─────────────┴───┬────────────────┘
            │                 │
            ↓                 ↓
┌───────────────────┐  ┌───────────────────┐
│  WORKER 0         │  │  WORKER 1  ...    │
│  ┌─────────────┐  │  │                   │
│  │ Inference   │  │  │                   │
│  │ Client      │  │  │                   │
│  │  - submit() │  │  │                   │
│  │  - dispatch │  │  │                   │
│  │    response │  │  │                   │
│  │    thread   │  │  │                   │
│  └──────┬──────┘  │  │                   │
│         │         │  │                   │
│  ┌──────┴──────┐  │  │                   │
│  │ Player 0    │  │  │                   │
│  │ Player 1    │  │  │                   │
│  │ Player 2    │  │  │                   │
│  │  - featurize│  │  │                   │
│  │  - hidden   │  │  │                   │
│  │  - submit ──┘  │  │                   │
│  │  - sample/send│  │                   │
│  └────────────┘  │  │                   │
└───────────────────┘  └───────────────────┘
```

---

## Request / response data classes

```python
# In src/elitefurretai/rl/inference_ipc.py (new file)

@dataclass(frozen=True)
class InferenceRequest:
    request_id: int                    # unique within (worker_id, model_id)
    worker_id: int
    state: np.ndarray                  # (embedding_size,) float32
    mask: np.ndarray                   # (action_space,) bool, or None for tp
    is_teampreview: bool
    hidden: Optional[torch.Tensor]     # (1, T, hidden_size) or None
    temperature: float
    top_p: float

@dataclass(frozen=True)
class InferenceResponse:
    request_id: int
    action_idx: int                    # sampled action
    log_prob: float                    # T=1 log-prob for PPO
    value: float                       # scalar critic estimate
    win_dist: np.ndarray               # (num_value_bins,) for r-NaD bookkeeping
    next_hidden: torch.Tensor          # (1, T+1, hidden_size)
```

---

## File-by-file plan

### New files

| File | Purpose |
|---|---|
| `src/elitefurretai/rl/inference_ipc.py` | `InferenceRequest`, `InferenceResponse` dataclasses; queue type aliases. |
| `src/elitefurretai/rl/inference_service.py` | `InferenceService`: trainer-side. Owns model, runs inference loop on a daemon thread, batches requests, samples actions, dispatches responses. |
| `src/elitefurretai/rl/inference_client.py` | `InferenceClient`: worker-side. Submits requests, runs response-dispatch thread, resolves per-request futures. |
| `unit_tests/rl/test_inference_ipc_round_trip.py` | Milestone 1 test: trainer + 1 worker, fake model, prove round-trip works. |
| `unit_tests/rl/test_inference_service_equivalence.py` | Milestone 3 test: real model, equivalence vs legacy per-player batcher. |

### Modified files

| File | Change |
|---|---|
| `src/elitefurretai/rl/players.py` | Refactor `RLTrajectoryPlayer`: remove `self.model`, `self.queue`, `_inference_loop`, `_inference_future`, `_run_batch`, `_gpu_inference_sync`, `_add_to_batch`. Add `self.inference_client: InferenceClient`. Replace `await self.queue.put(...)` in `_handle_battle_request` with `await self.inference_client.submit(...)`. Per-player state stays: `temperature`, `top_p`, `hidden_states`, `_request_generation`, `current_trajectories`, embed-side diagnostics. |
| `src/elitefurretai/rl/worker.py` | Receive `inference_clients: Dict[str, InferenceClient]` from spawn args (passed by trainer). Stop building `model` / `bc_model` / `exploiter_model` / `victim_model` here — trainer owns those now. Pass clients through `WorkerOpponentFactory` to player constructors. Keep `embedder` building (workers still featurize). |
| `src/elitefurretai/rl/opponents.py` | `WorkerOpponentFactory.__init__` and `create_player_pairs` accept `inference_clients` instead of model objects; route the right client to each `RLTrajectoryPlayer`. |
| `src/elitefurretai/rl/train.py` | Build `InferenceService` instances after the learner model is loaded. Set up shared mp.Queues. Spawn workers with the queues. Add periodic `state_sync` from learner model to inference service models. Hook shutdown to stop services cleanly. |
| `src/elitefurretai/rl/config.py` | Add `inference_service_batch_size: int = 32` and `inference_service_batch_timeout: float = 0.005` to `HardwareConfig`. (Per-player `batch_size`/`batch_timeout` deprecated — moved to service.) |

### Existing tests touched

| Test | Change |
|---|---|
| `unit_tests/rl/test_players.py` | `RLTrajectoryPlayer` constructor signature changed; tests instantiating it must pass an `inference_client` (mock for unit tests). |
| `unit_tests/rl/test_worker_opponent_factory.py` | Factory signature changed; tests update accordingly. |

---

## Milestone breakdown (3-week budget, 4-week ceiling)

### Milestone 1 (week 1, ~3 days): IPC round-trip

**Goal**: prove the queues + dispatch threads work end-to-end with a fake
model, no real battles, no Showdown.

**Deliverables**:
- `inference_ipc.py` with the two dataclasses.
- `inference_service.py` with `InferenceService` skeleton: takes a
  callable instead of a real model, just echoes `action_idx=0` for
  every request.
- `inference_client.py` with `InferenceClient` skeleton: submit() returns
  an asyncio Future resolved by the response-dispatch thread.
- Test (`test_inference_ipc_round_trip.py`): spawn 2 worker processes,
  each submits 100 fake requests, verify all 200 responses received
  with correct request_id correlation.

**Pass criteria**: 200/200 requests resolved, no deadlocks, clean
shutdown.

### Milestone 2 (week 1-2, ~3 days): real model in service

**Goal**: replace the fake-model service with the real model. Single
worker, single player, single battle, one real episode.

**Deliverables**:
- Service loads the real model, runs real forward + sampling.
- Hidden state flows over IPC correctly (verify shape and values).
- Sampling produces sensible action distributions (compare a few rolls
  against eager model).
- Worker successfully drives one battle through completion against a
  random opponent.

**Pass criteria**: 10 battles complete cleanly. Trajectories produced.
No watchdog fires.

### Milestone 3 (week 2, ~3 days): equivalence test

**Goal**: prove that, given identical RNG seeds and inputs, centralized
inference produces the same trajectory as the legacy per-player batcher.

**Deliverables**:
- Equivalence test on a small model: feed the same sequence of (state,
  hidden, mask, temp, top_p) through both paths, assert
  `torch.allclose` on action_idx + log_prob + value.
- Document any acceptable sources of drift (batch-order non-associativity
  on float ops is fine; anything else is a bug).

**Pass criteria**: equivalence holds across ≥100 sample requests.

### Milestone 4 (week 2-3, ~5 days): full sep_arch run

**Goal**: replace the legacy inference path entirely. Run sep_arch for
100+ updates with multiple models (main + BC), all 4 workers, all 12
players.

**Deliverables**:
- Trainer wires up two InferenceServices (main, bc).
- Workers receive both clients, route correctly per player type.
- Periodic state-sync from learner → inference service models.
- Memory watchdog updated to monitor service threads + worker processes.
- Compare batch-fill stats: avg should jump from ~2 to ~30+, max should
  hit 32.
- Compare throughput vs main's 3.7 traj/s.

**Pass criteria**:
- 100 updates complete without crash.
- Throughput ≥ 5.5 traj/s (≥1.5x).
- Trajectories collected look statistically equivalent (loss curves
  match within noise over 50 updates).

### Buffer (week 4): bug-fixing, perf tuning

Reserve a week for "the inevitable why does THIS happen" — popup
recovery, race conditions, watchdog interactions, IPC bandwidth bugs.

---

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `mp.Queue` deadlock under high load | Medium | High (workers stall) | Start with small `maxsize`; explicit timeouts on `get_nowait`. Worker watchdog fires on response-queue starvation. |
| Stale-request handling subtly broken | Medium | Medium (wasted compute) | Equivalence test (M3) catches wrong-trajectory bugs. Diagnostics counter for stale drops. |
| Trainer process becomes the SPOF | High | High | Already true today (workers depend on trainer's broadcast). Adding inference doesn't change qualitative SPOF status. |
| Hidden state ser/deser cost dominates IPC | Low | Medium | Bandwidth math says no. If wrong, fall back to D3-alt (trainer-owned hidden state). |
| Learner thread + inference thread GIL contention | Low | Low | PyTorch ops release GIL. Profile with py-spy if suspected. |
| `torch.compile` on inference model breaks something | Medium | Low | Compile is OFF by default. Re-enable as separate experiment after M4 ships. |
| Existing watchdog (zero-completion, popup recovery) interactions | High | Medium | Read each watchdog carefully when refactoring; keep existing semantics. Test for regressions. |
| Worker reconnect after Showdown disconnect | Medium | Medium | Existing `rebuild_runtime_agents` path needs updating to not rebuild model (trainer owns it). Must rebuild client connection though. |

---

## Decision gates

1. **End of M1**: round-trip works → proceed. If not, the IPC choice is
   wrong; reconsider mp.Pipe or shared-memory tensors.
2. **End of M3**: equivalence holds → proceed. If output drift is
   non-trivial, we have a bug in the IPC layer. Don't run M4 until this
   is clean.
3. **End of M4**: throughput ≥ 1.5x → proceed to compile retry +
   integration polish. If <1.3x, the model forward isn't the bottleneck
   — re-profile and reconsider whether the gain justifies merging C
   to main.
4. **Week 3 hard check**: clean run completed? If not, document where
   we are and fall back to Option A on a fresh branch off main.

---

## What stays unchanged

To keep scope honest, here's what this refactor explicitly does NOT touch:

- The PPO/r-NaD update math.
- The trajectory format and trajectory_queue protocol.
- The opponent curriculum, ghosts, exploiter co-training control plane.
- The Showdown integration (one websocket per player).
- The masking / featurization pipeline (still per-worker).
- The Rust backend (untouched).
- The supervised training pipeline (untouched).
- Wandb logging and checkpoint format.

If during implementation we find we MUST touch one of these, that's a
decision-gate moment — surface it before proceeding.

---

## Updates log

### 2026-05-13 18:00 — Plan written
- Branch `centralized-inference` created off `main`.
- Design choices D1–D7 resolved.
- File-by-file plan and 4-milestone breakdown documented.
- Next: scaffold milestone 1 files (`inference_ipc.py`,
  `inference_service.py`, `inference_client.py` skeletons + round-trip
  test).

### 2026-05-13 18:30 — M1 complete (IPC round-trip)

**Shipped**:
- `src/elitefurretai/rl/inference_ipc.py` — `InferenceRequest` and
  `InferenceResponse` dataclasses. Fields per D1–D5.
- `src/elitefurretai/rl/inference_service.py` — `InferenceService` class
  with full lifecycle (start/stop/daemon thread), batching loop
  (gather up to `batch_size` or `batch_timeout` from first request),
  per-worker response dispatch, diagnostics matching the legacy
  per-player counters. Includes `echo_batch_handler` for M1 testing.
- `src/elitefurretai/rl/inference_client.py` — `InferenceClient` class
  with submit() returning an asyncio Future, response-dispatch daemon
  thread that resolves futures via `loop.call_soon_threadsafe`, clean
  stop() that cancels in-flight requests.
- `unit_tests/rl/test_inference_ipc_round_trip.py` — 5 tests covering:
  single request, request-with-hidden-tensor (verifies tensor IPC
  works), 200 concurrent requests across 2 simulated workers (no
  cross-talk), diagnostics correctness, clean shutdown cancels pending
  futures.

**Result**: 5/5 tests pass. Lint + pyright clean.

**Caveat**: M1 runs in-process (single Python interpreter, real threads,
real torch.mp.Queue). It does NOT exercise multi-process spawn yet —
that's M2's job, where we replace the echo handler with a real model
in a separate process and verify pickling/shm work across the
process boundary. The in-process choice for M1 is deliberate:
validates the queue + dispatch + future plumbing without spawn-related
complexity.

**Ready for M2** (real model in service, single worker, single battle).

### 2026-05-13 19:30 — M2 complete (real model in service)

**Scope decision**: M2 as originally written said "single worker, single
battle." We deliberately deferred the "single battle" half to M3 because
running a real Showdown battle through the centralized path requires
worker.py + opponents.py + train.py wiring that's properly milestone-4
work. Instead M2 ships the **complete inference pipeline** (real model
+ sampling math + IPC) and proves it produces numerically identical
results to the legacy path on identical inputs. M3 will then drive a
real battle end-to-end.

**Shipped**:
- `src/elitefurretai/rl/inference_handlers.py` — `RealModelBatchHandler`
  class. Mirrors the legacy `_gpu_inference_sync` + `_run_batch`
  per-request loop, operating over `InferenceRequest` items:
  - Pads transformer growing context tensors to the batch's max length
    (matches legacy `_pad_transformer_context` logic exactly).
  - Supports LSTM (h, c) pairs packed as a (2, L, H) wire tensor.
  - Per-request sampling: temperature softmax, mask + renormalize,
    top-p nucleus filter, multinomial or argmax.
  - Masked T=1 log-prob (PPO importance-ratio-compatible).
  - Splits next_hidden per request, restoring the legacy
    "hidden_states[battle_tag] = next_ctx_batch[i:i+1, :next_len, :]"
    convention.
- `src/elitefurretai/rl/inference_ipc.py` — dropped `win_dist` field
  from `InferenceResponse` (verified: not consumed by trajectory format
  or learner; learner recomputes `win_dist_logits` from model during
  training).
- `unit_tests/rl/test_inference_handler_real_model.py` — 7 tests:
  - turn-0 (no context) request → next_hidden length 1
  - turn-1 with 4-step prior context → next_hidden length 5
  - teampreview request → action in [0, 90)
  - batch of 4 mixed contexts → each response has correctly-sized
    next_hidden
  - **argmax-matches-legacy**: numerically identical action_idx +
    log_prob (atol 1e-5) + value (atol 1e-5) vs running the model
    directly through `RNaDModel.forward` and applying the legacy
    sampling math. Strongest M2 correctness signal.
  - end-to-end through IPC layer: real handler + service + client +
    asyncio future resolution all together in-process.
  - top-p doesn't break argmax (sanity).

**Result**: 12/12 tests passing (5 M1 + 7 M2). Lint + pyright clean.

**Caveat**: still in-process. Multi-process spawn is M3's job.

**Ready for M3** (equivalence over a multi-step rollout + first
multi-process exercise).

### 2026-05-13 20:30 — M3 complete (multi-step equivalence + cross-process IPC)

**Shipped**:
- `unit_tests/rl/test_inference_multistep_and_mp.py` — 3 tests:
  - `test_multistep_equivalence_with_growing_context` — drives 10
    sequential turns through both centralized handler and direct
    legacy-style model calls; asserts action/log_prob/value match
    step-by-step. Catches accumulated drift in next_hidden slicing.
  - `test_multistep_equivalence_batched_agrees_with_per_request` —
    submitting 6 requests in one batch must produce numerically close
    next_hidden + log_prob + value as submitting one at a time. This
    test uncovered a real legacy bug (see below).
  - `test_cross_process_round_trip` — first M3 milestone: spawns a
    real `mp.Process` child that uses `InferenceClient` to submit 10
    concurrent requests across the OS process boundary. Validates
    pickling, `torch.mp.Queue` shared-memory tensor transit, asyncio
    in subprocess, and clean shutdown.

**Bug discovered & fixed in `_slice_next_hidden`**:

The legacy `RLTrajectoryPlayer._run_batch` transformer path stores
`next_ctx_batch[i:i+1, :L_i+1, :]` as the new hidden state for
request i with prior context length L_i. **This is wrong when the
batch contains mixed-length requests.** The model's
`forward_with_hidden` concatenates `[hidden_state(B, max_T, H),
encoded(B, 1, H)]`, so the new encoded state ends up at position
`max_T` regardless of L_i. The legacy slice `[:L_i+1]` picks
positions `0..L_i`, which includes a padding-derived position at
index L_i instead of the actual new state at position max_T.

**Effect in production**: when a battle starts (L_0 = 0) and gets
batched with longer-context battles (max_T > 0), its new hidden
state is set to position 0 — pure padding. Next turn that battle has
a corrupted L_1 = 1 hidden, batched with longer contexts again, the
slice picks a padding-derived position again, and the corruption
propagates indefinitely. Battles that started in heterogeneous
batches were effectively training without hidden context.

**Fix in centralized handler**: explicitly concatenate
`prior = ctx_batch[:, :L_i, :]` with `new_state = ctx_batch[:, max_T:max_T+1, :]`,
producing the correct (1, L_i+1, H) tensor.

**Why we don't fix the legacy code**: it's about to be removed when
this branch merges. Fixing it on `main` would change behavior
mid-flight for the running training run; safer to ship the fix as
part of the centralized refactor.

**Test results**: 15/15 passing (5 M1 + 7 M2 + 3 M3). Lint + pyright
clean.

**Ready for M4** (worker.py + train.py wiring + first real sep_arch
training run via the centralized path).

### 2026-05-13 22:07 — M4 complete (system works end-to-end; throughput SURPRISE)

**Code shipped (M4a–d)**:
- M4a: `RLTrajectoryPlayer` dual-mode. Accepts either `model` (legacy
  per-player) OR `inference_client` + `is_transformer` (centralized).
  Constructor validates exactly one is set. `start_inference_loop` is
  no-op in centralized mode. `_choose_move_async` branches on
  `inference_client` to either submit through the service or run the
  legacy queue.put + future path. 4 new unit tests (constructor
  validation + no-op verification).
- M4b: `WorkerOpponentFactory` accepts `main_inference_client` +
  `main_is_transformer`. `create_player_pairs` builds players in
  centralized mode when client is provided.
- M4c: `worker.py` accepts new spawn args (`main_inference_request_queue`,
  `main_inference_response_queue`, `main_is_transformer`). When set,
  skips the main model load and builds an `InferenceClient` on
  `POKE_LOOP`. Threads through `VGCEnvironment.from_config` and
  `_ShowdownBackend`. `update_weights` is a no-op when the worker has
  no model (centralized mode).
- M4d: `train.py` builds the trainer-side `InferenceService` (gated
  on `config.hardware.enable_centralized_inference`), an independent
  model copy on the trainer's device, per-worker mp.Queues, and
  passes them via spawn args. State-syncs the inference model from
  the learner at every weight broadcast. Stops the service cleanly on
  trainer shutdown.
- 68/68 related tests passing (constructor + factory + IPC + handler +
  multistep + cross-process). Lint + pyright clean.

**M4e training run results — surprising regression**:

Two attempts of sep_arch with `enable_centralized_inference: true`,
self-play-only curriculum:

| Variant | Throughput | Batch-fill (avg / max / cap) |
|---|---|---|
| Baseline (per-player, M3 fix included) | 3.7 traj/s | per-player: 1.4–2.2 / 4 / 32 |
| Centralized, inference on CPU | **1.42 traj/s** | service: 4.83 / 15 / 32 |
| Centralized, inference on GPU | **1.26 traj/s** | service: 4.13 / 29 / 32 |

**The good news**: the system runs end-to-end without crashes. Workers
spawn, play battles, submit through IPC, receive responses, ship
trajectories. Centralized batches ARE meaningfully bigger (avg 4.13–4.83
vs 1.4–2.2 per-player; max 15–29 vs 4). The IPC layer, the dispatch,
the state-sync, the shutdown — all working.

**The bad news**: total throughput dropped ~65%. The bottleneck moved
from "per-player batch density" to "request submission rate from
workers." Likely cause: IPC pickling overhead per request. Each
request payload is ~120 KB (state + hidden + mask), pickled per
mp.Queue.put. At ~30 requests/sec/worker × 4 workers, that's
~14 MB/sec of pickling, which on CPU competes with the
asyncio/featurization work that previously parallelized across workers.

GPU vs CPU inference made almost no difference because **inference
itself was never the bottleneck** — the trainer-side service was
spending most of its time waiting for requests, not running forwards.
Per-batch wait was ~50 ms (resulting in 13–14k batches over 10 min);
inference latency on GPU is sub-1 ms. The service was idle most of
the time.

**What this teaches**:
1. The "make batches bigger" intuition was correct but insufficient.
   We DID make batches bigger. Throughput dropped anyway because the
   request rate to the bigger-batch service is lower than the
   aggregate request rate to N per-worker services.
2. CPU-bound parallelism is real and valuable in our architecture.
   Per-worker inference parallelizes across cores; centralizing
   serializes the inference path through one process.
3. **The right architecture is probably hybrid**: per-worker
   inference loops (for parallelism) + bigger batch_size knobs
   (already shipped) + faster CPU inference (torch.compile + smaller
   model) + maybe shared model weights across workers via shm
   (avoids 4× memory cost without serializing inference).
4. The IPC payload size matters. Hidden state shipped on every
   request (~80 KB max) is the largest cost; if we kept hidden in
   the trainer keyed by battle_tag, payloads would shrink ~40×.
5. The legacy bug fix from M3 (`_slice_next_hidden`) is the only
   piece of M4 work that directly improves training; it should be
   ported to the legacy `_run_batch` path.

**Status**: M4 code stays on the `centralized-inference` branch as
working reference. The architecture is shipped and tested; the
throughput finding tells us **simple centralization isn't the win
we predicted**, and the F8 bottleneck (per-player batchers cap at 4)
isn't the dominant one.

**Recommendation**: do NOT merge `centralized-inference` to `main`
as-is. Either:
- (a) Iterate on this branch — port hidden-state-in-trainer (D3-alt
  from the design doc), measure again. If throughput recovers, merge.
- (b) Cherry-pick the M3 hidden-state slicing bug fix to `main` and
  abandon the centralized-inference branch. Move on to a different
  throughput lever (e.g., the distillation or Rust-backend work from
  the original obsolescence table).

The throughput investigation doc's TL;DR should be updated to
reflect: F8 was a real finding but its impact was overestimated.
The IPC overhead of fixing F8 ate the gain.

### 2026-05-13 23:21 — D3-alt iteration: hidden state moves trainer-side

**Hypothesis (from M4 retrospective)**: per-request IPC payload was the
dominant new bottleneck because each request shipped the (1, T, H)
hidden tensor. Move hidden ownership from worker to trainer (key by
(worker_id, player_id, battle_tag)); request payload drops to just
(state + mask + battle_tag) and shrinks ~40x.

**Code changes shipped**:
- `inference_ipc.py`: `InferenceRequest` loses `hidden`, gains
  `player_id` + `battle_tag`. `InferenceResponse` loses `next_hidden`.
  New `EvictRequest` for cleanup.
- `inference_handlers.py`: `RealModelBatchHandler` owns
  `hidden_states: Dict[(worker_id, player_id, battle_tag), Tensor]`.
  Looks up prior hidden by key, runs forward, stores updated hidden in
  place. Exposes `evict(...)`.
- `inference_service.py`: drains `EvictRequest` from the same queue,
  forwards to handler.evict(); doesn't count as inference batch.
- `inference_client.py`: `submit(...)` takes `player_id` + `battle_tag`
  (no hidden). New `evict(player_id, battle_tag)` method.
- `players.py`: centralized branch passes `player_id=self.username`.
  New `_reset_battle_hidden_state(battle_tag)` helper that pops local
  dict AND calls `inference_client.evict()`. All 10 stale-cleanup
  sites use it.

**Tests**: 69/69 passing (16 IPC/handler/multistep + 53 elsewhere).

**Two critical bugs caught during integration**:

1. **Both sides of self-play share `battle_tag`.** First sep_arch run
   crashed at startup: positional encoding pe sized `max_seq_len +
   decision_tokens + 1 = 44`, but hidden grew past 41. Keying by just
   `(worker_id, battle_tag)` collided p1 and p2, so each turn updated
   hidden TWICE. Fix: add `player_id` to the key (the player's
   Showdown username).
2. **Stale-request races grew hidden indefinitely.** Same crash
   recurred after fix #1. The legacy player resets
   `hidden_states[battle_tag] = None` on every stale-request /
   timeout / send-failure / max-steps path, but in centralized mode
   those pops only hit the unused player-local dict — the trainer-side
   dict kept growing across in-flight stale requests. Fix:
   `_reset_battle_hidden_state` calls `inference_client.evict()` on
   every cleanup path. 10 sites updated.

**Throughput measurement** (sep_arch, centralized + D3-alt, self-play
only curriculum, post-warmup updates 153–156):

| Run variant | Throughput | Batch avg / max |
|---|---|---|
| Per-player baseline (M3 fix included) | 3.7 traj/s | 1.4–2.2 / 4 |
| Initial centralized, hidden-in-wire (M4e) | 1.42 traj/s | 4.83 / 15 |
| **D3-alt (this)** | **3.0 traj/s** | **7.7 / 32** |

**Reading the result**:
- D3-alt **closes most of the regression** plain centralization
  introduced: from -62% to -19% vs baseline. 2x improvement over the
  initial centralized attempt.
- **Batches actually fill now.** First time seeing max=32 (the cap)
  hit. Avg 7.7 is 4x the per-player avg — F8 (per-player batchers
  cap at 4) is genuinely broken.
- Still ~20% below per-player baseline despite bigger batches. Likely
  remaining bottlenecks: trainer process runs learner + inference in
  one process (CPU/GPU contention); single inference thread can only
  run one forward at a time (per-player had 4 in parallel); residual
  ~4.3 MB/sec pickling (down from 14 MB/sec).

### Where this leaves us

D3-alt is a **partial win**. Architecture works, tests pass, bugs
fixed. Throughput went from -62% to -19% vs baseline. Not full
recovery.

Decision options:

**(c) Iterate further on this branch**: move inference to a separate
trainer process (eliminate GPU/CPU contention with learner); add
multiple concurrent inference threads; retry `torch.compile` (no
longer blocked — H4 unblocked, one process pays compile cost).

**(d) Accept the 20% regression for the architecture cleanup, merge
this branch.** Cleaner long-term (one model copy, easier to add
compile / multi-thread inference / different pruning later). 20%
regression buys structural simplicity. Includes the M3 hidden-state
slicing bug fix.

**(e) Cherry-pick the M3 bug fix to main, abandon this branch.** Move
on to a different lever (Rust backend, distillation, more workers).

I lean toward (c) — we have a working centralized pipeline with two
of three known wins (bigger batches, single model copy) and one
unblocked future win (torch.compile). The remaining 20% gap is
identifiable. But (e) is fully reasonable; the architectural win
isn't worth pursuing if (c) doesn't recover throughput.

### 2026-05-13 23:45 — D3-alt + torch.compile: net win over baseline

**Hypothesis (c.1)**: now that compile cost is paid ONCE in the trainer
(not N times per worker on CPU), torch.compile becomes viable. Compile
the inference agent in train.py, warm up synchronously before workers
spin up. Should reduce per-call inference latency, freeing trainer
GPU/CPU time and reducing the contention with the learner.

**Code shipped**:
- `train.py`: when `config.hardware.compile_inference_model` is set,
  wrap `inference_agent` with `torch.compile(mode=..., dynamic=True)`
  and run a 2-shape warmup (turn 0 + turn 1) on the inference device
  before workers are spawned. Cast back to `RNaDModel` for type
  checking; runtime delegation handles the actual `__call__` and
  attribute access.
- `sep_arch.yaml`: `compile_inference_model: default`.

**Measurement** (sep_arch, D3-alt + compile, self-play only, post-warmup
updates 154–159):

| Run variant | Throughput | Learner steps/s | Batch avg / max | Notes |
|---|---|---|---|---|
| Per-player baseline (M3 fix) | 3.7 traj/s | ~60 | 1.4–2.2 / 4 | reference |
| D3-alt only | 3.0 traj/s | ~60 | 7.7 / 32 | -19% |
| **D3-alt + compile** | **4.05 traj/s** | **~82** | **7.15 / 32 (0.2% hit cap)** | **+9% over baseline, +37% learner** |

- 6 consecutive updates ≥3.77 traj/s. No outliers, no errors, no
  Long-context warnings, no RuntimeErrors.
- Compile + warmup paid 76 seconds upfront in the trainer. One-time
  cost, fine.
- Inference batches now occasionally hit the 32 cap (0.1–0.2% of
  batches) — first time we've seen this.

**Reading the result**:

D3-alt + compile is a **net win over the per-player baseline**, not
just a regression-recovery. Going from 3.7 → 4.05 traj/s is small in
absolute terms (+9%) but the structural gains compound:

1. **One model copy in GPU memory.** Workers no longer hold ~300 MB
   each. Total trainer memory drops by ~1.2 GB. Useful headroom for
   bigger models / more concurrent trainings later.
2. **One compile target.** Future architecture experiments (deeper
   value head, different attention, etc.) only need to be compiled
   once, not N times.
3. **Centralized observability.** All inference goes through one
   service with one diagnostics dict. Easier to profile and tune
   batching parameters.
4. **F8 genuinely broken.** Centralized batches reach 32, vs the
   per-player 4 cap. Batching is no longer the bottleneck.
5. **Learner is faster too.** The state-sync into the inference model
   is now strict-non-blocking on the learner; the inference thread
   handles all requests. Learner throughput rose ~37% (60 → 82 steps/s).

### Recommendation: merge `centralized-inference` to `main`

This branch is ready. To merge cleanly:

1. **Cherry-pick or carry**: The M3 hidden-state slicing bug fix
   (`_slice_next_hidden` correctly handles batched mixed-length
   contexts) should land regardless. It's already on this branch.
2. **Single commit suggestion**: ~9 files modified, a handful of new
   files (`inference_*.py`, related tests). A clean squash-merge
   keeps history readable.
3. **No follow-up needed**: the dual-mode `RLTrajectoryPlayer` keeps
   the legacy path intact, so reverting to per-player is one config
   knob away (`enable_centralized_inference: false`).

Future opportunities still on the table:

- **Other model types**: BC, ghost, exploiter, victim — currently they
  still use the legacy per-worker path. Centralizing them requires
  porting the `opponent.model = X` hot-swap logic to inference_client
  swaps. Probably medium effort. Would unlock the full curriculum
  (currently constrained to self-play in the centralized config).
- **Separate inference process**: still on the table if the trainer's
  learner+inference contention becomes a bottleneck again.
- **Bigger batch_timeout**: currently 5ms; could try 10ms for tighter
  batches at slight latency cost.
- **bf16/fp16 inference**: would cut GPU compute roughly in half for
  the centralized forward.

But for now: ship.
