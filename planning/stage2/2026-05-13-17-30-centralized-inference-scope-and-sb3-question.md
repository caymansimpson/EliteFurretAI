# Centralized Inference Refactor — Scope, and the SB3 Question

**Date**: 2026-05-13
**Context**: After Tier 1 cheap throughput wins exhausted (vec `_dual_expand`,
`batch_size=32`, batch-fill logging), we're at 3.69 traj/s with all easy
levers pulled. `torch.compile` and `num_servers: 1` both turned out blocked
on architectural change. F8 (per-player batchers cap at ~4 due to
one-websocket-per-player serialization) is now the headline bottleneck.
See [throughput investigation doc](./2026-05-13-13-00-throughput-bottleneck-investigation.md).

---

## Part 1 — The SB3 question (decide first)

Before scoping any refactor: should we just adopt Stable Baselines 3 and
inherit VGCBench's whole architecture?

### What SB3 would give us

- `SubprocVecEnv` — centralized inference across all envs in one batched
  forward pass (the win we want).
- Battle-tested PPO, GAE, advantage normalization, value clipping.
- Mature callbacks, eval, checkpointing, wandb integration.
- Same architecture as VGCBench → directly comparable performance ceiling.
- Less code to maintain.

### What SB3 would cost us

| EliteFurretAI feature | SB3 fit |
|---|---|
| **r-NaD update rule** (regularized Nash dynamics, KL to anchor policy) | **No native support.** Would require monkey-patching PPO or forking. r-NaD is core to the project (per `project_approach_and_vision` memory). |
| **Distributional value head** (51-bin categorical) | Possible but awkward; SB3's value head is scalar. |
| **Autoregressive turn head with decision tokens** | Requires a custom `ActorCriticPolicy` subclass. Doable but invasive. |
| **Transformer hidden state across turns** | SB3 has RNN support but not the "growing context tensor" pattern; would need custom hooks. |
| **Curriculum + opponent rotation + ghosts + exploiters** | All bespoke — would need custom callbacks. SB3's eval framework doesn't model this. |
| **Trajectory collection format** (custom for r-NaD) | SB3's rollout buffer assumes PPO; would need to extend. |

### Verdict

**Don't adopt SB3 wholesale.** Every customization SB3 saves us is one we'd
have to re-add. We'd spend the next month fighting the framework instead
of doing science. The asymmetry is brutal: SB3 saves us ~2 weeks of
infrastructure work but costs us ~6 weeks of refactoring to fit r-NaD,
the autoregressive head, and the curriculum machinery into its
abstractions.

**But borrow the architectural pattern.** `SubprocVecEnv`-style centralized
inference is the right design. We can build a focused version of it
ourselves in ~1 week — no framework lock-in, no fighting abstractions.

This is also what the throughput investigation doc's TL;DR
recommendation already pointed at: "centralized inference (one shared
batcher across players within a worker)."

### When to revisit

If we ever decide r-NaD isn't paying off and we want to switch to
plain PPO, SB3 becomes a much better fit. Not now.

---

## Part 2 — Centralized inference architecture options

Three meaningful designs, in order of effort/reward.

### Option A — Per-worker centralization (recommended)

**Architecture**: each worker process has ONE shared inference batcher
serving ALL players in that worker. Players push requests to a
worker-shared queue; one inference loop per worker pulls and batches.

```
Per worker (×4):
  3 players → 1 shared queue → 1 inference loop → 1 model copy
  ~96 concurrent battles → real batches of 12–30
```

**Throughput leverage**:
- Combines 3 players' request streams per worker → batches grow from
  ~2 (current avg per F8) to ~6–18 per call.
- `batch_size=32` becomes meaningfully fillable.
- `torch.compile` becomes viable (one process, one compile) → unblocks H4.

**Risk**: low. Same model layout, same Showdown integration, same
trajectory format. The change is local to the inference path.

**Effort**: ~1 week.

### Option B — Cross-worker model sharing

**Architecture**: workers share ONE model copy in GPU memory via shared
tensors / inter-process model handles.

**Throughput leverage**: small. Just saves GPU memory (4 model copies →
1). Doesn't enlarge batches because each worker still batches
independently.

**Risk**: medium-high. PyTorch model-across-processes via shared memory
is finicky; race conditions on parameter reads during sync.

**Verdict**: not worth pursuing alone. Only valuable as a stepping
stone to Option C.

### Option C — Trainer-process inference (full VGCBench pattern)

**Architecture**: workers featurize and step environments only. ALL
inference happens in the trainer process. Workers send observation
arrays via shared memory; trainer runs ONE forward across all 384
battles' pending observations and sends actions back.

```
Trainer process:
  receives obs from 4 workers → ONE batched forward (batch≈30–100) → sends actions
Workers (×4):
  featurize → IPC send obs → IPC recv action → step Showdown
```

**Throughput leverage**: largest. Effective batch sizes of 30-100,
single model copy on GPU, single compile.

**Risk**: high. Major rewrite of worker.py, players.py, the IPC layer.
Have to handle: stale-request detection across IPC, worker reconnects,
graceful shutdowns, deadlock prevention, failure recovery.

**Effort**: 2-3 weeks.

### Recommendation

**Ship Option A first. Re-measure. Then decide on C.**

Rationale:
1. **A captures most of the gain at a fraction of the risk.** If A gets
   us to 5–7 traj/s, we're at the lower bound of VGCBench's 15–20.
   Combined with `torch.compile` (now unblocked), we might hit 7–10
   traj/s. That may be enough for the science we want to do.
2. **A is a stepping stone to C.** The work of unifying per-worker
   inference is exactly the same work needed to later move it to a
   trainer-side service. Nothing wasted.
3. **A keeps the failure surface small.** If something breaks, we
   isolate to a single worker; the trainer is unaffected.

If A's measured gain is below 1.5x, that's signal that the bottleneck
isn't just batching — and Option C wouldn't help proportionally either.
Reconsider the architecture from scratch in that case.

---

## Part 3 — Detailed scope for Option A

### Files touched

| File | Change |
|---|---|
| `src/elitefurretai/rl/players.py` | Refactor `RLTrajectoryPlayer`: remove per-instance `_inference_loop`, `_inference_future`, `queue`; replace with reference to a shared `WorkerInferenceService`. Player still owns `_handle_battle_request`, embed, masking, etc. |
| `src/elitefurretai/rl/players.py` (new class) | Add `WorkerInferenceService`: owns the model, the shared queue, the inference loop. Serves N players. Same `_run_batch` / `_gpu_inference_sync` internals as today. |
| `src/elitefurretai/rl/opponents.py` | `WorkerOpponentFactory.create_player_pairs` constructs the service once per worker, passes it to all players in the worker. |
| `src/elitefurretai/rl/worker.py` | No structural change; just confirm the factory wiring. |
| `src/elitefurretai/rl/config.py` | No new knobs needed (existing `batch_size`, `batch_timeout` apply to the shared batcher). |

### Class boundaries

```python
class WorkerInferenceService:
    """One per worker. Owns the model and a shared inference loop.
    Serves N RLTrajectoryPlayers."""
    def __init__(self, model, device, batch_size, batch_timeout, ...): ...
    async def submit(self, state, mask, hidden, battle_tag, player_id) -> Future: ...
    async def _inference_loop(self): ...  # gathers from shared queue
    def _run_batch(self, ...): ...
    def _gpu_inference_sync(self, ...): ...
    def get_diagnostics_snapshot(self) -> Dict[str, float]: ...

class RLTrajectoryPlayer(Player):
    def __init__(self, *, inference_service: WorkerInferenceService, ...):
        # No model attribute, no own queue, no own inference loop.
        self.inference_service = inference_service
        # Per-player state stays: temperature, top_p, hidden_states[battle_tag],
        # _request_generation, etc.
    
    async def _handle_battle_request(self, ...):
        # Same as today: embed, mask, look up hidden state.
        # Submit to shared service instead of own queue:
        future = await self.inference_service.submit(state, mask, hidden, ...)
        result = await asyncio.wait_for(future, timeout=...)
        # Same downstream: sample, send to Showdown, record trajectory.
```

### Data flow (single inference cycle)

1. Showdown sends battle state to player.
2. Player embeds, masks, looks up hidden state.
3. Player calls `service.submit(...)`, gets a Future.
4. Service's inference loop pulls request off shared queue.
5. Loop gathers up to `batch_size` requests within `batch_timeout`.
6. Loop runs ONE forward pass.
7. Loop resolves each Future with that request's slice of the output.
8. Players' awaiting coroutines wake up, sample actions, send to Showdown.

### Shared queue semantics

- One `asyncio.Queue` per `WorkerInferenceService`, created on the POKE_LOOP
  (consistent with today's pattern in `RLTrajectoryPlayer`).
- Items: `(state_array, mask, hidden, battle_tag, player_id, future)`.
- `player_id` flows through so the loop can route per-player diagnostics
  and (if needed) per-player temperature.

### Per-player state — what stays where

| State | Today | After A |
|---|---|---|
| Model weights | Per player (shared via Python ref) | Per service (one copy) |
| Inference queue | Per player | Per service |
| Inference loop task | Per player | Per service |
| `temperature`, `top_p` | Per player | Per player (passed through with request) |
| `hidden_states[battle_tag]` | Per player | Per player |
| `_request_generation` | Per player | Per player |
| `current_trajectories` | Per player | Per player |
| `_diagnostics` | Per player | Service holds inference-side counters; player keeps embed/request counters |

### Diagnostics changes

`get_diagnostics_snapshot` aggregates across players (`opponents.py:1580`).
Service's diagnostics need a separate aggregation entry, OR each player
forwards a slice of the service's counters proportional to its request
share. Cleanest: service exposes `get_diagnostics_snapshot()`, factory
aggregates separately and includes in the combined report.

The new batch-fill log line moves from `RLTrajectoryPlayer._inference_loop`
to `WorkerInferenceService._inference_loop` — same format, but now
shows the genuinely-shared batch sizes.

### Test plan

1. **Unit test** (new): `WorkerInferenceService` with N=2 players,
   submit synthetic requests concurrently, verify each player's Future
   resolves with the correct row of the batched output. Mocked model.
2. **Equivalence test** (new): on a small model, run M battles with
   N=1 player using the service vs N=1 player with the legacy
   per-player batcher. Outputs should be deterministic-equal given
   same RNG.
3. **Integration test** (manual): launch a 100-update sep_arch run
   with the new architecture. Compare batch-fill stats:
   - Pre-refactor (today): avg 1.4–2.2, max 4 per player
   - Post-refactor (target): avg 6–18, max approaching 32, filled%>0
4. **Regression checks**: zero-completion watchdog still fires
   correctly on stalls, popup recovery still works, graceful shutdown
   still saves checkpoint.

### Failure modes to guard

| Failure | Mitigation |
|---|---|
| Service stalls → all players in worker stall together | Existing zero-completion watchdog catches this; tier of granularity drops from per-player to per-worker (acceptable). |
| Service queue grows unbounded under load spike | Queue is unbounded today per-player (no `maxsize`); same for service queue. Add `maxsize` only if memory becomes an issue. |
| Player teardown leaves dangling Futures in service queue | Service must drain pending Futures with `set_exception(CancelledError)` on shutdown. Mirrors existing `stop_inference_loop` logic. |
| Shape mismatch when concatenating across players | All players in a worker share the same model and embedding shape — no mismatch by construction. Add an assertion in `_run_batch` regardless. |

### Effort estimate

| Phase | Effort |
|---|---|
| Implementation (refactor + new service class + factory wiring) | 2–3 days |
| Unit + equivalence tests | 1 day |
| Integration + benchmark (multiple sep_arch runs) | 1–2 days |
| Buffer for surprises | 1–2 days |
| **Total** | **5–8 days (~1 week)** |

### Decision gates (when to abort or pivot)

1. **After unit tests pass**: if equivalence test reveals subtle output
   drift, pause and decide whether the source of drift is acceptable
   (e.g., batch-order non-associativity in float ops is acceptable;
   anything else needs root-cause).
2. **After first integration run**: if batch-fill avg stays below 6,
   per-worker centralization isn't enough — F8's bottleneck is per-worker
   not per-player. Skip ahead to Option C planning.
3. **After throughput measurement**: if gain is <1.3x, the model
   forward isn't where the time is going. Re-profile and reconsider.

---

## Part 4 — What comes after A

In priority order, conditional on A's measured outcome:

1. **`torch.compile` retry** (was H4): now viable because one process
   does the compile. Expected +1.3–1.5x on top of A.
2. **Re-evaluate Option C**: if A + compile gets us to 7–10 traj/s,
   maybe we don't need C. If we're stuck below 6 traj/s, C becomes
   the next big lever.
3. **Distillation (D)**: smaller rollout-only model. Most attractive
   if model forward stays dominant after A + compile.

---

## Open questions for Cayman

1. **Stable Baselines 3** — agree with the verdict to NOT adopt
   wholesale, but borrow the centralized-inference pattern? (My
   recommendation is yes.)
2. **Refactor scope** — go straight to A, or build a feature flag so
   old + new paths coexist temporarily? (My recommendation: no flag.
   The change is local enough; we keep the old code in git history.)
3. **Timing** — start the A refactor now (this week), or pause to
   continue training for a few days first? Current run is at 3.69
   traj/s climbing toward step 200; pausing for a refactor costs ~1
   week of training time but unblocks 1.5–2x for everything after.
