# Ghost Centralization + Compile-Race + Legacy Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the registry plan's "Future work": centralize ghost inference through `ModelRegistry`, investigate the torch.compile multi-thread race (4-hour time-box), delete the legacy per-worker inference path, and measure throughput at each gate.

**Architecture:** Pre-register `max_ghosts` `ghost_<slot>` services in `ModelRegistry` at startup. Trainer maintains `active_ghost_slots: Set[int]` and broadcasts it to workers in the existing weight-broadcast payload. Workers route `GHOSTS` opponent samples by picking a random active slot and assigning `opponent.inference_client = clients.get(f"ghost_{slot}")`. After ghost lands and (conditionally) compile-race is solved, rip out the dual-mode `BatchInferencePlayer`, the legacy worker model-build path, and the unused `OpponentPool.sample_opponent` / `_create_*_opponent` methods.

**Tech Stack:** Python 3.11, PyTorch, torch.multiprocessing, pytest, ruff, pyright. Existing classes: `ModelRegistry` ([model_registry.py](../../src/elitefurretai/rl/model_registry.py)), `WorkerInferenceClients` ([worker_inference_clients.py](../../src/elitefurretai/rl/worker_inference_clients.py)), `OpponentPool` + `WorkerOpponentFactory` ([opponents.py](../../src/elitefurretai/rl/opponents.py)).

**Design doc:** [planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md](./2026-05-14-13-00-ghost-centralization-and-cleanup-design.md)

---

## File map

| File | Phase | Change |
|---|---|---|
| `src/elitefurretai/rl/train.py` | 1 | Add ghost slot pre-registration + LRU rotation + broadcast extension. Phase 3: drop legacy build branch + `enable_centralized_inference` references. |
| `src/elitefurretai/rl/opponents.py` | 1 | Add `set_active_ghost_slots` + `GHOSTS` branch in `configure_opponent_for_batch` that routes via `ghost_<slot>`. Phase 3: delete legacy ghost caching + `OpponentPool.sample_opponent` family. |
| `src/elitefurretai/rl/worker.py` | 1 | Receive initial `active_ghost_slots` in spawn args; handle `active_ghost_slots` in broadcast handler. Phase 3: drop legacy `if not centralized` branch + `update_weights`. |
| `src/elitefurretai/rl/players.py` | 3 | Drop dual-mode `BatchInferencePlayer` (only `inference_client` mode survives). |
| `src/elitefurretai/rl/config.py` | 3 | Remove `enable_centralized_inference`. |
| `src/elitefurretai/rl/configs/sep_arch.yaml` | 3 | Remove `enable_centralized_inference: true`; replace multi-line ghost comment. |
| `src/elitefurretai/rl/RL.md` | 3-4 | Update section 8b architecture description; remove "legacy ghost path retained" language. |
| `unit_tests/rl/test_model_registry.py` | 1 | New tests for ghost slot lifecycle. |
| `unit_tests/rl/test_opponent_pool.py` | 1, 3 | Tests for `set_active_ghost_slots` (1); delete legacy-mode tests (3). |
| `unit_tests/rl/test_worker_opponent_factory.py` | 1, 3 | Test ghost slot routing (1); delete legacy-mode tests (3). |
| `unit_tests/rl/test_players.py` | 3 | Delete legacy-mode tests. |
| `planning/stage2/2026-05-14-13-00-...design.md` | 4 | Final comparison table + "done" status. |
| `planning/stage2/2026-05-14-00-15-model-registry-plan.md` | 4 | Mark Future work closed. |

---

## Phase 0 — Baseline measurement

### Task 0.1: Run sep_arch baseline on current `main`

**Files:** none (read-only measurement)

- [ ] **Step 1: Confirm clean working tree**

Run:
```bash
git status --short
```
Expected: only the unstaged `M src/elitefurretai/rl/configs/sep_arch.yaml` from before (if it's still there). If anything else is dirty, stash or commit before starting.

- [ ] **Step 2: Launch sep_arch in the background**

Run:
```bash
source ../venv/bin/activate && \
nohup python -u src/elitefurretai/rl/train.py \
  --config src/elitefurretai/rl/configs/sep_arch.yaml \
  > /tmp/baseline_measurement.log 2>&1 &
```
Note the python PID (not the bash wrapper) via `pgrep -f "train.py.*sep_arch"`.

- [ ] **Step 3: Wait for warmup + measurement window**

Tail the log:
```bash
tail -f /tmp/baseline_measurement.log | grep -E "Update [0-9]+ "
```
Wait for ~50 update lines to appear (~15 minutes). The first 30 are warmup; updates 31–50 are the measurement window.

- [ ] **Step 4: Stop the run**

Run:
```bash
pkill -TERM -f "train.py.*sep_arch"
```
Verify with `pgrep -f train.py` → empty.

- [ ] **Step 5: Extract metrics**

Run:
```bash
grep -E "Update (3[1-9]|[45][0-9]) " /tmp/baseline_measurement.log | head -40
```
Compute mean traj/s and learner steps/s over updates 31–50. Record in a scratch note (for later comparison table).

- [ ] **Step 6: Confirm parity with prior measurement**

The registry plan reported 4.98 traj/s on this same config. If this baseline is within ±0.3 traj/s, proceed. If outside that band, investigate before going further — we need a trustworthy reference.

- [ ] **Step 7: Record Measurement 1**

Append to the design doc's Updates section:
```markdown
### 2026-05-14 [time] — Measurement 1 (baseline on main @ 1944eda)

| Metric | Value |
|---|---|
| traj/s mean | X.XX |
| traj/s stddev | X.XX |
| learner steps/s | XX |
| batch fill avg / max | X.X / XX |
| errors per 1k batches | 0 |
```

No git commit — this is a doc edit that will be commited together with later updates.

---

## Phase 1 — Ghost centralization

### Task 1.1: Test that `ModelRegistry` supports ghost slot registration

**Files:**
- Test: `unit_tests/rl/test_model_registry.py`

- [ ] **Step 1: Add failing test**

Add to `unit_tests/rl/test_model_registry.py`:
```python
def test_register_multiple_ghost_slots(make_registry, make_agent):
    """Registry accepts N ghost_<i> registrations with distinct services."""
    registry = make_registry(num_workers=2)
    for slot in range(3):
        registry.register(f"ghost_{slot}", make_agent(), compile=False)
    assert set(registry.names()) == {"ghost_0", "ghost_1", "ghost_2"}
    # Each slot has independent diagnostics
    diag = registry.get_diagnostics()
    assert set(diag.keys()) == {"ghost_0", "ghost_1", "ghost_2"}
    registry.stop_all()
```

If `make_registry` / `make_agent` fixtures don't already exist, mirror the pattern used by existing tests in this file (read it first).

- [ ] **Step 2: Run test, verify it passes**

Run:
```bash
source ../venv/bin/activate && pytest unit_tests/rl/test_model_registry.py::test_register_multiple_ghost_slots -v
```
Expected: PASS (the registry's existing `register()` already supports arbitrary names — this test pins that contract).

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_model_registry.py
git commit -m "test(registry): pin multi-ghost-slot registration contract"
```

### Task 1.2: Test that `WorkerOpponentFactory.set_active_ghost_slots` updates internal state

**Files:**
- Test: `unit_tests/rl/test_worker_opponent_factory.py`
- Modify: `src/elitefurretai/rl/opponents.py:1148-1170` (the `set_ghost_paths` / `set_exploiter_paths` methods area, where the new method will sit)

- [ ] **Step 1: Add failing test**

Add to `unit_tests/rl/test_worker_opponent_factory.py`:
```python
def test_set_active_ghost_slots_updates_state(make_worker_factory):
    factory = make_worker_factory()
    assert factory._active_ghost_slots == set()
    factory.set_active_ghost_slots([0, 2, 4])
    assert factory._active_ghost_slots == {0, 2, 4}
    factory.set_active_ghost_slots([])
    assert factory._active_ghost_slots == set()
```

If `make_worker_factory` doesn't exist, factor a helper that constructs a minimal `WorkerOpponentFactory` (look at existing tests in the file for the pattern — they likely already instantiate it).

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py::test_set_active_ghost_slots_updates_state -v
```
Expected: FAIL with AttributeError on `_active_ghost_slots` or `set_active_ghost_slots`.

- [ ] **Step 3: Implement**

In `src/elitefurretai/rl/opponents.py`, in `WorkerOpponentFactory.__init__` (around line 1032 where `self.ghosts = []` is set), add:
```python
        self._active_ghost_slots: Set[int] = set()
```

Then add a new method near `set_ghost_paths` (~line 1154):
```python
    def set_active_ghost_slots(self, slots: List[int]) -> None:
        """Update the set of populated ghost slots from a trainer broadcast.

        Workers use this to know which `ghost_<slot>` clients in the
        inference bundle correspond to real ghost weights vs placeholders.
        Only slots in this set are valid targets for GHOSTS opponent
        routing in `configure_opponent_for_batch`.
        """
        self._active_ghost_slots = set(slots)
```

Ensure `Set` and `List` are imported from `typing` at the top of the file (they likely already are; verify).

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py::test_set_active_ghost_slots_updates_state -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "feat(opponents): add set_active_ghost_slots to WorkerOpponentFactory"
```

### Task 1.3: Test + implement ghost slot routing in `configure_opponent_for_batch`

**Files:**
- Test: `unit_tests/rl/test_worker_opponent_factory.py`
- Modify: `src/elitefurretai/rl/opponents.py:1495-1504` (the GHOSTS branch in `configure_opponent_for_batch`)

- [ ] **Step 1: Add failing test**

Add to `unit_tests/rl/test_worker_opponent_factory.py`:
```python
def test_configure_opponent_for_batch_ghosts_routes_via_active_slot(
    make_worker_factory, mock_worker_inference_clients
):
    """When GHOSTS is sampled and slots are active, opponent gets a
    ghost_<slot> InferenceClient from the bundle."""
    clients = mock_worker_inference_clients(
        names=["main", "ghost_0", "ghost_1", "ghost_2"]
    )
    factory = make_worker_factory(
        worker_inference_clients=clients,
        curriculum={"ghosts": 1.0},  # force GHOSTS selection
    )
    factory.set_active_ghost_slots([0, 1, 2])

    player, opponent = make_player_pair()
    selected = factory.configure_opponent_for_batch(player, opponent)

    assert selected == factory.GHOSTS
    # opponent.inference_client should now point to one of the ghost clients
    assert opponent.inference_client in (
        clients.get("ghost_0"),
        clients.get("ghost_1"),
        clients.get("ghost_2"),
    )
```

If `mock_worker_inference_clients` / `make_player_pair` helpers don't exist, write minimal stand-ins. The mock just needs `.has(name)` / `.get(name)` returning sentinel objects.

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py::test_configure_opponent_for_batch_ghosts_routes_via_active_slot -v
```
Expected: FAIL (current code passes `None` as `centralized_name` for ghosts, taking the legacy `ghost_agent` path).

- [ ] **Step 3: Implement**

Replace the GHOSTS branch in `configure_opponent_for_batch` (currently `src/elitefurretai/rl/opponents.py:1495-1504`):
```python
        elif selected_type == self.GHOSTS:
            if not self._active_ghost_slots:
                # Curriculum sampled GHOSTS but no slots are populated —
                # treat as a desync bug and fall back to self-play loudly
                # via the standard path. (The curriculum's
                # _opponent_available guard should prevent this.)
                selected_type = self.SELF_PLAY
            else:
                slot = random.choice(tuple(self._active_ghost_slots))
                opponent_swapped = self._swap_to(
                    opponent, f"ghost_{slot}", None
                )
                if not opponent_swapped:
                    selected_type = self.SELF_PLAY
```

The `_swap_to(opponent, f"ghost_{slot}", None)` call uses `_resolve_centralized_client` to look up the ghost client; if the bundle is missing it (config error), `_swap_to` returns False and we fall back.

- [ ] **Step 4: Run the new test + the full factory test file**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -v
```
Expected: all PASS, including the new test.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "feat(opponents): route GHOSTS via centralized ghost_<slot> clients"
```

### Task 1.4: Add slot lifecycle state to `OpponentPool` (trainer-side)

**Files:**
- Test: `unit_tests/rl/test_opponent_pool.py`
- Modify: `src/elitefurretai/rl/opponents.py:209-280` (OpponentPool.__init__) and near `add_ghost` (line 655)

- [ ] **Step 1: Add failing test**

Add to `unit_tests/rl/test_opponent_pool.py`:
```python
def test_opponent_pool_tracks_active_ghost_slots(tmp_path, make_opponent_pool):
    """OpponentPool exposes active_ghost_slots reflecting which slots
    hold real ghost weights, and rotates LRU when full."""
    pool = make_opponent_pool(ghosts_dir=str(tmp_path), max_ghosts=3)
    # Empty at startup with empty dir
    assert pool.active_ghost_slots() == set()
    assert pool.slot_for_ghost_path == {}

    # First three add_ghost calls fill slots 0, 1, 2
    for step, name in [(10, "g10.pt"), (20, "g20.pt"), (30, "g30.pt")]:
        p = tmp_path / name
        p.write_bytes(b"x")  # add_ghost only stores the path
        pool.add_ghost(step, str(p))
    assert pool.active_ghost_slots() == {0, 1, 2}
    assert set(pool.slot_for_ghost_path.values()) == {0, 1, 2}

    # Fourth add evicts LRU (slot 0 — oldest by insertion)
    p4 = tmp_path / "g40.pt"
    p4.write_bytes(b"x")
    pool.add_ghost(40, str(p4))
    assert pool.active_ghost_slots() == {0, 1, 2}  # still full
    # slot 0 now holds the newest ghost
    assert pool.slot_for_ghost_path[str(p4)] == 0
```

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
pytest unit_tests/rl/test_opponent_pool.py::test_opponent_pool_tracks_active_ghost_slots -v
```
Expected: FAIL with AttributeError on `active_ghost_slots` / `slot_for_ghost_path`.

- [ ] **Step 3: Implement slot tracking in `OpponentPool`**

In `src/elitefurretai/rl/opponents.py`, in `OpponentPool.__init__` (after `self.ghosts: List[Tuple[int, str]] = []` line ~268), add:
```python
        # Ghost slot lifecycle: each path is assigned a slot 0..max_ghosts-1.
        # `slot_for_ghost_path` maps file path -> slot index. `_slot_lru` is
        # an insertion-ordered list of currently-occupied slots; head is
        # oldest. Loaded ghosts from disk on startup get slots 0..K-1.
        self.slot_for_ghost_path: Dict[str, int] = {}
        self._slot_lru: List[int] = []
```

Update `_load_ghosts` (line 364) to assign slots:
```python
    def _load_ghosts(self) -> None:
        files = self._list_model_checkpoints(self.ghosts_dir)
        models: List[Tuple[int, str]] = []
        for file in files:
            step = int(os.path.basename(file).split("_")[-1].split(".")[0])
            models.append((step, file))
        models.sort(key=lambda x: x[0], reverse=True)
        self.ghosts = models[: self.max_ghosts]
        # Assign slots in oldest-first order (slot 0 = oldest = first to
        # be evicted by LRU). `self.ghosts` is sorted newest-first, so
        # iterate in reverse for slot assignment.
        self.slot_for_ghost_path = {}
        self._slot_lru = []
        for slot, (_, path) in enumerate(reversed(self.ghosts)):
            self.slot_for_ghost_path[path] = slot
            self._slot_lru.append(slot)
```

Update `add_ghost` (line 655) to assign/rotate slot:
```python
    def add_ghost(self, step: int, filepath: str):
        # Determine slot: reuse if path already known, else allocate
        # next free or evict LRU.
        if filepath in self.slot_for_ghost_path:
            slot = self.slot_for_ghost_path[filepath]
        elif len(self._slot_lru) < self.max_ghosts:
            slot = len(self._slot_lru)
            self._slot_lru.append(slot)
        else:
            # Evict the LRU slot. Find which path currently holds it,
            # drop the mapping, and reuse the slot for the new path.
            slot = self._slot_lru.pop(0)
            evicted_path = next(
                p for p, s in self.slot_for_ghost_path.items() if s == slot
            )
            del self.slot_for_ghost_path[evicted_path]
            self._slot_lru.append(slot)
        self.slot_for_ghost_path[filepath] = slot

        self.ghosts.append((step, filepath))
        self.ghosts.sort(key=lambda x: x[0], reverse=True)
        self.ghosts = self.ghosts[: self.max_ghosts]

    def active_ghost_slots(self) -> Set[int]:
        """Slots currently populated with real ghost weights."""
        return set(self._slot_lru)
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
pytest unit_tests/rl/test_opponent_pool.py::test_opponent_pool_tracks_active_ghost_slots -v
```
Expected: PASS.

- [ ] **Step 5: Run full opponent_pool tests**

Run:
```bash
pytest unit_tests/rl/test_opponent_pool.py -v
```
Expected: all PASS. If any unrelated test broke, that's a regression in `add_ghost` / `_load_ghosts` semantics — fix before committing.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_opponent_pool.py
git commit -m "feat(opponents): OpponentPool tracks active ghost slots + LRU eviction"
```

### Task 1.5: Pre-register ghost slot services in `train.py` registry setup

**Files:**
- Modify: `src/elitefurretai/rl/train.py` (registry setup area around lines 1090–1200)

This is configuration plumbing — no unit test for the setup wiring itself (it's exercised by the integration test in Task 1.10 and the sep_arch run in Task 1.11).

- [ ] **Step 1: Locate registry setup**

Read `src/elitefurretai/rl/train.py:1090-1200`. Find the section where `bc`, `exploiter`, `victim` get conditionally registered. The new ghost block goes after those.

- [ ] **Step 2: Add ghost pre-registration block**

After the existing victim registration (look for `registry.register("victim", ...)` or similar), add:
```python
    # Ghost slots: pre-register max_ghosts services so the slot pool is
    # fixed-size and the registration plumbing never happens mid-run.
    # Slots start with main-agent weights as placeholders; only slots
    # listed in `opponent_pool.active_ghost_slots()` are valid routing
    # targets (workers filter on that set). compile=False per the known
    # torch.compile multi-thread race (see registry plan).
    for slot in range(config.curriculum.max_ghosts):
        # Clone main agent so each ghost slot owns independent params.
        # sync_weights below overwrites with real ghost weights for any
        # slot that has one loaded from disk.
        ghost_agent = clone_agent_for_ghost_slot(inference_agent)
        registry.register(f"ghost_{slot}", ghost_agent, compile=False)
    # Load weights for any pre-existing ghost checkpoints onto their
    # assigned slots. `slot_for_ghost_path` was populated by
    # OpponentPool._load_ghosts.
    for path, slot in opponent_pool.slot_for_ghost_path.items():
        state_dict = torch.load(path, map_location=registry.device)
        registry.sync_weights(f"ghost_{slot}", state_dict)
```

- [ ] **Step 3: Add `clone_agent_for_ghost_slot` helper**

Above the function where the registry setup lives in `train.py`, add (or place near other helpers in the file):
```python
def clone_agent_for_ghost_slot(template_agent: RNaDAgent) -> RNaDAgent:
    """Build a fresh RNaDAgent with the same architecture as `template_agent`
    for use as a ghost slot placeholder. Weights are copied at construction
    time; subsequent registry.sync_weights() replaces them when a real ghost
    checkpoint lands."""
    new_agent = type(template_agent)(
        # Mirror the constructor pattern used by build_main_agent in this
        # file — pass the same config-derived kwargs the template was built
        # with. Read template_agent's `_init_kwargs` if available, otherwise
        # inspect the existing build_main_agent call and copy its arguments.
        **template_agent.init_kwargs  # placeholder if attribute doesn't exist
    )
    new_agent.model.load_state_dict(template_agent.model.state_dict())
    return new_agent
```

> **Implementation note**: `RNaDAgent` may not have `init_kwargs` exposed. If not, the cleanest approach is to refactor `build_main_agent` (or whatever currently builds the main agent in `train.py`) into a `_build_agent_from_config(config) -> RNaDAgent` helper and call it once per ghost slot. The helper already exists conceptually — find it before adding `clone_agent_for_ghost_slot`. If it doesn't exist, lift it from the inline construction site in train.py first.

- [ ] **Step 4: Lint + type check**

Run:
```bash
ruff check src/elitefurretai/rl/train.py && \
pyright src/elitefurretai/rl/train.py
```
Expected: clean.

- [ ] **Step 5: Run full unit suite to catch obvious breakage**

Run:
```bash
pytest unit_tests -q
```
Expected: all PASS. (Registry setup is integration-level; unit suite shouldn't fail. If it does, the import broke somewhere — check the helper signature.)

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/train.py
git commit -m "feat(train): pre-register max_ghosts ghost slot services on registry init"
```

### Task 1.6: Sync ghost weights on `add_ghost` event in trainer loop

**Files:**
- Modify: `src/elitefurretai/rl/train.py` around line 1613 (where `opponent_pool.add_ghost(updates, ghost_checkpoint_path)` is called)

- [ ] **Step 1: Locate the ghost-save site**

Read `src/elitefurretai/rl/train.py:1600-1670`. The current call is:
```python
                    opponent_pool.add_ghost(updates, ghost_checkpoint_path)
```

- [ ] **Step 2: Add registry sync immediately after**

Replace that line with:
```python
                    opponent_pool.add_ghost(updates, ghost_checkpoint_path)
                    # Sync new weights into the registry slot. OpponentPool
                    # already assigned the slot in add_ghost; look it up.
                    new_slot = opponent_pool.slot_for_ghost_path[
                        ghost_checkpoint_path
                    ]
                    state_dict = torch.load(
                        ghost_checkpoint_path,
                        map_location=registry.device,
                    )
                    registry.sync_weights(f"ghost_{new_slot}", state_dict)
```

- [ ] **Step 3: Lint + type check**

Run:
```bash
ruff check src/elitefurretai/rl/train.py && \
pyright src/elitefurretai/rl/train.py
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/rl/train.py
git commit -m "feat(train): sync ghost weights into registry slot on save"
```

### Task 1.7: Broadcast `active_ghost_slots` to workers

**Files:**
- Modify: `src/elitefurretai/rl/train.py` around line 1669 (broadcast payload composition)
- Modify: `src/elitefurretai/rl/worker.py` (broadcast handler)

- [ ] **Step 1: Extend trainer broadcast payload**

In `src/elitefurretai/rl/train.py` find the broadcast composition block around line 1665-1670 (currently sends `"exploiter_paths"` and `"ghost_paths"`). Add a new key:
```python
                        "active_ghost_slots": sorted(
                            opponent_pool.active_ghost_slots()
                        ),
```
Keep `"ghost_paths"` for now (Phase 3 removes it once the worker no longer reads paths).

- [ ] **Step 2: Add failing test for worker broadcast handler**

Add to `unit_tests/rl/test_worker_opponent_factory.py`:
```python
def test_worker_handles_active_ghost_slots_broadcast(make_worker_factory):
    """When the broadcast handler receives active_ghost_slots, it
    propagates to the factory via set_active_ghost_slots."""
    factory = make_worker_factory()
    # Simulate the worker's broadcast handler dispatching to the factory.
    # The actual dispatch lives in worker.py — this test pins the
    # factory side of the contract.
    factory.set_active_ghost_slots([1, 3, 5])
    assert factory._active_ghost_slots == {1, 3, 5}
```

This duplicates Task 1.2's coverage on purpose for the broadcast pathway documentation. If it feels redundant, drop in favor of the worker.py-side test below.

- [ ] **Step 3: Wire up worker's broadcast handler**

In `src/elitefurretai/rl/worker.py`, find where `exploiter_paths` is consumed from incoming broadcast messages (grep for `exploiter_paths`). Adjacent to that, add:
```python
                if "active_ghost_slots" in msg:
                    opponent_factory.set_active_ghost_slots(
                        msg["active_ghost_slots"]
                    )
```

If the worker also calls `set_ghost_paths(msg["ghost_paths"])`, keep that line for now — Phase 3 deletes it.

- [ ] **Step 4: Lint + type check**

Run:
```bash
ruff check src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py && \
pyright src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py
```
Expected: clean.

- [ ] **Step 5: Run unit suite**

Run:
```bash
pytest unit_tests -q
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/train.py src/elitefurretai/rl/worker.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "feat(broadcast): plumb active_ghost_slots from trainer to workers"
```

### Task 1.8: Pass initial `active_ghost_slots` via spawn args

**Files:**
- Modify: `src/elitefurretai/rl/train.py` (worker spawn site)
- Modify: `src/elitefurretai/rl/worker.py` (spawn arg parsing + initial factory setup)

- [ ] **Step 1: Locate worker spawn**

Read the worker spawn site in `train.py` (grep for `mp.Process` or `torch_mp.Process` and `args=`). Note all current spawn args.

- [ ] **Step 2: Add `initial_active_ghost_slots` to spawn args**

In `train.py`, where the spawn-arg tuple is built, add:
```python
                initial_active_ghost_slots=sorted(
                    opponent_pool.active_ghost_slots()
                ),
```

If spawn args are positional (a tuple), append carefully — order matters. Prefer adding as a keyword arg if the spawn target accepts `**kwargs`.

- [ ] **Step 3: Consume on worker side**

In `worker.py`, accept the new arg in the worker entry point signature and pass it to the factory right after `WorkerOpponentFactory` is constructed:
```python
opponent_factory.set_active_ghost_slots(initial_active_ghost_slots)
```

- [ ] **Step 4: Lint + type check**

Run:
```bash
ruff check src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py && \
pyright src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py
```
Expected: clean.

- [ ] **Step 5: Run unit suite**

Run:
```bash
pytest unit_tests -q
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/train.py src/elitefurretai/rl/worker.py
git commit -m "feat(worker): receive initial active_ghost_slots in spawn args"
```

### Task 1.9: End-to-end smoke test of ghost centralization

**Files:** no edits (integration check)

- [ ] **Step 1: Run sep_arch for 5 updates as a smoke**

Edit `src/elitefurretai/rl/configs/sep_arch.yaml` temporarily to set `max_updates: 5` (we'll revert this in a moment). Then:
```bash
source ../venv/bin/activate && \
python -u src/elitefurretai/rl/train.py \
  --config src/elitefurretai/rl/configs/sep_arch.yaml \
  2>&1 | tee /tmp/ghost_smoke.log
```
Expected:
- 5 updates complete
- Log shows `Registry: registered 'ghost_0'` ... `ghost_4`
- No `KeyError: ghost_*` from workers
- No `IndexError: Cannot choose from an empty sequence` (would mean the active slots fallback fired)

- [ ] **Step 2: Revert the max_updates change**

Run:
```bash
git checkout src/elitefurretai/rl/configs/sep_arch.yaml
```
Confirm with `git diff src/elitefurretai/rl/configs/sep_arch.yaml` → empty.

- [ ] **Step 3: Inspect the smoke log for ghost traffic**

```bash
grep -E "(ghost_[0-9]|GHOSTS|active_ghost_slots)" /tmp/ghost_smoke.log | head -20
```
You should see: registry-side ghost registrations, broadcast-side `active_ghost_slots` mentions, and at least one routed GHOSTS battle if the curriculum sampled it.

### Task 1.10: Measurement 2 — sep_arch with ghost centralization

**Files:** none (read-only measurement)

- [ ] **Step 1: Launch full sep_arch**

Run:
```bash
nohup python -u src/elitefurretai/rl/train.py \
  --config src/elitefurretai/rl/configs/sep_arch.yaml \
  > /tmp/measurement_2.log 2>&1 &
```

- [ ] **Step 2: Wait for 50 updates**

Monitor `tail -f /tmp/measurement_2.log | grep -E "Update [0-9]+ "`. Wait until update 50 prints.

- [ ] **Step 3: Stop the run**

```bash
pkill -TERM -f "train.py.*sep_arch"
```

- [ ] **Step 4: Extract metrics from window updates 31–50**

Same as Task 0.1 step 5. Compute mean traj/s, learner steps/s, batch fill avg/max.

- [ ] **Step 5: Gate check**

Compare to Measurement 1. Expected: throughput ≥ baseline (ghost centralization should be a net win or neutral — the 10% of curriculum that was on the slow path now joins larger batches). If a regression > 0.3 traj/s: STOP, diagnose, do not proceed to Phase 2.

- [ ] **Step 6: Record Measurement 2**

Append to the design doc's Updates section. Format same as Measurement 1.

- [ ] **Step 7: Commit the doc update**

```bash
git add planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md
git commit -m "docs(plan): record Measurement 2 (ghost centralization)"
```

---

## Phase 2 — torch.compile race investigation (4-hour time-box)

**Time-box starts when Task 2.1 starts.** If you hit 4 hours without a working fix, jump to Task 2.5 (document + skip).

### Task 2.1: Build a minimal reproducer test

**Files:**
- Create: `unit_tests/rl/test_compile_race_reproducer.py`

- [ ] **Step 1: Write the reproducer**

Create `unit_tests/rl/test_compile_race_reproducer.py`:
```python
"""Reproducer for the torch.compile + concurrent-service dynamo race
documented in the model_registry_plan.

The race manifests as RuntimeError("Detected that you are using FX to
symbolically trace a dynamo-optimized function") when multiple compiled
models are invoked concurrently from separate threads.

This test is expected to FAIL on plain torch + concurrent calls; the
goal of Task 2.2-2.4 is to find a wrapper that makes it pass.
"""
import threading
from typing import List

import pytest
import torch
import torch.nn as nn


class TinyAgent(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


@pytest.mark.timeout(60)
def test_two_compiled_models_concurrent_calls_no_race():
    """Two distinct compiled models called from two threads should not
    raise. Fails today; the fix from Task 2.2/2.3/2.4 should make it pass."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    m2 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    # Warm up each on a fixed shape (single-threaded, no race)
    with torch.no_grad():
        m1(torch.zeros(1, 64, device=device))
        m2(torch.zeros(1, 64, device=device))

    errors: List[BaseException] = []

    def loop(m):
        try:
            with torch.no_grad():
                for _ in range(500):
                    m(torch.randn(4, 64, device=device))
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=loop, args=(m1,))
    t2 = threading.Thread(target=loop, args=(m2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors == [], f"compile race triggered: {errors[0]!r}"
```

- [ ] **Step 2: Confirm the reproducer fires**

Run:
```bash
pytest unit_tests/rl/test_compile_race_reproducer.py -v
```
Expected outcomes (any of):
- FAIL with the documented `RuntimeError`. ← target behavior, proceed.
- PASS — race doesn't reproduce in isolation. The race may need richer
  workload (variable shapes, longer threads). If so, vary `dynamic=True`,
  inputs across shapes, or increase iterations to 5000. If still no
  reproduction after 30 minutes of attempts, document this in the design
  doc and skip the rest of Phase 2 — we can't fix what we can't reproduce.

If the reproducer can't be made to fail: **stop the investigation here**, document, move to Phase 3.

- [ ] **Step 3: Commit the reproducer (failing or passing — both useful)**

```bash
git add unit_tests/rl/test_compile_race_reproducer.py
git commit -m "test(compile): add reproducer for torch.compile multi-thread race"
```

### Task 2.2: Attempt fix #1 — per-model `threading.Lock`

**Files:**
- Create or modify: `src/elitefurretai/rl/model_registry.py` (try adding a per-service compile lock)
- Test: `unit_tests/rl/test_compile_race_reproducer.py`

- [ ] **Step 1: Modify the reproducer to use a lock wrapper**

Add a second test function below the existing one:
```python
def test_two_compiled_models_with_per_model_lock():
    """Wrap each compiled call in a per-model lock. Does serializing
    entry into the compiled function bypass the dynamo race?"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    m2 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    lock1, lock2 = threading.Lock(), threading.Lock()

    def call_locked(m, lock, x):
        with lock:
            return m(x)

    with torch.no_grad():
        call_locked(m1, lock1, torch.zeros(1, 64, device=device))
        call_locked(m2, lock2, torch.zeros(1, 64, device=device))

    errors: List[BaseException] = []

    def loop(m, lock):
        try:
            with torch.no_grad():
                for _ in range(500):
                    call_locked(m, lock, torch.randn(4, 64, device=device))
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=loop, args=(m1, lock1))
    t2 = threading.Thread(target=loop, args=(m2, lock2))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors == [], f"per-model lock didn't fix it: {errors[0]!r}"
```

- [ ] **Step 2: Run the lock test**

```bash
pytest unit_tests/rl/test_compile_race_reproducer.py::test_two_compiled_models_with_per_model_lock -v
```
- If PASS: lock fix works. Jump to Task 2.5.
- If FAIL: move to Task 2.3.

- [ ] **Step 3: Commit the additional test regardless of outcome**

```bash
git add unit_tests/rl/test_compile_race_reproducer.py
git commit -m "test(compile): attempt per-model lock workaround"
```

### Task 2.3: Attempt fix #2 — `cudagraph_mark_step_begin`

**Files:**
- Test: `unit_tests/rl/test_compile_race_reproducer.py`

Only do this task if Task 2.2 failed.

- [ ] **Step 1: Add a third test variant**

Add:
```python
def test_two_compiled_models_with_cudagraph_mark_step():
    """Insert torch.compiler.cudagraph_mark_step_begin() between calls."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("cudagraph_mark_step_begin is a GPU-specific hook")
    m1 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    m2 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)

    with torch.no_grad():
        m1(torch.zeros(1, 64, device=device))
        m2(torch.zeros(1, 64, device=device))

    errors: List[BaseException] = []

    def loop(m):
        try:
            with torch.no_grad():
                for _ in range(500):
                    torch.compiler.cudagraph_mark_step_begin()
                    m(torch.randn(4, 64, device=device))
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=loop, args=(m1,))
    t2 = threading.Thread(target=loop, args=(m2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors == [], f"cudagraph_mark_step_begin didn't fix it: {errors[0]!r}"
```

- [ ] **Step 2: Run it**

```bash
pytest unit_tests/rl/test_compile_race_reproducer.py::test_two_compiled_models_with_cudagraph_mark_step -v
```
- If PASS: jump to Task 2.5.
- If FAIL: move to Task 2.4.

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_compile_race_reproducer.py
git commit -m "test(compile): attempt cudagraph_mark_step_begin workaround"
```

### Task 2.4: Attempt fix #3 — dedicated thread per compiled model

**Files:**
- Test: `unit_tests/rl/test_compile_race_reproducer.py`

Only do this task if Task 2.3 failed.

- [ ] **Step 1: Add fourth test variant**

Add:
```python
import queue


def test_two_compiled_models_each_in_own_thread():
    """Each compiled model is only ever called from its own dedicated
    thread (via a queue). This eliminates any concurrency on a single
    compiled model — the race is between models, but each model itself
    sees serial access."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m1 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    m2 = torch.compile(TinyAgent().to(device).eval(), mode="default", dynamic=True)
    with torch.no_grad():
        m1(torch.zeros(1, 64, device=device))
        m2(torch.zeros(1, 64, device=device))

    errors: List[BaseException] = []

    def serve(m, q_in, q_out):
        try:
            while True:
                x = q_in.get()
                if x is None:
                    return
                with torch.no_grad():
                    q_out.put(m(x))
        except BaseException as e:
            errors.append(e)

    q1_in, q1_out = queue.Queue(), queue.Queue()
    q2_in, q2_out = queue.Queue(), queue.Queue()
    s1 = threading.Thread(target=serve, args=(m1, q1_in, q1_out))
    s2 = threading.Thread(target=serve, args=(m2, q2_in, q2_out))
    s1.start(); s2.start()

    def produce(qi, qo):
        for _ in range(500):
            qi.put(torch.randn(4, 64, device=device))
            qo.get()

    p1 = threading.Thread(target=produce, args=(q1_in, q1_out))
    p2 = threading.Thread(target=produce, args=(q2_in, q2_out))
    p1.start(); p2.start(); p1.join(); p2.join()
    q1_in.put(None); q2_in.put(None); s1.join(); s2.join()
    assert errors == [], f"dedicated-thread didn't fix it: {errors[0]!r}"
```

- [ ] **Step 2: Run it**

```bash
pytest unit_tests/rl/test_compile_race_reproducer.py::test_two_compiled_models_each_in_own_thread -v
```
- If PASS: jump to Task 2.5.
- If FAIL: this exhausts the three attempts. Move directly to Task 2.6 (escalate / document).

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_compile_race_reproducer.py
git commit -m "test(compile): attempt dedicated-thread workaround"
```

### Task 2.5: Deploy the working fix to production code

**Files:**
- Modify: `src/elitefurretai/rl/inference_service.py` and/or `src/elitefurretai/rl/model_registry.py` depending on which fix worked

Only do this task if one of Tasks 2.2 / 2.3 / 2.4 passed.

- [ ] **Step 1: Apply the fix to the production inference path**

Choose the matching code change:

**If Task 2.2 (lock) won**: in `InferenceService.run_batch` (the code that calls the handler's compiled agent), add a `self._compile_lock = threading.Lock()` in `__init__`, then wrap the actual forward call site with `with self._compile_lock:`. Each service has its own lock — locks are not shared across services (per-model granularity).

**If Task 2.3 (cudagraph_mark_step) won**: in the same forward call site, prepend `torch.compiler.cudagraph_mark_step_begin()` (guarded by `if torch.cuda.is_available()`).

**If Task 2.4 (dedicated thread) won**: `InferenceService` already runs in a dedicated daemon thread, so this may already be the situation — verify by reading `InferenceService.start()`. If it's true (each service has its own thread, all compiled forwards happen on that thread), then the production code already does what Task 2.4 simulated, and the original race report was from a different code path. Investigate where else compiled forwards run — likely the eval/state-sync path. Apply fix there.

- [ ] **Step 2: Enable `compile=True` for secondary models**

In `train.py` registry setup, change the following `register(..., compile=False)` calls to `compile=True`:
- `bc`
- `exploiter` (if conditionally registered)
- `victim` (if conditionally registered)
- All `ghost_<slot>` registrations

- [ ] **Step 3: Run unit tests**

```bash
pytest unit_tests -q
```
Expected: all PASS, including new compile-race tests.

- [ ] **Step 4: Run sep_arch end-to-end smoke (5 updates)**

Temporarily set `max_updates: 5`. Run. Check log for zero compile-race RuntimeErrors. Revert max_updates.

- [ ] **Step 5: Measurement 3 — full sep_arch run**

Same protocol as Task 1.10. 50 updates, measure window 31–50.

- [ ] **Step 6: Gate check**

Compare to Measurement 2. Expected: throughput ≥ Measurement 2 (compile on more models should be a win or neutral). Critical: zero compile-race errors over the 20-update window. If any errors: revert the compile=True changes for that model, treat compile fix as failed, move to Task 2.6.

- [ ] **Step 7: Record Measurement 3 in the design doc**

- [ ] **Step 8: Commit**

```bash
git add src/elitefurretai/rl/inference_service.py src/elitefurretai/rl/model_registry.py src/elitefurretai/rl/train.py planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md
git commit -m "feat(inference): resolve torch.compile multi-thread race + enable compile on secondaries"
```

### Task 2.6: If no fix worked in 4 hours — document and skip

**Files:**
- Modify: `planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md`

- [ ] **Step 1: Append findings to the design doc's Updates section**

Add a section documenting:
- Which fixes were tried
- For each, what symptom it produced (still failed / passed in test but flaked in sep_arch / other)
- Why upstream investigation seems warranted
- Reproducer test status (kept as `@pytest.mark.skip(reason="known race, see plan")` to preserve the diagnostic)

- [ ] **Step 2: Mark reproducer tests skip**

In `unit_tests/rl/test_compile_race_reproducer.py`, add `@pytest.mark.skip(reason="known dynamo race, no user-space fix found 2026-05-14")` to the failing tests, OR keep them as-is if pytest already accepts the expected failure via `@pytest.mark.xfail`. Pick xfail if `dynamic=True` does reproduce reliably; skip if it's environment-dependent.

- [ ] **Step 3: Commit**

```bash
git add planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md unit_tests/rl/test_compile_race_reproducer.py
git commit -m "docs(compile): document failed attempts at torch.compile race fix"
```

---

## Phase 3 — Legacy inference path cleanup

Phase 3 happens regardless of Phase 2 outcome. Even if compile didn't get fixed, the legacy per-worker path can still be deleted: we just retain `compile=False` for secondaries.

### Task 3.1: Inventory legacy test cases

**Files:** none (read-only audit)

- [ ] **Step 1: Grep test files for legacy-mode patterns**

Run:
```bash
grep -nE "model=|legacy|main_inference_client|enable_centralized_inference" \
  unit_tests/rl/test_players.py \
  unit_tests/rl/test_worker_opponent_factory.py \
  unit_tests/rl/test_opponent_pool.py \
  unit_tests/rl/test_workers.py 2>/dev/null
```

- [ ] **Step 2: List legacy test cases to delete**

For each match, decide:
- Test exercises legacy-only code path → delete
- Test exercises shared code path with legacy arg → modernize to centralized arg
- Test exercises something orthogonal to centralization → keep

Write the list in a scratch buffer. This drives Tasks 3.2 / 3.3 / 3.4.

- [ ] **Step 3: Grep for callers of `OpponentPool.sample_opponent`**

Run:
```bash
grep -rn "sample_opponent\b\|_create_self_play_opponent\|_create_bc_opponent\|_create_exploiter_opponent\|_create_ghost_opponent" \
  src/ unit_tests/ 2>/dev/null
```

If any production caller exists (in `src/` outside `opponents.py` itself): do NOT delete those methods in Task 3.7. Update the design's "Drop it from scope" note and skip the OpponentPool method removal.

If only `test_opponent_pool.py` references them: add those test cases to the deletion list.

### Task 3.2: Delete legacy tests in test_players.py

**Files:**
- Modify: `unit_tests/rl/test_players.py`

- [ ] **Step 1: Delete identified legacy-mode test functions**

Use the list from Task 3.1. For each, delete the entire `def test_...():` block.

- [ ] **Step 2: Run remaining tests**

```bash
pytest unit_tests/rl/test_players.py -v
```
Expected: all remaining PASS.

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_players.py
git commit -m "test(players): drop legacy-mode test cases"
```

### Task 3.3: Delete legacy tests in test_worker_opponent_factory.py

**Files:**
- Modify: `unit_tests/rl/test_worker_opponent_factory.py`

- [ ] **Step 1: Delete identified legacy-mode test functions**

- [ ] **Step 2: Run remaining tests**

```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -v
```

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_worker_opponent_factory.py
git commit -m "test(factory): drop legacy-mode test cases"
```

### Task 3.4: Delete legacy tests in test_opponent_pool.py (if applicable)

**Files:**
- Modify: `unit_tests/rl/test_opponent_pool.py`

Only do this task if Task 3.1's grep showed `sample_opponent` etc. are tested but have no production caller.

- [ ] **Step 1: Delete tests for `sample_opponent` / `_create_*_opponent`**

- [ ] **Step 2: Run remaining tests**

```bash
pytest unit_tests/rl/test_opponent_pool.py -v
```

- [ ] **Step 3: Commit**

```bash
git add unit_tests/rl/test_opponent_pool.py
git commit -m "test(opponents): drop tests for unused sample_opponent family"
```

### Task 3.5: Remove `enable_centralized_inference` config flag

**Files:**
- Modify: `src/elitefurretai/rl/config.py:402` (the field)
- Modify: `src/elitefurretai/rl/train.py:1168` (the `centralized = ...` read site)
- Modify: `src/elitefurretai/rl/configs/sep_arch.yaml:158` (the YAML key)

- [ ] **Step 1: Delete the field from config.py**

In `src/elitefurretai/rl/config.py`, remove:
```python
    enable_centralized_inference: bool = False
```

- [ ] **Step 2: Delete the read site in train.py**

In `train.py:1168` and any other site that branches on `centralized`, hard-code the centralized path:
- Remove `centralized = config.hardware.enable_centralized_inference`
- Inline `centralized` consumers: take the True branch unconditionally, delete the False branch.

- [ ] **Step 3: Remove from sep_arch.yaml**

In `src/elitefurretai/rl/configs/sep_arch.yaml`, delete the `enable_centralized_inference: true` line (line 158). Also delete the multi-line comment above it about it being a centralized-inference flag.

- [ ] **Step 4: Search for stragglers**

```bash
grep -rn "enable_centralized_inference" src/ unit_tests/ planning/ docs/ 2>/dev/null
```
Anything that turns up: delete it (or update doc references to historical context only).

- [ ] **Step 5: Run unit suite**

```bash
pytest unit_tests -q
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/config.py src/elitefurretai/rl/train.py src/elitefurretai/rl/configs/sep_arch.yaml
git commit -m "refactor: remove enable_centralized_inference flag (centralized is the only mode)"
```

### Task 3.6: Strip dual-mode from `BatchInferencePlayer`

**Files:**
- Modify: `src/elitefurretai/rl/players.py`

- [ ] **Step 1: Read current `BatchInferencePlayer` structure**

Read the full class. Note where the dual-mode validation happens (constructor) and where the if/else branches on `inference_client` vs `self.model` occur (`_choose_move_async`, etc.).

- [ ] **Step 2: Delete legacy attributes**

Remove from `__init__`:
- The `model: Optional[RNaDAgent] = None` parameter
- Validation that "exactly one of model/inference_client is set"
- Setting `self.model`
- Setting `self.queue`, `self._inference_loop`, `self._inference_future`, anything else only used by the legacy path

Make `inference_client` a required argument (no default).

- [ ] **Step 3: Delete legacy methods**

Remove:
- `_inference_loop`
- `_run_batch`
- `_gpu_inference_sync`
- `_add_to_batch`
- `start_inference_loop` (currently a no-op stub for centralized mode)

If any of these are imported elsewhere (e.g., `from .players import _run_batch`), update those importers.

- [ ] **Step 4: Collapse `_choose_move_async`**

Remove the `if self.inference_client is not None: ... else: ...` branch. Keep only the centralized path.

- [ ] **Step 5: Run players tests**

```bash
pytest unit_tests/rl/test_players.py -v
```
Expected: all PASS.

- [ ] **Step 6: Run full unit suite**

```bash
pytest unit_tests -q
```
Expected: all PASS.

- [ ] **Step 7: Lint + type check**

```bash
ruff check src/elitefurretai/rl/players.py && \
pyright src/elitefurretai/rl/players.py
```
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/elitefurretai/rl/players.py
git commit -m "refactor(players): drop dual-mode legacy inference path from BatchInferencePlayer"
```

### Task 3.7: Strip legacy branch from `worker.py`

**Files:**
- Modify: `src/elitefurretai/rl/worker.py`

- [ ] **Step 1: Find legacy branches**

Grep `worker.py` for `if not centralized` / `if self.model is not None` / `update_weights` / model-build patterns.

- [ ] **Step 2: Delete the legacy model-build branch**

Remove the entire `if not centralized:` (or equivalent) block that builds `model`/`bc_model`/`exploiter_model`/`victim_model` per worker. Centralized is unconditional now.

- [ ] **Step 3: Delete `update_weights`**

Remove the `update_weights` method on the worker entry class. Update any caller (likely in the broadcast handler) to remove the call. The broadcast no longer needs to push weights to the worker — centralized handles weight sync via `registry.sync_weights` on the trainer side.

- [ ] **Step 4: Run unit tests**

```bash
pytest unit_tests/rl/test_workers.py -v 2>/dev/null
pytest unit_tests/rl/ -v
```
Expected: all PASS.

- [ ] **Step 5: Lint + type check**

```bash
ruff check src/elitefurretai/rl/worker.py && \
pyright src/elitefurretai/rl/worker.py
```

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/worker.py
git commit -m "refactor(worker): drop legacy per-worker model build path"
```

### Task 3.8: Strip legacy ghost path from `WorkerOpponentFactory`

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py`

- [ ] **Step 1: Delete fields**

Remove from `WorkerOpponentFactory.__init__`:
- `self.loaded_ghosts: Dict[str, RNaDAgent] = {}`
- `self.loaded_exploiters: Dict[str, RNaDAgent] = {}` (if present and only used by legacy)

- [ ] **Step 2: Delete methods**

Remove:
- `_get_cached_model`
- `_get_ghost_agent`
- `_get_exploiter_agent` (verify it's not used by centralized path — exploiter agent comes via registry, so the cached version is dead)
- `_load_ghosts` (the disk-scan version) — broadcast is now authoritative
- `set_ghost_paths` (replaced by `set_active_ghost_slots`)
- `set_exploiter_paths` (exploiter has no legacy path on this side)

- [ ] **Step 3: Drop `ghost_paths` from train.py broadcast**

In `train.py` broadcast composition (line ~1669), remove the `"ghost_paths": [...]` and `"exploiter_paths": [...]` keys, and remove the corresponding `set_ghost_paths(...)` / `set_exploiter_paths(...)` calls in `worker.py`'s broadcast handler.

- [ ] **Step 4: Run unit suite**

```bash
pytest unit_tests -q
```
Expected: all PASS.

- [ ] **Step 5: Lint + type check**

```bash
ruff check src/elitefurretai/rl/opponents.py src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py && \
pyright src/elitefurretai/rl/opponents.py src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py
```

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/opponents.py src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py
git commit -m "refactor(opponents): drop legacy ghost/exploiter caching from WorkerOpponentFactory"
```

### Task 3.9: Remove unused main-process player creation in OpponentPool

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py`

Only do this task if Task 3.1 confirmed no production caller exists.

- [ ] **Step 1: Delete dead methods**

Remove from main-process `OpponentPool`:
- `sample_opponent` (lines ~386)
- `_create_self_play_opponent` (line ~533)
- `_create_bc_opponent` (line ~555)
- `_create_exploiter_opponent` (line ~590)
- `_create_ghost_opponent` (line ~624)
- `_create_max_damage_opponent` / `_create_random_baseline_opponent` / `_create_max_base_power_baseline_opponent` / `_create_simple_heuristic_baseline_opponent` / `_create_vgc_bench_baseline_opponent` if also unused — grep first.

- [ ] **Step 2: Remove now-orphaned imports**

If e.g. `Player`-related imports are now unused in opponents.py, drop them. Let `ruff check` catch this.

- [ ] **Step 3: Final grep for dead symbols**

```bash
grep -rn "sample_opponent\|_create_self_play_opponent\|_create_bc_opponent\|_create_exploiter_opponent\|_create_ghost_opponent" \
  src/ unit_tests/ planning/ docs/ 2>/dev/null
```
Anything left: investigate (might be doc references, ok to leave; or might be a missed caller).

- [ ] **Step 4: Run unit suite**

```bash
pytest unit_tests -q
```

- [ ] **Step 5: Lint + type check**

```bash
ruff check src/elitefurretai/rl/opponents.py && \
pyright src/elitefurretai/rl/opponents.py
```

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/opponents.py
git commit -m "refactor(opponents): remove unused main-process sample_opponent family"
```

### Task 3.10: Update documentation

**Files:**
- Modify: `src/elitefurretai/rl/RL.md`
- Modify: `src/elitefurretai/rl/configs/sep_arch.yaml` (comments only, the flag was removed in Task 3.5)

- [ ] **Step 1: Update RL.md**

Search RL.md for these phrases and update:
- `"legacy ghost path retained"` → describe the registry slot pattern instead
- `"still use legacy inference"` → describe centralized as the only mode
- Section 8b (the centralized inference architecture description) — add a paragraph on ghost slot lifecycle

If you're unsure what the section should say, write 3-5 sentences referencing this plan's design doc.

- [ ] **Step 2: Update sep_arch.yaml comments**

Around the area where `enable_centralized_inference: true` used to be (already removed in Task 3.5) and the `max_ghosts: 5` line, replace the multi-line "ghosts use legacy on-demand worker-side loading" comment with a one-liner like:
```yaml
  # Ghost slots are pre-registered in the model registry at startup;
  # see RL.md §8b for the slot rotation lifecycle.
  max_ghosts: 5
```

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/rl/RL.md src/elitefurretai/rl/configs/sep_arch.yaml
git commit -m "docs: update RL.md + sep_arch comments for centralized-only inference"
```

### Task 3.11: Measurement 4 — sep_arch post-cleanup

**Files:** none (read-only measurement)

- [ ] **Step 1: Launch sep_arch**

```bash
nohup python -u src/elitefurretai/rl/train.py \
  --config src/elitefurretai/rl/configs/sep_arch.yaml \
  > /tmp/measurement_4.log 2>&1 &
```

- [ ] **Step 2: Wait for 50 updates, stop, extract metrics**

Same protocol as Task 1.10.

- [ ] **Step 3: Gate check**

Compare to the most recent previous measurement (3 if compile fix landed, otherwise 2). Expected: throughput equal within ±0.2 traj/s. Cleanup shouldn't change behavior. If significantly off: STOP, investigate — cleanup may have touched something load-bearing.

- [ ] **Step 4: Record Measurement 4 in the design doc**

---

## Phase 4 — Final documentation

### Task 4.1: Append comparison table to the design doc

**Files:**
- Modify: `planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md`

- [ ] **Step 1: Add a "Final comparison" section**

Append to the design doc's Updates section:
```markdown
### 2026-05-14 [time] — Final comparison

| Measurement | What | traj/s | learner steps/s | batch fill avg / max | notes |
|---|---|---|---|---|---|
| 1 | Baseline on main @ 1944eda | X.XX | XX | X.X / XX | reference |
| 2 | + ghost centralization | X.XX | XX | X.X / XX | gate: ≥ 1 |
| 3 | + compile race fix (if landed) | X.XX | XX | X.X / XX | gate: ≥ 2 |
| 4 | + cleanup pass | X.XX | XX | X.X / XX | gate: ≈ prev |

Net change from baseline: X.XX traj/s (+Y%).
```

### Task 4.2: Mark the registry plan's Future Work closed

**Files:**
- Modify: `planning/stage2/2026-05-14-00-15-model-registry-plan.md`

- [ ] **Step 1: Append a status update to the registry plan**

Append:
```markdown
### 2026-05-14 [time] — Future work closed

- **Ghost centralization**: SHIPPED. See plan
  `2026-05-14-13-30-ghost-centralization-implementation-plan.md` and
  design `2026-05-14-13-00-...-design.md`.
- **torch.compile multi-thread race**: [SHIPPED with fix X / DEFERRED
  with documented findings].
- **Eval-time inference**: DROPPED. No production caller of
  `OpponentPool.sample_opponent` existed; the methods were removed in
  the cleanup pass.
```

Fill in the bracketed status based on Phase 2's outcome.

### Task 4.3: Final commit

- [ ] **Step 1: Commit doc updates**

```bash
git add planning/stage2/2026-05-14-13-00-ghost-centralization-and-cleanup-design.md \
        planning/stage2/2026-05-14-00-15-model-registry-plan.md
git commit -m "docs(plan): record final measurements + close registry plan future work"
```

- [ ] **Step 2: Final verification**

```bash
ruff check src unit_tests
ruff format src unit_tests --check
pyright src unit_tests
pytest unit_tests -q
```
Expected: all clean.

- [ ] **Step 3: Show summary**

```bash
git log --oneline main^..HEAD  # actually, use the commit before Phase 0 work started
```

Done. Report the final comparison table to Cayman.

---

## Self-review notes

- **Spec coverage**: every section of the design doc maps to at least one
  task. Ghost design → Tasks 1.1–1.10. Compile race → Tasks 2.1–2.6.
  Cleanup → Tasks 3.1–3.11. Measurement protocol → Tasks 0.1, 1.10, 2.5,
  3.11. Sequencing/gates → present throughout.
- **Placeholders**: filenames, line numbers, code, and commit messages
  are all concrete. Two tasks (1.5 helper, 2.5 fix deployment) note
  alternative paths depending on real code state — annotated as
  implementation notes rather than placeholders.
- **Type consistency**: `active_ghost_slots` is a `Set[int]` throughout
  (returned as `sorted` list in broadcasts, restored to set on receive).
  `slot_for_ghost_path` is `Dict[str, int]`. Method names match across
  tasks.
- **Skipped-task semantics**: Phase 2 tasks 2.2/2.3/2.4 explicitly say
  "only do this if the previous task failed" to prevent over-execution.
