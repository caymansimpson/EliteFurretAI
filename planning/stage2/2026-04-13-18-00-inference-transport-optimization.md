# Inference Transport Optimization: Eliminating the Pipe+Pickle Bottleneck

## Context

### Where We Are

This is **Stage 2** of EliteFurretAI development (Single-Team League Mastery). The RL training system uses an IMPALA-style multiprocessing architecture with:
- **Worker processes**: Run battle simulation (Rust engine) + embedding (Python), and request inference from a central server
- **Central GPU inference server**: A separate `spawn`-based process that receives state tensors from workers, batches them, runs the ~100M parameter model on GPU, and returns logits + hidden states
- **Learner process**: Consumes completed trajectories from workers, runs RNaD gradient updates on GPU

The centralized GPU inference server was implemented in the prior session to move the dominant bottleneck (60-70% of wall time was CPU model inference) onto the idle GPU. The infrastructure works correctly, but **the transport overhead negated the GPU speed advantage**, resulting in a net throughput regression.

### What Led to This Plan

A phased throughput optimization effort documented in [2026-04-13-14-00-training-speed-recommendations.md](2026-04-13-14-00-training-speed-recommendations.md) implemented:

1. **Fast full embedding** (`embed_to_array` path): 0.320 → 0.738 battles/s on benchmarks
2. **Centralized GPU inference server**: Workers send state tensors over `multiprocessing.Pipe` to a GPU process instead of running CPU forward passes
3. **Step cap raised to 40** with truncation visibility: 0 truncated battles across all sweep runs
4. **Worker count sweep** (4/6/8 workers): Found 6 workers optimal, 8 workers regressed

Despite GPU inference being 10-20x faster per forward pass, the system got **slower**:

| Config | Learner steps/s | Battles/s |
|--------|----------------|-----------|
| **Baseline** (CPU inference, 4 workers, steady state) | **9.3** | **0.35** |
| GPU inference, 4 workers | 6.04 | 0.20 |
| GPU inference, 6 workers | 6.66 | 0.22 |
| GPU inference, 8 workers | 5.18 | 0.17 |

The 8-worker run confirmed the inference server was CPU-saturated: the server process was near 100% CPU utilization while worker CPU dropped, meaning the server was spending most of its time on **transport overhead** rather than GPU inference.

### Key Constraint

A fresh AI should be able to implement this plan without additional context. All file paths, function signatures, data structures, and expected changes are specified below.

---

## Problem Statement

The centralized GPU inference server's transport layer uses Python `multiprocessing.Pipe` with `pickle` serialization and `dict`-based message framing. Every inference request involves:

1. **Client** (worker): convert tensors to numpy → wrap in Python dict → pickle serialize → Pipe write (kernel copy)
2. **Server**: Pipe read (kernel copy) → pickle deserialize → extract numpy → convert to torch tensor → move to GPU → run forward pass → convert output to numpy → wrap in dict → pickle serialize → Pipe write
3. **Client**: Pipe read → pickle deserialize → extract numpy → convert to torch tensors

Each round trip serializes **~93 KB per sample** (or ~279 KB for a batched request of 3) through pickle **twice** (request + response), involving multiple memory copies and Python object allocation/destruction.

The GPU forward pass itself takes ~2-5ms for a batch. But the transport overhead adds ~5-10ms per request cycle (pickle + pipe + batch_timeout wait), making the server's per-cycle time dominated by CPU-side overhead rather than useful GPU compute.

---

## Evidence

### Measured Throughput Regression

From the worker sweep documented in the parent planning doc:

| Workers | Learner steps/s | Battles/s | Worker RSS | Server RSS |
|---------|----------------|-----------|------------|------------|
| 4 (CPU baseline) | 9.3 (steady) / 8.19 (1st update) | 0.35 / 0.30 | N/A | N/A |
| 4 (GPU server) | 6.04 | 0.20 | 2.42 GB | 1.68 GB |
| 6 (GPU server) | 6.66 | 0.22 | 2.42 GB | 1.68 GB |
| 8 (GPU server) | 5.18 | 0.17 | 2.47 GB | 1.62 GB |

Key observations:
- 6 workers improved over 4, but only by 10% (6.04 → 6.66 steps/s)
- 8 workers **regressed** below 4 workers (5.18 steps/s), confirming the server is the bottleneck
- Server RSS was stable across all runs, so it's not memory pressure — it's CPU saturation

### Calculated Payload Sizes

Model config: LSTM with `lstm_layers=4`, bidirectional (8 effective layer-directions), `lstm_hidden_size=512`.
Feature set: `full` with 5,222 features.

**Per-sample request (worker → server):**
| Component | Dimensions | Bytes | KB |
|-----------|-----------|-------|-----|
| State vector | (5222,) float32 | 20,888 | 20.4 |
| Hidden h | (8, 1, 512) float32 | 16,384 | 16.0 |
| Hidden c | (8, 1, 512) float32 | 16,384 | 16.0 |
| **Total data** | | **53,656** | **52.4** |

**Per-sample response (server → worker):**
| Component | Dimensions | Bytes | KB |
|-----------|-----------|-------|-----|
| Turn logits | (2025,) float32 | 8,100 | 7.9 |
| TP logits | (90,) float32 | 360 | 0.4 |
| Values | (1,) float32 | 4 | <0.1 |
| Win dist logits | (51,) float32 | 204 | 0.2 |
| Hidden h | (8, 1, 512) float32 | 16,384 | 16.0 |
| Hidden c | (8, 1, 512) float32 | 16,384 | 16.0 |
| **Total data** | | **41,436** | **40.5** |

**Round trip: 92.9 KB per sample, 278.6 KB for batch=3, 557.2 KB for batch=6.**

All of this passes through `pickle.dumps()` and `pickle.loads()` twice per round trip (once for request, once for response), plus kernel buffer copies for the Pipe.

### Code-Level Transport Path

**Request path** (`CentralGpuInferenceClient.forward()` in [central_inference.py](../../src/elitefurretai/rl/central_inference.py#L131)):
```python
self.connection.send({              # pickle serializes entire dict
    "kind": "infer",
    "model_key": model_key,
    "state": state_tensor.detach().cpu().numpy(),   # tensor → numpy copy
    "hidden": _serialize_hidden(hidden_state),       # tensor → numpy copy (recursive for LSTM tuple)
})
response = self.connection.recv()   # pickle deserializes entire dict
turn_logits = torch.as_tensor(response["turn_logits"])  # numpy → tensor
```

**Server path** (`process_infer_requests()` in [central_inference.py](../../src/elitefurretai/rl/central_inference.py#L207)):
```python
# For each request: deserialize state and hidden from numpy
state_tensor = _state_to_tensor(request["state"], device)           # numpy → tensor → GPU
hidden_tensor = _move_hidden_to_device(_deserialize_hidden(request.get("hidden")), device)
# GPU forward pass
turn_logits, tp_logits, values, win_dist_logits, next_hidden = model.forward_with_hidden(...)
# Serialize response
connection.send({
    "turn_logits": turn_logits.detach().cpu().numpy(),    # tensor → numpy copy
    ...
    "next_hidden": _serialize_hidden(next_hidden),        # tensor → numpy copy
})
```

**Server event loop** (`run_central_gpu_inference_server()` in [central_inference.py](../../src/elitefurretai/rl/central_inference.py#L276)):
```python
# Outer loop: wait for any connection to have data
ready_connections = wait(active_connections, timeout=0.05)
# Start batch collection with 5ms deadline
deadline = time.perf_counter() + batch_timeout  # batch_timeout=0.005
# Read all ready messages, wait for more up to deadline
# Process entire batch
process_infer_requests(batch_requests)
```

The 5ms `batch_timeout` adds artificial latency to every batch cycle. With ~30 requests/second across 6 workers, requests arrive every ~33ms, so the 5ms window captures only ~0.15 additional requests on average — minimal batching benefit at high per-request cost.

### Original cProfile Evidence

From the Showdown benchmark (pre-GPU-server), CPU inference breakdown:
- `torch._C._nn.linear`: 6.19s (36.4% of total) — pure matrix mult, 20-50x faster on GPU
- `{torch.lstm}`: 2.68s — LSTM forward, batches well on GPU
- Total inference: 15.97s / 16.96s = 94% was model forward pass

A single CPU forward pass: ~77ms (15.97s ÷ 208 calls).
A single GPU forward pass: ~2-5ms (batch of 16).

The GPU is 15-40x faster at the raw forward pass. The transport overhead must be consuming the entire speedup for the overall system to be slower.

---

## Current Architecture (Full Detail)

### File Map

| File | Role |
|------|------|
| `src/elitefurretai/rl/central_inference.py` | Inference server process + client class (~460 lines) |
| `src/elitefurretai/engine/sync_battle_driver.py` | `SyncPolicyPlayer` (uses inference client), `SyncRustBattleDriver` (~760 lines) |
| `src/elitefurretai/rl/train.py` | Training loop, worker process creation, Pipe wiring (~1600 lines) |
| `src/elitefurretai/rl/config.py` | `RNaDConfig` dataclass with inference settings |
| `src/elitefurretai/supervised/model_archs.py` | `FlexibleThreeHeadedModel` with `forward_with_hidden()` |
| `src/elitefurretai/rl/players.py` | `RNaDAgent` wrapper with `get_initial_state()` |

### Request Flow

```
Worker Process                          Inference Server Process (spawn, GPU)
─────────────────                       ────────────────────────────────────
SyncRustBattleDriver.run()
  └─ SyncPolicyPlayer.choose_actions_from_snapshots()
       ├─ embed all snapshots (CPU)
       ├─ batch state tensors (N, 1, 5222)
       ├─ concat LSTM hidden (8, N, 512) × 2
       └─ CentralGpuInferenceClient.forward()
            ├─ state_tensor.detach().cpu().numpy()     ─┐
            ├─ _serialize_hidden(hidden) → numpy        │ ~52.4 KB × N
            ├─ connection.send(dict)  ──────────────────┤─── pickle + pipe ───►
            │                                           │      ▼
            │                                           │ connection.recv() (pickle)
            │                                           │ _state_to_tensor() → GPU
            │                                           │ _deserialize_hidden() → GPU
            │                                           │ model.forward_with_hidden()  [~2-5ms GPU]
            │                                           │ output.cpu().numpy()
            │                                           │ connection.send(dict) ◄── pickle + pipe
            ├─ response = connection.recv() ◄───────────┘   ~40.5 KB × N
            ├─ torch.as_tensor(response[...])
            └─ return logits, values, hidden
```

### Key Classes

**`CentralGpuInferenceClient`** ([central_inference.py](../../src/elitefurretai/rl/central_inference.py#L101)):
- Holds a `multiprocessing.Connection` (one end of a Pipe)
- `register_model(model_key, model_config, state_dict)`: sends full state_dict over Pipe (one-time per weight update, ~400MB)
- `forward(model_key, state_tensor, hidden_state)`: sends dict with numpy arrays, receives dict with numpy arrays

**`SyncPolicyPlayer`** ([sync_battle_driver.py](../../src/elitefurretai/engine/sync_battle_driver.py#L139)):
- Has `inference_backend: Optional[CentralGpuInferenceClient]`
- `_uses_central_inference`: True when backend + model_key + model_config are all set
- Two inference paths that both use the backend:
  - `_sample_masked_action()`: single-request path (for Transformer or single snapshot)
  - `choose_actions_from_snapshots()`: batched path (batches all concurrent snapshots into one request)

**Server event loop** (`run_central_gpu_inference_server()` [central_inference.py](../../src/elitefurretai/rl/central_inference.py#L183)):
- Runs in a `spawn`-context child process (required for clean CUDA init)
- Sets `torch.set_num_threads(1)` to avoid stealing CPU from workers
- Batching: groups requests by `model_key`, concatenates LSTM hidden states, runs one GPU forward pass
- **Transformer path does NOT batch**: processes each request individually
- `batch_timeout=0.005` (5ms), `max_batch_size=64`

### Pipe Setup (in `train.py` lines 1530-1560)

```python
# For each worker, create a duplex Pipe
for _ in range(config.num_workers):
    parent_connection, child_connection = mp.Pipe(duplex=True)
    inference_parent_connections.append(parent_connection)
    worker_inference_connections[worker_index] = child_connection

# Server gets parent ends, workers get child ends
inference_server_process = start_central_gpu_inference_server(
    inference_parent_connections,
    device=config.device,
    batch_timeout=config.central_inference_batch_timeout,
    max_batch_size=config.central_inference_max_batch_size,
)
```

After spawning workers, the parent closes the child connections (correct lifecycle management).

---

## Root Cause Analysis

### Why GPU Inference Is Slower Than CPU Inference End-to-End

With **CPU inference** (baseline):
- Worker does everything locally: embed (4ms) → forward pass (77ms) → action select (0.5ms) = **~82ms per decision**
- 4 workers run **in parallel** with no IPC, no contention: effective throughput = 4 × ~12 decisions/s = ~48 decisions/s

With **GPU inference** (central server):
- Worker: embed (4ms) → serialize (0.5ms) → Pipe send → **wait for server** → Pipe recv → deserialize (0.5ms) → action select (0.5ms)
- Server per batch cycle: batch_timeout wait (5ms) + read/deserialize requests (0.5-1ms) + GPU forward (3ms) + serialize/send responses (0.5-1ms) = **~10ms per cycle**
- All 4-8 workers funnel through **one serial server process**, so the server is the throughput ceiling

The server can process ~100 cycles/second (1000ms / 10ms). Each cycle handles ~1-6 samples. So max throughput = ~100-600 decisions/second. This seems like enough headroom, but:

1. **The 5ms batch_timeout is dead time** where the server does nothing useful. At 30 requests/s, reducing to 1ms would barely affect batching but save 4ms per cycle.
2. **Pickle overhead scales with payload size**. For a batch=6 request (~314 KB) going through pickle twice, the CPU time is non-trivial.
3. **The Python dict protocol allocates/deallocates many objects per cycle**: dict construction, numpy array creation, pickle byte buffer creation, response dict, etc. This creates GC pressure.
4. **The server is single-threaded**: it reads connections, processes, sends responses, then reads again. No pipelining.

### The Hidden Cost: Per-Request Object Allocation

Each request-response cycle creates and destroys:
- 2 Python dicts (request + response)
- 4-8 numpy arrays (state, h, c, turn_logits, tp_logits, values, win_dist, hidden)
- 2 pickle byte buffers (~350 KB total)
- Intermediate torch tensors during conversion

At ~30-100 cycles/second, the Python memory allocator and garbage collector are doing significant work.

---

## Proposed Solution: Phased Transport Optimization

### Phase 0: Instrument the Transport Layer (Evidence Gathering)

**Goal**: Get exact timing for each component of the request cycle before making changes.

**What to add**: Lightweight `time.perf_counter()` instrumentation around:

1. In `CentralGpuInferenceClient.forward()`:
   - Time spent in `state_tensor.detach().cpu().numpy()` + `_serialize_hidden()`
   - Time spent in `connection.send()`
   - Time spent waiting in `connection.recv()`
   - Time spent in `torch.as_tensor()` + `_deserialize_hidden()` on response

2. In `run_central_gpu_inference_server()`:
   - Time from `wait()` return to first `connection.recv()` complete
   - Time spent in `_state_to_tensor()` + `_deserialize_hidden()` for all requests
   - Time spent in `model.forward_with_hidden()` (pure GPU compute)
   - Time spent in response serialization + `connection.send()` for all responses
   - Time spent in batch_timeout wait (idle time between first request and batch processing)

3. Print a summary every N cycles (e.g., every 100) with averages:
   ```
   [InferenceServer] Last 100 cycles: avg_batch_size=2.3, avg_idle_ms=4.8, avg_deser_ms=0.6, avg_gpu_ms=3.1, avg_ser_ms=0.5, avg_cycle_ms=9.0
   ```

**How to run**: Use the existing `worker_sweep_nw6.yaml` config with `max_updates=1`. The sweep produces enough inference cycles to get stable averages.

**Files to modify**:
- `src/elitefurretai/rl/central_inference.py`: Add timing in `CentralGpuInferenceClient.forward()` and `run_central_gpu_inference_server()`

**Expected output**: Exact breakdown confirming how much time goes to idle/batch_timeout vs. serialization vs. GPU. This data determines whether Phase 1 (binary protocol) or Phase 2 (batch tuning) is the higher-priority change.

#### Implementation Details

Add a `TransportMetrics` dataclass to track cumulative timing in the server:

```python
@dataclass
class TransportMetrics:
    cycles: int = 0
    total_idle_s: float = 0.0          # Time in batch_timeout wait
    total_recv_deser_s: float = 0.0    # Reading + deserializing requests
    total_gpu_s: float = 0.0           # model.forward_with_hidden 
    total_ser_send_s: float = 0.0      # Serializing + sending responses
    total_batch_samples: int = 0       # Number of samples processed

    def log_summary(self, every_n: int = 100) -> None:
        if self.cycles % every_n != 0 or self.cycles == 0:
            return
        n = every_n
        avg_batch = self.total_batch_samples / max(n, 1)
        # ... log averages, then reset
```

Add timing in `process_infer_requests()`:
- Before the loop: `gpu_start = time.perf_counter()`
- After `model.forward_with_hidden()`: accumulate GPU time
- Before/after `connection.send()`: accumulate send time

Add timing in the main event loop:
- After `deadline = time.perf_counter() + batch_timeout`: record idle start
- After batch collection completes: `metrics.total_idle_s += idle_time`

On the client side, add timing to `CentralGpuInferenceClient.forward()` and accumulate into a profile attribute (similar to `SyncPolicyProfile`):

```python
@dataclass
class InferenceClientProfile:
    serialize_seconds: float = 0.0
    send_seconds: float = 0.0
    wait_seconds: float = 0.0    # recv() blocking time
    deserialize_seconds: float = 0.0
    requests: int = 0
```

---

### Phase 1: Binary Protocol (Replace Pickle with Raw Bytes)

**Goal**: Eliminate pickle serialization overhead by switching to a fixed-format binary protocol using `connection.send_bytes()` / `connection.recv_bytes()`.

**Why this is the highest-impact change**:
- Pickle serialization of numpy arrays involves: object introspection → type dispatch → byte buffer allocation → data copy → framing. For ~300 KB payloads, this is measurable.
- `connection.send_bytes()` / `recv_bytes()` bypass pickle entirely: they send raw bytes with only a 4-byte length prefix.
- Numpy arrays can be serialized/deserialized as raw bytes via `array.tobytes()` / `np.frombuffer()`, which are essentially `memcpy` — much cheaper than pickle.

**Protocol Design**:

A request consists of a fixed **header** followed by contiguous **data bytes**:

```
REQUEST HEADER (28 bytes, little-endian):
  [0:1]   uint8   message_type     # 0=infer, 1=register, 2=stop
  [1:2]   uint8   model_key_len    # length of model_key string
  [2:4]   uint16  batch_size       # number of samples in this request
  [4:8]   uint32  state_dim        # feature vector dimension (5222)
  [8:10]  uint16  hidden_layers    # num_layers * num_directions (8)
  [10:12] uint16  hidden_size      # LSTM hidden size (512)
  [12:13] uint8   has_hidden       # 0 if hidden is None, 1 otherwise
  [13:28] padding/reserved

REQUEST DATA (contiguous float32):
  model_key bytes     [model_key_len bytes, UTF-8]
  state data          [batch_size × state_dim × 4 bytes]
  hidden_h data       [hidden_layers × batch_size × hidden_size × 4 bytes]  (if has_hidden)
  hidden_c data       [hidden_layers × batch_size × hidden_size × 4 bytes]  (if has_hidden)
```

```
RESPONSE HEADER (16 bytes, little-endian):
  [0:1]   uint8   status           # 0=ok, 1=error
  [1:3]   uint16  batch_size       # number of samples
  [3:5]   uint16  turn_dim         # 2025
  [5:7]   uint16  tp_dim           # 90
  [7:9]   uint16  value_dim        # 1
  [9:11]  uint16  win_dist_dim     # 51
  [11:13] uint16  hidden_layers    # 8
  [13:15] uint16  hidden_size      # 512
  [15:16] uint8   has_hidden       # 0 or 1

RESPONSE DATA (contiguous float32):
  turn_logits   [batch_size × turn_dim × 4]
  tp_logits     [batch_size × tp_dim × 4]
  values        [batch_size × value_dim × 4]
  win_dist      [batch_size × win_dist_dim × 4]
  hidden_h      [hidden_layers × batch_size × hidden_size × 4]  (if has_hidden)
  hidden_c      [hidden_layers × batch_size × hidden_size × 4]  (if has_hidden)
```

Error responses use `status=1` and data is a UTF-8 error message string.

**Implementation details**:

The `register_model` message type continues to use `connection.send()` with pickle, because it's a one-time operation per weight update (~400MB state_dict) and the register path doesn't need to be fast. Only the hot `infer` path switches to binary.

For the `stop` message, just send a 1-byte message: `connection.send_bytes(b'\x02')`.

#### Client-side changes (`CentralGpuInferenceClient`)

Replace `forward()` method:

```python
import struct

_REQUEST_HEADER_FMT = "<BBHIHHBxxxxxxxxxxxxx"  # 28 bytes, padded
_REQUEST_HEADER_SIZE = struct.calcsize(_REQUEST_HEADER_FMT)

_RESPONSE_HEADER_FMT = "<BHHHHHHB"  # 16 bytes
_RESPONSE_HEADER_SIZE = struct.calcsize(_RESPONSE_HEADER_FMT)

def forward(self, model_key, state_tensor, hidden_state=None, mask=None):
    # Prepare numpy arrays (still needed for data, but no pickle)
    state_np = state_tensor.detach().cpu().numpy()
    if state_np.ndim == 3:
        state_np = state_np.squeeze(1)  # (batch, seq=1, dim) → (batch, dim)
    batch_size = state_np.shape[0]
    state_dim = state_np.shape[1]
    
    key_bytes = model_key.encode("utf-8")
    has_hidden = hidden_state is not None
    
    if has_hidden and isinstance(hidden_state, tuple):
        h_np = hidden_state[0].detach().cpu().numpy()  # (layers, batch, hidden)
        c_np = hidden_state[1].detach().cpu().numpy()
        hidden_layers = h_np.shape[0]
        hidden_size = h_np.shape[2]
    else:
        h_np = c_np = None
        hidden_layers = hidden_size = 0
    
    # Pack header
    header = struct.pack(
        _REQUEST_HEADER_FMT,
        0,                  # message_type = infer
        len(key_bytes),
        batch_size,
        state_dim,
        hidden_layers,
        hidden_size,
        1 if has_hidden else 0,
    )
    
    # Build contiguous buffer
    parts = [header, key_bytes, state_np.tobytes()]
    if has_hidden:
        parts.append(h_np.tobytes())
        parts.append(c_np.tobytes())
    
    self.connection.send_bytes(b"".join(parts))
    
    # Receive response
    resp_bytes = self.connection.recv_bytes()
    resp_header = struct.unpack_from(_RESPONSE_HEADER_FMT, resp_bytes, 0)
    status, r_batch, turn_dim, tp_dim, value_dim, win_dist_dim, r_hidden_layers, r_hidden_size = resp_header[:8]
    r_has_hidden = resp_header[7] if len(resp_header) > 7 else 0
    
    if status != 0:
        raise RuntimeError(f"Inference error: {resp_bytes[_RESPONSE_HEADER_SIZE:].decode('utf-8', errors='replace')}")
    
    offset = _RESPONSE_HEADER_SIZE
    # Parse contiguous float32 arrays from buffer
    def read_array(shape):
        nonlocal offset
        size = 1
        for s in shape:
            size *= s
        arr = np.frombuffer(resp_bytes, dtype=np.float32, count=size, offset=offset).reshape(shape)
        offset += size * 4
        return torch.as_tensor(arr.copy())  # copy because frombuffer gives read-only view
    
    turn_logits = read_array((r_batch, turn_dim))
    tp_logits = read_array((r_batch, tp_dim))
    values = read_array((r_batch, value_dim))
    win_dist_logits = read_array((r_batch, win_dist_dim))
    
    next_hidden = None
    if r_has_hidden:
        h = read_array((r_hidden_layers, r_batch, r_hidden_size))
        c = read_array((r_hidden_layers, r_batch, r_hidden_size))
        next_hidden = (h, c)
    
    return turn_logits, tp_logits, values, win_dist_logits, next_hidden
```

#### Server-side changes (`run_central_gpu_inference_server`)

The server's main loop must distinguish between binary inference requests and pickle-based register/stop messages. Strategy:

1. Use `connection.recv_bytes()` to receive all messages as raw bytes
2. Check the first byte to determine message type:
   - `0x00` = binary inference request → parse binary protocol
   - Anything else = use `pickle.loads()` for backwards-compatible register_model/stop handling

Wait — this doesn't work cleanly because `connection.send()` (pickle) and `connection.send_bytes()` use different wire formats and `recv()` vs `recv_bytes()` aren't interchangeable.

**Better approach**: Use `send_bytes()` / `recv_bytes()` for ALL messages. For register_model, pickle the dict first, then send the pickled bytes via `send_bytes()`. For infer, use the binary protocol. Discriminate by the first byte.

```python
# Client sends register:
payload = pickle.dumps({"kind": "register_model", ...})
connection.send_bytes(b'\x01' + payload)  # 0x01 = pickle-framed register

# Client sends infer:
connection.send_bytes(b'\x00' + header + data)  # 0x00 = binary infer

# Client sends stop:
connection.send_bytes(b'\x02')  # 0x02 = stop

# Server receives:
raw = connection.recv_bytes()
msg_type = raw[0]
if msg_type == 0:    # binary infer
    header = struct.unpack_from(_REQUEST_HEADER_FMT, raw, 1)
    ...
elif msg_type == 1:  # pickle register
    message = pickle.loads(raw[1:])
    ...
elif msg_type == 2:  # stop
    remove_connection(connection)
```

This is clean, backward-compatible in spirit, and avoids mixing `send()`/`send_bytes()` on the same Pipe.

#### Server batching changes

The current server batches by reading all ready connections and grouping by model_key. With the binary protocol:

1. Read raw bytes from each ready connection
2. Parse binary headers to get batch_size, state_dim, etc.
3. For LSTM: concatenate state bytes directly (they're already contiguous float32) before converting to a single GPU tensor
4. Run GPU forward pass on the combined batch
5. Serialize response in binary format for each connection

**Optimization**: Instead of creating individual numpy arrays per request then concatenating, concatenate the raw bytes first, then create one numpy array:

```python
# Accumulate raw state bytes from all requests
all_state_bytes = b"".join(request.state_bytes for request in batch_requests)
total_samples = sum(request.batch_size for request in batch_requests)
combined_state_np = np.frombuffer(all_state_bytes, dtype=np.float32).reshape(total_samples, state_dim)
state_tensor = torch.as_tensor(combined_state_np, device=device).unsqueeze(1)
```

This avoids intermediate numpy array allocation per request.

#### Files to modify

| File | Changes |
|------|---------|
| `src/elitefurretai/rl/central_inference.py` | Replace `forward()` with binary protocol; update server recv loop; keep `register_model` as pickle-over-bytes; add 1-byte message type discriminator |

**No changes needed** to `sync_battle_driver.py`, `train.py`, or `config.py` — the `CentralGpuInferenceClient` API surface (`forward()`, `register_model()`) remains the same.

---

### Phase 2: Reduce Batch Timeout

**Goal**: Reduce or eliminate the `batch_timeout` wait that adds dead time to every server cycle.

**Current value**: `batch_timeout=0.005` (5ms), configurable via `RNaDConfig.central_inference_batch_timeout`.

**Analysis**: With 6 workers generating ~30 requests/second total, inter-request gap is ~33ms. In a 5ms window, the probability of a second request arriving is ~15%. The expected batch size improvement from waiting 5ms: +0.15 samples. Against GPU inference speed of ~3ms per batch regardless of batch size (for small batches), holding 5ms to maybe add 0.15 samples is a bad tradeoff.

**Proposed change**: Reduce to `0.001` (1ms), or implement **adaptive batch timeout**:

```python
# Adaptive: wait longer when requests are frequent, shorter when sparse
if len(batch_requests) > 0:
    # Already have requests; short deadline
    deadline = time.perf_counter() + 0.001
else:
    # No requests yet; use normal timeout
    deadline = time.perf_counter() + 0.005
```

However, the simplest effective change is just reducing the default to 0.001:

**Files to modify**:
| File | Changes |
|------|---------|
| `src/elitefurretai/rl/config.py` | Change `central_inference_batch_timeout` default from `0.005` to `0.001` |
| `src/elitefurretai/rl/central_inference.py` | Change `batch_timeout` parameter default from `0.005` to `0.001` |

---

### Phase 3: Pin Inference Server to Dedicated CPU Core

**Goal**: Prevent the inference server process from being scheduled across cores, improving cache locality and reducing context-switch overhead.

**Implementation**: Add one line at the start of `run_central_gpu_inference_server()`:

```python
import os
try:
    # Pin to the last logical core, leaving others for workers
    os.sched_setaffinity(0, {os.cpu_count() - 1})
except (OSError, AttributeError):
    pass  # Not available on all platforms
```

On the 8-core WSL box with 6 workers, this dedicates core 7 to the inference server while workers spread across cores 0-6.

**Files to modify**:
| File | Changes |
|------|---------|
| `src/elitefurretai/rl/central_inference.py` | Add `os.sched_setaffinity()` at start of `run_central_gpu_inference_server()` |

---

### Phase 4: Shared Memory Transport (If Needed)

**Goal**: If the binary protocol (Phase 1) doesn't provide sufficient improvement, replace Pipe transfers entirely with shared memory buffers, using the Pipe only for lightweight signaling.

**When to consider**: If Phase 0 instrumentation shows that the `send_bytes()`/`recv_bytes()` kernel copies still dominate (i.e., the data is large enough that kernel buffer management is slow), shared memory eliminates the kernel copy entirely.

**Design sketch**:

For each worker, pre-allocate two `multiprocessing.shared_memory.SharedMemory` regions:

```
request_shm:  max_batch_size × (state_dim + 2 × hidden_layers × hidden_size) × 4 bytes
response_shm: max_batch_size × (turn_dim + tp_dim + value_dim + win_dist_dim + 2 × hidden_layers × hidden_size) × 4 bytes
```

With max_batch_size=16 (reasonable for per-worker batch):
- request_shm: 16 × (5222 + 2 × 8 × 512) × 4 = 16 × 13,414 × 4 = **858,496 bytes** (~858 KB per worker)
- response_shm: 16 × (2025 + 90 + 1 + 51 + 2 × 8 × 512) × 4 = 16 × 10,359 × 4 = **662,976 bytes** (~663 KB per worker)

For 6 workers: 6 × (858 + 663) = **~9.1 MB total shared memory**. This is negligible.

**Flow**:
1. Worker writes state + hidden arrays directly to its `request_shm` region
2. Worker sends 4-byte signal over Pipe: `struct.pack("<HH", batch_size, state_dim)`
3. Server reads signal, gets a numpy view of the worker's `request_shm` via `np.ndarray(buffer=shm.buf)`
4. Server copies to GPU tensor (required — can't avoid this copy for CUDA)
5. Server writes results to `response_shm`
6. Server sends 2-byte signal over Pipe: `struct.pack("<H", batch_size)`
7. Worker reads signal, gets numpy view of `response_shm`

**Pipe data per request**: ~4 bytes (request signal) + ~2 bytes (response signal) = **6 bytes** instead of ~350 KB.

**Lifecycle**: SharedMemory must be `.unlink()`ed on shutdown. Track in the client's `__del__` or in a cleanup callback registered with `atexit`.

This is deferred because the binary protocol (Phase 1) may provide sufficient improvement. The shared memory path adds complexity (cleanup, error handling, buffer sizing) that's only justified if kernel pipe copies remain a bottleneck.

---

## Complete Implementation Order

1. **Phase 0**: Instrument transport layer → run 6-worker config → analyze timing breakdown
2. **Phase 1**: Binary protocol implementation → re-run same config → compare
3. **Phase 2**: Reduce batch_timeout to 0.001 → small config change → re-run
4. **Phase 3**: Pin server to core 7 → one-liner → re-run
5. **Validation**: Run full 6-worker sweep (same as `worker_sweep_nw6.yaml` with `max_updates=1`) and compare learner steps/s, battles/s against the current 6.66 / 0.22 baseline
6. **(If needed)** Phase 4: Shared memory → re-run

---

## Validation Plan

### Benchmark Config

Use the existing `worker_sweep_nw6.yaml` config with `max_updates=1`:
- `num_workers=6`, `num_players=18`, `num_battles_per_pair=32`
- `max_battle_steps=40`, `use_mixed_precision=true`, `device=cuda`
- This produces 256 completed battles and one learner update

### Success Metrics

| Metric | Current (6w GPU server) | Target (post-optimization) |
|--------|------------------------|---------------------------|
| Learner steps/s | 6.66 | ≥ 9.0 (recover baseline or better) |
| Battles/s | 0.22 | ≥ 0.30 |
| Truncated battles | 0 | 0 (no regression) |
| Server CPU % at 6 workers | near 100% | < 70% (headroom) |

The primary success criterion is **recovering the pre-GPU-server baseline of 9.3 steps/s** at steady state. If we exceed it, that means the GPU inference advantage is finally coming through.

### Secondary Validation

After the transport optimization, re-run the 8-worker config to see if the server can now handle 8 workers without regression:
- If 8 workers now exceeds 6 workers in steps/s, the server bottleneck has been resolved
- If 8 workers still regresses, the bottleneck has shifted elsewhere (embedding, GIL, etc.)

### Comparison Baseline

All benchmarks should compare against these established datapoints from the worker sweep:

| Workers | Steps/s | Battles/s | Duration (256 battles) |
|---------|---------|-----------|----------------------|
| 4 (GPU, current) | 6.04 | 0.20 | 1275s |
| 6 (GPU, current) | 6.66 | 0.22 | 1141s |
| 8 (GPU, current) | 5.18 | 0.17 | 1469s |

---

## Files To Modify (Summary)

| Phase | File | What Changes |
|-------|------|-------------|
| 0 | `src/elitefurretai/rl/central_inference.py` | Add `TransportMetrics` dataclass; instrument `forward()` and server loop; periodic logging |
| 1 | `src/elitefurretai/rl/central_inference.py` | Replace pickle dict protocol with binary `send_bytes`/`recv_bytes`; 1-byte message type discriminator; register_model uses pickle-over-bytes; binary header+data for infer |
| 2 | `src/elitefurretai/rl/config.py` | Reduce `central_inference_batch_timeout` default from 0.005 to 0.001 |
| 2 | `src/elitefurretai/rl/central_inference.py` | Reduce `batch_timeout` parameter default from 0.005 to 0.001 |
| 3 | `src/elitefurretai/rl/central_inference.py` | Add `os.sched_setaffinity(0, {os.cpu_count() - 1})` at start of server function |

**No changes needed to**: `sync_battle_driver.py`, `train.py`, `model_archs.py`, `players.py`, or any config YAML files. The `CentralGpuInferenceClient.forward()` and `register_model()` API surface remains identical. Workers are unaffected.

---

## Non-Goals

1. **Do not remove the Pipe**: Pipes are still used for lightweight signaling and register_model. Only the hot-path data transfer changes.
2. **Do not change the model architecture or feature set**: This is purely a transport optimization.
3. **Do not change the number of workers**: The existing 6-worker config is the benchmark target.
4. **Do not remove the Showdown websocket backend**: All changes are scoped to the centralized inference server.
5. **Do not change the learner or trajectory queue**: Those are separate from the inference transport.
6. **Do not clean up the shutdown ConnectionResetError**: That's a separate, cosmetic issue.
7. **Do not change how `sync_remote_model` / `register_model` works**: The one-time weight registration can remain pickle-based since it's infrequent (~once per learner update).

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Binary protocol introduces subtle serialization bugs (wrong byte offsets, endianness) | High | Write a unit test that round-trips a known state+hidden through the binary protocol and compares against pickle results |
| Shared memory cleanup failure on crash | Medium | Phase 4 only; use `atexit` + `try/finally` in server; SharedMemory names should be unique per run |
| Batch timeout reduction causes many single-sample batches, reducing GPU efficiency | Low | GPU forward pass at batch=1 is still ~2-3ms; any loss is offset by reduced idle time. Monitor batch size distribution in Phase 0 metrics. |
| `sched_setaffinity` not available in WSL | Low | Wrapped in try/except; degrades gracefully |

---

## Appendix: How a Fresh AI Should Execute This

### Prerequisites
1. Activate the venv: `source ../venv/bin/activate`
2. Verify the workspace builds: `ruff check src unit_tests && pyright src unit_tests`
3. Read `src/elitefurretai/rl/central_inference.py` in full (~460 lines) — this is the only file with significant changes

### Execution Steps

**Step 1**: Implement Phase 0 instrumentation in `central_inference.py`. Run `worker_sweep_nw6.yaml` with `max_updates=1`. Capture the timing summary from logs. Record in a planning doc update.

**Step 2**: Implement Phase 1 binary protocol in `central_inference.py`. The key is:
- All messages now use `send_bytes()` / `recv_bytes()` (no `send()` / `recv()`)
- First byte discriminates message type: 0x00=infer, 0x01=register(pickle), 0x02=stop
- Infer requests use struct-packed headers + contiguous `tobytes()` data
- Register requests wrap existing pickle dict in `b'\x01' + pickle.dumps(dict)`
- The `forward()` method's return type and values are IDENTICAL to the current implementation

**Step 3**: Write a unit test that:
- Creates a Pipe pair
- Sends a request through the new binary protocol
- Receives and parses the response
- Asserts all output tensors match the pickle-based path within float32 tolerance

**Step 4**: Run the 6-worker sweep config. Compare learner steps/s against 6.66 baseline.

**Step 5**: Apply Phase 2 (batch timeout) and Phase 3 (CPU pinning). Re-run sweep.

**Step 6**: Update this planning doc with results. If steps/s ≥ 9.0, we've recovered the baseline and the transport bottleneck is resolved. If not, proceed to Phase 4 (shared memory).

### Quality Gates
After all changes:
```bash
ruff check src unit_tests
pyright src unit_tests
pytest unit_tests -q
```

---

## Update: Transformer Batching Follow-Up

### What We Learned After Reprofiling

The transport plan above was written when the fresh Transformer backend comparison had already revealed a separate unfairness in the Rust path: Transformer requests were not being batched at all in either the Rust sync player or the central inference server.

That has now been fixed in:

- [src/elitefurretai/engine/sync_battle_driver.py](src/elitefurretai/engine/sync_battle_driver.py)
- [src/elitefurretai/rl/central_inference.py](src/elitefurretai/rl/central_inference.py)

The implementation batches Transformer requests **by shared context length** instead of forcing all Transformer calls through the single-request fallback.

### Reprofiled Numbers

Rust Transformer benchmark, `12 battles`, `max_concurrent=3`, `device=cpu`:

- before batching fix:
    - `duration_seconds=53.148`
    - `battles_per_second=0.226`
    - `decisions_per_second=7.639`
    - `policy_inference_seconds=49.326`
- after batching fix:
    - `duration_seconds=51.011`
    - `battles_per_second=0.235`
    - `decisions_per_second=9.331`
    - `policy_inference_seconds=46.622`

Interpretation:

- the missing Transformer batching path was a real bug and the fix reduced model-inference time
- the gain at battle-level throughput is small because the benchmark is still dominated by model time and by battle-quality effects
- decisions/s improved more than battles/s, which is the better indicator here because the post-fix run completed more decisions in slightly less time

### Why The Gain Was Smaller Than Expected

The shared-context-length batching strategy is the correct semantics-preserving fix, but it also exposes the next limit clearly:

1. Transformer contexts diverge as concurrent battles drift to different turn counts.
2. Once context lengths diverge, batching fragments into smaller groups.
3. That means we recover the obviously missing batching, but we do **not** get ideal large-batch Transformer utilization across all live battles.

So the next Rust-side limit is now:

- **context fragmentation**, not total absence of Transformer batching

### Central GPU Inference After The Fix

Rust Transformer benchmark with central GPU inference enabled, same `12 battles`, `max_concurrent=3` shape:

- `duration_seconds=58.017`
- `battles_per_second=0.207`
- `decisions_per_second=9.635`
- `policy_inference_seconds=53.004`

Interpretation:

- restoring Transformer batching in the server was necessary, but it did **not** make the central GPU path win
- the transport overhead is still large enough that centralized GPU inference remains slower than the local CPU path for this workload shape

### Updated Bottleneck Ordering

After the Transformer batching fix, the next bottlenecks are now ordered roughly as:

1. central inference transport overhead (for the GPU-server path)
2. Transformer context-length fragmentation limiting effective batch size
3. action-quality / legality issues under higher concurrency, visible as rejected choices and truncations

Evidence from the `max_concurrent=4` CPU probe after the batching fix:

- `battles_per_second=0.158`
- `truncated_battles=5`
- `p1_rejected_choices=64`
- `p2_rejected_choices=38`

That run is a useful warning: once concurrency increases, raw inference speed stops being the only limiter and battle-quality regressions begin to dominate throughput.

### What This Means For The Transport Plan

The transport plan remains valid and high priority.

What changed is the interpretation:

- a missing Transformer batching bug was part of the original slowdown story
- fixing that bug did **not** remove the need for the transport work
- the transport work is still required if we want the central GPU inference design to outperform local CPU inference in practice
