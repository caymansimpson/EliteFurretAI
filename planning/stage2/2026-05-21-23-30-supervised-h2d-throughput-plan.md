# Supervised Training H2D Throughput — Plan

## Context

The Stage II supervised pipeline ([train.py](../../src/elitefurretai/supervised/train.py))
trains the BC/value backbone (`TransformerThreeHeadedModel`, ~26.7M params)
used as the warm-start for RL. A long-running supervised job (PID 20945 on
2026-05-21, `may22.yaml` against `data/battles/regc_final_v4/`) showed RAM
creeping over time *and*, more importantly, the GPU is massively
under-utilized. This plan addresses the throughput bottleneck. The RAM
creep is partially addressed as a side-effect (the `wandb.watch(log="all")`
fix).

## Before State

180-second diagnostic capture against PID 20945 — outputs in `~/diag/`
(2026-05-21 ~22:48–22:51):

| Signal | Value |
|---|---|
| GPU SM utilization | **avg 14%**, max 82% (over 180s, 1 Hz sampling) |
| GPU memory-bandwidth util | ~0% average |
| GPU clock (pclk) | dips to 990–1200 MHz from 1935 MHz peak — downclocking |
| VRAM used | 2,319 MiB / 24,576 MiB (~9.4%) |
| py-spy `dump` main-thread samples | 5/8 at [train.py:62](../../src/elitefurretai/supervised/train.py#L62) (the `.cuda()` call), 2/8 forward ([train.py:79](../../src/elitefurretai/supervised/train.py#L79)), 1/8 backward ([train.py:240](../../src/elitefurretai/supervised/train.py#L240)) |
| py-spy flamegraph (`train.svg`, 90,611 samples) | 285 samples in `cudaMemcpyAsync` chain on main thread; **939 samples in `multiprocessing/connection.poll/get` from DataLoader workers** (workers blocked on a *full* output queue → consumer is the bottleneck, not producers) |
| `_load_file` (zstd decode) | 790 samples — workers active <7% per worker |
| Disk (`iostat -x`) | sd[abcd] mostly 0%, occasional bursts; not disk-bound |
| `/dev/shm` file count | 57 → 37 over the window — no tmpfile leak |
| Main process RSS | climbs slowly (separate issue — wandb log="all" suspected) |

Relevant code state on `main`:

- [train.py:62](../../src/elitefurretai/supervised/train.py#L62) — `batch = {k: v.cuda(non_blocking=True) ...}`. With WSL2-mandated `pin_memory=False`, `non_blocking=True` is silently a no-op; the call is synchronous H2D.
- [train.py:357-360](../../src/elitefurretai/supervised/train.py#L357-L360) — defaults `worker_batch_size=128`, `num_workers=3`, `persistent_workers=True`, `prefetch_factor=2`.
- [train.py:366-367](../../src/elitefurretai/supervised/train.py#L366-L367) — `batch_size=512` so [train.py:467](../../src/elitefurretai/supervised/train.py#L467) computes `accumulation_steps = 512/128 = 4` (4 H2D + 4 forward/backward per optimizer step).
- [train.py:570-571](../../src/elitefurretai/supervised/train.py#L570-L571) — `torch.compile(model)` (default mode).
- [train.py:573](../../src/elitefurretai/supervised/train.py#L573) — `wandb.watch(model, log="all", log_freq=1000)` — logs every gradient and every parameter histogram. Known to cause monotonic RAM growth and is suspect #1 for the observed RAM creep.
- [train.py:587-592](../../src/elitefurretai/supervised/train.py#L587-L592) — `AdamW` constructed without `fused=True`.
- [train.py:606](../../src/elitefurretai/supervised/train.py#L606) — fp16 path: `torch.amp.GradScaler("cuda")` + implicit fp16 in `torch.amp.autocast` ([train.py:76-77](../../src/elitefurretai/supervised/train.py#L76-L77)).
- No TF32 / `set_float32_matmul_precision("high")` flags set anywhere in [train.py](../../src/elitefurretai/supervised/train.py).

## Problem

The GPU is the constrained resource (we paid for the 3090), and it is idle
~85% of the time. The proximate cause is the **synchronous H2D copy at
[train.py:62](../../src/elitefurretai/supervised/train.py#L62)** — while
the main thread is staging the next batch into VRAM, the GPU has no work
to do. The workers have already produced the next batch (they're blocked
on the queue), so this is *not* a data-loading problem.

`pin_memory=True` would be the textbook fix but is objectively broken on
this WSL2 environment (project-wide hard constraint in
[CLAUDE.md](../../CLAUDE.md)). We need to hide the H2D wait by overlapping
it with compute, plus reduce per-step overhead so the GPU does more work
per H2D.

Secondary issue: `wandb.watch(log="all")` retains gradient/parameter
histograms in the wandb client process, growing main-process RSS over the
course of a run.

## Solution

Six changes, ordered by impact × ease. Each phase produces a measurable
delta — the engineer running this plan must record batches/sec before
proceeding to the next phase, so we can attribute gains.

1. **Throughput measurement harness** — without this, the rest of the
   plan is unfalsifiable.
2. **Free flags**: TF32 + matmul precision + fused AdamW. One-line
   changes; small individual gains compound.
3. **Drop `wandb.watch(log="all")`** — addresses RAM creep, marginal
   throughput effect.
4. **Bigger effective batch (kill gradient accumulation)** — bump
   `worker_batch_size` from 128 to 512 so `accumulation_steps=1`. Four
   fewer H2D + four fewer forward/backward per optimizer step.
5. **bf16 mixed precision** — replace fp16+`GradScaler` with bf16
   autocast. Drops the scaler overhead, more numerically stable, faster
   on Ampere bf16 paths.
6. **CUDA stream prefetcher** — overlap the next batch's H2D with the
   current batch's forward/backward. The structural fix that hides the
   H2D entirely.

Optional phase 7 (defer until 1–6 are measured):

7. **bf16 storage in collate** — cast `states` to bf16 in
   `_trajectory_collate_fn` to halve the H2D payload. Touches the data
   pipeline; do last so the throughput gain is attributable.

## Reasoning

**Why focus on H2D?** Evidence is direct: 5/8 py-spy `dump` samples show
the main thread parked at the `.cuda()` line, and `cudaMemcpyAsync` is
the largest single C++ frame in the flamegraph on the main thread. GPU
SM util at 14% confirms idle time.

**Why a stream prefetcher and not just `pin_memory`?** Per project
constraint, pin_memory is off the table on WSL2. The next-best technique
is to issue the H2D copy on a side CUDA stream and have the compute
stream wait on its completion event. Modern PyTorch supports this with a
small wrapper (~30 lines).

**Why bf16 instead of fp16?** The 3090 has native bf16 matmul. Drops the
`GradScaler` complexity (which itself adds CPU overhead per step), is
more stable for the value head's MSE loss, and runs the same speed as
fp16 on Ampere.

**Why drop gradient accumulation?** Each accumulation step is a full
forward+backward+H2D. With 22 GB of VRAM free, fitting `worker_batch_size=512`
directly is almost certainly possible — and reduces 4 round-trips to 1.
If it OOMs, we step down to 256 (still 2× over the current 128).

**What about more dataloader workers?** Already-collected evidence
(939 samples of workers blocked on a full output queue) shows the workers
are over-provisioned, not under. Adding more is wasted.

**Why measure first?** The throughput-bottleneck investigation on the RL
side already burned cycles on hypotheses that were wrong (the
`_COMPILE_LOCK` story — see
[memory/feedback_killing_train_processes.md](../../../.claude/projects/-home-cayman-Repositories-EliteFurretAI/memory/MEMORY.md)).
Each phase here is one knob; if it doesn't deliver, we know immediately
and can move on without compounding noise.

## Planned Next Steps

> Each task below is bite-sized (~5 min). Verify each step before moving
> on. Commit frequently. The numeric speedup targets in parentheses are
> *expectations* — record the actual measurement in the **Updates**
> section after each phase.

### Task 1 — Throughput measurement harness

**Files:**
- Modify: [src/elitefurretai/supervised/train.py:283-295](../../src/elitefurretai/supervised/train.py#L283-L295)

**Goal:** log a `batches_per_sec` metric to wandb so every subsequent
change can be A/B-measured against the same baseline.

- [ ] **Step 1.1: Add per-window timing variables.**

After [train.py:53](../../src/elitefurretai/supervised/train.py#L53) (the
existing `start = time.time()` inside `train_epoch`), add:

```python
    window_start = time.time()
    window_batches = 0
```

- [ ] **Step 1.2: Replace the 10-batch wandb-log block with a version that includes batches/sec.**

Replace [train.py:283-295](../../src/elitefurretai/supervised/train.py#L283-L295)
with:

```python
        if num_batches % (10 * accumulation_steps) == 0:
            now = time.time()
            elapsed = now - window_start
            window_batches += 10 * accumulation_steps
            batches_per_sec = window_batches / elapsed if elapsed > 0 else 0.0
            window_start = now
            window_batches = 0
            wandb.log(
                {
                    "Total Steps": prev_steps + steps,
                    "train_loss": running_loss / num_batches,
                    "train_turn_loss": running_turn_loss / num_batches,
                    "train_teampreview_loss": running_teampreview_loss / num_batches,
                    "train_win_loss": running_win_loss / num_batches,
                    "train_brier": (brier_sum / brier_count) if brier_count > 0 else 0.0,
                    "train_entropy": running_entropy / num_batches,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batches_per_sec": batches_per_sec,
                }
            )
```

- [ ] **Step 1.3: Run lint + type gates.**

```bash
source ../venv/bin/activate && ruff check src/elitefurretai/supervised/train.py && pyright src/elitefurretai/supervised/train.py
```

Expected: no errors.

- [ ] **Step 1.4: Capture baseline.**

Start a short training run (10 minutes is enough). Note steady-state
`batches_per_sec` value from wandb (skip the first 200 batches —
those include `torch.compile` warmup). Record this baseline in the
**Updates** section below as `phase0_baseline_batches_per_sec`.

- [ ] **Step 1.5: Commit.**

```bash
git add src/elitefurretai/supervised/train.py
git commit -m "supervised: log batches_per_sec for throughput tracking"
```

### Task 2 — Free flags (TF32, matmul precision, fused AdamW)

**Files:**
- Modify: [src/elitefurretai/supervised/train.py:345-349](../../src/elitefurretai/supervised/train.py#L345-L349) (in `initialize`)
- Modify: [src/elitefurretai/supervised/train.py:587-592](../../src/elitefurretai/supervised/train.py#L587-L592) (AdamW constructor)

**Goal (~1.1–1.3× cumulative):** turn on TF32 matmuls (free win on Ampere
for the MLP-heavy backbone) and use the fused AdamW kernel.

- [ ] **Step 2.1: Add TF32 + matmul precision setup in `initialize`.**

Modify [train.py:345-349](../../src/elitefurretai/supervised/train.py#L345-L349)
to add three lines below the existing seed setup. The block becomes:

```python
    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(config["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    random.seed(int(config["seed"]))
```

- [ ] **Step 2.2: Enable fused AdamW.**

Modify [train.py:587-592](../../src/elitefurretai/supervised/train.py#L587-L592):

```python
    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
        fused=(config["device"] == "cuda"),
    )
```

- [ ] **Step 2.3: Run lint + type gates.**

```bash
source ../venv/bin/activate && ruff check src/elitefurretai/supervised/train.py && pyright src/elitefurretai/supervised/train.py
```

Expected: no errors.

- [ ] **Step 2.4: Re-run training, capture `batches_per_sec`.**

Run for 10 min, record steady-state value. Should be 1.1–1.3× the
phase-0 baseline.

- [ ] **Step 2.5: Commit.**

```bash
git add src/elitefurretai/supervised/train.py
git commit -m "supervised: enable TF32 + fused AdamW for Ampere throughput"
```

### Task 3 — Drop `wandb.watch(log="all")`

**Files:**
- Modify: [src/elitefurretai/supervised/train.py:573](../../src/elitefurretai/supervised/train.py#L573)

**Goal:** stop the wandb client from holding per-step gradient/parameter
histograms. Addresses RAM creep observed during long runs.

- [ ] **Step 3.1: Replace with a lighter setting.**

Modify [train.py:573](../../src/elitefurretai/supervised/train.py#L573)
from:

```python
    wandb.watch(model, log="all", log_freq=1000)
```

to:

```python
    wandb.watch(model, log=None)
```

- [ ] **Step 3.2: Run lint + type gates.**

```bash
source ../venv/bin/activate && ruff check src/elitefurretai/supervised/train.py && pyright src/elitefurretai/supervised/train.py
```

Expected: no errors.

- [ ] **Step 3.3: Verify RAM is no longer climbing.**

Start a training run and watch main-process RSS over 30 minutes. Should
be ~flat after the first few minutes (one cache-warm-up jump is fine).

```bash
( for i in $(seq 1 60); do date +%s; ps -o pid,rss,cmd -p <NEW_PID>; sleep 30; done ) > ~/diag/rss_phase3.log
```

- [ ] **Step 3.4: Commit.**

```bash
git add src/elitefurretai/supervised/train.py
git commit -m "supervised: drop wandb.watch(log=all) to stop RAM creep"
```

### Task 4 — Kill gradient accumulation (worker_batch_size 128 → 512)

**Files:**
- Modify: training config (`src/elitefurretai/supervised/configs/may22.yaml` —
  or whichever config the user is running). Default in
  [train.py:356-357](../../src/elitefurretai/supervised/train.py#L356-L357).

**Goal (~1.3–1.5×):** one optimizer step per batch instead of four.

- [ ] **Step 4.1: Check that the chosen config sets `batch_size` and `worker_batch_size`.**

```bash
grep -n "worker_batch_size\|batch_size" src/elitefurretai/supervised/configs/may22.yaml
```

If `worker_batch_size` is set in the config, update it there. If not,
the default in [train.py:356](../../src/elitefurretai/supervised/train.py#L356)
applies.

- [ ] **Step 4.2: Update `worker_batch_size` to 512.**

Edit the config (or [train.py:356](../../src/elitefurretai/supervised/train.py#L356))
to set `worker_batch_size: 512`. With `batch_size: 512`, this makes
`accumulation_steps = 1`.

- [ ] **Step 4.3: Run training, watch VRAM peak.**

```bash
nvidia-smi --query-gpu=memory.used --format=csv -lms 500 -i 0 -f ~/diag/vram_phase4.csv &
# (then start training)
```

Expected peak: well under 24 GB. If OOM occurs, fall back to
`worker_batch_size: 256` (still 2× the original) and re-test.

- [ ] **Step 4.4: Capture `batches_per_sec`.**

Record. Should be 1.3–1.5× the post-Task-2 value.

- [ ] **Step 4.5: Commit.**

```bash
git add src/elitefurretai/supervised/configs/may22.yaml  # or train.py
git commit -m "supervised: drop gradient accumulation (worker_batch_size 128 -> 512)"
```

### Task 5 — bf16 mixed precision (replace fp16 + GradScaler)

**Files:**
- Modify: [src/elitefurretai/supervised/train.py:76-77](../../src/elitefurretai/supervised/train.py#L76-L77) (autocast call site)
- Modify: [src/elitefurretai/supervised/train.py:236-265](../../src/elitefurretai/supervised/train.py#L236-L265) (backward+step block)
- Modify: [src/elitefurretai/supervised/train.py:606](../../src/elitefurretai/supervised/train.py#L606) (scaler creation)
- Modify: [src/elitefurretai/supervised/train.py:614](../../src/elitefurretai/supervised/train.py#L614) (train_epoch call)

**Goal (~1.2–1.4×):** drop the `GradScaler` step entirely (bf16 doesn't
need it) and let the compute kernels run on Ampere bf16 paths.

- [ ] **Step 5.1: Change the autocast block to bf16.**

Modify [train.py:76-77](../../src/elitefurretai/supervised/train.py#L76-L77)
from:

```python
        autocast = torch.amp.autocast if config["device"] == "cuda" else torch.autocast  # type: ignore
        with autocast(config["device"]):
```

to:

```python
        with torch.amp.autocast(
            device_type=config["device"],
            dtype=torch.bfloat16 if config["device"] == "cuda" else torch.float32,
            enabled=(config["device"] == "cuda"),
        ):
```

- [ ] **Step 5.2: Remove the `GradScaler` instance.**

Delete [train.py:606](../../src/elitefurretai/supervised/train.py#L606)
entirely.

- [ ] **Step 5.3: Simplify the backward/step block.**

Replace [train.py:236-265](../../src/elitefurretai/supervised/train.py#L236-L265)
with the scaler-free version (collapsing the `if scaler is not None` /
`else` branches into the else-branch behavior):

```python
        # Backpropagation (bf16 autocast — no scaler needed)
        if loss.requires_grad:
            loss.backward()
        else:
            # No valid samples in batch — nothing to backprop
            continue

        accumulation_counter += 1

        # Only update weights every accumulation_steps
        if accumulation_counter >= accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["max_grad_norm"]
            )
            optimizer.step()

            optimizer.zero_grad()
            accumulation_counter = 0
```

- [ ] **Step 5.4: Update the `train_epoch` call signature.**

Modify [train.py:614](../../src/elitefurretai/supervised/train.py#L614)
to drop the `scaler` argument:

```python
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config)
```

And remove the `scaler=None` parameter from `train_epoch`'s signature
at [train.py:40](../../src/elitefurretai/supervised/train.py#L40).

- [ ] **Step 5.5: Run lint + type gates.**

```bash
source ../venv/bin/activate && ruff check src/elitefurretai/supervised/train.py && pyright src/elitefurretai/supervised/train.py
```

Expected: no errors.

- [ ] **Step 5.6: Sanity-check loss curve.**

Start a 5-minute training run. Loss should look qualitatively similar to
the phase-4 run (small differences in the 3rd decimal are OK; large
divergence — NaN, exploding — is a regression).

- [ ] **Step 5.7: Capture `batches_per_sec`.**

Should be 1.2–1.4× the post-Task-4 value.

- [ ] **Step 5.8: Commit.**

```bash
git add src/elitefurretai/supervised/train.py
git commit -m "supervised: switch fp16+GradScaler to bf16 autocast"
```

### Task 6 — CUDA stream prefetcher (the structural fix)

**Files:**
- Create: `src/elitefurretai/etl/cuda_prefetcher.py`
- Modify: [src/elitefurretai/supervised/train.py:59-64](../../src/elitefurretai/supervised/train.py#L59-L64) (the `for batch in dataloader` loop)
- Create: `unit_tests/etl/test_cuda_prefetcher.py`

**Goal (~1.5–2×):** overlap the next batch's H2D with the current
batch's forward+backward. Eliminates the visible H2D wait on the main
thread (which the diagnostic showed taking 60%+ of main-thread time).

- [ ] **Step 6.1: Write the failing unit test.**

Create `unit_tests/etl/test_cuda_prefetcher.py`:

```python
import pytest
import torch

from elitefurretai.etl.cuda_prefetcher import CudaStreamPrefetcher


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_yields_same_batches_in_order():
    batches = [
        {"x": torch.arange(4, dtype=torch.float32).view(2, 2) + i}
        for i in range(5)
    ]
    prefetcher = CudaStreamPrefetcher(iter(batches), device="cuda")
    out = list(prefetcher)
    assert len(out) == 5
    for i, batch in enumerate(out):
        assert batch["x"].device.type == "cuda"
        expected = torch.arange(4, dtype=torch.float32).view(2, 2) + i
        assert torch.allclose(batch["x"].cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_handles_empty_iterable():
    prefetcher = CudaStreamPrefetcher(iter([]), device="cuda")
    assert list(prefetcher) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefetcher_preserves_keys():
    batch = {"a": torch.zeros(2), "b": torch.ones(3), "c": torch.full((1,), 7.0)}
    prefetcher = CudaStreamPrefetcher(iter([batch]), device="cuda")
    out = next(iter(prefetcher))
    assert set(out.keys()) == {"a", "b", "c"}
```

- [ ] **Step 6.2: Run the test, confirm it fails.**

```bash
source ../venv/bin/activate && pytest unit_tests/etl/test_cuda_prefetcher.py -v
```

Expected: ModuleNotFoundError (`cuda_prefetcher` does not yet exist).

- [ ] **Step 6.3: Implement `CudaStreamPrefetcher`.**

Create `src/elitefurretai/etl/cuda_prefetcher.py`:

```python
"""
Overlap host-to-device tensor transfer with GPU compute.

The main-thread .cuda() call in supervised training is synchronous (we
cannot use pin_memory=True on WSL2). This wrapper issues each batch's
H2D copy on a side CUDA stream so the compute stream can run the
previous batch's forward/backward concurrently.

Usage:
    for batch in CudaStreamPrefetcher(dataloader, device="cuda"):
        # batch tensors are already on `device`; H2D for batch N+1 is
        # already in flight on the side stream
        ...
"""

from typing import Dict, Iterable, Iterator, Optional

import torch


class CudaStreamPrefetcher:
    def __init__(
        self,
        loader: Iterable[Dict[str, torch.Tensor]],
        device: str = "cuda",
    ):
        self._loader = loader
        self._device = torch.device(device)
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        it = iter(self._loader)

        if self._stream is None:
            # Non-CUDA fallback: just move tensors to the device.
            for batch in it:
                yield {k: v.to(self._device) for k, v in batch.items()}
            return

        next_batch = self._prefetch(it)
        while next_batch is not None:
            # Make the compute stream wait until the H2D finishes for
            # the batch we are about to hand out.
            torch.cuda.current_stream(self._device).wait_stream(self._stream)
            current = next_batch
            next_batch = self._prefetch(it)
            yield current

    def _prefetch(
        self, it: Iterator[Dict[str, torch.Tensor]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            host = next(it)
        except StopIteration:
            return None
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            return {k: v.to(self._device, non_blocking=True) for k, v in host.items()}
```

- [ ] **Step 6.4: Run the test, confirm it passes.**

```bash
source ../venv/bin/activate && pytest unit_tests/etl/test_cuda_prefetcher.py -v
```

Expected: 3 passed.

- [ ] **Step 6.5: Integrate into `train_epoch`.**

Modify the top of the `for batch in dataloader:` loop at
[train.py:59-64](../../src/elitefurretai/supervised/train.py#L59-L64).
Change:

```python
    for batch in dataloader:
        # Transfer data to the right device
        if config["device"] == "cuda":
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
```

to:

```python
    from elitefurretai.etl.cuda_prefetcher import CudaStreamPrefetcher

    iterator = (
        CudaStreamPrefetcher(dataloader, device=config["device"])
        if config["device"] == "cuda"
        else dataloader
    )

    for batch in iterator:
        if config["device"] != "cuda":
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
```

Move the import to the top of the file (alongside the other
`from elitefurretai.etl ...` imports near
[train.py:18-22](../../src/elitefurretai/supervised/train.py#L18-L22)).

- [ ] **Step 6.6: Do the same in `evaluate`.**

The validation/test loaders go through `evaluate()` in
[src/elitefurretai/supervised/utils.py](../../src/elitefurretai/supervised/utils.py).
Check whether `evaluate` also does `.cuda()` per batch — if so, wrap
its dataloader too. Find with:

```bash
grep -n "\.cuda\|\.to(" src/elitefurretai/supervised/utils.py
```

If a wrap is needed, mirror the change from Step 6.5.

- [ ] **Step 6.7: Run lint + type gates + tests.**

```bash
source ../venv/bin/activate && \
    ruff check src/elitefurretai/etl/cuda_prefetcher.py src/elitefurretai/supervised/train.py && \
    pyright src/elitefurretai/etl/cuda_prefetcher.py src/elitefurretai/supervised/train.py && \
    pytest unit_tests/etl/test_cuda_prefetcher.py -v
```

Expected: no lint/type errors, 3 tests pass.

- [ ] **Step 6.8: Run training, capture `batches_per_sec`.**

Should be 1.5–2× the post-Task-5 value. Watch the GPU SM utilization in
parallel:

```bash
nvidia-smi dmon -s u -d 1 -c 120 -o T > ~/diag/gpu_phase6.log
```

Expected: SM util now sustained > 70% (was 14% avg in the baseline).

- [ ] **Step 6.9: Commit.**

```bash
git add src/elitefurretai/etl/cuda_prefetcher.py unit_tests/etl/test_cuda_prefetcher.py src/elitefurretai/supervised/train.py
# also add src/elitefurretai/supervised/utils.py if Step 6.6 modified it
git commit -m "supervised: overlap H2D with compute via CUDA stream prefetcher"
```

### Task 7 — Update SUPERVISED.md

**Files:**
- Modify: [src/elitefurretai/supervised/SUPERVISED.md](../../src/elitefurretai/supervised/SUPERVISED.md)

**Goal:** document the new prefetcher and the bf16/TF32 baseline so the
next person doesn't undo it.

- [ ] **Step 7.1: Add a "Throughput" section to SUPERVISED.md.**

Add the new prefetcher to the architecture description, note that bf16
+ TF32 + fused AdamW are now the defaults, and link to this plan as
the rationale.

- [ ] **Step 7.2: Commit.**

```bash
git add src/elitefurretai/supervised/SUPERVISED.md
git commit -m "docs: document supervised throughput baseline (bf16 + prefetcher)"
```

### Task 8 (optional) — bf16 states in collate

**Files:**
- Modify: [src/elitefurretai/etl/battle_dataloader.py:15-40](../../src/elitefurretai/etl/battle_dataloader.py#L15-L40)
- Modify: [src/elitefurretai/supervised/train.py:66](../../src/elitefurretai/supervised/train.py#L66)

**Goal:** halve the per-batch H2D payload by sending bf16 states across
the wire instead of fp32. Only do this if Tasks 1–6 didn't hit the
2–4× target.

- [ ] **Step 8.1: Decide whether to do this phase.**

If the post-Task-6 `batches_per_sec` is ≥ 2× the phase-0 baseline,
**stop here.** Phase 8 increases preprocessing complexity and should
only be done if needed.

- [ ] **Step 8.2: Cast `states` to bf16 in `_trajectory_collate_fn`.**

In [battle_dataloader.py:33-39](../../src/elitefurretai/etl/battle_dataloader.py#L33-L39),
inside the per-key loop, special-case the `"states"` key:

```python
    for key in keys:
        tensors = [item[key] for item in batch]
        stacked = torch.stack(tensors, dim=0)
        if key == "states":
            stacked = stacked.to(torch.bfloat16)
        result[key] = stacked.contiguous()
```

- [ ] **Step 8.3: Update the dtype cast on the consumer side.**

[train.py:66](../../src/elitefurretai/supervised/train.py#L66) currently
does `states = batch["states"].to(torch.float32)`. Under bf16 autocast
this is wasteful; switch to:

```python
        states = batch["states"]
```

(autocast will handle dtype promotion where needed).

- [ ] **Step 8.4: Run lint + type gates + tests.**

```bash
source ../venv/bin/activate && \
    ruff check src/elitefurretai/etl/battle_dataloader.py src/elitefurretai/supervised/train.py && \
    pyright src/elitefurretai/etl/battle_dataloader.py src/elitefurretai/supervised/train.py && \
    pytest unit_tests/etl -q
```

- [ ] **Step 8.5: Sanity-check the loss curve.**

Make sure loss looks qualitatively similar to phase-6.

- [ ] **Step 8.6: Capture `batches_per_sec` and commit.**

```bash
git add src/elitefurretai/etl/battle_dataloader.py src/elitefurretai/supervised/train.py
git commit -m "supervised: bf16 states in collate to halve H2D payload"
```

### Final verification

- [ ] **Step F.1: Compare phase-0 baseline to final `batches_per_sec`.**

Record the multiplier in the **Updates** section. Target was 2–4×.

- [ ] **Step F.2: Run full test suite to confirm no regressions.**

```bash
source ../venv/bin/activate && pytest unit_tests -q
```

- [ ] **Step F.3: Long-run sanity check.**

Run a full epoch of training. Confirm:
- Final test loss is within ~1% of the phase-0 baseline (small
  differences from bf16/TF32 are acceptable).
- Main-process RSS is flat (RAM creep fix confirmed).
- GPU SM util is sustained > 70%.

## Updates

*(Each task should append a brief note with the measured value.)*

- **Phase 0 baseline (2026-05-21):** 10m 7s = **607s per full epoch** (13,191,018 samples). A/B metric for subsequent phases = full-epoch wall-clock on the same dataset (`data/battles/regc_final_v4/`).
- Post-Task 2 epoch time: TBD
- Post-Task 3 RSS slope: TBD
- Post-Task 4 epoch time: TBD
- Post-Task 5 epoch time: TBD
- Post-Task 6 epoch time and GPU SM util: TBD
- Final speedup multiplier vs phase 0: TBD
