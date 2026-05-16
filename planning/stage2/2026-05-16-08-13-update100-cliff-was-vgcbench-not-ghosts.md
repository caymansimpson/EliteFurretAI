# The "update-100 cliff" was vgcbench, not ghost activation

**Date**: 2026-05-16 08:13
**Trigger**: avid-star-62 (may15.yaml, instrumented) reproduced the same
"throughput collapses at update 100" pattern as firm-field-41 and
radiant-violet-57. Initial diagnosis pinned it on ghost-subprocess
activation. A fast-iteration follow-up run with `portfolio_add_interval`
+ `checkpoint_interval` dropped to 10 (so the cliff event fires at
update 10 instead of 100) **falsified that diagnosis** and identified
broken vgcbench as the actual cause.

## Context

Three consecutive may15.yaml / sep_arch.yaml runs showed the same shape:

| Run | Pre-100 window | Post-100 window | Drop |
|---|---|---|---|
| firm-field-41 | ~8 b/s | (collapsed; loss-vs-capability diverged) | n/a |
| radiant-violet-57 | (similar) | crashed at update 100 with CUDA tensor error | n/a |
| **avid-star-62** | **8.33 b/s** | **0.51 b/s** | **16×** |

The CUDA-tensor fix landed earlier (ship CPU shadow's state_dict over
mp.Queue, not CUDA tensors). avid-star-62 confirmed the crash was
gone, but the *slowdown* survived.

## Before State

### Initial (wrong) attribution

The slowdown coincided with `[Update 100] Adding new reference to
portfolio` + `[Update 100] Saving checkpoint and updating curriculum`,
which together do five things at once:

1. Deep-copy main model into a new portfolio reference (extra ref
   forward pass per learner update from this point on).
2. Save checkpoint to disk.
3. Add the snapshot to `opponent_pool` as ghost_0.
4. `registry.sync_weights("ghost_0", ...)` — ships the new weights to
   the frozen inference subprocess.
5. `registry.sync_weights("main", cpu_weights)` — syncs the trainer's
   in-process inference service.
6. Broadcast weights + curriculum + sampling knobs to workers.

The natural hypothesis: ghost_0 going live for the first time floods
the frozen subprocess with inference traffic it wasn't seeing before
(10% of new battles route through it via curriculum `ghosts: 0.1`).

### Block-level instrumentation said "block is fine"

Adding `time.perf_counter()` timing in [model_registry.sync_weights](src/elitefurretai/rl/model_registry.py),
[inference_subprocess SyncWeightsMsg handler](src/elitefurretai/rl/inference_subprocess.py),
[learner ref_models forward loop](src/elitefurretai/rl/learners.py),
and the [train.py portfolio-add block](src/elitefurretai/rl/train.py)
gave clean numbers at update 100 of avid-star-62:

| Operation | Time |
|---|---|
| add_reference_model (deepcopy) | 82.8 ms |
| save_checkpoint | 588.4 ms |
| sync_weights[ghost_0→frozen] | 55.3 ms total (load 52.8 + get_state_dict 2.2 + enqueue 0.4) |
| main→cpu state_dict | 74.2 ms |
| sync_weights[main] in-process | 55.7 ms |
| worker queue broadcast (4 workers) | 3.7 ms |
| **Block total** | **1191.3 ms** |

So the cliff event itself is a one-time ~1.2 s hiccup. The *sustained*
16× throughput drop afterward had to be in steady-state behavior, not
in this block.

## Problem

What was the actual driver of the sustained post-update-100 throughput
floor at ~0.5 b/s?

## Solution

### Fast-iteration config

Created [src/elitefurretai/rl/configs/may15_fast.yaml](src/elitefurretai/rl/configs/may15_fast.yaml)
identical to may15.yaml except:

- `portfolio.portfolio_add_interval`: 100 → 10
- `training.checkpoint_interval`: 100 → 10
- (later) `vgc_bench_baseline`: 0.2 → 0 (curriculum weight redistributed to self_play to free RAM)

This made the cliff event reproduce at update 10 (~4 min run-in)
instead of update 100 (~30 min), enabling tight diagnostic loops.

### Orthogonal vgcbench bug

While inspecting why `win_rate_vgc_bench_baseline` was 0 in avid-star-62,
found that workers were challenging the bare username `VGCBENCH` while
the external runners had been launched with port-suffixed names
(`VGCBENCH_8000`, `VGCBENCH_8001`, ...).

Root cause: [vgc_environment.py:setup()](src/elitefurretai/engine/vgc_environment.py)
gated the port-suffix derivation on `cur.auto_launch_external_vgcbench`,
but [train.py:1085-1099](src/elitefurretai/rl/train.py) had previously
been switched to launch runners based on `curriculum_weights[VGC_BENCH_BASELINE] > 0`
(not the legacy flag). may15.yaml has `auto_launch_external_vgcbench: false`
("legacy; no longer consulted by train.py") — so the launcher launched
suffixed names but the worker kept the bare name. Every challenge
yielded an instant `|popup|The user 'VGCBENCH' was not found.`

**Fix**: replace the `cur.auto_launch_external_vgcbench` check with a
curriculum-weight check, mirroring train.py's launcher logic exactly.

### Worker-side amplification

Why did broken vgcbench produce a 16× throughput floor (not just a
"vgcbench battles fail" warning)? Looked at [worker.py:482-494](src/elitefurretai/rl/worker.py):

```python
if (
    not vgcbench_disabled_locally
    and external_vgcbench_usernames
    and consecutive_vgcbench_timeouts >= 2
):
    updated_curriculum[OpponentPool.VGC_BENCH_BASELINE] = 0.0
    env.update_curriculum(updated_curriculum)
    vgcbench_disabled_locally = True
```

The disable-locally guard only trips on **timeouts**, not on instant
"user not found" popups. So the worker:

1. Picks vgc_bench_baseline (20% of every batch by curriculum weight).
2. Sends challenge to bare `VGCBENCH`.
3. Server replies instantly with "user not found" popup.
4. Worker doesn't count it as a timeout — retries next batch.

Each cycle wastes a battle slot. Across 4 workers × 48 concurrent
battles, ~20% of slot capacity is permanently bottled up in dead
challenges. **Until ghost_0 also goes live at update 100**, the
remaining slots are fast enough to maintain ~6-8 b/s. The moment ghost
inference adds latency to the remaining slots, the system saturates and
throughput collapses to ~0.5 b/s.

### Validation

Re-ran may15_fast.yaml with the vgcbench fix applied AND vgcbench
disabled in curriculum (to also avoid a separate RAM-pressure issue —
working vgcbench runners now actually load SB3 models and consume
~4.88 GB combined, which trips the 24 GB watchdog on WSL2).

Result:

| Update | Window b/s | Notes |
|---|---|---|
| 9 | 7.97 | Peak baseline |
| 10 | 7.13 | Portfolio-add block fires (84.5 ms ref-add + 617 ms save + 57+75+55 ms syncs) |
| 11 | **3.59** | One-update transient (~2× dip) |
| 12 | **8.26** | Fully recovered |
| 13 | 7.92 | Sustained |
| 14 | 9.03 | Sustained (above pre-cliff peak) |
| 15 | 7.81 | Sustained |
| 20 | 8.56 | Pre-update-20 block |
| 25 | 8.69 | Sustained through 2 cliffs |
| 30, 40, 50, 60, 70, 80 | block fires each time | No throughput collapse |

`ref_fwd[last 50 updates]: portfolio_size=5 avg_total=400.4ms avg_refs=3.00 per_ref=133.5ms`
at update 50 — per-ref forward cost stayed roughly flat (~133-160 ms)
through portfolio growth from 1 to 5. The portfolio cost grew linearly
with size as expected.

**Conclusion**: With vgcbench working correctly, ghost activation
causes only a one-update transient that immediately recovers.

## Reasoning

**Why the wrong hypothesis was tempting**: the block timings clustered
exactly at the moment of throughput collapse, and the new code path
(frozen subprocess inference via mp.Queue) is the most architecturally
complex thing that activates at update 100. The fix was always one
layer below — a popup-vs-timeout distinction in worker.py that had been
working fine until vgcbench broke for an unrelated reason.

**Why fast-iteration mattered**: at portfolio_add_interval=100,
verifying a hypothesis takes ~30 minutes per run. Dropping it to 10
made each verification ~3-4 minutes, and we were able to test "what if
vgcbench off" decisively in a single iteration. The instrumentation
already in place from the slow-iteration diagnosis kept working at the
faster cadence — no code changes needed beyond config.

**Why the diagnostic still produced value**: even though the original
hypothesis was wrong, the block-timing instrumentation we added
(`sync_weights[…]`, `synced weights for`, `ref_fwd[last N]`,
`[Update N] add_reference_model/save_checkpoint/worker queue`)
remains useful telemetry going forward. Keeping it in the production
code path.

## Planned Next Steps

1. **Commit the bundle**: vgc_environment.py vgcbench fix + may15_fast.yaml +
   the instrumentation across model_registry.py, inference_subprocess.py,
   learners.py, train.py + showdown_server_manager.py launcher arg
   `--no-battle-retention` + this planning doc.
2. **Harden worker.py** (follow-up): make `consecutive_vgcbench_failures`
   bump on *any* failure (popup or timeout), not just timeouts. Defense
   in depth — even if the username derivation drifts again, the local
   disable will trip after 2 dead challenges instead of running forever.
3. **Relaunch may15.yaml** for real training (portfolio_add_interval=100,
   vgc_bench_baseline=0.2, with both fixes in place). With the broken-
   vgcbench amplifier gone, expectation is multi-hour healthy training
   that capitalizes on the showdown-fork battle-drop patch as well.
4. **Memory profiling** (lower priority): the "other=7.19 GB" in the
   watchdog breakdown is most likely the frozen inference subprocess
   (11 model services on CUDA). If we want headroom for a future
   exploiter pipeline on the same box, characterize that footprint.

## Updates

*(none yet — will be filled in after relaunch)*
