# Throughput Correction and PokeJAX-Paper Context

## Context

While discussing how to maximize Stage II training speed on the workstation, two
prior claims I made in conversation were either wrong or dated. This doc
corrects them against the existing planning record and the live system, and
frames the relevance of Karten et al., "Automatic Generation of High-Performance
RL Environments" ([arXiv:2603.12145](https://arxiv.org/abs/2603.12145)) for the
project's medium-term throughput strategy.

## Before State (what I said in chat)

1. "Switching to the in-process Rust backend would give 5–10× per-machine
   throughput." — wrong direction on this hardware.
2. "The single-machine bottleneck is the Pokémon Showdown JS engine + websocket
   roundtrip." — partially wrong; the dominant cost is actually the actor-side
   CPU transformer plus asyncio polling, with websocket-related work being a
   third-tier contributor.
3. "Moving actor inference to GPU on this box would not help." — *this one is
   correct for our specific architecture*, and is worth carefully separating
   from the paper's findings, which look like they say the opposite at a glance.

## Problem

Without correcting the record, the next round of throughput decisions will
target the wrong levers (e.g., reviving the Rust path or chasing a centralized
GPU inference design) instead of the actually-leveraged ones (legality-fix
churn, smaller / quantized actor, more cores via cloud, or — at the extreme — a
full env port).

## Solution

### Correction 1: Rust ≠ faster on this machine

[planning/stage2/2026-04-13-23-20-rl-training-bottleneck-profile.md](2026-04-13-23-20-rl-training-bottleneck-profile.md)
ran paired 10-update training comparisons:

| Backend | Wall clock for 10 updates | b/s | learner steps / battle |
|---|---|---|---|
| Rust | 35m 23s | 0.30 | 29.7 |
| Showdown websocket | 13m 27s | 0.79 | 10.1 |

Showdown is **2.63× faster wall-clock**. Rust runs ~3× more inference calls per
battle, and since actor inference is the dominant cost on this CPU, Rust
*ends up slower* end-to-end despite the faster simulator. The "Rust is faster"
intuition only holds if the actor is cheap. It currently isn't.

**Implication for medium-term plans:** the Rust backend is still strategically
valuable (it's the path to in-process vectorization later — see Correction 3),
but it is *not* the right next throughput lever on this workstation.

### Correction 2: where wall-clock actually goes

[planning/stage2/2026-04-13-23-55-showdown-training-profile-and-speed-plan.md](2026-04-13-23-55-showdown-training-profile-and-speed-plan.md)
profiled the Showdown backend on this machine. Top hotspots:

| Hotspot | Time | Share |
|---|---|---|
| `selectors.select` (asyncio + websocket polling) | 23 s | ~33% |
| `RLTrajectoryPlayer._gpu_inference_sync` (actor forward) | 22 s | ~31% |
| `torch._C._nn.linear` (matmul inside the forward) | 7.6 s | ~11% |
| `poke_env.player.Player._handle_battle_message` | 5.7 s | ~8% |
| `RLTrajectoryPlayer._embed_battle_state` | 4.7 s | ~7% |
| `Embedder.embed_to_array` | 4.6 s | ~7% |
| `calculate_damage` | 3.3 s | ~5% |

Two things follow from this:

- **The 33% `selectors.select` share is the asyncio event loop blocking on the
  next websocket message.** That is *not* "Showdown is slow"; it is "we are
  waiting for the next request to arrive before we have anything for the actor
  to do." Websocket bandwidth itself is not saturated.
- **Actor inference is the second dominant cost (31% incl. matmul).** Even at
  zero inference cost the wall-clock ceiling is `1 / (1 − 0.31) ≈ 1.45×`, *not*
  the orders of magnitude I claimed earlier.

The combined realistic per-machine improvement budget on the i7-7700K is
**roughly 1.5–2×**, stacking legality fixes + cheaper actor + minor
embedding/feature work. This is a *single-knob ceiling*, not the whole story
(see Correction 3).

### Live resource verification (2026-05-06 00:27)

Sampled while easy_test.yaml is running with K3 topology
(6 workers / 18 players / 6 servers, ppo_epochs=3):

```
=== CPU per-core (5-second mpstat sample) ===
all cores:  92–94 % busy   ~5 % idle
load avg:   10.78  (1.35× oversubscribed on 8 logical CPUs)

Top procs by %CPU:
  6 worker pythons   106–113 % each   ← multi-threaded, > 1 logical core each
  6 vgcbench runners 4–7 % each
  Showdown node servers ~3–5 % each + helpers
  Trainer process    ~50 %

Memory: 19/23 GiB used, 567 MiB swap engaged, 3.5 GiB available
```

**Conclusion: the box is fully CPU-bound and starting to feel memory pressure.**
Adding workers will hurt, not help. Per-machine parallelism is exhausted.

### Correction 3: PokeJAX paper says the opposite — but addresses a different system

Karten, Appapogu, Jin (arXiv:2603.12145, "Automatic Generation of
High-Performance RL Environments") report:

- **PokeJAX**: 22,320× speedup over TypeScript reference, 500M steps/sec random
  actions.
- **EmuRust**: 1.5× PPO speedup over TypeScript baseline.
- "At 200M parameters, the environment overhead drops below 4 % of training
  time."

These numbers are real and large, but the regime is different from ours:

- They run **in-process JAX simulators** that vectorize thousands of envs per
  GPU step. Their "step" is a fused GPU kernel; effective batch size is
  thousands.
- Our current setup is **out-of-process Showdown via websocket**, batch size 8
  per worker. Inference latency is amortized over a much smaller batch and is
  chained behind a slow async loop.
- In their regime, the policy *does* live on the GPU profitably — because the
  env is on the same device, the batch is large, and there is no
  cross-device transport per step.
- In our regime, the
  [2026-04-13-23-55 doc](2026-04-13-23-55-showdown-training-profile-and-speed-plan.md)
  measured **CPU actors 0.424 b/s vs CUDA actors 0.388 b/s** on this box —
  CUDA was *slightly slower* because per-batch transport cost exceeded the
  matmul savings. That measurement is dated (it used the older 125M-param
  model; the current cool-bee-85-finetune is ~27M), but the underlying reason —
  selectors.select dominates, batch is small, learner contends for the same
  GPU — has not changed.

So both statements are simultaneously true:

> "Move the actor to GPU" gives 22,000× when the env is on the GPU too.
> "Move the actor to GPU" gives ~0× (or negative) when the env is on the
> network and batch sizes are small.

The paper's path *is* what would give us order-of-magnitude gains: port the
Pokémon engine itself to a JAX/Rust in-process vectorized simulator. PokeJAX
already exists. We have not validated whether its scope covers
`gen9vgc2024regg` (Tera, Commander, open-team-sheets, etc.). That validation is
the only path I see to closing the multi-OOM gap to AlphaStar-class throughput
*without* renting a fleet.

## Reasoning

This rewrite of the throughput picture matters because the project's compute
budget is finite and the wrong lever costs days. The corrected order of merit
on *this hardware* is:

1. Legality-churn reduction (the fuzz harness landed 2026-05-05 is the right
   tool; estimated 1.3–1.6× from removing PS_ERROR retries).
2. Cheaper actor per call: quantize the 27M model fp16/int8 on CPU, or distill
   to ~8M (estimated 1.1–1.3×).
3. Embedding / damage-calc micro-optimizations (1.1–1.2×).
4. Move to a 16- or 32-core cloud box; per-worker actor cost is the cap,
   adding workers is linear up to roughly 16 (then asyncio re-caps).

And the corrected order of merit *across hardware*:

1. Validate PokeJAX coverage for `gen9vgc2024regg` (paper-supported path, 100×+
   if scope is sufficient).
2. Distributed Showdown training across many cheap CPU boxes (linear in
   $$).
3. In-house JAX/Rust env port if PokeJAX coverage is insufficient.

Avoided wrong levers: re-prioritizing the existing Rust backend; centralized GPU
inference on the same 3090 as the learner.

## Live throughput measurements — current run

Config: `easy_test.yaml` with K3 topology, ppo_epochs=3, on the i7-7700K + 3090.

| Update | b/s | learner steps/s | bc_player | max_damage | simple_heur | vgc_bench | ghosts |
|--------|-----|-----------------|-----------|------------|-------------|-----------|--------|
| 1 | 2.05 | 34.38 | 55.6% | 46.7% | 14.3% | 4.8% | — |
| 5 | 2.52 | 43.80 | 51.0% | 36.0% | 21.0% | 4.0% | — |
| 10 | 2.64 | 45.82 | 46.0% | 31.0% | 22.0% | 3.0% | — |
| 25 | 2.62 | 45.80 | 54.0% | 43.0% | 17.0% | 9.0% | — |
| 50 | 2.54 | 45.52 | 41.0% | 26.0% | 16.0% | 8.0% | 49.0% |

Wall-clock from start to update 50: **1h 24m 5s** (~84 min, vs my chat-time
estimate of 2.4 hours — actual was 1.42 hr).

Warm-window aggregate (update 4–50, parsed via
[scripts/parse_rl_benchmark.py](../../scripts/parse_rl_benchmark.py)):
**b/s = 2.580 ± 0.039**, **learner steps/s = 45.47 ± 0.55**,
**sec/update = 100.7**.

Comparison vs prior benchmarks (full-100-update aggregates, warm window only;
see `data/benchmarks/2026-05-03-K1/summary.json` and `…-K3/summary.json`):

| Run | Topology | b/s mean | learner steps/s | sec/update |
|-----|----------|----------|-----------------|------------|
| K=1 baseline (2026-05-03) | 4 wkr / 12 ply / 4 srv, batch=4 | 1.497 ± 0.008 | 29.19 ± 0.15 | 171.5 |
| K=3 baseline (2026-05-03) | same | 1.516 ± 0.013 | 29.61 ± 0.23 | 167.0 |
| **easy_test K3 (this run, n=47 warm updates)** | **6 / 18 / 6, batch=8** | **2.580 ± 0.039** | **45.47 ± 0.55** | **100.7** |

That is **1.72× b/s** and **1.66× wall-clock per update** on identical
hardware. Drivers, in order of contribution: (a) topology winner 6/18/6/16 vs
the K-benchmark's 4/12/4 (~1.4×), (b) batch_size 8 / max_battle_steps 40 vs
4 / 30 (~1.15×), (c) ppo_epochs=3 (multiplies *learning per battle*, not b/s,
but reduces sec/update at fixed quality vs K=1 by amortizing inference), and
(d) the VGCBench username plumbing fix — without the fix, ~30% of curriculum
opponent slots fail and workers self-disable VGCBench locally, wasting wall
time on retries before falling back.

**Learning signal at update 50:**

- Total loss: 3.74 → 3.14 (−16% from update 1)
- bc_player: 56% → 41% (model differentiating from frozen BC anchor —
  expected r-NaD behavior)
- vgc_bench: 5% → 8% (slowly improving against the strongest fixed baseline)
- max_damage: 47% → 26% (drop is mild concern; consistent with early-policy
  thrash where the model loses to deterministic heuristics that exploit
  brittleness; K=1 baseline showed `max_damage = 37%` at update 50 for
  comparison)
- simple_heuristic: 14% → 16% (roughly flat)
- ghosts: 49% (the first ghost checkpoint was added at update 25 per
  `portfolio_add_interval`; pool now has one entry; 49% means we're a coin
  flip vs our own past — sensible)

The run is learning by every standard metric: monotone-ish total loss decrease,
sensible BC drift, value-loss stable around 5.0, no NaNs, no divergence in
policy or RNaD loss. It is **not** yet beating max_damage / simple_heuristic by
big margins at update 50 — that is normal at this stage; K1/K3 baselines
required the full 100-update window to climb the simpler-baseline win rates
above ~30%.

## Planned Next Steps

1. Let the run reach update 50; fill in TBD rows above with real numbers.
2. Confirm whether b/s plateaus or continues climbing past update 25 (steady
   state usually arrives by update 8–12 in past runs; we may already be there).
3. Open a separate planning doc on PokeJAX scope evaluation: confirm or
   refute coverage of `gen9vgc2024regg` (Tera, Commander, ITP open-team-sheet
   semantics, all moves used in the curated team pool). If coverage is
   sufficient, this becomes the primary medium-term throughput lever.
4. After that doc lands, decide between (a) cloud scale-out on existing
   Showdown stack vs (b) PokeJAX-based env port, depending on scope answers.
5. Sequence legality-churn fixes (fuzz-harness driven) ahead of any of the
   above; they are independent and stack.

## Updates

### 2026-05-06 01:42 — Update 50 reached, placeholders filled

Run reached update 50 in 1h 24m 5s. Warm-window aggregate
**2.580 ± 0.039 b/s, 45.47 ± 0.55 learner steps/s, 100.7 sec/update**.
Throughput is **1.72× the K1/K3 baseline mean** on identical hardware.
Learning signal healthy by all standard metrics; nothing diverging. Run is
continuing under nohup overnight.
