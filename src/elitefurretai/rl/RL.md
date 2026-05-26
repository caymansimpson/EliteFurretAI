# RL Training System

Current-state reference for EliteFurretAI's reinforcement learning training system. Historical narrative (optimization journey, pre-merge architectures, ablation runs) lives in `planning/stage2/`.

## Table of Contents

1.  [Overview](#1-overview)
2.  [Infrastructure & Hardware](#2-infrastructure--hardware)
3.  [Architecture & Throughput](#3-architecture--throughput)
4.  [RNaD Algorithm](#4-rnad-algorithm)
5.  [Exploiters & Curriculum](#5-exploiters--curriculum)
6.  [Model Architecture & Config](#6-model-architecture--config)
7.  [Known Issues & Workarounds](#7-known-issues--workarounds)
8.  [Training Workflow](#8-training-workflow)
9.  [Quick Start](#9-quick-start)
10. [Module Reference](#10-module-reference)
11. [Future Directions](#11-future-directions)

---

## 1. Overview

EliteFurretAI's RL trainer builds off of a behavior-cloned `TransformerThreeHeadedModel` against itself and a curated opponent pool using **Regularized Nash Dynamics (RNaD)**. The system is built around two constraints:

- **Pokémon Showdown is single-threaded Node.js** where one server pegs one CPU core. Thus, maximizing throughput requires several Showdown servers on different ports.
- **Python's GIL forces a tradeoff between battle-stepping and inference inside a single process,** forcing us to do multiprocessing instead of multithreading.

The architecture is therefore IMPALA-style: one **trainer process** owns the learner and centralized inference; **N worker processes** drive battles through dedicated Showdown servers and never touch model weights where communication is `mp.Queue` only.

**Current state (as of May 2026)**:

- Backend: `showdown_websocket` (poke-env `Player` over local Showdown servers).
- Inference: centralized via `ModelRegistry`. Services can run in-process (daemon thread inside trainer) or in dedicated subprocesses.
- Algorithm: PPO + KL-to-reference + C51 distributional value, with portfolio regularization (multiple frozen reference models, regularize to the closest).
- Current best supervised checkpoint feeding RL: `data/models/supervised/cool-bee-85-finetune_best.pt` (~26.7M params, raw featureset).
- Current throughput on full curriculum (balmy-cloud-70, 48h continuous): **~8.5 traj/s overall**, per-update range ~6–10 traj/s.
- Current optimal topology (sep_arch.yaml, used by balmy-cloud-70): `num_workers=4`, `num_players=16`, `num_servers=4`, `num_battles_per_pair=48`, `max_concurrent_battles_per_player=48`.
- Stage II graduation criterion: simultaneously ≥60% win rate vs `vgc_bench_baseline`, `max_damage`, `bc_player`, and `simple_heuristic_baseline`. We're also exploring a 50% win rate against FoulPlay.

## 2. Infrastructure & Hardware

### The Two Fundamental Problems

Most of the architecture exists to work around two constraints that the rest of the codebase inherits.

**1. Pokémon Showdown is slow and single-threaded.** Showdown is a Node.js simulator, and one server process can only use one CPU core regardless of how many concurrent battles it is handling. The only way to get more simulation throughput out of a multi-core machine is to run several Showdown server processes in parallel on different ports.

**2. Python's GIL forces a tradeoff between battle-stepping speed and inference speed inside a single process.** The Global Interpreter Lock (GIL) is a mutex inside CPython that only lets one thread execute Python bytecode at a time, so even on an 8-core machine Python threads doing Python work end up taking turns on a single core rather than running in parallel.

The work in this pipeline splits into two phases that are both heavily Python-bound:

- **Battle work**: parsing Showdown protocol messages, embedding battle states into feature vectors, computing action masks, and managing the asyncio loop that drives WebSocket I/O.
- **Inference work**: packing input tensors, launching the forward pass (CUDA releases the GIL but the Python wrapper around it does not), unpacking outputs, and sampling actions.

Inside one process these two phases compete for the same lock, which means time the interpreter spends doing inference is time it is not stepping battles, and vice versa. The usual workarounds that look like they should help don't, because:

- **async/await** is still just one thread cooperatively yielding to itself, with no actual parallelism.
- **threading** does not help either, since 16 "concurrent" battles in 16 threads still share one Python interpreter and serialize on the GIL.

The way out is **multiprocessing**: each process has its own Python interpreter and its own GIL, so spawning N worker processes gives us N independent Python execution streams running truly in parallel on N cores.

### Reference Hardware

- **GPU**: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- **CPU**: 8 cores
- **RAM**: 32 GB (24 GB available in WSL2)
- **Storage**: 2 TB NVMe SSD
- **OS**: Linux via WSL2

### WSL2 Notes

```python
# REQUIRED at the start of any training script
torch.multiprocessing.set_sharing_strategy('file_system')

# DataLoader settings
pin_memory=False  # MUST be False on WSL2 — pinned-memory paths OOM the kernel
```

### Six Design Decisions

These six choices follow directly from the two constraints above, and they end up shaping the rest of the code.

**1. Workers are separate processes rather than threads.** Because of the GIL, threads in Python share one interpreter and cannot run the CPU-heavy parts of a battle in parallel. Using `mp.Process` gives each worker its own Python interpreter and its own GIL instead, which is the IMPALA pattern of separate actor and learner processes that communicate only through queues.

**2. One Showdown server per worker, on different ports.** Because Showdown is single-threaded Node.js, one server pegged at 100% CPU is what actually limits us once enough battles are in flight. `launch_servers.py` brings up 4–8 servers on different ports so each worker has its own and they distribute simulation load across CPU cores.

**3. Inference is centralized on the trainer.** All workers submit batched inference requests to trainer-side `InferenceService`s, which means there is only one model copy per registered name and the forward pass batches across every worker's concurrent battles. Workers never hold model weights themselves. We do pay some GPU contention with the learner for this, but the learner on its own does not saturate the 3090, so the spare capacity covers inference.

**4. Inference services can live in subprocesses (Plan C).** When many services run concurrently (main + bc + exploiter + victim + ghosts), they end up contending on the trainer's GIL, so `ModelRegistry` supports grouping services into dedicated subprocesses via a `process_group` argument (e.g. `process_group="ghosts"`). Each subprocess gets its own Python interpreter and CUDA context. This costs roughly 1 GB RSS and 150 MB of GPU memory per subprocess, and is worth paying when the service-side Python work would otherwise serialize behind the trainer.

**5. Trajectories are the IPC currency between worker and learner.** Rather than send raw battle state across processes, workers ship completed trajectory dicts (`steps`, `opponent_type`, `won`, `battle_length`, `forfeited`). This lets the learner stay agnostic about how a given trajectory was produced — it just sees `(state, action, reward)` tuples regardless of the opponent type.

**6. The opponent pool and curriculum run inside each worker.** Self-play against a single opponent tends to collapse into a degenerate optimum, so `opponents.py` maintains a mix of types (`self`, `bc`, `max_damage`, `simple_heuristic_baseline`, `vgc_bench_baseline`, `ghosts`, `exploiters`, `train_exploiter`). Each worker has a `WorkerOpponentFactory` that picks an opponent per battle pair from curriculum weights the trainer pushes down. Because inference is centralized, switching opponents is essentially free — `opponent.inference_client = clients.get("bc")` re-points to a different model without reloading anything.

---

## 3. Architecture & Throughput

### Picture

```
┌──────────────────────── TRAINER PROCESS (owns the GPU) ─────────────────────────┐
│                                                                                 │
│   ┌─ Learner ─────────┐       ┌─ ModelRegistry ──────────────────────────────┐  │
│   │ main_model (FP32) │       │   InferenceService["main"] ──┐               │  │
│   │ ref_model(s)      │◄──────┤   InferenceService["bc"]   ──┤ in-proc       │  │
│   │ optimizer state   │       │   InferenceService["expl"] ──┤ daemon thread │  │
│   └─────────▲─────────┘       │   InferenceService["vict"] ──┘               │  │
│             │ trajectories    │                  AND/OR                      │  │
│             │                 │   InferenceSubprocessHandle["ghosts"]        │  │
│             │                 │       (own python + CUDA process)            │  │
│             │                 └────────────────▲──────────────────┬──────────┘  │
│             │                                  │ requests         │ responses   │
└─────────────┼──────────────────────────────────┼──────────────────┼─────────────┘
              │                                  │                  ▼
              │                                  │                  │
   ┌──────────┴───────┐          ┌───────────────┴─────────┐    ┌───┴──────────────┐
   │ traj_queue       │          │ inference req queue     │    │ per-worker       │
   │ (mp.Queue)       │          │ (mp.Queue, one/service) │    │ response queues  │
   └──────────▲───────┘          └───────────▲─────────────┘    └─────────┬────────┘
              │                              │                            │
              │ pushed by workers            │ pushed by workers          │ routed per worker
              │                              │                            ▼
   ┌──────────┴───────────┐  ┌───────────────┴──────┐  ┌──────────────────┴───┐
   │ WORKER 0             │  │ WORKER 1             │  │ WORKER N             │
   │ (own process)        │  │ (own process)        │  │ (own process)        │
   │                      │  │                      │  │                      │
   │ RLTrajectoryPlayer   │  │ RLTrajectoryPlayer   │  │ RLTrajectoryPlayer   │
   │ × many battles       │  │ × many battles       │  │ × many battles       │
   │           │          │  │           │          │  │           │          │
   │           ▼          │  │           ▼          │  │           ▼          │
   │ Showdown :8000       │  │ Showdown :8001       │  │ Showdown :800N       │
   └──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### How A Single Battle Flows

1. **Worker N** has up to `max_concurrent_battles_per_player` battles running against its dedicated Showdown server (`localhost:800N`).
2. Showdown sends "it's your turn" over WebSocket to a `RLTrajectoryPlayer`.
3. The player computes the **action mask** (`masking.py`) — directly enumerates legal (move, target) and switch pairs from `battle.last_request` instead of probing all 2,025 actions.
4. The player **embeds** the battle state into a feature vector via the Embedder (in `etl/`).
5. The player calls `client.submit(features, mask, ...)` — packages an `InferenceRequest` and puts it on an `mp.Queue` heading back to the trainer process (or to a Plan C subprocess hosting the right service).
6. The player **awaits an asyncio future** for the response. Other concurrent battles in this worker continue meanwhile.
7. Trainer-side, the `InferenceService` daemon thread drains the queue, **batches** up to `batch_size` requests (or waits at most `batch_timeout`), and runs ONE forward pass on the GPU.
8. Results go back through per-worker response queues. The future resolves. The player picks an action (with temperature + top-p sampling) and sends the order to Showdown.
9. After each step, the player saves `(state, action, log_prob, value, mask, …)` to a per-battle buffer.
10. When the battle ends, the worker pushes the **full trajectory** onto `mp_traj_queue` and sends an `EvictRequest` so the trainer can free the hidden-state slot for that `(worker_id, player_id, battle_tag)`.

The **learner** drains `mp_traj_queue`, accumulates trajectories into training batches, computes the RNaD loss, backprops, and at the broadcast cadence calls `registry.sync_weights(name, state_dict)` — which updates each `InferenceService`'s model copy in place (for in-process services) or ships a state_dict over the control queue (for subprocess services).

### Centralized Inference Details

- **Hidden state lives trainer-side**, in `RealModelBatchHandler.hidden_states` keyed by `(worker_id, player_id, battle_tag)`. The wire payload only carries the lightweight battle_tag rather than the full `(1, T, hidden_size)` tensor, which keeps inference requests small. Workers send an `EvictRequest` when a battle finishes so the corresponding slot can be freed.
- **One model copy per name.** `ModelRegistry` registers `main` (always), as well as `bc`, `exploiter`, and `victim` when the curriculum calls for them, plus ghost and exploiter-snapshot slot pools.
- **Weight sync.** When the trainer calls `registry.sync_weights(name, state_dict)`, the registry routes the update either to the in-process model (in-place `load_state_dict`) or to the right subprocess via that group's control queue, depending on where the service is hosted.
- **Subprocess backend (Plan C).** Passing `process_group` to `register()` places a service in a subprocess group instead of the trainer process, giving it its own Python interpreter (which gets it out from under the trainer's GIL) and its own CUDA context.
- **torch.compile.** Compiled forward calls are serialized by a process-wide `_COMPILE_LOCK` in `inference_trainer.py`, because dynamo's trace state is global across instances of the same class and per-model locks turned out not to be enough to prevent cross-instance races.

### Hot-Swap (Multi-Model Curriculum)

Dispatch is split across two methods on `WorkerOpponentFactory`:

- `sample_opp_type_for(player)` picks the opp type from the curriculum, validates it (falls back to `self_play` if e.g. `ghosts` was sampled but no slots are populated), and tags `player.opponent_type`. Does not touch inference clients.
- `apply_opp_type_to_pair(player, opponent, opp_type)` runs only when the opp type uses the BIP (built-in player) pool — `self_play` / `bc` / `ghosts` / `exploiters` / `train_exploiter`. It re-points `opponent.inference_client` (and, for `train_exploiter`, `player.inference_client`) via `_swap_to(slot, name)`. Heuristic and VGCBench types skip this — they have their own Player pools or external usernames.

### Throughput (Current)

Measured on `balmy-cloud-70` (sep_arch.yaml lineage, full curriculum: self_play / bc / ghosts / max_damage / simple_heuristic / vgc_bench / exploiters), 48h continuous run:

| Metric | Value |
|---|---|
| Total trajectories | 1.49M over 48h 30m |
| Overall traj/s (cumulative) | ~8.5 |
| Per-update range | ~6–10 traj/s |
| Learner steps/s (overall) | ~146 |

Optimal topology (from current configs):

```yaml
num_workers: 4
num_players: 16
num_servers: 4
num_battles_per_pair: 48
max_concurrent_battles_per_player: 48  # = num_battles_per_pair removes the 5–30s slow-handler tail
```

The current throughput is the result of several optimization passes (fast action masking, embedder move caching, centralized inference, Plan C), each documented in `planning/stage2/` if you want the full history. The throughput investigation doc is the best entry point.

### Memory Footprint

| Component | Approx |
|---|---|
| Trainer process (learner + ModelRegistry, in-process services only) | ~3 GB RSS + ~3 GB VRAM |
| Worker process | ~700 MB RSS each |
| Plan C subprocess (per group) | ~1 GB RSS + ~150 MB VRAM |
| VGCBench external runner | ~5.5 GB RSS each |
| Showdown server | ~100–200 MB RSS each |

The memory watchdog in `sep_arch.yaml` is set to 22 GB and should be raised if you add more subprocess groups or run additional VGCBench bots beyond the current setup.

---

## 4. RNaD Algorithm

### Why RNaD?

Standard RL can be unstable in games like Pokémon. An agent might discover a simple exploitative strategy, over-optimize for it, and forget robust strategies learned from Behavioral Cloning.

**Regularized Nash Dynamics (RNaD)** forces the learning agent to stay "close" to a stable reference policy, preventing catastrophic forgetting and ensuring:

- **Stability**: smoother training and less prone to sudden collapses.
- **Retention of priors**: human-like strategies from BC are preserved.
- **Robustness**: agent finds improvements that generalize well.

### Loss

$$L_{total} = L_{policy} + \beta \cdot L_{value} - \gamma \cdot H + \alpha \cdot L_{RNaD}$$

| Component | Description |
|---|---|
| $L_{policy}$ | PPO policy loss — increases probability of high-advantage actions |
| $L_{value}$ | C51 distributional value loss — cross-entropy against two-hot encoded targets |
| $H$ | Entropy bonus — encourages exploration |
| $L_{RNaD}$ | KL divergence from reference model — the core of RNaD |

```python
ref_probs = softmax(ref_model(state))
curr_probs = softmax(main_model(state))
kl_divergence = sum(curr_probs * (log(curr_probs) - log(ref_probs)))
```

In words: the agent is allowed to improve, but each gradient step is penalized in proportion to how much its policy distribution has moved away from the reference distribution. Improvements that look like the reference policy are cheap; improvements that look like a totally new policy are expensive, which keeps the agent from abandoning what it already knows.

### Portfolio Regularization

Instead of one reference, keep 3–5 diverse past models and regularize to the closest:

```python
kl_losses = [KL(current_policy || ref_i) for ref_i in portfolio]
kl_loss = min(kl_losses)
```

Regularizing against a single reference tends to drift over time as the reference itself is refreshed from main, which can quietly erase competence against older strategies that no recent reference happens to encode. Keeping a portfolio of several reference snapshots and regularizing to the closest one means the agent has to stay near at least one of them, which preserves more of the strategic surface area across refreshes.

### Inspiration from Ataraxos

DeepMind's Ataraxos (superhuman Stratego):

- **Separate heads** for setup (teampreview: 90 actions) and gameplay (turns: 2,025 actions).
- **Belief states** via `BattleInference` for hidden information.
- **Portfolio regularization** — multiple reference models instead of one.

---

## 5. Exploiters & Curriculum

### Why a Curriculum (and Why Exploiters)

Self-play with one opponent collapses into a degenerate optimum: the agent learns to beat a copy of itself and forgets how to handle anything else. To prevent this, the curriculum mixes **four kinds of opponents**, each plugging a different hole:

1. **Past selves (`ghosts`)** — keeps the agent honest against its own history; prevents cyclical strategy churn.
2. **Behavioral cloning (`bc`)** — anchors to human-like play; prevents drift into RL-discovered nonsense that wouldn't survive against a real human.
3. **Hand-crafted heuristics (`max_damage`, `simple_heuristic_baseline`) and external baselines (`vgc_bench_baseline`)** — catch agents that win against learned policies but lose to dumb deterministic ones (a common failure mode).
4. **Learned exploitation (`exploiters`)** — adversarial policies whose only goal is to beat the current main. They actively probe for weaknesses the other three buckets can't surface.

The Stage II graduation criterion (≥60% vs `vgc_bench_baseline`, `max_damage`, `bc_player`, *and* `simple_heuristic_baseline` simultaneously) is structured around this same logic. It is fairly easy to clear one or two of those buckets in isolation by overfitting to that style of opponent, so demanding all four simultaneously forces the agent to be genuinely general rather than narrowly tuned.

Exploiters specifically are **single-team and unregularized**:

- **One team, learned deeply.** Each exploiter is trained with a single fixed team rather than the full sampling pool. This lets it specialize quickly on one playstyle, which complements the main agent's role of generalizing across many teams.
- **No RNaD regularization.** The whole purpose of an exploiter is to find a strategy that beats the main agent, so it should be free to discover any winning policy without being pulled back toward a reference distribution.
- **Frozen victim.** The exploiter trains against a frozen snapshot of main rather than live main. If the target moved with every learner step, the reward landscape would shift underneath the exploiter and it would oscillate instead of converging. Freezing the victim turns each refresh window into a stable ~1000-update RL problem where exploitation can actually happen.

An exploiter that graduates into the curriculum reveals a hole that none of `ghosts` / `bc` / heuristics happened to catch, so the exploiter bucket is what keeps fresh adversarial pressure flowing into training. Dropping it would leave the agent training against its own history indefinitely, which is the curriculum failure mode this design is most concerned about.

### Exploiter Pipeline

```
MAIN TRAINING LOOP
        │
        ├─ Every exploiter_check_interval updates
        ▼
  Save current main as victim snapshot
        │
        ▼
┌─────────────────────────────────────────────┐
│     EXPLOITER TRAINING SUBPROCESS           │
│                                             │
│  1. Sample ONE team for the exploiter       │
│  2. Train against victim (PPO only, no KL)  │
│  3. Save model + team together              │
└─────────────────────────────────────────────┘
        │
        ▼
  If rolling win_rate > graduation_threshold (default 0.65,
  computed over the last 1000 train_exploiter battles):
    Add exploiter (+ team) to the pool, start fresh generation
        │
        ▼
  WorkerOpponentFactory now samples this exploiter
  (always with its training team)
```

Key knobs (in `RNaDConfig.exploiter_pipeline`):

- `graduation_win_rate_threshold=0.65` — at window=1000, SE ≈ 1.5%, so a graduating exploiter has true win rate ≥ 62% with high confidence.
- `graduation_window=1000` — recent battles to compute rolling win rate.
- `max_updates_per_generation` — stall safeguard. If a generation can't graduate, force a reset (main is likely robust against this basin; try another).
- `victim_refresh_interval` — how often to copy live main's weights into the victim.

### Opponent Pool

`OPP_TYPE` enum (in `opponents.py`):

| Type | Description |
|---|---|
| `self_play` | A second copy of the current learning policy. |
| `bc` | Frozen BC model — anchors human-like play. |
| `ghosts` | Past checkpoints of main, rotated into a slot pool (default `max_ghosts=10`). |
| `exploiters` | Graduated adversarial policies, each with its specific training team. |
| `max_damage` | Fixed heuristic: highest-damage move. Sanity baseline. |
| `simple_heuristic_baseline` | poke-env's `SimpleHeuristicsPlayer`. |
| `vgc_bench_baseline` | External SB3-trained agent (runs in its own venv — see below). |
| `train_exploiter` | The active-generation exploiter, training in parallel. |

Curriculum weights are configured in YAML (`opponents.curriculum`) and pushed down to workers via the trainer's control queue. Win rates are tracked per category and logged to WandB; sampling does NOT currently adapt to weaknesses automatically — graduation criteria are the feedback loop.

### Multi-Format Training

`CurriculumConfig.battle_formats: Dict[str, float]` declares a probability distribution over battle formats. At each `WorkerOpponentFactory.create_agents` call, formats are apportioned across pairs deterministically via the largest-remainder (Hamilton) method — every pair is pinned to one format for the run, and the apportioned distribution is as close as possible to the configured weights.

Constraints (enforced by `CurriculumConfig.__post_init__`):

- Weights must be positive and sum to 1.0 within 1e-6.
- All formats must share the same gen (format string char [3]). This is required because the embedder vocab is gen-keyed.
- Path fields (`agent_team_path`, `opponent_team_pool_path`) accept either a single string (broadcasts to all formats), a dict keyed by format, or null. Dict form must cover every format exactly — no missing or extra keys.

The embedder is built once per worker against `primary_format` (the highest-weight format). Vocab is gen-shared, so every species/move/item in the gen's pokedex is embeddable regardless of which doubles format is sampled at runtime.

VGCBench v1 is single-format; off-`primary_format` pairs cannot challenge it (Showdown rejects mismatched formats). `VGCBenchManager` logs a warning at launch when this configuration is detected. VGCBench v2 (multi-format trained) will replace it later.

For the multi-format graduation check, see `src/elitefurretai/scripts/multi_format_graduation_eval.py` and the `q_format_opp_type_win_rate` / `graduation_summary` metrics.

### Team-Axis Curriculum

The curriculum biases agent team selection per-format in addition to opponent sampling. `OpponentPool` tracks per-`(battle_format, team_name)` EWMA win rate using the same decay rule planned for the opponent-axis EWMA (Change 5, not yet landed). At each curriculum broadcast cadence, `update_team_distribution()` computes per-format `{team_name: weight}` distributions using asymmetric PFSP `(1 - wr) ** p` (reserving `p` for the opponent-axis curriculum from Change 4), with a per-team floor enforced via water-filling and a per-format warm-up gate.

Workers receive the per-format distributions over the same broadcast mechanism that carries the opponent curriculum. `WorkerOpponentFactory.sample_team(fmt, biased=True)` draws a team from the broadcast distribution when one exists for `fmt`, or falls back to uniform `team_repo.sample_team_name(...)` during warm-up. `RLTrajectoryPlayer.current_team_name` flows the sampled name onto the trajectory dict (alongside `battle_format` read from `battle.format`) so the trainer can route the EWMA update back to the right cell.

Bias scope: every training call site uses the biased distribution. Self-play, ghosts, exploiters, and baseline matchups all draw from the same per-format distribution. The `biased=False` opt-out exists for future eval-at-checkpoint paths that want the natural uniform team distribution.

Config knobs (on `CurriculumConfig`):

- `team_axis_enabled` — master switch. When False the broadcast emits `None` for every format and workers stay on uniform sampling.
- `team_warmup_threshold` — minimum non-forfeit battles per `(format, team)` before that format's distribution flips on. Each format latches independently.
- `team_per_team_floor` — minimum post-renormalization weight any team can receive. Enforced by water-filling rather than naive clamp-and-renormalize so the floor invariant is preserved through normalization.
- `pfsp_exponent` — asymmetric PFSP shape, reserved for the opponent-axis curriculum (Change 4) which has not landed.
- `half_life` — EWMA decay rate, reserved for the opponent-axis curriculum (Change 5) which has not landed.

Design spec: `planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md`.

### VGCBench: Fork-Safe External Opponent

`vgc-bench` depends on a different fork of `poke-env` than EliteFurretAI. You can't import both into one Python process without API conflicts.

The fix: run `vgc-bench` in its own venv (`../venv-vgcbench/`) as a separate process. From the trainer side, we don't import `vgc-bench` code at all — we challenge the VGCBench player's hardcoded Showdown username over the Showdown server like any other opponent. From the VGCBench side, `agents/_vgcbench_subprocess.py` (spawned by `VGCBenchManager.launch()` when the curriculum gives `vgc_bench_baseline` positive weight) sits in a loop accepting challenges.

The pattern here is the same one that motivates multiprocessing in the first place: when shared state (Python GIL, conflicting package versions) makes two pieces of code unable to coexist in one process, you isolate them by process boundary and have them talk over a clean protocol. In this case the protocol is the Showdown WebSocket itself, which both sides already speak.

This costs around 5.5 GB of RAM per external runner and adds a measurable throughput tax on full-curriculum runs, so VGCBench-enabled runs are consistently slower than runs that have VGCBench weighted out.

Manual invocation (debugging):

```bash
source ../venv-vgcbench/bin/activate
python src/elitefurretai/agents/_vgcbench_subprocess.py \
    --username VGCBENCHX \
    --server localhost:8000 \
    --battle-format gen9vgc2024regg \
    --checkpoint-path data/models/vgc-bench-sb3-model.zip \
    --team-file data/teams/gen9vgc2024regg/vgcbench.txt \
    --n-challenges 100
```

Usernames are class constants on `VGCBenchManager.USERNAMES` (`["VGCBENCH"]`).

### FoulPlay: Periodic Ground-Truth Eval Opponent

`foul-play-doubles` is a search-based bot (top-100 in human ladder play) we use as a ground-truth eval signal at checkpoint cadence — NOT as a curriculum opponent. The constraint: FoulPlay's search uses 8 cores at ~750 ms/move, which saturates the machine. Curriculum integration would serialize the entire training loop around its search; throughput math doesn't close. So FoulPlay sits in eval-only territory: every `eval_every_n_updates`, the trainer pauses, runs N battles per active format against FoulPlay, and logs win-rate to wandb.

Like VGCBench, FoulPlay is isolated in its own venv (`../venv-foulplay/`) because it depends on `poke-engine-doubles` (a Rust extension) and an older `poke_env`. The same Showdown-as-protocol trick keeps the two Python worlds from colliding. From the trainer side, [`analyze/foulplay_eval.py`](analyze/foulplay_eval.py) iterates `CurriculumConfig.battle_formats` and runs one self-contained cycle per format via the existing `kind="external"` flow ([`launch_external_player`](analyze/player_factory.py) dispatches to `_launch_foulplay_subprocess`). Trajectory parquet + gzipped replays land in the same place as every other eval — so if FoulPlay BC or distillation ever becomes worth doing, the data is on disk for free.

Design and scoping rationale: [`planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md`](../../../planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md). FoulPlay is currently an informational signal, not a Stage II graduation requirement.

#### One-time setup

```bash
# 1. Rust toolchain (poke-engine-doubles builds via cargo on first install).
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Create the venv and install dependencies.
python3 -m venv ../venv-foulplay
../venv-foulplay/bin/pip install --upgrade pip
git clone https://github.com/pmariglia/foul-play-doubles ../foul-play-doubles
../venv-foulplay/bin/pip install -v -r ../foul-play-doubles/requirements.txt

# 3. Make EFA's team pool visible to FoulPlay's load_team(). FoulPlay
#    looks under foul-play-doubles/teams/<format>/. Symlink the EFA pool:
mkdir -p ../foul-play-doubles/teams/gen9vgc2024regg
ln -sft ../foul-play-doubles/teams/gen9vgc2024regg \
    "$(pwd)/data/teams/gen9vgc2024regg/constrained/"*.txt

# 4. Sanity check.
../venv-foulplay/bin/python -c "import fp.run_battle, poke_engine; print('foulplay ok')"
```

#### Inline during training

```yaml
foulplay_eval:
  enabled: true
  eval_every_n_updates: 50
  n_battles_per_format: 100
  search_time_ms: 750
  python_executable: /home/cayman/Repositories/venv-foulplay/bin/python
  # foulplay_team_pool_paths: null → falls back to opponent_team_pool_paths[fmt].
```

Each eval pass pauses training for `n_battles_per_format * |active formats|` battles at full search (~30–60 min for 1 format, 100 battles). Per-format and weight-averaged `eval/foulplay/*` metrics land on wandb. Eval failures are caught and logged — they do not kill training.

#### Manual eval (CLI)

```bash
source ../venv/bin/activate
python -m elitefurretai.rl.analyze.foulplay_eval \
    --checkpoint data/models/rl/<run>/main_model_step_500.pt \
    --battle-formats gen9vgc2024regg:1.0 \
    --n-battles-per-format 50 \
    --python-executable /home/cayman/Repositories/venv-foulplay/bin/python \
    --foulplay-team-pool data/teams/gen9vgc2024regg/constrained \
    --launch-servers
```

---

## 6. Model Architecture & Config

### Model

- **Backbone**: `TransformerThreeHeadedModel` — `TransformerEncoder` over a growing sequence of encoded battle states, with learned `[ACTOR]` / `[CRITIC]` / `[FIELD]` decision tokens prepended. ACTOR → turn head, CRITIC → value head.
- **Parameters**: ~26.7M (cool-bee-85-finetune; raw featureset).
- **Embedding dimensions**: 9,223 input features.
- **Action space**: 2,025 turn actions + 90 teampreview actions.
- **Value head**: C51 distributional — predicts a categorical distribution over 51 bins spanning [-1, 1], trained with cross-entropy against two-hot encoded targets. This gives the value head a richer gradient signal than a scalar MSE head would. The actor side does not need the full distribution, so it consumes the scalar expected value computed as `(softmax(logits) * support).sum(-1)`.
- **Hidden state**: a context tensor of past encoded features that grows by one position each turn as new state is appended. The tensor is held trainer-side in `RealModelBatchHandler.hidden_states`, so workers only need to carry the lightweight battle_tag identifying which context to use.
- **Causal mask**: enabled (`use_causal_mask=True`).

### Exploration

Sampling-side temperature anneal and loss-side entropy anneal share one horizon:

- `temperature_start=1.5`, `temperature_end=0.5`, `exploration_anneal_steps=50000`.
- `top_p=0.95` (nucleus sampling — filters low-probability actions).
- `ent_coef_end` anneals on the same schedule.

Log-probs for PPO ratios are computed at T=1 (unscaled) to avoid biasing importance weights.

### Optimizer

- **Type**: AdamW.
- **Separate param groups**: `lr_backbone` and `lr_heads` are tuned independently. Prevents the value head from under-training (a known PPO failure mode) while regularizing the larger backbone.
- **Scheduler**: linear warmup → cosine or linear decay (`lr_warmup_steps`, `lr_schedule`).
- **Weight decay**: configurable per-group.

### PPO Mini-Epochs

`ppo_epochs` (default 3): how many gradient passes to take over the same batch per RL update. Standard PPO uses 3–10. Old log-probs from collection are reused across epochs; reference-model forwards happen once per update (frozen across epochs). Optional `target_kl` early-stops the inner loop when approximate KL exceeds threshold.

### Number Banks (Off by Default)

`use_number_banks=False` (default). When enabled, numerical features (HP%, stats, base power) are discretized and embedded inside `GroupedFeatureEncoder` via `NumberBankEncoder` instead of being passed as raw floats. The Embedder output format is unchanged — feature identification happens by pattern matching on `Embedder.feature_names` (`HP_PATTERNS`, `STAT_PATTERNS`, `POWER_PATTERNS`).

Enabling number banks changes input dimensions and requires a fresh training run.

### Config Knob Groups (`RNaDConfig`)

| Group | Knobs |
|---|---|
| Hardware | `num_workers`, `num_servers`, `num_players`, `num_battles_per_pair`, `max_concurrent_battles_per_player`, `batch_size`, `batch_timeout`, `device`, `compile_inference_model` |
| Algorithm | `learning_rate`, `ppo_epochs`, `target_kl`, `rnad_alpha`, `gradient_clip` |
| Exploration | `temperature_start/end`, `exploration_anneal_steps`, `top_p`, `ent_coef_end` |
| Optimizer | `optimizer.type`, `weight_decay`, `lr_backbone`, `lr_heads`, `lr_warmup_steps`, `lr_schedule` |
| Distributional Value | `num_value_bins`, `value_min`, `value_max` |
| Number Banks | `use_number_banks`, `number_bank_embedding_dim`, `number_bank_hp/stat/power_bins` |
| Transformer | `transformer_layers/heads/ff_dim/dropout`, `use_decision_tokens`, `use_causal_mask` |
| Portfolio | `use_portfolio_regularization`, `max_portfolio_size`, `portfolio_update_strategy`, `portfolio_add_interval` |
| Exploiter Pipeline | `exploiter_check_interval`, `graduation_win_rate_threshold`, `graduation_window`, `max_updates_per_generation`, `victim_refresh_interval` |
| Curriculum | weight per opp type, `external_vgcbench_python_executable`, `external_vgcbench_team_file` |

---

## 7. Known Issues & Workarounds

| Issue | Workaround |
|---|---|
| WSL2 DataLoader OOM | `pin_memory=False` (always) |
| Loss → NaN | Increase `gradient_clip` or disable mixed precision |
| Showdown timeouts | Reduce concurrent battles per server; add more servers |
| `torch.compile` cross-instance race | `_COMPILE_LOCK` in `inference_trainer.py` serializes all compiled forwards (dynamo trace state is global across instances of the same class — per-model locks were insufficient) |
| Memory ceiling on full curriculum + VGCBench | Memory watchdog at 22 GB in `sep_arch.yaml`; bump if adding Plan C groups |
| `Player.battles` dict leak (long-running eval) | One Python process per opponent type; let process exit reset the leak (see `feedback_poke_env_battles_leak`) |

---

## 8. Training Workflow

### Multi-Stage Process

1. **Behavioral Cloning (BC)**: Pre-train on 1M+ human battles (see `supervised/SUPERVISED.md`).
2. **RL Fine-tuning**: Load BC checkpoint, fine-tune with RNaD against the curriculum.

### Config-Driven

```bash
# Create default config
python src/elitefurretai/rl/config.py my_config.yaml

# Edit my_config.yaml

# Train
python src/elitefurretai/rl/train.py --config my_config.yaml
```

### Resume

```bash
# Most recent checkpoint
python src/elitefurretai/rl/train.py --resume

# Specific checkpoint
python src/elitefurretai/rl/train.py --checkpoint path/to/checkpoint.pt
```

### Exploiter Training

Configured in YAML:

```yaml
exploiter_pipeline:
  enabled: true
  exploiter_check_interval: 5000
  graduation_win_rate_threshold: 0.65
  graduation_window: 1000
  max_updates_per_generation: 10000
  victim_refresh_interval: 1000
```

See [Section 5](#5-exploiters--curriculum) for the design.

### Monitoring (WandB)

Logged per-update:

- **Loss components**: `policy_loss`, `value_loss`, `entropy`, `rnad_loss`.
- **Win rates**: per opponent type — `win_rate/self_play`, `win_rate/bc`, `win_rate/exploiters`, `win_rate/ghosts`, `win_rate/max_damage`, `win_rate/simple_heuristic_baseline`, `win_rate/vgc_bench_baseline`.
- **Curriculum weights**: current sampling probabilities per opp type.
- **Throughput**: `traj/s`, `learner_steps/s`, `batches_per_sec`.

---

## 9. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Showdown servers (6 recommended)
python src/elitefurretai/rl/launch_servers.py --num-servers 6

# 3. Create config
python -c "from elitefurretai.rl.config import RNaDConfig; RNaDConfig().save('config.yaml')"

# 4. Edit config.yaml — set checkpoint_path to your BC model

# 5. Train
python src/elitefurretai/rl/train.py --config config.yaml
```

### Minimal Config (Testing)

```yaml
checkpoint_path: "data/models/supervised/cool-bee-85-finetune_best.pt"
hardware:
  num_workers: 2
  num_servers: 2
  num_players: 6
  num_battles_per_pair: 8
training:
  train_batch_size: 16
  max_updates: 10000
logging:
  use_wandb: false
exploiter_pipeline:
  enabled: false
```

### Production Config (Approximate Current Optimum)

```yaml
checkpoint_path: "data/models/supervised/cool-bee-85-finetune_best.pt"

curriculum:
  battle_formats:
    gen9vgc2024regg: 1.0
  # opponent_team_pool_path: <subdir under data/teams/<format>/, or dict-per-format>

hardware:
  num_workers: 4
  num_servers: 4
  num_players: 16
  num_battles_per_pair: 48
  max_concurrent_battles_per_player: 48

training:
  learning_rate: 0.0001
  rnad_alpha: 0.01
  train_batch_size: 256
  ppo_epochs: 3

portfolio:
  use_portfolio_regularization: true
  max_portfolio_size: 5

exploiter_pipeline:
  enabled: true
  exploiter_check_interval: 5000
  graduation_win_rate_threshold: 0.65

logging:
  use_wandb: true
  wandb_project: "elitefurretai-production"
```

See `src/elitefurretai/rl/configs/sep_arch.yaml` for the running reference.

---

## 10. Module Reference

### `rl_trajectory_player.py`

`RLTrajectoryPlayer` — poke-env `Player` subclass each worker runs. Routes every decision through a centralized `InferenceClient`: computes mask, embeds state, `await client.submit(...)`, picks action with temperature + top-p sampling. Accumulates per-step `(state, action, log_prob, value, mask, …)` and pushes a completed-battle dict (`steps`, `opponent_type`, `won`, `battle_length`, `forfeited`) onto `trajectory_queue` when the battle ends. Sends `EvictRequest` on completion / timeout / stale-request / send-failure paths to free trainer-side hidden state.

### `rnad_model.py`

`RNaDModel` — `torch.nn.Module` wrapper around `TransformerThreeHeadedModel` providing a uniform `forward(x, hidden_state)` API. Carries the growing transformer context between turns; `get_initial_state` returns `None` so the first turn starts with an empty context.

### `learners.py`

`PortfolioRNaDLearner` — holds `main_model` + frozen reference models, pulls trajectories from the queue, computes the RNaD loss (PPO policy + C51 distributional value + entropy + KL), backprops with the topology-aware optimizer and LR scheduler, and periodically copies weights into the reference portfolio. Model construction helpers (`build_model_from_config`, `load_model_from_checkpoint`, `save_checkpoint`, `load_checkpoint`, `is_checkpoint_compatible_with_model_config`) live in this module — used by both trainer and workers.

### `worker.py`

`mp_worker_process` — function each worker subprocess runs. Lifecycle: build `WorkerInferenceClients` from spawn-time mp.Queue handles → build a `VGCEnvironment` over the Showdown websocket backend → run battles → push completed trajectories to the learner via `mp_traj_queue`. Workers never load model weights.

### `train.py`

Trainer entrypoint. `main()` owns the full trainer side: builds the learner via `initialize_learner`, sets up the `ModelRegistry` via `setup_model_registry`, spawns `num_workers` worker processes, runs the trajectory collection loop, calls `learner.update(...)`, periodically calls `registry.sync_weights(...)`, and writes checkpoints. Other helpers: `initialize_training_state`, `collate_trajectories`, `start_memory_watchdog`, `_maybe_run_exploiter_update`, `get_dead_workers`.

### `config.py`

`RNaDConfig` dataclass — all hyperparameters, grouped by domain (hardware / training / exploration / optimizer / portfolio / exploiter_pipeline / curriculum / logging). YAML load/save round-trip.

### `opponents.py`

`OpponentPool` (trainer-side curriculum manager — tracks weights, win rates, ghost slot pool, exploiter pool) and `WorkerOpponentFactory` (worker-side player builder — `sample_opp_type_for` picks the type, `apply_opp_type_to_pair` hot-swaps `inference_client` via `_swap_to`).

### `exploiters.py`

Exploiter pipeline: `ExploiterPipelineState`, `train_exploiter_weight`, `build_exploiter_config`, `initialize_exploiter_pipeline`, `mask_curriculum_during_warmup`, `maybe_run_exploiter_update`. The active-generation exploiter trains in parallel with main; on graduation (rolling win rate ≥ threshold over the last `graduation_window` train_exploiter battles), its weights + team are saved and a fresh generation starts.

### `masking.py`

`fast_get_action_mask(battle: DoubleBattle) -> np.ndarray` — the fast path for valid-action masking. Reads `battle.last_request` and directly enumerates valid (move, target) and switch pairs instead of probing every action.

### `model_registry.py`

Trainer-side `ModelRegistry`: registers `main` (always), `bc` / `exploiter` / `victim` (conditional), and ghost / exploiter-snapshot slot pools. Supports two backends per registration: in-process `InferenceService` thread, or subprocess group (Plan C — pass `process_group="<name>"`). Lifecycle: `register(...)` per service → `start_all()` once → `queues_for_workers()` to get the bundle for worker spawn → `sync_weights(name, sd)` at broadcast cadence → `stop_all()` on shutdown.

### `inference_trainer.py`

Trainer-side: `InferenceService` (daemon thread, drains request queue, batches up to `batch_size` or `batch_timeout`, calls handler once, dispatches per-request responses) + `RealModelBatchHandler` (model-forward path, owns `hidden_states` keyed by `(worker_id, player_id, battle_tag)`, exposes `evict(...)`) + `echo_batch_handler` for plumbing tests. The process-wide `_COMPILE_LOCK` serializes compiled forwards.

### `inference_subprocess.py`

Plan C subprocess host. `run_subprocess(specification: SubprocessSpecification)` is the `mp.Process` entrypoint — each subprocess hosts one or more `InferenceService`s, drains a `control_queue` for `SyncWeightsMsg` / `ShutdownMsg`, and serves traffic on per-service request/response queues. `InferenceSubprocessHandle` wraps the spawned process for the trainer's `ModelRegistry`.

### `inference_worker.py`

Worker-side: `InferenceClient` (submits `InferenceRequest`, awaits response via per-request asyncio future) + `WorkerInferenceClients` (per-worker bundle keyed by model name; counterpart to `ModelRegistry`).

### `inference_ipc.py`

Wire protocol dataclasses: `InferenceRequest`, `InferenceResponse`, `EvictRequest`.

### `launch_servers.py`

Multi-server Showdown launcher. Starts N Showdown servers on consecutive ports.

### `analyze/`

Evaluation utilities, plotters, VGCBench external runner glue.

---

## 11. Future Directions

1. **Adaptive Exploiter Allocation**: Dynamic adjustment of exploiter check interval based on win-rate stability.
2. **Multi-Format Adaptive Curriculum**: Format weights are currently static. Dynamic re-weighting (à la `adaptive_curriculum`) based on per-format performance is a natural follow-up to the shipped multi-format infrastructure (see "Multi-Format Training" section under Curriculum).
3. **Team Generation**: Generate novel teams instead of sampling from a fixed pool.
4. **Native Battle Engine**: Port Showdown to Python to eliminate WebSocket overhead.
5. **Number Bank Tuning**: Optimize bin counts and embedding dimensions for production training.
6. **Batched Transformer Inference**: Pad variable-length contexts for true batched inference per battle (currently sequential within a battle, batched across battles).
7. **Curriculum-Adaptive Sampling**: Have the curriculum react to per-opponent win-rate trends automatically, rather than fixed weights + graduation criteria as the only feedback loop.
