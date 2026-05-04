# Context

This note records the first full safe CPU-only Showdown batching sweep after implementing true batched transformer actor inference for the websocket backend.

The goal was to choose a runtime configuration that improves throughput on this WSL2 machine without moving into settings that are likely to be fragile for long-running Stage 2 training.

# Before State

- transformer actor batching had been implemented and verified
- short benchmarks showed meaningful improvement, but the best production settings were still unknown
- `20`-battle tests were already known to be noisy enough that they could only be used for coarse filtering

# Problem

We needed to answer two practical questions before locking in a Stage 2 Showdown runtime configuration:

1. which batching settings are actually fastest over a longer horizon
2. which of those settings are still conservative enough for WSL2 and week-long CPU actor runs

The core risk was choosing a configuration based on a `20`-battle outlier and then carrying a slower or less stable setup into longer training runs.

# Solution

Ran a constrained CPU sweep over:

- `batch_size ∈ {4, 8, 16}`
- `batch_timeout ∈ {0.003, 0.005, 0.01, 0.02}`
- `max_concurrent_battles ∈ {2, 4}`

Used `20` battles only to eliminate clearly weak settings, then promoted a conservative finalist set to `100`-battle confirmation.

Final promoted candidates:

- `8 / 0.01 / 2`
- `8 / 0.005 / 4`
- `4 / 0.01 / 2`
- `8 / 0.02 / 4`

Final `100`-battle winner:

- `batch_size=8`
- `batch_timeout=0.02`
- `max_concurrent_battles=4`

Winning result:

- `duration_seconds=122.300`
- `battles_per_second=0.818`
- `battle_loop_seconds=116.040`

Profile evidence on the winner showed that the remaining runtime is dominated by:

- async websocket/event-loop waiting
- CPU transformer inference
- embedding and damage calculation

# Reasoning

This helps Stage 2 because it converts the batching work from a code-level optimization into an operationally usable runtime recommendation.

The important outcome is not just that batching exists, but that the project now has a tested default setting for the Showdown websocket path that is both faster and still conservative enough for this WSL2 environment.

The `100`-battle reruns also validated that the coarse `20`-battle winner was not the real winner, which prevents the team from overfitting to noisy short runs.

# Planned Next Steps/Implementation Plan

1. Use `batch_size=8`, `batch_timeout=0.02`, `max_concurrent_battles=4` as the default safe CPU Showdown actor setting unless a future benchmark disproves it.
2. When resuming performance work, prioritize reducing embedder and damage-calc overhead before attempting larger actor batch sizes.
3. Investigate whether websocket/event-loop wait time can be reduced without shrinking effective batch fill.
4. If GPU actors are reconsidered later, benchmark them separately rather than mixing that question into this CPU-safe baseline.