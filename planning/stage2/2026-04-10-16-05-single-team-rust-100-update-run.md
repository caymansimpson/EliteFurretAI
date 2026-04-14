# Single-Team Rust 100-Update Run

## Context

We wanted a real `train.py` run based on the single-team RL setup, but using the Rust engine backend with wandb enabled, and we wanted evidence that the run is:

- operational through the real worker/learner path
- actually reaching learner updates
- not immediately slowing down or exploding memory

This is Stage 2 operational work because it validates the long-running Rust training path rather than one-off benchmark scripts.

## Before State

Before this pass:

- the Rust backend had already been recommended as the primary training runtime
- a new one-off config did not yet exist for a dedicated 100-update single-team Rust run with isolated outputs
- a fresh long single-team Rust run crashed before the first learner update

The first crash was not a backend legality issue. It was a transformer-context issue in the synchronous policy player.

## Problem

The initial 100-update run based on the single-team configuration failed before the first update with:

- `RuntimeError: The size of tensor a (45) must match the size of tensor b (44) at non-singleton dimension 1`

Root cause:

- `SyncPolicyPlayer` kept accumulating Transformer hidden/context state across turns
- once battles exceeded the model's positional-encoding budget, `forward_with_hidden()` produced a sequence longer than the transformer's positional encoding supported
- the rollout recorder already stopped storing learner steps after `max_battle_steps`, but hidden-state growth continued anyway

That meant the run was not yet safe for long-lived Rust training.

## Solution

1. Added a dedicated config for the experiment:
   - `src/elitefurretai/rl/configs/single_team_rust_100_updates.yaml`
   - derived from `single_team.yaml`
   - `battle_backend: rust_engine`
   - `max_updates: 100`
   - `use_wandb: true`
   - isolated save directory and run name

2. Fixed the Transformer context overflow in:
   - `src/elitefurretai/engine/sync_battle_driver.py`

3. Added focused regression coverage in:
   - `unit_tests/engine/test_sync_battle_driver.py`

The concrete fix is that transformer hidden/context tensors are now trimmed to the model's `max_seq_len` before reuse and after each forward step.

## Reasoning

This is the correct fix because it addresses the real state-growth bug rather than downgrading the experiment to a non-transformer config or hiding the failure with a smaller run.

The model's positional encoding is a hard architectural limit. If online RL inference stores more than that amount of context, the worker will eventually crash on long battles. Long-running training therefore requires bounded Transformer context regardless of whether learner trajectory recording has already stopped for that battle.

## Run Status

Active run:

- config: `src/elitefurretai/rl/configs/single_team_rust_100_updates.yaml`
- wandb run: `single-team-rust-100-updates-2026-04-10`
- wandb id: `9q8jjlez`
- wandb URL: `https://wandb.ai/caymansimpson/elitefurretai-rnad/runs/9q8jjlez`

## Validation

- Focused regression test:
  - `python -m pytest unit_tests/engine/test_sync_battle_driver.py -q`
  - result: passed

- The rerun no longer died at worker startup / early-battle Transformer overflow.

- First live learner update recorded from the real `train.py` path:
  - `Update 1: Loss=2.4577, Policy=0.2742, Value=4.4968, RNaD=1.0954`
  - `Total Battles=256 in 0h 14m 8s (0.30 b/s)`
  - `Learner Steps=6950 (8.19 steps/s)`
  - `Learner Trajectories=256 (0.30 traj/s)`
  - observed win rates in the console at update 1:
    - `self_play: 48.0%`
    - `bc_player: 40.6%`

- Worker batch summaries before update 1 show the Rust actor path is running normally and feeding trajectories:
  - truncation ranged from `15.6%` to `37.5%` across early worker batches
  - per-batch sent-trajectory counts ranged from `20` to `30`

## Early Memory Notes

Process RSS samples during the active run:

- around 3 minutes:
  - main trainer: about `1.14 GB`
  - workers: about `2.07 GB` each

- around 6 minutes:
  - main trainer: about `1.59 GB`
  - workers: about `2.15 GB` each

- around 10 minutes:
  - main trainer: about `1.91 GB`
  - workers: about `2.17 GB` to `2.22 GB`

- around 15 minutes / after the first learner update:
  - main trainer: about `2.22 GB`
  - workers: about `2.14 GB` to `2.19 GB`

Interpretation so far:

- worker RSS appears to be warming into a fairly stable band rather than climbing monotonically every sample
- the main process is still rising during startup + first-update warmup, so we do not yet have enough wall-clock evidence to claim long-horizon stability
- this is a materially better state than the pre-fix run, which crashed before update 1

## Other Observations

- The run emits repeated poke-env warnings for `SANDTOMB` being treated as `Effect.UNKNOWN`.
  - This did not stop the run.
  - It is worth tracking separately if it correlates with truncation or state-quality issues later.

- The run also emits a PyTorch warning that `lr_scheduler.step()` is called before `optimizer.step()`.
  - This is not new Rust-backend behavior, but it is worth cleaning up because it changes the effective first LR schedule step.

## Planned Next Steps

1. Let the active 100-update wandb run continue and monitor whether learner steps/s stays in the same range across later updates.
2. Watch `mem_rss_gb` and `mem_workers_total_rss_gb` in wandb over more updates before claiming week-scale memory stability.
3. If later updates slow materially, compare worker batch truncation and trajectory-send counts against update-1-era behavior before changing throughput settings.
4. Clean up the scheduler call order warning in `train.py` in a separate pass, because it affects all backends and is orthogonal to the Rust overflow fix.