# Fuzz Masking Harness Design

## Context

Despite the recent fixes for the Uproar (`randomNormal` target) and Commander (`forceSwitch=[False,False]`) bugs ([planning/stage2/2026-04-25-23-11-showdown-uproar-and-commander-fix.md](2026-04-25-23-11-showdown-uproar-and-commander-fix.md)), residual invalid-choice errors continue to surface during Showdown self-play, especially in less-common interactions ([planning/stage2/2026-04-26-22-00-two-residual-bugs.md](2026-04-26-22-00-two-residual-bugs.md)). Each error blocks a training step and corrupts the trajectory.

This design specifies a closed-loop fuzz harness that exercises [src/elitefurretai/rl/masking.py](../../src/elitefurretai/rl/masking.py) against the live Showdown websocket backend on `gen9vgc2024regg`, halts on the first invalid-choice error, and produces a self-contained failure report so a follow-up Claude session can diagnose root cause, write a regression test, and implement a fix.

## Before State

Existing related code:

- [src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py](../../src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py) — already does Showdown server launch, random team sampling, websocket clients, error capture via `_handle_battle_error`, and rich human-readable rendering of battle/request state. Uses `RLTrajectoryPlayer` (model-driven).
- [src/elitefurretai/inference/analyze/fuzz_inference.py](../../src/elitefurretai/inference/analyze/fuzz_inference.py) — sequential and concurrent fuzz patterns with `RandomPlayer` subclasses, but used for inference-correctness fuzzing, not masking.
- [src/elitefurretai/rl/masking.py](../../src/elitefurretai/rl/masking.py) — the system under test.
- [src/elitefurretai/etl/encoder.py](../../src/elitefurretai/etl/encoder.py) — `MDBO.from_int(idx, type)` decodes a 0..2024 action index back to a `/choose ...` command.
- [unit_tests/rl/test_fast_action_mask.py](../../unit_tests/rl/test_fast_action_mask.py) — existing masking pytests, two patterns: `BattleIterator(json_log)` replay and `MagicMock(spec=DoubleBattle)` synthetic.

Gap: no harness drives `masking.py` through random legal-only play against live Showdown, so we have no systematic way to find masking bugs.

## Problem

We need a tool that:

1. Plays masked-random vs masked-random battles on `gen9vgc2024regg` until Showdown rejects a choice.
2. On the first rejection, halts and dumps enough state for offline diagnosis (battle, request, attempted action, mask, last 5 turns, full event log).
3. Hands off to a debug/fix workflow that writes a synthetic regression test, fixes the root cause, and restarts the fuzz.

## Solution

### Component 1 — `MaskedRandomPlayer` (added to `showdown_invalid_choice_diagnostics.py`)

A new `Player` subclass alongside the existing `DiagnosticRLTrajectoryPlayer`. Activated by a new `--player random-masked` CLI flag.

**Behavior:**

- **Teampreview:** returns `/team {random permutation of 1234}`. No invalid-choice risk on teampreview itself; randomization diversifies which mons start active across battles.
- **Each turn (regular and force-switch):** call `fast_get_action_mask(battle)`, sample a single index uniformly from `{i : mask[i] == 1}`, then `MDBO.from_int(idx, type).message` to produce the `/choose ...` command.
- **Tera:** emerges naturally — Tera variants occupy specific offsets in the 2025-action space and are sampled whenever the mask permits.
- **No inference / no embedding:** purely exercises the mask-and-decode pipeline.
- **Empty-mask case:** treated as itself a bug. Capture and exit via the same path as an invalid-choice rejection.
- **Error capture:** override `_handle_battle_error` (same hook `DiagnosticRLTrajectoryPlayer` uses) to record `(battle_state, last_request, attempted_command, mask, sampled_index, error_msg, recent_events)`.

### Component 2 — Outer fuzz loop (added to `_run` in `showdown_invalid_choice_diagnostics.py`, gated on `--player random-masked`)

Pseudocode:

```
install SIGINT handler → set sigint_received

resamplings = 0
total_battles = 0
while not sigint_received:
    resamplings += 1
    team_a, team_b = sample 2 distinct teams uniformly from data/teams/gen9vgc2024regg/
    p1, p2 = MaskedRandomPlayer(team_a), MaskedRandomPlayer(team_b)
    error_seen = asyncio.Event()

    # run up to 100 battles per pair, with up to fuzz_concurrent_battles in flight
    for batch_start in range(0, 100, fuzz_concurrent_battles):
        if error_seen.is_set() or sigint_received: break
        batch_size = min(fuzz_concurrent_battles, 100 - batch_start)
        await asyncio.gather(*(run_battle(p1, p2) for _ in range(batch_size)))
        total_battles += batch_size

        # If the batch produced one or more captured invalid-choice errors,
        # write the artifacts for the FIRST captured failure (deterministic by
        # battle_tag string sort), set error_seen, and exit cleanly. Any
        # additional failures from the same batch are dropped — we re-fuzz
        # after the fix.
        if either player captured ≥1 invalid-choice error this batch:
            error_seen.set()
            write artifacts for first failure (Component 3) → shut down server → exit 0

    print f"[resample {resamplings}] still clean after {total_battles} battles (pair: {team_a_name} vs {team_b_name})"

# graceful exit on Ctrl-C
print summary; exit 0
```

**Defaults:**

- `--fuzz-concurrent-battles` default 1 (knob exists for later parallel runs).
- Battles per pair: 100 (matches user spec).
- `--max-battle-steps` 80 (raised from existing default of 40 — masked-random play drags without efficient KOs).
- Server lifecycle: launch one Showdown server at startup via existing `launch_showdown_servers`, reuse across all resamplings, shut down on exit.
- Distinct teams only (no mirror matches).

### Component 3 — Failure-report artifacts

Written to `data/fuzz_results/{ISO timestamp}-{p1_username}-vs-{p2_username}.txt` and a sibling `.json`.

**Text file structure:**

```
================ FUZZ FAILURE REPORT ================
format:        gen9vgc2024regg
timestamp:     2026-05-05T22:42:01
battle_tag:    battle-gen9vgc2024regg-...
failing player: p1 (received the invalid-choice rejection)
sigint hint:   resampling #237, battle 12/100, total battles 23612

team_a (p1): <full Showdown team paste>
team_b (p2): <full Showdown team paste>

attempted command:    /choose move surgingstrikes 1, switch ironhands
decoded action index: 1389  (slot0=27, slot1=42)
mask[1389]:           1.0   ← bug: should have been 0 OR mask was 0 but action was sent anyway
showdown error msg:   "Can't move: Surging Strikes needs a target"

────────── LAST REQUEST PAYLOAD ──────────
<pretty-printed JSON of battle.last_request seen by masking>

────────── BATTLE STATE AT FAILURE ──────────
<output of inference_utils.battle_to_str(battle)>

────────── LAST 5 TURNS OF EVENTS ──────────
turn N-4: ...
turn N-3: ...
turn N-2: ...
turn N-1: ...
turn N (FAILURE TURN): ...

────────── FULL EVENT LOG ──────────
(complete protocol log from start of battle)
```

**JSON sidecar** holds the raw `request`, `events`, `attempted_command`, `sampled_index`, `mask` for any future programmatic replay.

### Component 4 — Per-bug debug & regression workflow

Executed by Claude in a follow-up session, given the failure report:

1. **Diagnose**, focusing on the failure turn + 5 prior turns. Classify as:
   - **(a)** poke-env internal-representation drift (a `DoubleBattle` field disagrees with Showdown's true state), or
   - **(b)** `masking.py` over-permissive (mask=1 when action was illegal) or over-strict (mask=0 when action was legal).
2. **Document** — `planning/stage2/YYYY-MM-DD-hh-mm-fuzz-{short-description}.md` with Context, Before State, Problem, Solution, Reasoning, Updates.
3. **Reproduce in a synthetic test first** — add a `test_fuzz_regression_{short_description}` test to [unit_tests/rl/test_fast_action_mask.py](../../unit_tests/rl/test_fast_action_mask.py) using `MagicMock(spec=DoubleBattle)`. The test must fail before the fix and pass after.
4. **Implement the fix** — narrowly scoped to root cause. `masking.py` for class (b); the relevant poke-env-side module (likely `request_targeting.py` or similar) for class (a). No drive-by refactors.
5. **Guardrail** — for any mask change that switches a 1 → 0, judge whether *additional* nearby actions should also be masked given the same root cause, and extend the test to cover the conceptually-correct breadth (not just the literal one index that triggered the failure). Conversely, add a positive-case test asserting masks stay 1 for adjacent legal actions, to catch over-strictness.
6. **Verify** — run the new test alone, then quality gates: `ruff check src unit_tests`, `ruff format src unit_tests --check`, `pyright src unit_tests`, `pytest unit_tests/rl -q`.
7. **Restart fuzz** — prompt the user to re-run the harness (Claude does not autonomously launch long-running processes).

## Reasoning

**Why extend `showdown_invalid_choice_diagnostics.py` instead of new file?** It already does ~80% of the heavy lifting — server lifecycle, websocket clients, random team sampling, error capture, rendering. A `--player random-masked` flag adds the new player class and a small outer loop without duplicating infrastructure.

**Why synthetic `MagicMock` regression tests instead of JSON-log replay?** `BattleIterator` replays Showdown protocol events but does not faithfully reconstruct `battle.last_request` at an arbitrary turn (request is a separate channel). Synthetic tests with `MagicMock(spec=DoubleBattle)` give us exact control over the failure scenario, are immune to poke-env replay drift, and match the precedent set by `test_single_target_move_requires_explicit_target` in the existing test file.

**Why halt on first error rather than collect many?** Each masking bug typically has a class of related triggers — fixing one often resolves several. Halting forces serial debug-fix-verify cycles, which keeps the diagnosis tight and prevents trying to fix multiple bugs simultaneously.

**Why uniform sampling over the legal action space (rather than structured "pick a move, then pick a target")?** Uniform-over-legal-indices is exactly the distribution `RLTrajectoryPlayer` uses post-mask, so we exercise the same code path as production self-play. A structured sampler would test a different distribution and miss bugs.

**Why 100 battles per pair?** Small enough that one resampling cycle is fast (~100 sec at 1s/battle), large enough to give each pair a fair chance to surface team-specific edge cases. User-specified.

**Why `--max-battle-steps 80` (vs the existing 40)?** Masked-random play has no objective so battles drag. Truncating at 40 turns biases the harness toward early-game edge cases and hides anything that requires status accumulation, late-game force-switches, etc.

## Planned Next Steps

1. After this design doc is approved, transition to the writing-plans skill to produce a step-by-step implementation plan.
2. Implement components 1–3 in the existing `showdown_invalid_choice_diagnostics.py`.
3. Smoke-test the harness with a 2-resampling run to confirm wiring.
4. Hand off to the user for the first long-running fuzz run.
5. On the first failure report, execute the per-bug workflow (Component 4). Loop until the user is satisfied.

## Updates

### 2026-05-05 — Implementation landed (commit ea59853)

Components 1–3 implemented in [src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py](../../src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py). The file's module docstring now back-links to this design doc.

**How to run the fuzz harness** (invocation Claude or the user can execute):

```bash
source ../venv/bin/activate
python -u -m elitefurretai.engine.analyze.showdown_invalid_choice_diagnostics \
  --player random-masked \
  --format gen9vgc2024regg \
  --port 8765 \
  --fuzz-battles-per-pair 1000 \
  --max-battle-steps 80 \
  --log-level 50 \
  --seed <int>
```

Long-running. Stops on the first invalid-choice rejection (or empty mask) OR on Ctrl-C.

**Use `python -u`** when redirecting stdout to a log file. Without it Python block-buffers stdout to ~8 KB, so progress lines (one per resampling round) won't appear in the log until the buffer fills or the process exits. This was learned the hard way during the first launch — a 20-minute background run had a 0-byte log because no flush had happened. Failure-report artifacts go directly to disk (independent of stdout buffering) so they always appear in `data/fuzz_results/` immediately.

**Artifact location:** `data/fuzz_results/{ISO timestamp}-{battle_tag}.txt` (human-readable failure report) and `.artifacts.json` (machine-readable sidecar with raw mask, request, observations).

**Cycle entry point for a future Claude session:**

If `data/fuzz_results/` contains an unprocessed report when you sit down, that means the harness caught a bug and exited cleanly. Process it via the per-bug workflow (Component 4 above):

1. Read the most recent `*.txt` report in `data/fuzz_results/`.
2. Read this design doc (you are here).
3. Diagnose, classify (a) representation drift vs (b) masking, document in `planning/stage2/YYYY-MM-DD-hh-mm-fuzz-{short-description}.md`.
4. Author a `test_fuzz_regression_{short_description}` test in [unit_tests/rl/test_fast_action_mask.py](../../unit_tests/rl/test_fast_action_mask.py) BEFORE the fix. Verify it fails.
5. Fix narrowly; rerun the test; quality gates.
6. After all tests are green, archive the report (move it to a `processed/` subdirectory or delete) and re-launch the harness.

**Throughput observed:** ~0.65 sec/battle at `--max-battle-steps 20`; expect ~1.5–2 sec/battle at `--max-battle-steps 80` (random play drags). 1000 battles per pair ≈ 25–35 minutes per resampling round before validating the pair as clean.

**Amendment to Component 4 step 7:** the design doc originally said "Claude does not autonomously launch long-running processes." The user has since asked Claude to launch fuzz runs in the background. So: when the user explicitly asks, Claude may launch the fuzz harness as a background process and monitor it; otherwise prompt the user.

**Smoke-test result:** 92 clean battles in 60 sec at `--max-battle-steps 20` on `gen9vgc2024regg`. No bugs surfaced at this small scale — expected, since the harness is intended to find rare bugs that take many battles to trigger.
