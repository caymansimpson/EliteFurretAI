# Play Human vs Model — Design

**Date**: 2026-05-16
**Status**: Pre-implementation. Design agreed; ready for plan writing.

A CLI tool that lets Cayman play a single VGC battle against any RNaD
checkpoint from the terminal, with the model's per-turn reasoning surfaced
inline (configurable: pre- or post-turn).

---

## Context

Two existing pieces cover most of what's needed:

- `examples/human_player.py` defines `HumanPlayer`: a `poke_env.player.Player`
  that prompts the user via stdin each turn, supports doubles/singles, team
  preview, tera, dynamax, switches, targeting, and forfeit.
- `src/elitefurretai/rl/analyze/play_model.py` defines `VerboseModelPlayer`:
  loads an RNaD checkpoint via `load_agent_from_checkpoint`, runs the model
  with per-battle hidden state, prints top-k action probs and the value
  estimate each turn, and supports `challenge`, `ladder`, and `vs-bot`
  modes. The `vs-bot` mode's `--opponent` choices are
  `model | maxdamage | maxbasepower | shp | random` — no `human` option.

What's missing: a way to pit `HumanPlayer` against `VerboseModelPlayer`
locally with both teams configurable and the model's reasoning revealed at
a sensible point in the turn cycle.

Browser-based play (Option B from brainstorming) already works today via
`play_model.py --mode=challenge --launch-server`. This design covers the
pure-terminal path (Option A).

## Before state

- Branch: `main`.
- `examples/human_player_example.py` shows `HumanPlayer` battling a
  `RandomPlayer`/`MaxBasePowerPlayer` via `human.battle_against(opponent, n_battles=1)`.
- `play_model.py`'s `--mode=vs-bot` accepts scripted opponents only. To
  battle a human, the user must use `--mode=challenge` and open the browser.
- No existing module imports both `HumanPlayer` and `VerboseModelPlayer`.

## Problem

Cayman wants to play one RL model from the terminal — no browser, no
spectator view, no replay save. The goal is fast iteration and being able
to see what the model is "thinking" alongside the game state, both for
debugging policy quality and for fair head-to-head play.

Requirements (from brainstorming):

1. Pure terminal experience; no Showdown UI.
2. Model checkpoint passed as required CLI arg; no default.
3. Both human and model team paths are required CLI args (no default; even
   smoke runs must pass both explicitly).
4. Deterministic argmax sampling by default; `--probabilistic` flag flips to
   stochastic.
5. Model debug printout is configurable: hidden until the next state is
   visible (default, post-turn) or shown before the user submits (`--reveal`).
6. Script lives at `src/elitefurretai/rl/analyze/play_human_vs_model.py`
   (next to `play_model.py`, not in `examples/`).

## Solution

A single new file at `src/elitefurretai/rl/analyze/play_human_vs_model.py`
that wires together two thin subclasses and runs `battle_against`.

### Components

**`DeferredVerboseModelPlayer(VerboseModelPlayer)`** — buffers debug
output instead of printing it during `choose_move`.

- New ctor kwarg `reveal: bool`.
- New instance attr `pending_debug: list[str]`.
- Overrides `_print_debug` to either `print(...)` (when `reveal=True`) or
  append the formatted block to `pending_debug` (when `reveal=False`).
- New method `flush_debug() -> str` that joins and clears `pending_debug`.
- Everything else inherited unchanged: model load, hidden-state mgmt,
  embedder, masking, sampling.

**`HumanVsModelPlayer(HumanPlayer)`** — drains the model's pending debug
at the start of each turn so the user sees "model's reasoning from the
turn that just resolved" alongside the new battle state.

- New ctor kwarg `model_player: DeferredVerboseModelPlayer`.
- Overrides `choose_move`: first call `model_player.flush_debug()` and
  print any non-empty result, then defer to `super().choose_move(battle)`.
- All input prompts, formatting, forfeit handling inherited unchanged.

### CLI

```
python -m elitefurretai.rl.analyze.play_human_vs_model <model> \
    --human-team <path> --model-team <path> \
    [--battle-format gen9vgc2023regc] \
    [--device cuda] \
    [--probabilistic] \
    [--reveal] \
    [--num-battles 1] \
    [--top-k 5] \
    [--print-summary] \
    [--launch-server] [--start-port 8000] \
    [--server localhost:8000]
```

Required positional: `model` (path to RNaD checkpoint).
Required flags: `--human-team`, `--model-team`.
All other flags have defaults matching `play_model.py` where applicable.

### Data flow per turn

```
turn N:
  state_N arrives at both players
  ├─ HumanVsModelPlayer.choose_move(state_N):
  │    print(model_player.flush_debug())  # model's turn N-1 reasoning
  │    display state_N, prompt user, return BattleOrder
  └─ DeferredVerboseModelPlayer.choose_move(state_N):
       run model → top-k, value, action
       if reveal: print debug
       else:      append to pending_debug
       return BattleOrder
  both orders submitted → showdown resolves → state_N+1 arrives → repeat
```

Turn 1's model reasoning shows at the top of turn 2. The final turn of
each battle would otherwise leave a residual entry in `pending_debug` (no
next `choose_move` to flush it, or — with `--num-battles > 1` — it would
incorrectly bleed into the next battle's turn 1). To prevent both:

- `DeferredVerboseModelPlayer.choose_move` clears `pending_debug` whenever
  it detects `battle.finished` (same hook that already clears
  `hidden_states` in the base class), and prints the buffer first.
- After `battle_against` returns, orchestration calls
  `flush_debug()` once more and prints any remainder before the record
  summary.

### Server lifecycle

Reuse `launch_showdown_servers(1, start_port)` and
`shutdown_showdown_servers(...)` from
`elitefurretai.engine.showdown_server_manager`. When `--launch-server` is
set, launch one server, override `--server` to `localhost:{start_port}`,
and shut it down in a `finally` block. Otherwise connect to the existing
server at `--server`.

### Account configurations

Two accounts on the local server, no passwords:
- Human: `AccountConfiguration("Human", None)`
- Model: `AccountConfiguration("Model", None)`

Showdown allows arbitrary local usernames without auth.

## Reasoning

- **Subclass over modify.** `VerboseModelPlayer` is already used by
  `play_model.py` in three modes; adding a `reveal` kwarg with default
  `True` to the base class would still couple it to a buffering concept
  that only this script needs. A thin subclass keeps the production
  inference path untouched.
- **Drain-on-next-turn over a separate thread or callback.** Showdown
  doesn't expose a clean "turn fully resolved" hook on the client side;
  the most reliable cross-platform anchor is "the next state has arrived
  and we're being asked to decide again." This also means the user never
  sees stale debug from a battle that just ended — the post-battle flush
  in orchestration handles that case explicitly.
- **`battle_against` over `accept_challenges`/`send_challenges`.** The
  brainstorming-confirmed scope is one-at-a-time local battles. Letting
  poke-env orchestrate the challenge eliminates manual handshake code.
- **No backwards-compat dance.** This is a new file, no existing callers.
- **Both teams required.** Per Cayman's preference — explicit beats
  surprising defaults, especially for evaluation contexts where
  mirror-match vs. cross-team makes a big difference to readouts.

## Error handling

- Argparse validates `Path(...).exists()` for `model`, `--human-team`,
  `--model-team`. Missing path → argparse error message and exit 2.
- `launch_showdown_servers` failure → exception propagates; `finally`
  block still calls `shutdown_showdown_servers` on whatever was returned
  (None-safe).
- `battle_against` exception → propagates; `finally` ensures the server
  is killed and the final record is still printed (if any battles
  completed) inside its own try.
- User types `quit` → existing `HumanPlayer` path returns
  `ForfeitBattleOrder`; recorded as a loss in the final tally.
- Model produces an invalid action (no mask-legal action) → existing
  `VerboseModelPlayer` path falls back to `DefaultBattleOrder`.

## Testing

- **Smoke import.** Add a single test to `unit_tests/rl/test_smoke.py`
  (or a new `unit_tests/rl/test_play_human_vs_model.py`) that imports
  the module and instantiates the two subclasses with a real checkpoint
  (gated behind a `pytest.mark.skipif` if the default checkpoint is
  missing locally). No battle execution — interactive stdin makes that
  impractical to test.
- **Manual verification.** One full battle on a local server using:
  - `model`: `data/models/supervised/cool-bee-85-finetune_best.pt`
  - `--human-team` and `--model-team`:
    `data/teams/gen9vgc2024regg/constrained/naicchampion.txt`
  - default sampling (deterministic), default reveal (off).
  Verify: (a) team preview prompts work; (b) doubles targeting parses;
  (c) model debug appears at start of turn 2 onward; (d) final record
  prints; (e) server cleanly shuts down.

## Out of scope

- Browser/Showdown UI (covered by existing `play_model.py --mode=challenge`).
- Replay save (Showdown server already writes per-battle logs).
- Rust backend support (matches `play_model.py`'s scope).
- ELO / rating tracking across sessions.
- Multiple concurrent battles (`max_concurrent_battles=1` always — the
  human can only play one game at a time).
- Spectator view / hybrid mode (Option C from brainstorming).

## Planned next steps

1. Write the implementation plan to
   `planning/stage2/2026-05-16-XX-XX-play-human-vs-model-implementation-plan.md`.
2. Implement the two subclasses + orchestration + argparse in the new
   file.
3. Add the smoke import test.
4. Manual smoke battle with `cool-bee-85-finetune_best.pt` and the
   naicchampion team on both sides.
5. Quality gates: `ruff check`, `ruff format --check`, `pyright`,
   `pytest unit_tests/rl/test_play_human_vs_model.py`.

## Updates

_(none yet)_
