# laddering.py — official Showdown ladder runner for RL checkpoints

## Context

Stage II evaluation has so far been against local baselines (`max_damage`,
`simple_heuristic`, `vgc_bench`, `foul_play`) on private servers managed by
`engine.showdown_server_manager`. The most recent full eval
([2026-05-19 MODEL_EVALUATION.md](../../src/elitefurretai/rl/analyze/MODEL_EVALUATION.md))
showed the production checkpoint far below the 60%×4 graduation bar against
those baselines.

What is missing is a way to run a checkpoint on the **official**
`sim3.psim.us` Showdown ladder for an external sanity check — ladder rating,
GXE, and replay links provide a population-level signal that the local
baselines can't. This doc designs a small script,
`src/elitefurretai/rl/analyze/laddering.py`, that takes a checkpoint, a
single team file, and a Showdown account, and plays N rated ladder battles
in a chosen format.

## Before State

* `src/elitefurretai/rl/analyze/` contains `evaluate.py`, `foulplay_eval.py`,
  `play_human_vs_model.py`, `eval_collector.py`, `eval_analysis.py`. All
  target *local* Showdown servers — no path connects a checkpoint to the
  official ladder.
* `agents/simple_model_player.py` already loads a checkpoint and runs
  inference inline (deterministic argmax when `probabilistic=False`).
* `poke-env` ships `Player.ladder(n_games)` for laddering and a
  `ShowdownServerConfiguration` constant pointing at
  `wss://sim3.psim.us/showdown/websocket`.
* `poke-env` exposes `_battle_finished_callback` and
  `_handle_battle_message` as the two override points relevant to
  capturing ladder-specific signals (rating, GXE, replay URL).

## Problem

`Player.ladder(n)` plays battles but only exposes aggregate counters
(`n_won_battles`, etc.). Per-battle ladder metadata (opponent username,
pre/post rating, GXE delta, replay URL) arrives over the WebSocket as
`|raw|<html>` lines and as the `|player|` battle-init line. None of it is
surfaced through poke-env's public API.

To run a useful ladder session we need to:

1. Authenticate against the official server with a registered account.
2. Play `n` rated battles on a chosen format with a fixed team using a
   checkpoint's deterministic argmax policy.
3. Capture per-battle: opponent, outcome, pre-rating, post-rating, GXE,
   replay URL, final turn.
4. Emit one jsonl record per battle to stdout (and optionally to a file).

## Solution

Single file: **`src/elitefurretai/rl/analyze/laddering.py`**, containing:

### Class — `SimpleModelLadderPlayer(SimpleModelPlayer)`

Three additions over the base class:

* `_handle_battle_message(split_messages)`: dispatches to the parent for
  state updates, then for each split message also extracts:
  - `|player|p1|<user>|<avatar>|<rating>` → opponent username + pre-rating
    (rating field empty for unrated).
  - `|raw|...` at battle end → post-rating + GXE via regexes targeting
    Showdown's standard rating-change HTML (`...&rarr;<strong>NNNN</strong>...`
    and `GXE NN.N%`).
  - `|raw|<a ... href="https://replay.pokemonshowdown.com/<tag>...">` →
    replay URL.
  Records are accumulated in `self.ladder_records: dict[str, LadderRecord]`
  keyed by `battle_tag`.
* `_battle_finished_callback(battle)`: sends `/savereplay` to the battle
  room so Showdown publishes the replay, sets `outcome` and `final_turn`
  on the record, then invokes the user-supplied `on_record` callback if
  set. After the callback fires, the record can be popped from the dict.
* No `choose_move` change — inherited argmax (`probabilistic=False`) is
  exactly what Stage II evals use.

### Data — `LadderRecord` dataclass

```python
@dataclass
class LadderRecord:
    battle_tag: str
    opponent: str
    outcome: Literal["win", "loss", "tie"]  # from the agent's perspective
    final_turn: int
    pre_rating: Optional[int]
    post_rating: Optional[int]
    gxe: Optional[float]
    replay_url: Optional[str]
    timestamp: str  # ISO8601, set when record completes
```

`Optional` fields stay `None` when Showdown didn't surface them (unrated
format, provisional account, replay save failed, etc.). No try/catch
wraps message parsing — regex misses simply leave the field unset.

### CLI driver

`main()` does, in order:

1. argparse for `--checkpoint`, `--battle-format`, `--team`,
   `--credentials`, `--n-games`, `--device` (default `cuda`),
   `--output` (optional jsonl path).
2. Load credentials JSON: `{"username": ..., "password": ...}`. Fail fast
   on missing file, malformed JSON, or missing keys.
3. Read team file contents.
4. Build `AccountConfiguration(username, password)` and use
   `ShowdownServerConfiguration` from `poke_env.ps_client`.
5. Instantiate `SimpleModelLadderPlayer` with `probabilistic=False`. Pass
   an `on_record` callback that prints the record as jsonl to stdout and
   (if `--output` set) appends to that file.
6. `asyncio.run(player.ladder(args.n_games))`.
7. Print a final aggregate summary to stderr: total played, W/L/T,
   final rating, final GXE (last non-None values seen).

### Output shape

Per battle, one line to stdout:

```json
{"battle_tag":"battle-gen9vgc2024regg-...","opponent":"...","outcome":"win","final_turn":18,"pre_rating":1500,"post_rating":1512,"gxe":54.3,"replay_url":"https://replay.pokemonshowdown.com/...","timestamp":"2026-05-29T17:00:00Z"}
```

### Example invocation

```
python -m elitefurretai.rl.analyze.laddering \
    --checkpoint data/models/supervised/cool-bee-85-finetune_best.pt \
    --battle-format gen9vgc2024regg \
    --team data/teams/gen9vgc2024regg/constrained/38dessert.txt \
    --credentials ~/.config/elitefurretai/showdown_credentials.json \
    --n-games 50 \
    --output data/ladder/cool-bee-85-finetune_2026-05-29.jsonl
```

## Reasoning

**Why a Player subclass over post-hoc HTTP queries.** Showdown's
post-battle rating/GXE deltas arrive on the same WebSocket as the battle
itself, inside `|raw|` lines that reference the exact battle. Hooking
`_handle_battle_message` is one place; doing it post-hoc requires
matching battles to a separate ladder query and risks race conditions
when other battles complete between the query and the parse.

**Why deterministic argmax.** Matches the sampling mode used in every
other Stage II eval (evaluate.py, foulplay_eval.py), so a ladder rating
from this script is directly comparable to the local-baseline WRs in
[MODEL_EVALUATION.md](../../src/elitefurretai/rl/analyze/MODEL_EVALUATION.md).
Probabilistic sampling could be added later as a flag without changing
the file layout.

**Why one battle at a time.** Concurrent ladder play increases
account-flagging risk for negligible throughput gain — a VGC ladder
battle is ~5-15 minutes; 50 battles at concurrency 1 is a several-hour
overnight run, which is fine. The script never holds compute idle
waiting on Showdown matchmaking because the in-process model only runs
during `choose_move`.

**Why stdout + optional file, no per-run directory.** The user
explicitly chose this over a `data/ladder/<run>/` tree. One jsonl line
per battle is greppable and trivially fed back into pandas without a
schema dependency. If a structured run-dir is needed later, it can be
added.

**Why `/savereplay` rather than `Player.save_replay` (local HTML).**
The desired artifact is a shareable URL on Showdown's public replay
server. `Player.save_replay` writes the replay HTML to a local file
instead, which doesn't expose anything externally. `/savereplay` is
the in-chat command that triggers Showdown to publish; the resulting
URL comes back as a `|raw|` line we already parse.

**Why fields stay `Optional` instead of guaranteed.** Pre-provisional
accounts get no rating in the `|player|` line until Showdown decides
the rating is stable; some unrated practice formats omit GXE entirely; if
Showdown's reply markup ever shifts, regex misses leave a `None`
instead of crashing the whole session. Down-stream analysis filters by
`pre_rating is not None` when it needs rated battles only.

## Planned Next Steps

1. Hand off to writing-plans to produce a step-by-step implementation
   plan with TDD slices (parser unit tests against captured `|raw|`
   fixtures; integration smoke against a local server before pointing
   at the official ladder).
2. After implementation: dry-run against a freshly-registered throwaway
   account on the official server with `n_games=2` to confirm the
   parsers fire and replay URLs come back populated.
3. Once verified, run `cool-bee-85-finetune_best.pt` on
   `gen9vgc2024regg` for an overnight 50-game session to get a first
   real-world ladder signal.

## Updates

(none)
