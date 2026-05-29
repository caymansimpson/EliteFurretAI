# HumanPlayer CLI Overhaul + Battle Renderer Extraction

**Date:** 2026-05-29 12:10
**Author:** Cayman (with Claude)
**Status:** Plan — implementation queued immediately after sign-off

## Context

[`HumanPlayer`](../../src/elitefurretai/agents/human_player.py) is the CLI agent used for interactive doubles play (see [`rl/analyze/play_human_vs_model.py`](../../src/elitefurretai/rl/analyze/play_human_vs_model.py)). It also overlapped with Singles handling, which we just removed in the prior edit pass. Separately, [`inference/inference_utils.py`](../../src/elitefurretai/inference/inference_utils.py) hosts two state-rendering helpers (`observation_to_str`, `battle_to_str`) used by inference debug paths. Both render the same battle structures into text from different angles.

The two rendering paths grew independently — the human-readable CLI inside `HumanPlayer`, and the dense debug dumps in `inference_utils`. They duplicate work (format Pokemon, weather, side conditions, events) but with different formats and audiences.

## Before State

**[`src/elitefurretai/agents/human_player.py`](../../src/elitefurretai/agents/human_player.py)** (doubles-only after the prior edit):

- `_display_battle_state(battle)` — single-snapshot render: active mons, opp actives, no field, no effects, HP bar.
- `_format_pokemon`, `_format_move`, `_create_hp_bar` — leaf primitives, private to HumanPlayer.
- `_handle_team_preview` — prompts for `1 2 3 4`, marks `_selected_in_teampreview`, emits `/team 1234`.
- `_handle_doubles_turn` — action grammar is position-based: `1a 2b` where `1/2` = acting slot, `a/b/c` = target letter. Tera is `t1`/`t2` mixed into the input string. Reprints the targeting cheatsheet + special-options block every turn.

**[`src/elitefurretai/inference/inference_utils.py`](../../src/elitefurretai/inference/inference_utils.py:751)**:

- `observation_to_str(obs)` — renders one observation: active mons, weather, fields, side conditions, per-mon (speed, item, speed boost, effects, status), raw events.
- `battle_to_str(battle, opp=None)` — renders both teampreview teams (with item + speed) plus every observation in `battle.observations`.

**Six consumers of `battle_to_str`** today:

- [`inference/battle_inference.py`](../../src/elitefurretai/inference/battle_inference.py) (7 prints)
- [`inference/item_inference.py`](../../src/elitefurretai/inference/item_inference.py) (7 prints)
- [`inference/analyze/fuzz_inference.py`](../../src/elitefurretai/inference/analyze/fuzz_inference.py) (5 appends)
- [`agents/verbose_model_player.py`](../../src/elitefurretai/agents/verbose_model_player.py) (1 print)

No backwards-compat concern per [CLAUDE.md](../../CLAUDE.md).

## Problem

1. **HumanPlayer's CLI is hard to read at speed**: species/items/moves are unnormalized poke-env ids, no field state, no effects/abilities/boosts, HP bars are noise at full HP, position-based targeting (`a/b/c`) doesn't map to Showdown's wire convention, cheatsheet repeats every turn.
2. **No information for the opponent side**: no abilities (when known), no revealed moves, no tera state.
3. **Action grammar is awkward**: `1a 2b` + `t1` mixed in is hard to parse mentally; doesn't read like a move.
4. **No event log between turns** — the user can't see what just happened without scrolling back through Showdown's own log window.
5. **Force-switch path is undefined**: the current code doesn't separate force-switch turns from action turns, and would silently fail when only switches are legal.
6. **Duplicated rendering logic**: HumanPlayer's `_format_pokemon` and inference_utils's per-mon line both extract the same structure with different formatting.

## Solution

### High-level

Move battle-state rendering into the engine package as the single source of truth, redesign the CLI around Showdown's native target convention, and rebuild the action grammar to be readable. Delete the old `inference_utils` renderers (no shims).

### New file: `src/elitefurretai/engine/battle_renderer.py`

Pure functions, no state. Reads `DoubleBattle` / `Observation` / `Pokemon` and emits strings. No `input()`, no I/O beyond return values.

**Leaf primitives:**

```python
def format_pokemon_line(mon: Pokemon, *, is_opponent: bool, show_tera_availability: Optional[bool] = None) -> str
def format_field(source: Union[DoubleBattle, Observation]) -> str
def format_moves_oneline(moves: Sequence[Move], *, assume_max_pp: bool = False) -> str
def format_events(events: Sequence[Sequence[str]]) -> str
```

**Composite views:**

```python
def format_action_reference() -> str                    # the one-shot help banner
def format_teampreview(battle: DoubleBattle) -> str     # both teams, no HP bars
def format_battle_state(battle: DoubleBattle) -> str    # per-turn snapshot for HumanPlayer
def format_observation(obs: Observation) -> str         # replaces inference_utils.observation_to_str
def format_battle_log(battle, opp: Optional[Player] = None) -> str  # replaces inference_utils.battle_to_str
```

### Agreed CLI design

**Action reference (printed once at battle start, before teampreview):**

```
================================================================================
ACTION REFERENCE  (printed once at battle start)
================================================================================
Target codes (Showdown convention):
   1,  2   = opp slot 1, opp slot 2
  -1, -2   = your slot 1, your slot 2  (for ally-target moves; omit for self-target)

Action grammar: comma-separated, slot 1 then slot 2.

  Move:     "<move_name|number> [<target>] [tera|mega]"
            e.g.  "astralbarrage 1 tera, surgingstrikes 1"
            e.g.  "protect, helpinghand -1"

  Switch:   use species name
            e.g.  "incineroar, surgingstrikes 1"

  Pass:     "pass"      (when a slot has no legal action)
  Forfeit:  "quit"

Notes:
  - No target needed for self-target / spread moves (protect, nasty plot, eq, etc.)
  - During a force switch you'll be prompted with only the slot(s) that must switch.
```

**Teampreview** (no HP bars; opp team printed as species list):

```
================================================================================
TEAM PREVIEW — gen9vgc2025regg
================================================================================

Your team:
  1. calyrexshadow @ lifeorb       | Ability: asoneglastrier  | Tera: ghost
  2. urshifurapidstrike @ mysticwater | Ability: unseenfist   | Tera: water
  ...

Opponent's team:
  1. miraidon
  2. fluttermane
  ...

Select 4 Pokemon for your team (you have 6).
Your selection:
```

**Turn snapshot** — field, both actives with effects/ability/boosts/status, opp moves seen, condensed moves, bench, last-turn raw events:

```
--- LAST TURN ---
  |move|p2a: Calyrex|Astral Barrage|p1a: Miraidon|[spread] p1a,p1b
  |-damage|p1a: Miraidon|0 fnt
  ...

--- FIELD ---
  Weather: (none)              Terrain: (none)
  Your side conditions: (none)
  Opp side conditions:  (none)

--- ACTIVE POKEMON ---
Your side:
  Slot 1 (-1): calyrexshadow @ lifeorb (tera'd: ghost)  | Ability: asoneglastrier  | HP: 90%
               Boosts: (none)   Status: ok   Effects: (none)
  Slot 2 (-2): urshifurapidstrike @ mysticwater  | Ability: unseenfist  | HP: 100%
               Tera: water (ally used)   Boosts: (none)   Status: ok   Effects: (none)

Opp side:
  Slot 1 (1):  miraidon  | Ability: hadronengine  | HP: 100%
               Boosts: (none) Status: ok Effects: (none) Moves seen: (none yet)
  ...

--- YOUR OPTIONS ---
Slot 1 calyrexshadow moves:
   (1) astralbarrage (8/8)  (2) psyshock (16/16)  (3) nastyplot (32/32)  (4) protect (16/16)
Slot 2 urshifurapidstrike moves:
   (1) surgingstrikes (8/8)  ...

Bench:
  incineroar @ sitrusberry   | Ability: intimidate   | HP: 100%  Status: ok  Effects: (none)
  rillaboom @ assaultvest    | Ability: grassysurge  | HP: 100%  Status: ok  Effects: (none)
```

**Force-switch turn** — distinct header, no action menu, only bench listing:

```
================================================================================
Turn 1 — FORCE SWITCH
================================================================================

--- EVENTS SO FAR THIS TURN ---
  |move|p1a: Miraidon|Electro Drift|p2a: Calyrex
  |faint|p2a: Calyrex

--- FORCE SWITCH REQUIRED ---
Your slot 1 must switch (calyrexshadow fainted). Slot 2 takes no action this prompt.

Bench:
  incineroar @ sitrusberry   | ...

Your action (just the switch): incineroar
```

### Action grammar (parser spec)

Tokens: `pass`, `quit`, integer, species name, move id/number, target int, `tera`, `mega`.

```
input          := <slot_input> ("," <slot_input>)?
slot_input     := "pass" | <switch> | <move_input>
switch         := <species_name>                          # matches a bench mon
move_input     := <move_ref> <whitespace_token>*
move_ref       := <move_id> | <move_number_1_to_4>
whitespace_token := <target> | "tera" | "mega"
target         := "1" | "2" | "-1" | "-2"
```

Resolution rules:

1. Split top-level by `,`. If 1 token in normal turn → error. If 1 token in force-switch with only 1 forced slot → OK, the other is `PassBattleOrder()`.
2. For each slot input: lower-case, split by whitespace. First token disambiguates:
   - `"pass"` → `PassBattleOrder()`
   - matches a bench mon species → `SingleBattleOrder(order=bench_mon)` (switch)
   - matches an available move id or is `1`..`len(moves)` → move path
3. Remaining tokens (zero or more) optionally specify target / tera / mega. Tera flag set only if `battle.can_tera[slot]`. Mega similarly.
4. If target absent and move is self/spread, that's fine; pass `0` (no target).
5. `quit` at top level (before split) → `ForfeitBattleOrder()`.

### Force-switch detection

Inside `choose_move`, before the normal action path:

```python
if isinstance(battle, DoubleBattle) and any(battle.force_switch):
    return self._handle_force_switch(battle)
```

`_handle_force_switch` prompts only for the forced slot(s), parses switch by species, fills `PassBattleOrder()` for the unforced slot.

### inference_utils cleanup

Delete `observation_to_str` and `battle_to_str` from [`inference_utils.py`](../../src/elitefurretai/inference/inference_utils.py:751-816). Update all six importers to import `format_observation` / `format_battle_log` from `engine.battle_renderer` instead. No re-exports.

### ENGINE.md ownership update

Add a one-paragraph section: *"engine owns rendering of battle structures (battle/observation/pokemon → str). Rendering is a read-only view of state engine already understands."*

## Reasoning

**Why engine/ over a new package**: rendering reads exclusively from battle structures the engine already owns. A new `display/` package would be one file; the engine already has the right concept boundary. Cost of a new top-level package > benefit here.

**Why delete (no shims)**: per [CLAUDE.md](../../CLAUDE.md) we don't carry backwards compat, and only six call sites need updating. Shims would rot.

**Why Showdown-native target codes**: matches the protocol the user already reads in event logs (`p1a: Miraidon`, target `1`). Keeps mental model consistent across reading and writing actions.

**Why raw event lines (not natural-language render)**: NL rendering is open-ended work — every move/ability/effect has its own template, and parser bugs would mislead the user. Raw lines are already terse and the user reads them today in Showdown's UI.

**Why pure functions in `battle_renderer.py`**: testable without a live battle, no global state, composes cleanly. The HumanPlayer becomes a thin shell: render + input + parse.

**Why TDD on the renderer leaves but not the input loop**: leaf renderers (`format_pokemon_line`, `format_moves_oneline`, `format_events`) take plain objects and emit strings — perfect unit-test surface. The `input()` loop in `_handle_doubles_turn` isn't unit-testable in isolation (see [`test_play_human_vs_model.py`](../../unit_tests/rl/test_play_human_vs_model.py)) — the parser, on the other hand, is.

## Planned Next Steps

Each task ends with a commit. Frequent commits per [CLAUDE.md](../../CLAUDE.md) discipline. TDD where the surface is testable (leaf renderers, parser). Quality gates (`ruff check`, `ruff format --check`, `pyright`, `pytest unit_tests -q`) run at the end of every task.

### Task 1 — Scaffold `engine/battle_renderer.py` with leaf primitives + tests

**Files:**
- Create: [`src/elitefurretai/engine/battle_renderer.py`](../../src/elitefurretai/engine/battle_renderer.py)
- Create: [`unit_tests/engine/test_battle_renderer.py`](../../unit_tests/engine/test_battle_renderer.py)
- Create: [`unit_tests/engine/__init__.py`](../../unit_tests/engine/__init__.py) if absent

**Steps:**

- [ ] Write failing tests for `format_pokemon_line` covering: (a) own mon with item + ability + HP + status `ok`, (b) tera'd own mon emits `(tera'd: <type>)` after item and omits `Tera:` availability line, (c) opp mon with no item shown, with `Moves seen:` listing revealed moves (assume max PP per agreed UI), (d) fainted mon → `(fainted)`.
- [ ] Implement `format_pokemon_line` to pass.
- [ ] Write failing tests for `format_field` covering weather + terrain + side conditions on both sides (and empty `(none)` cases).
- [ ] Implement `format_field` to pass.
- [ ] Write failing tests for `format_moves_oneline` — own moves emit `(N) id (cur/max)`; opp moves with `assume_max_pp=True` emit `(N) id (max/max)`.
- [ ] Implement to pass.
- [ ] Write failing tests for `format_events` — given a list of `["|move|...", "|-damage|...", "|faint|..."]` it joins them with newline + leading indent, one per line, no filtering. Empty list → `"  (no events yet)"`.
- [ ] Implement to pass.
- [ ] Commit `feat(engine): add battle_renderer leaf primitives`

### Task 2 — Composite views: `format_action_reference`, `format_teampreview`, `format_battle_state`

**Files:**
- Modify: [`src/elitefurretai/engine/battle_renderer.py`](../../src/elitefurretai/engine/battle_renderer.py)
- Modify: [`unit_tests/engine/test_battle_renderer.py`](../../unit_tests/engine/test_battle_renderer.py)

**Steps:**

- [ ] Test: `format_action_reference()` is a constant string starting with the banner header `ACTION REFERENCE` and containing both `1, 2 = opp slot` and `-1, -2 = your slot` substrings.
- [ ] Implement as a module-level constant string returned by the function (simple — no logic).
- [ ] Test: `format_teampreview(battle)` includes both team headers, lists 6 own mons with item + ability + tera, and lists 6 opp species (no item).
- [ ] Implement (using a stub `DoubleBattle` for the test — see `unit_tests/inference/` for examples of stubbing battle objects).
- [ ] Test: `format_battle_state(battle)` includes the FIELD, ACTIVE POKEMON, YOUR OPTIONS, Bench sections in order.
- [ ] Implement.
- [ ] Commit `feat(engine): add battle_renderer composite views`

### Task 3 — Replacements for `observation_to_str` / `battle_to_str`

**Files:**
- Modify: [`src/elitefurretai/engine/battle_renderer.py`](../../src/elitefurretai/engine/battle_renderer.py)
- Modify: [`unit_tests/engine/test_battle_renderer.py`](../../unit_tests/engine/test_battle_renderer.py)

**Steps:**

- [ ] Test: `format_observation(obs)` includes the same key data the legacy `observation_to_str` did (active mons line, weather, fields, side conditions, team status, events) and is functionally equivalent for a snapshot fixture.
- [ ] Implement (port [`inference_utils.py:751-775`](../../src/elitefurretai/inference/inference_utils.py#L751) into the new module verbatim, with light cleanup).
- [ ] Test: `format_battle_log(battle, opp=None)` reproduces the legacy header + teampreview blocks + per-turn observations.
- [ ] Implement (port [`inference_utils.py:778-816`](../../src/elitefurretai/inference/inference_utils.py#L778)).
- [ ] Commit `feat(engine): port observation/battle log renderers to engine`

### Task 4 — Cut over inference_utils consumers

**Files:**
- Modify: [`src/elitefurretai/inference/inference_utils.py`](../../src/elitefurretai/inference/inference_utils.py) — delete `observation_to_str` (lines 751-775) and `battle_to_str` (lines 778-816).
- Modify: [`src/elitefurretai/inference/battle_inference.py`](../../src/elitefurretai/inference/battle_inference.py) — replace import + 7 call sites.
- Modify: [`src/elitefurretai/inference/item_inference.py`](../../src/elitefurretai/inference/item_inference.py) — replace import + 7 call sites.
- Modify: [`src/elitefurretai/inference/analyze/fuzz_inference.py`](../../src/elitefurretai/inference/analyze/fuzz_inference.py) — replace import + 5 call sites.
- Modify: [`src/elitefurretai/agents/verbose_model_player.py`](../../src/elitefurretai/agents/verbose_model_player.py) — replace import + 1 call site.

**Steps:**

- [ ] Replace each `from elitefurretai.inference.inference_utils import battle_to_str` with `from elitefurretai.engine.battle_renderer import format_battle_log`.
- [ ] Replace each `battle_to_str(...)` call with `format_battle_log(...)`.
- [ ] Delete `observation_to_str` and `battle_to_str` from [`inference_utils.py`](../../src/elitefurretai/inference/inference_utils.py).
- [ ] Run quality gates: `ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q`.
- [ ] Commit `refactor(inference): cut over to engine.battle_renderer`

### Task 5 — Action grammar parser

**Files:**
- Create: [`src/elitefurretai/agents/_human_action_parser.py`](../../src/elitefurretai/agents/_human_action_parser.py) — pure parser (no I/O).
- Create: [`unit_tests/agents/test_human_action_parser.py`](../../unit_tests/agents/test_human_action_parser.py)
- Create: [`unit_tests/agents/__init__.py`](../../unit_tests/agents/__init__.py) if absent

**Steps:**

- [ ] Test: `parse_action("astralbarrage 1 tera, surgingstrikes 1", battle)` returns `DoubleBattleOrder` with slot 1 move = astralbarrage, target=1, tera=True; slot 2 move = surgingstrikes, target=1.
- [ ] Test: `parse_action("protect, helpinghand -1", battle)` — slot 1 protect (target 0), slot 2 helpinghand (target -1).
- [ ] Test: switch by species: `parse_action("incineroar, surgingstrikes 1", battle)` → slot 1 switch to incineroar.
- [ ] Test: numeric move ref: `parse_action("1 1, 4", battle)` → slot 1 move 1 targeting opp 1, slot 2 move 4 (self target).
- [ ] Test: `parse_action("pass, protect", battle)` → slot 1 PassBattleOrder, slot 2 protect.
- [ ] Test: `parse_action("quit", battle)` → `ForfeitBattleOrder`.
- [ ] Test: force switch: `parse_action("incineroar", battle, force_switch=[True, False])` → slot 1 switch, slot 2 pass.
- [ ] Test: invalid move id raises a custom `ActionParseError` with a clear message.
- [ ] Test: tera flag silently ignored when `battle.can_tera[slot]` is False.
- [ ] Implement `parse_action` to pass all the above.
- [ ] Commit `feat(agents): add HumanPlayer action grammar parser`

### Task 6 — Rewrite HumanPlayer around the renderer + parser

**Files:**
- Modify: [`src/elitefurretai/agents/human_player.py`](../../src/elitefurretai/agents/human_player.py)

**Steps:**

- [ ] Drop `_format_pokemon`, `_format_move`, `_create_hp_bar`, `_display_battle_state`, `_handle_doubles_turn` body.
- [ ] Add `_action_ref_shown: bool = False` class-side flag; in `choose_move`, on first call print `format_action_reference()` and flip the flag.
- [ ] Replace the body of `choose_move` with: print state via `format_battle_state(battle)`, then dispatch:
  - if `battle.teampreview` → `_handle_team_preview` (kept, but now uses `format_teampreview` for display)
  - elif `any(battle.force_switch)` → `_handle_force_switch`
  - else → `_handle_action_turn` (prompt → `parse_action(...)` → return).
- [ ] Add `_handle_force_switch(battle)`: print force-switch banner + bench, prompt, parse with `force_switch` arg.
- [ ] Add `_handle_action_turn(battle)`: prompt, call `parse_action(input_str, battle)`, on `ActionParseError` print the message and re-prompt.
- [ ] Keep `teampreview()` and `choose_default_move()` as-is.
- [ ] Commit `feat(agents): rewrite HumanPlayer CLI around battle_renderer`

### Task 7 — Update tests + ENGINE.md

**Files:**
- Modify: [`src/elitefurretai/engine/ENGINE.md`](../../src/elitefurretai/engine/ENGINE.md) — add a short "Battle rendering" paragraph and an ownership row.
- Modify: [`unit_tests/rl/test_play_human_vs_model.py`](../../unit_tests/rl/test_play_human_vs_model.py) — verify the existing tests still pass and HumanPlayer still imports clean.

**Steps:**

- [ ] Update [`ENGINE.md`](../../src/elitefurretai/engine/ENGINE.md): add `### Battle rendering` subsection under the layout list pointing to [`battle_renderer.py`](../../src/elitefurretai/engine/battle_renderer.py) and a line under "Ownership Guide" — *"engine owns text rendering of battle structures it owns (battle/observation/pokemon → str)."*
- [ ] Run quality gates one last time.
- [ ] Commit `docs(engine): document battle_renderer ownership`

### Task 8 — Update this planning doc's Updates section

- [ ] On each task commit, append a single-line entry to the Updates section of this doc with the SHA and a one-line summary.

## Updates

<!-- Populated during implementation. Format: YYYY-MM-DD hh:mm — <sha> — <one-line summary> -->
