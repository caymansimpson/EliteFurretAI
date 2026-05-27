# Mega Action Layer (Unified Gimmick Slot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing tera "gimmick" action slot also represent mega-evolution, so a single 2025-dim action space and one policy serve both tera and mega formats.

**Architecture:** The action *index* (move offsets 5–9 within each 10-wide move group) is already a generic once-per-battle gimmick slot. Only two places are tera-specific: the legality mask (which gates the gimmick offset on `canTerastallize`) and the order emission (which hardcodes `terastallize=True`). We make the mask also recognize `canMegaEvo`, and make emission choose mega vs. tera from `battle.can_mega_evolve`. The integer label space, BC label parsing, and policy head are untouched, so tera-format behavior and supervised data are unaffected.

**Tech Stack:** Python, poke-env (local editable checkout at `/home/cayman/Repositories/poke-env/`), numpy, pytest.

**Scope:** This is Plan 1 of the mega-format effort (action layer only). It does NOT add mega-form state features (embedder), opponent mega-enumeration, curriculum changes, or engine validation — those are separate plans (see Roadmap at the end). This plan produces a working, unit-tested action layer on its own.

**Reference design:** `planning/stage2/2026-05-27-16-15-mega-format-cross-format-support-design.md`

---

## Grounding facts (verified 2026-05-27)

- `SingleBattleOrder(order, mega=False, z_move=False, dynamax=False, terastallize=False, move_target=0)` — `mega=True` emits `/choose move <id> mega`. (`../poke-env/src/poke_env/battle/battle_order.py`)
- `DoubleBattle.can_mega_evolve` and `.can_tera` are both `List[bool]` properties, populated from the request fields `canMegaEvo` and `canTerastallize`. (`../poke-env/src/poke_env/battle/double_battle.py:221-228,471-484`)
- Tera and mega are mutually exclusive per format (regMA has mega, no tera; regH etc. have tera, no mega), so a single gimmick offset is unambiguous within any battle.
- The mask's pair-level "no double gimmick" rule (`masking.py:501-505`) is already purely offset-based (`(action % 10) >= 5`), so it applies to mega with no logic change — only naming/comments need updating.

## File Structure

- Modify: `src/elitefurretai/rl/masking.py`
  - `get_valid_slot_actions` — recognize `canMegaEvo` so the gimmick offset is legal in mega formats.
  - `_mark_valid_action_pairs` — rename the local `is_tera_action` → `is_gimmick_action` and update docstring/comments (behavior unchanged).
  - Module header comments (lines ~61, ~214) — generalize "tera" wording to "gimmick (tera/mega)".
- Modify: `src/elitefurretai/etl/encoder.py`
  - `MDBO.to_double_battle_order` — emit `mega=True` instead of `terastallize=True` when `battle.can_mega_evolve[i]` is set for the moving slot.
- Test: `unit_tests/rl/test_fast_action_mask.py` — add mega-availability tests.
- Test: `unit_tests/etl/test_model_double_battle_order.py` — add mega-emission test.

---

## Task 1: Mask recognizes mega availability for the gimmick offset

**Files:**
- Test: `unit_tests/rl/test_fast_action_mask.py` (add new test functions at end of file)
- Modify: `src/elitefurretai/rl/masking.py:290` and `:306-307`

- [ ] **Step 1: Write the failing tests**

Append to `unit_tests/rl/test_fast_action_mask.py`:

```python
def test_mega_availability_enables_gimmick_offset():
    """A mega-format request (canMegaEvo) must make the gimmick offset (+5) legal,
    exactly as canTerastallize does for tera formats."""
    from unittest.mock import MagicMock

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]

    ally_left = MagicMock()
    ally_left.fainted = False
    ally_right = MagicMock()
    ally_right.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [ally_left, ally_right]
    battle.opponent_active_pokemon = [opp_left, opp_right]

    request = {
        "active": [
            {
                "moves": [{"target": "normal", "pp": 8, "disabled": False}],
                "trapped": False,
                "canMegaEvo": True,
            },
            {"moves": [], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    slot_actions = get_valid_slot_actions(battle, 0, request)
    # move 0 with each opponent target (offsets 3,4) plus their gimmick variants (+5)
    assert {3, 4}.issubset(slot_actions)
    assert {8, 9}.issubset(slot_actions), "mega should enable the gimmick offset"


def test_no_gimmick_offset_without_tera_or_mega():
    """Without canTerastallize or canMegaEvo, the gimmick offset (+5) must be illegal."""
    from unittest.mock import MagicMock

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]

    ally_left = MagicMock()
    ally_left.fainted = False
    ally_right = MagicMock()
    ally_right.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [ally_left, ally_right]
    battle.opponent_active_pokemon = [opp_left, opp_right]

    request = {
        "active": [
            {"moves": [{"target": "normal", "pp": 8, "disabled": False}], "trapped": False},
            {"moves": [], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    slot_actions = get_valid_slot_actions(battle, 0, request)
    assert {3, 4}.issubset(slot_actions)
    assert slot_actions.isdisjoint({8, 9}), "no gimmick offset when neither tera nor mega"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_fast_action_mask.py::test_mega_availability_enables_gimmick_offset -v`
Expected: FAIL — `{8, 9}` not in slot_actions (mask ignores `canMegaEvo` today).

(`test_no_gimmick_offset_without_tera_or_mega` should PASS already — it documents the existing guard and pins it.)

- [ ] **Step 3: Implement — recognize mega in the mask**

In `src/elitefurretai/rl/masking.py`, replace the single `can_tera` line in `get_valid_slot_actions` (currently line 290):

```python
    can_tera = active_info.get("canTerastallize") is not None
```

with:

```python
    # The once-per-battle "gimmick" offset (+5) is shared across formats: tera in
    # tera formats, mega in the mega format. They are mutually exclusive per format,
    # so a single offset is unambiguous. Showdown signals availability via
    # canTerastallize (tera type string) or canMegaEvo (bool).
    can_tera = active_info.get("canTerastallize") is not None
    can_mega = bool(active_info.get("canMegaEvo", False))
    can_gimmick = can_tera or can_mega
```

Then replace the gimmick-offset guard (currently lines 306-307):

```python
            valid_actions.add(base_action)
            if can_tera:
                valid_actions.add(base_action + 5)
```

with:

```python
            valid_actions.add(base_action)
            if can_gimmick:
                valid_actions.add(base_action + 5)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_fast_action_mask.py -k "mega or gimmick" -v`
Expected: PASS (both new tests).

- [ ] **Step 5: Run the full mask suite for regressions**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_fast_action_mask.py -q`
Expected: all PASS (tera-format behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/masking.py unit_tests/rl/test_fast_action_mask.py
git commit -m "feat(masking): recognize canMegaEvo for the shared gimmick offset

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Generalize the pair-constraint naming/comments to "gimmick"

This task changes naming and comments only; the offset-based logic already covers mega. It keeps the code honest so a future reader does not believe mega double-use is unguarded.

**Files:**
- Modify: `src/elitefurretai/rl/masking.py` — `_mark_valid_action_pairs` (lines ~496-523), module header comments (lines ~61, ~214-216).

- [ ] **Step 1: Run the existing double-gimmick test to confirm current behavior**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_fast_action_mask.py::test_no_double_terastallization -q`
Expected: PASS (this guards the offset-based rule we are renaming around; it must keep passing).

- [ ] **Step 2: Rename the inner helper and update the docstring**

In `_mark_valid_action_pairs`, update the docstring:

```python
    """Mark valid action pairs, filtering out two pair-level constraints:
    - Both slots cannot use their once-per-battle gimmick (tera or mega) in the same turn.
    - Both slots cannot switch to the same bench Pokemon.
    """
```

Rename the local helper and its call site (logic unchanged — offset 5-9 is the gimmick):

```python
    def is_gimmick_action(action: int) -> bool:
        if action >= SWITCH_ACTION_BASE:
            return False
        # Within each move group of 10, offsets 5-9 are gimmick variants (tera/mega)
        return (action % 10) >= 5
```

and the pair loop:

```python
    for a0 in slot0_actions:
        a0_gimmick = is_gimmick_action(a0)
        a0_switch_target = get_switch_target(a0)

        for a1 in slot1_actions:
            if a0_gimmick and is_gimmick_action(a1):
                continue
```

- [ ] **Step 3: Update the module header comments**

In the file's top docstring, change the edge-case bullet (currently line ~61):

```
- Gimmick lock (only one slot may use its once-per-battle gimmick — tera or mega — per turn).
```

In the "Slot-Level Legality" comment block (currently lines ~214-216), change:

```
#       gimmick offsets: 0=no gimmick, 5=gimmick (tera in tera formats, mega in regMA)
```

- [ ] **Step 4: Run the full mask suite for regressions**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_fast_action_mask.py -q`
Expected: all PASS (pure rename + comments).

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/masking.py
git commit -m "refactor(masking): rename tera pair-constraint to generic gimmick (tera/mega)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Encoder emits mega vs. tera from battle legality

**Files:**
- Test: `unit_tests/etl/test_model_double_battle_order.py` (add new test at end of file)
- Modify: `src/elitefurretai/etl/encoder.py:284-290` (the move-branch `SingleBattleOrder` construction in `to_double_battle_order`)

- [ ] **Step 1: Write the failing test**

Append to `unit_tests/etl/test_model_double_battle_order.py` (the file already imports `MDBO`, `SingleBattleOrder`, `List`, `Any`, and defines `DummyBattle`):

```python
def test_to_double_battle_order_emits_mega_when_can_mega_evolve():
    """In a mega format, a gimmick-offset action must emit mega=True (not terastallize)
    for the slot whose can_mega_evolve flag is set; the other slot stays normal."""
    # MDBO encodes a combined order as slot0 * 45 + slot1. Per-slot action 7 is
    # "move 1 terastallize" (the gimmick offset) and action 2 is plain "move 1", so
    # the combined int is 7 * 45 + 2 = 317.
    mdbo = MDBO.from_int(317, MDBO.TURN)
    assert mdbo.message == "/choose move 1 terastallize, move 1"

    battle = DummyBattle()
    battle.player_role = "p1"
    battle.active_pokemon = [
        type("DummyMon", (), {"moves": {"a": 1}})(),
        type("DummyMon", (), {"moves": {"a": 1}})(),
    ]
    battle.team = {str(i): i for i in range(6)}
    battle.available_moves = [
        [type("DummyMove", (), {"id": "a"})()],
        [type("DummyMove", (), {"id": "a"})()],
    ]
    # Mega format: slot 0 may mega-evolve this turn.
    battle.can_mega_evolve = [True, False]

    dbo = mdbo.to_double_battle_order(battle)  # type: ignore
    assert dbo.first_order.mega is True
    assert dbo.first_order.terastallize is False
    assert dbo.second_order.mega is False
    assert dbo.second_order.terastallize is False


def test_to_double_battle_order_emits_tera_when_not_mega_format():
    """When can_mega_evolve is unset (tera format / supervised replay), the gimmick
    offset must keep emitting terastallize=True — existing behavior preserved."""
    mdbo = MDBO.from_int(317, MDBO.TURN)  # slot0 gimmick offset, slot1 plain

    battle = DummyBattle()  # __getattr__ returns None, so can_mega_evolve is None
    battle.player_role = "p1"
    battle.active_pokemon = [
        type("DummyMon", (), {"moves": {"a": 1}})(),
        type("DummyMon", (), {"moves": {"a": 1}})(),
    ]
    battle.team = {str(i): i for i in range(6)}
    battle.available_moves = [
        [type("DummyMove", (), {"id": "a"})()],
        [type("DummyMove", (), {"id": "a"})()],
    ]

    dbo = mdbo.to_double_battle_order(battle)  # type: ignore
    assert dbo.first_order.terastallize is True
    assert dbo.first_order.mega is False
```

- [ ] **Step 2: Run the tests to verify the mega one fails**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_model_double_battle_order.py -k "emits_mega or emits_tera" -v`
Expected: `test_to_double_battle_order_emits_mega_when_can_mega_evolve` FAILS (currently always emits `terastallize=True`, `mega=False`). `test_to_double_battle_order_emits_tera_when_not_mega_format` PASSES (pins existing behavior).

- [ ] **Step 3: Implement — choose mega vs. tera at emission**

In `src/elitefurretai/etl/encoder.py`, in `to_double_battle_order`, replace the final `SingleBattleOrder` construction in the move branch (currently lines 284-290):

```python
                orders.append(
                    SingleBattleOrder(
                        order=move,
                        terastallize="terastallize" in order,
                        move_target=target,
                    )
                )
```

with:

```python
                # The gimmick offset decodes to the literal token "terastallize" in the
                # internal mapping, but in the mega format that same offset means mega.
                # They are mutually exclusive per format, so we pick based on the battle's
                # mega legality for this slot. can_mega_evolve is absent in supervised
                # replay / tera formats, in which case we emit terastallize as before.
                is_gimmick = "terastallize" in order
                can_mega = getattr(battle, "can_mega_evolve", None)
                use_mega = bool(
                    is_gimmick
                    and can_mega is not None
                    and i < len(can_mega)
                    and can_mega[i]
                )
                orders.append(
                    SingleBattleOrder(
                        order=move,
                        terastallize=is_gimmick and not use_mega,
                        mega=use_mega,
                        move_target=target,
                    )
                )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_model_double_battle_order.py -k "emits_mega or emits_tera" -v`
Expected: both PASS.

- [ ] **Step 5: Run the full encoder suite for regressions**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_model_double_battle_order.py unit_tests/etl/test_encoder_edge_cases.py -q`
Expected: all PASS (tera emission and round-trips unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/etl/encoder.py unit_tests/etl/test_model_double_battle_order.py
git commit -m "feat(encoder): emit mega vs terastallize from battle.can_mega_evolve

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Quality gates and end-to-end sanity

**Files:** none (verification only).

- [ ] **Step 1: Lint**

Run: `source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check`
Expected: no errors.

- [ ] **Step 2: Type-check the touched modules**

Run: `source ../venv/bin/activate && pyright src/elitefurretai/rl/masking.py src/elitefurretai/etl/encoder.py`
Expected: no new errors.

- [ ] **Step 3: Run the full ETL + RL mask/encoder test subset**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_model_double_battle_order.py unit_tests/etl/test_encoder_edge_cases.py unit_tests/rl/test_fast_action_mask.py -q`
Expected: all PASS.

- [ ] **Step 4: Update the design doc's Updates section**

Add a dated line to `planning/stage2/2026-05-27-16-15-mega-format-cross-format-support-design.md` Updates section recording that the action layer (Plan 1) shipped, with the commit range.

- [ ] **Step 5: Commit the doc update**

```bash
git add planning/stage2/2026-05-27-16-15-mega-format-cross-format-support-design.md
git commit -m "docs(planning): mark mega action layer (Plan 1) shipped

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (against the design's Solution section):**
- "Action space stays 40 / unified gimmick offset" → Tasks 1-3 keep the integer space fixed and reuse offset +5. ✓
- "MDBO.to_double_battle_order emits mega vs terastallize by battle legality" → Task 3. ✓
- "Generalize the hardcoded tera-offset logic in masking into one gimmick rule" → Tasks 1-2. ✓
- Out of scope here (deferred to later plans, intentionally): mega-form state features (embedder), opponent mega-enumeration, curriculum mix, engine validation, inference updates. Listed in Roadmap below.

**Placeholder scan:** No TBD/TODO; every code step shows full code and exact commands.

**Type/name consistency:** `can_gimmick`, `is_gimmick_action`, `use_mega`, `can_mega` used consistently across tasks; field names `canMegaEvo` (request) and `can_mega_evolve` (DoubleBattle property) match poke-env source verified above; `SingleBattleOrder(mega=...)` matches the verified signature.

## Roadmap (subsequent plans — not part of this plan)

1. **Plan 2 — Embedder mega-form features:** gimmick-spent flag, mega-availability feature (and `can_tera` for symmetry), mega-form stat/typing/ability representation gated on the Mega Stone.
2. **Plan 3 — Opponents/baselines:** mega-aware action enumeration for `max_damage` and `simple_heuristic`; confirm `vgc_bench` regMA support.
3. **Plan 4 — Curriculum + engine validation:** add regMA to the sampling mix from the start; validate end-to-end on `showdown_websocket`; sweep for new invalid-choice families.
4. **Plan 5 — Inference (if in path):** `speed_inference` / `item_inference` for mega forms.
5. **Open decisions to settle before/within the above:** tera:mega curriculum ratio; regMA graduation bar (bc_player structurally absent).
