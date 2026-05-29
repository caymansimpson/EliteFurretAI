# laddering.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/elitefurretai/rl/analyze/laddering.py` so a checkpoint can play N rated battles on the official `sim3.psim.us` ladder for a given format with a fixed team, emitting per-battle jsonl records (opponent, outcome, pre/post rating, GXE, replay URL).

**Architecture:** Single file. A `SimpleModelLadderPlayer` subclass of `SimpleModelPlayer` overrides `_handle_battle_message` to capture ladder-specific signals from Showdown's `|player|` and `|raw|` lines into a per-battle `LadderRecord`, and `_battle_finished_callback` to fire `/savereplay` and dispatch the completed record to a sink callback. Parsing helpers are pure functions (regex over the `|raw|` HTML) and unit-testable independently of any live connection. A `main()` CLI driver loads credentials, instantiates the player, runs `Player.ladder(n_games)`, and writes records to stdout (+ optional file).

**Tech Stack:** Python 3.12, poke-env, PyTorch (via `SimpleModelPlayer`), pytest, ruff, pyright.

**Spec:** [2026-05-29-10-51-laddering-script-design.md](2026-05-29-10-51-laddering-script-design.md)

---

## File Structure

* **Create** `src/elitefurretai/rl/analyze/laddering.py` — all production code:
  - `LadderRecord` dataclass
  - Pure parser helpers: `_parse_player_line`, `_parse_rating_change`, `_parse_gxe`, `_parse_replay_url`
  - `SimpleModelLadderPlayer(SimpleModelPlayer)` — message + battle-finished hooks
  - `_load_credentials(path)` helper
  - `main()` argparse driver and `if __name__ == "__main__"` entry
* **Create** `unit_tests/rl/analyze/test_laddering.py` — pure-function tests for the parsers and credentials loader; the live-connection path is exercised manually per the spec's "Planned Next Steps".

No existing file is modified. Production exclusion list in `src/elitefurretai/scripts/` does not apply — `analyze/` is part of the quality-gate set per CLAUDE.md.

---

## Task 1: Module scaffold + LadderRecord dataclass

**Files:**
- Create: `src/elitefurretai/rl/analyze/laddering.py`

- [ ] **Step 1: Create the file with header, imports, and `LadderRecord` dataclass.**

```python
# -*- coding: utf-8 -*-
"""Official-ladder runner for an RL checkpoint.

Plays N rated ladder battles on sim3.psim.us against the live Showdown
matchmaker using a fixed team and a deterministic argmax policy from a
trained checkpoint. Emits one jsonl record per battle to stdout and,
optionally, to a file.

See planning/stage2/2026-05-29-10-51-laddering-script-design.md for
design rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from poke_env.battle import AbstractBattle
from poke_env.ps_client import (
    AccountConfiguration,
    ShowdownServerConfiguration,
)

from elitefurretai.agents.simple_model_player import SimpleModelPlayer


@dataclass
class LadderRecord:
    """One ladder battle's metadata.

    Optional fields stay None when Showdown didn't surface the data
    (unrated practice format, provisional account, replay save failed).
    """

    battle_tag: str
    opponent: str = ""
    outcome: Optional[Literal["win", "loss", "tie"]] = None  # from agent's POV
    final_turn: int = 0
    pre_rating: Optional[int] = None
    post_rating: Optional[int] = None
    gxe: Optional[float] = None
    replay_url: Optional[str] = None
    timestamp: Optional[str] = None
```

- [ ] **Step 2: Verify the file imports cleanly.**

Run: `source ../venv/bin/activate && python -c "from elitefurretai.rl.analyze import laddering; print(laddering.LadderRecord(battle_tag='x'))"`
Expected: prints `LadderRecord(battle_tag='x', opponent='', outcome=None, ...)` with no traceback.

- [ ] **Step 3: Run quality gates on the new file.**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/laddering.py && ruff format src/elitefurretai/rl/analyze/laddering.py --check && pyright src/elitefurretai/rl/analyze/laddering.py`
Expected: all three commands exit 0. If ruff format reports differences, run without `--check` to apply.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py
git commit -m "rl/analyze: scaffold laddering.py with LadderRecord"
```

---

## Task 2: Parser — `_parse_player_line`

Showdown sends `|player|<slot>|<username>|<avatar>|<rating>` at battle init. The rating field is the integer Elo for rated battles, empty otherwise.

**Files:**
- Create: `unit_tests/rl/analyze/test_laddering.py`
- Modify: `src/elitefurretai/rl/analyze/laddering.py`

- [ ] **Step 1: Write the failing test.**

Create `unit_tests/rl/analyze/test_laddering.py` (or append, if it already exists):

```python
# -*- coding: utf-8 -*-
"""Tests for the laddering script's pure parsers and credentials loader."""

from __future__ import annotations

import json

import pytest

from elitefurretai.rl.analyze.laddering import (
    _load_credentials,
    _parse_gxe,
    _parse_player_line,
    _parse_rating_change,
    _parse_replay_url,
)


def test_parse_player_line_rated():
    """A |player| line for a rated battle yields username and integer rating."""
    msg = ["player", "p2", "OpponentUser", "169", "1500"]
    assert _parse_player_line(msg) == ("OpponentUser", 1500)


def test_parse_player_line_unrated():
    """An empty rating field yields None for the rating."""
    msg = ["player", "p2", "OpponentUser", "169", ""]
    assert _parse_player_line(msg) == ("OpponentUser", None)


def test_parse_player_line_missing_rating_field():
    """Older formats omit the rating field entirely."""
    msg = ["player", "p2", "OpponentUser", "169"]
    assert _parse_player_line(msg) == ("OpponentUser", None)


def test_parse_player_line_not_a_player_line():
    """Returns None when the message isn't a |player| message."""
    assert _parse_player_line(["turn", "5"]) is None
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py::test_parse_player_line_rated -v`
Expected: FAIL — `ImportError: cannot import name '_parse_player_line'`.

- [ ] **Step 2: Implement `_parse_player_line` in `laddering.py`.**

Insert below the `LadderRecord` dataclass:

```python
def _parse_player_line(
    split_message: List[str],
) -> Optional[tuple[str, Optional[int]]]:
    """Extract (username, rating) from a `|player|<slot>|<user>|<avatar>|<rating>` line.

    Returns ``None`` for non-player messages. ``rating`` is ``None`` for
    unrated battles or when the field is absent.
    """
    if not split_message or split_message[0] != "player":
        return None
    if len(split_message) < 3:
        return None
    username = split_message[2]
    rating: Optional[int] = None
    if len(split_message) >= 5 and split_message[4].strip():
        rating = int(split_message[4])
    return username, rating
```

- [ ] **Step 3: Run all four tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k parse_player_line`
Expected: 4 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: parse |player| line into (username, rating)"
```

---

## Task 3: Parser — `_parse_rating_change` and `_parse_gxe`

Showdown emits a `|raw|` line at the end of a rated battle with HTML like:

```
<small>YourUser's rating: 1500 &rarr; <strong>1512</strong></small><br />(+12 for winning)
```

GXE appears nearby (sometimes the same `|raw|`, sometimes the next) as `<small>...GXE 54.3%...</small>`.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`
- Modify: `unit_tests/rl/analyze/test_laddering.py`

- [ ] **Step 1: Write the failing tests.**

Append to `unit_tests/rl/analyze/test_laddering.py`:

```python
def test_parse_rating_change_typical():
    html = (
        "<small>EliteFurret's rating: 1500 &rarr; "
        "<strong>1512</strong></small><br />(+12 for winning)"
    )
    assert _parse_rating_change(html) == (1500, 1512)


def test_parse_rating_change_loss():
    html = (
        "<small>EliteFurret's rating: 1500 &rarr; "
        "<strong>1488</strong></small><br />(-12 for losing)"
    )
    assert _parse_rating_change(html) == (1500, 1488)


def test_parse_rating_change_no_match():
    assert _parse_rating_change("<p>some unrelated raw line</p>") is None


def test_parse_gxe_present():
    html = "<small>Provisional GXE: 54.3%</small>"
    assert _parse_gxe(html) == 54.3


def test_parse_gxe_absent():
    assert _parse_gxe("<p>nothing relevant</p>") is None
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k "rating_change or gxe"`
Expected: 5 FAILED with `ImportError`.

- [ ] **Step 2: Implement the parsers.**

Add to `laddering.py` after `_parse_player_line`:

```python
_RATING_CHANGE_RE = re.compile(
    r"rating:\s*(\d+)\s*&rarr;\s*<strong>\s*(\d+)\s*</strong>",
    re.IGNORECASE,
)
_GXE_RE = re.compile(r"GXE[:\s]*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _parse_rating_change(raw_html: str) -> Optional[tuple[int, int]]:
    """Extract (pre_rating, post_rating) from a Showdown rating-change `|raw|` line."""
    match = _RATING_CHANGE_RE.search(raw_html)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_gxe(raw_html: str) -> Optional[float]:
    """Extract a GXE percentage (e.g. 54.3) from a Showdown `|raw|` line."""
    match = _GXE_RE.search(raw_html)
    if match is None:
        return None
    return float(match.group(1))
```

- [ ] **Step 3: Run the new tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k "rating_change or gxe"`
Expected: 5 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: parse post-battle rating change and GXE"
```

---

## Task 4: Parser — `_parse_replay_url`

Showdown publishes a replay link in a `|raw|` line like:

```
<a class="ilink" href="https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456">https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456</a>
```

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`
- Modify: `unit_tests/rl/analyze/test_laddering.py`

- [ ] **Step 1: Write the failing tests.**

Append to `unit_tests/rl/analyze/test_laddering.py`:

```python
def test_parse_replay_url_present():
    html = (
        '<a class="ilink" '
        'href="https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456">'
        "Open replay</a>"
    )
    assert (
        _parse_replay_url(html)
        == "https://replay.pokemonshowdown.com/gen9vgc2024regg-2189123456"
    )


def test_parse_replay_url_absent():
    assert _parse_replay_url("<p>nothing</p>") is None
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k replay_url`
Expected: 2 FAILED with `ImportError`.

- [ ] **Step 2: Implement `_parse_replay_url`.**

Add to `laddering.py`:

```python
_REPLAY_URL_RE = re.compile(
    r"https://replay\.pokemonshowdown\.com/[A-Za-z0-9_\-]+",
)


def _parse_replay_url(raw_html: str) -> Optional[str]:
    """Extract the Showdown replay URL from a `|raw|` line, if present."""
    match = _REPLAY_URL_RE.search(raw_html)
    if match is None:
        return None
    return match.group(0)
```

- [ ] **Step 3: Run the new tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k replay_url`
Expected: 2 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: parse replay URL from |raw| lines"
```

---

## Task 5: `_update_record` — merge a parsed split message into the per-battle record

Pure function so the message-loop logic stays testable without a live Showdown connection.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`
- Modify: `unit_tests/rl/analyze/test_laddering.py`

- [ ] **Step 1: Write the failing tests.**

Append to `unit_tests/rl/analyze/test_laddering.py`:

```python
from elitefurretai.rl.analyze.laddering import LadderRecord, _update_record


def test_update_record_with_player_line_sets_opponent_and_pre_rating():
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        ["player", "p2", "OpponentUser", "169", "1500"],
        agent_role="p1",
    )
    assert record.opponent == "OpponentUser"
    assert record.pre_rating == 1500


def test_update_record_ignores_own_player_line():
    """The agent's own |player| line shouldn't overwrite opponent/pre_rating."""
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        ["player", "p1", "EliteFurret", "169", "1500"],
        agent_role="p1",
    )
    assert record.opponent == ""
    assert record.pre_rating is None


def test_update_record_with_raw_rating_change():
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        [
            "raw",
            "<small>EliteFurret's rating: 1500 &rarr; "
            "<strong>1512</strong></small>",
        ],
        agent_role="p1",
    )
    # pre_rating may already be set from a |player| line; the raw line
    # is authoritative for post_rating and confirms pre_rating.
    assert record.pre_rating == 1500
    assert record.post_rating == 1512


def test_update_record_with_raw_gxe():
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record, ["raw", "<small>GXE: 54.3%</small>"], agent_role="p1"
    )
    assert record.gxe == 54.3


def test_update_record_with_raw_replay_url():
    record = LadderRecord(battle_tag="battle-x-1")
    _update_record(
        record,
        [
            "raw",
            '<a class="ilink" '
            'href="https://replay.pokemonshowdown.com/gen9vgc2024regg-1">x</a>',
        ],
        agent_role="p1",
    )
    assert (
        record.replay_url
        == "https://replay.pokemonshowdown.com/gen9vgc2024regg-1"
    )
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k update_record`
Expected: 5 FAILED with `ImportError`.

- [ ] **Step 2: Implement `_update_record`.**

Add to `laddering.py`:

```python
def _update_record(
    record: LadderRecord,
    split_message: List[str],
    *,
    agent_role: str,
) -> None:
    """Merge a single Showdown split message into ``record`` in place.

    ``agent_role`` is the agent's own slot (e.g. ``"p1"``) — used to skip
    the agent's own ``|player|`` line so ``opponent`` / ``pre_rating``
    track the *other* player only. Unknown messages are ignored.
    """
    if not split_message:
        return
    tag = split_message[0]
    if tag == "player":
        # |player|<slot>|<user>|<avatar>|<rating>
        if len(split_message) < 3 or split_message[1] == agent_role:
            return
        parsed = _parse_player_line(split_message)
        if parsed is None:
            return
        username, rating = parsed
        if username:
            record.opponent = username
        if rating is not None:
            record.pre_rating = rating
        return
    if tag == "raw" and len(split_message) >= 2:
        raw_html = split_message[1]
        change = _parse_rating_change(raw_html)
        if change is not None:
            pre, post = change
            record.pre_rating = pre
            record.post_rating = post
        gxe = _parse_gxe(raw_html)
        if gxe is not None:
            record.gxe = gxe
        replay = _parse_replay_url(raw_html)
        if replay is not None:
            record.replay_url = replay
```

- [ ] **Step 3: Run the new tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k update_record`
Expected: 5 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: _update_record merges parsed lines into LadderRecord"
```

---

## Task 6: `_finalize_record` — set outcome, final_turn, timestamp from a finished battle

Separated from the live `_battle_finished_callback` so we can unit-test the finalization logic against a fake battle without needing a live PSClient.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`
- Modify: `unit_tests/rl/analyze/test_laddering.py`

- [ ] **Step 1: Write the failing tests.**

Append to `unit_tests/rl/analyze/test_laddering.py`:

```python
from elitefurretai.rl.analyze.laddering import _finalize_record


class _FakeBattle:
    def __init__(self, won, lost, turn, battle_tag):
        self.won = won
        self.lost = lost
        self.turn = turn
        self.battle_tag = battle_tag


def test_finalize_record_win():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(
        record, _FakeBattle(True, False, 18, "battle-x-1")
    )
    assert record.outcome == "win"
    assert record.final_turn == 18
    assert record.timestamp is not None  # ISO-8601 set


def test_finalize_record_loss():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(
        record, _FakeBattle(False, True, 25, "battle-x-1")
    )
    assert record.outcome == "loss"
    assert record.final_turn == 25


def test_finalize_record_tie():
    record = LadderRecord(battle_tag="battle-x-1")
    _finalize_record(
        record, _FakeBattle(False, False, 60, "battle-x-1")
    )
    assert record.outcome == "tie"
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k finalize_record`
Expected: 3 FAILED with `ImportError`.

- [ ] **Step 2: Implement `_finalize_record`.**

Add to `laddering.py`:

```python
def _finalize_record(record: LadderRecord, battle: AbstractBattle) -> None:
    """Populate outcome, final_turn, and timestamp from a finished battle."""
    if battle.won:
        record.outcome = "win"
    elif battle.lost:
        record.outcome = "loss"
    else:
        record.outcome = "tie"
    record.final_turn = int(battle.turn)
    record.timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
```

- [ ] **Step 3: Run the new tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k finalize_record`
Expected: 3 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: _finalize_record populates outcome/turn/timestamp"
```

---

## Task 7: `SimpleModelLadderPlayer` — wire helpers into the live message loop

The class extends `SimpleModelPlayer` to (a) maintain `self.ladder_records`, (b) merge incoming split messages into the matching record via `_update_record`, (c) on battle-finished, finalize the record, send `/savereplay`, and invoke an `on_record` callback. Inheriting `SimpleModelPlayer` keeps the deterministic-argmax `choose_move` unchanged.

We don't unit-test the async hooks themselves (matches the existing `RecordingModelPlayer` precedent in `eval_collector.py` — those parts are exercised only by the end-to-end smoke). The helpers they call are covered by Tasks 5 and 6.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`

- [ ] **Step 1: Implement the class.**

Add to `laddering.py`:

```python
class SimpleModelLadderPlayer(SimpleModelPlayer):
    """SimpleModelPlayer extension that records ladder-specific signals.

    Captures opponent username, pre/post rating, GXE, and the public replay
    URL for each finished battle and dispatches a finalized ``LadderRecord``
    to ``on_record`` when each battle ends.

    Parsing helpers (``_parse_*``, ``_update_record``, ``_finalize_record``)
    are pure module-level functions so they're testable without a live
    Showdown connection; the async hooks below only orchestrate calls
    into those helpers.
    """

    def __init__(
        self,
        *args: Any,
        on_record: Optional[Callable[[LadderRecord], None]] = None,
        **kwargs: Any,
    ) -> None:
        # SimpleModelPlayer forces probabilistic=False unless overridden,
        # which is the deterministic-argmax behavior we want for laddering.
        kwargs.setdefault("probabilistic", False)
        super().__init__(*args, **kwargs)
        self._on_record = on_record
        self.ladder_records: Dict[str, LadderRecord] = {}

    def _get_or_create_record(self, battle_tag: str) -> LadderRecord:
        if battle_tag not in self.ladder_records:
            self.ladder_records[battle_tag] = LadderRecord(battle_tag=battle_tag)
        return self.ladder_records[battle_tag]

    async def _handle_battle_message(
        self, split_messages: List[List[str]]
    ) -> None:
        # Delegate to the parent so battle state / requests are processed
        # as usual; our record-merging runs in addition.
        await super()._handle_battle_message(split_messages)

        if not split_messages or not split_messages[0]:
            return
        battle_tag = split_messages[0][0].lstrip(">")
        if not battle_tag.startswith("battle-"):
            return
        battle = self._battles.get(battle_tag)
        agent_role = battle.player_role if battle and battle.player_role else "p1"
        record = self._get_or_create_record(battle_tag)
        for split_message in split_messages[1:]:
            _update_record(record, split_message, agent_role=agent_role)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        record = self._get_or_create_record(battle.battle_tag)
        _finalize_record(record, battle)
        # Publish the replay so the URL becomes externally shareable.
        # |raw| line carrying the URL arrives moments later and is
        # captured by _handle_battle_message before the battle room
        # closes; if it never arrives, replay_url stays None.
        asyncio.ensure_future(
            self.ps_client.send_message("/savereplay", room=battle.battle_tag)
        )
        if self._on_record is not None:
            self._on_record(record)
```

- [ ] **Step 2: Verify the file still imports cleanly.**

Run: `source ../venv/bin/activate && python -c "from elitefurretai.rl.analyze.laddering import SimpleModelLadderPlayer; print(SimpleModelLadderPlayer)"`
Expected: prints the class object, no traceback.

- [ ] **Step 3: Run quality gates.**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/laddering.py && ruff format src/elitefurretai/rl/analyze/laddering.py --check && pyright src/elitefurretai/rl/analyze/laddering.py`
Expected: all three exit 0.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py
git commit -m "rl/analyze: SimpleModelLadderPlayer hooks for ladder records"
```

---

## Task 8: `_load_credentials` — read JSON credentials file

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`
- Modify: `unit_tests/rl/analyze/test_laddering.py`

- [ ] **Step 1: Write the failing tests.**

Append to `unit_tests/rl/analyze/test_laddering.py`:

```python
def test_load_credentials_valid(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"username": "u", "password": "pw"}))
    assert _load_credentials(p) == ("u", "pw")


def test_load_credentials_missing_file(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        _load_credentials(missing)


def test_load_credentials_missing_keys(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"username": "u"}))  # no password
    with pytest.raises(ValueError):
        _load_credentials(p)


def test_load_credentials_malformed_json(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        _load_credentials(p)
```

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k load_credentials`
Expected: 4 FAILED with `ImportError`.

- [ ] **Step 2: Implement `_load_credentials`.**

Add to `laddering.py`:

```python
def _load_credentials(path: Path) -> tuple[str, str]:
    """Read ``{"username": ..., "password": ...}`` from ``path``.

    Raises ``FileNotFoundError`` if the path doesn't exist,
    ``json.JSONDecodeError`` for malformed JSON, and ``ValueError`` if
    either required key is missing or empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"credentials file not found: {path}")
    data = json.loads(path.read_text())
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        raise ValueError(
            f"credentials file {path} must contain non-empty 'username' "
            f"and 'password' keys"
        )
    return str(username), str(password)
```

- [ ] **Step 3: Run the new tests, expect PASS.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v -k load_credentials`
Expected: 4 passed.

- [ ] **Step 4: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py
git commit -m "rl/analyze: _load_credentials reads username/password JSON"
```

---

## Task 9: CLI `main()` driver

Wires everything together: argparse, credentials, team, player construction, `Player.ladder(n_games)`, and the jsonl sink.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/laddering.py`

- [ ] **Step 1: Implement `main()` and module entry.**

Add to `laddering.py`:

```python
def _make_record_sink(
    output_path: Optional[Path],
) -> Callable[[LadderRecord], None]:
    """Return a callback that prints each record as a jsonl line.

    Always writes to stdout; if ``output_path`` is set, also appends
    to that file. The file is opened in append mode so partial runs
    don't lose history.
    """
    file_handle = open(output_path, "a", encoding="utf-8") if output_path else None

    def sink(record: LadderRecord) -> None:
        line = json.dumps(asdict(record))
        print(line, flush=True)
        if file_handle is not None:
            file_handle.write(line + "\n")
            file_handle.flush()

    return sink


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laddering",
        description=(
            "Play N rated battles on the official Showdown ladder with a "
            "given checkpoint, format, and team."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the RL checkpoint (.pt).",
    )
    parser.add_argument(
        "--battle-format",
        required=True,
        help="Showdown battle format (e.g. gen9vgc2024regg).",
    )
    parser.add_argument(
        "--team",
        required=True,
        type=Path,
        help="Path to the team file (Showdown export format).",
    )
    parser.add_argument(
        "--credentials",
        required=True,
        type=Path,
        help='JSON file containing {"username": ..., "password": ...}.',
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=1,
        help="Number of rated battles to play (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional jsonl file to append per-battle records to.",
    )
    return parser


async def _run_ladder(
    player: SimpleModelLadderPlayer, n_games: int
) -> None:
    await player.ladder(n_games)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.team.exists():
        raise FileNotFoundError(f"team file not found: {args.team}")
    username, password = _load_credentials(args.credentials)
    team_text = args.team.read_text()

    sink = _make_record_sink(args.output)
    account = AccountConfiguration(username, password)
    player = SimpleModelLadderPlayer(
        model_path=str(args.checkpoint),
        device=args.device,
        battle_format=args.battle_format,
        team=team_text,
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        on_record=sink,
    )
    asyncio.run(_run_ladder(player, args.n_games))

    final = list(player.ladder_records.values())
    won = sum(1 for r in final if r.outcome == "win")
    lost = sum(1 for r in final if r.outcome == "loss")
    tied = sum(1 for r in final if r.outcome == "tie")
    last_rating = next(
        (r.post_rating for r in reversed(final) if r.post_rating is not None),
        None,
    )
    last_gxe = next(
        (r.gxe for r in reversed(final) if r.gxe is not None), None
    )
    print(
        f"=== ladder run done: {won}W-{lost}L-{tied}T, "
        f"final_rating={last_rating}, final_gxe={last_gxe}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify `--help` works.**

Run: `source ../venv/bin/activate && python -m elitefurretai.rl.analyze.laddering --help`
Expected: prints the argparse usage text including all flags; exit 0.

- [ ] **Step 3: Run quality gates on the final file.**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py && ruff format src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py --check && pyright src/elitefurretai/rl/analyze/laddering.py unit_tests/rl/analyze/test_laddering.py`
Expected: all three exit 0.

- [ ] **Step 4: Run the full new-file test suite.**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_laddering.py -v`
Expected: all tests pass (4 + 5 + 2 + 5 + 3 + 4 = 23 passing).

- [ ] **Step 5: Commit.**

```bash
git add src/elitefurretai/rl/analyze/laddering.py
git commit -m "rl/analyze: CLI main() for the laddering script"
```

---

## Task 10: Verification and planning-doc update

- [ ] **Step 1: Run the project-wide quality gates against changed paths.**

Run: `source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests`
Expected: all exit 0 (or fix any incidental issues introduced by formatter on neighbour files — should be none since we only touched two files).

- [ ] **Step 2: Update the design doc's Updates section.**

Edit `planning/stage2/2026-05-29-10-51-laddering-script-design.md`. Replace the `(none)` under `## Updates` with a single bullet noting the date the implementation landed and the commit range. Example:

```markdown
## Updates

- 2026-05-29: Implementation landed across commits `<first>..<last>`. All
  parser unit tests passing; live-ladder smoke test still pending —
  needs a registered Showdown account.
```

- [ ] **Step 3: Commit the doc update.**

```bash
git add planning/stage2/2026-05-29-10-51-laddering-script-design.md
git commit -m "planning: log laddering.py implementation in design doc"
```

- [ ] **Step 4: Hand back to the user for manual smoke.**

Per the spec's "Planned Next Steps", the live-ladder smoke requires a Showdown account and access to `sim3.psim.us`. Suggest the user run:

```
python -m elitefurretai.rl.analyze.laddering \
    --checkpoint data/models/supervised/cool-bee-85-finetune_best.pt \
    --battle-format gen9vgc2024regg \
    --team data/teams/gen9vgc2024regg/constrained/38dessert.txt \
    --credentials ~/.config/elitefurretai/showdown_credentials.json \
    --n-games 2
```

Expected: 2 jsonl lines on stdout with populated `battle_tag`, `opponent`, `outcome`, `final_turn`. `pre_rating`/`post_rating`/`gxe`/`replay_url` populated for rated formats.

---

## Self-Review Notes

* **Spec coverage:** every spec section maps to a task — `LadderRecord` (Task 1); `_parse_*` helpers (Tasks 2-4); `_handle_battle_message` hook (Tasks 5, 7); `_battle_finished_callback` + `/savereplay` (Tasks 6, 7); credentials (Task 8); CLI driver + jsonl sink + summary (Task 9); Updates log (Task 10).
* **Placeholder scan:** no TBDs, no "implement appropriately"; every step has the code or command needed.
* **Type consistency:** `LadderRecord` field names defined in Task 1 are referenced unchanged in Tasks 5, 6, 9 (`opponent`, `outcome`, `final_turn`, `pre_rating`, `post_rating`, `gxe`, `replay_url`, `timestamp`). Helper names (`_parse_player_line`, `_parse_rating_change`, `_parse_gxe`, `_parse_replay_url`, `_update_record`, `_finalize_record`, `_load_credentials`) used identically across tasks.
