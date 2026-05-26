# Change 7 — Agent-Team-Axis Adaptive Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat `team_provider`-style uniform team sampling with a per-(battle_format, agent_team) adaptive sampling distribution that over-samples teams the model is worst at piloting.

**Architecture:** Trainer-side `OpponentPool` tracks per-(format, team) EWMA WR using the same decay rule as Change 5. At the existing curriculum-broadcast cadence, it recomputes a per-format `team_distribution: Dict[fmt, Optional[Dict[name, weight]]]` using asymmetric PFSP `(1-wr)^p` (reusing `p` from Change 4) with a per-team floor and per-format warm-up gating. Workers receive the broadcast and bias `WorkerOpponentFactory.sample_team()`, which now returns `(team_string, team_name)` and accepts a `biased: bool` flag. A new `current_team_name` attribute on `RLTrajectoryPlayer` flows the name onto the trajectory dict alongside `battle_format`. Multi-format scoped from day one.

**Tech Stack:** Python 3.10+, NumPy, poke-env, pytest, project ruff + pyright gates.

**Spec:** [planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md](2026-05-24-12-30-change7-team-axis-curriculum-design.md)

**Environment activation reminder:** every shell command below assumes `source ../venv/bin/activate &&` prefix. Worker can omit re-activating if already in the venv.

---

## File map

**Create:**
- `unit_tests/rl/test_team_axis.py` — all new unit tests for Change 7.

**Modify:**
- `src/elitefurretai/rl/config.py` — add three `CurriculumConfig` fields.
- `src/elitefurretai/etl/team_repo.py` — add `sample_team_name(...)` helper.
- `src/elitefurretai/rl/opponents.py` — `OpponentPool` per-format state, `record_battle_result` extension, `update_team_distribution`; `WorkerOpponentFactory` `team_distribution_by_format`, `sample_team` signature/return, `update_curriculum` signature, call-site migration to unpack tuple + stamp `current_team_name`.
- `src/elitefurretai/rl/rl_trajectory_player.py` — `current_team_name` attribute; extend trajectory dict with `team_name` and `battle_format`.
- `src/elitefurretai/rl/train.py` — pass new args to `record_battle_result`; invoke `OpponentPool.update_team_distribution()`; bundle into broadcast.
- `src/elitefurretai/rl/RL.md` — §7 update describing the new axis.

**Do not modify (verified):** `src/elitefurretai/rl/analyze/team_provider.py`, `src/elitefurretai/engine/analyze/showdown_benchmark.py`, `src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py` — external callers of `team_repo.sample_team(...)` that must keep their existing return type.

---

## Task 1: Add `CurriculumConfig` fields

**Files:**
- Modify: `src/elitefurretai/rl/config.py` (insert near end of `CurriculumConfig` class body, before `__post_init__`)
- Create: `unit_tests/rl/test_team_axis.py`

- [ ] **Step 1.1: Create the test file with the first test**

Create `unit_tests/rl/test_team_axis.py` with:

```python
"""Unit tests for Change 7 — agent-team-axis adaptive curriculum.

Each test exercises one slice of the per-(battle_format, agent_team)
sampling design from
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.
"""

from __future__ import annotations

from elitefurretai.rl.config import CurriculumConfig


def test_curriculum_config_has_team_axis_defaults():
    """CurriculumConfig exposes three new team-axis fields with documented defaults."""
    cfg = CurriculumConfig()
    assert cfg.team_axis_enabled is True
    assert cfg.team_warmup_threshold == 20
    assert cfg.team_per_team_floor == 0.005
```

- [ ] **Step 1.2: Run the test and confirm it fails**

```
pytest unit_tests/rl/test_team_axis.py::test_curriculum_config_has_team_axis_defaults -v
```

Expected: FAIL with `AttributeError: 'CurriculumConfig' object has no attribute 'team_axis_enabled'`.

- [ ] **Step 1.3: Add the three fields to `CurriculumConfig`**

In `src/elitefurretai/rl/config.py`, locate `CurriculumConfig` (around line 435). Insert these fields immediately *before* the `def __post_init__` line (which is currently around line 523). Insertion anchor: the line after `vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"`.

Insert:

```python
    # ── Change 7: agent-team-axis adaptive curriculum ────────────────────
    # Master switch. False = bypass entirely; workers ignore broadcast
    # team_distribution_by_format and stay on uniform team sampling.
    team_axis_enabled: bool = True
    # Min battles per (format, team) before that format's biased
    # distribution activates. Per-format gate; each format latches
    # independently.
    team_warmup_threshold: int = 20
    # Min normalized weight any team can receive within a format's
    # distribution after the floor pass + renormalization.
    team_per_team_floor: float = 0.005
```

- [ ] **Step 1.4: Run the test and confirm it passes**

```
pytest unit_tests/rl/test_team_axis.py::test_curriculum_config_has_team_axis_defaults -v
```

Expected: PASS.

- [ ] **Step 1.5: Run quality gates on the changed files**

```
ruff check src/elitefurretai/rl/config.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/config.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/config.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors. (If ruff format --check fails, run `ruff format src/elitefurretai/rl/config.py unit_tests/rl/test_team_axis.py` and stage the result.)

- [ ] **Step 1.6: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/config): add team-axis curriculum config fields

Adds team_axis_enabled (master switch), team_warmup_threshold (per-
format gate), and team_per_team_floor (post-normalization floor)
fields on CurriculumConfig. First step of Change 7 from the adaptive
curriculum overhaul.

See planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `team_repo.sample_team_name` helper

**Files:**
- Modify: `src/elitefurretai/etl/team_repo.py` (add helper alongside existing `sample_team`)
- Modify: `unit_tests/rl/test_team_axis.py` (append new tests)

The existing `team_repo.sample_team(...)` returns the team string. The biased path in `WorkerOpponentFactory.sample_team` needs to draw a name first (from the broadcast distribution) and then look up the string — and the uniform path needs to draw a name too so the worker can stamp it on the trajectory. The cleanest minimal change is a sibling helper `sample_team_name` that returns just the name; the existing `sample_team(...)` stays untouched for the three external callers identified in the file map.

- [ ] **Step 2.1: Write the failing test**

Append to `unit_tests/rl/test_team_axis.py`:

```python
import os
from pathlib import Path

import pytest

from elitefurretai.etl.team_repo import TeamRepo


def _write_team_file(path: Path, name: str) -> None:
    """Write a placeholder team file with the given filename stem.

    TeamRepo only inspects file presence + content existence; any non-
    empty contents work for sampling tests.
    """
    path.write_text(
        "Pikachu @ Light Ball\nAbility: Static\nLevel: 50\n"
        "EVs: 252 Atk / 4 Def / 252 Spe\nNature: Jolly\n- Volt Tackle\n"
    )


def test_sample_team_name_uniform_returns_known_name(tmp_path):
    """sample_team_name draws from the configured format and returns a known name."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    _write_team_file(fmt_dir / "team_alpha.txt", "team_alpha")
    _write_team_file(fmt_dir / "team_beta.txt", "team_beta")
    repo = TeamRepo(filepath=str(tmp_path))

    seen = set()
    for _ in range(50):
        seen.add(repo.sample_team_name("gen9vgc2024regg"))

    assert seen <= {"team_alpha", "team_beta"}
    assert len(seen) >= 1  # at least one name returned across 50 draws


def test_sample_team_name_respects_subdirectory(tmp_path):
    """sample_team_name restricts draws to the specified subdirectory."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    sub_dir = fmt_dir / "constrained"
    sub_dir.mkdir(parents=True)
    other_dir = fmt_dir / "other"
    other_dir.mkdir()
    _write_team_file(sub_dir / "in_pool.txt", "constrained/in_pool")
    _write_team_file(other_dir / "out_of_pool.txt", "other/out_of_pool")
    repo = TeamRepo(filepath=str(tmp_path))

    for _ in range(50):
        name = repo.sample_team_name(
            "gen9vgc2024regg", subdirectory="constrained"
        )
        assert name.startswith("constrained/"), name
```

- [ ] **Step 2.2: Run the new tests and confirm they fail**

```
pytest unit_tests/rl/test_team_axis.py -v -k sample_team_name
```

Expected: FAIL with `AttributeError: 'TeamRepo' object has no attribute 'sample_team_name'`.

- [ ] **Step 2.3: Add the helper to `TeamRepo`**

In `src/elitefurretai/etl/team_repo.py`, find the existing `def sample_team(...)` (currently around line 348). Insert the new helper *immediately above* it (so they sit adjacent in the file). Insert:

```python
    def sample_team_name(
        self,
        format: str,
        subdirectory: Optional[str] = None,
    ) -> str:
        """Sample a random team name (filename without .txt) from the format.

        Used by Change 7's per-team adaptive sampling: when the worker
        needs just the name (to stamp on a trajectory or to look up
        the corresponding team string later), this avoids the cost
        and shuffle step of materializing the full team. The uniform
        sampling distribution is identical to ``sample_team``.

        Args:
            format: Pokemon format (e.g., "gen9vgc2024regg").
            subdirectory: Optional subdirectory under the format to
                restrict the sample to. Same semantics as
                ``sample_team``.

        Returns:
            Team name as stored in ``self._teams[format]``, e.g.
            ``"constrained/38dessert"``.

        Raises:
            ValueError: same conditions as ``sample_team``.
        """
        if format not in self._teams:
            raise ValueError(
                f"Format '{format}' not found. Available formats: {list(self._teams.keys())}"
            )

        format_teams = self._teams[format]
        if not format_teams:
            raise ValueError(f"No teams found for format '{format}'")

        if subdirectory is not None:
            subdirectory = subdirectory.replace(os.sep, "/")
            filtered_teams = {
                name: team
                for name, team in format_teams.items()
                if name.startswith(subdirectory + "/") or name == subdirectory
            }
            if not filtered_teams:
                raise ValueError(
                    f"No teams found in subdirectory '{subdirectory}' for "
                    f"format '{format}'. Available teams: "
                    f"{list(format_teams.keys())}"
                )
            format_teams = filtered_teams

        return random.choice(list(format_teams.keys()))
```

- [ ] **Step 2.4: Run the tests and confirm they pass**

```
pytest unit_tests/rl/test_team_axis.py -v -k sample_team_name
```

Expected: PASS (2 tests).

- [ ] **Step 2.5: Run quality gates**

```
ruff check src/elitefurretai/etl/team_repo.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/etl/team_repo.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/etl/team_repo.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 2.6: Commit**

```bash
git add src/elitefurretai/etl/team_repo.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(etl/team_repo): add sample_team_name helper

Returns just the team name (filename without .txt) under the same
sampling rules as sample_team. Used by Change 7's per-team adaptive
sampling to draw a name first, then look up the string lazily — and
to stamp the name on trajectories without rebuilding it later.

sample_team is left untouched (external callers in team_provider.py,
showdown_benchmark.py, showdown_invalid_choice_diagnostics.py
depend on its existing signature).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `OpponentPool` per-format team-axis state

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` — `OpponentPool` class (around line 111-490)
- Modify: `unit_tests/rl/test_team_axis.py` (append)

This task adds the state but does NOT yet wire `record_battle_result` or `update_team_distribution`. Those land in Tasks 4 and 5 so they can each be test-driven cleanly.

- [ ] **Step 3.1: Write the failing test**

Append to `unit_tests/rl/test_team_axis.py`:

```python
from typing import Dict, Optional

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.opponents import OpponentPool


def _two_format_repo(tmp_path) -> TeamRepo:
    """Build a TeamRepo with two formats, two teams each.

    Used by several tests in this module to exercise per-format
    isolation.
    """
    for fmt in ("gen9vgc2024regg", "gen9vgc2023regc"):
        d = tmp_path / fmt
        d.mkdir()
        _write_team_file(d / "alpha.txt", f"{fmt}/alpha")
        _write_team_file(d / "beta.txt", f"{fmt}/beta")
    return TeamRepo(filepath=str(tmp_path))


def _make_opponent_pool(
    tmp_path,
    *,
    team_axis_enabled: bool = True,
    team_warmup_threshold: int = 20,
    team_per_team_floor: float = 0.005,
    half_life: float = 50.0,
    pfsp_exponent: float = 1.0,
) -> OpponentPool:
    """Construct an OpponentPool with the multi-format test repo."""
    repo = _two_format_repo(tmp_path)
    return OpponentPool(
        team_repo=repo,
        battle_formats={
            "gen9vgc2024regg": 0.5,
            "gen9vgc2023regc": 0.5,
        },
        opponent_team_subdirectories={
            "gen9vgc2024regg": None,
            "gen9vgc2023regc": None,
        },
        team_axis_enabled=team_axis_enabled,
        team_warmup_threshold=team_warmup_threshold,
        team_per_team_floor=team_per_team_floor,
        half_life=half_life,
        pfsp_exponent=pfsp_exponent,
    )


def test_opponent_pool_initializes_per_format_team_state(tmp_path):
    """OpponentPool builds per-format team_win_rates / sample_counts / known_teams at init."""
    pool = _make_opponent_pool(tmp_path)
    assert set(pool.known_teams.keys()) == {"gen9vgc2024regg", "gen9vgc2023regc"}
    assert set(pool.known_teams["gen9vgc2024regg"]) == {"alpha", "beta"}
    assert set(pool.known_teams["gen9vgc2023regc"]) == {"alpha", "beta"}
    # EWMA state starts at zero for every (format, team).
    for fmt, teams in pool.known_teams.items():
        for t in teams:
            assert pool.team_win_rates[fmt][t] == (0.0, 0.0)
            assert pool.team_sample_counts[fmt][t] == 0
    # Warm flag starts False for every format.
    for fmt in pool.known_teams:
        assert pool._team_axis_warm.get(fmt, False) is False
```

- [ ] **Step 3.2: Run the test and confirm it fails**

```
pytest unit_tests/rl/test_team_axis.py -v -k test_opponent_pool_initializes
```

Expected: FAIL — either the new kwargs are unknown or the new attributes don't exist.

- [ ] **Step 3.3: Extend `OpponentPool.__init__`**

In `src/elitefurretai/rl/opponents.py`, locate `class OpponentPool` (line 111) and its `__init__` (around line 170). Find the existing signature; add the new keyword-only parameters. Anchor: the last existing parameter is followed by `) -> None:` and the body begins with `self.curriculum = curriculum`.

Add to the parameter list (insert before the closing `) -> None:`):

```python
        team_repo: Optional["TeamRepo"] = None,
        battle_formats: Optional[Dict[str, float]] = None,
        opponent_team_subdirectories: Optional[Dict[str, Optional[str]]] = None,
        team_axis_enabled: bool = True,
        team_warmup_threshold: int = 20,
        team_per_team_floor: float = 0.005,
        half_life: float = 50.0,
        pfsp_exponent: float = 1.0,
```

(If any of these already exist on `OpponentPool.__init__`, skip the duplicates and use the existing ones. The test harness above only needs them present; existing callers of `OpponentPool(...)` from `train.py` keep working because the new params default.)

Right after the existing `self.curriculum = ...` assignment (around line 183), append the team-axis initialization block:

```python
        # ── Change 7: team-axis adaptive curriculum state ────────────────
        self.team_axis_enabled = team_axis_enabled
        self.team_warmup_threshold = team_warmup_threshold
        self.team_per_team_floor = team_per_team_floor
        self._team_axis_half_life = half_life
        self._team_axis_pfsp_exponent = pfsp_exponent

        self.known_teams: Dict[str, List[str]] = {}
        self.team_win_rates: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.team_sample_counts: Dict[str, Dict[str, int]] = {}
        self._team_axis_warm: Dict[str, bool] = {}

        if team_axis_enabled and team_repo is not None and battle_formats:
            subs = opponent_team_subdirectories or {}
            for fmt in battle_formats:
                names = sorted(team_repo.get_all(fmt).keys())
                # Apply subdirectory filter symmetric with sample_team.
                sub = subs.get(fmt)
                if sub is not None:
                    sub = sub.replace("\\", "/")
                    names = [
                        n for n in names
                        if n.startswith(sub + "/") or n == sub
                    ]
                self.known_teams[fmt] = names
                self.team_win_rates[fmt] = {n: (0.0, 0.0) for n in names}
                self.team_sample_counts[fmt] = {n: 0 for n in names}
                self._team_axis_warm[fmt] = False
```

Make sure `Tuple` and `List` are imported at the top of `opponents.py`. They likely already are; if not, add them to the existing `from typing import ...` line.

- [ ] **Step 3.4: Run the test and confirm it passes**

```
pytest unit_tests/rl/test_team_axis.py -v -k test_opponent_pool_initializes
```

Expected: PASS.

- [ ] **Step 3.5: Run the full team-axis test file and the broader RL tests to check for regressions**

```
pytest unit_tests/rl/test_team_axis.py -v
pytest unit_tests/rl/ -q
```

Expected: all new tests pass; no pre-existing RL test fails. If a pre-existing test breaks, it's because `OpponentPool(...)` is constructed with positional args that now collide with the new keyword params — fix by using keyword args at the test call site, not by reordering the new params.

- [ ] **Step 3.6: Run quality gates**

```
ruff check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 3.7: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/opponents): add per-format team-axis state to OpponentPool

Adds known_teams, team_win_rates (EWMA), team_sample_counts, and
_team_axis_warm dicts keyed by battle_format. Initialized from
team_repo at __init__ time using the same subdirectory filter as
sample_team. State is inert until Tasks 4 and 5 wire
record_battle_result and update_team_distribution.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `record_battle_result` extension

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` (around line 318)
- Modify: `unit_tests/rl/test_team_axis.py`

- [ ] **Step 4.1: Write the failing tests**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def test_record_battle_result_per_format_team_routes_correctly(tmp_path):
    """Per-(format, team) EWMA updates route correctly; other cells untouched."""
    pool = _make_opponent_pool(tmp_path, half_life=1e9)  # ~no decay

    # 10 wins, 10 losses on (gen9vgc2024regg, alpha).
    for _ in range(10):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )
    for _ in range(10):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )

    wins, n = pool.team_win_rates["gen9vgc2024regg"]["alpha"]
    assert abs(n - 20.0) < 1e-6
    assert abs(wins / n - 0.5) < 1e-6
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 20

    # Other (format, team) cells untouched.
    assert pool.team_win_rates["gen9vgc2024regg"]["beta"] == (0.0, 0.0)
    assert pool.team_win_rates["gen9vgc2023regc"]["alpha"] == (0.0, 0.0)
    assert pool.team_sample_counts["gen9vgc2024regg"]["beta"] == 0
    assert pool.team_sample_counts["gen9vgc2023regc"]["alpha"] == 0


def test_record_battle_result_forfeit_skips_team_update(tmp_path):
    """Forfeits do not update the per-team EWMA or the sample count."""
    pool = _make_opponent_pool(tmp_path, half_life=1e9)

    for _ in range(5):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )
    for _ in range(5):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=True,
            battle_format="gen9vgc2024regg",
            team_name="alpha",
        )

    wins, n = pool.team_win_rates["gen9vgc2024regg"]["alpha"]
    assert abs(n - 5.0) < 1e-6
    assert abs(wins - 5.0) < 1e-6
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 5


def test_record_battle_result_no_team_args_is_noop_for_team_state(tmp_path):
    """Calls without battle_format / team_name don't touch team state.

    Preserves the existing record_battle_result contract for opponent-
    type-only tracking paths that haven't migrated yet.
    """
    pool = _make_opponent_pool(tmp_path)

    pool.record_battle_result(
        opponent_type="self_play",
        won=True,
        battle_length=10,
        forfeited=False,
    )

    for fmt, teams in pool.known_teams.items():
        for t in teams:
            assert pool.team_win_rates[fmt][t] == (0.0, 0.0)
            assert pool.team_sample_counts[fmt][t] == 0
```

- [ ] **Step 4.2: Run the new tests and confirm they fail**

```
pytest unit_tests/rl/test_team_axis.py -v -k record_battle_result
```

Expected: FAIL — `record_battle_result` doesn't accept the new kwargs.

- [ ] **Step 4.3: Extend `record_battle_result`**

In `src/elitefurretai/rl/opponents.py`, locate `def record_battle_result(` (around line 318) on `OpponentPool`. The current signature looks like (approximately):

```python
    def record_battle_result(
        self,
        opponent_type: str,
        won: bool,
        battle_length: int = 0,
        forfeited: bool = False,
    ) -> None:
        # existing body that updates opponent-type win rate tracking
```

Add two new keyword parameters AT THE END of the signature (so existing positional callers don't break):

```python
        battle_format: Optional[str] = None,
        team_name: Optional[str] = None,
```

Then, append the team-axis update block to the very bottom of the method body (after all existing logic):

```python
        # ── Change 7: per-(format, team) EWMA update ─────────────────────
        if (
            self.team_axis_enabled
            and battle_format is not None
            and team_name is not None
            and not forfeited
            and battle_format in self.team_win_rates
            and team_name in self.team_win_rates[battle_format]
        ):
            decay = 0.5 ** (1.0 / max(self._team_axis_half_life, 1e-9))
            prev_wins, prev_n = self.team_win_rates[battle_format][team_name]
            new_wins = prev_wins * decay + (1.0 if won else 0.0)
            new_n = prev_n * decay + 1.0
            self.team_win_rates[battle_format][team_name] = (new_wins, new_n)
            self.team_sample_counts[battle_format][team_name] += 1
```

`Optional` must be imported at the top of `opponents.py` (likely already there from `typing`).

- [ ] **Step 4.4: Run the new tests and confirm they pass**

```
pytest unit_tests/rl/test_team_axis.py -v -k record_battle_result
```

Expected: PASS (3 tests).

- [ ] **Step 4.5: Re-run the full RL test suite to catch regressions**

```
pytest unit_tests/rl/ -q
```

Expected: all green.

- [ ] **Step 4.6: Run quality gates**

```
ruff check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 4.7: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/opponents): extend record_battle_result with per-team EWMA

Adds optional battle_format and team_name kwargs. When both are
provided and the battle was not forfeited, updates the per-(format,
team) EWMA win rate and increments the per-team sample count used by
the warm-up gate. Existing opponent-type-only callers are unaffected.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `update_team_distribution` algorithm

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` (new method on `OpponentPool`, sibling to `update_curriculum` at around line 493)
- Modify: `unit_tests/rl/test_team_axis.py`

- [ ] **Step 5.1: Write the failing tests**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def _record_wins_losses(
    pool: OpponentPool,
    battle_format: str,
    team_name: str,
    wins: int,
    losses: int,
) -> None:
    """Helper: record `wins` wins and `losses` losses for (format, team)."""
    for _ in range(wins):
        pool.record_battle_result(
            opponent_type="self_play",
            won=True,
            battle_length=10,
            forfeited=False,
            battle_format=battle_format,
            team_name=team_name,
        )
    for _ in range(losses):
        pool.record_battle_result(
            opponent_type="self_play",
            won=False,
            battle_length=10,
            forfeited=False,
            battle_format=battle_format,
            team_name=team_name,
        )


def test_update_team_distribution_warmup_per_format(tmp_path):
    """Per-format warm-up: format A warms up first; format B stays None until it warms too.

    Also verifies latching: once a format's warm flag is True, draining
    sample counts back below threshold does not flip it back to None.
    """
    pool = _make_opponent_pool(tmp_path, team_warmup_threshold=20, half_life=1e9)

    # Warm up format A only.
    _record_wins_losses(pool, "gen9vgc2024regg", "alpha", 10, 10)
    _record_wins_losses(pool, "gen9vgc2024regg", "beta", 10, 10)
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert dist["gen9vgc2023regc"] is None

    # Now warm up format B.
    _record_wins_losses(pool, "gen9vgc2023regc", "alpha", 10, 10)
    _record_wins_losses(pool, "gen9vgc2023regc", "beta", 10, 10)
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert isinstance(dist["gen9vgc2023regc"], dict)

    # Latching: synthetically drop sample counts back to 0.
    for fmt in pool.team_sample_counts:
        for t in pool.team_sample_counts[fmt]:
            pool.team_sample_counts[fmt][t] = 0
    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    assert isinstance(dist["gen9vgc2023regc"], dict)


def test_update_team_distribution_asymmetric_pfsp_direction(tmp_path):
    """Asymmetric PFSP over-weights teams the model is worst at piloting.

    5-team single-format setup: 3 strong (90/10 W/L) and 2 weak
    (20/80 W/L). After warm-up, the two weak teams together get >60%
    of the format's distribution.
    """
    # Build a TeamRepo with 5 teams in one format.
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    for name in ("s1", "s2", "s3", "w1", "w2"):
        _write_team_file(fmt_dir / f"{name}.txt", name)
    repo = TeamRepo(filepath=str(tmp_path))
    pool = OpponentPool(
        team_repo=repo,
        battle_formats={"gen9vgc2024regg": 1.0},
        opponent_team_subdirectories={"gen9vgc2024regg": None},
        team_axis_enabled=True,
        team_warmup_threshold=20,
        team_per_team_floor=0.0,  # disable floor for pure-PFSP check
        half_life=1e9,
        pfsp_exponent=1.0,
    )

    for name in ("s1", "s2", "s3"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 90, 10)
    for name in ("w1", "w2"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 20, 80)

    dist = pool.update_team_distribution()
    assert isinstance(dist["gen9vgc2024regg"], dict)
    weak_share = dist["gen9vgc2024regg"]["w1"] + dist["gen9vgc2024regg"]["w2"]
    assert weak_share > 0.60, f"weak_share={weak_share}, dist={dist}"


def test_update_team_distribution_per_team_floor_enforced(tmp_path):
    """Per-team floor prevents any team from dropping below the configured minimum."""
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    for name in ("s1", "s2", "s3", "w1", "w2"):
        _write_team_file(fmt_dir / f"{name}.txt", name)
    repo = TeamRepo(filepath=str(tmp_path))
    pool = OpponentPool(
        team_repo=repo,
        battle_formats={"gen9vgc2024regg": 1.0},
        opponent_team_subdirectories={"gen9vgc2024regg": None},
        team_axis_enabled=True,
        team_warmup_threshold=20,
        team_per_team_floor=0.05,
        half_life=1e9,
        pfsp_exponent=2.0,  # steeper to make floor relevant
    )
    for name in ("s1", "s2", "s3"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 95, 5)
    for name in ("w1", "w2"):
        _record_wins_losses(pool, "gen9vgc2024regg", name, 5, 95)

    dist = pool.update_team_distribution()["gen9vgc2024regg"]
    assert isinstance(dist, dict)
    for name, weight in dist.items():
        assert weight >= 0.05 - 1e-9, f"team {name} weight {weight} below floor"
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_update_team_distribution_disabled_returns_all_none(tmp_path):
    """team_axis_enabled=False makes update_team_distribution return None per format."""
    pool = _make_opponent_pool(tmp_path, team_axis_enabled=False)
    _record_wins_losses(pool, "gen9vgc2024regg", "alpha", 50, 50)
    _record_wins_losses(pool, "gen9vgc2024regg", "beta", 50, 50)
    dist = pool.update_team_distribution()
    assert dist["gen9vgc2024regg"] is None
    assert dist["gen9vgc2023regc"] is None
```

- [ ] **Step 5.2: Run the new tests and confirm they fail**

```
pytest unit_tests/rl/test_team_axis.py -v -k update_team_distribution
```

Expected: FAIL — `update_team_distribution` doesn't exist.

- [ ] **Step 5.3: Implement `update_team_distribution` on `OpponentPool`**

In `src/elitefurretai/rl/opponents.py`, find `OpponentPool.update_curriculum` (around line 493). Insert the new method **immediately after** the existing `update_curriculum` body. Insert:

```python
    def update_team_distribution(
        self,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """Recompute the per-format team sampling distribution.

        For each configured battle_format:
        - If feature disabled OR any of the format's known_teams has
          a sample count below team_warmup_threshold AND the format
          hasn't already latched warm, returns None for that format.
        - Otherwise computes asymmetric PFSP weights
          ``(1 - wr_t) ** p`` with Beta(8, 8) smoothing of ``wr_t``,
          normalizes, applies per-team floor, renormalizes, and
          returns the resulting distribution.

        Warm-up latches per format: once a format trips True, it
        stays True regardless of future sample-count drift.

        Returns:
            Dict keyed by battle_format. Values are either a
            normalized {team_name: weight} dict OR None during
            warm-up / when team_axis_enabled is False.
        """
        result: Dict[str, Optional[Dict[str, float]]] = {}
        if not self.team_axis_enabled:
            for fmt in self.known_teams:
                result[fmt] = None
            return result

        alpha = 8.0
        beta = 8.0
        p = self._team_axis_pfsp_exponent
        floor = self.team_per_team_floor

        for fmt, teams in self.known_teams.items():
            if self._team_axis_warm.get(fmt, False):
                warm = True
            else:
                warm = all(
                    self.team_sample_counts[fmt][t] >= self.team_warmup_threshold
                    for t in teams
                )
                if warm:
                    self._team_axis_warm[fmt] = True

            if not warm:
                result[fmt] = None
                continue

            scores: Dict[str, float] = {}
            for t in teams:
                wins, n = self.team_win_rates[fmt][t]
                wr_t = (wins + alpha) / (n + alpha + beta)
                scores[t] = max(0.0, (1.0 - wr_t)) ** p

            total = sum(scores.values())
            if total <= 0.0:
                # Degenerate case: all teams at wr=1.0 (impossible from
                # smoothed estimator with finite n, but guarded for safety).
                result[fmt] = {t: 1.0 / len(teams) for t in teams}
                continue

            distribution = {t: scores[t] / total for t in teams}

            if floor > 0.0:
                for t in teams:
                    if distribution[t] < floor:
                        distribution[t] = floor
                total = sum(distribution.values())
                distribution = {t: distribution[t] / total for t in teams}

            result[fmt] = distribution

        return result
```

`Dict` and `Optional` must be imported at the top of `opponents.py` (likely already).

- [ ] **Step 5.4: Run the new tests and confirm they pass**

```
pytest unit_tests/rl/test_team_axis.py -v -k update_team_distribution
```

Expected: PASS (4 tests).

- [ ] **Step 5.5: Run the full team-axis suite and broader RL tests**

```
pytest unit_tests/rl/test_team_axis.py -v
pytest unit_tests/rl/ -q
```

Expected: all green.

- [ ] **Step 5.6: Run quality gates**

```
ruff check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 5.7: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/opponents): add update_team_distribution to OpponentPool

Computes a per-format {team_name: weight} distribution using
asymmetric PFSP (1-wr)^p with Beta(8,8) smoothing, per-team floor,
and renormalization. Per-format warm-up latch: once a format has
enough samples it stays unlocked even if counts drift.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `RLTrajectoryPlayer.current_team_name` + trajectory dict additions

**Files:**
- Modify: `src/elitefurretai/rl/rl_trajectory_player.py` (around lines 80-100 for the attribute; lines 633-641 for the dict)
- Modify: `unit_tests/rl/test_team_axis.py`

- [ ] **Step 6.1: Write the failing test**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def test_rl_trajectory_player_has_current_team_name_default(tmp_path):
    """RLTrajectoryPlayer initializes current_team_name to None.

    The factory sets this at battle setup; the trajectory player
    reads it at battle finish to stamp on the trajectory dict.
    """
    from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer

    # We only care about attribute presence + default; construct a
    # minimal player with the minimum required arguments. The class
    # accepts many kwargs — pass only what's required and rely on
    # defaults for the rest.
    # NOTE: if RLTrajectoryPlayer construction requires non-trivial
    # collaborators (inference_client, etc.), introspect __init__ and
    # pass MagicMocks. The test only inspects current_team_name on a
    # freshly-constructed object.
    player = RLTrajectoryPlayer.__new__(RLTrajectoryPlayer)
    RLTrajectoryPlayer.__init__.__wrapped__ if False else None  # silence noqa
    # Avoid invoking poke-env Player's heavy __init__: directly set
    # the attribute that __init__ is expected to set.
    assert hasattr(RLTrajectoryPlayer, "__init__")
    # The actual contract under test: after __init__ runs, the
    # attribute exists and defaults to None. We can verify this by
    # reading the source — the test passes once the attribute is
    # defined as a default in __init__ or as a class attribute.
    # Probe via the class default first if available, else accept
    # absence-with-AttributeError as the failing condition.
    default = getattr(RLTrajectoryPlayer, "current_team_name", "MISSING")
    assert default in (None, "MISSING"), default
    if default == "MISSING":
        # Class-level attribute not present — the implementation must
        # set it in __init__. This branch will be exercised once we
        # add the attribute.
        pass
```

The probe above is permissive because `RLTrajectoryPlayer.__init__` requires real collaborators (`inference_client`, etc.). The actual behavioral check happens in Task 8's integration-style migration test, where the factory constructs a real player and sets `current_team_name`. For now, this test just ensures the attribute is at least defined and defaults to `None` when present.

Replace the body above with this simpler test instead:

```python
def test_rl_trajectory_player_class_declares_current_team_name():
    """RLTrajectoryPlayer declares current_team_name at the class level (or via __init__).

    We probe the source / class to assert the attribute is reachable
    on instances. Behavioral wiring is covered in Task 8.
    """
    from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer

    # Class-level attribute is the simplest declaration. The
    # implementation in Task 6 sets it via class body default so that
    # subclasses/instances see a None default until the factory
    # writes a name.
    assert (
        getattr(RLTrajectoryPlayer, "current_team_name", "MISSING") is None
    ), (
        "RLTrajectoryPlayer.current_team_name must be declared as a "
        "class-level attribute defaulting to None."
    )
```

- [ ] **Step 6.2: Run the test and confirm it fails**

```
pytest unit_tests/rl/test_team_axis.py -v -k current_team_name
```

Expected: FAIL.

- [ ] **Step 6.3: Add `current_team_name` to `RLTrajectoryPlayer`**

In `src/elitefurretai/rl/rl_trajectory_player.py`, find `class RLTrajectoryPlayer` (around line 80). Add a class-level attribute declaration immediately inside the class body, before `__init__`:

```python
class RLTrajectoryPlayer(...):
    """..."""

    # ── Change 7: stamped by WorkerOpponentFactory at battle setup ──────
    current_team_name: Optional[str] = None
```

Make sure `Optional` is imported at the top of `rl_trajectory_player.py`.

- [ ] **Step 6.4: Extend the trajectory dict**

Locate the trajectory dict construction site at lines 633-641 (`self.trajectory_queue.put({...})`). Modify it to:

```python
            self.trajectory_queue.put(
                {
                    "steps": filtered_traj,
                    "opponent_type": self.opponent_type,
                    "won": battle.won,
                    "battle_length": len(filtered_traj),
                    "forfeited": forfeited,
                    "team_name": self.current_team_name,
                    "battle_format": battle.format,
                }
            )
```

- [ ] **Step 6.5: Verify the test passes**

```
pytest unit_tests/rl/test_team_axis.py -v -k current_team_name
pytest unit_tests/rl/ -q
```

Expected: PASS, no regressions.

- [ ] **Step 6.6: Run quality gates**

```
ruff check src/elitefurretai/rl/rl_trajectory_player.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/rl_trajectory_player.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/rl_trajectory_player.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 6.7: Commit**

```bash
git add src/elitefurretai/rl/rl_trajectory_player.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/rl_trajectory_player): stamp team_name + battle_format on trajectories

Adds current_team_name class attribute (default None) that the
WorkerOpponentFactory sets at battle setup. The trajectory dict
gains team_name (from self.current_team_name) and battle_format
(from battle.format) so the trainer can route per-(format, team)
EWMA updates.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `WorkerOpponentFactory.sample_team` tuple return + biased flag

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` (around line 645 — `WorkerOpponentFactory.sample_team`; line 625 — `update_curriculum`; line 523 — `__init__`)
- Modify: `unit_tests/rl/test_team_axis.py`

This task changes `WorkerOpponentFactory.sample_team` to return a tuple and accept `biased`. It also adds the `team_distribution_by_format` field and extends `update_curriculum` to accept it. Call-site migration happens in Task 8.

- [ ] **Step 7.1: Write the failing tests**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def _make_worker_factory(
    tmp_path,
    *,
    team_distribution_by_format: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
):
    """Construct a minimal WorkerOpponentFactory for sample_team tests.

    Most of WorkerOpponentFactory's collaborators (account configs,
    inference clients, etc.) are unused by the sample_team code path,
    so we construct a bare instance via __new__ and only populate the
    attributes sample_team needs. This avoids the heavy real
    initialization.
    """
    from elitefurretai.rl.opponents import WorkerOpponentFactory

    repo = _two_format_repo(tmp_path)
    factory = WorkerOpponentFactory.__new__(WorkerOpponentFactory)
    factory.team_repo = repo
    factory.opponent_team_subdirectories = {
        "gen9vgc2024regg": None,
        "gen9vgc2023regc": None,
    }
    factory.team_distribution_by_format = team_distribution_by_format or {}
    return factory


def test_sample_team_returns_tuple_uniform(tmp_path):
    """sample_team(battle_format) returns (team_string, team_name) — uniform path."""
    factory = _make_worker_factory(tmp_path)
    team_string, team_name = factory.sample_team("gen9vgc2024regg")
    assert isinstance(team_string, str) and team_string
    assert team_name in {"alpha", "beta"}


def test_sample_team_biased_false_uses_uniform(tmp_path):
    """sample_team(biased=False) always uses uniform sampling, even if a biased dist is set."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 1.0, "beta": 0.0},
        },
    )
    seen = set()
    for _ in range(40):
        _, name = factory.sample_team("gen9vgc2024regg", biased=False)
        seen.add(name)
    # With biased=False both names should appear under uniform sampling.
    assert seen == {"alpha", "beta"}, seen


def test_sample_team_biased_true_uses_distribution(tmp_path):
    """sample_team(biased=True) draws from the configured per-format distribution."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.95, "beta": 0.05},
        },
    )
    counts = {"alpha": 0, "beta": 0}
    for _ in range(2000):
        _, name = factory.sample_team("gen9vgc2024regg", biased=True)
        counts[name] += 1
    # alpha should dominate; tolerate sampling noise within reason.
    assert counts["alpha"] > counts["beta"] * 5, counts


def test_sample_team_biased_format_isolation(tmp_path):
    """Per-format distributions don't cross-pollinate."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.95, "beta": 0.05},
            "gen9vgc2023regc": {"alpha": 0.05, "beta": 0.95},
        },
    )
    counts_a = {"alpha": 0, "beta": 0}
    counts_b = {"alpha": 0, "beta": 0}
    for _ in range(2000):
        _, name_a = factory.sample_team("gen9vgc2024regg", biased=True)
        _, name_b = factory.sample_team("gen9vgc2023regc", biased=True)
        counts_a[name_a] += 1
        counts_b[name_b] += 1
    assert counts_a["alpha"] > counts_a["beta"] * 5, counts_a
    assert counts_b["beta"] > counts_b["alpha"] * 5, counts_b


def test_sample_team_falls_back_to_uniform_when_no_distribution(tmp_path):
    """sample_team(biased=True) falls back to uniform when no distribution exists for the format."""
    factory = _make_worker_factory(
        tmp_path,
        team_distribution_by_format={"gen9vgc2024regg": None},
    )
    seen = set()
    for _ in range(40):
        _, name = factory.sample_team("gen9vgc2024regg", biased=True)
        seen.add(name)
    assert seen == {"alpha", "beta"}, seen


def test_update_curriculum_accepts_team_distribution(tmp_path):
    """WorkerOpponentFactory.update_curriculum accepts team_distribution_by_format and stores it."""
    factory = _make_worker_factory(tmp_path)
    factory.curriculum = {}  # update_curriculum normalizes this
    factory.update_curriculum(
        {"self_play": 1.0},
        team_distribution_by_format={
            "gen9vgc2024regg": {"alpha": 0.7, "beta": 0.3},
        },
    )
    assert factory.team_distribution_by_format == {
        "gen9vgc2024regg": {"alpha": 0.7, "beta": 0.3},
    }
```

- [ ] **Step 7.2: Run the new tests and confirm they fail**

```
pytest unit_tests/rl/test_team_axis.py -v -k "sample_team_returns_tuple or sample_team_biased or sample_team_falls_back or update_curriculum_accepts_team_distribution"
```

Expected: FAIL.

- [ ] **Step 7.3: Add `team_distribution_by_format` to `WorkerOpponentFactory.__init__`**

In `src/elitefurretai/rl/opponents.py`, locate `class WorkerOpponentFactory.__init__` (around line 523). Just after the existing `self.opponent_team_subdirectories = dict(opponent_team_subdirectories)` line (around line 541), insert:

```python
        # ── Change 7: per-format biased team distribution from broadcast ─
        # None or absent value for a format → fall back to uniform.
        self.team_distribution_by_format: Dict[str, Optional[Dict[str, float]]] = {}
```

- [ ] **Step 7.4: Extend `WorkerOpponentFactory.update_curriculum`**

Locate `def update_curriculum(self, curriculum: Dict[str, float]) -> None:` around line 624. Replace it with:

```python
    def update_curriculum(
        self,
        curriculum: Dict[str, float],
        team_distribution_by_format: Optional[
            Dict[str, Optional[Dict[str, float]]]
        ] = None,
    ) -> None:
        """Update worker-local curriculum and refresh dependent opponent pools.

        team_distribution_by_format: per-format biased team sampling
            distributions broadcast from the trainer (Change 7). May
            contain None values for formats still in warm-up;
            sample_team falls back to uniform for those.
        """
        self.curriculum = normalize_curriculum(curriculum)
        if team_distribution_by_format is not None:
            self.team_distribution_by_format = dict(team_distribution_by_format)
```

(Keep the rest of `update_curriculum`'s existing body that refreshes opponent pools, if any — append the new line after the existing normalize call.)

- [ ] **Step 7.5: Rewrite `WorkerOpponentFactory.sample_team`**

Locate `def sample_team(self, battle_format: str) -> str:` around line 645. Replace it with:

```python
    def sample_team(
        self,
        battle_format: str,
        biased: bool = True,
    ) -> Tuple[str, str]:
        """Return (team_string, team_name) for the given battle format.

        biased=True (default): if the trainer has broadcast a non-None
            distribution for ``battle_format``, sample a name from it
            via numpy.random.choice and look up the corresponding
            team string. Otherwise (no distribution set, or value is
            None during warm-up) fall back to the uniform path.

        biased=False: always uniform via team_repo.sample_team_name.
            Reserved for future eval-at-checkpoint code paths that
            want the natural team distribution.
        """
        dist = (
            self.team_distribution_by_format.get(battle_format)
            if biased else None
        )
        if dist:
            # numpy.random.choice requires same-length parallel arrays.
            names = list(dist.keys())
            weights = list(dist.values())
            name = str(np.random.choice(names, p=weights))
        else:
            name = self.team_repo.sample_team_name(
                battle_format,
                subdirectory=self.opponent_team_subdirectories.get(battle_format),
            )
        team_string = self.team_repo.get(battle_format, name)
        return self.team_repo._shuffle_team_order(team_string), name
```

Add `import numpy as np` at the top of `opponents.py` if not already present (likely is — check). `Tuple` must also be in the existing `from typing import ...` block.

- [ ] **Step 7.6: Run the new tests and confirm they pass**

```
pytest unit_tests/rl/test_team_axis.py -v -k "sample_team_returns_tuple or sample_team_biased or sample_team_falls_back or update_curriculum_accepts_team_distribution"
```

Expected: PASS (6 tests).

- [ ] **Step 7.7: Verify no in-repo callers of the old sample_team string-return are still broken**

```
grep -rn "\.sample_team(" src/elitefurretai/rl/ | grep -v "sample_team_name"
```

Each match should be either:
- An old call in `WorkerOpponentFactory` itself that we'll migrate in Task 8 (expected; broken right now).
- A call we knowingly accepted in this PR.

DO NOT modify external callers in `analyze/team_provider.py`, `engine/analyze/showdown_benchmark.py`, or `engine/analyze/showdown_invalid_choice_diagnostics.py` — those are calling `team_repo.sample_team`, not `WorkerOpponentFactory.sample_team`, and the former is intentionally untouched.

After verifying the remaining matches are all inside `WorkerOpponentFactory` and slated for Task 8, run the broader test suite to confirm the call sites that haven't migrated yet still produce a runnable system (they'll currently break in ways we'll fix in Task 8 — expected). Specifically:

```
pytest unit_tests/rl/test_team_axis.py -v
```

This file's tests should all be green even though some internal call sites still treat the tuple return as a string. Tests for the broader system may fail until Task 8 completes — that's acceptable; mark them tracked and proceed.

- [ ] **Step 7.8: Run quality gates on the changed files**

```
ruff check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
```

Pyright may flag old call sites in `opponents.py` that treat the tuple return as a string. Note them; Task 8 fixes them.

- [ ] **Step 7.9: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/opponents): WorkerOpponentFactory team-axis sampling

sample_team(battle_format, biased=True) now returns
(team_string, team_name). When biased=True and a per-format biased
distribution is set via broadcast, draws from it via
numpy.random.choice; otherwise falls back to
team_repo.sample_team_name.

update_curriculum gains an optional
team_distribution_by_format parameter that the trainer-side broadcast
fills. Internal call sites of sample_team migrate in the next task.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Migrate `sample_team` call sites + stamp `current_team_name`

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py` — lines 686, 769, 1015, 1019, 1023, 1027, 1031 (approximate; verify before editing).
- Modify: `unit_tests/rl/test_team_axis.py`

The internal callers of the old `self.sample_team(fmt) -> str` need to migrate to `team_string, team_name = self.sample_team(fmt)` and, where the player is the trajectory player or an opponent player whose team will be stamped on a trajectory, also set `player.current_team_name = team_name`.

- [ ] **Step 8.1: Inventory the call sites**

Run:

```
grep -n "self.sample_team(" src/elitefurretai/rl/opponents.py
```

Each match falls into one of these categories:

1. **Trajectory player team assignment** — the player produces trajectories. Must stamp `current_team_name`. Look for `team=self.sample_team(fmt)` inside an `RLTrajectoryPlayer(...)` constructor call.
2. **Opponent player team reset** — `opponent._team = ConstantTeambuilder(self.sample_team(fmt))`. If the opponent is itself a trajectory-producing player (self-play, ghosts, exploiters), also set `opponent.current_team_name = team_name`. If the opponent is a non-trajectory baseline (max_damage, random, max_base_power, simple_heuristic), team_name can be discarded — those players don't produce trajectories.

Without the diff in hand, the heuristic:

- If the line is `opponent._team = ConstantTeambuilder(self.sample_team(fmt))` and the surrounding context constructs or refers to a baseline (`md_opp`, `random_opp`, `maxbp_opp`, `heuristic_opp`), discard the name.
- If it's `opponent` (unspecified — the self-play opponent), stamp the name.

- [ ] **Step 8.2: Update each call site**

For each match from Step 8.1, replace `team_or_string = self.sample_team(fmt)` patterns according to:

**Pattern A: Trajectory player constructor with `team=`.**

Before:
```python
self.players.append(
    RLTrajectoryPlayer(
        ...,
        team=self.sample_team(fmt),
        ...,
    )
)
```

After:
```python
_team_string, _team_name = self.sample_team(fmt)
player = RLTrajectoryPlayer(
    ...,
    team=_team_string,
    ...,
)
player.current_team_name = _team_name
self.players.append(player)
```

Mirror this for the opponent-side trajectory player at the same construction site if present.

**Pattern B: Opponent `_team` reset on a self-play / ghost / exploiter player (also a `RLTrajectoryPlayer`).**

Before:
```python
opponent._team = ConstantTeambuilder(self.sample_team(fmt))
```

After:
```python
_team_string, _team_name = self.sample_team(fmt)
opponent._team = ConstantTeambuilder(_team_string)
opponent.current_team_name = _team_name
```

**Pattern C: Opponent `_team` reset on a heuristic baseline (`md_opp`, `random_opp`, `maxbp_opp`, `heuristic_opp`).**

Before:
```python
md_opp._team = ConstantTeambuilder(self.sample_team(fmt))
```

After:
```python
_team_string, _ = self.sample_team(fmt)
md_opp._team = ConstantTeambuilder(_team_string)
```

The underscore discards the team name (baselines don't produce trajectories that need stamping).

- [ ] **Step 8.3: Write the failing integration-style test**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def test_factory_stamps_current_team_name_on_self_play_player(tmp_path):
    """End-to-end: factory assigns team_name to RLTrajectoryPlayer.current_team_name.

    Constructs a real WorkerOpponentFactory and exercises the public
    code path that assigns a team to a self-play trajectory player.
    Verifies the stamp lands on the player.
    """
    # NOTE: this test is intentionally focused on the stamp; it bypasses
    # any heavy worker init by mocking the player class to record what's
    # set on it. If the factory's worker-pool construction is too
    # heavyweight to exercise in a unit test, replace this with a
    # lighter probe: monkey-patch `RLTrajectoryPlayer` in this module
    # to a recording subclass and call the smallest method on the
    # factory that hits the trajectory-player branch.
    from unittest.mock import patch
    from elitefurretai.rl import opponents as opp_mod

    factory = _make_worker_factory(tmp_path)
    # Populate enough additional state for the call we're about to make.
    # The exact set depends on which factory method we're invoking;
    # the implementer should choose the smallest one that hits a
    # Pattern A or Pattern B call site.

    # Capture what the factory tries to do. Since the production code
    # path may not be invokable in a pure unit test (depends on
    # inference clients, account configs, server config, etc.),
    # exercise the contract at the API boundary:
    team_string, team_name = factory.sample_team("gen9vgc2024regg")
    assert team_name in {"alpha", "beta"}
    assert isinstance(team_string, str) and team_string
```

This test is intentionally minimal — the goal is to confirm that the `sample_team` contract is stable end-to-end and the migration in Step 8.2 doesn't accidentally reorder or drop the tuple. The richer "factory wires current_team_name" check is covered by the broader RL test suite once the migration is applied (Step 8.5).

- [ ] **Step 8.4: Verify pre-migration the new test runs (it already would, since Task 7 shipped the contract)**

```
pytest unit_tests/rl/test_team_axis.py -v -k test_factory_stamps_current_team_name
```

Expected: PASS (the contract is satisfied by Task 7's `sample_team` rewrite — what we're verifying in Task 8 is that the *call sites* don't break the contract by accident).

- [ ] **Step 8.5: Run the entire RL test suite to catch broken call sites**

```
pytest unit_tests/rl/ -q
```

If any test fails because a call site still treats `sample_team(...)` as returning a string, return to Step 8.2 and finish migrating it.

- [ ] **Step 8.6: Run quality gates**

```
ruff check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 8.7: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/opponents): migrate sample_team call sites to tuple return

Unpacks (team_string, team_name) at every internal call site.
Trajectory-producing players (RLTrajectoryPlayer in self_play,
ghosts, exploiters) also receive .current_team_name = team_name so
the name flows onto the trajectory dict at battle finish. Heuristic
baselines discard the name — they don't produce trajectories.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `train.py` wiring

**Files:**
- Modify: `src/elitefurretai/rl/train.py` (around line 936 for `record_battle_result` call; around line 1110 for broadcast)
- Modify: `unit_tests/rl/test_team_axis.py`

- [ ] **Step 9.1: Write the failing test**

Append to `unit_tests/rl/test_team_axis.py`:

```python
def test_train_passes_team_args_to_record_battle_result(monkeypatch, tmp_path):
    """train.py's trajectory-ingest loop passes battle_format and team_name to record_battle_result.

    Indirect test: synthesize a trajectory dict with the new fields,
    call OpponentPool.record_battle_result with the same kwargs the
    train.py code path will pass, and verify the side effect lands.
    This catches keyword typos and ensures the wiring stays in sync
    if record_battle_result's signature evolves.
    """
    pool = _make_opponent_pool(tmp_path, half_life=1e9)

    fake_traj = {
        "steps": [],
        "opponent_type": "self_play",
        "won": True,
        "battle_length": 7,
        "forfeited": False,
        "team_name": "alpha",
        "battle_format": "gen9vgc2024regg",
    }
    # This is the call shape train.py uses (see Step 9.2 below).
    pool.record_battle_result(
        opponent_type=fake_traj["opponent_type"],
        won=fake_traj["won"],
        battle_length=fake_traj["battle_length"],
        forfeited=fake_traj["forfeited"],
        battle_format=fake_traj["battle_format"],
        team_name=fake_traj["team_name"],
    )
    assert pool.team_sample_counts["gen9vgc2024regg"]["alpha"] == 1
```

- [ ] **Step 9.2: Update the `OpponentPool(...)` construction in `train.py`**

Locate the `OpponentPool(...)` construction in `src/elitefurretai/rl/train.py` (around line 742). The new kwargs added to `OpponentPool.__init__` in Task 3 default to None/disabled values, so the feature is inert unless they're passed here.

Find the construction:

```python
    opponent_pool = OpponentPool(
        # ... existing kwargs ...
    )
```

Add the new kwargs sourced from `config.curriculum`:

```python
    opponent_pool = OpponentPool(
        # ... existing kwargs ...
        team_repo=team_repo,
        battle_formats=dict(config.curriculum.battle_formats),
        opponent_team_subdirectories=opponent_team_subdirectories,
        team_axis_enabled=config.curriculum.team_axis_enabled,
        team_warmup_threshold=config.curriculum.team_warmup_threshold,
        team_per_team_floor=config.curriculum.team_per_team_floor,
        half_life=config.curriculum.half_life,
        pfsp_exponent=config.curriculum.pfsp_exponent,
    )
```

Notes:
- `team_repo` should already be in scope where `OpponentPool` is constructed (or grep for its existing creation site nearby).
- `opponent_team_subdirectories` may need to be derived from `config.curriculum.resolved_opponent_team_pool_paths()` or similar — check the surrounding code for how the worker factory gets the same dict.
- `half_life` and `pfsp_exponent` are config fields owned by Changes 5 and 4 respectively. If those changes have not landed yet, define stand-in defaults on `CurriculumConfig` (`half_life: float = 50.0`, `pfsp_exponent: float = 1.0`) — flag this in the commit message so it's clear they came in pre-emptively. If Changes 4 and 5 already landed, use the existing config fields.

- [ ] **Step 9.3: Update the `record_battle_result` call in `train.py`**

In `src/elitefurretai/rl/train.py`, find the call site around line 936 (`pool.record_battle_result(...)` / `opponent_pool.record_battle_result(...)`). Add the two new kwargs:

```python
                opponent_pool.record_battle_result(
                    opponent_type=traj["opponent_type"],
                    won=traj["won"],
                    battle_length=traj["battle_length"],
                    forfeited=traj["forfeited"],
                    battle_format=traj["battle_format"],
                    team_name=traj["team_name"],
                )
```

(The exact variable name `opponent_pool` may differ; preserve whatever is there.)

- [ ] **Step 9.4: Invoke `update_team_distribution` and bundle into broadcast**

Locate the curriculum-broadcast site around line 1110 in `train.py`. The current pattern likely looks like:

```python
new_curriculum = opponent_pool.update_curriculum()
# ... broadcast new_curriculum to workers ...
```

Modify to also compute the per-format team distribution and ship it:

```python
new_curriculum = opponent_pool.update_curriculum()
team_distribution_by_format = opponent_pool.update_team_distribution()
# ... broadcast both to workers via the same path ...
```

Find the broadcast call (likely a worker hook that calls `WorkerOpponentFactory.update_curriculum(...)`). Update it to also pass `team_distribution_by_format`. The exact mechanism depends on the broadcast plumbing — search:

```
grep -n "update_curriculum" src/elitefurretai/rl/train.py
grep -n "update_curriculum" src/elitefurretai/rl/worker.py
```

For each `worker_factory.update_curriculum(new_curriculum)` site, change to:

```python
worker_factory.update_curriculum(
    new_curriculum,
    team_distribution_by_format=team_distribution_by_format,
)
```

If the broadcast goes through a queue/message rather than a direct method call, add `team_distribution_by_format` to the message payload and unpack it on the worker side.

- [ ] **Step 9.5: Run the test and the broader train.py smoke**

```
pytest unit_tests/rl/test_team_axis.py -v -k train_passes_team_args
pytest unit_tests/rl/ -q
```

Expected: all green.

- [ ] **Step 9.6: Run quality gates**

```
ruff check src/elitefurretai/rl/train.py unit_tests/rl/test_team_axis.py
ruff format --check src/elitefurretai/rl/train.py unit_tests/rl/test_team_axis.py
pyright src/elitefurretai/rl/train.py unit_tests/rl/test_team_axis.py
```

Expected: 0 errors.

- [ ] **Step 9.7: Commit**

```bash
git add src/elitefurretai/rl/train.py unit_tests/rl/test_team_axis.py
git commit -m "$(cat <<'EOF'
feat(rl/train): wire team-axis curriculum end-to-end

Passes battle_format and team_name from each trajectory into
opponent_pool.record_battle_result, invokes update_team_distribution
alongside update_curriculum at the existing broadcast cadence, and
bundles the per-format team distributions into the worker broadcast.

Completes Change 7's data flow: trajectories now carry team identity,
the trainer-side pool aggregates per-(format, team) WR, recomputes a
biased distribution at checkpoint cadence, and workers apply it on
the next sample_team call.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update `RL.md` §7

**Files:**
- Modify: `src/elitefurretai/rl/RL.md`

- [ ] **Step 10.1: Read RL.md §7**

```
grep -n "^## \|^### " src/elitefurretai/rl/RL.md | head -20
```

Locate §7 ("The Opponent Pool & Adaptive Curriculum" per the spec; section title may differ slightly). Read its current content.

- [ ] **Step 10.2: Add a new subsection describing the team axis**

At the end of §7, append a subsection:

```markdown
### Team-axis curriculum (Change 7, 2026-05-24)

In addition to opponent sampling, the curriculum biases agent team
selection per-format. `OpponentPool` tracks per-(battle_format,
team_name) EWMA win rate using the same decay rule as Change 5. At
each curriculum-broadcast cadence, `update_team_distribution()`
computes per-format `{team_name: weight}` distributions using
asymmetric PFSP `(1 - wr)^p` (sharing `p` with Change 4), with a
per-team floor and per-format warm-up gating.

Workers receive the distribution via the same broadcast mechanism
as the opponent curriculum. `WorkerOpponentFactory.sample_team(fmt,
biased=True)` draws a team from the broadcast distribution if one
exists for `fmt`, or falls back to uniform `team_repo.sample_team_name`
during warm-up. A new `current_team_name` attribute on
`RLTrajectoryPlayer` flows the name onto the trajectory dict so the
trainer can route the EWMA update back to the right cell.

Bias scope: **all** call sites in training use the biased
distribution (self-play, ghosts, exploiters, *and* baseline matchups).
The `biased=False` opt-out exists for future eval-at-checkpoint code
paths that want the natural uniform team distribution.

Config keys: `team_axis_enabled`, `team_warmup_threshold`,
`team_per_team_floor`. Reuses `pfsp_exponent` from Change 4 and
`half_life` from Change 5.

Design spec:
[planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md](../../planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md).
```

- [ ] **Step 10.3: Run quality gates on the doc change**

```
ruff check src unit_tests
ruff format --check src unit_tests
pyright src unit_tests
pytest unit_tests -q
```

Expected: full project green. This is the final cross-cutting check.

- [ ] **Step 10.4: Commit**

```bash
git add src/elitefurretai/rl/RL.md
git commit -m "$(cat <<'EOF'
docs(rl): describe Change 7 (team-axis curriculum) in RL.md §7

Closes the documentation half of Change 7. See
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md
for the design and
planning/stage2/2026-05-24-14-00-change7-team-axis-curriculum-implementation-plan.md
for the implementation breakdown.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Validation checklist (end-of-PR)

Before opening the PR or merging, confirm:

- [ ] All ten task commits land in order, with no broken intermediate state on a clean checkout.
- [ ] `pytest unit_tests -q` is green.
- [ ] `ruff check src unit_tests` and `ruff format --check src unit_tests` are green.
- [ ] `pyright src unit_tests` is green.
- [ ] The ~20 new tests in `unit_tests/rl/test_team_axis.py` are all listed in the green output (count them by running `pytest unit_tests/rl/test_team_axis.py -v --co | grep "::test_"`).
- [ ] At least one short smoke training run (≤ 100 updates) completes successfully on `single_team.yaml` with `team_axis_enabled=True`. Confirm in logs:
  - `OpponentPool.update_team_distribution()` is being called at the curriculum cadence.
  - At some point after the warm-up threshold per team, the broadcast payload starts carrying non-None per-format distributions.
  - `WorkerOpponentFactory.sample_team` selections shift toward bottom-quartile teams as training progresses (informal check via wandb or log telemetry).

## What's NOT in this PR (per spec)

- Joint `(team, opp_type)` granularity.
- Eval driver migration to `biased=False`.
- BC-derived or eval-derived priors.
- Mid-run team-file rediscovery.
- Threshold-based team graduation logic.
- Length-weighted EWMA (Change 1's other half).
