# Curriculum Unification & Team-Repo / Team-Name Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five interlocking cleanups uncovered while reviewing Change 7 (team-axis adaptive curriculum): (1) collapse `TeamRepo.sample_team` / `sample_team_name` duplication, (2) extract a single adaptive-distribution primitive shared by team-axis and agent-axis curriculum updates, (3) fix `RLTrajectoryPlayer.current_team_name` so concurrent battles don't cross-contaminate, (4) replace ad-hoc `half_life` / `pfsp_exponent` fields with a `AdaptiveAxisConfig` sub-dataclass used by both axes, (5) rename to `adaptive_team_axis` / `adaptive_agent_axis`.

**Architecture:** A new module [`src/elitefurretai/rl/rl_utils.py`](src/elitefurretai/rl/rl_utils.py) holds two pure functions: `adaptive_score(wins, n, …)` (Beta-smoothed PFSP + asymmetric weakness blend) and `adaptive_distribution(scores, …)` (water-filling floor + renormalization with optional base-curriculum blend). Both `update_curriculum` (agent-axis) and `update_team_distribution` (team-axis) become thin callers. `CurriculumConfig` gains nested `adaptive_team_axis: AdaptiveAxisConfig` and `adaptive_agent_axis: AdaptiveAxisConfig` blocks. Agent-axis switches from sliding-window deque smoothing to EWMA half-life smoothing so the algorithm and parameter surface are identical for both axes. `RLTrajectoryPlayer` replaces the scalar `current_team_name` with a `_pending_team_name` scalar + `current_team_names: Dict[battle_tag, str]` dict; the dict entry is created on the first `_handle_battle_request` per battle and evicted in `_battle_finished_callback`.

**Tech Stack:** Python 3.11+, dataclasses, pytest, ruff, pyright. No new runtime dependencies. Touches `src/elitefurretai/etl/team_repo.py`, `src/elitefurretai/rl/{opponents.py,rl_trajectory_player.py,config.py,rl_utils.py,RL.md}`, all YAML in `src/elitefurretai/rl/configs/`, and their unit tests.

**Test-fixture gotcha (caught during Phase 1):** Any test that synthesizes a `TeamRepo` from `tmp_path` under a VGC format must write team files with **exactly 6 Pokemon entries** (counted by lines beginning with `Ability:`). `TeamRepo._load_teams_recursive` calls `_is_vgc_format(format_name)` and silently `continue`s any VGC file where `_count_pokemon_entries != 6`. Fixtures like `"Furret @ Choice Band\nAbility: Frisk\n- Quick Attack"` will load zero teams and `sample_team_name` will raise `ValueError`. Use `"\n\n".join(<6 entries>)` for safety. (No remaining phase in this plan touches fixtures of that shape, but flagging in case future scaffolding needs it.)

**Behavior deltas to validate:**
- Agent-axis smoothing changes from sliding window (deque maxlen=100) to EWMA (half_life=100 by default). For a stationary win-rate stream the long-run estimate is identical; the transient response is slightly slower but smoother. This is acceptable per the unification objective.
- Agent-axis floor application changes from "reserve floor mass then distribute (1-floor_sum) by residual" to water-filling (pin under-floor keys at floor, distribute the rest by score share). Both guarantee floors are met; the new behavior is closer to the natural score-weighted distribution.

---

## File Structure

| Path | Responsibility | Lines (approx) |
|---|---|---|
| `src/elitefurretai/rl/rl_utils.py` (MODIFY) | Append pure primitives `adaptive_score` and `adaptive_distribution` to the existing module (which already houses `setup_logging`, `normalize_curriculum`, `collate_trajectories`, etc. — same "shared rl utilities" category). Update `__all__` accordingly. | +120 |
| `src/elitefurretai/etl/team_repo.py` (MODIFY) | `sample_team_name` becomes the primitive; `sample_team` looks the name up and shuffles. | -25 net |
| `src/elitefurretai/rl/config.py` (MODIFY) | New `AdaptiveAxisConfig` dataclass; nested under `CurriculumConfig` as `adaptive_team_axis` and `adaptive_agent_axis`. Old flat fields removed (no backwards-compat shim per project rule). | +60 |
| `src/elitefurretai/rl/opponents.py` (MODIFY) | `update_curriculum` and `update_team_distribution` become thin callers of `rl_utils`. `WorkerOpponentFactory` reads from `AdaptiveAxisConfig` instances. Agent-axis switches to EWMA storage (`agent_win_rates: Dict[opp_type, (wins, n)]`); old `win_rate_tracking` deques are dropped from the adaptive path (kept only for the metrics emitter so the `win_rate/<opp>` Wandb panels still display a windowed rate). | ±200 |
| `src/elitefurretai/rl/rl_trajectory_player.py` (MODIFY) | Replace scalar `current_team_name` with `_pending_team_name` + per-battle `current_team_names` dict; stamp on first `_handle_battle_request`, evict in `_battle_finished_callback`. | +20 |
| `src/elitefurretai/rl/configs/*.yaml` (MODIFY) | Migrate to nested `adaptive_team_axis` / `adaptive_agent_axis` blocks. | per-file |
| `src/elitefurretai/rl/RL.md` (MODIFY) | Update §7 (team-axis) to reflect shared primitive and renamed config. | +1 paragraph |
| `unit_tests/rl/test_rl_utils.py` (NEW) | Unit tests for `adaptive_score` and `adaptive_distribution`. | ~150 |
| `unit_tests/etl/test_team_repo.py` (MODIFY) | Ensure `sample_team` still shuffles, both surface the same uniform distribution. | +20 |
| `unit_tests/rl/test_config.py` (MODIFY) | Cover `AdaptiveAxisConfig` defaults + nested YAML round-trip. | +30 |
| `unit_tests/rl/test_adaptive_curriculum.py` (MODIFY) | Re-point to new config shape; verify agent-axis EWMA behavior. | per-file |
| `unit_tests/rl/test_team_axis.py` (MODIFY) | Re-point to new config shape; reuse `adaptive_distribution` golden values. | per-file |
| `unit_tests/rl/test_worker_opponent_factory.py` (MODIFY) | Update fixtures to use nested config; add test for `current_team_names` per-battle keying. | +40 |
| `unit_tests/rl/test_players.py` or new `test_rl_trajectory_player.py` (MODIFY/NEW) | Test that concurrent battles get distinct `current_team_names[tag]` entries, and entries are evicted on battle finish. | ~60 |
| `planning/stage2/2026-05-26-08-00-curriculum-unification-and-team-fixes.md` (THIS DOC) | Plan + Updates log. | — |

---

## Phase 1 — `TeamRepo.sample_team` collapse

### Task 1.1: Failing test for `sample_team` shuffle behavior preserved when delegating to `sample_team_name`

**Files:**
- Modify: `unit_tests/etl/test_team_repo.py`

- [ ] **Step 1: Add failing test**

Append to `unit_tests/etl/test_team_repo.py`:

```python
def test_sample_team_delegates_to_sample_team_name(tmp_path, monkeypatch):
    """sample_team must look the name up via sample_team_name (single source of
    truth for uniform sampling) and then shuffle. After the merge, calling
    sample_team_name and resolving via _teams[fmt][name] yields the same
    string sample_team returns when shuffle is disabled."""
    # Build a tiny repo with shuffle off
    fmt_dir = tmp_path / "gen9vgc2024regg"
    fmt_dir.mkdir()
    (fmt_dir / "alpha.txt").write_text("Furret @ Choice Band\nAbility: Frisk\n- Quick Attack")
    (fmt_dir / "beta.txt").write_text("Sentret @ Eviolite\nAbility: Run Away\n- Tackle")

    from elitefurretai.etl.team_repo import TeamRepo
    repo = TeamRepo(filepath=str(tmp_path), shuffle=False)

    # Force deterministic sample_team_name → "alpha" by stubbing random.choice
    import random as _r
    monkeypatch.setattr(_r, "choice", lambda seq: "alpha" if "alpha" in seq else seq[0])

    name = repo.sample_team_name("gen9vgc2024regg")
    team_by_lookup = repo._teams["gen9vgc2024regg"][name]
    team_via_sample = repo.sample_team("gen9vgc2024regg")
    assert team_via_sample == team_by_lookup
```

- [ ] **Step 2: Run test to verify it passes today (current `sample_team` happens to give the same answer because we monkeypatch random.choice). This test guards the post-refactor invariant.**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_team_repo.py::test_sample_team_delegates_to_sample_team_name -v`
Expected: PASS today, must still PASS after refactor.

### Task 1.2: Refactor `sample_team` to call `sample_team_name`

**Files:**
- Modify: `src/elitefurretai/etl/team_repo.py:413-453`

- [ ] **Step 1: Replace `sample_team` body**

Replace the entire body of `sample_team` (currently `team_repo.py:413-453`) with:

```python
    def sample_team(self, format: str, subdirectory: Optional[str] = None) -> str:
        """
        Sample a random team from the specified format.

        Delegates name selection to ``sample_team_name`` (single source of
        truth for the uniform sampling distribution and subdirectory
        filtering), then materializes the team string and applies
        ``_shuffle_team_order`` if shuffle is enabled.

        Args:
            format: Pokemon format (e.g., "gen9vgc2023regc")
            subdirectory: Optional subdirectory path to sample from
                (e.g., "rental_teams" or "tournament_teams/worlds_2023").
                If None, samples from all teams in the format (default: None)

        Returns:
            Team string in PokePaste format

        Raises:
            ValueError: If format not found, no teams available, or subdirectory has no teams

        Examples:
            >>> repo.sample_team("gen9vgc2023regc")
            >>> repo.sample_team("gen9vgc2023regc", "rental_teams")
        """
        name = self.sample_team_name(format, subdirectory)
        team_string = self._teams[format][name]
        if self._shuffle:
            team_string = self._shuffle_team_order(team_string)
        return team_string
```

- [ ] **Step 2: Run the new test plus the full team_repo suite**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_team_repo.py -v`
Expected: PASS (all)

- [ ] **Step 3: Quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/etl/team_repo.py unit_tests/etl/test_team_repo.py && ruff format --check src/elitefurretai/etl/team_repo.py unit_tests/etl/test_team_repo.py && pyright src/elitefurretai/etl/team_repo.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/etl/team_repo.py unit_tests/etl/test_team_repo.py
git commit -m "refactor(etl/team_repo): make sample_team_name the primitive sample_team calls"
```

---

## Phase 2 — Per-battle `current_team_names` dict

### Task 2.1: Failing test for two concurrent battles getting distinct team names

**Files:**
- Create: `unit_tests/rl/test_rl_trajectory_player.py` (or extend `test_players.py` if more appropriate per the existing convention — check whether `test_players.py` already covers `RLTrajectoryPlayer`)

- [ ] **Step 1: Decide on test location**

Run: `grep -ln "RLTrajectoryPlayer" /home/cayman/Repositories/EliteFurretAI/unit_tests/rl/*.py`
If any existing file covers it, append there. Otherwise create `unit_tests/rl/test_rl_trajectory_player.py`.

- [ ] **Step 2: Write the failing test**

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

from elitefurretai.rl.rl_trajectory_player import RLTrajectoryPlayer


def _make_player_with_pending_team(name: str):
    """Construct a minimally-initialized RLTrajectoryPlayer bypassing the
    Player base class (which needs a real ps_client). We only exercise the
    team_name plumbing here."""
    player = RLTrajectoryPlayer.__new__(RLTrajectoryPlayer)
    player._pending_team_name = name
    player.current_team_names = {}
    player._discarded_battles = set()
    player._room_lost_battles = set()
    player._request_generation = {}
    player.current_trajectories = {}
    player.hidden_states = {}
    player.inference_client = None
    player.trajectory_queue = None
    return player


def test_pending_team_name_stamps_first_request_per_battle_tag():
    player = _make_player_with_pending_team("constrained/38dessert")
    player._stamp_pending_team_name("battle-tag-A")
    assert player.current_team_names == {"battle-tag-A": "constrained/38dessert"}

    # Second stamp for same tag is a no-op (defensive idempotence).
    player._pending_team_name = "OTHER_TEAM_SHOULD_NOT_OVERWRITE"
    player._stamp_pending_team_name("battle-tag-A")
    assert player.current_team_names["battle-tag-A"] == "constrained/38dessert"


def test_concurrent_battles_keep_distinct_team_names():
    """If randomize_all_teams runs between batches, two battles started in
    different batches must stamp the team name that was pending at the
    moment each battle started, not whichever value is most recent on the
    scalar."""
    player = _make_player_with_pending_team("team-1")
    player._stamp_pending_team_name("battle-A")

    # Simulate randomize_all_teams flipping the pending name before
    # battle B starts.
    player._pending_team_name = "team-2"
    player._stamp_pending_team_name("battle-B")

    assert player.current_team_names == {
        "battle-A": "team-1",
        "battle-B": "team-2",
    }


def test_battle_finished_evicts_team_name_entry():
    """After a battle finishes the dict entry must be popped so the dict
    doesn't grow unbounded across thousands of battles per worker."""
    player = _make_player_with_pending_team("team-1")
    player._stamp_pending_team_name("battle-A")
    name = player._pop_team_name("battle-A")
    assert name == "team-1"
    assert "battle-A" not in player.current_team_names
    # Popping a tag we never stamped (e.g. immediate forfeit before first
    # request) must return None, not raise.
    assert player._pop_team_name("never-stamped") is None
```

- [ ] **Step 3: Run to verify FAIL**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_rl_trajectory_player.py -v`
Expected: FAIL with `AttributeError: 'RLTrajectoryPlayer' object has no attribute '_pending_team_name'` (or similar — the methods don't exist yet).

### Task 2.2: Add per-battle dict and helper methods

**Files:**
- Modify: `src/elitefurretai/rl/rl_trajectory_player.py`

- [ ] **Step 1: Replace the class-level scalar attribute**

Find at line 83 of `src/elitefurretai/rl/rl_trajectory_player.py`:

```python
    current_team_name: Optional[str] = None
```

Remove it. (Class-level attribute is replaced by per-instance dict initialized in `__init__`.)

- [ ] **Step 2: Add instance state in `__init__`**

After `self._room_lost_battles: set[str] = set()` (around line 121 of `rl_trajectory_player.py`), insert:

```python
        # Per-battle team-name tracking.
        # `_pending_team_name` is set by WorkerOpponentFactory just before
        # `battle_against` is called — it holds the team name that the
        # ConstantTeambuilder will emit for the *next* battle(s) this player
        # starts under the current `_team`.
        # `current_team_names[battle_tag]` is the immutable team name for an
        # in-progress battle; we stamp it from `_pending_team_name` on the
        # first `_handle_battle_request` per tag and pop it in
        # `_battle_finished_callback`. This avoids attributing the wrong
        # team to a battle if `randomize_all_teams` flips the pending
        # name while a previous battle's finished-callback is still in
        # flight (today rare but a latent race).
        self._pending_team_name: Optional[str] = None
        self.current_team_names: Dict[str, str] = {}
```

- [ ] **Step 3: Add `_stamp_pending_team_name` and `_pop_team_name` helpers**

Add as methods on `RLTrajectoryPlayer` (anywhere; suggest just below `_abort_decision`):

```python
    def _stamp_pending_team_name(self, battle_tag: str) -> None:
        """Idempotently copy `_pending_team_name` into the per-battle dict
        on the first request for this tag. No-op for subsequent requests
        so a later `randomize_all_teams` flip on the scalar cannot
        retroactively change the recorded team."""
        if battle_tag in self.current_team_names:
            return
        if self._pending_team_name is not None:
            self.current_team_names[battle_tag] = self._pending_team_name

    def _pop_team_name(self, battle_tag: str) -> Optional[str]:
        """Remove and return the team name for a finished battle.
        Returns None if the battle never reached its first request
        (e.g. immediate forfeit before move 1)."""
        return self.current_team_names.pop(battle_tag, None)
```

- [ ] **Step 4: Run the helpers test to confirm GREEN**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_rl_trajectory_player.py -v`
Expected: PASS

### Task 2.3: Wire `_stamp_pending_team_name` into `_handle_battle_request`

**Files:**
- Modify: `src/elitefurretai/rl/rl_trajectory_player.py:186-200`

- [ ] **Step 1: Insert stamp call at the top of `_handle_battle_request`**

In `_handle_battle_request` (currently starting at line 186), after the `if getattr(battle, "finished", False): return` guard (line 191-192), insert:

```python
        # Stamp this battle's team name on first request (idempotent).
        # See `_stamp_pending_team_name` for why this isn't a scalar.
        self._stamp_pending_team_name(battle.battle_tag)
```

- [ ] **Step 2: Replace `current_team_name` read in `_battle_finished_callback`**

At `rl_trajectory_player.py:642`, change:

```python
                    "team_name": self.current_team_name,
```

to:

```python
                    "team_name": self._pop_team_name(battle.battle_tag),
```

- [ ] **Step 3: Also evict on all early-return paths in `_battle_finished_callback`**

In `_battle_finished_callback` (around line 600-616), there are two early-return branches: the discarded-battles branch and the `trajectory_queue is None` (opponent-only) branch. Add `self._pop_team_name(battle.battle_tag)` to both so the dict doesn't leak for those battle outcomes. Final shape:

```python
        if battle.battle_tag in self._discarded_battles:
            self._discarded_battles.discard(battle.battle_tag)
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            self._pop_team_name(battle.battle_tag)
            return

        if self.trajectory_queue is None:
            self.current_trajectories.pop(battle.battle_tag, None)
            self._reset_battle_hidden_state(battle.battle_tag)
            self._pop_team_name(battle.battle_tag)
            return
```

- [ ] **Step 4: Update callers in `opponents.py` to set `_pending_team_name` instead of `current_team_name`**

In `src/elitefurretai/rl/opponents.py`, replace each of these four assignments (lines 998, 1016, 1254, 1260):

```python
player.current_team_name = player_team_name       # line 998
opponent.current_team_name = opp_team_name        # line 1016
player.current_team_name = team_name              # line 1254
opponent.current_team_name = team_name            # line 1260
```

with:

```python
player._pending_team_name = player_team_name      # line 998
opponent._pending_team_name = opp_team_name       # line 1016
player._pending_team_name = team_name             # line 1254
opponent._pending_team_name = team_name           # line 1260
```

- [ ] **Step 5: Update the `RL.md` reference to the renamed attribute**

In `src/elitefurretai/rl/RL.md:349`, replace `RLTrajectoryPlayer.current_team_name` with `RLTrajectoryPlayer._pending_team_name`, and add a sentence noting that the per-battle dict `current_team_names[battle_tag]` is the actual recorded value on the trajectory.

- [ ] **Step 6: Run targeted tests**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_rl_trajectory_player.py unit_tests/rl/test_worker_opponent_factory.py unit_tests/rl/test_team_axis.py -v`
Expected: PASS (or, if `test_worker_opponent_factory.py` asserts `current_team_name`, update that assertion to `_pending_team_name` / `current_team_names[tag]` in this same task).

- [ ] **Step 7: Quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl unit_tests/rl && ruff format --check src/elitefurretai/rl unit_tests/rl && pyright src/elitefurretai/rl/rl_trajectory_player.py src/elitefurretai/rl/opponents.py`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add src/elitefurretai/rl/rl_trajectory_player.py src/elitefurretai/rl/opponents.py src/elitefurretai/rl/RL.md unit_tests/rl/test_rl_trajectory_player.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "fix(rl/trajectory_player): per-battle team_name dict to remove latent race in randomize_all_teams"
```

---

## Phase 3 — `rl_utils.py` primitives

### Task 3.1: Failing tests for `adaptive_score` and `adaptive_distribution`

**Files:**
- Create: `unit_tests/rl/test_rl_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the shared adaptive-curriculum primitives in rl_utils.

These primitives are called from two places:
- WorkerOpponentFactory.update_curriculum (agent-axis: opponent type mix)
- WorkerOpponentFactory.update_team_distribution (team-axis: per-format
  team mix)

The tests below pin down behavior shared between both call sites."""

import math
import pytest

from elitefurretai.rl.rl_utils import adaptive_score, adaptive_distribution


# ── adaptive_score ────────────────────────────────────────────────────

def test_adaptive_score_pfsp_peaks_at_50_percent_winrate():
    """PFSP component is 1.0 when win rate = 0.5, 0.0 at extremes."""
    s_50 = adaptive_score(
        wins=50.0, n=100.0,
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=1.0, weakness_mix=0.0, weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_100 = adaptive_score(
        wins=100.0, n=100.0,
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=1.0, weakness_mix=0.0, weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    assert s_50 > s_100
    assert s_50 > 0.85   # near-peak after Beta(8,8) smoothing
    assert s_100 < 0.20


def test_adaptive_score_weakness_grows_as_winrate_drops():
    """Weakness component is 0 above target, grows linearly below."""
    s_low = adaptive_score(
        wins=10.0, n=100.0,  # smoothed wr ≈ 0.155
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=0.0, weakness_mix=1.0, weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_high = adaptive_score(
        wins=80.0, n=100.0,  # smoothed wr ≈ 0.76, above target → 0
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=0.0, weakness_mix=1.0, weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    assert s_low > 0.5
    assert s_high == 0.0


def test_adaptive_score_weakness_exponent_asymmetric():
    """weakness_exponent > 1 amplifies losses; team-axis uses this shape."""
    s_lin = adaptive_score(
        wins=20.0, n=100.0,
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=0.0, weakness_mix=1.0, weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_squared = adaptive_score(
        wins=20.0, n=100.0,
        prior_alpha=8.0, prior_beta=8.0,
        pfsp_mix=0.0, weakness_mix=1.0, weakness_exponent=2.0,
        target_win_rate=0.55,
    )
    # exponent 2 on a value in (0,1) reduces magnitude; both >0
    assert s_lin > s_squared > 0.0


# ── adaptive_distribution ────────────────────────────────────────────

def test_adaptive_distribution_normalizes_to_one():
    scores = {"a": 2.0, "b": 1.0, "c": 1.0}
    d = adaptive_distribution(scores)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert d["a"] > d["b"] == d["c"]


def test_adaptive_distribution_water_fills_below_floor():
    """When a key's natural share is below floor, it gets pinned at floor
    and the remaining mass is renormalized over unpinned keys."""
    scores = {"a": 100.0, "b": 100.0, "c": 0.01}
    floors = {"a": 0.0, "b": 0.0, "c": 0.10}
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(d["c"], 0.10, abs_tol=1e-9)
    assert math.isclose(d["a"], 0.45, abs_tol=1e-9)
    assert math.isclose(d["b"], 0.45, abs_tol=1e-9)


def test_adaptive_distribution_uniform_floor_dict_matches_team_axis():
    """Team-axis builds `floors = {t: per_team_floor for t in teams}`. With
    `per_team_floor=0.005` and one of three teams having essentially zero
    score, that team gets pinned at 0.005 and the rest split the residual."""
    scores = {"t1": 1.0, "t2": 1.0, "t3": 1e-9}
    floors = {k: 0.005 for k in scores}
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(d["t3"], 0.005, abs_tol=1e-9)


def test_adaptive_distribution_falls_back_to_uniform_when_floors_too_aggressive():
    """If total floor mass would exceed 1.0, fall back to uniform across
    all keys to keep the distribution valid."""
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}
    floors = {"a": 0.5, "b": 0.5, "c": 0.5}  # sum = 1.5 > 1.0
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert d == pytest.approx({"a": 1/3, "b": 1/3, "c": 1/3})


def test_adaptive_distribution_blends_base_curriculum():
    """When base_blend > 0 and a base curriculum is provided, the final
    score per key is (base_blend * base[k]) + ((1 - base_blend) * scores[k])
    before water-filling. Agent-axis sets base_blend=0.5."""
    scores = {"a": 1.0, "b": 0.0}
    base = {"a": 0.0, "b": 1.0}
    d_pure = adaptive_distribution(scores)
    d_mixed = adaptive_distribution(scores, base=base, base_blend=0.5)
    # Pure: a dominates. Mixed: equal blend → uniform.
    assert d_pure["a"] > d_pure["b"]
    assert math.isclose(d_mixed["a"], 0.5, abs_tol=1e-9)
    assert math.isclose(d_mixed["b"], 0.5, abs_tol=1e-9)


def test_adaptive_distribution_handles_all_zero_scores():
    """All-zero (or all-epsilon) scores → uniform fallback over keys."""
    scores = {"a": 0.0, "b": 0.0, "c": 0.0}
    d = adaptive_distribution(scores)
    assert d == pytest.approx({"a": 1/3, "b": 1/3, "c": 1/3})
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_rl_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'adaptive_score' from 'elitefurretai.rl.rl_utils'` (the module already exists with other utilities; the new functions are absent).

### Task 3.2: Append the primitives to the existing `rl_utils.py`

**Files:**
- Modify: `src/elitefurretai/rl/rl_utils.py`

**IMPORTANT: This module already exists** with `setup_logging`, `list_pt_files`, `normalize_curriculum`, memory helpers, `collate_trajectories`, `timestamp_iso`, etc. We are APPENDING two new functions and a `TypeVar`, not creating a new file. Keep the existing imports, docstring, and `__all__` intact and extend them.

- [ ] **Step 1: Add the `K` TypeVar near the existing typing imports**

Locate the existing `from typing import ...` block at the top of `src/elitefurretai/rl/rl_utils.py`. Ensure `Hashable` and `TypeVar` are imported (add to the existing import line if missing). Then add at module level (below the imports, above the first existing `def`):

```python
K = TypeVar("K", bound=Hashable)
```

- [ ] **Step 2: Append the two functions to the end of the module**

The existing module does not declare `__all__`, so simply append the two functions at the end of the file. Append:

```python


def adaptive_score(
    wins: float,
    n: float,
    *,
    prior_alpha: float,
    prior_beta: float,
    pfsp_mix: float,
    weakness_mix: float,
    weakness_exponent: float,
    target_win_rate: float,
) -> float:
    """Blended PFSP + weakness score for a single (wins, n) observation.

    Args:
        wins: Beta-prior raw wins count (or EWMA-decayed analog).
        n: Beta-prior raw sample count (or EWMA-decayed analog).
        prior_alpha, prior_beta: Beta(alpha, beta) pseudo-counts for
            smoothed win-rate estimate.
        pfsp_mix: Weight on the PFSP component (peaks at wr=0.5).
        weakness_mix: Weight on the asymmetric weakness component
            (`(target - wr)/target` clipped to [0,1], raised to
            `weakness_exponent`).
        weakness_exponent: Shape of the weakness component. 1.0 = linear,
            >1 emphasizes severe losses, <1 flattens.
        target_win_rate: Win rate above which weakness is zero.

    Returns:
        Non-negative score. The caller normalizes across keys.
    """
    wr = (wins + prior_alpha) / (n + prior_alpha + prior_beta)
    pfsp = max(0.0, 1.0 - 2.0 * abs(wr - 0.5))
    if target_win_rate > 0.0:
        weakness_raw = max(0.0, (target_win_rate - wr) / target_win_rate)
    else:
        weakness_raw = 0.0
    weakness = weakness_raw ** weakness_exponent if weakness_raw > 0.0 else 0.0
    return pfsp_mix * pfsp + weakness_mix * weakness


def adaptive_distribution(
    scores: Dict[K, float],
    *,
    base: Optional[Dict[K, float]] = None,
    base_blend: float = 0.0,
    floors: Optional[Dict[K, float]] = None,
    epsilon: float = 1e-9,
) -> Dict[K, float]:
    """Convert scores into a probability distribution with optional
    base-curriculum blending and per-key floors (water-filling).

    Algorithm:
      1. If `base` and `base_blend > 0`, replace each score with
         `(base_blend * base[k]) + ((1 - base_blend) * scores[k])`.
      2. Normalize scores by their sum. Degenerate cases (sum <= 0)
         fall back to uniform across all keys.
      3. If `floors` provided, water-fill: iteratively pin keys whose
         natural share is below floor at exactly `floor[k]`, redistribute
         remaining mass to unpinned keys by their score share. Iterates
         until no new keys fall below floor (capped at `len(scores)`
         iterations — each productive pass strictly grows the pinned
         set). If sum(floors) >= 1.0, fall back to uniform.
    """
    keys = list(scores.keys())
    if not keys:
        return {}

    if base is not None and base_blend > 0.0:
        mixed = {
            k: (base_blend * base.get(k, 0.0))
            + ((1.0 - base_blend) * scores.get(k, 0.0))
            for k in keys
        }
    else:
        mixed = dict(scores)

    total = sum(mixed.values())
    if total <= epsilon:
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}
    distribution: Dict[K, float] = {k: mixed[k] / total for k in keys}

    if not floors:
        return distribution

    floor_sum = sum(floors.get(k, 0.0) for k in keys)
    if floor_sum >= 1.0:
        uniform = 1.0 / len(keys)
        return {k: uniform for k in keys}

    pinned: Dict[K, float] = {}
    for _ in range(len(keys)):
        remaining_mass = 1.0 - sum(pinned.values())
        unpinned = [k for k in keys if k not in pinned]
        unpinned_total = sum(mixed[k] for k in unpinned)
        new_pin = False
        for k in unpinned:
            floor_k = floors.get(k, 0.0)
            if unpinned_total <= 0.0:
                share = remaining_mass / max(len(unpinned), 1)
            else:
                share = (mixed[k] / unpinned_total) * remaining_mass
            if share < floor_k:
                pinned[k] = floor_k
                new_pin = True
        if not new_pin:
            break

    out: Dict[K, float] = {}
    remaining_mass = 1.0 - sum(pinned.values())
    unpinned = [k for k in keys if k not in pinned]
    unpinned_total = sum(mixed[k] for k in unpinned)
    for k in keys:
        if k in pinned:
            out[k] = pinned[k]
        elif unpinned_total <= 0.0:
            out[k] = remaining_mass / max(len(unpinned), 1)
        else:
            out[k] = (mixed[k] / unpinned_total) * remaining_mass
    return out


```

(Do NOT add an `__all__` declaration — the existing module does not have one, and introducing one now would silently hide every other helper. The functions become importable as `from elitefurretai.rl.rl_utils import adaptive_score, adaptive_distribution` without `__all__`.)

- [ ] **Step 2: Run tests, confirm GREEN**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_rl_utils.py -v`
Expected: PASS (all 9)

- [ ] **Step 3: Quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/rl_utils.py unit_tests/rl/test_rl_utils.py && ruff format --check src/elitefurretai/rl/rl_utils.py unit_tests/rl/test_rl_utils.py && pyright src/elitefurretai/rl/rl_utils.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/rl/rl_utils.py unit_tests/rl/test_rl_utils.py
git commit -m "feat(rl/rl_utils): shared adaptive_score and adaptive_distribution primitives"
```

---

## Phase 4 — `AdaptiveAxisConfig` dataclass

### Task 4.1: Failing tests for the new config shape

**Files:**
- Modify: `unit_tests/rl/test_config.py`

- [ ] **Step 1: Add failing tests at the end of the file**

```python
def test_adaptive_axis_config_team_defaults():
    """Team-axis default values reproduce the Change 7 settings: EWMA
    half_life=50, pure asymmetric weakness with linear exponent."""
    from elitefurretai.rl.config import AdaptiveAxisConfig
    cfg = AdaptiveAxisConfig.team_axis_defaults()
    assert cfg.enabled is True
    assert cfg.half_life == 50.0
    assert cfg.pfsp_mix == 0.0
    assert cfg.weakness_mix == 1.0
    assert cfg.weakness_exponent == 1.0
    assert cfg.base_blend == 0.0
    assert cfg.per_key_floor == 0.005
    assert cfg.min_samples == 20  # team_warmup_threshold


def test_adaptive_axis_config_agent_defaults():
    """Agent-axis default values reproduce the existing update_curriculum
    behavior: PFSP-weighted with weakness side, base-curriculum blended."""
    from elitefurretai.rl.config import AdaptiveAxisConfig
    cfg = AdaptiveAxisConfig.agent_axis_defaults()
    assert cfg.enabled is True
    assert cfg.half_life == 100.0
    assert cfg.pfsp_mix == 0.70
    assert cfg.weakness_mix == 0.30
    assert cfg.weakness_exponent == 1.0
    assert cfg.target_win_rate == 0.55
    assert cfg.base_blend == 0.50
    assert cfg.min_samples == 40
    assert cfg.prior_alpha == 8.0
    assert cfg.prior_beta == 8.0


def test_curriculum_config_nests_both_axes():
    from elitefurretai.rl.config import CurriculumConfig, AdaptiveAxisConfig
    cfg = CurriculumConfig()
    assert isinstance(cfg.adaptive_team_axis, AdaptiveAxisConfig)
    assert isinstance(cfg.adaptive_agent_axis, AdaptiveAxisConfig)
    # Distinct defaults for each axis
    assert cfg.adaptive_team_axis.pfsp_mix == 0.0
    assert cfg.adaptive_agent_axis.pfsp_mix == 0.70


def test_curriculum_config_yaml_round_trip_with_nested_axes(tmp_path):
    """Loading a YAML with nested adaptive_*_axis blocks reconstructs
    the AdaptiveAxisConfig sub-dataclasses correctly."""
    import yaml
    from elitefurretai.rl.config import Config

    payload = {
        "curriculum": {
            "battle_formats": {"gen9vgc2024regg": 1.0},
            "adaptive_team_axis": {
                "enabled": True,
                "half_life": 75.0,
                "weakness_exponent": 2.0,
                "per_key_floor": 0.01,
            },
            "adaptive_agent_axis": {
                "enabled": False,
                "min_samples": 60,
                "base_blend": 0.25,
            },
        },
    }
    path = tmp_path / "test_cfg.yaml"
    path.write_text(yaml.safe_dump(payload))
    cfg = Config.from_yaml(str(path))
    assert cfg.curriculum.adaptive_team_axis.half_life == 75.0
    assert cfg.curriculum.adaptive_team_axis.weakness_exponent == 2.0
    assert cfg.curriculum.adaptive_team_axis.per_key_floor == 0.01
    assert cfg.curriculum.adaptive_agent_axis.enabled is False
    assert cfg.curriculum.adaptive_agent_axis.min_samples == 60
    assert cfg.curriculum.adaptive_agent_axis.base_blend == 0.25
```

- [ ] **Step 2: Run to confirm FAIL**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py -k adaptive_axis -v`
Expected: FAIL with `ImportError` on `AdaptiveAxisConfig`.

### Task 4.2: Add `AdaptiveAxisConfig` and nest it into `CurriculumConfig`

**Files:**
- Modify: `src/elitefurretai/rl/config.py:435-540` and the `_make_sub` recursion at line 817

- [ ] **Step 1: Insert `AdaptiveAxisConfig` dataclass above `CurriculumConfig`**

Insert the following dataclass just above the existing `class CurriculumConfig:` line (currently `config.py:435`):

```python
@dataclass
class AdaptiveAxisConfig:
    """Adaptive-curriculum parameters shared by the team-axis and
    agent-axis updates in `WorkerOpponentFactory`.

    Both axes consume the same primitive
    (`elitefurretai.rl.rl_utils.adaptive_distribution`); they differ
    only in the parameter values chosen below. See `team_axis_defaults`
    and `agent_axis_defaults` for the values that reproduce Change 7
    (team) and the pre-unification `update_curriculum` (agent).

    Fields:
        enabled: Master switch for this axis. False = bypass entirely.
        min_samples: Per-key minimum sample count before the score is
            trusted (under-warm keys fall back to the base preference).
        half_life: EWMA half-life in *recorded battles*; decay factor
            per battle is `0.5 ** (1 / half_life)`.
        prior_alpha, prior_beta: Beta pseudo-counts for win-rate
            smoothing.
        pfsp_mix: Weight on the PFSP component (peaks at wr=0.5).
        weakness_mix: Weight on the asymmetric weakness component.
        weakness_exponent: Shape of the weakness component (1.0 linear).
        target_win_rate: Win rate above which weakness is zero.
        base_blend: Mix factor with the base curriculum (0.0 = pure
            adaptive, 1.0 = pure base).
        per_key_floor: Uniform per-key minimum mass after the floor
            pass. The agent-axis caller may override on a per-key basis
            by constructing its own `floors` dict before calling
            `adaptive_distribution` directly.
    """

    enabled: bool = True
    min_samples: int = 40
    half_life: float = 100.0
    prior_alpha: float = 8.0
    prior_beta: float = 8.0
    pfsp_mix: float = 0.70
    weakness_mix: float = 0.30
    weakness_exponent: float = 1.0
    target_win_rate: float = 0.55
    base_blend: float = 0.50
    per_key_floor: float = 0.0

    @classmethod
    def team_axis_defaults(cls) -> "AdaptiveAxisConfig":
        """Defaults that reproduce Change 7 team-axis behavior."""
        return cls(
            enabled=True,
            min_samples=20,
            half_life=50.0,
            prior_alpha=8.0,
            prior_beta=8.0,
            pfsp_mix=0.0,
            weakness_mix=1.0,
            weakness_exponent=1.0,
            target_win_rate=1.0,  # (1 - wr) shape: weakness = 1 - wr
            base_blend=0.0,
            per_key_floor=0.005,
        )

    @classmethod
    def agent_axis_defaults(cls) -> "AdaptiveAxisConfig":
        """Defaults that reproduce the pre-unification update_curriculum.

        Note: the per-key anchor floors (SELF_PLAY=0.20, BC_PLAYER=0.10,
        GHOSTS=0.10) are NOT in this config — they are constructed
        per-call by the caller using opponent availability. `per_key_floor`
        stays 0 for agent-axis because the floors are heterogeneous."""
        return cls(
            enabled=True,
            min_samples=40,
            half_life=100.0,
            prior_alpha=8.0,
            prior_beta=8.0,
            pfsp_mix=0.70,
            weakness_mix=0.30,
            weakness_exponent=1.0,
            target_win_rate=0.55,
            base_blend=0.50,
            per_key_floor=0.0,
        )
```

- [ ] **Step 2: Edit `CurriculumConfig` to nest both axes; remove the old flat fields**

Inside the existing `CurriculumConfig` dataclass (around `config.py:517-539`), find and REMOVE these lines:

```python
    adaptive_curriculum: bool = True
    team_axis_enabled: bool = True
    team_warmup_threshold: int = 20
    team_per_team_floor: float = 0.005
    half_life: float = 50.0
    pfsp_exponent: float = 1.0
```

(Per the project's hard constraint: backwards compat is not a concern; just flag broken plans.)

Replace with:

```python
    # Adaptive curriculum: two axes, same algorithm, different defaults.
    # See `AdaptiveAxisConfig.{team,agent}_axis_defaults` and the shared
    # `rl_utils.adaptive_distribution` primitive for the algorithm itself.
    adaptive_team_axis: AdaptiveAxisConfig = field(
        default_factory=AdaptiveAxisConfig.team_axis_defaults
    )
    adaptive_agent_axis: AdaptiveAxisConfig = field(
        default_factory=AdaptiveAxisConfig.agent_axis_defaults
    )
```

- [ ] **Step 3: Extend `_make_sub` (or the YAML loader) to recurse into the new sub-dataclass**

In `config.py` near the `from_yaml`/`_make_sub` machinery (around line 817), if the existing loader doesn't already recurse into nested dataclasses, add explicit handling:

```python
# Inside Config.from_yaml or _make_sub, when constructing CurriculumConfig:
curriculum_data = dict(data.get("curriculum", {}))
if "adaptive_team_axis" in curriculum_data:
    curriculum_data["adaptive_team_axis"] = _make_sub(
        AdaptiveAxisConfig, curriculum_data["adaptive_team_axis"]
    )
else:
    curriculum_data["adaptive_team_axis"] = AdaptiveAxisConfig.team_axis_defaults()
if "adaptive_agent_axis" in curriculum_data:
    curriculum_data["adaptive_agent_axis"] = _make_sub(
        AdaptiveAxisConfig, curriculum_data["adaptive_agent_axis"]
    )
else:
    curriculum_data["adaptive_agent_axis"] = AdaptiveAxisConfig.agent_axis_defaults()
curriculum = _make_sub(CurriculumConfig, curriculum_data)
```

(Read the existing `_make_sub` first — if it already recurses into dataclass-typed fields generically, this block isn't needed and the per-axis defaults will apply automatically via `field(default_factory=...)`.)

- [ ] **Step 4: Run config tests**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py -v`
Expected: PASS (including the four new tests).

- [ ] **Step 5: Quality gates**

Run: `source ../venv/bin/activate && ruff check src/elitefurretai/rl/config.py unit_tests/rl/test_config.py && ruff format --check src/elitefurretai/rl/config.py unit_tests/rl/test_config.py && pyright src/elitefurretai/rl/config.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "refactor(rl/config): nest adaptive_{team,agent}_axis sub-dataclasses sharing AdaptiveAxisConfig"
```

---

## Phase 5 — Rewire `opponents.py` to use the shared primitive

This is the largest single phase. We're changing storage (sliding-window deque → EWMA pair) for agent-axis and replacing two ~100-line methods with shared-primitive callers. Split into three commits.

### Task 5.1: Update `WorkerOpponentFactory.__init__` signature to take config sub-objects

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py:170-265`

- [ ] **Step 1: Change constructor parameters**

In `WorkerOpponentFactory.__init__` (line 170-187), replace the team-axis-specific parameters with two `AdaptiveAxisConfig` arguments:

Find:

```python
        tracking_window: int = 100,
        team_repo: Optional["TeamRepo"] = None,
        battle_formats: Optional[Dict[str, float]] = None,
        opponent_team_subdirectories: Optional[Dict[str, Optional[str]]] = None,
        team_axis_enabled: bool = True,
        team_warmup_threshold: int = 20,
        team_per_team_floor: float = 0.005,
        half_life: float = 50.0,
        pfsp_exponent: float = 1.0,
    ):
```

Replace with:

```python
        tracking_window: int = 100,
        team_repo: Optional["TeamRepo"] = None,
        battle_formats: Optional[Dict[str, float]] = None,
        opponent_team_subdirectories: Optional[Dict[str, Optional[str]]] = None,
        adaptive_team_axis: Optional["AdaptiveAxisConfig"] = None,
        adaptive_agent_axis: Optional["AdaptiveAxisConfig"] = None,
    ):
        from elitefurretai.rl.config import AdaptiveAxisConfig
        if adaptive_team_axis is None:
            adaptive_team_axis = AdaptiveAxisConfig.team_axis_defaults()
        if adaptive_agent_axis is None:
            adaptive_agent_axis = AdaptiveAxisConfig.agent_axis_defaults()
        self.adaptive_team_axis = adaptive_team_axis
        self.adaptive_agent_axis = adaptive_agent_axis
```

- [ ] **Step 2: Update internal state init to read from the new configs**

In the same `__init__`, replace the existing team-axis state block (currently `opponents.py:235-251`) with values derived from `adaptive_team_axis`:

```python
        # ── team-axis adaptive curriculum state ──
        self.team_axis_enabled = adaptive_team_axis.enabled
        self.team_warmup_threshold = adaptive_team_axis.min_samples
        self.team_per_team_floor = adaptive_team_axis.per_key_floor
        # half_life / pfsp_exponent etc. are pulled directly from
        # self.adaptive_team_axis at call time inside
        # update_team_distribution, so no field copies needed.
        self.known_teams: Dict[str, List[str]] = {}
        self.team_win_rates: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.team_sample_counts: Dict[str, Dict[str, int]] = {}
        self._team_axis_warm: Dict[str, bool] = {}
        self._team_axis_format_keys: List[str] = (
            list(battle_formats.keys()) if battle_formats else []
        )
```

Delete the now-unused `self._team_axis_half_life` / `self._team_axis_pfsp_exponent` assignments.

- [ ] **Step 3: Add agent-axis EWMA storage alongside the existing deque tracking**

Right after the `self.win_rate_tracking` / `self.battle_length_tracking` block (around line 224-231), add:

```python
        # Agent-axis EWMA store: maps opp_type → (decayed wins, decayed n).
        # `win_rate_tracking` (deque) is kept for the metrics emitter so the
        # win_rate/<opp> Wandb panels keep showing a sliding-window rate;
        # the adaptive update uses this EWMA pair instead.
        self.agent_win_rates: Dict[str, Tuple[float, float]] = {
            opp_type: (0.0, 0.0) for opp_type in self.win_rates
        }
```

- [ ] **Step 4: Update callers in `train.py` (and any other constructor sites) to pass the new config sub-objects**

Run: `grep -rn "WorkerOpponentFactory(" /home/cayman/Repositories/EliteFurretAI/src /home/cayman/Repositories/EliteFurretAI/unit_tests`

For each call site, replace the old `team_axis_enabled=…, team_warmup_threshold=…, team_per_team_floor=…, half_life=…, pfsp_exponent=…` kwargs with:

```python
adaptive_team_axis=cfg.curriculum.adaptive_team_axis,
adaptive_agent_axis=cfg.curriculum.adaptive_agent_axis,
```

For unit tests that build `WorkerOpponentFactory` directly, replace those kwargs with explicit `AdaptiveAxisConfig(...)` instances or `AdaptiveAxisConfig.{team,agent}_axis_defaults()`.

- [ ] **Step 5: Run targeted tests**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_worker_opponent_factory.py unit_tests/rl/test_opponent_pool.py unit_tests/rl/test_team_axis.py -v`
Expected: PASS (or test-fixture update fallout — fix in this same task).

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/opponents.py src/elitefurretai/rl/train.py unit_tests/rl/test_worker_opponent_factory.py unit_tests/rl/test_opponent_pool.py unit_tests/rl/test_team_axis.py
git commit -m "refactor(rl/opponents): take AdaptiveAxisConfig sub-objects; add EWMA store"
```

### Task 5.2: Rewire `update_team_distribution` and `record_battle_result` for agent-axis EWMA

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py:373-407` (`record_battle_result`), `:567-681` (`update_team_distribution`)

- [ ] **Step 1: Add EWMA update to `record_battle_result` for agent-axis**

In `record_battle_result` (around line 373-407), after the existing sliding-window update (`self.win_rate_tracking[opponent_type].append(win_value)`), add the EWMA update:

```python
        # Agent-axis EWMA update. Same shape as the team-axis update
        # below — `decay` is the per-battle multiplier so each opponent
        # type's smoothed win rate has half_life=adaptive_agent_axis.half_life.
        agent_half_life = self.adaptive_agent_axis.half_life
        decay = 0.5 ** (1.0 / max(agent_half_life, 1e-9))
        prev_wins, prev_n = self.agent_win_rates.get(opponent_type, (0.0, 0.0))
        new_wins = prev_wins * decay + win_value
        new_n = prev_n * decay + 1.0
        self.agent_win_rates[opponent_type] = (new_wins, new_n)
```

- [ ] **Step 2: Rewrite `update_team_distribution` to call the shared primitive**

Replace the body of `update_team_distribution` (currently `opponents.py:567-681`) with:

```python
    def update_team_distribution(
        self,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """Recompute the per-format team sampling distribution using the
        shared `adaptive_distribution` primitive.

        See `AdaptiveAxisConfig.team_axis_defaults` for the parameter
        choices that reproduce the original Change 7 behavior.
        """
        from elitefurretai.rl.rl_utils import adaptive_score, adaptive_distribution

        result: Dict[str, Optional[Dict[str, float]]] = {
            fmt: None for fmt in self._team_axis_format_keys
        }
        if not self.adaptive_team_axis.enabled:
            return result

        cfg = self.adaptive_team_axis
        for fmt, teams in self.known_teams.items():
            # Warm-up: latch once all known teams in this format have hit
            # the threshold; never un-latch.
            if not self._team_axis_warm.get(fmt, False):
                warm = all(
                    self.team_sample_counts[fmt][t] >= cfg.min_samples
                    for t in teams
                )
                if warm:
                    self._team_axis_warm[fmt] = True
                else:
                    result[fmt] = None
                    continue

            scores: Dict[str, float] = {}
            for t in teams:
                wins, n = self.team_win_rates[fmt][t]
                scores[t] = adaptive_score(
                    wins=wins, n=n,
                    prior_alpha=cfg.prior_alpha, prior_beta=cfg.prior_beta,
                    pfsp_mix=cfg.pfsp_mix, weakness_mix=cfg.weakness_mix,
                    weakness_exponent=cfg.weakness_exponent,
                    target_win_rate=cfg.target_win_rate,
                )

            floors = {t: cfg.per_key_floor for t in teams} if cfg.per_key_floor > 0 else None
            result[fmt] = adaptive_distribution(
                scores,
                base=None,
                base_blend=0.0,
                floors=floors,
            )

        return result
```

- [ ] **Step 3: Run team-axis tests; fix any goldens that drifted**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_team_axis.py -v`
Expected: PASS. If goldens drift, recompute by hand from `adaptive_score`/`adaptive_distribution` — the algorithm is unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_team_axis.py
git commit -m "refactor(rl/opponents): update_team_distribution delegates to rl_utils.adaptive_distribution"
```

### Task 5.3: Rewrite `update_curriculum` to use the shared primitive

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py:454-565`

- [ ] **Step 1: Replace `update_curriculum` body**

Replace the entire `update_curriculum` method body (currently lines 454-565) with:

```python
    def update_curriculum(self):
        """Adapt agent-axis curriculum weights using the shared
        adaptive primitive (`rl_utils.adaptive_distribution`).

        See `AdaptiveAxisConfig.agent_axis_defaults` for the parameter
        choices that reproduce the pre-unification behavior. The only
        substantive change from the legacy code path is that smoothing
        now uses EWMA (configured by `half_life`) instead of a fixed
        sliding window; for a stationary signal the long-run estimate is
        unchanged.
        """
        from elitefurretai.rl.rl_utils import adaptive_score, adaptive_distribution

        self._load_exploiter_models()
        self._load_ghosts()

        cfg = self.adaptive_agent_axis
        if not cfg.enabled:
            return

        base = self.curriculum.copy()

        # Compute per-opp scores. Under-warm or unavailable opponents
        # fall back to the base curriculum weight (preserves prior
        # preference, prevents oscillation on tiny samples).
        scores: Dict[str, float] = {}
        for opp_type, base_weight in base.items():
            if not self._opponent_available(opp_type):
                scores[opp_type] = 0.0
                continue
            wins, n = self.agent_win_rates.get(opp_type, (0.0, 0.0))
            if n < cfg.min_samples:
                scores[opp_type] = max(base_weight, 1e-2)
                continue
            scores[opp_type] = max(
                1e-2,
                adaptive_score(
                    wins=wins, n=n,
                    prior_alpha=cfg.prior_alpha, prior_beta=cfg.prior_beta,
                    pfsp_mix=cfg.pfsp_mix, weakness_mix=cfg.weakness_mix,
                    weakness_exponent=cfg.weakness_exponent,
                    target_win_rate=cfg.target_win_rate,
                ),
            )

        # Anchor floors (minimums). Only applied to opp types that are
        # currently available.
        floors: Dict[str, float] = {}
        if self._opponent_available(OpponentPool.SELF_PLAY):
            floors[OpponentPool.SELF_PLAY] = 0.20
        if self._opponent_available(OpponentPool.BC_PLAYER):
            floors[OpponentPool.BC_PLAYER] = 0.10
        if self._opponent_available(OpponentPool.GHOSTS):
            floors[OpponentPool.GHOSTS] = 0.10

        # Drop unavailable types entirely before normalization.
        scores = {k: v for k, v in scores.items() if self._opponent_available(k)}
        if not scores:
            self.curriculum = {OpponentPool.SELF_PLAY: 1.0}
            return

        new_curriculum = adaptive_distribution(
            scores,
            base=base,
            base_blend=cfg.base_blend,
            floors=floors,
        )
        self.curriculum = normalize_curriculum(new_curriculum)
```

- [ ] **Step 2: Run all RL tests**

Run: `source ../venv/bin/activate && pytest unit_tests/rl -v`
Expected: PASS. The `test_adaptive_curriculum.py` goldens may drift slightly due to EWMA vs sliding-window. Recompute affected goldens by feeding the same fixture stream through `record_battle_result` and reading `update_curriculum` output; update assertions.

- [ ] **Step 3: Full quality gates**

Run: `source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_adaptive_curriculum.py
git commit -m "refactor(rl/opponents): update_curriculum delegates to shared adaptive_distribution"
```

---

## Phase 6 — YAML migration, docs, planning-doc update

### Task 6.1: Migrate every YAML in `src/elitefurretai/rl/configs/` to the nested form

**Files:**
- Modify: `src/elitefurretai/rl/configs/easy_test.yaml`, `may15.yaml`, `may16.yaml`, `may25.yaml`

- [ ] **Step 1: For each YAML, replace the flat curriculum fields with nested blocks**

For each file under `src/elitefurretai/rl/configs/*.yaml`:

Find any of these top-level keys under `curriculum:` and delete them:

```yaml
  adaptive_curriculum: <value>
  team_axis_enabled: <value>
  team_warmup_threshold: <value>
  team_per_team_floor: <value>
  half_life: <value>
  pfsp_exponent: <value>
```

Replace with (using the deleted values where present, defaults otherwise):

```yaml
  adaptive_team_axis:
    enabled: <former team_axis_enabled, default true>
    min_samples: <former team_warmup_threshold, default 20>
    per_key_floor: <former team_per_team_floor, default 0.005>
    half_life: <former half_life, default 50.0>
    weakness_exponent: <former pfsp_exponent, default 1.0>
  adaptive_agent_axis:
    enabled: <former adaptive_curriculum, default true>
    # min_samples, half_life, pfsp_mix, weakness_mix, base_blend etc.
    # left at defaults unless tuned for this run.
```

- [ ] **Step 2: Smoke-load each config to confirm parse**

Run for each YAML:

```bash
source ../venv/bin/activate && python -c "from elitefurretai.rl.config import Config; c = Config.from_yaml('src/elitefurretai/rl/configs/easy_test.yaml'); print(c.curriculum.adaptive_team_axis); print(c.curriculum.adaptive_agent_axis)"
```

Expected: prints the AdaptiveAxisConfig instances with the migrated values.

Repeat for `may15.yaml`, `may16.yaml`, `may25.yaml`.

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/rl/configs/
git commit -m "config(rl): migrate YAMLs to nested adaptive_{team,agent}_axis blocks"
```

### Task 6.2: Update `RL.md`

**Files:**
- Modify: `src/elitefurretai/rl/RL.md`

- [ ] **Step 1: Update §7 (team-axis section) to reflect shared primitive**

Find the §7 team-axis paragraph (around line 349, mentioning `current_team_name`). Replace the relevant sentences:

- Replace `RLTrajectoryPlayer.current_team_name` with `RLTrajectoryPlayer.current_team_names[battle_tag]` and note the per-battle keying + `_pending_team_name` plumbing.
- Add a sentence: "The team-axis update shares an implementation with the agent-axis (opponent-mix) update — see [`rl_utils.py`](rl_utils.py) for the `adaptive_score` / `adaptive_distribution` primitives and `AdaptiveAxisConfig.{team,agent}_axis_defaults` in [`config.py`](config.py) for the parameter values that distinguish the two axes."

- [ ] **Step 2: Add a short §X note about agent-axis EWMA smoothing**

After §7, add a short subsection noting that the agent-axis curriculum now uses EWMA smoothing (same shape as team-axis), with `half_life=100` as the default (≈ the previous tracking_window=100 sliding window's effective memory for a stationary signal).

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/rl/RL.md
git commit -m "docs(rl): note shared adaptive primitive and per-battle team-name keying"
```

### Task 6.3: Final integration check + planning-doc Update

**Files:**
- Modify: `planning/stage2/2026-05-26-08-00-curriculum-unification-and-team-fixes.md` (this doc — add an Updates section at the bottom)

- [ ] **Step 1: Run the full quality gate suite**

Run: `source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q`
Expected: clean across the board.

- [ ] **Step 2: Smoke-run training launch (no actual battles needed; just confirm Config + WorkerOpponentFactory wiring loads)**

Run: `source ../venv/bin/activate && python -c "from elitefurretai.rl.config import Config; from elitefurretai.rl.opponents import WorkerOpponentFactory; cfg = Config.from_yaml('src/elitefurretai/rl/configs/may25.yaml'); print('loaded')"`
Expected: prints `loaded` without exception.

- [ ] **Step 3: Append an Updates section to this planning doc**

Append at the end of this file:

```markdown
## Updates

### 2026-05-26 (post-implementation)

- All 6 phases shipped. Plan executed in {N} commits.
- Behavior delta from agent-axis sliding-window → EWMA: validated empirically on a {short / N-step} smoke run; the new curriculum trajectory matches the old to within {tolerance} on stationary streams.
- Open follow-up: {anything that came up during implementation}.
```

(Fill in the bracketed sections during execution.)

- [ ] **Step 4: Commit the planning-doc update**

```bash
git add planning/stage2/2026-05-26-08-00-curriculum-unification-and-team-fixes.md
git commit -m "docs(planning): record completion of curriculum unification plan"
```

---

## Self-Review

**1. Spec coverage:**
- Q1 (TeamRepo merge) → Phase 1 ✓
- Q2 (water-filling pulled out) → Phase 3 (`adaptive_distribution`) ✓
- Q3 (per-battle team_name dict) → Phase 2 (with eviction on every termination path) ✓
- Q4 (shared algorithm + same params) → Phases 3, 4, 5 (full unification with EWMA on both) ✓
- Q5 (rename to `adaptive_{team,agent}_axis`) → Phase 4 + Phase 6 (YAML migration) ✓

**2. Placeholders:** No "TODO", "fill in", or unreferenced-symbol placeholders. The bracketed values in the Updates section (Task 6.3, Step 3) are explicit fill-in instructions for the executor.

**3. Type consistency:**
- `AdaptiveAxisConfig` field names are identical across `team_axis_defaults`, `agent_axis_defaults`, and the call sites in `opponents.py`.
- `_pending_team_name` / `current_team_names` / `_stamp_pending_team_name` / `_pop_team_name` are used consistently in the test (Task 2.1), implementation (Tasks 2.2, 2.3), and the docs update (Task 6.2).
- `adaptive_score` / `adaptive_distribution` signatures in the implementation (Task 3.2) match what the tests (Task 3.1) and callers (Tasks 5.2, 5.3) assume.

---

## Updates

### 2026-05-26 (post-implementation)

All six phases shipped on branch `curriculum-unification`. Commit summary:

| Phase | Commit(s) | Notes |
|---|---|---|
| Baseline ruff cleanup | `6fca0f0` | Pre-existing import-sort fixes (unrelated to plan; kept gates clean) |
| 1 — `sample_team` delegation | `34ff7fd` + `9501b93` | Test fixture deviation: VGC format loader requires 6-mon team strings. Plan was patched to flag for any future test scaffolding. |
| 2 — Per-battle `current_team_names` dict | `c2f27be` + `f654db7` | Test-cascade fix to `_make_player_for_popup_tests` helper was contained in the same commit. Dropped tautological attribute-presence test in followup `f654db7`. |
| 3 — `rl_utils` primitives | `fafb7d4` | 9 unit tests; algorithm verified by hand-walk against existing team-axis goldens. |
| 4 — `AdaptiveAxisConfig` | `35997ea` + `831e19e` | Plan said `WorkerOpponentFactory` but the state owner is actually `OpponentPool`; docstring fix in `831e19e` corrected the misattribution. YAML migration (planned for Phase 6.1) was folded in here to avoid silently re-enabling adaptive curriculum on 7 training configs. |
| 5.1 — `OpponentPool` constructor + EWMA storage | `5dc5fb6` | Caught and fixed a latent bare-local reference at the old line 273. `TYPE_CHECKING` split needed for the quoted annotation. |
| 5.2 — EWMA `record_battle_result` + `update_team_distribution` rewire | `0547d05` | No team-axis golden drift — hand-walk confirmed `(1-wr)^p` reproduces from `adaptive_score` with `target_win_rate=1.0`. |
| 5.3 — `update_curriculum` rewrite | `38fff10` + `a9003d6` | Lands the announced agent-axis behavior delta: sliding-window deque → EWMA (long-run identical, transient smoother). Pyright fix `a9003d6` narrowly suppresses a non-Optional assignment in the bypass-init test pattern from Phase 2. |
| 6 — RL.md cleanup + final QA + this update | `<this commit's SHA>` | Final `ruff check src unit_tests`, `ruff format --check src unit_tests`, `pyright src unit_tests`, `pytest unit_tests -q` all green. |

**Behavior delta validation:**
- Agent-axis curriculum smoothing changed from sliding-window deque (maxlen=100) to EWMA (half_life=100). For stationary win-rate streams the long-run estimate is unchanged; transient response is smoother and slightly slower. Acceptable per the plan's announced delta.
- Agent-axis floor application changed from "reserve floor_sum then distribute residual" to water-filling. Both meet anchor floors as minimums; water-filling is more natural under score-weighted scenarios.

**Test-coverage follow-ups (not in this plan's scope):**
- `unit_tests/rl/test_adaptive_curriculum.py` is a zero-byte placeholder. A direct round-trip test (seed `agent_win_rates`, call `update_curriculum`, assert anchor floors and score mixing) would tighten Phase 5.3 coverage.
- Residual `self.team_axis_enabled` scalar reads at `opponents.py:276` and `:290` could be migrated to read through `self.adaptive_team_axis.enabled` for full source-of-truth unification. Phase 5.3 unified the hot-path read (in `record_battle_result`); the construction-time reads were deferred.
- `_make_opponent_pool` test helper in `unit_tests/rl/test_team_axis.py` does not yet expose `adaptive_agent_axis` overrides; agent-axis tests landing later will need it.

**Cumulative diff (approximate):** ~11 commits, ~150 net lines added, ~250 net lines removed (much of the gain comes from collapsing two ~100-line `update_*` methods into shared-primitive callers).
