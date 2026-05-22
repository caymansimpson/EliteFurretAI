# Multi-Format Doubles RL Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a single RL model across multiple gen9 doubles formats (e.g. `gen9vgc2024regg`, `gen9vgc2024regh`, `gen9vgc2024rege`) by promoting `battle_format` from a single string to a weighted distribution. Each rollout pair is pinned to one sampled format for the duration of training; opponent and team sampling routes through the matching format folder.

**Architecture:**

- `CurriculumConfig.battle_format: str` becomes `battle_formats: Dict[str, float]` (a probability distribution that sums to 1.0). A `primary_format` property returns the highest-weight format and is used for embedder construction only — vocab building uses `format_str[3]` (the gen number), which is identical across all gen9 doubles formats, so the embedder remains shared.
- `WorkerOpponentFactory` is rebuilt to hold a format distribution. At `create_agents(num_pairs)` it samples one format per pair via largest-remainder apportionment (deterministic), tags each pair with its format, and uses that tag for team sampling and Player construction.
- `agent_team_path` and `opponent_team_pool_path` become per-format dicts when multiple formats are active; `resolved_agent_team_path()` returns a `Dict[str, str]`.
- VGCBench (v2 multi-format) and the single shared BC checkpoint stay as-is — both already generalize across formats per the user's confirmation.
- Per-format evaluation aggregation is sketched as a follow-up task (Task 7), not implemented in this plan.

**Tech Stack:** Python 3.12, PyTorch, poke-env, ruff, pyright, pytest. Project convention: `source ../venv/bin/activate` before any command.

**Reference docs:**
- [CLAUDE.md](../../CLAUDE.md) — project conventions; backwards compat not a concern
- [src/elitefurretai/rl/RL.md](../../src/elitefurretai/rl/RL.md) — RL architecture overview
- [src/elitefurretai/etl/ETL.md](../../src/elitefurretai/etl/ETL.md) — embedder + team_repo contracts

---

## Pre-flight

- [ ] **Step 1: Confirm clean working tree for files this plan touches**

Run:
```bash
git status -s src/elitefurretai/rl/config.py src/elitefurretai/rl/opponents.py src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py src/elitefurretai/engine/vgc_environment.py src/elitefurretai/rl/configs/single_team.yaml unit_tests/rl/test_config.py unit_tests/rl/test_worker_opponent_factory.py
```
Expected: empty output. Pre-existing modifications elsewhere (e.g. `src/elitefurretai/supervised/`, `data/`, `planning/`) are acceptable — do not touch them.

- [ ] **Step 2: Activate venv and confirm baseline quality gates pass**

Run:
```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: all green. If anything is red here, stop — fix or revert before starting so we don't confuse pre-existing failures with regressions introduced by this plan.

- [ ] **Step 3: Snapshot the call sites that read `battle_format`**

Run:
```bash
grep -rn "battle_format" src unit_tests --include='*.py' --include='*.yaml' > /tmp/multi_format_before.txt
wc -l /tmp/multi_format_before.txt
```
Expected: ~30–60 matches. Keep the file — Task 5 reruns the grep and the count of remaining `\.battle_format\b` reads (outside the new `primary_format` property body) must drop to zero in `rl/`, `engine/`, and the relevant test files.

---

## Task 1: Config schema — `battle_formats` distribution + `primary_format`

**Files:**
- Modify: `src/elitefurretai/rl/config.py:435-510` (`CurriculumConfig`)
- Modify: `unit_tests/rl/test_config.py:38, 119-134, 244, 274, 540` (existing `battle_format` assertions)
- Test: `unit_tests/rl/test_config.py` (add new cases)

**Design:**

```python
# Replace
battle_format: str = "gen9vgc2023regc"

# With
battle_formats: Dict[str, float] = field(
    default_factory=lambda: {"gen9vgc2023regc": 1.0}
)

@property
def primary_format(self) -> str:
    """Highest-weight format. Used for embedder construction (gen-keyed vocab only)."""
    return max(self.battle_formats.items(), key=lambda kv: kv[1])[0]
```

Add a `__post_init__` validator: weights must be positive, sum to 1.0 ± 1e-6, and keys must be non-empty strings.

- [ ] **Step 1.1: Write the failing test for `battle_formats` default + `primary_format`**

In `unit_tests/rl/test_config.py`, add:

```python
def test_battle_formats_default_is_single_format_distribution():
    """Default CurriculumConfig has battle_formats == {default: 1.0}."""
    config = get_default_config()
    assert config.curriculum.battle_formats == {"gen9vgc2023regc": 1.0}
    assert config.curriculum.primary_format == "gen9vgc2023regc"


def test_battle_formats_validation_rejects_non_unit_sum():
    from elitefurretai.rl.config import CurriculumConfig
    with pytest.raises(ValueError, match="must sum to 1.0"):
        CurriculumConfig(battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.4})


def test_battle_formats_validation_rejects_negative_weight():
    from elitefurretai.rl.config import CurriculumConfig
    with pytest.raises(ValueError, match="positive"):
        CurriculumConfig(battle_formats={"gen9vgc2024regg": 1.2, "gen9vgc2024regh": -0.2})


def test_primary_format_returns_highest_weight():
    from elitefurretai.rl.config import CurriculumConfig
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3}
    )
    assert cur.primary_format == "gen9vgc2024regg"
```

- [ ] **Step 1.2: Run the new tests to verify they fail**

Run:
```bash
pytest unit_tests/rl/test_config.py::test_battle_formats_default_is_single_format_distribution unit_tests/rl/test_config.py::test_battle_formats_validation_rejects_non_unit_sum unit_tests/rl/test_config.py::test_battle_formats_validation_rejects_negative_weight unit_tests/rl/test_config.py::test_primary_format_returns_highest_weight -v
```
Expected: 4 FAILs (`AttributeError: ... no attribute 'battle_formats'` on the first two; `TypeError` or absent property on the others).

- [ ] **Step 1.3: Replace `battle_format` field with `battle_formats` in `CurriculumConfig`**

In `src/elitefurretai/rl/config.py`, locate the `CurriculumConfig` block at line 435 and apply:

```python
@dataclass
class CurriculumConfig:
    """Opponent sampling, team pools, BC models, and ghost/exploiter directories.

    Path conventions
    ----------------
    `opponent_team_pool_path` and `agent_team_path` are resolved relative to
    `<base_team_path>/<format>/` per their format key (see Task 2). Both may
    be a single string (used for every format) or a `Dict[str, str]` keyed by
    format. `resolved_agent_team_paths()` returns a `Dict[str, str]` keyed
    by format.
    """

    agent_team_path: Optional[Union[str, Dict[str, str]]] = None
    base_team_path: str = "data/teams"
    # Probability distribution over battle formats. Must sum to 1.0.
    # Each rollout pair is pinned to one sampled format at create_agents time
    # (see WorkerOpponentFactory). The embedder is built once against
    # primary_format — vocab is gen-keyed (format_str[3]) so all entries must
    # share the same gen.
    battle_formats: Dict[str, float] = field(
        default_factory=lambda: {"gen9vgc2023regc": 1.0}
    )
    opponent_team_pool_path: Optional[Union[str, Dict[str, str]]] = None
    # ... (rest of the existing fields unchanged: bc_model_path, curriculum_weights, etc.)
```

Add `Union` to the existing `typing` import at the top of the file if it isn't already imported.

Add a `__post_init__` validator immediately after the field declarations:

```python
    def __post_init__(self) -> None:
        if not self.battle_formats:
            raise ValueError("battle_formats must not be empty")
        for fmt, weight in self.battle_formats.items():
            if not isinstance(fmt, str) or not fmt:
                raise ValueError(f"battle_formats key must be non-empty string, got {fmt!r}")
            if weight <= 0:
                raise ValueError(
                    f"battle_formats weight for {fmt!r} must be positive, got {weight}"
                )
        total = sum(self.battle_formats.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"battle_formats weights must sum to 1.0, got {total} "
                f"({self.battle_formats})"
            )
        gens = {fmt[3] for fmt in self.battle_formats}
        if len(gens) > 1:
            raise ValueError(
                f"All battle_formats must share the same gen (format[3]); got {gens}"
            )
```

Add the `primary_format` property after the `__post_init__`:

```python
    @property
    def primary_format(self) -> str:
        """Highest-weight format. Used for embedder construction only.

        Vocab is gen-keyed (see embedder.build_*_to_id), so the chosen
        primary_format determines the gen but not which species are
        embeddable — every species in the gen's pokedex is reachable
        regardless of which doubles format is sampled at runtime.
        """
        return max(self.battle_formats.items(), key=lambda kv: kv[1])[0]
```

- [ ] **Step 1.4: Update existing `battle_format` assertions in `test_config.py`**

Replace every `config.curriculum.battle_format` access (lines 38, 119, 134, 244, 274, 540) with the new schema. Pattern:

```python
# Before
assert config.curriculum.battle_format == "gen9vgc2023regc"

# After
assert config.curriculum.primary_format == "gen9vgc2023regc"
assert config.curriculum.battle_formats == {"gen9vgc2023regc": 1.0}
```

For the `to_dict`/`from_dict` round-trip cases at line 119 (`config.curriculum.battle_format = "gen9vgc2024regg"`), use:

```python
config.curriculum.battle_formats = {"gen9vgc2024regg": 1.0}
```

For the existence-in-dict assertions at lines 244 and 274 (`assert "battle_format" in data["curriculum"]`), update to:

```python
assert "battle_formats" in data["curriculum"]
```

- [ ] **Step 1.5: Run the full config test suite to verify it passes**

Run:
```bash
pytest unit_tests/rl/test_config.py -v
```
Expected: all PASS (existing + 4 new tests).

- [ ] **Step 1.6: Commit**

Run:
```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "rl/config: replace battle_format with battle_formats distribution + primary_format

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Per-format `agent_team_paths` and `opponent_team_pool_paths` resolution

**Files:**
- Modify: `src/elitefurretai/rl/config.py:500-510` (`resolved_agent_team_path`)
- Modify: `unit_tests/rl/test_config.py` (add resolution tests)

**Design:**

Current `resolved_agent_team_path() -> Optional[str]` returns one path under `<base_team_path>/<battle_format>/`. After this task it becomes `resolved_agent_team_paths() -> Dict[str, str]` returning one resolved path per format (or `{}` when `agent_team_path is None`).

Symmetric helper for the opponent pool: `resolved_opponent_team_pool_paths() -> Dict[str, Optional[str]]`. The dict is keyed by every format in `battle_formats`; each value is the subdirectory string to pass to `TeamRepo.sample_team(format, subdirectory=...)`, or `None` when no subdirectory was configured.

- [ ] **Step 2.1: Write the failing tests for `resolved_agent_team_paths` and `resolved_opponent_team_pool_paths`**

Append to `unit_tests/rl/test_config.py`:

```python
def test_resolved_agent_team_paths_string_form_broadcasts_to_all_formats():
    from elitefurretai.rl.config import CurriculumConfig
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
        base_team_path="data/teams",
        agent_team_path="constrained",
    )
    paths = cur.resolved_agent_team_paths()
    assert paths == {
        "gen9vgc2024regg": "data/teams/gen9vgc2024regg/constrained",
        "gen9vgc2024regh": "data/teams/gen9vgc2024regh/constrained",
    }


def test_resolved_agent_team_paths_dict_form_used_verbatim_per_format():
    from elitefurretai.rl.config import CurriculumConfig
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.7, "gen9vgc2024regh": 0.3},
        base_team_path="data/teams",
        agent_team_path={"gen9vgc2024regg": "constrained", "gen9vgc2024regh": "rentals"},
    )
    paths = cur.resolved_agent_team_paths()
    assert paths == {
        "gen9vgc2024regg": "data/teams/gen9vgc2024regg/constrained",
        "gen9vgc2024regh": "data/teams/gen9vgc2024regh/rentals",
    }


def test_resolved_agent_team_paths_dict_form_rejects_missing_format():
    from elitefurretai.rl.config import CurriculumConfig
    with pytest.raises(ValueError, match="missing entry for format"):
        CurriculumConfig(
            battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5},
            agent_team_path={"gen9vgc2024regg": "constrained"},  # missing regh
        )


def test_resolved_agent_team_paths_none_returns_empty_dict():
    from elitefurretai.rl.config import CurriculumConfig
    cur = CurriculumConfig(battle_formats={"gen9vgc2024regg": 1.0}, agent_team_path=None)
    assert cur.resolved_agent_team_paths() == {}


def test_resolved_opponent_team_pool_paths_matches_format_keys():
    from elitefurretai.rl.config import CurriculumConfig
    cur = CurriculumConfig(
        battle_formats={"gen9vgc2024regg": 0.6, "gen9vgc2024regh": 0.4},
        opponent_team_pool_path={"gen9vgc2024regg": "ranked", "gen9vgc2024regh": None},
    )
    assert cur.resolved_opponent_team_pool_paths() == {
        "gen9vgc2024regg": "ranked",
        "gen9vgc2024regh": None,
    }
```

- [ ] **Step 2.2: Run the new tests to verify they fail**

Run:
```bash
pytest unit_tests/rl/test_config.py -k "resolved_agent_team_paths or resolved_opponent_team_pool_paths" -v
```
Expected: 5 FAILs (`AttributeError: ... no attribute 'resolved_agent_team_paths'` etc.).

- [ ] **Step 2.3: Extend `__post_init__` to validate dict-form per-format paths**

In `CurriculumConfig.__post_init__` (added in Task 1.3), append:

```python
        for attr in ("agent_team_path", "opponent_team_pool_path"):
            value = getattr(self, attr)
            if isinstance(value, dict):
                missing = set(self.battle_formats) - set(value)
                if missing:
                    raise ValueError(
                        f"{attr} (dict form) is missing entry for format(s) "
                        f"{sorted(missing)}; got keys {sorted(value)}"
                    )
```

- [ ] **Step 2.4: Replace `resolved_agent_team_path` with `resolved_agent_team_paths` and add the opponent helper**

In `src/elitefurretai/rl/config.py:500-510`, replace the existing `resolved_agent_team_path` method with:

```python
    def resolved_agent_team_paths(self) -> Dict[str, str]:
        """Return absolute agent-team paths per format, or {} if unset.

        - `agent_team_path is None` -> {}
        - `agent_team_path: str` -> {fmt: <base>/<fmt>/<path> for fmt in battle_formats}
        - `agent_team_path: Dict[str, str]` -> {fmt: <base>/<fmt>/<path[fmt]>}
        """
        if self.agent_team_path is None:
            return {}
        if isinstance(self.agent_team_path, dict):
            return {
                fmt: os.path.join(self.base_team_path, fmt, self.agent_team_path[fmt])
                for fmt in self.battle_formats
            }
        return {
            fmt: os.path.join(self.base_team_path, fmt, self.agent_team_path)
            for fmt in self.battle_formats
        }

    def resolved_opponent_team_pool_paths(self) -> Dict[str, Optional[str]]:
        """Return per-format opponent-team subdirectory values for TeamRepo.

        These are subdirectory strings (NOT full paths) since TeamRepo.sample_team
        takes (format, subdirectory=...). When opponent_team_pool_path is None,
        every format maps to None. When it's a string, it broadcasts to all
        formats. When it's a dict, it's used verbatim.
        """
        if self.opponent_team_pool_path is None:
            return {fmt: None for fmt in self.battle_formats}
        if isinstance(self.opponent_team_pool_path, dict):
            return {fmt: self.opponent_team_pool_path[fmt] for fmt in self.battle_formats}
        return {fmt: self.opponent_team_pool_path for fmt in self.battle_formats}
```

- [ ] **Step 2.5: Update the one external caller of `resolved_agent_team_path` (singular)**

The current call site is in `engine/vgc_environment.py:231`:
```python
agent_team_path=cur.resolved_agent_team_path(),
```

It will be replaced wholesale in Task 5. For now (to keep this task's commits self-contained and the suite green), add a thin backwards-compat alias right after `resolved_opponent_team_pool_paths`:

```python
    def resolved_agent_team_path(self) -> Optional[str]:
        """Legacy single-format alias. Returns the path for `primary_format`,
        or None if `agent_team_path` is unset. Removed by Task 5.
        """
        paths = self.resolved_agent_team_paths()
        return paths.get(self.primary_format)
```

This will be deleted in Task 5 once `vgc_environment.py` is updated.

- [ ] **Step 2.6: Run the new tests + the existing config suite**

Run:
```bash
pytest unit_tests/rl/test_config.py -v
```
Expected: all PASS.

- [ ] **Step 2.7: Commit**

Run:
```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "rl/config: per-format resolved_agent_team_paths + opponent_team_pool_paths

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `WorkerOpponentFactory` accepts format distribution and assigns per-pair format

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py:491-560` (`WorkerOpponentFactory.__init__`)
- Modify: `unit_tests/rl/test_worker_opponent_factory.py` (the existing fixture at line 48)

**Design:**

Replace `battle_format: str` with `battle_formats: Dict[str, float]`. Replace `team_subdirectory: Optional[str]` with `opponent_team_subdirectories: Dict[str, Optional[str]]` (one entry per format). Replace `agent_team_path: Optional[str]` with `agent_team_paths: Dict[str, str]` (one entry per format, or empty dict when unset).

Add `self.pair_formats: List[str]` (empty until `create_agents` is called). Populate it via `largest_remainder_apportionment(num_pairs, self.battle_formats)` so distribution across pairs is deterministic and as close as possible to the configured weights.

- [ ] **Step 3.1: Write the failing test for largest-remainder apportionment**

In `src/elitefurretai/rl/opponents.py` we'll add a small helper `largest_remainder_apportionment`. Add tests for it first.

In `unit_tests/rl/test_worker_opponent_factory.py`, add at the top of the file (after existing imports):

```python
from elitefurretai.rl.opponents import largest_remainder_apportionment


def test_largest_remainder_apportionment_exact_split():
    """8 pairs across two formats 50/50 -> [fmt_a]*4 + [fmt_b]*4."""
    result = largest_remainder_apportionment(
        num_items=8, weights={"a": 0.5, "b": 0.5}
    )
    assert sorted(result) == ["a"] * 4 + ["b"] * 4


def test_largest_remainder_apportionment_handles_remainder():
    """5 pairs across 70/30 -> 4 of 'a', 1 of 'b' (round 3.5 -> 4 via remainder)."""
    result = largest_remainder_apportionment(
        num_items=5, weights={"a": 0.7, "b": 0.3}
    )
    counts = {fmt: result.count(fmt) for fmt in result}
    assert counts == {"a": 4, "b": 1}


def test_largest_remainder_apportionment_deterministic():
    """Same inputs -> same output (order matters for reproducibility)."""
    r1 = largest_remainder_apportionment(num_items=7, weights={"x": 0.4, "y": 0.6})
    r2 = largest_remainder_apportionment(num_items=7, weights={"x": 0.4, "y": 0.6})
    assert r1 == r2


def test_largest_remainder_apportionment_single_format():
    """Single-format distribution -> every slot gets that format."""
    result = largest_remainder_apportionment(num_items=4, weights={"only": 1.0})
    assert result == ["only"] * 4
```

- [ ] **Step 3.2: Run the new tests to verify they fail**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -k largest_remainder -v
```
Expected: 4 FAILs (`ImportError: cannot import name 'largest_remainder_apportionment'`).

- [ ] **Step 3.3: Add `largest_remainder_apportionment` helper to `opponents.py`**

In `src/elitefurretai/rl/opponents.py`, add this module-level function near the top (after imports, before any class definition):

```python
def largest_remainder_apportionment(
    num_items: int, weights: Dict[str, float]
) -> List[str]:
    """Distribute `num_items` slots across `weights.keys()` proportionally.

    Uses the largest-remainder (Hamilton) method: floor every quota first,
    then assign leftover slots one-at-a-time to the keys with the largest
    fractional remainders. Deterministic — ties are broken by sorted key
    order. Returns a list of length `num_items`.

    Example
    -------
    >>> largest_remainder_apportionment(5, {"a": 0.7, "b": 0.3})
    ['a', 'a', 'a', 'a', 'b']
    """
    if num_items <= 0 or not weights:
        return []
    quotas = {fmt: weights[fmt] * num_items for fmt in weights}
    floors = {fmt: int(quotas[fmt]) for fmt in weights}
    remainders = sorted(
        ((quotas[fmt] - floors[fmt], fmt) for fmt in weights),
        key=lambda pair: (-pair[0], pair[1]),
    )
    leftover = num_items - sum(floors.values())
    for _, fmt in remainders[:leftover]:
        floors[fmt] += 1
    result: List[str] = []
    for fmt in sorted(weights):
        result.extend([fmt] * floors[fmt])
    return result
```

- [ ] **Step 3.4: Run the apportionment tests and confirm they pass**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -k largest_remainder -v
```
Expected: 4 PASS.

- [ ] **Step 3.5: Rewrite the `WorkerOpponentFactory.__init__` signature for format distribution**

In `src/elitefurretai/rl/opponents.py:491-557`, replace the constructor signature + body with:

```python
    def __init__(
        self,
        team_repo: TeamRepo,
        battle_formats: Dict[str, float],
        opponent_team_subdirectories: Dict[str, Optional[str]],
        server_config: ServerConfiguration,
        curriculum: Optional[Dict[str, float]],
        embedder: Embedder,
        worker_id: int,
        run_id: str,
        worker_inference_clients: WorkerInferenceClients,
        max_battle_steps: int = 40,
        external_vgcbench_usernames: Optional[List[str]] = None,
        agent_team_paths: Optional[Dict[str, str]] = None,
        max_concurrent_battles_per_player: Optional[int] = None,
    ):
        self.team_repo = team_repo
        self.battle_formats = dict(battle_formats)
        self.opponent_team_subdirectories = dict(opponent_team_subdirectories)
        self.server_config = server_config
        self.worker_inference_clients = worker_inference_clients
        self.max_concurrent_battles_per_player = max_concurrent_battles_per_player
        self.curriculum = curriculum or {OpponentPool.SELF_PLAY: 1.0}
        self.embedder = embedder
        self.worker_id = worker_id
        self.run_id = run_id
        self.max_battle_steps = max_battle_steps

        # Per-format agent team strings. Keys match self.battle_formats; each
        # value is the list of team strings loaded from disk for that format
        # (or [] if no agent_team_paths entry was given for that format,
        # which means "sample from opponent_team_subdirectories instead").
        self._agent_teams_by_format: Dict[str, List[str]] = {
            fmt: [] for fmt in self.battle_formats
        }
        if agent_team_paths:
            for fmt, path in agent_team_paths.items():
                teams: List[str] = []
                if os.path.isdir(path):
                    for fname in sorted(os.listdir(path)):
                        if fname.endswith(".txt"):
                            with open(os.path.join(path, fname)) as f:
                                teams.append(f.read())
                    logger.info(
                        "Loaded %d agent teams for %s from directory %s",
                        len(teams), fmt, path,
                    )
                else:
                    with open(path) as f:
                        teams.append(f.read())
                self._agent_teams_by_format[fmt] = teams

        self.external_vgcbench_usernames = [
            username.strip()
            for username in (external_vgcbench_usernames or [])
            if username and username.strip()
        ]

        self.players: List[RLTrajectoryPlayer] = []
        self.opponents: List[RLTrajectoryPlayer] = []
        self.max_damage_opponents: List[MaxDamagePlayer] = []
        self.random_baseline_opponents: List[RandomPlayer] = []
        self.max_base_power_baseline_opponents: List[MaxBasePowerPlayer] = []
        self.simple_heuristic_baseline_opponents: List[Player] = []
        self._active_ghost_slots: Set[int] = set()
        self._active_exploiter_slots: Set[int] = set()
        self._batch_count = 0
        self._rebuild_generation = 0
        # Per-pair format assignment. Populated in create_agents().
        self.pair_formats: List[str] = []

        cleaned_run_id = "".join(ch for ch in str(self.run_id) if ch.isalnum()).upper()
        self._run_tag = cleaned_run_id[-4:] if cleaned_run_id else "0000"
        self._factory_tag = f"{random.getrandbits(8):02X}"
```

Also delete the now-removed `self.battle_format` and `self.team_subdirectory` attributes and the `self._agent_teams: List[str]` field. (Anything that referenced `self.battle_format` directly is rewritten in Task 4.)

- [ ] **Step 3.6: Update the existing factory fixture to use the new signature**

In `unit_tests/rl/test_worker_opponent_factory.py` around line 48, replace:

```python
return WorkerOpponentFactory(
    team_repo=...,
    battle_format="gen9vgc2023regc",
    team_subdirectory=...,
    ...
)
```

with:

```python
return WorkerOpponentFactory(
    team_repo=...,
    battle_formats={"gen9vgc2023regc": 1.0},
    opponent_team_subdirectories={"gen9vgc2023regc": None},
    ...
    agent_team_paths=None,
)
```

If the fixture's `MockTeamRepo` signature (line 14) takes `battle_format`, leave it — `TeamRepo.sample_team(format, ...)` is still the call we use.

- [ ] **Step 3.7: Add a new test for per-pair format apportionment in `create_agents`**

Append to `unit_tests/rl/test_worker_opponent_factory.py`:

```python
def test_create_agents_assigns_pair_formats_via_apportionment(monkeypatch):
    """4 pairs at 50/50 -> 2 of each format in self.pair_formats."""
    factory = make_factory_for_test(
        battle_formats={"gen9vgc2024regg": 0.5, "gen9vgc2024regh": 0.5}
    )
    # Avoid network: monkeypatch RLTrajectoryPlayer to a stub that records
    # the battle_format it was constructed with.
    constructed_formats: List[str] = []

    class _StubPlayer:
        def __init__(self, *, battle_format, **kwargs):
            constructed_formats.append(battle_format)

    monkeypatch.setattr(
        "elitefurretai.rl.opponents.RLTrajectoryPlayer", _StubPlayer
    )
    factory.create_agents(num_pairs=4, local_traj_queue=queue.Queue())
    assert sorted(factory.pair_formats) == [
        "gen9vgc2024regg",
        "gen9vgc2024regg",
        "gen9vgc2024regh",
        "gen9vgc2024regh",
    ]
    # Each pair contributes one self-player and one opponent-player (both
    # share the pair format), so we expect 2 entries in constructed_formats
    # per pair format.
    assert sorted(constructed_formats) == [
        "gen9vgc2024regg", "gen9vgc2024regg", "gen9vgc2024regg", "gen9vgc2024regg",
        "gen9vgc2024regh", "gen9vgc2024regh", "gen9vgc2024regh", "gen9vgc2024regh",
    ]
```

If `make_factory_for_test` does not exist in the test file, add it next to the existing fixture as a thin wrapper that takes a `battle_formats` kwarg and produces a `WorkerOpponentFactory` with mock dependencies, mirroring the existing fixture at line 48.

- [ ] **Step 3.8: Run the new test and confirm it fails (create_agents not updated yet)**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py::test_create_agents_assigns_pair_formats_via_apportionment -v
```
Expected: FAIL — likely `AssertionError` on pair_formats (empty list, since create_agents doesn't yet populate it) or on constructed_formats.

(`create_agents` will be updated in Task 4. The test stays red until then. Skip with `xfail` if needed: tag the test with `@pytest.mark.xfail(reason="create_agents per-pair format wired in Task 4", strict=True)` so the test suite stays green between tasks.)

- [ ] **Step 3.9: Run the full opponents test file to confirm fixture migration is clean**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -v --deselect unit_tests/rl/test_worker_opponent_factory.py::test_create_agents_assigns_pair_formats_via_apportionment
```
Expected: all PASS (i.e. fixture rewrite didn't break existing factory tests).

- [ ] **Step 3.10: Commit**

Run:
```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "rl/opponents: WorkerOpponentFactory accepts battle_formats distribution

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `WorkerOpponentFactory` — per-pair format wiring in `create_agents`, `sample_team`, baseline pools

**Files:**
- Modify: `src/elitefurretai/rl/opponents.py:598-720` (`sample_team`, `get_agent_team`, `_make_baseline_pool`, `create_agents`)
- Modify: `unit_tests/rl/test_worker_opponent_factory.py` (remove the `xfail` from Step 3.8)

**Design:**

1. `sample_team(self, fmt)` takes a format arg, routes to `team_repo.sample_team(fmt, subdirectory=self.opponent_team_subdirectories[fmt])`.
2. `get_agent_team(self, fmt)` returns a shuffled agent team for the given format if `self._agent_teams_by_format[fmt]` is non-empty, else falls back to `self.sample_team(fmt)`.
3. `_make_baseline_pool` takes a `pair_formats: List[str]` argument and constructs one player per pair, using `battle_format=pair_formats[i]` and `team=self.sample_team(pair_formats[i])`.
4. `create_agents(num_pairs, local_traj_queue)` calls `largest_remainder_apportionment(num_pairs, self.battle_formats)` once, stores as `self.pair_formats`, then iterates `for i, fmt in enumerate(self.pair_formats)` for the main player + opponent loop and passes `self.pair_formats` to `_make_baseline_pool`.

- [ ] **Step 4.1: Rewrite `sample_team` to take a format arg**

In `src/elitefurretai/rl/opponents.py:598-602`, replace:

```python
    def sample_team(self) -> str:
        return self.team_repo.sample_team(
            self.battle_format,
            subdirectory=self.team_subdirectory,
        )
```

with:

```python
    def sample_team(self, battle_format: str) -> str:
        return self.team_repo.sample_team(
            battle_format,
            subdirectory=self.opponent_team_subdirectories.get(battle_format),
        )
```

- [ ] **Step 4.2: Rewrite `get_agent_team` to take a format arg**

In `src/elitefurretai/rl/opponents.py:604-610`, replace:

```python
    def get_agent_team(self) -> str:
        if self._agent_teams:
            return self.team_repo._shuffle_team_order(random.choice(self._agent_teams))
        return self.sample_team()
```

with:

```python
    def get_agent_team(self, battle_format: str) -> str:
        """Return the agent's team for the given format, shuffled.

        Uses fixed team(s) from `_agent_teams_by_format[battle_format]` when
        any are loaded for this format; otherwise falls back to sampling from
        the opponent team pool.
        """
        teams = self._agent_teams_by_format.get(battle_format, [])
        if teams:
            return self.team_repo._shuffle_team_order(random.choice(teams))
        return self.sample_team(battle_format)
```

- [ ] **Step 4.3: Rewrite `_make_baseline_pool` to take per-pair formats**

In `src/elitefurretai/rl/opponents.py:612-638`, replace the body with:

```python
    def _make_baseline_pool(
        self,
        opp_type: str,
        player_cls: Type[_P],
        role: str,
        pair_formats: List[str],
    ) -> List[_P]:
        """Create one `player_cls` per entry in `pair_formats`, each pinned
        to that pair's format and team. Returns [] when curriculum weight is 0.
        """
        if self.curriculum.get(opp_type, 0) <= 0:
            return []
        return [
            player_cls(
                battle_format=fmt,
                account_configuration=AccountConfiguration(
                    self._account_name(role, i), None
                ),
                server_configuration=self.server_config,
                team=self.sample_team(fmt),
            )
            for i, fmt in enumerate(pair_formats)
        ]
```

- [ ] **Step 4.4: Rewrite `create_agents` to apportion per-pair formats**

In `src/elitefurretai/rl/opponents.py:640-720` (and the following baseline-pool block), replace the main loop with:

```python
    def create_agents(
        self,
        num_pairs: int,
        local_traj_queue: queue.Queue,
    ) -> Tuple[List[RLTrajectoryPlayer], List[RLTrajectoryPlayer], List[MaxDamagePlayer]]:
        extra_player_kwargs: Dict[str, Any] = {}
        if self.max_concurrent_battles_per_player is not None:
            extra_player_kwargs["max_concurrent_battles"] = (
                self.max_concurrent_battles_per_player
            )
        main_kwargs: Dict[str, Any] = {
            "inference_client": self.worker_inference_clients["main"]
        }

        # Apportion formats across pairs deterministically.
        self.pair_formats = largest_remainder_apportionment(
            num_items=num_pairs, weights=self.battle_formats
        )

        self.players = []
        self.opponents = []
        for i, fmt in enumerate(self.pair_formats):
            self.players.append(
                RLTrajectoryPlayer(
                    account_configuration=AccountConfiguration(
                        self._account_name("Self", i), None
                    ),
                    server_configuration=self.server_config,
                    trajectory_queue=local_traj_queue,
                    battle_format=fmt,
                    team=self.get_agent_team(fmt),
                    worker_id=self.worker_id,
                    embedder=self.embedder,
                    max_battle_steps=self.max_battle_steps,
                    opponent_type=OpponentPool.SELF_PLAY,
                    **main_kwargs,
                    **extra_player_kwargs,
                )
            )
            self.opponents.append(
                RLTrajectoryPlayer(
                    account_configuration=AccountConfiguration(
                        self._account_name("Opp", i), None
                    ),
                    server_configuration=self.server_config,
                    trajectory_queue=None,
                    battle_format=fmt,
                    team=self.sample_team(fmt),
                    worker_id=self.worker_id,
                    embedder=self.embedder,
                    max_battle_steps=self.max_battle_steps,
                    **main_kwargs,
                    **extra_player_kwargs,
                )
            )

        self.max_damage_opponents = self._make_baseline_pool(
            OpponentPool.MAX_DAMAGE, MaxDamagePlayer, "MaxD", self.pair_formats
        )
        # ... (apply the same pattern to random_baseline_opponents,
        # max_base_power_baseline_opponents, simple_heuristic_baseline_opponents:
        # replace the existing num_pairs argument with self.pair_formats)
        # The remainder of create_agents is unchanged.
        ...
```

Update every other `_make_baseline_pool` call in `create_agents` to pass `self.pair_formats` instead of `num_pairs`.

Locate any remaining `self.battle_format` reads inside `opponents.py` (Step 3.5 removed the attribute, so these are now `AttributeError`s waiting to happen). Each one corresponds to a path that wants the *pair's* format — replace with the appropriate `pair_formats[i]` or `fmt` from context.

- [ ] **Step 4.5: Remove the `xfail` marker from `test_create_agents_assigns_pair_formats_via_apportionment` (Step 3.8) and run it**

Edit `unit_tests/rl/test_worker_opponent_factory.py` to remove the `@pytest.mark.xfail(...)` decorator added in Step 3.8 (if applied). Then run:

```bash
pytest unit_tests/rl/test_worker_opponent_factory.py::test_create_agents_assigns_pair_formats_via_apportionment -v
```
Expected: PASS — pair_formats reflects 50/50 apportionment and the stub player was constructed with each format twice.

- [ ] **Step 4.6: Run the full factory test file**

Run:
```bash
pytest unit_tests/rl/test_worker_opponent_factory.py -v
```
Expected: all PASS.

- [ ] **Step 4.7: Quality gates**

Run:
```bash
ruff check src/elitefurretai/rl/opponents.py src/elitefurretai/rl/config.py && ruff format src/elitefurretai/rl/opponents.py src/elitefurretai/rl/config.py --check && pyright src/elitefurretai/rl/opponents.py src/elitefurretai/rl/config.py
```
Expected: all clean. If pyright complains about `Dict[str, Optional[str]]` type compatibility, add `# pyright: ignore[reportXxx]` only as a last resort — prefer fixing the annotation.

- [ ] **Step 4.8: Commit**

Run:
```bash
git add src/elitefurretai/rl/opponents.py unit_tests/rl/test_worker_opponent_factory.py
git commit -m "rl/opponents: per-pair format assignment via largest-remainder apportionment

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire `VGCEnvironment`, `worker.py`, `train.py` to use the new schema

**Files:**
- Modify: `src/elitefurretai/engine/vgc_environment.py:218-234` (`WorkerOpponentFactory` construction in `from_config`)
- Modify: `src/elitefurretai/rl/worker.py:143, 181` (embedder construction, control payload)
- Modify: `src/elitefurretai/rl/train.py:163, 209, 233, 341` (4 embedder constructions)
- Modify: `src/elitefurretai/rl/config.py` (remove the legacy `resolved_agent_team_path` alias from Task 2.5)

**Design:**

Replace every `config.curriculum.battle_format` read with `config.curriculum.primary_format` for embedder construction. Replace `cur.battle_format` / `cur.team_subdirectory` / `cur.resolved_agent_team_path()` in `vgc_environment.py` with the new dict-returning helpers.

- [ ] **Step 5.1: Update `VGCEnvironment.from_config` to pass format distribution to the factory**

In `src/elitefurretai/engine/vgc_environment.py:218-234`, replace:

```python
self._factory = WorkerOpponentFactory(
    team_repo=team_repo,
    battle_format=cur.battle_format,
    team_subdirectory=cur.opponent_team_pool_path,
    server_config=ServerConfiguration(...),
    ...
    agent_team_path=cur.resolved_agent_team_path(),
    ...
)
```

with:

```python
self._factory = WorkerOpponentFactory(
    team_repo=team_repo,
    battle_formats=cur.battle_formats,
    opponent_team_subdirectories=cur.resolved_opponent_team_pool_paths(),
    server_config=ServerConfiguration(
        f"ws://localhost:{server_port}/showdown/websocket", ""
    ),
    curriculum=self._curriculum,
    embedder=embedder,
    worker_id=worker_id,
    run_id=run_id,
    max_battle_steps=hw.max_battle_steps,
    external_vgcbench_usernames=external_vgcbench_usernames,
    agent_team_paths=cur.resolved_agent_team_paths() or None,
    max_concurrent_battles_per_player=hw.max_concurrent_battles_per_player,
    worker_inference_clients=worker_inference_clients,
)
```

(`or None` collapses `{}` to `None` for the factory's `agent_team_paths: Optional[Dict[str, str]]` signature.)

- [ ] **Step 5.2: Update `worker.py` embedder construction**

In `src/elitefurretai/rl/worker.py:143-184`, replace:

```python
battle_format = config.curriculum.battle_format
...
embedder = Embedder(
    format=battle_format,
    feature_set=embedder_feature_set,
    omniscient=False,
)
```

with:

```python
primary_format = config.curriculum.primary_format
...
embedder = Embedder(
    format=primary_format,
    feature_set=embedder_feature_set,
    omniscient=False,
)
```

The `base_team_path` line and TeamRepo construction stay unchanged — TeamRepo already auto-discovers every format folder under `base_team_path`.

- [ ] **Step 5.3: Update the 4 embedder sites in `train.py`**

In `src/elitefurretai/rl/train.py` at lines 163, 209, 233, 341, replace every:

```python
Embedder(
    format=config.curriculum.battle_format,
    feature_set=...,
    omniscient=False,
)
```

with:

```python
Embedder(
    format=config.curriculum.primary_format,
    feature_set=...,
    omniscient=False,
)
```

- [ ] **Step 5.4: Delete the legacy `resolved_agent_team_path` alias**

In `src/elitefurretai/rl/config.py` (added in Step 2.5), remove the `def resolved_agent_team_path(self) -> Optional[str]:` method. All callers were migrated in Step 5.1.

- [ ] **Step 5.5: Confirm no stale `battle_format` reads remain in core paths**

Run:
```bash
grep -rn "\.battle_format\b" src/elitefurretai/rl src/elitefurretai/engine --include='*.py' | grep -v '^.*config\.py:' | grep -v 'primary_format'
```
Expected: empty output. (Matches in `config.py` are allowed — that's the `primary_format` body. Comments mentioning the old field name are also acceptable but flag them for cleanup.)

- [ ] **Step 5.6: Run the full RL test suite + quality gates**

Run:
```bash
pytest unit_tests/rl -q && ruff check src unit_tests && ruff format src unit_tests --check && pyright src/elitefurretai/rl src/elitefurretai/engine
```
Expected: all green.

- [ ] **Step 5.7: Run the worker-level integration tests specifically**

Run:
```bash
pytest unit_tests/rl/test_worker.py -v
```
Expected: PASS (these tests exercise the worker bootstrap with a real Embedder; primary_format must route correctly).

- [ ] **Step 5.8: Commit**

Run:
```bash
git add src/elitefurretai/engine/vgc_environment.py src/elitefurretai/rl/worker.py src/elitefurretai/rl/train.py src/elitefurretai/rl/config.py
git commit -m "rl: wire VGCEnvironment + worker + train to multi-format config

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update `single_team.yaml` example + smoke training run

**Files:**
- Modify: `src/elitefurretai/rl/configs/single_team.yaml:87` (and the comment block on lines 116-120)
- Modify: any other yaml configs in `src/elitefurretai/rl/configs/` that still reference `battle_format`

**Design:**

Replace `battle_format: gen9vgc2024regg` with `battle_formats:\n  gen9vgc2024regg: 1.0`. Add a commented-out multi-format example so the schema is discoverable.

- [ ] **Step 6.1: Find every yaml config that references `battle_format`**

Run:
```bash
grep -rln "^[[:space:]]*battle_format:" src/elitefurretai/rl/configs/
```
Expected: returns `single_team.yaml`, `may15.yaml`, `may16.yaml`, `may22.yaml`, `easy_test.yaml`.

- [ ] **Step 6.2: Update `single_team.yaml`**

In `src/elitefurretai/rl/configs/single_team.yaml`, replace line 87:

```yaml
  battle_format: gen9vgc2024regg
```

with:

```yaml
  battle_formats:
    gen9vgc2024regg: 1.0
  # ──────────────────────────────────────────────────────────────────────
  # Multi-format example (all entries must share the same gen):
  # battle_formats:
  #   gen9vgc2024regg: 0.6
  #   gen9vgc2024regh: 0.4
  #
  # Per-format team subdirectories (sampled from
  # data/teams/<format>/<subdir>/). agent_team_path and
  # opponent_team_pool_path each accept three shapes:
  #
  #   1) null / unset — TeamRepo samples across the whole format folder
  #   2) a single string — broadcast to every active format:
  #        opponent_team_pool_path: constrained
  #   3) a dict keyed by format — different subdir per format:
  #        opponent_team_pool_path:
  #          gen9vgc2024regg: constrained
  #          gen9vgc2024regh: rentals
  #        agent_team_path:
  #          gen9vgc2024regg: constrained/38dessert.txt
  #          gen9vgc2024regh: setup
  #
  # In dict form every active format MUST appear as a key — the config
  # validator raises ValueError on missing entries (no implicit fallback).
  # ──────────────────────────────────────────────────────────────────────
```

- [ ] **Step 6.3: Update the remaining yamls with the same single-format conversion**

For each of `may15.yaml`, `may16.yaml`, `may22.yaml`, `easy_test.yaml`, replace the `battle_format: <fmt>` line with:

```yaml
  battle_formats:
    <fmt>: 1.0
```

(use the same `<fmt>` value the file originally had).

- [ ] **Step 6.4: Verify each config round-trips through `RNaDConfig.from_yaml`**

Run:
```bash
python -c "
from elitefurretai.rl.config import RNaDConfig
for name in ['single_team', 'may15', 'may16', 'may22', 'easy_test']:
    cfg = RNaDConfig.from_yaml(f'src/elitefurretai/rl/configs/{name}.yaml')
    print(name, cfg.curriculum.battle_formats, cfg.curriculum.primary_format)
"
```
Expected: one line per config, each showing the format dict and primary format.

- [ ] **Step 6.5: Run the full unit-test suite + quality gates one more time**

Run:
```bash
ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: all green.

- [ ] **Step 6.6: Smoke training run — single-format config (regression check)**

Run:
```bash
source ../venv/bin/activate
python src/elitefurretai/rl/train.py --config src/elitefurretai/rl/configs/easy_test.yaml --max-updates 10
```
Expected: training launches, completes 10 updates, no exceptions. Confirms the single-format path still works after the refactor.

- [ ] **Step 6.7: Smoke training run — multi-format config**

Create `src/elitefurretai/rl/configs/multi_format_smoke.yaml` as a copy of `easy_test.yaml` with the format block replaced by:

```yaml
  battle_formats:
    gen9vgc2024regg: 0.5
    gen9vgc2024regh: 0.5
```

Pre-check that team folders exist:

```bash
ls data/teams/gen9vgc2024regg/ data/teams/gen9vgc2024regh/ | head
```
Expected: both folders have at least one `.txt` team.

Then run:

```bash
python src/elitefurretai/rl/train.py --config src/elitefurretai/rl/configs/multi_format_smoke.yaml --max-updates 10 2>&1 | tee /tmp/multi_format_smoke.log
```
Expected: completes 10 updates. Look in the log for `Loaded N agent teams for gen9vgc2024regg` / `Loaded N agent teams for gen9vgc2024regh` (if `agent_team_path` was set) or for evidence that both formats appear in worker startup logs (each pair's Player will log its `battle_format`).

Then verify both formats appeared in actual battles:

```bash
grep -c "battle-gen9vgc2024regg-" /tmp/multi_format_smoke.log
grep -c "battle-gen9vgc2024regh-" /tmp/multi_format_smoke.log
```
Expected: both counts > 0. If `regh` count is 0, apportionment may have rounded a small dataset down to all-`regg`; reduce `num_pairs` proof window or rerun with explicit `--num-pairs 4`.

- [ ] **Step 6.8: Commit the yaml updates + smoke-test config**

Run:
```bash
git add src/elitefurretai/rl/configs/
git commit -m "rl/configs: migrate to battle_formats schema + add multi-format smoke config

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Per-format graduation evaluation

**Files:**
- Modify: `src/elitefurretai/rl/analyze/eval_analysis.py` (add per-format aggregation + graduation check)
- Modify: `unit_tests/rl/analyze/test_eval_analysis.py` (or create if absent — there's no current file by that name; create `unit_tests/rl/analyze/test_eval_analysis.py`)
- Create: `src/elitefurretai/scripts/multi_format_graduation_eval.py` (driver — under `scripts/`, which is intentionally excluded from quality gates per CLAUDE.md)

**Design:**

The eval pipeline already records `battle_format` per battle (see `eval_schema.py:60, 111`). Per-format aggregation is therefore a thin slice over the existing `battles` DataFrame — no changes needed to `evaluate.py`, `eval_collector.py`, or `eval_schema.py`. The new pieces:

1. `q_format_opp_type_win_rate(battles)` — groups by `(battle_format, opp_player_name)` and returns a DataFrame with `n_battles, wins, win_rate, ci_low, ci_high` per cell.
2. `graduation_summary(battles, threshold=0.60, required_opp_types=(...))` — applies a per-cell pass/fail threshold and returns a summary with overall `passed: bool`.
3. `multi_format_graduation_eval.py` — driver script that reads `battle_formats` from an RL config (or a CLI list), invokes `evaluate.py` once per format writing parquet shards into a shared run dir, concatenates the shards, runs `graduation_summary`, and prints the matrix.

VGCBench v1 is acceptable as a baseline across all formats per user confirmation; off-format VGCBench results will be slightly noisier but still informative. No dependency on VGCBench v2.

- [ ] **Step 7.1: Write the failing test for `q_format_opp_type_win_rate`**

Create `unit_tests/rl/analyze/test_eval_analysis.py` (the file does not currently exist — verify with `ls unit_tests/rl/analyze/test_eval_analysis.py`) with:

```python
"""Tests for elitefurretai.rl.analyze.eval_analysis aggregation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from elitefurretai.rl.analyze.eval_analysis import (
    graduation_summary,
    q_format_opp_type_win_rate,
)


def _battles_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal battles DataFrame with just the columns the
    aggregation reads. outcome=1 -> win, 0 -> loss, NaN -> tie.
    """
    return pd.DataFrame(rows)


def test_q_format_opp_type_win_rate_groups_by_format_and_opp():
    battles = _battles_df([
        {"battle_format": "gen9vgc2024regg", "opp_player_name": "max_damage", "outcome": 1.0},
        {"battle_format": "gen9vgc2024regg", "opp_player_name": "max_damage", "outcome": 1.0},
        {"battle_format": "gen9vgc2024regg", "opp_player_name": "max_damage", "outcome": 0.0},
        {"battle_format": "gen9vgc2024regg", "opp_player_name": "vgc_bench", "outcome": 1.0},
        {"battle_format": "gen9vgc2024regh", "opp_player_name": "max_damage", "outcome": 0.0},
        {"battle_format": "gen9vgc2024regh", "opp_player_name": "max_damage", "outcome": 1.0},
    ])
    result = q_format_opp_type_win_rate(battles)
    # Pivot to dict-of-dicts for stable comparison.
    keyed = {
        (r["battle_format"], r["opp_player_name"]): r["win_rate"]
        for _, r in result.iterrows()
    }
    assert keyed[("gen9vgc2024regg", "max_damage")] == pytest.approx(2 / 3)
    assert keyed[("gen9vgc2024regg", "vgc_bench")] == pytest.approx(1.0)
    assert keyed[("gen9vgc2024regh", "max_damage")] == pytest.approx(0.5)


def test_graduation_summary_passes_when_all_cells_meet_threshold():
    battles = _battles_df([
        {"battle_format": fmt, "opp_player_name": opp, "outcome": outcome}
        for fmt in ("gen9vgc2024regg", "gen9vgc2024regh")
        for opp in ("max_damage", "vgc_bench", "bc_player", "simple_heuristic")
        for outcome in [1.0] * 7 + [0.0] * 3  # 70% win rate per cell
    ])
    summary = graduation_summary(
        battles,
        threshold=0.60,
        required_opp_types=("max_damage", "vgc_bench", "bc_player", "simple_heuristic"),
    )
    assert summary["passed"] is True
    assert len(summary["cells"]) == 2 * 4  # 2 formats × 4 opp types
    for cell in summary["cells"]:
        assert cell["passed"] is True
        assert cell["win_rate"] == pytest.approx(0.70)


def test_graduation_summary_fails_when_any_cell_below_threshold():
    battles = _battles_df(
        # regg: all four opps at 70% — passes
        [
            {"battle_format": "gen9vgc2024regg", "opp_player_name": opp, "outcome": o}
            for opp in ("max_damage", "vgc_bench", "bc_player", "simple_heuristic")
            for o in [1.0] * 7 + [0.0] * 3
        ]
        # regh: vgc_bench at 50% — fails
        + [
            {"battle_format": "gen9vgc2024regh", "opp_player_name": opp, "outcome": o}
            for opp in ("max_damage", "bc_player", "simple_heuristic")
            for o in [1.0] * 7 + [0.0] * 3
        ]
        + [
            {"battle_format": "gen9vgc2024regh", "opp_player_name": "vgc_bench", "outcome": o}
            for o in [1.0] * 5 + [0.0] * 5
        ]
    )
    summary = graduation_summary(
        battles,
        threshold=0.60,
        required_opp_types=("max_damage", "vgc_bench", "bc_player", "simple_heuristic"),
    )
    assert summary["passed"] is False
    failing = [c for c in summary["cells"] if not c["passed"]]
    assert len(failing) == 1
    assert failing[0]["battle_format"] == "gen9vgc2024regh"
    assert failing[0]["opp_player_name"] == "vgc_bench"


def test_graduation_summary_flags_missing_required_opp_type():
    battles = _battles_df([
        {"battle_format": "gen9vgc2024regg", "opp_player_name": "max_damage", "outcome": 1.0},
    ])
    summary = graduation_summary(
        battles,
        threshold=0.60,
        required_opp_types=("max_damage", "vgc_bench"),
    )
    assert summary["passed"] is False
    assert any(
        c["opp_player_name"] == "vgc_bench" and c.get("missing")
        for c in summary["cells"]
    )
```

- [ ] **Step 7.2: Run the new tests to verify they fail**

Run:
```bash
pytest unit_tests/rl/analyze/test_eval_analysis.py -v
```
Expected: 4 FAILs (`ImportError: cannot import name 'q_format_opp_type_win_rate' / 'graduation_summary'`).

- [ ] **Step 7.3: Add `q_format_opp_type_win_rate` to `eval_analysis.py`**

In `src/elitefurretai/rl/analyze/eval_analysis.py`, locate the existing `_group_win_rate(battles, group_cols)` helper (line 867) and `q3_opp_type_win_rate` (line 68). Append a new function next to `q3_opp_type_win_rate`:

```python
def q_format_opp_type_win_rate(battles: pd.DataFrame) -> pd.DataFrame:
    """Win rate grouped by (battle_format, opp_player_name).

    Used for the Stage II per-format graduation check. Returns
    ``n_battles, wins, losses, ties, win_rate, ci_low, ci_high`` per cell.
    Requires a ``battle_format`` column on the input DataFrame, which the
    eval pipeline already populates (see eval_schema.py).
    """
    return _group_win_rate(battles, ["battle_format", "opp_player_name"])
```

- [ ] **Step 7.4: Add `graduation_summary` to `eval_analysis.py`**

Append (next to `q_format_opp_type_win_rate`):

```python
def graduation_summary(
    battles: pd.DataFrame,
    threshold: float = 0.60,
    required_opp_types: tuple[str, ...] = (
        "max_damage",
        "vgc_bench",
        "bc_player",
        "simple_heuristic",
    ),
) -> dict:
    """Stage II graduation check across (format × opp_type) cells.

    For each (battle_format, opp_type) present in ``battles`` whose
    ``opp_player_name`` is in ``required_opp_types``, emit a cell:
    ``{battle_format, opp_player_name, n_battles, win_rate, passed}``.
    For any (format, opp_type) pair where opp_type is required but no
    battles exist, emit a cell with ``missing=True, passed=False``.

    Overall ``passed`` is True iff every cell passes (no missing
    required opp_types, every win_rate >= threshold).
    """
    per_cell = q_format_opp_type_win_rate(battles)
    formats = sorted({str(f) for f in battles["battle_format"].unique()})
    cells: list[dict] = []
    for fmt in formats:
        for opp in required_opp_types:
            row = per_cell[
                (per_cell["battle_format"] == fmt)
                & (per_cell["opp_player_name"] == opp)
            ]
            if row.empty:
                cells.append({
                    "battle_format": fmt,
                    "opp_player_name": opp,
                    "n_battles": 0,
                    "win_rate": float("nan"),
                    "passed": False,
                    "missing": True,
                })
            else:
                wr = float(row["win_rate"].iloc[0])
                cells.append({
                    "battle_format": fmt,
                    "opp_player_name": opp,
                    "n_battles": int(row["n_battles"].iloc[0]),
                    "win_rate": wr,
                    "passed": wr >= threshold,
                    "missing": False,
                })
    return {
        "threshold": threshold,
        "required_opp_types": list(required_opp_types),
        "formats": formats,
        "cells": cells,
        "passed": all(c["passed"] for c in cells),
    }
```

- [ ] **Step 7.5: Run the tests and confirm they pass**

Run:
```bash
pytest unit_tests/rl/analyze/test_eval_analysis.py -v
```
Expected: 4 PASS.

- [ ] **Step 7.6: Write the multi-format driver script**

Create `src/elitefurretai/scripts/multi_format_graduation_eval.py`:

```python
"""Run the Stage II graduation matrix (4 baselines × N formats) for a checkpoint.

Reads `battle_formats` from an RL config (or accepts `--formats` directly),
shells out to `evaluate.py` once per format with each of the four required
baselines, concatenates the parquet shards, and prints a graduation matrix.

VGCBench v1 is used uniformly across formats — cross-format VGCBench numbers
are slightly noisier but acceptable per the user's spec for this plan.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from elitefurretai.rl.analyze.eval_analysis import graduation_summary
from elitefurretai.rl.config import RNaDConfig

BASELINES = [
    "max_damage",
    "vgc_bench",
    "bc_player",
    "simple_heuristic",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to RL .pt checkpoint")
    parser.add_argument(
        "--config",
        default=None,
        help="RL yaml config. If given, --formats defaults to its battle_formats keys.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Override format list (space-separated).",
    )
    parser.add_argument("--battles-per-cell", type=int, default=200)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    if args.formats:
        formats = args.formats
    elif args.config:
        cfg = RNaDConfig.from_yaml(args.config)
        formats = list(cfg.curriculum.battle_formats.keys())
    else:
        sys.exit("Must provide --formats or --config")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        for baseline in BASELINES:
            print(f"[run] {fmt} vs {baseline}")
            subprocess.run(
                [
                    sys.executable,
                    "src/elitefurretai/rl/analyze/evaluate.py",
                    "--player1", f"model:{args.checkpoint}",
                    "--player2", baseline,
                    "--team1", f"data/teams/{fmt}/constrained",
                    "--team2", f"data/teams/{fmt}/constrained",
                    "--battle-format", fmt,
                    "--battles", str(args.battles_per_cell),
                    "--collect-trajectories", str(run_dir),
                ],
                check=True,
            )

    # Concatenate all parquet shards under run_dir.
    shards = list(run_dir.rglob("*.parquet"))
    if not shards:
        sys.exit(f"No parquet shards found under {run_dir}")
    battles = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)

    summary = graduation_summary(
        battles,
        threshold=args.threshold,
        required_opp_types=tuple(BASELINES),
    )
    print("\n=== Graduation Matrix ===")
    print(f"Threshold: {summary['threshold']:.0%}")
    for cell in summary["cells"]:
        mark = "✓" if cell["passed"] else "✗"
        wr = cell["win_rate"]
        wr_str = "MISSING" if cell.get("missing") else f"{wr:.1%}"
        print(
            f"  {mark} {cell['battle_format']:>22} vs {cell['opp_player_name']:>18}: "
            f"{wr_str:>8}  (n={cell['n_battles']})"
        )
    print(f"\nOverall: {'PASS' if summary['passed'] else 'FAIL'}")
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.7: Smoke-check that the driver script is syntactically valid**

Run:
```bash
python -c "import ast; ast.parse(open('src/elitefurretai/scripts/multi_format_graduation_eval.py').read())"
```
Expected: no output, exit 0.

A full end-to-end smoke run (actually launching Showdown and battling) is not required as a plan step — it costs hours and depends on a working checkpoint. The aggregation function is unit-tested in Steps 7.1–7.5; the driver is a thin shell around `evaluate.py`, which is already exercised by the existing eval tests.

- [ ] **Step 7.8: Update `MODEL_EVALUATION.md` to document the per-format matrix**

In `src/elitefurretai/rl/analyze/MODEL_EVALUATION.md`, search for "graduation" (line ~233). Add a new subsection describing the per-format matrix output:

```markdown
### Multi-format graduation matrix

For a multi-format RL run, the Stage II 60%×4 criterion expands to
`(format × opp_type)` cells. Run:

    python src/elitefurretai/scripts/multi_format_graduation_eval.py \
        --checkpoint data/models/<run>/checkpoint_<step>.pt \
        --config src/elitefurretai/rl/configs/<your_config>.yaml \
        --run-dir data/eval/<run>_<step>_graduation \
        --battles-per-cell 200

This shells out to `evaluate.py` once per (format, baseline) cell and
prints a matrix with one row per cell. Overall pass requires every cell
≥ threshold (default 60%). VGCBench v1 is reused across formats — its
cross-format generalization is acceptable per the multi-format
implementation plan.
```

- [ ] **Step 7.9: Quality gates**

Run:
```bash
ruff check src/elitefurretai/rl/analyze unit_tests/rl/analyze && ruff format src/elitefurretai/rl/analyze unit_tests/rl/analyze --check && pyright src/elitefurretai/rl/analyze unit_tests/rl/analyze && pytest unit_tests/rl/analyze -q
```
Expected: all green. (The `scripts/` driver is intentionally outside the quality gates per CLAUDE.md, so it isn't included here.)

- [ ] **Step 7.10: Commit**

Run:
```bash
git add src/elitefurretai/rl/analyze/eval_analysis.py src/elitefurretai/rl/analyze/MODEL_EVALUATION.md unit_tests/rl/analyze/test_eval_analysis.py src/elitefurretai/scripts/multi_format_graduation_eval.py
git commit -m "rl/analyze: per-format graduation matrix + multi-format eval driver

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Post-flight

- [ ] **Step P.1: Full clean run of every quality gate**

Run:
```bash
source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src unit_tests && pytest unit_tests -q
```
Expected: all green.

- [ ] **Step P.2: Update `RL.md` to document the new format schema**

In `src/elitefurretai/rl/RL.md`, locate the section that describes `CurriculumConfig` (search for `battle_format` — expected hit around line 521 per `grep`). Replace any single-format documentation with the dict-form schema and add the per-pair apportionment paragraph (one sentence is enough: "When `battle_formats` contains multiple entries, `WorkerOpponentFactory` apportions formats across pairs via largest-remainder; each pair is pinned to a single format for the entire run.").

- [ ] **Step P.3: Write the planning update**

Create `planning/stage2/YYYY-MM-DD-hh-mm-multi-format-doubles-shipped.md` (use the current date/time) with the sections required by `CLAUDE.md`: Context, Before State, Problem, Solution, Reasoning, Planned Next Steps, Updates. Reference this implementation plan in the Context section.

- [ ] **Step P.4: Commit docs**

Run:
```bash
git add src/elitefurretai/rl/RL.md planning/stage2/
git commit -m "docs: multi-format training schema + post-implementation planning note

$(cat <<'EOF'
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Risk register

- **Apportionment with very small num_pairs:** at `num_pairs=4` and weights `{regg: 0.6, regh: 0.4}`, you get `[regg, regg, regh, regh]` — distribution is exact. At `num_pairs=2` you'd get `[regg, regh]`. At `num_pairs=1` you'd get only the higher-weight format. Document this in the smoke test if observed; recommend `num_pairs >= len(battle_formats) * 2` for stable distributions.
- **Showdown server format support:** confirm the Showdown server has every target format registered (e.g. `gen9vgc2024regh` requires the appropriate ruleset). If a format string isn't recognized, the worker will hang on `|nametaken|` or `|formaterror|`. Resolve by ensuring `pokemon-showdown` is up-to-date in the engine setup.
- **TeamRepo cold-start cost:** TeamRepo loads every format folder under `base_team_path` at startup. Adding more formats grows worker memory linearly — log the team count per format at startup so regressions are visible.
- **`agent_team_path` dict form silently broadcasts only configured formats:** the validator (Step 2.3) catches missing keys, so this is enforced; do not loosen it later.
- **VGCBench v1 used across formats:** per user, VGCBench's cross-format generalization is acceptable. Off-format VGCBench numbers will be slightly noisier than same-format, but no config gating is required. When VGCBench v2 (multi-format trained) ships, swap the checkpoint in `external_vgcbench_team_file` / `vgc_bench_checkpoint_path` and rerun the graduation matrix.

---

## Self-review

**Spec coverage:**
- Format distribution schema → Task 1
- Per-format team-path resolution → Task 2
- Per-pair format assignment + Showdown format routing → Tasks 3, 4
- Embedder using gen-shared vocab → Task 5 (`primary_format` everywhere)
- YAML migration → Task 6
- Smoke verification (both single- and multi-format) → Task 6
- Per-format eval → Task 7 (per-cell graduation matrix + driver script)
- VGCBench v1 cross-format use + BC adaptation → user-confirmed acceptable (no tasks needed)

**Placeholder scan:** All "..." in this plan appear inside *examples of unchanged code* (e.g. `# ... (rest of the existing fields unchanged: ...)`). No TODO/TBD/"fill in details" placeholders are used to defer real work.

**Type consistency:**
- `battle_formats: Dict[str, float]` — used identically in `CurriculumConfig`, `WorkerOpponentFactory.__init__`, and the apportionment helper.
- `opponent_team_subdirectories: Dict[str, Optional[str]]` (factory) ↔ `resolved_opponent_team_pool_paths() -> Dict[str, Optional[str]]` (config) — match.
- `agent_team_paths: Optional[Dict[str, str]]` (factory) ↔ `resolved_agent_team_paths() -> Dict[str, str]` (config) — match, with `or None` collapsing the empty case at the call site (Step 5.1).
- `pair_formats: List[str]` — populated in `create_agents` (Step 4.4), consumed by `_make_baseline_pool` (Step 4.3) — match.
- `primary_format: str` — property on `CurriculumConfig`, used at 5 embedder sites — match.
