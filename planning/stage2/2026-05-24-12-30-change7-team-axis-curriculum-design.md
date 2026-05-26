# Change 7 — Agent-Team-Axis Adaptive Sampling — Design

**Date**: 2026-05-24 12:30
**Predecessor**: [adaptive curriculum overhaul (2026-05-22 reprioritization)](./2026-05-20-15-00-adaptive-curriculum-overhaul.md), Group A+
**Status**: Pre-implementation. Design agreed; ready for plan writing.

This doc fills in the Group A+ stub from the curriculum overhaul: an
adaptive curriculum over the *agent team* (the team the model pilots),
co-priority with Group A (Changes 2 + 4 + 5) and independent of it. The
opponent-axis curriculum and the team-axis curriculum operate on
orthogonal axes and ship as separate PRs.

---

## Context

[MODEL_EVALUATION.md](../../src/elitefurretai/rl/analyze/MODEL_EVALUATION.md)
(balmy70 step-10700, n = 476,650 battles) surfaced that the dominant
source of WR variance is the agent team, not the opponent team:

- Per-agent-team WR (Q1): 35-point spread vs heuristics, 28-point
  spread vs vgc_bench. Only one cell (`38dessert` ×
  `simple_heuristic`) has a CI upper bound > 50%.
- Per-opponent-team WR std vs vgc_bench (Q2): 0.006 — the model has
  effectively zero per-opp-team adaptation.
- Cross-pattern summary: the model can pilot direct-offensive cores
  (Calyrex + Urshifu-Rapid-Strike + weather) and cannot pilot
  setup-reliant teams (Calm Mind, Trick Room, Friend Guard). Bottom-5
  piloting teams are all setup-reliant.

The current curriculum samples opponents adaptively but samples
agent teams uniformly via `WorkerOpponentFactory.sample_team(battle_format)` at
[opponents.py:645](../../src/elitefurretai/rl/opponents.py#L645).
Training pressure on the under-performed teams is therefore diluted by
the ~35 teams the policy already handles consistently.

This change adds an agent-team axis to the curriculum so the policy
spends more wall-clock practicing the teams it cannot pilot.

## Before state

- `WorkerOpponentFactory.sample_team(battle_format)` returns a team
  string via `team_repo.sample_team(battle_format, subdirectory=...)`
  — uniform `random.choice` under the configured subdirectory.
- Both p1 and p2 sample teams independently from this single
  uniform distribution. Call sites:
  [opponents.py:686, 769, 1015, 1019, 1023, 1027, 1031](../../src/elitefurretai/rl/opponents.py).
- The trajectory dict carries `opponent_type`, `won`, `battle_length`,
  `forfeited`. It does not carry a team identity.
- `OpponentPool.record_battle_result(...)` at
  [opponents.py:318](../../src/elitefurretai/rl/opponents.py#L318)
  records per-opponent-type WR; no per-team tracking exists.
- A debug/ablation override `_agent_teams_by_format: Dict[str, List[str]]`
  exists on `WorkerOpponentFactory` at
  [opponents.py:555](../../src/elitefurretai/rl/opponents.py#L555).
  `get_agent_team(battle_format)` at
  [opponents.py:651](../../src/elitefurretai/rl/opponents.py#L651)
  short-circuits to a fixed team from this dict when populated,
  bypassing `sample_team()`.
- External callers of `team_repo.sample_team(...)` live outside
  `opponents.py` in `rl/analyze/team_provider.py`,
  `engine/analyze/showdown_benchmark.py`, and
  `engine/analyze/showdown_invalid_choice_diagnostics.py`. The
  existing return type must remain `str` to avoid breaking them.

## Decisions (from 2026-05-24 brainstorm)

1. **Granularity**: per-(battle_format, agent_team), marginalized
   over `opp_type`. Each configured training format gets its own
   WR tracking and its own sampling distribution. Within a format,
   teams are tracked individually (42 cells under the constrained
   pool for the current Stage II format). Joint
   `(team, opp_type)` tracking deferred until integration
   validation shows it's needed.
2. **Score function**: asymmetric PFSP `weight_t ∝ (1 - wr_t) ** p`,
   reusing `p` from Change 4. No separate exponent.
3. **Bias scope**: applies to all call sites during training. Both
   sides of self-play, ghost matchups, exploiter matchups, *and*
   baseline matchups draw teams from the biased distribution. An opt-out
   `biased=False` is exposed for future eval-at-checkpoint code paths.
4. **Sampler location**: worker-side, mirroring the opponent-axis
   curriculum. Main process tracks WR, recomputes the team
   distribution at the same cadence as `update_curriculum()`, and
   broadcasts it to workers.
5. **Initialization**: uniform prior, global warm-up gate.
   Workers stay on uniform `team_repo.sample_team()` until every team
   has accumulated ≥ `team_warmup_threshold` battles; once tripped,
   stays tripped.
6. **Forfeit handling**: forfeited battles skip the EWMA update and
   do not count toward warm-up sample counts. Same principle as
   Change 1, included here unconditionally regardless of whether
   Change 1 ships first.

### Acknowledged tradeoffs

- Biasing all opp sides (including baselines) means the per-team WR
  estimate mixes "model vs model" battles with "model vs
  baseline-piloting-a-team-it-can't-execute" battles. The latter
  inflates WR for setup teams when the baseline is on the other side.
  Relative ordering of teams should still be preserved (setup teams
  remain hardest in aggregate), but absolute per-team WR values will
  skew higher than they would in uniform-opp regimes. Future
  threshold-based logic on per-team WR will need calibration against
  this biased-mix distribution. We accept this tradeoff in exchange
  for maximal exposure to hard team-axis cells during training.

## Architecture

### Component changes

#### `OpponentPool` (main process)

New state (multi-format from day one, matching existing
`_agent_teams_by_format` convention):

```python
self.team_win_rates: Dict[str, Dict[str, Tuple[float, float]]] = {}
    # battle_format → team_name (filename without .txt) → (ewma_wins, ewma_n)
self.team_sample_counts: Dict[str, Dict[str, int]] = {}
    # battle_format → team_name → number of non-forfeit battles recorded
self.known_teams: Dict[str, List[str]] = {}
    # battle_format → list of team_names snapshotted at __init__ from
    # team_repo entries under the configured (format, subdirectory).
    # Mid-run file additions are ignored until restart, same
    # convention as ghost/exploiter slots.
self._team_axis_warm: Dict[str, bool] = {}
    # battle_format → cached "this format has tripped the warm-up
    # gate" flag. Once True for a format, stays True.
```

**Format discovery**: `known_teams` is populated at `OpponentPool`
`__init__` by iterating `self.battle_formats.keys()`
([opponents.py:540](../../src/elitefurretai/rl/opponents.py#L540)).
For each format, list `team_repo` entries under
`self.opponent_team_subdirectories.get(battle_format)`
([opponents.py:541](../../src/elitefurretai/rl/opponents.py#L541)).
Single-format training is the degenerate case (one format key);
multi-format is the general case.

**Warm-up is per-format**: each format independently waits until
all its known teams have ≥ `team_warmup_threshold` battles before
its distribution flips on. A format that warms up first does not
wait for slower-converging formats. This avoids one rare format
holding back curriculum activation across the others.

New method:

```python
def update_team_distribution(
    self,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Recompute per-format team sampling distributions.

    Returns a dict keyed by battle_format. Value is None for any
    format still in warm-up (some team with sample_count <
    team_warmup_threshold); a normalized {team_name: weight} dict
    for any format past warm-up. Empty top-level dict if the
    feature is disabled.
    """
```

`record_battle_result(...)` gains two new optional parameters:
`battle_format: Optional[str] = None` and
`team_name: Optional[str] = None`. When both are non-None and
`forfeited=False`, the per-(format, team) EWMA is updated with the
same decay rule as Change 5.

#### `WorkerOpponentFactory` (worker process)

New state:

```python
self.team_distribution_by_format: Dict[str, Optional[Dict[str, float]]] = {}
    # Set by broadcast. For each format, None or absent = warm-up
    # not yet tripped; falls back to uniform team_repo sampling.
    # A dict value = past warm-up; biased sampling active.
```

Signature change:

```python
def sample_team(
    self,
    battle_format: str,
    biased: bool = True,
) -> Tuple[str, str]:
    """Return (team_string, team_name).

    team_name is the filename without .txt — used by the worker to
    stamp the trajectory dict.

    biased=True (default): look up
    self.team_distribution_by_format.get(battle_format); if a
    non-None dict, use numpy.random.choice over its keys with the
    weights. Otherwise fall back to the uniform path.

    biased=False: always uniform. Reserved for future
    eval-at-checkpoint paths.
    """
```

The two call-site flavors discussed in brainstorm collapse to a
single method with a `biased` flag: all training call sites use the
default `biased=True`; future eval code passes `biased=False`.

All existing call sites at lines 686, 769, 1015, 1019, 1023, 1027,
1031 migrate from `team_string = self.sample_team(fmt)` to
`team_string, team_name = self.sample_team(fmt)`. Sites that pilot a
model (the trajectory player and any model-flavored opponent) cache
`team_name` somewhere reachable from the trajectory finish hook;
baseline-only sites can drop the name.

`update_curriculum(curriculum, team_distribution_by_format)`
accepts the broadcast per-format distributions and replaces
`self.team_distribution_by_format` wholesale.

`get_agent_team(battle_format)` at
[opponents.py:651](../../src/elitefurretai/rl/opponents.py#L651)
continues to short-circuit to `_agent_teams_by_format` when set,
fully bypassing biased sampling. No reconciliation logic — the
override remains a debug affordance.

### Protocol change

Trajectory dict adds two fields (current shape at
[rl_trajectory_player.py:633-641](../../src/elitefurretai/rl/rl_trajectory_player.py#L633-L641)):

```python
traj["team_name"]: str          # filename (without .txt) of the
                                # trajectory player's team
traj["battle_format"]: str      # e.g. "gen9vgc2024regg" — required
                                # for routing the EWMA update to the
                                # right per-format bucket
```

Sources:
- `team_name`: `WorkerOpponentFactory.sample_team(battle_format)`
  returns `(team_string, team_name)`. The factory sets a new
  attribute `current_team_name` on `RLTrajectoryPlayer` at the same
  site where it sets `_team = ConstantTeambuilder(team_string)`
  (opponents.py lines 1015, 1019, 1023, 1027, 1031 — and the
  corresponding sites for the trajectory players themselves at
  686 and 769). The trajectory player reads
  `self.current_team_name` at battle finish and stamps it.
- `battle_format`: read directly from `battle.format` (poke-env
  `Battle` object) at trajectory-finish time in
  `rl_trajectory_player.py`. No new player attribute needed.

The trajectory dict construction site is
[rl_trajectory_player.py:633-641](../../src/elitefurretai/rl/rl_trajectory_player.py#L633-L641).
That's the file the protocol change actually modifies.

`train.py` at [line 936](../../src/elitefurretai/rl/train.py#L936)
passes both `battle_format=traj["battle_format"]` and
`team_name=traj["team_name"]` to `record_battle_result`.

### Broadcast

The existing curriculum broadcast at
[train.py:1110](../../src/elitefurretai/rl/train.py#L1110)
gains one extra field:
`team_distribution_by_format: Dict[str, Optional[Dict[str, float]]]`.
Keyed by battle_format; value is `None` for any format still in
warm-up and a normalized `{team_name: weight}` for any format past
warm-up. The worker hook applies this inside the same update path
as the opponent curriculum.

## Algorithm

### Per-trajectory WR update (`record_battle_result`)

```
if team_name is None or battle_format is None:
    return                              # not enough info to route
if forfeited:                           # forfeits are not legitimate signal
    return
decay = 0.5 ** (1 / half_life)          # half_life from Change 5, in battles

fmt_wins  = ewma_wins[battle_format]    # per-format inner dict
fmt_n     = ewma_n[battle_format]
fmt_count = team_sample_counts[battle_format]

fmt_wins[team_name]  = fmt_wins[team_name]  * decay + float(won)
fmt_n[team_name]     = fmt_n[team_name]     * decay + 1.0
fmt_count[team_name] = fmt_count[team_name] + 1
```

### Distribution recompute (`update_team_distribution`)

```
result: Dict[str, Optional[Dict[str, float]]] = {}
for fmt, teams in known_teams.items():
    if self._team_axis_warm.get(fmt, False):
        warm = True
    else:
        warm = all(
            team_sample_counts[fmt][t] >= team_warmup_threshold
            for t in teams
        )
        if warm:
            self._team_axis_warm[fmt] = True

    if not warm:
        result[fmt] = None
        continue

    scores = {}
    for t in teams:
        wr_t = (
            (ewma_wins[fmt][t] + alpha)
            / (ewma_n[fmt][t] + alpha + beta)
        )
        scores[t] = (1.0 - wr_t) ** pfsp_exponent     # p from Change 4

    total = sum(scores.values())
    distribution = {t: scores[t] / total for t in teams}

    # Apply per-team floor, then renormalize.
    floor = team_per_team_floor
    for t in teams:
        if distribution[t] < floor:
            distribution[t] = floor
    total = sum(distribution.values())
    distribution = {t: distribution[t] / total for t in teams}

    result[fmt] = distribution

return result
```

The Beta(α, β) smoothing reuses the (α=8, β=8) prior from Change 5.
`half_life` units are battles, matching Change 5.

**Per-format warm-up latching**: once `_team_axis_warm[fmt]`
becomes True for a format, it stays True for the rest of training.
Each format latches independently.

### Worker-side sampling (`sample_team(battle_format, biased)`)

```
dist = (
    self.team_distribution_by_format.get(battle_format)
    if biased else None
)
if dist is not None:
    name = numpy.random.choice(
        list(dist.keys()),
        p=list(dist.values()),
    )
else:
    name = self.team_repo.sample_team_name(
        battle_format,
        subdirectory=self.team_subdirectory,
    )
team_string = self.team_repo.get(battle_format, name)
return self.team_repo._shuffle_team_order(team_string), name
```

A new helper `team_repo.sample_team_name(format, subdirectory=None)`
returns just the filename (without `.txt`) of a uniformly-sampled
team. The existing `team_repo.sample_team(...)` is left untouched
(it has external callers in `team_provider.py`, `showdown_benchmark.py`,
and `showdown_invalid_choice_diagnostics.py` that should not be
disturbed by this change). Internally, `team_repo.sample_team` may
delegate to `sample_team_name` + `get` to avoid duplicating the
subdirectory-filter logic — but that refactor is optional and not
load-bearing for Change 7.

## Config

New fields on `CurriculumConfig` in
[src/elitefurretai/rl/config.py](../../src/elitefurretai/rl/config.py):

```python
team_axis_enabled: bool = True
    # Master switch. False = bypass entirely; workers ignore broadcast
    # team_distribution_by_format and stay on uniform sampling.
team_warmup_threshold: int = 20
    # Min battles per (format, team) before that format's bias
    # activates. Per-format gate; each format latches independently.
team_per_team_floor: float = 0.005
    # Min normalized weight any team can receive within a format's
    # distribution after renormalization.
```

Reused from elsewhere:

- `pfsp_exponent` (Change 4) — shape of the asymmetric PFSP curve.
- `half_life` (Change 5) — EWMA decay rate.
- Beta(α=8, β=8) smoothing — Change 5.

## Files touched

| File | Change |
|---|---|
| `src/elitefurretai/rl/opponents.py` | `OpponentPool`: add per-format `team_win_rates`, `team_sample_counts`, `known_teams`, `_team_axis_warm`, `update_team_distribution()`; extend `record_battle_result` with `battle_format` and `team_name` params. `WorkerOpponentFactory`: add `team_distribution_by_format` field; extend `update_curriculum` signature; modify `sample_team` to return `(string, name)` and accept `biased` flag. At call sites (lines 686, 769, 1015, 1019, 1023, 1027, 1031), unpack the tuple and set `player.current_team_name = name` alongside `player._team = ConstantTeambuilder(team_string)`. |
| `src/elitefurretai/rl/rl_trajectory_player.py` | Add `current_team_name: Optional[str] = None` attribute. At trajectory-build site (lines 633-641), add `"team_name": self.current_team_name` and `"battle_format": battle.format` to the trajectory dict. |
| `src/elitefurretai/rl/train.py` | Pass `battle_format=traj["battle_format"]` and `team_name=traj["team_name"]` to `record_battle_result` (currently called at line 936). Invoke `OpponentPool.update_team_distribution()` adjacent to `update_curriculum()` (currently at line 1110). Bundle the per-format distributions into the broadcast payload. |
| `src/elitefurretai/rl/config.py` | Add three new fields to `CurriculumConfig`. |
| `src/elitefurretai/etl/team_repo.py` | Add `sample_team_name(format, subdirectory=None) -> str` helper. Do not modify the existing `sample_team(...)` signature (external callers in `team_provider.py`, `showdown_benchmark.py`, `showdown_invalid_choice_diagnostics.py` depend on it). |
| `unit_tests/rl/test_team_axis.py` (new) | Unit tests below. |
| `src/elitefurretai/rl/RL.md` | §7 update describing the new axis after PR lands. |

## Acceptance criteria

### Unit tests

1. **Per-(format, team) WR update routes correctly.** Record 10
   wins and 10 losses for `("gen9vgc2024regg", "38dessert")`;
   assert that format/team's `ewma_wins / ewma_n ≈ 0.5` within
   EWMA tolerance and other (format, team) cells are untouched
   (including the same team name under a different format).
2. **Per-format warm-up gate.** With `team_warmup_threshold=20`
   and two configured formats A and B: record 20 battles per team
   for format A only. Assert `update_team_distribution()` returns
   `{A: <dict>, B: None}`. Then warm up B. Assert subsequent call
   returns dicts for both. Verify the warm-up flag latches:
   draining samples back to <20 after warm-up does not flip the
   format back to None.
3. **Asymmetric PFSP direction.** Single-format synthetic
   5-team setup: 3 strong (90/10 W/L), 2 weak (20/80 W/L). After
   warm-up, assert the two weak teams' weights sum to > 60% of
   the distribution.
4. **Per-team floor enforced.** Same setup as #3 with `floor=0.05`;
   assert no team's weight is below 0.05 after normalization.
5. **Forfeit skip.** Record 5 normal wins, then 5 forfeits for a
   single (format, team). Assert `ewma_wins / ewma_n` reflects
   only the 5 normal wins; `team_sample_counts` reflects only 5.
6. **`team_axis_enabled=False` bypasses the feature.** Flag off.
   Record many battles across multiple formats. Assert
   `update_team_distribution()` returns an empty dict (or all
   `None` values) unconditionally; worker `sample_team(biased=True)`
   falls back to uniform for every format.
7. **`biased=False` is always uniform.** With a biased distribution
   set for some format on the worker,
   `sample_team(battle_format, biased=False)` calls the uniform
   path. Test for both a format that has a distribution set and a
   format that does not.
8. **Format isolation in broadcast.** Build two format-keyed
   distributions where format A heavily weights team X and format
   B heavily weights team Y. Assert `sample_team("A", biased=True)`
   draws X more often than Y, and `sample_team("B", biased=True)`
   draws Y more often than X — distributions don't cross-pollinate.

### Integration validation

Two runs from the same BC checkpoint, identical hyperparameters,
identical opponent-axis curriculum (Group A applied to both):

- Run A: `team_axis_enabled=True`
- Run B: `team_axis_enabled=False`

After K updates (chosen at launch time to give ≥ a few thousand
battles per team), run the MODEL_EVALUATION per-team WR eval on each
checkpoint.

**Pass criterion**: the worst-quartile-team WR (averaged across the
11 weakest teams against the 4 graduation baselines) climbs at
least 5 percentage points faster on Run A than Run B, holding
wall-clock fixed.

**Secondary signal**: variance across the 42 per-team WRs shrinks
on Run A (movement toward uniform competence rather than the
current bimodal distribution).

## Out of scope

- **Joint `(team, opp_type)` tracking.** Granularity stays at
  per-team. Revisit only if integration validation shows team-axis
  isn't enough to move the limiting baseline toward 60%.
- **Eval driver changes.** The `biased=False` path is plumbing-only;
  no eval call sites are migrated as part of Change 7.
- **BC-derived or eval-derived priors.** Uniform + warm-up gate is
  the chosen initialization.
- **Re-discovering team files mid-run.** Snapshot at init; new team
  files require a restart.
- **Threshold-based graduation logic** on per-team WR.
- **Interaction shim with `_agent_teams`.** The override bypasses
  biased sampling unconditionally — no reconciliation.
- **Length-weighted EWMA** (Change 1's other half). Out of scope here;
  if Change 1 ships, the per-team EWMA path picks up the same
  length-weight logic in that change.

## Open questions for later

- After Group A and Change 7 both ship and the first paired
  validation run completes: is the worst-quartile-team WR lift large
  enough to chase Stage II graduation on team-axis alone, or do we
  need joint `(team, opp_type)` granularity to keep climbing? This
  determines whether per-team is the final design or a stepping
  stone.
- Once Change 1 ships, do we keep the unconditional forfeit-skip in
  the team-axis EWMA, or migrate to length-weighted updates for
  consistency?
- Does the per-team floor need to be config-driven per team (some
  teams may need higher floors to prevent forgetting), or is the
  global floor sufficient? Likely sufficient; revisit only if eval
  shows forgetting on previously-strong teams.
- **Cross-format interaction**: per-format warm-up and distributions
  are independent in this design. Does cross-format transfer (e.g.
  the model improves on team X in format A while training on
  format B) need explicit accounting, or is letting each format
  warm up independently good enough? Likely good enough — formats
  are sampled by a separate mechanism and each format's WR signal
  is grounded in its own battles.

## Updates

- **2026-05-24 14:00** — Implementation Task 5
  (`update_team_distribution`) replaced the planned naive
  floor-then-renormalize step with a water-filling algorithm.
  Reasoning: the naive version cannot mathematically guarantee the
  floor invariant after renormalization (lifting below-floor weights
  up to floor and dividing by the new sum can push the lifted
  weights back below floor). Water-filling pins below-floor teams at
  exactly floor and redistributes the remaining mass over unpinned
  teams, iterating until a fixed point. The implementation caps
  iterations at `len(teams)`; in practice it terminates in ≤6
  iterations even with 100 teams. See `update_team_distribution` in
  `src/elitefurretai/rl/opponents.py`.
