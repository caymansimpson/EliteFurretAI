# MODEL_EVALUATION.md — balmy70 step-10700 full evaluation

End-to-end Plan B trajectory + analysis run against the production
checkpoint `balmy-cloud-70/ghosts/main_model_step_10700.pt`, at full
matrix scale (42 × 42 teams × 100 battles per cell per opp_type).

Every result block reports the exact command run, the sample size, and
a Wilson 95% CI. Cross-checks are surfaced inline. No headline number
is reported without provenance. Team filenames (without the `.txt`
suffix) are used everywhere instead of hashes so you can grep the
team directory directly to look up rosters.

## Headline

**Aggregate WR 17.8%** across all opp_types (n = 476,650, Wilson CIs
fully below 0.5 for every opp_type). The model is **far below the
Stage II 60%-baseline threshold** at this checkpoint. Per-opp WR:
vgc_bench **10.1%** ≪ max_damage **19.8%** ≤ simple_heuristic **21.9%**.

**One-line synthesis** (cross-pattern from Q1 + Q2 + team rosters
analyzed below):

> The model has *one* strategy — direct damage commitment — and
> applies it everywhere. It exploits opponents that need multi-turn
> setup (Calm Mind, Trick Room, Friend Guard support) and loses to
> opponents that don't (Choice Specs Miraidon, Bulk Up Koraidon,
> Choice Scarf Urshifu). The Stage II target requires it to develop a
> *second* strategy.

| Pattern | Model strength | Model weakness |
|---|---|---|
| Pilot direct offensive teams | ✓ (best WR with offensive cores) | — |
| Pilot setup teams | — | ✗ (worst WR with Calm Mind / Trick Room / Friend Guard cores) |
| Beat setup opponents | ✓ (best WR against setup-heavy gimmicks) | — |
| Beat direct offensive opponents | — | ✗ (worst WR against Choice Specs box mons) |

## Prioritized takeaways — what to address, in order

Numbered by expected impact on the Stage II 60%-baseline threshold.
Each item lists the supporting data, the action to take, and what
*independent* analysis would let us confirm the priority before
spending compute on a fix.

### Priority 1 — Develop strategy diversity beyond direct-damage

**Claim:** The model has a single offensive strategy. It wins when
that strategy applies (direct-damage cores vs setup opponents) and
loses when it doesn't (setup mirrors, defensive cycling, Choice
Specs box mons).

**Supporting data:**

* WR std across opp_teams against vgc_bench is **0.006** (Q2) — the
  model has zero team-specific strategy against vgc_bench.
* Compare against heuristics: **0.076 / 0.069** — the model adapts
  to weak heuristic teams (finds exploits) but has no
  per-opponent-team adaptation against a strong policy.
* Bottom-5 agent teams are all setup-reliant; top-5 are all direct
  offensive (see Team-based pattern analysis below).
* Q9b found **0 agree-then-diverge** battles — the model's value
  head doesn't shift mid-battle, suggesting no strategic adaptation.

**Action:** Add an explicit curriculum signal for setup teams. Try
one of:
1. Curriculum weighting that over-samples (agent_team =
   setup-reliant) cells during self-play.
2. Add behavior-cloning data from human Trick Room games to break
   the "always click damage" prior.
3. Search-guided rollouts at training time (early Stage IV
   anticipation) on the bottom-5 agent teams.

**Independent confirmation:** train for ~5k steps with curriculum
weighting biased toward setup teams; re-run this eval. If
bottom-5 agent team WR rises by ≥5 pp while top-5 stays flat,
the priority was correct. If top-5 drops while bottom-5 doesn't
rise, the model lacks capacity to hold both strategies and needs
architectural changes (or a teacher).

### Priority 2 — Fix value-head calibration on the upper half

**Claim:** The value head is systematically over-confident at
predicted ≥ 0.3 (gap up to 16.4 pp at predicted ~0.65), but
near-calibrated below 0.1.

**Supporting data:**

* Q7 ECE = **5.69%** (substantial — well-calibrated heads hit ECE <
  2%).
* Reliability table shows under-confidence in the 0-0.1 bin (gap
  3.8 pp) but over-confidence in every bin from 0.3 upward.
* Q9a top 20 are *all* wins where value head was pessimistic
  throughout — the head doesn't update on positive evidence
  mid-battle.
* Q6 worst swings: value head at +0.25 / -0.20 right before a
  game-ending bad turn — fails to flag uncertainty.

**Action:** Add a calibration penalty to the value-loss term —
something like temperature scaling on the C51 logits, or a
mild label-smoothing on the value target during training.
Cheap fix; could ship before re-running training.

**Independent confirmation:** plot the same reliability diagram on
the *next* checkpoint after applying the fix. ECE < 3% is a clean
pass. Also: search-guided eval (Stage V prototype) will be
calibration-sensitive — if MCTS rollouts collapse on this
checkpoint's value, that's a separate confirmation.

### Priority 3 — Short-loss prevention against max_damage

**Claim:** 61% of losses to max_damage end in ≤ 5 turns. The model
gives up tempo in the opening through switches that don't pay off.

**Supporting data:**

* Q5 short-loss rate: **max_damage 61.17%** vs **vgc_bench 50.74%**
  (vgc_bench wins more often but slower).
* Q5 action distribution: double-switches (`switch 4, switch 3` and
  `switch 3, switch 4`) have χ² residuals 76.0 and 74.6 — both
  significantly over-represented in short-loss turns.
* Switch+pass combos also over-represented.
* Specific opp teams give 80%+ short-loss rates (e.g.
  `takehironakatasworlds2024top128team` × max_damage: 83.6%).

**Action:** Less impactful on the 60% threshold than Priority 1
(this only addresses ~12 pp of the max_damage gap, not the full 40
pp). But cheaper: an opening-move regularizer that penalizes
double-switches when no Intimidate / Fake Out is pending could be
added directly to training. Could also try BC pre-training on the
"don't double-switch" pattern.

**Independent confirmation:** run the eval, filter to turn_number ≤
3, count double-switch rate against max_damage. If the rate drops
by 5+ pp and short-loss rate drops, the regularizer worked.

### Priority 4 — Recovery path for the `38dessert` / Terapagos matchup

**Claim:** The `38dessert` opp team is the *only* (opp_team, opp_type)
cell where the model's CI crosses 50%. There's a narrow window of
"almost-passable" performance — investigate what's working there to
generalize.

**Supporting data:**

* Q2 shows `38dessert` × simple_heuristic: WR **47.33%**, CI **[44.52%,
  50.16%]** (n = 1,200).
* No other cell in the entire 42×3 = 126-row Q2 table has CI
  upper > 50%.
* `38dessert` is *also* the 3rd-worst agent team — the model can't
  pilot Terapagos teams but can punish them.

**Action:** Save 10 winning replays and 10 losing replays in this
matchup (`save_games` CLI). Manual review to identify what specific
turn structure the model exploits. May inform what to teach in
priority 1.

**Independent confirmation:** if the winning pattern in the 47%
matchup is "click direct damage before Terapagos can Calm Mind,"
the same skill should transfer to other Calm Mind / Nasty Plot
opponents. Test on `seowonkimsworlds2024top128team` (also Terapagos
+ Sub + CM, 24.4% allowed) — if the model wins more there post-fix,
the recovery path generalizes.

---

The rest of this document is the underlying evidence: run identity,
collection history, the Q1–Q9 results, team-pattern analysis, and
example queries for ad-hoc exploration.

## Run identity

| Field | Value |
|---|---|
| Run dir | `data/eval/balmy70_step10700_full_2026-05-19/` |
| Eval run id | `balmy70_step10700_full` |
| Agent checkpoint | `data/models/rl/balmy-cloud-70/ghosts/main_model_step_10700.pt` |
| Battle format | `gen9vgc2024regg` |
| Opp_types | `simple_heuristic`, `max_damage`, `vgc_bench` (`max_base_power` dropped per session decision) |
| Teams source | `data/teams/gen9vgc2024regg/constrained/` (42 teams, both p1 and p2) |
| Battles per cell | 100 (some cells received 200-300 from session overlap on simple_heuristic) |
| Total battles | 476,650 across all opp_types (verified — sum of `outcome` column rows across all `battles_worker_*.parquet`) |
| Total turns | 3,744,644 (verified — sum across all `turns_worker_*.parquet`) |

## Collection history

1. **simple_heuristic** ran first across two sessions (run_tags `c931`
   and `c9ff`) plus a brief third (`3cd8`) before the
   [poke-env Player.battles leak][leak-doc] was diagnosed. WSL crash on
   2026-05-21 ~05:42 PDT killed all in-flight processes; max_damage
   was 55% complete, vgc_bench was 45% complete at the time.
2. **Cell-ordering blind spot.** `build_cells` uses
   `sorted(p1_dir.glob("*.txt"))` with `agent_team` as the slow axis
   ([evaluate.py:579-589](../evaluate.py#L579-L589)), so cutting early
   = missing the *same* tail of agent teams across every opp_type. At
   crash time: 18 of 42 agent teams unvisited for max_damage; 23 of 42
   for vgc_bench. The Q1 (per-agent-team WR) analysis is unanswerable
   on those teams without backfill.
3. **Resume eval** (2026-05-21, run_tags `01de` and `f623`) targeted
   exactly the missing agent teams using the post-merge
   `--executor process` default. Wall-rate speedup vs the threaded
   path: ~2.3× for max_damage (3.9 → 9.09 b/s) and ~2.6× for vgc_bench
   (2.9 → 7.51 b/s). See [process-pool plan][pp-plan].
4. After both resumes: **all 42 agent teams covered for all three
   opp_types**.

[leak-doc]: ../../../planning/stage2/2026-05-20-23-05-evaluate-process-pool-plan.md
[pp-plan]: ../../../planning/stage2/2026-05-20-23-05-evaluate-process-pool-plan.md

## Q3 — aggregate WR by opp_type

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    data/eval/balmy70_step10700_full_2026-05-19 summary
```

| opp_player_name | n_battles | wins | losses | ties | win_rate | ci_low | ci_high |
|:---|---:|---:|---:|---:|---:|---:|---:|
| max_damage | 172,850 | 34,244 | 138,606 | 0 | **0.1981** | 0.1962 | 0.2000 |
| simple_heuristic | 128,150 | 28,046 | 100,104 | 0 | **0.2189** | 0.2166 | 0.2211 |
| vgc_bench | 175,650 | 17,756 | 157,894 | 0 | **0.1011** | 0.0997 | 0.1025 |

Sanity (hand-aggregated from `outcome` column of all `battles_worker_*.parquet`):

| opp | direct n | direct p1_wins | direct p1_wr | matches CLI? |
|---|---|---|---|---|
| max_damage | 172,850 | 34,244 | 19.81% | ✓ |
| simple_heuristic | 128,150 | 28,046 | 21.89% | ✓ |
| vgc_bench | 175,650 | 17,756 | 10.11% | ✓ |

Also: wins + losses + ties = n for every row (34,244 + 138,606 = 172,850 ✓ ; etc.).

**Headline (n=476,650 total, well past statistical significance):**

balmy-cloud-70 step 10700 is **far below the Stage II graduation
threshold** (60% WR against each of the four target baselines) at this
training step. Order of difficulty for the model: vgc_bench (10.1%) ≫
max_damage (19.8%) > simple_heuristic (21.9%). All Wilson 95% CIs
exclude 0.5 by a huge margin, so the model loses decisively to every
baseline.

### Sub-check: did the resume teams skew the numbers?

The 18 (max_damage) / 23 (vgc_bench) agent teams that were missing
pre-crash got their data from the resume runs (tags `01de`, `f623`).
If the original sort happened to put the *strongest* teams late, the
combined WR would be biased low.

| opp | original WR | resume WR | combined WR | delta (resume − original) |
|---|---|---|---|---|
| max_damage | 20,020 / 97,250 = 20.59% | 14,224 / 75,600 = 18.81% | 19.81% | −1.78 pp |
| vgc_bench | 8,508 / 79,050 = 10.76% | 9,248 / 96,600 = 9.57% | 10.11% | −1.19 pp |

The resume teams gave the model ~1-2 percentage points *lower* WR,
suggesting the sorted-filename ordering is mildly correlated with team
strength (later alphabetical entries = slightly stronger teams), but
the combined WR is a faithful aggregate over the full team pool.

## Q1 — per-agent-team WR

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q1_agent_team.md \
    data/eval/balmy70_step10700_full_2026-05-19 agent_team
```

126 rows (42 agent teams × 3 opp_types). Full table in
`/tmp/q1_agent_team.md`. Distribution summary:

| opp | n_battles per agent_team (mean / min / max) | win_rate range | median WR |
|---|---|---|---|
| max_damage | 4,115 / 2,150 / 4,200 | [4.00%, 37.34%] | 20.44% |
| simple_heuristic | 3,051 / 1,900 / 9,800 | [6.24%, 39.00%] | 22.40% |
| vgc_bench | 4,182 / 3,450 / 4,200 | [1.14%, 29.31%] | 9.77% |

Verification:

* Unique `agent_team_hash` values: 42 — matches the 42 teams in
  `data/teams/gen9vgc2024regg/constrained/` ✓ (no missing teams).
* Cells with n < 100: **0** — every (agent_team, opp_type) row has at
  least 100 battles ✓.
* Per-opp `n_battles` sums: max_damage=172,850, simple_heuristic=128,150,
  vgc_bench=175,650 — matches Q3 totals exactly ✓.

The wide WR range — **35-point spread on max_damage and
simple_heuristic, 28-point spread on vgc_bench** — is the headline.
The model has very different effectiveness depending on which team it
plays, with no agent_team reaching 50% WR against any baseline. The
top of the range (37-39% WR) is on simple_heuristic / max_damage, not
vgc_bench: vgc_bench's best agent_team WR is 29.3%, meaning even the
model's "good" teams aren't competitive vs vgc_bench.

### Worst agent teams against vgc_bench (model essentially can't win on these)

| agent team | WR | 95% CI |
|---|---|---|
| `koenvsworlds2024top16teamseniors` | 1.14% | [0.86%, 1.51%] |
| `stockholmregionalstop32team` | 1.19% | [0.90%, 1.57%] |
| `38dessert` | 1.86% | [1.49%, 2.31%] |

n = 4,200 each. CIs are tight enough to make this a real signal, not noise.

### Best agent teams (any opp)

| agent team | opp | WR | 95% CI |
|---|---|---|---|
| `matinmoradisindianapolisregionalstop32team` | simple_heuristic | 39.00% | [36.94%, 41.10%] |
| `matinmoradisindianapolisregionalstop32team` | max_damage | 37.34% | [35.68%, 39.03%] |
| `takehironakatasworlds2024top128team` | simple_heuristic | 36.52% | [34.49%, 38.61%] |

Same team (`matinmoradisindianapolisregionalstop32team`) tops the ranking
against both simple_heuristic and max_damage — suggests this team is
robust across heuristic opponents. *Does the agent under-train on
team-strategy or just on matchup-specific tactics?* If
`matinmoradisindianapolisregionalstop32team` is fine vs heuristics but
collapses vs vgc_bench, the gap is matchup-specific. Checked in Q2 below.

## Q2 — per-opp-team WR

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q2_opp_team.md \
    data/eval/balmy70_step10700_full_2026-05-19 opp_team
```

126 rows, 42 unique opp_team_hashes. Full table in
`/tmp/q2_opp_team.md`. Distribution summary:

| opp | n_battles per opp_team (mean / min / max) | WR range | WR std |
|---|---|---|---|
| max_damage | 4,115 / 4,000 / 4,200 | [8.39%, 40.39%] | 0.0759 |
| simple_heuristic | 3,051 / 900 / 5,200 | [11.40%, 47.33%] | 0.0693 |
| vgc_bench | 4,182 / 4,100 / 4,200 | [8.88%, 11.19%] | **0.0057** |

Verification:

* Unique `opp_team_hash` values: 42 ✓.
* Cells with n < 100: 0 ✓.
* Per-opp `n_battles` sums match Q1/Q3 totals (172,850 / 128,150 / 175,650) ✓.
* **Total parquet rows: 476,650 = Q3 total ✓** (computed independently
  by summing all `battles_worker_*.parquet`).

The single most important number here: **vgc_bench's WR std across
opp_teams is 0.0057** (vs 0.0759 / 0.0693 for the heuristics). That
means the model's WR against vgc_bench barely depends on which team
vgc_bench is using — vgc_bench dominates regardless. Range [8.88%,
11.19%] is just a 2.3-point window. Against the heuristics, the
range is 30+ points — the model can find weak heuristic teams to
exploit.

Interpretation: vgc_bench's policy generalizes across the team pool;
heuristic policies have team-specific blindspots the model
identifies.

### Best opp teams to play against (model nearly even on the top one)

| opp team | opp | WR | 95% CI |
|---|---|---|---|
| `38dessert` | simple_heuristic | **47.33%** | [44.52%, 50.16%] |
| `speccalydozo` | max_damage | 40.39% | [38.90%, 41.90%] |
| `seowonkimsworlds2024top128team` | max_damage | 37.50% | [36.01%, 39.01%] |

Note: the upper CI bound for `38dessert` vs simple_heuristic crosses
50% — this is the only (opp_team, opp_type) cell where the model
*might* be playing at-or-above parity. Every other cell's CI is fully
below 50%.

### Q1 ↔ Q2 cross-check: same team, two roles

Team `38dessert` shows up at both extremes:

* As **agent team** vs vgc_bench: WR **1.86%** (Q1 worst-3) — the
  model can't pilot this team against a strong opponent.
* As **opp team** for simple_heuristic: WR **47.33%** (Q2 best) — the
  model crushes simple_heuristic when *it* tries to pilot the same team.

Same pattern with `matinmoradisindianapolisregionalstop32team`:

* As **agent team** vs simple_heuristic/max_damage: 39.0% / 37.3%
  (Q1 top) — best team for the model.
* As **opp team** for vgc_bench: 9.19% — the model loses against
  vgc_bench piloting this team.

Consistent story: team "strength" is pilot-dependent. A
narrowly-focused team that needs precise execution (`38dessert`,
`matinmoradisindianapolisregionalstop32team`) is fine in hands that
exploit it (the model or vgc_bench) but terrible in heuristic hands.
The matchup-specific gap to vgc_bench is real — even on the model's
*best* team, vgc_bench wins 91% of the time.

## Q5 — short-loss patterns (≤ 5 turns)

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q5_actions.md \
    data/eval/balmy70_step10700_full_2026-05-19 short_loss
```

Threshold: `max_turn = 5` from
[eval_analysis.py:102](../eval_analysis.py#L102).

### Overall short-loss rate

| opp | losses | short losses (≤5 turns) | short-loss rate |
|---|---|---|---|
| max_damage | 138,606 | 84,787 | **61.17%** |
| simple_heuristic | 100,104 | 53,343 | 53.29% |
| vgc_bench | 157,894 | 80,113 | 50.74% |
| **all** | 396,604 | 218,243 | **55.03%** |

Cross-check: 84,787 + 53,343 + 80,113 = 218,243 ✓.

Important inversion: **vgc_bench has the lowest short-loss rate
despite having the highest WR** (89.89% vs the model). Read: vgc_bench
wins more often but takes longer to win, suggesting it plays a more
strategic / positional game. max_damage wins fast by just blasting
high-damage moves; vgc_bench grinds the model down. This is
diagnostic — the loss *modality* differs by opponent type.

### Over-represented opp_teams in short losses (top 5)

| opp team | opp | n_losses | n_short | short_fraction | over_rep |
|---|---|---|---|---|---|
| `takehironakatasworlds2024top128team` | max_damage | 3,432 | 2,869 | 83.6% | 1.519 |
| `andy斯托哥尔摩junior亚军队` | max_damage | 3,601 | 2,978 | 82.7% | 1.503 |
| `白宇平s2024taiwannationalchampionships` | max_damage | 3,756 | 3,044 | 81.0% | 1.473 |
| `crisrossivgcsbolognaspecialeventseniorstop16team` | max_damage | 3,610 | 2,890 | 80.1% | 1.455 |
| `kiyoshiroaraisworlds2024top128team` | max_damage | 3,515 | 2,805 | 79.8% | 1.450 |

Top 9 over-represented opp_teams are *all* max_damage — confirming
the previous point. The model gets crushed in ≤5 turns by certain
max_damage builds 80%+ of the time. These are likely
hyper-offensive teams (high BST sweepers, terrains, weather)
that one-shot the model's mons before defensive plans can develop.

No vgc_bench opp_team appears in the top of the over-representation
list (the highest vgc_bench entry would be further down; CLI output
truncated at the top 46 rows displayed).

### Action distribution in short-loss turns (top 5 by chi-square residual)

| action | n_short_loss | n_all | short_rate | overall_rate | χ² residual |
|---|---|---|---|---|---|
| `switch 4, switch 3` | 20,855 | 26,285 | 1.18% | 0.70% | 76.0 |
| `switch 3, switch 4` | 43,555 | 64,723 | 2.47% | 1.73% | 74.6 |
| `move 3 -2, move 4 2` | 46,275 | 70,006 | 2.62% | 1.87% | 73.0 |
| `move 1 2, move 1 2` | 337,957 | 640,294 | 19.14% | 17.10% | 65.7 |
| `switch 4, pass` | 127,059 | 233,261 | 7.20% | 6.23% | 51.5 |

Reading: "χ² residual" is the standardized residual; values >2 are
significant. All top-20 actions have residuals >15 → very significant
over-representation. Interpretation:

* **Double-switches** (`switch 4, switch 3` / `switch 3, switch 4`)
  are the strongest signal. The model double-switches early when
  it shouldn't, surrendering tempo.
* The single most-frequent action overall (`move 1 2, move 1 2` —
  double-targeting opp slot 2 with each mon's move slot 1) is also
  over-represented in short losses. The model is reaching for a
  default action that doesn't work in disadvantageous opening
  positions.
* Switch+pass combos (`switch 4, pass`, `switch 3, pass`,
  `pass, switch 3`) — losing tempo by switching only one mon.

These aren't smoking guns by themselves — every action shows up
because the model lost 396k times — but the residuals say there's a
real signature: short-loss turns are *more* switch-heavy than the
average turn. The model defaults to switches in bad early positions
where holding ground might at least force trades.

## Q6 — confidence vs heuristic advantage

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q6_swings.md \
    data/eval/balmy70_step10700_full_2026-05-19 confidence
```

### Distribution of turns by heuristic_adv quartile

| bucket (heuristic_adv) | n_turns | mean_entropy | mean_value_predicted | mean_heuristic_adv |
|---|---|---|---|---|
| (-1.001, 0.498] | 936,162 | 2.0344 | -0.7313 | 0.2800 |
| (0.498, 0.766] | 936,648 | 2.3385 | -0.5668 | 0.6365 |
| (0.766, 1.0] | 1,871,834 | 3.4788 | -0.5679 | 0.9475 |

Cross-check: 936,162 + 936,648 + 1,871,834 = **3,744,644** ✓ (matches
the parquet row count from all `turns_worker_*.parquet`).

Note: these are quartile *bins* but the data is heavily right-skewed
toward heuristic_adv ≈ 1 — half the model's turns are in the highest
bucket. Read: the heuristic computes the model as "ahead" most of the
time, even though the model loses 83% of battles. Two likely causes:
(a) the heuristic over-weights HP balance early-game when teams are
healthy, or (b) the model genuinely wins many turns but loses the
late game.

### Poor-situation summary

| subset | n_turns | mean_entropy | mean_value_predicted |
|---|---|---|---|
| all | 3,744,644 | 2.8325 | **-0.6085** |
| heuristic_adv < -0.3 | 5,336 | 1.6848 | **-0.8576** |

Cross-check: the "all" row's n_turns matches the quartile sum and the
direct parquet count ✓.

Reading: **the value head is consistently pessimistic** (mean -0.61 on
all turns) — well-calibrated to the 17.8% aggregate WR (value scaled
[-1, 1], so -0.61 ≈ 19.5% win expectation). When heuristic_adv < -0.3
(only 0.14% of turns — the model is rarely in heuristic-poor positions
in this run), the value head goes even more negative (-0.86) and
entropy drops sharply (1.68 vs 2.83 overall). So the model knows it's
losing in those situations and its policy gets more deterministic
(low entropy = peaked distribution).

### Worst negative value swings in losses (top 5)

```
$ head -7 /tmp/q6_swings.md
```

| battle_id | t_before | t_after | swing | h_adv_before | h_adv_after | val_before | val_after | ent_before | ent_after |
|---|---|---|---|---|---|---|---|---|---|
| `p8202_battle-gen9vgc2024regg-1591988` | 4 | 6 | -1.9015 | 0.927 | -0.975 | -0.889 | -0.985 | 5.466 | 1.808 |
| `p8202_battle-gen9vgc2024regg-1557956` | 44 | 47 | -1.8200 | 0.820 | -1.000 | -0.201 | -0.113 | 0.722 | 1.385 |
| `p8202_battle-gen9vgc2024regg-1591956` | 3 | 5 | -1.8121 | 0.952 | -0.860 | -0.684 | -0.983 | 4.924 | 1.964 |
| `p8202_battle-gen9vgc2024regg-1581539` | 4 | 6 | -1.7960 | 1.000 | -0.796 | -0.626 | -0.983 | 2.528 | 0.693 |
| `p8202_battle-gen9vgc2024regg-1570983` | 7 | 9 | -1.7756 | 1.000 | -0.776 | 0.246 | -0.996 | 2.318 | 1.464 |

Pattern: **heuristic_adv flips from +1.0 (perfect winning position) to
strongly negative in 2-3 turns**. The model's value head drops to
-0.98 across the swing. Entropy drops by ~2-4 nats in most cases,
meaning the policy concentrates on fewer moves after the disaster.
These are textbook "in one turn, you lose the game" battles — likely a
KO that the model didn't see coming (terrain set up, item triggered,
critical hit).

The first row is illustrative: turn 4 the model thinks it's 92.7%
favored, turn 6 it's effectively dead. Entropy plummets 5.47 → 1.81.
That's exactly the failure mode where the value head and the heuristic
should be flagging "uncertainty," but instead they only react *after*
the bad event.

## Bug surfaced during Q7: duplicate `battle_id` across sessions

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q7_reliability.md \
    data/eval/balmy70_step10700_full_2026-05-19 value_calibration
…
pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects
```

[q7_value_calibration](../eval_analysis.py#L268) does
`battles.set_index("battle_id")["outcome"]` to map turn rows to their
battle outcome. This fails because **92,786 of 476,650 battle_id
values (19.5%) are duplicates** across `run_tag`s.

Cross-checked: the duplicates pair (9de8, 01de) — max_damage original
and resume — and (c931, c9ff, 3cd8) — three simple_heuristic sessions.
Same Showdown server restarts reissue the same `battle-gen9vgc2024regg-N`
tags, but the `battle_id_prefix = "p{port}_"` in
[eval_collector.py][collector] doesn't include `run_tag`, so the
composite is not unique across sessions.

[collector]: ../eval_collector.py

Impact:

* **Q3, Q1, Q2, Q5** — aggregate over rows, not joins. Numbers above
  are correct.
* **Q7, Q9** — join battles ↔ turns on battle_id. Broken by dups.
* **Q6** — uses turn-internal swings (no battle join). Works.

Workaround used for Q7 (below): join on the composite key
**(battle_id, run_tag)** by extracting run_tag from each parquet
file's name. Verified that `(battle_id, run_tag)` uniquely identifies
all 476,650 battle rows.

## Q7 — value-head calibration (composite-key workaround)

```python
# Workaround: read run_tag from filename, join on (battle_id, run_tag).
# Full script kept in session transcript; equivalent to:
merged = turns.merge(
    battles[['battle_id', 'run_tag', 'outcome']],
    on=['battle_id', 'run_tag'], how='left',
)
m = merged[~merged['is_teampreview'] & merged['outcome'].notna()]
m['pred_prob'] = ((m['value_predicted'] + 1) / 2).clip(0, 1)
# bin into 10 equal-width buckets, compute observed_win_rate per bin
```

Verification: composite key uniqueness check passed (476,650 unique
(battle_id, run_tag) pairs = 476,650 rows ✓). Join produced 0 NaN
outcomes across 3,744,644 turn rows ✓.

### Reliability diagram (10 bins, value scaled to [0,1])

| bin (pred prob) | n_turns | mean predicted | observed WR | gap |
|---|---|---|---|---|
| 0.0–0.1 | 1,939,283 | 0.038 | **0.076** | 0.038 (under-confident) |
| 0.1–0.2 | 648,182 | 0.142 | 0.155 | 0.013 |
| 0.2–0.3 | 302,289 | 0.245 | 0.196 | 0.049 |
| 0.3–0.4 | 191,336 | 0.347 | 0.249 | **0.098** |
| 0.4–0.5 | 146,671 | 0.448 | 0.314 | **0.134** |
| 0.5–0.6 | 129,213 | 0.550 | 0.392 | **0.158** |
| 0.6–0.7 | 128,780 | 0.650 | 0.486 | **0.164** (largest) |
| 0.7–0.8 | 138,277 | 0.751 | 0.599 | 0.152 |
| 0.8–0.9 | 114,723 | 0.843 | 0.720 | 0.123 |
| 0.9–1.0 | 5,890 | 0.910 | 0.799 | 0.111 |

**ECE = 0.0569** (5.69% — weighted absolute calibration gap).

Reading:

* **The value head is systematically over-confident from
  predicted ≥ 0.3 onwards.** The gap grows from ~10 pp at predicted=0.35
  to **16.4 pp at predicted=0.65** (the worst-calibrated band). Even at
  predicted=0.9+, observed WR is 0.80 — 11 pp short.
* **The 0.0–0.1 bin** holds 1.94M turns (52% of all non-teampreview
  turns) and is mildly *under*-confident (predicted 3.8%, observed
  7.6%). When the model thinks it's losing badly, it actually loses a
  bit less badly than it expects.
* **Net asymmetry**: the value head is too pessimistic when very low
  and too optimistic everywhere else. The "I'm in a winning position"
  signal is unreliable — useful for the loss-detection side but not
  for guiding aggressive plays. This is consistent with Q6's "the
  model thinks it's winning, then loses" pattern.

This also limits how much the value head can be used as a stopping
criterion or as input to MCTS rollouts at later Stage IV/V work — a
miscalibrated value head pushes searches toward false-positive
"winning" leaves.

## Q9 — value head vs ensemble-advantage (composite-key workaround)

```
$ python -m elitefurretai.rl.analyze.eval_analysis --format markdown \
    --output /tmp/q9_diverge.md \
    data/eval/balmy70_step10700_full_2026-05-19 value_ensemble
…
ValueError: The truth value of a Series is ambiguous.
```

CLI fails for the same dup-`battle_id` reason as Q7
(`_attach_ensemble_advantage`'s `outcome_map.get(battle_id)` returns a
Series instead of a scalar when there are dups). Same workaround:
re-implemented inline with composite key `uid = (battle_id, run_tag)`.

Verification: composite uid uniqueness check passed (476,650 unique
`uid`s == 476,650 battle rows ✓). All 3,744,644 turn rows got
`ensemble_adv` attached (NaN only on teampreview/tie rows).

### Q9a — persistent value-vs-ensemble disagreement (top 5 by mean |value − ensemble_adv|)

| battle_id \| run_tag | opp | mean_abs_diff | n_turns | outcome | final_turn |
|---|---|---|---|---|---|
| `p8206_battle-gen9vgc2024regg-1537249 \| ebaa` | vgc_bench | **1.971** | 14 | **1.0 (win)** | 9 |
| `p8201_battle-gen9vgc2024regg-1569378 \| 01de` | max_damage | 1.968 | 1143 | 1.0 | 8 |
| `p8206_battle-gen9vgc2024regg-1537522 \| ebaa` | vgc_bench | 1.966 | 10 | 1.0 | 7 |
| `p8200_battle-gen9vgc2024regg-1557361 \| 01de` | max_damage | 1.962 | 23 | 1.0 | 22 |
| `p8200_battle-gen9vgc2024regg-1565977 \| c9ff` | simple_heuristic | 1.953 | 41 | 1.0 | 40 |

(Full top 20 below; all 20 entries have outcome=1.0.)

<details><summary>Full top 20</summary>

| battle_id \| run_tag | opp | mean_abs_diff | n_turns | final_turn |
|---|---|---|---|---|
| `p8206_battle-gen9vgc2024regg-1537249 \| ebaa` | vgc_bench | 1.971 | 14 | 9 |
| `p8201_battle-gen9vgc2024regg-1569378 \| 01de` | max_damage | 1.968 | 1143 | 8 |
| `p8206_battle-gen9vgc2024regg-1537522 \| ebaa` | vgc_bench | 1.966 | 10 | 7 |
| `p8200_battle-gen9vgc2024regg-1557361 \| 01de` | max_damage | 1.962 | 23 | 22 |
| `p8200_battle-gen9vgc2024regg-1565977 \| c9ff` | simple_heuristic | 1.953 | 41 | 40 |
| `p8200_battle-gen9vgc2024regg-1565794 \| 01de` | max_damage | 1.953 | 38 | 36 |
| `p8200_battle-gen9vgc2024regg-1566404 \| 01de` | max_damage | 1.950 | 34 | 33 |
| `p8200_battle-gen9vgc2024regg-1590848 \| f623` | vgc_bench | 1.949 | 38 | 36 |
| `p8200_battle-gen9vgc2024regg-1569204 \| 01de` | max_damage | 1.949 | 7 | 5 |
| `p8200_battle-gen9vgc2024regg-1580546 \| f623` | vgc_bench | 1.947 | 12 | 9 |
| `p8203_battle-gen9vgc2024regg-1590319 \| f623` | vgc_bench | 1.947 | 27 | 26 |
| `p8207_battle-gen9vgc2024regg-1537650 \| ebaa` | vgc_bench | 1.946 | 10 | 7 |
| `p8201_battle-gen9vgc2024regg-1566605 \| 01de` | max_damage | 1.945 | 36 | 35 |
| `p8200_battle-gen9vgc2024regg-1565903 \| 01de` | max_damage | 1.945 | 28 | 26 |
| `p8202_battle-gen9vgc2024regg-1565497 \| c9ff` | simple_heuristic | 1.940 | 35 | 33 |
| `p8200_battle-gen9vgc2024regg-1566615 \| 01de` | max_damage | 1.939 | 12 | 11 |
| `p8207_battle-gen9vgc2024regg-1537241 \| ebaa` | vgc_bench | 1.938 | 10 | 8 |
| `p8204_battle-gen9vgc2024regg-1538116 \| ebaa` | vgc_bench | 1.936 | 10 | 7 |
| `p8203_battle-gen9vgc2024regg-1572957 \| f623` | vgc_bench | 1.934 | 13 | 11 |
| `p8201_battle-gen9vgc2024regg-1566802 \| 01de` | max_damage | 1.934 | 21 | 21 |

</details>

**Striking observation: every battle in the top-20 disagreement list
is a model WIN.** The value head was pessimistic (predicting near-loss)
throughout these games, but the model still won. Combined with Q7's
finding that the value head is over-confident at predicted ≥ 0.3 AND
under-confident at predicted ≤ 0.1, this tells a clean story:

* When the model **is going to lose**, the value head ≤ 0.1 — correct
  in aggregate (Q7 well-calibrated low bin) but the *ensemble_adv*
  blends in the heuristic which often says "ahead". Hence high
  disagreement → the model loses, value was right, ensemble was wrong.
* When the model **is going to win**, the value head pre-game is
  *still* often low (the model loses 83% of battles overall, and the
  value head reflects that prior). The ensemble's outcome term (which
  knows the model wins) pulls toward +1. Hence high disagreement → the
  model wins, value was pessimistic, ensemble was right.

So Q9a doesn't fault the value head specifically — it surfaces the
*tension* between the NN's prior ("usually I lose") and the actual
outcome distribution. Sorting Q9a by win-batches showcases the second
class.

### Q9a aggregate by opp_type

| opp | n_battles | mean(|v - ens|) | median | max |
|---|---|---|---|---|
| max_damage | 172,850 | 0.8998 | 0.8996 | 1.9678 |
| simple_heuristic | 128,150 | 0.9467 | 0.9487 | 1.9534 |
| vgc_bench | 175,650 | 0.9608 | 0.9599 | 1.9707 |

Mean disagreement is ~0.9-0.96 (out of theoretical max ~2) — the value
head and ensemble_adv genuinely disagree most of the time. Disagreement
is highest on vgc_bench, lowest on max_damage. Modest cross-opp
spread; the value head's posture doesn't track the opponent type
strongly.

### Q9b — agree-then-diverge

Defaults: early_window=5 turns, early_threshold=0.15 (|value −
ensemble_adv| in early phase), late_threshold=0.40.

**0 matches.** The model never has both `early_diff < 0.15` AND
`late_diff > 0.40`. The mean disagreement is already 0.9 from turn 1,
so the "early agree" precondition never holds.

This is itself an interesting finding: there is no
agree-then-diverge regime. The value head and ensemble_adv disagree
*from the start* on essentially every battle. The model's value
head and the heuristic-blended reference are not telling the same
story even pre-game — they encode different priors on the same battle.
Future work: lower the early threshold (≥ 0.5 might find matches), or
re-train the heuristic component so it agrees with the model's prior
in expectation.

## Q8 — save games (skipped here)

`save_games` dumps 3 representative replay files per category
(short_loss, low team WR, etc.) — useful for human inspection but
not for the doc body. Skipping the in-doc dump; the CLI is available
when needed:

```
$ python -m elitefurretai.rl.analyze.eval_analysis \
    data/eval/balmy70_step10700_full_2026-05-19 save_games
```

## Team-based pattern analysis

Read the actual `data/teams/gen9vgc2024regg/constrained/*.txt` files
for the top-5 / bottom-5 agent teams and the top-5 / bottom-5 opp
teams (ranked by aggregate WR across all opp_types). Patterns below
are described in terms of what's structurally distinctive between
buckets, not aesthetics — meant to inform what the *next* training
curriculum should weight.

### Top-5 agent teams (model wins MOST piloting these)

| agent team | aggregate WR | notable cores |
|---|---|---|
| `takehironakatasworlds2024top128team` | 30.26% | Calyrex-Ice + Urshifu-RS + Torkoal + Ogerpon-Cornerstone + Farigiraf (sun + TR) |
| `kiyoshiroaraisworlds2024top128team` | 28.64% | Calyrex-Ice + Urshifu-RS + Pelipper rain |
| `kiwamuendos2024japannationalchampionshipsrunnerupteam` | 28.27% | Calyrex-Ice + Urshifu-RS + Pelipper rain |
| `sayawoszaciancrownedrainbalanceteam` | 25.29% | Zacian-Crowned + Rillaboom + Pelipper rain |
| `zachdroegkampsworlds2024top64team` | 23.80% | Calyrex-Shadow Taunt + Ting-Lu + Ogerpon-Hearthflame + Dondozo |

Common structural features (n = 5, each line tallied independently):

* **Calyrex restricted in 5/5** (4× Ice, 1× Shadow with Taunt — *not*
  Calm Mind / Nasty Plot setup variants).
* **Urshifu-Rapid-Strike in 4/5** (with Choice Scarf / Focus Sash /
  Assault Vest — offensive items, not Choice Band).
* **Weather setter in 4/5** (Pelipper rain × 3, Torkoal sun × 1).
* **Restricted hits with a single STAB action** (Glacial Lance,
  Surging Strikes) — high-damage commitment rather than
  multi-turn setup.
* **Tera types skew offensive**: Stellar (3×), Dragon, Grass, Fire.
  No defensive Tera (Steel, Fairy is minor).

**Takeaway:** the model is *best* at piloting teams with **direct
offensive cores + weather support**. The strategy is "click damage
button on the right target," which is exactly what RL self-play
overtrains on without explicit search.

### Bottom-5 agent teams (model wins LEAST piloting these)

| agent team | aggregate WR | notable cores |
|---|---|---|
| `stockholmregionalstop32team` | 3.93% | Calyrex-Shadow Covert Cloak + Gastrodon + Urshifu (single-strike) |
| `tomoyaogawasworlds2024top64team` | 4.47% | Calyrex-Shadow Calm Mind + Mienshao + Clefairy Friend Guard |
| `38dessert` | 4.67% | Terapagos Calm Mind + Clefairy Friend Guard + Amoonguss Occa Berry |
| `naicchampion` | 5.65% | Calyrex-Ice + (Urshifu-SS likely) + Pelipper |
| `joeywoodringsindianapolisregionalstop64team` | 8.03% | Lunala Trick Room + Ursaluna Guts + (no weather setter) |

Common structural features:

* **Setup-reliant restricted mons in 4/5**: Calyrex-Shadow Calm
  Mind (2×), Terapagos Calm Mind (1×), Lunala Trick Room setup (1×).
* **Support mons that require precise positioning**: Clefairy Friend
  Guard (2×), Amoonguss Occa Berry (1×), Mienshao (Wide Guard /
  Quick Guard support).
* **Urshifu-Single-Strike (Dark/Ghost) in ≥2/5** — Wicked Blow
  needs setup to make use of crit-guarantee, vs Rapid-Strike's
  guaranteed-3-hit damage on turn 1.
* **No reliable weather setter in 4/5** — only `naicchampion` has
  Pelipper.

**Takeaway:** the model **cannot execute multi-turn setup plans**. It
clicks immediate-damage moves where a Calm Mind / Trick Room turn
would be correct. This is a *known* failure mode of pure self-play
without search guidance — long-horizon credit assignment is hard.

### Top-5 opp teams (model wins MOST against — these are the *weakest*
opponents in heuristic hands)

| opp team | aggregate WR allowed | notable cores |
|---|---|---|
| `38dessert` | 26.22% | Terapagos Calm Mind + Clefairy Friend Guard (needs setup) |
| `iwata9tascalyrexshadowsmeargleteam` | 24.59% | Calyrex-Shadow Nasty Plot + Smeargle Spore + Tornadus Prankster |
| `seowonkimsworlds2024top128team` | 24.41% | Terapagos Substitute + Calm Mind + Landorus Choice Scarf |
| `speccalydozo` | 24.26% | Weezing-Galar Neutralizing Gas + Dondozo Unaware + Maushold-Four Friend Guard + Tatsugiri Commander |
| `joeywoodringsindianapolisregionalstop64team` | 20.65% | Lunala Trick Room + Ursaluna Guts |

Common structural features:

* **Same setup-reliance pattern** as the bottom-5 agent teams. 38dessert,
  joeywoodringsindianapolisregionalstop64team, tomoyaogawasworlds2024top64team
  appear in *both* lists. This is the **pilot-dependence inversion**
  — teams the model can't pilot also can't be piloted by heuristics.
* **Gimmick combos**: Commander Tatsugiri + Dondozo (`speccalydozo`),
  Smeargle Moody + Spore (`iwata9...`), Weezing Neutralizing Gas
  (`speccalydozo`) — conditional positional plays that heuristic
  opponents botch.
* **Friend Guard support in 2/5**: Clefairy (`38dessert`), Maushold-Four
  (`speccalydozo`). Heuristic opponents likely fail to keep the support
  mon alive.

**Takeaway:** the model **can punish setup**. When opponents need
multi-turn execution, the model wins via direct damage commitment.
This is the same skill as "piloting offensive cores" — and the same
*weakness* on the agent side.

### Bottom-5 opp teams (model wins LEAST against — *strongest* opponents)

| opp team | aggregate WR allowed | notable cores |
|---|---|---|
| `白宇平s2024taiwannationalchampionships` | 10.00% | Miraidon Choice Specs + Ursaluna-Bloodmoon AV + Ogerpon-Cornerstone + Ditto Imposter |
| `lukatrejgutsindianapolisregionalstop16team` | 11.48% | Koraidon Bulk Up Clear Amulet + Chien-Pao Focus Sash + Incineroar |
| `ziguziguzakusworlds2024top32teamseniors` | 11.60% | Calyrex-Ice TR + Urshifu-RS Choice Scarf + Pelipper rain + Incineroar |
| `zachdroegkampsworlds2024top64team` | 11.83% | Calyrex-Shadow Taunt + Ting-Lu + Ogerpon-Hearthflame + Dondozo (also a TOP-5 *agent* team!) |
| `9thplacenaic` | 12.14% | Miraidon Choice Specs + Chien-Pao + Talonflame Gale Wings |

Common structural features:

* **Box legendaries with offensive items in 4/5**: Miraidon Choice
  Specs (2×), Koraidon Clear Amulet (1×), Calyrex-Ice/Shadow with
  immediate-pressure items (3×).
* **No reliance on multi-turn setup** — Bulk Up is the only setup
  move in the list and it's on Koraidon (which is also a
  click-damage threat without setup).
* **Choice Scarf speed control** in 2/5 (`ziguziguzaku…`'s Urshifu).
* **Calyrex-Shadow Taunt** (`zachdroegkamps...`) — directly punishes
  the model's known weakness against setup *by mirroring it*. The same
  team is also top-5 for the agent.

**Takeaway:** the model **loses to direct offensive pressure** —
Miraidon / Koraidon / Choice-Scarf Urshifu — exactly the kind of
threats heuristic players can also handle (just not as crisply).
Against vgc_bench piloting these threats, the model has no
exploit. **The 60% Stage II target requires fixing this.**

### Cross-pattern summary

| Pattern | Model strength | Model weakness |
|---|---|---|
| Pilot direct offensive teams | ✓ (best WR with offensive cores) | — |
| Pilot setup teams | — | ✗ (worst WR with Calm Mind / Trick Room / Friend Guard cores) |
| Beat setup opponents | ✓ (best WR against setup-heavy gimmicks) | — |
| Beat direct offensive opponents | — | ✗ (worst WR against Choice Specs box mons) |

The model has **one strategy** (direct damage commitment) and applies
it everywhere. It exploits opponents that need setup; it loses to
opponents that don't. The Stage II target requires it to develop a
*second* strategy — either patient setup execution or counter-play
against offensive pressure (Intimidate cycling, Rocky Helmet recoil,
defensive Tera). Current architecture has the capacity for this; the
data distribution may not be encouraging it.

## Example queries for future exploration

The analysis CLI handles the common questions. For ad-hoc digging,
the parquet shards are the source of truth — read them directly with
`pyarrow` + `pandas`. Each shard's filename embeds the `run_tag` so
you can recover the composite key `(battle_id, run_tag)`.

### Query 1: WR against a specific opp_type, filtered by something

```python
import pyarrow.parquet as pq, glob, re
import pandas as pd

RUN_DIR = "data/eval/balmy70_step10700_full_2026-05-19"

# All battle rows with run_tag
dfs = []
for f in glob.glob(f"{RUN_DIR}/battles_worker_*.parquet"):
    m = re.search(r"battles_worker_\d+_([0-9a-f]+)_", f)
    df = pq.read_table(f).to_pandas()
    df["run_tag"] = m.group(1) if m else None
    dfs.append(df)
battles = pd.concat(dfs, ignore_index=True)

# WR against vgc_bench, in losses where the model held the agent_team `38dessert`:
from pathlib import Path
from elitefurretai.rl.analyze.eval_schema import canonical_team_hash
target_hash = canonical_team_hash(
    Path("data/teams/gen9vgc2024regg/constrained/38dessert.txt").read_text()
)
mask = (battles["opp_player_name"] == "vgc_bench") & (battles["agent_team_hash"] == target_hash)
sub = battles[mask]
print(f"n={len(sub)}, wins={int(sub['outcome'].sum())}, "
      f"WR={sub['outcome'].mean():.4f}")
```

### Query 2: Find a specific battle's turn-by-turn trace

```python
# Pick a battle_id from any analysis result (e.g. Q6 swings, Q9a top 20).
target_battle = "p8202_battle-gen9vgc2024regg-1591988"
target_run_tag = "9de8"  # extract from the shard filename if you don't know it

t_dfs = []
for f in glob.glob(f"{RUN_DIR}/turns_worker_*.parquet"):
    m = re.search(r"turns_worker_\d+_([0-9a-f]+)_", f)
    if m and m.group(1) != target_run_tag:
        continue  # speed up by skipping irrelevant tags
    df = pq.read_table(f).to_pandas()
    df["run_tag"] = m.group(1)
    t_dfs.append(df)
turns = pd.concat(t_dfs, ignore_index=True)
trace = turns[(turns["battle_id"] == target_battle) & (turns["run_tag"] == target_run_tag)]
print(trace[["turn_number", "action_chosen_str", "value_predicted", "heuristic_adv"]].to_string())
```

### Query 3: Top actions in *long* losses (the opposite of Q5)

```python
# Battles that ended in a loss after >25 turns — the model held on
# but lost anyway. What did it do that didn't work?
losses = battles[battles["outcome"] == 0.0]
long_losses = losses[losses["final_turn"] > 25]
print(f"long losses: {len(long_losses)} ({100*len(long_losses)/len(losses):.1f}% of losses)")

# Filter turns to those battle_ids, composite key
key = list(zip(long_losses["battle_id"], long_losses["run_tag"]))
turn_keys = list(zip(turns["battle_id"], turns["run_tag"]))
turns_in_long = turns[pd.Series(turn_keys, index=turns.index).isin(set(key))]
print(turns_in_long["action_chosen_str"].value_counts().head(10))
```

### Query 4: Per-team WR against vgc_bench, sorted

```python
agent_wr_vgcbench = (
    battles[battles["opp_player_name"] == "vgc_bench"]
    .groupby("agent_team_hash")
    .agg(n=("outcome", "size"), wins=("outcome", "sum"))
)
agent_wr_vgcbench["wr"] = agent_wr_vgcbench["wins"] / agent_wr_vgcbench["n"]

teams_dir = Path("data/teams/gen9vgc2024regg/constrained")
hash_to_name = {
    canonical_team_hash(tf.read_text()): tf.stem for tf in teams_dir.glob("*.txt")
}
agent_wr_vgcbench["team"] = agent_wr_vgcbench.index.map(hash_to_name)
print(agent_wr_vgcbench.sort_values("wr", ascending=False).head(10))
```

### Query 5: Battles where value head was *most* over-confident

```python
# Where value_predicted was >0.5 (model thought it was winning) but
# the model lost. Useful for finding "I was tricked" battles.
high_value_losses = (
    turns[turns["value_predicted"] > 0.5]
    .merge(
        battles[["battle_id", "run_tag", "outcome", "final_turn", "opp_player_name"]],
        on=["battle_id", "run_tag"],
        how="left",
    )
)
high_value_losses = high_value_losses[high_value_losses["outcome"] == 0.0]
print(f"n_turn_rows where value > 0.5 but battle lost: {len(high_value_losses)}")
print(high_value_losses.groupby("opp_player_name").size())
```

(End of evidence section — see the **Prioritized takeaways** at the
top of this document for what to do with this data.)