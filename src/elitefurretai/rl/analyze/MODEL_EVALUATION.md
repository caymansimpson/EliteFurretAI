# MODEL_EVALUATION.md — running the Plan B analysis pipeline

Operational log for the first end-to-end run of the Plan B trajectory
collection + analysis pipeline (commits `0245183` through `1a08551`
implement the pipeline; this document is the first-run report).

## Run identity

| Field | Value |
|---|---|
| Run dir | `data/eval/initial_run_2026-05-19/` |
| Eval run id | `initial_run_001` |
| Agent checkpoint | `data/models/rl/zany-cloud-67/main_model_step_2204.pt` |
| Battle format | `gen9vgc2024regg` |
| Schedule | `simple_heuristic` → `max_damage` → `max_base_power` → `vgc_bench` |
| Cells | 3 agent teams × 3 opp teams = 9 per opp_type |
| Battles per cell | 10 |
| Replay sample rate | 1.0 (every battle) |
| Total battles | 4 × 90 = 360 |
| Total replays on disk | 360 (1.9 MB run dir) |
| Wall clock | ~2 minutes total |

The checkpoint `zany-cloud-67` was chosen because the active may16
training run (`balmy-cloud-70`) is no longer running here, but
zany-cloud-67's checkpoint file is stable and predates the embedder
drift that broke earlier checkpoints. It is an early-RL snapshot
(step 2204 ≈ ~2k learner updates) — these results characterize the
*pipeline*, not the model's capability.

## Topology

* Showdown servers: localhost ports 8200–8203 (4 servers), launched
  manually outside the eval CLI to avoid the boot-race condition we
  hit when `--launch-servers` returns before sockets accept.
* Workers: 4 for in-process opp_types; 1 for `vgc_bench` (the
  external subprocess shares a single Showdown server per worker).
* Device: `cuda` (GPU was idle — `balmy-cloud-70` was stopped).
* Cell iteration: ON.

## Per-opp_type results

| Opp player | Battles | P1 wins | P1 losses | P1 WR | Duration | Errors |
|---|---|---|---|---|---|---|
| simple_heuristic | 90 | 12 | 78 | 13.3% | ~30s | 0 |
| max_damage | 90 | 24 | 66 | 26.7% | ~30s | 0 |
| max_base_power | 90 | 10 | 80 | 11.1% | ~30s | 0 |
| vgc_bench | 90 | 4 | 86 | 4.4% | ~40s | 0 |
| **Total** | **360** | **50** | **310** | **13.9%** | ~2m | **0** |

zany-cloud-67 is undertrained: even against `max_damage` (the
weakest baseline), it manages only 27% WR.

## Analysis subcommand outputs

All eight subcommands ran cleanly. Highlights below; full tables in
the per-section files at `data/eval/initial_run_2026-05-19/`.

### Q3 — summary (WR by opp_type, with Wilson CIs)

```
 opp_player_name  n  wins  losses  win_rate   ci_low   ci_high
  max_base_power  90    10      80    0.111   0.0615    0.193
simple_heuristic  90    12      78    0.133   0.0779    0.219
       vgc_bench  90     4      86    0.044   0.0174    0.109
      max_damage  90    24      66    0.267   0.1862    0.366
```

### Q1 — agent_team breakdown (sorted worst-first)

| agent_team_hash | opp_player_name | n | WR |
|---|---|---|---|
| 158f8a2cd230 | vgc_bench | 30 | 0.00 |
| 50c69f75599d | vgc_bench | 30 | 0.00 |
| 158f8a2cd230 | max_damage | 30 | 0.03 |
| 158f8a2cd230 | max_base_power | 30 | 0.03 |
| 50c69f75599d | simple_heuristic | 30 | 0.07 |
| … | … | … | … |
| 50c69f75599d | max_damage | 30 | 0.40 |

`158f8a2cd230` is the worst-performing agent team in this run; it
goes 0-30 against `vgc_bench`. `29fd1bdc1602` is the best (40% WR vs
max_damage). The diagnostic value is real even at this scale.

### Q2 — opp_team breakdown (sorted worst-first)

| opp_team_hash | opp_player_name | n | WR |
|---|---|---|---|
| da6cea47f462 | vgc_bench | 30 | 0.00 |
| 2f253bc0634d | vgc_bench | 30 | 0.03 |
| f51186a1be48 | simple_heuristic | 30 | 0.07 |
| … | … | … | … |

`da6cea47f462` is the toughest opponent team for our agent.

### Q5 — short-loss patterns (battles ≤ 5 turns)

Top over-represented opp_team matchups (over_representation > 1
means "this team beats us faster than average"):

| opp_team_hash | opp_player_name | n_short_losses | over_representation |
|---|---|---|---|
| da6cea47f462 | max_base_power | 21/27 | 1.87× |
| da6cea47f462 | simple_heuristic | 17/23 | 1.78× |
| f51186a1be48 | vgc_bench | 18/27 | 1.60× |

Action distribution highlight: `/choose move 4, move 1 -1` is +4.6σ
over-represented in short losses, suggesting the model picks a
specific bad doubles combination when collapsing fast. Worth manual
inspection of the saved replays.

### Q6 — confidence vs heuristic_adv

```
         bucket  n_turns  mean_entropy  mean_value_predicted  mean_heuristic_adv
(-0.917, 0.447]      775         2.07               -0.83                  0.18
 (0.447, 0.726]      774         2.43               -0.74                  0.59
   (0.726, 1.0]     1549         3.48               -0.73                  0.93
```

**Critical finding**: in "poor situations" (`heuristic_adv < -0.3`,
n=34) mean entropy is **1.90** vs **2.86** overall — the model is
*more confident* when losing. This is the inverse of healthy
calibration. Combined with the consistently-negative
`mean_value_predicted` across all buckets (-0.83 → -0.73), the value
head is essentially predicting "we're losing" almost regardless of
the actual position. zany-cloud-67's value head has not yet learned
to read board state.

Worst negative swings (most happen with `heuristic_adv_before` ≈ 1.0
and `value_at_t_minus_k` ≈ -0.95 — heuristic says we're crushing,
model says we're losing, then we actually do collapse). This is
diagnostic gold: the model is value-blind to advantageous positions.

### Q7 — value-head calibration

```
ECE = 0.0682
 bin_lower  n_turns  mean_predicted  observed_win_rate
       0.0     2251           0.025              0.069
       0.1      332           0.139              0.069
       0.2      123           0.244              0.171
       0.3       61           0.346              0.295
       0.4       45           0.441              0.289
       0.5       60           0.551              0.300
       0.6       82           0.657              0.427
       0.7       66           0.747              0.364
       0.8       74           0.842              0.703
       0.9        4           0.909              0.500
```

Two patterns: (i) **the model is wildly biased toward predicting
loss** — 72% of turns (2251/3098) land in the 0.0–0.1 bin, where
predicted=0.03 vs actual=0.07. The value head doesn't even use the
upper half of its support in 70%+ of cases. (ii) where it does
predict confidence, it's over-confident — at predicted 0.66 actual
is 0.43, at predicted 0.75 actual is 0.36. ECE 0.068 is moderate;
the dominant signal is the bias toward predicting loss.

### Q9 — value vs ensemble disagreement

**Q9a** (top 20 persistent disagreement) — all top battles have
`mean_abs_diff > 1.3`. Inspection: nearly every top battle is an
*outcome=1.0 win* where the model's `value_predicted ≈ -0.95`
throughout (it thought it was losing badly) but the ensemble (which
blends heuristic + final outcome) sat at +0.5 to +1.0. Same finding
as Q6/Q7: value head says "loss" even mid-victory.

**Q9b** (agree-then-diverge) — **empty**. With ~8.5 turns/battle
and `early_window=5`, very few battles have enough late turns to
detect a split. At realistic ~15 turn battles the detector would
have signal; at our 5-loss bias it's data-starved. Not a bug —
just a sample-size limitation.

### Q8 — saved games (5 categories × 3 each = up to 15 games)

| Category | Saved | Notes |
|---|---|---|
| short_loss | 3 ✓ | battles ≤ 5 turns, loss |
| team_we_lose_with | 3 ✓ | agent_team overall WR < 25% |
| team_we_lose_to | 3 ✓ | opp_team's WR over us > 75% |
| value_vs_ensemble_persistent | 3 ✓ | top Q9a candidates |
| value_vs_ensemble_diverge | 0 | Q9b pool empty (above) |

Files: `data/eval/initial_run_2026-05-19/saved_games/<category>/<battle_id>.log.gz`
(Showdown replay) plus matching `.json` sidecar with per-turn
records. 12/15 games saved — the 3 missing are from Q9b having no
candidates. Categories all populate at realistic-scale runs.

## Data location

```
data/eval/initial_run_2026-05-19/
├── manifest.json                       # eval_run_id, ckpt, schedule, timestamps
├── battles_worker_<i>_<call_id>.parquet # 7 shards, 360 rows total
├── turns_worker_<i>_<call_id>.parquet   # 7 shards, 3098 rows total
├── replays/
│   └── p<port>_battle-<format>-<id>.log.gz   # 360 files, ~3 KB each
├── saved_games/
│   ├── short_loss/                     # 3 .log.gz + 3 .json
│   ├── team_we_lose_with/              # 3 + 3
│   ├── team_we_lose_to/                # 3 + 3
│   ├── value_vs_ensemble_persistent/   # 3 + 3
│   └── value_vs_ensemble_diverge/      # empty
└── result_<opp>.json                   # 4 per-opp_type summary JSONs
```

Total: ~1.9 MB on disk (extrapolating to the full 42×42×100×4 = 706k
battle schedule: ~720 MB, well within the 1.6 TB free on `/home`).

## Bugs found + fixes (during the run)

Three independent data-integrity bugs surfaced. None were caught by
unit tests at smoke scale (8 battles) — they all need multi-worker
multi-server multi-call execution to reproduce.

### Bug 1: Embedder `calculate_damage` assertion hangs the worker

* **Symptom**: `AssertionError: defender stats not defined` (and its
  `attacker` twin) raised inside `poke_env.calc.calculate_damage`
  during `SimpleModelPlayer._select_action` → `Embedder.embed` →
  `generate_feature_engineered_features`. The exception propagates
  up an async task and is logged as
  "Task exception was never retrieved", but the player never returns
  a choice, so the battle hangs. With 4 workers all hitting this on
  different battles, the run gets stuck at near-0% CPU.
* **Root cause**: the embedder unconditionally calls
  `calculate_damage(...)` for feature-engineered damage features
  (lines ~505 and ~584 of `etl/embedder.py`). `calculate_damage`
  asserts both sides have every stat defined; mid-battle, the
  opponent's stats can be partially `None` until they reveal a
  move/item that pins down the spread.
* **Why pre-existing checks failed**: an earlier guard
  (`opp_mon.stats is None or opp_mon.stats["hp"] is None`)
  recomputes stats only when *the whole stats dict* is None or HP
  specifically is missing. It misses the case where stats is a dict
  with HP set but Atk = None.
* **Fix** (commits to follow): two checks in `embedder.py`:
  1. `_stats_fully_defined(mon)` — every `stats.value()` is `int`/`float`.
  2. `_calc_damage_args_safe(battle, attacker_id, defender_id)` —
     mirrors `calculate_damage`'s internal `battle.get_pokemon(id)`
     lookup and validates the *resolved* objects. Necessary because
     `calculate_damage` re-looks up by id; the `opp_mon` we hold in
     the embedder loop is not always the same object.
  Falls through to the existing `dmg = (-1, -1)` sentinel that the
  rest of the code already handles.

### Bug 2: Parquet shard names collide across opp_type calls

* **Symptom**: After four successive `evaluate.py` invocations into
  the same `--collect-trajectories` dir, the `battles.parquet` glob
  loaded only 150 / 360 rows, with only `max_base_power` and
  `vgc_bench` (the last two opp_types) represented. The earlier
  opp_types' rows were silently lost.
* **Root cause**: `TrajectoryCollector.flush` wrote shards as
  `battles_worker_<i>.parquet`. Each subsequent call's worker 0
  overwrote the previous call's worker 0 shard.
* **Fix**: `TrajectoryCollector` accepts a `call_id` (the per-call
  `run_tag`, a 4-hex-char nonce already generated in `evaluate.py`).
  Shards are `battles_worker_<i>_<call_id>.parquet`. The
  `read_battles` / `read_turns` glob remains `battles_worker_*` so
  it transparently catches both old and new naming.

### Bug 3: Showdown's per-server battle counter collides across workers

* **Symptom**: After fixing Bug 2, the run produced 360 rows but
  only 269 unique `battle_id`s. Replays on disk: 269 (down from the
  expected 360 — file collisions overwrote ~91 logs).
* **Root cause**: Each Showdown server (port) maintains its own
  battle counter. Two workers on two servers can each play their
  own battle and get labels `battle-gen9vgc2024regg-1511502` — the
  *same* tag for *different* battles. The collector's idempotency
  set (`_recorded_battle_tags`) is per-instance; two workers'
  collectors each record their own row independently. Replay file
  paths (`replays/<battle_tag>.log.gz`) are content-addressed by
  the colliding tag, so one overwrites the other.
* **Fix**: `TrajectoryCollector` accepts a `battle_id_prefix`
  (conventionally `f"p{port}_"`) prepended to every `battle.battle_tag`
  before storing or filing. The canonical id used as BattleRecord's
  `battle_id`, TurnRecord's `battle_id`, the dedup-set key, and the
  replay filename is now globally unique across servers. `evaluate.py`
  builds the prefix from the worker's `server_url`.

## Caveats and learnings

* **Sample size**: 3×3 teams × 10 battles = 90 per opp_type is too
  small for many of the analysis questions to surface confidence-
  worthy results. Wilson CIs on per-team WR are wide (e.g. 0/30
  gives CI [0.000, 0.114]). Q9b especially needs more turns/battle.
  Full schedule (42×42 × 100 = 176,400 per opp_type) will fix this
  but takes ~24h.
* **Replay rate 1.0**: produced 360 / 360 replays at 1.9 MB total
  (gzipped ~5 KB each on average — the storage projection in commit
  `6d72b1f` of "~3 GB at full schedule" still holds).
* **Server launch**: `--launch-servers` has a race where the eval
  starts connecting before Showdown's TCP listen actually accepts.
  Manual server launch + 8s wait works reliably; the CLI's built-in
  startup wait (`time.sleep(2)`) is too short. Worth a follow-up:
  poll for port-listening readiness in `launch_showdown_servers`.
* **Cell iteration with vgc_bench**: external subprocess can only
  serve one worker at a time, so `--workers 1 --num-servers 1` is
  the working topology for vgc_bench. The in-process opp_types use
  `--workers 4 --num-servers 4`.
* **GPU device**: when training is paused, `--device cuda` works
  fine and the per-worker SimpleModelPlayer instances each load
  their own copy of the model. With training running, fall back to
  `--device cpu` to avoid OOM.
* **balmy-cloud-70 active training note**: the may16 RL training
  was *not* running when this evaluation completed. If running it
  again while training is live, use `--device cpu`, ports 8200+
  (training uses 8000–8003), and a different
  `--vgc-bench-checkpoint-path` if the training's own external
  vgc_bench is busy on `VGCBENCH_8000`.

## What works, what doesn't (full pipeline coverage)

| Subcommand | Status | Notes |
|---|---|---|
| `summary` (Q3) | ✓ working | Wilson CIs computed correctly |
| `agent_team` (Q1) | ✓ working | sorted worst-first per CLI default |
| `opp_team` (Q2) | ✓ working | same |
| `short_loss` (Q5) | ✓ working | opp_team chi-square + action dist |
| `confidence` (Q6) | ✓ working | quartiles + swing detection + poor-situation summary |
| `value_calibration` (Q7) | ✓ working | ECE = 0.068 |
| `value_ensemble` (Q9) | ✓ Q9a / data-starved Q9b | Q9b needs longer battles |
| `save_games` (Q8) | ✓ working | 12/15 saved (Q9b empty) |
| `report` | TODO | not implemented (each subcommand is individually runnable) |

## Reproducing this run

```bash
# 1. Launch 4 Showdown servers on 8200-8203 (wait 8s for binding)
for port in 8200 8201 8202 8203; do
    (cd /home/cayman/Repositories/pokemon-showdown && \
     nohup node pokemon-showdown start --no-security \
     --no-battle-retention --port $port \
     > /tmp/showdown-eval-$port.log 2>&1 &)
done
sleep 8

# 2. Prepare team subsets (these specific files were used here):
mkdir -p /tmp/eval_run_teams_agent /tmp/eval_run_teams_opp
for t in 38dessert.txt 8989takoyaki.txt 9thplacenaic.txt; do
    cp data/teams/gen9vgc2024regg/constrained/$t /tmp/eval_run_teams_agent/
done
for t in bfi24championteam.txt danielyusworlds2024top32team.txt frederiknielsensstockholmregionalstop8team.txt; do
    cp data/teams/gen9vgc2024regg/constrained/$t /tmp/eval_run_teams_opp/
done

# 3. Run all 4 opp_types into the same RUN_DIR:
RUN_DIR=data/eval/initial_run_2026-05-19
rm -rf $RUN_DIR
for OPP in simple_heuristic max_damage max_base_power; do
    python -m elitefurretai.rl.analyze.evaluate \
        --player1 data/models/rl/zany-cloud-67/main_model_step_2204.pt \
        --player2 $OPP \
        --team1 /tmp/eval_run_teams_agent --team2 /tmp/eval_run_teams_opp \
        --cell-iteration --battles 10 \
        --workers 4 --num-servers 4 --start-port 8200 \
        --device cuda --battle-format gen9vgc2024regg \
        --collect-trajectories $RUN_DIR \
        --eval-run-id initial_run_001 \
        --output $RUN_DIR/result_${OPP}.json
done
python -m elitefurretai.rl.analyze.evaluate \
    --player1 data/models/rl/zany-cloud-67/main_model_step_2204.pt \
    --player2 vgc_bench \
    --team1 /tmp/eval_run_teams_agent --team2 /tmp/eval_run_teams_opp \
    --cell-iteration --battles 10 \
    --workers 1 --num-servers 1 --start-port 8200 \
    --device cuda --battle-format gen9vgc2024regg \
    --collect-trajectories $RUN_DIR \
    --eval-run-id initial_run_001 \
    --output $RUN_DIR/result_vgc_bench.json

# 4. Run all analyses
for SUB in summary agent_team opp_team short_loss confidence \
           value_calibration value_ensemble save_games; do
    python -m elitefurretai.rl.analyze.eval_analysis $RUN_DIR $SUB
done
```

## Scaling to the full schedule

The 360-battle initial run completed in ~2 minutes. Extrapolating:

* 42×42 × 100 battles per opp_type = 176,400 battles
* × 4 opp_types = 705,600 battles total
* At ~3 b/s sustained (the smoke-run number), ≈ 65 hours wall-clock
* At ~10 b/s (achievable with parallel workers on a quiet host), ≈
  20 hours

Recommended sequencing: run one opp_type at a time, sequentially.
Each opp_type's call writes its own `<call_id>.parquet` shards into
the same RUN_DIR, so a mid-run failure only loses the in-flight
opp_type. The `report` subcommand is the only ergonomic piece still
missing — Q1–Q9 each work individually right now.
