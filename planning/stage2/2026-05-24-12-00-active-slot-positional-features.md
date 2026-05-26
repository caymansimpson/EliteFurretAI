# 2026-05-24 12:00 — Active-slot positional features (Option A + opponent mirror)

## Context

The action diagnostics work in [2026-05-23-12-00](
2026-05-23-12-00-action-diagnostics-slot-decomposition.md) surfaced a
fundamental representation gap: the embedder stores `MON:i:active`
(which 2 of 6 mons are on the field) but no feature distinguishing
**which active position** each mon occupies. The MDBO action references
moves by active position (slot 0 = `battle.active_pokemon[0]`, slot 1
= `battle.active_pokemon[1]`), but the state tensor doesn't carry that
mapping. The opponent side has the symmetric gap with target
resolution.

This is fundamental info the BC and RL models need to predict
well — currently they have to learn the mapping implicitly from
conventions (e.g. `MON:0` is usually the lead at battle start) and
indirect signals (e.g. `sent` + `active`). Mid-battle after switches,
those conventions break.

The user picked **Option A + opponent-side mirror**: add 4 boolean
features per timestep (`MON:i:active_slot_0`, `MON:i:active_slot_1`
for i in 0..5, mirrored on `OPP_MON:i:`). Smallest change that
captures the positional info directly; preserves cross-time
identity (MON:i stays bound to a fixed teampreview pick); cheap in
embedding size (+24 features, +0.5% on the 5,032-feature raw set).

## Plan

### Code changes — embedder.py only

Two insertions in `src/elitefurretai/etl/embedder.py`:

1. **Player side** — `generate_pokemon_features` (~line 1006, right
   after `emb[prefix + "active"]`):

   ```python
   # Distinguish which active slot this mon is in (if any).
   # MDBO slot k = battle.active_pokemon[k], so this lets the model
   # disambiguate "move 2, move 1" → which mon does which.
   active_pos_0_name = active_names[0] if len(active_names) > 0 else None
   active_pos_1_name = active_names[1] if len(active_names) > 1 else None
   emb[prefix + "active_slot_0"] = (
       -1 if mon is None
       else int(active_pos_0_name is not None and mon.name == active_pos_0_name)
   )
   emb[prefix + "active_slot_1"] = (
       -1 if mon is None
       else int(active_pos_1_name is not None and mon.name == active_pos_1_name)
   )
   ```

2. **Opponent side** — `generate_opponent_pokemon_features` (~line
   1163, right after `emb[prefix + "active"]`):

   ```python
   opp_active_names = list(
       map(lambda x: x.name if x else None, battle.opponent_active_pokemon)
   )
   opp_pos_0_name = opp_active_names[0] if len(opp_active_names) > 0 else None
   opp_pos_1_name = opp_active_names[1] if len(opp_active_names) > 1 else None
   emb[prefix + "active_slot_0"] = (
       -1 if mon is None
       else int(opp_pos_0_name is not None and mon.name == opp_pos_0_name)
   )
   emb[prefix + "active_slot_1"] = (
       -1 if mon is None
       else int(opp_pos_1_name is not None and mon.name == opp_pos_1_name)
   )
   ```

### Why this is safe everywhere downstream

Audited every place that touches feature indices or group shapes:

| Site | What it does | Effect of new features |
|---|---|---|
| `Embedder._compute_grouped_feature_names` | Calls `generate_pokemon_features` / `generate_opponent_pokemon_features` 6× each, sorts within each group | New features automatically picked up, sorted into the MON:i / OPP_MON:i group |
| `Embedder.group_embedding_sizes` | `[pokemon_size]*6 + [opp_size]*6 + ...` | Both `pokemon_embedding_size` and `opponent_pokemon_embedding_size` grow by 2; group sizes update automatically |
| `Embedder.embedding_size` | Computed from `len(self.embed(dummy_battle))` at `__init__` | +24 (= 12 player + 12 opponent) |
| All training scripts (`train.py`, `fine_tune.py`, `train_sweep.py`, `train_non_traj.py`) | Set `state_input_dim = embedder.embedding_size` | New, larger input dim flows through to model construction |
| `TransformerThreeHeadedModel` | Uses `embedder.embedding_size` and `embedder.group_embedding_sizes` | Constructs encoders to match new sizes |
| Diagnostic / analyzer scripts | Look up feature indices by string name | New features findable by name (e.g. `MON:0:active_slot_0`) — already pick up the existing positional features the same way |
| `MON:{j}:force_switch` lookups (`train.py:509`, `fine_tune.py:239`, `train_sweep.py:176`, `train_non_traj.py:149`, `reasonable_filter_on_action_model.py:413`) | All loop over the existing per-mon force_switch features | Untouched by this change |
| Tests | `test_embedder.py` checks `embedding_size == sum(group_embedding_sizes)` (invariant) and `pokemon_embedding_size` consistency across raw/omni instances | Still pass — invariants hold |

### Things that will break (by design)

- **All existing checkpoints become incompatible** for inference and
  fine-tuning. Their stored input dim (5,032 for raw, larger for full)
  no longer matches the embedder's new embedding_size. This is
  intentional — we're retraining.
- **All preprocessed `.pt.zst` trajectory files** in
  `data/battles/regc_final_v4/` are obsolete. The states encoded in
  them don't have the new features. We will regenerate into
  `regc_final_v5/`.

### `regc_final_v5` preprocessing parameters

Inferred from `regc_final_v4/`:

| Param | Value | Source |
|---|---|---|
| `mode` | `trajectories` | user spec |
| `chunk_size` | **512** | regc_final_v4 train files have 512 trajectories each (last partial = 256) |
| `train_pct` / `test_pct` / `val_pct` | **0.9 / 0.05 / 0.05** | default; matches the 1523/86/86 file counts in v4 (95% / 2.5% / 2.5% of files = ~5%/2.5% of total because train chunks are full) |
| `seed` | 21 | default; reproduces same split if data list is identical |
| `batch_size` | 32 | default |
| `num_workers` | **7** | user spec |
| `steps_per_battle` | 40 | default |
| `augment_teampreview` | True | default + matters for memorization fix |
| `output_folder` | `data/battles/regc_final_v5` | user spec |
| `battle_filepath` | `data/battles/supervised_battle_files_w_commander.json` | user spec |

Command:

```bash
python src/elitefurretai/etl/process_training_data.py \
    data/battles/supervised_battle_files_w_commander.json \
    data/battles/regc_final_v5 \
    --mode trajectories \
    --chunk-size 512 \
    --num-workers 7
```

### Retraining plan

Per the user's "retrain cool-bee and the current run" — interpreted as
**retrain may24 (the active in-flight config) and produce a fine-tuned
descendant** (the cool-bee-85-finetune analogue). may24's config notes
that it was deliberately set up as the head-to-head replacement for
the cool-bee-85 family, sharing the same architecture knobs, so a
single `may24.yaml` retrain + `may22_finetune.yaml` fine-tune captures
both runs cleanly with the new data.

Sequential chain (preprocess → train → fine-tune) wrapped in a single
`nohup` script:

```bash
set -e
# 1. Preprocess raw battles → regc_final_v5
python src/elitefurretai/etl/process_training_data.py \
    data/battles/supervised_battle_files_w_commander.json \
    data/battles/regc_final_v5 \
    --mode trajectories --chunk-size 512 --num-workers 7

# 2. Retrain may24 from scratch on v5 (the new BC reference)
python src/elitefurretai/supervised/train.py \
    --config src/elitefurretai/supervised/configs/may24.yaml \
    --save-best data/battles/regc_final_v5/

# 3. Fine-tune the result (cool-bee-finetune analogue)
#    The train.py step saves <run_name>_best.pt in data/models/supervised/
#    fine_tune.py takes <data_dir> <model_path> <wandb_run_name> --config-override
python src/elitefurretai/supervised/fine_tune.py \
    data/battles/regc_final_v5/ \
    data/models/supervised/<may24_run_name>_best.pt \
    <may24_run_name>-finetune \
    --config-override src/elitefurretai/supervised/configs/may22_finetune.yaml \
    --save-best
```

The `<may24_run_name>` is wandb-assigned and only known after step 2
finishes. The orchestration script will discover it by reading the
saved checkpoint path produced by step 2.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stopping the in-flight may24 loses uncommitted progress | The user explicitly directed this; the new-feature retrain supersedes it |
| Preprocessing takes hours and competes with anything else on the box | Run niced, single-process — pre-stop the GPU training so CPU is free |
| Disk space — regc_final_v5 will be similar size to v4 (~tens of GB) | regc_final_v4 stays in place; user can rm later if confident |
| New `active_slot_*` features have integer values {-1, 0, 1} just like `active` — model treats them as ordinary binary-ish inputs | None — matches existing convention |
| Augmentation seed isn't preserved across v4 → v5; specific battles may land in different splits | Default seed=21 will be used. If the user wants identical splits, we'd need to record v4's exact seed — skipped here since the goal is fresh data |
| Test/val data leakage — augment_teampreview shuffles team order; we don't want it on test/val | Existing `process_training_data.py` handles this correctly (verified — `augment_teampreview` is a constructor knob on `BattleDataset`, called from `trajectories()`). Check it before kickoff. |

## Verification

After implementation:

1. `ruff check src` + `ruff format src --check` + `pyright src/elitefurretai/etl/embedder.py` — clean.
2. `pytest unit_tests/etl/test_embedder.py unit_tests/etl/test_embedder_improvements.py -q` — passes.
3. Smoke-test by instantiating `Embedder(gen=9, feature_set='raw')` and confirming:
   - `'MON:0:active_slot_0'` and `'MON:0:active_slot_1'` exist in `feature_names`
   - Same for `OPP_MON:0:active_slot_0/1`
   - `embedding_size == sum(group_embedding_sizes)`
   - `pokemon_embedding_size` grew by exactly 2
   - `opponent_pokemon_embedding_size` grew by exactly 2

## Planned Next Steps

1. Write planning doc (this) ✓
2. Edit `embedder.py` (player + opponent insertions)
3. Quality gates + smoke test + targeted unit tests
4. Stop in-flight may24 training (PID 19056) cleanly
5. Launch the chained preprocess → train → fine-tune script via `nohup` with `setsid`
6. Monitor via log files; report headline diagnostics on the new
   model once steady-state checkpoints exist

After the new BC model exists:
- Re-run [action_model_diagnostics.py](
  ../../src/elitefurretai/supervised/analyze/action_model_diagnostics.py)
  on it. Expectation: `ambiguous_pct` drops to near-zero on both
  category and move-name analyses, since the positional bit is now
  in the state tensor and the diagnostic can resolve MDBO slot →
  MON:i unambiguously by reading `MON:i:active_slot_0/1`.
- Direct apples-to-apples comparison vs cool-bee-85-finetune on the
  same `regc_final_v5/test/` split — expected gains driven by:
  (a) post-switch turn accuracy (the previously-ambiguous cases)
  (b) MOVE Top-1 (currently 16% on cool-bee).

## Updates

* 2026-05-24 12:00 — plan written, about to implement.
* 2026-05-24 00:28 — **implementation shipped**:
  * Embedder edits landed in
    [src/elitefurretai/etl/embedder.py](
    ../../src/elitefurretai/etl/embedder.py) — 4 new feature insertions
    (player active_slot_0/1 + opponent active_slot_0/1).
  * Quality gates clean (`ruff`, `pyright` 0 errors).
  * Embedder unit tests: 80 / 80 pass.
  * Embedding-size change: **raw 5,032 → 5,056** (+24), full 5,222 → 5,246.
  * `pokemon_embedding_size` and `opponent_pokemon_embedding_size` each
    grew by 2 (player 418→420, opponent 410→412).
  * `MON:0:active_slot_0`, `MON:0:active_slot_1`, etc., all present in
    `feature_names`.
* 2026-05-24 00:28 — **may24 in-flight training stopped** (PID 19056, SIGTERM).
  GPU freed (24,326 MiB free → confirmed).
* 2026-05-24 00:28 — **pipeline launched** via
  `setsid nohup bash scripts/v5_pipeline.sh > /tmp/v5_pipeline/orchestrator.log 2>&1 &`.
  Stage 1 (preprocess) is running. Confirmed:
  * 432,646 battle files loaded, split 90/5/5 (seed=21):
    train 389,381 / val 21,632 / test 21,632.
  * Chunk size 512, num_workers 7, mode trajectories.
* Expected total runtime: preprocess ~few hours + train ~6–12h + fine-tune ~3–4h.
* Logs at `/tmp/v5_pipeline/orchestrator.log` (top-level), with per-stage
  detail at `01_preprocess.log`, `02_train.log`, `03_finetune.log`.

### Existing checkpoints made obsolete

All `data/models/supervised/*.pt` checkpoints trained on
`regc_final_v4/` have input dim 5,032 (or full=5,222). They cannot
load with the new embedder (5,056 raw / 5,246 full). Affected
files include but are not limited to:

* `cool-bee-85-finetune_best.pt` (BC reference until now)
* `curious-darkness-77_best.pt`
* `classic-firebrand-90_best.pt`
* `sandy-firefly-98_best.pt`
* `lyric-feather-99_best.pt`
* `balmy-cloud-70_best.pt`

These files remain on disk (no destructive action taken). After the
pipeline produces new BC checkpoints on v5 data, they can be
archived or removed at the user's discretion.
