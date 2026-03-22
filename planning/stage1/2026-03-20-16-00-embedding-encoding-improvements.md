# 2026-03-20: Embedding Encoding Improvements (NumberBank + EntityID)

## Context

Stage I — Supervised Baseline Models. Before the next round of BC retraining, we audited every feature in the `Embedder` class against the `NumberBankEncoder` and `EntityIDEncoder` to find features that would benefit from learned discrete embeddings vs raw scalars.

Training is using the `full` feature set (~9,223 features) with `FlexibleThreeHeadedModel` + `GroupedFeatureEncoder`. The encoders already handle: `current_hp_fraction`, `PERC_HP_LEFT`, `STAT:*`, `STAT_MIN/MAX:*`, `base_power` (NumberBank) and `ability_id`, `item_id` (EntityID).

## Before State

- **9,223 features** in the `full` embedding
- ~3,500 features are extremely sparse one-hot encodings on per-move metadata (volatileStatus, sideCondition, field, weather OHE per move) — >95% zeros
- `EST_DAMAGE_MIN/MAX` (288 features) are raw floats 0–600, semantically identical to stats but without NumberBank
- No species or move identity features — model must reconstruct identity from stat/type/ability combinations
- `turn`, `p1rating`, `p2rating` passed as raw scalars
- `level` is always 50 in VGC — zero information
- `TYPE_MATCHUP` features are strictly redundant with `EST_DAMAGE` features

## Problem

1. **Missing identity features**: No `species_id` or `move_id`. The model must learn to reconstruct "this is Incineroar" from types+stats+ability, which wastes capacity.
2. **Raw damage values**: `EST_DAMAGE_MIN/MAX` are the richest continuous features but treated as raw floats. NumberBank discretization would let the model learn qualitative damage thresholds.
3. **Sparse per-move OHE**: Move volatile status (51 per move), side conditions (12), fields (6), weather (4) are >95% zeros. With `move_id` EntityID, the model learns these implicitly.
4. **Uninformative features**: `level` is always 50; `TYPE_MATCHUP` is redundant with damage calcs.
5. **Raw scalars for turn/ratings**: Game phase and skill level have meaningful discrete breakpoints that NumberBank captures.

## Solution

### Additions (5 new feature types)

| Feature | Encoding | Details |
|---------|----------|---------|
| `species_id` | EntityID | embed_dim=32, vocab≈1300, added to player+opponent pokemon features |
| `move_id` | EntityID | embed_dim=16, vocab≈900, added to move features |
| `EST_DAMAGE_MIN/MAX` | NumberBank | Shares stat bank (0–600 range, same discretization) |
| `turn` | NumberBank | New turn_bank, bins=40, range 0–40 |
| `p1rating`/`p2rating` | NumberBank | New rating_bank, bins=100, range 0–2000 |

### Pruning (comment out, not delete)

| Feature Group | Features Removed | Reason |
|--------------|-----------------|--------|
| Move `EFFECT:*` (volatile status OHE) | ~2,448 | >95% zeros, captured by `move_id` |
| Move `SC:*` (side condition OHE) | ~576 | >99% zeros, captured by `move_id` |
| Move `FIELD:*` (field OHE) | ~288 | >99% zeros, captured by `move_id` |
| Move `WEATHER:*` (weather OHE) | ~192 | >99% zeros, captured by `move_id` |
| `level` | 12 | Always 50 in VGC |
| `TYPE_MATCHUP:*` | 36 | Redundant with `EST_DAMAGE_*` |
| **Total removed** | **~3,552** | |

### Net result
- Embedding tensor: **~9,223 → ~5,721 features** (38% reduction in storage/preprocessing)
- New EntityID embeddings expand inside the model forward pass (no storage impact)
- NumberBank for damage uses embed_dim=4 to keep expansion manageable

## Reasoning

- **Species identity** is the single most informative signal for Pokemon strategy. Every experienced player's decision-making starts with "what species is this?" — the model should too.
- **Move identity** captures move-specific mechanics (Fake Out turn restrictions, Sucker Punch failure conditions, pivot sequencing) that can't be represented by the current feature set.
- **Damage NumberBank** lets the model learn qualitative damage thresholds (2HKO vs 3HKO) rather than interpolating from raw floats.
- **Pruning sparse OHE** reduces preprocessing time and storage by 38% while the information is fully captured by move_id embeddings.
- **Rating NumberBank** enables skill-tier-aware imitation during supervised learning.
- **Turn NumberBank** enables game-phase-aware strategy (early positioning vs late closing).

## Planned Implementation

### Files to modify:
1. **`src/elitefurretai/etl/embedder.py`**: Add `species_id`, `move_id`, `SPECIES_TO_ID`, `MOVE_TO_ID` mappings. Comment out pruned features. Export new constants.
2. **`src/elitefurretai/etl/__init__.py`**: Export new constants (`NUM_SPECIES`, `NUM_MOVES`, `SPECIES_TO_ID`, `MOVE_TO_ID`).
3. **`src/elitefurretai/supervised/model_archs.py`**: Add species/move to `EntityIDEncoder.ID_PATTERNS`. Add damage/turn/rating pattern banks to `NumberBankEncoder`. Add `num_species`/`num_moves` params to constructors.
4. **Downstream consumers** (train.py, behavior_clone_player.py, model_io.py, opponents.py, fine_tune.py, diagnostics): Pass new `num_species`/`num_moves` kwargs.
5. **Tests**: Update test_embedder.py, test_model_archs.py, add dry-run battle test.
6. **Reprocess data**: Run `process_training_data.py` with new embedder.

### Not backwards compatible
Old preprocessed `.pt.zst` data is incompatible — embedding size changes. Must reprocess.

## Updates

### Implementation Complete (2026-03-20)

**All changes implemented and tested. Quality gates pass (ruff, pyright, pytest).**

#### Files modified:

1. **`src/elitefurretai/etl/embedder.py`**:
   - Added `build_species_to_id()` and `build_move_to_id()` functions
   - Module-level constants: `SPECIES_TO_ID`, `MOVE_TO_ID`, `NUM_SPECIES`, `NUM_MOVES`
   - Instance properties: `species_id`, `move_id`, `num_species`, `num_moves`
   - `species_id` added to `generate_pokemon_features()` and `generate_opponent_pokemon_features()`
   - `move_id` added to `generate_move_features()` and `_generate_null_move_features()`
   - Commented out (not deleted): move EFFECT OHE, SC OHE, FIELD OHE, WEATHER OHE, `level`, `TYPE_MATCHUP`

2. **`src/elitefurretai/etl/__init__.py`**: Exported `SPECIES_TO_ID`, `MOVE_TO_ID`, `NUM_SPECIES`, `NUM_MOVES`

3. **`src/elitefurretai/supervised/model_archs.py`**:
   - `NumberBankEncoder`: Added `DAMAGE_PATTERNS`, `TURN_PATTERNS`, `RATING_PATTERNS`; new banks (`damage_bank`, `turn_bank`, `rating_bank`) with per-bank embed_dims; updated `_classify_feature`, `_embed_dim_for_bank`, `_embed_dim_for_entry`, `_get_bank`
   - `EntityIDEncoder`: Added `species_id` → `species` and `move_id` → `move` to `ID_PATTERNS`; new `num_species`, `num_moves`, `species_embed_dim`, `move_embed_dim` params; new `species_emb`, `move_emb` nn.Embedding tables; updated `_embed_dim_for`, `_get_embedding`
   - `GroupedFeatureEncoder`: New params forwarded to sub-encoders
   - `FlexibleThreeHeadedModel`: New params forwarded to `GroupedFeatureEncoder`
   - `TransformerThreeHeadedModel`: Same as above

4. **Downstream consumers (7 files)**: Added `NUM_SPECIES`/`NUM_MOVES` imports and `num_species`/`num_moves` constructor kwargs to:
   - `supervised/train.py`
   - `supervised/behavior_clone_player.py`
   - `supervised/fine_tune.py`
   - `supervised/analyze/action_model_diagnostics.py`
   - `supervised/analyze/win_model_diagnostics.py`
   - `rl/model_io.py` (also added damage/turn/rating bank params)
   - `rl/opponents.py`

5. **Tests updated**:
   - `unit_tests/etl/test_embedder.py`: Updated assertions for pruned features; added `test_species_id_feature`, `test_move_id_feature`, `test_species_to_id_mapping`, `test_move_to_id_mapping`, `test_pruned_features_absent`, `test_feature_consistency_across_feature_sets`, `test_embedding_size_consistency`, `test_dry_run_battle_embedding`
   - `unit_tests/etl/test_embedder_improvements.py`: Updated `EntityIDEncoder` and `GroupedFeatureEncoder` test classes for new entity types and constructor API

#### Feature size post-change:
- **RAW embedding**: reduced from ~9,223 to ~5,721 features (38% reduction)
- **NUM_SPECIES**: ~1,300 (Gen 9 Pokedex + 1 for unknown)
- **NUM_MOVES**: ~900 (Gen 9 moves + 1 for unknown)

#### Next step:
- Reprocess training data with `process_training_data.py` using the updated embedder
