# Model Utilities
This folder contains several useful utilities that should be helpful for building supervised or reinforcement learning models.

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Pokemon Battle played on Showdown. This class can be used to recreate `Battle` and `DoubleBattle` objects turn-by-turn. It is compatible with `Battle` objects collected by an agent from either self-play (where we have omniscience) or on Showdown against other agents.
2. `BattleIterator` -- this is the main class that you will find useful if you have records of offline battles. Given a `BattleData` object, it iterates through the logs of a battle to reconstruct and populate a battle object, used like an iterator. It supports commands to iterate to the agent's next required input so that you can recreate a `Battle` object at any important part of the battle w/out a request (which it can also roughly recreate).
3. `Embedder` -- converts a `Battle` object into an array of floats for input into a neural networ. You can use three types of featuresets -- a `simple` one for testing, `raw` one that has all informaiton, and a `full` one that includes feature engineered features (e.g. w/ damage calcs). If you want to edit the featureset yourself, it also outputs a `dict` that maps each feature to a name for custom filtering.
4. `MDBO` (in encoder) --  a container for decision-making for model-based agents (short for ModelDoubleBattleOrder). It should be used to translate `int`s to `BattleOrder`s and vice versa for training. `BattleIterator`'s `last_order()` function returns a `ModelBattleOrder` object. This is used to generate training data (actions in BattleDataset).
5. `BattleDataset` -- implementing the pytorch wrapper so you can generate battle data's at random and easily. It uses all the above classes to return fully embedded battle states for each turn, in batches of battles. It is not optimized for performance because it reads the BattleData file from memory for every perspective. This can be fixed by `@cache`ing `__getitem__` in `BattleDataset` and then reading the indices sequentially with `shuffle=False` in `DataLoader`.
6. `BattleDataLoader` -- full on dataloader that abstracts the BattleDataset with optimized parameters (for my Windows machine). Leverages compressed and preprocessed files from `src/elitefurretai/scripts/prepare/process_training_data.py` with ability to change `num_workers`, `files_per_worker` and `prefetch_factor` for balancing I/O, RAM and GPU needs, respectively.

There are also `train_utils` which holds a variety of helper methods (including `analyze` and `evaluate`) that assess model performance, and are flexible enough to support whichever model architecture you have.

### Example Usage:
```python
# Load data
files = ['battledata.json', 'battledata2.json']
dataset = BattleDataset(files)
data_loader = DataLoader(dataset)

# Iterate through batches of battles with data_loader
for metrics in data_loader:
    states = batch["states"]
    actions = batch["actions"]
    action_masks = batch["action_masks"]
    wins = batch["wins"]
    masks = batch["masks"]
    move_orders = batch["move_orders"]
    kos = batch["kos"]
    switches = batch["switches"]

    # Do training things
    pass

```

# Teampreview Data Augmentation Fix

**Date**: November 30, 2024  
**Issue**: Severe data leakage causing 99.9% validation accuracy through memorization

---

## Problem Discovered

Analysis of the teampreview training data revealed **severe data leakage** where the model was memorizing team compositions rather than learning strategic decision-making.

### Key Findings

**Evidence from 20,000 battles**:
- **88.6%** of unique team patterns (specific 4 Pokemon in MON:0-3) deterministically map to exactly one teampreview action
- Only **11.4%** of team patterns showed any strategic variation
- Model achieved **99.9% validation accuracy** through lookup table memorization
- Same team composition always made the same teampreview choice regardless of opponent

**Evidence from 200 battles**:
- **98.7%** determinism rate (even higher on smaller sample)
- Only **5 out of 394** unique team patterns showed variation in actions
- **100%** of unique state embeddings mapped to exactly one action

### Root Cause

Competitive Pokemon players using specific team archetypes consistently bring the same 4 Pokemon and lead with the same pair:
- Rain teams always lead Pelipper + Barraskewda
- Trick Room teams always lead their TR setter + slow attacker  
- Beat-up teams always lead Annihilape + Murkrow

The dataset lacked **counterfactual examples** where the same team makes different teampreview choices against different opponents.

### Impact

1. **No strategic learning**: Model memorizes "Team X → Action Y" without understanding matchups
2. **Poor generalization**: Against novel team compositions, model makes essentially random predictions
3. **Misleading metrics**: 99.9% accuracy doesn't reflect strategic understanding
4. **Wasted capacity**: 198M parameter transformer functioning as a glorified dictionary lookup

---

## Solution Implemented

### Strategy: Team Order Randomization with Label Remapping

**Concept**: Shuffle the Pokemon team order before embedding, then adjust action labels accordingly.

**Implementation** (in `battle_dataset.py`):

1. **Before creating BattleIterator**: Randomly permute `bd.p1_team` and `bd.p2_team` 
2. **Track permutation**: Store mapping like `[2,0,5,1,3,4]` (new position → old position)
3. **Remap input logs**: Update teampreview choices in `bd.input_logs` to reflect new ordering
4. **Convert labels**: Ground truth action is automatically adjusted when iterator parses remapped input logs

**Example**:
```
Original team: [Incineroar, Rillaboom, Urshifu, FlutterMane, Amoonguss, Gholdengo]
Ground truth: /team 1234 (select positions 0,1,2,3) → Action 0

After shuffle [2,3,0,1,4,5]: [Urshifu, FlutterMane, Incineroar, Rillaboom, Amoonguss, Gholdengo]
New input log: /team 3412 (select positions 2,3,0,1 which are original 0,1,2,3) → Action 54
```

### Benefits

1. **Breaks positional memorization**: Model can't learn "always select MON:0-3"
2. **Creates diversity**: Effectively 720x data augmentation per battle (6! permutations sampled)
3. **Forces position-invariant learning**: Model must reason about matchups, not positions
4. **Maintains label correctness**: Actions are properly remapped to match new ordering

### Files Modified

1. **`src/elitefurretai/model_utils/battle_dataset.py`**:
   - `BattleDataset.__getitem__`: Added full augmentation with detailed comments
   - `BattleIteratorDataset.__getitem__`: Added same augmentation for consistency

2. **`src/elitefurretai/scripts/prepare/process_training_data.py`**:
   - Added warning that augmentation happens during preprocessing
   - Old preprocessed data must be regenerated to benefit from fix

---

## Next Steps

### Required Actions

1. **Regenerate all preprocessed data**:
   ```bash
   python src/elitefurretai/scripts/prepare/process_training_data.py \
       <battles.json> \
       data/battles/regc_trajectory_train \
       --mode full --compressed True
   ```

2. **Retrain models** from scratch with new augmented data

3. **Validate fix**: Run `verify_teampreview_leakage.py` on new data to confirm determinism drops below 50%

### Expected Results After Fix

- Teampreview top-1 accuracy should drop to **20-40%** (realistic for human imitation)
- Teampreview top-5 accuracy should be **40-60%** (legitimate strategic variation)
- Model should show **genuine strategic reasoning** about matchups
- Performance should **generalize better** to novel team compositions

### Optional Future Enhancements

If augmentation alone is insufficient, consider:

1. **Counterfactual augmentation**: Generate synthetic examples where same team makes different choices
2. **Opponent-aware augmentation**: Vary teampreview choices based on opponent team composition
3. **Multi-task learning**: Predict "which 4 Pokemon" (unordered set) instead of ordered action

---

## Technical Details

### Label Remapping Algorithm

```python
# Given: original positions [0,1,2,3], permutation [2,0,5,1,3,4]
# Want: new positions that map to same Pokemon

# 1. Compute inverse permutation (old_pos → new_pos)
inverse_perm = [0] * 6
for new_idx, old_idx in enumerate(permutation):
    inverse_perm[old_idx] = new_idx
# Result: [1, 3, 0, 4, 5, 2]

# 2. Map original positions through inverse
new_positions = [inverse_perm[pos] for pos in [0,1,2,3]]
# Result: [1, 3, 0, 4]

# 3. Convert to teampreview string
new_team_choice = "2413"  # (add 1 for 1-indexing)
```

### Code Location

All augmentation logic is in:
- `src/elitefurretai/model_utils/battle_dataset.py` lines 71-162 (BattleDataset)
- `src/elitefurretai/model_utils/battle_dataset.py` lines 399-445 (BattleIteratorDataset)

Search for "TEAMPREVIEW DATA AUGMENTATION" comment block for full explanation.

---

## Analysis Scripts Used

For reproducibility, these scripts were created during investigation:

1. `src/elitefurretai/scripts/analyze/diagnose_teampreview_leakage.py`
   - Checks if states deterministically map to actions
   - Reports percentage of deterministic mappings

2. `src/elitefurretai/scripts/analyze/verify_teampreview_leakage.py`
   - Comprehensive analysis of team pattern → action mappings
   - Shows examples of variation and determinism
   - Confirms the 88.6% determinism finding

3. `src/elitefurretai/scripts/analyze/inspect_leaked_features.py`
   - Inspects which features differ between samples
   - Helped rule out simple positional encoding

Run these on newly preprocessed data to verify the fix worked.

---

## References

- Original issue: 99.9% teampreview validation accuracy in `three_headed_transformer.py` training run
- Analysis conversations: November 30, 2024
- Related: VGC team archetype standardization in competitive Pokemon
