# 2026-02-28: Observation Embedding Improvements

## Context

After implementing the 5 ps-ppo-inspired improvements (temperature annealing, topology-aware optimizer, C51 distributional value head, number bank embeddings, transformer architecture), we compared EliteFurretAI's observation embedding (`Embedder` class) against ps-ppo's observation space to identify further improvements.

BC training is currently running with the `full` feature set (9,223 features) using `FlexibleThreeHeadedModel` (138.8M params, C51 value head). The goal is to identify and plan embedding improvements that can be implemented before the next round of data preprocessing and retraining.

**Key constraint**: All changes are to the `full` feature set only. The `raw` feature set is kept as-is for compatibility/testing. We do not care about backwards compatibility with existing preprocessed data — we'll reprocess.

## Before State

Current `full` feature set (9,223 features) structure:
- **6 own Pokémon** × `pokemon_embedding_size` features each
- **6 opponent Pokémon** × `opponent_pokemon_embedding_size` features each
- **Battle features**: weather (binary), fields (binary), side conditions (binary), format OHE, turn count, ratings
- **Engineered features**: type matchups (6×6 grid), damage calcs (6×6×4 moves × min/max/ko), faint counts, HP percentages, status counts

### What we embed well (keep as-is):
- **Damage calculations**: Explicit min/max damage estimates for every attacker×defender×move combo — ps-ppo doesn't have this
- **Type matchups**: 6×6 grid of offensive type multipliers
- **Stat range inference**: LP-solver-derived speed bounds from SpeedInference — unique to us
- **Rich move metadata**: 50+ features per move (drain, heal, force_switch, boosts, secondary effects, targeting types, volatile statuses)
- **State summaries**: Faint counts, HP percentages, status counts, revealed move counts

### What's missing (the 4 improvements below):

## Problem 1: No Transition Features

**Problem**: The embedding is a pure snapshot of the current state. There's no information about *what happened last turn* — who moved first, whether attacks were critical hits, whether they were super-effective, whether moves missed, etc. This temporal information helps the model reason about:
- Speed tiers (who outsped whom, confirming Choice Scarf / Tailwind calcs)
- Damage calibration (a crit does 1.5× — without knowing it was a crit, the model can't learn accurate damage expectations)
- Move selection adjustments (if an attack missed, should we retry or pivot?)
- Opponent tendencies (did they switch? did they protect?)

ps-ppo tracks: who moved first, critical hits, super-effective hits, and uses these as explicit features.

**Solution**: Add a `generate_transition_features()` method to `Embedder` that extracts per-turn events from `battle.observations`.

### Implementation Plan

#### Data Source
poke-env's `battle.observations` dict maps turn numbers to `Observation` objects. Each `Observation.events` contains raw Showdown protocol messages like:
```

## Implementation Status Update (2026-03-02)

### Context

Follow-up refactor requested to simplify architecture wiring: always compose grouped encoding with both entity-ID and number-bank expansions internally, and remove compatibility paths for toggling those encoders independently.

### Before State

- `GroupedFeatureEncoder` accepted optional `entity_id_encoder` / `number_bank` objects and had branchy forward paths.
- Callers passed `use_number_banks` / `use_entity_embeddings` flags, and several constructor call-sites only passed grouped parameters without `feature_names` / vocab sizes.

### Problem

Optional composition paths made model construction and checkpoint wiring brittle; after the refactor landing in `model_archs.py`, downstream constructors needed to be aligned to avoid runtime argument mismatch when grouped mode is enabled.

### Solution

- Updated `src/elitefurretai/rl/model_io.py` to the unified grouped interface (no legacy toggle kwargs in model construction).
- Updated grouped constructor call-sites to pass `feature_names`, `num_abilities`, and `num_items`:
    - `src/elitefurretai/supervised/behavior_clone_player.py`
    - `src/elitefurretai/rl/opponents.py`
    - `src/elitefurretai/supervised/train.py`
    - `src/elitefurretai/supervised/fine_tune.py`
    - `src/elitefurretai/supervised/analyze/action_model_diagnostics.py`
    - `src/elitefurretai/supervised/analyze/win_model_diagnostics.py`
- Fixed transformer decision-token typing path in `src/elitefurretai/supervised/model_archs.py` by using `register_parameter(..., None)` when decision tokens are disabled.
- Updated `unit_tests/supervised/test_model_archs.py` to the new grouped constructor contract (required `feature_names`, `num_abilities`, `num_items`; unified number-bank configuration).

### Validation

- `runTests unit_tests/supervised/test_model_archs.py`: 40 passed
- `runTests unit_tests/supervised/test_behavior_clone_player.py`: 18 passed
- `get_errors` on touched files: no remaining diagnostics

### Reasoning

The unified grouped-expansion path reduces configuration skew between training, RL loading, diagnostics, and inference code paths. This improves reliability of experimentation and deployment while ensuring grouped models consistently use both numerical and entity-ID representations.

### Planned Next Steps

1. Run broader RL/supervised test slices when convenient (outside this focused patch).
2. Optionally clean stale docs/config text that still references deprecated number-bank toggle semantics.
['', '-crit', 'p2a: Rillaboom']
['', '-supereffective', 'p1a: Incineroar']
['', 'move', 'p1a: Incineroar', 'Flare Blitz', 'p2a: Rillaboom']
['', '-miss', 'p2a: Rillaboom', 'p1b: Flutter Mane']
['', 'faint', 'p2a: Rillaboom']
['', 'switch', 'p1a: Incineroar', 'Incineroar, L50, M', '100/167']
```

#### Feature Design (~30 new features)

**Per active slot (4 slots: my_slot_0, my_slot_1, opp_slot_0, opp_slot_1):**

| Feature | Type | Description |
|---------|------|-------------|
| `used_move` | binary | Whether this slot used a move last turn |
| `used_switch` | binary | Whether this slot switched last turn |
| `used_protect` | binary | Whether this slot used a protect-type move |
| `was_crit` | binary | Whether this slot's attack was a critical hit |
| `hit_super_effective` | binary | Whether this slot's attack was super effective |
| `hit_resisted` | binary | Whether this slot's attack was resisted |
| `move_missed` | binary | Whether this slot's attack missed |
| `move_failed` | binary | Whether this slot's move failed |

**4 slots × 8 features = 32 features**

**Global transition features (not per-slot):**

| Feature | Type | Description |
|---------|------|-------------|
| `my_slot0_moved_first` | binary | Whether my slot 0 moved before any opponent |
| `my_slot1_moved_first` | binary | Whether my slot 1 moved before any opponent |
| `any_faint_last_turn` | binary | Whether any Pokémon fainted last turn |
| `my_faint_last_turn` | int | Number of my Pokémon that fainted last turn |
| `opp_faint_last_turn` | int | Number of opponent Pokémon that fainted last turn |

**5 more features → total ~37 new features**

#### Implementation Details

```python
def generate_transition_features(self, battle: DoubleBattle) -> Dict[str, float]:
    """Extract features from the previous turn's events."""
    features: Dict[str, float] = {}

    # Default all to -1 (no data / turn 0 / teampreview)
    slots = ["MY:0:", "MY:1:", "OPP:0:", "OPP:1:"]
    for slot in slots:
        for feat in ["used_move", "used_switch", "used_protect",
                      "was_crit", "hit_super_effective", "hit_resisted",
                      "move_missed", "move_failed"]:
            features[f"TRANSITION:{slot}{feat}"] = -1

    features["TRANSITION:my_slot0_moved_first"] = -1
    features["TRANSITION:my_slot1_moved_first"] = -1
    features["TRANSITION:any_faint_last_turn"] = -1
    features["TRANSITION:my_faint_last_turn"] = -1
    features["TRANSITION:opp_faint_last_turn"] = -1

    # If it's turn 0/1 or teampreview, return defaults
    prev_turn = battle.turn - 1
    if prev_turn < 1 or prev_turn not in battle.observations:
        return features

    events = battle.observations[prev_turn].events

    # Parse events to populate features...
    # (see detailed parsing logic below)
    return features
```

#### Event Parsing Logic

The core challenge is mapping raw protocol messages to slots. Key patterns:

```python
# Identify which slot performed what
for event in events:
    if event[1] == "move":
        mon_ident = event[2]  # e.g., "p1a: Incineroar"
        slot = self._ident_to_slot(mon_ident, battle)
        if slot is not None:
            features[f"TRANSITION:{slot}used_move"] = 1
            move_name = event[3]
            # Check if it's a protect move
            if move_name.lower().replace(" ", "") in PROTECT_MOVES:
                features[f"TRANSITION:{slot}used_protect"] = 1

    elif event[1] == "switch":
        mon_ident = event[2]
        slot = self._ident_to_slot(mon_ident, battle)
        if slot is not None:
            features[f"TRANSITION:{slot}used_switch"] = 1

    elif event[1] == "-crit":
        # The crit message refers to the TARGET, but we want to mark
        # the ATTACKER. We need to track who last used "move".
        # Track with a "last_attacker" variable
        pass

    elif event[1] == "-supereffective":
        # Same as crit — refers to the target
        pass

    elif event[1] == "-miss":
        # event[2] is the attacker, event[3] is the target
        attacker_slot = self._ident_to_slot(event[2], battle)
        if attacker_slot:
            features[f"TRANSITION:{attacker_slot}move_missed"] = 1
```

**The tricky part**: `-crit` and `-supereffective` messages follow the `move` event and reference the *target*, not the *attacker*. The solution is to maintain a `last_attacker` variable while iterating:

```python
last_attacker_slot = None
for event in events:
    if event[1] == "move":
        last_attacker_slot = self._ident_to_slot(event[2], battle)
        # ... set used_move features
    elif event[1] == "-crit":
        if last_attacker_slot:
            features[f"TRANSITION:{last_attacker_slot}was_crit"] = 1
    elif event[1] == "-supereffective":
        if last_attacker_slot:
            features[f"TRANSITION:{last_attacker_slot}hit_super_effective"] = 1
    elif event[1] == "-resisted":
        if last_attacker_slot:
            features[f"TRANSITION:{last_attacker_slot}hit_resisted"] = 1
    elif event[1] == "-miss":
        attacker_slot = self._ident_to_slot(event[2], battle)
        if attacker_slot:
            features[f"TRANSITION:{attacker_slot}move_missed"] = 1
    elif event[1] == "-fail":
        if last_attacker_slot:
            features[f"TRANSITION:{last_attacker_slot}move_failed"] = 1
    elif event[1] == "faint":
        fainted_slot = self._ident_to_slot(event[2], battle)
        if fainted_slot and fainted_slot.startswith("MY:"):
            features["TRANSITION:my_faint_last_turn"] += 1
        elif fainted_slot and fainted_slot.startswith("OPP:"):
            features["TRANSITION:opp_faint_last_turn"] += 1
```

**Helper method** needed:
```python
def _ident_to_slot(self, ident: str, battle: DoubleBattle) -> Optional[str]:
    """Map a Showdown identifier like 'p1a: Incineroar' to a slot string."""
    # ident format: "pXY: Nickname" where X is player (1/2), Y is slot (a/b)
    player = ident[:2]  # "p1" or "p2"
    slot_letter = ident[2]  # "a" or "b"
    slot_idx = 0 if slot_letter == "a" else 1

    if player == battle.player_role:
        return f"MY:{slot_idx}:"
    else:
        return f"OPP:{slot_idx}:"
```

**Move order**: Parse move events in sequence, track which player moves appeared first:
```python
my_first_move_turn = None
opp_first_move_turn = None
move_order_idx = 0

for event in events:
    if event[1] == "move":
        slot = self._ident_to_slot(event[2], battle)
        if slot and slot.startswith("MY:") and my_first_move_turn is None:
            my_first_move_turn = move_order_idx
        elif slot and slot.startswith("OPP:") and opp_first_move_turn is None:
            opp_first_move_turn = move_order_idx
        move_order_idx += 1

if my_first_move_turn is not None and opp_first_move_turn is not None:
    features["TRANSITION:my_slot0_moved_first"] = int(my_first_move_turn < opp_first_move_turn)
```

#### Integration into Embedder

1. Add `generate_transition_features()` as a method on `Embedder`
2. Call it in `embed()` when `feature_set == FULL`:
   ```python
   if self._feature_set == self.FULL:
       emb.update(self.generate_feature_engineered_features(battle, bi))
       emb.update(self.generate_transition_features(battle))
   ```
3. Add `self._transition_embedding_size` to `__init__` and update `group_embedding_sizes` to include it as a new group
4. Update `_generate_dummy_battle()` if needed (transition features should all be -1 for a dummy battle, which is the default)

#### Files to Modify
- [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py): Add `generate_transition_features()`, `_ident_to_slot()`, update `embed()`, update `group_embedding_sizes`

#### Effort Estimate
Medium — ~100 lines of new code, but the event parsing needs careful testing against real battle logs to handle edge cases (e.g., moves that hit multiple targets generate multiple `-supereffective` messages).

---

## Problem 2: No Weather/Terrain/Trick Room Duration Encoding

**Problem**: Currently, weather, terrain, and Trick Room are encoded as binary (active or not). The model gets no information about *how many turns remain*. This is critical for decision-making:
- Should I set up Trick Room if the opponent's will expire next turn?
- Should I attack now or protect to stall out their Tailwind?
- Is it worth switching to a sun abuser if sun only has 1 turn left?

ps-ppo uses a two-hot binned duration encoding (bins for 0-5+ turns remaining).

**Solution**: Add remaining-turn features for weather, terrain, Trick Room, and key side conditions.

### Implementation Plan

#### Duration Computation

poke-env stores the turn a field/weather was set in the dict value. Remaining turns:
```python
turns_remaining = max_duration - (battle.turn - set_turn)
```

Standard VGC durations:
| Effect | Normal Duration | Extended Duration | Extension Item |
|--------|----------------|-------------------|----------------|
| Weather | 5 turns | 8 turns | Heat/Damp/Smooth/Icy Rock |
| Terrain | 5 turns | 8 turns | Terrain Extender |
| Trick Room | 5 turns | N/A | — |
| Tailwind | 4 turns | N/A | — |
| Aurora Veil | 5 turns | 8 turns | Light Clay |
| Reflect | 5 turns | 8 turns | Light Clay |
| Light Screen | 5 turns | 8 turns | Light Clay |

**Pragmatic approach**: Use 5-turn default duration. We can't always know if the setter held a duration-extending item (though `BattleInference` can sometimes infer this). Using 5 instead of 8 when extension is present means we predict "ends sooner" — this is conservative and still way more useful than binary.

#### Feature Design (~20 new features)

Replace the current binary encoding with a normalized duration:

**For weather, fields, and key side conditions, add:**
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `WEATHER:{name}:remaining` | float | 0-1 | `turns_remaining / max_duration`, 0 if not active |
| `FIELD:{name}:remaining` | float | 0-1 | Same |
| `SIDE_CONDITION:{name}:remaining` | float | 0-1 | Same for Tailwind/Reflect/Light Screen/Aurora Veil |
| `OPP_SIDE_CONDITION:{name}:remaining` | float | 0-1 | Same for opponent |

**Keep the binary features** — whether the effect is active at all is still useful as a simple signal. The duration adds precision on top.

This adds:
- 4 weather durations (sun, rain, sand, snow)
- 6 field durations (4 terrains + Trick Room + Gravity)
- 4 side condition durations × 2 sides (Tailwind, Reflect, Light Screen, Aurora Veil) = 8
- **Total: ~18 new features**

#### Implementation

```python
# In generate_battle_features():

FIELD_DURATIONS = {
    Field.TRICK_ROOM: 5,
    Field.GRASSY_TERRAIN: 5,
    Field.ELECTRIC_TERRAIN: 5,
    Field.MISTY_TERRAIN: 5,
    Field.PSYCHIC_TERRAIN: 5,
    Field.GRAVITY: 5,
}
WEATHER_DURATION = 5
SIDE_CONDITION_DURATIONS = {
    SideCondition.TAILWIND: 4,
    SideCondition.REFLECT: 5,
    SideCondition.LIGHT_SCREEN: 5,
    SideCondition.AURORA_VEIL: 5,
}

# Weather duration
for weather in self._knowledge["Weather"]:
    if weather in battle.weather:
        set_turn = battle.weather[weather]
        remaining = max(0, WEATHER_DURATION - (battle.turn - set_turn))
        emb["WEATHER:" + weather.name + ":remaining"] = remaining / WEATHER_DURATION
    else:
        emb["WEATHER:" + weather.name + ":remaining"] = 0

# Field duration
for field in self._knowledge["Field"]:
    duration = FIELD_DURATIONS.get(field, 5)
    if field in battle.fields:
        set_turn = battle.fields[field]
        remaining = max(0, duration - (battle.turn - set_turn))
        emb["FIELD:" + field.name + ":remaining"] = remaining / duration
    else:
        emb["FIELD:" + field.name + ":remaining"] = 0

# Side condition duration
for sc in SIDE_CONDITION_DURATIONS:
    duration = SIDE_CONDITION_DURATIONS[sc]
    # Player's side
    if sc in battle.side_conditions:
        set_turn = battle.side_conditions[sc]
        remaining = max(0, duration - (battle.turn - set_turn))
        emb["SIDE_CONDITION:" + sc.name + ":remaining"] = remaining / duration
    else:
        emb["SIDE_CONDITION:" + sc.name + ":remaining"] = 0
    # Opponent's side
    if sc in battle.opponent_side_conditions:
        set_turn = battle.opponent_side_conditions[sc]
        remaining = max(0, duration - (battle.turn - set_turn))
        emb["OPP_SIDE_CONDITION:" + sc.name + ":remaining"] = remaining / duration
    else:
        emb["OPP_SIDE_CONDITION:" + sc.name + ":remaining"] = 0
```

#### Caveat: Stackable Side Conditions
For stackable conditions (Spikes, Toxic Spikes), the dict value is the **layer count**, not the turn set. These don't have durations anyway, so they stay binary (or count-valued). Only the non-stackable, duration-based conditions get the new features.

#### Files to Modify
- [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py): Update `generate_battle_features()` to add duration features

#### Effort Estimate
Low — ~40 lines of new code. Straightforward computation, no tricky parsing.

---

## Problem 3: Raw Integer Boosts vs One-Hot Boost Encoding

**Problem**: Currently, boosts are encoded as raw integers in [-6, 6]. This is a 13-valued ordinal feature, but the model must learn non-linear effects:
- +1 Atk = 1.5× damage, +2 = 2×, +3 = 2.5×, ... the relationship isn't linear
- The difference between +0 and +1 boost is much more strategically important than between +5 and +6
- Negative boosts (Intimidate's -1 Atk) are fundamentally different in meaning from positive boosts

ps-ppo one-hot encodes each stat boost into 13 bins (-6 through +6), giving the model a richer representation.

**Solution**: Replace raw integer boost encoding with 13-bin one-hot encoding for each stat.

### Implementation Plan

#### Feature Design

Current: 6 boosts per Pokémon × 1 float each = 6 features per mon
New: 6 boosts per Pokémon × 13 bins each = 78 features per mon

For 6 own + 6 opponent Pokémon = 12 × (78 - 6) = **+864 features total**

#### Implementation

```python
# Replace this (in generate_pokemon_features and generate_opponent_pokemon_features):
for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
    emb[prefix + "BOOST:" + stat] = mon.boosts[stat] if mon else -1

# With this:
BOOST_RANGE = range(-6, 7)  # -6 to +6 inclusive = 13 bins
for stat in ["accuracy", "atk", "def", "spa", "spd", "spe"]:
    boost_val = mon.boosts[stat] if mon else None
    for b in BOOST_RANGE:
        emb[prefix + f"BOOST:{stat}:{b}"] = (
            -1 if boost_val is None else int(boost_val == b)
        )
```

Also update the same pattern in the `_generate_null_move_features` and `_generate_static_move_features` methods for move boosts (though those represent "boost effect of a move" which is different — they're already small integers representing the boost amount the move *would apply*, not the current in-battle boost. These should stay as raw ints since they're describing the move, not the current state).

#### Impact on Feature Count

- Remove: 6 raw boost features × 12 mons = 72 features
- Add: 6 stats × 13 bins × 12 mons = 936 features
- Net: **+864 features** (9,223 → ~10,087)

This is a meaningful increase in dimensionality, but each one-hot feature is sparse and easy for the model to learn from.

#### Files to Modify
- [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py): Update `generate_pokemon_features()` and `generate_opponent_pokemon_features()` boost sections

#### Effort Estimate
Low — ~20 lines changed. The main concern is the increased feature count and potential impact on training speed (should be minimal since more features but sparser).

---

## Problem 4: OHE Abilities/Items/Types vs Entity ID Embeddings

**Problem**: Currently, abilities (83 tracked), items (40 tracked), and species/types are one-hot encoded directly in the observation vector. This leads to:
- Very wide observation vectors (9,223 features, many from OHE)
- No shared representation between similar items/abilities (Life Orb and Choice Specs both boost damage but share no representational similarity)
- Fixed vocabulary — adding a new ability/item requires retraining from scratch
- Sparse, high-dimensional inputs that are harder for the first layer to learn from

ps-ppo uses integer IDs (species_id, item_id, ability_id, move_id) and passes them through `nn.Embedding` layers in the model. This:
- Dramatically reduces observation dimensionality (~3K vs our ~9K)
- Learns dense, meaningful representations (similar items cluster together)
- Scales to new vocabulary entries without changing observation format

**Solution**: Replace OHE with integer ID encoding for abilities, items, species, and move IDs. Add corresponding `nn.Embedding` layers in the model architecture.

### Implementation Plan

#### Phase 1: Embedder Changes (Observation Format)

Replace OHE features with single integer IDs:

```python
# Current (83 features for ability):
for ability in self._knowledge["Ability"]:
    emb[prefix + "ABILITY:" + ability] = int(mon.ability == ability) if mon else -1

# New (1 feature):
emb[prefix + "ability_id"] = ABILITY_TO_ID.get(mon.ability, 0) if mon else -1
# 0 = unknown/unseen, 1-N = known abilities
```

Create ID mapping dicts:
```python
ABILITY_TO_ID = {ability: i+1 for i, ability in enumerate(sorted(TRACKED_ABILITIES))}
# 0 reserved for unknown
ITEM_TO_ID = {item: i+1 for i, item in enumerate(sorted(TRACKED_ITEMS))}
MOVE_TO_ID = {move_id: i+1 for i, move_id in enumerate(sorted(ALL_MOVE_IDS))}
SPECIES_TO_ID = {species: i+1 for i, species in enumerate(sorted(ALL_SPECIES))}
```

**Feature count impact on abilities alone**: Replace 83 OHE features with 1 int = -82 features per mon × 12 mons = **-984 features**. Similar reductions for items (40→1 = -39 per mon) and potentially moves.

#### Phase 2: Model Architecture Changes

Add `nn.Embedding` layers to `FlexibleThreeHeadedModel` and `TransformerThreeHeadedModel`:

```python
class FlexibleThreeHeadedModel(nn.Module):
    def __init__(self, ...):
        # New embedding layers
        self.ability_emb = nn.Embedding(num_abilities + 1, ability_emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        self.move_emb = nn.Embedding(num_moves + 1, move_emb_dim, padding_idx=0)
        self.species_emb = nn.Embedding(num_species + 1, species_emb_dim, padding_idx=0)
        # ...

    def forward(self, x):
        # Extract integer ID columns from x
        ability_ids = x[:, ability_id_indices].long()  # shape: (batch, 12)
        item_ids = x[:, item_id_indices].long()        # shape: (batch, 12)
        # ...

        # Look up embeddings
        ability_embs = self.ability_emb(ability_ids)    # shape: (batch, 12, emb_dim)
        item_embs = self.item_emb(item_ids)

        # Concatenate with continuous features
        # ...
```

#### Complexity & Dependencies
This is the most invasive change because it requires:
1. **Embedder**: Switch OHE to integer IDs
2. **Model architecture**: Add `nn.Embedding` layers, handle mixed integer/float inputs
3. **GroupedFeatureEncoder**: Must know which features are IDs vs floats
4. **Preprocessing**: Reprocess all training data
5. **NumberBankEncoder**: May interact with this (both are learned embedding approaches)

#### Why This Should Be Last
- Largest effort, most files touched
- Requires careful coordination between embedder and model
- The other 3 changes are additive (just adding features); this one restructures existing features
- Should validate that items 1-3 improve performance before taking on this restructuring

#### Recommended Embedding Dimensions
| Entity | Vocab Size | Embedding Dim | Reasoning |
|--------|-----------|---------------|-----------|
| Ability | 84 | 16 | Smallish vocab, abilities are functionally diverse |
| Item | 41 | 16 | Small vocab, items have clear functional clusters |
| Move | ~800 | 32 | Large vocab, moves have many attributes to encode |
| Species | ~400 (VGC) | 32 | Medium vocab, species have complex stat/type/ability combos |

#### Files to Modify
- [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py): Replace OHE with integer IDs
- [src/elitefurretai/supervised/model_archs.py](src/elitefurretai/supervised/model_archs.py): Add `nn.Embedding` layers to both model classes
- [src/elitefurretai/supervised/train.py](src/elitefurretai/supervised/train.py): Pass entity vocab sizes to model constructor
- [src/elitefurretai/rl/model_io.py](src/elitefurretai/rl/model_io.py): Update model construction for RL

#### Effort Estimate
High — touches 4+ files, requires careful index management between embedder and model. Recommend implementing after confirming items 1-3 improve performance.

---

## Reasoning: Why These Changes Help Build the Best VGC Bot

1. **Transition features** (high priority): VGC is a game of information. Knowing what happened last turn is crucial for adjusting strategy. Speed tier confirmation alone is worth the effort — it's the most common piece of hidden information that determines outcomes.

2. **Duration encoding** (high priority): Duration-aware play is a hallmark of strong VGC players. Knowing Trick Room expires next turn completely changes whether you should switch to a slow or fast Pokémon. This is low-effort, high-impact.

3. **One-hot boosts** (medium priority): Boosts are strategically critical but non-linear. One-hot encoding lets the model learn distinct responses to each boost level (e.g., "at +2, I can OHKO" vs "at +1, I need to boost again"). The cost is more features, but they're sparse and easy.

4. **Entity ID embeddings** (low priority, high upside): This is the biggest architectural improvement but also the most disruptive. It would dramatically reduce input dimensionality and enable the model to learn meaningful entity representations. But it's the riskiest change — it restructures the input format and requires coordinated changes across embedder and model.

## Implementation Order

1. **Duration encoding** (low effort, high impact) — do first
2. **Transition features** (medium effort, high impact) — do second
3. **One-hot boosts** (low effort, medium impact) — do third
4. **Entity ID embeddings** (high effort, high upside) — do last, after validating 1-3

## Planned Next Steps

1. ~~Wait for current BC training to finish (~6-7 hours from start, ~20 epochs)~~
2. ~~Validate metrics against benchmarks (~16.6% top-1, ~42.4% top-5 action accuracy)~~
3. ~~Implement items 1-3 in the embedder~~
4. ~~Proceed with item 4 (entity ID embeddings)~~
5. Reprocess training data with new embedder
6. Retrain BC model
7. Compare metrics to see if embedding improvements help
8. Launch RL fine-tuning with best BC model

---

## Implementation Status (2026-03-01)

### Completed: All 4 Improvements Implemented and Tested

#### 1. Duration Encoding ✅
- Added `:remaining` features for weather (4), fields (6), and duration-based side conditions (4×2 sides = 8) — **18 new features**
- Normalized to [0, 1] range: `remaining_turns / max_duration`
- Constants defined: `WEATHER_DURATION=5`, `FIELD_DURATIONS`, `SIDE_CONDITION_DURATIONS`
- Only non-stackable side conditions get durations (Tailwind, Reflect, Light Screen, Aurora Veil). Stackable ones (Stealth Rock, Spikes, Toxic Spikes) remain presence-only.
- Tests: `TestDurationEncoding` — 9 tests covering turn 0 defaults, just-set, decay, expiry, field/weather/SC, opponent SC, non-duration SCs, and feature count.

#### 2. Transition Features ✅
- New `generate_transition_features()` method (~120 lines) and `_ident_to_slot()` helper
- 37 features total: 4 slots × 8 per-slot features + 5 global features
- Per-slot: `used_move`, `used_switch`, `used_protect`, `was_crit`, `hit_super_effective`, `hit_resisted`, `move_missed`, `move_failed`
- Global: `my_slot0_moved_first`, `my_slot1_moved_first`, `any_faint_last_turn`, `my_faint_last_turn`, `opp_faint_last_turn`
- All default to -1 when no previous turn data available
- Parses `battle.observations[prev_turn].events` using raw Showdown protocol messages
- `-crit` and `-supereffective` correctly attributed to attacker via `last_attacker_slot` tracking
- Added as 15th group in `group_embedding_sizes` for FULL feature set
- Constants: `PROTECT_MOVES` set with all protect variants
- Tests: `TestTransitionFeatures` — 9 tests covering turn 0/1 defaults, feature keys/count, real battle parsing (move, switch, miss, faint, supereffective, resisted), FULL vs RAW embedding, group sizes.

#### 3. One-Hot Boost Encoding ✅
- Replaced raw integer boosts with 13-bin one-hot per stat (BOOST_RANGE = [-6, +6])
- Applied in both `generate_pokemon_features()` and `generate_opponent_pokemon_features()`
- 6 stats × 13 bins = 78 features per pokemon (was 6)
- Move-level BOOST features (`BOOST:atk` in `generate_move_features`) intentionally left as raw integers — they describe the move's boost effect, not current battle state
- Constant: `BOOST_RANGE = list(range(-6, 7))`
- Tests: `TestOneHotBoostEncoding` — 11 tests covering range constant, key format, default zero, positive/negative/boundary boosts, None mon, total features, opponent boosts, exactly-one-hot invariant, move boost unchanged.

#### 4. Entity ID Embeddings ✅
- **Embedder**: Replaced OHE abilities (83 features → 1 int) and items (40 features → 1 int) with integer IDs
- Constants: `ABILITY_TO_ID`, `ITEM_TO_ID` (sorted alphabetically, 1-indexed), `NUM_ABILITIES`, `NUM_ITEMS` (include +1 for unknown token at index 0)
- Exported from `etl/__init__.py`
- **Model architecture**: New `EntityIDEncoder` class in `model_archs.py` (~110 lines):
  - Pattern-matches `ability_id` and `item_id` in feature names
  - `nn.Embedding` with `padding_idx=0` for both ability and item
  - Sentinel -1 clamped to 0 during lookup
  - Same API as `NumberBankEncoder`: `group_output_sizes` property, `embed_group()` method
- **GroupedFeatureEncoder**: New `entity_id_encoder` parameter, `_dual_expand()` method for when both EntityIDEncoder and NumberBankEncoder are active simultaneously (coordinated single-pass to avoid index drift)
- **Model constructors**: Both `FlexibleThreeHeadedModel` and `TransformerThreeHeadedModel` accept `use_entity_embeddings`, `num_abilities`, `num_items`, `ability_embed_dim`, `item_embed_dim`
- **model_io.py**: Updated `build_model_from_config` to pass entity ID params
- Tests: `TestEntityIDEmbeddings` — 14 tests covering ID mappings, ordering, known/unknown abilities/items, None mon, no OHE keys, opponent known/unknown, size consistency, deterministic ordering. `TestEntityIDEncoderModel` — 7 tests covering creation, group position identification, output size expansion, forward pass shapes, sentinel handling, padding gradient blocking. `TestGroupedFeatureEncoderWithEntityIDs` — 2 tests covering forward pass and backprop.

#### 5. Ability Mapping Source Update ✅ (2026-03-01)
- `ABILITY_TO_ID` is now generated from poke-env `GenData` based on the embedder format generation instead of the static `TRACKED_ABILITIES` subset.
- `Embedder` now stores format-specific `self._ability_to_id` and `self._num_abilities`, exposed via `ability_to_id` and `num_abilities` properties.
- Ability IDs in pokemon/opponent features now use the format-specific mapping (`self._ability_to_id.get(...)`).
- This removes dependence on the hand-maintained tracked ability list for ability-ID encoding and ensures coverage of all legal abilities in the generation.

### Key Learnings & Design Decisions

1. **NumberBankEncoder BOOST conflict**: `STAT_PATTERNS` originally included `"BOOST:"` which would have matched the new one-hot boost feature names (`BOOST:atk:-6`), incorrectly treating them as continuous scalars. Fixed by removing `"BOOST:"` from `STAT_PATTERNS` — boosts are now binary one-hot features, not continuous.

2. **Dual encoder index drift**: When both `EntityIDEncoder` and `NumberBankEncoder` are active in `GroupedFeatureEncoder`, applying them sequentially would cause index drift (first expansion shifts tensor positions for the second). Solved with `_dual_expand()` — a single-pass method that merges both replacement maps and processes all positions left-to-right.

3. **Alphabetical feature sorting broke group boundaries (CRITICAL BUG FIXED 2026-03-01)**: The original code globally sorted all feature names alphabetically before converting to a vector. This caused `GroupedFeatureEncoder` (which splits the flat vector by `group_embedding_sizes`) to receive semantically scrambled groups. For example, Group 0 (expected: MON:0) actually contained a mix of MON:0 features, battle features (FIELD:, FORMAT:), and feature-engineered features (EST_DAMAGE:). Every group was ~20-25% wrong features. The cross-attention over "player Pokemon" (groups 0-5) was attending over jumbled features. **Fix**: Replaced global alphabetical sort with group-preserving order — features are sorted alphabetically *within* each semantic group but groups are concatenated in their natural order (MON:0-5, OPP_MON:0-5, battle, engineered, transition). New `_compute_grouped_feature_names()` method ensures this. All 15 groups are now 100% correctly aligned.

4. **Opponent ability uncertainty trade-off**: OHE allowed marking multiple possible abilities for unknown opponent mons. With entity IDs, unknown ability → ID 0 (padding). This loses explicit uncertainty representation but the model can learn appropriate "unknown" semantics through the embedding. The species features (type, stats) already provide strong priors about possible abilities.

5. **Move boosts left as raw integers**: Move-level BOOST features (e.g., `BOOST:atk` in `generate_move_features`) describe the boost a move *would apply*, not the current in-battle boost. These are small fixed integers (-3 to +3 typically) and don't need one-hot encoding.

### Feature Count Changes

- Duration encoding: +18 features
- Transition features: +37 features (FULL only)
- One-hot boosts: +864 features (6 stats × 12 new bins × 12 mons)
- Entity ID abilities: -984 features (83 OHE → 1 int × 12 mons)
- Entity ID items: -468 features (40 OHE → 1 int × 12 mons)
- **Net change for FULL**: -533 features (from ~9,223 to ~8,690)
- **Net change for RAW**: -570 features (no transition features in RAW)

### Validation Results

- **ruff**: All checks passed (0 errors)
- **pyright**: 0 errors, 0 warnings
- **pytest**: 70/71 embedder tests pass; 1 pre-existing failure (`test_embed_turn` — `Pokemon._data` AttributeError in `battle_iterator.py:642`, unrelated to our changes)
- **New tests**: 64 tests in `test_embedder_improvements.py`, all passing (63 original + 1 group alignment test)
- **Existing tests**: 6 updated tests in `test_embedder.py`, all passing

### Files Modified

- [src/elitefurretai/etl/embedder.py](src/elitefurretai/etl/embedder.py): All 4 improvements + constants
- [src/elitefurretai/supervised/model_archs.py](src/elitefurretai/supervised/model_archs.py): `EntityIDEncoder`, `GroupedFeatureEncoder` updates, `NumberBankEncoder` BOOST fix, model constructor updates
- [src/elitefurretai/rl/model_io.py](src/elitefurretai/rl/model_io.py): `build_model_from_config` entity ID params
- [src/elitefurretai/etl/__init__.py](src/elitefurretai/etl/__init__.py): Export new constants
- [unit_tests/etl/test_embedder.py](unit_tests/etl/test_embedder.py): Updated assertions for new feature formats
- [unit_tests/etl/test_embedder_improvements.py](unit_tests/etl/test_embedder_improvements.py): **NEW** — 63 comprehensive tests
