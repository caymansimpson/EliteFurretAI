# Model Utilities
Here, we have modules which we'll use to train our various models, listed below:

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Battle. This class can be used to recreate battle objects turn-by-turn. This class is meant to expedite model-training, by storing info from the json, and being able to convert it into training data (it's a read-only class)
2. `DataProcessor` -- takes in either self-play data, showdown data or anonymized showdown data and converts them into a `BattleData` that can be used for training. Also supports reading and writing compressed `BattleData` for storage.
3. `Embedder` -- prepares a battlestate (which can be replicated from BattleData) for input into a neural network by translating the state into an array of floats. It also supports constructing a simplified Embeddings for testing purposes.

Example Usage:
```python
# Initialize Data Processor
dp = DataProcessor()

# Read File into a BattleData object
bd0 = dp.convert_anonymous_showdown_log_to_battledata(double_battle_json_str)

# Write BattleData object to file
dp.write_battle_data_to_file(filename)

# Read BattleData object
bd1 = [for bd in dp.stream_data([filename])][0]

# They are the same
assert bd0 == bd1

# Convert it into a poke-env Battle Object
battle = BattleData.to_battle(bd)

# Initialize my Embedder
embedder = Embedder()

# Go through each turn
for turn in bd.observations:

    # Update battle object with observations in BattleData
    for msg in bd.observations.events:
        inference_utils.update_battle(msg, battle)

    # Embed into a dict with feature name -> value
    feature_dict = embedder.featurize_double_battle(battle)

    # Collapse into a vector for neural network input
    battle_state_emb = embedder.feature_dict_to_vector(feature_dict)

    # Yield this to whatever model wants to ingest these
    yield battle_state_emb

```
