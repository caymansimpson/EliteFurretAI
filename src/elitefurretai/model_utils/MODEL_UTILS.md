# Model Utilities
Here, we have modules which we'll use to train our various models, listed below:

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Pokemon Battle played on Showdown. This class can be used to recreate battle objects turn-by-turn. This class is meant to expedite model-training, by storing info from the json, and being able to convert it into training data. It also has helper functions to produce `poke-env` `Battle` objects from `BattleData`s, as well as converting either self-play or Showdown logs to `BattleData`s.
2. `Embedder` -- prepares a battlestate (which can be replicated from BattleData) for input into a neural network by translating the state into an array of floats. It also supports constructing a simplified Embeddings for testing purposes.
3. `BattleIterator` -- iterates through the logs of a battle so to reconstruct and populate a battle object, used like an iterator. It can also iterate through the battle to the next agent's input so that you can either train a model based on inputs, or re-evaluate and re-examine the battle state at an agent's decision.
4. `ModelBattleOrder` -- a subclass of `BattleOrder` that is a container for decision-making for model-based `Player`s. It can be used to convert outputs of a neural network (`int`s) to `BattleOrder`s. It can also convert inputs from `BattleData`'s `input_log`s into ints for comparisons. It also has action_masking functionality.
5. `BattleDataset` -- implementing the pytorch wrapper so you can generate battle data's at random and easily. It uses all the above classes to return fully embedded battle states for each turn, in batches of battles.

Example Usage:
```python
# Generate batches
dataset = BattleDataset(
    files=['battledata.json', 'battledata2.json'],
    format="gen9vgc2024regc",
    label_type="win",
    bd_eligibility_func=BattleData.is_valid_for_supervised_learning,
)

# Generate Data Loader
data_loader = DataLoader(
    dataset,
    batch_size=32, # return 32 battles at a time
    num_workers=min(os.cpu_count() or 1, 4),
    shuffle=True
)

for batch_X, batch_Y, batch_mask in data_loader:

    # Since each X, Y and mask is a battle, we need to flatten them
    batch_X = batch_X.flatten(start_dim=0, end_dim=1)
    batch_Y = batch_Y.flatten(start_dim=0, end_dim=1)
    batch_mask = batch_mask.flatten(start_dim=0, end_dim=1)

    # Update progress
    batch_num += 1
    steps += len(batch_X)

    # Train an arbitrary pytorch `model` based on the batch
    loss = model.train_batch(batch_X, batch_Y, batch_mask)

```
