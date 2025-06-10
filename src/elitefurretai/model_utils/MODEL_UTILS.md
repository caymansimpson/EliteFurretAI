# Model Utilities
This folder contains several useful utilities that should be helpful for building supervised or reinforcement learning models.

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Pokemon Battle played on Showdown. This class can be used to recreate `Battle` and `DoubleBattle` objects turn-by-turn. It is compatible with `Battle` objects collected by an agent from either self-play (where we have omniscience) or on Showdown against other agents.
2. `BattleIterator` -- this is the main class that you will find useful if you have records of offline battles. Given a `BattleData` object, it iterates through the logs of a battle to reconstruct and populate a battle object, used like an iterator. It supports commands to iterate to the agent's next required input so that you can recreate a `Battle` object at any important part of the battle w/out a request (which it can also roughly recreate).
3. `Embedder` -- converts a `Battle` object into an array of floats for input into a neural networ. You can use three types of featuresets -- a `simple` one for testing, `raw` one that has all informaiton, and a `full` one that includes feature engineered features (e.g. w/ damage calcs). If you want to edit the featureset yourself, it also outputs a `dict` that maps each feature to a name for custom filtering.
4. `MDBO` --  a container for decision-making for model-based agents (short for ModelDoubleBattleOrder). It should be used to translate `int`s to `BattleOrder`s and vice versa for training. `BattleIterator`'s `last_order()` function returns a `ModelBattleOrder` object. This is used to generate training data (actions in BattleDataset).
5. `BattleDataset` -- implementing the pytorch wrapper so you can generate battle data's at random and easily. It uses all the above classes to return fully embedded battle states for each turn, in batches of battles. It is not optimized for performance because it reads the BattleData file from memory for every perspective. This can be fixed by `@cache`ing `__getitem__` in `BattleDataset` and then reading the indices sequentially with `shuffle=False` in `DataLoader`.

Example Usage:
```python
# Load data
files = ['battledata.json', 'battledata2.json']
embedder = Embedder(format="gen9vgc2024regc", feature_set=Embedder.FULL)
dataset = BattleDataset(files, embedder=embedder)
data_loader = DataLoader(dataset)

# Iterate through batches of battles with data_loader
for batch_states, batch_actions, batch_action_masks, batch_wins, batch_masks in data_loader:

    # Do training things
    pass

```
