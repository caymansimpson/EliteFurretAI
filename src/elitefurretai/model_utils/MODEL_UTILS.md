# Model Utilities
Here, we have modules which we'll use to train our various models, listed below:

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Pokemon Battle played on Showdown. This class can be used to recreate battle objects turn-by-turn. This class is meant to expedite model-training, by storing info from the json, and being able to convert it into training data. It also has helper functions to produce `poke-env` `Battle` objects from `BattleData`s, as well as converting either self-play or Showdown logs to `BattleData`s.
2. `Embedder` -- prepares a battlestate (which can be replicated from BattleData) for input into a neural network by translating the state into an array of floats. It also supports constructing a simplified Embeddings for testing purposes.
3. `BattleIterator` -- iterates through the logs of a battle so to reconstruct and populate a battle object, used like an iterator. It can also iterate through the battle to the next agent's input so that you can either train a model based on inputs, or re-evaluate and re-examine the battle state at an agent's decision.
4. `ModelBattleOrder` -- a subclass of `BattleOrder` that is a container for decision-making for model-based `Player`s. It can be used to convert outputs of a neural network (`int`s) to `BattleOrder`s. It can also convert inputs from `BattleData`'s `input_log`s into ints for comparisons. It also has action_masking functionality.
5. `training_generator` -- just some helper functions to read batches of `BattleData`s at a time.

Example Usage:
```python
bd = None
with open("filename_to_read.json", 'r') as f:
    # Read File into a BattleData object
    bd = BattleData.from_showdown_json(orjson.load(f.read()))

    # Write BattleData object to file
    bd.save("filename_to_save.json")

# Read BattleData object, which is now saved in the elitefurretai format
bd1 = BattleData.from_elite_furret_ai_json("filename_to_save.json")

# They are the same
assert bd0 == bd1

# Convert it into a poke-env Battle Object
battle = BattleData.to_battle(bd)

# Initialize my Embedder
embedder = Embedder()

# Generate batches
for bd, filename in file_generator("path/to/BattleDatas"):

    # Look at battle from each player's perspective
    for perspective in ["p1", "p2"]:

        # Generate Battle and its BattleIterator
        battle = bd.to_battle(perspective)
        iter = BattleIterator(battle, bd, perspective=perspective)

        # Go through Battle and stop at each agent decision point
        while not battle.finished and iter.next_input():

            # Get the last input command found by the iterator by the player, and stop if there's no more
            input = iter.last_input
            if input is None:
                continue

            # Simulate a Battle's Request
            request = iter.simulate_request()
            battle.parse_request(request)

            # Convert training data into embedding and state into a label (in this case, the winner of the battle)
            x = embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle))
            y = int(bd.winnder == perspective)

            # Give whoever is calling this their training data/labels
            yield x, y

```
