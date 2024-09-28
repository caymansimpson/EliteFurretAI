# Model Utilities
Here, we are going to build modules which we'll use to train our various models, listed below:

### Stage II Models
- **Information Network**: Infostate + history → E[Worldstate]
- **“Value” Network**: Infostate + E[Worldstate] → P(win)
- **Human Policy Network**: PBS + E[Worldstate] → P(action) over all opponent legal actions

### Stage IV Models
- **Regret Network**: Predicts Regret of MCCFR in ESCHER
- **Value Network**: Train a value network based on Regret Network results, one per iteration
- **Policy Network**: Train a policy network based on MCCFR results

## Classes in this folder
1. `BattleData` -- a dataclass that stores all relevant information about a Battle. This class can be used to recreate battle objects turn-by-turn. This class is meant to expedite model-training, by storing info from the json, and being able to convert it into training data (it's a read-only class)
2. `DataProcessor` -- takes in either self-play data, showdown data or anonymized showdown data and converts them into a `BattleData` that can be used for training. Also supports reading and writing compressed `BattleData` for storage.
3. `Embedder` -- prepares a battlestate for input into a neural network by translating the state into a series of floats
