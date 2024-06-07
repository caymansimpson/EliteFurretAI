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
2. `DataProcessor` -- takes in showdown data (via json returned from showdown, ingested into poke-env) and converts these to BattleData classes.

## How BattleData, JSON logs and Battle work
We store battle data recorded in Json in either of two formats:
1. Anonymized Showdown data
2. Our own version of data, which is similar to Showdown data with four exceptions:
    - We dont store nature/iv/ev, just the mon's raw stats
    - We store teampreview teams directly instead of through inputlogs
    - We don't record inputlogs
    - We mark each one we wrote with `eliteFurretAIGenerated`

We use `DataProcessor.json_to_battle:` to convert JSONs to battle. We use this to process the json objects and turn them into `BattleData` which can be used by models for learning

In self-play, we can use `DataProcessor.battle_to_json` to convert our battles to json for storage/training.
