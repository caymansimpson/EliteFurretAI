# Model Utilities
Here, we are going to build modules to train our various models, listed below:

### Stage II Models
- **Information Network**: Infostate + history → E[Worldstate]
- **“Value” Network**: Infostate + E[Worldstate] → P(win)
- **Human Policy Network**: PBS + E[Worldstate] → P(action) over all opponent legal actions

### Stage IV Models
- **Regret Network**: Predicts Regret of MCCFR in ESCHER
- **Value Network**: Train a value network based on Regret Network results, one per iteration
- **Policy Network**: Train a policy network based on MCCFR results

## Classes in this folder
1. BattleData -- a dataclass that stores all relevant information about a Battle. This class can be used to recreate battle objects turn-by-turn. It also has functions that embeds the Battle State for model training.
2. DataProcessor -- takes in showdown data (via json returned from showdown, ingested into poke-env) and converts these to BattleData classes.
3. ModelTrainer -- w/ model structure as input, data input from DataProcessor, and a function to create values, trains various models. This has yet to be created.

### Example Usage
```
nn, params = create_random_tensorflow_model()
dp = DataProcessor(omniscient=True, double_data=True)
trainer = ModelTrainer(network = nn)

files = glob.glob("~/EliteFurretAI/data/static/*/*.json", recursive=True)
data = dp.load_data(files=files)

trainer.train(data=data.values(), *params)
trainer.save_model("~/EliteFurretAI/data/models/{run}.json".format(run=trainer.run_id))
```
