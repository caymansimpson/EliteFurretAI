# Getting Started
For coding n00bs like me, here is a [step-by-step guide](https://docs.google.com/document/d/1QlrOuvNUQYl4ZLsBvRUHDqjQ9ibm61Nq4lrAJfjQjpE/edit?tab=t.0#heading=h.zawc9657hsqc) for how to get EliteFurretAI up and running on your local machine. It also includes details on how to contribute if you would like; I wouldn't recommend it though since I don't build with backwards compatibility in mind.

## General workflow and Setup
Ultimately, this project is just a collection of extensions built on top of poke-env to facilitate supervised and RL-based laerning; it isn't completely stable so I would recommend you either fork or reach out to me if you would like to build off of it.

Right now, much of this work is built to support offline supervised learning. The general workflow is:
- Take battle logs stored in [pkmn's showdown log format](https://github.com/pkmn/stats/blob/main/anon/src/testdata/anon.json) and store them in a [`BattleData`](./src/elitefurretai/etl/battle_data.py) object
- Use [`BattleIterator`](./src/elitefurretai/etl/battle_iterator.py) to convert `BattleData` into a poke-env `DoubleBattle` object, and recreate the battle object by iterating through the logs
- Use [`BattleDataset`](./src/elitefurretai/etl/battle_dataset.py) to generate state representations (from [`Embedder`](./src/elitefurretai/etl/embedder.py)) with action/win labels (from [`MDBO`](./src/elitefurretai/etl/encoder.py)) by leveraging `BattleIterator` to go through the battle turn by turn
- Train BC model using [`supervised/train.py`](./src/elitefurretai/supervised/train.py)
- Train RL model using [`rl/train.py`](./src/elitefurretai/rl/train.py)

## Table of Contents
- [`./data`](./data) contains all the data to run and train EliteFurretAI. Note that you will have to generate/port your own.
- [`./docs`](./docs) contains pdf's of useful and relevant research that can help you understand a bit more about the problem space. Not well maintained.
- [`./examples`](./examples) working examples that will help you jumpstart working with EliteFurretAI
- [`src`](./src) all major code
    - [`inference`](./src/elitefurretai/inference) contains classes that infer information from the battle state (e.g. choice scarf if you move provably faster than you can)
    - [`etl`](./src/elitefurretai/etl) contains data-related classes that help process and train models
    - [`supervised`](./src/elitefurretai/supervised) contains everything you need to train a prediction model on VGC
    - [`rl`](./src/elitefurretai/rl) contains everything you need to train a RL bot on VGC
- [`unit_tests`](./unit_tests) contains all my unit tests in pytest