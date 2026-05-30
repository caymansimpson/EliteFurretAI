# Getting Started
For coding n00bs like me, here is a [step-by-step guide](https://docs.google.com/document/d/1QlrOuvNUQYl4ZLsBvRUHDqjQ9ibm61Nq4lrAJfjQjpE/edit?tab=t.0#heading=h.zawc9657hsqc) (maybe outdated) for how to get EliteFurretAI up and running on your local machine. It also includes details on how to contribute if you would like; I wouldn't recommend it though since I don't build with backwards compatibility in mind.

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


## [FYI] Forks of pokemon-showdown and poke-env

EliteFurretAI runs against personal forks of both [pokemon-showdown](https://github.com/smogon/pokemon-showdown) and [poke-env](https://github.com/hsahovic/poke-env). If you want to reproduce training or supervised data generation, clone these forks rather than the upstream repos, and adjust your PYTHONPATHs to point to them in your own virtual environment.

### [caymansimpson/pokemon-showdown](https://github.com/caymansimpson/pokemon-showdown) (branch `elitefurretai-patches`)

Patches on top of `smogon/pokemon-showdown`:

- **`--no-battle-retention` CLI flag.** Upstream keeps `GameRoom` objects alive for 10-40 minutes after a battle ends (controlled by `TIMEOUT_EMPTY_DEALLOCATE` / `TIMEOUT_INACTIVE_DEALLOCATE` in `rooms.ts`), which accumulates ~1.3 GB/hr of resident-set growth during high-throughput RL training where the server finishes hundreds of battles per minute. The flag follows the `--no-security` precedent: it sets `Config.nobattleretention`, and `RoomBattle.end()` destroys the `GameRoom` on the next tick after the `|win|`/`|tie|` broadcast (or after the replay upload settles). Off by default; only training environments that don't need post-battle replay browsing should enable it. This enables training to not load up on RAM.

### [caymansimpson/poke-env](https://github.com/caymansimpson/poke-env) (branch `master`)

Patches on top of `hsahovic/poke-env`:

- **Restore `battle.observations` with an events-only `Observation` class.** Upstream removed the observations feature in commit `d9781e5` ("Move observation feature to examples/"). The fork re-implements a minimal version: each turn's protocol messages are collected into `Observation(events=[...])` and archived in `battle.observations[turn]` when a `|turn|` message arrives, with `battle.current_observation` holding the in-progress turn. No state snapshots (weather, etc.) are stored since EliteFurretAI only consumes the events list.
- **Restore `ObservedPokemon`.** Removed in the same upstream commit. `BattleData`, `MetaDB`, and inference accuracy analysis all import `ObservedPokemon` from `poke_env.battle`, so the fork restores the pre-deletion source to keep EliteFurretAI drop-in.
- **Keep `DoubleBattle.team` order in sync with Showdown on switch/swap.** When the player switches or swaps active mons, upstream updates the active slots but not the underlying `self.team` dict ordering. The fork reorders `self.team` so positional indexing matches the server's view, which the embedder and action encoder rely on. This is particularly important for ingesting game and input logs from Showdown itself.
- **`get_pokemon`: match the identifier's name against species and nickname.** Protocol events that don't carry a `details` field (the `[of]` field inside `-activate Trick` / Symbiosis, Pressure `target_str`, etc.) used to fall through every lookup branch and hit a `species=identifier[4:]` fallback, which raises a pokedex `KeyError` when `identifier[4:]` is a nickname. The fork extends the nickname-match block to also try the identifier's own name part against both species (via `identifies_as`) and any stored `Pokemon._name`.
