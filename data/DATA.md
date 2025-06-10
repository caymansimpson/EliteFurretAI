# Data
This folder hosts all the data we use in EliteFurretAI development. Some of it is stored in the github repo, and some isn't (e.g. if it's proprietary Showdown data, or too big). Here, we record each type of data and what they are:

| Directory | Data Format | How it was Generated | Use-Cases | Additional Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| battles | Serialized BattleData objects organized in folders by format | From Showdown (anonymized) | Training Supervised Models for Imitiation Learning | Not pushed to github due to both sensitivity of data and size. These can be generated manually via self-play if desired (for other use-cases) |
| database | SQLite DB of usage stats | Generated from Showdown data | Used to make real-time inferences on opponent sets. See [here](../src/elitefurretai/utils/UTILS.md) for more details | Not pushed to github due to both sensitivity of data and size |
| fixture | Various | Manually curated | Used by [conftest.py](../conftest.py) to generate fixtures for pytest | N/A |
| teams | Pokepastes of teams organized by formats | Manually curated | Load into Players to use. See more on how to use this via TeamRepo [here](../src/elitefurretai/utils/UTILS.md) | Not all teams are guaranteed to be valid. You can validate them before use with TeamRepo |
