# Utils

This folder contains general utilities for EliteFurretAI. So far, these include:

1. **BattleOrderValidator**: This module simply validates which moves are available moves we should pick from. It has simple logic that first validates whether a move will even work. Note that this is _not_ representative of what will pass the showdown protocol -- it is a stricter protocol syntax (relative to what Showdown accepts) that we will force the AI to use. It separately contains simple heuristics that eliminate moves that are unlikely to help (e.g. self-attacks if the move doesnt heal or activate weakness policy).
2. [**TeamRepo**](#3-teamrepo): This module simply just retrieves several pre-built teams to use, stored in data/teams. More on usage below.

## TeamRepo

This module provides functionality for reading and managing Pokémon teams in Showdown Export ([PokePaste](https://pokepast.es/syntax.html)) format. It includes features for team validation and organization.

### Features

- Read teams from a specified directory structure
- Optional team validation using [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)
- Organize teams by format
- Retrieve individual teams or all teams for a specific format

### Usage

#### Initializing TeamRepo

```python
from team_repo import TeamRepo

repo = TeamRepo(
    filepath="path/to/team/directory",
    showdown_path="path/to/pokemon-showdown",
    validate=False,
    verbose=False
)
```

#### Parameters

- `filepath`: Path to the directory containing team files (default: "data/teams")
- `showdown_path`: Path to the Pokémon Showdown directory (default: "../pokemon-showdown")
- `validate`: Whether to validate teams using Pokémon Showdown (default: False)
- `verbose`: Whether to print verbose output (default: False)

#### Accessing Teams

```python
# Get all formats
formats = repo.formats

# Get all teams for a specific format
gen8ou_teams = repo.get_all("vgc2023regd")

# Get a specific team
team = repo.get("vgc2023regd", "team_name")

# Get all teams for all formats
all_teams = repo.teams
```

#### Team Validation

If `validate=True` is set when initializing TeamRepo, it will attempt to validate each team using Pokémon Showdown. Make sure you have Pokémon Showdown and that `showdown_path` is correctly set for this feature to work.

### File Structure

The module expects teams to be organized in the following directory structure:

```
data/teams/
    ├── vgc2023regd/
    │   ├── team1.txt
    │   └── team2.txt
    ├── vgc2024regg/
    │   ├── team3.txt
    │   └── team4.txt
    └── ...
```

Each `.txt` file should contain a single team in Showdown Export format.
