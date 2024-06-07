# Utils

This folder contains general utilities for EliteFurretAI. So far, these include:
1. **MetaDB**: a module that calls a database built on raw data (dependent on anonymized showdown data availability) that will allow us to make usage-based inferences (e.g. likelihood that Incineroar has assault vest | observations). The database will not be sharable. `predict_vgc_team` will take in what the AI has observed about the opponent's pokemon (stored in `ObservedPokemon`) and return either the most common team that matches your observations, or a probability distribution of all teams that matches its observations based on data stored in the database. The probability distributions can then be used for probabilistic search for AI; the primary downside is that this method relies on having previously seen data -- it only memorizes. I have not yet tested this method's coverage.
2. **BattleOrderValidator**: This module simply validates which moves are available moves we should pick from. It has simple logic that first validates whether a move will even work. Note that this is _not_ representative of what will pass the showdown protocol -- it is a stricter protocol syntax (relative to what Showdown accepts) that we will force the AI to use.
   - It separately contains simple heuristics that eliminate moves that are unlikely to help (e.g. self-attacks if the move doesnt heal or activate weakness policy).
3. **InfostateNetwork** -- a module that predicts the likelihood of opponent infostates. We will be using opponent’s past actions (e.g. our PBS/history) to infer infostates using our value network when created. Because Pokemon is a two-player zerosum game, our oppponent's predicted value can be trivially inferred by our predicted value -- they sum to 0. We can calculate the probability of an infostate based on the probability an opponent was likely to make previous moves (derived from the value they'd get from those moves). As an example, if I'm going to hydro pump an Incineroar, they probably let the Incineroar take it if the only mon they in back is a Heatran. This method is superior to MetaDB's because it can make inferences using opponent's actions and make guesses on unseen data. This has yet to be implemented.

For `MetaDB``, the database is structured as follows:

| battle | team | pokemon | team_counts |
| ------------- | ------------- | ------------- | ------------- |
| battle_id (key) | team_id | mon_id (key) | team_id |
| player1  | mon_id | species | format |
| player2  | | gender | num |
| elo1  | | tera_type |  |
| elo2  | | item |  |
| team1_id  | | hp |  |
| team2_id  | | atk |  |
| sent_team1_id  | | def |  |
| sent_team2_id  | | spa |  |
| result  | | spd |  |
| score  | | spe |  |
| format | | ability |  |
| | | level |  |
| | | shiny |  |
| | | move1_id |  |
| | | move2_id |  |
| | | move3_id |  |
| | | move3_id |  |
