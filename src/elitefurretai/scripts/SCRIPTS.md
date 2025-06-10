# SCRIPTS

This folder contains a bunch of scripts I've used on and off throughout development:
1. **Analyze** -- generate statistics of training data or EliteFurretAI's performance in ingesting the training data. 
2. **Fuzz** -- play a bunch of players against each other to find errors in our Inference classes.
3. **Pastes** -- scrape pokepast.es and clean them into text files for EliteFurretAI to ingest.
4. **Prepare** -- prepare training data for supervised learning by cleaning and aggregating it.
5. **Train** -- train supervised models for win and move prediction. I've left an example to train a sklearn model with utilities in `model_utils`.