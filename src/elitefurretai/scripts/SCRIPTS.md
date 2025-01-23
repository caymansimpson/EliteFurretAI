# SCRIPTS

This folder contains scripts I run as one-offs as I have gone through my development process. Descriptions of each:
- Scrape Pastes: scrapes a list of pokepastes using multithreading from a CSV and then stores them in a file structure that we use for EliteFurretAI
- Standardize VGC Levels: standardizes the pastes that we scraped so that we can accurately calculate pokemons' stats from teampreview using pokepaste information
- Fuzz Test: Runs our various inference programs tens of thousands of times with various teams to identify any poke-env or inference bugs. If you do any development, I encourage you to lean on this type of testing given VGC's complexity.
- Fuzz Test Threaded: Same deal as Fuzz Test, but goes 2x as fast
- Training Data Dry Run: Goes through training data to make sure there are no surprises or bugs when parsing
- Train Move Prediction Model: Trains a neural network to play like humans. If you have a bunch of replay data (saved, community sourced or scraped from Showdown public replays), you can use this script to build a model that behaves like players in the dataset.