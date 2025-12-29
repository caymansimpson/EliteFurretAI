# ETL (Extract, Transform, Load)

This folder contains the core data pipeline for EliteFurretAI. It handles everything from scraping raw data and parsing battle logs to transforming game states into neural network features and loading them efficiently for training.

## Common Workflows

### 1. Supervised Learning Pipeline (From Showdown Logs)
This workflow transforms raw JSON battle logs from Pokemon Showdown into training-ready tensors.

1.  **Filter Data**:
    *   **Script**: `filter_battle_data.py`
    *   **Input**: Directory of raw JSON battle logs.
    *   **Action**: Scans files to remove low-quality battles (low Elo), protocol errors, corrupted logs, or battles with unsupported mechanics (e.g., specific abilities like Illusion).
    *   **Output**: A JSON file containing a list of valid file paths.
    *   **Command**: `python src/elitefurretai/etl/filter_battle_data.py`

2.  **Process Data**:
    *   **Script**: `process_training_data.py`
    *   **Input**: The JSON list of valid files from step 1.
    *   **Action**:
        *   Replays each battle using `BattleIterator`.
        *   Embeds states using `Embedder` and encodes actions using `MDBO`.
        *   Applies data augmentation (e.g., randomizing team order to prevent leakage).
        *   Saves data in compressed chunks (512 battles per file) to optimize I/O.
    *   **Output**: A directory of `.pt.zst` (Zstandard compressed PyTorch tensors) files.
    *   **Command**: `python src/elitefurretai/etl/process_training_data.py`

3.  **Train Model**:
    *   **Location**: `src/elitefurretai/supervised/`
    *   **Action**: Use training scripts (e.g., `three_headed_transformer.py`) which load the processed data using `BattleDataset`.

### 2. Team Collection Pipeline
This workflow gathers and prepares competitive teams for the AI to use in self-play or evaluation.

1.  **Collect Links**:
    *   **Action**: Manually gather PokePaste URLs (e.g., from Twitter, Victory Road, Limitless) and save them into a CSV file. The CSV should map format/regulation to the URL.

2.  **Scrape & Clean**:
    *   **Script**: `scrape_pastes.py`
    *   **Input**: The CSV file containing PokePaste links.
    *   **Action**:
        *   Downloads the HTML from PokePaste.
        *   Parses the team data.
        *   **Standardization**: Crucially, it enforces `Level: 50` for all VGC teams. This is required because the `Embedder` relies on accurate stats, and missing level info defaults to Level 100.
        *   Saves cleaned text files to `data/teams/<format>/`.
    *   **Command**: `python src/elitefurretai/etl/scrape_pastes.py <path_to_csv>`

3.  **Use Teams**:
    *   **Module**: `team_repo.py`
    *   **Action**: Use the `TeamRepo` class to load, validate, and sample these teams in your agent code.

---

## Detailed File Documentation

### Scripts & Pipelines

#### `filter_battle_data.py`
*   **Purpose**: Quality control for the training dataset.
*   **How it works**: It iterates through raw log files using multithreading. For each file, it checks:
    *   **Elo**: Is the average rating above the threshold?
    *   **Protocol**: Does the log contain deprecated or broken protocol messages?
    *   **Mechanics**: Does the battle involve mechanics we explicitly exclude (e.g., Zoroark's Illusion, Ditto, or specific items like Iron Ball that complicate inference)?
*   **Design Choice**: We filter *before* processing to save compute time. We output a list of files rather than copying them to avoid duplicating terabytes of data.

#### `process_training_data.py`
*   **Purpose**: Converts human-readable logs into machine-learning tensors.
*   **How it works**:
    *   Loads a `BattleData` object.
    *   Uses `BattleIterator` to step through the battle.
    *   At each step, calls `Embedder.embed_battle()` to get the state tensor and `MDBO.battle_order_to_action()` to get the label.
    *   **Chunking**: Accumulates 512 battles worth of data and saves them as a single `.pt.zst` file.
*   **Design Choice**:
    *   **Compression**: Zstandard is used because it offers high decompression speeds, which is critical when the dataloader needs to feed the GPU.
    *   **Augmentation**: It randomizes the order of Pokemon in the party during processing. This fixes a "teampreview leakage" issue where the model could memorize team slots instead of learning the actual Pokemon features.

#### `scrape_pastes.py`
*   **Purpose**: Ingests external team data.
*   **How it works**: A threaded web scraper that parses PokePaste HTML. It uses regex to clean up inconsistent formatting (e.g., spacing, nature definitions).
*   **Key Feature**: It includes a `standardize_vgc_levels` function. If a team is for VGC, it ensures every Pokemon has `Level: 50`. Without this, `poke-env` calculates stats at Level 100, breaking the `Embedder`'s damage calculation features.

### Core Libraries

#### `battle_data.py` (`BattleData`)
*   **Purpose**: The source of truth for a battle's history.
*   **How it works**: Parses the JSON log structure from Showdown. It extracts the initial seed, team details, and the sequence of protocol messages.
*   **Design Choice**: It is designed to be immutable and serializable. It serves as the input for `BattleIterator`.

#### `battle_iterator.py` (`BattleIterator`)
*   **Purpose**: A time-machine for battles.
*   **How it works**: It maintains a `poke_env.Battle` object. It iterates through the logs from `BattleData`, applying each event to the state.
*   **Key Methods**:
    *   `next_input()`: Fast-forwards the state until the agent needs to make a decision.
    *   `last_order()`: Returns the action the player actually took at the current state (the ground truth label).

#### `embedder.py` (`Embedder`)
*   **Purpose**: The bridge between game objects and tensors.
*   **How it works**: It inspects the `DoubleBattle` object and extracts hundreds of features (HP %, status, weather, field effects, etc.).
*   **Feature Sets**:
    *   `simple`: Basic stats (HP, status). Good for unit tests.
    *   `full`: The production feature set. Includes "Omniscient" features (if enabled) and "Inferred" features (guesses about opponent items/sets). It also runs damage calculations to add "expected damage" features to the state.

#### `encoder.py` (`MDBO`)
*   **Purpose**: Defines the action space.
*   **How it works**: Maps `BattleOrder` objects (e.g., "Move 1 targeting slot 2") to an integer ID (0-2024).
*   **Design Choice**: We use a discrete action space of size 2025. This covers all valid combinations of moves, targets, and switches in a Double Battle. This allows us to use standard Cross Entropy Loss for classification.

#### `battle_dataset.py` & `battle_dataloader.py`
*   **Purpose**: Efficient data streaming for PyTorch.
*   **How it works**:
    *   `BattleDataset` maps indices to specific trajectories within the compressed files.
    *   `OptimizedBattleDataLoader` uses a custom sampler (`OptimizedPreprocessedSampler`).
*   **Design Choice**: **Locality**. The sampler ensures that a worker reads all trajectories from `file_A.pt.zst` before moving to `file_B.pt.zst`. This prevents "thrashing" where the worker constantly opens and closes files, which would kill training performance.

#### `team_repo.py` (`TeamRepo`)
*   **Purpose**: Organized access to team files.
*   **How it works**: Scans the `data/teams` directory. It can optionally spin up a local Showdown process to validate that the teams are legal for their format.
