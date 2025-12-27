"""
ETL (Extract, Transform, Load) Module

This module contains data processing, embedding, and validation utilities for EliteFurretAI.

Data Processing:
    - battle_data: BattleData dataclass for structured battle storage
    - battle_iterator: Replay battles turn-by-turn
    - battle_dataset: PyTorch datasets for training
    - battle_dataloader: Optimized dataloader with compression support

Feature Engineering:
    - embedder: Convert game state to neural network features
    - encoder: MDBO action space encoding/decoding

Validation & Utilities:
    - battle_order_validator: Validate BattleOrders are legal
    - evaluate_state: Heuristic position evaluation
    - team_repo: Team validation and management

Scripts:
    - filter_battle_data: Filter raw battle logs
    - process_training_data: Convert battles to compressed tensors
    - scrape_pastes: Scrape team data from PokePaste
    - standardize_vgc_levels: Normalize Pokemon levels for VGC
"""

# Core data structures
from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.battle_iterator import BattleIterator
from elitefurretai.etl.battle_dataset import (
    BattleDataset,
    PreprocessedBattleDataset,
)
from elitefurretai.etl.battle_dataloader import OptimizedBattleDataLoader

# Feature engineering
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO, MoveOrderEncoder

# Validation & utilities
from elitefurretai.etl.battle_order_validator import is_valid_order
from elitefurretai.etl.evaluate_state import evaluate_position_advantage
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.etl.compress_utils import save_compressed, load_compressed

__all__ = [
    # Data structures
    "BattleData",
    "BattleIterator",
    "BattleDataset",
    "PreprocessedBattleDataset",
    "OptimizedBattleDataLoader",
    # Feature engineering
    "Embedder",
    "MDBO",
    "MoveOrderEncoder",
    # Validation & utilities
    "is_valid_order",
    "evaluate_position_advantage",
    "TeamRepo",
    # Compression utilities
    "save_compressed",
    "load_compressed",
]
