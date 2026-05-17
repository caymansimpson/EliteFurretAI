"""
Smoke test for supervised training (src/elitefurretai/supervised/train.py).

Runs 1 epoch of supervised training with a tiny architecture and a synthetic
in-memory dataset so no real replay data or wandb credentials are required.

Marked `smoke` — run with `pytest -m smoke` or skip with `pytest -m "not smoke"`.
Expected wall-clock time: 5–30 s.
"""

import types
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch


class _FakeBattleDataLoader:
    """Minimal stand-in for OptimizedBattleDataLoader.

    Returns one tiny synthetic batch so train/evaluate/analyze can exercise
    the full code path without real data files on disk.
    """

    batch_size: int = 2
    dataset = None  # accessed by analyze() but not actually used
    _seq_len: int = 5
    _n_batches: int = 1

    def __init__(self, path, embedder=None, **kwargs):
        self._embedding_size = embedder.embedding_size if embedder is not None else 16

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator:
        bs = self.batch_size
        sl = self._seq_len
        es = self._embedding_size
        batch = {
            # All turn decisions (teampreview bit = 0), no force-switch.
            "states": torch.zeros(bs, sl, es),
            # Action 0 is always valid for both turn and teampreview heads.
            "actions": torch.zeros(bs, sl, dtype=torch.long),
            # All actions legal (mask = 1 everywhere).
            "action_masks": torch.ones(bs, sl, 2025),
            "wins": torch.zeros(bs, sl),
            # All positions valid (no padding).
            "masks": torch.ones(bs, sl),
        }
        yield batch


@pytest.mark.smoke
def test_supervised_train_smoke(tmp_path):
    """Supervised training runs 1 epoch end-to-end without errors and saves a checkpoint."""
    from elitefurretai.supervised.train import main

    config = {
        "num_epochs": 1,
        "device": "cpu",
        "save_path": str(tmp_path),
        # Tiny architecture to keep the test fast.
        "grouped_encoder_hidden_dim": 32,
        "grouped_encoder_aggregated_dim": 64,
        "pokemon_attention_heads": 1,
        "early_layers": [64],
        "transformer_layers": 1,
        "transformer_heads": 1,
        "transformer_ff_dim": 64,
        "late_layers": [64],
        "teampreview_head_layers": [64],
        "teampreview_attention_heads": 1,
        "turn_head_layers": [64],
        # Match the fake loader's batch size.
        "batch_size": 2,
        "worker_batch_size": 2,
        # No real workers needed — FakeBattleDataLoader is synchronous.
        "num_workers": 0,
        "files_per_worker": 1,
        "persistent_workers": False,
    }

    wandb_run = types.SimpleNamespace(name="smoke-test")

    with (
        patch(
            "elitefurretai.supervised.train.OptimizedBattleDataLoader",
            _FakeBattleDataLoader,
        ),
        patch("wandb.init"),
        patch("wandb.save"),
        patch("wandb.watch"),
        patch("wandb.log"),
        patch("wandb.finish"),
        patch("wandb.Settings", return_value=MagicMock()),
        patch("wandb.run", wandb_run),
    ):
        main("unused_train", "unused_test", "unused_val", config, save_best=False)

    # main() always saves a final checkpoint named "{run.name}.pt".
    checkpoints = list(tmp_path.glob("*.pt"))
    assert checkpoints, f"No checkpoint written to {tmp_path} after training completed"
