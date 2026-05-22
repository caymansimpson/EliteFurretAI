"""RNaDModel — torch.nn.Module wrapper around TransformerThreeHeadedModel.

Used by the inference subprocess, the learner, the trainer, and the
model registry to present a uniform ``forward(x, hidden_state)`` API
over the underlying transformer architecture.

Renamed from ``RNaDAgent`` on 2026-05-19; was moved here from
``rl/players.py`` on the same day as part of the agents/ directory
reorganization (see planning/stage2/2026-05-19-09-30-agents-directory-reorg.md).
"""

from __future__ import annotations

import torch

from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel


class RNaDModel(torch.nn.Module):
    """torch.nn.Module wrapper around TransformerThreeHeadedModel.

    Why this exists
    ---------------
    The supervised (BC) model maintains a growing context tensor across turns.
    RNaDModel presents a uniform `forward(x, hidden_state)` API that callers
    use without caring about the underlying architecture details.

    `get_initial_state(batch_size, device)` returns None (empty context) to
    start a fresh battle.

    This is a thin wrapper — it has no parameters of its own beyond those
    of the wrapped model.
    """

    def __init__(self, model: TransformerThreeHeadedModel):
        super().__init__()
        self.model = model

    def get_initial_state(self, batch_size: int, device: str):
        # Transformer has no initial hidden state — context starts as None.
        return None

    def forward(self, x, hidden_state=None, mask=None, hidden_mask=None):
        assert isinstance(self.model, TransformerThreeHeadedModel)
        turn_logits, tp_logits, value, win_dist_logits, next_hidden = (
            self.model.forward_with_hidden(x, hidden_state, mask, hidden_mask)
        )
        return turn_logits, tp_logits, value, win_dist_logits, next_hidden
