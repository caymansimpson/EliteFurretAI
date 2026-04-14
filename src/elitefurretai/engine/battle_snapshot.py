"""Policy-facing observation container for the Rust fallback runtime.

This structure is deliberately small and stable. It helped the Stage 2 Rust work
by separating runtime stepping from policy evaluation and by giving the sync
driver one place to attach legal actions, sanitized requests, cached vectors,
and optional binding-side metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from elitefurretai.engine.rust_battle_engine import RustBattleSideSnapshot


@dataclass
class BattleSnapshot:
    """Policy-facing battle observation for the Rust self-play path.

    Today this wraps the synchronized poke-env DoubleBattle plus the legal-action
    metadata already derived by the Python driver. In the future, the Rust
    binding can populate the same structure directly without changing the policy
    interface again.
    """

    battle_tag: str
    side: str
    battle: Any
    request: Dict[str, Any]
    legal_actions: List[Tuple[int, str]]
    action_mask: Optional[np.ndarray]
    action_to_choice: Dict[int, str]
    is_teampreview: bool
    opponent_fainted: int
    state_vector: Optional[Any] = None
    binding_snapshot: Optional["RustBattleSideSnapshot"] = None
