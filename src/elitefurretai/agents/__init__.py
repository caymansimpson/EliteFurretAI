"""EliteFurretAI agents — user-facing, instantiable battle participants.

See AGENTS.md in this directory for what's here and how to use each.

Re-exports the public classes so ``from elitefurretai.agents import X``
works, but per-file imports (``from elitefurretai.agents.bc_player
import BCPlayer``) are also fine and slightly faster at import time
because they skip loading sibling modules.
"""

from elitefurretai.agents.bc_player import BCPlayer
from elitefurretai.agents.human_player import HumanPlayer
from elitefurretai.agents.max_damage_player import MaxDamagePlayer
from elitefurretai.agents.simple_model_player import SimpleModelPlayer
from elitefurretai.agents.verbose_model_player import VerboseModelPlayer
from elitefurretai.agents.vgcbench_manager import VGCBenchManager

__all__ = [
    "BCPlayer",
    "HumanPlayer",
    "MaxDamagePlayer",
    "SimpleModelPlayer",
    "VerboseModelPlayer",
    "VGCBenchManager",
]
