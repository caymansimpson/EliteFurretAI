# agents/

User-facing, instantiable battle participants — the things you grab to run a battle in EFA.

Current contents:
- `human_player.py` — `HumanPlayer`, a terminal-driven manual control surface used by `rl/analyze/play_human_vs_model.py`.

Additional Player subclasses and subprocess managers land here during the
ongoing reorg (see planning/stage2/2026-05-19-10-00-agents-directory-reorg-implementation-plan.md).
Full contents/usage docs fill in during phase 6 of that plan.
