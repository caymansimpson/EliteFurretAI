"""Official-ladder runner for an RL checkpoint.

Plays N rated ladder battles on sim3.psim.us against the live Showdown
matchmaker using a fixed team and a deterministic argmax policy from a
trained checkpoint. Emits one jsonl record per battle to stdout and,
optionally, to a file.

See planning/stage2/2026-05-29-10-51-laddering-script-design.md for
design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class LadderRecord:
    """One ladder battle's metadata.

    Optional fields stay None when Showdown didn't surface the data
    (unrated practice format, provisional account, replay save failed).
    """

    battle_tag: str
    opponent: str = ""
    outcome: Optional[Literal["win", "loss", "tie"]] = None  # from agent's POV
    final_turn: int = 0
    pre_rating: Optional[int] = None
    post_rating: Optional[int] = None
    gxe: Optional[float] = None
    replay_url: Optional[str] = None
    timestamp: Optional[str] = None


def _parse_player_line(
    split_message: List[str],
) -> Optional[tuple[str, Optional[int]]]:
    """Extract (username, rating) from a `|player|<slot>|<user>|<avatar>|<rating>` line.

    Returns ``None`` for non-player messages. ``rating`` is ``None`` for
    unrated battles or when the field is absent.
    """
    if not split_message or split_message[0] != "player":
        return None
    if len(split_message) < 3:
        return None
    username = split_message[2]
    rating: Optional[int] = None
    if len(split_message) >= 5 and split_message[4].strip():
        rating = int(split_message[4])
    return username, rating
