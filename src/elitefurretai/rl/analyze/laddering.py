"""Official-ladder runner for an RL checkpoint.

Plays N rated ladder battles on sim3.psim.us against the live Showdown
matchmaker using a fixed team and a deterministic argmax policy from a
trained checkpoint. Emits one jsonl record per battle to stdout and,
optionally, to a file.

See planning/stage2/2026-05-29-10-51-laddering-script-design.md for
design rationale.
"""

from __future__ import annotations

import re
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


_RATING_CHANGE_RE = re.compile(
    r"rating:\s*(\d+)\s*&rarr;\s*<strong>\s*(\d+)\s*</strong>",
    re.IGNORECASE,
)
_GXE_RE = re.compile(r"GXE[:\s]*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def _parse_rating_change(raw_html: str) -> Optional[tuple[int, int]]:
    """Extract (pre_rating, post_rating) from a Showdown rating-change `|raw|` line."""
    match = _RATING_CHANGE_RE.search(raw_html)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_gxe(raw_html: str) -> Optional[float]:
    """Extract a GXE percentage (e.g. 54.3) from a Showdown `|raw|` line."""
    match = _GXE_RE.search(raw_html)
    if match is None:
        return None
    return float(match.group(1))


_REPLAY_URL_RE = re.compile(
    r"https://replay\.pokemonshowdown\.com/[A-Za-z0-9_\-]+",
)


def _parse_replay_url(raw_html: str) -> Optional[str]:
    """Extract the Showdown replay URL from a `|raw|` line, if present."""
    match = _REPLAY_URL_RE.search(raw_html)
    if match is None:
        return None
    return match.group(0)
