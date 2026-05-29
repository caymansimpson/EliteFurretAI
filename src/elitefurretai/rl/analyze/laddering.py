"""Official-ladder runner for an RL checkpoint.

Plays N rated ladder battles on sim3.psim.us against the live Showdown
matchmaker using a fixed team and a deterministic argmax policy from a
trained checkpoint. Emits one jsonl record per battle to stdout and,
optionally, to a file.

See planning/stage2/2026-05-29-10-51-laddering-script-design.md for
design rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from poke_env.battle import AbstractBattle
from poke_env.ps_client import AccountConfiguration, ShowdownServerConfiguration

from elitefurretai.agents.simple_model_player import SimpleModelPlayer


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

    ``split_message`` is the result of splitting a Showdown protocol line on
    ``"|"``.  Because the line starts with ``"|"``, ``split_message[0]`` is
    always the empty string ``""``, the type tag sits at ``[1]``, and the
    positional args begin at ``[2]``.

    Returns ``None`` for non-player messages. ``rating`` is ``None`` for
    unrated battles or when the field is absent.
    """
    if not split_message or split_message[1] != "player":
        return None
    if len(split_message) < 4:
        return None
    username = split_message[3]
    rating: Optional[int] = None
    if len(split_message) >= 6 and split_message[5].strip():
        rating = int(split_message[5])
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


def _update_record(
    record: LadderRecord,
    split_message: List[str],
    *,
    agent_role: str,
) -> None:
    """Merge a single Showdown split message into ``record`` in place.

    ``agent_role`` is the agent's own slot (e.g. ``"p1"``) — used to skip
    the agent's own ``|player|`` line so ``opponent`` / ``pre_rating``
    track the other player only. Unknown messages are ignored.
    """
    if not split_message:
        return
    tag = split_message[1]
    if tag == "player":
        if len(split_message) < 4 or split_message[2] == agent_role:
            return
        parsed = _parse_player_line(split_message)
        if parsed is None:
            return
        username, rating = parsed
        if username:
            record.opponent = username
        if rating is not None:
            record.pre_rating = rating
        return
    if tag == "raw" and len(split_message) >= 3:
        raw_html = split_message[2]
        change = _parse_rating_change(raw_html)
        if change is not None:
            pre, post = change
            record.pre_rating = pre
            record.post_rating = post
        gxe = _parse_gxe(raw_html)
        if gxe is not None:
            record.gxe = gxe
        replay = _parse_replay_url(raw_html)
        if replay is not None:
            record.replay_url = replay


def _finalize_record(record: LadderRecord, battle: AbstractBattle) -> None:
    """Populate outcome, final_turn, and timestamp from a finished battle."""
    if battle.won:
        record.outcome = "win"
    elif battle.lost:
        record.outcome = "loss"
    else:
        record.outcome = "tie"
    record.final_turn = int(battle.turn)
    record.timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _load_credentials(path: Path) -> tuple[str, str]:
    """Read ``{"username": ..., "password": ...}`` from ``path``.

    Raises ``FileNotFoundError`` if the path doesn't exist,
    ``json.JSONDecodeError`` for malformed JSON, and ``ValueError`` if
    either required key is missing or empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"credentials file not found: {path}")
    data = json.loads(path.read_text())
    username = data.get("username")
    password = data.get("password")
    if not username or not password:
        raise ValueError(
            f"credentials file {path} must contain non-empty 'username' "
            f"and 'password' keys"
        )
    return str(username), str(password)


class SimpleModelLadderPlayer(SimpleModelPlayer):
    """SimpleModelPlayer extension that records ladder-specific signals.

    Captures opponent username, pre/post rating, GXE, and the public replay
    URL for each finished battle and dispatches a finalized ``LadderRecord``
    to ``on_record`` when each battle ends.

    Parsing helpers (``_parse_*``, ``_update_record``, ``_finalize_record``)
    are pure module-level functions so they're testable without a live
    Showdown connection; the async hooks below only orchestrate calls
    into those helpers.
    """

    def __init__(
        self,
        *args: Any,
        on_record: Optional[Callable[[LadderRecord], None]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("probabilistic", False)
        super().__init__(*args, **kwargs)
        self._on_record = on_record
        self.ladder_records: Dict[str, LadderRecord] = {}
        self._pending_replay_tasks: set[asyncio.Task[Any]] = set()

    def _get_or_create_record(self, battle_tag: str) -> LadderRecord:
        if battle_tag not in self.ladder_records:
            self.ladder_records[battle_tag] = LadderRecord(battle_tag=battle_tag)
        return self.ladder_records[battle_tag]

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        await super()._handle_battle_message(split_messages)

        if not split_messages or not split_messages[0]:
            return
        battle_tag = split_messages[0][0].lstrip(">")
        if not battle_tag.startswith("battle-"):
            return
        battle = self._battles.get(battle_tag)
        # super()._handle_battle_message above runs parse_message on the
        # |player| line first, so player_role is set by the time we read it.
        agent_role = battle.player_role if battle and battle.player_role else "p1"
        record = self._get_or_create_record(battle_tag)
        for split_message in split_messages[1:]:
            _update_record(record, split_message, agent_role=agent_role)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        record = self._get_or_create_record(battle.battle_tag)
        _finalize_record(record, battle)
        task = asyncio.create_task(
            self.ps_client.send_message("/savereplay", room=battle.battle_tag)
        )
        self._pending_replay_tasks.add(task)
        task.add_done_callback(self._pending_replay_tasks.discard)
        if self._on_record is not None:
            self._on_record(record)


def _make_record_sink(
    output_path: Optional[Path],
) -> tuple[Callable[[LadderRecord], None], Callable[[], None]]:
    """Return ``(sink, close)`` — sink writes jsonl, close flushes the file handle.

    Always writes to stdout; if ``output_path`` is set, also appends
    to that file. The file is opened in append mode so partial runs
    don't lose history.  The caller must invoke ``close()`` (e.g. in a
    ``finally`` block) to ensure the handle is flushed and released even
    on SIGTERM.
    """
    file_handle = open(output_path, "a", encoding="utf-8") if output_path else None

    def sink(record: LadderRecord) -> None:
        line = json.dumps(asdict(record))
        print(line, flush=True)
        if file_handle is not None:
            file_handle.write(line + "\n")
            file_handle.flush()

    def close() -> None:
        if file_handle is not None:
            file_handle.flush()
            file_handle.close()

    return sink, close


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laddering",
        description=(
            "Play N rated battles on the official Showdown ladder with a "
            "given checkpoint, format, and team."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to the RL checkpoint (.pt).",
    )
    parser.add_argument(
        "--battle-format",
        required=True,
        help="Showdown battle format (e.g. gen9vgc2024regg).",
    )
    parser.add_argument(
        "--team",
        required=True,
        type=Path,
        help="Path to the team file (Showdown export format).",
    )
    parser.add_argument(
        "--credentials",
        required=True,
        type=Path,
        help='JSON file containing {"username": ..., "password": ...}.',
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=1,
        help="Number of rated battles to play (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional jsonl file to append per-battle records to.",
    )
    return parser


async def _run_ladder(player: SimpleModelLadderPlayer, n_games: int) -> None:
    await player.ladder(n_games)
    if player._pending_replay_tasks:
        # Give post-final-battle /savereplay requests a chance to flush
        # before the event loop closes.  This only ensures the send was
        # delivered; the |raw| URL response may still arrive later, which
        # is a known inherent race with the Showdown protocol.
        await asyncio.gather(*player._pending_replay_tasks, return_exceptions=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.team.exists():
        raise FileNotFoundError(f"team file not found: {args.team}")
    username, password = _load_credentials(args.credentials)
    team_text = args.team.read_text()

    sink, close_sink = _make_record_sink(args.output)
    account = AccountConfiguration(username, password)
    player = SimpleModelLadderPlayer(
        model_path=str(args.checkpoint),
        device=args.device,
        battle_format=args.battle_format,
        team=team_text,
        account_configuration=account,
        server_configuration=ShowdownServerConfiguration,
        on_record=sink,
    )
    try:
        asyncio.run(_run_ladder(player, args.n_games))
    finally:
        close_sink()

    final = list(player.ladder_records.values())
    won = sum(1 for r in final if r.outcome == "win")
    lost = sum(1 for r in final if r.outcome == "loss")
    tied = sum(1 for r in final if r.outcome == "tie")
    last_rating = next(
        (r.post_rating for r in reversed(final) if r.post_rating is not None),
        None,
    )
    last_gxe = next((r.gxe for r in reversed(final) if r.gxe is not None), None)
    print(
        f"=== ladder run done: {won}W-{lost}L-{tied}T, "
        f"final_rating={last_rating}, final_gxe={last_gxe}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
