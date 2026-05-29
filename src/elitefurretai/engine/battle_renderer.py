# -*- coding: utf-8 -*-
"""Battle structure rendering.

Pure functions that read poke-env battle structures (``DoubleBattle``,
``Observation``, ``Pokemon``, ``Move``) and emit human-readable strings.
Used by [`HumanPlayer`](../agents/human_player.py) for the interactive
CLI and by the inference debug paths for state dumps.

No I/O here; nothing in this module calls ``print`` or ``input``.
"""

from typing import Optional, Sequence

from poke_env.battle import (
    DoubleBattle,
    Move,
    Pokemon,
)


def format_pokemon_line(
    mon: Optional[Pokemon],
    *,
    is_opponent: bool,
    tera_status: Optional[str] = None,
) -> str:
    """Render a single Pokemon's state across two lines.

    Args:
        mon: The Pokemon to render. ``None`` or fainted → ``"(fainted)"``.
        is_opponent: When True, hide item and append a ``Moves seen:`` block.
        tera_status: If provided and the mon is not yet tera'd, render a
            ``Tera: <type> (<tera_status>)`` line. Typical values: ``"available"``,
            ``"ally used"``. Ignored when ``mon.is_terastallized`` is True.
    """
    if mon is None:
        return "(fainted)"
    if mon._current_hp is not None and mon._current_hp == 0 and mon._max_hp:
        return "(fainted)"

    header_parts = [mon.species]
    if not is_opponent and mon.item and mon.item != "unknown_item":
        header_parts.append(f"@ {mon.item}")
    if mon.is_terastallized and mon.tera_type is not None:
        header_parts.append(f"(tera'd: {mon.tera_type.name.lower()})")
    header = " ".join(header_parts)

    ability = mon.ability if mon.ability else "?"
    hp_pct = int(mon.current_hp_fraction * 100)
    line1 = f"{header}  | Ability: {ability}  | HP: {hp_pct}%"

    detail_parts = []

    if not mon.is_terastallized and tera_status and mon.tera_type is not None:
        detail_parts.append(f"Tera: {mon.tera_type.name.lower()} ({tera_status})")

    nonzero_boosts = [(k, v) for k, v in mon.boosts.items() if v != 0]
    if nonzero_boosts:
        boost_str = ", ".join(f"{k}:{v:+d}" for k, v in nonzero_boosts)
        detail_parts.append(f"Boosts: {boost_str}")
    else:
        detail_parts.append("Boosts: (none)")

    status_name = mon.status.name if mon.status else "ok"
    detail_parts.append(f"Status: {status_name}")

    if mon.effects:
        effect_str = ", ".join(e.name for e in mon.effects)
        detail_parts.append(f"Effects: {effect_str}")
    else:
        detail_parts.append("Effects: (none)")

    if is_opponent:
        if mon.moves:
            moves_str = ", ".join(mon.moves.keys())
            detail_parts.append(f"Moves seen: {moves_str}")
        else:
            detail_parts.append("Moves seen: (none yet)")

    line2 = "        " + "   ".join(detail_parts)
    return f"{line1}\n{line2}"


def format_field(battle: DoubleBattle) -> str:
    """Render weather, terrain, and both sides' side conditions."""
    weather = ", ".join(w.name for w in battle.weather) if battle.weather else "(none)"
    terrain = ", ".join(f.name for f in battle.fields) if battle.fields else "(none)"
    own_sc = (
        ", ".join(sc.name for sc in battle.side_conditions)
        if battle.side_conditions
        else "(none)"
    )
    opp_sc = (
        ", ".join(sc.name for sc in battle.opponent_side_conditions)
        if battle.opponent_side_conditions
        else "(none)"
    )

    return (
        f"  Weather: {weather}              Terrain: {terrain}\n"
        f"  Your side conditions: {own_sc}\n"
        f"  Opp side conditions:  {opp_sc}"
    )


def format_moves_oneline(
    moves: Sequence[Move],
    *,
    assume_max_pp: bool,
) -> str:
    """Render moves as ``(1) id (cur/max)  (2) id (cur/max) ...``.

    Args:
        moves: Sequence of Move objects in display order.
        assume_max_pp: When True, render the current PP as ``max_pp`` (used
            for opponent moves where we don't track PP).
    """
    if not moves:
        return "(none)"
    parts = []
    for i, move in enumerate(moves, 1):
        max_pp = move.max_pp
        cur_pp = max_pp if assume_max_pp else move.current_pp
        parts.append(f"({i}) {move.id} ({cur_pp}/{max_pp})")
    return "  ".join(parts)


def format_events(events: Sequence[Sequence[str]]) -> str:
    """Render Showdown protocol events as one ``|...|...`` line per event.

    No filtering — every event is rendered verbatim. Each line is indented
    by two spaces for readability under the ``LAST TURN`` banner.
    """
    if not events:
        return "  (no events yet)"
    return "\n".join("  " + "|".join(e) for e in events)
