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
    Observation,
    Pokemon,
)
from poke_env.player.player import Player


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


_ACTION_REFERENCE = """\
================================================================================
ACTION REFERENCE  (printed once at battle start)
================================================================================

Target codes (Showdown convention):
   1,  2   = opp slot 1, opp slot 2
  -1, -2   = your slot 1, your slot 2  (for ally-target moves; omit for self-target)

Action grammar: comma-separated, slot 1 then slot 2.

  Move:     "<move_name|number> [<target>] [tera|mega]"
            e.g.  "astralbarrage 1 tera, surgingstrikes 1"
            e.g.  "protect, helpinghand -1"   (Slot 1 protects; Slot 2 buffs Slot 1)

  Switch:   use species name
            e.g.  "incineroar, surgingstrikes 1"

  Pass:     "pass"      (when a slot has no legal action)
  Forfeit:  "quit"

Notes:
  - No target needed for self-target / spread moves (protect, nasty plot, eq, etc.)
  - During a force switch you'll be prompted with only the slot(s) that must switch.
"""


def format_action_reference() -> str:
    """The action grammar / target-code reference, printed once at battle start."""
    return _ACTION_REFERENCE


def format_teampreview(battle: DoubleBattle) -> str:
    """Render the teampreview banner + both teams.

    Own team shows item + ability + tera type (we know all of these).
    Opponent team shows only species (everything else is hidden info pre-battle).
    No HP bars — at teampreview everyone is at full HP.
    """
    fmt = getattr(battle, "_format", "") or ""
    lines = [
        "=" * 80,
        f"TEAM PREVIEW — {fmt}",
        "=" * 80,
        "",
        "Your team:",
    ]
    for i, mon in enumerate(battle.teampreview_team, 1):
        item = mon.item if mon.item and mon.item != "unknown_item" else "?"
        ability = mon.ability or "?"
        tera = mon.tera_type.name.lower() if mon.tera_type is not None else "?"
        lines.append(
            f"  {i}. {mon.species} @ {item}   | Ability: {ability}   | Tera: {tera}"
        )
    lines.append("")
    lines.append("Opponent's team:")
    for i, mon in enumerate(battle.teampreview_opponent_team, 1):
        lines.append(f"  {i}. {mon.species}")
    return "\n".join(lines)


def _own_slot_target(slot_idx: int) -> str:
    """Target code for one of *our* slots — used for ally-target moves."""
    return "-1" if slot_idx == 0 else "-2"


def _opp_slot_target(slot_idx: int) -> str:
    """Target code for one of the *opponent's* slots — used for attacking."""
    return "1" if slot_idx == 0 else "2"


def _tera_status_for_slot(battle: DoubleBattle, slot_idx: int) -> Optional[str]:
    """When the slot's mon is not tera'd, decide what to show in its tera line."""
    try:
        if battle.can_tera[slot_idx]:
            return "available"
    except (IndexError, AttributeError):
        return None
    return None


def _format_active_section(battle: DoubleBattle) -> str:
    own = battle.active_pokemon
    opp = battle.opponent_active_pokemon

    lines = ["--- ACTIVE POKEMON ---", "Your side:"]
    for i, mon in enumerate(own):
        label = f"Slot {i + 1} ({_own_slot_target(i)})"
        if mon is None:
            lines.append(f"  {label}: (fainted)")
        else:
            rendered = format_pokemon_line(
                mon,
                is_opponent=False,
                tera_status=_tera_status_for_slot(battle, i),
            )
            head, _, rest = rendered.partition("\n")
            lines.append(f"  {label}: {head}")
            if rest:
                lines.append(f"             {rest.strip()}")
    lines.append("")
    lines.append("Opp side:")
    for i, mon in enumerate(opp):
        label = f"Slot {i + 1} ({_opp_slot_target(i)})"
        if mon is None:
            lines.append(f"  {label}: (fainted)")
        else:
            rendered = format_pokemon_line(mon, is_opponent=True)
            head, _, rest = rendered.partition("\n")
            lines.append(f"  {label}:  {head}")
            if rest:
                lines.append(f"              {rest.strip()}")
    return "\n".join(lines)


def _format_options_section(battle: DoubleBattle) -> str:
    own = battle.active_pokemon
    lines = ["--- YOUR OPTIONS ---"]
    for i, mon in enumerate(own):
        if mon is None:
            lines.append(f"Slot {i + 1}: (fainted — must switch)")
            continue
        moves_line = format_moves_oneline(battle.available_moves[i], assume_max_pp=False)
        lines.append(f"Slot {i + 1} {mon.species} moves:")
        lines.append(f"   {moves_line}")
    return "\n".join(lines)


def _format_bench_section(battle: DoubleBattle) -> str:
    # Slot 0's available_switches is representative of the bench;
    # both slots share the same bench.
    try:
        switches = list(battle.available_switches[0])
    except (IndexError, AttributeError):
        switches = []
    if not switches:
        return "Bench: (none)"
    lines = ["Bench:"]
    for mon in switches:
        rendered = format_pokemon_line(mon, is_opponent=False)
        head, _, rest = rendered.partition("\n")
        lines.append(f"  {head}")
        if rest:
            lines.append(f"    {rest.strip()}")
    return "\n".join(lines)


def _format_last_turn_section(battle: DoubleBattle) -> str:
    lines = ["--- LAST TURN ---"]
    prior_turn = battle.turn - 1
    prior_obs = battle.observations.get(prior_turn) if prior_turn >= 1 else None
    if prior_obs is None:
        lines.append("  (battle just started)")
    else:
        lines.append(format_events(prior_obs.events))
    return "\n".join(lines)


def format_battle_state(battle: DoubleBattle) -> str:
    """Render a per-turn snapshot of battle state for the human CLI.

    Includes last turn's raw events, field state, both sides' active mons
    with Showdown-convention target codes, the player's move options per
    slot, and the player's bench.
    """
    fmt = getattr(battle, "_format", "") or ""
    sections = [
        "=" * 80,
        f"Turn {battle.turn} — {fmt}",
        "=" * 80,
        "",
        _format_last_turn_section(battle),
        "",
        "--- FIELD ---",
        format_field(battle),
        "",
        _format_active_section(battle),
        "",
        _format_options_section(battle),
        "",
        _format_bench_section(battle),
    ]
    return "\n".join(sections)


def format_observation(obs: Observation) -> str:
    """Render one Observation. The current poke-env Observation only carries
    raw protocol events, so this is just a thin wrapper around format_events.

    (The legacy version in inference_utils accessed per-mon and field data on
    Observation; those attributes were removed upstream and the old function
    is broken — see planning/stage2/2026-05-29-12-10-human-player-cli-and-battle-renderer.md.)
    """
    return format_events(obs.events)


def _format_teampreview_team_block(header: str, team: Sequence[Pokemon]) -> str:
    """Render one side's teampreview team for the battle-log dump."""
    lines = [f"{header}: ["]
    for mon in team:
        speed = mon.stats.get("spe") if mon.stats else None
        lines.append(f"\t{mon.name} => [Speed: {speed} // Item: {mon.item}]")
    lines.append("]")
    return "\n".join(lines)


def format_battle_log(battle: DoubleBattle, opp: Optional[Player] = None) -> str:
    """Whole-battle debug dump: header + both teampreview teams + every
    turn's events. Optionally enrich the opponent's teampreview team with
    poke-env state from the given opponent Player (used by fuzz harnesses
    that have both sides' Player instances in process).
    """
    header = f"============= Battle [{battle.battle_tag}] =============\n"
    header += (
        f"The battle is between {battle.player_username} and "
        f"{battle.opponent_username} from {battle.player_username}'s perspective.\n"
    )

    own_team = battle.teampreview_team
    own_role = battle.player_role
    own_enriched = []
    for mon in own_team:
        ident = mon.identifier(own_role) if own_role else None
        own_enriched.append(battle.team.get(ident, mon) if ident else mon)

    opp_team = battle.teampreview_opponent_team
    opp_team_dict = battle.opponent_team
    if opp is not None:
        opp_team = opp.battles[battle.battle_tag].teampreview_team
        opp_team_dict = opp.battles[battle.battle_tag].team
    opp_role = battle.opponent_role
    opp_enriched = []
    for mon in opp_team:
        ident = mon.identifier(opp_role) if opp_role else None
        opp_enriched.append(opp_team_dict.get(ident, mon) if ident else mon)

    body = [
        header,
        _format_teampreview_team_block("P1 Teampreview Team (omniscient)", own_enriched),
        _format_teampreview_team_block(
            "P2 Teampreview Team (not omniscient)", opp_enriched
        ),
    ]

    last_obs: Optional[Observation] = None
    for turn, obs in battle.observations.items():
        body.append(f"\nTurn #{turn}:")
        body.append(format_observation(obs))
        last_obs = obs

    current = battle._current_observation
    if last_obs is not None and current.events != last_obs.events:
        body.append(f"\nCurrent Observation; Turn #{battle.turn}:")
        body.append(format_observation(current))

    return "\n".join(body)
