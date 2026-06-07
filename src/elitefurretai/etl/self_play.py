# -*- coding: utf-8 -*-
"""self_play.py

Helpers to serialize a *self-play* battle into a replayable ``BattleData``.

Showdown only writes a server-side ``inputLog`` for battles it hosts; when we run
our own agents against each other we instead have to reconstruct that input log
from the orders the agents made. This module provides:

* ``battle_order_to_input`` / ``teampreview_order_to_input`` — convert a poke-env
  order made on a battle into the ``">pX ..."`` line format used by
  ``BattleData.input_logs`` (and parsed back by ``BattleIterator.last_order``).
* ``merge_input_logs`` — interleave each player's own ordered inputs into the
  single, correctly-ordered ``input_logs`` that ``BattleIterator`` expects. The
  interleaving (which player acts, and in what order, at each decision — including
  one- or two-sided force switches) is driven by the protocol via ``BattleIterator``
  rather than guessed, so it stays correct across pivots, faints and revival blessing.
* ``build_self_play_battle_data`` — glue: build a ``BattleData`` from two finished
  battles plus each player's recorded inputs (see ``input_log_recorder.InputLogRecorder``).
"""

from collections import deque
from typing import List, Optional

from poke_env.battle import AbstractBattle, Move, Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.battle_iterator import BattleIterator


def _switch_slot(mon: Pokemon, battle: AbstractBattle, role: str) -> int:
    """Showdown switch orders reference the 1-indexed position of the target in the
    request's ``side.pokemon`` list -- which is what ``BattleIterator``/``MDBO`` decode
    against. We resolve the slot from the live request and do NOT fall back to
    ``battle.team`` dict order: per ``encoder.py`` that order is not guaranteed to match
    the request, so guessing could emit a wrong (silently mislabeled) switch."""
    target = mon.identifier(role)
    request = battle.last_request or {}
    side_pokemon = request.get("side", {}).get("pokemon", []) if request else []
    for i, entry in enumerate(side_pokemon):
        if entry.get("ident") == target:
            return i + 1
    raise ValueError(
        f"Could not resolve switch slot for {target} in {battle.battle_tag}: not found "
        f"in request side.pokemon ({len(side_pokemon)} entries). A live request is "
        "required to serialize a switch order."
    )


def _single_order_to_str(
    order: SingleBattleOrder, battle: AbstractBattle, role: str
) -> str:
    o = order.order
    # Pass/default are stored as raw strings on the SingleBattleOrder (e.g. "/choose pass").
    if isinstance(o, str):
        return o.replace("/choose ", "")
    if isinstance(o, Move):
        part = f"move {o.id}"
        # Foe targets are written "+1"/"+2", ally targets "-1"/"-2", self/spread omit it.
        if order.move_target != 0:
            part += (
                f" +{order.move_target}"
                if order.move_target > 0
                else f" {order.move_target}"
            )
        # Gimmick token comes last (matches the MDBO mapping + BattleIterator.last_order).
        if order.terastallize:
            part += " terastallize"
        elif order.mega:
            part += " mega"
        elif order.dynamax:
            part += " dynamax"
        elif order.z_move:
            part += " zmove"
        return part
    if isinstance(o, Pokemon):
        return f"switch {_switch_slot(o, battle, role)}"
    raise ValueError(f"Unrecognized single order {order!r} in {battle.battle_tag}")


def battle_order_to_input(order: BattleOrder, battle: AbstractBattle) -> Optional[str]:
    """Convert an order made on ``battle`` into a ``">pX ..."`` input-log line.

    Returns ``None`` for orders we don't serialize (default / forfeit), so callers
    can simply skip them.
    """
    role = battle.player_role
    assert role is not None, "battle.player_role must be set to serialize an order"
    if isinstance(order, (DefaultBattleOrder, ForfeitBattleOrder)):
        return None
    if isinstance(order, DoubleBattleOrder):
        first = _single_order_to_str(order.first_order, battle, role)
        second = _single_order_to_str(order.second_order, battle, role)
        return f">{role} {first}, {second}"
    if isinstance(order, SingleBattleOrder):
        return f">{role} {_single_order_to_str(order, battle, role)}"
    return None


def teampreview_order_to_input(order: str, role: str) -> str:
    """Convert a poke-env teampreview string ("/team 4513" or "/team 4, 5, 1, 3")
    into the ``">pX team a, b, c, d"`` form used by ``BattleData.input_logs``."""
    digits = [c for c in order if c.isdigit()]
    return f">{role} team " + ", ".join(digits)


def merge_input_logs(
    bd: BattleData, p1_input_log: List[str], p2_input_log: List[str]
) -> List[str]:
    """Interleave each player's own ordered inputs into ``bd.input_logs``.

    ``bd`` must already have ``logs`` and teams populated (e.g. from
    ``BattleData.from_self_play``); ``bd.input_logs`` is rebuilt in place and also
    returned. ``p1_input_log`` / ``p2_input_log`` are each player's ``">pX ..."``
    inputs in the order that player made them (teampreview first).

    Misalignment detection is best-effort, not a guarantee. We raise on over-supply
    (inputs left unconsumed) and on a mid-battle gap (a decision with no recorded
    input that is followed by a later one). We CANNOT detect a missing *final* input:
    an empty queue on the last decision is indistinguishable from the legitimate
    trailing decision the game ends on. Callers must record one input per decision
    (see ``InputLogRecorder``); these guards catch many mistakes, but not that one.
    """
    queues = {"p1": deque(p1_input_log), "p2": deque(p2_input_log)}
    bd.input_logs = []  # grown just-in-time as the iterator walks the protocol

    it = BattleIterator(bd, perspective="p1", omniscient=True)
    last_nums = list(it.input_nums)
    # An empty owner queue at a decision means no input was recorded for it. That is
    # only legitimate for the trailing decision the game ends on -- a force switch (a
    # faint with the win already decided) or even a bare "|turn|N" emitted just before
    # "|win|". So a skip is fine ONLY if nothing is placed afterwards; if we skip and
    # then place a later input, the stream is under-supplied / misaligned (e.g. a
    # default/timeout order that was never recorded -- poke-env can send
    # `choose_default_move` without ever calling `choose_move`). Fail loudly then.
    underflowed = False
    guard = 0
    while not it.battle.finished and guard < 1_000_000:
        guard += 1
        try:
            it.next()
        except StopIteration:
            break
        # A new decision was classified when the input slice advances to a non-empty range.
        if it.input_nums != last_nums and it.input_nums[1] > it.input_nums[0]:
            last_nums = list(it.input_nums)
            for owner in it.last_input_owners:
                if queues[owner]:
                    if underflowed:
                        raise ValueError(
                            f"merge_input_logs: a decision in {bd.battle_tag} had no recorded "
                            "input but a later one was placed -- the input streams are "
                            "under-supplied or misaligned (e.g. an unrecorded default order)."
                        )
                    bd.input_logs.append(queues[owner].popleft())
                else:
                    underflowed = True

    # And both streams must be fully consumed; leftover inputs mean the opposite
    # (over-supply): more was recorded than the protocol asked for.
    leftover = {role: list(q) for role, q in queues.items() if q}
    if leftover:
        raise ValueError(
            f"merge_input_logs left inputs unconsumed for {bd.battle_tag}: {leftover}. "
            f"Merged {len(bd.input_logs)} of {len(p1_input_log) + len(p2_input_log)} inputs."
        )
    return bd.input_logs


def build_self_play_battle_data(
    p1_battle: AbstractBattle,
    p2_battle: AbstractBattle,
    p1_input_log: List[str],
    p2_input_log: List[str],
) -> BattleData:
    """Build a replayable ``BattleData`` from two finished self-play battles and each
    player's recorded inputs (e.g. from two ``InputLogRecorder`` players)."""
    bd = BattleData.from_self_play(p1_battle, p2_battle, input_logs=[])
    merge_input_logs(bd, p1_input_log, p2_input_log)
    return bd
