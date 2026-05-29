# -*- coding: utf-8 -*-
"""Parse human-typed action strings into poke-env BattleOrders.

Grammar (see [`HumanPlayer`](human_player.py) docstring for examples):

    input        := "quit" | <slot_input> ("," <slot_input>)?
    slot_input   := "pass" | <switch> | <move_input>
    switch       := <species_name>                  # matches a bench mon
    move_input   := <move_ref> <token>*
    move_ref     := <move_id> | <move_number_1_to_4>
    token        := <target> | "tera" | "mega"
    target       := "1" | "2" | "-1" | "-2"

The parser is intentionally pure: it never reads stdin or mutates the
battle. The HumanPlayer owns I/O and re-prompts on ``ActionParseError``.
"""

from typing import List, Optional

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player.battle_order import (
    BattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

_VALID_TARGETS = {"1", "2", "-1", "-2"}


class ActionParseError(ValueError):
    """Raised when an input string cannot be parsed as a legal action."""


def parse_action(
    text: str,
    battle: DoubleBattle,
    *,
    force_switch: Optional[List[bool]] = None,
) -> BattleOrder:
    """Parse one line of human input into a BattleOrder.

    Args:
        text: The raw input line (case-insensitive; whitespace trimmed).
        battle: The current battle, used to resolve move/switch references
            and to gate tera/mega flags.
        force_switch: When set, restricts each slot to a switch (or pass on
            unforced slots). When a single slot is forced, accepting one
            action is allowed; the unforced slot is auto-padded with
            ``PassBattleOrder``.
    """
    raw = (text or "").strip().lower()
    if not raw:
        raise ActionParseError("Empty input")
    if raw == "quit":
        return ForfeitBattleOrder()

    parts = [p.strip() for p in raw.split(",")]

    if force_switch is not None and sum(force_switch) == 1 and len(parts) == 1:
        forced_slot = 0 if force_switch[0] else 1
        orders: List[SingleBattleOrder] = [PassBattleOrder(), PassBattleOrder()]
        orders[forced_slot] = _parse_slot(parts[0], battle, forced_slot)
        return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])

    if len(parts) != 2:
        raise ActionParseError(
            f"Expected two comma-separated actions, got {len(parts)}: {raw!r}"
        )

    first = _parse_slot(parts[0], battle, 0)
    second = _parse_slot(parts[1], battle, 1)
    return DoubleBattleOrder(first_order=first, second_order=second)


def _parse_slot(slot_text: str, battle: DoubleBattle, slot_idx: int) -> SingleBattleOrder:
    tokens = slot_text.split()
    if not tokens:
        raise ActionParseError(f"Empty slot {slot_idx + 1} input")

    head = tokens[0]
    rest = tokens[1:]

    if head == "pass":
        return PassBattleOrder()

    switch_target = _resolve_switch(head, battle, slot_idx)
    if switch_target is not None:
        return SingleBattleOrder(order=switch_target)

    move = _resolve_move(head, battle, slot_idx)
    if move is None:
        raise ActionParseError(
            f"Slot {slot_idx + 1}: '{head}' is not a known move id, "
            f"move number 1-{len(battle.available_moves[slot_idx])}, "
            f"or bench species name"
        )

    target = 0
    tera = False
    mega = False
    for tok in rest:
        if tok in _VALID_TARGETS:
            target = int(tok)
        elif tok == "tera":
            tera = True
        elif tok == "mega":
            mega = True
        else:
            raise ActionParseError(
                f"Slot {slot_idx + 1}: unrecognized token {tok!r} (expected "
                f"target 1/2/-1/-2, 'tera', or 'mega')"
            )

    if tera and not battle.can_tera[slot_idx]:
        tera = False
    if mega and not battle.can_mega_evolve[slot_idx]:
        mega = False

    return SingleBattleOrder(
        order=move,
        move_target=target,
        terastallize=tera,
        mega=mega,
    )


def _resolve_switch(token: str, battle: DoubleBattle, slot_idx: int) -> Optional[Pokemon]:
    try:
        switches = list(battle.available_switches[slot_idx])
    except (IndexError, AttributeError):
        return None
    for mon in switches:
        if mon.species == token:
            return mon
    return None


def _resolve_move(token: str, battle: DoubleBattle, slot_idx: int) -> Optional[Move]:
    try:
        moves = list(battle.available_moves[slot_idx])
    except (IndexError, AttributeError):
        return None
    if token.isdigit():
        i = int(token) - 1
        if 0 <= i < len(moves):
            return moves[i]
        return None
    for move in moves:
        if move.id == token:
            return move
    return None
