# -*- coding: utf-8 -*-
"""
masking.py — VGC action-space legality

What this is
------------
A pure-Python module that answers a single question: "Of the 2,025 possible
actions the model could output, which ones are actually legal in this battle
state?" The answer is a `np.ndarray` of shape (2025,) full of 0.0 / 1.0 values,
called the action mask.

Why this matters
----------------
The neural network is trained to output logits over a fixed 2025-dim action
space (45 single-slot actions × 45 single-slot actions = 2025 pair actions).
But in a real Pokemon battle, only a small fraction of those are legal — most
moves can't target both opponents, you can't switch into a fainted Pokemon,
both slots can't terastallize in the same turn, etc.

Without masking, the network would happily sample illegal moves, the Showdown
server would reject them, and we'd get cascading errors and wasted battles.
With masking we multiply illegal logits by 0 (or set them to a large negative
number) before softmax, so the policy *cannot* sample them.

Where this fits in the bigger picture
-------------------------------------
Every turn:
    1. The actor gets a request payload from Showdown describing the current
       state of the battle (active mons, available moves, force-switch flags).
    2. `RLTrajectoryPlayer` snapshots that request, calls
       `fast_get_action_mask()` here to get a (2025,) mask.
    3. The mask is sent into the inference batch alongside the embedded state.
    4. The learner's logits are multiplied by the mask; softmax is taken;
       an action is sampled.
    5. The chosen action is decoded back into a Showdown command and sent.

Pipeline within this file (lowest → highest)
--------------------------------------------
  1. get_valid_targets_for_target_type()  [Target Resolution section]
     "Which battlefield positions can Move X reach, given its target-type tag?"

  2. get_valid_slot_actions()  [Slot-Level Legality section]
     "Which of the 45 per-slot actions are structurally legal right now?"

  3. fast_get_action_mask()  [Full Pair Mask section]
     "Give me the full 2025-length binary mask for this turn."

Why "fast" in the name
----------------------
The naive approach — try every one of 2,025 actions, ask poke-env if each is
legal — took 3-4 seconds per turn (this used to be 99% of our runtime!).
The current approach reads the Showdown request payload and *constructs*
only the legal actions in O(moves × targets), bringing this down to ~0.05 ms.
That's a ~52,000× speedup. See RL.md for the historical context.

Edge cases this code handles
----------------------------
- Force-switch turns (a Pokemon fainted; we must switch in a replacement).
- Tatsugiri / Dondozo "Commander" ability (one slot is forced to pass).
- Trapped Pokemon (cannot switch out).
- Tera lock (only one slot may terastallize per turn — pair-level constraint).
- Both slots switching to the same bench Pokemon (illegal — pair-level).
"""

from typing import Any, Dict, List, Optional, Set

import numpy as np
from poke_env.battle import Move
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon

from elitefurretai.etl.encoder import MDBO

# ── Target Resolution ─────────────────────────────────────────────────────────
#
# These functions answer: "Given the move's target-type tag from the Showdown
# request payload, which battlefield positions are reachable?"
#
# The Showdown request is the source of truth. Centralizing this logic here
# ensures the mask generator, heuristic player, and sync driver do not drift.
# ─────────────────────────────────────────────────────────────────────────────

# Moves whose target is always the user / a side-effect — no positional target needed.
# Pass EMPTY_TARGET_POSITION so the action encoding can treat them uniformly.
_NO_TARGET_TARGET_TYPES = {
    "self",
    "allySide",
    "allAdjacent",
    "allAdjacentFoes",
    "all",
    "scripted",
    "randomNormal",
}

# Moves that pick a single arbitrary target from opponents + ally-other-slot.
_SINGLE_TARGET_TARGET_TYPES = {
    "adjacentFoe",
    "normal",
    "any",
}


def get_valid_targets_for_target_type(
    battle: DoubleBattle,
    slot: int,
    target_type: str,
) -> List[int]:
    """Resolve legal Showdown targets for a target type on the current board.

    The Showdown request payload is the source of truth whenever it is available.
    This helper centralizes the target-shape rules so the fast mask, heuristic
    player path, and sync driver do not drift.
    """

    normalized_target_type = str(target_type or "normal")

    opp_active = battle.opponent_active_pokemon
    ally_active = battle.active_pokemon
    opponent_positions = [
        DoubleBattle.OPPONENT_1_POSITION,
        DoubleBattle.OPPONENT_2_POSITION,
    ]
    ally_positions = [
        DoubleBattle.POKEMON_1_POSITION,
        DoubleBattle.POKEMON_2_POSITION,
    ]

    opp_1_exists = opp_active[0] is not None and not opp_active[0].fainted
    opp_2_exists = opp_active[1] is not None and not opp_active[1].fainted
    ally_exists = [
        ally_active[0] is not None and not ally_active[0].fainted,
        ally_active[1] is not None and not ally_active[1].fainted,
    ]

    if normalized_target_type in _NO_TARGET_TARGET_TYPES:
        return [DoubleBattle.EMPTY_TARGET_POSITION]

    if normalized_target_type == "adjacentFoe":
        targets: List[int] = []
        if opp_1_exists:
            targets.append(opponent_positions[0])
        if opp_2_exists:
            targets.append(opponent_positions[1])
        return targets

    if normalized_target_type in _SINGLE_TARGET_TARGET_TYPES:
        targets = []
        if opp_1_exists:
            targets.append(opponent_positions[0])
        if opp_2_exists:
            targets.append(opponent_positions[1])
        other_slot = 1 - slot
        if ally_exists[other_slot]:
            targets.append(ally_positions[other_slot])
        return targets

    if normalized_target_type == "adjacentAllyOrSelf":
        targets = [ally_positions[slot]]
        other_slot = 1 - slot
        if ally_exists[other_slot]:
            targets.append(ally_positions[other_slot])
        return targets

    if normalized_target_type == "adjacentAlly":
        other_slot = 1 - slot
        if ally_exists[other_slot]:
            return [ally_positions[other_slot]]
        return []

    return [DoubleBattle.EMPTY_TARGET_POSITION]


def get_valid_targets_for_request_move(
    battle: DoubleBattle,
    slot: int,
    request_move: Dict[str, Any],
) -> List[int]:
    """Resolve valid targets directly from a Showdown request move payload."""

    return get_valid_targets_for_target_type(
        battle,
        slot,
        str(request_move.get("target", "normal")),
    )


def get_valid_targets(
    battle: DoubleBattle,
    slot: int,
    *,
    request_move: Optional[Dict[str, Any]] = None,
    move: Optional[Move] = None,
    active_mon: Optional[Pokemon] = None,
) -> List[int]:
    """Single entry point for target legality in RL battle action generation."""

    if request_move is not None:
        return get_valid_targets_for_request_move(battle, slot, request_move)

    if move is not None and active_mon is not None:
        return list(battle.get_possible_showdown_targets(move, active_mon))

    return [DoubleBattle.EMPTY_TARGET_POSITION]


# ── Slot-Level Legality ───────────────────────────────────────────────────────
#
# These functions answer: "Which of the 45 per-slot actions are legal right now?"
#
# Action encoding (must match encoder.py / MDBO):
#   Each slot has 45 actions:
#     Actions  0-39: move actions  (move_idx × 10 + target_offset + tera_offset)
#       target offsets: 0=-2, 1=-1, 2=no_target, 3=+1, 4=+2
#       tera offsets:   0=no tera, 5=tera
#     Actions 40-43: switch actions (SWITCH_ACTION_BASE + 0-indexed bench slot)
#     Action  44:    pass
#
#   Full action space: 45 × 45 = 2025 pairs (slot0_action * 45 + slot1_action)
# ─────────────────────────────────────────────────────────────────────────────

ACTIONS_PER_SLOT = 45

MOVE_ACTION_BASE = 0  # Actions 0-39 are move actions
SWITCH_ACTION_BASE = 40  # Actions 40-43 are switch actions
PASS_ACTION = 44  # Action 44 is pass

# Maps poke-env/Showdown battlefield position → offset within the move action block.
# The offsets are fixed by the MDBO encoding in encoder.py.
TARGET_TO_OFFSET = {
    DoubleBattle.POKEMON_2_POSITION: 0,
    DoubleBattle.POKEMON_1_POSITION: 1,
    DoubleBattle.EMPTY_TARGET_POSITION: 2,
    DoubleBattle.OPPONENT_1_POSITION: 3,
    DoubleBattle.OPPONENT_2_POSITION: 4,
}


def get_valid_slot_actions(
    battle: DoubleBattle,
    slot: int,
    request: Dict,
) -> Set[int]:
    """Get all valid action indices [0..44] for a single slot.

    Args:
        battle: Current DoubleBattle state
        slot:   Slot index (0 or 1)
        request: battle.last_request dict (should be a frozen snapshot)

    Returns:
        Set of valid action indices for this slot.
    """
    valid_actions: Set[int] = set()

    # Handle force switch
    request_force_switch = request.get("forceSwitch")
    force_switch = (
        [bool(value) for value in request_force_switch]
        if isinstance(request_force_switch, list)
        else [bool(value) for value in battle.force_switch]
    )
    forced_slot_count = sum(force_switch)

    if force_switch[slot]:
        available_switches = _get_available_switch_indices(battle, slot, request)
        for switch_idx in available_switches:
            valid_actions.add(SWITCH_ACTION_BASE + switch_idx)
        allow_pass = len(available_switches) < max(1, forced_slot_count)
        if not valid_actions or allow_pass:
            valid_actions.add(PASS_ACTION)
        return valid_actions

    # Slot has no active pokemon — only pass is valid
    if not battle.active_pokemon[slot]:
        valid_actions.add(PASS_ACTION)
        return valid_actions

    # No active info in request — fall back to pass
    if "active" not in request or slot >= len(request["active"]):
        valid_actions.add(PASS_ACTION)
        return valid_actions

    active_info = request["active"][slot]

    # Commanding Pokemon (Dondozo under Tatsugiri's Commander) passes automatically
    if slot_is_commanding(battle, slot, request):
        valid_actions.add(PASS_ACTION)
        return valid_actions

    # The once-per-battle "gimmick" offset (+5) is shared across formats: tera in
    # tera formats, mega in the mega format. They are mutually exclusive per format,
    # so a single offset is unambiguous. Showdown signals availability via
    # canTerastallize (tera type string) or canMegaEvo (bool).
    can_tera = active_info.get("canTerastallize") is not None
    can_mega = bool(active_info.get("canMegaEvo", False))
    can_gimmick = can_tera or can_mega

    # Enumerate legal move actions
    moves = active_info.get("moves", [])
    for move_idx, move_info in enumerate(moves):
        if move_info.get("disabled", False):
            continue
        if move_info.get("pp", 1) <= 0:
            continue

        valid_targets = get_valid_targets_for_request_move(battle, slot, move_info)

        for target in valid_targets:
            target_offset = TARGET_TO_OFFSET.get(target, 2)
            base_action = move_idx * 10 + target_offset
            valid_actions.add(base_action)
            if can_gimmick:
                valid_actions.add(base_action + 5)

    # Enumerate legal switch actions (only if not trapped)
    if not battle.trapped[slot]:
        available_switches = _get_available_switch_indices(battle, slot, request)
        for switch_idx in available_switches:
            valid_actions.add(SWITCH_ACTION_BASE + switch_idx)

    return valid_actions


def _get_available_switch_indices(
    battle: DoubleBattle,
    slot: int,
    request: Dict,
) -> List[int]:
    """Return 0-indexed bench positions that are legal to switch to for this slot.

    In the Showdown protocol "switch N" is 1-indexed, but internally we use
    0-indexed positions that map to SWITCH_ACTION_BASE + idx.
    """
    switch_indices: List[int] = []

    if "side" not in request:
        return switch_indices

    side_pokemon = request["side"].get("pokemon", [])

    for i, mon in enumerate(side_pokemon):
        if mon.get("active", False):
            continue
        condition = mon.get("condition", "0 fnt")
        if condition == "0 fnt" or "fnt" in condition:
            continue
        switch_indices.append(i)

    return switch_indices


def _get_valid_targets_for_move(
    battle: DoubleBattle,
    slot: int,
    target_type: str,
) -> List[int]:
    """Thin wrapper kept for backwards-compatibility with direct callers."""
    return get_valid_targets_for_target_type(battle, slot, target_type)


def slot_is_commanding(
    battle: DoubleBattle,
    slot: int,
    request: Optional[Dict[str, Any]],
) -> bool:
    """Return whether the active slot is marked as commanding in the request.

    A Pokemon is commanding when Tatsugiri is in Dondozo's mouth (Commander ability).
    Dondozo effectively gets two turns; Tatsugiri is forced to pass.
    We check both the request's active section and the side.pokemon section because
    the flag can appear in either depending on when in the turn cycle we're called.
    """
    if request is None:
        request = battle.last_request
    if not request:
        return False

    active_entries = request.get("active") or []
    if slot < len(active_entries):
        active_entry = active_entries[slot]
        if isinstance(active_entry, dict) and bool(active_entry.get("commanding")):
            return True

    side = request.get("side") or {}
    side_pokemon = side.get("pokemon") or []
    active_side_pokemon = [pokemon for pokemon in side_pokemon if pokemon.get("active")]
    if slot < len(active_side_pokemon):
        active_side_entry = active_side_pokemon[slot]
        if isinstance(active_side_entry, dict) and bool(
            active_side_entry.get("commanding")
        ):
            return True

    active_pokemon = (
        battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
    )
    commander_flag = getattr(active_pokemon, "is_commanding", False)
    return bool(commander_flag) if isinstance(commander_flag, bool) else False


# ── Full Pair Mask ────────────────────────────────────────────────────────────
#
# fast_get_action_mask() is the top-level entry point. It combines per-slot
# legality into a 2025-dimensional binary mask over all (slot0_action,
# slot1_action) pairs. Pair-level constraints (no double-tera, no switching
# both slots to the same Pokemon) are applied after per-slot enumeration.
#
# The mask is indexed as: action_index = slot0_action * ACTIONS_PER_SLOT + slot1_action
# ─────────────────────────────────────────────────────────────────────────────


def fast_get_action_mask(
    battle: DoubleBattle,
    request_override: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Generate the full 2025-dimensional action mask for the current turn.

    This is the main entry point used by RLTrajectoryPlayer and SyncPolicyPlayer.
    It enumerates valid actions directly from the request (~0.05ms) rather than
    validating each of the 2025 candidates (~3-4s) — a ~50,000× speedup.

    Args:
        battle:           Current DoubleBattle state.
        request_override: If provided, use this instead of battle.last_request.
                          RLTrajectoryPlayer passes a frozen snapshot here to
                          prevent mask/decode drift when the request changes mid-flight
                          while we're waiting on async inference.

    Returns:
        np.ndarray of shape (2025,) — 1.0 for valid actions, 0.0 for invalid.
        Falls back to all-ones if the request is missing (permissive fallback —
        better to let the model try and have Showdown reject than to block entirely).
    """
    mask = np.zeros(MDBO.action_space(), dtype=np.float32)

    # Permissive fallback: if we have no request at all, return all-ones rather
    # than a zero-vector. A zero-vector would cause softmax-on-zero downstream;
    # all-ones lets the model pick anything and we let Showdown be the final
    # arbiter of legality. This path should only fire on the very first turn or
    # in error-recovery situations.
    request = request_override if request_override is not None else battle.last_request
    if not request:
        return np.ones(MDBO.action_space(), dtype=np.float32)

    # Determine force-switch: check both the request field and battle.force_switch
    request_fs = request.get("forceSwitch")
    if isinstance(request_fs, list):
        is_force_switch = any(bool(v) for v in request_fs)
    else:
        is_force_switch = bool(request_fs)
    is_force_switch = is_force_switch or any(battle.force_switch)

    if is_force_switch:
        # battle.force_switch is the ground truth (more reliable than the request field)
        force_switch = battle.force_switch

        slot0_actions = (
            get_valid_slot_actions(battle, 0, request)
            if force_switch[0]
            else {PASS_ACTION}
        )
        slot1_actions = (
            get_valid_slot_actions(battle, 1, request)
            if force_switch[1]
            else {PASS_ACTION}
        )

        if force_switch[0] and force_switch[1]:
            # Both slots must switch to different targets
            _mark_valid_switch_pairs(mask, slot0_actions, slot1_actions, request)
        else:
            # Only one slot switching — at least one must actually switch (not both pass)
            for a0 in slot0_actions:
                for a1 in slot1_actions:
                    if a0 == PASS_ACTION and a1 == PASS_ACTION:
                        continue
                    mask[a0 * ACTIONS_PER_SLOT + a1] = 1.0

        if mask.sum() == 0:
            return np.ones(MDBO.action_space(), dtype=np.float32)
        return mask

    # Normal turn
    slot0_actions = get_valid_slot_actions(battle, 0, request)
    slot1_actions = get_valid_slot_actions(battle, 1, request)

    _mark_valid_action_pairs(mask, slot0_actions, slot1_actions, battle, request)

    if mask.sum() == 0:
        return np.ones(MDBO.action_space(), dtype=np.float32)

    return mask


def _mark_valid_action_pairs(
    mask: np.ndarray,
    slot0_actions: Set[int],
    slot1_actions: Set[int],
    battle: DoubleBattle,
    request: Dict,
) -> None:
    """Mark valid action pairs, filtering out two pair-level constraints:
    - Both slots cannot terastallize in the same turn.
    - Both slots cannot switch to the same bench Pokemon.
    """

    def is_tera_action(action: int) -> bool:
        if action >= SWITCH_ACTION_BASE:
            return False
        # Within each move group of 10, offsets 5-9 are tera variants
        return (action % 10) >= 5

    def get_switch_target(action: int) -> Optional[int]:
        if action < SWITCH_ACTION_BASE or action >= PASS_ACTION:
            return None
        return action - SWITCH_ACTION_BASE

    for a0 in slot0_actions:
        a0_tera = is_tera_action(a0)
        a0_switch_target = get_switch_target(a0)

        for a1 in slot1_actions:
            if a0_tera and is_tera_action(a1):
                continue
            if a0_switch_target is not None:
                a1_switch_target = get_switch_target(a1)
                if a1_switch_target is not None and a0_switch_target == a1_switch_target:
                    continue
            mask[a0 * ACTIONS_PER_SLOT + a1] = 1.0


def _mark_valid_switch_pairs(
    mask: np.ndarray,
    slot0_actions: Set[int],
    slot1_actions: Set[int],
    request: Dict,
) -> None:
    """Mark valid force-switch pairs: different targets, and at least one real switch."""

    def get_pokemon_idx(action: int) -> Optional[int]:
        if action < SWITCH_ACTION_BASE or action >= PASS_ACTION:
            return None
        return action - SWITCH_ACTION_BASE

    for a0 in slot0_actions:
        idx0 = get_pokemon_idx(a0)
        for a1 in slot1_actions:
            idx1 = get_pokemon_idx(a1)
            # Both switching to same Pokemon is illegal
            if idx0 is not None and idx1 is not None and idx0 == idx1:
                continue
            # Both passing is illegal during force-switch
            if a0 == PASS_ACTION and a1 == PASS_ACTION:
                continue
            mask[a0 * ACTIONS_PER_SLOT + a1] = 1.0
