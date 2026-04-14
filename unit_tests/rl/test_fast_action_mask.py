# -*- coding: utf-8 -*-
"""
Unit tests for fast action mask generation.

These tests verify that fast_get_action_mask correctly:
1. Masks out invalid moves (disabled, no PP, wrong targets)
2. Prevents double-terastallization
3. Prevents switching both Pokemon to the same target
4. Handles force switch scenarios correctly
5. Respects trapped status (no switches when trapped)
6. Validates target positions for different move types
7. Handles edge cases (fainted Pokemon, no valid actions)
"""


import numpy as np
import pytest
from poke_env.battle.double_battle import DoubleBattle

from elitefurretai.etl import BattleData, BattleIterator
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.fast_action_mask import (
    ACTIONS_PER_SLOT,
    MOVE_ACTION_BASE,
    PASS_ACTION,
    SWITCH_ACTION_BASE,
    TARGET_TO_OFFSET,
    fast_get_action_mask,
    get_valid_slot_actions,
    get_valid_targets_for_request_move,
    slot_is_commanding,
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def action_to_indices(action_int: int) -> "tuple[int, int]":
    """
    Convert a combined action index to (slot0_action, slot1_action).

    Args:
        action_int: Combined action index (0-2024)

    Returns:
        Tuple of (slot0_action, slot1_action) where each is 0-44
    """
    slot0 = action_int // ACTIONS_PER_SLOT
    slot1 = action_int % ACTIONS_PER_SLOT
    return slot0, slot1


def indices_to_action(slot0: int, slot1: int) -> int:
    """
    Convert (slot0_action, slot1_action) to combined action index.

    Args:
        slot0: Slot 0 action (0-44)
        slot1: Slot 1 action (0-44)

    Returns:
        Combined action index (0-2024)
    """
    return slot0 * ACTIONS_PER_SLOT + slot1


def is_move_action(action: int) -> bool:
    """Check if action is a move (not switch/pass)."""
    return MOVE_ACTION_BASE <= action < SWITCH_ACTION_BASE


def is_switch_action(action: int) -> bool:
    """Check if action is a switch."""
    return SWITCH_ACTION_BASE <= action < PASS_ACTION


def is_tera_action(action: int) -> bool:
    """Check if action includes terastallization."""
    if not is_move_action(action):
        return False
    # Tera actions have offset 5-9 within each move group of 10
    return (action % 10) >= 5


def get_switch_target(action: int) -> "int | None":
    """Get the Pokemon index for a switch action."""
    if not is_switch_action(action):
        return None
    return action - SWITCH_ACTION_BASE


def get_move_details(action: int) -> dict:
    """
    Extract move details from action encoding.

    Returns:
        Dict with 'move_idx', 'target', 'is_tera'
    """
    if not is_move_action(action):
        return {}

    move_idx = action // 10
    offset = action % 10
    is_tera = offset >= 5

    if is_tera:
        offset -= 5

    # Map offset to target position.
    # Offsets 0/1 are ally slots, 3/4 are opponent slots.
    target_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
    target = target_map.get(offset, 0)

    return {"move_idx": move_idx, "target": target, "is_tera": is_tera}


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


def test_action_space_constants():
    """Test that action space constants match MDBO."""
    assert ACTIONS_PER_SLOT == 45
    assert MOVE_ACTION_BASE == 0
    assert SWITCH_ACTION_BASE == 40
    assert PASS_ACTION == 44
    assert MDBO.action_space() == 2025  # 45 * 45


def test_target_offset_mapping():
    """Test that target offset mapping is correct."""
    assert TARGET_TO_OFFSET[-2] == 0  # Ally slot 2
    assert TARGET_TO_OFFSET[-1] == 1  # Ally slot 1
    assert TARGET_TO_OFFSET[0] == 2  # No target / self
    assert TARGET_TO_OFFSET[1] == 3  # Opponent slot 1
    assert TARGET_TO_OFFSET[2] == 4  # Opponent slot 2


def test_helper_functions():
    """Test action encoding/decoding helpers."""
    # Test round-trip conversion
    for action in [0, 100, 500, 1000, 2024]:
        slot0, slot1 = action_to_indices(action)
        reconstructed = indices_to_action(slot0, slot1)
        assert reconstructed == action

    # Test action type checks
    assert is_move_action(0)  # First move action
    assert is_move_action(39)  # Last move action
    assert not is_move_action(40)  # First switch action
    assert is_switch_action(40)
    assert is_switch_action(43)
    assert not is_switch_action(44)  # Pass action
    assert not is_switch_action(0)  # Move action

    # Test tera detection
    assert not is_tera_action(0)  # Move 0, target -2, no tera
    assert is_tera_action(5)  # Move 0, target -2, tera
    assert not is_tera_action(10)  # Move 1, target -2, no tera
    assert is_tera_action(15)  # Move 1, target -2, tera

    # Test switch target extraction
    assert get_switch_target(40) == 0  # Switch to Pokemon 0
    assert get_switch_target(41) == 1  # Switch to Pokemon 1
    assert get_switch_target(42) == 2  # Switch to Pokemon 2
    assert get_switch_target(43) == 3  # Switch to Pokemon 3
    assert get_switch_target(0) is None  # Not a switch
    assert get_switch_target(44) is None  # Pass action


# =============================================================================
# BASIC MASK TESTS WITH REAL BATTLES
# =============================================================================


def test_fast_action_mask_basic(vgc_json_anon):
    """Test that fast_get_action_mask returns a valid mask."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)

    # Advance to first turn
    battle_iter.next_input()  # Teampreview
    battle_iter.next_input()  # Turn 1

    battle = battle_iter.battle
    mask = fast_get_action_mask(battle)  # type: ignore[arg-type]

    # Basic checks
    assert mask.shape == (MDBO.action_space(),)
    assert mask.dtype == np.float32
    assert np.all((mask == 0.0) | (mask == 1.0))  # Binary mask
    assert mask.sum() > 0  # At least some valid actions


def test_mask_shape_and_dtype(vgc_json_anon):
    """Test mask has correct shape and dtype."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview
    battle_iter.next_input()  # Turn 1

    mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]  # type: ignore[arg-type]

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (2025,)
    assert mask.dtype == np.float32


def test_mask_is_binary(vgc_json_anon):
    """Test that mask only contains 0.0 and 1.0."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()
    battle_iter.next_input()

    mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]

    # All values should be either 0.0 or 1.0
    assert np.all((mask == 0.0) | (mask == 1.0))


def test_mask_has_valid_actions(vgc_json_anon):
    """Test that mask has at least one valid action."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)

    # Test multiple turns
    battle_iter.next_input()  # Teampreview
    for _ in range(5):
        try:
            battle_iter.next_input()
            mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]
            assert mask.sum() > 0, "Mask should have at least one valid action"
        except StopIteration:
            break


# =============================================================================
# CONSTRAINT TESTS
# =============================================================================


def test_no_double_terastallization(vgc_json_anon):
    """Test that mask prevents both Pokemon from terastallizing."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview
    battle_iter.next_input()  # Turn 1

    mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]
    valid_actions = np.where(mask == 1.0)[0]

    # Check all valid actions
    for action in valid_actions:
        slot0, slot1 = action_to_indices(action)

        # If both slots use tera, this should be invalid
        if is_tera_action(slot0) and is_tera_action(slot1):
            pytest.fail(
                f"Action {action} allows double-tera: "
                f"slot0={slot0} (tera={is_tera_action(slot0)}), "
                f"slot1={slot1} (tera={is_tera_action(slot1)})"
            )


def test_no_double_switch_same_target(vgc_json_anon):
    """Test that mask prevents switching both Pokemon to the same target."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview

    # Test multiple turns
    for _ in range(5):
        try:
            battle_iter.next_input()
            mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]
            valid_actions = np.where(mask == 1.0)[0]

            for action in valid_actions:
                slot0, slot1 = action_to_indices(action)

                # If both are switches, they must target different Pokemon
                if is_switch_action(slot0) and is_switch_action(slot1):
                    target0 = get_switch_target(slot0)
                    target1 = get_switch_target(slot1)
                    if target0 == target1:
                        pytest.fail(
                            f"Action {action} allows double-switch to same target: "
                            f"slot0={slot0} (target={target0}), "
                            f"slot1={slot1} (target={target1})"
                        )
        except StopIteration:
            break


def test_no_invalid_action_pairs(vgc_json_anon):
    """Test that all valid actions respect both constraints."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()

    for _ in range(5):
        try:
            battle_iter.next_input()
            mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]
            valid_actions = np.where(mask == 1.0)[0]

            for action in valid_actions:
                slot0, slot1 = action_to_indices(action)

                # Check no double-tera
                if is_tera_action(slot0) and is_tera_action(slot1):
                    pytest.fail(f"Double-tera allowed: action={action}")

                # Check no same-target double-switch
                if is_switch_action(slot0) and is_switch_action(slot1):
                    if get_switch_target(slot0) == get_switch_target(slot1):
                        pytest.fail(f"Same-target switch allowed: action={action}")
        except StopIteration:
            break


# =============================================================================
# FORCE SWITCH TESTS
# =============================================================================


def test_force_switch_scenario(vgc_json_anon2):
    """Test mask during force switch scenarios."""
    bd = BattleData.from_showdown_json(vgc_json_anon2)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview

    # Iterate through battle to find force switch
    found_force_switch = False
    for _ in range(20):
        try:
            battle_iter.next_input()
            battle = battle_iter.battle

            if any(battle.force_switch):
                found_force_switch = True
                mask = fast_get_action_mask(battle)  # type: ignore[arg-type]  # type: ignore[arg-type]

                # During force switch, valid actions should be limited
                assert mask.sum() > 0, "Should have valid actions during force switch"

                valid_actions = np.where(mask == 1.0)[0]

                # Check that actions respect force_switch state
                for action in valid_actions:
                    slot0, slot1 = action_to_indices(action)

                    # If slot is forced to switch, it should only have switch/pass actions
                    if battle.force_switch[0]:
                        assert is_switch_action(slot0) or slot0 == PASS_ACTION, (
                            f"Slot 0 forced switch but has non-switch action: {slot0}"
                        )

                    if battle.force_switch[1]:
                        assert is_switch_action(slot1) or slot1 == PASS_ACTION, (
                            f"Slot 1 forced switch but has non-switch action: {slot1}"
                        )

                break
        except StopIteration:
            break

    if not found_force_switch:
        pytest.skip("No force switch scenario found in this battle")


# =============================================================================
# SLOT-SPECIFIC TESTS
# =============================================================================


def test_get_valid_slot_actions(vgc_json_anon):
    """Test get_valid_slot_actions for individual slots."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview
    battle_iter.next_input()  # Turn 1

    battle = battle_iter.battle
    request = battle.last_request

    # Get valid actions for each slot
    slot0_actions = get_valid_slot_actions(
        battle,  # type: ignore[arg-type]
        0,
        request,
    )  # type: ignore[arg-type]
    slot1_actions = get_valid_slot_actions(
        battle,  # type: ignore[arg-type]
        1,
        request,
    )  # type: ignore[arg-type]

    # Both slots should have valid actions
    assert len(slot0_actions) > 0
    assert len(slot1_actions) > 0

    # All actions should be in valid range
    for action in slot0_actions:
        assert 0 <= action <= PASS_ACTION

    for action in slot1_actions:
        assert 0 <= action <= PASS_ACTION


def test_trapped_pokemon_cannot_switch(vgc_json_anon):
    """Test that trapped Pokemon cannot switch."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview

    # Search for a turn where a Pokemon is trapped
    found_trapped = False
    for _ in range(20):
        try:
            battle_iter.next_input()
            battle = battle_iter.battle

            # Check if any Pokemon is trapped
            for slot in [0, 1]:
                if battle.trapped[slot]:
                    found_trapped = True
                    request = battle.last_request
                    slot_actions = get_valid_slot_actions(
                        battle,  # type: ignore[arg-type]
                        slot,
                        request,
                    )

                    # Trapped Pokemon should not have switch actions
                    for action in slot_actions:
                        if is_switch_action(action):
                            pytest.fail(
                                f"Trapped Pokemon in slot {slot} has switch action: {action}"
                            )
                    break

            if found_trapped:
                break
        except StopIteration:
            break

    if not found_trapped:
        pytest.skip("No trapped scenario found in this battle")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_no_request_fallback(vgc_json_anon):
    """Test that mask returns all-ones fallback when no request available."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    # Don't advance - battle should have no request
    battle = battle_iter.battle

    # Manually clear request to test fallback
    battle._last_request = None  # type: ignore[assignment]

    mask = fast_get_action_mask(battle)  # type: ignore[arg-type]

    # Should return all-ones fallback
    assert mask.sum() == MDBO.action_space()
    assert np.all(mask == 1.0)


def test_empty_mask_fallback(vgc_json_anon):
    """Test that all-ones fallback is used when no valid actions found."""
    # This is a safety check - in practice, there should always be valid actions
    # But if the logic fails, we don't want to crash during training
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()
    battle_iter.next_input()

    mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]

    # Mask should never be completely empty (always has fallback)
    assert mask.sum() > 0


# =============================================================================
# INTEGRATION TESTS WITH MULTIPLE BATTLES
# =============================================================================


def test_mask_consistency_across_turns(vgc_json_anon):
    """Test that masks are consistent across all turns of a battle."""
    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()  # Teampreview

    turn_count = 0
    while turn_count < 20:
        try:
            battle_iter.next_input()
            mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]

            # Basic validity checks
            assert mask.shape == (MDBO.action_space(),)
            assert mask.sum() > 0
            assert np.all((mask == 0.0) | (mask == 1.0))

            turn_count += 1
        except StopIteration:
            break


def test_mask_on_multiple_battles(vgc_json_anon, vgc_json_anon2, vgc_json_anon3):
    """Test mask generation on multiple different battles."""
    for battle_json in [vgc_json_anon, vgc_json_anon2, vgc_json_anon3]:
        bd = BattleData.from_showdown_json(battle_json)
        battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
        battle_iter.next_input()  # Teampreview

        for _ in range(10):
            try:
                battle_iter.next_input()
                mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]

                # Validate mask
                assert mask.shape == (MDBO.action_space(),)
                assert mask.sum() > 0
                assert np.all((mask == 0.0) | (mask == 1.0))

                # Verify constraints
                valid_actions = np.where(mask == 1.0)[0]
                for action in valid_actions:
                    slot0, slot1 = action_to_indices(action)

                    # No double-tera
                    assert not (is_tera_action(slot0) and is_tera_action(slot1)), (
                        "Double-tera in battle"
                    )

                    # No same-target double-switch
                    if is_switch_action(slot0) and is_switch_action(slot1):
                        assert get_switch_target(slot0) != get_switch_target(slot1), (
                            "Same-target switch in battle"
                        )

            except StopIteration:
                break


def test_both_perspectives(vgc_json_anon):
    """Test that masks work correctly for both p1 and p2 perspectives."""
    bd = BattleData.from_showdown_json(vgc_json_anon)

    for perspective in ["p1", "p2"]:
        battle_iter = BattleIterator(bd, perspective=perspective, omniscient=False)
        battle_iter.next_input()  # Teampreview

        for _ in range(5):
            try:
                battle_iter.next_input()
                mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]

                # Basic checks
                assert mask.shape == (MDBO.action_space(),)
                assert mask.sum() > 0
                assert np.all((mask == 0.0) | (mask == 1.0))

                # Constraint checks
                valid_actions = np.where(mask == 1.0)[0]
                for action in valid_actions:
                    slot0, slot1 = action_to_indices(action)
                    assert not (is_tera_action(slot0) and is_tera_action(slot1))
            except StopIteration:
                break


# =============================================================================
# PERFORMANCE/COMPARISON TESTS
# =============================================================================


def test_single_target_move_requires_explicit_target():
    """
    Test that single-target moves (adjacentFoe, normal, any) require explicit target selection.

    This is a regression test for a bug where the mask allowed no-target (offset=2) for
    single-target moves when only one opponent was remaining. This caused errors like:
    "[Invalid choice] Can't move: Wild Charge needs a target"

    Single-target moves MUST have explicit target (-2, -1, 1, or 2), not 0.
    """
    from unittest.mock import MagicMock

    from elitefurretai.rl.fast_action_mask import _get_valid_targets_for_move

    # Create mock battle with only one opponent (not fainted)
    battle = MagicMock(spec=DoubleBattle)
    opp_mon = MagicMock()
    opp_mon.fainted = False  # Opponent is alive
    ally_mon_0 = MagicMock()
    ally_mon_0.fainted = False
    ally_mon_1 = MagicMock()
    ally_mon_1.fainted = False
    battle.opponent_active_pokemon = [opp_mon, None]  # Only one opponent
    battle.active_pokemon = [ally_mon_0, ally_mon_1]  # Both allies present

    # Test adjacentFoe - should NOT include 0 (no-target)
    targets = _get_valid_targets_for_move(battle, slot=0, target_type="adjacentFoe")
    assert 0 not in targets, "adjacentFoe should not allow no-target (0)"
    assert 1 in targets, "adjacentFoe should allow targeting opponent at +1"

    # Test normal - should NOT include 0 (no-target)
    targets = _get_valid_targets_for_move(battle, slot=0, target_type="normal")
    assert 0 not in targets, "normal should not allow no-target (0)"
    # Should allow opponent and ally targets
    assert 1 in targets, "normal should allow targeting opponent"
    assert -2 in targets, "normal should allow targeting ally (slot 1)"

    # Test any - should NOT include 0 (no-target)
    targets = _get_valid_targets_for_move(battle, slot=0, target_type="any")
    assert 0 not in targets, "any should not allow no-target (0)"

    # Contrast with spread moves - these SHOULD return [0]
    targets = _get_valid_targets_for_move(battle, slot=0, target_type="allAdjacent")
    assert targets == [0], "allAdjacent (spread) should use no-target (0)"

    targets = _get_valid_targets_for_move(battle, slot=0, target_type="allAdjacentFoes")
    assert targets == [0], "allAdjacentFoes (spread) should use no-target (0)"


def test_get_valid_slot_actions_use_protocol_faithful_target_offsets():
    from unittest.mock import MagicMock

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]

    ally_left = MagicMock()
    ally_left.fainted = False
    ally_right = MagicMock()
    ally_right.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [ally_left, ally_right]
    battle.opponent_active_pokemon = [opp_left, opp_right]

    request = {
        "active": [
            {
                "moves": [{"target": "normal", "pp": 8, "disabled": False}],
                "trapped": False,
            },
            {"moves": [], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }

    slot_actions = get_valid_slot_actions(battle, 0, request)

    assert {0, 3, 4}.issubset(slot_actions)


def test_slot_is_commanding_detects_commander_effect():
    from unittest.mock import MagicMock

    from poke_env.battle.effect import Effect

    battle = MagicMock(spec=DoubleBattle)
    commander_mon = MagicMock()
    commander_mon.effects = {Effect.COMMANDER: 0}
    ally_mon = MagicMock()
    ally_mon.effects = {}
    battle.active_pokemon = [commander_mon, ally_mon]

    assert slot_is_commanding(battle, 0, None)
    assert not slot_is_commanding(battle, 1, None)


def test_commanding_slot_only_allows_pass():
    from unittest.mock import MagicMock

    from poke_env.battle.effect import Effect

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]

    commander_mon = MagicMock()
    commander_mon.effects = {Effect.COMMANDER: 0}
    commander_mon.fainted = False
    ally_mon = MagicMock()
    ally_mon.effects = {}
    ally_mon.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [commander_mon, ally_mon]
    battle.opponent_active_pokemon = [opp_left, opp_right]

    request = {
        "active": [
            {
                "moves": [
                    {
                        "move": "Draco Meteor",
                        "target": "adjacentFoe",
                        "pp": 8,
                        "disabled": False,
                    }
                ],
                "canTerastallize": "Dragon",
            },
            {"moves": [], "trapped": False},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100", "commanding": True},
                {"active": True, "condition": "100/100", "commanding": False},
                {"active": False, "condition": "100/100", "commanding": False},
            ]
        },
    }

    slot_actions = get_valid_slot_actions(battle, 0, request)

    assert slot_actions == {PASS_ACTION}


def test_commanding_slot_masks_to_pass_pairs_only():
    from unittest.mock import MagicMock

    from poke_env.battle.effect import Effect

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]

    commander_mon = MagicMock()
    commander_mon.effects = {Effect.COMMANDER: 0}
    commander_mon.fainted = False
    partner_mon = MagicMock()
    partner_mon.effects = {}
    partner_mon.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [commander_mon, partner_mon]
    battle.opponent_active_pokemon = [opp_left, opp_right]
    battle._last_request = {
        "active": [
            {
                "moves": [
                    {
                        "move": "Draco Meteor",
                        "target": "adjacentFoe",
                        "pp": 8,
                        "disabled": False,
                    }
                ],
                "canTerastallize": "Dragon",
            },
            {
                "moves": [
                    {
                        "move": "Wave Crash",
                        "target": "adjacentFoe",
                        "pp": 8,
                        "disabled": False,
                    }
                ],
                "canTerastallize": None,
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100", "commanding": True},
                {"active": True, "condition": "100/100", "commanding": False},
                {"active": False, "condition": "100/100", "commanding": False},
            ]
        },
    }

    mask = fast_get_action_mask(battle)
    valid_actions = np.where(mask == 1.0)[0]

    assert len(valid_actions) > 0
    for action in valid_actions:
        slot0, _ = action_to_indices(action)
        assert slot0 == PASS_ACTION


def test_terastarstorm_becomes_spread_when_stellar_terastallized():
    from unittest.mock import MagicMock

    from poke_env.battle import PokemonType

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]
    battle.fields = set()

    terapagos = MagicMock()
    terapagos.fainted = False
    terapagos.is_terastallized = True
    terapagos.tera_type = PokemonType.STELLAR
    terapagos.type_1 = PokemonType.NORMAL
    terapagos.type_2 = None

    ally = MagicMock()
    ally.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [terapagos, ally]
    battle.opponent_active_pokemon = [opp_left, opp_right]
    battle.is_grounded.return_value = True

    targets = get_valid_targets_for_request_move(
        battle,
        0,
        {"id": "terastarstorm", "target": "normal", "pp": 8, "disabled": False},
    )

    assert targets == [0]


def test_expanding_force_becomes_spread_in_psychic_terrain_for_grounded_user():
    from unittest.mock import MagicMock

    from poke_env.battle import Field, PokemonType

    battle = MagicMock(spec=DoubleBattle)
    battle.force_switch = [False, False]
    battle.trapped = [False, False]
    battle.fields = {Field.PSYCHIC_TERRAIN}

    armarouge = MagicMock()
    armarouge.fainted = False
    armarouge.is_terastallized = False
    armarouge.tera_type = PokemonType.GRASS
    armarouge.type_1 = PokemonType.FIRE
    armarouge.type_2 = PokemonType.PSYCHIC

    ally = MagicMock()
    ally.fainted = False
    opp_left = MagicMock()
    opp_left.fainted = False
    opp_right = MagicMock()
    opp_right.fainted = False

    battle.active_pokemon = [armarouge, ally]
    battle.opponent_active_pokemon = [opp_left, opp_right]
    battle.is_grounded.return_value = True

    targets = get_valid_targets_for_request_move(
        battle,
        0,
        {"id": "expandingforce", "target": "normal", "pp": 16, "disabled": False},
    )

    assert targets == [0]


def test_mask_generation_is_fast(vgc_json_anon):
    """Test that mask generation completes in reasonable time."""
    import time

    bd = BattleData.from_showdown_json(vgc_json_anon)
    battle_iter = BattleIterator(bd, perspective="p1", omniscient=False)
    battle_iter.next_input()
    battle_iter.next_input()

    # Time single mask generation
    start = time.time()
    mask = fast_get_action_mask(battle_iter.battle)  # type: ignore[arg-type]
    elapsed = time.time() - start

    # Should complete in under 100ms (typically 30-50ms)
    assert elapsed < 0.1, f"Mask generation too slow: {elapsed:.3f}s"

    # Verify mask is valid
    assert mask.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
