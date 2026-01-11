# -*- coding: utf-8 -*-
"""
Unit tests for MDBO encoder edge cases discovered during RL training debugging.

These tests verify:
1. Switch encoding uses request order (not battle.team dict order)
2. Struggle/recharge moves are handled correctly
3. MDBO roundtrip conversions work for all edge cases

See src/elitefurretai/rl/DEBUG_LEARNINGS.md for detailed explanations.
"""

from unittest.mock import MagicMock

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player.battle_order import (
    DoubleBattleOrder,
)

from elitefurretai.etl import MDBO


class TestSwitchEncodingWithRequestOrder:
    """
    Tests that switch encoding uses battle.last_request order, not battle.team order.

    Root cause: After teampreview, Showdown reorders side.pokemon so positions 1-4
    are the brought Pokemon, but battle.team dict keeps original teampreview order.
    """

    def test_switch_uses_request_order_not_team_order(self):
        """
        Verify switch target is looked up from request, not team dict.

        Scenario: Team dict has Pokemon in order [A, B, C, D, E, F]
        but request has them as [C, A, E, B] (only 4 brought, reordered).
        "switch 1" should give C, not A.
        """
        # Create mock battle
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        # Create Pokemon objects
        mon_a = MagicMock(spec=Pokemon)
        mon_a.species = "amoonguss"
        mon_b = MagicMock(spec=Pokemon)
        mon_b.species = "dragonite"
        mon_c = MagicMock(spec=Pokemon)
        mon_c.species = "ironhands"
        mon_d = MagicMock(spec=Pokemon)
        mon_d.species = "palafin"

        # Team dict has original teampreview order
        battle.team = {
            "p1: Amoonguss": mon_a,
            "p1: Dragonite": mon_b,
            "p1: Iron Hands": mon_c,
            "p1: Palafin": mon_d,
        }

        # Request has different order (Iron Hands first, Amoonguss second)
        battle.last_request = {
            "side": {
                "pokemon": [
                    {"ident": "p1: Iron Hands"},
                    {"ident": "p1: Amoonguss"},
                    {"ident": "p1: Dragonite"},
                    {"ident": "p1: Palafin"},
                ]
            }
        }

        # Active pokemon for the move conversion part
        battle.active_pokemon = [mon_c, mon_a]
        battle.available_moves = [[], []]

        # "switch 1" should get Iron Hands (from request order), not Amoonguss (from dict order)
        mdbo = MDBO(MDBO.FORCE_SWITCH, "/choose switch 1, pass")
        order = mdbo.to_double_battle_order(battle)

        assert isinstance(order, DoubleBattleOrder)
        assert isinstance(order.first_order.order, Pokemon)
        assert order.first_order.order.species == "ironhands"

    def test_switch_3_with_mismatched_orders(self):
        """
        Test switch 3 uses third position in request, not team dict.
        """
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        mon_a = MagicMock(spec=Pokemon)
        mon_a.species = "growlithe"
        mon_b = MagicMock(spec=Pokemon)
        mon_b.species = "fletchling"
        mon_c = MagicMock(spec=Pokemon)
        mon_c.species = "palafin"
        mon_d = MagicMock(spec=Pokemon)
        mon_d.species = "dragonite"

        # Team dict order
        battle.team = {
            "p1: Growlithe": mon_a,
            "p1: Fletchling": mon_b,
            "p1: Palafin": mon_c,
            "p1: Dragonite": mon_d,
        }

        # Request order is different
        battle.last_request = {
            "side": {
                "pokemon": [
                    {"ident": "p1: Dragonite"},
                    {"ident": "p1: Palafin"},
                    {"ident": "p1: Growlithe"},
                    {"ident": "p1: Fletchling"},
                ]
            }
        }

        battle.active_pokemon = [mon_d, mon_c]
        battle.available_moves = [[], []]

        # "switch 3" should give Growlithe (position 3 in request)
        mdbo = MDBO(MDBO.FORCE_SWITCH, "/choose pass, switch 3")
        order = mdbo.to_double_battle_order(battle)

        assert isinstance(order.second_order.order, Pokemon)  # type: ignore[attr-defined]
        assert order.second_order.order.species == "growlithe"  # type: ignore[attr-defined, union-attr]


class TestStruggleRechargeHandling:
    """
    Tests that struggle/recharge moves are handled correctly.

    Root cause: These moves appear in available_moves but not in Pokemon.moves dict.
    """

    def test_struggle_uses_available_moves(self):
        """
        When only struggle is available, use it directly from available_moves.
        """
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"
        battle.last_request = None  # No request needed for move tests

        # Create Pokemon and struggle move
        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "amoonguss"
        active_mon.moves = {
            "pollenpuff": MagicMock(spec=Move),
            "spore": MagicMock(spec=Move),
        }

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "dragonite"
        ally_mon.moves = {"extremespeed": MagicMock(spec=Move)}

        struggle = MagicMock(spec=Move)
        struggle.id = "struggle"

        protect = MagicMock(spec=Move)
        protect.id = "protect"

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [
            [struggle],
            [protect],
        ]  # Only struggle available for slot 0

        # "move 1" for slot 0 should use struggle from available_moves
        mdbo = MDBO(MDBO.TURN, "/choose move 1, move 1")
        order = mdbo.to_double_battle_order(battle)

        assert isinstance(order.first_order.order, Move)  # type: ignore[attr-defined]
        assert order.first_order.order.id == "struggle"  # type: ignore[attr-defined, union-attr]

    def test_recharge_uses_available_moves(self):
        """
        When only recharge is available (after Hyper Beam), use it directly.
        """
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"
        battle.last_request = None

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "dragonite"
        active_mon.moves = {"hyperbeam": MagicMock(spec=Move)}

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "amoonguss"
        ally_mon.moves = {"spore": MagicMock(spec=Move)}

        recharge = MagicMock(spec=Move)
        recharge.id = "recharge"

        spore = MagicMock(spec=Move)
        spore.id = "spore"

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[recharge], [spore]]

        mdbo = MDBO(MDBO.TURN, "/choose move 1, move 1")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "recharge"  # type: ignore[attr-defined, union-attr]

    def test_normal_move_uses_pokemon_moveset(self):
        """
        When multiple moves available, use Pokemon's moveset for lookup.
        """
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"
        battle.last_request = None

        move1 = MagicMock(spec=Move)
        move1.id = "drainpunch"
        move2 = MagicMock(spec=Move)
        move2.id = "wildcharge"

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "ironhands"
        active_mon.moves = {"drainpunch": move1, "wildcharge": move2}

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "amoonguss"
        ally_mon.moves = {"spore": MagicMock(spec=Move)}

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[move1, move2], []]  # Multiple moves = normal case

        # "move 2" should use second move from Pokemon's moveset
        mdbo = MDBO(MDBO.TURN, "/choose move 2, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "wildcharge"  # type: ignore[attr-defined, union-attr]


class TestMDBORoundtrip:
    """
    Tests that MDBO encoding/decoding roundtrips correctly for edge cases.
    """

    def test_force_switch_roundtrip(self):
        """Test force_switch orders encode and decode correctly."""
        test_cases = [
            "/choose switch 1, pass",
            "/choose switch 2, pass",
            "/choose switch 3, pass",
            "/choose switch 4, pass",
            "/choose pass, switch 1",
            "/choose pass, switch 2",
            "/choose pass, switch 3",
            "/choose pass, switch 4",
            "/choose switch 3, switch 4",
            "/choose switch 4, switch 3",
        ]

        for msg in test_cases:
            mdbo = MDBO(MDBO.FORCE_SWITCH, msg)
            int_val = mdbo.to_int()
            reconstructed = MDBO.from_int(int_val, MDBO.FORCE_SWITCH)
            assert reconstructed.message == msg, f"Roundtrip failed for {msg}"

    def test_switch_action_indices(self):
        """Verify switch actions map to expected indices."""
        # Single slot actions: 40-43 are switch 1-4
        for switch_idx in range(1, 5):
            msg = f"/choose switch {switch_idx}, pass"
            mdbo = MDBO(MDBO.FORCE_SWITCH, msg)
            int_val = mdbo.to_int()

            # First slot action is (40 + switch_idx - 1), second is pass (44)
            expected_first = 40 + switch_idx - 1
            expected = expected_first * 45 + 44  # first * 45 + second
            assert int_val == expected, (
                f"switch {switch_idx} mapped to {int_val}, expected {expected}"
            )
