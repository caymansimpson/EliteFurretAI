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


class TestRequestSnapshotDecoding:
    def test_request_override_keeps_original_move_slot_mapping(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        original_move = Move("uproar", gen=9)
        mutated_move = Move("protect", gen=9)
        ally_move = Move("knockoff", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "farigiraf"
        active_mon.moves = {
            "uproar": original_move,
            "protect": mutated_move,
        }
        active_mon.available_moves_from_request.side_effect = lambda request: [
            active_mon.moves[request["moves"][0]["id"]]
        ]

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "incineroar"
        ally_mon.moves = {"knockoff": ally_move}
        ally_mon.available_moves_from_request.side_effect = lambda request: [
            ally_mon.moves[request["moves"][0]["id"]]
        ]

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[original_move], [ally_move]]
        battle.team = {}
        battle.last_request = {
            "active": [
                {"moves": [{"move": "Protect", "id": "protect", "target": "self"}]},
                {"moves": [{"move": "Knock Off", "id": "knockoff", "target": "normal"}]},
            ]
        }

        request_snapshot = {
            "active": [
                {"moves": [{"move": "Uproar", "id": "uproar", "target": "normal"}]},
                {"moves": [{"move": "Knock Off", "id": "knockoff", "target": "normal"}]},
            ]
        }

        mdbo = MDBO(MDBO.TURN, "/choose move 1 -2, move 1 1")
        order = mdbo.to_double_battle_order(battle, request=request_snapshot)

        assert order.first_order.order.id == "uproar"  # type: ignore[attr-defined, union-attr]


class TestMoveOrderInvariantDecoding:
    """
    Tests that MDBO decoding follows the pokemon moveset order invariant.

    The source fix is to maintain `Pokemon.moves` in request order inside poke-env.
    MDBO decoding should then be able to index `moving_mon.moves` directly.
    """

    def test_slot_three_decodes_to_protect_when_moves_are_request_ordered(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        behemoth_bash = Move("behemothbash", gen=9)
        body_press = Move("bodypress", gen=9)
        protect = Move("protect", gen=9)
        iron_defense = Move("irondefense", gen=9)
        ally_move = Move("tailwind", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "zamazentacrowned"
        active_mon.moves = {
            "behemothbash": behemoth_bash,
            "bodypress": body_press,
            "protect": protect,
            "irondefense": iron_defense,
        }

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "moltresgalar"
        ally_mon.moves = {"tailwind": ally_move}

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [
            [behemoth_bash, body_press, protect, iron_defense],
            [ally_move],
        ]
        battle.last_request = None
        battle.valid_orders = [
            [
                MagicMock(message="/choose move behemothbash 1"),
                MagicMock(message="/choose move behemothbash 2"),
                MagicMock(message="/choose move bodypress 1"),
                MagicMock(message="/choose move bodypress 2"),
                MagicMock(message="/choose move protect"),
                MagicMock(message="/choose move irondefense"),
            ],
            [MagicMock(message="/choose move tailwind")],
        ]

        mdbo = MDBO(MDBO.TURN, "/choose move 3, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "protect"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == 0  # type: ignore[attr-defined, union-attr]
        assert order.first_order.message == "/choose move protect"  # type: ignore[attr-defined, union-attr]


class TestRestrictedRequestMoveDecoding:
    def test_single_legal_request_move_uses_request_slot_index(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        iron_head = Move("ironhead", gen=9)
        aerial_ace = Move("aerialace", gen=9)
        extreme_speed = Move("extremespeed", gen=9)
        outrage = Move("outrage", gen=9)
        protect = Move("protect", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "dragonite"
        active_mon.moves = {
            "ironhead": iron_head,
            "aerialace": aerial_ace,
            "extremespeed": extreme_speed,
            "outrage": outrage,
        }
        active_mon.available_moves_from_request.return_value = [outrage]

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "dondozo"
        ally_mon.moves = {"protect": protect}
        ally_mon.available_moves_from_request.return_value = [protect]

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[outrage], [protect]]
        battle.last_request = {
            "active": [
                {"moves": [{"move": "Outrage", "id": "outrage"}], "trapped": True},
                {
                    "moves": [
                        {"move": "Protect", "id": "protect", "disabled": False, "pp": 16}
                    ]
                },
            ]
        }

        mdbo = MDBO(MDBO.TURN, "/choose move 1 2, move 1")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "outrage"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == 2  # type: ignore[attr-defined, union-attr]

    def test_partial_request_move_subset_uses_request_order(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        dragon_claw = Move("dragonclaw", gen=9)
        stomping_tantrum = Move("stompingtantrum", gen=9)
        protect = Move("protect", gen=9)
        heavy_slam = Move("heavyslam", gen=9)
        ally_move = Move("tailwind", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "archaludon"
        active_mon.moves = {
            "dragonclaw": dragon_claw,
            "stompingtantrum": stomping_tantrum,
            "protect": protect,
            "heavyslam": heavy_slam,
        }
        active_mon.available_moves_from_request.return_value = [heavy_slam]

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "tornadus"
        ally_mon.moves = {"tailwind": ally_move}
        ally_mon.available_moves_from_request.return_value = [ally_move]

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[dragon_claw, heavy_slam], [ally_move]]
        battle.last_request = {
            "active": [
                {
                    "moves": [
                        {
                            "move": "Dragon Claw",
                            "id": "dragonclaw",
                            "disabled": False,
                            "pp": 16,
                        },
                        {
                            "move": "Heavy Slam",
                            "id": "heavyslam",
                            "disabled": False,
                            "pp": 16,
                        },
                    ]
                },
                {
                    "moves": [
                        {"move": "Tailwind", "id": "tailwind", "disabled": False, "pp": 24}
                    ]
                },
            ]
        }

        mdbo = MDBO(MDBO.TURN, "/choose move 2 1, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "heavyslam"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == 1  # type: ignore[attr-defined, union-attr]

    def test_disabled_request_slot_preserves_raw_slot_index(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"
        battle.gen = 9

        protect = Move("protect", gen=9)
        draco_meteor = Move("dracometeor", gen=9)
        ally_move = Move("tailwind", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "tatsugiri"
        active_mon.moves = {
            "protect": protect,
            "dracometeor": draco_meteor,
        }
        active_mon.available_moves_from_request.side_effect = [
            [draco_meteor],
            [ally_move],
        ]

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "murkrow"
        ally_mon.moves = {"tailwind": ally_move}
        ally_mon.available_moves_from_request.return_value = [ally_move]

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[draco_meteor], [ally_move]]
        battle.last_request = {
            "active": [
                {
                    "moves": [
                        {"move": "Protect", "id": "protect", "disabled": True, "pp": 16},
                        {
                            "move": "Draco Meteor",
                            "id": "dracometeor",
                            "disabled": False,
                            "pp": 8,
                            "target": "adjacentFoe",
                        },
                    ]
                },
                {
                    "moves": [
                        {"move": "Tailwind", "id": "tailwind", "disabled": False, "pp": 24}
                    ]
                },
            ]
        }

        mdbo = MDBO(MDBO.TURN, "/choose move 2 1, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "dracometeor"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == 1  # type: ignore[attr-defined, union-attr]

    def test_request_slot_decoding_preserves_targeted_ally_move(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"
        battle.gen = 9

        helping_hand = Move("helpinghand", gen=9)
        ally_move = Move("surf", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "furret"
        active_mon.moves = {"helpinghand": helping_hand}
        active_mon.available_moves_from_request.return_value = [helping_hand]

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "pelipper"
        ally_mon.moves = {"surf": ally_move}
        ally_mon.available_moves_from_request.return_value = [ally_move]

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [[helping_hand], [ally_move]]
        battle.last_request = {
            "active": [
                {
                    "moves": [
                        {
                            "move": "Helping Hand",
                            "id": "helpinghand",
                            "disabled": False,
                            "pp": 32,
                            "target": "adjacentAlly",
                        }
                    ]
                },
                {"moves": [{"move": "Surf", "id": "surf", "disabled": False, "pp": 24}]},
            ]
        }

        mdbo = MDBO(MDBO.TURN, "/choose move 1 -2, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "helpinghand"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == -2  # type: ignore[attr-defined, union-attr]

    def test_slot_one_targeted_move_stays_targeted_when_moves_are_request_ordered(self):
        battle = MagicMock(spec=DoubleBattle)
        battle.player_role = "p1"

        behemoth_bash = Move("behemothbash", gen=9)
        body_press = Move("bodypress", gen=9)
        protect = Move("protect", gen=9)
        iron_defense = Move("irondefense", gen=9)
        ally_move = Move("tailwind", gen=9)

        active_mon = MagicMock(spec=Pokemon)
        active_mon.species = "zamazentacrowned"
        active_mon.moves = {
            "behemothbash": behemoth_bash,
            "bodypress": body_press,
            "protect": protect,
            "irondefense": iron_defense,
        }

        ally_mon = MagicMock(spec=Pokemon)
        ally_mon.species = "moltresgalar"
        ally_mon.moves = {"tailwind": ally_move}

        battle.active_pokemon = [active_mon, ally_mon]
        battle.available_moves = [
            [behemoth_bash, body_press, protect, iron_defense],
            [ally_move],
        ]
        battle.last_request = None
        battle.valid_orders = [
            [
                MagicMock(message="/choose move behemothbash 1"),
                MagicMock(message="/choose move behemothbash 2"),
                MagicMock(message="/choose move bodypress 1"),
                MagicMock(message="/choose move bodypress 2"),
                MagicMock(message="/choose move protect"),
                MagicMock(message="/choose move irondefense"),
            ],
            [MagicMock(message="/choose move tailwind")],
        ]

        mdbo = MDBO(MDBO.TURN, "/choose move 1 2, pass")
        order = mdbo.to_double_battle_order(battle)

        assert order.first_order.order.id == "behemothbash"  # type: ignore[attr-defined, union-attr]
        assert order.first_order.move_target == 2  # type: ignore[attr-defined, union-attr]
        assert order.first_order.message == "/choose move behemothbash 2"  # type: ignore[attr-defined, union-attr]


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
