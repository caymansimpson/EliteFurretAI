# -*- coding: utf-8 -*-
import itertools
from unittest.mock import MagicMock

from poke_env.battle import Battle, DoubleBattle, Move, Pokemon, Status, Target
from poke_env.player import DoubleBattleOrder, PassBattleOrder, SingleBattleOrder

from elitefurretai.etl.battle_order_validator import is_valid_order


# Create helper functions to represent BattleOrder and DobuleBattleOrder; this will allow us to store
# DoubleBattleOrders in a set, which enables faster lookup when looking for comparisons
def repr_bo(bo: SingleBattleOrder):
    if bo is None:
        return "There is no order"
    main = "None"
    if isinstance(bo.order, Pokemon):
        main = bo.order.species
    elif isinstance(bo.order, Move):
        main = bo.order.id
    return (
        main
        + str(bo.mega)
        + str(bo.z_move)
        + str(bo.dynamax)
        + str(bo.terastallize)
        + str(bo.move_target)
    )


def repr_dbo(dbo: DoubleBattleOrder):
    return repr_bo(dbo.first_order) + repr_bo(dbo.second_order)


def test_is_valid_singles_order(example_singles_request):
    logger = MagicMock()
    battle = Battle("battle-gen8singlesou-1", "username", logger, gen=8)

    battle.parse_request(example_singles_request)

    assert is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma")), battle
    )
    assert is_valid_order(SingleBattleOrder(order=Move("sludgebomb", gen=8)), battle)

    assert not is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="furret")), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma"), terastallize=True),
        battle,
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma"), dynamax=True), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma"), mega=True), battle
    )

    assert not is_valid_order(
        SingleBattleOrder(order=Move("sludgebomb", gen=8), terastallize=True), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Move("sludgebomb", gen=8), dynamax=True), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Move("sludgebomb", gen=8), mega=True), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Move("sludgebomb", gen=8), z_move=True), battle
    )
    assert not is_valid_order(
        SingleBattleOrder(order=Move("icywind", gen=8), z_move=True), battle
    )

    battle.trapped = True
    assert not is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma")), battle
    )

    battle._force_switch = True
    assert is_valid_order(
        SingleBattleOrder(order=Pokemon(gen=8, species="necrozma")), battle
    )
    assert not is_valid_order(SingleBattleOrder(order=Move("sludgebomb", gen=8)), battle)


# This tests all possible orders for a double battle, and their validity under major conditions
# It tests 2v2, force_switch, trapped and 1v1 scenarios. This method really demonstrates how
# much more complicated VGC is as a game
def test_is_valid_doubles_order(example_doubles_request):
    # Initiate battle and create an opponent team
    logger = MagicMock()
    battle = DoubleBattle("battle-gen8doublesou-1", "username", logger, gen=8)
    battle.parse_request(example_doubles_request)
    mon1 = Pokemon(gen=8, species="furret")
    mon1._active = True
    mon2 = Pokemon(gen=8, species="sentret")
    mon2._active = True
    battle._opponent_active_pokemon = {"p2a": mon1, "p2b": mon2}

    # ========================== CREATING ALL POSSIBLE  ORDERS IN A 2v2 BATTLE ==========================
    # Now create all possible DoubleBattleOrders; I purposefully add dummy pokemon and moves to the list
    # to ensure that we can catch bad orders
    possible_orders = (
        list(battle.team.values())
        + [Pokemon(gen=8, species="furret")]
        + battle.available_moves[0]
        + battle.available_moves[1]
    )
    targets = [-2, -1, 0, 1, 2]
    dynamax = [True, False]
    mega = [True, False]
    z_move = [True, False]
    tera = [True, False]

    battleorders = []
    for order, dyn, meg, z, ter, target in itertools.product(
        possible_orders, dynamax, mega, z_move, tera, targets
    ):
        battleorders.append(
            SingleBattleOrder(
                order=order,
                dynamax=dyn,
                mega=meg,
                z_move=z,
                terastallize=ter,
                move_target=target,
            )
        )
    battleorders.append(PassBattleOrder())

    doublebattleorders = [
        DoubleBattleOrder(o1, o2)
        for o1, o2 in itertools.product(battleorders, battleorders)
    ]

    # ========================== CREATING ALL POSSIBLE VALID ORDERS IN A 2v2 BATTLE ==========================
    # Create lists of all of the right moves and switches
    switches: list[list] = [[], []]
    moves: list[list] = [[], []]
    dynamaxes: list[list] = [[], []]
    teras: list[list] = [[], []]
    megas: list[list] = [[], []]
    z_moves: list[list] = [[], []]

    # i is the pokemon index
    for i in range(2):
        # Get switches
        switches[i] = [
            SingleBattleOrder(order=mon) for mon in battle.available_switches[i]
        ]

        # Get normal moves
        for move in battle.available_moves[i]:
            if move.target in (Target.ANY, Target.NORMAL):
                moves[i].append(
                    SingleBattleOrder(
                        order=move,
                        move_target=battle.to_showdown_target(
                            move, battle.active_pokemon[1 - i]
                        ),
                    )
                )
                moves[i].append(SingleBattleOrder(order=move, move_target=1))
                moves[i].append(SingleBattleOrder(order=move, move_target=2))
            elif move.target == Target.ADJACENT_FOE:
                moves[i].append(SingleBattleOrder(order=move, move_target=1))
                moves[i].append(SingleBattleOrder(order=move, move_target=2))
            else:
                moves[i].append(SingleBattleOrder(order=move))

            # Get Dynamax moves
            if battle.can_dynamax[i]:
                if move.base_power > 0:
                    dynamaxes[i].append(
                        SingleBattleOrder(order=move, dynamax=True, move_target=1)
                    )
                    dynamaxes[i].append(
                        SingleBattleOrder(order=move, dynamax=True, move_target=2)
                    )
                    dynamaxes[i].append(
                        SingleBattleOrder(order=move, dynamax=True, move_target=1)
                    )
                    dynamaxes[i].append(
                        SingleBattleOrder(order=move, dynamax=True, move_target=2)
                    )
                else:
                    dynamaxes[i].append(SingleBattleOrder(order=move, dynamax=True))

            # Get Mega moves
            if battle.can_mega_evolve[i]:
                for single_order in moves[i]:
                    megas[i].append(
                        SingleBattleOrder(
                            order=single_order.order,
                            mega=True,
                            dynamax=single_order.dynamax,
                            move_target=single_order.move_target,
                        )
                    )

            # Get Z-Move moves
            if battle.can_z_move[i]:
                if move.base_power > 0:
                    z_moves[i].append(
                        SingleBattleOrder(order=move, z_move=True, move_target=1)
                    )
                    z_moves[i].append(
                        SingleBattleOrder(order=move, z_move=True, move_target=2)
                    )
                    z_moves[i].append(
                        SingleBattleOrder(order=move, z_move=True, move_target=1)
                    )
                    z_moves[i].append(
                        SingleBattleOrder(order=move, z_move=True, move_target=2)
                    )
                else:
                    z_moves[i].append(SingleBattleOrder(order=move, z_move=True))

            # Get Tera moves
            if battle.can_tera[i]:
                for single_order in moves[i]:
                    teras[i].append(
                        SingleBattleOrder(
                            order=single_order.order,
                            terastallize=True,
                            move_target=single_order.target,
                        )
                    )

    # Create set of all valid moves
    valid_moves = set()
    mon1_moves = switches[0] + moves[0] + dynamaxes[0] + teras[0] + megas[0] + z_moves[0]
    mon2_moves = switches[1] + moves[1] + dynamaxes[1] + teras[1] + megas[1] + z_moves[1]
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            if move1.dynamax and move2.dynamax:
                continue
            if move1.terastallize and move2.terastallize:
                continue
            if move1.mega and move2.mega:
                continue
            if move1.z_move and move2.z_move:
                continue
            if isinstance(move1.order, Pokemon) and move1.order == move2.order:
                continue
            valid_moves.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    # Go through all orders
    for dbo in doublebattleorders:
        if repr_dbo(dbo) in valid_moves:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)

    # ========================== TEST TRAPPED SCENARIO ==========================
    battle._trapped = [False, True]
    trapped_valid_moves = set()
    mon1_moves = switches[0] + moves[0] + dynamaxes[0] + teras[0] + megas[0] + z_moves[0]
    mon2_moves = moves[1] + dynamaxes[1] + teras[1] + megas[1] + z_moves[1]
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            if move1.dynamax and move2.dynamax:
                continue
            if move1.terastallize and move2.terastallize:
                continue
            if move1.mega and move2.mega:
                continue
            if move1.z_move and move2.z_move:
                continue
            if isinstance(move1.order, Pokemon) and move1.order == move2.order:
                continue

            trapped_valid_moves.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    for dbo in doublebattleorders:
        if repr_dbo(dbo) in trapped_valid_moves:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)

    # ========================== TEST FORCE SWITCH SCENARIO ==========================
    battle._trapped = [False, False]
    battle._force_switch = [False, True]

    force_switch_valid_moves = set()
    mon1_moves = [PassBattleOrder()]
    mon2_moves = switches[1]
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            force_switch_valid_moves.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    for dbo in doublebattleorders:
        if repr_dbo(dbo) in force_switch_valid_moves:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)

    # ========================== TEST FORCE SWITCH (OTHER MON) SCENARIO ==========================
    battle._force_switch = [True, False]
    mon1_moves = switches[0]
    mon2_moves = [PassBattleOrder()]
    force_switch_other_valid_moves = set()
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            force_switch_other_valid_moves.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    for dbo in doublebattleorders:
        if repr_dbo(dbo) in force_switch_other_valid_moves:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)

    # ========================== TEST ONE OPP MON SCENARIO ==========================
    battle._force_switch = [False, False]
    mon1._status = Status.FNT

    # If there is one mon left, any target on the other side of the field is valid
    for i in range(2):
        for move in battle.available_moves[i]:
            if move.target in (Target.ANY, Target.NORMAL):
                moves[i].append(SingleBattleOrder(order=move, move_target=0))

    opp_one_faint_mon_moves = set()
    mon1_moves = switches[0] + moves[0] + dynamaxes[0] + teras[0] + megas[0] + z_moves[0]
    mon2_moves = switches[1] + moves[1] + dynamaxes[1] + teras[1] + megas[1] + z_moves[1]
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            if move1.dynamax and move2.dynamax:
                continue
            if move1.terastallize and move2.terastallize:
                continue
            if move1.mega and move2.mega:
                continue
            if move1.z_move and move2.z_move:
                continue
            if isinstance(move1.order, Pokemon) and move1.order == move2.order:
                continue
            if move1.move_target == 1 or move2.move_target == 1:
                continue

            opp_one_faint_mon_moves.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    for dbo in doublebattleorders:
        if repr_dbo(dbo) in opp_one_faint_mon_moves:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)

    # ========================== TEST 1v1 SCENARIO ==========================
    my_mon = battle.active_pokemon[0]
    assert my_mon is not None
    my_mon._status = Status.FNT
    battle._available_switches = [[], []]

    moves_1v1 = set()
    mon1_moves = [PassBattleOrder()]
    mon2_moves = moves[1] + dynamaxes[1] + teras[1] + megas[1] + z_moves[1]
    for move1 in mon1_moves:
        for move2 in mon2_moves:
            if move2.move_target == 1:
                continue
            if move2.move_target == -1:
                continue

            moves_1v1.add(repr_dbo(DoubleBattleOrder(move1, move2)))

    for dbo in doublebattleorders:
        if repr_dbo(dbo) in moves_1v1:
            assert is_valid_order(dbo, battle)
        else:
            assert not is_valid_order(dbo, battle)


class TestDoubleForceSwitch:
    """
    Tests for force_switch=[True, True] edge cases.

    When both active Pokemon faint but limited replacements are available,
    one slot must switch and the other must pass.

    See src/elitefurretai/rl/DEBUG_LEARNINGS.md for details.
    """

    def test_double_force_switch_with_one_pokemon_allows_pass(self):
        """
        When force_switch=[True, True] but only 1 Pokemon available,
        one slot can pass because there's nothing to switch to.
        """
        logger = MagicMock()
        battle = DoubleBattle("battle-gen9vgc2023regc-1", "username", logger, gen=9)
        battle._player_role = "p1"  # Required for active_pokemon property

        # Both slots need to force switch
        battle._force_switch = [True, True]

        # Only one Pokemon available (e.g., all others fainted)
        remaining_mon = Pokemon(gen=9, species="dragonite")
        remaining_mon._active = False

        battle._available_switches = [[remaining_mon], [remaining_mon]]
        battle._active_pokemon = {"p1a": None, "p1b": None}

        # switch + pass should be valid
        switch_order = SingleBattleOrder(order=remaining_mon)
        pass_order = PassBattleOrder()

        order1 = DoubleBattleOrder(first_order=switch_order, second_order=pass_order)
        order2 = DoubleBattleOrder(first_order=pass_order, second_order=switch_order)

        assert is_valid_order(order1, battle), "switch, pass should be valid with 1 mon"
        assert is_valid_order(order2, battle), "pass, switch should be valid with 1 mon"

    def test_double_force_switch_with_two_pokemon_requires_both_switch(self):
        """
        When force_switch=[True, True] and 2+ Pokemon available,
        both slots must switch (pass is not valid).
        """
        logger = MagicMock()
        battle = DoubleBattle("battle-gen9vgc2023regc-2", "username", logger, gen=9)
        battle._player_role = "p1"

        battle._force_switch = [True, True]

        mon1 = Pokemon(gen=9, species="amoonguss")
        mon1._active = False
        mon2 = Pokemon(gen=9, species="dragonite")
        mon2._active = False

        battle._available_switches = [[mon1, mon2], [mon1, mon2]]
        battle._active_pokemon = {"p1a": None, "p1b": None}

        # Both switching should be valid
        order_both_switch = DoubleBattleOrder(
            first_order=SingleBattleOrder(order=mon1),
            second_order=SingleBattleOrder(order=mon2),
        )
        assert is_valid_order(order_both_switch, battle)

        # One pass should be invalid when 2 mons available
        order_with_pass = DoubleBattleOrder(
            first_order=SingleBattleOrder(order=mon1), second_order=PassBattleOrder()
        )
        assert not is_valid_order(order_with_pass, battle)

    def test_single_force_switch_pass_behavior(self):
        """
        When force_switch=[False, True], the switching slot cannot pass
        and the non-switching slot must pass.
        """
        logger = MagicMock()
        battle = DoubleBattle("battle-gen9vgc2023regc-3", "username", logger, gen=9)
        battle._player_role = "p1"

        battle._force_switch = [False, True]

        active_mon = Pokemon(gen=9, species="ironhands")
        active_mon._active = True

        switch_target = Pokemon(gen=9, species="amoonguss")
        switch_target._active = False

        battle._available_switches = [[], [switch_target]]
        battle._active_pokemon = {"p1a": active_mon, "p1b": None}

        # Correct: pass for non-switching slot, switch for switching slot
        valid_order = DoubleBattleOrder(
            first_order=PassBattleOrder(),
            second_order=SingleBattleOrder(order=switch_target),
        )
        assert is_valid_order(valid_order, battle)

        # Invalid: switching slot tries to pass
        invalid_order = DoubleBattleOrder(
            first_order=PassBattleOrder(), second_order=PassBattleOrder()
        )
        assert not is_valid_order(invalid_order, battle)
