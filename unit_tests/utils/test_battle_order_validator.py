# -*- coding: utf-8 -*-
import itertools
from unittest.mock import MagicMock

from poke_env.environment import Battle, DoubleBattle, Move, Pokemon, Status, Target
from poke_env.player import BattleOrder, DoubleBattleOrder

from elitefurretai.utils.battle_order_validator import is_valid_order


def test_is_valid_singles_order(example_singles_request):
    logger = MagicMock()
    battle = Battle("battle-gen8singlesou-1", "username", logger, gen=8)

    battle.parse_request(example_singles_request)

    assert is_valid_order(BattleOrder(order=Pokemon(gen=8, species="necrozma")), battle)
    assert is_valid_order(BattleOrder(order=Move("sludgebomb", gen=8)), battle)

    assert not is_valid_order(BattleOrder(order=Pokemon(gen=8, species="furret")), battle)
    assert not is_valid_order(
        BattleOrder(order=Pokemon(gen=8, species="necrozma"), terastallize=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Pokemon(gen=8, species="necrozma"), dynamax=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Pokemon(gen=8, species="necrozma"), mega=True), battle
    )

    assert not is_valid_order(
        BattleOrder(order=Move("sludgebomb", gen=8), terastallize=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Move("sludgebomb", gen=8), dynamax=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Move("sludgebomb", gen=8), mega=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Move("sludgebomb", gen=8), z_move=True), battle
    )
    assert not is_valid_order(
        BattleOrder(order=Move("icywind", gen=8), z_move=True), battle
    )

    battle.trapped = True
    assert not is_valid_order(
        BattleOrder(order=Pokemon(gen=8, species="necrozma")), battle
    )

    battle._force_switch = True
    assert is_valid_order(BattleOrder(order=Pokemon(gen=8, species="necrozma")), battle)
    assert not is_valid_order(BattleOrder(order=Move("sludgebomb", gen=8)), battle)

    assert not is_valid_order(BattleOrder(order=None), battle)
    assert not is_valid_order(None, battle)


# TODO: make faster
# This tests all possible orders for a double battle, and their validity under major conditions
# It tests 2v2, force_switch, trapped and 1v1 scenarios. This method really demonstrates how
# much more complicated VGC is as a game
def test_is_valid_doubles_order(example_doubles_request):

    # Create helper functions to represent BattleOrder and DobuleBattleOrder; this will allow us to store
    # DoubleBattleOrders in a set, which enables faster lookup when looking for comparisons
    def repr_bo(bo: BattleOrder) -> str:
        if bo is None:
            return "None"
        return "BattleOrder({o}, mega={m}, z_move={z}, dynamax={d}, terastallize={t}, move_target={mt})".format(
            o=bo.order if bo.order else "None",
            m=bo.mega,
            z=bo.z_move,
            d=bo.dynamax,
            t=bo.terastallize,
            mt=bo.move_target,
        )

    def repr_dbo(dbo: DoubleBattleOrder) -> str:
        return "DoubleBattleOrder(\n\tFirst Order: {first}\n\tSecond Order: {second}\n)".format(
            first=repr_bo(dbo.first_order), second=repr_bo(dbo.second_order)
        )

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

    battleorders = [None]
    for order, dynamax, mega, z_move, tera, target in itertools.product(
        possible_orders, dynamax, mega, z_move, tera, targets
    ):
        battleorders.append(
            BattleOrder(
                order=order,
                dynamax=dynamax,
                mega=mega,
                z_move=z_move,
                terastallize=tera,
                move_target=target,
            )
        )

    doublebattleorders = [
        DoubleBattleOrder(o1, o2)
        for o1, o2 in itertools.product(battleorders, battleorders)
    ]

    # ========================== CREATING ALL POSSIBLE VALID ORDERS IN A 2v2 BATTLE ==========================
    # Create lists of all of the right moves and switches
    switches, moves, dynamaxes, teras, megas, z_moves = (
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
    )

    # i is the pokemon index
    for i in range(2):

        # Get switches
        switches[i] = [BattleOrder(order=mon) for mon in battle.available_switches[i]]

        # Get normal moves
        for move in battle.available_moves[i]:
            if move.target in (Target.ANY, Target.NORMAL):
                moves[i].append(
                    BattleOrder(
                        order=move,
                        move_target=battle.to_showdown_target(
                            move, battle.active_pokemon[1 - i]
                        ),
                    )
                )
                moves[i].append(BattleOrder(order=move, move_target=1))
                moves[i].append(BattleOrder(order=move, move_target=2))
            elif move.target == Target.ADJACENT_FOE:
                moves[i].append(BattleOrder(order=move, move_target=1))
                moves[i].append(BattleOrder(order=move, move_target=2))
            else:
                moves[i].append(BattleOrder(order=move))

            # Get Dynamax moves
            if battle.can_dynamax[i]:
                if move.base_power > 0:
                    dynamaxes[i].append(
                        BattleOrder(order=move, dynamax=True, move_target=1)
                    )
                    dynamaxes[i].append(
                        BattleOrder(order=move, dynamax=True, move_target=2)
                    )
                    dynamaxes[i].append(
                        BattleOrder(order=move, dynamax=True, move_target=1)
                    )
                    dynamaxes[i].append(
                        BattleOrder(order=move, dynamax=True, move_target=2)
                    )
                else:
                    dynamaxes[i].append(BattleOrder(order=move, dynamax=True))

            # Get Mega moves
            if battle.can_mega_evolve[i]:
                for order in moves[i]:
                    megas[i].append(
                        BattleOrder(
                            order=order.order,
                            mega=True,
                            dynamax=order.dynamax,
                            move_target=order.target,
                        )
                    )

            # Get Z-Move moves
            if battle.can_z_move[i]:
                if move.base_power > 0:
                    z_moves[i].append(BattleOrder(order=move, z_move=True, move_target=1))
                    z_moves[i].append(BattleOrder(order=move, z_move=True, move_target=2))
                    z_moves[i].append(BattleOrder(order=move, z_move=True, move_target=1))
                    z_moves[i].append(BattleOrder(order=move, z_move=True, move_target=2))
                else:
                    z_moves[i].append(BattleOrder(order=move, z_move=True))

            # Get Tera moves
            if battle.can_tera[i]:
                for order in moves[i]:
                    teras[i].append(
                        BattleOrder(
                            order=order.order, terastallize=True, move_target=order.target
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
    mon1_moves = [None]
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
    mon2_moves = [None]
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
    my_mon._status = Status.FNT
    battle._available_switches = [[], []]

    moves_1v1 = set()
    mon1_moves = [None]
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
