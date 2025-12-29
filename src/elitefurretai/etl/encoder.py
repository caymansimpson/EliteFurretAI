# -*- coding: utf-8 -*-
"""
encoder.py

This module defines the MDBO (Model Double Battle Order) class and related utilities for encoding,
decoding, and manipulating Pokémon Showdown double battle orders for use in machine learning models.

Key features:
- Maps between integer representations and string protocol orders for double battles.
- Handles teampreview, force switch, and turn orders.
- Provides conversion between model-friendly integer actions and Showdown protocol strings.
- Supports conversion to and from PokéEnv's BattleOrder/DoubleBattleOrder objects.
- Encodes all possible move/switch/terastallize combinations for both active Pokémon.

It also defines the move_encoder that encodes the move order of pokemon for a classification task
"""

import itertools
import re
from typing import List, Union

from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)

# Mapping from integer to order string for all possible move/switch/terastallize actions
_INT_TO_ORDER_MAPPINGS = {
    0: "move 1 -2",
    1: "move 1 -1",
    2: "move 1",
    3: "move 1 1",
    4: "move 1 2",
    5: "move 1 -2 terastallize",
    6: "move 1 -1 terastallize",
    7: "move 1 terastallize",
    8: "move 1 1 terastallize",
    9: "move 1 2 terastallize",
    10: "move 2 -2",
    11: "move 2 -1",
    12: "move 2",
    13: "move 2 1",
    14: "move 2 2",
    15: "move 2 -2 terastallize",
    16: "move 2 -1 terastallize",
    17: "move 2 terastallize",
    18: "move 2 1 terastallize",
    19: "move 2 2 terastallize",
    20: "move 3 -2",
    21: "move 3 -1",
    22: "move 3",
    23: "move 3 1",
    24: "move 3 2",
    25: "move 3 -2 terastallize",
    26: "move 3 -1 terastallize",
    27: "move 3 terastallize",
    28: "move 3 1 terastallize",
    29: "move 3 2 terastallize",
    30: "move 4 -2",
    31: "move 4 -1",
    32: "move 4",
    33: "move 4 1",
    34: "move 4 2",
    35: "move 4 -2 terastallize",
    36: "move 4 -1 terastallize",
    37: "move 4 terastallize",
    38: "move 4 1 terastallize",
    39: "move 4 2 terastallize",
    40: "switch 1",
    41: "switch 2",
    42: "switch 3",
    43: "switch 4",
    44: "pass",
}

# Reverse mapping from order string to integer
_ORDER_MAPPINGS_TO_INT = {v: k for k, v in _INT_TO_ORDER_MAPPINGS.items()}

# All possible teampreview orderings (choose 4 out of 6, order matters, but treat pairs the same)
_INT_TO_TEAMPREVIEW_ORDER = {
    i: "".join([str(x) for x in permutation])
    for i, permutation in enumerate(
        [v for v in itertools.permutations(range(1, 7), 4) if v[3] > v[2] and v[1] > v[0]]
    )
}
_TEAMPREVIEW_ORDER_TO_INT = {v: k for k, v in _INT_TO_TEAMPREVIEW_ORDER.items()}


# Model Double Battle Order
class MDBO(BattleOrder):
    """
    Encodes and decodes double battle orders for use in ML models.
    Supports teampreview, force switch, turn, and default orders.
    """

    # Order types
    FORCE_SWITCH = "force_switch"
    TEAMPREVIEW = "teampreview"
    TURN = "turn"
    DEFAULT = "default"

    def __init__(self, type: str, msg: str = ""):
        """
        Initialize an MDBO object.
        type: one of FORCE_SWITCH, TEAMPREVIEW, TURN, DEFAULT
        msg: showdown protocol string (e.g., "/choose move 1, move 2")
        """
        assert type in [
            self.FORCE_SWITCH,
            self.TEAMPREVIEW,
            self.TURN,
            self.DEFAULT,
        ], "mdbo type not recognized"
        self._type = type
        self._msg = msg

        # For teampreview, standardize the order so that pairs are treated the same
        if self._type == MDBO.TEAMPREVIEW:
            msg = "".join(char for char in msg.replace("/team ", "") if char.isdigit())
            first_less = msg[0] if msg[0] < msg[1] else msg[1]
            first_most = msg[0] if msg[0] > msg[1] else msg[1]
            second_less = msg[2] if msg[2] < msg[3] else msg[3]
            second_most = msg[2] if msg[2] > msg[3] else msg[3]
            self._msg = f"/team {first_less}{first_most}{second_less}{second_most}"

    @property
    def message(self) -> str:
        """
        Returns the showdown protocol string for this order.
        """
        if self._type == self.DEFAULT:
            return "/choose default"
        else:
            assert self._msg is not None
            return self._msg

    def to_int(self) -> int:
        """
        Converts this MDBO to an integer for model use.
        """
        if self._type == self.DEFAULT:
            return -1

        assert (
            self._msg is not None
        ), "msg is None and not default; cant turn it into an int"
        if self._type == self.TEAMPREVIEW:
            # Remove spaces/commas and look up in teampreview mapping
            key = re.sub(r"[\s,]", "", self._msg[5:])
            assert (
                key in _TEAMPREVIEW_ORDER_TO_INT
            ), f"{key} not in TEAMPREVIEW_ORDER_TO_INT. Message: {self._msg}"
            return _TEAMPREVIEW_ORDER_TO_INT[key]
        else:
            # For turn/force_switch: parse both orders and encode as a single int
            orders = self._msg[8:].split(", ")
            assert (
                len(orders) == 2
            ), f"Only got one order for a double battle order: {self._msg}"
            assert (
                orders[0] in _ORDER_MAPPINGS_TO_INT
            ), f"{orders[0]} not in ORDER_MAPPINGS_TO_INT. Label: {self._msg}"
            assert (
                orders[1] in _ORDER_MAPPINGS_TO_INT
            ), f"{orders[1]} not in ORDER_MAPPINGS_TO_INT. Label: {self._msg}"
            return (
                _ORDER_MAPPINGS_TO_INT[orders[0]] * len(_ORDER_MAPPINGS_TO_INT)
                + _ORDER_MAPPINGS_TO_INT[orders[1]]
            )

    def to_double_battle_order(
        self, battle: DoubleBattle
    ) -> Union[DoubleBattleOrder, DefaultBattleOrder]:
        """
        Converts this MDBO to a PyPokéEnv DoubleBattleOrder or DefaultBattleOrder.
        """
        assert not self.message.startswith(
            "/team"
        ), "MDBO cannot be converted into DBO when it is a teampreview"
        assert (
            battle.player_role is not None
        ), "Cannot convert MDBO to DBO when player_role is None because we assume we have the right perspective"
        orders: List[SingleBattleOrder] = []
        for i, order in enumerate(self.message.replace("/choose ", "").split(", ")):
            if order == "pass":
                orders.append(PassBattleOrder())
            elif order == "default":
                return DefaultBattleOrder()
            elif order.startswith("switch "):
                # Switch to a specific Pokémon in the team
                key = list(battle.team.keys())[int(order[7]) - 1]
                orders.append(SingleBattleOrder(order=battle.team[key]))

            # move
            else:
                moving_mon = battle.active_pokemon[i]
                assert (
                    moving_mon is not None
                ), "Cannot convert an order for a pokemon when it is not active"
                move_key = list(moving_mon.moves.keys())[int(order[5]) - 1]

                target = DoubleBattle.EMPTY_TARGET_POSITION
                if len(order) > 7 and order.replace(" terastallize", "")[7:] in [
                    "-1",
                    "-2",
                    "1",
                    "2",
                ]:
                    target = int(order.replace(" terastallize", "")[7:])

                orders.append(
                    SingleBattleOrder(
                        order=moving_mon.moves[move_key],
                        terastallize="terastallize" in order,
                        move_target=target,
                    )
                )

        return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])

    @staticmethod
    def action_space() -> int:
        """
        Returns the total number of possible double battle orders (for model output size).
        """
        return len(_INT_TO_ORDER_MAPPINGS) * len(_INT_TO_ORDER_MAPPINGS)

    @staticmethod
    def teampreview_space() -> int:
        """
        Returns the total number of possible teampreview orders (for model output size).
        """
        return len(_INT_TO_TEAMPREVIEW_ORDER)

    @staticmethod
    def from_int(i: int, type: str):
        """
        Converts an integer and type back to an MDBO object.
        """
        assert type in [MDBO.FORCE_SWITCH, MDBO.TEAMPREVIEW, MDBO.TURN, MDBO.DEFAULT]
        if i == -1:
            return MDBO(MDBO.DEFAULT)
        elif type == MDBO.TEAMPREVIEW:
            assert i < len(_INT_TO_TEAMPREVIEW_ORDER)
            return MDBO(type, "/team " + _INT_TO_TEAMPREVIEW_ORDER[i])
        else:
            assert i < len(_INT_TO_ORDER_MAPPINGS) * len(_INT_TO_ORDER_MAPPINGS)
            order_1 = _INT_TO_ORDER_MAPPINGS[i // len(_INT_TO_ORDER_MAPPINGS)]
            order_2 = _INT_TO_ORDER_MAPPINGS[i % len(_INT_TO_ORDER_MAPPINGS)]
            return MDBO(type, "/choose " + order_1 + ", " + order_2)


_MOVE_ORDERS = (
    list(itertools.permutations(["p1a", "p1b", "p2a", "p2b"], 4))
    + [x + ("",) for x in itertools.permutations(["p1a", "p1b", "p2a", "p2b"], 3)]
    + [x + ("", "") for x in itertools.permutations(["p1a", "p1b", "p2a", "p2b"], 2)]
    + [x + ("", "", "") for x in itertools.permutations(["p1a", "p1b", "p2a", "p2b"], 1)]
    + [("", "", "", "")]
)
_MOVE_ORDER_MAPPINGS = {order: i for i, order in enumerate(_MOVE_ORDERS)}


class MoveOrderEncoder:
    @staticmethod
    def action_space() -> int:
        """
        Returns the total number of possible move orders (for model output size).
        """
        return len(_MOVE_ORDER_MAPPINGS)

    @staticmethod
    def encode(move_order: List[str]) -> int:
        """
        Encodes a list of up to 4 Pokémon IDs into an integer representing the move order.

        move_order: List of Pokémon IDs (up to 4; eg [p1a, p2b, p3a])
        """
        assert len(move_order) <= 4, "move_order can have at most 4 Pokémon"
        if len(move_order) < 4:
            move_order = move_order + [""] * (
                4 - len(move_order)
            )  # Pad with empty strings
        order: tuple[str, str, str, str] = (
            move_order[0],
            move_order[1],
            move_order[2],
            move_order[3],
        )
        return _MOVE_ORDER_MAPPINGS.get(order, -1)
