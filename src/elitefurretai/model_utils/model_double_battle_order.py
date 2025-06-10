import itertools
import re
from typing import Optional, Union

from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)

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

_ORDER_MAPPINGS_TO_INT = {v: k for k, v in _INT_TO_ORDER_MAPPINGS.items()}

_INT_TO_TEAMPREVIEW_ORDER = {
    i: "".join([str(x) for x in permutation])
    for i, permutation in enumerate(itertools.permutations(range(1, 7), 4))
}

_TEAMPREVIEW_ORDER_TO_INT = {v: k for k, v in _INT_TO_TEAMPREVIEW_ORDER.items()}


# Model Double Battle Order
class MDBO(BattleOrder):

    # type, used in battle_iterator
    FORCE_SWITCH = "force_switch"
    TEAMPREVIEW = "teampreview"
    TURN = "turn"
    DEFAULT = "default"

    # TODO: write down what MDBO expects. Basically a showdown order in test_model_battle_order
    def __init__(self, type: str, msg: Optional[str] = None):
        assert type in [
            self.FORCE_SWITCH,
            self.TEAMPREVIEW,
            self.TURN,
            self.DEFAULT,
        ], "mdbo type not recognized"
        self._type = type
        self._msg = msg

    # Returns the standardized protocol in string form
    @property
    def message(self) -> str:
        if self._type == self.DEFAULT:
            return "/choose default"
        else:
            assert self._msg is not None
            return self._msg

    # Translates a MDBO to an int
    def to_int(self) -> int:
        if self._type == self.DEFAULT:
            return -1

        assert (
            self._msg is not None
        ), "msg is None and not default; cant turn it into an int"
        if self._type == self.TEAMPREVIEW:
            key = re.sub(r"[\s,]", "", self._msg[5:])
            assert (
                key in _TEAMPREVIEW_ORDER_TO_INT
            ), f"{key} not in TEAMPREVIEW_ORDER_TO_INT. Message: {self._msg}"
            return _TEAMPREVIEW_ORDER_TO_INT[key]
        else:
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
        assert not self.message.startswith(
            "/team"
        ), "MDBO cannot be converted into DBO when it is a teampreview"
        assert (
            battle.player_role is not None
        ), "Cannot convert MDBO to DBO when player_role is None because we assume we have the right perspective"
        orders = []
        for i, order in enumerate(self.message.replace("/choose ", "").split(", ")):
            if order == "pass":
                orders.append(None)
            elif order == "default":
                return DefaultBattleOrder()
            elif order.startswith("switch "):
                key = list(battle.team.keys())[int(order[7]) - 1]
                orders.append(BattleOrder(order=battle.team[key]))

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
                    BattleOrder(
                        order=moving_mon.moves[move_key],
                        terastallize="terastallize" in order,
                        move_target=target,
                    )
                )

        return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])

    # Gets total number of possible orders
    @staticmethod
    def action_space() -> int:
        return len(_INT_TO_ORDER_MAPPINGS) * len(_INT_TO_ORDER_MAPPINGS)

    @staticmethod
    def teampreview_space() -> int:
        return len(_INT_TO_TEAMPREVIEW_ORDER)

    # Translates an integer into a ModelBattleOrder
    @staticmethod
    def from_int(i: int, type: str):
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
