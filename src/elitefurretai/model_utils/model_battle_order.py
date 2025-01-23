import itertools
import re
from typing import List

from poke_env.data.normalize import to_id_str
from poke_env.environment import AbstractBattle, Move, Pokemon
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder

from elitefurretai.model_utils.battle_data import BattleData

_INT_TO_ORDER_MAPPINGS = {
    0: "move 1 -2",
    1: "move 1 -1",
    2: "move 1 0",
    3: "move 1 1",
    4: "move 1 2",
    5: "move 1 -2 terastallize",
    6: "move 1 -1 terastallize",
    7: "move 1 0 terastallize",
    8: "move 1 1 terastallize",
    9: "move 1 2 terastallize",
    10: "move 2 -2",
    11: "move 2 -1",
    12: "move 2 0",
    13: "move 2 1",
    14: "move 2 2",
    15: "move 2 -2 terastallize",
    16: "move 2 -1 terastallize",
    17: "move 2 0 terastallize",
    18: "move 2 1 terastallize",
    19: "move 2 2 terastallize",
    20: "move 3 -2",
    21: "move 3 -1",
    22: "move 3 0",
    23: "move 3 1",
    24: "move 3 2",
    25: "move 3 -2 terastallize",
    26: "move 3 -1 terastallize",
    27: "move 3 0 terastallize",
    28: "move 3 1 terastallize",
    29: "move 3 2 terastallize",
    30: "move 4 -2",
    31: "move 4 -1",
    32: "move 4 0",
    33: "move 4 1",
    34: "move 4 2",
    35: "move 4 -2 terastallize",
    36: "move 4 -1 terastallize",
    37: "move 4 0 terastallize",
    38: "move 4 1 terastallize",
    39: "move 4 2 terastallize",
    40: "switch 3",
    41: "switch 4",
    42: "pass",
}

_ORDER_MAPPINGS_TO_INT = {v: k for k, v in _INT_TO_ORDER_MAPPINGS.items()}

_INT_TO_TEAMPREVIEW_ORDER = {
    i: "".join([str(x) for x in permutation])
    for i, permutation in enumerate(itertools.permutations(range(1, 7), 4))
}

_TEAMPREVIEW_ORDER_TO_INT = {v: k for k, v in _INT_TO_TEAMPREVIEW_ORDER.items()}


class ModelBattleOrder(BattleOrder):

    # Takes in either an int or a string and converts it into a protocol
    def __init__(self, msg: str, battle: AbstractBattle, standardize=True):
        self._battle = battle
        self._message = self._standardize(msg) if standardize else msg

    # Returns the standardized protocol in string form
    @property
    def message(self) -> str:
        if self._message.startswith("team"):
            return "/" + self._message
        else:
            return "/choose " + self._message

    # Store standardized protocol
    def set_message(self, msg: str, standardize=True):
        self._message = self._standardize(msg) if standardize else msg

    # Translates a standardized protocol into an integer for model output
    def to_int(self) -> int:
        if self._message.startswith("team"):
            key = re.sub(r"[\s,]", "", self.message[5:])
            if key not in _TEAMPREVIEW_ORDER_TO_INT:
                raise ValueError(
                    f"{key} not in TEAMPREVIEW_ORDER_TO_INT. Message: {self._message}"
                )
            return _TEAMPREVIEW_ORDER_TO_INT[key]
        else:
            orders = self._message.split(", ")
            if len(orders) < 2:
                raise ValueError("Only got one order for a double battle order")
            elif len(orders) == 2 and orders[0] not in _ORDER_MAPPINGS_TO_INT:
                raise ValueError(
                    f"{orders[0]} not in ORDER_MAPPINGS_TO_INT. Label: {self._message}"
                )
            elif len(orders) == 2 and orders[1] not in _ORDER_MAPPINGS_TO_INT:
                raise ValueError(
                    f"{orders[1]} not in ORDER_MAPPINGS_TO_INT. Label: {self._message}"
                )
            return (
                _ORDER_MAPPINGS_TO_INT[orders[0]] * len(_ORDER_MAPPINGS_TO_INT)
                + _ORDER_MAPPINGS_TO_INT[orders[1]]
            )

    # Convert this to a regular double battle order
    def to_double_battle_order(self) -> DoubleBattleOrder:
        if self._message.startswith("team"):
            raise ValueError(
                "Cannot convert to a double battle order with a team order: {self._message}"
            )

        orders = []
        for i, order in enumerate(self._message.split(", ")):
            if order.startswith("switch "):
                index = int(order[7])
                mon_entry = self._battle.last_request["side"]["pokemon"][index - 1]
                orders.append(
                    BattleOrder(
                        order=Pokemon(gen=self._battle.gen, details=mon_entry["details"])
                    )
                )
            elif order.startswith("move "):

                # Extract Move index
                move_index = int(order[6])

                # Get the mon_entry of the ith active pokemon from the last request, and get the move
                mon_entry = self._battle.last_request["active"][i]
                move = Move(mon_entry["moves"][move_index - 1]["id"], gen=self._battle.gen)

                # Get target
                target = int(order[8])

                # Return the order with the right mechanic (if any)
                if order.endswith(" mega"):
                    orders.append(BattleOrder(order=move, mega=True, move_target=target))
                elif order.endswith("dynamax"):
                    orders.append(
                        BattleOrder(order=move, dynamax=True, move_target=target)
                    )
                elif order.endswith("zmove"):
                    orders.append(BattleOrder(order=move, z_move=True, move_target=target))
                elif order.endswith("terastallize"):
                    orders.append(
                        BattleOrder(order=move, terastallize=True, move_target=target)
                    )
                else:
                    orders.append(BattleOrder(order=move, move_target=target))

        return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])

    # Translates a protocol method into a standardized protocol where moves are to_id_str'ed
    # switches are the index of the mon in the request and targets are the index
    def _standardize(self, msg: str) -> str:
        label = msg
        # Remove player identifier; some inputs start with ">p1 " or ">p2 " if it exists
        if msg.startswith(">p"):
            label = label[4:]

        # If teampreview order
        if re.search(r"^(/?team)", label):

            # Remove "team " or "/team "
            label = re.sub(r"^(/?team)\s+", "", label)

            # Replace all spaces and commas with just order of numbers, and find the mapping int
            # we want the output of the neural net to be
            return "team " + ",".join(re.findall(r"[1-6]", label))

        # If move order
        else:
            # Remove "/choose "
            label = re.sub(r"^(/?choose)\s+", "", label)

            # Remove all the +[num] and convert to [num]
            label = re.sub(r"[\+]([0-2]])", r"\1", label)

            # To store both orders and figure out orders of mons
            orders = []
            json = self._battle.last_request
            assert json
            assert self._battle.player_role

            # Parse each single order separately, where i is the index of battle.active_pokemon
            for i, order in enumerate(label.split(", ")):
                order = order.strip()

                # If the single order is a move
                if order.startswith("move "):

                    # If the move is a name of the move, and not the index
                    if re.search("move [A-Za-z]+", order):

                        # Get the mon_entry of the ith active pokemon from json
                        mon_entry = self._battle.last_request["active"][i]

                        # Get the move_index of the move
                        move_index = -1
                        for j, move in enumerate(mon_entry["moves"]):

                            if to_id_str(order[5:]).startswith(to_id_str(move)):
                                move_index = j + 1  # Moves are 1-based
                    else:
                        # Extract move index if already standardized
                        move_index = int(re.findall(r"([1-4])", order[5:])[0])

                    # Find whether there's a mechanic, and if so, remove it
                    mechanic = None
                    for ending in ["mega", "dynamax", "zmove", "terastallize"]:
                        if order.endswith(ending):
                            mechanic = ending
                            order = order[: -(len(ending) + 1)]
                            break

                    # Extract the target from the string, and default to 0
                    target = 0
                    if re.search(r"move\s[1-4]$", order.strip()):
                        target = 0
                    elif re.search(r".*?(-?[0-2])$", order.strip()):
                        target = int(re.findall(r".*?(-?[0-2])$", order.strip())[0])

                    # Replace move name with move index
                    orders.append(
                        "move "
                        + str(move_index)
                        + " "
                        + str(target)
                        + (" " + mechanic if mechanic is not None else "")
                    )

                # If we need to translate a pokemon switch name into the mon's index
                elif re.search("switch [A-Za-z]+", order):

                    mon = re.findall(r"switch (.*)", order)[0]

                    # Find the index of the mon, according to the battle_json. Will match the name or the species
                    mon_index = None
                    for j, entry_mon in enumerate(json["side"]["pokemon"]):
                        if entry_mon["name"] == mon:
                            mon_index = j + 1
                        elif to_id_str(entry_mon["species"]) == to_id_str(mon):
                            mon_index = j + 1

                    if mon_index is None:
                        raise ValueError(f"Could not find mon {mon} in json of {json}")

                    # Replace mon name with mon index
                    orders.append("switch " + str(mon_index))

                # This is going to be either "pass" or "default" or an already sanitized switch
                else:
                    orders.append(order.strip())

            return ", ".join(orders)

    # TODO: implement
    @staticmethod
    def action_mask(battle: AbstractBattle) -> List[bool]:
        raise NotImplementedError

    # Gets total number of possible orders
    @staticmethod
    def action_space() -> int:
        return len(_INT_TO_ORDER_MAPPINGS) * len(_INT_TO_ORDER_MAPPINGS)

    # Translates an integer into a ModelBattleOrder
    @staticmethod
    def from_int(i: int, battle: AbstractBattle):
        if battle.teampreview:
            assert i < len(_INT_TO_TEAMPREVIEW_ORDER)
            return ModelBattleOrder(_INT_TO_TEAMPREVIEW_ORDER[i], battle)

        else:
            if i >= len(_INT_TO_ORDER_MAPPINGS) * len(_INT_TO_ORDER_MAPPINGS):
                raise ValueError(f"Invalid order: {i}")
            msg = (
                _INT_TO_ORDER_MAPPINGS[i // len(_INT_TO_ORDER_MAPPINGS)]
                + ", "
                + _INT_TO_ORDER_MAPPINGS[i % len(_INT_TO_ORDER_MAPPINGS)]
            )
            return ModelBattleOrder(msg, battle)

    @staticmethod
    def from_battle_data(msg: str, battle: AbstractBattle, bd: BattleData):
        # Remove player identifier; inputs start with ">p1 " or ">p2
        label = msg[4:]

        # If teampreview order: Remove "team "; Replace all spaces and commas with just order of numbers
        if label.startswith("team "):
            return ModelBattleOrder(
                "team " + ",".join(re.findall(r"[1-6]", label[5:])), battle
            )

        # If move order
        else:

            # Remove the all the +[num] and convert to [num]
            label = re.sub(r"\+([012])", r"\1", label)
            orders = []

            # Parse each single order separately, where i is the index of battle.active_pokemon
            for i, order in enumerate(label.split(", ")):
                order = order.strip()

                # If the single order is a move
                if order.startswith("move "):

                    # Get the mon_entry of the ith active pokemon from json
                    actor = None
                    team = bd.p1_team if battle.player_role == "p1" else bd.p2_team
                    for mon in team:
                        if (
                            battle.active_pokemon[i]
                            and mon.name == battle.active_pokemon[i].name
                        ):
                            actor = mon
                            break

                    # Check to see if we could find the mon
                    assert (
                        actor is not None
                    ), f"Could not find actor [{battle.active_pokemon[i]}] from the order [{order}] in team [{list(map(lambda x: x.name, team))}]"

                    # Get the move_index of the move
                    move_index = -1
                    for j, move in enumerate(actor.moves):
                        if to_id_str(order[5:]).startswith(to_id_str(move)):
                            move_index = j + 1  # Moves are 1-based
                            break

                    # Edge case for struggle, where any input is the right one. We should technically
                    # skip these, since any input is correct
                    if to_id_str(order[5:]).startswith("struggle") or to_id_str(
                        order[5:]
                    ).startswith("recharge"):
                        move_index = 1

                    # Check to see if we could find the move
                    assert (
                        move_index > -1
                    ), f"Could not find move [{order[5:]}] from the order [{order}] in the actor [{actor.name}] moves [{list(map(lambda x: x, actor.moves))}]"

                    # Find whether there's a mechanic, and if so, remove it
                    mechanic = None
                    for ending in ["mega", "dynamax", "zmove", "terastallize"]:
                        if order.endswith(ending):
                            mechanic = ending
                            order = order[: -(len(ending) + 1)]
                            break

                    # Extract the target from the string, and default to 0
                    target = 0
                    if re.search(r".*?\s(-?[0-2])$", order):
                        target = int(re.findall(r".*?\s(-?[0-2])$", order)[0])

                    # Replace move name with move index
                    orders.append(
                        f"move {str(move_index)} {str(target)} {mechanic if mechanic is not None else ''}".strip()
                    )

                # This is going to be either "pass" or "default" or an already sanitized switch
                else:

                    # Just an additional check for correctness of the order (if we are using requests)
                    if order.startswith("pass"):
                        assert (
                            any(battle.force_switch)
                            or any(
                                map(
                                    lambda x: x is None or x.species == "tatsugiri",
                                    battle.active_pokemon,
                                )
                            )
                        ) or len(battle.last_request)

                    orders.append(order.strip())

            return ModelBattleOrder(", ".join(orders), battle, standardize=False)
