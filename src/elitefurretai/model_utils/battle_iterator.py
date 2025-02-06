import re
from typing import Any, Dict, List, Optional

from poke_env.environment import (
    AbstractBattle,
    Effect,
    MoveCategory,
    Pokemon,
    PokemonGender,
)

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.utils.inference_utils import get_showdown_identifier


# Iterates through BattleData; cannot switch perspectives
class BattleIterator:

    FORCE_SWITCH = "force_switch"
    TEAMPREVIEW = "teampreview"
    TURN = "turn"

    def __init__(
        self,
        battle: AbstractBattle,
        bd: BattleData,
        perspective: str = "p1",
        custom_parse=BattleData.showdown_translation,
    ):
        self._battle: AbstractBattle = battle
        self._bd: BattleData = bd
        self._index: int = 0
        self._input_nums: List[int] = [0, 0]
        self._prev_turn_index = 0
        self._perspective = perspective
        self._last_input_type: Optional[str] = None

        # We use this to parse messages
        self._custom_parse = custom_parse

    def __str__(self):
        return f"""
        BattleIterator()
        Battle: {self._battle.battle_tag}
        Index: {self.index}
        BattleData: {self.bd}
        Input nums: {self._input_nums}
        """

    def next(self):
        assert (
            self._battle.player_role == self._perspective
        ), "Battle's player role was switched"

        if self._index >= len(self.bd.logs):
            raise StopIteration("Attempt to iterate when no more logs\n", str(self))
        split_message = self.bd.logs[self._index].split("|")

        # Implement parts that parse_message can't deal with
        if (
            len(split_message) == 1
            or split_message == ["", ""]
            or split_message[1] == "t:"
        ):
            pass
        elif split_message[1] == "win":
            self._battle.won_by(split_message[2])
        elif split_message[1] == "tie":
            self._battle.tied()
        else:
            self._battle.parse_message(self._custom_parse(split_message))

        # Find input and set _last_input to this; use _input_index to track where we are
        # in the input log; we should leave it where we should start from next time
        if self._is_pivot_trigger():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = self.FORCE_SWITCH

        elif self._is_teampreview():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            self._last_input_type = self.TEAMPREVIEW

        elif self._is_turn():

            # If we're starting a turn, the first input_num of the new turn should always start
            # with p1 since a player needs to have an input at the beginning of each turn
            # if this goes off, it likely means a problem with the last input
            assert self._input_nums[1] >= len(self._bd.input_logs) or self._bd.input_logs[
                self._input_nums[1]
            ].startswith(
                ">p1"
            ), "We are starting a turn, but the next input in the inputlog doesnt start with p1, or starts at the end of the logs, both of which are invalid"

            self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            self._prev_turn_index = self._index
            self._last_input_type = self.TURN

        elif self._is_revival_blessing():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = self.FORCE_SWITCH

        # Can either be two or one, depending on who faints; we deduce whether we need
        # two or one by the order of the player inputs in the player log
        elif self._is_preturn_switch():
            if self._need_switch("p1") and self._need_switch("p2"):
                self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            else:
                self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = self.FORCE_SWITCH

        self._index += 1

    def next_turn(self):
        while self._index < len(self.bd.logs) and not self._is_turn():
            self.next()

        # Go past the turn
        if not self._battle.finished and self._index < len(self.bd.logs):
            self.next()

    # Continues going until we find the next input for our perspective
    def next_input(self) -> Optional[str]:
        while (
            self._index < len(self.bd.logs)
            and not self._is_turn()
            and not self._is_pivot_trigger()
            and not self._is_teampreview()
            and not self._is_preturn_switch()
            and not self._is_revival_blessing()
        ):
            self.next()

        while self._index < len(self.bd.logs) and not (
            self.log == "|" or self._is_preturn_switch()
        ):
            self.next()

        if self._index < len(self.bd.logs) and self._is_preturn_switch():
            self.next()

        # Last input can be None if we havent iterated through the battle. It can also be
        # another opponent's input if they have a pivot. If this is the case, we need to go again
        # until we fine the input of the player (self._perspective) we're looking for
        if self.last_input is not None and not self.last_input.startswith(
            ">" + self._perspective
        ):

            if self._input_nums[0] >= len(self.bd.input_logs) or self._battle.finished:
                return None
            else:
                return self.next_input()

        # We have the next input
        elif self.last_input is not None and self._index < len(self.bd.logs):
            return self.last_input

        # We're at the end of the battle
        elif self._index == len(self.bd.logs):
            return None

        else:
            return None

    def finish(self):
        while self.index < len(self.bd.logs):
            self.next()

    # Assumes order of inputs is p1, p2
    @property
    def last_input(self) -> Optional[str]:
        assert (
            self._battle.player_role == self._perspective
        ), "Battle's player role was switched"
        inputs = self.bd.input_logs[self._input_nums[0] : self._input_nums[1]]
        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        elif self._perspective == "p1":
            return inputs[0]
        else:
            return inputs[1]

    @property
    def last_input_type(self) -> Optional[str]:
        return self._last_input_type

    @property
    def last_opponent_input(self) -> Optional[str]:
        assert (
            self._battle.player_role == self._perspective
        ), "Battle's player role was switched"
        inputs = self.bd.input_logs[self._input_nums[0] : self._input_nums[1]]
        if len(inputs) <= 1:
            return None
        elif self._perspective == "p1":
            return inputs[1]
        else:
            return inputs[0]

    @property
    def log(self) -> str:
        return self.bd.logs[self._index]

    @property
    def battle(self) -> AbstractBattle:
        return self._battle

    @property
    def bd(self) -> BattleData:
        return self._bd

    @property
    def index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._battle.finished

    @property
    def perspective(self) -> str:
        return self._perspective

    def _is_turn(self) -> bool:
        return self.log.startswith("|turn|")

    def _is_revival_blessing(self) -> bool:
        if self.index + 1 >= len(self.bd.logs):
            return False  # Can't Revival Blessing in last log
        next_log = self.bd.logs[self.index + 1]
        return "|Revival Blessing|" in self.log and "|-fail|" not in next_log

    def _is_pivot_trigger(self) -> bool:
        split_message = self.log.split("|")
        # So we don't waste anymore time with calculating mon_in_back
        if len(split_message) < 4:
            return False

        if split_message[3] not in [
            "Eject Button",
            "Eject Pack",
            "Baton Pass",
            "Teleport",
            "Chilly Reception",
            "Parting Shot",
            "Shed Tail",
            "Flip Turn",
            "U-turn",
            "Volt Switch",
            "ability: Wimp Out",
            "ability: Emergency Exit",
        ]:
            return False

        # No one to pivot to
        if not self._mon_in_back(split_message[2][0:2]):
            return False

        # Hard Pivot
        if (
            len(split_message) >= 4
            and split_message[1] == "move"
            and split_message[3]
            in [
                "Baton Pass",
                "Teleport",
                "Chilly Reception",  # Won't trigger weather if already snowing
            ]
        ):
            next_log = (
                self.bd.logs[self.index + 1].split("|")
                if self.index + 1 < len(self.bd.logs)
                else None
            )
            if next_log is not None and next_log[1] != "-fail":
                return True

        # Activate pivot
        elif (
            len(split_message) >= 4
            and split_message[1] == "move"
            and split_message[3]
            in [
                "Chilly Reception",
                "Parting Shot",
                "Shed Tail",
            ]
        ):
            i, next_logs = self.index + 1, []
            while (
                i < len(self.bd.logs)
                and "|" in self.bd.logs[i]
                and self.bd.logs[i].split("|")[1] not in ["move", "", "upkeep", "switch"]
            ):
                next_logs.append(self.bd.logs[i].split("|"))
                i += 1

            # -boost could come from Contrary
            if any(map(lambda x: x[1] in ["-unboost", "-start", "-weather", "-boost"], next_logs)):
                return True
            # super edge-casE: if there is mirror armor activation, but I am holding a Clear Amulet (so I get "-fail")
            elif len(next_logs) > 0 and len(next_logs[0]) > 3 and next_logs[0][3] == "Mirror Armor":
                return True

        # Damaging Move-based pivot
        elif (
            len(split_message) >= 4
            and split_message[1] == "move"
            and split_message[3]
            in [
                "Flip Turn",
                "U-turn",
                "Volt Switch",
            ]
        ):
            # Look until an empty log or a switch, a move or we're out of logs
            next_logs, i = [], self.index + 1
            while (
                i < len(self.bd.logs)
                and "|" in self.bd.logs[i]
                and self.bd.logs[i].split("|")[1] not in ["", "upkeep", "switch", "move"]
            ):
                next_logs.append(self.bd.logs[i].split("|"))
                i += 1

            damage_messages = ["-supereffective", "-damage", "-resisted", "-crit"]
            target = split_message[4] if len(split_message) > 4 else ""

            # If we hit the move, we will say we switch. Two edge-cases:
            # We activate redcard (which will take away our choice of switch), in which case this isnt a pivot
            # We faint due to rockyhelmet/roughskin/lifeorb, in which case this isnt a pivot
            hit_item_prevent_switch = any(
                map(
                    lambda x: x[1] == "-enditem" and x[3] in ["Red Card", "Eject Button"],
                    next_logs,
                )
            )
            pivot_mon_fainted = any(
                map(lambda x: x[1] == "faint" and x[2] == split_message[2], next_logs)
            )

            if (
                any(
                    map(
                        lambda x: x[1] in damage_messages and x[2] == target,
                        next_logs,
                    )
                )
                and not hit_item_prevent_switch
                and not pivot_mon_fainted
            ):
                return True

            # Handle substitute cases
            elif (
                any(
                    map(
                        lambda x: (x[1] == "-activate" and x[3] == "move: Substitute")
                        or (x[1] == "-end" and x[3] == "Substitute"),
                        next_logs,
                    )
                )
                and not hit_item_prevent_switch
                and not pivot_mon_fainted
            ):
                return True

            # Edge-case where we get a "-block" message if we hit disguise
            elif (
                any(
                    map(
                        lambda x: x is not None
                        and x[1] == "-block"
                        and x[3] == "ability: Disguise",
                        next_logs,
                    )
                )
                and not hit_item_prevent_switch
                and not pivot_mon_fainted
            ):
                return True

        # Item-based pivot
        elif (
            len(split_message) >= 4
            and split_message[1] == "-enditem"
            and split_message[3]
            in [
                "Eject Button",
                "Eject Pack",
            ]
        ):
            # Not an activation when an -enditem triggers cuz its either affected by a move
            # or if its activated but overriden by a redcard, or if its fainted from roughskin/spikyshield
            non_activations = [
                "[from] move: Knock Off",
                "[from] move: Corrosive Gas",
                "[from] move: Switcheroo",
                "[from] move: Thief",
                "[from] move: Trick"
            ]
            i, next_logs = self.index + 1, []
            while (
                i < len(self.bd.logs)
                and "|" in self.bd.logs[i]
                and self.bd.logs[i].split("|")[1] not in ["move", "", "upkeep", "switch"]
            ):
                next_logs.append(self.bd.logs[i].split("|"))
                i += 1

            if any(map(lambda x: len(x) > 2 and x[1] == "faint" and x[2] == split_message[2], next_logs)):
                return False
            if len(split_message) >= 5 and split_message[4] in non_activations:
                return False
            elif len(split_message) >= 6 and split_message[5] in non_activations:
                return False
            elif any(map(lambda x: len(x) > 3 and x[3] == "Red Card", next_logs)):
                return False
            else:
                return True

        # Ability-based pivot
        elif (
            len(split_message) >= 4
            and split_message[1] == "-ability"
            and split_message[3] in ["ability: Wimp Out", "ability: Emergency Exit"]
        ):
            return True

        return False

    # I have to look at actual battle logs to see if any mon has fainted or not. I cant just assume this
    # from input logs unfortunately, since p1 and p2 could both request a switch
    def _is_preturn_switch(self) -> bool:
        if self.log == "|upkeep":
            return self._need_switch("p1") or self._need_switch("p2")
        else:
            return False

    # We pass in a perspective to see if they need a switch on a preturn switch
    # Checks if a pokemon is fainted (or is about to faint from perish, an edge-case since perish triggers
    # after "upkeep"), and then looks to see if we have a mon to switch in.
    # NOTE: assumes as a default the max team is 4 for VGC
    def _need_switch(self, perspective: str) -> bool:
        logs = self.bd.logs[self._prev_turn_index : self._index]

        # Looks to see if a mon is fainted from the team (or is about to faint due to perish)
        needs_replacement = any(
            map(
                lambda x: x.startswith("|faint|" + perspective)
                or (x.startswith("|-start|" + perspective) and x.endswith("|perish0")),
                logs,
            )
        ) or (
            # Edge-case where we revival blessing with one fainted mon; then we need to switch the
            # revived one back in
            any(
                map(
                    lambda x: x.startswith("|-heal|" + perspective)
                    and "|[from] move: Revival Blessing" in x,
                    logs,
                )
            )
            and None
            in (
                self._battle.active_pokemon
                if self._battle.player_role == perspective
                else self._battle.opponent_active_pokemon
            )
        )

        # Return if a player needs to replace a mon, and there is a mon to replace
        return needs_replacement and self._mon_in_back(perspective)

    def _mon_in_back(self, perspective: str) -> bool:
        team_size = (
            self._battle.max_team_size if self._battle.max_team_size is not None else 4
        )
        # Get the team and active pokemon from the player
        team = (
            self._battle.team
            if self._battle.player_role == perspective
            else self._battle.opponent_team
        )
        actives = (
            self._battle.active_pokemon
            if self._battle.player_role == perspective
            else self._battle.opponent_active_pokemon
        )

        # Calculate non-fainted and non-fainted actives
        non_fainted = team_size - len(
            list(filter(lambda x: x.fainted or Effect.PERISH0 in x.effects, team.values()))
        )
        actives_non_fainted = len(
            list(
                filter(
                    lambda x: x is not None
                    and not (x.fainted or Effect.PERISH0 in x.effects),
                    actives,
                )
            )
        )
        return non_fainted - actives_non_fainted > 0

    def _is_teampreview(self) -> bool:
        return self.log.startswith("|teampreview")

    # Used to simulate requests that would be sent by the server when replaying omniscient
    # battle logs (via BattleIterator)
    def _generate_active_mon(self, mon: Pokemon) -> Dict[str, Any]:
        json = {"moves": []}

        for move_id, move in mon.moves.items():
            disabled = False
            if (mon.item == "assaultvest" and move.category == MoveCategory.STATUS) or (
                Effect.TAUNT in mon.effects and move.category == MoveCategory.STATUS
            ):
                disabled = True

            # TODO: implement these
            if Effect.DISABLE in mon.effects:
                pass  # |-start|p1a: Vikavolt|Disable|Flash Cannon

            # TODO: implement these
            if Effect.ENCORE in mon.effects:
                pass  # |-start|p1a: Vikavolt|Encore|Flash Cannon

            json["moves"].append(
                {
                    "move": move.entry["name"],
                    "id": move_id,
                    "pp": move.current_pp,
                    "maxpp": move.max_pp,
                    "target": move.entry["target"],
                    "disabled": disabled,
                }
            )

        # Choice items
        if not mon.first_turn and (
            mon.ability == "geurillatactics"
            or (mon.item is not None and mon.item.startswith("choice"))
        ):
            index = self.index
            while index > 0 and not re.search(
                f"|switch|{self.perspective}[ab]: {mon.name}", self.bd.logs[index]
            ):
                if re.search(
                    f"|move|{self.perspective}[ab]: {mon.name}|", self.bd.logs[index]
                ):
                    locked_move = self.bd.logs[index].split("|")[3]
                    for move_json in json["moves"]:
                        move_json["disabled"] = not (move_json["move"] == locked_move)
                    break
                index -= 1

        someone_terastallized = any(
            map(lambda x: x.is_terastallized, self.battle.team.values())
        )
        if not someone_terastallized and mon.tera_type is not None:
            json["canTerastallize"] = mon.tera_type.name.title()  # type: ignore

        return json

    def _generate_request_mon(self, mon: Pokemon) -> Dict[str, Any]:
        gender = None
        if mon.gender == PokemonGender.MALE:
            gender = "M"
        elif mon.gender == PokemonGender.FEMALE:
            gender = "F"

        details = f"{mon._data.pokedex[mon.species]['name']}, L{mon.level}"
        if gender is not None:
            details += f", {gender}"

        condition = f"{mon.current_hp}/{mon.max_hp}"
        if mon.status is not None:
            condition += f" {mon.status.name.lower()}"

        return {
            "ident": get_showdown_identifier(mon, self.perspective),
            "details": details,
            "condition": condition,
            "stats": mon.stats,
            "moves": list(mon.moves.keys()),
            "baseAbility": mon._ability,
            "item": mon.item,
            "ability": mon.ability,
            "teraType": mon.tera_type.name.title() if mon.tera_type else None,
            "terastallized": (
                mon.tera_type.name.title()
                if mon.is_terastallized and mon.tera_type
                else ""
            ),
            "active": mon in self.battle.active_pokemon,
        }

    def simulate_request(self) -> Dict[str, Any]:
        req = {}
        assert self.last_input is not None
        if self.last_input_type == BattleIterator.TEAMPREVIEW:
            assert self.last_input[4:].startswith("team")
            req["maxChosenTeamSize"] = 4  # What Showdown uses now
            req["maxTeamSize"] = 4  # Backwards compatibility
            req["teamPreview"] = True

        elif self.last_input_type == BattleIterator.FORCE_SWITCH:
            assert (
                self.last_input[1:3] == self.battle.player_role
                and "move" not in self.last_input
            )
            orders = self.last_input[4:].split(",")
            req["forceSwitch"] = [orders[0].strip() != "pass", orders[1].strip() != "pass"]

        elif self.last_input_type == BattleIterator.TURN:
            assert not self.last_input[4:].startswith("team")
            req["actives"] = []
            active_pokemon = self.battle._active_pokemon  # type: ignore
            req["actives"].append(
                self._generate_active_mon(active_pokemon.get(f"{self.perspective}a"))
            )
            req["actives"].append(
                self._generate_active_mon(active_pokemon.get(f"{self.perspective}b"))
            )

        req["side"] = {
            "name": self.battle.player_username,
            "id": self.battle.player_role,
            "pokemon": [
                self._generate_request_mon(mon) for mon in self.battle.team.values()
            ],
        }

        req["rqid"] = self.index

        return req
