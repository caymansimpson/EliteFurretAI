import re
from typing import Any, Dict, List, Optional

from poke_env.battle import AbstractBattle, Effect, MoveCategory, Pokemon, PokemonGender
from poke_env.data.normalize import to_id_str

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.encoder import MDBO


# Iterates through BattleData; cannot switch perspectives
class BattleIterator:

    def __init__(
        self,
        bd: BattleData,
        perspective: str = "p1",
        omniscient: bool = False,
    ):
        self._battle: AbstractBattle = bd.to_battle(perspective, omniscient)
        self._bd: BattleData = bd
        self._index: int = 0
        self._input_nums: List[int] = [0, 0]
        self._prev_turn_index = 0
        self._perspective = perspective
        self._omniscient = omniscient
        self._last_input_type: Optional[str] = None

    def __str__(self):
        return f"""
        BattleIterator()
        Battle: {self._battle.battle_tag}
        Index: {self.index}
        BattleData: {self.bd}
        Input nums: {self._input_nums}
        """

    def next(self):
        assert self._battle.player_role == self._perspective, (
            "Battle's player role was switched; " + self.bd.battle_tag
        )

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
            self._battle.parse_message(split_message)

        # If we were just passed teampreview, fill in information from player's choices
        if len(split_message) > 1 and split_message[1] == "start":

            # Assumes an order in inputLog
            index = 0 if self._perspective == "p1" else 1
            choices = map(
                lambda x: int(x) - 1,
                self.bd.input_logs[index]
                .replace(f">{self._perspective} team ", "")
                .split(", "),
            )
            team = self.bd.p1_team if self._perspective == "p1" else self.bd.p2_team
            sent_team = {}
            for omon in [team[choice] for choice in choices]:
                mon = omon.to_pokemon()
                sent_team[mon.identifier(self._perspective)] = mon
            self._battle.team = sent_team

            # Fill opponent team if we have omniscient perspective
            if self._omniscient:
                opp_perspective = "p1" if self._perspective == "p2" else "p2"
                opp_team = self.bd.p1_team if opp_perspective == "p1" else self.bd.p2_team

                index = 0 if opp_perspective == "p1" else 1
                choices = map(
                    lambda x: int(x) - 1,
                    self.bd.input_logs[index]
                    .replace(f">{opp_perspective} team ", "")
                    .split(", "),
                )
                opp_sent_team = {}
                for omon in [opp_team[choice] for choice in choices]:
                    mon = omon.to_pokemon()
                    opp_sent_team[mon.identifier(opp_perspective)] = mon
                self._battle._opponent_team = opp_sent_team

        # Find input and set _last_input to this; use _input_index to track where we are
        # in the input log; we should leave it where we should start from next time
        if self._is_pivot_trigger():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = MDBO.FORCE_SWITCH
        elif self._is_teampreview():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            self._last_input_type = MDBO.TEAMPREVIEW

            # if omniscient, populate opponent teampreview
            if self._omniscient:
                opp_perspective = "p1" if self._perspective == "p2" else "p2"
                opp_team = self.bd.p1_team if opp_perspective == "p1" else self.bd.p2_team
                self._battle._teampreview_opponent_team = [
                    omon.to_pokemon() for omon in opp_team
                ]
        elif self._is_turn():

            # If we're starting a turn, the first input_num of the new turn should always start
            # with p1 since a player needs to have an input at the beginning of each turn
            # if this goes off, it likely means a problem with the last input
            assert self._input_nums[1] >= len(self._bd.input_logs) or self._bd.input_logs[
                self._input_nums[1]
            ].startswith(">p1"), (
                "We are starting a turn, but the next input in the inputlog doesnt start with p1, or starts at the end of the logs, both of which are invalid; "
                + self.bd.battle_tag
            )

            self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            self._prev_turn_index = self._index
            self._last_input_type = MDBO.TURN
        elif self._is_revival_blessing():
            self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = MDBO.FORCE_SWITCH
        # Can either be two or one, depending on who faints; we deduce whether we need
        # two or one by the order of the player inputs in the player log
        elif self._is_preturn_switch():
            if self._need_switch("p1") and self._need_switch("p2"):
                self._input_nums = [self._input_nums[1], self._input_nums[1] + 2]
            else:
                self._input_nums = [self._input_nums[1], self._input_nums[1] + 1]
            self._last_input_type = MDBO.FORCE_SWITCH

        # If I hit a new request for my perspective, I should simulate the request to fill in the
        # battle object with synthesized request information
        if (
            self.last_input is not None
            and (
                self._is_pivot_trigger()
                or self._is_revival_blessing()
                or self._is_preturn_switch()
                or self._is_turn()
                or self._is_teampreview()
            )
            and self.last_input.startswith(">" + self._perspective)
        ):
            self.simulate_request()

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
        assert self._battle.player_role == self._perspective, (
            "Battle's player role was switched " + self.bd.battle_tag
        )
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
        assert self._battle.player_role == self._perspective, (
            "Battle's player role was switched " + self.bd.battle_tag
        )
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

    @property
    def omniscient(self) -> bool:
        return self._omniscient

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
            if any(
                map(
                    lambda x: x[1] in ["-unboost", "-start", "-weather", "-boost"],
                    next_logs,
                )
            ):
                return True
            # super edge-casE: if there is mirror armor activation, but I am holding a Clear Amulet (so I get "-fail")
            elif (
                len(next_logs) > 0
                and len(next_logs[0]) > 3
                and next_logs[0][3] == "Mirror Armor"
            ):
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
                "[from] move: Trick",
            ]
            i, next_logs = self.index + 1, []
            while (
                i < len(self.bd.logs)
                and "|" in self.bd.logs[i]
                and self.bd.logs[i].split("|")[1] not in ["move", "", "upkeep", "switch"]
            ):
                next_logs.append(self.bd.logs[i].split("|"))
                i += 1

            if any(
                map(
                    lambda x: len(x) > 2 and x[1] == "faint" and x[2] == split_message[2],
                    next_logs,
                )
            ):
                return False
            if len(split_message) >= 5 and split_message[4] in non_activations:
                return False
            elif len(split_message) >= 6 and split_message[5] in non_activations:
                return False

            # Red Card will negate an Eject Pack/Button from allowing the user to switch. Here, we need to make sure
            # that the Red Card is ejecting the Eject Pack mon
            elif any(
                map(
                    lambda x: len(x) > 4
                    and x[3] == "Red Card"
                    and split_message[2] == x[4].replace("[of] ", ""),
                    next_logs,
                )
            ):
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
        json: Dict[str, Any] = {"moves": []}

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
            "ident": mon.identifier(self.perspective),
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

    def simulate_request(self):
        req: Dict[str, Any] = {}
        assert self.last_input is not None
        if self.last_input_type == MDBO.TEAMPREVIEW:
            assert self.last_input[4:].startswith("team"), self.bd.battle_tag
            req["maxChosenTeamSize"] = 4  # What Showdown uses now
            req["maxTeamSize"] = 4  # Backwards compatibility
            req["teamPreview"] = True

        elif self.last_input_type == MDBO.FORCE_SWITCH:
            assert (
                self.last_input[1:3] == self.battle.player_role
                and "move" not in self.last_input
            ), f"{self.last_input} {self.bd.battle_tag} {self.battle.player_role}"
            orders = self.last_input[4:].split(",")
            req["forceSwitch"] = [orders[0].strip() != "pass", orders[1].strip() != "pass"]

        elif self.last_input_type == MDBO.TURN:
            assert not self.last_input[4:].startswith("team"), self.bd.battle_tag
            req["active"] = []
            active_pokemon = self.battle._active_pokemon  # type: ignore
            req["active"].append(
                self._generate_active_mon(active_pokemon.get(f"{self.perspective}a"))
            )
            req["active"].append(
                self._generate_active_mon(active_pokemon.get(f"{self.perspective}b"))
            )

        # Get identifiers of mons we chose in teampreview, so we only populate request with the team we sent
        choice = ""
        team = []
        if self._perspective == "p1":
            choice, team = self.bd.input_logs[0], self.bd.p1_team
        else:
            choice, team = self.bd.input_logs[1], self.bd.p2_team
        team_choice = list(map(lambda x: int(x) - 1, choice[9:].split(", ")))
        identifiers = list(
            map(lambda x: self._perspective + ": " + team[x].name, team_choice)
        )

        req["side"] = {
            "name": self.battle.player_username,
            "id": self.battle.player_role,
            "pokemon": [
                self._generate_request_mon(mon)
                for mon in self.battle.team.values()
                if mon.identifier(self._perspective) in identifiers
            ],
        }

        req["rqid"] = self.index

        self.battle.parse_request(req)

    # Creates a MDBO from the last input
    def last_order(self) -> MDBO:
        # Remove player identifier; inputs start with ">p1 " or ">p2
        if self.last_input is None or self.last_input_type is None:
            return MDBO(MDBO.DEFAULT)

        label = self.last_input[4:]

        # If teampreview order: Remove "team "; Replace all spaces and commas with just order of numbers
        if label.startswith("team "):
            return MDBO(MDBO.TEAMPREVIEW, "/" + label)

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
                    team = (
                        self.bd.p1_team
                        if self.battle.player_role == "p1"
                        else self.bd.p2_team
                    )
                    for mon in team:
                        if (
                            self.battle.active_pokemon[i]
                            and mon.name == self.battle.active_pokemon[i].name
                        ):
                            actor = mon
                            break

                    # Check to see if we could find the mon
                    assert actor is not None, (
                        f"Could not find actor [{self.battle.active_pokemon[i]}] from the order [{order}] in team [{list(map(lambda x: x.name, team))}]; "
                        + self.bd.battle_tag
                    )

                    # Get the move_index of the move
                    move_index = -1
                    for j, move in enumerate(actor.moves):
                        if to_id_str(order[5:].split(" ")[0]) == (to_id_str(move)):
                            move_index = j + 1  # Moves are 1-based
                            break

                    # Edge case for struggle, where any input is the right one. We should technically
                    # skip these, since any input is correct
                    if to_id_str(order[5:]) == "struggle" or to_id_str(
                        order[5:]
                    ).startswith("recharge"):
                        move_index = 1

                    # Check to see if we could find the move
                    assert move_index > -1, (
                        f"Could not find move [{order[5:]}] from the order [{order}] in the actor [{actor.name}] moves [{list(map(lambda x: x, actor.moves))}]; "
                        + self.bd.battle_tag
                    )

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

                    o = "move " + str(move_index)
                    if target != 0:
                        o += " " + str(target)
                    if mechanic is not None:
                        o += " " + mechanic
                    # Replace move name with move index
                    orders.append(o)

                # This is going to be either "pass", "default" or "switch" which is already good to go
                else:
                    orders.append(order.strip())

            return MDBO(self.last_input_type, "/choose " + ", ".join(orders))
