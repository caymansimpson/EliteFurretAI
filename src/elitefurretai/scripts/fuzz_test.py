# -*- coding: utf-8 -*-
"""This module plays a bunch of players against each other to find errors in Inference classes.
"""

import asyncio
import itertools
import os.path
import re
import sys
import time
from typing import Optional, Union

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import (
    get_showdown_identifier,
    observation_to_str,
)
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference
from elitefurretai.utils.team_repo import TeamRepo

from poke_env.data.gen_data import GenData
from poke_env.environment.move_category import MoveCategory
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder


global players  # TODO: remove after debug


class FuzzTestPlayer(RandomPlayer):

    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        avatar: Optional[str] = None,
        battle_format: str = "gen9randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = False,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = None,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
    ):
        super().__init__(
            account_configuration=account_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            accept_open_team_sheet=accept_open_team_sheet,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team,
        )
        self._inferences = {}
        self._speed_inferences = {}
        self._item_inferences = {}

    def teampreview(self, battle):
        inferences = BattleInference(battle)
        self._inferences[battle.battle_tag] = inferences

        player = None  # TODO: remove after debugging
        for p in players:
            if p.username == battle.opponent_username:
                player = p
                break

        # Speed and Item Inferences will fill inferences
        if battle.player_role == "p1":
            self._speed_inferences[battle.battle_tag] = SpeedInference(
                battle, inferences, verbose=0, player=player
            )
            self._item_inferences[battle.battle_tag] = ItemInference(
                battle, inferences, verbose=0
            )

            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)
        return "/team 1234"

    def choose_move(self, battle):
        if battle.battle_tag not in self._inferences:
            raise ValueError("SHOWDOWN IS BROKEN WITH TEAMPREVIEW BUG! RESTART IT!")

        # Don't deal with battle.force_switch until
        if not any(battle.force_switch) and battle.player_role == "p1":
            key = max(battle.observations.keys())
            # print("fuzz_test.choose_move", battle.observations[key].events)
            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)
        return self.choose_random_doubles_move(battle)  # pyright: ignore


# TODO: move this over utils and import it; can be omniscient or not
def print_battle(p1, p2, battle_tag):

    battle = p1.battles[battle_tag]
    message = f"============= Battle [{battle_tag} =============\n"
    message += f"The battle is between {p1.username} and {p2.username} from {p1.username}'s perspective.\n"

    message += "P1 Teampreview Team (omniscient): ["
    for mon in p1.battles[battle_tag].teampreview_team:
        ident = get_showdown_identifier(mon, p1.battles[battle_tag].player_role)
        mon = p1.battles[battle_tag].team.get(ident, mon)
        message += f"\n\t{mon.name} => [Speed: {mon.stats['spe']} // Item: {mon.item}]"
    message += "]\n"

    message += "P2 Teampreview Team (omniscient): ["
    for mon in p2.battles[battle_tag].teampreview_team:
        ident = get_showdown_identifier(mon, p2.battles[battle_tag].player_role)
        mon = p2.battles[battle_tag].team.get(ident, mon)
        message += f"\n\t{mon.name} => [Speed: {mon.stats['spe']} // Item: {mon.item}]"
    message += "]\n"

    for turn, obs in battle.observations.items():
        message += f"\n\nTurn #{turn}:"
        message += observation_to_str(obs)

    if battle._current_observation not in battle.observations.values():
        message += f"\n\nCurrent Observation; Turn #{battle.turn}:"
        message += observation_to_str(battle._current_observation)

    return message


def check_ground_truth(p1, p2, battle_tag):
    msg = ""
    counts = {}

    # Check P2's BattleInference
    for mon in p1.battles[battle_tag].teampreview_team:

        ident = get_showdown_identifier(mon, p2.battles[battle_tag].opponent_role)
        if not p2._inferences[battle_tag].is_tracking(ident):
            continue

        flags = p2._inferences[battle_tag].get_flags(ident)
        mon_in_battle = p1.battles[battle_tag].team[
            ident
        ]  # Checking in case of knockoff

        # zazmenta crown ed is causing problems; not sure if its getting recorded as steel type
        # handling of trick is broken too
        if (
            flags["item"]
            not in [mon.item, None, GenData.UNKNOWN_ITEM, mon_in_battle.item]
            and flags["item"] == "choicescarf"
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type: (error_speed_item) Erroneously found item that the mon didn't have, due to incorrect Speed Calculations\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_speed_item"] = counts.get("speed_item", 0) + 1

        elif flags["item"] not in [
            mon.item,
            None,
            GenData.UNKNOWN_ITEM,
            mon_in_battle.item,
        ]:
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_item): Erroneously found item that the mon didn't have\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_item"] = counts.get("error_item", 0) + 1

        elif (
            not flags["can_be_choice"]
            and mon.item
            and mon_in_battle.item
            and mon_in_battle.item.startswith("choice")
            and mon.item.startswith("choice")
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_can_be_choice): Erroneously found a mon can't have a choice item when they do\n"
            msg += f"{mon_in_battle.name} was found not to have choice, but has the item '{mon.item}'\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_can_be_choice"] = counts.get("error_can_be_choice", 0) + 1

        elif flags["has_status_move"] and not any(
            map(lambda x: x.category == MoveCategory.STATUS, mon.moves.values())
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_has_status_move): Erroneously found a mon can't be assault vest (e.g. they used a status move) when they could be\n"
            msg += f"{mon_in_battle.name} has the moves: [{', '.join(map(lambda x: x.id, mon.moves.values()))}]\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_has_status_move"] = counts.get("error_has_status_move", 0) + 1

        elif (
            mon_in_battle.stats["spe"] < flags["spe"][0]
            or mon_in_battle.stats["spe"] > flags["spe"][1]
        ) and not (
            mon_in_battle.item == "choicescarf"
            and flags["spe"][1] == BattleInference.load_opponent_set(mon)["spe"][1]  # We guess theyre maxspeed
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_speed): Found an erroneous speed.\n"
            msg += f"{mon_in_battle.name} has a {mon_in_battle.stats['spe']} Speed stat, but we've bounded it between {flags['spe'][0]} and {flags['spe'][1]}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_speed"] = counts.get("error_speed", 0) + 1

    return msg, counts


# Arguments passed via CLI can be number of battles to run, or "print" to print the errors to the console
async def main():

    print("\n\n\n\n\n\n\n\n\033[92mStarting Fuzz Test!\033[0m\n")
    num = None
    for arg in sys.argv:
        if arg.isdigit():
            num = arg
    total_battles = int(num) if num is not None else 1000
    print("Loading and validating teams, then creating players...")
    tr = TeamRepo(validate=False, verbose=False)
    print(
        f"Finished loading {len(tr.teams['gen9vgc2024regh'])} teams to battle against each other!"
    )

    global players  # TODO: remove after debug
    players = []
    i = 0
    for team_name, team in tr.teams["gen9vgc2024regh"].items():

        # ef this noise
        if (
            "Zoroark" in team
            or ("Dondozo" in team and "Tatsugiri" in team)
            or "Lagging Tail" in team
            or "Iron Ball" in team
        ):
            continue

        # Don't want to deal with non-english characters
        if bool(re.search(r"[^a-zA-Z0-9\- _]", team_name)):
            continue

        if not any(map(lambda x: team_name[:17] == x.username, players)):
            players.append(
                FuzzTestPlayer(
                    AccountConfiguration(team_name[:17], None),
                    battle_format="gen9vgc2024regg",
                    team=team,
                )
            )

        i += 1

    # Change min to max to remove debugging
    num_matchups = max(
        1, int(total_battles / len(list(itertools.combinations(players, 2))))
    )

    print(f"Done! Now starting Player battles with {num_matchups} each")

    total_errors = ""
    total = 0
    original_time = time.time()
    filename = os.path.expanduser("~/Desktop/fuzz_errors.txt")

    for p1, p2 in itertools.combinations(players, 2):
        t0 = time.time()

        for i in range(num_matchups):
            total += 1
            print(
                f"\rStarting battle #{total} between {p1.username} and {p2.username}...\r",
                end="",
                flush=True,
            )
            await p1.battle_against(p2)
            if total >= total_battles:
                break

        print(
            f"Finished {num_matchups} battles between {p1.username} and {p2.username} in {round((time.time() - t0), 2)}"
            + f"s ({round((time.time() - t0) * 1. / num_matchups, 2)}s each)"
        )

        if total >= total_battles:
            break

    print("\n\n======== Finished all battles!! ========")
    print(f"Finished {total} battles in {round(time.time() - original_time, 2)} sec")
    print(
        f"\tfor {round((time.time() - original_time) * 1. / total, 2)} sec per battle"
    )

    print(
        "Now going through Inference to check the right ground truth! This method won't catch cases when"
        + " I should have inferred items, but didn't. It will only catch when I infer incorrectly..."
    )

    # Check for implicit inference errors. Doesn't catch when I should have inferred an error and I didn't
    total_success = 0
    total = 0
    counts = {}
    for p1, p2 in itertools.combinations(players, 2):

        # Go through battle from each player's perspective; first have to find battles
        # between them
        for battle_tag, battle in p1.battles.items():
            if battle.opponent_username == p2.username:
                total += 2

                msg, counts1 = check_ground_truth(p1, p2, battle_tag)
                if msg == "":
                    total_success += 1
                else:
                    total_errors += msg

                msg, counts2 = check_ground_truth(p2, p1, battle_tag)
                if msg == "":
                    total_success += 1
                else:
                    total_errors += msg

                for key in counts1:
                    counts[key] = counts.get(key, 0) + counts1[key]
                for key in counts2:
                    counts[key] = counts.get(key, 0) + counts2[key]

    print("\n======== Finished checking all battles for inferences ========")
    print(
        f"Finished {total} battles with {round(total_success * 1.0 / total * 100, 2)}% success rate"
    )
    for error_type in counts:
        print(f"\tError Type [{error_type}] was found {counts[error_type]} times")

    print(
        f"Writing these to {filename}). {'Also, printing them here:' if 'print' in sys.argv else ''}"
    )
    with open(filename, "w") as f:
        f.write(total_errors)

    print_errors = False
    if print_errors or "print" in sys.argv:
        print(total_errors)


if __name__ == "__main__":
    asyncio.run(main())
