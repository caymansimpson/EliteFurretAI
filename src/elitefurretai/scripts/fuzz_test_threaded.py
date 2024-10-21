# -*- coding: utf-8 -*-
"""Threaded version of fuzz_test, where we play a bunch of players against each
other to find errors in our Inference classes. This threaded version is
specifically for when you get to the last 0.01% of errors, and need to find those errors
more quickly.
"""

import asyncio
import os.path
import re
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple

from poke_env.data.gen_data import GenData
from poke_env.environment.move_category import MoveCategory
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import get_showdown_identifier
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference
from elitefurretai.utils.team_repo import TeamRepo


@dataclass
class BattleResult:
    p1_username: str
    p2_username: str
    battle_tag: str
    errors: str
    error_counts: Dict[str, int]
    success: bool


class ResultAggregator:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_battles = 0
        self.successful_battles = 0
        self.error_counts = {}
        self.error_log = ""

    def add_result(self, result: BattleResult):
        with self.lock:
            self.total_battles += 1
            if result.success:
                self.successful_battles += 1

            for error_type, count in result.error_counts.items():
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + count

            if result.errors:
                self.error_log += result.errors


class CustomPlayer(RandomPlayer):

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

        # Speed and Item Inferences will fill inferences
        if battle.player_role == "p1":
            self._speed_inferences[battle.battle_tag] = SpeedInference(battle, inferences, verbose=0)
            self._item_inferences[battle.battle_tag] = ItemInference(battle, inferences, verbose=0)

            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)
        return "/team 1234"

    def choose_move(self, battle):
        if battle.battle_tag not in self._inferences:
            raise ValueError("SHOWDOWN IS BROKEN WITH TEAMPREVIEW BUG! RESTART IT!")

        # Don't deal with battle.force_switch until
        if not any(battle.force_switch) and battle.player_role == "p1":
            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)
        return self.choose_random_doubles_move(battle)  # pyright: ignore


class PlayerPool:
    """Manages a pool of players, ensuring each player is only used in one battle at a time"""
    def __init__(self, players: List[CustomPlayer]):
        self.players = set(players)
        self.in_use = set()
        self.lock = threading.Lock()

    async def acquire_pair(self) -> Optional[Tuple[CustomPlayer, CustomPlayer]]:
        """Attempt to acquire a pair of available players"""
        with self.lock:
            available = self.players - self.in_use
            if len(available) < 2:
                return None

            # Get first two available players
            p1, p2 = list(available)[:2]
            self.in_use.add(p1)
            self.in_use.add(p2)
            return p1, p2

    def release_pair(self, p1: CustomPlayer, p2: CustomPlayer):
        """Release a pair of players back to the available pool"""
        with self.lock:
            self.in_use.remove(p1)
            self.in_use.remove(p2)


def print_observation(obs):
    message = ""
    message += f"\n\tMy Active Mon:  [{', '.join(map(lambda x: x.species if x else 'None', obs.active_pokemon)) if obs.active_pokemon else ''}]"
    message += f"\n\tOpp Active Mon: [{', '.join(map(lambda x: x.species if x else 'None', obs.opponent_active_pokemon)) if obs.opponent_active_pokemon else ''}]"
    message += f"\n\tWeather: [{', '.join(map(lambda x: x.name, obs.weather))}]"
    message += f"\tFields: [{', '.join(map(lambda x: x.name, obs.fields))}]"
    message += f"\n\tMy Side Conditions:  [{', '.join(map(lambda x: x.name, obs.side_conditions))}]"
    message += f"\n\tOpp Side Conditions: [{', '.join(map(lambda x: x.name, obs.opponent_side_conditions))}]"

    message += "\n\tMy Team:"
    for ident, mon in obs.team.items():
        message += f"\n\t\t{mon.species} => [Speed: {mon.stats['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"

    message += "\n\tOpp Team:"
    for ident, mon in obs.opponent_team.items():
        message += f"\n\t\t{mon.species} => [Speed: {mon.stats['spe']}], [Item: {mon.item}], [Speed Boost: {mon.boosts['spe']}], [Effects: {list(map(lambda x: x.name, mon.effects))}], [Status: {mon.status.name if mon.status else 'None'}]"

    message += "\n\n\tEvents:"
    for event in obs.events:
        message += f"\n\t\t{event}"

    return message


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
        message += print_observation(obs)

    if battle._current_observation not in battle.observations.values():
        message += f"\n\nCurrent Observation; Turn #{battle.turn}:"
        message += print_observation(battle._current_observation)

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
        mon_in_battle = p1.battles[battle_tag].team[ident]  # Checking in case of knockoff

        # zazmenta crown ed is causing problems; not sure if its getting recorded as steel type
        # handling of trick is broken too
        if flags["item"] not in [mon.item, None, GenData.UNKNOWN_ITEM, mon_in_battle.item] and flags["item"] == "choicescarf":
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type: (error_speed_item) Erroneously found item that the mon didn't have, due to incorrect Speed Calculations\n"
            msg += f"This was done on {flags['debug_item_found_turn']} turn\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_speed_item"] = counts.get("speed_item", 0) + 1

        elif flags["item"] not in [mon.item, None, GenData.UNKNOWN_ITEM, mon_in_battle.item]:
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_item): Erroneously found item that the mon didn't have\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_item"] = counts.get("error_item", 0) + 1

        elif not flags["can_be_choice"] and mon.item and mon_in_battle.item and mon_in_battle.item.startswith("choice") and mon.item.startswith("choice"):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_can_be_choice): Erroneously found a mon can't have a choice item when they do\n"
            msg += f"{mon_in_battle.name} was found not to have choice, but has the item '{mon.item}'\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_can_be_choice"] = counts.get("error_can_be_choice", 0) + 1

        elif flags["has_status_move"] and not any(map(lambda x: x.category == MoveCategory.STATUS, mon.moves.values())):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_has_status_move): Erroneously found a mon can't be assault vest (e.g. they used a status move) when they could be\n"
            msg += f"{mon_in_battle.name} has the moves: [{', '.join(map(lambda x: x.id, mon.moves.values()))}]\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_has_status_move"] = counts.get("error_has_status_move", 0) + 1

        # Something going wrong here; giving mons choicescarf when they already have revealed an item
        elif mon_in_battle.stats['spe'] < flags["spe"][0] or mon_in_battle.stats['spe'] > flags['spe'][1]:
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_speed): Found an erroneous speed.\n"
            msg += f"{mon_in_battle.name} has a {mon_in_battle.stats['spe']} Speed stat, but we've bounded it between {flags['spe'][0]} and {flags['spe'][1]}\n\n"
            msg += print_battle(p2, p1, battle_tag)
            counts["error_speed"] = counts.get("error_speed", 0) + 1

    return msg, counts


async def run_battle(p1: CustomPlayer, p2: CustomPlayer) -> BattleResult:
    """Run a single battle between two players and return the results"""
    await p1.battle_against(p2)

    # Find the battle tag for this matchup
    battle_tag = None
    for tag, battle in p1.battles.items():
        if battle.opponent_username == p2.username:
            battle_tag = tag
            break

    if not battle_tag:
        return BattleResult(p1.username, p2.username, "", "", {}, False)

    # Check ground truth from both perspectives
    msg1, counts1 = check_ground_truth(p1, p2, battle_tag)
    msg2, counts2 = check_ground_truth(p2, p1, battle_tag)

    # Combine error counts
    combined_counts = {}
    for key in set(counts1.keys()) | set(counts2.keys()):
        combined_counts[key] = counts1.get(key, 0) + counts2.get(key, 0)

    return BattleResult(
        p1_username=p1.username,
        p2_username=p2.username,
        battle_tag=battle_tag,
        errors=msg1 + msg2,
        error_counts=combined_counts,
        success=not bool(msg1 or msg2)
    )


async def battle_worker(player_pool: PlayerPool, result_aggregator: ResultAggregator, remaining_battles: asyncio.Event):
    """Worker function that processes battles using available players from the pool"""
    while remaining_battles.is_set():
        try:
            # Try to get an available pair of players
            player_pair = await player_pool.acquire_pair()
            if not player_pair:
                # No players available, wait a bit and try again
                await asyncio.sleep(0.1)
                continue

            p1, p2 = player_pair
            try:
                result = await run_battle(p1, p2)
                result_aggregator.add_result(result)
            finally:
                # Always release the players back to the pool
                player_pool.release_pair(p1, p2)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in battle worker: {e}")


async def main():
    print("\n\n\n\n\n\n\n\n\033[92mStarting Multithreaded Fuzz Test!\033[0m\n")

    # Parse command line arguments
    num = None
    for arg in sys.argv:
        if arg.isdigit():
            num = arg
    total_battles = int(num) if num is not None else 1000

    # Number of concurrent battles to run - limited by number of player pairs available
    num_threads = min(os.cpu_count() or 4, 8)  # Limit to 8 threads maximum

    print(f"Running with up to {num_threads} concurrent battles")
    print("Loading and validating teams, then creating players...")

    tr = TeamRepo(validate=False, verbose=False)
    print(f"Finished loading {len(tr.teams['gen9vgc2024regg'])} teams!")

    # Create players
    players = []
    for team_name, team in tr.teams["gen9vgc2024regg"].items():
        if "Ditto" in team or "Zoroark" in team or ("Dondozo" in team and "Tatsugiri" in team):
            continue
        if bool(re.search(r'[^a-zA-Z0-9\- _]', team_name)):
            continue

        players.append(CustomPlayer(
            AccountConfiguration(team_name[:17], None),
            battle_format="gen9vgc2024regg",
            team=team,
        ))

    print(f"Created {len(players)} legal players (filtered out Dondozo, Ditto and Zoroark teams)")

    # Initialize player pool and result aggregator
    player_pool = PlayerPool(players)
    result_aggregator = ResultAggregator()

    # Create an event to signal when we should stop creating new battles
    remaining_battles = asyncio.Event()
    remaining_battles.set()

    print(f"Starting {total_battles} total battles! Will update progress every second:")

    # Create and start worker tasks
    original_time = time.time()
    workers = []
    for _ in range(num_threads):
        worker = asyncio.create_task(
            battle_worker(player_pool, result_aggregator, remaining_battles)
        )
        workers.append(worker)

    # Monitor progress and stop when we reach total_battles
    while result_aggregator.total_battles < total_battles:
        pct = result_aggregator.total_battles * 100. / total_battles
        num = result_aggregator.total_battles
        print(f"\r\tCompleted {pct}% battles with {num} done...", end="", flush=True)
        await asyncio.sleep(1)

    # Signal workers to stop
    remaining_battles.clear()

    # Cancel workers
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # Print results
    print("\r\n\n======== Finished all battles!! ========")
    print(f"Finished {result_aggregator.total_battles} battles in {round(time.time() - original_time, 2)} sec")
    print(f"\tfor {round((time.time() - original_time) * 1. / result_aggregator.total_battles, 2)} sec per battle")

    success_rate = round(result_aggregator.successful_battles * 100.0 / result_aggregator.total_battles, 2)
    print(f"\nFinished with {success_rate}% success rate")

    for error_type, count in result_aggregator.error_counts.items():
        print(f"\tError Type [{error_type}] was found {count} times")

    # Write errors to file
    filename = os.path.expanduser('~/Desktop/fuzz_errors.txt')
    print(f"\nWriting errors to {filename}")
    with open(filename, 'w') as f:
        f.write(result_aggregator.error_log)

    if "print" in sys.argv:
        print(result_aggregator.error_log)

if __name__ == "__main__":
    asyncio.run(main())
