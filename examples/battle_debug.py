# -*- coding: utf-8 -*-
"""This module plays one player against another. This is what I use to examine and print what's
happening in a battle to debug; the purpose of this file is to provide starter code for others.

Now automatically launches and shuts down Showdown server, so no need to run it separately.
"""

import asyncio
import os
import signal
import subprocess
import time
from typing import List, Optional, Union

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.rl.players import MaxDamagePlayer

# =============================================================================
# SERVER MANAGEMENT
# =============================================================================


def launch_showdown_servers(num_servers: int = 1, start_port: int = 8000) -> List[subprocess.Popen]:
    """Launch Showdown servers on consecutive ports.

    Args:
        num_servers: Number of servers to launch
        start_port: Starting port number (default 8000)

    Returns:
        List of subprocess.Popen objects for each server
    """
    print(f"\n{'=' * 70}")
    print(f"LAUNCHING {num_servers} SHOWDOWN SERVER(S)")
    print(f"Ports: {start_port}-{start_port + num_servers - 1}")
    print(f"{'=' * 70}\n")

    # Determine path to pokemon-showdown (relative to repo root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    showdown_dir = os.path.join(repo_root, "..", "pokemon-showdown")

    if not os.path.exists(showdown_dir):
        raise FileNotFoundError(
            f"Pokemon Showdown not found at {showdown_dir}\n"
            "Please clone it: git clone https://github.com/smogon/pokemon-showdown.git"
        )

    server_processes = []
    for i in range(num_servers):
        port = start_port + i
        try:
            # Launch server with stdout/stderr redirected to suppress logs
            process = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=showdown_dir,
                preexec_fn=os.setsid
            )
            server_processes.append(process)
            print(f"✓ Launched Showdown server on port {port} (PID: {process.pid})")
            time.sleep(0.5)
        except FileNotFoundError:
            print(f"ERROR: 'node' or 'pokemon-showdown' not found at {showdown_dir}")
            shutdown_showdown_servers(server_processes)
            raise
        except Exception as e:
            print(f"ERROR launching server on port {port}: {e}")
            shutdown_showdown_servers(server_processes)
            raise

    # Give servers time to fully start
    print("\nWaiting for servers to initialize...")
    time.sleep(5)  # Increased wait time for websocket server to be ready
    print(f"All {num_servers} server(s) ready!\n")
    return server_processes


def shutdown_showdown_servers(server_processes: List[subprocess.Popen]) -> None:
    """Gracefully shut down all Showdown server processes.

    Args:
        server_processes: List of subprocess.Popen objects to terminate
    """
    if not server_processes:
        return

    print(f"\n{'=' * 70}")
    print(f"SHUTTING DOWN {len(server_processes)} SHOWDOWN SERVER(S)")
    print(f"{'=' * 70}\n")

    for i, process in enumerate(server_processes):
        if process.poll() is None:  # Process is still running
            try:
                print(f"Terminating server {i + 1}/{len(server_processes)} (PID: {process.pid})...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                # Wait up to 3 seconds for graceful shutdown
                try:
                    process.wait(timeout=3)
                    print(f"✓ Server on PID {process.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    print(f"⚠ Server on PID {process.pid} didn't respond, force killing...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                print(f"Server on PID {process.pid} already terminated")
            except Exception as e:
                print(f"Error terminating server on PID {process.pid}: {e}")

    print("\n✓ All servers shut down\n")


# =============================================================================
# CUSTOM PLAYER
# =============================================================================


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
        server_configuration: ServerConfiguration = LocalhostServerConfiguration,
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

    # Place where I can implement basic logic; right now I just print that I'm in choose_move
    def choose_move(self, battle):
        if self.username == "elitefurretai":
            print()
            print("in choose_move")
            print("team:", list(map(lambda x: x.species, battle.team.values())))
            print("request:", battle.last_request)
            # return StringBattleOrder(input("Choose move: "))

        return self.choose_random_doubles_move(battle)  # pyright: ignore

    # Place where I can implement basic logic; right now I just print that I'm in teampreview
    def teampreview(self, battle):
        if self.username == "elitefurretai":
            print("in teampreview")
            print(list(map(lambda x: x.species, battle.team.values())))

        return "/team 1234"

    # Print the battle upon battle completion, and save the observations in a BattleData object to the Desktop
    def _battle_finished_callback(self, battle: AbstractBattle):
        if self.username == "elitefurretai":
            print(battle_to_str(battle))


async def main():
    pokepaste = """
        Oranguru @ Safety Goggles
        Ability: Symbiosis
        Tera Type: Steel
        EVs: 1 HP
        - Psychic

        Rillaboom @ Grassy Seed
        Ability: Grassy Surge
        Tera Type: Ground
        EVs: 1 HP
        - Grassy Glide
        - Wood Hammer

        Smeargle @ Wiki Berry
        Ability: Moody
        Tera Type: Rock
        EVs: 1 HP
        - Icy Wind

        Furret @ Mago Berry
        Ability: Frisk
        Tera Type: Rock
        EVs: 1 HP
        - Follow Me
        - Double Edge
        """

    # Launch Showdown server
    server_processes = []
    try:
        server_processes = launch_showdown_servers(num_servers=1, start_port=8000)

        p1 = CustomPlayer(
            AccountConfiguration("elitefurretai", None),
            battle_format="gen9vgc2025regi",
            team=pokepaste,
            server_configuration=LocalhostServerConfiguration,
        )

        p2 = MaxDamagePlayer(
            battle_format="gen9vgc2025regi",
            debug=True,
            team=pokepaste,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration("maxdamage", None),
        )

        # Run the battle
        print("Starting battle...")
        await p1.battle_against(p2, n_battles=100)
        print("Battle completed!")

    finally:
        # Always shut down servers, even if battle fails
        shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    asyncio.run(main())
