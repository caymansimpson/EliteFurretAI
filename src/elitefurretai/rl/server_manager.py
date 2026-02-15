"""Pokemon Showdown server lifecycle and allocation helpers."""

import os
import signal
import subprocess
import time
from typing import List, Optional, Tuple


def launch_showdown_servers(num_servers: int, start_port: int = 8000) -> List[subprocess.Popen]:
    print(f"\n{'=' * 60}")
    print(f"LAUNCHING {num_servers} SHOWDOWN SERVERS")
    print(f"Ports: {start_port}-{start_port + num_servers - 1}")
    print(f"{'=' * 60}\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    showdown_dir = os.path.join(repo_root, "..", "pokemon-showdown")

    if not os.path.exists(showdown_dir):
        raise FileNotFoundError(f"Pokemon Showdown not found at {showdown_dir}")

    server_processes = []
    for i in range(num_servers):
        port = start_port + i
        try:
            process = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=showdown_dir,
                preexec_fn=os.setsid,
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

    print("\nWaiting for servers to initialize...")
    time.sleep(2)
    print(f"All {num_servers} servers ready!\n")
    return server_processes


def shutdown_showdown_servers(server_processes: List[subprocess.Popen]) -> None:
    if not server_processes:
        return

    print(f"\n{'=' * 60}")
    print(f"SHUTTING DOWN {len(server_processes)} SHOWDOWN SERVERS")
    print(f"{'=' * 60}\n")

    for i, process in enumerate(server_processes):
        if process.poll() is None:
            try:
                print(f"Terminating server {i + 1}/{len(server_processes)} (PID: {process.pid})...")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)

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
                print(f"Error terminating server PID {process.pid}: {e}")

    print("\nAll servers shut down.\n")


def allocate_server_ports(
    num_workers: int,
    players_per_worker: int,
    num_showdown_servers: int,
    max_players_per_server: int,
    showdown_start_port: int,
) -> Tuple[List[int], List[int]]:
    total_players = num_workers * players_per_worker
    total_capacity = num_showdown_servers * max_players_per_server

    if total_players > total_capacity:
        raise ValueError(
            "Not enough Showdown server capacity. "
            f"Need {total_players} concurrent players but only have capacity for {total_capacity}. "
            "Increase num_showdown_servers or max_players_per_server (see OPTIMIZATIONS.md)."
        )

    server_loads = [0 for _ in range(num_showdown_servers)]
    worker_ports: List[int] = []

    for _ in range(num_workers):
        selected_idx: Optional[int] = None
        for idx in range(num_showdown_servers):
            projected = server_loads[idx] + players_per_worker
            if projected <= max_players_per_server:
                if selected_idx is None or server_loads[idx] < server_loads[selected_idx]:
                    selected_idx = idx

        if selected_idx is None:
            raise ValueError(
                "Unable to allocate worker to a Showdown server without exceeding max_players_per_server. "
                "Try increasing num_showdown_servers or relaxing the per-server cap."
            )

        server_loads[selected_idx] += players_per_worker
        worker_ports.append(showdown_start_port + selected_idx)

    return worker_ports, server_loads


__all__ = [
    "launch_showdown_servers",
    "shutdown_showdown_servers",
    "allocate_server_ports",
]
