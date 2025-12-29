import argparse
import os
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple Pokemon Showdown servers on different ports"
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=None,
        help="Number of Showdown servers to launch (or read from config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (reads num_showdown_servers if --num-servers not provided)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=8000,
        help="Starting port number (default: 8000), or read by config's showdown_start_port",
    )
    parser.add_argument(
        "--showdown-path",
        type=str,
        default="/home/cayman/Repositories/pokemon-showdown",
        help="Path to pokemon-showdown repository",
    )

    args = parser.parse_args()

    # Determine number of servers
    num_servers = args.num_servers
    if args.config is not None:
        # Read from config file
        try:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            num_servers = config.get("num_showdown_servers", 4)
            start_port = config.get("showdown_start_port", args.start_port)
        except Exception as e:
            print(f"Warning: Could not read config file {args.config}: {e}")
            sys.exit(1)
    else:
        print("Couldnt find config or num-servers argument. Exiting.")
        sys.exit(1)

    start_port = args.start_port
    REPO_ROOT = args.showdown_path

    if not os.path.exists(REPO_ROOT):
        print(f"Error: Pokemon Showdown not found at {REPO_ROOT}")
        return

    procs = []
    for i in range(num_servers):
        port = start_port + i
        # Command to start showdown on a specific port
        # Usually: node pokemon-showdown start --port 8000
        cmd = ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)]

        print(f"Starting server on port {port}...")
        try:
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            procs.append(proc)
        except Exception as e:
            print(f"Failed to start server on port {port}: {e}")

    print(f"Started {len(procs)} servers. Press Ctrl+C to stop.")
    print(f"Servers running on ports: {start_port} to {start_port + len(procs) - 1}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        for p in procs:
            p.terminate()
        print("All servers stopped.")


if __name__ == "__main__":
    main()
