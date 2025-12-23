import subprocess
import time
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch multiple Pokemon Showdown servers on different ports")
    parser.add_argument(
        "--num-servers", 
        type=int, 
        default=4, 
        help="Number of Showdown servers to launch (default: 4)"
    )
    parser.add_argument(
        "--start-port", 
        type=int, 
        default=8000, 
        help="Starting port number (default: 8000)"
    )
    parser.add_argument(
        "--showdown-path",
        type=str,
        default="/home/cayman/Repositories/pokemon-showdown",
        help="Path to pokemon-showdown repository"
    )
    
    args = parser.parse_args()
    
    num_servers = args.num_servers
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
        cmd = ["node", "pokemon-showdown", "start", "--port", str(port)]
        
        print(f"Starting server on port {port}...")
        try:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
