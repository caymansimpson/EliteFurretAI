#!/usr/bin/env python3
"""
Profile BCPlayer inference to understand the bottleneck.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Configure logging before importing poke-env
import logging

logging.getLogger("poke_env").setLevel(logging.WARNING)

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.supervised.behavior_clone_player import BCPlayer

MODEL_PATH = "data/models/supervised/dauntless-hill-95.pt"
BATTLE_FORMAT = "gen9vgc2023regc"
SHOWDOWN_DIR = os.path.expanduser("~/Repositories/pokemon-showdown")
START_PORT = 8000


def wait_for_port(port: int, timeout: float = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except socket.error:
            pass
        time.sleep(0.5)
    return False


def start_server(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=SHOWDOWN_DIR,
    )
    return proc


def stop_server(proc: subprocess.Popen):
    proc.terminate()
    time.sleep(1)
    if proc.poll() is None:
        proc.kill()


async def profile_bc_player():
    """Profile BCPlayer to find bottlenecks."""

    # Start server
    print("Starting server...")
    server_proc = start_server(START_PORT)
    if not wait_for_port(START_PORT):
        print("ERROR: Server failed to start")
        stop_server(server_proc)
        return

    print("Server ready")

    try:
        # Load team
        team_repo = TeamRepo("data/teams")
        team = team_repo.sample_team(BATTLE_FORMAT, subdirectory="straightforward")

        # Create server config
        server_config = ServerConfiguration(
            f"ws://localhost:{START_PORT}/showdown/websocket",
            None,
        )

        # Time model loading
        print("\n=== MODEL LOADING ===")
        load_start = time.time()

        player = BCPlayer(
            unified_model_filepath=MODEL_PATH,
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("ProfileP", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            probabilistic=True,
            max_concurrent_battles=1,
            team=team,
            verbose=True,
        )

        load_time = time.time() - load_start
        print(f"Model loading time: {load_time:.2f}s")

        # Create random opponent
        opponent = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("ProfileO", None),
            max_concurrent_battles=1,
            team=team,
        )

        # Run a few battles with profiling
        print("\n=== BATTLE PROFILING ===")
        num_battles = 3  # Reduced for faster testing

        battle_start = time.time()

        # Add a timeout
        try:
            await asyncio.wait_for(
                player.battle_against(opponent, n_battles=num_battles),
                timeout=180.0  # 3 minute timeout
            )
        except asyncio.TimeoutError:
            print("WARNING: Battle timed out!")

        battle_time = time.time() - battle_start

        print(f"\nCompleted {num_battles} battles in {battle_time:.1f}s")
        print(f"Battles/sec: {num_battles/battle_time:.3f}")
        print(f"Seconds/battle: {battle_time/num_battles:.2f}")

        # Cleanup
        await player.stop_listening()
        await opponent.stop_listening()

    finally:
        stop_server(server_proc)


async def profile_embedder():
    """Profile the embedder separately."""
    from elitefurretai.etl.embedder import Embedder

    print("\n=== EMBEDDER PROFILING ===")

    # Test different feature sets
    for feature_set in [Embedder.SIMPLE, Embedder.RAW, Embedder.FULL]:
        embedder = Embedder(
            format=BATTLE_FORMAT,
            feature_set=feature_set,
            omniscient=False,
        )

        print(f"\nFeature set: {feature_set}")
        print(f"Embedding size: {embedder.embedding_size}")

    # To properly time, we'd need a real battle state
    # Let's create a quick test with the actual battles


async def test_inference_time():
    """Test raw model inference time."""
    from elitefurretai.rl.train import load_model

    print("\n=== RAW INFERENCE TIMING ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = load_model(MODEL_PATH, device)
    model.eval()

    # Get input size from embedder
    from elitefurretai.etl.embedder import Embedder
    embedder = Embedder(format=BATTLE_FORMAT, feature_set=Embedder.FULL, omniscient=False)
    input_size = embedder.embedding_size

    # Create dummy input
    batch_size = 1
    seq_len = 20

    dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # Time inference
    if device == "cuda":
        torch.cuda.synchronize()

    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_time = total_time / num_iterations

    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Inferences per second: {num_iterations/total_time:.1f}")

    # Test with larger batch
    print("\n--- Batched inference ---")
    for batch_size in [1, 4, 8, 16, 32]:
        dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations

        print(f"Batch size {batch_size:3d}: {avg_time*1000:.2f}ms/batch, {batch_size*num_iterations/total_time:.1f} samples/sec")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["battle", "inference", "all"], default="all")
    args = parser.parse_args()

    if args.test in ["inference", "all"]:
        asyncio.run(test_inference_time())

    if args.test in ["battle", "all"]:
        asyncio.run(profile_bc_player())
