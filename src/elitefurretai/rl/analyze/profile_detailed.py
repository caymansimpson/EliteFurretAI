#!/usr/bin/env python3
"""
Detailed profiling of embedding time vs inference time.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from functools import wraps

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Configure logging before importing poke-env
import logging
logging.getLogger("poke_env").setLevel(logging.WARNING)

from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.supervised.behavior_clone_player import BCPlayer


MODEL_PATH = "data/models/supervised/dauntless-hill-95.pt"
BATTLE_FORMAT = "gen9vgc2023regc"
SHOWDOWN_DIR = os.path.expanduser("~/Repositories/pokemon-showdown")
START_PORT = 8000


# Global timing accumulators
EMBED_TIME = 0.0
EMBED_COUNT = 0
INFERENCE_TIME = 0.0
INFERENCE_COUNT = 0


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


class TimingBCPlayer(BCPlayer):
    """BCPlayer with timing instrumentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_times = []
        self.predict_times = []
        self.total_choose_move_times = []
    
    def embed_battle_state(self, battle):
        """Timed version of embed_battle_state."""
        start = time.perf_counter()
        result = super().embed_battle_state(battle)
        elapsed = time.perf_counter() - start
        self.embed_times.append(elapsed)
        return result
    
    def predict(self, traj, battle, action_type=None):
        """Timed version of predict."""
        start = time.perf_counter()
        result = super().predict(traj, battle, action_type)
        elapsed = time.perf_counter() - start
        self.predict_times.append(elapsed)
        return result


async def profile_with_timing():
    """Profile with detailed timing breakdown."""
    
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
        
        print("\n=== TIMING BREAKDOWN ===")
        
        # Create timed player
        player = TimingBCPlayer(
            unified_model_filepath=MODEL_PATH,
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("TimingP", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            probabilistic=True,
            max_concurrent_battles=1,
            team=team,
            verbose=False,
        )
        
        # Create random opponent
        opponent = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("TimingO", None),
            max_concurrent_battles=1,
            team=team,
        )
        
        # Run battles
        num_battles = 5
        print(f"Running {num_battles} battles...")
        
        total_start = time.time()
        await player.battle_against(opponent, n_battles=num_battles)
        total_time = time.time() - total_start
        
        # Report timing
        print(f"\nTotal time: {total_time:.1f}s for {num_battles} battles")
        print(f"Battles/sec: {num_battles/total_time:.3f}")
        
        if player.embed_times:
            avg_embed = sum(player.embed_times) / len(player.embed_times)
            total_embed = sum(player.embed_times)
            print(f"\nEmbedding:")
            print(f"  Count: {len(player.embed_times)}")
            print(f"  Avg: {avg_embed*1000:.1f}ms")
            print(f"  Total: {total_embed:.1f}s ({total_embed/total_time*100:.1f}% of battle time)")
        
        if player.predict_times:
            avg_predict = sum(player.predict_times) / len(player.predict_times)
            total_predict = sum(player.predict_times)
            print(f"\nInference:")
            print(f"  Count: {len(player.predict_times)}")
            print(f"  Avg: {avg_predict*1000:.1f}ms")
            print(f"  Total: {total_predict:.1f}s ({total_predict/total_time*100:.1f}% of battle time)")
        
        overhead = total_time - sum(player.embed_times) - sum(player.predict_times)
        print(f"\nOther overhead: {overhead:.1f}s ({overhead/total_time*100:.1f}%)")
        
        # Cleanup
        await player.stop_listening()
        await opponent.stop_listening()
        
    finally:
        stop_server(server_proc)


async def test_embedding_speed():
    """Test embedding speed directly with a real battle state."""
    
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
        
        # Create players and run one battle to get a battle state
        player = BCPlayer(
            unified_model_filepath=MODEL_PATH,
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("EmbedP", None),
            device="cuda" if torch.cuda.is_available() else "cpu",
            probabilistic=True,
            max_concurrent_battles=1,
            team=team,
            verbose=False,
        )
        
        opponent = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            account_configuration=AccountConfiguration("EmbedO", None),
            max_concurrent_battles=1,
            team=team,
        )
        
        print("Running one battle to capture battle states...")
        await player.battle_against(opponent, n_battles=1)
        
        # Get a battle state
        if player._battles:
            battle = list(player._battles.values())[0]
            
            print("\n=== EMBEDDING SPEED TEST ===")
            
            # Test different feature sets
            for feature_set in [Embedder.SIMPLE, Embedder.RAW, Embedder.FULL]:
                embedder = Embedder(
                    format=BATTLE_FORMAT,
                    feature_set=feature_set,
                    omniscient=False,
                )
                
                # Warmup
                for _ in range(3):
                    _ = embedder.embed(battle)
                
                # Time it
                num_iterations = 20
                start = time.perf_counter()
                for _ in range(num_iterations):
                    _ = embedder.embed(battle)
                elapsed = time.perf_counter() - start
                
                avg_time = elapsed / num_iterations
                print(f"{feature_set:8s}: {avg_time*1000:.1f}ms avg, {embedder.embedding_size} features")
        
        # Cleanup
        await player.stop_listening()
        await opponent.stop_listening()
        
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["timing", "embedding", "all"], default="all")
    args = parser.parse_args()
    
    if args.test in ["embedding", "all"]:
        asyncio.run(test_embedding_speed())
    
    if args.test in ["timing", "all"]:
        asyncio.run(profile_with_timing())
