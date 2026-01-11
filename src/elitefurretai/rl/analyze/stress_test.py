"""
Stress test to find hardware limits.
Runs battles continuously while monitoring CPU/memory.
"""

import asyncio
import gc
import threading
import time
import uuid
from typing import List

import psutil
import torch
from poke_env import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.worker import BatchInferencePlayer
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


class ResourceMonitor:
    """Monitor CPU and memory usage in a separate thread."""

    def __init__(self):
        self.running = False
        self.cpu_samples = []
        self.mem_samples = []
        self.thread = None

    def start(self):
        self.running = True
        self.cpu_samples = []
        self.mem_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _monitor_loop(self):
        while self.running:
            self.cpu_samples.append(psutil.cpu_percent(interval=0.5))
            self.mem_samples.append(psutil.Process().memory_info().rss / (1024**3))

    def get_stats(self):
        if not self.cpu_samples:
            return {"cpu_avg": 0, "cpu_max": 0, "mem_avg": 0, "mem_max": 0}
        return {
            "cpu_avg": sum(self.cpu_samples) / len(self.cpu_samples),
            "cpu_max": max(self.cpu_samples),
            "mem_avg": sum(self.mem_samples) / len(self.mem_samples),
            "mem_max": max(self.mem_samples),
        }


def create_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        None,  # type: ignore
    )


async def run_stress_test(
    num_servers: int,
    pairs_per_server: int,
    duration_seconds: int,
    available_ports: List[int],
    model,
    format_: str,
    team: str,
    run_id: str,
):
    """Run battles for a fixed duration and measure throughput."""

    total_pairs = num_servers * pairs_per_server

    # Create players distributed across servers
    players = []
    opponents = []

    for pair_idx in range(total_pairs):
        server_idx = pair_idx % num_servers
        port = available_ports[server_idx]
        server_config = create_server_config(port)

        agent = RNaDAgent(model=model)

        player = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"P{run_id}_{pair_idx}", None),
            server_configuration=server_config,
            battle_format=format_,
            team=team,
            accept_open_team_sheet=True,
        )

        opponent = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"O{run_id}_{pair_idx}", None),
            server_configuration=server_config,
            battle_format=format_,
            team=team,
            accept_open_team_sheet=True,
        )

        players.append(player)
        opponents.append(opponent)

    # Start inference loops
    for p in players:
        p.start_inference_loop()
    for o in opponents:
        o.start_inference_loop()

    await asyncio.sleep(0.3)

    # Start resource monitor
    monitor = ResourceMonitor()
    monitor.start()

    # Run battles for fixed duration
    start_time = time.time()
    battle_count = 0
    battles_per_batch = 50  # Each pair runs this many battles per batch
    remaining = 0.0

    print(
        f"  Running for {duration_seconds}s with {total_pairs} concurrent pairs...",
        flush=True,
    )

    while time.time() - start_time < duration_seconds:
        tasks = []
        for player, opponent in zip(players, opponents):
            remaining = duration_seconds - (time.time() - start_time)
            if remaining < 10:  # Not enough time for more battles
                break
            tasks.append(player.battle_against(opponent, n_battles=battles_per_batch))

        if not tasks:
            break

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=min(300, remaining + 30),  # 5 min or remaining time + buffer
            )
        except asyncio.TimeoutError:
            print("  [batch timeout]", end="", flush=True)

        # Update count
        new_count = sum(p.n_finished_battles for p in players)
        battle_count = new_count
        elapsed = time.time() - start_time
        rate = (battle_count / elapsed) * 3600 if elapsed > 0 else 0
        print(f"  [{elapsed:.0f}s] {battle_count} battles, {rate:.0f}/hr", flush=True)

    monitor.stop()
    stats = monitor.get_stats()

    elapsed = time.time() - start_time
    final_count = sum(p.n_finished_battles for p in players)

    # Cleanup
    for p in players:
        if hasattr(p, "_inference_future"):
            p._inference_future.cancel()
    for o in opponents:
        if hasattr(o, "_inference_future"):
            o._inference_future.cancel()

    del players, opponents
    gc.collect()

    return {
        "config": f"{num_servers}srv × {pairs_per_server}pairs",
        "total_pairs": total_pairs,
        "duration": elapsed,
        "battles": final_count,
        "rate_hr": (final_count / elapsed) * 3600 if elapsed > 0 else 0,
        "cpu_avg": stats["cpu_avg"],
        "cpu_max": stats["cpu_max"],
        "mem_avg_gb": stats["mem_avg"],
        "mem_max_gb": stats["mem_max"],
    }


async def main():
    print("=" * 70)
    print("STRESS TEST: Finding Hardware Limits")
    print("=" * 70)

    # Configuration
    available_ports = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
    duration_per_test = 90  # 90 seconds per config
    battle_format = "gen9vgc2023regc"
    embedder_format = "gen9vgc2023regulationc"
    team_path = (
        "/home/cayman/Repositories/EliteFurretAI/data/teams/gen9vgc2023regc/easy/basic.txt"
    )
    checkpoint_path = "/home/cayman/Repositories/EliteFurretAI/data/models/supervised/ethereal-flower-59-extended.pt"

    with open(team_path) as f:
        team = f.read()

    # Load model once
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = Embedder(format=embedder_format, omniscient=False, feature_set="full")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config.get("lstm_layers", 2),
        lstm_hidden_size=config.get("lstm_hidden_size", 512),
        dropout=config.get("dropout", 0.1),
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 8),
        late_attention_heads=config.get("late_attention_heads", 8),
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=embedder.group_embedding_sizes
        if config.get("use_grouped_encoder", False)
        else None,
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        turn_head_layers=config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=17,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test duration per config: {duration_per_test}s")

    # Test configs: push beyond 16 concurrent
    configs = [
        (8, 2),  # 16 concurrent (current best: 2756/hr)
        (8, 3),  # 24 concurrent
        (8, 4),  # 32 concurrent
    ]

    results = []
    run_id = uuid.uuid4().hex[:4]

    print("\n" + "=" * 70)

    for num_servers, pairs_per_server in configs:
        gc.collect()
        torch.cuda.empty_cache()
        await asyncio.sleep(2)

        print(
            f"\n>>> Testing {num_servers} servers × {pairs_per_server} pairs/server = {num_servers * pairs_per_server} concurrent"
        )

        try:
            result = await run_stress_test(
                num_servers=num_servers,
                pairs_per_server=pairs_per_server,
                duration_seconds=duration_per_test,
                available_ports=available_ports,
                model=model,
                format_=battle_format,
                team=team,
                run_id=f"{run_id}_{num_servers}_{pairs_per_server}",
            )
            results.append(result)
            print(
                f"  RESULT: {result['rate_hr']:.0f}/hr | CPU avg/max: {result['cpu_avg']:.0f}%/{result['cpu_max']:.0f}% | RAM: {result['mem_max_gb']:.1f}GB"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Hardware Scaling Results")
    print("=" * 70)
    print(
        f"{'Config':<20} {'Pairs':>6} {'Rate/hr':>10} {'CPU avg':>8} {'CPU max':>8} {'RAM max':>8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['config']:<20} {r['total_pairs']:>6} {r['rate_hr']:>10.0f} {r['cpu_avg']:>7.0f}% {r['cpu_max']:>7.0f}% {r['mem_max_gb']:>7.1f}GB"
        )

    if results:
        best = max(results, key=lambda r: r["rate_hr"])
        print("-" * 70)
        print(f"BEST: {best['config']} → {best['rate_hr']:.0f} battles/hour")

        # Find scaling efficiency
        baseline = next((r for r in results if r["total_pairs"] == 4), None)
        if baseline:
            print("\nScaling efficiency (vs 4 pairs baseline):")
            for r in results:
                if r["total_pairs"] > 4:
                    expected = baseline["rate_hr"] * (r["total_pairs"] / 4)
                    actual = r["rate_hr"]
                    efficiency = (actual / expected) * 100 if expected > 0 else 0
                    print(
                        f"  {r['total_pairs']} pairs: {actual:.0f}/hr (expected: {expected:.0f}, efficiency: {efficiency:.0f}%)"
                    )


if __name__ == "__main__":
    asyncio.run(main())
