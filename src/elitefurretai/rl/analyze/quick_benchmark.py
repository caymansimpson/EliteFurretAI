"""
Quick benchmark to measure battle throughput after ThreadPoolExecutor fix.
"""

import asyncio
import gc
import os
import signal
import subprocess
import time

import torch
from poke_env import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.players import BatchInferencePlayer, RNaDAgent
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


def launch_showdown_server(port: int) -> subprocess.Popen:
    """Launch a Showdown server on the specified port."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    showdown_dir = os.path.join(repo_root, "..", "pokemon-showdown")

    process = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=showdown_dir,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    return process


def shutdown_server(process: subprocess.Popen):
    """Gracefully shut down a server process."""
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except Exception:
        process.kill()


def create_server_config(port: int) -> ServerConfiguration:
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        None,  # type: ignore
    )


async def run_benchmark(
    num_pairs: int,
    battles_per_pair: int,
    port: int,
    model,
    team: str,
    run_id: int = 0,
):
    """Run a fixed number of battles and measure time."""

    server_config = create_server_config(port)

    players = []
    opponents = []

    for i in range(num_pairs):
        agent = RNaDAgent(model=model)

        # Use unique names per run to avoid name collision issues
        player = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"BP{run_id}_{i}", None),
            server_configuration=server_config,
            battle_format="gen9vgc2023regc",
            team=team,
            accept_open_team_sheet=True,
        )

        opponent = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"BO{run_id}_{i}", None),
            server_configuration=server_config,
            battle_format="gen9vgc2023regc",
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

    await asyncio.sleep(0.5)

    print(f"Starting {num_pairs} pairs × {battles_per_pair} battles = {num_pairs * battles_per_pair} total...")

    start_time = time.time()

    tasks = []
    for player, opponent in zip(players, opponents):
        tasks.append(player.battle_against(opponent, n_battles=battles_per_pair))

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    total_battles = sum(p.n_finished_battles for p in players)
    rate_hr = (total_battles / elapsed) * 3600

    print(f"Completed {total_battles} battles in {elapsed:.1f}s")
    print(f"Rate: {rate_hr:.0f} battles/hour")
    print(f"Per battle: {(elapsed / total_battles) * 1000:.0f}ms")

    # Cleanup - properly disconnect players from server
    for p in players:
        if hasattr(p, "_inference_future") and p._inference_future:
            p._inference_future.cancel()
        try:
            await p.stop_listening()
        except Exception:
            pass
    for o in opponents:
        if hasattr(o, "_inference_future") and o._inference_future:
            o._inference_future.cancel()
        try:
            await o.stop_listening()
        except Exception:
            pass

    # Give server time to cleanup connections
    await asyncio.sleep(1)

    return {
        "pairs": num_pairs,
        "battles": total_battles,
        "elapsed": elapsed,
        "rate_hr": rate_hr,
    }


async def main():
    print("=" * 60)
    print("BATTLE THROUGHPUT BENCHMARK (with ThreadPoolExecutor fix)")
    print("=" * 60)

    # Start a showdown server
    print("\nStarting Showdown server on port 8000...")
    server_proc = launch_showdown_server(8000)
    await asyncio.sleep(3)
    print(f"Server ready! (PID: {server_proc.pid})")

    # Load model
    print("\nLoading model...")
    device = torch.device("cuda")

    checkpoint_path = "/home/cayman/Repositories/EliteFurretAI/data/models/supervised/magic-resonance-88.pt"
    team_path = "/home/cayman/Repositories/EliteFurretAI/data/teams/gen9vgc2023regc/easy/basic.txt"

    with open(team_path) as f:
        team = f.read()

    embedder = Embedder(format="gen9vgc2023regc", omniscient=False, feature_set="full")
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
        group_sizes=embedder.group_embedding_sizes if config.get("use_grouped_encoder", False) else None,
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        turn_head_layers=config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=config.get("max_seq_len", 40),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run benchmark with different concurrency levels
    results = []

    configs = [
        (1, 20),   # 1 pair, 20 battles - baseline
        (2, 20),   # 2 pairs, 20 each
        (4, 20),   # 4 pairs, 20 each
    ]

    for run_idx, (num_pairs, battles_per_pair) in enumerate(configs):
        gc.collect()
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

        print(f"\n>>> Testing {num_pairs} concurrent pair(s)...")
        try:
            result = await run_benchmark(
                num_pairs=num_pairs,
                battles_per_pair=battles_per_pair,
                port=8000,
                model=model,
                team=team,
                run_id=run_idx,
            )
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Pairs':>6} {'Battles':>8} {'Time':>8} {'Rate/hr':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['pairs']:>6} {r['battles']:>8} {r['elapsed']:>7.1f}s {r['rate_hr']:>10.0f}")

    if results:
        best = max(results, key=lambda r: r["rate_hr"])
        print("-" * 40)
        print(f"Best: {best['pairs']} pairs → {best['rate_hr']:.0f} battles/hour")

    # Cleanup
    print("\nStopping server...")
    shutdown_server(server_proc)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
