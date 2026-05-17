# -*- coding: utf-8 -*-
"""Microbench: measure cross-process sync_weights latency.

Plan C Step 3 — before integrating multi-process inference into train.py,
verify that shipping a state_dict across an mp.Queue is fast enough to
not dominate the broadcast tick.

Context: the trainer calls `registry.sync_weights("name", state_dict)`
every `checkpoint_interval=100` updates for the main agent (and more
often for exploiter). The current in-process path is just
`load_state_dict` — ~ms. The Plan C subprocess path adds:
  1. Pickle state_dict (CPU tensors, 26M params ≈ 100 MB)
  2. mp.Queue.put — shared-memory for tensor storages, but the
     dict + metadata are still pickled
  3. Subprocess receives + unpickles
  4. load_state_dict in subprocess

Step 3's pass criterion: trainer-side `sync_weights()` returns in
≤ 1 second on average for a production-shape model. If slower, switch
to shared-memory handles (`Tensor.share_memory_()`) instead of pickling.

The "trainer-side latency" measured here is what blocks the broadcast
tick. The subprocess's apply happens asynchronously after.

What this does NOT measure: end-to-end latency (when do future forwards
actually reflect the new weights?). That's harder to instrument without
adding diagnostic plumbing to the subprocess; if trainer-side is fast,
we can verify end-to-end correctness via the existing unit tests.

Run:
  python -u src/elitefurretai/rl/analyze/bench_sync_weights_latency.py
"""

from __future__ import annotations

import logging
import statistics
import time
from typing import List

import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.model_registry import ModelRegistry
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

logger = logging.getLogger(__name__)


def _make_production_shape_agent() -> RNaDAgent:
    """Construct an RNaDAgent with sep_arch.yaml's production architecture.

    Targets ~26M parameters to match the cool-bee-85-finetune model that
    sep_arch fine-tunes from.
    """
    embedder = Embedder(feature_set="raw")
    model = TransformerThreeHeadedModel(
        embedder=embedder,
        early_layers=[1024, 512, 512],
        late_layers=[512, 512],
        transformer_layers=4,
        transformer_heads=8,
        transformer_ff_dim=1024,
        dropout=0.0,
        max_seq_len=40,
        grouped_encoder_hidden_dim=256,
        grouped_encoder_aggregated_dim=2048,
        pokemon_attention_heads=8,
        teampreview_head_layers=[256, 128],
        teampreview_head_dropout=0.0,
        teampreview_attention_heads=2,
        turn_head_layers=[512, 256, 256],
    )
    model.eval()
    return RNaDAgent(model)


def _measure_sync_latencies(
    registry: ModelRegistry,
    service_name: str,
    agent: RNaDAgent,
    n_iters: int = 20,
    new_state_dicts: List[dict] | None = None,
) -> List[float]:
    """Call registry.sync_weights N times; return per-call latency in seconds."""
    if new_state_dicts is None:
        # Pre-build N distinct state_dicts so we don't measure
        # state_dict construction cost — only the sync cost.
        new_state_dicts = []
        for _ in range(n_iters):
            torch.manual_seed(int(time.time_ns()) % (2**31))
            ag = _make_production_shape_agent()
            new_state_dicts.append(
                {k: v.clone() for k, v in ag.model.state_dict().items()}
            )

    latencies: List[float] = []
    for sd in new_state_dicts[:n_iters]:
        start = time.perf_counter()
        registry.sync_weights(service_name, sd)
        end = time.perf_counter()
        latencies.append(end - start)
    return latencies


def _print_stats(label: str, latencies: List[float]) -> None:
    if not latencies:
        print(f"{label}: (no samples)")
        return
    n = len(latencies)
    mean_ms = statistics.mean(latencies) * 1000
    median_ms = statistics.median(latencies) * 1000
    min_ms = min(latencies) * 1000
    max_ms = max(latencies) * 1000
    if n >= 2:
        stdev_ms = statistics.stdev(latencies) * 1000
    else:
        stdev_ms = 0.0
    print(
        f"{label}: n={n}  mean={mean_ms:.1f} ms  median={median_ms:.1f} ms  "
        f"min={min_ms:.1f}  max={max_ms:.1f}  stdev={stdev_ms:.1f}"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    print("Building production-shape agent (sep_arch dimensions)...")
    initial_agent = _make_production_shape_agent()
    n_params = sum(p.numel() for p in initial_agent.model.parameters())
    state_dict_bytes = sum(
        v.numel() * v.element_size() for v in initial_agent.model.state_dict().values()
    )
    print(f"  num_params: {n_params:,}")
    print(f"  state_dict size: {state_dict_bytes / (1024 * 1024):.1f} MB")

    # Pre-build N state_dicts so latency measurements don't include
    # construction cost.
    n_iters = 20
    print(f"\nPre-building {n_iters} fresh state_dicts for sync test...")
    new_sds: List[dict] = []
    for i in range(n_iters):
        torch.manual_seed(1000 + i)
        ag = _make_production_shape_agent()
        new_sds.append({k: v.clone() for k, v in ag.model.state_dict().items()})

    # ── In-process baseline ───────────────────────────────────────────
    print("\n=== In-process baseline (process_group=None) ===")
    registry = ModelRegistry(
        num_workers=1,
        batch_size=4,
        batch_timeout=0.005,
        device="cpu",
    )
    try:
        registry.register("test_inproc", initial_agent, compile=False)
        registry.start_all()
        latencies_in = _measure_sync_latencies(
            registry,
            "test_inproc",
            initial_agent,
            n_iters=n_iters,
            new_state_dicts=new_sds,
        )
        _print_stats("in_process", latencies_in)
    finally:
        registry.stop_all()

    # ── Subprocess (Plan C path) ──────────────────────────────────────
    print("\n=== Subprocess (process_group='ghosts') ===")
    initial_agent_sub = _make_production_shape_agent()
    registry2 = ModelRegistry(
        num_workers=1,
        batch_size=4,
        batch_timeout=0.005,
        device="cpu",
    )
    try:
        registry2.register(
            "test_subproc", initial_agent_sub, compile=False, process_group="ghosts"
        )
        registry2.start_all()
        # Brief wait for subprocess startup before timing — we don't
        # want startup cost contaminating the first sample.
        time.sleep(5.0)
        latencies_sub = _measure_sync_latencies(
            registry2,
            "test_subproc",
            initial_agent_sub,
            n_iters=n_iters,
            new_state_dicts=new_sds,
        )
        _print_stats("subprocess (trainer-side)", latencies_sub)
    finally:
        registry2.stop_all()

    # ── Summary + verdict ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("VERDICT — Plan C Step 3 acceptance: subprocess sync ≤ 1.0 s mean")
    print("═" * 70)
    mean_in = statistics.mean(latencies_in) * 1000
    mean_sub = statistics.mean(latencies_sub) * 1000
    print(f"  In-process mean:  {mean_in:.1f} ms  (load_state_dict only)")
    print(
        f"  Subprocess mean:  {mean_sub:.1f} ms  (pickle + queue.put + CPU shadow update)"
    )
    print(f"  Overhead:         +{mean_sub - mean_in:.1f} ms  ({mean_sub / mean_in:.2f}x)")
    if mean_sub < 1000:
        print(f"  ✓ PASS — subprocess sync mean < 1.0 s ({mean_sub:.0f} ms)")
        print("    Pickling + queue path is fast enough for production. Ship as-is.")
    else:
        print(f"  ✗ FAIL — subprocess sync mean > 1.0 s ({mean_sub:.0f} ms)")
        print("    Switch to Tensor.share_memory_() handles before Step 4.")


if __name__ == "__main__":
    main()
