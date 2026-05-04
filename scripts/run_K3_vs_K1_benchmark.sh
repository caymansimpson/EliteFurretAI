#!/usr/bin/env bash
# run_K3_vs_K1_benchmark.sh
#
# Orchestrates the K=3 (new default) vs K=1 (pre-Phase-3 behavior) RL training
# comparison. Runs K=3 first, then K=1, both for 200 updates with vgc-bench
# disabled and wandb off. Parses each log into summary.json. Writes a
# comparison.json + comparison.txt at the end.
#
# Total wall-clock estimate (RTX 3090 + 8-core WSL2): ~5-7 hours sequentially.
# Logs land in:
#   data/benchmarks/2026-05-03-K3/run.log + summary.json
#   data/benchmarks/2026-05-03-K1/run.log + summary.json
#   data/benchmarks/2026-05-03-K3-vs-K1/comparison.{json,txt}

set -u

K3_DIR=data/benchmarks/2026-05-03-K3
K1_DIR=data/benchmarks/2026-05-03-K1
COMPARE_DIR=data/benchmarks/2026-05-03-K3-vs-K1

mkdir -p "$K3_DIR" "$K1_DIR" "$COMPARE_DIR"

cd "$(git rev-parse --show-toplevel)"

echo "=== Phase 1/2: K=3 (new default) ==="
bash scripts/run_rl_throughput_benchmark.sh \
    src/elitefurretai/rl/configs/benchmark_K3.yaml \
    "$K3_DIR"
K3_EXIT=$?

echo ""
echo "=== Phase 2/2: K=1 (pre-Phase-3 behavior) ==="
bash scripts/run_rl_throughput_benchmark.sh \
    src/elitefurretai/rl/configs/benchmark_K1.yaml \
    "$K1_DIR"
K1_EXIT=$?

echo ""
echo "=== Parsing logs ==="
source ../venv/bin/activate
python scripts/parse_rl_benchmark.py "$K3_DIR" "$K1_DIR"

echo ""
echo "=== Building comparison ==="
python <<EOF
import json, pathlib
k3 = json.loads(pathlib.Path("$K3_DIR/summary.json").read_text())
k1 = json.loads(pathlib.Path("$K1_DIR/summary.json").read_text())

def line(label, k1v, k3v):
    return f"  {label:35s}  K=1: {k1v:>14}   K=3: {k3v:>14}"

lines = [
    "RL throughput benchmark — K=3 vs K=1",
    "=" * 70,
    "",
    f"  K=1 config:   {k1['config']}",
    f"  K=3 config:   {k3['config']}",
    f"  Git HEAD:     {k3['git_head']}",
    "",
    line("Updates completed",
         k1["aggregate"]["n_updates_total"],
         k3["aggregate"]["n_updates_total"]),
    line("Wall-clock total (s)",
         k1["elapsed_seconds"], k3["elapsed_seconds"]),
    line("Sec / update (warm)",
         f"{k1['aggregate']['wall_clock_per_update_mean_s']:.1f}",
         f"{k3['aggregate']['wall_clock_per_update_mean_s']:.1f}"),
    line("b/s mean (warm)",
         f"{k1['aggregate']['b_per_s_mean']:.3f}",
         f"{k3['aggregate']['b_per_s_mean']:.3f}"),
    line("learner steps/s (warm)",
         f"{k1['aggregate']['learner_steps_per_s_mean']:.2f}",
         f"{k3['aggregate']['learner_steps_per_s_mean']:.2f}"),
    "",
    "Win rates at milestone updates (lower wall-clock at equal win rate = win)",
    "",
]

for milestone in ("50", "100", "150", "200"):
    if milestone in k1.get("win_rate_at_update", {}) and milestone in k3.get(
        "win_rate_at_update", {}
    ):
        lines.append(f"  Update {milestone}:")
        # Time elapsed at that update (lookup in updates list)
        def t_at(summary, n):
            for u in summary["updates"]:
                if u["update"] == int(n):
                    return u["elapsed_s_total"]
            return None

        t1 = t_at(k1, milestone)
        t3 = t_at(k3, milestone)
        lines.append(f"    wall-clock (s)    K=1: {t1}    K=3: {t3}")
        opps = sorted(
            set(k1["win_rate_at_update"][milestone]) | set(k3["win_rate_at_update"][milestone])
        )
        for opp in opps:
            v1 = k1["win_rate_at_update"][milestone].get(opp, None)
            v3 = k3["win_rate_at_update"][milestone].get(opp, None)
            v1s = f"{v1:.1%}" if v1 is not None else "—"
            v3s = f"{v3:.1%}" if v3 is not None else "—"
            lines.append(f"    {opp:30s}  K=1: {v1s:>7}   K=3: {v3s:>7}")
        lines.append("")

text = "\n".join(lines)
print(text)
pathlib.Path("$COMPARE_DIR/comparison.txt").write_text(text + "\n")
pathlib.Path("$COMPARE_DIR/comparison.json").write_text(json.dumps(
    {"k1": k1, "k3": k3}, indent=2
))
print(f"\nWritten: $COMPARE_DIR/comparison.{{txt,json}}")
EOF

echo ""
echo "=== Done. K3_exit=$K3_EXIT  K1_exit=$K1_EXIT ==="
exit "$((K3_EXIT | K1_EXIT))"
