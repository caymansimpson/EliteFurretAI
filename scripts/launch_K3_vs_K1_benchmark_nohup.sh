#!/usr/bin/env bash
# launch_K3_vs_K1_benchmark_nohup.sh
#
# Detaches the K=3 vs K=1 RL benchmark from the calling shell so it survives
# session termination (SSH disconnect, Claude session end, etc.).
#
# Writes:
#   data/benchmarks/launch.log           — top-level log (PID, start time)
#   data/benchmarks/2026-05-03-K3/run.log
#   data/benchmarks/2026-05-03-K1/run.log
#   data/benchmarks/2026-05-03-K3-vs-K1/comparison.{json,txt}
#
# To check status later:
#   tail -f data/benchmarks/2026-05-03-K3/run.log
#   ps -p $(cat data/benchmarks/launch.pid)
#   kill $(cat data/benchmarks/launch.pid)   # to stop early

set -u
cd "$(git rev-parse --show-toplevel)"

mkdir -p data/benchmarks
LAUNCH_LOG=data/benchmarks/launch.log
PID_FILE=data/benchmarks/launch.pid

# setsid + nohup → fully detached. Stdout/err to launch.log.
nohup setsid bash scripts/run_K3_vs_K1_benchmark.sh </dev/null \
    >> "$LAUNCH_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launched run_K3_vs_K1_benchmark.sh as PID $PID" \
    | tee -a "$LAUNCH_LOG"
echo "  PID file:  $PID_FILE"
echo "  Top log:   $LAUNCH_LOG"
echo "  K3 log:    data/benchmarks/2026-05-03-K3/run.log"
echo "  K1 log:    data/benchmarks/2026-05-03-K1/run.log"
echo "  Compare:   data/benchmarks/2026-05-03-K3-vs-K1/comparison.txt"
echo ""
echo "Estimated total wall-clock: 12-16 hours (K=3 first, then K=1, sequential)"
echo "To stop: kill \$(cat $PID_FILE)"
