#!/usr/bin/env bash
# launch_K1_only_nohup.sh
#
# Runs ONLY the K=1 benchmark (the K=3 partial run from 2026-05-04 produced
# 116 updates of clean data before WSL2 crashed at 5h+ of sustained training).
# K=1 is capped at 100 updates / ~4 hours to stay well clear of whatever
# accumulation pressure took down WSL the last time.
#
# Writes:
#   data/benchmarks/launch_K1.log
#   data/benchmarks/launch_K1.pid
#   data/benchmarks/2026-05-03-K1/run.log
#   data/benchmarks/2026-05-03-K1/summary.json   (after run completes)
#
# To monitor / stop:
#   tail -f data/benchmarks/2026-05-03-K1/run.log
#   ps -p $(cat data/benchmarks/launch_K1.pid)
#   kill $(cat data/benchmarks/launch_K1.pid)

set -u
cd "$(git rev-parse --show-toplevel)"

mkdir -p data/benchmarks/2026-05-03-K1
LAUNCH_LOG=data/benchmarks/launch_K1.log
PID_FILE=data/benchmarks/launch_K1.pid

# Inner script: runs the benchmark, then parses the log into summary.json.
INNER=$(cat <<'INNER_EOF'
set -u
cd "$(git rev-parse --show-toplevel)"
bash scripts/run_rl_throughput_benchmark.sh \
    src/elitefurretai/rl/configs/benchmark_K1.yaml \
    data/benchmarks/2026-05-03-K1
EXIT=$?
source ../venv/bin/activate
python scripts/parse_rl_benchmark.py data/benchmarks/2026-05-03-K1 \
                                     data/benchmarks/2026-05-03-K3
exit $EXIT
INNER_EOF
)

nohup setsid bash -c "$INNER" </dev/null >> "$LAUNCH_LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launched K=1-only benchmark as PID $PID" \
    | tee -a "$LAUNCH_LOG"
echo "  PID file:  $PID_FILE"
echo "  Top log:   $LAUNCH_LOG"
echo "  K1 log:    data/benchmarks/2026-05-03-K1/run.log"
echo "  K3 data:   data/benchmarks/2026-05-03-K3/run.log (already collected)"
echo ""
echo "Estimated wall-clock: ~4 hours (100 updates at ~150s each)"
echo "To stop: kill \$(cat $PID_FILE)"
