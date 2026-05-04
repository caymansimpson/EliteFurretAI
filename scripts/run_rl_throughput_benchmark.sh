#!/usr/bin/env bash
# run_rl_throughput_benchmark.sh <config> <log_dir>
#
# Runs `python src/elitefurretai/rl/train.py --config <config>` and tees stdout/stderr
# to <log_dir>/run.log. Designed to be invoked from another script that handles
# ordering and parsing. Idempotent — overwrites the log on each invocation.

set -u

CONFIG=${1:?"Usage: $0 <config_yaml> <log_dir>"}
LOG_DIR=${2:?"Usage: $0 <config_yaml> <log_dir>"}

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"

source ../venv/bin/activate

START=$(date +%s)
echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG_FILE"
echo "config=$CONFIG" | tee -a "$LOG_FILE"
echo "log_dir=$LOG_DIR" | tee -a "$LOG_FILE"
echo "git_head=$(git rev-parse HEAD)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python -u src/elitefurretai/rl/train.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
EXIT=${PIPESTATUS[0]}

END=$(date +%s)
echo "" | tee -a "$LOG_FILE"
echo "=== END $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$EXIT, elapsed=$((END-START))s) ===" | tee -a "$LOG_FILE"

exit "$EXIT"
