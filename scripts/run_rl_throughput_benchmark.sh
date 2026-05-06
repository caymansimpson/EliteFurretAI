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

# Drop poke-env per-player verbose lines as a belt-and-suspenders measure
# (train.py also silences these at the source by setting root logger to
# WARNING, but this guards against any third-party future regression that
# re-enables Showdown stdout chatter). Patterns dropped:
#   - "[INFO] M\d{2}..."  per-player logger lines (player usernames begin "M\d\d")
#   - lines starting with "|"  raw Showdown websocket payload continuations
#   - common ANSI-colored Showdown tags ([93m[1m>>>... etc)
# Drop poke-env per-player verbose lines as a belt-and-suspenders measure
# (train.py also silences these at the source by setting root logger to
# WARNING, but this guards against any third-party future regression that
# re-enables Showdown stdout chatter). Patterns dropped:
#   - "[INFO] M\d{2}..."  per-player logger lines (player usernames begin "M\d\d")
#   - lines starting with "|"  raw Showdown websocket payload continuations
#   - common ANSI-colored Showdown tags ([93m[1m>>>... etc)
#   - PS_ERROR Invalid choice retries (known residual bugs documented in
#     planning/stage2/2026-04-26-22-00-two-residual-bugs.md; poke-env retries
#     internally so training succeeds — but they were ~95% of log volume on
#     the previous K=3 run, contributing to memory pressure that crashed WSL2)
python -u src/elitefurretai/rl/train.py --config "$CONFIG" 2>&1 \
    | grep -E --line-buffered -v '^\||\[INFO\] M[0-9]{2}|\[1m(>>>|<<<)|PS_ERROR|Invalid choice|rejected open team sheets' \
    | tee -a "$LOG_FILE"
EXIT=${PIPESTATUS[0]}

END=$(date +%s)
echo "" | tee -a "$LOG_FILE"
echo "=== END $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$EXIT, elapsed=$((END-START))s) ===" | tee -a "$LOG_FILE"

exit "$EXIT"
