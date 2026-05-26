#!/bin/bash
# Restart stage 2 fresh on the existing regc_final_v5 data, then run the
# action diagnostic on the resulting checkpoint. Does NOT auto-fine-tune —
# the user wants the diagnostic results before deciding on fine-tune config.
#
# Launch via:
#   setsid nohup bash scripts/v5_stage2_diag.sh > /tmp/v5_stage2/orchestrator.log 2>&1 &

set -euo pipefail

cd /home/cayman/Repositories/EliteFurretAI
source ../venv/bin/activate

LOGDIR=/tmp/v5_stage2
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== STAGE 0: sanity check ==="
log "  pwd=$(pwd)"
log "  v5 chunks: train=$(ls data/battles/regc_final_v5/train/ | grep -c '\.pt\.zst$'), val=$(ls data/battles/regc_final_v5/val/ | grep -c '\.pt\.zst$'), test=$(ls data/battles/regc_final_v5/test/ | grep -c '\.pt\.zst$')"
log "  embedder embedding_size: $(python -c "from elitefurretai.etl import Embedder; print(Embedder(gen=9, feature_set='raw').embedding_size)")"

# Marker for identifying the checkpoint produced by this run
touch "$LOGDIR/pre_train.marker"
sleep 2

# ---------- STAGE 2: train may24.yaml on v5 ----------
log "=== STAGE 2: train may24.yaml on regc_final_v5 ==="
python src/elitefurretai/supervised/train.py \
    data/battles/regc_final_v5/ \
    --config src/elitefurretai/supervised/configs/may24.yaml \
    --save-best \
    > "$LOGDIR/02_train.log" 2>&1

NEWEST_CKPT=$(find data/models/supervised -maxdepth 1 -name "*_best.pt" -newer "$LOGDIR/pre_train.marker" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$NEWEST_CKPT" ]]; then
    log "ERROR: no new *_best.pt found after stage 2. Aborting."
    exit 1
fi
log "  train produced: $NEWEST_CKPT"
RUN_NAME=$(basename "$NEWEST_CKPT" _best.pt)
log "  wandb run name: $RUN_NAME"

# ---------- STAGE 2.5: action diagnostic on the new checkpoint ----------
# Run on CPU (--safe) so it's robust regardless of whether anything else is
# already using the GPU. ~3 min wall time on a 26M-param model.
log "=== STAGE 2.5: action diagnostic on $RUN_NAME ==="
nice -n 19 python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
    "$NEWEST_CKPT" \
    data/battles/regc_final_v5/test/ \
    --safe --max-batches 200 \
    > "$LOGDIR/02d_diagnostic.log" 2>&1

log "=== STAGE 2 + DIAGNOSTIC COMPLETE ==="
log "  checkpoint: $NEWEST_CKPT"
log "  diagnostic log: $LOGDIR/02d_diagnostic.log"
log "  diagnostic JSON saved alongside checkpoint as ${NEWEST_CKPT%.pt}_action_diagnostics.json"
log "  next: user reviews diagnostic, decides on fine-tune config"
