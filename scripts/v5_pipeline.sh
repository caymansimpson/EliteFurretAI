#!/bin/bash
# Active-slot positional feature pipeline (see planning/stage2/2026-05-24-12-00-active-slot-positional-features.md).
# Steps: preprocess raw battles -> regc_final_v5, train may24, fine-tune the result.
# Sequential chain via `set -e` — if any step fails, downstream steps are skipped.
# Launch via:
#   setsid nohup bash scripts/v5_pipeline.sh > /tmp/v5_pipeline/orchestrator.log 2>&1 &

set -euo pipefail

cd /home/cayman/Repositories/EliteFurretAI
source ../venv/bin/activate

LOGDIR=/tmp/v5_pipeline
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== STAGE 0: env snapshot ==="
log "  pwd=$(pwd)"
log "  python=$(which python)"
log "  embedder embedding_size: $(python -c "from elitefurretai.etl import Embedder; print(Embedder(gen=9, feature_set='raw').embedding_size)")"

# ---------- STAGE 0.5: re-filter existing JSON with smoke-parse gate ----------
# The existing JSON was built with static-log filters only; some replays still
# trip poke-env's parser at preprocess time (e.g. nickname-as-species on Trick).
# The smoke-parse check now lives in filter_battle_data; re-validate the 432K
# entries to produce a guaranteed-clean JSON.
JSON_ORIG=data/battles/supervised_battle_files_w_commander.json
JSON_CLEAN=data/battles/supervised_battle_files_w_commander_smoke.json
if [[ -e "$JSON_ORIG" && ! -e "${JSON_ORIG}.bak" ]]; then
    cp "$JSON_ORIG" "${JSON_ORIG}.bak"
    log "  backed up $JSON_ORIG -> ${JSON_ORIG}.bak"
fi

log "=== STAGE 0.5: smoke-parse re-validation -> $JSON_CLEAN ==="
python src/elitefurretai/etl/filter_battle_data.py \
    "$JSON_CLEAN" \
    --input-json "$JSON_ORIG" \
    --num-threads 8 \
    > "$LOGDIR/00_filter.log" 2>&1
log "  filter output:"
tail -3 "$LOGDIR/00_filter.log" | sed 's/^/    /'

# ---------- STAGE 1: preprocess ----------
log "=== STAGE 1: preprocess -> data/battles/regc_final_v5 (using $JSON_CLEAN) ==="
python src/elitefurretai/etl/process_training_data.py \
    "$JSON_CLEAN" \
    data/battles/regc_final_v5 \
    --mode trajectories \
    --chunk-size 512 \
    --num-workers 7 \
    > "$LOGDIR/01_preprocess.log" 2>&1

log "Preprocess done. v5 contents:"
ls data/battles/regc_final_v5/ | sed 's/^/    /' | tee -a "$LOGDIR/orchestrator.log"
log "  train files: $(ls data/battles/regc_final_v5/train/ 2>/dev/null | grep -c '\.pt\.zst$')"
log "  val files:   $(ls data/battles/regc_final_v5/val/ 2>/dev/null | grep -c '\.pt\.zst$')"
log "  test files:  $(ls data/battles/regc_final_v5/test/ 2>/dev/null | grep -c '\.pt\.zst$')"

# ---------- STAGE 2: train may24 on v5 ----------
log "=== STAGE 2: train may24.yaml on regc_final_v5 ==="
# Marker so we can identify the new checkpoint produced by this stage
touch "$LOGDIR/pre_train.marker"
sleep 2

python src/elitefurretai/supervised/train.py \
    data/battles/regc_final_v5/ \
    --config src/elitefurretai/supervised/configs/may24.yaml \
    --save-best \
    > "$LOGDIR/02_train.log" 2>&1

# Find the *_best.pt produced after the marker
NEWEST_CKPT=$(find data/models/supervised -maxdepth 1 -name "*_best.pt" -newer "$LOGDIR/pre_train.marker" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$NEWEST_CKPT" ]]; then
    log "ERROR: no new *_best.pt checkpoint found after stage 2. Aborting."
    exit 1
fi
log "  train produced: $NEWEST_CKPT"
RUN_NAME=$(basename "$NEWEST_CKPT" _best.pt)
log "  wandb run name: $RUN_NAME"

# ---------- STAGE 3: fine-tune ----------
log "=== STAGE 3: fine-tune $RUN_NAME (may22_finetune override) ==="
python src/elitefurretai/supervised/fine_tune.py \
    data/battles/regc_final_v5/ \
    "$NEWEST_CKPT" \
    "${RUN_NAME}-finetune" \
    --config-override src/elitefurretai/supervised/configs/may22_finetune.yaml \
    --save-best \
    > "$LOGDIR/03_finetune.log" 2>&1

log "=== PIPELINE COMPLETE ==="
log "  final checkpoint candidates:"
find data/models/supervised -maxdepth 1 -name "*_best.pt" -newer "$LOGDIR/pre_train.marker" -printf "    %T@ %p\n" | sort -n
