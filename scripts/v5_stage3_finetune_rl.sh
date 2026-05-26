#!/bin/bash
# Stage 3: fine-tune rose-sun-108, pick best BC checkpoint by diagnostic score,
# then kick off an RL run with may22.yaml using the chosen BC as initialize_path.
#
# Notes:
#   * `set -e` is intentionally NOT enabled — fine_tune.py and train.py have
#     occasionally crashed in the *post-training* validation pass while still
#     writing a valid *_best.pt during training. We check checkpoint existence
#     explicitly to decide whether to continue.
#   * The RL stage runs in the foreground of this script (the script stays
#     alive for the full RL duration). setsid+nohup on the orchestrator
#     protects the whole chain.
#
# Launch via:
#   setsid nohup bash scripts/v5_stage3_finetune_rl.sh > /tmp/v5_stage3/orchestrator.log 2>&1 &

set -uo pipefail

cd /home/cayman/Repositories/EliteFurretAI
source ../venv/bin/activate

LOGDIR=/tmp/v5_stage3
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

BC_BASE=data/models/supervised/rose-sun-108_best.pt

# ---------- STAGE 3a: fine-tune rose-sun-108 ----------
log "=== STAGE 3a: fine-tune $BC_BASE (may22_finetune.yaml override) ==="
touch "$LOGDIR/pre_finetune.marker"
sleep 2

python src/elitefurretai/supervised/fine_tune.py \
    data/battles/regc_final_v5/ \
    "$BC_BASE" \
    rose-sun-108-finetune \
    --config-override src/elitefurretai/supervised/configs/may22_finetune.yaml \
    --save-best \
    > "$LOGDIR/fine_tune.log" 2>&1
FT_EXIT=$?
log "  fine_tune exit code: $FT_EXIT (non-zero usually means post-training eval crash; checkpoint may still exist)"

NEW_CKPT=$(find data/models/supervised -maxdepth 1 -name "*_best.pt" -newer "$LOGDIR/pre_finetune.marker" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$NEW_CKPT" ]]; then
    log "ERROR: no new *_best.pt produced by fine-tune. Aborting before diagnostic/RL."
    exit 1
fi
log "  fine-tune produced: $NEW_CKPT"

# ---------- STAGE 3b: diagnostics on BOTH checkpoints (fixed-attribution) ----------
log "=== STAGE 3b: diagnostic on $BC_BASE (fixed attribution; overwrites earlier report) ==="
nice -n 19 python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
    "$BC_BASE" \
    data/battles/regc_final_v5/test/ \
    --safe --max-batches 200 \
    > "$LOGDIR/diag_rose_sun.log" 2>&1
log "  diagnostic 1 done"

log "=== STAGE 3b: diagnostic on $NEW_CKPT ==="
nice -n 19 python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
    "$NEW_CKPT" \
    data/battles/regc_final_v5/test/ \
    --safe --max-batches 200 \
    > "$LOGDIR/diag_finetune.log" 2>&1
log "  diagnostic 2 done"

# ---------- STAGE 3c: pick best by BC quality score ----------
log "=== STAGE 3c: pick best BC checkpoint ==="
BEST_CKPT=$(python <<PY
import json, sys
candidates = [
    ("$BC_BASE", "${BC_BASE%.pt}_action_diagnostics.json"),
    ("$NEW_CKPT", "${NEW_CKPT%.pt}_action_diagnostics.json"),
]
def bc_score(rep):
    # Mirrors train.py's action_score (drops win_corr since the diagnostic
    # doesn't compute it; rolls that 0.1 into switch_top1's weight).
    topk = rep.get("topk_accuracy_by_type", {})
    return (
        topk.get("MOVE", {}).get("top3", 0) * 0.5
        + topk.get("BOTH", {}).get("top3", 0) * 0.3
        + topk.get("SWITCH", {}).get("top1", 0) * 0.2
    )
best_path, best_score = None, -1
for path, jpath in candidates:
    with open(jpath) as f:
        rep = json.load(f)
    s = bc_score(rep)
    print(f"  {path}: bc_score={s:.4f}", file=sys.stderr)
    if s > best_score:
        best_score, best_path = s, path
print(best_path)
PY
)
log "  best checkpoint: $BEST_CKPT"
if [[ -z "$BEST_CKPT" ]]; then
    log "ERROR: could not pick a best checkpoint. Aborting before RL."
    exit 1
fi

# ---------- STAGE 3d: write an RL config that points at the chosen BC ----------
log "=== STAGE 3d: write RL config with initialize_path=$BEST_CKPT ==="
RL_CONFIG=src/elitefurretai/rl/configs/may22_v5_init.yaml
sed "s#initialize_path: data/models/supervised/cool-bee-85-finetune_best.pt#initialize_path: $BEST_CKPT#" \
    src/elitefurretai/rl/configs/may22.yaml > "$RL_CONFIG"
if ! grep -q "initialize_path: $BEST_CKPT" "$RL_CONFIG"; then
    log "ERROR: initialize_path substitution failed in $RL_CONFIG"
    exit 1
fi
log "  wrote $RL_CONFIG (initialize_path verified)"

# ---------- STAGE 3e: launch RL ----------
log "=== STAGE 3e: launch RL training (foreground; this script blocks for the full RL duration) ==="
python src/elitefurretai/rl/train.py --config "$RL_CONFIG" \
    > "$LOGDIR/rl_train.log" 2>&1

log "=== STAGE 3 ORCHESTRATOR COMPLETE (RL training process has exited) ==="
