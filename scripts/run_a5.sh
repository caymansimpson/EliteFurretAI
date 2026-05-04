#!/usr/bin/env bash
# A5: A3 architecture + RAW featureset, 30 epochs
# Run inside tmux so it survives SSH disconnects:
#   tmux new -s a5
#   bash scripts/run_a5.sh 2>&1 | tee logs/a5_run.log
#   tmux detach  (Ctrl-b d)

set -uo pipefail

REPO="/home/cayman/Repositories/EliteFurretAI"
VENV="/home/cayman/Repositories/venv"
DATA="$REPO/data/battles/regc_final_v4"
MODELS="$REPO/data/models/supervised"
CONFIGS="$REPO/src/elitefurretai/supervised/configs"
VAL_DATA="$DATA/val"
LOG_DIR="$REPO/logs"
LABEL="a5_small5x_raw_30ep"

mkdir -p "$LOG_DIR"
source "$VENV/bin/activate"
cd "$REPO"

echo "================================================================"
echo " A5 (5x smaller + raw, 30 epochs)  $(date)"
echo "================================================================"

PATH_FILE="$LOG_DIR/${LABEL}_model_path.txt"

# Resume-aware training
if [ -f "$PATH_FILE" ] && [ -s "$PATH_FILE" ] && [ -f "$(cat $PATH_FILE)" ]; then
    MODEL_PATH=$(cat "$PATH_FILE")
    echo "[$LABEL] SKIP training — checkpoint exists: $MODEL_PATH"
else
    BEFORE_LIST="$LOG_DIR/${LABEL}_before.txt"
    ls -1 "$MODELS"/*_best.pt 2>/dev/null > "$BEFORE_LIST" || touch "$BEFORE_LIST"

    python src/elitefurretai/supervised/train.py "$DATA" \
        --config "$CONFIGS/ablation_a5_small5x_raw_30ep.yaml" \
        --save-best 2>&1 | tee "$LOG_DIR/${LABEL}_train.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" != "0" ]; then
        echo "[$LABEL] ERROR: training exited with code $rc"
        rm -f "$BEFORE_LIST"
        exit 1
    fi

    MODEL_PATH=$(ls -t "$MODELS"/*_best.pt 2>/dev/null | grep -vFxf "$BEFORE_LIST" | head -1 || true)
    [ -z "$MODEL_PATH" ] && MODEL_PATH=$(ls -t "$MODELS"/*_best.pt 2>/dev/null | head -1 || true)
    rm -f "$BEFORE_LIST"
    if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
        echo "[$LABEL] ERROR: could not locate trained checkpoint"
        exit 1
    fi
    echo "$MODEL_PATH" > "$PATH_FILE"
    echo "[$LABEL] Training complete  $(date)  model=$MODEL_PATH"
fi

# Action diagnostics
ACTION_JSON="${MODEL_PATH%.pt}_action_diagnostics.json"
if [ -f "$ACTION_JSON" ]; then
    echo "[$LABEL] SKIP action diagnostics — already exists"
else
    echo "[$LABEL] Running ACTION diagnostics  $(date)"
    python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
        "$MODEL_PATH" "$VAL_DATA" 2>&1 | tee "$LOG_DIR/${LABEL}_action_diagnostics.log"
fi

# Win diagnostics
WIN_JSON="${MODEL_PATH%.pt}_win_diagnostics.json"
if [ -f "$WIN_JSON" ]; then
    echo "[$LABEL] SKIP win diagnostics — already exists"
else
    echo "[$LABEL] Running WIN diagnostics  $(date)"
    python src/elitefurretai/supervised/analyze/win_model_diagnostics.py \
        "$MODEL_PATH" "$VAL_DATA" --num-workers 3 \
        2>&1 | tee "$LOG_DIR/${LABEL}_win_diagnostics.log"
fi

echo ""
echo "================================================================"
echo " A5 done  $(date)"
echo "================================================================"
echo "Model:           $MODEL_PATH"
echo "Action diag:     ${MODEL_PATH%.pt}_action_diagnostics.json"
echo "Win diag:        ${MODEL_PATH%.pt}_win_diagnostics.json"
