#!/usr/bin/env bash
# BC Ablation Study — runs B0, A1, A2, A3, A4 sequentially, then runs diagnostics on each.
# Resumable: if a model_path.txt file exists for a run, training is skipped.
#            if a *_action_diagnostics.json exists for that model, diagnostics are skipped.
#
# Run inside tmux so it survives SSH disconnects:
#   tmux new -s ablations
#   bash scripts/run_ablations.sh 2>&1 | tee -a logs/ablation_study.log
#   tmux detach  (Ctrl-b d)

set -uo pipefail
# NOTE: deliberately NOT using `set -e` — we want to keep going through all runs even if
# one diagnostics call hiccups, and we report failures explicitly.

REPO="/home/cayman/Repositories/EliteFurretAI"
VENV="/home/cayman/Repositories/venv"
DATA="$REPO/data/battles/regc_final_v4"
MODELS="$REPO/data/models/supervised"
CONFIGS="$REPO/src/elitefurretai/supervised/configs"
VAL_DATA="$DATA/val"
LOG_DIR="$REPO/logs"

mkdir -p "$LOG_DIR"

source "$VENV/bin/activate"
cd "$REPO"

echo "================================================================"
echo " BC Ablation Study — $(date)"
echo "================================================================"

run_one() {
    local config_name="$1"
    local label="$2"

    local path_file="$LOG_DIR/${label}_model_path.txt"
    local model_path=""

    # ── Training (skip if already done) ────────────────────────────────
    if [ -f "$path_file" ] && [ -s "$path_file" ] && [ -f "$(cat $path_file)" ]; then
        model_path=$(cat "$path_file")
        echo ""
        echo "[$label] SKIP training — checkpoint already exists: $model_path"
    else
        echo ""
        echo "================================================================"
        echo " [$label] Training  config=$config_name  $(date)"
        echo "================================================================"
        # Snapshot which best models exist before training so we can identify the new one.
        local before_list="$LOG_DIR/${label}_before.txt"
        ls -1 "$MODELS"/*_best.pt 2>/dev/null > "$before_list" || touch "$before_list"

        python src/elitefurretai/supervised/train.py "$DATA" \
            --config "$CONFIGS/$config_name" \
            --save-best 2>&1 | tee "$LOG_DIR/${label}_train.log"
        local rc=${PIPESTATUS[0]}
        if [ "$rc" != "0" ]; then
            echo "[$label] ERROR: training exited with code $rc — moving on"
            rm -f "$before_list"
            return 1
        fi

        # New model = newest *_best.pt that wasn't in before_list
        model_path=$(ls -t "$MODELS"/*_best.pt 2>/dev/null | grep -vFxf "$before_list" | head -1 || true)
        if [ -z "$model_path" ]; then
            # Fallback: just take the most recently modified
            model_path=$(ls -t "$MODELS"/*_best.pt 2>/dev/null | head -1 || true)
        fi
        rm -f "$before_list"

        if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
            echo "[$label] ERROR: could not locate trained checkpoint — skipping diagnostics"
            return 1
        fi
        echo "$model_path" > "$path_file"
        echo "[$label] Training complete  $(date)  model=$model_path"
    fi

    # ── Action diagnostics (skip if already produced) ──────────────────
    local action_json="${model_path%.pt}_action_diagnostics.json"
    if [ -f "$action_json" ]; then
        echo "[$label] SKIP action diagnostics — already exists: $action_json"
    else
        echo "[$label] Running ACTION diagnostics  $(date)"
        python src/elitefurretai/supervised/analyze/action_model_diagnostics.py \
            "$model_path" "$VAL_DATA" 2>&1 | tee "$LOG_DIR/${label}_action_diagnostics.log"
        local rc=${PIPESTATUS[0]}
        if [ "$rc" != "0" ]; then
            echo "[$label] WARNING: action diagnostics exited with code $rc"
        fi
    fi

    # ── Win diagnostics (skip if already produced) ─────────────────────
    local win_json="${model_path%.pt}_win_diagnostics.json"
    if [ -f "$win_json" ]; then
        echo "[$label] SKIP win diagnostics — already exists: $win_json"
    else
        echo "[$label] Running WIN diagnostics  $(date)"
        python src/elitefurretai/supervised/analyze/win_model_diagnostics.py \
            "$model_path" "$VAL_DATA" --num-workers 3 \
            2>&1 | tee "$LOG_DIR/${label}_win_diagnostics.log"
        local rc=${PIPESTATUS[0]}
        if [ "$rc" != "0" ]; then
            echo "[$label] WARNING: win diagnostics exited with code $rc"
        fi
    fi

    echo "[$label] DONE  $(date)"
}

# ── Sequential runs ────────────────────────────────────────────────────
run_one "ablation_b0_baseline.yaml"     "b0_baseline"       || true
run_one "ablation_a1_no_transition.yaml" "a1_no_transition" || true
run_one "ablation_a2_raw.yaml"           "a2_raw"           || true
run_one "ablation_a3_small5x.yaml"       "a3_small5x"       || true
run_one "ablation_a4_small10x.yaml"      "a4_small10x"      || true

# ── Final summary ──────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " All runs complete  $(date)"
echo "================================================================"
for label in b0_baseline a1_no_transition a2_raw a3_small5x a4_small10x; do
    pf="$LOG_DIR/${label}_model_path.txt"
    if [ -f "$pf" ]; then
        mp=$(cat "$pf")
        echo "  $label: $mp"
        [ -f "${mp%.pt}_action_diagnostics.json" ] && echo "    ✓ action diagnostics" || echo "    ✗ action diagnostics MISSING"
        [ -f "${mp%.pt}_win_diagnostics.json" ]    && echo "    ✓ win diagnostics"    || echo "    ✗ win diagnostics MISSING"
    else
        echo "  $label: (not run)"
    fi
done
echo "Done."
