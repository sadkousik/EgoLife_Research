#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fio_sweep.sh — run a single-hyperparameter sweep on the FULL model in tmux
#
#   ./fio_sweep.sh start | status | attach | stop | ls
#
# Sweeps ONE knob across a value list (ABLATION=full, everything else fixed).
# Each value -> its own BUILD_7day_ablation_full__<param>=<val>/ dir + log, so
# nothing clobbers. Same parallel + thread-cap behaviour as fio_ablation.sh.
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG ───────────────────────────────────────────────────────────────────
SESSION="fio_sweep"
ENV_NAME="egogpt"
PROJECT_DIR="/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG"   # verify
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"             # verify

# ── PICK ONE SWEEP (uncomment exactly one) ───────────────────────────────────
#SWEEP_PARAM="delta_g_threshold"; SWEEP_VALUES=(0.60 0.65 0.70 0.75 0.80 0.85)
# SWEEP_PARAM="max_prototypes";  SWEEP_VALUES=(400 500) #(100 200 300 400 500)
# SWEEP_PARAM="tau_threshold";   SWEEP_VALUES=(0.1 0.3 0.5)
# SWEEP_PARAM="history_window";   SWEEP_VALUES=(3 5 10)
#SWEEP_PARAM="context_history_weight"; SWEEP_VALUES=(0.9) # SWEEP_VALUES=(0.3 0.5 0.7 0.9)
#SWEEP_PARAM="lambda_decay"; SWEEP_VALUES=(2.6742e-6 1.6045e-6 8.9139e-7) #(3d,5d,9dya)
SWEEP_PARAM="delta_h_threshold"; SWEEP_VALUES=(0.2)

MAX_PARALLEL=0          # 0 = all at once | 1 = sequential | N = at most N
THREADS_PER_JOB=4       # (concurrent jobs) x THREADS_PER_JOB <= physical cores (24)

# 1 = remove each value's BUILD dir + t_sne exports before (re)building, so a
#     re-run starts clean (no Chroma corruption, no duplicate merge-stats rows).
# 0 = leave existing outputs alone (first run, or you want to keep them).
CLEAN_FIRST=0
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_PATH="$(readlink -f "$0")"

start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[start] session '$SESSION' exists. './fio_sweep.sh status' or 'stop' first."
        exit 1
    fi
    [ -f "$CONDA_SH" ] || echo "[start] WARNING: CONDA_SH not found: $CONDA_SH"

    MASTER_LOG="$PROJECT_DIR/sweep_${SWEEP_PARAM}_$(date +%F_%H%M).log"
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "bash '$SCRIPT_PATH' _run 2>&1 | tee '$MASTER_LOG'" C-m

    local mode="all at once"
    [ "$MAX_PARALLEL" = "1" ] && mode="sequential"
    [ "$MAX_PARALLEL" -gt 1 ] 2>/dev/null && mode="up to $MAX_PARALLEL concurrent"
    echo "[start] sweeping $SWEEP_PARAM over: ${SWEEP_VALUES[*]}  (mode: $mode)"
    echo "        master log: $MASTER_LOG"
    echo "        check: ./fio_sweep.sh status     watch: ./fio_sweep.sh attach"
}

_run() {
    echo "######## SWEEP $SWEEP_PARAM START $(date) ########"
    echo "values: ${SWEEP_VALUES[*]}   MAX_PARALLEL=$MAX_PARALLEL  THREADS_PER_JOB=$THREADS_PER_JOB"
    source "$CONDA_SH" && conda activate "$ENV_NAME" && cd "$PROJECT_DIR" || {
        echo "[_run] FATAL: conda activate / cd failed."; exit 1; }

    local cap="$MAX_PARALLEL"
    [ "$cap" -le 0 ] 2>/dev/null && cap=${#SWEEP_VALUES[@]}
    local stamp; stamp="$(date +%F_%H%M)"

    for V in "${SWEEP_VALUES[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do sleep 5; done
        local TAG="full__${SWEEP_PARAM}=${V}"
        if [ "$CLEAN_FIRST" = "1" ]; then
            rm -rf "$PROJECT_DIR/BUILD_7day_ablation_${TAG}"
            rm -f  "$PROJECT_DIR/t_sne/merge_stats_${TAG}.json" \
                   "$PROJECT_DIR/t_sne/prototype_tsne_data_${TAG}.json"
        fi
        local LOG="$PROJECT_DIR/build_sweep_${SWEEP_PARAM}_${V}_${stamp}.log"
        echo "[$(date +%H:%M:%S)] launching $SWEEP_PARAM=$V (threads=$THREADS_PER_JOB) -> $LOG"
        (
            OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
            OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
            ABLATION=full SWEEP_PARAM="$SWEEP_PARAM" SWEEP_VAL="$V" \
            python -u fio_v3.py > "$LOG" 2>&1
            echo "___FINISHED___ ${SWEEP_PARAM}=$V exit=$? $(date)" >> "$LOG"
        ) &
    done
    wait

    echo ""
    echo "──────── sweep summary ($SWEEP_PARAM) ────────"
    for V in "${SWEEP_VALUES[@]}"; do
        local log; log="$(ls -t "$PROJECT_DIR"/build_sweep_"${SWEEP_PARAM}"_"${V}"_*.log 2>/dev/null | head -1)"
        local line; line="$(grep '___FINISHED___' "$log" 2>/dev/null | tail -1)"
        printf "  %-10s %s\n" "$V" "${line:-<no FINISHED marker>}"
    done
    echo "___ALL_DONE___ $(date)"
}

status() {
    echo "──────── sweep status ($SWEEP_PARAM) ────────"
    if pgrep -af "fio_v3.py" >/dev/null; then
        echo "  ▶ build(s) RUNNING:"; pgrep -af "fio_v3.py" | sed 's/^/      /'
    elif tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  ■ session alive, no python (between waves / finished)"
    else
        echo "  ✗ NOT running"
    fi
    for V in "${SWEEP_VALUES[@]}"; do
        log=$(ls -t "$PROJECT_DIR"/build_sweep_"${SWEEP_PARAM}"_"${V}"_*.log 2>/dev/null | head -1)
        if [ -z "$log" ]; then printf "      %-10s · pending\n" "$V"
        elif grep -q "___FINISHED___" "$log" 2>/dev/null; then
            rc=$(grep "___FINISHED___" "$log" | tail -1 | sed -n 's/.*exit=\([0-9]*\).*/\1/p')
            [ "$rc" = "0" ] && printf "      %-10s ✔ done\n" "$V" || printf "      %-10s ✗ FAILED (exit %s)\n" "$V" "$rc"
        else printf "      %-10s ▶ in progress\n" "$V"; fi
    done
    echo "─────────────────────────────────────────────"
}

stop() {
    pkill -f "fio_v3.py" 2>/dev/null && echo "[stop] killed builds" || echo "[stop] none running"
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "[stop] killed session" || echo "[stop] no session"
}

case "${1:-}" in
    start)  start  ;;
    status) status ;;
    attach) tmux attach -t "$SESSION" ;;
    stop)   stop   ;;
    ls)     tmux ls 2>/dev/null || echo "no tmux server" ;;
    _run)   _run   ;;
    *)      echo "usage: $0 {start|status|attach|stop|ls}" ;;
esac