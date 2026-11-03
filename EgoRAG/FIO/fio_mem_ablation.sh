#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fio_mem_ablation.sh — run the STAGE ablation (full / no_s1 / no_s2 / no_s3 /
#   s1s2_no_s3 / s3_only) on ONE cumulative cache window, in tmux.
#
#   ./fio_mem_ablation.sh start  [day-window]
#   ./fio_mem_ablation.sh status [day-window]
#   ./fio_mem_ablation.sh attach [day-window]
#   ./fio_mem_ablation.sh stop   [day-window]
#   ./fio_mem_ablation.sh ls
#
# day-window is one of:  day1 day1-2 day1-3 day1-4 day1-5 day1-6
#   - pass it as the 2nd arg, OR edit DAY_WINDOW below.
#   - status/stop/attach must use the SAME window as start (it sets the session).
#
# Each stage runs fio_v3.py with ABLATION=<stage> and writes to:
#     <MEM_ROOT>/<stage>/            e.g. fio_cache_day1-3_mem/no_s2/
#         chroma_fio_opt/  build_report.json  ...
#
# REQUIRES the 4 env-override edits in fio_v3.py (FIO_CACHE_DIR, TARGET_DAY,
# OUTPUT_DIR, RUN_TAG). Without them every stage builds the same hardcoded
# path. See notes at the bottom of this file.
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG ───────────────────────────────────────────────────────────────────
ENV_NAME="egogpt"
PROJECT_DIR="/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG"   # verify
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"             # verify

# Which cumulative window to ablate. Edit here OR pass as 2nd CLI arg.
#   day1 | day1-2 | day1-3 | day1-4 | day1-5 | day1-6
DAY_WINDOW="${DAY_WINDOW:-day1-3}"

# Stages to run = ABLATION keys in fio_v3.py's ABLATIONS dict.
#   NOTE: no_s1 and s3_only need the opti_fio_test.py uniform-retention patch,
#         or fio_v3.py will sys.exit on them.
#   NOTE: no_s3 and s1s2_no_s3 are the SAME build — drop one to save a slot.
STAGES=(full no_s1 no_s2 no_s3 s3_only)

# Where the split windows live (output of split_fio_cache_by_day.py),
# relative to PROJECT_DIR.
WINDOWS_ROOT="memory_build_v2_7days/fio_cache_windows"

# Per-window output parent. WINDOW is replaced with the chosen window.
#   -> memory_build_v2_7days/fio_cache_day1-3_mem/<stage>/
MEM_ROOT_TMPL="memory_build_v2_7days/fio_cache_WINDOW_mem"

MAX_PARALLEL=0 #3       # 0 = all stages at once | 1 = sequential | N = at most N
THREADS_PER_JOB=4    # (concurrent jobs) x THREADS_PER_JOB <= physical cores (24)
CLEAN_FIRST=0        # 1 = wipe each stage's output dir + its t_sne exports first
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_PATH="$(readlink -f "$0")"

# 2nd CLI arg overrides the window for any command.
[ -n "$2" ] && DAY_WINDOW="$2"

# Session name — change freely. Defaults to include the window so different
# windows can run side by side. Override:  FIO_SESSION=myname ./fio_mem_ablation.sh ...
SESSION="${FIO_SESSION:-fio_mem_${DAY_WINDOW}}"

# Derive TARGET_DAY, cache dir, and output root from the window.
case "$DAY_WINDOW" in
    day1)   TARGET_DAY=1 ;;
    day1-*) TARGET_DAY="${DAY_WINDOW#day1-}" ;;
    *) echo "[config] bad DAY_WINDOW='$DAY_WINDOW' (use day1 | day1-2 .. day1-6)"; exit 1 ;;
esac
FIO_CACHE_DIR="${WINDOWS_ROOT}/fio_cache_${DAY_WINDOW}"
MEM_ROOT="${MEM_ROOT_TMPL/WINDOW/$DAY_WINDOW}"

# ─────────────────────────────────────────────────────────────────────────────

start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[start] session '$SESSION' exists. './fio_mem_ablation.sh status $DAY_WINDOW' or 'stop' first."
        echo "        (or change DAY_WINDOW / set FIO_SESSION to run another window alongside.)"
        exit 1
    fi
    [ -f "$CONDA_SH" ] || echo "[start] WARNING: CONDA_SH not found: $CONDA_SH"
    [ -d "$PROJECT_DIR/$FIO_CACHE_DIR" ] || \
        echo "[start] WARNING: cache dir not found: $PROJECT_DIR/$FIO_CACHE_DIR"

    MASTER_LOG="$PROJECT_DIR/mem_ablation_${DAY_WINDOW}_$(date +%F_%H%M).log"
    tmux new-session -d -s "$SESSION"
    # Pass the chosen window + session into the detached _run (fresh bash).
    tmux send-keys -t "$SESSION" \
        "DAY_WINDOW='$DAY_WINDOW' FIO_SESSION='$SESSION' bash '$SCRIPT_PATH' _run 2>&1 | tee '$MASTER_LOG'" C-m

    local mode="all at once"
    [ "$MAX_PARALLEL" = "1" ] && mode="sequential"
    [ "$MAX_PARALLEL" -gt 1 ] 2>/dev/null && mode="up to $MAX_PARALLEL concurrent"
    echo "[start] window=$DAY_WINDOW  TARGET_DAY=$TARGET_DAY  session=$SESSION"
    echo "        stages : ${STAGES[*]}   (mode: $mode)"
    echo "        cache  : $FIO_CACHE_DIR"
    echo "        out    : $MEM_ROOT/<stage>/"
    echo "        master log: $MASTER_LOG"
    echo "        check: ./fio_mem_ablation.sh status $DAY_WINDOW    watch: ./fio_mem_ablation.sh attach $DAY_WINDOW"
}

_run() {
    echo "######## MEM ABLATION  window=$DAY_WINDOW (TARGET_DAY=$TARGET_DAY)  START $(date) ########"
    echo "stages: ${STAGES[*]}   MAX_PARALLEL=$MAX_PARALLEL  THREADS_PER_JOB=$THREADS_PER_JOB"
    echo "cache : $FIO_CACHE_DIR"
    echo "out   : $MEM_ROOT/<stage>/"
    source "$CONDA_SH" && conda activate "$ENV_NAME" && cd "$PROJECT_DIR" || {
        echo "[_run] FATAL: conda activate / cd failed."; exit 1; }

    if [ ! -d "$FIO_CACHE_DIR" ]; then
        echo "[_run] FATAL: cache window not found: $PROJECT_DIR/$FIO_CACHE_DIR"
        echo "        run split_fio_cache_by_day.py first."; exit 1
    fi

    local cap="$MAX_PARALLEL"
    [ "$cap" -le 0 ] 2>/dev/null && cap=${#STAGES[@]}
    local stamp; stamp="$(date +%F_%H%M)"

    for STAGE in "${STAGES[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do sleep 5; done
        local OUT="${MEM_ROOT}/${STAGE}"
        if [ "$CLEAN_FIRST" = "1" ]; then
            rm -rf "$PROJECT_DIR/$OUT"
            rm -f  "$PROJECT_DIR/t_sne/merge_stats_${STAGE}_${DAY_WINDOW}.json" \
                   "$PROJECT_DIR/t_sne/prototype_tsne_data_${STAGE}_${DAY_WINDOW}.json"
        fi
        local LOG="$PROJECT_DIR/build_mem_${DAY_WINDOW}_${STAGE}_${stamp}.log"
        echo "[$(date +%H:%M:%S)] launching stage=$STAGE -> $OUT  (log $LOG)"
        (
            OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
            OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
            ABLATION="$STAGE" \
            FIO_CACHE_DIR="$FIO_CACHE_DIR" \
            TARGET_DAY="$TARGET_DAY" \
            OUTPUT_DIR="$OUT" \
            RUN_TAG="${STAGE}_${DAY_WINDOW}" \
            python -u fio_v3.py > "$LOG" 2>&1
            echo "___FINISHED___ stage=$STAGE exit=$? $(date)" >> "$LOG"
        ) &
    done
    wait

    echo ""
    echo "──────── mem-ablation summary (window=$DAY_WINDOW) ────────"
    for STAGE in "${STAGES[@]}"; do
        local log; log="$(ls -t "$PROJECT_DIR"/build_mem_"${DAY_WINDOW}"_"${STAGE}"_*.log 2>/dev/null | head -1)"
        local line; line="$(grep '___FINISHED___' "$log" 2>/dev/null | tail -1)"
        printf "  %-12s %s\n" "$STAGE" "${line:-<no FINISHED marker>}"
    done
    echo "___ALL_DONE___ $(date)"
}

status() {
    echo "──────── mem-ablation status (window=$DAY_WINDOW, session=$SESSION) ────────"
    if pgrep -af "fio_v3.py" >/dev/null; then
        echo "  ▶ build(s) RUNNING (note: matches ALL fio_v3.py across windows):"
        pgrep -af "fio_v3.py" | sed 's/^/      /'
    elif tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  ■ session alive, no python (between waves / finished)"
    else
        echo "  ✗ NOT running"
    fi
    for STAGE in "${STAGES[@]}"; do
        log=$(ls -t "$PROJECT_DIR"/build_mem_"${DAY_WINDOW}"_"${STAGE}"_*.log 2>/dev/null | head -1)
        if [ -z "$log" ]; then printf "      %-12s · pending\n" "$STAGE"
        elif grep -q "___FINISHED___" "$log" 2>/dev/null; then
            rc=$(grep "___FINISHED___" "$log" | tail -1 | sed -n 's/.*exit=\([0-9]*\).*/\1/p')
            [ "$rc" = "0" ] && printf "      %-12s ✔ done\n" "$STAGE" \
                            || printf "      %-12s ✗ FAILED (exit %s)\n" "$STAGE" "$rc"
        else printf "      %-12s ▶ in progress\n" "$STAGE"; fi
    done
    echo "──────────────────────────────────────────────"
}

stop() {
    # NOTE: pkill matches ALL fio_v3.py runs (env vars aren't in the cmdline),
    # so this also stops other windows' builds if any are running.
    pkill -f "fio_v3.py" 2>/dev/null && echo "[stop] killed builds" || echo "[stop] none running"
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "[stop] killed session $SESSION" \
        || echo "[stop] no session $SESSION"
}

case "${1:-}" in
    start)  start  ;;
    status) status ;;
    attach) tmux attach -t "$SESSION" ;;
    stop)   stop   ;;
    ls)     tmux ls 2>/dev/null || echo "no tmux server" ;;
    _run)   _run   ;;
    *)      echo "usage: $0 {start|status|attach|stop|ls} [day-window]"
            echo "  day-window: day1 day1-2 day1-3 day1-4 day1-5 day1-6  (default: $DAY_WINDOW)"
            echo "  stages    : ${STAGES[*]}"
            echo "  session   : $SESSION  (override with FIO_SESSION=...)"
            ;;
esac