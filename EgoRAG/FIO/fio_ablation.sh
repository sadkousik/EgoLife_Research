#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fio_ablation.sh — run the stage-ablation sweep inside a tmux session
#
#   ./fio_ablation.sh start     launch the sweep in a detached tmux session
#   ./fio_ablation.sh status    which ablations are done / running / pending
#   ./fio_ablation.sh attach    watch the launcher  (detach: Ctrl-b then d)
#   ./fio_ablation.sh stop      kill the sweep and the session
#   ./fio_ablation.sh ls        list tmux sessions
#
# PHASE 2 ONLY → SAFE TO PARALLELISE:
#   These runs use the cached FIOs (SKIP_PHASE1=True) and make no LLM calls,
#   so the 7B model is never loaded onto the GPU. The ablations are also fully
#   independent — each reads the same read-only FIO cache and writes to its own
#   BUILD_7day_ablation_<name>/ dir — so they can run concurrently.
#
#   MAX_PARALLEL controls how many run at once:
#       0  = all at once   (default; one wave)
#       1  = sequential    (one after another)
#       N  = throttle to N concurrent
#   Dial it down if RAM / CPU contention bites (each run loads a small
#   SentenceTransformer + processes the FIO stream).
#
# OWNERSHIP:
#   python runs INSIDE the tmux pane (via a `_run` re-invocation of this
#   script), so the session owns the process tree and `stop` actually kills
#   every running build — no orphans.
#
# Per-ablation output goes to build_<name>_*.log (not interleaved on the pane).
# Watch one live with:  tail -f build_no_s3_*.log
# ─────────────────────────────────────────────────────────────────────────────

# ── CONFIG — edit these to match your machine ───────────────────────────────
SESSION="fio_abl"                                            # distinct from your "fio" session
ENV_NAME="egogpt"
PROJECT_DIR="/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG" # <-- verify this path
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"           # <-- verify this path

# ── WHICH ABLATIONS TO RUN ──────────────────────────────────────────────────
# Valid names are the keys in fio_v3.py's ABLATIONS dict:
#   full   no_ctx   no_s2   no_s3   no_s1
# Your full model (33%) is the locked anchor, so it is NOT in the default list.
# Add "full" only if you want to re-measure the anchor on the current config.
ABLATIONS=(no_s2 no_ctx) #(no_ctx no_s2 no_s3 no_s1 s3_only)

# 0 = all at once | 1 = sequential | N = at most N concurrent
MAX_PARALLEL=0

# Threads each build may use (caps PyTorch/BLAS CPU parallelism).
# RULE: (concurrent jobs) × THREADS_PER_JOB ≤ PHYSICAL cores, or the processes
# oversubscribe the cores and thrash. This box = 2 sockets × 12C = 24 physical
# cores, and there are 5 ablations, so 5 × 4 = 20 ≤ 24. Confirm core count with
# `lscpu` (Core(s) per socket × Socket(s)); if it's 12 physical, use 2.
THREADS_PER_JOB=12 #4
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_PATH="$(readlink -f "$0")"

start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[start] tmux session '$SESSION' already exists."
        echo "        run './fio_ablation.sh status', or 'stop' first."
        exit 1
    fi

    if [ ! -f "$CONDA_SH" ]; then
        echo "[start] WARNING: CONDA_SH not found at: $CONDA_SH"
        echo "        'conda activate' will fail. Find it with: grep -m1 conda.sh ~/.bashrc"
    fi

    for a in "${ABLATIONS[@]}"; do
        if [ "$a" = "no_s1" ]; then
            echo "[start] NOTE: 'no_s1' is in the run list — make sure target_keep_rate"
            echo "        in fio_v3.py (ABLATIONS['no_s1']) is the full model's real"
            echo "        keep-rate, not the 0.40 placeholder."
        fi
    done

    MASTER_LOG="$PROJECT_DIR/ablation_sweep_$(date +%F_%H%M).log"
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" \
        "bash '$SCRIPT_PATH' _run 2>&1 | tee '$MASTER_LOG'" C-m

    local mode="all at once"
    [ "$MAX_PARALLEL" = "1" ] && mode="sequential"
    [ "$MAX_PARALLEL" -gt 1 ] 2>/dev/null && mode="up to $MAX_PARALLEL concurrent"

    echo "[start] launched in tmux session '$SESSION'  (mode: $mode)"
    echo "        ablations : ${ABLATIONS[*]}"
    echo "        master log: $MASTER_LOG"
    echo "        check: ./fio_ablation.sh status     watch: ./fio_ablation.sh attach"
}

# Internal worker — executed INSIDE the tmux pane by start(). Not for direct use.
_run() {
    echo "######## ABLATION SWEEP START $(date) ########"
    echo "ablations: ${ABLATIONS[*]}   MAX_PARALLEL=$MAX_PARALLEL"

    source "$CONDA_SH" && conda activate "$ENV_NAME" && cd "$PROJECT_DIR" || {
        echo "[_run] FATAL: conda activate / cd failed. Check CONDA_SH, ENV_NAME, PROJECT_DIR."
        exit 1
    }

    # Resolve 0 → run them all in one wave.
    local cap="$MAX_PARALLEL"
    [ "$cap" -le 0 ] 2>/dev/null && cap=${#ABLATIONS[@]}

    local stamp; stamp="$(date +%F_%H%M)"
    for A in "${ABLATIONS[@]}"; do
        # Wait for a free slot before launching the next job.
        while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do sleep 5; done

        local LOG="$PROJECT_DIR/build_${A}_${stamp}.log"
        echo "[$(date +%H:%M:%S)] launching ABLATION=$A (threads=$THREADS_PER_JOB) -> $LOG"
        (
            OMP_NUM_THREADS="$THREADS_PER_JOB" \
            MKL_NUM_THREADS="$THREADS_PER_JOB" \
            OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
            ABLATION="$A" python -u fio_v3.py > "$LOG" 2>&1
            echo "___FINISHED___ ablation=$A exit=$? $(date)" >> "$LOG"
        ) &
    done

    wait   # block until every background build is done

    echo ""
    echo "──────── sweep summary ────────"
    for A in "${ABLATIONS[@]}"; do
        local log; log="$(ls -t "$PROJECT_DIR"/build_"${A}"_*.log 2>/dev/null | head -1)"
        local line; line="$(grep '___FINISHED___' "$log" 2>/dev/null | tail -1)"
        printf "  %-8s %s\n" "$A" "${line:-<no FINISHED marker>}"
    done
    echo "___ALL_DONE___ $(date)"
    echo "######## ABLATION SWEEP COMPLETE ########"
}

status() {
    echo "──────── FIO ablation sweep status ────────"
    if pgrep -af "fio_v3.py" >/dev/null; then
        echo "  ▶ build(s) RUNNING:"
        pgrep -af "fio_v3.py" | sed 's/^/      /'
    elif tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "  ■ session '$SESSION' alive but NO python process (finished / between waves)"
    else
        echo "  ✗ NOT running (no python, no tmux session)"
    fi

    echo "  per-ablation progress:"
    for A in "${ABLATIONS[@]}"; do
        log=$(ls -t "$PROJECT_DIR"/build_"${A}"_*.log 2>/dev/null | head -1)
        if [ -z "$log" ]; then
            printf "      %-8s · pending (no log yet)\n" "$A"
        elif grep -q "___FINISHED___" "$log" 2>/dev/null; then
            rc=$(grep "___FINISHED___" "$log" | tail -1 | sed -n 's/.*exit=\([0-9]*\).*/\1/p')
            if [ "$rc" = "0" ]; then
                printf "      %-8s ✔ done (exit 0)\n" "$A"
            else
                printf "      %-8s ✗ FAILED (exit %s)\n" "$A" "$rc"
            fi
        else
            printf "      %-8s ▶ in progress\n" "$A"
        fi
    done

    master=$(ls -t "$PROJECT_DIR"/ablation_sweep_*.log 2>/dev/null | head -1)
    if [ -n "$master" ]; then
        echo "  master log: $master"
        grep -q "___ALL_DONE___" "$master" 2>/dev/null && \
            echo "  ✔ sweep COMPLETE: $(grep '___ALL_DONE___' "$master" | tail -1)"
        echo "  last 4 lines:"; tail -n 4 "$master" | sed 's/^/      /'
    fi
    echo "───────────────────────────────────────────"
}

stop() {
    pkill -f "fio_v3.py" 2>/dev/null \
        && echo "[stop] terminated fio_v3.py build(s)" \
        || echo "[stop] no fio_v3.py process was running"
    tmux kill-session -t "$SESSION" 2>/dev/null \
        && echo "[stop] killed tmux session '$SESSION'" \
        || echo "[stop] no tmux session '$SESSION'"
}

case "${1:-}" in
    start)  start  ;;
    status) status ;;
    attach) tmux attach -t "$SESSION" ;;
    stop)   stop   ;;
    ls)     tmux ls 2>/dev/null || echo "no tmux server running" ;;
    _run)   _run   ;;   # internal — invoked inside the tmux pane by start()
    *)      echo "usage: $0 {start|status|attach|stop|ls}" ;;
esac