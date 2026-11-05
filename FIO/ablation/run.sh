#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ablation/run.sh — run FIO memory-build ablations in tmux (Phase 2 only, CPU).
#
#   ablation/run.sh stages        plan|start|status|attach|stop|qa|report
#   ablation/run.sh sweep         plan|start|status|attach|stop|qa|report
#   ablation/run.sh window <W>    plan|start|status|attach|stop|qa|report    W = day1 | day1-2 .. day1-7
#   ablation/run.sh ls
#
#   stages  one build per row in STAGES         (rows: python ablation/configs.py)
#   sweep   one build per value in SWEEP_VALUES  for SWEEP_PARAM, on row SWEEP_BASE
#   window  every row in STAGES on one cumulative day window of the FIO cache
#
# `plan` prints the runs without starting anything. Each run writes to
# OUT_ROOT/<tag>/ (window: OUT_ROOT/<W>/<row>/) and logs to log/ablation/.
# `qa` then evaluates every finished build with scripts/qa.py, one after the
# other in the foreground (needs the GPU — run it inside a Slurm job), and
# `report` prints GA / Acc. for each build from the saved results.
# Paper table → command recipes: ablation/README.md.
# Every CONFIG value below can also be set from the environment, e.g.
#   WEARER=A1_JAKE STAGES="full no_s3" ablation/run.sh stages start
# ─────────────────────────────────────────────────────────────────────────────

FIO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
REPO_DIR="$(dirname "$FIO_DIR")"
SCRIPT_PATH="$(readlink -f "$0")"

# ── CONFIG ───────────────────────────────────────────────────────────────────
WEARER="${WEARER:-A1_JAKE}"
ENV_NAME="${ENV_NAME:-egogpt}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

# Rows for `stages` and `window` (names from ablation/configs.py).
read -r -a STAGES <<< "${STAGES:-no_ctx no_s1 no_s2 no_s3 s3_only}"

# `sweep`: one knob of BASE_OPT_KWARGS over a value list, on top of SWEEP_BASE.
#   (paper grids)
#   delta_g_threshold      0.60 0.65 0.70 0.75 0.80 0.85        Fig. 6b  (δ_g)
#   max_prototypes         100 200 300 400 500                  Fig. 6a  (N_max)
#   tau_threshold          0.1 0.2 0.3                          Fig. 7b  (τ)
#   history_window         3 5 10                               Table 6  (k)
#   context_history_weight 0.3 0.5 0.7                          Fig. 7c  (α_h)
#   lambda_decay           2.6742e-6 1.6045e-6 1.146e-6 5.73e-7 Fig. 7a  (λ, per second)
#   delta_h_threshold      0.2 0.3 0.5 0.7                      Table 7  (δ_h)
SWEEP_BASE="${SWEEP_BASE:-full}"
SWEEP_PARAM="${SWEEP_PARAM:-delta_h_threshold}"
read -r -a SWEEP_VALUES <<< "${SWEEP_VALUES:-0.2 0.3 0.5 0.7}"

MAX_PARALLEL="${MAX_PARALLEL:-0}"        # 0 = all at once | 1 = sequential | N = at most N
THREADS_PER_JOB="${THREADS_PER_JOB:-4}"  # (concurrent jobs) x threads <= physical cores
CLEAN_FIRST="${CLEAN_FIRST:-0}"          # 1 = delete a run's output dir + t-SNE files first

# Data. Phase 1 (LLM extraction) is skipped: the FIO cache must already exist.
CAPTIONS_PATH="${CAPTIONS_PATH:-$REPO_DIR/generated_caption/${WEARER}_meta.json}"
FIO_CACHE_DIR="${FIO_CACHE_DIR:-$REPO_DIR/FIO_mem/${WEARER}/fio_cache}"            # 7-day Phase-1 cache
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/FIO_mem/${WEARER}_ablation}"
LOG_DIR="${LOG_DIR:-$FIO_DIR/log/ablation}"
QA_DATA_PATH="${QA_DATA_PATH:-$REPO_DIR/benchmarks/QA/EgoLifeQA_A1_JAKE.json}"   # paper: 500 questions, Jake
RESULTS_DIR="${RESULTS_DIR:-$FIO_DIR/results/ablation}"
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    sed -n '3,9p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'
    exit 1
}

MODE="${1:-}"; shift || true
WINDOW=""
if [ "$MODE" = "window" ]; then WINDOW="${1:-}"; shift || true; fi
CMD="${1:-}"

# ── Build the run list: TAGS / ROWS / SVALS, plus per-mode cache + out root ──
TAGS=(); ROWS=(); SVALS=()
TARGET_DAY=7
CACHE="$FIO_CACHE_DIR"
RUN_OUT_ROOT="$OUT_ROOT"
case "$MODE" in
    stages)
        SESSION="fio_abl_stages"
        for r in "${STAGES[@]}"; do TAGS+=("$r"); ROWS+=("$r"); SVALS+=(""); done ;;
    sweep)
        SESSION="fio_abl_sweep"
        for v in "${SWEEP_VALUES[@]}"; do
            TAGS+=("${SWEEP_BASE}__${SWEEP_PARAM}=${v}"); ROWS+=("$SWEEP_BASE"); SVALS+=("$v")
        done ;;
    window)
        case "$WINDOW" in
            day1)          TARGET_DAY=1 ;;
            day1-[2-7])    TARGET_DAY="${WINDOW#day1-}" ;;
            *) echo "window must be day1 | day1-2 .. day1-7 (got '${WINDOW}')"; exit 1 ;;
        esac
        # The full cache serves every window: build_memory.py only reads the
        # cache files of captions in DAY1..DAY<TARGET_DAY>.
        RUN_OUT_ROOT="$OUT_ROOT/$WINDOW"
        SESSION="fio_abl_${WINDOW}"
        for r in "${STAGES[@]}"; do TAGS+=("${r}_${WINDOW}"); ROWS+=("$r"); SVALS+=(""); done ;;
    ls) tmux ls 2>/dev/null || echo "no tmux server"; exit 0 ;;
    *)  usage ;;
esac
SESSION="${FIO_SESSION:-$SESSION}"
PIDFILE="$LOG_DIR/${SESSION}.pids"

out_dir() {   # output dir for run i
    if [ "$MODE" = "window" ]; then echo "$RUN_OUT_ROOT/${ROWS[$1]}"; else echo "$RUN_OUT_ROOT/${TAGS[$1]}"; fi
}

plan() {
    echo "mode=$MODE${WINDOW:+ window=$WINDOW}  session=$SESSION  wearer=$WEARER"
    echo "  cache      : $CACHE  (TARGET_DAY=$TARGET_DAY)"
    echo "  captions   : $CAPTIONS_PATH"
    echo "  parallel   : MAX_PARALLEL=$MAX_PARALLEL  THREADS_PER_JOB=$THREADS_PER_JOB  CLEAN_FIRST=$CLEAN_FIRST"
    for i in "${!TAGS[@]}"; do
        local sw=""; [ -n "${SVALS[$i]}" ] && sw="  SWEEP_PARAM=$SWEEP_PARAM SWEEP_VAL=${SVALS[$i]}"
        printf "  %-34s ABLATION=%s%s\n  %-34s -> %s\n" "${TAGS[$i]}" "${ROWS[$i]}" "$sw" "" "$(out_dir "$i")"
    done
}

start() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "[start] tmux session '$SESSION' already exists — '$0 $MODE ${WINDOW:+$WINDOW }status' or 'stop' first."
        exit 1
    fi
    [ -f "$CONDA_SH" ] || echo "[start] WARNING: CONDA_SH not found: $CONDA_SH"
    [ -d "$CACHE" ]    || echo "[start] WARNING: FIO cache not found: $CACHE"
    for r in "${ROWS[@]}"; do
        [ "$r" = "no_s1" ] && echo "[start] NOTE: no_s1 uses target_keep_rate from ablation/configs.py — set it to the full model's measured keep-rate."
    done
    mkdir -p "$LOG_DIR"; : > "$PIDFILE"
    local master="$LOG_DIR/${SESSION}_$(date +%F_%H%M).log"
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "bash '$SCRIPT_PATH' $MODE ${WINDOW:+$WINDOW }_run 2>&1 | tee '$master'" C-m
    plan
    echo "[start] launched in tmux session '$SESSION'.  master log: $master"
    echo "        check: $0 $MODE ${WINDOW:+$WINDOW }status     watch: $0 $MODE ${WINDOW:+$WINDOW }attach"
}

# Internal: runs INSIDE the tmux pane.
_run() {
    echo "######## $SESSION START $(date) ########"
    source "$CONDA_SH" && conda activate "$ENV_NAME" && cd "$FIO_DIR" || {
        echo "[_run] FATAL: conda activate / cd failed. Check CONDA_SH, ENV_NAME."; exit 1; }
    [ -d "$CACHE" ] || { echo "[_run] FATAL: FIO cache not found: $CACHE"; exit 1; }
    mkdir -p "$LOG_DIR"; touch "$PIDFILE"

    local cap="$MAX_PARALLEL"
    [ "$cap" -le 0 ] 2>/dev/null && cap=${#TAGS[@]}
    local stamp; stamp="$(date +%F_%H%M)"

    for i in "${!TAGS[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do sleep 5; done
        local tag="${TAGS[$i]}" row="${ROWS[$i]}" sval="${SVALS[$i]}" out; out="$(out_dir "$i")"
        if [ "$CLEAN_FIRST" = "1" ] && [[ -n "$out" && "$out" == "$OUT_ROOT"/* ]]; then
            rm -rf -- "$out"
            rm -f  -- "$FIO_DIR/t_sne/merge_stats_${tag}.json" "$FIO_DIR/t_sne/prototype_tsne_data_${tag}.json"
        fi
        local log="$LOG_DIR/${SESSION}_${tag}_${stamp}.log"
        echo "[$(date +%H:%M:%S)] launching $tag (ABLATION=$row${sval:+ $SWEEP_PARAM=$sval}) -> $out"
        (
            OMP_NUM_THREADS="$THREADS_PER_JOB" MKL_NUM_THREADS="$THREADS_PER_JOB" \
            OPENBLAS_NUM_THREADS="$THREADS_PER_JOB" \
            ABLATION="$row" SWEEP_PARAM="${sval:+$SWEEP_PARAM}" SWEEP_VAL="$sval" \
            RUN_TAG="$tag" OUTPUT_DIR="$out" FIO_CACHE_DIR="$CACHE" TARGET_DAY="$TARGET_DAY" \
            CAPTIONS_PATH="$CAPTIONS_PATH" SKIP_PHASE1=1 \
            python -u scripts/build_memory.py > "$log" 2>&1 &
            local pid=$!; echo "$pid" >> "$PIDFILE"
            wait "$pid"; local rc=$?
            echo "___FINISHED___ tag=$tag exit=$rc $(date)" >> "$log"
        ) &
    done
    wait
    echo; status
    echo "######## $SESSION COMPLETE $(date) ########"
}

status() {
    echo "──────── $SESSION ────────"
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo "  tmux session: alive"; else echo "  tmux session: none"; fi
    for tag in "${TAGS[@]}"; do
        local log; log="$(ls -t "$LOG_DIR/${SESSION}_${tag}_"*.log 2>/dev/null | head -1)"
        local state="not started"
        if [ -n "$log" ]; then
            local fin; fin="$(grep '___FINISHED___' "$log" 2>/dev/null | tail -1)"
            if [ -n "$fin" ]; then state="${fin#___FINISHED___ tag=$tag }"; else state="running  ($(tail -c 200 "$log" | tr '\r' '\n' | tail -1 | cut -c1-60))"; fi
        fi
        printf "  %-34s %s\n" "$tag" "$state"
    done
}

# Evaluate every build of this mode with scripts/qa.py (sequential, foreground).
qa() {
    mkdir -p "$RESULTS_DIR"
    cd "$FIO_DIR" || exit 1
    for i in "${!TAGS[@]}"; do
        local tag="${TAGS[$i]}" row="${ROWS[$i]}" out; out="$(out_dir "$i")"
        local res="$RESULTS_DIR/${SESSION}_${tag}.json"
        if [ ! -d "$out/chroma_fio_opt" ]; then echo "[qa] skip $tag: no build at $out"; continue; fi
        if [ -f "$res" ]; then echo "[qa] skip $tag: $res exists"; continue; fi
        echo "[qa] $tag  (build $out)"
        QA_BUILD_DIR="$out" QA_DATA_PATH="$QA_DATA_PATH" BUILD_CONFIG="$row" MAX_DAY="$TARGET_DAY" \
        MEM_WINDOW="${WINDOW:-day1-7}" QA_RESULTS="$res" \
            python -u scripts/qa.py > "$LOG_DIR/qa_${SESSION}_${tag}.log" 2>&1 \
            || echo "[qa] $tag FAILED (log: $LOG_DIR/qa_${SESSION}_${tag}.log)"
    done
    report
}

# GA / Acc. per build, from the saved QA results.
report() {
    printf "%-34s %7s %7s   %s\n" "run" "GA" "Acc." "results"
    for tag in "${TAGS[@]}"; do
        local res="$RESULTS_DIR/${SESSION}_${tag}.json"
        if [ -f "$res" ]; then
            python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print(f\"{sys.argv[2]:<34} {m['GA']:7.2f} {m['accuracy']:7.2f}   {sys.argv[1]}\")" "$res" "$tag"
        else
            printf "%-34s %7s %7s\n" "$tag" "—" "—"
        fi
    done
}

stop() {
    if [ -s "$PIDFILE" ]; then
        while read -r pid; do kill "$pid" 2>/dev/null && echo "[stop] killed build pid $pid"; done < "$PIDFILE"
        : > "$PIDFILE"
    else
        echo "[stop] no recorded builds for $SESSION"
    fi
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "[stop] killed session $SESSION" || echo "[stop] no session"
}

case "$CMD" in
    plan)   plan ;;
    start)  start ;;
    status) status ;;
    attach) tmux attach -t "$SESSION" ;;
    stop)   stop ;;
    qa)     mkdir -p "$LOG_DIR"; qa ;;
    report) report ;;
    _run)   _run ;;
    *)      usage ;;
esac
