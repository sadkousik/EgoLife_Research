#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# jobs/build_tmux.sh — run ONE scripts/build_memory.py build in a tmux session.
#
#   jobs/build_tmux.sh                                  # settings from build_memory.py defaults
#   ABLATION=no_s3 SKIP_PHASE1=1 OUTPUT_DIR=../FIO_mem/x jobs/build_tmux.sh
#   tmux attach -t fio_build                            # watch (detach: Ctrl-b then d)
#
# Build settings are passed through explicitly: a tmux session inherits the
# tmux *server's* environment, not this shell's. For many builds at once use
# ablation/run.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FIO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-egogpt}"
SESSION="${SESSION:-fio_build}"
LOG_DIR="${LOG_DIR:-$FIO_DIR/log/build}"
LOG_FILE="$LOG_DIR/build_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

# Every variable scripts/build_memory.py (and the libraries it uses) reads.
PASS_VARS=(ABLATION SWEEP_PARAM SWEEP_VAL RUN_TAG SEED
           CAPTIONS_PATH FIO_CACHE_DIR OUTPUT_DIR TARGET_DAY SKIP_PHASE1 STORE_RAW
           LOCAL_DB_ROOT QWEN_BACKEND QWEN_MODEL_ID QWEN_REVISION EXP_ROOT EXP_DESC
           EXP_NOTIFY_EMAIL EXP_NOTIFY_APP_PASSWORD CUDA_VISIBLE_DEVICES
           OMP_NUM_THREADS MKL_NUM_THREADS HF_HOME HF_HUB_CACHE)
EXPORTS=""
for v in "${PASS_VARS[@]}"; do
    [ -n "${!v+x}" ] && EXPORTS+="export $v=$(printf '%q' "${!v}"); "
done

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Attach: tmux attach -t $SESSION   (or: tmux kill-session -t $SESSION)"
    exit 1
fi

INNER="cd $(printf '%q' "$FIO_DIR") && source $(printf '%q' "$CONDA_BASE/etc/profile.d/conda.sh") \
&& conda activate $(printf '%q' "$CONDA_ENV") && ${EXPORTS} echo \"python: \$(which python)\" \
&& python -u scripts/build_memory.py; echo \"___FINISHED___ exit=\$? \$(date)\""

tmux new-session -d -s "$SESSION" \
    "bash -lc $(printf '%q' "{ $INNER ; } 2>&1 | tee $(printf '%q' "$LOG_FILE"); echo; echo '[done — press enter to close]'; read")"

echo "Launched tmux session '$SESSION' (env: $CONDA_ENV)."
echo "  Passed: ${EXPORTS:-<nothing — build_memory.py defaults>}" | sed 's/export //g; s/; / /g' | sed -E 's/(EXP_NOTIFY_APP_PASSWORD=)[^ ]*/\1***/'
echo "  Watch:  tmux attach -t $SESSION"
echo "  Log:    $LOG_FILE"
