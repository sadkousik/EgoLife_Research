#!/usr/bin/env bash
# ==============================================================================
# run_phase2_imprint.sh
# Runs Imprint Phase 2 build only (fio_v3.py, SKIP_PHASE1=True) in a detached
# tmux session. Edit hyperparameters directly in fio_v3.py before launching.
#
# Run this FROM your activated env (the (egogpt) prompt) so the conda base
# resolves correctly.
#
#   ./run_phase2_imprint.sh
#   tmux attach -t imprint     # watch    (detach: Ctrl-b then d)
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rs/24CS92R01/EGOLIFE/EgoLife_copy/EgoRAG}"
CONDA_ENV="${CONDA_ENV:-egogpt}"
SESSION="imprint"
LOG_DIR="${PROJECT_ROOT}/run_logs"
LOG_FILE="${LOG_DIR}/imprint_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

# Resolve conda base now, while the env is active in this shell.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"

tmux has-session -t "${SESSION}" 2>/dev/null && {
  echo "Session '${SESSION}' already running. Attach: tmux attach -t ${SESSION}"
  echo "Or kill it:  tmux kill-session -t ${SESSION}"
  exit 1
}

read -r -d '' INNER <<EOF || true
cd "${PROJECT_ROOT}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "python: \$(which python)"

echo "Phase 2 build ..."
python fio_v3.py 2>&1 || { echo "BUILD FAILED"; exit 1; }

echo "DONE."
EOF

tmux new-session -d -s "${SESSION}" \
  "bash -lc '{ ${INNER} ; } 2>&1 | tee \"${LOG_FILE}\"; echo; echo \"[done — press enter to close]\"; read'"

echo "Launched tmux session '${SESSION}' (env: ${CONDA_ENV})."
echo "  Watch: tmux attach -t ${SESSION}"
echo "  Log:   ${LOG_FILE}"