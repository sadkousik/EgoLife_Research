#!/bin/bash
#SBATCH --job-name=jake_qa
#SBATCH --output=log/tempo/%x_%j.log
#SBATCH --error=log/tempo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=research
#SBATCH --partition=gpu_l40
#SBATCH --qos=limited
#SBATCH --chdir=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO

# jobs/qw.sh — Slurm GPU job: build one FIO memory, then run EgoMem QA.
#
#   1. scripts/build_memory.py   with the local Qwen backend (Phase 1 + 2;
#                                settings come from the script defaults / env)
#   2. scripts/qa_egomem.py --wearer jake  with QWEN_BACKEND=openai, which
#                                needs a vLLM server already listening on
#                                localhost:8001 — this script does not start one.
#
# Edit --job-name and EXP_DESC per run. Logs: log/tempo/<job>_<id>.log/.err
# EXP_DESC, RUN_BUILD, RUN_QA, QA_BUILD_DIR, QA_DATA_PATH and QA_WEARER can also be
# set at submission time instead of editing this file, e.g.
#   sbatch --job-name=jake_qa_heur_lb \
#     --export=ALL,QA_BUILD_DIR=$REPO/FIO_mem/A1_JAKE_heuristic_lb,EXP_DESC="QA heuristic_lb" jobs/qw.sh
# Can be submitted from anywhere (e.g. FIO/ or FIO/jobs/): --chdir above and the
# cd below pin the working dir to FIO/, which all relative paths assume.
FIO_DIR=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO
cd "$FIO_DIR" || { echo "cannot cd to $FIO_DIR" >&2; exit 1; }
mkdir -p log/tempo

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt

# 👇 EXPERIMENT DESCRIPTION (CHANGE THIS ONLY)
export EXP_DESC="${EXP_DESC:-QA A1_JAKE_learnable_full}"
export SLURM_STDOUT=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
export SLURM_STDERR=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err

# for together AI API
# Ensure env variable is available
export TOGETHER_API_KEY=$TOGETHER_API_KEY

#test
python3 -c "print('Hello, World!')"
#
nvidia-smi
#python3 fio/llm.py

# 👇 WHICH STEPS TO RUN (1 = run, 0 = skip)
RUN_BUILD=${RUN_BUILD:-0}   # step 1: scripts/build_memory.py
RUN_QA=${RUN_QA:-1}         # step 2: scripts/qa.py

REPO=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research

# ── Step 1: build memory (batched local Transformers generation) ──
if [ "$RUN_BUILD" = "1" ]; then
    export QWEN_BACKEND=local
    export STORE_RAW=1
    # export CAPTIONS_PATH=$REPO/generated_caption/A6_SHURE_meta.json
    # export OUTPUT_DIR=$REPO/FIO_mem/A6_SHURE_RE_full
    python3 scripts/build_memory.py || { echo "build_memory.py failed" >&2; exit 1; }
fi

# ── Step 2: QA on a built memory ──
if [ "$RUN_QA" = "1" ]; then
    export QWEN_BACKEND=local                               # Qwen on this job's GPU (openai => needs vLLM on :8001)
    export QA_BUILD_DIR=${QA_BUILD_DIR:-$REPO/FIO_mem/A1_JAKE_learnable_full}   # must contain chroma_fio_opt/ (or chroma_fio_raw/ for DB_MODE=raw)
    export QA_DATA_PATH=${QA_DATA_PATH:-$REPO/benchmarks/QA/EgoLifeQA_A1_JAKE.json}
    export QA_WEARER=${QA_WEARER:-Jake}                     # must match the questions file (Jake | Tasha ...)
    export DB_MODE=opt                                      # opt | raw
    export BUILD_CONFIG=full                                # full|no_s1|no_s2|s3_only|no_s3|no_ctx|no_freq|no_recency|no_distinct
    export MEM_WINDOW=day1-7
    export MAX_DAY=7
    export EVIDENCE_K=5
    # export QA_RANKER=full                                 # optional override: full|no_e|no_a|no_imp|equal|only_e
    # export QA_RESULTS=$REPO/FIO/results/my_run.json       # optional; default FIO/results/qa_<build>_<exp>_<ts>.json
    if [ ! -d "$QA_BUILD_DIR/chroma_fio_$DB_MODE" ]; then
        echo "no $QA_BUILD_DIR/chroma_fio_$DB_MODE — build not finished?" >&2; exit 1
    fi
    echo "[qw] QA on $QA_BUILD_DIR ($EXP_DESC)"
    python3 scripts/qa.py
fi

