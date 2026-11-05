#!/bin/bash
#SBATCH --job-name=ada_build_qa
#SBATCH --output=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO/log/tempo/%x_%j.log
#SBATCH --error=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research/FIO/log/tempo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=research
#SBATCH --partition=gpu_l40
#SBATCH --qos=limited
#SBATCH --chdir=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research
#
# Build one wearer's memory with learnable/scripts/build_memory.py, then run
# EgoLifeQA on it through FIO/jobs/qw.sh. Runs in the cluster's `egogpt` env so
# the Chroma store has the pinned chromadb format (0.5.20) that QA reads.
#
#   CKPT     checkpoint for a learned build; empty = heuristic           (default: empty)
#   WEARER   e.g. A1_JAKE                                                (default: A1_JAKE)
#   OUT      output build folder                                         (required)
#   MAX_PROTOTYPES  override FIO's max_prototypes; must match the checkpoint
#                   (traces and models so far use 300)                   (default: FIO config)
#   QA_WEARER / QA_DATA_PATH   passed on to qw.sh                        (default: Jake's)
#
#   sbatch --job-name=jake_heur_lb --export=ALL,OUT=$REPO/FIO_mem/A1_JAKE_heuristic_lb learnable/jobs/build_qa.sh
#   sbatch --job-name=jake_learn   --export=ALL,CKPT=$REPO/learnable/outputs/ckpt_full_valA1_JAKE/best.pt,OUT=$REPO/FIO_mem/A1_JAKE_learnable_full learnable/jobs/build_qa.sh
set -euo pipefail
REPO=/home/rs/24CS92R01/EGOLIFE/EgoLife_Research
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt
export PYTHONUNBUFFERED=1

WEARER=${WEARER:-A1_JAKE}
CKPT=${CKPT:-}
: "${OUT:?set OUT to the build folder}"
if [[ -e $OUT ]]; then echo "$OUT already exists; refusing to overwrite" >&2; exit 1; fi

python -c "import chromadb, torch; print('[build_qa] chromadb', chromadb.__version__, '| torch', torch.__version__)"
python -m learnable.scripts.build_memory ${CKPT:+--ckpt "$CKPT"} \
    --captions "generated_caption/${WEARER}_meta.json" \
    --cache-dir "FIO_mem/${WEARER}/fio_cache" \
    --out "$OUT" ${MAX_PROTOTYPES:+--opt max_prototypes=$MAX_PROTOTYPES}
echo "[build_qa] build done: $OUT"

export QA_BUILD_DIR="$OUT"
export EXP_DESC="QA_$(basename "$OUT")"
bash FIO/jobs/qw.sh
