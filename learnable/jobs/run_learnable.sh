#!/usr/bin/env bash
# AdaImprint: record traces -> train -> build.
# Run from the repo root:  bash learnable/jobs/run_learnable.sh [trace|train|build|all]
#
# One wearer (train DAY1..DAY(SPLIT_DAY-1), validate DAY(SPLIT_DAY)..DAY7):
#   WEARERS="A4_LUCIA" SPLIT_DAY=6 EPOCHS=3 bash learnable/jobs/run_learnable.sh all
# Two or more wearers (leave VAL_WEARER out):
#   WEARERS="A4_LUCIA A5_KATRINA" VAL_WEARER=A5_KATRINA EPOCHS=3 bash learnable/jobs/run_learnable.sh all
set -euo pipefail
export PYTHONUNBUFFERED=1                                # logs show progress as it happens

FIO_MEM_ROOT=${FIO_MEM_ROOT:-FIO_mem}                   # <wearer>/fio_cache
CAPTIONS_DIR=${CAPTIONS_DIR:-generated_caption}         # <wearer>_meta.json
OUT=${OUT:-learnable/outputs}
MODE=${MODE:-full}                                      # full | params
EPOCHS=${EPOCHS:-10}
THREADS=${THREADS:-4}                                   # CPU threads per training run
TRAIN_ARGS=${TRAIN_ARGS:-}                              # extra train.py flags, e.g. "--w-past 5"
read -r -a WEARERS <<< "${WEARERS:-A1_JAKE A2_ALICE A3_TASHA A4_LUCIA A5_KATRINA A6_SHURE}"
VAL_WEARER=${VAL_WEARER:-${WEARERS[0]}}                 # held out / wearer to build
SPLIT_DAY=${SPLIT_DAY:-6}                               # single-wearer mode only

if [[ ${#WEARERS[@]} -eq 1 ]]; then
  SINGLE=1
  VAL_WEARER=${WEARERS[0]}
  CKPT_DIR=${CKPT_DIR:-$OUT/ckpt_${MODE}_${VAL_WEARER}_split${SPLIT_DAY}}
else
  SINGLE=0
  CKPT_DIR=${CKPT_DIR:-$OUT/ckpt_${MODE}_val${VAL_WEARER}}
fi

step=${1:-all}

if [[ $step == trace || $step == all ]]; then
  for w in "${WEARERS[@]}"; do
    if [[ -f $OUT/traces/$w.pt ]]; then echo "[skip] $w trace exists"; continue; fi
    if [[ ! -d $FIO_MEM_ROOT/$w/fio_cache ]]; then
      echo "[skip] $w: no $FIO_MEM_ROOT/$w/fio_cache"; continue
    fi
    python -m learnable.scripts.record_trace --wearer "$w" \
      --captions "$CAPTIONS_DIR/${w}_meta.json" \
      --cache-dir "$FIO_MEM_ROOT/$w/fio_cache" \
      --out "$OUT/traces/$w.pt"
  done
fi

if [[ $step == train || $step == all ]]; then
  if [[ ! -f $OUT/traces/$VAL_WEARER.pt ]]; then echo "no trace for $VAL_WEARER"; exit 1; fi
  if [[ $SINGLE == 1 ]]; then
    python -m learnable.scripts.train --mode "$MODE" --epochs "$EPOCHS" --threads "$THREADS" ${TRAIN_ARGS:-} \
      --train "$OUT/traces/$VAL_WEARER.pt" --split-day "$SPLIT_DAY" --out "$CKPT_DIR"
  else
    TRAIN=()
    for w in "${WEARERS[@]}"; do
      if [[ $w != "$VAL_WEARER" && -f $OUT/traces/$w.pt ]]; then TRAIN+=("$OUT/traces/$w.pt"); fi
    done
    if [[ ${#TRAIN[@]} -eq 0 ]]; then echo "no training traces in $OUT/traces"; exit 1; fi
    python -m learnable.scripts.train --mode "$MODE" --epochs "$EPOCHS" --threads "$THREADS" ${TRAIN_ARGS:-} \
      --train "${TRAIN[@]}" --val "$OUT/traces/$VAL_WEARER.pt" --out "$CKPT_DIR"
  fi
fi

if [[ $step == build || $step == all ]]; then
  # Builds the opt Chroma store for VAL_WEARER with the learned rule
  # (learnable/scripts/build_memory.py: Phase 2 only, CPU, FIO/scripts/build_memory.py untouched).
  # Extra flags go through BUILD_ARGS, e.g. BUILD_ARGS="--budget 1500 --doc-fix".
  # Single-wearer mode builds all 7 days, so only QA on DAY>=SPLIT_DAY is held out.
  python -m learnable.scripts.build_memory \
    --ckpt "$CKPT_DIR/best.pt" \
    --captions "$CAPTIONS_DIR/${VAL_WEARER}_meta.json" \
    --cache-dir "$FIO_MEM_ROOT/$VAL_WEARER/fio_cache" \
    --out "${BUILD_OUT:-$FIO_MEM_ROOT/${VAL_WEARER}_learnable_${MODE}}" ${BUILD_ARGS:-}
fi