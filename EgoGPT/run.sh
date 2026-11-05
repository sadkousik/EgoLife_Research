#!/bin/bash
#SBATCH --job-name=shure_meta
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
##SBATCH --mem=64G
#SBATCH --cpus-per-task=5      # Use enough CPU for faster compilation
#SBATCH --partition=gpu_l40 #gpupart12hr#gpupart48hr  #gpupart12hr


# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt


# 👇 EXPERIMENT DESCRIPTION (CHANGE THIS ONLY)
export EXP_DESC="Shure | EgoGPT "

export SLURM_STDOUT=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
export SLURM_STDERR=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err


python3  inference_egolife.py
