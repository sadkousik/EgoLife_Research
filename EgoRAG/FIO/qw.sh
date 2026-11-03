#!/bin/bash
#SBATCH --job-name=A6_shure_MEM
#SBATCH --output=log/tempo/%x_%j.log
#SBATCH --error=log/tempo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --account=research
#SBATCH --partition=gpu_l40
#SBATCH --qos=limited

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt

# 👇 EXPERIMENT DESCRIPTION (CHANGE THIS ONLY)
export EXP_DESC="build mem for A6_shure"
export SLURM_STDOUT=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
export SLURM_STDERR=$(pwd)/log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err

# for together AI API
# Ensure env variable is available
export TOGETHER_API_KEY=$TOGETHER_API_KEY

#test
python3 -c "print('Hello, World!')"
#
nvidia-smi
#python3 Q_decompose.py
python3 fio_v3.py
#python3 build_fio_from_recovered.py
#python3 parse_Q.py
#python3 q_a_v6.py
# for K in 2 5 10 15 20; do
#   EVIDENCE_K=$K DB_MODE=opt python q_a_v62706.py
# done
#python -u q_a_v62706.py
#python q_a_egomem.py
#python q_a_egomem.py --wearer tasha
nvidia-smi
#bash submit_all_8.sh 
#bash run_ablation_abl.sh
#python3 build_fio_from_recovered.py