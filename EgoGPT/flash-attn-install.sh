####################
#Added by Kousik
# for installing flash-attention on gpu node
####################
#!/bin/bash
#SBATCH --job-name=flash-attn-install_h100
#SBATCH --output=log/fa_m_h100_%j.log
#SBATCH --error=log/fa_m_h100_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodelist=gnode2
#SBATCH --cpus-per-task=8          # Use enough CPU for faster compilation

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egogpt

# Use system-installed CUDA toolkit
# Update this path to match your system's actual CUDA location



# Ensure nvcc is found
nvcc --version || { echo "nvcc not found"; exit 1; }
#pip install -e ".[train]"
# Install FlashAttention from pip without build isolation
pip uninstall flash-attn -y
pip show flash-attn
#pip install flash-attn --no-build-isolation
pip install flash-attn==2.4.2 --no-build-isolation
