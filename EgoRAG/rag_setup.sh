#!/bin/bash
#SBATCH --job-name=egorag-setup
#SBATCH --output=log/inst_rag_%j.log
#SBATCH --error=log/install_rag_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodelist=gnode1

set -e  # Exit on any error
set -x  # Echo commands to log

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment if not exists
#conda env list | grep egogpt || conda create -n egogpt python=3.10 -y
conda activate egogpt

# Step 1: Install PyTorch with CUDA 12.1 support
#conda install -c pytorch -c nvidia \
#    pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -y

# Step 2: Install project dependencies and training extras
pip install -e .        # Local package

