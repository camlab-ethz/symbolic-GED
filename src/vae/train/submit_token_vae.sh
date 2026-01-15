#!/bin/bash
#SBATCH --job-name=token_vae_full
#SBATCH --output=logs/training/token_vae_%j.out
#SBATCH --error=logs/training/token_vae_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

set -euo pipefail

# Load modules
module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set working directory
WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

# Create directories
mkdir -p logs/training
mkdir -p checkpoints/token_vae

echo "================================================================================"
echo "Token VAE Full Training"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "================================================================================"

# Run training with default config from config_vae.yaml
python3 -m vae.train.train \
    --tokenization token \
    --seed 42 \
    --gpus 1 \
    --batch_size 256 \
    --epochs 1000

echo ""
echo "================================================================================"
echo "Training completed at: $(date)"
echo "Checkpoints: checkpoints/token_vae/"
echo "Logs: logs/training/token_vae_${SLURM_JOB_ID}.{out,err}"
echo "================================================================================"
