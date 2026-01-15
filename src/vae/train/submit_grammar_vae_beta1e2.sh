#!/bin/bash
#SBATCH --job-name=grammar_vae_beta1e2
#SBATCH --output=logs/training/grammar_vae_beta1e2_%j.out
#SBATCH --error=logs/training/grammar_vae_beta1e2_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

set -eo pipefail

# Load modules
module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Set working directory
WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

# Create directories
mkdir -p logs/training
mkdir -p checkpoints/grammar_vae

echo "================================================================================"
echo "Grammar VAE Training (beta=1e-2)"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Start time: $(date)"
echo "Working directory: $WORK_DIR"
echo "Beta: 0.01 (1e-2) - Strong VAE regularization"
echo "================================================================================"

# Run training with beta=0.01 override
python3 -m vae.train.train \
    --tokenization grammar \
    --seed 42 \
    --gpus 1 \
    --batch_size 256 \
    --epochs 1000 \
    --beta 0.01

echo ""
echo "================================================================================"
echo "Training completed at: $(date)"
echo "Checkpoints: checkpoints/grammar_vae/"
echo "Logs: logs/training/grammar_vae_beta1e2_${SLURM_JOB_ID:-<job_id>}.{out,err}"
echo "================================================================================"
