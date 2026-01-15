#!/bin/bash
# Submit both Grammar and Token VAE training jobs with beta=5e-4
# 
# Usage: bash vae/train/submit_beta5e4.sh

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

echo "================================================================================"
echo "Submitting VAE Training Jobs with beta=5e-4"
echo "================================================================================"
echo ""

# Submit Grammar VAE with beta override
echo "Submitting Grammar VAE training (beta=5e-4)..."
GRAMMAR_JOB=$(sbatch --job-name=grammar_vae_beta5e4 \
    --output=logs/training/grammar_vae_beta5e4_%j.out \
    --error=logs/training/grammar_vae_beta5e4_%j.err \
    --time=48:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=nvidia_geforce_rtx_4090:1 \
    --wrap="cd $WORK_DIR && module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python3 -m vae.train.train --tokenization grammar --beta 0.0005" | awk '{print $4}')

echo "  → Grammar VAE Job ID: $GRAMMAR_JOB"
echo "  → Logs: logs/training/grammar_vae_beta5e4_${GRAMMAR_JOB}.{out,err}"
echo "  → Checkpoints: checkpoints/grammar_vae/"
echo ""

# Submit Token VAE with beta override
echo "Submitting Token VAE training (beta=5e-4)..."
TOKEN_JOB=$(sbatch --job-name=token_vae_beta5e4 \
    --output=logs/training/token_vae_beta5e4_%j.out \
    --error=logs/training/token_vae_beta5e4_%j.err \
    --time=48:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem-per-cpu=8G \
    --gpus=nvidia_geforce_rtx_4090:1 \
    --wrap="cd $WORK_DIR && module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 && source ~/miniconda3/etc/profile.d/conda.sh && conda activate base && python3 -m vae.train.train --tokenization token --beta 0.0005" | awk '{print $4}')

echo "  → Token VAE Job ID: $TOKEN_JOB"
echo "  → Logs: logs/training/token_vae_beta5e4_${TOKEN_JOB}.{out,err}"
echo "  → Checkpoints: checkpoints/token_vae/"
echo ""

echo "================================================================================"
echo "Both jobs submitted with beta=5e-4!"
echo "================================================================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check progress:"
echo "  tail -f logs/training/grammar_vae_beta5e4_${GRAMMAR_JOB}.out"
echo "  tail -f logs/training/token_vae_beta5e4_${TOKEN_JOB}.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $GRAMMAR_JOB"
echo "  scancel $TOKEN_JOB"
echo "================================================================================"
