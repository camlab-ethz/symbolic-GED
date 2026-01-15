#!/bin/bash
# Submit both Grammar and Token VAE training jobs to SLURM
# 
# Usage: bash vae/train/submit_both.sh

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

echo "================================================================================"
echo "Submitting Full-Scale VAE Training Jobs"
echo "================================================================================"
echo ""

# Submit Grammar VAE
echo "Submitting Grammar VAE training..."
GRAMMAR_JOB=$(sbatch vae/train/submit_grammar_vae.sh | awk '{print $4}')
echo "  → Grammar VAE Job ID: $GRAMMAR_JOB"
echo "  → Logs: logs/training/grammar_vae_${GRAMMAR_JOB}.{out,err}"
echo "  → Checkpoints: checkpoints/grammar_vae/"
echo ""

# Submit Token VAE
echo "Submitting Token VAE training..."
TOKEN_JOB=$(sbatch vae/train/submit_token_vae.sh | awk '{print $4}')
echo "  → Token VAE Job ID: $TOKEN_JOB"
echo "  → Logs: logs/training/token_vae_${TOKEN_JOB}.{out,err}"
echo "  → Checkpoints: checkpoints/token_vae/"
echo ""

echo "================================================================================"
echo "Both jobs submitted!"
echo "================================================================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check progress:"
echo "  tail -f logs/training/grammar_vae_${GRAMMAR_JOB}.out"
echo "  tail -f logs/training/token_vae_${TOKEN_JOB}.out"
echo ""
echo "Cancel jobs if needed:"
echo "  scancel $GRAMMAR_JOB"
echo "  scancel $TOKEN_JOB"
echo "================================================================================"
