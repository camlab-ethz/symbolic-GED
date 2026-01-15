#!/bin/bash
#SBATCH --job-name=vae_s1_b0.001
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/logs/vae_s1-%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/logs/vae_s1-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

# VAE Scenario 1: Light Regularization (β=0.001)
# Focus: Mostly reconstruction with slight latent space structure

echo "=============================================="
echo "VAE Scenario 1: Beta = 0.001 (Token)"
echo "=============================================="
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at $(date)"
echo ""

cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Run training with beta=0.001
python -m vae.train --tokenization token --beta 0.001

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "VAE Scenario 1 (β=0.001) completed successfully!"
else
    echo "VAE Scenario 1 failed with exit code: $EXIT_CODE"
fi
echo "Finished at $(date)"

exit $EXIT_CODE
