#!/bin/bash
#SBATCH --job-name=vae_s3_b0.1
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/logs/vae_s3-%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/logs/vae_s3-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

# VAE Scenario 3: Standard VAE (β=0.1)
# Focus: True VAE behavior with structured latent space

echo "=============================================="
echo "VAE Scenario 3: Beta = 0.1 (Token)"
echo "=============================================="
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at $(date)"
echo ""

cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Run training with beta=0.1
python -m vae.train --tokenization token --beta 0.1

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "VAE Scenario 3 (β=0.1) completed successfully!"
else
    echo "VAE Scenario 3 failed with exit code: $EXIT_CODE"
fi
echo "Finished at $(date)"

exit $EXIT_CODE
