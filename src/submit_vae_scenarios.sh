#!/bin/bash
# Submit all VAE scenarios for both tokenization methods
# 4 beta values × 2 tokenizations = 8 jobs total

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "Submitting VAE Scenarios"
echo "4 beta values × 2 tokenizations = 8 jobs"
echo "=============================================="
echo ""

# Beta values to test
BETAS=(0.001 0.01 0.1 0.5)

# Submit jobs for both tokenization methods
for beta in "${BETAS[@]}"; do
    for tok in token grammar; do
        job_name="vae_${tok}_b${beta}"

        echo "Submitting: $job_name"

        sbatch --job-name="$job_name" \
               --output="$WORK_DIR/logs/${job_name}-%j.out" \
               --error="$WORK_DIR/logs/${job_name}-%j.err" \
               --nodes=1 \
               --ntasks=1 \
               --cpus-per-task=4 \
               --mem-per-cpu=8G \
               --time=24:00:00 \
               --gpus=nvidia_geforce_rtx_4090:1 \
               --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta $beta"
    done
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $WORK_DIR/logs/"
