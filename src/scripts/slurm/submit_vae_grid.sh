#!/bin/bash
# Submit VAE training grid job array (2x2 grid: 2 tokenizations x 2 betas)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBGEN_DIR="$SCRIPT_DIR/.."

cd "$LIBGEN_DIR" || { echo "Error: Could not cd to $LIBGEN_DIR"; exit 1; }

# Create log directory
mkdir -p slurm_logs

# Submit job array
echo "Submitting VAE training grid job array..."
JOB_ID=$(sbatch "$SCRIPT_DIR/vae_train_grid.sbatch" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Job submitted successfully!"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo ""
echo "Grid mapping (SLURM_ARRAY_TASK_ID -> tokenization, beta):"
echo "  0 -> grammar, 2e-4"
echo "  1 -> grammar, 1e-2"
echo "  2 -> token, 2e-4"
echo "  3 -> token, 1e-2"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f slurm_logs/vae_grid_${JOB_ID}_0.out  # Task 0"
echo "  tail -f slurm_logs/vae_grid_${JOB_ID}_1.out  # Task 1"
echo "  tail -f slurm_logs/vae_grid_${JOB_ID}_2.out  # Task 2"
echo "  tail -f slurm_logs/vae_grid_${JOB_ID}_3.out  # Task 3"
echo "=========================================="
