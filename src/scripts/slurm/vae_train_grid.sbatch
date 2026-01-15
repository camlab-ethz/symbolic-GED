#!/bin/bash
#SBATCH --job-name=vae_grid
#SBATCH --array=0-3
# Use absolute path for logs to ensure consistency regardless of submission directory
# Note: #SBATCH directives don't support variable expansion, so we use hardcoded absolute path
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/vae_grid_%A_%a.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/vae_grid_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

# Use absolute path to src directory
LIBGEN_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"

cd "$LIBGEN_DIR" || { echo "Error: Could not cd to $LIBGEN_DIR"; exit 1; }
export PYTHONPATH="$LIBGEN_DIR:$PYTHONPATH"
echo "Working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# Load modules
module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6

# Activate conda if CONDA_ENV is set
if [ -n "$CONDA_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
fi

# Map array task ID to (tokenization, beta)
# Task 0: grammar, 2e-4
# Task 1: grammar, 1e-2
# Task 2: token, 2e-4
# Task 3: token, 1e-2

TOKENIZATIONS=("grammar" "grammar" "token" "token")
BETAS=("2e-4" "1e-2" "2e-4" "1e-2")

TOK=${TOKENIZATIONS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "VAE Training Grid Job"
echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Tokenization: $TOK"
echo "Beta: $BETA"
echo "Seed: 42"
echo "Working directory: $LIBGEN_DIR"
echo "=========================================="

# Run training
python -m vae.train.train \
    --tokenization "$TOK" \
    --beta "$BETA" \
    --seed 42 \
    --gpus 1 \
    --num_workers 4

echo "Training complete for $TOK beta=$BETA"
