#!/bin/bash
#SBATCH --job-name=vae_generate
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/test_generative_abilities_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/test_generative_abilities_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
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

# Default parameters
N_SAMPLES=${N_SAMPLES:-1000}
SEED=${SEED:-42}

echo "=========================================="
echo "Testing Generative Abilities of VAE Models"
echo "=========================================="
echo "Number of samples: $N_SAMPLES"
echo "Random seed: $SEED"
echo "Working directory: $LIBGEN_DIR"
echo "=========================================="

# Run the test script
python -m vae.test_generative_abilities \
    --n_samples "$N_SAMPLES" \
    --seed "$SEED" \
    --device cuda

echo ""
echo "=========================================="
echo "Test complete!"
echo "Results saved to: generation_results/"
echo "=========================================="
