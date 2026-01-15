#!/bin/bash
#SBATCH --job-name=vae_compare_same_latents
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/compare_same_latents_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/compare_same_latents_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /cluster/work/math/ooikonomou/symbolic-GED/src

echo "=========================================="
echo "Comparing Models with Same Latent Vectors"
echo "(Using IMPROVED validation function)"
echo "=========================================="
echo "Number of samples: 1000"
echo "Random seed: 42"
echo "Working directory: $(pwd)"
echo "=========================================="

# Run the comparison script
python -m vae.compare_models_same_latents \
    --n_samples 1000 \
    --seed 42 \
    --device cuda

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "Results saved to: generation_results/"
echo "=========================================="
