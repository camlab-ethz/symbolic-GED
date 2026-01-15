#!/bin/bash
#SBATCH --job-name=test_large_sample
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/test_large_sample_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/test_large_sample_%j.err
#SBATCH --time=04:00:00
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
echo "Testing Generative Abilities with 20k samples"
echo "=========================================="
echo "This verifies validity rates are not due to cherry-picking"
echo "=========================================="

python -m vae.test_generative_large_sample \
    --n_samples 20000 \
    --seed 42 \
    --device cuda \
    --batch_size 500

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
