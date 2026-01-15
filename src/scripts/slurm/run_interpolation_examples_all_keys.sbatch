#!/bin/bash
#SBATCH --job-name=vae_interp_examples
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/interp_examples_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/interp_examples_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /cluster/work/math/ooikonomou/symbolic-GED/src

echo "=========================================="
echo "Interpolation examples (all dim × spatial_order keys)"
echo "Config: configs/interpolation_examples_all_keys.yaml"
echo "Outputs: analysis_results/interpolation_examples_all_keys_48000/"
echo "=========================================="

python3 -m vae.interpolate_family_pairs --config configs/interpolation_examples_all_keys.yaml

echo "=========================================="
echo "Done."
echo "See: analysis_results/interpolation_examples_all_keys/INDEX.csv"
echo "=========================================="

