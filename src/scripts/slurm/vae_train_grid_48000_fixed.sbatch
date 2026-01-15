#!/bin/bash
#SBATCH --job-name=vae48000_fix
#SBATCH --array=0-3
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/vae48000_fix_%A_%a.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/vae48000_fix_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /cluster/work/math/ooikonomou/symbolic-GED/src || exit 1
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Task mapping:
# 0: grammar, beta=2e-4
# 1: grammar, beta=1e-2
# 2: token,   beta=2e-4
# 3: token,   beta=1e-2
TOKENIZATIONS=("grammar" "grammar" "token" "token")
BETAS=("2e-4" "1e-2" "2e-4" "1e-2")

TOK=${TOKENIZATIONS[$SLURM_ARRAY_TASK_ID]}
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training on operator-only 48000 dataset"
echo "Tokenization: $TOK"
echo "Beta: $BETA"
echo "Seed: 42"
echo "Config: configs/config_vae_48000_operator.yaml"
echo "Outputs: checkpoints_48000_fixed/"
echo "=========================================="

python3 -m vae.train.train \
  --config configs/config_vae_48000_operator.yaml \
  --tokenization "$TOK" \
  --beta "$BETA" \
  --seed 42 \
  --gpus 1 \
  --num_workers 4

echo "Done: $TOK beta=$BETA"

