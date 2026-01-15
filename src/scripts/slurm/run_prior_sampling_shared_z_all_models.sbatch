#!/bin/bash
#SBATCH --job-name=prior_shared_z_all
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/prior_shared_z_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/prior_shared_z_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Single source of truth for dataset/output roots (override by exporting PATHS_CONFIG before sbatch)
PATHS_CONFIG="${PATHS_CONFIG:-configs/paths_48000_fixed.yaml}"
eval "$(python3 scripts/print_paths_env.py --paths-config ${PATHS_CONFIG})"

# Current model checkpoints (48000_fixed operator-only dataset)
GRAMMAR_BETA2E4="checkpoints_48000_fixed/grammar_vae/beta_2e-4_seed_42/best-epoch=411-val/seq_acc=0.9983.ckpt"
GRAMMAR_BETA1E2="checkpoints_48000_fixed/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt"
TOKEN_BETA2E4="checkpoints_48000_fixed/token_vae/beta_2e-4_seed_42/best-epoch=307-val/seq_acc=0.9992.ckpt"
TOKEN_BETA1E2="checkpoints_48000_fixed/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt"

N_SAMPLES=20000
SEED=42
BATCH_SIZE=512
OUTDIR="${ANALYSIS_ROOT}/prior_sampling_shared_z_48000/n${N_SAMPLES}_seed${SEED}"

echo "=========================================="
echo "Prior sampling with SHARED Z (all 4 models)"
echo "=========================================="
echo "N_SAMPLES=${N_SAMPLES}"
echo "SEED=${SEED}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "OUTDIR=${OUTDIR}"
echo "GRAMMAR_BETA2E4=${GRAMMAR_BETA2E4}"
echo "GRAMMAR_BETA1E2=${GRAMMAR_BETA1E2}"
echo "TOKEN_BETA2E4=${TOKEN_BETA2E4}"
echo "TOKEN_BETA1E2=${TOKEN_BETA1E2}"
echo "=========================================="

python3 -m vae.prior_sampling_shared_z \
  --grammar_beta2e4 "${GRAMMAR_BETA2E4}" \
  --grammar_beta1e2 "${GRAMMAR_BETA1E2}" \
  --token_beta2e4 "${TOKEN_BETA2E4}" \
  --token_beta1e2 "${TOKEN_BETA1E2}" \
  --outdir "${OUTDIR}" \
  --n_samples "${N_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device cuda \
  --save_z

echo ""
echo "=========================================="
echo "Done. Outputs in: ${OUTDIR}"
echo "=========================================="

