#!/bin/bash
#SBATCH --job-name=clustering_metrics
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/clustering_metrics_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/clustering_metrics_%j.err
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

# Single source of truth for dataset paths (override by exporting PATHS_CONFIG before sbatch)
PATHS_CONFIG="${PATHS_CONFIG:-configs/paths_48000_fixed.yaml}"
eval "$(python3 scripts/print_paths_env.py --paths-config ${PATHS_CONFIG})"

# Current model checkpoints (same as t-SNE script)
GRAMMAR_BETA2E4="checkpoints_48000_fixed/grammar_vae/beta_2e-4_seed_42/best-epoch=411-val/seq_acc=0.9983.ckpt"
GRAMMAR_BETA1E2="checkpoints_48000_fixed/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt"
TOKEN_BETA2E4="checkpoints_48000_fixed/token_vae/beta_2e-4_seed_42/best-epoch=307-val/seq_acc=0.9992.ckpt"
TOKEN_BETA1E2="checkpoints_48000_fixed/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt"

echo "=========================================="
echo "Clustering metrics (NMI) for 4 models"
echo "Splits: train/val/test"
echo "Outputs: analysis_results/clustering/"
echo "=========================================="

RUN_ID_ARG=""
if [ -n "${RUN_ID}" ]; then
  RUN_ID_ARG="--run-id ${RUN_ID}"
fi

python3 -m analysis.run_clustering_metrics \
  --csv-metadata ${CSV_METADATA} \
  --split-dir ${SPLIT_DIR} \
  --splits train,val,test \
  --device cuda \
  --batch-size 256 \
  --max-samples 10000 \
  --outdir ${ANALYSIS_ROOT}/clustering_48000 \
  ${RUN_ID_ARG} \
  --ckpt grammar_beta2e4:${GRAMMAR_BETA2E4} \
  --ckpt grammar_beta1e2:${GRAMMAR_BETA1E2} \
  --ckpt token_beta2e4:${TOKEN_BETA2E4} \
  --ckpt token_beta1e2:${TOKEN_BETA1E2}

echo "=========================================="
echo "Done."
echo "See: analysis_results/clustering_48000/summary.csv"
echo "=========================================="

