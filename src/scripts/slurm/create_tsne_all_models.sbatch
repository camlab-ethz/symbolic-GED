#!/bin/bash
#SBATCH --job-name=tsne_all_models
#SBATCH --output=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/tsne_all_models_%j.out
#SBATCH --error=/cluster/work/math/ooikonomou/symbolic-GED/src/slurm_logs/tsne_all_models_%j.err
#SBATCH --time=03:00:00
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

# Current model checkpoints
GRAMMAR_BETA2E4="checkpoints_48000_fixed/grammar_vae/beta_2e-4_seed_42/best-epoch=411-val/seq_acc=0.9983.ckpt"
GRAMMAR_BETA1E2="checkpoints_48000_fixed/grammar_vae/beta_0.01_seed_42/best-epoch=517-val/seq_acc=0.0755.ckpt"
TOKEN_BETA2E4="checkpoints_48000_fixed/token_vae/beta_2e-4_seed_42/best-epoch=307-val/seq_acc=0.9992.ckpt"
TOKEN_BETA1E2="checkpoints_48000_fixed/token_vae/beta_0.01_seed_42/best-epoch=136-val/seq_acc=0.0016.ckpt"

echo "=========================================="
echo "Creating t-SNE plots for all 4 models"
echo "on all 3 dataset splits (train/val/test)"
echo "=========================================="

# Generate for each split
for SPLIT in train val test; do
    echo ""
    echo "Processing ${SPLIT} split..."
    echo "----------------------------------------"
    
    RUN_ID_SUFFIX=""
    if [ -n "${RUN_ID}" ]; then
      RUN_ID_SUFFIX="_${RUN_ID}"
    fi
    OUTDIR="${FIGURES_ROOT}/tsne_48000_comparison_${SPLIT}${RUN_ID_SUFFIX}"
    
    python3 scripts/plot_tsne_comparison.py \
        --split ${SPLIT} \
        --csv-metadata ${CSV_METADATA} \
        --split-dir ${SPLIT_DIR} \
        --grammar-ckpt-beta1 ${GRAMMAR_BETA2E4} \
        --token-ckpt-beta1 ${TOKEN_BETA2E4} \
        --grammar-ckpt-beta2 ${GRAMMAR_BETA1E2} \
        --token-ckpt-beta2 ${TOKEN_BETA1E2} \
        --outdir ${OUTDIR} \
        --seed 42 \
        --device cuda
    
    echo "Completed ${SPLIT} split"
done

echo ""
echo "=========================================="
echo "t-SNE generation complete!"
echo "Results saved to: experiments/reports/tsne_48000_comparison_*/"
echo "=========================================="
