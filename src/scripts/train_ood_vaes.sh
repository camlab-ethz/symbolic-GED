#!/bin/bash
# Train VAE models for OOD experiments
# 
# This script trains both Grammar and Token VAEs on data that EXCLUDES
# KdV and Schrödinger families, enabling true OOD evaluation.
#
# Usage:
#   chmod +x scripts/train_ood_vaes.sh
#   ./scripts/train_ood_vaes.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBGEN_DIR="$(dirname "$SCRIPT_DIR")"
cd "$LIBGEN_DIR"

echo "============================================================"
echo "OOD VAE TRAINING PIPELINE"
echo "============================================================"

# Step 1: Create OOD splits
echo ""
echo "[1/3] Creating OOD data splits..."
python scripts/create_ood_splits.py \
    --exclude-families kdv schrodinger \
    --output-dir splits/ood_kdv_schrodinger \
    --seed 42

# Step 2: Train Grammar VAE on OOD split
echo ""
echo "[2/3] Training Grammar VAE (excluding KdV, Schrödinger)..."
python vae/train.py \
    --tokenization grammar \
    --prod_path data/production_ids.npy \
    --masks_path data/production_masks.npy \
    --split_dir splits/ood_kdv_schrodinger \
    --batch_size 256 \
    --max_epochs 200 \
    --z_dim 26 \
    --beta 1e-5 \
    --lr 0.001 \
    --encoder_hidden 128 \
    --encoder_conv_layers 3 \
    --decoder_hidden 80 \
    --decoder_layers 3 \
    --checkpoint_dir checkpoints/grammar_vae_ood \
    --experiment_name grammar_vae_ood

# Step 3: Train Token VAE on OOD split
echo ""
echo "[3/3] Training Token VAE (excluding KdV, Schrödinger)..."
python vae/train.py \
    --tokenization token \
    --token_path data/token_ids.npy \
    --masks_path data/token_masks.npy \
    --split_dir splits/ood_kdv_schrodinger \
    --batch_size 256 \
    --max_epochs 200 \
    --z_dim 26 \
    --beta 1e-5 \
    --lr 0.001 \
    --encoder_hidden 128 \
    --encoder_conv_layers 3 \
    --decoder_hidden 80 \
    --decoder_layers 3 \
    --checkpoint_dir checkpoints/token_vae_ood \
    --experiment_name token_vae_ood

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run OOD evaluation:"
echo "     python scripts/run_ood_evaluation.py"
echo ""
