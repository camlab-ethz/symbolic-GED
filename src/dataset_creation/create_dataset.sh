#!/bin/bash
# Dataset Creation Script
# Generates PDE dataset, fixes labels, creates splits, and tokenizes
# This is ONLY for dataset preparation - VAE training comes later

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

echo "========================================================================"
echo "PDE Dataset Creation Pipeline"
echo "========================================================================"

# Configuration
NUM_PER_FAMILY=${NUM_PER_FAMILY:-3000}
RANDOM_STATE=${RANDOM_STATE:-42}

# Single source of truth for "reporting-ready" pipeline paths (override via env var).
PATHS_CONFIG="${PATHS_CONFIG:-configs/paths_48000_fixed.yaml}"
eval "$(python3 scripts/print_paths_env.py --paths-config ${PATHS_CONFIG})"

# Derive the raw CSV path from the fixed CSV path in the config.
FIXED_CSV="${CSV_METADATA}"
RAW_CSV="${FIXED_CSV%_fixed.csv}.csv"

# Step 1: Generate dataset
echo ""
echo "Step 1: Generating PDE dataset..."
mkdir -p data/raw
python dataset_creation/generator.py \
    --output "${RAW_CSV}" \
    --num_per_family "$NUM_PER_FAMILY" \
    --seed "$RANDOM_STATE"

# Step 2: Fix labels
echo ""
echo "Step 2: Fixing dataset labels..."
python dataset_creation/fix_dataset_labels.py \
    --input "${RAW_CSV}" \
    --output "${FIXED_CSV}"

# Step 3: Validate dataset (continue even if warnings)
echo ""
echo "Step 3: Validating dataset..."
python dataset_creation/validate_dataset.py \
    --dataset "${FIXED_CSV}" || true

# Step 4: Create train/val/test splits
echo ""
echo "Step 4: Creating train/val/test splits..."
python dataset_creation/create_data_splits.py \
    --dataset "${FIXED_CSV}" \
    --output "${SPLIT_DIR}" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed "$RANDOM_STATE"

# Step 5: Tokenize dataset
echo ""
echo "Step 5: Tokenizing dataset..."
python dataset_creation/create_tokenized_data.py \
    --dataset "${FIXED_CSV}" \
    --splits "${SPLIT_DIR}" \
    --output "${TOKENIZED_DIR}"

echo ""
echo "========================================================================"
echo "✅ Dataset creation complete!"
echo "========================================================================"
echo ""
echo "Note: This script only creates the dataset. VAE training is separate."
echo ""
echo "Generated files:"
echo "  ${FIXED_CSV}  (dataset CSV)"
echo "  ${SPLIT_DIR}/"
echo "    ├── train_indices.npy"
echo "    ├── val_indices.npy"
echo "    └── test_indices.npy"
echo "  ${TOKENIZED_DIR}/"
echo "    ├── grammar/"
echo "    │   ├── train.npy"
echo "    │   ├── val.npy"
echo "    │   └── test.npy"
echo "    └── token/"
echo "        ├── train.npy"
echo "        ├── val.npy"
echo "        └── test.npy"
echo ""
