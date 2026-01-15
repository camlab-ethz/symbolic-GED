#!/bin/bash
# VAE Training Script
# Trains Grammar or Token VAE with proper logging and reproducibility
#
# Usage:
#   bash vae/train/train_vae.sh grammar
#   bash vae/train/train_vae.sh token
#   bash vae/train/train_vae.sh grammar --lr 0.0005 --batch_size 512

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$BASE_DIR"

# Parse arguments
TOKENIZATION=${1:-grammar}  # Default to grammar
shift || true  # Remove first arg, keep rest for python script

# Validate tokenization type
if [[ "$TOKENIZATION" != "grammar" && "$TOKENIZATION" != "token" ]]; then
    echo "Error: tokenization must be 'grammar' or 'token'"
    exit 1
fi

# Configuration
SEED=${SEED:-42}
OUTPUT_DIR="logs/training"
OUTPUT_LOG="$OUTPUT_DIR/train_${TOKENIZATION}_$(date +%Y%m%d_%H%M%S).out"
ERROR_LOG="$OUTPUT_DIR/train_${TOKENIZATION}_$(date +%Y%m%d_%H%M%S).err"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "checkpoints/${TOKENIZATION}_vae"

echo "================================================================================"
echo "VAE Training: $TOKENIZATION"
echo "================================================================================"
echo "Base directory: $BASE_DIR"
echo "Output log:     $OUTPUT_LOG"
echo "Error log:      $ERROR_LOG"
echo "Seed:           $SEED"
echo "Additional args: $@"
echo "================================================================================"

# Run training
python3 -m vae.train.train \
    --tokenization "$TOKENIZATION" \
    --seed "$SEED" \
    "$@" \
    > "$OUTPUT_LOG" 2> "$ERROR_LOG"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ Training completed successfully!"
    echo "================================================================================"
    echo "Output log: $OUTPUT_LOG"
    echo "Error log:  $ERROR_LOG"
    echo "Checkpoints: checkpoints/${TOKENIZATION}_vae/"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo "================================================================================"
    echo "Check error log: $ERROR_LOG"
    echo "================================================================================"
    exit $EXIT_CODE
fi
