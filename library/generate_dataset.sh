#!/usr/bin/env bash
# Configurable script for generating PDE datasets

# ======= CONFIGURATION SETTINGS =======
# Edit these values to change the dataset generation

# Basic settings
OUTPUT_FILE="../datasets/generated_triplets.csv"  # Path to output CSV file
MODE="comprehensive"                                  # "comprehensive" or "random"
NUM_PER_OPERATOR=1                             # Number of examples per operator (for random mode)
CONFIG_FILE="config_dataset.yaml"              # Path to configuration YAML file

# Optional filters - leave empty for no filtering
OPERATORS=""                     # Space-separated list, e.g. "diffusion wave"
DIMENSIONS=""                    # Space-separated list, e.g. "1 2"
SOLUTIONS=""                     # Space-separated list, e.g. "sine_cosine exp_decay"

# ======= END OF CONFIGURATION ======= 

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create datasets directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Build the command
CMD="python $SCRIPT_DIR/generate_triplets.py --config $CONFIG_FILE --output $OUTPUT_FILE --mode $MODE"

# Add num-per-operator for random mode
if [[ "$MODE" == "random" ]]; then
    CMD="$CMD --num-per-operator $NUM_PER_OPERATOR"
fi

# Add operators if specified
if [[ -n "$OPERATORS" ]]; then
    CMD="$CMD --operators $OPERATORS"
fi

# Add dimensions if specified
if [[ -n "$DIMENSIONS" ]]; then
    CMD="$CMD --dimensions $DIMENSIONS"
fi

# Add solution types if specified
if [[ -n "$SOLUTIONS" ]]; then
    CMD="$CMD --solutions $SOLUTIONS"
fi

# Print the command
echo "Executing: $CMD"

# Run the command
eval "$CMD"

# Check result
if [ $? -eq 0 ]; then
    echo "Dataset generation completed successfully!"
    echo "Output file: $OUTPUT_FILE"
else
    echo "Error: Dataset generation failed."
    exit 1
fi