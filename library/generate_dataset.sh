#!/usr/bin/env bash
# All-in-one script for generating PDE datasets
# Can run locally or create/submit a Slurm job for ETH Euler

# ======= CONFIGURATION SETTINGS =======
# Dataset generation settings
OUTPUT_FILE="../datasets/generated_triplets_smaller.csv"    # Path to output CSV file
MODE="comprehensive"                                # "comprehensive" or "random"
NUM_PER_OPERATOR=10                              # Number of examples per operator
CONFIG_FILE="config_dataset.yaml"                  # Path to configuration YAML file
OPERATORS=""                                       # Space-separated list, e.g. "diffusion wave"
DIMENSIONS=""                                      # Space-separated list, e.g. "1 2"
SOLUTIONS=""                                       # Space-separated list, e.g. "sine_cosine exp_decay"

# Execution mode
RUN_MODE="slurm"                                   # "local" or "slurm"

# Slurm settings (only used if RUN_MODE="slurm")
SLURM_JOB_NAME="pde_dataset"                       # Job name
SLURM_CPUS=8                                       # Number of CPU cores
SLURM_MEM="16G"                                    # Memory request
SLURM_TIME="12:00:00"                              # Time limit (HH:MM:SS)
AUTO_SUBMIT=false                                  # Set to true to auto-submit the job
# ======= END OF CONFIGURATION =======

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Create datasets directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Build the base command
CMD="python $SCRIPT_DIR/generate_triplets.py --config $CONFIG_FILE --output $OUTPUT_FILE --mode $MODE --num-per-operator $NUM_PER_OPERATOR"

# Add filters if specified
if [[ -n "$OPERATORS" ]]; then
    CMD="$CMD --operators $OPERATORS"
fi

if [[ -n "$DIMENSIONS" ]]; then
    CMD="$CMD --dimensions $DIMENSIONS"
fi

if [[ -n "$SOLUTIONS" ]]; then
    CMD="$CMD --solutions $SOLUTIONS"
fi

# Execute based on mode
if [[ "$RUN_MODE" == "local" ]]; then
    # Run locally
    echo "Executing locally: $CMD"
    eval "$CMD"
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "Dataset generation completed successfully!"
        echo "Output file: $OUTPUT_FILE"
    else
        echo "Error: Dataset generation failed."
        exit 1
    fi
    
elif [[ "$RUN_MODE" == "slurm" ]]; then
    # Create a Slurm job script
    SLURM_SCRIPT="${SLURM_JOB_NAME}_job.sh"
    
    echo "Creating Slurm job script: $SLURM_SCRIPT"
    
    # Write the job script
    cat > "$SLURM_SCRIPT" << EOL
#!/bin/bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH --output=${SLURM_JOB_NAME}-%j.out
#SBATCH --error=${SLURM_JOB_NAME}-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${SLURM_CPUS}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --time=${SLURM_TIME}


# Print job information
echo "Running on host: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Using \$SLURM_CPUS_PER_TASK CPU cores"
echo "Started at \$(date)"
echo "Command: $CMD"

# Create output directory if needed
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Run the command
$CMD

# Check result and print completion info
if [ \$? -eq 0 ]; then
  echo "Dataset generation completed successfully!"
  echo "Output file: $OUTPUT_FILE"
  echo "Finished at \$(date)"
  exit 0
else
  echo "Error: Dataset generation failed."
  echo "Finished at \$(date)"
  exit 1
fi
EOL

    # Make the script executable
    chmod +x "$SLURM_SCRIPT"
    
    # Auto-submit or provide instructions
    if [[ "$AUTO_SUBMIT" == true ]]; then
        echo "Auto-submitting job to Slurm..."
        sbatch "$SLURM_SCRIPT"
        echo "Job submitted. Check status with 'squeue -u \$USER'"
    else
        echo "Slurm job script created. To submit the job, run:"
        echo "sbatch $SLURM_SCRIPT"
    fi
    
else
    echo "Error: Invalid RUN_MODE. Choose 'local' or 'slurm'."
    exit 1
fi