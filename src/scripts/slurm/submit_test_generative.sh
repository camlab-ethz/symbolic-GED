#!/bin/bash
# Submit generative abilities test job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIBGEN_DIR="$SCRIPT_DIR/.."

cd "$LIBGEN_DIR" || { echo "Error: Could not cd to $LIBGEN_DIR"; exit 1; }

# Create log directory
mkdir -p slurm_logs
mkdir -p generation_results

# Default parameters (can be overridden)
N_SAMPLES=${1:-1000}
SEED=${2:-42}

echo "Submitting generative abilities test job..."
echo "  Number of samples: $N_SAMPLES"
echo "  Random seed: $SEED"
echo ""

# Export parameters for sbatch script
export N_SAMPLES
export SEED

JOB_ID=$(sbatch "$SCRIPT_DIR/test_generative_abilities.sbatch" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Job submitted successfully!"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo ""
echo "Parameters:"
echo "  Number of samples: $N_SAMPLES"
echo "  Random seed: $SEED"
echo ""
echo "Monitor job:"
echo "  squeue -u \$USER -j $JOB_ID"
echo ""
echo "View logs:"
echo "  tail -f slurm_logs/test_generative_abilities_${JOB_ID}.out"
echo "  tail -f slurm_logs/test_generative_abilities_${JOB_ID}.err"
echo ""
echo "Results will be saved to:"
echo "  generation_results/generation_summary_n${N_SAMPLES}_seed${SEED}.txt"
echo "  generation_results/{model}_all_pdes_n${N_SAMPLES}_seed${SEED}.txt"
echo "  generation_results/{model}_valid_pdes_n${N_SAMPLES}_seed${SEED}.txt"
echo "=========================================="
