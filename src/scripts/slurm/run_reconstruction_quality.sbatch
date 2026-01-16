#!/bin/bash
#SBATCH --job-name=recon_quality
#SBATCH --output=logs/reconstruction_quality_%j.out
#SBATCH --error=logs/reconstruction_quality_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

# Reconstruction Quality Analysis
# ================================
# Analyzes what happens when models reconstruct PDEs:
# - Exact match: decoded = input
# - Structure preserved: same family, different coefficients
# - Family changed: valid but different family
# - Invalid: syntax error or corruption

echo "=============================================="
echo "Reconstruction Quality Analysis"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

# Setup
cd /cluster/work/math/ooikonomou/symbolic-GED/src
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Create logs directory
mkdir -p logs

# Run analysis on both val and test splits
echo "Running reconstruction quality analysis..."
python -m analysis.reconstruction_quality \
    --config configs/paths_48000_fixed.yaml \
    --splits val,test \
    --outdir analysis_results/reconstruction_quality_48000 \
    --batch_size 256 \
    --device cuda

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="

# Display summary
echo ""
echo "Summary:"
if [ -f analysis_results/reconstruction_quality_48000/summary.csv ]; then
    cat analysis_results/reconstruction_quality_48000/summary.csv
fi
