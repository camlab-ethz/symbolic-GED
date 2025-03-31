#!/bin/bash
#SBATCH --job-name=python_PDE5      # Job name
#SBATCH --output=python_PDE5-%j.out   # Output file
#SBATCH --error=python_PDE5-%j.err    # Error file
#SBATCH --nodes=1                   # Number of nodes       
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00             # Maximum execution time (48 hours)

# Change to the directory where this script (and train.py) is located
cd "$(dirname "$SLURM_SUBMIT_DIR/model/")"

# Run the training script using srun
srun python3 train.py -A es_chatzi