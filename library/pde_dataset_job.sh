#!/bin/bash
#SBATCH --job-name=pde_dataset
#SBATCH --output=pde_dataset-%j.out
#SBATCH --error=pde_dataset-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00


# Print job information
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Using $SLURM_CPUS_PER_TASK CPU cores"
echo "Started at $(date)"
echo "Command: python /cluster/work/math/camlab-data/symbolic-GED/library/generate_triplets.py --config config_dataset.yaml --output ../datasets/generated_triplets.csv --mode comprehensive --num-per-operator 1"

# Create output directory if needed
mkdir -p "../datasets"

# Run the command
srun python /cluster/work/math/camlab-data/symbolic-GED/library/generate_triplets.py --config config_dataset.yaml --output ../datasets/test-fix-Derivative-1000.csv --mode comprehensive --num-per-operator 1000
# Check result and print completion info
if [ $? -eq 0 ]; then
  echo "Dataset generation completed successfully!"
  echo "Output file: ../datasets/generated_triplets.csv"
  echo "Finished at $(date)"
  exit 0
else
  echo "Error: Dataset generation failed."
  echo "Finished at $(date)"
  exit 1
fi
