#!/bin/bash

#SBATCH --job-name=python_train_200 # Job name
#SBATCH --output=python_train_200-%j.out      # Output file
#SBATCH --error=python_train_200-%j.err       # Error file
#SBATCH --nodes=2                      # Number of nodes       
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx3090:1


#SBATCH --time=1-00:00:00                  # Maximum execution time (1 hour)
srun python3 /cluster/home/ooikonomou/LoDE/src/Discovery/PDEs/train_pde.py
# srun  python3 make_expressions_pde_dif_lengths.py

#SBATCH --gpus-per-node=1   

# srun python3 predict.py


# module load python/3.8.5                 # Load the Python module (adjust version if needed)
