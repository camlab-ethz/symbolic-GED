#!/bin/bash

#SBATCH --job-name=python_predict_circular # Job name
#SBATCH --output=pythonp_predict_circular-%j.out      # Output file
#SBATCH --error=python_predict_circular-%j.err       # Error file
#SBATCH --nodes=1                     # Number of nodes       
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512G



#SBATCH --time=1-00:00:00                  # Maximum execution time (1 hour)
srun python3 /cluster/home/ooikonomou/LoDE/src/Discovery/PDEs/predict_pde.py
# srun  python3 make_expressions_pde_dif_lengths.py

#SBATCH --gpus-per-node=1   

# srun python3 predict.py


# module load python/3.8.5                 # Load the Python module (adjust version if needed)
