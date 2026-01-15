#!/bin/bash
# Submit VAE retraining with beta=1e-4

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

mkdir -p logs

echo "=============================================="
echo "Submitting VAE Training Jobs (beta=1e-4)"
echo "=============================================="

# Grammar VAE
echo "Submitting Grammar VAE..."
sbatch << 'GRAMMAR_EOF'
#!/bin/bash
#SBATCH --job-name=grammar_b1e4
#SBATCH --output=logs/grammar_b1e4_%j.out
#SBATCH --error=logs/grammar_b1e4_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

# Load modules
module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Navigate to directory
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Run training
echo "Starting Grammar VAE training with beta=0.0001..."
python -m vae.train --tokenization grammar --beta 0.0001

echo "Grammar VAE training complete!"
GRAMMAR_EOF

# Token VAE
echo "Submitting Token VAE..."
sbatch << 'TOKEN_EOF'
#!/bin/bash
#SBATCH --job-name=token_b1e4
#SBATCH --output=logs/token_b1e4_%j.out
#SBATCH --error=logs/token_b1e4_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

# Load modules
module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Navigate to directory
cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Run training
echo "Starting Token VAE training with beta=0.0001..."
python -m vae.train --tokenization token --beta 0.0001

echo "Token VAE training complete!"
TOKEN_EOF

echo ""
echo "=============================================="
echo "Both jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs:   tail -f logs/*b1e4*.out"
