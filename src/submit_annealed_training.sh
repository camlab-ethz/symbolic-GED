#!/bin/bash
# Submit VAE training with higher beta + annealing
# Running alongside beta=1e-4 fixed jobs for comparison

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

mkdir -p logs
mkdir -p checkpoints/grammar_vae_b0.01_ann50
mkdir -p checkpoints/token_vae_b0.01_ann50

echo "=============================================="
echo "Submitting Annealed VAE Training Jobs"
echo "=============================================="

# Grammar VAE: beta=0.01, 50 epoch annealing
echo "Submitting Grammar VAE (beta=0.01, anneal=50)..."
sbatch << 'GRAMMAR_EOF'
#!/bin/bash
#SBATCH --job-name=grammar_ann
#SBATCH --output=logs/grammar_b0.01_ann50_%j.out
#SBATCH --error=logs/grammar_b0.01_ann50_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /cluster/work/math/ooikonomou/symbolic-GED/src

echo "Starting Grammar VAE: beta=0.01, kl_anneal_epochs=50"
python -m vae.train --tokenization grammar --beta 0.01 --kl_anneal_epochs 50

echo "Training complete!"
GRAMMAR_EOF

# Token VAE: beta=0.01, 50 epoch annealing
echo "Submitting Token VAE (beta=0.01, anneal=50)..."
sbatch << 'TOKEN_EOF'
#!/bin/bash
#SBATCH --job-name=token_ann
#SBATCH --output=logs/token_b0.01_ann50_%j.out
#SBATCH --error=logs/token_b0.01_ann50_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /cluster/work/math/ooikonomou/symbolic-GED/src

echo "Starting Token VAE: beta=0.01, kl_anneal_epochs=50"
python -m vae.train --tokenization token --beta 0.01 --kl_anneal_epochs 50

echo "Training complete!"
TOKEN_EOF

echo ""
echo "=============================================="
echo "Annealed jobs submitted!"
echo "=============================================="
echo ""
echo "Now running:"
echo "  1. grammar_b1e4 / token_b1e4     : beta=0.0001 fixed"
echo "  2. grammar_ann / token_ann       : beta=0.01, 50 epoch annealing"
echo ""
echo "Monitor: squeue -u \$USER"
