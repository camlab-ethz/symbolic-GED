#!/bin/bash
#SBATCH --job-name=token_eval_b1e2
#SBATCH --output=logs/tokenization_eval_beta1e-2_%j.out
#SBATCH --error=logs/tokenization_eval_beta1e-2_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:12g

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /cluster/work/math/ooikonomou/symbolic-GED/src

python3 scripts/run_tokenization_eval.py \
    --grammar-ckpt checkpoints/grammar_vae/best-epoch=362-seqacc=val/seq_acc=0.2510.ckpt \
    --token-ckpt checkpoints/token_vae/best-epoch=463-val/seq_acc=0.0083.ckpt \
    --csv-metadata data/raw/pde_dataset_45672.csv \
    --split test \
    --outdir experiments/reports/tokenization_eval_test_beta1e-2 \
    --n_pairs 50 \
    --n_steps 11 \
    --sigma 0.1 \
    --n_samples 500 \
    --seed 42 \
    --device cuda

echo "Evaluation complete! Results in: experiments/reports/tokenization_eval_test_beta1e-2/"
