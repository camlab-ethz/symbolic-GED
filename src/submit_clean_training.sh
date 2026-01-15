#!/bin/bash
# Submit all 8 VAE training jobs on clean data
# Training matrix:
# | Beta   | Annealing      | Grammar | Token |
# |--------|----------------|---------|-------|
# | 0.001  | Linear 50      | ✓       | ✓     |
# | 0.001  | Cyclical 20    | ✓       | ✓     |
# | 0.01   | Linear 50      | ✓       | ✓     |
# | 0.1    | Linear 50      | ✓       | ✓     |

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

mkdir -p logs
mkdir -p checkpoints/grammar_vae
mkdir -p checkpoints/token_vae

echo "=============================================="
echo "Submitting VAE Training Jobs (Clean Data)"
echo "=============================================="
echo ""

# Job 1-2: β=0.001 with linear 50 epoch annealing
for tok in grammar token; do
    job_name="vae_${tok}_b0.001_lin50"
    ckpt_dir="checkpoints/${tok}_vae_b0.001_lin50"
    mkdir -p "$ckpt_dir"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.001 --kl_anneal_epochs 50 --epochs 500"
done

# Job 3-4: β=0.001 with cyclical 20 epoch annealing
for tok in grammar token; do
    job_name="vae_${tok}_b0.001_cyc20"
    ckpt_dir="checkpoints/${tok}_vae_b0.001_cyc20"
    mkdir -p "$ckpt_dir"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.001 --cyclical_beta --cycle_epochs 20 --epochs 500"
done

# Job 5-6: β=0.01 with linear 50 epoch annealing
for tok in grammar token; do
    job_name="vae_${tok}_b0.01_lin50"
    ckpt_dir="checkpoints/${tok}_vae_b0.01_lin50"
    mkdir -p "$ckpt_dir"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.01 --kl_anneal_epochs 50 --epochs 500"
done

# Job 7-8: β=0.1 with linear 50 epoch annealing
for tok in grammar token; do
    job_name="vae_${tok}_b0.1_lin50"
    ckpt_dir="checkpoints/${tok}_vae_b0.1_lin50"
    mkdir -p "$ckpt_dir"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.1 --kl_anneal_epochs 50 --epochs 500"
done

echo ""
echo "=============================================="
echo "All 8 jobs submitted!"
echo "=============================================="
echo ""
echo "Training Matrix:"
echo "  - grammar_b0.001_lin50: β=0.001, linear annealing 50 epochs"
echo "  - token_b0.001_lin50:   β=0.001, linear annealing 50 epochs"
echo "  - grammar_b0.001_cyc20: β=0.001, cyclical annealing 20 epochs"
echo "  - token_b0.001_cyc20:   β=0.001, cyclical annealing 20 epochs"
echo "  - grammar_b0.01_lin50:  β=0.01, linear annealing 50 epochs"
echo "  - token_b0.01_lin50:    β=0.01, linear annealing 50 epochs"
echo "  - grammar_b0.1_lin50:   β=0.1, linear annealing 50 epochs"
echo "  - token_b0.1_lin50:     β=0.1, linear annealing 50 epochs"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs:   tail -f logs/vae_*.out"
