#!/bin/bash
# Submit VAE scenarios with β annealing
# This allows high β models to learn reconstruction first, then gradually increase regularization

WORK_DIR="/cluster/work/math/ooikonomou/symbolic-GED/src"
cd "$WORK_DIR"

mkdir -p logs

echo "=============================================="
echo "Submitting VAE Scenarios with β Annealing"
echo "=============================================="
echo ""

# Scenarios with annealing (anneal over first N epochs)
# Format: beta, anneal_epochs, tokenization

# Scenario A: β=0.1 with 50 epoch annealing
for tok in token grammar; do
    job_name="vae_${tok}_b0.1_anneal50"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.1 --kl_anneal_epochs 50"
done

# Scenario B: β=0.5 with 100 epoch annealing
for tok in token grammar; do
    job_name="vae_${tok}_b0.5_anneal100"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.5 --kl_anneal_epochs 100"
done

# Scenario C: β=0.01 with cyclical annealing (cycle every 20 epochs)
for tok in token grammar; do
    job_name="vae_${tok}_b0.01_cyclic20"
    echo "Submitting: $job_name"
    sbatch --job-name="$job_name" \
           --output="$WORK_DIR/logs/${job_name}-%j.out" \
           --error="$WORK_DIR/logs/${job_name}-%j.err" \
           --nodes=1 --ntasks=1 --cpus-per-task=4 \
           --mem-per-cpu=8G --time=24:00:00 \
           --gpus=nvidia_geforce_rtx_4090:1 \
           --wrap="cd $WORK_DIR && python -m vae.train --tokenization $tok --beta 0.01 --cyclical_beta --cycle_epochs 20"
done

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
