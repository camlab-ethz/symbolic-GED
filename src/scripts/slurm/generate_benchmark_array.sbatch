#!/bin/bash
#SBATCH --job-name=gen_benchmark
#SBATCH --array=0-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err

# Generate manufactured solutions benchmark - SLURM array job
# Each array task processes 1000 operators
# Total: 48 tasks × 1000 = 48,000 operators

module load stack/2024-06 python/3.11.6

cd /cluster/work/math/ooikonomou/symbolic-GED/src

# Calculate operator range for this task
START=$((SLURM_ARRAY_TASK_ID * 1000))
END=$((START + 1000))

# Create temp CSV for this chunk
CHUNK_CSV="/tmp/chunk_${SLURM_ARRAY_TASK_ID}.csv"
head -1 data/raw/pde_dataset_48000_fixed.csv > $CHUNK_CSV
sed -n "$((START+2)),$((END+1))p" data/raw/pde_dataset_48000_fixed.csv >> $CHUNK_CSV

# Output file for this chunk
OUT_DIR="data/manufactured/chunks"
mkdir -p $OUT_DIR
OUT_FILE="${OUT_DIR}/benchmark_chunk_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Task ${SLURM_ARRAY_TASK_ID}: Processing operators ${START}-${END}"
echo "Output: ${OUT_FILE}"

python -m datasets.manufactured.generate_parallel \
    --csv $CHUNK_CSV \
    --out $OUT_FILE \
    --k 8 \
    --track both \
    --seed $((42 + SLURM_ARRAY_TASK_ID * 1000)) \
    --complexity_level 2 \
    --workers 8

# Cleanup
rm -f $CHUNK_CSV

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
