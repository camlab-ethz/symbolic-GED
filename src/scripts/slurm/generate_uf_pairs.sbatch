#!/bin/bash
#SBATCH --job-name=gen_uf
#SBATCH --array=0-47
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/gen_uf_%A_%a.out
#SBATCH --error=logs/gen_uf_%A_%a.err

# Generate manufactured (u, f=L(u)) pairs
# 48 tasks × 1000 operators = 48,000 operators
# Both tracks (A, B), k=8 pairs each

module load stack/2024-06 python/3.11.6

cd /cluster/work/math/ooikonomou/symbolic-GED/src

CHUNK_SIZE=1000
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

OUT_DIR="${SCRATCH}/symbolic-GED/datasets/manufactured/chunks"
mkdir -p $OUT_DIR
OUT_FILE="${OUT_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Task ${SLURM_ARRAY_TASK_ID}: Operators ${START}-${END}"
echo "Output: ${OUT_FILE}"

python -m datasets.manufactured.generate_chunk \
    --csv data/raw/pde_dataset_48000_fixed.csv \
    --start $START \
    --end $END \
    --k 8 \
    --track both \
    --seed 42 \
    --complexity 2 \
    --out $OUT_FILE

echo "Task ${SLURM_ARRAY_TASK_ID} complete"
