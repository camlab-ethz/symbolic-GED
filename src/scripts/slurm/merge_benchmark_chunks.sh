#!/bin/bash
# Merge all benchmark chunks into a single file

cd /cluster/work/math/ooikonomou/symbolic-GED/src

OUT_FILE="data/manufactured/benchmark_48k_full.jsonl"
CHUNK_DIR="data/manufactured/chunks"

echo "Merging chunks from ${CHUNK_DIR}..."

# Concatenate all chunks
cat ${CHUNK_DIR}/benchmark_chunk_*.jsonl > $OUT_FILE

# Count records
N_RECORDS=$(wc -l < $OUT_FILE)
echo "Merged ${N_RECORDS} records into ${OUT_FILE}"

# Verify
echo "Sample record:"
head -1 $OUT_FILE | python -c "import json,sys; d=json.load(sys.stdin); print(f\"  Family: {d['family']}, Track: {d['track']}, k={d['k']}\")"
