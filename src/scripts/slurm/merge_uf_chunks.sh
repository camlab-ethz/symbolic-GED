#!/bin/bash
# Merge all (u,f) chunks into final dataset

CHUNK_DIR="${SCRATCH}/symbolic-GED/datasets/manufactured/chunks"
OUT_DIR="${SCRATCH}/symbolic-GED/datasets/manufactured"
OUT_FILE="${OUT_DIR}/uf_pairs_48k.jsonl"

echo "Merging chunks..."
cat ${CHUNK_DIR}/chunk_*.jsonl > $OUT_FILE

N=$(wc -l < $OUT_FILE)
echo "Total records: $N"
echo "Output: $OUT_FILE"

# Quick validation
echo ""
echo "Sample record:"
head -1 $OUT_FILE | python -c "import json,sys; d=json.load(sys.stdin); print(f\"  Family: {d['family']}, Track: {d['track']}, k={d['k']}\")"
