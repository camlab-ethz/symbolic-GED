#!/bin/bash
# Quick script to check evaluation job status

JOB_ID=${1:-53927218}

echo "=== Job Status ==="
squeue -u $USER -j $JOB_ID 2>/dev/null || echo "Job $JOB_ID not in queue (may have finished)"

echo ""
echo "=== Recent Log Output (last 40 lines) ==="
tail -40 logs/tokenization_eval_${JOB_ID}.out 2>/dev/null || echo "Log file not found"

echo ""
echo "=== Output Files Created ==="
ls -lh experiments/reports/tokenization_eval/*.json experiments/reports/tokenization_eval/REPORT.md 2>/dev/null | head -10 || echo "Output files not created yet"

echo ""
echo "=== To Monitor in Real-Time ==="
echo "tail -f logs/tokenization_eval_${JOB_ID}.out"
