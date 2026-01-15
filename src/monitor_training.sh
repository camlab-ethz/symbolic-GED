#!/bin/bash
# Monitor both Grammar and Token VAE training jobs
# Shows sequence accuracy progress for both trainings

GRAMMAR_LOG=$1
TOKEN_LOG=$2

if [ -z "$GRAMMAR_LOG" ] || [ -z "$TOKEN_LOG" ]; then
    echo "Usage: $0 <grammar_log> <token_log>"
    echo "Example: $0 logs/training/grammar_vae_12345.out logs/training/token_vae_67890.out"
    exit 1
fi

echo "================================================================================"
echo "Monitoring VAE Training - Sequence Accuracy"
echo "================================================================================"
echo ""
echo "Grammar VAE: $GRAMMAR_LOG"
echo "Token VAE:   $TOKEN_LOG"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "================================================================================"
echo ""

# Function to extract latest sequence accuracy from a log file
extract_seq_acc() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        # Get the last validation sequence accuracy line
        grep -E "val/seq_acc|Epoch.*val/seq_acc" "$log_file" | tail -1 | sed 's/.*val\/seq_acc[=:]\s*\([0-9.]*\).*/\1/' || echo "N/A"
    else
        echo "N/A (file not found)"
    fi
}

# Function to get current epoch
get_epoch() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        grep -E "^Epoch [0-9]+:" "$log_file" | tail -1 | sed 's/^Epoch \([0-9]*\):.*/\1/' || echo "N/A"
    else
        echo "N/A"
    fi
}

while true; do
    clear
    echo "================================================================================"
    echo "VAE Training Progress - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""
    
    GRAMMAR_EPOCH=$(get_epoch "$GRAMMAR_LOG")
    GRAMMAR_ACC=$(extract_seq_acc "$GRAMMAR_LOG")
    TOKEN_EPOCH=$(get_epoch "$TOKEN_LOG")
    TOKEN_ACC=$(extract_seq_acc "$TOKEN_LOG")
    
    printf "Grammar VAE:\n"
    printf "  Epoch: %s\n" "$GRAMMAR_EPOCH"
    printf "  Val Seq Acc: %s\n" "$GRAMMAR_ACC"
    printf "\n"
    printf "Token VAE:\n"
    printf "  Epoch: %s\n" "$TOKEN_EPOCH"
    printf "  Val Seq Acc: %s\n" "$TOKEN_ACC"
    echo ""
    echo "================================================================================"
    echo "Last 3 lines from each log:"
    echo "================================================================================"
    echo "Grammar VAE:"
    tail -3 "$GRAMMAR_LOG" 2>/dev/null || echo "  (log file not ready yet)"
    echo ""
    echo "Token VAE:"
    tail -3 "$TOKEN_LOG" 2>/dev/null || echo "  (log file not ready yet)"
    echo ""
    
    sleep 10
done
