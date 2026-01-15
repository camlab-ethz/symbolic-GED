#!/bin/bash
# Quick check of sequence accuracy for both trainings

GRAMMAR_LOG="logs/training/grammar_vae_53908580.out"
TOKEN_LOG="logs/training/token_vae_53908581.out"

echo "================================================================================"
echo "Sequence Accuracy Progress - $(date '+%H:%M:%S')"
echo "================================================================================"
echo ""

# Grammar VAE
if [ -f "$GRAMMAR_LOG" ]; then
    echo "Grammar VAE:"
    LAST_EPOCH=$(grep "^Epoch" "$GRAMMAR_LOG" | tail -1 | sed 's/Epoch \([0-9]*\):.*/\1/' 2>/dev/null || echo "N/A")
    VAL_SEQ=$(grep "val/seq_acc" "$GRAMMAR_LOG" | tail -1 | sed 's/.*val\/seq_acc[=:]\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "N/A")
    TRAIN_SEQ=$(grep "train/seq_acc" "$GRAMMAR_LOG" | tail -1 | sed 's/.*train\/seq_acc[=:]\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "N/A")
    echo "  Epoch: $LAST_EPOCH"
    echo "  Val Seq Acc:  $VAL_SEQ"
    echo "  Train Seq Acc: $TRAIN_SEQ"
else
    echo "Grammar VAE: Log file not ready"
fi

echo ""

# Token VAE
if [ -f "$TOKEN_LOG" ]; then
    echo "Token VAE:"
    LAST_EPOCH=$(grep "^Epoch" "$TOKEN_LOG" | tail -1 | sed 's/Epoch \([0-9]*\):.*/\1/' 2>/dev/null || echo "N/A")
    VAL_SEQ=$(grep "val/seq_acc" "$TOKEN_LOG" | tail -1 | sed 's/.*val\/seq_acc[=:]\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "N/A")
    TRAIN_SEQ=$(grep "train/seq_acc" "$TOKEN_LOG" | tail -1 | sed 's/.*train\/seq_acc[=:]\s*\([0-9.]*\).*/\1/' 2>/dev/null || echo "N/A")
    echo "  Epoch: $LAST_EPOCH"
    echo "  Val Seq Acc:  $VAL_SEQ"
    echo "  Train Seq Acc: $TRAIN_SEQ"
else
    echo "Token VAE: Log file not ready"
fi

echo ""
echo "================================================================================"
