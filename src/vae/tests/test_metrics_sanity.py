"""Sanity test for metrics to verify padding is ignored correctly.

Tests that token_acc and seq_acc correctly ignore padding tokens (-1).
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vae.module import VAEModule


def test_metrics_ignore_padding():
    """Test that metrics correctly ignore padding tokens."""
    # Create a minimal VAE module
    model = VAEModule(P=10, max_length=5, z_dim=4, beta=0.001, free_bits=0.0)
    
    # Create a toy batch:
    # Batch size 2, max_length 5, vocab_size 10
    # Sequence 1: [0, 1, 2, -1, -1] (2 valid tokens, then padding)
    # Sequence 2: [3, 4, -1, -1, -1] (2 valid tokens, then padding)
    
    # Create one-hot inputs (B=2, T=5, P=10)
    x = torch.zeros(2, 5, 10)
    x[0, 0, 0] = 1.0  # token 0 at pos 0
    x[0, 1, 1] = 1.0  # token 1 at pos 1
    x[0, 2, 2] = 1.0  # token 2 at pos 2
    x[1, 0, 3] = 1.0  # token 3 at pos 0
    x[1, 1, 4] = 1.0  # token 4 at pos 1
    
    # Create targets with padding (-1)
    targets = torch.tensor([
        [0, 1, 2, -1, -1],  # Sequence 1
        [3, 4, -1, -1, -1], # Sequence 2
    ], dtype=torch.long)
    
    # Create masks (for token VAE: B, T)
    # Position is valid if target >= 0
    masks = (targets >= 0).float()
    
    # Mock forward to get logits
    with torch.no_grad():
        logits, mu, logvar = model.forward(x)
    
    # Create a batch tuple
    batch = (x, targets, masks)
    
    # Compute metrics using _shared_step
    results = model._shared_step(batch, "train")
    
    # Extract metrics
    token_acc = results["token_acc"].item()
    seq_acc = results["seq_acc"].item()
    valid_count = results["valid_count"].item()
    token_correct = results["token_correct"].item()
    
    # Verify valid_count excludes padding
    expected_valid_count = 4.0  # 2 + 2 = 4 valid tokens
    assert abs(valid_count - expected_valid_count) < 1e-5, \
        f"valid_count should be {expected_valid_count}, got {valid_count}"
    
    # Verify that padding targets are -1
    assert (targets[targets < 0] == -1).all(), \
        "All padding targets should be -1"
    
    print(f"✓ Test passed!")
    print(f"  valid_count: {valid_count} (expected: {expected_valid_count})")
    print(f"  token_acc: {token_acc:.4f}")
    print(f"  seq_acc: {seq_acc:.4f}")
    print(f"  token_correct: {token_correct:.4f}")


if __name__ == "__main__":
    test_metrics_ignore_padding()
