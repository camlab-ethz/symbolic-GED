"""Small demo: show masked-softmax decoding using grammar masks.

This script demonstrates applying per-step masks produced by
`pde_grammar.build_masks_from_production_sequence` to logits before softmax.
It also shows greedy decoding constrained by masks.
"""

from __future__ import annotations
import torch
import numpy as np

from src.pde import grammar as pde_grammar
from src import onehot


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Apply a mask (1=allowed, 0=forbidden) to logits and return softmax probs."""
    neg_inf = -1e9
    logits_masked = logits.masked_fill(mask == 0, neg_inf)
    return torch.nn.functional.softmax(logits_masked, dim=dim)


def greedy_decode_with_masks(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Greedy decode per-timestep using masks.

    Args:
        logits: (T, P) tensor of logits for each timestep
        masks:  (T, P) mask tensor (1 allowed, 0 forbidden)

    Returns:
        Tensor of shape (T,) with selected production ids
    """
    T, P = logits.shape
    outs = []
    for t in range(T):
        logit = logits[t : t + 1]  # (1, P)
        mask = masks[t : t + 1]
        probs = masked_softmax(logit, mask, dim=-1)
        choice = torch.argmax(probs, dim=-1).item()
        outs.append(choice)
    return torch.tensor(outs, dtype=torch.int64)


def demo():
    s = "dt(u) - 1.935*dxx(u) = 0"
    seq = pde_grammar.parse_to_productions(s)
    masks = pde_grammar.build_masks_from_production_sequence(seq)
    P = pde_grammar.PROD_COUNT
    T = len(seq)

    # Build logits that strongly prefer the true production at each step
    logits = torch.randn((T, P), dtype=torch.float32) * 0.1
    for t, pid in enumerate(seq):
        logits[t, pid] = 5.0  # make true production highly likely

    masks_t = torch.tensor(np.array(masks, dtype=np.float32))

    probs = masked_softmax(logits, masks_t)
    decoded = greedy_decode_with_masks(logits, masks_t)

    print("Original production seq:", seq)
    print("Decoded seq          :", decoded.tolist())
    # compare
    ok = decoded.tolist() == seq
    print("Greedy decode matches original?", ok)


if __name__ == "__main__":
    demo()
