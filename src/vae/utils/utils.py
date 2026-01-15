"""Utility functions for VAE training.

This module provides loss functions and helper utilities:
- reparameterize: VAE reparameterization trick
- masked_cross_entropy: Cross-entropy loss with padding mask
- kl_divergence: KL divergence with optional free bits
"""

from typing import Optional

import torch
import torch.nn.functional as F


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick for VAE.

    Samples from N(mu, var) using the reparameterization trick:
    z = mu + std * epsilon, where epsilon ~ N(0, 1)

    Args:
        mu: Mean of the posterior (B, z_dim)
        logvar: Log variance of the posterior (B, z_dim)

    Returns:
        Sampled latent vector (B, z_dim)
    """
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cross-entropy loss with optional padding mask.

    Only computes loss over positions where valid_mask == 1 and targets >= 0.
    Automatically filters out padding tokens (targets == -1).

    Args:
        logits: Prediction logits (B, T, P)
        targets: Target indices (B, T), may contain -1 for padding
        valid_mask: Padding mask (B, T) with 1 for valid positions (optional)

    Returns:
        Average cross-entropy loss over valid positions

    Note:
        No label smoothing is applied as it's incompatible with -inf masked logits.
    """
    B, T, P = logits.shape
    logits_flat = logits.reshape(B * T, P)
    targets_flat = targets.reshape(B * T)

    # Convention assumed: padding targets are < 0 (e.g. -1)
    pad_target = -1

    # Per-token CE (ignore padding targets)
    ce = F.cross_entropy(
        logits_flat.float(),  # compute CE in fp32 for stability
        targets_flat,
        reduction="none",
        ignore_index=pad_target,
    )  # (B*T,)

    if valid_mask is not None:
        vm = valid_mask.reshape(B * T).bool()
        # Ensure we never count padding even if vm is accidentally True there
        vm = vm & (targets_flat != pad_target)
        denom = vm.sum()
        if denom.item() == 0:
            # Return a zero that is connected to the graph (no weird "free tensor" loss)
            return logits.sum() * 0.0
        ce = ce.masked_fill(~vm, 0.0)
        return ce.sum() / denom
    else:
        denom = (targets_flat != pad_target).sum()
        if denom.item() == 0:
            return logits.sum() * 0.0
        return ce.sum() / denom


def kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0
) -> torch.Tensor:
    """KL divergence between posterior N(mu, var) and prior N(0, 1).

    Computes:
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(var) - mu^2 - var)

    Args:
        mu: Posterior mean (B, z_dim)
        logvar: Posterior log variance (B, z_dim)
        free_bits: Minimum total KL (prevents posterior collapse)
                   Applied to total KL, not per-dimension

    Returns:
        KL divergence averaged over batch (after free_bits clamp)

    Note:
        Free bits is applied to total KL following Kingma et al. 2016.
        This allows individual dimensions to collapse as long as
        total information is preserved.
    """
    # KL per dimension: (B, z_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Sum over dimensions: (B,)
    kl_total = torch.sum(kl_per_dim, dim=1)

    # Apply free bits threshold
    if free_bits > 0:
        kl_total = torch.clamp(kl_total, min=free_bits)

    # Mean over batch
    return torch.mean(kl_total)


def kl_divergence_raw(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Compute raw KL divergence without free_bits clamp.
    
    Returns the total KL before any clamping.
    
    Args:
        mu: Posterior mean (B, z_dim)
        logvar: Posterior log variance (B, z_dim)
        
    Returns:
        Raw KL divergence averaged over batch (before free_bits clamp)
    """
    # KL per dimension: (B, z_dim)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Sum over dimensions: (B,)
    kl_total = torch.sum(kl_per_dim, dim=1)
    
    # Mean over batch
    return torch.mean(kl_total)
