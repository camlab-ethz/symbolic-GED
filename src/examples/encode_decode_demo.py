"""
Encode/Decode with trained Grammar-VAE model.

This script shows how to:
1. Encode PDE strings to latent vectors (z)
2. Decode latent vectors back to PDE strings
3. Sample from the latent space
4. Interpolate between PDEs in latent space
"""

import torch
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.grammar_vae.module import GrammarVAEModule
from src.pde import grammar as pde_grammar


def prod_ids_to_string(prod_ids):
    """Convert production ID sequence to PDE string via left-most derivation."""
    try:
        nonterminals = set(pde_grammar.LHS_TO_PRODS.keys())
        derivation = ["PDE"]

        for pid in prod_ids:
            if pid >= len(pde_grammar.Productions):
                break

            lhs, rhs = pde_grammar.Productions[pid]

            # Find leftmost nonterminal
            found_idx = None
            for i, sym in enumerate(derivation):
                if sym == lhs:
                    found_idx = i
                    break

            if found_idx is None:
                remaining_nts = [s for s in derivation if s in nonterminals]
                if not remaining_nts:
                    break
                return f"[MISMATCH: {lhs} not found]"

            # Expand nonterminal
            derivation = (
                derivation[:found_idx] + list(rhs) + derivation[found_idx + 1 :]
            )

        # Extract terminals
        result = []
        for sym in derivation:
            if not sym:  # epsilon
                continue
            elif sym == "NUMBER":
                result.append("N")
            elif sym not in nonterminals:
                result.append(sym)

        return "".join(result).strip()
    except Exception as e:
        return f"[ERROR: {e}]"


class GrammarVAEEncoder:
    """Wrapper for encoding/decoding with Grammar-VAE."""

    def __init__(self, checkpoint_path, device="auto"):
        """Load trained Grammar-VAE model."""
        print(f"Loading model from {checkpoint_path}...")
        self.model = GrammarVAEModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        print(f"Model loaded on {device}")
        print(f"  z_dim: {self.model.hparams.z_dim}")
        print(f"  max_length: {self.model.hparams.max_length}")
        print(f"  encoder_hidden: {self.model.hparams.encoder_hidden}")
        print(f"  decoder_hidden: {self.model.hparams.decoder_hidden}")

    def encode(self, prod_onehot, return_variance=False):
        """
        Encode production sequences to latent vectors.

        Args:
            prod_onehot: (N, T, P) tensor of one-hot encoded productions
            return_variance: If True, return (z, mu, logvar). If False, return just z.

        Returns:
            z: (N, z_dim) latent vectors (sampled from q(z|x))
            mu: (N, z_dim) mean of q(z|x) [if return_variance=True]
            logvar: (N, z_dim) log variance of q(z|x) [if return_variance=True]
        """
        with torch.no_grad():
            x = prod_onehot.to(self.device)
            mu, logvar = self.model.encoder(x)

            # Sample z from N(mu, exp(logvar))
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            if return_variance:
                return z.cpu(), mu.cpu(), logvar.cpu()
            return z.cpu()

    def decode(self, z, apply_grammar_mask=False, masks=None):
        """
        Decode latent vectors to production sequences.

        Args:
            z: (N, z_dim) latent vectors
            apply_grammar_mask: If True, apply grammar constraints (requires masks)
            masks: (N, T, P) grammar masks [optional, only if apply_grammar_mask=True]

        Returns:
            prod_ids: (N, T) tensor of predicted production IDs
            logits: (N, T, P) raw logits before argmax
        """
        with torch.no_grad():
            z = z.to(self.device)
            logits = self.model.decoder(z)  # (N, T, P)

            if apply_grammar_mask and masks is not None:
                # Apply grammar constraints
                masks = masks.to(self.device)
                logits_masked = logits.clone()
                valid_mask = (masks.sum(dim=-1) > 0).float()
                non_padding = valid_mask.unsqueeze(-1) > 0
                invalid_mask = (masks == 0) & non_padding
                logits_masked[invalid_mask] = float("-inf")
                prod_ids = logits_masked.argmax(dim=-1)
            else:
                prod_ids = logits.argmax(dim=-1)

            return prod_ids.cpu(), logits.cpu()

    def reconstruct(self, prod_onehot, apply_grammar_mask=False, masks=None):
        """
        Reconstruct production sequences (encode then decode).

        Args:
            prod_onehot: (N, T, P) one-hot encoded productions
            apply_grammar_mask: Apply grammar constraints
            masks: Grammar masks (optional)

        Returns:
            prod_ids: (N, T) reconstructed production IDs
            z: (N, z_dim) latent vectors
        """
        z = self.encode(prod_onehot)
        prod_ids, _ = self.decode(z, apply_grammar_mask=apply_grammar_mask, masks=masks)
        return prod_ids, z

    def sample(self, n_samples=1, apply_grammar_mask=False, masks=None):
        """
        Sample random PDEs from the prior p(z) = N(0, I).

        Args:
            n_samples: Number of samples to generate
            apply_grammar_mask: Apply grammar constraints
            masks: If apply_grammar_mask=True, provide masks for one sample and repeat

        Returns:
            prod_ids: (n_samples, T) production IDs
            z: (n_samples, z_dim) sampled latent vectors
        """
        z_dim = self.model.hparams.z_dim
        z = torch.randn(n_samples, z_dim)

        if apply_grammar_mask and masks is not None:
            # Repeat masks for all samples
            masks_repeated = (
                masks.repeat(n_samples, 1, 1) if len(masks.shape) == 3 else masks
            )
            prod_ids, _ = self.decode(z, apply_grammar_mask=True, masks=masks_repeated)
        else:
            prod_ids, _ = self.decode(z, apply_grammar_mask=False)

        return prod_ids, z

    def interpolate(self, z1, z2, steps=10, apply_grammar_mask=False, masks=None):
        """
        Interpolate between two latent vectors.

        Args:
            z1: (z_dim,) or (1, z_dim) first latent vector
            z2: (z_dim,) or (1, z_dim) second latent vector
            steps: Number of interpolation steps
            apply_grammar_mask: Apply grammar constraints
            masks: Grammar masks (optional)

        Returns:
            prod_ids: (steps, T) production IDs for interpolated points
            z_interp: (steps, z_dim) interpolated latent vectors
        """
        if len(z1.shape) == 1:
            z1 = z1.unsqueeze(0)
        if len(z2.shape) == 1:
            z2 = z2.unsqueeze(0)

        # Linear interpolation
        alphas = torch.linspace(0, 1, steps).unsqueeze(1)
        z_interp = (1 - alphas) * z1 + alphas * z2

        if apply_grammar_mask and masks is not None:
            masks_repeated = masks.repeat(steps, 1, 1)
            prod_ids, _ = self.decode(
                z_interp, apply_grammar_mask=True, masks=masks_repeated
            )
        else:
            prod_ids, _ = self.decode(z_interp, apply_grammar_mask=False)

        return prod_ids, z_interp


def demo_encode_decode(checkpoint_path, data_path, masks_path):
    """Demonstrate encoding/decoding functionality."""

    # Load encoder
    encoder = GrammarVAEEncoder(checkpoint_path)

    # Load data
    print("\nLoading data...")
    prod_onehot = torch.load(data_path)  # (N, T, P)
    masks = torch.load(masks_path)  # (N, T, P)
    targets = prod_onehot.argmax(dim=-1)  # (N, T)

    print(f"Data shape: {prod_onehot.shape}")

    # Example 1: Encode single PDE
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Encode/Decode Single PDE")
    print("=" * 80)

    idx = 0
    x = prod_onehot[idx : idx + 1]
    mask = masks[idx : idx + 1]
    target = targets[idx]

    original_str = prod_ids_to_string(target.tolist())
    print(f"Original PDE: {original_str}")

    # Encode to latent vector
    z, mu, logvar = encoder.encode(x, return_variance=True)
    print(f"\nLatent vector z: shape={z.shape}")
    print(f"  Mean (mu): {mu[0,:10].numpy()}")
    print(f"  Variance: {torch.exp(logvar[0,:10]).numpy()}")

    # Decode back
    prod_ids, logits = encoder.decode(z, apply_grammar_mask=True, masks=mask)
    reconstructed_str = prod_ids_to_string(prod_ids[0].tolist())
    print(f"\nReconstructed PDE: {reconstructed_str}")
    print(f"Match: {'✓' if reconstructed_str == original_str else '✗'}")

    # Example 2: Sample from prior
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Sample from Prior p(z) = N(0,I)")
    print("=" * 80)

    # Use first sample's mask as template
    sample_mask = masks[0:1]
    sampled_ids, sampled_z = encoder.sample(
        n_samples=5, apply_grammar_mask=True, masks=sample_mask
    )

    print("Generated PDEs:")
    for i in range(5):
        pde_str = prod_ids_to_string(sampled_ids[i].tolist())
        print(f"  {i+1}. {pde_str}")

    # Example 3: Interpolation
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Latent Space Interpolation")
    print("=" * 80)

    # Encode two different PDEs
    idx1, idx2 = 0, 10
    x1 = prod_onehot[idx1 : idx1 + 1]
    x2 = prod_onehot[idx2 : idx2 + 1]

    z1 = encoder.encode(x1)
    z2 = encoder.encode(x2)

    pde1 = prod_ids_to_string(targets[idx1].tolist())
    pde2 = prod_ids_to_string(targets[idx2].tolist())

    print(f"PDE 1: {pde1}")
    print(f"PDE 2: {pde2}")
    print("\nInterpolation (7 steps):")

    interp_mask = masks[0:1]  # Use template mask
    interp_ids, interp_z = encoder.interpolate(
        z1[0], z2[0], steps=7, apply_grammar_mask=True, masks=interp_mask
    )

    for i, ids in enumerate(interp_ids):
        pde_str = prod_ids_to_string(ids.tolist())
        alpha = i / 6.0
        print(f"  α={alpha:.2f}: {pde_str}")

    # Example 4: Latent space statistics
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Latent Space Statistics")
    print("=" * 80)

    # Encode all samples
    batch_size = 100
    all_z = []
    for i in range(0, len(prod_onehot), batch_size):
        batch = prod_onehot[i : i + batch_size]
        z_batch = encoder.encode(batch)
        all_z.append(z_batch)

    all_z = torch.cat(all_z, dim=0)

    print(f"Encoded {len(all_z)} PDEs")
    print(f"Latent space shape: {all_z.shape}")
    print(f"\nStatistics per dimension:")
    print(
        f"  Mean: {all_z.mean(dim=0).abs().mean():.4f} (should be ~0 if VAE is well-trained)"
    )
    print(
        f"  Std:  {all_z.std(dim=0).mean():.4f} (should be ~1 if matching N(0,I) prior)"
    )
    print(f"  Min:  {all_z.min(dim=0).values.min():.4f}")
    print(f"  Max:  {all_z.max(dim=0).values.max():.4f}")

    return encoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode/Decode with Grammar-VAE")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--prod", type=str, required=True, help="Path to production onehot data (.pt)"
    )
    parser.add_argument(
        "--masks", type=str, required=True, help="Path to masks data (.pt)"
    )

    args = parser.parse_args()

    demo_encode_decode(args.checkpoint, args.prod, args.masks)
