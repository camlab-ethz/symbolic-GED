"""Perturbation stability and prior sampling analysis.

This module provides functions for:
- Testing VAE robustness to latent perturbations
- Sampling from the prior z ~ N(0, I)
- Evaluating generated PDE quality
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from collections import Counter

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.physics import (
    classify_pde_type, classify_spatial_dim, classify_linearity,
    classify_order, is_valid_pde
)
from analysis.interpolation_analysis import load_vae_model, decode_latent


def perturb_latents(
    latents: np.ndarray,
    sigma: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """Add Gaussian noise to latent vectors.

    Args:
        latents: (N, z_dim) latent vectors
        sigma: Standard deviation of noise
        seed: Random seed

    Returns:
        Perturbed latent vectors
    """
    np.random.seed(seed)
    noise = np.random.normal(0, sigma, latents.shape).astype(np.float32)
    return latents + noise


def analyze_decoded_pdes(pdes: List[str]) -> Dict:
    """Analyze a list of decoded PDEs for physics properties.

    Returns:
        Dictionary with counts and rates for each property
    """
    n_total = len(pdes)

    valid = []
    dims = []
    types = []
    linearities = []
    orders = []

    for pde in pdes:
        if pde and pde != '[INVALID]':
            v = is_valid_pde(pde)
            valid.append(v)
            if v:
                dims.append(classify_spatial_dim(pde))
                types.append(classify_pde_type(pde))
                linearities.append(classify_linearity(pde))
                orders.append(classify_order(pde))
            else:
                dims.append(-1)
                types.append('invalid')
                linearities.append('invalid')
                orders.append(-1)
        else:
            valid.append(False)
            dims.append(-1)
            types.append('invalid')
            linearities.append('invalid')
            orders.append(-1)

    return {
        'n_total': n_total,
        'n_valid': sum(valid),
        'validity_rate': sum(valid) / n_total if n_total > 0 else 0,
        'dims': dims,
        'types': types,
        'linearities': linearities,
        'orders': orders,
        'dim_counts': dict(Counter(dims)),
        'type_counts': dict(Counter(types)),
        'linearity_counts': dict(Counter(linearities)),
        'order_counts': dict(Counter(orders)),
    }


def run_perturbation_analysis(
    model,
    latents: np.ndarray,
    original_families: np.ndarray,
    tokenization: str,
    sigma: float = 0.1,
    n_samples: int = 1000,
    seed: int = 42,
    use_constrained: bool = False,
    device: str = 'cuda'
) -> Dict:
    """Run perturbation stability analysis.

    Args:
        model: VAE model
        latents: (N, z_dim) original latent vectors
        original_families: (N,) family labels for originals
        tokenization: 'grammar' or 'token'
        sigma: Noise standard deviation
        n_samples: Number of samples to analyze
        seed: Random seed
        use_constrained: Use constrained decoding
        device: Device

    Returns:
        Perturbation analysis results
    """
    np.random.seed(seed)

    # Sample indices
    if n_samples < len(latents):
        indices = np.random.choice(len(latents), n_samples, replace=False)
    else:
        indices = np.arange(len(latents))

    sampled_latents = latents[indices]
    sampled_families = original_families[indices]

    # Perturb
    perturbed = perturb_latents(sampled_latents, sigma, seed)

    # Decode originals and perturbed
    print(f"  Decoding {len(indices)} original and perturbed latents...")
    original_pdes = []
    perturbed_pdes = []

    for i, (orig_z, pert_z) in enumerate(zip(sampled_latents, perturbed)):
        if i % 200 == 0:
            print(f"    {i}/{len(indices)}...")

        orig_z_tensor = torch.from_numpy(orig_z[np.newaxis, :]).float().to(device)
        pert_z_tensor = torch.from_numpy(pert_z[np.newaxis, :]).float().to(device)

        orig_pde = decode_latent(model, orig_z_tensor, tokenization, use_constrained)
        pert_pde = decode_latent(model, pert_z_tensor, tokenization, use_constrained)

        original_pdes.append(orig_pde if orig_pde else '[INVALID]')
        perturbed_pdes.append(pert_pde if pert_pde else '[INVALID]')

    # Analyze both
    original_analysis = analyze_decoded_pdes(original_pdes)
    perturbed_analysis = analyze_decoded_pdes(perturbed_pdes)

    # Compare original vs perturbed
    n_valid_orig = original_analysis['n_valid']
    n_valid_pert = perturbed_analysis['n_valid']

    # Count preserved properties (among valid pairs)
    dim_preserved = 0
    type_preserved = 0
    linearity_preserved = 0
    order_preserved = 0
    n_both_valid = 0

    for i in range(len(indices)):
        orig_valid = original_analysis['dims'][i] != -1
        pert_valid = perturbed_analysis['dims'][i] != -1

        if orig_valid and pert_valid:
            n_both_valid += 1
            if original_analysis['dims'][i] == perturbed_analysis['dims'][i]:
                dim_preserved += 1
            if original_analysis['types'][i] == perturbed_analysis['types'][i]:
                type_preserved += 1
            if original_analysis['linearities'][i] == perturbed_analysis['linearities'][i]:
                linearity_preserved += 1
            if original_analysis['orders'][i] == perturbed_analysis['orders'][i]:
                order_preserved += 1

    results = {
        'sigma': sigma,
        'n_samples': len(indices),
        'original': {
            'validity_rate': original_analysis['validity_rate'],
            'dim_counts': original_analysis['dim_counts'],
            'type_counts': original_analysis['type_counts'],
        },
        'perturbed': {
            'validity_rate': perturbed_analysis['validity_rate'],
            'dim_counts': perturbed_analysis['dim_counts'],
            'type_counts': perturbed_analysis['type_counts'],
        },
        'preservation': {
            'n_both_valid': n_both_valid,
            'dim_preserved': dim_preserved / n_both_valid if n_both_valid > 0 else 0,
            'type_preserved': type_preserved / n_both_valid if n_both_valid > 0 else 0,
            'linearity_preserved': linearity_preserved / n_both_valid if n_both_valid > 0 else 0,
            'order_preserved': order_preserved / n_both_valid if n_both_valid > 0 else 0,
        },
        'validity_degradation': original_analysis['validity_rate'] - perturbed_analysis['validity_rate'],
    }

    return results


def run_prior_sampling(
    model,
    tokenization: str,
    z_dim: int = 26,
    n_samples: int = 500,
    seed: int = 42,
    use_constrained: bool = False,
    device: str = 'cuda',
    training_pdes: List[str] = None
) -> Dict:
    """Sample from prior z ~ N(0, I) and analyze generated PDEs.

    Args:
        model: VAE model
        tokenization: 'grammar' or 'token'
        z_dim: Latent dimension
        n_samples: Number of samples
        seed: Random seed
        use_constrained: Use constrained decoding
        device: Device
        training_pdes: List of training PDEs for novelty check

    Returns:
        Prior sampling analysis results
    """
    np.random.seed(seed)

    # Sample from prior
    z_samples = np.random.randn(n_samples, z_dim).astype(np.float32)

    print(f"  Decoding {n_samples} prior samples...")
    sampled_pdes = []

    for i, z in enumerate(z_samples):
        if i % 100 == 0:
            print(f"    {i}/{n_samples}...")

        z_tensor = torch.from_numpy(z[np.newaxis, :]).float().to(device)
        pde = decode_latent(model, z_tensor, tokenization, use_constrained)
        sampled_pdes.append(pde if pde else '[INVALID]')

    # Analyze
    analysis = analyze_decoded_pdes(sampled_pdes)

    # Check novelty
    novelty_rate = 0
    if training_pdes:
        training_set = set(training_pdes)
        novel_count = sum(1 for pde in sampled_pdes if pde not in training_set and pde != '[INVALID]')
        novelty_rate = novel_count / analysis['n_valid'] if analysis['n_valid'] > 0 else 0

    results = {
        'n_samples': n_samples,
        'validity_rate': analysis['validity_rate'],
        'novelty_rate': novelty_rate,
        'dim_distribution': analysis['dim_counts'],
        'type_distribution': analysis['type_counts'],
        'linearity_distribution': analysis['linearity_counts'],
        'order_distribution': analysis['order_counts'],
        'sample_pdes': sampled_pdes[:20],  # Store first 20 samples
    }

    return results


def compare_perturbation_results(
    grammar_results: Dict,
    token_results: Dict
) -> Dict:
    """Compare perturbation results between tokenizations."""
    comparison = {
        'grammar': grammar_results,
        'token': token_results,
        'summary': {}
    }

    # Compare preservation rates
    for metric in ['dim_preserved', 'type_preserved', 'linearity_preserved', 'order_preserved']:
        g_val = grammar_results['preservation'].get(metric, 0)
        t_val = token_results['preservation'].get(metric, 0)
        comparison['summary'][metric] = {
            'grammar': g_val,
            'token': t_val,
            'winner': 'Grammar' if g_val > t_val else 'Token'
        }

    # Compare validity rates
    g_valid = grammar_results['perturbed']['validity_rate']
    t_valid = token_results['perturbed']['validity_rate']
    comparison['summary']['validity_rate'] = {
        'grammar': g_valid,
        'token': t_valid,
        'winner': 'Grammar' if g_valid > t_valid else 'Token'
    }

    return comparison


def compare_sampling_results(
    grammar_results: Dict,
    token_results: Dict
) -> Dict:
    """Compare prior sampling results between tokenizations."""
    comparison = {
        'grammar': grammar_results,
        'token': token_results,
        'summary': {}
    }

    # Compare validity
    comparison['summary']['validity_rate'] = {
        'grammar': grammar_results['validity_rate'],
        'token': token_results['validity_rate'],
        'winner': 'Grammar' if grammar_results['validity_rate'] > token_results['validity_rate'] else 'Token'
    }

    # Compare novelty
    comparison['summary']['novelty_rate'] = {
        'grammar': grammar_results['novelty_rate'],
        'token': token_results['novelty_rate'],
        'winner': 'Grammar' if grammar_results['novelty_rate'] > token_results['novelty_rate'] else 'Token'
    }

    return comparison


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Run perturbation and sampling analysis')
    parser.add_argument('--grammar-ckpt', type=str, required=True)
    parser.add_argument('--token-ckpt', type=str, required=True)
    parser.add_argument('--latent-file', type=str, default='visualization_maps/latent_data.npz')
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--output', type=str, default='experiments/reports/perturbation_results.json')

    args = parser.parse_args()

    # Load latents
    data = np.load(args.latent_file, allow_pickle=True)
    grammar_latents = data['grammar_latents']
    token_latents = data['token_latents']
    families = data['families']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    print("Loading Grammar VAE...")
    grammar_model, grammar_hparams = load_vae_model(args.grammar_ckpt, device)
    z_dim = grammar_hparams.get('z_dim', 26)

    print("Loading Token VAE...")
    token_model, token_hparams = load_vae_model(args.token_ckpt, device)

    # Run perturbation analysis
    print(f"\nRunning Grammar perturbation analysis (σ={args.sigma})...")
    grammar_pert = run_perturbation_analysis(
        grammar_model, grammar_latents, families, 'grammar',
        sigma=args.sigma, n_samples=args.n_samples,
        use_constrained=True, device=device
    )

    print(f"\nRunning Token perturbation analysis (σ={args.sigma})...")
    token_pert = run_perturbation_analysis(
        token_model, token_latents, families, 'token',
        sigma=args.sigma, n_samples=args.n_samples,
        use_constrained=False, device=device
    )

    # Run prior sampling
    print(f"\nRunning Grammar prior sampling...")
    grammar_sample = run_prior_sampling(
        grammar_model, 'grammar', z_dim=z_dim,
        n_samples=args.n_samples // 2,
        use_constrained=True, device=device
    )

    print(f"\nRunning Token prior sampling...")
    token_sample = run_prior_sampling(
        token_model, 'token', z_dim=z_dim,
        n_samples=args.n_samples // 2,
        use_constrained=False, device=device
    )

    # Compare
    pert_comparison = compare_perturbation_results(grammar_pert, token_pert)
    sample_comparison = compare_sampling_results(grammar_sample, token_sample)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(args.output, 'w') as f:
        json.dump(convert({
            'perturbation': pert_comparison,
            'sampling': sample_comparison,
        }), f, indent=2)

    print(f"\nResults saved to {args.output}")

    print(f"\nPerturbation Summary (σ={args.sigma}):")
    print(f"  Grammar validity: {grammar_pert['perturbed']['validity_rate']:.1%}")
    print(f"  Token validity: {token_pert['perturbed']['validity_rate']:.1%}")
    print(f"  Grammar dim preserved: {grammar_pert['preservation']['dim_preserved']:.1%}")
    print(f"  Token dim preserved: {token_pert['preservation']['dim_preserved']:.1%}")

    print(f"\nPrior Sampling Summary:")
    print(f"  Grammar validity: {grammar_sample['validity_rate']:.1%}")
    print(f"  Token validity: {token_sample['validity_rate']:.1%}")
