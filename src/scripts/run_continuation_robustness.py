#!/usr/bin/env python3
"""Run continuation and robustness analysis (Phase 2).

This script runs:
1. Interpolation analysis - measuring smoothness in latent traversals
2. Perturbation stability - robustness to latent noise
3. Prior sampling - quality of z ~ N(0,I) generation

Usage:
    python scripts/run_continuation_robustness.py

    # With custom parameters:
    python scripts/run_continuation_robustness.py \
        --grammar-ckpt checkpoints/grammar_vae/best.ckpt \
        --token-ckpt checkpoints/token_vae/best.ckpt \
        --n-pairs 100 \
        --sigma 0.1
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find checkpoint with highest seq_acc."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None

    best_ckpt = None
    best_acc = 0

    for ckpt in ckpt_path.rglob("*.ckpt"):
        name = str(ckpt)
        if "seq_acc=" in name:
            try:
                acc_str = name.split("seq_acc=")[-1].split(".ckpt")[0]
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = str(ckpt)
            except:
                pass

    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description='Run continuation and robustness analysis')

    parser.add_argument('--grammar-ckpt', type=str, default=None,
                        help='Grammar VAE checkpoint')
    parser.add_argument('--token-ckpt', type=str, default=None,
                        help='Token VAE checkpoint')
    parser.add_argument('--latent-file', type=str,
                        default='visualization_maps/latent_data.npz',
                        help='Path to latent file')
    parser.add_argument('--n-pairs', type=int, default=50,
                        help='Number of interpolation pairs')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples for perturbation/sampling')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Perturbation noise std')
    parser.add_argument('--output-dir', type=str, default='experiments/reports',
                        help='Output directory')
    parser.add_argument('--skip-interpolation', action='store_true',
                        help='Skip interpolation analysis')
    parser.add_argument('--skip-perturbation', action='store_true',
                        help='Skip perturbation analysis')
    parser.add_argument('--skip-sampling', action='store_true',
                        help='Skip prior sampling')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Find checkpoints
    grammar_ckpt = args.grammar_ckpt or find_best_checkpoint(base_dir / 'checkpoints/grammar_vae')
    token_ckpt = args.token_ckpt or find_best_checkpoint(base_dir / 'checkpoints/token_vae')

    if not grammar_ckpt or not token_ckpt:
        print("Error: Could not find checkpoints")
        return

    print(f"\nGrammar checkpoint: {grammar_ckpt}")
    print(f"Token checkpoint: {token_ckpt}")

    # Load latent data
    latent_path = base_dir / args.latent_file
    print(f"\nLoading latents from {latent_path}")
    data = np.load(str(latent_path), allow_pickle=True)

    grammar_latents = data['grammar_latents']
    token_latents = data['token_latents']
    families = data['families']

    print(f"  Grammar latents: {grammar_latents.shape}")
    print(f"  Token latents: {token_latents.shape}")
    print(f"  Families: {len(families)}")

    # Import analysis modules
    from analysis.interpolation_analysis import (
        load_vae_model, run_interpolation_suite, compare_tokenizations
    )
    from analysis.perturbation_analysis import (
        run_perturbation_analysis, run_prior_sampling,
        compare_perturbation_results, compare_sampling_results
    )
    from analysis.report_generator import (
        generate_continuation_report, save_report
    )

    results = {
        'interpolation': None,
        'perturbation': None,
        'sampling': None,
        'timestamp': datetime.now().isoformat()
    }

    # Load models once
    print("\nLoading Grammar VAE...")
    grammar_model, grammar_hparams = load_vae_model(grammar_ckpt, device)
    z_dim = grammar_hparams.get('z_dim', 26)

    print("Loading Token VAE...")
    token_model, token_hparams = load_vae_model(token_ckpt, device)

    # ============================================================
    # INTERPOLATION ANALYSIS
    # ============================================================
    if not args.skip_interpolation:
        print("\n" + "=" * 60)
        print("INTERPOLATION ANALYSIS")
        print("=" * 60)

        print(f"\nRunning Grammar interpolation ({args.n_pairs} pairs)...")
        grammar_interp = run_interpolation_suite(
            grammar_model, grammar_latents, families, 'grammar',
            n_pairs=args.n_pairs, use_constrained=True, device=device
        )

        print(f"\nRunning Token interpolation ({args.n_pairs} pairs)...")
        token_interp = run_interpolation_suite(
            token_model, token_latents, families, 'token',
            n_pairs=args.n_pairs, use_constrained=False, device=device
        )

        interp_comparison = compare_tokenizations(grammar_interp, token_interp)
        results['interpolation'] = interp_comparison

        print("\nInterpolation Summary:")
        print(f"  Grammar avg type changes: {grammar_interp['summary']['avg_type_changes']:.2f}")
        print(f"  Token avg type changes: {token_interp['summary']['avg_type_changes']:.2f}")
        print(f"  Grammar validity: {grammar_interp['summary']['validity_rate']:.1%}")
        print(f"  Token validity: {token_interp['summary']['validity_rate']:.1%}")

        for dim_key in ['1D→1D', '2D→2D', '3D→3D', 'Overall']:
            g_val = grammar_interp['summary']['dim_preservation'].get(dim_key, 0)
            t_val = token_interp['summary']['dim_preservation'].get(dim_key, 0)
            print(f"  {dim_key} dim preserved: Grammar {g_val:.1%}, Token {t_val:.1%}")

    # ============================================================
    # PERTURBATION ANALYSIS
    # ============================================================
    if not args.skip_perturbation:
        print("\n" + "=" * 60)
        print(f"PERTURBATION ANALYSIS (σ={args.sigma})")
        print("=" * 60)

        print(f"\nRunning Grammar perturbation ({args.n_samples} samples)...")
        grammar_pert = run_perturbation_analysis(
            grammar_model, grammar_latents, families, 'grammar',
            sigma=args.sigma, n_samples=args.n_samples,
            use_constrained=True, device=device
        )

        print(f"\nRunning Token perturbation ({args.n_samples} samples)...")
        token_pert = run_perturbation_analysis(
            token_model, token_latents, families, 'token',
            sigma=args.sigma, n_samples=args.n_samples,
            use_constrained=False, device=device
        )

        pert_comparison = compare_perturbation_results(grammar_pert, token_pert)
        results['perturbation'] = pert_comparison

        print("\nPerturbation Summary:")
        print(f"  Grammar validity after noise: {grammar_pert['perturbed']['validity_rate']:.1%}")
        print(f"  Token validity after noise: {token_pert['perturbed']['validity_rate']:.1%}")
        print(f"  Grammar dim preserved: {grammar_pert['preservation']['dim_preserved']:.1%}")
        print(f"  Token dim preserved: {token_pert['preservation']['dim_preserved']:.1%}")
        print(f"  Grammar type preserved: {grammar_pert['preservation']['type_preserved']:.1%}")
        print(f"  Token type preserved: {token_pert['preservation']['type_preserved']:.1%}")

    # ============================================================
    # PRIOR SAMPLING
    # ============================================================
    if not args.skip_sampling:
        print("\n" + "=" * 60)
        print("PRIOR SAMPLING z ~ N(0, I)")
        print("=" * 60)

        n_prior = args.n_samples // 2

        print(f"\nRunning Grammar prior sampling ({n_prior} samples)...")
        grammar_sample = run_prior_sampling(
            grammar_model, 'grammar', z_dim=z_dim,
            n_samples=n_prior, use_constrained=True, device=device
        )

        print(f"\nRunning Token prior sampling ({n_prior} samples)...")
        token_sample = run_prior_sampling(
            token_model, 'token', z_dim=z_dim,
            n_samples=n_prior, use_constrained=False, device=device
        )

        sample_comparison = compare_sampling_results(grammar_sample, token_sample)
        results['sampling'] = sample_comparison

        print("\nPrior Sampling Summary:")
        print(f"  Grammar validity: {grammar_sample['validity_rate']:.1%}")
        print(f"  Token validity: {token_sample['validity_rate']:.1%}")

    # ============================================================
    # GENERATE REPORT
    # ============================================================
    print("\n" + "=" * 60)
    print("GENERATING MARKDOWN REPORT")
    print("=" * 60)

    # Prepare report data
    interpolation_data = None
    perturbation_data = None
    sampling_data = None

    if results['interpolation']:
        interpolation_data = {
            'grammar': results['interpolation']['grammar'],
            'token': results['interpolation']['token'],
        }

    if results['perturbation']:
        g_pert = results['perturbation']['grammar']
        t_pert = results['perturbation']['token']
        perturbation_data = {
            'grammar': {
                'validity_rate': g_pert['perturbed']['validity_rate'],
                'dim_preserved': g_pert['preservation']['dim_preserved'],
                'type_preserved': g_pert['preservation']['type_preserved'],
                'linearity_preserved': g_pert['preservation']['linearity_preserved'],
            },
            'token': {
                'validity_rate': t_pert['perturbed']['validity_rate'],
                'dim_preserved': t_pert['preservation']['dim_preserved'],
                'type_preserved': t_pert['preservation']['type_preserved'],
                'linearity_preserved': t_pert['preservation']['linearity_preserved'],
            },
        }

    if results['sampling']:
        sampling_data = {
            'grammar': results['sampling']['grammar'],
            'token': results['sampling']['token'],
        }

    report = generate_continuation_report(
        interpolation_data,
        perturbation_data,
        sampling_data
    )

    # Save report
    report_path = output_dir / 'continuation_and_robustness_summary.md'
    save_report(report, str(report_path))

    # Save JSON results
    json_path = output_dir / 'continuation_results.json'

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

    with open(json_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved JSON results to {json_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nReport: {report_path}")
    print(f"JSON: {json_path}")


if __name__ == '__main__':
    main()
