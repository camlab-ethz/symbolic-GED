#!/usr/bin/env python3
"""Run interpolation experiments with rigorous PDEClassifier.

This script runs interpolation analysis using the robust rule-based
PDEClassifier instead of simple heuristics.

Usage:
    python scripts/run_interpolation_rigorous.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Setup paths
SCRIPT_DIR = Path(__file__).parent
LIBGEN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LIBGEN_DIR))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import analysis modules
from analysis.interpolation_analysis import (
    load_vae_model, run_interpolation_suite, compare_tokenizations
)


def main():
    """Run interpolation experiments."""
    
    print("=" * 70)
    print("INTERPOLATION ANALYSIS WITH RIGOROUS PDECLASSIFIER")
    print("=" * 70)
    
    # Configuration
    output_dir = LIBGEN_DIR / "experiments/interpolation_rigorous"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_pairs = 100  # Number of random pairs
    n_steps = 11   # Interpolation steps
    seed = 42
    
    # Find best checkpoints
    grammar_ckpt_candidates = [
        LIBGEN_DIR / "checkpoints/grammar_vae/best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt",
        LIBGEN_DIR / "checkpoints/grammar_vae/best-epoch=155-seqacc=val/seq_acc=0.9905.ckpt",
        LIBGEN_DIR / "checkpoints/grammar_simple/last.ckpt",
    ]
    
    token_ckpt_candidates = [
        LIBGEN_DIR / "checkpoints/token_vae/best-epoch=314-seqacc=val/seq_acc=0.9841.ckpt",
        LIBGEN_DIR / "checkpoints/token_vae/best-epoch=303-seqacc=val/seq_acc=0.9589.ckpt",
    ]
    
    grammar_ckpt = None
    for ckpt in grammar_ckpt_candidates:
        if ckpt.exists():
            grammar_ckpt = ckpt
            break
    
    token_ckpt = None
    for ckpt in token_ckpt_candidates:
        if ckpt.exists():
            token_ckpt = ckpt
            break
    
    print(f"\nConfiguration:")
    print(f"  Grammar checkpoint: {grammar_ckpt if grammar_ckpt else 'NOT FOUND'}")
    print(f"  Token checkpoint: {token_ckpt if token_ckpt else 'NOT FOUND'}")
    print(f"  Output: {output_dir}")
    print(f"  Pairs: {n_pairs}, Steps: {n_steps}")
    
    if grammar_ckpt is None and token_ckpt is None:
        print("\nERROR: No checkpoints found!")
        return
    
    # Load latent vectors
    print("\n[1/4] Loading latent vectors...")
    
    latent_paths = [
        LIBGEN_DIR / "experiments/comparison/latents/grammar_latents.npz",
        LIBGEN_DIR / "visualization_maps/latent_data.npz",
    ]
    
    latent_file = None
    for p in latent_paths:
        if p.exists():
            latent_file = p
            break
    
    if latent_file is None:
        print("ERROR: No latent file found!")
        print("Expected one of:", latent_paths)
        return
    
    print(f"  Loading from: {latent_file}")
    
    data = np.load(latent_file, allow_pickle=True)
    
    # Handle different file formats
    if 'grammar_latents' in data:
        grammar_latents = data['grammar_latents']
        token_latents = data['token_latents']
        families = data['families']
    elif 'mu' in data:
        # Single model format - load both
        grammar_data = np.load(LIBGEN_DIR / "experiments/comparison/latents/grammar_latents.npz")
        token_data = np.load(LIBGEN_DIR / "experiments/comparison/latents/token_latents.npz")
        grammar_latents = grammar_data['mu']
        token_latents = token_data['mu']
        
        # Load families from dataset
        import pandas as pd
        df = pd.read_csv(LIBGEN_DIR / "pde_dataset_48444_clean.csv")
        families = df['family'].values
    else:
        print("ERROR: Unknown latent file format")
        return
    
    print(f"  Grammar latents: {grammar_latents.shape}")
    print(f"  Token latents: {token_latents.shape}")
    print(f"  Families: {len(families)} samples, {len(set(families))} unique")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Load models
    print("\n[2/4] Loading VAE models...")
    
    try:
        grammar_model, grammar_hparams = load_vae_model(str(grammar_ckpt), device)
        print(f"  Grammar VAE loaded: z_dim={grammar_hparams.get('z_dim', 26)}")
    except Exception as e:
        print(f"  ERROR loading Grammar VAE: {e}")
        grammar_model = None
    
    try:
        token_model, token_hparams = load_vae_model(str(token_ckpt), device)
        print(f"  Token VAE loaded: z_dim={token_hparams.get('z_dim', 26)}")
    except Exception as e:
        print(f"  ERROR loading Token VAE: {e}")
        token_model = None
    
    if grammar_model is None and token_model is None:
        print("ERROR: Could not load any models!")
        return
    
    results = {}
    
    # Run Grammar interpolation
    if grammar_model is not None:
        print("\n[3/4] Running Grammar VAE interpolation...")
        grammar_results = run_interpolation_suite(
            grammar_model, grammar_latents, families, 'grammar',
            n_pairs=n_pairs, n_steps=n_steps, seed=seed,
            use_constrained=True, device=device
        )
        results['grammar'] = grammar_results
    
    # Run Token interpolation
    if token_model is not None:
        print("\n[4/4] Running Token VAE interpolation...")
        token_results = run_interpolation_suite(
            token_model, token_latents, families, 'token',
            n_pairs=n_pairs, n_steps=n_steps, seed=seed,
            use_constrained=False, device=device
        )
        results['token'] = token_results
    
    # Compare if both available
    if 'grammar' in results and 'token' in results:
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)
        
        comparison = compare_tokenizations(results['grammar'], results['token'])
        results['comparison'] = comparison
        
        print("\n| Metric | Grammar | Token | Winner |")
        print("|--------|---------|-------|--------|")
        for metric, data in comparison['comparison'].items():
            g_val = data['grammar']
            t_val = data['token']
            winner = data['winner']
            
            if isinstance(g_val, float):
                print(f"| {metric} | {g_val:.3f} | {t_val:.3f} | {winner} |")
            else:
                print(f"| {metric} | {g_val} | {t_val} | {winner} |")
        
        print(f"\n**Overall Winner: {comparison['summary']['overall_winner']}**")
        print(f"  Grammar wins: {comparison['summary']['grammar_wins']}")
        print(f"  Token wins: {comparison['summary']['token_wins']}")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    def convert(obj):
        """Convert numpy types for JSON serialization."""
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
    
    # Save full results
    results_file = output_dir / "interpolation_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Full results: {results_file}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_pairs': n_pairs,
        'n_steps': n_steps,
        'seed': seed,
    }
    
    if 'grammar' in results:
        summary['grammar'] = results['grammar']['summary']
    if 'token' in results:
        summary['token'] = results['token']['summary']
    if 'comparison' in results:
        summary['comparison'] = results['comparison']['summary']
        summary['detailed_comparison'] = {
            k: {'grammar': v['grammar'], 'token': v['token'], 'winner': v['winner']}
            for k, v in results['comparison']['comparison'].items()
        }
    
    summary_file = output_dir / "interpolation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(convert(summary), f, indent=2)
    print(f"  Summary: {summary_file}")
    
    # Generate markdown report
    report = f"""# Interpolation Analysis Results (Rigorous Classifier)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Random pairs: {n_pairs}
- Interpolation steps: {n_steps}
- Random seed: {seed}

## Summary

"""
    
    if 'grammar' in results:
        g = results['grammar']['summary']
        report += f"""### Grammar VAE
- Validity rate: {g['validity_rate']:.1%}
- Avg type changes: {g['avg_type_changes']:.2f}
- Avg family changes: {g.get('avg_family_changes', 0):.2f}
- Avg classifier confidence: {g.get('avg_confidence', 0):.2f}

"""
    
    if 'token' in results:
        t = results['token']['summary']
        report += f"""### Token VAE
- Validity rate: {t['validity_rate']:.1%}
- Avg type changes: {t['avg_type_changes']:.2f}
- Avg family changes: {t.get('avg_family_changes', 0):.2f}
- Avg classifier confidence: {t.get('avg_confidence', 0):.2f}

"""
    
    if 'comparison' in results:
        c = results['comparison']
        report += f"""## Comparison

**Overall Winner: {c['summary']['overall_winner']}**
- Grammar wins: {c['summary']['grammar_wins']}
- Token wins: {c['summary']['token_wins']}

### Detailed Metrics

| Metric | Grammar | Token | Winner |
|--------|---------|-------|--------|
"""
        for metric, data in c['comparison'].items():
            g_val = data['grammar']
            t_val = data['token']
            winner = data['winner']
            if isinstance(g_val, float):
                report += f"| {metric} | {g_val:.3f} | {t_val:.3f} | {winner} |\n"
            else:
                report += f"| {metric} | {g_val} | {t_val} | {winner} |\n"
    
    report_file = output_dir / "INTERPOLATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"  Report: {report_file}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
