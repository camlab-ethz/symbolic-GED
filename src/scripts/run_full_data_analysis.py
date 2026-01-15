#!/usr/bin/env python3
"""Run full-data VAE representation analysis (Phase 1).

This script:
1. Loads or extracts latents from existing checkpoints
2. Computes clustering and classification metrics for both tokenizations
3. Generates a Markdown report comparing Grammar vs Token

Usage:
    python scripts/run_full_data_analysis.py

    # With custom checkpoint paths:
    python scripts/run_full_data_analysis.py \
        --grammar-ckpt checkpoints/grammar_vae/best.ckpt \
        --token-ckpt checkpoints/token_vae/best.ckpt

    # Use existing latent data:
    python scripts/run_full_data_analysis.py --use-existing-latents
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.run_analysis import load_latent_file, analyze_latents
from analysis.report_generator import generate_full_data_report, save_report


def find_best_checkpoint(ckpt_dir: str, pattern: str = "*.ckpt") -> str:
    """Find checkpoint with highest seq_acc in filename."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None

    best_ckpt = None
    best_acc = 0

    for ckpt in ckpt_path.rglob(pattern):
        # Parse accuracy from filename like "best-epoch=189-seqacc=val/seq_acc=0.9907.ckpt"
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


def load_existing_latents(latent_path: str) -> dict:
    """Load existing latent data and convert to expected format."""
    data = np.load(latent_path, allow_pickle=True)

    # Check what keys exist
    keys = list(data.keys())
    print(f"  Found keys: {keys}")

    result = {}

    # Handle different formats
    if 'grammar_latents' in data and 'token_latents' in data:
        # Combined file format
        result['grammar'] = {
            'latents': data['grammar_latents'],
            'family': data['families'] if 'families' in data else None,
            'dim': data['dimensions'] if 'dimensions' in data else None,
        }
        result['token'] = {
            'latents': data['token_latents'],
            'family': data['families'] if 'families' in data else None,
            'dim': data['dimensions'] if 'dimensions' in data else None,
        }
    elif 'latents' in data or 'mu' in data:
        # Single model format
        result['single'] = {
            'latents': data['latents'] if 'latents' in data else data['mu'],
            'family': data['family'] if 'family' in data else None,
            'dim': data['dim'] if 'dim' in data else None,
        }

    return result


def augment_with_physics_labels(data: dict, dataset_csv: str = 'pde_dataset_48444_clean.csv'):
    """Add physics labels from dataset CSV to latent data."""
    import pandas as pd

    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / dataset_csv

    if not csv_path.exists():
        print(f"  Warning: Dataset CSV not found at {csv_path}")
        return data

    df = pd.read_csv(csv_path)
    n = len(data['latents'])

    # Add labels from CSV
    if 'family' not in data or data['family'] is None:
        data['family'] = df['family'].values[:n] if 'family' in df.columns else None

    if 'dim' not in data or data['dim'] is None:
        data['dim'] = df['dim'].values[:n] if 'dim' in df.columns else None

    # Add physics labels from PDE_PHYSICS
    from analysis.physics import PDE_PHYSICS

    if data['family'] is not None:
        types = []
        nonlinear = []
        spatial_orders = []
        temporal_orders = []

        for fam in data['family']:
            if fam in PDE_PHYSICS:
                props = PDE_PHYSICS[fam]
                types.append(props.get('type', 'unknown'))
                nonlinear.append(props.get('linearity', 'linear') == 'nonlinear')
                spatial_orders.append(props.get('order', 2))
                temporal = props.get('temporal', 'first')
                temporal_orders.append(0 if temporal == 'none' else (1 if temporal == 'first' else 2))
            else:
                types.append('unknown')
                nonlinear.append(False)
                spatial_orders.append(2)
                temporal_orders.append(1)

        data['type'] = np.array(types)
        data['nonlinear'] = np.array(nonlinear)
        data['spatial_order'] = np.array(spatial_orders)
        data['temporal_order'] = np.array(temporal_orders)

    return data


def main():
    parser = argparse.ArgumentParser(description='Run full-data VAE representation analysis')

    parser.add_argument('--grammar-ckpt', type=str, default=None,
                        help='Grammar VAE checkpoint path')
    parser.add_argument('--token-ckpt', type=str, default=None,
                        help='Token VAE checkpoint path')
    parser.add_argument('--use-existing-latents', action='store_true',
                        help='Use existing latent file instead of extracting')
    parser.add_argument('--latent-file', type=str,
                        default='visualization_maps/latent_data.npz',
                        help='Path to existing latent file')
    parser.add_argument('--output-dir', type=str, default='experiments/reports',
                        help='Output directory for reports')
    parser.add_argument('--dataset', type=str, default='pde_dataset_48444_clean.csv',
                        help='Dataset CSV path')

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FULL-DATA VAE REPRESENTATION ANALYSIS")
    print("=" * 60)

    # Load latents
    if args.use_existing_latents:
        latent_path = base_dir / args.latent_file
        print(f"\nLoading existing latents from {latent_path}")
        combined_data = load_existing_latents(str(latent_path))

        if 'grammar' in combined_data and 'token' in combined_data:
            grammar_data = combined_data['grammar']
            token_data = combined_data['token']
        else:
            print("Error: Expected combined latent file with grammar_latents and token_latents")
            return

        # Augment with physics labels
        print("\nAugmenting with physics labels...")
        grammar_data = augment_with_physics_labels(grammar_data, args.dataset)
        token_data = augment_with_physics_labels(token_data, args.dataset)

    else:
        # Find or use specified checkpoints
        grammar_ckpt = args.grammar_ckpt
        token_ckpt = args.token_ckpt

        if grammar_ckpt is None:
            grammar_ckpt = find_best_checkpoint(base_dir / 'checkpoints/grammar_vae')
        if token_ckpt is None:
            token_ckpt = find_best_checkpoint(base_dir / 'checkpoints/token_vae')

        if not grammar_ckpt or not token_ckpt:
            print("Error: Could not find checkpoints. Use --grammar-ckpt and --token-ckpt")
            return

        print(f"\nGrammar checkpoint: {grammar_ckpt}")
        print(f"Token checkpoint: {token_ckpt}")

        # Extract latents (would call extract_latents.py here)
        print("\nNote: Latent extraction not implemented in this script.")
        print("Use --use-existing-latents or run training/extract_latents.py first")
        return

    # Run analysis
    print("\n" + "-" * 60)
    print("Analyzing Grammar VAE latents...")
    grammar_results = analyze_latents(grammar_data)

    print("\n" + "-" * 60)
    print("Analyzing Token VAE latents...")
    token_results = analyze_latents(token_data)

    # Generate comparison
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)

    # Generate Markdown report
    report = generate_full_data_report(grammar_results, token_results)

    # Save report
    report_path = output_dir / 'full_data_representation_summary.md'
    save_report(report, str(report_path))

    # Save JSON results
    json_path = output_dir / 'full_data_results.json'
    results = {
        'grammar': grammar_results,
        'token': token_results,
        'timestamp': datetime.now().isoformat()
    }

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
