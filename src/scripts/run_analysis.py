#!/usr/bin/env python3
"""Run analysis on extracted latent files.

This script loads latent .npz files and computes:
- Clustering metrics (ARI, NMI, Purity, Silhouette)
- Classification accuracy (z -> physics label)
- IID vs OOD comparison (if is_ood mask present)

Usage:
    # Analyze single latent file:
    python scripts/run_analysis.py --latent experiments/latents/hard_families_grammar.npz

    # Analyze all latents in directory:
    python scripts/run_analysis.py --latent-dir experiments/latents/

    # Compare Grammar vs Token:
    python scripts/run_analysis.py \
        --grammar experiments/latents/full_grammar.npz \
        --token experiments/latents/full_token.npz

    # Generate report:
    python scripts/run_analysis.py --latent experiments/latents/full_grammar.npz \
        --output experiments/reports/grammar_analysis.txt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import (
    compute_clustering_metrics,
    compute_all_clustering,
    compute_ood_aware_clustering,
    train_classifier,
    train_ood_aware_classifier,
    assign_physics_labels,
)


def load_latent_file(path: str) -> dict:
    """Load latent .npz file and return contents.

    Args:
        path: Path to .npz file

    Returns:
        Dictionary with latents and metadata
    """
    data = np.load(path, allow_pickle=True)

    result = {
        'path': path,
        'latents': data['latents'] if 'latents' in data else data['mu'],
        'mu': data['mu'] if 'mu' in data else None,
        'logvar': data['logvar'] if 'logvar' in data else None,
    }

    # Load labels
    for key in ['is_ood', 'family', 'dim', 'type', 'nonlinear', 'spatial_order', 'temporal_order', 'pde_strings']:
        if key in data:
            result[key] = data[key]

    # Load config
    if 'config' in data:
        result['config'] = data['config'].item() if hasattr(data['config'], 'item') else dict(data['config'])

    return result


def analyze_latents(data: dict) -> dict:
    """Run full analysis on latent data.

    Args:
        data: Dictionary from load_latent_file()

    Returns:
        Analysis results dictionary
    """
    latents = data['latents']
    n_samples, z_dim = latents.shape

    results = {
        'n_samples': n_samples,
        'z_dim': z_dim,
        'timestamp': datetime.now().isoformat(),
        'clustering': {},
        'classification': {},
    }

    # Prepare labels dict
    labels_dict = {}

    if 'family' in data:
        labels_dict['family'] = data['family']

    if 'type' in data:
        labels_dict['type'] = data['type']

    if 'dim' in data:
        labels_dict['dim'] = [str(d) for d in data['dim']]

    if 'nonlinear' in data:
        labels_dict['nonlinear'] = ['nonlinear' if x else 'linear' for x in data['nonlinear']]

    if 'spatial_order' in data:
        labels_dict['spatial_order'] = [str(o) for o in data['spatial_order']]

    if 'temporal_order' in data:
        labels_dict['temporal_order'] = [str(o) for o in data['temporal_order']]

    # Check for OOD mask
    has_ood = 'is_ood' in data and data['is_ood'].any()

    print(f"\nAnalyzing {n_samples} samples with {z_dim}-dim latents")
    if has_ood:
        n_ood = data['is_ood'].sum()
        print(f"  OOD samples: {n_ood} ({100*n_ood/n_samples:.1f}%)")

    # Clustering metrics
    print("\nComputing clustering metrics...")
    for label_name, labels in labels_dict.items():
        if has_ood:
            metrics = compute_ood_aware_clustering(latents, labels, data['is_ood'], label_name)
        else:
            metrics = compute_clustering_metrics(latents, labels, label_name)
        results['clustering'][label_name] = metrics

    # Classification metrics
    print("Computing classification accuracy...")
    for label_name, labels in labels_dict.items():
        if has_ood:
            metrics = train_ood_aware_classifier(latents, labels, data['is_ood'], label_name)
        else:
            metrics = train_classifier(latents, labels, label_name)
        results['classification'][label_name] = metrics

    return results


def format_results(results: dict, title: str = "Analysis Results") -> str:
    """Format results as text report.

    Args:
        results: Analysis results
        title: Report title

    Returns:
        Formatted string
    """
    lines = [
        "=" * 80,
        title,
        "=" * 80,
        f"\nSamples: {results['n_samples']}",
        f"Latent dim: {results['z_dim']}",
        f"Timestamp: {results['timestamp']}",
        "",
    ]

    # Clustering results
    lines.append("\n" + "=" * 60)
    lines.append("CLUSTERING METRICS")
    lines.append("=" * 60)

    lines.append(f"\n{'Label':<20} {'Split':<8} {'ARI':>10} {'NMI':>10} {'Purity':>10} {'Silhouette':>12}")
    lines.append("-" * 70)

    for label_name, metrics in results['clustering'].items():
        if isinstance(metrics, dict) and 'all' in metrics:
            # OOD-aware results
            for split in ['all', 'iid', 'ood']:
                if split in metrics:
                    m = metrics[split]
                    ari = m.get('ari', np.nan)
                    nmi = m.get('nmi', np.nan)
                    pur = m.get('purity', np.nan)
                    sil = m.get('silhouette', np.nan)
                    lines.append(f"{label_name:<20} {split:<8} {ari:>10.4f} {nmi:>10.4f} {pur:>10.4f} {sil:>12.4f}")
            lines.append("")
        else:
            # Single result
            ari = metrics.get('ari', np.nan)
            nmi = metrics.get('nmi', np.nan)
            pur = metrics.get('purity', np.nan)
            sil = metrics.get('silhouette', np.nan)
            lines.append(f"{label_name:<20} {'all':<8} {ari:>10.4f} {nmi:>10.4f} {pur:>10.4f} {sil:>12.4f}")

    # Classification results
    lines.append("\n" + "=" * 60)
    lines.append("CLASSIFICATION ACCURACY (5-fold CV)")
    lines.append("=" * 60)

    lines.append(f"\n{'Label':<20} {'Split':<8} {'Accuracy':>12} {'Std':>10} {'Classes':>10}")
    lines.append("-" * 60)

    for label_name, metrics in results['classification'].items():
        if isinstance(metrics, dict) and 'iid' in metrics:
            # OOD-aware results
            for split in ['iid', 'ood']:
                if split in metrics:
                    m = metrics[split]
                    if 'accuracy_mean' in m:
                        acc = m['accuracy_mean']
                        std = m.get('accuracy_std', 0)
                        n_cls = m.get('n_classes', '?')
                        lines.append(f"{label_name:<20} {split:<8} {acc:>11.1%} {std:>9.1%} {n_cls:>10}")
                    elif 'accuracy' in m:
                        acc = m['accuracy']
                        lines.append(f"{label_name:<20} {split:<8} {acc:>11.1%} {'N/A':>10} {'N/A':>10}")
            lines.append("")
        else:
            # Single result
            acc = metrics.get('accuracy_mean', np.nan)
            std = metrics.get('accuracy_std', 0)
            n_cls = metrics.get('n_classes', '?')
            lines.append(f"{label_name:<20} {'all':<8} {acc:>11.1%} {std:>9.1%} {n_cls:>10}")

    lines.append("")
    lines.append("=" * 80)

    return '\n'.join(lines)


def compare_grammar_token(grammar_path: str, token_path: str) -> dict:
    """Compare Grammar VAE vs Token VAE latents.

    Args:
        grammar_path: Path to grammar latents .npz
        token_path: Path to token latents .npz

    Returns:
        Comparison results
    """
    print(f"Loading Grammar VAE: {grammar_path}")
    g_data = load_latent_file(grammar_path)
    print(f"Loading Token VAE: {token_path}")
    t_data = load_latent_file(token_path)

    print("\nAnalyzing Grammar VAE...")
    g_results = analyze_latents(g_data)

    print("\nAnalyzing Token VAE...")
    t_results = analyze_latents(t_data)

    # Compare
    comparison = {
        'grammar': g_results,
        'token': t_results,
        'summary': {}
    }

    print("\n" + "=" * 80)
    print("COMPARISON: GRAMMAR VAE vs TOKEN VAE")
    print("=" * 80)

    g_wins = 0
    t_wins = 0

    print(f"\n{'Label':<20} {'Metric':<12} {'Grammar':>12} {'Token':>12} {'Winner':>10}")
    print("-" * 66)

    # Compare clustering NMI
    for label_name in g_results['clustering']:
        g_m = g_results['clustering'][label_name]
        t_m = t_results['clustering'][label_name]

        # Handle OOD-aware
        if isinstance(g_m, dict) and 'all' in g_m:
            g_nmi = g_m['all'].get('nmi', 0)
            t_nmi = t_m['all'].get('nmi', 0)
        else:
            g_nmi = g_m.get('nmi', 0)
            t_nmi = t_m.get('nmi', 0)

        winner = "Grammar" if g_nmi > t_nmi else "Token"
        if g_nmi > t_nmi:
            g_wins += 1
        else:
            t_wins += 1

        print(f"{label_name:<20} {'NMI':<12} {g_nmi:>12.4f} {t_nmi:>12.4f} {winner:>10}")

        comparison['summary'][f'{label_name}_nmi'] = {
            'grammar': g_nmi,
            'token': t_nmi,
            'winner': winner
        }

    # Compare classification
    for label_name in g_results['classification']:
        g_m = g_results['classification'][label_name]
        t_m = t_results['classification'][label_name]

        if isinstance(g_m, dict) and 'iid' in g_m:
            g_acc = g_m['iid'].get('accuracy_mean', 0)
            t_acc = t_m['iid'].get('accuracy_mean', 0)
        else:
            g_acc = g_m.get('accuracy_mean', 0)
            t_acc = t_m.get('accuracy_mean', 0)

        winner = "Grammar" if g_acc > t_acc else "Token"
        if g_acc > t_acc:
            g_wins += 1
        else:
            t_wins += 1

        print(f"{label_name:<20} {'Accuracy':<12} {g_acc:>11.1%} {t_acc:>11.1%} {winner:>10}")

        comparison['summary'][f'{label_name}_acc'] = {
            'grammar': g_acc,
            'token': t_acc,
            'winner': winner
        }

    print("-" * 66)
    print(f"\nTotal wins: Grammar={g_wins}, Token={t_wins}")

    overall_winner = "Grammar VAE" if g_wins > t_wins else ("Token VAE" if t_wins > g_wins else "Tie")
    print(f"\n>>> Overall winner: {overall_winner} <<<")

    comparison['overall'] = {
        'grammar_wins': g_wins,
        'token_wins': t_wins,
        'winner': overall_winner
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Run analysis on latent files')

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--latent', type=str, help='Single latent .npz file')
    input_group.add_argument('--latent-dir', type=str, help='Directory of latent files')
    input_group.add_argument('--grammar', type=str, help='Grammar VAE latent file (for comparison)')

    parser.add_argument('--token', type=str, help='Token VAE latent file (for comparison)')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output report file')
    parser.add_argument('--output-json', type=str, default=None, help='Output JSON file')

    args = parser.parse_args()

    # Comparison mode
    if args.grammar:
        if not args.token:
            parser.error("--grammar requires --token for comparison")

        comparison = compare_grammar_token(args.grammar, args.token)

        if args.output_json:
            with open(args.output_json, 'w') as f:
                # Convert numpy to python types
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    if isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    return obj

                json.dump(convert(comparison), f, indent=2)
                print(f"\nSaved JSON to {args.output_json}")

        return

    # Single file mode
    if args.latent:
        data = load_latent_file(args.latent)
        results = analyze_latents(data)
        report = format_results(results, f"Analysis: {args.latent}")
        print(report)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nSaved report to {args.output}")

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Saved JSON to {args.output_json}")

        return

    # Directory mode
    if args.latent_dir:
        latent_dir = Path(args.latent_dir)
        npz_files = list(latent_dir.glob('*.npz'))

        if not npz_files:
            print(f"No .npz files found in {latent_dir}")
            return

        print(f"Found {len(npz_files)} latent files:")
        for f in npz_files:
            print(f"  {f.name}")

        all_results = {}
        for npz_file in npz_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {npz_file.name}")
            print('='*60)

            data = load_latent_file(str(npz_file))
            results = analyze_latents(data)
            all_results[npz_file.stem] = results

            report = format_results(results, f"Analysis: {npz_file.name}")
            print(report)

        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nSaved combined JSON to {args.output_json}")


if __name__ == '__main__':
    main()
